"""Export a frozen trained model to ONNX and validate it on validation only."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from dataset import CastagNetDataset, EXPECTED_MAPPING, check_paths, reject_test_manifest
from models.pretrained import build_pretrained
from transforms import build_transforms
from utils.reproducibility import seed_everything


TRAINING_DIR = Path(__file__).resolve().parent
ROOT = TRAINING_DIR.parent
VALIDATION_MANIFEST = ROOT / "analysis" / "output" / "validation_manifest.csv"
DEFAULT_IMAGE_ROOT = ROOT / "images"
MODEL_SOURCES = {
    "resnet18": {
        "run_id": "db0ee7873903411e899e2f85384ae58a",
        "checkpoint_epoch": 13,
        "checkpoint_sha256": "2aff26da95785868bbead488531874265f242e3d178ee9937cb16ff47b67348d",
    },
    "efficientnet_b0": {
        "run_id": "a434cccec8624ad2bcdae5fe952726e9",
        "checkpoint_epoch": 10,
        "checkpoint_sha256": "ece11b08bcd9bfc3407e18bea4a5a1c5161a37ba5f5f5b0aee419b0459eb00be",
    },
    "mobilenet_v3_small": {
        "run_id": "8b9b161b9dd24e208af9be3baa532f2e",
        "checkpoint_epoch": 11,
        "checkpoint_sha256": "f225e8e1e1d3839c79f82c9fbd86be88115a9d5b9702beba498d1a71f83ee42b",
    },
}
EXPECTED_VALIDATION_SIZE = 5289
CONFORME_THRESHOLD = 0.55
OPSET_VERSION = 17
INPUT_SHAPE = [1, 3, 224, 224]
OUTPUT_SHAPE = [1, 4]
MAX_LOGIT_TOLERANCE = 1e-3
MAX_PROBABILITY_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_SOURCES, default="resnet18")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--validation-samples", type=int, default=16)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def threshold_decision(probabilities: np.ndarray) -> np.ndarray:
    alternatives = probabilities[:, 1:].argmax(axis=1) + 1
    return np.where(probabilities[:, 0] >= CONFORME_THRESHOLD, 0, alternatives)


def tensor_shape(value_info) -> list[int | str | None]:
    result = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            result.append(dimension.dim_param)
        else:
            result.append(None)
    return result


def model_paths(model_name: str) -> dict[str, Path]:
    model_dir = ROOT / "analysis" / "output" / "modeling" / model_name
    output_dir = model_dir / "onnx"
    return {
        "checkpoint": model_dir / "best_model.pt",
        "config": model_dir / "config.json",
        "training_summary": model_dir / "metrics_summary.json",
        "output_dir": output_dir,
        "onnx": output_dir / f"{model_name}_castagnet.onnx",
        "export_summary": output_dir / "export_summary.json",
        "details": output_dir / "numerical_validation_details.csv",
    }


def validate_frozen_sources(model_name: str, paths: dict[str, Path]) -> tuple[dict, dict]:
    expected = MODEL_SOURCES[model_name]
    if file_sha256(paths["checkpoint"]) != expected["checkpoint_sha256"]:
        raise RuntimeError(f"L'empreinte du checkpoint final {model_name} a change")
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    training_summary = json.loads(paths["training_summary"].read_text(encoding="utf-8"))
    expected_config = {
        "model_name": model_name,
        "pretrained": True,
        "num_classes": 4,
        "input_size": 224,
        "normalization": "ImageNet",
        "seed": 20260902,
    }
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise RuntimeError(f"La configuration finale {model_name} ou son preprocessing a change")
    if training_summary.get("run_id") != expected["run_id"]:
        raise RuntimeError("Le run MLflow ne correspond pas au run final fige")
    if training_summary.get("best_epoch") != expected["checkpoint_epoch"]:
        raise RuntimeError("Le meilleur epoch documente ne correspond plus au checkpoint fige")
    return config, training_summary


def main() -> None:
    args = parse_args()
    if not 1 <= args.validation_samples <= 128:
        raise ValueError("--validation-samples doit etre compris entre 1 et 128")
    paths = model_paths(args.model)
    expected = MODEL_SOURCES[args.model]
    config, training_summary = validate_frozen_sources(args.model, paths)
    reject_test_manifest(VALIDATION_MANIFEST)
    if VALIDATION_MANIFEST.name != "validation_manifest.csv":
        raise RuntimeError("La validation numerique accepte uniquement validation_manifest.csv")
    seed_everything(config["seed"])
    image_root = args.image_root.resolve()
    expected_audit = {"expected": EXPECTED_VALIDATION_SIZE, "found": EXPECTED_VALIDATION_SIZE, "missing": 0}
    path_audit = check_paths(VALIDATION_MANIFEST, image_root)
    if path_audit != expected_audit:
        raise RuntimeError(f"Audit des images validation inattendu : {path_audit}")

    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != expected["checkpoint_epoch"]:
        raise RuntimeError("Le checkpoint charge n'est pas celui de l'epoch fige")
    if checkpoint.get("class_mapping") != EXPECTED_MAPPING:
        raise RuntimeError("Le mapping du checkpoint a change")
    model = build_pretrained(args.model, num_classes=4, use_weights=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            model, dummy, paths["onnx"], export_params=True, opset_version=OPSET_VERSION,
            do_constant_folding=True, input_names=["input"], output_names=["logits"],
            dynamic_axes=None, dynamo=False,
        )

    onnx_model = onnx.load(paths["onnx"])
    onnx.checker.check_model(onnx_model)
    graph_input_shape = tensor_shape(onnx_model.graph.input[0])
    graph_output_shape = tensor_shape(onnx_model.graph.output[0])
    if graph_input_shape != INPUT_SHAPE or graph_output_shape != OUTPUT_SHAPE:
        raise RuntimeError(f"Shapes ONNX inattendues : input={graph_input_shape}, output={graph_output_shape}")
    if onnx_model.graph.input[0].name != "input" or onnx_model.graph.output[0].name != "logits":
        raise RuntimeError("Noms des tenseurs ONNX inattendus")
    softmax_nodes = sum(node.op_type == "Softmax" for node in onnx_model.graph.node)
    if softmax_nodes:
        raise RuntimeError("Le graphe ONNX contient un Softmax")

    available_providers = ort.get_available_providers()
    session = ort.InferenceSession(str(paths["onnx"]), providers=["CPUExecutionProvider"])
    if list(session.get_inputs()[0].shape) != INPUT_SHAPE or list(session.get_outputs()[0].shape) != OUTPUT_SHAPE:
        raise RuntimeError("Les shapes exposees par ONNX Runtime sont incorrectes")

    dataset = CastagNetDataset(
        VALIDATION_MANIFEST, image_root, EXPECTED_MAPPING,
        build_transforms(config["input_size"], train=False),
    )
    sample_indices = np.linspace(0, len(dataset) - 1, num=args.validation_samples, dtype=int).tolist()
    loader = DataLoader(Subset(dataset, sample_indices), batch_size=1, shuffle=False, num_workers=0)
    detail_rows, all_torch_logits, all_ort_logits = [], [], []
    filenames = dataset.frame.iloc[sample_indices].filename.tolist()
    with torch.inference_mode():
        for sample_number, ((images, _, _), filename) in enumerate(zip(loader, filenames), start=1):
            torch_logits = model(images).cpu().numpy()
            ort_logits = session.run(["logits"], {"input": images.numpy()})[0]
            torch_probabilities = stable_softmax(torch_logits)
            ort_probabilities = stable_softmax(ort_logits)
            logit_difference = np.abs(torch_logits - ort_logits)
            probability_difference = np.abs(torch_probabilities - ort_probabilities)
            detail_rows.append({
                "sample_number": sample_number,
                "validation_index": sample_indices[sample_number - 1],
                "filename": filename,
                "max_abs_difference_logits": float(logit_difference.max()),
                "mean_abs_difference_logits": float(logit_difference.mean()),
                "max_abs_difference_probabilities": float(probability_difference.max()),
                "argmax_pytorch": int(torch_logits.argmax(axis=1)[0]),
                "argmax_onnx": int(ort_logits.argmax(axis=1)[0]),
                "threshold_decision_pytorch": int(threshold_decision(torch_probabilities)[0]),
                "threshold_decision_onnx": int(threshold_decision(ort_probabilities)[0]),
            })
            all_torch_logits.append(torch_logits)
            all_ort_logits.append(ort_logits)
    pd.DataFrame(detail_rows).to_csv(paths["details"], index=False)
    torch_logits = np.concatenate(all_torch_logits)
    ort_logits = np.concatenate(all_ort_logits)
    torch_probabilities = stable_softmax(torch_logits)
    ort_probabilities = stable_softmax(ort_logits)
    logits_difference = np.abs(torch_logits - ort_logits)
    probabilities_difference = np.abs(torch_probabilities - ort_probabilities)
    argmax_differences = int(np.count_nonzero(torch_logits.argmax(axis=1) != ort_logits.argmax(axis=1)))
    threshold_differences = int(np.count_nonzero(
        threshold_decision(torch_probabilities) != threshold_decision(ort_probabilities)
    ))
    max_logits = float(logits_difference.max())
    max_probabilities = float(probabilities_difference.max())
    numerically_valid = (
        max_logits <= MAX_LOGIT_TOLERANCE and max_probabilities <= MAX_PROBABILITY_TOLERANCE
        and argmax_differences == 0 and threshold_differences == 0
    )
    initializer_parameters = int(sum(np.prod(list(item.dims)) for item in onnx_model.graph.initializer))
    summary = {
        "status": "valid" if numerically_valid else "invalid",
        "created_at": datetime.now().astimezone().isoformat(),
        "model": args.model,
        "checkpoint": str(paths["checkpoint"].resolve()),
        "checkpoint_sha256": expected["checkpoint_sha256"],
        "checkpoint_epoch": expected["checkpoint_epoch"],
        "training_run_id": expected["run_id"],
        "training_best_val_loss": training_summary["best_val_loss"],
        "onnx_path": str(paths["onnx"].resolve()),
        "onnx_sha256": file_sha256(paths["onnx"]),
        "opset": OPSET_VERSION,
        "input_name": "input", "input_shape": graph_input_shape,
        "output_name": "logits", "output_shape": graph_output_shape,
        "output_is_logits": True,
        "softmax_nodes_in_graph": softmax_nodes,
        "business_threshold_in_graph": False,
        "onnx_size_bytes": paths["onnx"].stat().st_size,
        "onnx_size_mib": paths["onnx"].stat().st_size / (1024 ** 2),
        "pytorch_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "onnx_initializer_values": initializer_parameters,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "onnxruntime_available_providers": available_providers,
        "validation_provider": session.get_providers(),
        "numerical_validation": {
            "manifest": str(VALIDATION_MANIFEST.resolve()),
            "test_manifest_loaded": False,
            "sample_selection": "indices evenly spaced with numpy.linspace",
            "sample_count": len(sample_indices), "sample_indices": sample_indices,
            "max_absolute_difference_logits": max_logits,
            "mean_absolute_difference_logits": float(logits_difference.mean()),
            "max_absolute_difference_probabilities": max_probabilities,
            "mean_absolute_difference_probabilities": float(probabilities_difference.mean()),
            "argmax_prediction_differences": argmax_differences,
            "threshold_055_decision_differences": threshold_differences,
            "max_logit_tolerance": MAX_LOGIT_TOLERANCE,
            "max_probability_tolerance": MAX_PROBABILITY_TOLERANCE,
            "valid": numerically_valid,
        },
    }
    paths["export_summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not numerically_valid:
        raise RuntimeError("La validation numerique PyTorch / ONNX a echoue")


if __name__ == "__main__":
    main()
