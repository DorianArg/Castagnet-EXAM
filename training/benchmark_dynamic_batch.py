"""Export and benchmark frozen ResNet18 ONNX with a dynamic batch dimension."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from PIL import Image

from dataset import EXPECTED_MAPPING, check_paths, reject_test_manifest
from models.pretrained import build_pretrained
from transforms import build_transforms
from utils.reproducibility import seed_everything


ROOT = Path(__file__).resolve().parent.parent
VALIDATION_MANIFEST = ROOT / "analysis" / "output" / "validation_manifest.csv"
DEFAULT_IMAGE_ROOT = ROOT / "images"
MODEL_DIR = ROOT / "analysis" / "output" / "modeling" / "resnet18"
CHECKPOINT = MODEL_DIR / "best_model.pt"
CONFIG = MODEL_DIR / "config.json"
OUTPUT_DIR = MODEL_DIR / "onnx" / "batch_benchmark"
ONNX_PATH = MODEL_DIR / "onnx" / "resnet18_castagnet_dynamic_batch.onnx"
EXPECTED_CHECKPOINT_SHA256 = "2aff26da95785868bbead488531874265f242e3d178ee9937cb16ff47b67348d"
EXPECTED_CHECKPOINT_EPOCH = 13
EXPECTED_VALIDATION_SIZE = 5289
OPSET_VERSION = 17
THRESHOLD = 0.55
BATCH_SIZES = (1, 2, 4, 8, 12)
VALIDATION_BATCH_SIZES = (1, 4, 12)
FPS_PER_CAMERA = (1, 2, 5, 10, 25)
MAX_LOGIT_TOLERANCE = 1e-3
MAX_PROBABILITY_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=50)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def threshold_decision(probabilities: np.ndarray) -> np.ndarray:
    alternatives = probabilities[:, 1:].argmax(axis=1) + 1
    return np.where(probabilities[:, 0] >= THRESHOLD, 0, alternatives)


def latency_summary(seconds: list[float], batch_size: int) -> dict[str, float | int]:
    values = np.asarray(seconds, dtype=np.float64)
    mean_seconds = statistics.fmean(seconds)
    return {
        "iterations": len(seconds),
        "batch_size": batch_size,
        "mean_ms_per_batch": mean_seconds * 1000,
        "median_ms_per_batch": statistics.median(seconds) * 1000,
        "p90_ms_per_batch": float(np.percentile(values, 90) * 1000),
        "p95_ms_per_batch": float(np.percentile(values, 95) * 1000),
        "p99_ms_per_batch": float(np.percentile(values, 99) * 1000),
        "min_ms_per_batch": float(values.min() * 1000),
        "max_ms_per_batch": float(values.max() * 1000),
        "mean_ms_per_image": mean_seconds * 1000 / batch_size,
        "images_per_second": batch_size / mean_seconds,
    }


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


def decode_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def paths_for_iteration(paths: list[Path], iteration: int, batch_size: int) -> list[Path]:
    start = (iteration * batch_size) % len(paths)
    return [paths[(start + offset) % len(paths)] for offset in range(batch_size)]


def export_dynamic(model: torch.nn.Module) -> dict:
    dummy = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            ONNX_PATH,
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
            dynamo=False,
        )
    graph = onnx.load(ONNX_PATH)
    onnx.checker.check_model(graph)
    input_shape = tensor_shape(graph.graph.input[0])
    output_shape = tensor_shape(graph.graph.output[0])
    softmax_nodes = sum(node.op_type == "Softmax" for node in graph.graph.node)
    if input_shape != ["batch_size", 3, 224, 224] or output_shape != ["batch_size", 4]:
        raise RuntimeError(f"Dynamic shapes invalid: input={input_shape}, output={output_shape}")
    if softmax_nodes:
        raise RuntimeError("The ONNX graph must expose logits without Softmax")
    return {
        "onnx_path": str(ONNX_PATH.resolve()),
        "onnx_sha256": file_sha256(ONNX_PATH),
        "onnx_size_bytes": ONNX_PATH.stat().st_size,
        "onnx_size_mib": ONNX_PATH.stat().st_size / (1024 ** 2),
        "opset": OPSET_VERSION,
        "input_name": graph.graph.input[0].name,
        "input_shape": input_shape,
        "output_name": graph.graph.output[0].name,
        "output_shape": output_shape,
        "softmax_nodes_in_graph": softmax_nodes,
        "business_threshold_in_graph": False,
    }


def validate_numerically(
    model: torch.nn.Module,
    session: ort.InferenceSession,
    image_paths: list[Path],
    transform,
) -> list[dict]:
    rows = []
    for batch_size in VALIDATION_BATCH_SIZES:
        batch = np.stack([
            transform(decode_rgb(image_paths[index])).numpy()
            for index in range(batch_size)
        ]).astype(np.float32, copy=False)
        with torch.inference_mode():
            torch_logits = model(torch.from_numpy(batch)).numpy()
        ort_logits = session.run(["logits"], {"input": batch})[0]
        torch_probabilities = stable_softmax(torch_logits)
        ort_probabilities = stable_softmax(ort_logits)
        logits_difference = np.abs(torch_logits - ort_logits)
        probability_difference = np.abs(torch_probabilities - ort_probabilities)
        argmax_differences = int(np.count_nonzero(
            torch_logits.argmax(axis=1) != ort_logits.argmax(axis=1)
        ))
        decision_differences = int(np.count_nonzero(
            threshold_decision(torch_probabilities) != threshold_decision(ort_probabilities)
        ))
        row = {
            "batch_size": batch_size,
            "max_absolute_difference_logits": float(logits_difference.max()),
            "mean_absolute_difference_logits": float(logits_difference.mean()),
            "max_absolute_difference_probabilities": float(probability_difference.max()),
            "mean_absolute_difference_probabilities": float(probability_difference.mean()),
            "argmax_differences": argmax_differences,
            "threshold_055_decision_differences": decision_differences,
        }
        row["valid"] = (
            row["max_absolute_difference_logits"] <= MAX_LOGIT_TOLERANCE
            and row["max_absolute_difference_probabilities"] <= MAX_PROBABILITY_TOLERANCE
            and argmax_differences == 0
            and decision_differences == 0
        )
        rows.append(row)
    if not all(row["valid"] for row in rows):
        raise RuntimeError(f"Dynamic numerical validation failed: {rows}")
    return rows


def benchmark_batch(
    session: ort.InferenceSession,
    paths: list[Path],
    transform,
    batch_size: int,
    iterations: int,
    warmup: int,
) -> dict:
    decoded_pool = [decode_rgb(path) for path in paths]
    tensor_pool = [transform(image).numpy() for image in decoded_pool]
    prepared_batches = [
        np.stack([tensor_pool[(start + offset) % len(tensor_pool)] for offset in range(batch_size)]).astype(np.float32, copy=False)
        for start in range(min(len(tensor_pool), 16))
    ]
    for index in range(warmup):
        session.run(["logits"], {"input": prepared_batches[index % len(prepared_batches)]})

    inference_times = []
    for index in range(iterations):
        batch = prepared_batches[index % len(prepared_batches)]
        started = time.perf_counter()
        session.run(["logits"], {"input": batch})
        inference_times.append(time.perf_counter() - started)

    decode_times = []
    for index in range(iterations):
        selected = paths_for_iteration(paths, index, batch_size)
        started = time.perf_counter()
        _ = [decode_rgb(path) for path in selected]
        decode_times.append(time.perf_counter() - started)

    preprocessing_times = []
    for index in range(iterations):
        selected = [decoded_pool[(index * batch_size + offset) % len(decoded_pool)] for offset in range(batch_size)]
        started = time.perf_counter()
        _ = [transform(image).numpy() for image in selected]
        preprocessing_times.append(time.perf_counter() - started)

    batch_construction_times = []
    for index in range(iterations):
        selected = [tensor_pool[(index * batch_size + offset) % len(tensor_pool)] for offset in range(batch_size)]
        started = time.perf_counter()
        _ = np.stack(selected).astype(np.float32, copy=False)
        batch_construction_times.append(time.perf_counter() - started)

    end_to_end_times = []
    for index in range(iterations):
        selected = paths_for_iteration(paths, index, batch_size)
        started = time.perf_counter()
        decoded = [decode_rgb(path) for path in selected]
        tensors = [transform(image).numpy() for image in decoded]
        batch = np.stack(tensors).astype(np.float32, copy=False)
        session.run(["logits"], {"input": batch})
        end_to_end_times.append(time.perf_counter() - started)

    return {
        "batch_size": batch_size,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "decode_only": latency_summary(decode_times, batch_size),
        "preprocessing_without_decode": latency_summary(preprocessing_times, batch_size),
        "batch_construction": latency_summary(batch_construction_times, batch_size),
        "inference_only": latency_summary(inference_times, batch_size),
        "preprocessing_plus_inference": latency_summary(end_to_end_times, batch_size),
        "measurement_scope": {
            "decode_only": "PIL open, decode, RGB conversion and in-memory copy",
            "preprocessing_without_decode": "deterministic resize, padding, tensor conversion and ImageNet normalization",
            "batch_construction": "numpy.stack of already preprocessed CHW arrays",
            "inference_only": "ONNX Runtime session.run; includes host-to-device input transfer, CUDA execution, synchronization and output return",
            "preprocessing_plus_inference": "decode + preprocessing + batch construction + session.run",
            "cpu_to_gpu_copy_separately_measured": False,
        },
    }


def create_figures(rows: list[dict]) -> None:
    batch_sizes = [row["batch_size"] for row in rows]
    plt.figure(figsize=(7, 4))
    plt.plot(batch_sizes, [row["inference_only"]["images_per_second"] for row in rows], marker="o", label="Inference seule")
    plt.plot(batch_sizes, [row["preprocessing_plus_inference"]["images_per_second"] for row in rows], marker="o", label="End-to-end")
    plt.xlabel("Taille du batch")
    plt.ylabel("Images/seconde")
    plt.xticks(batch_sizes)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "batch_throughput.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(batch_sizes, [row["inference_only"]["mean_ms_per_image"] for row in rows], marker="o", label="Inference seule")
    plt.plot(batch_sizes, [row["preprocessing_plus_inference"]["mean_ms_per_image"] for row in rows], marker="o", label="End-to-end")
    plt.xlabel("Taille du batch")
    plt.ylabel("Latence moyenne par image (ms)")
    plt.xticks(batch_sizes)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "batch_latency_per_image.png", dpi=160)
    plt.close()


def write_report(summary: dict, rows: list[dict]) -> None:
    lines = [
        "# Benchmark ONNX ResNet18 a batch dynamique",
        "",
        "Export du checkpoint fige epoch 13, opset 17, logits uniquement. Validation et benchmark utilisent exclusivement des images du manifeste validation.",
        "",
        "## Validation numerique",
        "",
        "| Batch | Max ecart logits | Ecart moyen logits | Max ecart probabilites | Argmax differents | Decisions seuil 0.55 differentes |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["numerical_validation"]:
        lines.append(
            f"| {row['batch_size']} | {row['max_absolute_difference_logits']:.9f} | "
            f"{row['mean_absolute_difference_logits']:.9f} | {row['max_absolute_difference_probabilities']:.9f} | "
            f"{row['argmax_differences']} | {row['threshold_055_decision_differences']} |"
        )
    lines += [
        "",
        "## Benchmark CUDA",
        "",
        "| Batch | Inf. moyenne ms/batch | Inf. p95 | Inf. ms/image | Inf. img/s | E2E moyenne ms/batch | E2E p95 | E2E ms/image | E2E img/s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        inf = row["inference_only"]
        e2e = row["preprocessing_plus_inference"]
        lines.append(
            f"| {row['batch_size']} | {inf['mean_ms_per_batch']:.3f} | {inf['p95_ms_per_batch']:.3f} | "
            f"{inf['mean_ms_per_image']:.3f} | {inf['images_per_second']:.2f} | "
            f"{e2e['mean_ms_per_batch']:.3f} | {e2e['p95_ms_per_batch']:.3f} | "
            f"{e2e['mean_ms_per_image']:.3f} | {e2e['images_per_second']:.2f} |"
        )
    lines += [
        "",
        "## Etapes isolees",
        "",
        "| Batch | Decodage ms/batch | Preprocessing sans decodage ms/batch | Constitution batch ms/batch | session.run ms/batch |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch_size']} | {row['decode_only']['mean_ms_per_batch']:.3f} | "
            f"{row['preprocessing_without_decode']['mean_ms_per_batch']:.3f} | "
            f"{row['batch_construction']['mean_ms_per_batch']:.3f} | "
            f"{row['inference_only']['mean_ms_per_batch']:.3f} |"
        )
    lines += [
        "",
        "## Capacite theorique pour 12 cameras",
        "",
        "| Batch | Mesure | 12 img/s | 24 img/s | 60 img/s | 120 img/s | 300 img/s |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for scope, label in (("inference_only", "Inference seule"), ("preprocessing_plus_inference", "End-to-end")):
            throughput = row[scope]["images_per_second"]
            cells = ["OK" if throughput >= 12 * fps else "NON" for fps in FPS_PER_CAMERA]
            lines.append(
                f"| {row['batch_size']} | {label} ({throughput:.2f} img/s) | "
                + " | ".join(cells) + " |"
            )
    lines += [
        "",
        "## Portee des mesures",
        "",
        "Le decodage, le preprocessing sans decodage et la constitution NumPy du batch sont mesures separement. `session.run` regroupe la copie de l'entree CPU vers le GPU, l'execution CUDA, la synchronisation et le retour de la sortie; la copie CPU-GPU n'est donc pas isolee.",
        "",
        "La mesure end-to-end (decodage + preprocessing + batch + session.run) est la reference applicative. Un batch de 12 mesure uniquement le potentiel de mutualisation du GPU et ne simule pas douze flux video concurrents.",
        "",
        "## Limites",
        "",
        "Mesures realisees sur RTX 3080 Laptop 8 Go. Aucune extrapolation numerique n'est faite vers la GTX 1060 3 Go / ancien i7. Le test ne couvre ni capture video, ni files d'attente, ni ordonnancement de 12 flux, ni contention memoire d'une application complete.",
    ]
    (OUTPUT_DIR / "batch_benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.warmup < 50:
        raise ValueError("Au moins 50 warmups sont requis")
    if args.iterations < 200:
        raise ValueError("Au moins 200 iterations sont requises")
    if file_sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Le checkpoint ResNet18 fige a change")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected_config = {"model_name": "resnet18", "num_classes": 4, "input_size": 224, "normalization": "ImageNet", "seed": 20260902}
    if any(config.get(key) != value for key, value in expected_config.items()):
        raise RuntimeError("La configuration ResNet18 figee a change")
    reject_test_manifest(VALIDATION_MANIFEST)
    validation = pd.read_csv(VALIDATION_MANIFEST, usecols=["filename"])
    if len(validation) != EXPECTED_VALIDATION_SIZE or not validation.filename.is_unique:
        raise RuntimeError("Le manifeste validation fige n'est plus conforme")
    image_root = args.image_root.resolve()
    path_audit = check_paths(VALIDATION_MANIFEST, image_root)
    if path_audit != {"expected": 5289, "found": 5289, "missing": 0}:
        raise RuntimeError(f"Audit images validation inattendu: {path_audit}")

    seed_everything(config["seed"])
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != EXPECTED_CHECKPOINT_EPOCH or checkpoint.get("class_mapping") != EXPECTED_MAPPING:
        raise RuntimeError("Epoch ou mapping du checkpoint inattendu")
    model = build_pretrained("resnet18", num_classes=4, use_weights=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export = export_dynamic(model)
    _ = torch.cuda.is_available()
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError("CUDAExecutionProvider indisponible")
    validation_session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CUDAExecutionProvider"])
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"CUDA non actif: {session.get_providers()}")

    pool_indices = np.linspace(0, len(validation) - 1, num=64, dtype=int)
    image_paths = [image_root / validation.iloc[index].filename for index in pool_indices]
    transform = build_transforms(224, train=False)
    numerical_validation = validate_numerically(model, validation_session, image_paths, transform)
    export_summary = {
        "status": "valid",
        "created_at": datetime.now().astimezone().isoformat(),
        "model": "resnet18",
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "checkpoint_epoch": EXPECTED_CHECKPOINT_EPOCH,
        "validation_only": True,
        "test_manifest_loaded": False,
        "numerical_validation_provider": validation_session.get_providers(),
        **export,
        "numerical_validation": numerical_validation,
        "tolerances": {"max_logits": MAX_LOGIT_TOLERANCE, "max_probabilities": MAX_PROBABILITY_TOLERANCE},
    }
    (OUTPUT_DIR / "dynamic_export_summary.json").write_text(json.dumps(export_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    for batch_size in BATCH_SIZES:
        print(f"benchmark batch={batch_size}", flush=True)
        rows.append(benchmark_batch(session, image_paths, transform, batch_size, args.iterations, args.warmup))
    capacities = []
    for row in rows:
        for fps in FPS_PER_CAMERA:
            required = 12 * fps
            capacities.append({
                "batch_size": row["batch_size"],
                "fps_per_camera": fps,
                "required_images_per_second": required,
                "inference_only_met": row["inference_only"]["images_per_second"] >= required,
                "end_to_end_met": row["preprocessing_plus_inference"]["images_per_second"] >= required,
            })
    result = {
        "status": "completed",
        "created_at": datetime.now().astimezone().isoformat(),
        "model": "resnet18",
        "provider": "CUDAExecutionProvider",
        "providers_active": session.get_providers(),
        "validation_only": True,
        "test_manifest_loaded": False,
        "training_performed": False,
        "iterations_per_batch_size": args.iterations,
        "warmup_iterations_per_batch_size": args.warmup,
        "batch_results": rows,
        "theoretical_12_stream_capacity": capacities,
        "target_hardware_is_current_machine": False,
        "target_hardware": "GTX 1060 3 GB + older i7",
        "no_numeric_extrapolation_to_target": True,
    }
    (OUTPUT_DIR / "batch_benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    flat_rows = []
    for row in rows:
        flat = {"batch_size": row["batch_size"]}
        for scope in ("decode_only", "preprocessing_without_decode", "batch_construction", "inference_only", "preprocessing_plus_inference"):
            for key, value in row[scope].items():
                if key not in {"iterations", "batch_size"}:
                    flat[f"{scope}_{key}"] = value
        flat_rows.append(flat)
    with (OUTPUT_DIR / "batch_benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    create_figures(rows)
    write_report(export_summary, rows)
    print(json.dumps({
        "status": "completed",
        "dynamic_export_valid": True,
        "output_dir": str(OUTPUT_DIR.resolve()),
        "results": [{
            "batch_size": row["batch_size"],
            "inference_images_per_second": row["inference_only"]["images_per_second"],
            "end_to_end_images_per_second": row["preprocessing_plus_inference"]["images_per_second"],
        } for row in rows],
    }, indent=2))


if __name__ == "__main__":
    main()
