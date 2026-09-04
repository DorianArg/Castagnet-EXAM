"""Calibre le seuil Conforme sur la validation uniquement, sans entraînement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import CastagNetDataset, EXPECTED_MAPPING, check_paths, reject_test_manifest
from metrics import classification_metrics
from models.pretrained import build_pretrained
from transforms import build_transforms
from utils.reproducibility import seed_everything


TRAINING_DIR = Path(__file__).resolve().parent
ROOT = TRAINING_DIR.parent
DEFAULT_VALIDATION = ROOT / "analysis" / "output" / "validation_manifest.csv"
DEFAULT_IMAGES = ROOT / "images"
DEFAULT_CHECKPOINT = ROOT / "analysis" / "output" / "modeling" / "resnet18" / "best_model.pt"
DEFAULT_CONFIG = ROOT / "analysis" / "output" / "modeling" / "resnet18" / "config.json"
DEFAULT_OUTPUT = ROOT / "analysis" / "output" / "modeling" / "resnet18" / "calibration"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def select_threshold(curve: pd.DataFrame) -> tuple[pd.Series, bool]:
    eligible = curve[
        (curve.precision_conforme >= 0.95) & (curve.recall_conforme >= 0.85)
    ]
    if not eligible.empty:
        selected = eligible.sort_values(
            ["recall_conforme", "macro_f1", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
        return selected, True
    selected = curve.sort_values(
        ["distance_to_objectives", "macro_f1", "recall_conforme"],
        ascending=[True, False, False],
    ).iloc[0]
    return selected, False


def row_to_dict(row: pd.Series) -> dict[str, int | float]:
    return {
        key: int(value) if key in {
            "non_conforme_to_conforme",
            "pietra_to_conforme",
            "false_conforme_total",
            "conforme_rejected",
        } else float(value)
        for key, value in row.items()
    }


def main() -> None:
    args = parse_args()
    validation_path = args.validation_manifest.resolve()
    reject_test_manifest(validation_path)
    if validation_path.name != "validation_manifest.csv":
        raise ValueError("La calibration accepte uniquement validation_manifest.csv")

    checkpoint_path = args.checkpoint.resolve()
    config_path = args.config.resolve()
    image_root = args.image_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("model_name") != "resnet18" or not config.get("pretrained", False):
        raise ValueError("La calibration demandée requiert la configuration ResNet18 pré-entraînée")

    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_audit = check_paths(validation_path, image_root)
    if path_audit["missing"]:
        raise RuntimeError(f"Images de validation manquantes : {path_audit}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("class_mapping") != EXPECTED_MAPPING:
        raise ValueError("Le mapping du checkpoint ne correspond pas au mapping figé")
    model = build_pretrained("resnet18", num_classes=4, use_weights=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = CastagNetDataset(
        validation_path,
        image_root,
        EXPECTED_MAPPING,
        build_transforms(config["input_size"], train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=bool(config["pin_memory"] and device.type == "cuda"),
    )
    probabilities, targets = [], []
    with torch.inference_mode():
        for images, labels, _ in loader:
            logits = model(images.to(device, non_blocking=True))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Logits non finis pendant la calibration")
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(labels.numpy())
    probabilities_array = np.concatenate(probabilities)
    y_true = np.concatenate(targets)
    if len(y_true) != 5289:
        raise RuntimeError(f"Validation incomplète : {len(y_true)} images au lieu de 5289")

    rows = []
    for threshold_integer in range(50, 100):
        threshold = threshold_integer / 100
        non_conforme_predictions = probabilities_array[:, 1:].argmax(axis=1) + 1
        y_pred = np.where(
            probabilities_array[:, EXPECTED_MAPPING["Conforme"]] >= threshold,
            EXPECTED_MAPPING["Conforme"],
            non_conforme_predictions,
        )
        metrics = classification_metrics(y_true, y_pred)
        conforme = metrics["per_class"]["Conforme"]
        errors = metrics["business_errors"]
        precision_shortfall = max(0.0, 0.95 - conforme["precision"])
        recall_shortfall = max(0.0, 0.85 - conforme["recall"])
        rows.append({
            "threshold": threshold,
            "precision_conforme": conforme["precision"],
            "recall_conforme": conforme["recall"],
            "f1_conforme": conforme["f1"],
            "non_conforme_to_conforme": errors["false_conforme_non_conforme"],
            "pietra_to_conforme": errors["false_conforme_pietra"],
            "false_conforme_total": errors["false_conforme_total"],
            "conforme_rejected": errors["conforme_rejected"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "precision_shortfall": precision_shortfall,
            "recall_shortfall": recall_shortfall,
            "distance_to_objectives": float(np.hypot(precision_shortfall, recall_shortfall)),
        })
    curve = pd.DataFrame(rows)
    curve_path = output_dir / "conforme_threshold_curve.csv"
    curve.to_csv(curve_path, index=False)
    selected, objectives_met = select_threshold(curve)
    selected_values = row_to_dict(selected)
    eligible_count = int(
        ((curve.precision_conforme >= 0.95) & (curve.recall_conforme >= 0.85)).sum()
    )
    summary = {
        "status": "completed",
        "selection": "objectives_met" if objectives_met else "minimum_euclidean_shortfall",
        "objectives": {"precision_conforme": 0.95, "recall_conforme": 0.85},
        "eligible_threshold_count": eligible_count,
        "recommended": selected_values,
        "validation_only": True,
        "test_loaded": False,
        "validation_images": len(y_true),
        "thresholds_tested": 50,
        "threshold_min": 0.50,
        "threshold_max": 0.99,
        "threshold_step": 0.01,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_val_loss": float(checkpoint["val_loss"]),
        "validation_manifest": str(validation_path),
        "validation_manifest_sha256": file_sha256(validation_path),
        "curve_csv": str(curve_path),
    }
    summary_path = output_dir / "conforme_threshold_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    metric_lines = "\n".join(
        f"- `{key}` : {value}" for key, value in selected_values.items()
    )
    report = f"""# Calibration du seuil Conforme — ResNet18

Calibration effectuée uniquement sur les {len(y_true)} images de validation avec le
checkpoint inchangé de l'epoch {checkpoint['epoch']}. Le manifeste test n'a pas été chargé.

## Règle

Si `P(Conforme) >= seuil`, la prédiction est Conforme. Sinon, elle est l'argmax
parmi NON Conforme, PIETRA et Vide. Les 50 seuils de 0,50 à 0,99 ont été testés.

## Résultat

- Seuils satisfaisant simultanément les deux objectifs : {eligible_count}
- Méthode de sélection : {summary['selection']}

{metric_lines}

En l'absence de seuil admissible, la distance est la norme euclidienne des seuls
écarts manquants à 95 % de précision et 85 % de rappel. Aucun réentraînement et
aucune modification du checkpoint, des images ou des splits n'ont été effectués.
"""
    report_path = output_dir / "conforme_threshold_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
