"""Évaluation finale unique du test avec la configuration ResNet18 figée."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataset import EXPECTED_MAPPING
from metrics import CLASS_NAMES, classification_metrics, normalized_confusion
from models.pretrained import build_pretrained
from transforms import build_transforms
from utils.reproducibility import seed_everything


TRAINING_DIR = Path(__file__).resolve().parent
ROOT = TRAINING_DIR.parent
TEST_MANIFEST = ROOT / "analysis" / "output" / "test_manifest.csv"
CHECKPOINT = ROOT / "analysis" / "output" / "modeling" / "resnet18" / "best_model.pt"
MODEL_CONFIG = ROOT / "analysis" / "output" / "modeling" / "resnet18" / "config.json"
MODEL_SUMMARY = ROOT / "analysis" / "output" / "modeling" / "resnet18" / "metrics_summary.json"
DEFAULT_IMAGE_ROOT = ROOT / "images"
OUTPUT_DIR = ROOT / "analysis" / "output" / "modeling" / "resnet18" / "final_test"

EXPECTED_TEST_SIZE = 5288
EXPECTED_RUN_ID = "db0ee7873903411e899e2f85384ae58a"
EXPECTED_CHECKPOINT_EPOCH = 13
EXPECTED_CHECKPOINT_SHA256 = "2aff26da95785868bbead488531874265f242e3d178ee9937cb16ff47b67348d"
CONFORME_THRESHOLD = 0.55
PRECISION_TARGET = 0.95
RECALL_TARGET = 0.85


class FinalTestDataset(Dataset):
    """Dataset local dédié : seul le manifeste test figé est accepté."""

    def __init__(self, manifest_path: Path, image_root: Path, transform) -> None:
        manifest_path = manifest_path.resolve()
        if manifest_path != TEST_MANIFEST.resolve() or manifest_path.name != "test_manifest.csv":
            raise ValueError("Seul le test_manifest.csv figé peut être évalué")
        self.frame = pd.read_csv(manifest_path, low_memory=False)
        required = {"filename", "effective_label"}
        if not required.issubset(self.frame.columns):
            raise ValueError(f"Colonnes test manquantes : {required - set(self.frame.columns)}")
        if len(self.frame) != EXPECTED_TEST_SIZE:
            raise ValueError(
                f"Taille test inattendue : {len(self.frame)} au lieu de {EXPECTED_TEST_SIZE}"
            )
        if not self.frame.filename.is_unique:
            raise ValueError("filename doit rester unique dans le manifeste test")
        if not self.frame.effective_label.isin(EXPECTED_MAPPING).all():
            raise ValueError("effective_label test contient une classe inconnue")
        self.image_root = image_root.resolve()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.frame.iloc[index]
        path = self.image_root / row.filename
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, EXPECTED_MAPPING[row.effective_label], index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serializable_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cross_entropy_loss: float,
) -> tuple[dict[str, object], np.ndarray]:
    metrics = classification_metrics(y_true, y_pred)
    matrix = metrics.pop("confusion_matrix")
    metrics["loss"] = cross_entropy_loss
    metrics["confusion_matrix"] = matrix.tolist()
    return metrics, matrix


def save_confusion_figure(matrix: np.ndarray, path: Path, normalized: bool) -> None:
    values = normalized_confusion(matrix) if normalized else matrix
    fmt = ".2f" if normalized else "d"
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    image = ax.imshow(values, cmap="Blues", vmin=0)
    for row in range(4):
        for column in range(4):
            ax.text(column, row, format(values[row, column], fmt), ha="center", va="center")
    ax.set_xticks(range(4), CLASS_NAMES, rotation=25, ha="right")
    ax.set_yticks(range(4), CLASS_NAMES)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title("Test final — " + ("normalisée" if normalized else "brute"))
    fig.colorbar(image, ax=ax, fraction=.046)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def verify_frozen_configuration() -> tuple[dict, dict, dict]:
    if CONFORME_THRESHOLD != 0.55:
        raise RuntimeError("Le seuil métier figé a été altéré")
    if file_sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("L'empreinte du checkpoint ResNet18 ne correspond plus")
    config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    summary = json.loads(MODEL_SUMMARY.read_text(encoding="utf-8"))
    if config.get("model_name") != "resnet18" or not config.get("pretrained", False):
        raise RuntimeError("La configuration figée n'est pas ResNet18 pré-entraîné")
    frozen_preprocessing = {
        "input_size": 224,
        "normalization": "ImageNet",
        "seed": 20260902,
    }
    if any(config.get(key) != value for key, value in frozen_preprocessing.items()):
        raise RuntimeError("Le preprocessing ou la seed de la configuration figée a été altéré")
    if summary.get("run_id") != EXPECTED_RUN_ID:
        raise RuntimeError("Le run MLflow du modèle ne correspond pas au run figé")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != EXPECTED_CHECKPOINT_EPOCH:
        raise RuntimeError("Le checkpoint n'est pas celui de l'epoch 13")
    if checkpoint.get("class_mapping") != EXPECTED_MAPPING:
        raise RuntimeError("Le mapping du checkpoint a été modifié")
    return config, summary, checkpoint


def main() -> None:
    args = parse_args()
    expected_outputs = (
        "final_test_metrics.json",
        "final_test_report.md",
        "confusion_matrix_argmax.csv",
        "confusion_matrix_threshold_055.csv",
        "predictions_test.csv",
    )
    existing = [name for name in expected_outputs if (OUTPUT_DIR / name).exists()]
    if existing:
        raise FileExistsError(
            "Évaluation finale déjà présente ; aucun écrasement autorisé : " + ", ".join(existing)
        )

    config, training_summary, checkpoint = verify_frozen_configuration()
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_root = args.image_root.resolve()
    dataset = FinalTestDataset(
        TEST_MANIFEST,
        image_root,
        build_transforms(config["input_size"], train=False),
    )
    missing = [name for name in dataset.frame.filename if not (image_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images test sont absentes")
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=bool(config["pin_memory"] and device.type == "cuda"),
    )

    model = build_pretrained("resnet18", num_classes=4, use_weights=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    probabilities, targets, row_indices = [], [], []
    total_loss = 0.0
    seen = 0

    # Passe unique : les deux règles de décision réutilisent exactement ces probabilités.
    with torch.inference_mode():
        for images, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            logits = model(images)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Logits non finis pendant l'évaluation finale")
            loss = criterion(logits, labels_device)
            count = labels.size(0)
            total_loss += float(loss) * count
            seen += count
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            targets.append(labels.numpy())
            row_indices.append(indices.numpy())

    if seen != EXPECTED_TEST_SIZE:
        raise RuntimeError(f"Passe test incomplète : {seen} images")
    probabilities_array = np.concatenate(probabilities)
    y_true = np.concatenate(targets)
    indices_array = np.concatenate(row_indices)
    average_loss = total_loss / seen
    y_argmax = probabilities_array.argmax(axis=1)
    y_non_conforme = probabilities_array[:, 1:].argmax(axis=1) + 1
    y_threshold = np.where(
        probabilities_array[:, EXPECTED_MAPPING["Conforme"]] >= CONFORME_THRESHOLD,
        EXPECTED_MAPPING["Conforme"],
        y_non_conforme,
    )

    argmax_metrics, argmax_matrix = serializable_metrics(y_true, y_argmax, average_loss)
    threshold_metrics, threshold_matrix = serializable_metrics(y_true, y_threshold, average_loss)
    threshold_conforme = threshold_metrics["per_class"]["Conforme"]
    requirements = {
        "precision_conforme_target": PRECISION_TARGET,
        "recall_conforme_target": RECALL_TARGET,
        "precision_conforme_pass": threshold_conforme["precision"] >= PRECISION_TARGET,
        "recall_conforme_pass": threshold_conforme["recall"] >= RECALL_TARGET,
        "both_pass": (
            threshold_conforme["precision"] >= PRECISION_TARGET
            and threshold_conforme["recall"] >= RECALL_TARGET
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(argmax_matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        OUTPUT_DIR / "confusion_matrix_argmax.csv", index_label="true_label"
    )
    pd.DataFrame(threshold_matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        OUTPUT_DIR / "confusion_matrix_threshold_055.csv", index_label="true_label"
    )
    save_confusion_figure(argmax_matrix, OUTPUT_DIR / "confusion_matrix_argmax.png", False)
    save_confusion_figure(
        argmax_matrix, OUTPUT_DIR / "confusion_matrix_argmax_normalized.png", True
    )
    save_confusion_figure(
        threshold_matrix, OUTPUT_DIR / "confusion_matrix_threshold_055.png", False
    )
    save_confusion_figure(
        threshold_matrix, OUTPUT_DIR / "confusion_matrix_threshold_055_normalized.png", True
    )

    index_to_label = {value: key for key, value in EXPECTED_MAPPING.items()}
    ordered = dataset.frame.iloc[indices_array].reset_index(drop=True)
    predictions = pd.DataFrame({
        "filename": ordered.filename,
        "true_label": [index_to_label[value] for value in y_true],
        "predicted_argmax": [index_to_label[value] for value in y_argmax],
        "predicted_threshold_055": [index_to_label[value] for value in y_threshold],
        "probability_Conforme": probabilities_array[:, 0],
        "probability_NON_Conforme": probabilities_array[:, 1],
        "probability_PIETRA": probabilities_array[:, 2],
        "probability_Vide": probabilities_array[:, 3],
    })
    predictions.to_csv(OUTPUT_DIR / "predictions_test.csv", index=False)

    result = {
        "status": "completed",
        "evaluation_kind": "single_final_test_evaluation",
        "test_images": seen,
        "single_inference_pass": True,
        "device": str(device),
        "model": "resnet18",
        "checkpoint_epoch": EXPECTED_CHECKPOINT_EPOCH,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "training_run_id": EXPECTED_RUN_ID,
        "frozen_conforme_threshold": CONFORME_THRESHOLD,
        "class_mapping": EXPECTED_MAPPING,
        "argmax": argmax_metrics,
        "threshold_055": threshold_metrics,
        "business_requirements": requirements,
        "anti_leakage": {
            "test_used_for_architecture_selection": False,
            "test_used_for_hyperparameter_tuning": False,
            "threshold_selected_on_validation_before_test": True,
            "model_or_threshold_adaptation_after_test": False,
            "train_manifest_loaded": False,
            "validation_manifest_loaded": False,
        },
        "training_best_validation": training_summary["best_validation"],
    }
    (OUTPUT_DIR / "final_test_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def metric_block(name: str, metrics: dict[str, object]) -> str:
        conforme = metrics["per_class"]["Conforme"]
        errors = metrics["business_errors"]
        return f"""## {name}

- Loss : {metrics['loss']:.6f}
- Accuracy : {metrics['accuracy']:.4%}
- Macro precision : {metrics['macro_precision']:.4%}
- Macro recall : {metrics['macro_recall']:.4%}
- Macro F1 : {metrics['macro_f1']:.4%}
- Precision Conforme : {conforme['precision']:.4%}
- Recall Conforme : {conforme['recall']:.4%}
- F1 Conforme : {conforme['f1']:.4%}
- NON Conforme → Conforme : {errors['false_conforme_non_conforme']}
- PIETRA → Conforme : {errors['false_conforme_pietra']}
- Faux Conforme total : {errors['false_conforme_total']}
- Conforme rejetées : {errors['conforme_rejected']}
"""

    report = f"""# Évaluation finale unique — test CastagNet

Modèle ResNet18, checkpoint epoch 13 du run MLflow `{EXPECTED_RUN_ID}`.
Le test compte {seen} images et a fait l'objet d'une seule passe d'inférence.

Le test n'a servi à aucune sélection d'architecture ni aucun réglage
d'hyperparamètre. Le seuil Conforme `{CONFORME_THRESHOLD:.2f}` a été déterminé
uniquement sur validation avant l'ouverture du test. Il n'est pas adapté selon
les résultats ci-dessous. Cette exécution constitue l'évaluation finale unique.

{metric_block('Classification standard — argmax', argmax_metrics)}

{metric_block('Configuration métier — seuil Conforme 0,55', threshold_metrics)}

## Contraintes métier sur la configuration figée

- Precision Conforme ≥ 95 % : {'PASS' if requirements['precision_conforme_pass'] else 'FAIL'}
- Recall Conforme ≥ 85 % : {'PASS' if requirements['recall_conforme_pass'] else 'FAIL'}
- Respect simultané : {'PASS' if requirements['both_pass'] else 'FAIL'}

Quelle que soit cette conclusion, aucun ajustement du modèle ou du seuil ne doit
être réalisé à partir du jeu test.
"""
    (OUTPUT_DIR / "final_test_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
