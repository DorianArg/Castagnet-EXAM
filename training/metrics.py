"""Métriques de classification et exports d'analyse validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


CLASS_NAMES = ["Conforme", "NON Conforme", "PIETRA", "Vide"]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(4), zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(4), average="macro", zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(4))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            name: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, name in enumerate(CLASS_NAMES)
        },
        "business_errors": {
            "false_conforme_non_conforme": int(matrix[1, 0]),
            "false_conforme_pietra": int(matrix[2, 0]),
            "false_conforme_total": int(matrix[1, 0] + matrix[2, 0]),
            "conforme_rejected": int(matrix[0, :].sum() - matrix[0, 0]),
        },
        "confusion_matrix": matrix,
    }


def normalized_confusion(matrix: np.ndarray) -> np.ndarray:
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0)


def save_confusion_figures(matrix: np.ndarray, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for values, suffix, fmt in (
        (matrix, "raw", "d"), (normalized_confusion(matrix), "normalized", ".2f")
    ):
        fig, ax = plt.subplots(figsize=(6.5, 5.6))
        image = ax.imshow(values, cmap="Blues", vmin=0)
        for row in range(4):
            for col in range(4):
                value = format(values[row, col], fmt)
                ax.text(col, row, value, ha="center", va="center")
        ax.set_xticks(range(4), CLASS_NAMES, rotation=25, ha="right")
        ax.set_yticks(range(4), CLASS_NAMES)
        ax.set_xlabel("Classe prédite")
        ax.set_ylabel("Classe réelle")
        ax.set_title(f"Matrice de confusion validation — {suffix}")
        fig.colorbar(image, ax=ax, fraction=.046)
        fig.tight_layout()
        path = output_dir / f"confusion_matrix_{suffix}.png"
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs


def save_history_figures(history: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specifications = [
        (["train_loss", "val_loss"], "Loss", "loss.png"),
        (["train_accuracy", "val_accuracy"], "Accuracy", "accuracy.png"),
        (["val_precision_conforme", "val_recall_conforme"], "Conforme — validation", "conforme_precision_recall.png"),
    ]
    outputs = []
    for columns, title, filename in specifications:
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for column in columns:
            ax.plot(history.epoch, history[column], marker="o", label=column)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        outputs.append(path)
    return outputs


def subgroup_metrics(frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    annotated = frame.reset_index(drop=True).copy()
    annotated["_y_true"] = y_true
    annotated["_y_pred"] = y_pred
    definitions = {
        "year": ["2025", "2026"],
        "cam_position": ["T", "B"],
        "cam_num": ["1", "2", "3", "4", "5", "6"],
        "chunk": [False, True],
        "multiple": [False, True],
    }
    rows = []
    for dimension, categories in definitions.items():
        series = annotated[dimension]
        if dimension in {"year", "cam_num"}:
            series = series.astype(str)
        elif dimension in {"chunk", "multiple"} and series.dtype == object:
            series = series.astype(str).str.lower().isin({"true", "1", "yes"})
        for category in categories:
            subset = annotated[series == category]
            if subset.empty:
                continue
            metrics = classification_metrics(subset._y_true.to_numpy(), subset._y_pred.to_numpy())
            rows.append({
                "dimension": dimension,
                "group": str(category),
                "n_images": len(subset),
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "precision_conforme": metrics["per_class"]["Conforme"]["precision"],
                "recall_conforme": metrics["per_class"]["Conforme"]["recall"],
            })
    return pd.DataFrame(rows)
