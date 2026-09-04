"""Smoke test, benchmark et entraînement de la baseline CastagNet, sans jeu test."""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import CastagNetDataset, EXPECTED_MAPPING, check_paths, reject_test_manifest
from metrics import (
    CLASS_NAMES,
    classification_metrics,
    normalized_confusion,
    save_confusion_figures,
    save_history_figures,
    subgroup_metrics,
)
from models.custom_cnn import CustomCNN, parameter_summary
from models.pretrained import (
    build_pretrained,
    classification_head,
    set_backbone_trainable,
    set_training_phase_mode,
)
from transforms import build_transforms
from utils.mlflow_utils import configure_mlflow
from utils.reproducibility import environment_summary, seed_everything


TRAINING_DIR = Path(__file__).resolve().parent
ROOT = TRAINING_DIR.parent
DEFAULT_CONFIG = TRAINING_DIR / "configs" / "custom_cnn_baseline.json"
DEFAULT_MAPPING = TRAINING_DIR / "configs" / "class_mapping.json"
DEFAULT_TRAIN = ROOT / "analysis" / "output" / "train_manifest.csv"
DEFAULT_VALIDATION = ROOT / "analysis" / "output" / "validation_manifest.csv"
DEFAULT_IMAGE_ROOT = ROOT / "images"
DEFAULT_OUTPUT = ROOT / "analysis" / "output" / "modeling" / "custom_cnn_baseline_v1"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_allowed_manifests(train_path: Path, validation_path: Path) -> None:
    for path in (train_path, validation_path):
        reject_test_manifest(path)
    if train_path.resolve() == validation_path.resolve():
        raise ValueError("Les manifestes train et validation doivent être distincts")
    if train_path.name != "train_manifest.csv" or validation_path.name != "validation_manifest.csv":
        raise ValueError("Cette étape accepte uniquement train_manifest.csv et validation_manifest.csv")


def make_loaders(
    config: dict,
    mapping: dict[str, int],
    train_path: Path,
    validation_path: Path,
    image_root: Path,
) -> tuple[CastagNetDataset, CastagNetDataset, DataLoader, DataLoader]:
    train_dataset = CastagNetDataset(
        train_path, image_root, mapping, build_transforms(config["input_size"], train=True)
    )
    validation_dataset = CastagNetDataset(
        validation_path, image_root, mapping, build_transforms(config["input_size"], train=False)
    )
    generator = torch.Generator().manual_seed(config["seed"])
    common = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": bool(config["pin_memory"] and torch.cuda.is_available()),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **common)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    return train_dataset, validation_dataset, train_loader, validation_loader


def to_device(batch, device: torch.device):
    images, labels, indices = batch
    return images.to(device, non_blocking=True), labels.to(device, non_blocking=True), indices


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_name: str | None = None,
    head_only: bool = False,
) -> tuple[float, dict[str, object]]:
    if head_only:
        if model_name is None:
            raise ValueError("model_name est requis pour entraîner uniquement la tête")
        set_training_phase_mode(model, model_name, head_only=True)
    else:
        model.train()
    total_loss = 0.0
    seen = 0
    truth, predictions = [], []
    for batch in loader:
        images, labels, _ = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Loss non finie pendant train")
        loss.backward()
        optimizer.step()
        count = labels.size(0)
        total_loss += float(loss.detach()) * count
        seen += count
        truth.append(labels.detach().cpu().numpy())
        predictions.append(logits.detach().argmax(dim=1).cpu().numpy())
    metrics = classification_metrics(np.concatenate(truth), np.concatenate(predictions))
    return total_loss / seen, metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    seen = 0
    truth, predictions, probabilities, indices = [], [], [], []
    for batch in loader:
        images, labels, batch_indices = to_device(batch, device)
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise RuntimeError("Valeur non finie pendant validation")
        probs = torch.softmax(logits, dim=1)
        count = labels.size(0)
        total_loss += float(loss) * count
        seen += count
        truth.append(labels.cpu().numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        probabilities.append(probs.cpu().numpy())
        indices.append(batch_indices.numpy())
    y_true = np.concatenate(truth)
    y_pred = np.concatenate(predictions)
    probs = np.concatenate(probabilities)
    row_indices = np.concatenate(indices)
    return total_loss / seen, classification_metrics(y_true, y_pred), y_true, y_pred, probs, row_indices


def smoke_and_benchmark(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: dict,
    device: torch.device,
    output_dir: Path,
) -> dict[str, object]:
    criterion = nn.CrossEntropyLoss()
    is_pretrained = bool(config.get("pretrained", False))
    model_name = config["model_name"] if is_pretrained else None
    phase_checks: dict[str, object] = {}
    if is_pretrained:
        set_backbone_trainable(model, model_name, trainable=False)
        head = classification_head(model, model_name)
        head_parameter_ids = {id(parameter) for parameter in head.parameters()}
        phase_checks = {
            "phase_1_backbone_frozen": all(
                not parameter.requires_grad
                for parameter in model.parameters()
                if id(parameter) not in head_parameter_ids
            ),
            "phase_1_head_trainable": all(parameter.requires_grad for parameter in head.parameters()),
            "phase_1_trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        }
        learning_rate = config["head_learning_rate"]
        set_training_phase_mode(model, model_name, head_only=True)
    else:
        learning_rate = config["learning_rate"]
        model.train()
    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=config["weight_decay"],
    )
    smoke_losses = []
    logits_shape = None
    smoke_batch_device = None
    for batch_index, batch in enumerate(train_loader):
        images, labels, _ = to_device(batch, device)
        smoke_batch_device = str(images.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        if logits.shape != (images.shape[0], 4):
            raise RuntimeError(f"Dimensions logits invalides : {tuple(logits.shape)}")
        loss = criterion(logits, labels)
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise RuntimeError("NaN/Inf pendant smoke test")
        loss.backward()
        optimizer.step()
        smoke_losses.append(float(loss.detach()))
        logits_shape = list(logits.shape)
        if batch_index >= 1:
            break
    # Validation déterministe : forward uniquement.
    model.eval()
    with torch.no_grad():
        validation_batch = next(iter(validation_loader))
        val_images, val_labels, _ = to_device(validation_batch, device)
        val_logits = model(val_images)
        val_loss = criterion(val_logits, val_labels)
    checkpoint = output_dir / "smoke_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "mapping": EXPECTED_MAPPING}, checkpoint)
    reloaded = copy.deepcopy(model).cpu()
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"])

    # Benchmark train (forward + backward + optimizer) après un batch de chauffe.
    if is_pretrained:
        set_backbone_trainable(model, model_name, trainable=True)
        phase_checks.update({
            "phase_2_all_parameters_trainable": all(
                parameter.requires_grad for parameter in model.parameters()
            ),
            "phase_2_trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        })
        optimizer = AdamW(
            model.parameters(),
            lr=config["finetune_learning_rate"],
            weight_decay=config["weight_decay"],
        )
        set_training_phase_mode(model, model_name, head_only=False)
    else:
        model.train()
    train_images = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for batch_index, batch in enumerate(train_loader):
        images, labels, _ = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        train_images += labels.size(0)
        if batch_index + 1 >= config["benchmark_train_batches"]:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start

    model.eval()
    validation_images = 0
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        for batch_index, batch in enumerate(validation_loader):
            images, labels, _ = to_device(batch, device)
            model(images)
            validation_images += labels.size(0)
            if batch_index + 1 >= config["benchmark_validation_batches"]:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    validation_seconds = time.perf_counter() - start
    train_ips = train_images / train_seconds
    validation_ips = validation_images / validation_seconds
    estimated_epoch_seconds = len(train_loader.dataset) / train_ips + len(validation_loader.dataset) / validation_ips
    estimated_epochs = (
        config["head_epochs"] + config["finetune_epochs"]
        if is_pretrained else config["epochs_max"]
    )
    estimated_total_seconds = estimated_epoch_seconds * estimated_epochs
    return {
        "status": "pass",
        "train_batches_smoke": 2,
        "validation_batches_smoke": 1,
        "logits_shape": logits_shape,
        "model_device": str(next(model.parameters()).device),
        "batch_device": smoke_batch_device,
        "train_smoke_losses": smoke_losses,
        "validation_smoke_loss": float(val_loss),
        "finite_values": True,
        "backward_ok": True,
        "optimizer_step_ok": True,
        "checkpoint_save_reload_ok": True,
        "checkpoint": str(checkpoint),
        "transfer_phase_checks": phase_checks,
        "benchmark": {
            "train_batches": config["benchmark_train_batches"],
            "train_images": train_images,
            "train_seconds": train_seconds,
            "train_mean_batch_seconds": train_seconds / config["benchmark_train_batches"],
            "train_images_per_second": train_ips,
            "validation_batches": config["benchmark_validation_batches"],
            "validation_images": validation_images,
            "validation_seconds": validation_seconds,
            "validation_mean_batch_seconds": validation_seconds / config["benchmark_validation_batches"],
            "validation_images_per_second": validation_ips,
            "estimated_epoch_seconds": estimated_epoch_seconds,
            "estimated_epochs": estimated_epochs,
            "estimated_total_seconds": estimated_total_seconds,
            "estimated_total_hours": estimated_total_seconds / 3600,
            "estimated_30_epochs_seconds": estimated_total_seconds,
            "estimated_30_epochs_hours": estimated_total_seconds / 3600,
            "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda" else None,
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device)
            if device.type == "cuda" else None,
        },
    }


def flatten_mlflow_params(config: dict, environment: dict, parameters: dict, paths: dict) -> dict:
    values = {
        **config,
        **{f"env_{key}": value for key, value in environment.items()},
        **parameters,
        **paths,
        "class_mapping": json.dumps(EXPECTED_MAPPING, ensure_ascii=False),
    }
    return {key: str(value)[:500] for key, value in values.items() if value is not None}


def metric_scalars(metrics: dict[str, object], prefix: str) -> dict[str, float]:
    conforme = metrics["per_class"]["Conforme"]
    return {
        f"{prefix}_accuracy": metrics["accuracy"],
        f"{prefix}_macro_precision": metrics["macro_precision"],
        f"{prefix}_macro_recall": metrics["macro_recall"],
        f"{prefix}_macro_f1": metrics["macro_f1"],
        f"{prefix}_precision_conforme": conforme["precision"],
        f"{prefix}_recall_conforme": conforme["recall"],
        f"{prefix}_f1_conforme": conforme["f1"],
    }


def run_full_training(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    validation_dataset: CastagNetDataset,
    config: dict,
    mapping: dict[str, int],
    device: torch.device,
    output_dir: Path,
    mlflow,
) -> dict[str, object]:
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=config["scheduler_factor"], patience=config["scheduler_patience"]
    )
    best_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows = []
    checkpoint_path = output_dir / "best_model.pt"
    training_start = time.perf_counter()
    for epoch in range(1, config["epochs_max"] + 1):
        epoch_start = time.perf_counter()
        # Conserver le LR réellement utilisé pendant cette epoch. Le scheduler
        # est mis à jour après la validation et ne s'applique qu'à l'epoch suivante.
        learning_rate_used = optimizer.param_groups[0]["lr"]
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics, *_ = evaluate(model, validation_loader, criterion, device)
        scheduler.step(val_loss)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_precision_conforme": val_metrics["per_class"]["Conforme"]["precision"],
            "val_recall_conforme": val_metrics["per_class"]["Conforme"]["recall"],
            "learning_rate": learning_rate_used,
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history_rows.append(row)
        mlflow.log_metrics({key: value for key, value in row.items() if key != "epoch"}, step=epoch)
        pd.DataFrame(history_rows).to_csv(output_dir / "training_history.csv", index=False)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "class_mapping": mapping,
                "config": config,
            }, checkpoint_path)
        else:
            epochs_without_improvement += 1
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f}"
        )
        if epochs_without_improvement >= config["early_stopping_patience"]:
            break
    training_seconds = time.perf_counter() - training_start
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, val_metrics, y_true, y_pred, probabilities, row_indices = evaluate(
        model, validation_loader, criterion, device
    )
    history = pd.DataFrame(history_rows)
    history.to_csv(output_dir / "training_history.csv", index=False)
    figures_dir = output_dir / "figures"
    save_history_figures(history, figures_dir)
    matrix = val_metrics.pop("confusion_matrix")
    normalized = normalized_confusion(matrix)
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output_dir / "confusion_matrix.csv")
    pd.DataFrame(normalized, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix_normalized.csv"
    )
    save_confusion_figures(matrix, figures_dir)
    ordered_frame = validation_dataset.frame.iloc[row_indices].reset_index(drop=True)
    index_to_label = {value: key for key, value in mapping.items()}
    errors = ordered_frame.copy()
    errors["predicted_label"] = [index_to_label[value] for value in y_pred]
    errors["predicted_confidence"] = probabilities.max(axis=1)
    errors = errors[y_true != y_pred]
    error_columns = [
        "filename", "effective_label", "predicted_label", "predicted_confidence", "year",
        "cam_position", "cam_num", "chunk", "multiple", "historical_pair_id",
    ]
    errors[error_columns].to_csv(output_dir / "validation_errors.csv", index=False)
    subgroup = subgroup_metrics(ordered_frame, y_true, y_pred)
    subgroup.to_csv(output_dir / "subgroup_metrics.csv", index=False)
    final_train = history.iloc[-1]
    best_epoch_train = history.loc[history.epoch == best_epoch].iloc[0]
    # Comparer train et validation à la même epoch. Après rechargement du meilleur
    # checkpoint, val_metrics correspond à best_epoch et non à la dernière epoch.
    overfitting_gap = float(best_epoch_train.train_accuracy - val_metrics["accuracy"])
    diagnosis = (
        "overfitting_probable" if overfitting_gap > 0.08 else
        "underfitting_possible" if best_epoch_train.train_accuracy < 0.70 else
        "no_strong_divergence_detected"
    )
    summary = {
        "status": "completed",
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "training_time_seconds": training_seconds,
        "best_val_loss": val_loss,
        "final_train": {key: float(final_train[key]) for key in (
            "train_loss", "train_accuracy", "train_macro_f1"
        )},
        "best_epoch_train": {key: float(best_epoch_train[key]) for key in (
            "train_loss", "train_accuracy", "train_macro_f1"
        )},
        "best_validation": val_metrics,
        "business_errors": val_metrics["business_errors"],
        "overfitting_accuracy_gap": overfitting_gap,
        "diagnosis": diagnosis,
    }
    write_json(output_dir / "metrics_summary.json", summary)
    mlflow.log_metrics({
        "best_epoch": best_epoch,
        "best_val_loss": val_loss,
        "best_val_accuracy": val_metrics["accuracy"],
        "best_val_macro_f1": val_metrics["macro_f1"],
        "best_val_precision_conforme": val_metrics["per_class"]["Conforme"]["precision"],
        "best_val_recall_conforme": val_metrics["per_class"]["Conforme"]["recall"],
        "training_time_seconds": training_seconds,
        **{key: float(value) for key, value in val_metrics["business_errors"].items()},
    })
    return summary


def run_transfer_training(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    validation_dataset: CastagNetDataset,
    config: dict,
    mapping: dict[str, int],
    device: torch.device,
    output_dir: Path,
    mlflow,
) -> dict[str, object]:
    """Deux phases dans un même run : tête gelée, puis fine-tuning intégral."""
    criterion = nn.CrossEntropyLoss()
    model_name = config["model_name"]
    checkpoint_path = output_dir / "best_model.pt"
    history_rows: list[dict[str, object]] = []
    best_loss = math.inf
    best_epoch = 0
    best_phase = ""
    global_epoch = 0
    training_start = time.perf_counter()
    phases = (
        ("head", config["head_epochs"], config["head_learning_rate"], True),
        ("finetune", config["finetune_epochs"], config["finetune_learning_rate"], False),
    )

    for phase_index, (phase, epochs, learning_rate, head_only) in enumerate(phases, start=1):
        set_backbone_trainable(model, model_name, trainable=not head_only)
        optimizer = AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=learning_rate,
            weight_decay=config["weight_decay"],
        )
        scheduler = None if head_only else ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["scheduler_factor"],
            patience=config["scheduler_patience"],
        )
        phase_best_loss = math.inf
        phase_best_path = output_dir / f"best_{phase}_model.pt"
        epochs_without_improvement = 0

        for phase_epoch in range(1, epochs + 1):
            global_epoch += 1
            epoch_start = time.perf_counter()
            learning_rate_used = optimizer.param_groups[0]["lr"]
            train_loss, train_metrics = train_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                model_name=model_name,
                head_only=head_only,
            )
            val_loss, val_metrics, *_ = evaluate(model, validation_loader, criterion, device)
            if scheduler is not None:
                scheduler.step(val_loss)
            row = {
                "epoch": global_epoch,
                "phase": phase,
                "phase_index": phase_index,
                "phase_epoch": phase_epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_precision_conforme": val_metrics["per_class"]["Conforme"]["precision"],
                "val_recall_conforme": val_metrics["per_class"]["Conforme"]["recall"],
                "learning_rate": learning_rate_used,
                "epoch_seconds": time.perf_counter() - epoch_start,
            }
            history_rows.append(row)
            mlflow.log_metrics(
                {key: value for key, value in row.items() if key not in {"epoch", "phase"}},
                step=global_epoch,
            )
            pd.DataFrame(history_rows).to_csv(output_dir / "training_history.csv", index=False)

            checkpoint = {
                "epoch": global_epoch,
                "phase": phase,
                "phase_epoch": phase_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "class_mapping": mapping,
                "config": config,
            }
            if val_loss < phase_best_loss:
                phase_best_loss = val_loss
                epochs_without_improvement = 0
                torch.save(checkpoint, phase_best_path)
            else:
                epochs_without_improvement += 1
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = global_epoch
                best_phase = phase
                torch.save(checkpoint, checkpoint_path)

            print(
                f"epoch={global_epoch:02d} phase={phase} phase_epoch={phase_epoch:02d} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
            )
            if not head_only and epochs_without_improvement >= config["early_stopping_patience"]:
                break

        # Le fine-tuning part de la meilleure tête parmi les trois epochs.
        if head_only:
            phase_checkpoint = torch.load(phase_best_path, map_location=device, weights_only=False)
            model.load_state_dict(phase_checkpoint["model_state_dict"])

    training_seconds = time.perf_counter() - training_start
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, val_metrics, y_true, y_pred, probabilities, row_indices = evaluate(
        model, validation_loader, criterion, device
    )
    history = pd.DataFrame(history_rows)
    history.to_csv(output_dir / "training_history.csv", index=False)
    figures_dir = output_dir / "figures"
    save_history_figures(history, figures_dir)
    matrix = val_metrics.pop("confusion_matrix")
    normalized = normalized_confusion(matrix)
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(output_dir / "confusion_matrix.csv")
    pd.DataFrame(normalized, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix_normalized.csv"
    )
    save_confusion_figures(matrix, figures_dir)
    ordered_frame = validation_dataset.frame.iloc[row_indices].reset_index(drop=True)
    index_to_label = {value: key for key, value in mapping.items()}
    errors = ordered_frame.copy()
    errors["predicted_label"] = [index_to_label[value] for value in y_pred]
    errors["predicted_confidence"] = probabilities.max(axis=1)
    errors = errors[y_true != y_pred]
    error_columns = [
        "filename", "effective_label", "predicted_label", "predicted_confidence", "year",
        "cam_position", "cam_num", "chunk", "multiple", "historical_pair_id",
    ]
    errors[error_columns].to_csv(output_dir / "validation_errors.csv", index=False)
    subgroup_metrics(ordered_frame, y_true, y_pred).to_csv(
        output_dir / "subgroup_metrics.csv", index=False
    )
    final_train = history.iloc[-1]
    best_epoch_train = history.loc[history.epoch == best_epoch].iloc[0]
    overfitting_gap = float(best_epoch_train.train_accuracy - val_metrics["accuracy"])
    diagnosis = (
        "overfitting_probable" if overfitting_gap > 0.08 else
        "underfitting_possible" if best_epoch_train.train_accuracy < 0.70 else
        "no_strong_divergence_detected"
    )
    summary = {
        "status": "completed",
        "protocol": "transfer_learning_head_then_finetune",
        "epochs_completed": len(history),
        "head_epochs_completed": int((history.phase == "head").sum()),
        "finetune_epochs_completed": int((history.phase == "finetune").sum()),
        "best_epoch": best_epoch,
        "best_phase": best_phase,
        "training_time_seconds": training_seconds,
        "best_val_loss": val_loss,
        "final_train": {key: float(final_train[key]) for key in (
            "train_loss", "train_accuracy", "train_macro_f1"
        )},
        "best_epoch_train": {key: float(best_epoch_train[key]) for key in (
            "train_loss", "train_accuracy", "train_macro_f1"
        )},
        "best_validation": val_metrics,
        "business_errors": val_metrics["business_errors"],
        "overfitting_accuracy_gap": overfitting_gap,
        "diagnosis": diagnosis,
    }
    write_json(output_dir / "metrics_summary.json", summary)
    mlflow.log_metrics({
        "best_epoch": best_epoch,
        "best_val_loss": val_loss,
        "best_val_accuracy": val_metrics["accuracy"],
        "best_val_macro_f1": val_metrics["macro_f1"],
        "best_val_precision_conforme": val_metrics["per_class"]["Conforme"]["precision"],
        "best_val_recall_conforme": val_metrics["per_class"]["Conforme"]["recall"],
        "training_time_seconds": training_seconds,
        **{key: float(value) for key, value in val_metrics["business_errors"].items()},
    })
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--train", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--class-mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-slow-training", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path = args.train_manifest.resolve()
    validation_path = args.validation_manifest.resolve()
    ensure_allowed_manifests(train_path, validation_path)
    config = load_json(args.config.resolve())
    mapping = load_json(args.class_mapping.resolve())
    if mapping != EXPECTED_MAPPING:
        raise ValueError("class_mapping.json ne correspond pas au mapping figé")
    seed_everything(config["seed"])
    environment = environment_summary()
    device = torch.device(environment["device"])
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    image_root = args.image_root.resolve()
    path_audit = {
        "train": check_paths(train_path, image_root),
        "validation": check_paths(validation_path, image_root),
    }
    if any(values["missing"] for values in path_audit.values()):
        raise RuntimeError(f"Images manquantes : {path_audit}")
    train_dataset, validation_dataset, train_loader, validation_loader = make_loaders(
        config, mapping, train_path, validation_path, image_root
    )
    if config.get("pretrained", False):
        model = build_pretrained(
            config["model_name"],
            num_classes=config["num_classes"],
            use_weights=config.get("weights") == "DEFAULT",
        ).to(device)
    else:
        model = CustomCNN(num_classes=4, dropout=config["dropout"]).to(device)
    parameters = parameter_summary(model)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "class_mapping.json", mapping)
    write_json(output_dir / "environment.json", environment)
    write_json(output_dir / "image_path_audit.json", path_audit)
    mlflow, tracking_dir = configure_mlflow(ROOT, config["experiment_name"])
    paths = {
        "train_manifest": str(train_path.relative_to(ROOT)),
        "validation_manifest": str(validation_path.relative_to(ROOT)),
        "image_root": str(image_root),
        "train_size": len(train_dataset),
        "validation_size": len(validation_dataset),
        "test_loaded": False,
    }
    run_name = config["run_name"] + ("_smoke" if args.smoke_test else "")
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(flatten_mlflow_params(config, environment, parameters, paths))
        if args.smoke_test:
            summary = smoke_and_benchmark(
                model, train_loader, validation_loader, config, device, output_dir
            )
            summary.update({
                "run_id": run.info.run_id,
                "mlflow_tracking_dir": str(tracking_dir),
                "device": str(device),
                "model_parameters": parameters,
                "image_path_audit": path_audit,
            })
            write_json(output_dir / "smoke_test_summary.json", summary)
            mlflow.log_metrics({
                "smoke_train_loss_last": summary["train_smoke_losses"][-1],
                "smoke_validation_loss": summary["validation_smoke_loss"],
                "benchmark_train_images_per_second": summary["benchmark"]["train_images_per_second"],
                "benchmark_validation_images_per_second": summary["benchmark"]["validation_images_per_second"],
                "estimated_epoch_seconds": summary["benchmark"]["estimated_epoch_seconds"],
                "estimated_30_epochs_hours": summary["benchmark"]["estimated_30_epochs_hours"],
            })
            mlflow.log_artifacts(str(output_dir), artifact_path="smoke")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        else:
            smoke_path = output_dir / "smoke_test_summary.json"
            if not smoke_path.exists():
                raise RuntimeError("Smoke test requis avant l'entraînement complet")
            smoke = load_json(smoke_path)
            estimate = smoke["benchmark"].get(
                "estimated_total_hours", smoke["benchmark"]["estimated_30_epochs_hours"]
            )
            limit = config["max_estimated_training_hours"]
            if estimate > limit and not args.allow_slow_training:
                raise RuntimeError(
                    f"Entraînement refusé : estimation {estimate:.2f} h > limite {limit:.2f} h"
                )
            training_function = (
                run_transfer_training if config.get("pretrained", False) else run_full_training
            )
            summary = training_function(
                model, train_loader, validation_loader, validation_dataset, config, mapping,
                device, output_dir, mlflow,
            )
            summary["run_id"] = run.info.run_id
            write_json(output_dir / "metrics_summary.json", summary)
            mlflow.log_artifacts(str(output_dir), artifact_path="baseline")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        write_json(output_dir / "latest_run.json", {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "experiment": config["experiment_name"],
            "test_loaded": False,
        })


if __name__ == "__main__":
    main()
