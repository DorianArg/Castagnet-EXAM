"""Dataset basé exclusivement sur effective_label."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


EXPECTED_MAPPING = {"Conforme": 0, "NON Conforme": 1, "PIETRA": 2, "Vide": 3}


def reject_test_manifest(path: Path) -> None:
    if "test_manifest" in path.name.lower():
        raise ValueError("Le manifeste test est interdit pendant cette étape")


class CastagNetDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        image_root: Path,
        class_mapping: dict[str, int],
        transform: Callable | None = None,
    ) -> None:
        manifest_path = manifest_path.resolve()
        reject_test_manifest(manifest_path)
        if class_mapping != EXPECTED_MAPPING:
            raise ValueError(f"Mapping de classes inattendu : {class_mapping}")
        self.manifest_path = manifest_path
        self.image_root = image_root.resolve()
        self.class_mapping = class_mapping
        self.transform = transform
        self.frame = pd.read_csv(manifest_path, low_memory=False)
        required = {"filename", "effective_label"}
        if not required.issubset(self.frame.columns):
            raise ValueError(f"Colonnes manquantes : {required - set(self.frame.columns)}")
        if not self.frame.filename.is_unique:
            raise ValueError("filename doit être unique dans un manifeste")
        if not self.frame.effective_label.isin(class_mapping).all():
            raise ValueError("effective_label contient une valeur inconnue")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        row = self.frame.iloc[index]
        path = self.image_root / row.filename
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        label = self.class_mapping[row.effective_label]
        return image, label, index


def check_paths(manifest_path: Path, image_root: Path) -> dict[str, int]:
    reject_test_manifest(manifest_path)
    frame = pd.read_csv(manifest_path, usecols=["filename"])
    found = sum((image_root / filename).is_file() for filename in frame.filename)
    return {"expected": len(frame), "found": found, "missing": len(frame) - found}
