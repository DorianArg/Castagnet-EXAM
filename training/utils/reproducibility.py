"""Configuration déterministe et inventaire de l'environnement."""

from __future__ import annotations

import os
import platform
import random

import numpy as np
import torch
import torchvision


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def environment_summary() -> dict[str, object]:
    cuda = torch.cuda.is_available()
    result: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "cpu_threads": os.cpu_count(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": "cuda" if cuda else "cpu",
    }
    if cuda:
        props = torch.cuda.get_device_properties(0)
        result.update({"gpu_name": props.name, "gpu_vram_bytes": props.total_memory})
    else:
        result.update({"gpu_name": None, "gpu_vram_bytes": None})
    return result
