"""Benchmark reproductible ONNX Runtime, validation uniquement et batch 1."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import psutil
import torch
from PIL import Image

from transforms import build_transforms


TRAINING_DIR = Path(__file__).resolve().parent
ROOT = TRAINING_DIR.parent
VALIDATION_MANIFEST = ROOT / "analysis" / "output" / "validation_manifest.csv"
DEFAULT_IMAGE_ROOT = ROOT / "images"
MODEL_DIRECTORIES = {
    "resnet18": ROOT / "analysis" / "output" / "modeling" / "resnet18",
    "mobilenet_v3_small": ROOT / "analysis" / "output" / "modeling" / "mobilenet_v3_small",
    "efficientnet_b0": ROOT / "analysis" / "output" / "modeling" / "efficientnet_b0",
}
ONNX_FILENAMES = {
    "resnet18": "resnet18_castagnet.onnx",
    "mobilenet_v3_small": "mobilenet_v3_small_castagnet.onnx",
    "efficientnet_b0": "efficientnet_b0_castagnet.onnx",
}
EXPECTED_VALIDATION_SIZE = 5289
EXPECTED_INPUT_SHAPE = [1, 3, 224, 224]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_DIRECTORIES, default="resnet18")
    parser.add_argument("--provider", choices=("cpu", "cuda", "all"), default="all")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values), quantile))


def latency_summary(seconds: list[float]) -> dict[str, float]:
    total = sum(seconds)
    return {
        "iterations": len(seconds),
        "mean_ms": statistics.fmean(seconds) * 1000,
        "median_ms": statistics.median(seconds) * 1000,
        "p90_ms": percentile(seconds, 90) * 1000,
        "p95_ms": percentile(seconds, 95) * 1000,
        "p99_ms": percentile(seconds, 99) * 1000,
        "min_ms": min(seconds) * 1000,
        "max_ms": max(seconds) * 1000,
        "images_per_second": len(seconds) / total,
        "total_measured_seconds": total,
    }


def gpu_information() -> dict[str, object] | None:
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
        name, memory, driver = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        return {"name": name, "memory_total_mib": float(memory), "driver_version": driver}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def process_gpu_memory_mib() -> float | None:
    try:
        command = [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
        for line in result.stdout.splitlines():
            pid, memory = [part.strip() for part in line.split(",")]
            if int(pid) == os.getpid():
                return float(memory)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return None


def requested_providers(mode: str, available: list[str]) -> list[tuple[str, str]]:
    providers = []
    if mode in {"cpu", "all"}:
        if "CPUExecutionProvider" not in available:
            raise RuntimeError("CPUExecutionProvider indisponible")
        providers.append(("cpu", "CPUExecutionProvider"))
    if mode in {"cuda", "all"}:
        if "CUDAExecutionProvider" not in available:
            if mode == "cuda":
                raise RuntimeError("CUDAExecutionProvider indisponible")
        else:
            providers.append(("cuda", "CUDAExecutionProvider"))
    return providers


def load_image_tensor(path: Path, transform) -> np.ndarray:
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB"))
    return tensor.unsqueeze(0).numpy()


def benchmark_provider(
    provider_label: str,
    provider_name: str,
    model_path: Path,
    image_paths: list[Path],
    transform,
    iterations: int,
    warmup: int,
) -> dict[str, object]:
    process = psutil.Process()
    ram_before_session = process.memory_info().rss
    session_start = time.perf_counter()
    session = ort.InferenceSession(str(model_path), providers=[provider_name])
    session_load_seconds = time.perf_counter() - session_start
    active = session.get_providers()
    if not active or active[0] != provider_name:
        raise RuntimeError(f"Provider demandé non actif : demandé={provider_name}, actifs={active}")
    input_metadata = session.get_inputs()[0]
    output_metadata = session.get_outputs()[0]
    if list(input_metadata.shape) != EXPECTED_INPUT_SHAPE:
        raise RuntimeError(f"Shape d'entrée inattendue : {input_metadata.shape}")
    prepared = [load_image_tensor(path, transform) for path in image_paths]
    for index in range(warmup):
        session.run([output_metadata.name], {input_metadata.name: prepared[index % len(prepared)]})

    ram_peak = process.memory_info().rss
    inference_times = []
    for index in range(iterations):
        array = prepared[index % len(prepared)]
        start = time.perf_counter()
        session.run([output_metadata.name], {input_metadata.name: array})
        inference_times.append(time.perf_counter() - start)
        ram_peak = max(ram_peak, process.memory_info().rss)

    preprocessing_times = []
    for index in range(iterations):
        start = time.perf_counter()
        load_image_tensor(image_paths[index % len(image_paths)], transform)
        preprocessing_times.append(time.perf_counter() - start)
        ram_peak = max(ram_peak, process.memory_info().rss)

    total_times = []
    for index in range(iterations):
        start = time.perf_counter()
        array = load_image_tensor(image_paths[index % len(image_paths)], transform)
        session.run([output_metadata.name], {input_metadata.name: array})
        total_times.append(time.perf_counter() - start)
        ram_peak = max(ram_peak, process.memory_info().rss)

    ram_after = process.memory_info().rss
    return {
        "provider_requested": provider_name,
        "providers_active": active,
        "session_load_seconds_excluded_from_latencies": session_load_seconds,
        "warmup_iterations": warmup,
        "batch_size": 1,
        "preprocessing_includes_image_decode": True,
        "preprocessing": latency_summary(preprocessing_times),
        "inference_only": latency_summary(inference_times),
        "preprocessing_plus_inference": latency_summary(total_times),
        "memory": {
            "process_ram_before_session_bytes": ram_before_session,
            "process_ram_after_bytes": ram_after,
            "process_ram_peak_observed_bytes": ram_peak,
            "process_ram_peak_observed_mib": ram_peak / (1024 ** 2),
            "gpu_process_memory_observed_mib": (
                process_gpu_memory_mib() if provider_label == "cuda" else None
            ),
            "gpu_memory_measurement_note": (
                "Valeur nvidia-smi du processus si accessible ; ce n'est pas la mémoire d'une GTX 1060."
                if provider_label == "cuda"
                else "Non applicable au provider CPU."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    iterations = 5 if args.smoke_test else args.iterations
    warmup = 2 if args.smoke_test else args.warmup
    if not args.smoke_test and iterations < 200:
        raise ValueError("Le benchmark complet exige au moins 200 inférences")
    if warmup < 1 or iterations < 1:
        raise ValueError("Les nombres d'itérations et de warm-up doivent être positifs")
    model_dir = MODEL_DIRECTORIES[args.model]
    onnx_dir = model_dir / "onnx"
    model_path = onnx_dir / ONNX_FILENAMES[args.model]
    export_summary_path = onnx_dir / "export_summary.json"
    if not model_path.is_file() or not export_summary_path.is_file():
        raise FileNotFoundError(f"Export ONNX absent pour {args.model} : {model_path}")
    export_summary = json.loads(export_summary_path.read_text(encoding="utf-8"))
    if export_summary.get("status") != "valid":
        raise RuntimeError("L'export ONNX n'est pas marqué valide")
    if export_summary.get("business_threshold_in_graph") is not False:
        raise RuntimeError("Le seuil métier ne doit pas être intégré au graphe")

    validation = pd.read_csv(VALIDATION_MANIFEST, usecols=["filename"])
    if len(validation) != EXPECTED_VALIDATION_SIZE or not validation.filename.is_unique:
        raise RuntimeError("Le manifeste validation figé n'est plus conforme")
    image_root = args.image_root.resolve()
    sample_count = min(64, len(validation))
    sample_indices = np.linspace(0, len(validation) - 1, sample_count, dtype=int)
    image_paths = [(image_root / validation.iloc[index].filename) for index in sample_indices]
    if not all(path.is_file() for path in image_paths):
        raise FileNotFoundError("Une ou plusieurs images du pool de benchmark sont absentes")
    transform = build_transforms(224, train=False)

    # Importer torch avant la création d'une session CUDA rend ses DLL CUDA/cuDNN
    # disponibles à ONNX Runtime dans l'environnement Windows courant.
    _ = torch.cuda.is_available()
    available = ort.get_available_providers()
    selected = requested_providers(args.provider, available)
    results = {}
    failures = {}
    for label, provider in selected:
        try:
            results[label] = benchmark_provider(
                label, provider, model_path, image_paths, transform, iterations, warmup
            )
        except Exception as exc:
            failures[label] = f"{type(exc).__name__}: {exc}"
            if args.provider != "all":
                raise
    if not results:
        raise RuntimeError(f"Aucun provider benchmarké : {failures}")
    output = {
        "status": "smoke_pass" if args.smoke_test else "completed",
        "created_at": datetime.now().astimezone().isoformat(),
        "model": args.model,
        "onnx_path": str(model_path.resolve()),
        "onnx_sha256": export_summary["onnx_sha256"],
        "validation_only": True,
        "test_manifest_loaded": False,
        "batch_size": 1,
        "input_shape": EXPECTED_INPUT_SHAPE,
        "iterations": iterations,
        "warmup": warmup,
        "available_providers": available,
        "provider_failures": failures,
        "machine_measured": {
            "platform": platform.platform(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "logical_cpu_count": os.cpu_count(),
            "gpu": gpu_information(),
            "target_hardware_is_current_machine": False,
            "target_hardware": "NVIDIA GTX 1060 3 Go, ancien Intel i7, 12 flux",
            "no_numeric_extrapolation_to_target": True,
        },
        "results": results,
    }
    output_path = onnx_dir / ("benchmark_smoke.json" if args.smoke_test else "benchmark_results.json")
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
