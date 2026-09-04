"""Consolidate frozen validation quality and ONNX benchmarks without ML inference."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODELING = ROOT / "analysis" / "output" / "modeling"
OUTPUT_DIR = MODELING / "onnx_comparison"
MODELS = ("resnet18", "efficientnet_b0", "mobilenet_v3_small")
QUALITY = {
    "resnet18": {
        "accuracy": 0.8815, "macro_f1": 0.8521,
        "precision_conforme": 0.8836, "recall_conforme": 0.8782,
        "parameters": 11178564,
    },
    "efficientnet_b0": {
        "accuracy": 0.8661, "macro_f1": 0.8360,
        "precision_conforme": 0.8778, "recall_conforme": 0.8357,
        "parameters": 4012672,
    },
    "mobilenet_v3_small": {
        "accuracy": 0.8421, "macro_f1": 0.8049,
        "precision_conforme": 0.8731, "recall_conforme": 0.8043,
        "parameters": 1521956,
    },
}
CAMERAS = 12
FPS_PER_CAMERA = (1, 2, 5, 10, 25)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    detail = {}
    machine = None
    for model in MODELS:
        onnx_dir = MODELING / model / "onnx"
        export = load_json(onnx_dir / "export_summary.json")
        benchmark = load_json(onnx_dir / "benchmark_results.json")
        if export.get("status") != "valid" or benchmark.get("status") != "completed":
            raise RuntimeError(f"Artefacts ONNX non valides pour {model}")
        if benchmark.get("test_manifest_loaded") is not False:
            raise RuntimeError("Le benchmark ne doit jamais charger le test")
        if machine is None:
            machine = benchmark["machine_measured"]
        quality = QUALITY[model]
        if export["pytorch_parameters"] != quality["parameters"]:
            raise RuntimeError(f"Nombre de parametres inattendu pour {model}")
        row = {
            "model": model,
            **quality,
            "onnx_size_bytes": export["onnx_size_bytes"],
            "onnx_size_mib": export["onnx_size_mib"],
            "validation_max_abs_difference_logits": export["numerical_validation"]["max_absolute_difference_logits"],
            "validation_mean_abs_difference_logits": export["numerical_validation"]["mean_absolute_difference_logits"],
            "validation_max_abs_difference_probabilities": export["numerical_validation"]["max_absolute_difference_probabilities"],
            "validation_argmax_differences": export["numerical_validation"]["argmax_prediction_differences"],
            "validation_decision_differences": export["numerical_validation"]["threshold_055_decision_differences"],
        }
        for provider in ("cpu", "cuda"):
            result = benchmark["results"][provider]
            for scope, prefix in (("inference_only", "inference"), ("preprocessing_plus_inference", "end_to_end")):
                values = result[scope]
                for metric in ("mean_ms", "median_ms", "p90_ms", "p95_ms", "p99_ms", "min_ms", "max_ms", "images_per_second"):
                    row[f"{provider}_{prefix}_{metric}"] = values[metric]
            row[f"{provider}_preprocessing_mean_ms"] = result["preprocessing"]["mean_ms"]
            row[f"{provider}_process_ram_peak_mib"] = result["memory"]["process_ram_peak_observed_mib"]
            row[f"{provider}_gpu_process_memory_mib"] = result["memory"]["gpu_process_memory_observed_mib"]
        capacity = []
        for fps in FPS_PER_CAMERA:
            required = CAMERAS * fps
            for provider in ("cpu", "cuda"):
                for scope, key in (("inference_only", "inference_only"), ("end_to_end", "preprocessing_plus_inference")):
                    throughput = benchmark["results"][provider][key]["images_per_second"]
                    capacity.append({
                        "fps_per_camera": fps,
                        "required_images_per_second": required,
                        "provider": provider,
                        "scope": scope,
                        "measured_images_per_second": throughput,
                        "utilization_percent": required / throughput * 100,
                        "theoretical_capacity_met": throughput >= required,
                    })
        rows.append(row)
        detail[model] = {
            "quality_validation_frozen": quality,
            "export": export,
            "benchmark": benchmark,
            "theoretical_12_stream_capacity": capacity,
        }

    csv_path = OUTPUT_DIR / "onnx_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    output = {
        "status": "completed",
        "created_at": datetime.now().astimezone().isoformat(),
        "validation_metrics_reused_without_recalculation": True,
        "test_manifest_loaded": False,
        "training_performed": False,
        "benchmark_protocol": {"batch_size": 1, "input_shape": [1, 3, 224, 224], "iterations": 500, "warmup": 50},
        "machine_measured": machine,
        "models": detail,
        "interpretation": {
            "gpu_current_machine": "ResNet18 is both the best validation model and the fastest measured CUDA model.",
            "cpu_current_machine": "MobileNetV3 Small has the lowest CPU latency and smallest ONNX file, with lower validation quality.",
            "target_limit": "No numerical extrapolation to the GTX 1060 3 GB / older i7 target is valid.",
        },
    }
    (OUTPUT_DIR / "onnx_comparison.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Comparaison ONNX CastagNet",
        "",
        "Les metriques de qualite proviennent des resultats validation figes du paragraphe 4.2; elles n'ont pas ete recalculees. Aucun manifeste test n'a ete charge et aucun entrainement n'a ete execute.",
        "",
        "## Qualite, taille et cout mesure",
        "",
        "| Modele | Accuracy | Macro F1 | Precision Conforme | Recall Conforme | Parametres | ONNX (Mio) | CPU inf. (ms/img/s) | CUDA inf. (ms/img/s) | CPU E2E (ms/img/s) | CUDA E2E (ms/img/s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['accuracy']*100:.2f}% | {row['macro_f1']*100:.2f}% | "
            f"{row['precision_conforme']*100:.2f}% | {row['recall_conforme']*100:.2f}% | "
            f"{row['parameters']:,} | {row['onnx_size_mib']:.2f} | "
            f"{row['cpu_inference_mean_ms']:.3f} / {row['cpu_inference_images_per_second']:.2f} | "
            f"{row['cuda_inference_mean_ms']:.3f} / {row['cuda_inference_images_per_second']:.2f} | "
            f"{row['cpu_end_to_end_mean_ms']:.3f} / {row['cpu_end_to_end_images_per_second']:.2f} | "
            f"{row['cuda_end_to_end_mean_ms']:.3f} / {row['cuda_end_to_end_images_per_second']:.2f} |"
        )
    lines += [
        "",
        "## Capacite theorique de 12 flux sur la machine mesuree",
        "",
        "La comparaison ci-dessous est une division sequentielle simplifiee. Elle ne simule ni concurrence reelle, ni files d'attente, ni capture de 12 flux.",
        "",
        "| Modele | Provider | Mesure | 12 img/s | 24 img/s | 60 img/s | 120 img/s | 300 img/s |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        entries = detail[model]["theoretical_12_stream_capacity"]
        for provider in ("cpu", "cuda"):
            for scope in ("inference_only", "end_to_end"):
                selected = [x for x in entries if x["provider"] == provider and x["scope"] == scope]
                cells = ["OK" if x["theoretical_capacity_met"] else "NON" for x in selected]
                throughput = selected[0]["measured_images_per_second"]
                lines.append(f"| {model} | {provider.upper()} | {scope} ({throughput:.2f} img/s) | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Validation numerique PyTorch / ONNX Runtime",
        "",
        "L'echantillon comprend 16 indices repartis deterministiquement dans le manifeste validation. La colonne decision 0,55 est un controle de preservation numerique; elle ne calibre pas de seuil pour EfficientNet ou MobileNet.",
        "",
        "| Modele | Max logits | Moyenne logits | Max probabilites | Argmax differents | Decisions differentes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['validation_max_abs_difference_logits']:.9f} | "
            f"{row['validation_mean_abs_difference_logits']:.9f} | "
            f"{row['validation_max_abs_difference_probabilities']:.9f} | "
            f"{row['validation_argmax_differences']} | {row['validation_decision_differences']} |"
        )
    lines += [
        "",
        "## Cout du preprocessing",
        "",
        "Les mesures preprocessing, inference seule et end-to-end sont des boucles distinctes. Leur moyenne n'est donc pas additive terme a terme. La boucle combinee inclut notamment les allocations, le decodage, les conversions, les copies memoire, l'orchestration Python, les effets de cache et, avec CUDA, les synchronisations imposees par ONNX Runtime. Ces contributions n'ont pas ete instrumentees individuellement; aucune cause unique ne peut etre attribuee. La latence end-to-end reste la reference applicative.",
        "",
        "## Conclusion",
        "",
        "Sur la RTX 3080 Laptop mesuree, ResNet18 est favorise: il offre la meilleure qualite validation et aussi le meilleur debit CUDA mesure, y compris end-to-end. Sur CPU, MobileNetV3 Small est nettement le plus rapide et le plus compact, au prix d'une baisse de qualite. EfficientNet-B0 constitue un compromis intermediaire sur CPU, mais n'est pas favorise par les mesures CUDA actuelles.",
        "",
        "Ces resultats ne valident pas la cible GTX 1060 3 Go / ancien i7. La VRAM par processus n'a pas ete obtenue et aucun vrai essai concurrent a 12 flux n'a ete realise. Aucun chiffre n'est extrapole vers cette cible.",
    ]
    (OUTPUT_DIR / "onnx_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "output_dir": str(OUTPUT_DIR.resolve())}, indent=2))


if __name__ == "__main__":
    main()
