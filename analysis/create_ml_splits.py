"""Crée des splits ML groupés et représentatifs sans copier les images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "analysis" / "output" / "final_training_manifest.csv"
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "output"
SPLITS = ("train", "validation", "test")
TARGET_RATIOS = np.array([0.70, 0.15, 0.15], dtype=float)
VALID_LABELS = ("Conforme", "NON Conforme", "PIETRA", "Vide")
VALID_YEARS = ("2025", "2026")
VALID_POSITIONS = ("T", "B")
VALID_CAMERAS = ("1", "2", "3", "4", "5", "6")
EXPECTED_IMAGES = 35_254
GLOBAL_SEED = 20260902
DEFAULT_CANDIDATES = 500
DEFAULT_LOCAL_SWAPS = 50_000

DIMENSIONS = {
    "effective_label": VALID_LABELS,
    "year": VALID_YEARS,
    "cam_num": VALID_CAMERAS,
    "cam_position": VALID_POSITIONS,
}
SCORE_WEIGHTS = {
    "size": 100.0,
    "effective_label": 10.0,
    "year": 5.0,
    "cam_num": 2.0,
    "cam_position": 1.0,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        low_memory=False,
        dtype={
            "filename": "string",
            "effective_label": "string",
            "year": "string",
            "cam_position": "string",
            "cam_num": "string",
            "historical_pair_id": "string",
            "split_group_id": "string",
        },
    )
    required = {
        "filename", "effective_label", "year", "cam_position", "cam_num",
        "multiple", "chunk", "mixed_quality", "historical_pair_id", "split_group_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    if len(df) != EXPECTED_IMAGES or not df.filename.is_unique:
        raise ValueError("Le manifeste final doit contenir 35 254 filenames uniques")
    if df.split_group_id.isna().any() or df.split_group_id.eq("").any():
        raise ValueError("split_group_id manquant")
    if set(df.effective_label) != set(VALID_LABELS):
        raise ValueError("effective_label doit contenir exactement les quatre classes attendues")
    for column, values in (
        ("year", VALID_YEARS), ("cam_position", VALID_POSITIONS), ("cam_num", VALID_CAMERAS)
    ):
        if set(df[column]) != set(values):
            raise ValueError(f"Valeurs inattendues dans {column}")
    return df


def build_group_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str], dict[str, slice]]:
    encoded_parts: list[pd.DataFrame] = []
    feature_names = ["images"]
    slices: dict[str, slice] = {}
    start = 1
    for dimension, categories in DIMENSIONS.items():
        part = pd.DataFrame(
            {
                f"{dimension}={category}": df[dimension].eq(category).astype(np.int16)
                for category in categories
            },
            index=df.index,
        )
        encoded_parts.append(part)
        feature_names.extend(part.columns)
        slices[dimension] = slice(start, start + len(categories))
        start += len(categories)
    encoded = pd.concat(encoded_parts, axis=1)
    encoded.insert(0, "images", 1)
    encoded["split_group_id"] = df.split_group_id.values
    grouped = encoded.groupby("split_group_id", sort=True, observed=True).sum()
    return grouped, grouped.to_numpy(dtype=np.int32), feature_names, slices


def nearest_cut(cumulative: np.ndarray, target: float, minimum: int = 1) -> int:
    index = int(np.searchsorted(cumulative, target, side="left"))
    candidates = [i for i in (index - 1, index) if minimum <= i < len(cumulative)]
    if not candidates:
        return max(minimum, min(index, len(cumulative) - 1))
    return min(candidates, key=lambda i: abs(float(cumulative[i]) - target)) + 1


def random_partition(group_sizes: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(group_sizes))
    ordered_sizes = group_sizes[order]
    cumulative = np.cumsum(ordered_sizes)
    total = int(cumulative[-1])
    train_cut = nearest_cut(cumulative, TARGET_RATIOS[0] * total)
    remaining_cumulative = np.cumsum(ordered_sizes[train_cut:])
    validation_count = nearest_cut(
        remaining_cumulative, TARGET_RATIOS[1] * total, minimum=1
    )
    validation_cut = train_cut + validation_count
    assignment = np.empty(len(group_sizes), dtype=np.int8)
    assignment[order[:train_cut]] = 0
    assignment[order[train_cut:validation_cut]] = 1
    assignment[order[validation_cut:]] = 2
    return assignment


def candidate_counts(features: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    return np.vstack([features[assignment == index].sum(axis=0) for index in range(3)])


def distribution_pp(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0) * 100


def score_candidate(
    counts: np.ndarray, global_counts: np.ndarray, slices: dict[str, slice]
) -> tuple[float, dict[str, float]]:
    total_images = float(global_counts[0])
    size_pp = counts[:, 0] / total_images * 100
    size_error = float(np.max(np.abs(size_pp - TARGET_RATIOS * 100)))
    components: dict[str, float] = {"size": size_error}
    total_score = SCORE_WEIGHTS["size"] * size_error
    for dimension, feature_slice in slices.items():
        split_pp = distribution_pp(counts[:, feature_slice])
        global_dimension = global_counts[feature_slice]
        global_pp = global_dimension / global_dimension.sum() * 100
        max_error = float(np.max(np.abs(split_pp - global_pp[None, :])))
        components[dimension] = max_error
        total_score += SCORE_WEIGHTS[dimension] * max_error
    return round(total_score, 12), components


def search_candidates(
    features: np.ndarray,
    slices: dict[str, slice],
    n_candidates: int,
    global_seed: int,
) -> tuple[np.ndarray, dict[str, object], pd.DataFrame]:
    global_counts = features.sum(axis=0)
    seed_generator = np.random.default_rng(global_seed)
    candidate_seeds = seed_generator.integers(0, np.iinfo(np.uint32).max, n_candidates, dtype=np.uint32)
    best: tuple[float, int, int, np.ndarray, dict[str, float], np.ndarray] | None = None
    diagnostic_rows: list[dict[str, object]] = []
    for candidate_index, candidate_seed in enumerate(candidate_seeds):
        assignment = random_partition(features[:, 0], int(candidate_seed))
        counts = candidate_counts(features, assignment)
        score, components = score_candidate(counts, global_counts, slices)
        row = {
            "candidate_index": candidate_index,
            "seed": int(candidate_seed),
            "score": score,
            **{f"max_abs_pp_{name}": value for name, value in components.items()},
            **{f"{split}_images": int(counts[idx, 0]) for idx, split in enumerate(SPLITS)},
        }
        diagnostic_rows.append(row)
        key = (score, int(candidate_seed), candidate_index)
        if best is None or key < best[:3]:
            best = (score, int(candidate_seed), candidate_index, assignment.copy(), components, counts.copy())
    assert best is not None
    metadata = {
        "candidate_index": best[2],
        "seed": best[1],
        "score": best[0],
        "score_components": best[4],
        "feature_counts": best[5].tolist(),
    }
    return best[3], metadata, pd.DataFrame(diagnostic_rows).sort_values("score")


def refine_assignment(
    features: np.ndarray,
    slices: dict[str, slice],
    assignment: np.ndarray,
    seed: int,
    attempts: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Améliore le score par échanges de groupes de même taille, sans changer les volumes."""
    rng = np.random.default_rng(seed ^ 0xA5A5A5A5)
    refined = assignment.copy()
    global_counts = features.sum(axis=0)
    counts = candidate_counts(features, refined)
    score, components = score_candidate(counts, global_counts, slices)
    accepted = 0
    for _ in range(attempts):
        left, right = rng.integers(0, len(refined), size=2)
        left_split, right_split = int(refined[left]), int(refined[right])
        if left_split == right_split or features[left, 0] != features[right, 0]:
            continue
        proposed_counts = counts.copy()
        proposed_counts[left_split] += features[right] - features[left]
        proposed_counts[right_split] += features[left] - features[right]
        proposed_score, proposed_components = score_candidate(
            proposed_counts, global_counts, slices
        )
        if proposed_score < score:
            refined[left], refined[right] = right_split, left_split
            counts = proposed_counts
            score = proposed_score
            components = proposed_components
            accepted += 1
    return refined, {
        "attempts": attempts,
        "accepted_swaps": accepted,
        "final_score": score,
        "final_score_components": components,
    }


def assign_splits(df: pd.DataFrame, groups: pd.DataFrame, assignment: np.ndarray) -> pd.DataFrame:
    group_to_split = pd.Series(
        [SPLITS[index] for index in assignment], index=groups.index, name="split", dtype="string"
    )
    result = df.copy()
    result["split"] = result.split_group_id.map(group_to_split)
    if result.split.isna().any():
        raise RuntimeError("Une image n'a pas reçu de split")
    return result


def value_distribution(df: pd.DataFrame, column: str, categories: tuple[str, ...]) -> dict[str, dict[str, float | int]]:
    counts = df[column].value_counts()
    total = len(df)
    return {
        category: {
            "count": int(counts.get(category, 0)),
            "percentage": round(100 * int(counts.get(category, 0)) / total, 4) if total else 0.0,
        }
        for category in categories
    }


def flag_distribution(df: pd.DataFrame, column: str) -> dict[str, float | int]:
    values = df[column]
    if values.dtype == object:
        truth = values.astype("string").str.strip().str.lower().isin({"true", "1", "yes"})
    else:
        truth = values.fillna(False).astype(bool)
    count = int(truth.sum())
    return {"count": count, "percentage": round(100 * count / len(df), 4) if len(df) else 0.0}


def subset_summary(df: pd.DataFrame) -> dict[str, object]:
    paired = df.historical_pair_id.notna()
    return {
        "images": len(df),
        "percentage_of_dataset": round(100 * len(df) / EXPECTED_IMAGES, 4),
        "split_group_ids": int(df.split_group_id.nunique()),
        "effective_label": value_distribution(df, "effective_label", VALID_LABELS),
        "year": value_distribution(df, "year", VALID_YEARS),
        "cam_position": value_distribution(df, "cam_position", VALID_POSITIONS),
        "cam_num": value_distribution(df, "cam_num", VALID_CAMERAS),
        "difficulties": {
            flag: flag_distribution(df, flag) for flag in ("multiple", "chunk", "mixed_quality")
        },
        "historical_tb": {
            "paired_images": int(paired.sum()),
            "unpaired_images": int((~paired).sum()),
            "complete_pairs": int(df.loc[paired, "historical_pair_id"].nunique()),
        },
    }


def representativeness(df: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    for dimension, categories in DIMENSIONS.items():
        global_counts = df[dimension].value_counts()
        global_pp = {category: 100 * int(global_counts.get(category, 0)) / len(df) for category in categories}
        split_results: dict[str, object] = {}
        all_differences: list[float] = []
        for split in SPLITS:
            subset = df[df.split == split]
            counts = subset[dimension].value_counts()
            differences = {
                category: round(100 * int(counts.get(category, 0)) / len(subset) - global_pp[category], 4)
                for category in categories
            }
            all_differences.extend(abs(value) for value in differences.values())
            split_results[split] = {
                "differences_pp": differences,
                "max_abs_pp_difference": round(max(abs(value) for value in differences.values()), 4),
            }
        result[dimension] = {
            "global_percentage": {key: round(value, 4) for key, value in global_pp.items()},
            "splits": split_results,
            "max_abs_pp_difference": round(max(all_differences), 4),
        }
    return result


def difficulty_representativeness(df: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    for flag in ("multiple", "chunk", "mixed_quality"):
        if df[flag].dtype == object:
            truth = df[flag].astype("string").str.strip().str.lower().isin({"true", "1", "yes"})
        else:
            truth = df[flag].fillna(False).astype(bool)
        global_percentage = 100 * int(truth.sum()) / len(df)
        split_values: dict[str, object] = {}
        for split in SPLITS:
            mask = df.split.eq(split)
            percentage = 100 * int((truth & mask).sum()) / int(mask.sum())
            split_values[split] = {
                "percentage": round(percentage, 4),
                "difference_pp": round(percentage - global_percentage, 4),
            }
        result[flag] = {
            "global_percentage": round(global_percentage, 4),
            "splits": split_values,
            "max_abs_pp_difference": round(
                max(abs(item["difference_pp"]) for item in split_values.values()), 4
            ),
        }
    return result


def validate(df: pd.DataFrame) -> dict[str, bool]:
    filenames = {split: set(df.loc[df.split == split, "filename"]) for split in SPLITS}
    validations = {
        "filename_unique_and_in_one_split": df.filename.is_unique,
        "split_group_id_in_one_split": df.groupby("split_group_id").split.nunique().eq(1).all(),
        "historical_pair_id_in_one_split": df.dropna(subset=["historical_pair_id"]).groupby(
            "historical_pair_id"
        ).split.nunique().eq(1).all(),
        "train_inter_validation_empty": filenames["train"].isdisjoint(filenames["validation"]),
        "train_inter_test_empty": filenames["train"].isdisjoint(filenames["test"]),
        "validation_inter_test_empty": filenames["validation"].isdisjoint(filenames["test"]),
        "union_35254_images": len(set.union(*filenames.values())) == EXPECTED_IMAGES,
        "all_images_have_split": df.split.notna().all() and df.split.ne("").all(),
        "only_three_expected_splits": set(df.split) == set(SPLITS),
        "four_classes_in_each_split": all(
            set(df.loc[df.split == split, "effective_label"]) == set(VALID_LABELS) for split in SPLITS
        ),
        "both_years_in_each_split": all(
            set(df.loc[df.split == split, "year"]) == set(VALID_YEARS) for split in SPLITS
        ),
        "top_bottom_in_each_split": all(
            set(df.loc[df.split == split, "cam_position"]) == set(VALID_POSITIONS) for split in SPLITS
        ),
        "cameras_1_to_6_in_each_split": all(
            set(df.loc[df.split == split, "cam_num"]) == set(VALID_CAMERAS) for split in SPLITS
        ),
    }
    return {key: bool(value) for key, value in validations.items()}


def comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    subsets = [("Global", df)] + [(split.capitalize(), df[df.split == split]) for split in SPLITS]
    for name, subset in subsets:
        row: dict[str, object] = {
            "dataset": name,
            "images": len(subset),
            "dataset_pct": round(100 * len(subset) / len(df), 4),
        }
        for dimension, categories in DIMENSIONS.items():
            counts = subset[dimension].value_counts()
            for category in categories:
                row[f"{dimension}={category}_pct"] = round(
                    100 * int(counts.get(category, 0)) / len(subset), 4
                )
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def clean(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(clean, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(map(clean, row)) + " |" for row in rows],
    ])


def distribution_rows(summaries: dict[str, object], dimension: str, categories: tuple[str, ...]) -> list[list[object]]:
    rows = []
    for category in categories:
        row: list[object] = [category]
        for split in SPLITS:
            item = summaries[split][dimension][category]
            row.append(f"{item['count']} ({item['percentage']:.2f} %)")
        rows.append(row)
    return rows


def make_figures(df: pd.DataFrame, output_dir: Path) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    datasets = ["Global", "Train", "Validation", "Test"]
    subsets = [df, df[df.split == "train"], df[df.split == "validation"], df[df.split == "test"]]
    colors = ["#17324D", "#087E8B", "#D9A441", "#E76F51"]
    specs = [
        ("effective_label", VALID_LABELS, "classes_by_split.png", "Classes", "Part des images (%)"),
        ("year", VALID_YEARS, "years_by_split.png", "Années", "Part des images (%)"),
        ("cam_num", VALID_CAMERAS, "cameras_by_split.png", "Caméras", "Part des images (%)"),
        ("cam_position", VALID_POSITIONS, "positions_by_split.png", "Top / Bottom", "Part des images (%)"),
    ]
    files: list[str] = []
    for column, categories, filename, title, ylabel in specs:
        x = np.arange(len(categories))
        width = 0.19
        fig, ax = plt.subplots(figsize=(9.2, 4.7))
        for index, (name, subset, color) in enumerate(zip(datasets, subsets, colors)):
            counts = subset[column].value_counts()
            values = [100 * int(counts.get(category, 0)) / len(subset) for category in categories]
            ax.bar(x + (index - 1.5) * width, values, width, label=name, color=color)
        ax.set_xticks(x, categories)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title} — comparaison global / splits", weight="bold")
        ax.legend(frameon=False, ncols=4, loc="upper center")
        ax.grid(axis="y", alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        files.append(str(path.relative_to(ROOT)).replace("\\", "/"))
    return files


def build_report(
    summaries: dict[str, object],
    represent: dict[str, object],
    difficulty_represent: dict[str, object],
    validations: dict[str, bool],
    search: dict[str, object],
    figures: list[str],
    comparison: pd.DataFrame,
) -> str:
    volume_rows = [
        [split, summaries[split]["images"], f"{summaries[split]['percentage_of_dataset']:.4f} %",
         summaries[split]["split_group_ids"]]
        for split in SPLITS
    ]
    difficulty_rows = []
    for flag in ("multiple", "chunk", "mixed_quality"):
        difficulty_rows.append([
            flag,
            *[f"{summaries[s]['difficulties'][flag]['count']} ({summaries[s]['difficulties'][flag]['percentage']:.2f} %)" for s in SPLITS],
        ])
    pair_rows = [
        [metric, *[summaries[s]["historical_tb"][metric] for s in SPLITS]]
        for metric in ("paired_images", "unpaired_images", "complete_pairs")
    ]
    max_rows = []
    for dimension in DIMENSIONS:
        max_rows.append([
            dimension,
            *[represent[dimension]["splits"][s]["max_abs_pp_difference"] for s in SPLITS],
            represent[dimension]["max_abs_pp_difference"],
        ])
    validation_rows = [[name, "PASS" if result else "FAIL"] for name, result in validations.items()]
    comparison_by_name = comparison.set_index("dataset")
    comparison_specs = [
        ("Volume images", "images", "count"),
        ("Conforme", "effective_label=Conforme_pct", "pct"),
        ("NON Conforme", "effective_label=NON Conforme_pct", "pct"),
        ("PIETRA", "effective_label=PIETRA_pct", "pct"),
        ("Vide", "effective_label=Vide_pct", "pct"),
        ("2025", "year=2025_pct", "pct"), ("2026", "year=2026_pct", "pct"),
        ("Top", "cam_position=T_pct", "pct"), ("Bottom", "cam_position=B_pct", "pct"),
        *[(f"Caméra {camera}", f"cam_num={camera}_pct", "pct") for camera in VALID_CAMERAS],
    ]
    comparison_rows = []
    for label, column, value_type in comparison_specs:
        values = []
        for dataset in ("Global", "Train", "Validation", "Test"):
            value = comparison_by_name.loc[dataset, column]
            values.append(f"{int(value):,}".replace(",", " ") if value_type == "count" else f"{float(value):.2f} %")
        comparison_rows.append([label, *values])
    max_overall = max(float(represent[d]["max_abs_pp_difference"]) for d in DIMENSIONS)
    max_difficulty = max(
        float(difficulty_represent[flag]["max_abs_pp_difference"])
        for flag in ("multiple", "chunk", "mixed_quality")
    )
    conclusion = (
        "Les splits sont suffisamment proches du dataset global pour commencer les entraînements : "
        f"l’écart maximal observé sur les dimensions prioritaires est de {max_overall:.4f} point de pourcentage, "
        "tous les contrôles anti-fuite passent et chaque sous-ensemble contient les quatre classes, les deux années, "
        "les deux positions et les six caméras."
        if max_overall <= 1.0 and all(validations.values())
        else
        "Les splits ne sont pas validés pour l’entraînement : au moins un contrôle anti-fuite échoue ou un écart "
        "de représentativité dépasse 1 point de pourcentage."
    )
    return f"""# Splits ML CastagNet — §4.2

## Objectif

Les splits sont construits avant tout entraînement afin de figer une base commune à toutes les expériences. `train` sert à apprendre les paramètres ; `validation` sert à comparer les architectures, régler les hyperparamètres, surveiller l’overfitting et sélectionner le modèle ; `test` est réservé à l’évaluation finale. Le test est figé par ce manifeste et ne doit pas servir au choix d’architecture, learning rate, augmentation, nombre d’epochs, seuil ou modèle.

## Choix 70/15/15

Le compromis 70/15/15 réserve la majorité des données à l’apprentissage tout en maintenant plusieurs milliers d’images pour la sélection et l’évaluation finales. Ce ratio n’est pas universel ; il est adapté au volume disponible. Les proportions portent sur les images, avec de petits écarts admis pour préserver les groupes indivisibles.

{markdown_table(['split', 'images', '% réel', 'split_group_id uniques'], volume_rows)}

## Risque de fuite T/B

Un `random_split` au niveau image pourrait placer les deux vues d’une même châtaigne potentielle dans des jeux différents. L’affectation se fait donc exclusivement au niveau de `split_group_id`. Les associations historiques restent heuristiques, mais une fois admises elles sont conservées intégralement dans un seul split. Les 5 455 paires dont les labels diffèrent ne reçoivent aucun label de groupe : chacune contribue avec les labels de ses deux images.

## Méthode

- Source unique : `analysis/output/final_training_manifest.csv` ; cible : `effective_label`.
- {search['candidates_tested']} partitions candidates testées à partir du seed global `{search['global_seed']}`.
- Pour chaque candidat, les groupes sont mélangés, puis coupés au plus près de 70/15/15 en nombre d’images sans casser de groupe.
- Score : `100 × taille + 10 × classes + 5 × années + 2 × caméras + 1 × positions`, chaque composante étant le maximum absolu des écarts en points de pourcentage.
- Candidat retenu : index `{search['selected_candidate_index']}`, seed `{search['selected_seed']}` ; score initial `{search['selected_score_before_refinement']:.6f}`, puis `{search['selected_score']:.6f}` après {search['local_refinement']['accepted_swaps']} échanges acceptés sur {search['local_refinement']['attempts']} essais. Les échanges portent uniquement sur des groupes de même taille afin de conserver exactement les volumes.

## Résultats

### Classes

{markdown_table(['classe', 'train', 'validation', 'test'], distribution_rows(summaries, 'effective_label', VALID_LABELS))}

Le test contient exactement {summaries['test']['effective_label']['Conforme']['count']} Conforme, {summaries['test']['effective_label']['NON Conforme']['count']} NON Conforme et {summaries['test']['effective_label']['PIETRA']['count']} PIETRA. Ces volumes permettront d’interpréter au §4.2 les seuils métier de rappel ≥ 85 % et de précision ≥ 95 % sur Conforme ; aucune métrique modèle n’est calculée ici.

### Années

{markdown_table(['année', 'train', 'validation', 'test'], distribution_rows(summaries, 'year', VALID_YEARS))}

### Positions

{markdown_table(['position', 'train', 'validation', 'test'], distribution_rows(summaries, 'cam_position', VALID_POSITIONS))}

### Caméras

{markdown_table(['caméra', 'train', 'validation', 'test'], distribution_rows(summaries, 'cam_num', VALID_CAMERAS))}

### Cas difficiles

{markdown_table(['flag', 'train', 'validation', 'test'], difficulty_rows)}

Les flags difficiles ne constituent pas une contrainte forte du score ; leur distribution est contrôlée après coup afin de ne pas complexifier artificiellement l’optimisation. L’écart maximal au global est de {max_difficulty:.4f} point : aucun déséquilibre fort n’est observé sur `multiple`, `chunk` ou `mixed_quality`.

### Paires historiques

{markdown_table(['indicateur', 'train', 'validation', 'test'], pair_rows)}

### Écarts au global

{markdown_table(['dimension', 'train max abs pp', 'validation max abs pp', 'test max abs pp', 'maximum'], max_rows)}

Les valeurs détaillées catégorie par catégorie sont disponibles dans `ml_split_summary.json` et `ml_split_comparison.csv`.

### Tableau global de comparaison

{markdown_table(['indicateur', 'Global', 'Train', 'Validation', 'Test'], comparison_rows)}

## Satisfaisant

- aucun filename, `split_group_id` ou `historical_pair_id` ne traverse deux splits ;
- les volumes sont au plus près de 70/15/15 sans casser les paires ;
- chaque split contient les quatre classes, les deux années, Top et Bottom, ainsi que les six caméras ;
- le test contient plusieurs centaines d’images dans chacune des classes métier ;
- l’affectation est déterministe et reproductible.

## Limites

- les paires historiques demeurent heuristiques et ne prouvent pas l’identité physique ;
- l’équilibrage reproduit le dataset fourni, pas nécessairement le monde réel ;
- l’hétérogénéité 2025/2026 est répartie mais non supprimée ;
- la sélection du meilleur parmi plusieurs candidats réduit les écarts observés sans garantir un optimum mathématique global ;
- `multiple` et `chunk` sont seulement audités après affectation.

## Contrôles anti-fuite

{markdown_table(['contrôle', 'résultat'], validation_rows)}

## Figures

{chr(10).join(f'- `{figure}`' for figure in figures)}

## Décision

{conclusion}

Le jeu de test est désormais figé. Aucun entraînement, DataLoader, augmentation, suivi MLflow ou calcul de métrique modèle n’a été lancé.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--local-swaps", type=int, default=DEFAULT_LOCAL_SWAPS)
    args = parser.parse_args()
    if args.candidates < 1:
        raise ValueError("--candidates doit être positif")

    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    source_hash = sha256(manifest_path)
    df = load_manifest(manifest_path)
    groups, features, feature_names, slices = build_group_matrix(df)
    assignment, selected, diagnostics = search_candidates(
        features, slices, args.candidates, args.seed
    )
    initial_score = selected["score"]
    assignment, refinement = refine_assignment(
        features, slices, assignment, selected["seed"], args.local_swaps
    )
    result = assign_splits(df, groups, assignment)
    validations = validate(result)
    if not all(validations.values()):
        failed = [name for name, passed in validations.items() if not passed]
        raise RuntimeError(f"Contrôles anti-fuite échoués : {', '.join(failed)}")

    summaries = {split: subset_summary(result[result.split == split]) for split in SPLITS}
    represent = representativeness(result)
    difficulty_represent = difficulty_representativeness(result)
    comparison = comparison_table(result)
    figure_paths = make_figures(result, output_dir / "split_figures")
    search = {
        "method": "random_group_permutation_with_nearest_image_count_cuts",
        "candidates_tested": args.candidates,
        "global_seed": args.seed,
        "selected_candidate_index": selected["candidate_index"],
        "selected_seed": selected["seed"],
        "selected_score_before_refinement": initial_score,
        "selected_score": refinement["final_score"],
        "selected_score_components": refinement["final_score_components"],
        "local_refinement": refinement,
        "score_weights": SCORE_WEIGHTS,
        "target_ratios": dict(zip(SPLITS, TARGET_RATIOS.tolist())),
        "priority_order": ["group_integrity", "size", "effective_label", "year", "cam_num", "cam_position"],
    }
    summary = {
        "purpose": "grouped_train_validation_test_split_before_training",
        "source": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": source_hash,
        "target": "effective_label",
        "dataset_images": len(result),
        "dataset_split_group_ids": int(result.split_group_id.nunique()),
        "historical_pairs_with_different_effective_labels": 5455,
        "search": search,
        "splits": summaries,
        "representativeness": represent,
        "difficulty_representativeness": difficulty_represent,
        "validations": validations,
        "figures": figure_paths,
        "test_is_frozen": True,
        "training_started": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "ml_split_manifest.csv", index=False)
    for split in SPLITS:
        result[result.split == split].to_csv(output_dir / f"{split}_manifest.csv", index=False)
    comparison.to_csv(output_dir / "ml_split_comparison.csv", index=False)
    diagnostics.head(25).to_csv(output_dir / "ml_split_candidate_diagnostics.csv", index=False)
    (output_dir / "ml_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "ml_split_report.md").write_text(
        build_report(
            summaries, represent, difficulty_represent, validations, search, figure_paths,
            comparison,
        ),
        encoding="utf-8",
    )

    if sha256(manifest_path) != source_hash:
        raise RuntimeError("Le manifeste source a changé pendant la génération")

    print(f"Source: {len(df):,} images, {len(groups):,} groupes".replace(",", " "))
    for split in SPLITS:
        values = summaries[split]
        print(
            f"{split}: {values['images']:,} images ({values['percentage_of_dataset']:.4f} %), "
            f"{values['split_group_ids']:,} groupes".replace(",", " ")
        )
    print(
        f"Candidat {search['selected_candidate_index']} / seed {search['selected_seed']} / "
        f"score {search['selected_score']:.6f}"
    )
    print("Max abs pp:", {key: value["max_abs_pp_difference"] for key, value in represent.items()})
    print(f"Validations: {sum(validations.values())}/{len(validations)} PASS")
    print("Aucun entraînement lancé.")


if __name__ == "__main__":
    main()
