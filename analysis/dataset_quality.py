"""Génère les statistiques et cas à examiner pour le volet qualité du dataset."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from build_dataset_manifest import (
    DEFAULT_OUTPUT,
    MASKED_CSV,
    PRINCIPAL_CSV,
    build_manifest,
)


OUTPUT_DIR = DEFAULT_OUTPUT.parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _distribution(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groupings: list[tuple[str, str | None]] = [
        ("global", None),
        ("year", "year"),
        ("camera", "cam_num"),
        ("position", "cam_position"),
    ]

    for grouping, column in groupings:
        groups: Iterable[tuple[object, pd.DataFrame]]
        groups = [("all", df)] if column is None else df.groupby(column, dropna=False)
        for group_value, subset in groups:
            counts = subset[label_column].value_counts(dropna=False)
            total = len(subset)
            for label, count in counts.items():
                rows.append(
                    {
                        "grouping": grouping,
                        "group": group_value,
                        "label": label,
                        "count": int(count),
                        "percentage": round(100 * int(count) / total, 2) if total else 0.0,
                    }
                )
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame) -> str:
    printable = df.fillna("").astype(str)
    header = "| " + " | ".join(printable.columns) + " |"
    separator = "| " + " | ".join("---" for _ in printable.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in printable.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *body])


def _with_ambiguity_reasons(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    reasons: list[str] = []
    for row in result.itertuples(index=False):
        current: list[str] = []
        if row.multiple:
            current.append("multiple")
        if row.chunk:
            current.append("chunk")
        if row.mixed_quality:
            current.append("mixed_quality")
        if row.label_filename != row.label_principal:
            current.append("label_changed")
        if row.masked_disagreement:
            current.append("masked_disagreement")
        reasons.append(";".join(current))
    result.insert(1, "ambiguity_reasons", reasons)
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(DEFAULT_OUTPUT)

    principal_distribution = _distribution(manifest, "label_principal")
    filename_distribution = _distribution(manifest, "label_filename")
    principal_distribution.to_csv(OUTPUT_DIR / "label_principal_distribution.csv", index=False)
    filename_distribution.to_csv(OUTPUT_DIR / "label_filename_distribution.csv", index=False)

    reviewed_counts = manifest["reviewed"].value_counts().reindex([True, False], fill_value=0)
    reviewed_distribution = pd.DataFrame(
        {
            "reviewed": [True, False],
            "count": [int(reviewed_counts[True]), int(reviewed_counts[False])],
            "percentage": [
                round(100 * int(reviewed_counts[True]) / len(manifest), 2),
                round(100 * int(reviewed_counts[False]) / len(manifest), 2),
            ],
        }
    )
    reviewed_distribution.to_csv(OUTPUT_DIR / "reviewed_distribution.csv", index=False)

    raw_labelers = manifest["labeled_by"].replace("", "(missing)")
    raw_labeled_by_distribution = (
        raw_labelers.value_counts(dropna=False)
        .rename_axis("labeled_by")
        .reset_index(name="count")
    ).round(2)
    raw_labeled_by_distribution["percentage"] = (
        100 * raw_labeled_by_distribution["count"] / len(manifest)
    ).round(2)
    raw_labeled_by_distribution.to_csv(
        OUTPUT_DIR / "labeled_by_raw_distribution.csv", index=False
    )

    canonical_labelers = manifest["labeled_by_canonical"].replace("", "(missing)")
    labeled_by_distribution = (
        canonical_labelers.value_counts(dropna=False)
        .rename_axis("labeled_by_canonical")
        .reset_index(name="count")
    )
    labeled_by_distribution["percentage"] = (
        100 * labeled_by_distribution["count"] / len(manifest)
    ).round(2)
    labeled_by_distribution.to_csv(OUTPUT_DIR / "labeled_by_distribution.csv", index=False)

    raw_labeler_count = int(manifest.loc[manifest["labeled_by"].str.strip().ne(""), "labeled_by"].nunique())
    canonical_labeler_count = int(
        manifest.loc[
            manifest["labeled_by_canonical"].str.strip().ne(""),
            "labeled_by_canonical",
        ].nunique()
    )
    nico_image_count = int(manifest["labeled_by_canonical"].eq("Nico").sum())

    label_changes = manifest[manifest["label_filename"] != manifest["label_principal"]].copy()
    masked_disagreements = manifest[manifest["masked_disagreement"]].copy()
    missing_labeled_by = manifest[manifest["reviewed"] & manifest["labeled_by"].str.strip().eq("")].copy()

    ambiguity_mask = (
        manifest["multiple"]
        | manifest["chunk"]
        | manifest["mixed_quality"]
        | (manifest["label_filename"] != manifest["label_principal"])
        | manifest["masked_disagreement"]
    )
    ambiguous_cases = _with_ambiguity_reasons(manifest[ambiguity_mask])

    masked_disagreements.to_csv(OUTPUT_DIR / "masked_disagreements.csv", index=False)
    label_changes.to_csv(OUTPUT_DIR / "label_changes.csv", index=False)
    missing_labeled_by.to_csv(OUTPUT_DIR / "missing_labeled_by.csv", index=False)
    ambiguous_cases.to_csv(OUTPUT_DIR / "ambiguous_cases.csv", index=False)

    masked_covered = int(manifest["masked_covered"].sum())
    masked_distribution = (
        manifest.loc[manifest["masked_covered"], "masked_label"]
        .value_counts()
        .rename_axis("masked_label")
        .reset_index(name="count")
    )
    masked_distribution["percentage_of_covered"] = (
        100 * masked_distribution["count"] / masked_covered
    ).round(2)
    masked_distribution.to_csv(OUTPUT_DIR / "masked_label_distribution.csv", index=False)

    ambiguous_stats = {
        "multiple": int(manifest["multiple"].sum()),
        "chunk": int(manifest["chunk"].sum()),
        "mixed_quality": int(manifest["mixed_quality"].sum()),
        "label_filename_vs_principal": len(label_changes),
        "masked_vs_principal": len(masked_disagreements),
        "unique_ambiguous_images": len(ambiguous_cases),
    }
    summary = {
        "dataset_images": len(manifest),
        "unique_filenames": int(manifest["filename"].nunique()),
        "batch_num_present": int(manifest["batch_num"].notna().sum()),
        "reviewed_true": int(reviewed_counts[True]),
        "reviewed_false": int(reviewed_counts[False]),
        "reviewed_true_missing_labeled_by": len(missing_labeled_by),
        "labelers": {
            "raw_identifier_count": raw_labeler_count,
            "canonical_labeler_count": canonical_labeler_count,
            "nico_images": nico_image_count,
        },
        "ambiguous_cases": ambiguous_stats,
        "labels_masked": {
            "covered_images": masked_covered,
            "coverage_percentage": round(100 * masked_covered / len(manifest), 2),
            "disagreements": len(masked_disagreements),
        },
        "source_sha256": {
            PRINCIPAL_CSV.name: _sha256(PRINCIPAL_CSV),
            MASKED_CSV.name: _sha256(MASKED_CSV),
        },
    }
    (OUTPUT_DIR / "dataset_quality_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    global_principal = principal_distribution[principal_distribution["grouping"] == "global"]
    global_filename = filename_distribution[filename_distribution["grouping"] == "global"]
    report = f"""# Rapport qualité du dataset

## Vue d'ensemble

- Images : {len(manifest)}
- Filenames uniques : {manifest['filename'].nunique()}
- Images avec `batch_num` : {manifest['batch_num'].notna().sum()}

## Distribution globale de label_principal

{_markdown_table(global_principal[['label', 'count', 'percentage']])}

Les distributions par année, caméra et position sont disponibles dans
`label_principal_distribution.csv`.

## Distribution globale de label_filename

{_markdown_table(global_filename[['label', 'count', 'percentage']])}

Les distributions par année, caméra et position sont disponibles dans
`label_filename_distribution.csv`.

## Relecture

{_markdown_table(reviewed_distribution)}

- Images relues sans `labeled_by` : {len(missing_labeled_by)}

## Identifiants des labeliseurs

Les identifiants `NicoG`, `Nico`, `nico` et `nico h` correspondent au même
annotateur. Ils sont regroupés sous `Nico` uniquement pour les statistiques ;
les valeurs brutes restent conservées dans le manifeste.

- Identifiants non vides avant normalisation : {raw_labeler_count}
- Labeliseurs après normalisation : {canonical_labeler_count}
- Images attribuées à `Nico` après regroupement : {nico_image_count}

La distribution brute est disponible dans `labeled_by_raw_distribution.csv`
et la distribution canonisée dans `labeled_by_distribution.csv`.

## Cas ambigus

| indicateur | nombre |
| --- | ---: |
| multiple | {ambiguous_stats['multiple']} |
| chunk | {ambiguous_stats['chunk']} |
| mixed_quality | {ambiguous_stats['mixed_quality']} |
| label_filename différent de label_principal | {ambiguous_stats['label_filename_vs_principal']} |
| labels_masked en désaccord vide/non-vide avec label_principal | {ambiguous_stats['masked_vs_principal']} |
| images uniques avec au moins un indicateur | {ambiguous_stats['unique_ambiguous_images']} |

## labels_masked

- Images couvertes : {masked_covered} ({summary['labels_masked']['coverage_percentage']:.2f} %)
- Désaccords vide/non-vide avec `label_principal` : {len(masked_disagreements)}

{_markdown_table(masked_distribution)}

`masked_label` est conservé comme information distincte et ne remplace jamais
`label_principal`.
"""
    (OUTPUT_DIR / "dataset_quality_report.md").write_text(report, encoding="utf-8")

    print("Analyse qualité terminée")
    print(f"  Images: {len(manifest):,}".replace(",", " "))
    print(f"  Filenames uniques: {manifest['filename'].nunique():,}".replace(",", " "))
    print(f"  Reviewed=True: {int(reviewed_counts[True]):,} ({reviewed_distribution.iloc[0]['percentage']:.2f} %)".replace(",", " "))
    print(f"  Reviewed=True sans labeled_by: {len(missing_labeled_by):,}".replace(",", " "))
    print(f"  Identifiants labeliseur avant normalisation: {raw_labeler_count}")
    print(f"  Labeliseurs après normalisation: {canonical_labeler_count}")
    print(f"  Images attribuées à Nico: {nico_image_count:,}".replace(",", " "))
    print(f"  Changements label_filename -> label_principal: {len(label_changes):,}".replace(",", " "))
    print(f"  Couverture labels_masked: {masked_covered:,} ({summary['labels_masked']['coverage_percentage']:.2f} %)".replace(",", " "))
    print(f"  Désaccords masked/principal: {len(masked_disagreements):,}".replace(",", " "))
    print(f"  Résultats: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
