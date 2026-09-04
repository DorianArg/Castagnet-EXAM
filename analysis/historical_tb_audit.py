"""Audit exploratoire des clés possibles d'appariement Top/Bottom historiques.

Ce script ne produit volontairement aucune table de paires. Il mesure seulement
ce que les métadonnées permettent (ou non) de distinguer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "analysis" / "output" / "dataset_manifest.csv"
DEFAULT_OUTPUT_DIR = ROOT / "analysis" / "output"
NO_BATCH = "<NO_BATCH>"

FILENAME_PATTERN = re.compile(
    r"^(?P<year>\d{4})_(?P<label>NON_Conforme|Conforme|PIETRA)"
    r"(?:_(?P<batch>\d+))?_Cam_(?P<position>[TB])_"
    r"(?P<camera>\d+)_(?P<sample>\d+)\.jpg$"
)

KEYS = {
    "A": ["year", "batch_key", "cam_num", "sample_num"],
    "B": ["year", "batch_key", "cam_num", "sample_num", "label_filename"],
    "C": ["year", "cam_num", "sample_num", "label_filename"],
    "D": ["year", "cam_num", "sample_num"],
}
DISPLAY_KEYS = {
    key: ["batch_num" if column == "batch_key" else column for column in columns]
    for key, columns in KEYS.items()
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_bool(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    values = series.astype("string").str.strip().str.lower()
    invalid = values[~values.isin(["true", "false"]) & values.notna()].unique()
    if len(invalid):
        raise ValueError(f"Valeurs booléennes inattendues dans {column}: {invalid.tolist()}")
    return values.eq("true")


def _load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {
        "filename", "year", "cam_position", "cam_num", "sample_num",
        "batch_num", "label_filename", "label_principal", "multiple", "chunk",
        "mixed_quality",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Colonnes absentes du manifeste: {', '.join(missing)}")
    if df["filename"].duplicated().any():
        raise ValueError("filename n'est pas unique dans le manifeste")
    invalid_positions = sorted(set(df["cam_position"].dropna()) - {"T", "B"})
    if invalid_positions:
        raise ValueError(f"Positions caméra inattendues: {invalid_positions}")
    for column in ("multiple", "chunk", "mixed_quality"):
        df[column] = _as_bool(df[column], column)
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df["cam_num"] = pd.to_numeric(df["cam_num"], errors="raise").astype(int)
    df["sample_num"] = pd.to_numeric(df["sample_num"], errors="raise").astype(int)
    batch = pd.to_numeric(df["batch_num"], errors="coerce").astype("Int64")
    df["batch_num"] = batch
    df["batch_key"] = batch.astype("string").fillna(NO_BATCH)
    return df


def _group_counts(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(columns + ["cam_position"], dropna=False)
        .size()
        .unstack("cam_position", fill_value=0)
    )
    for position in ("T", "B"):
        if position not in grouped:
            grouped[position] = 0
    grouped = grouped[["T", "B"]].astype(int)
    grouped["image_count"] = grouped["T"] + grouped["B"]
    return grouped


def _masks(groups: pd.DataFrame) -> dict[str, pd.Series]:
    top = groups["T"]
    bottom = groups["B"]
    return {
        "exact_1t1b": top.eq(1) & bottom.eq(1),
        "top_only": top.gt(0) & bottom.eq(0),
        "bottom_only": top.eq(0) & bottom.gt(0),
        "multiple_top": top.gt(1),
        "multiple_bottom": bottom.gt(1),
        "multiple_both": top.gt(1) & bottom.gt(1),
        "ambiguous": top.gt(1) | bottom.gt(1),
    }


def _metrics(df: pd.DataFrame, key_id: str, scope: str) -> dict[str, object]:
    groups = _group_counts(df, KEYS[key_id])
    masks = _masks(groups)
    total_images = len(df)
    exact_groups = int(masks["exact_1t1b"].sum())
    row: dict[str, object] = {
        "scope": scope,
        "key_id": key_id,
        "key_columns": " + ".join(DISPLAY_KEYS[key_id]),
        "total_images": total_images,
        "total_groups": len(groups),
    }
    for category, mask in masks.items():
        row[f"{category}_groups"] = int(mask.sum())
        row[f"{category}_images"] = int(groups.loc[mask, "image_count"].sum())
        row[f"{category}_image_rate_pct"] = round(
            100 * int(groups.loc[mask, "image_count"].sum()) / total_images, 2
        )
    paired_images = 2 * exact_groups
    row["potential_pairs"] = exact_groups
    row["potentially_paired_images"] = paired_images
    row["remaining_images"] = total_images - paired_images
    row["potential_pair_rate_pct"] = round(100 * paired_images / total_images, 2)
    row["remaining_rate_pct"] = round(100 * (total_images - paired_images) / total_images, 2)
    return row


def _group_record_table(df: pd.DataFrame, key_id: str) -> pd.DataFrame:
    columns = KEYS[key_id]
    grouped = _group_counts(df, columns).reset_index().rename(
        columns={"T": "top_count", "B": "bottom_count"}
    )
    top_names = (
        df.loc[df.cam_position.eq("T")]
        .groupby(columns, dropna=False, sort=False).filename
        .agg(" | ".join).rename("top_filenames").reset_index()
    )
    bottom_names = (
        df.loc[df.cam_position.eq("B")]
        .groupby(columns, dropna=False, sort=False).filename
        .agg(" | ".join).rename("bottom_filenames").reset_index()
    )
    grouped = grouped.merge(top_names, on=columns, how="left", validate="one_to_one")
    grouped = grouped.merge(bottom_names, on=columns, how="left", validate="one_to_one")
    grouped[["top_filenames", "bottom_filenames"]] = grouped[
        ["top_filenames", "bottom_filenames"]
    ].fillna("")
    grouped["image_count"] = grouped.top_count + grouped.bottom_count
    grouped.insert(0, "key_columns", " + ".join(DISPLAY_KEYS[key_id]))
    grouped.insert(0, "key_id", key_id)
    return grouped.rename(columns={"batch_key": "batch_num"})


def _exact_pairs(df: pd.DataFrame, key_id: str) -> pd.DataFrame:
    columns = KEYS[key_id]
    groups = _group_counts(df, columns)
    exact_keys = groups.loc[_masks(groups)["exact_1t1b"]].reset_index()[columns]
    members = df.merge(exact_keys, on=columns, how="inner", validate="many_to_one")
    fields = columns + ["label_principal", "multiple", "chunk"]
    top = members.loc[members.cam_position.eq("T"), fields].rename(
        columns={"label_principal": "top_label", "multiple": "top_multiple", "chunk": "top_chunk"}
    )
    bottom = members.loc[members.cam_position.eq("B"), fields].rename(
        columns={"label_principal": "bottom_label", "multiple": "bottom_multiple", "chunk": "bottom_chunk"}
    )
    pairs = top.merge(bottom, on=columns, how="inner", validate="one_to_one")
    pairs["same_label"] = pairs.top_label.eq(pairs.bottom_label)
    pairs["top_vide"] = pairs.top_label.astype("string").str.casefold().eq("vide")
    pairs["bottom_vide"] = pairs.bottom_label.astype("string").str.casefold().eq("vide")
    return pairs


def _pair_diagnostics(pairs: pd.DataFrame, key_id: str, scope: str) -> dict[str, object]:
    count = len(pairs)
    same = int(pairs["same_label"].sum()) if count else 0
    return {
        "scope": scope,
        "key_id": key_id,
        "potential_pairs": count,
        "same_label_principal": same,
        "different_label_principal": count - same,
        "pairs_with_multiple": int((pairs.top_multiple | pairs.bottom_multiple).sum()) if count else 0,
        "multiple_differs_between_sides": int((pairs.top_multiple != pairs.bottom_multiple).sum()) if count else 0,
        "pairs_with_chunk": int((pairs.top_chunk | pairs.bottom_chunk).sum()) if count else 0,
        "chunk_differs_between_sides": int((pairs.top_chunk != pairs.bottom_chunk).sum()) if count else 0,
        "pairs_with_vide": int((pairs.top_vide | pairs.bottom_vide).sum()) if count else 0,
        "vide_differs_between_sides": int((pairs.top_vide != pairs.bottom_vide).sum()) if count else 0,
    }


def _batch_impact(df: pd.DataFrame, with_batch: str, without_batch: str, scope: str) -> dict[str, object]:
    without_columns = KEYS[without_batch]
    with_columns = KEYS[with_batch]
    without_groups = _group_counts(df, without_columns)
    without_table = without_groups.reset_index()
    without_table["original_ambiguous"] = (
        without_table["T"].gt(1) | without_table["B"].gt(1)
    )
    split = _group_counts(df, with_columns).reset_index()
    split["subgroup_ambiguous"] = split["T"].gt(1) | split["B"].gt(1)
    split_status = split.groupby(without_columns, dropna=False)["subgroup_ambiguous"].any().reset_index()
    status = without_table.merge(split_status, on=without_columns, validate="one_to_one")
    original = status.loc[status.original_ambiguous]
    resolved = int((~original.subgroup_ambiguous).sum())
    still_ambiguous = int(original.subgroup_ambiguous.sum())
    before = int(without_table.original_ambiguous.sum())
    after = int(split.subgroup_ambiguous.sum())
    return {
        "scope": scope,
        "comparison": f"{without_batch}->{with_batch}",
        "without_batch_key": without_batch,
        "with_batch_key": with_batch,
        "ambiguous_groups_without_batch": before,
        "ambiguous_groups_with_batch": after,
        "ambiguous_groups_resolved_by_batch_split": resolved,
        "original_ambiguous_groups_still_ambiguous": still_ambiguous,
        "net_ambiguous_group_reduction": before - after,
    }


def _label_effect(df: pd.DataFrame, structural: str, with_label: str, scope: str) -> dict[str, object]:
    structural_groups = _group_counts(df, KEYS[structural])
    exact_keys = structural_groups.loc[_masks(structural_groups)["exact_1t1b"]].reset_index()[KEYS[structural]]
    members = df.merge(exact_keys, on=KEYS[structural], how="inner", validate="many_to_one")
    top_labels = members.loc[members.cam_position.eq("T"), KEYS[structural] + ["label_filename"]]
    bottom_labels = members.loc[members.cam_position.eq("B"), KEYS[structural] + ["label_filename"]]
    paired_labels = top_labels.merge(
        bottom_labels, on=KEYS[structural], suffixes=("_top", "_bottom"), validate="one_to_one"
    )
    exact_split = int(paired_labels.label_filename_top.ne(paired_labels.label_filename_bottom).sum())
    structural_metrics = _metrics(df, structural, scope)
    label_metrics = _metrics(df, with_label, scope)
    return {
        "scope": scope,
        "comparison": f"{structural}->{with_label}",
        "structural_key": structural,
        "with_label_filename_key": with_label,
        "exact_pairs_without_label": structural_metrics["potential_pairs"],
        "exact_pairs_with_label": label_metrics["potential_pairs"],
        "ambiguous_groups_without_label": structural_metrics["ambiguous_groups"],
        "ambiguous_groups_with_label": label_metrics["ambiguous_groups"],
        "structural_exact_pairs_split_by_different_label_filename": exact_split,
    }


def _sample_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (year, cam_num, batch_key), subset in df.groupby(
        ["year", "cam_num", "batch_key"], dropna=False
    ):
        values = sorted(subset.sample_num.unique())
        frequencies = subset.sample_num.value_counts()
        differences = pd.Series(values).diff().dropna()
        rows.append(
            {
                "year": int(year),
                "cam_num": int(cam_num),
                "batch_num": batch_key,
                "images": len(subset),
                "sample_num_min": int(min(values)),
                "sample_num_max": int(max(values)),
                "distinct_sample_num": len(values),
                "repeated_sample_values": int(frequencies.gt(1).sum()),
                "max_images_for_one_sample_num": int(frequencies.max()),
                "missing_values_inside_min_max": int(max(values) - min(values) + 1 - len(values)),
                "numbering_breaks": int(differences.gt(1).sum()),
                "largest_forward_gap": int(differences.max()) if len(differences) else 0,
                "sample_zero_images": int(subset.sample_num.eq(0).sum()),
                "label_filename_values": int(subset.label_filename.nunique()),
            }
        )
    per_year: dict[str, object] = {}
    for year, subset in df.groupby("year"):
        frequencies = subset.sample_num.value_counts()
        contexts = subset.groupby("sample_num").agg(
            cameras=("cam_num", "nunique"),
            batches=("batch_key", "nunique"),
            historical_labels=("label_filename", "nunique"),
        )
        per_year[str(int(year))] = {
            "images": len(subset),
            "sample_num_min": int(subset.sample_num.min()),
            "sample_num_max": int(subset.sample_num.max()),
            "distinct_sample_num": int(subset.sample_num.nunique()),
            "sample_values_repeated": int(frequencies.gt(1).sum()),
            "maximum_images_sharing_one_sample_num": int(frequencies.max()),
            "sample_values_used_by_multiple_cameras": int(contexts.cameras.gt(1).sum()),
            "sample_values_used_by_multiple_batches": int(contexts.batches.gt(1).sum()),
            "sample_values_used_by_multiple_historical_labels": int(contexts.historical_labels.gt(1).sum()),
        }
    return pd.DataFrame(rows), per_year


def _sequence_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare les séquences demandées avec et sans label_filename."""
    variants = {
        "with_label_filename": [
            "year", "batch_key", "cam_num", "label_filename", "cam_position"
        ],
        "without_label_filename": ["year", "batch_key", "cam_num", "cam_position"],
    }
    rows: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    for variant, columns in variants.items():
        variant_rows: list[dict[str, object]] = []
        for values, subset in df.groupby(columns, dropna=False, sort=True):
            values = values if isinstance(values, tuple) else (values,)
            samples = sorted(int(value) for value in subset.sample_num.unique())
            holes = int(max(samples) - min(samples) + 1 - len(samples))
            record: dict[str, object] = {
                "namespace_variant": variant,
                "year": int(dict(zip(columns, values))["year"]),
                "batch_num": dict(zip(columns, values))["batch_key"],
                "cam_num": int(dict(zip(columns, values))["cam_num"]),
                "label_filename": dict(zip(columns, values)).get("label_filename", ""),
                "cam_position": dict(zip(columns, values))["cam_position"],
                "sample_num_min": min(samples),
                "sample_num_max": max(samples),
                "distinct_sample_num": len(samples),
                "holes": holes,
                "continuous": holes == 0,
                "starts_at_zero": min(samples) == 0,
            }
            rows.append(record)
            variant_rows.append(record)
        table = pd.DataFrame(variant_rows)
        by_year: dict[str, object] = {}
        for year, subset in table.groupby("year"):
            by_year[str(int(year))] = {
                "contexts": len(subset),
                "continuous_contexts": int(subset.continuous.sum()),
                "contexts_starting_at_zero": int(subset.starts_at_zero.sum()),
                "total_holes": int(subset.holes.sum()),
                "maximum_holes_in_one_context": int(subset.holes.max()),
            }
        summary[variant] = {
            "contexts": len(table),
            "continuous_contexts": int(table.continuous.sum()),
            "contexts_starting_at_zero": int(table.starts_at_zero.sum()),
            "total_holes": int(table.holes.sum()),
            "maximum_holes_in_one_context": int(table.holes.max()),
            "by_year": by_year,
        }
    return pd.DataFrame(rows), summary


def _cross_label_reuse(df: pd.DataFrame) -> dict[str, object]:
    columns = ["year", "batch_key", "cam_num", "sample_num"]
    slots = df.groupby(columns, dropna=False).agg(
        label_filename_count=("label_filename", "nunique"),
        image_count=("filename", "size"),
    ).reset_index()

    def summarize(subset: pd.DataFrame) -> dict[str, object]:
        reused = subset.label_filename_count.gt(1)
        reused_images = int(subset.loc[reused, "image_count"].sum())
        return {
            "sample_slots": len(subset),
            "slots_reused_between_labels": int(reused.sum()),
            "slot_reuse_rate_pct": round(100 * float(reused.mean()), 2),
            "images_in_reused_slots": reused_images,
            "image_coverage_rate_pct": round(100 * reused_images / int(subset.image_count.sum()), 2),
            "label_count_distribution": {
                str(int(label_count)): int(count)
                for label_count, count in subset.label_filename_count.value_counts().sort_index().items()
            },
        }

    return {
        "definition": "same year + batch_num + cam_num + sample_num in multiple label_filename",
        "all": summarize(slots),
        "by_year": {
            str(int(year)): summarize(subset)
            for year, subset in slots.groupby("year")
        },
    }


def _exact_members(df: pd.DataFrame, key_id: str) -> pd.DataFrame:
    columns = KEYS[key_id]
    groups = _group_counts(df, columns)
    exact_keys = groups.loc[_masks(groups)["exact_1t1b"]].reset_index()[columns]
    return df.merge(exact_keys, on=columns, how="inner", validate="many_to_one")


def _b_filename_consistency(df: pd.DataFrame) -> dict[str, object]:
    columns = KEYS["B"]
    members = _exact_members(df, "B")
    top = members.loc[members.cam_position.eq("T"), columns + ["filename"]]
    bottom = members.loc[members.cam_position.eq("B"), columns + ["filename"]]
    pairs = top.merge(bottom, on=columns, suffixes=("_top", "_bottom"), validate="one_to_one")
    consistent = 0
    invalid_filenames = 0
    for row in pairs.itertuples(index=False):
        top_match = FILENAME_PATTERN.fullmatch(row.filename_top)
        bottom_match = FILENAME_PATTERN.fullmatch(row.filename_bottom)
        if top_match is None or bottom_match is None:
            invalid_filenames += 1
            continue
        top_parts = top_match.groupdict()
        bottom_parts = bottom_match.groupdict()
        top_position = top_parts.pop("position")
        bottom_position = bottom_parts.pop("position")
        if top_position == "T" and bottom_position == "B" and top_parts == bottom_parts:
            consistent += 1
    total = len(pairs)
    return {
        "potential_b_pairs": total,
        "filenames_differing_only_by_cam_position": consistent,
        "rate_pct": round(100 * consistent / total, 2) if total else 0.0,
        "inconsistent_pairs": total - consistent,
        "invalid_filenames": invalid_filenames,
    }


def _a_pairs_split_by_label(df: pd.DataFrame) -> list[dict[str, object]]:
    columns = KEYS["A"]
    members = _exact_members(df, "A")
    top_fields = columns + ["filename", "label_filename", "label_principal"]
    bottom_fields = top_fields
    top = members.loc[members.cam_position.eq("T"), top_fields]
    bottom = members.loc[members.cam_position.eq("B"), bottom_fields]
    pairs = top.merge(bottom, on=columns, suffixes=("_top", "_bottom"), validate="one_to_one")
    pairs = pairs.loc[pairs.label_filename_top.ne(pairs.label_filename_bottom)].copy()
    pairs = pairs.rename(columns={"batch_key": "batch_num"})
    wanted = [
        "filename_top", "filename_bottom", "year", "batch_num", "cam_num", "sample_num",
        "label_filename_top", "label_filename_bottom", "label_principal_top",
        "label_principal_bottom",
    ]
    return pairs[wanted].sort_values(["year", "cam_num", "sample_num"]).to_dict("records")


def _markdown_table(df: pd.DataFrame) -> str:
    printable = df.fillna("").astype(str)
    header = "| " + " | ".join(printable.columns) + " |"
    separator = "| " + " | ".join("---" for _ in printable.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in printable.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *body])


def _write_report(
    path: Path,
    comparison: pd.DataFrame,
    diagnostics: pd.DataFrame,
    batch_impact: list[dict[str, object]],
    label_effect: list[dict[str, object]],
    sample_summary: dict[str, object],
    sequence_summary: dict[str, object],
    cross_label_reuse: dict[str, object],
    filename_consistency: dict[str, object],
    split_cases: list[dict[str, object]],
) -> None:
    compact = comparison[
        ["scope", "key_id", "exact_1t1b_groups", "potential_pair_rate_pct",
         "top_only_groups", "top_only_image_rate_pct", "bottom_only_groups",
         "bottom_only_image_rate_pct", "ambiguous_groups", "ambiguous_image_rate_pct"]
    ].rename(columns={
        "potential_pair_rate_pct": "paired_images_pct",
        "top_only_image_rate_pct": "top_only_images_pct",
        "bottom_only_image_rate_pct": "bottom_only_images_pct",
        "ambiguous_image_rate_pct": "ambiguous_images_pct",
    })
    bdiag = diagnostics[(diagnostics.scope == "all") & (diagnostics.key_id == "B")].iloc[0]
    batch_table = pd.DataFrame(batch_impact)[
        ["scope", "comparison", "ambiguous_groups_without_batch", "ambiguous_groups_with_batch",
         "ambiguous_groups_resolved_by_batch_split", "original_ambiguous_groups_still_ambiguous"]
    ]
    label_table = pd.DataFrame(label_effect)
    sequence_rows: list[dict[str, object]] = []
    for variant, values in sequence_summary.items():
        for year, year_values in values["by_year"].items():
            sequence_rows.append({"variant": variant, "year": year, **year_values})
    sequence_table = pd.DataFrame(sequence_rows)
    reuse_rows = []
    for scope, values in [("all", cross_label_reuse["all"]), *cross_label_reuse["by_year"].items()]:
        reuse_rows.append(
            {
                "scope": scope,
                "sample_slots": values["sample_slots"],
                "reused_between_labels": values["slots_reused_between_labels"],
                "reuse_pct": values["slot_reuse_rate_pct"],
                "images_in_reused_slots": values["images_in_reused_slots"],
                "images_pct": values["image_coverage_rate_pct"],
            }
        )
    split_table = pd.DataFrame(split_cases).fillna("")
    text = f"""# Audit exploratoire de l'appariement Top/Bottom historique

Ce rapport est un audit de métadonnées. Il ne constitue pas une vérité terrain. La table historique produite séparément ne contient que des correspondances qualifiées d'heuristiques.

## Hypothèses testées

- **A** : `year + batch_num + cam_num + sample_num`
- **B** : clé A + `label_filename`
- **C** : `year + cam_num + sample_num + label_filename` (batch ignoré)
- **D** : `year + cam_num + sample_num` (batch et label historique ignorés)

La valeur `{NO_BATCH}` est utilisée comme sentinelle stable quand `batch_num` est absent. `cam_position` sert uniquement à compter les côtés. `label_principal`, l'ordre des fichiers et l'offset vidéo +7 ne sont jamais utilisés dans les clés.

## Résultats

{_markdown_table(compact)}

Le taux représente les images appartenant à un groupe exactement 1 Top + 1 Bottom. Les catégories « plusieurs Top/Bottom » se recouvrent volontairement dans le CSV détaillé ; `ambiguous_groups` est leur union.

### Effet de `batch_num`

{_markdown_table(batch_table)}

`ambiguous_groups_resolved_by_batch_split` compte les groupes ambigus sans batch qui, une fois séparés par batch, ne contiennent plus aucun sous-groupe ambigu. Ce chiffre mesure une collision de métadonnées résolue, pas une paire physique confirmée.

### Effet de `label_filename`

{_markdown_table(label_table)}

L'ajout du diagnostic historique augmente fortement l'unicité apparente. Cela peut séparer de vraies collisions, mais aussi empêcher une association entre deux vues dont le diagnostic historique diffère. Il ne s'agit donc pas d'une preuve d'identité.

## Origine documentée de `sample_num`

La recherche dans le dépôt ne retrouve **aucun code de création ou de renommage des 35 254 images**, ni aucune règle explicite d'incrément ou de remise à zéro du compteur. Le README documente seulement le format `annee_label_Cam_{{T|B}}_{{numCamera}}_{{numEchantillon}}.jpg`. `labeling_tool/build_labels_csv.py` et `analysis/build_dataset_manifest.py` parsèrent des noms déjà existants ; ils n'attribuent pas `sample_num`. L'application de labellisation ne renomme pas les images. Le rapport PDF historique répète le format sans expliquer le compteur. Aucun historique Git exploitable n'est présent dans cette copie du dépôt.

La logique de namespace n'est donc pas démontrée par le code ou la documentation disponible ; elle doit être évaluée empiriquement.

## Vérification empirique du namespace

{_markdown_table(sequence_table)}

Les métriques détaillées pour chacun des contextes sont enregistrées dans `historical_tb_sample_num_analysis.csv`. Avec `label_filename`, 78 contextes sur 84 commencent à zéro. Sans `label_filename`, 47 contextes sur 48 commencent à zéro. La réunion de plusieurs labels bouche artificiellement certains trous : elle ne constitue donc pas un meilleur compteur unique.

### Réutilisation d'un même numéro entre labels

{_markdown_table(pd.DataFrame(reuse_rows))}

Distribution globale : {cross_label_reuse['all']['label_count_distribution']}. La réutilisation est particulièrement forte en 2026 et soutient l'hypothèse de compteurs locaux au label historique.

### Cohérence littérale des noms pour la clé B

Sur {filename_consistency['potential_b_pairs']} groupes B exactement 1T/1B, {filename_consistency['filenames_differing_only_by_cam_position']} ({filename_consistency['rate_pct']:.2f} %) ont des noms dont toutes les composantes sont identiques sauf `Cam_T` / `Cam_B`. Il y a {filename_consistency['inconsistent_pairs']} exception et {filename_consistency['invalid_filenames']} nom non conforme au format attendu.

### Huit groupes A séparés par `label_filename`

{_markdown_table(split_table)}

Ces cas associent seulement un Top et un Bottom parce que leur numéro local se rencontre dans deux labels historiques différents. Leurs noms ne suivent pas le motif symétrique T/B de la clé B ; les considérer comme des couples serait donc une collision numérique plausible, pas une association démontrée.

### Diagnostic de la clé B, meilleure au seul critère numérique

- Paires potentielles : {int(bdiag.potential_pairs)} ({float(comparison[(comparison.scope == 'all') & (comparison.key_id == 'B')].potential_pair_rate_pct.iloc[0]):.2f} % des images).
- Même `label_principal` : {int(bdiag.same_label_principal)}.
- `label_principal` différent : {int(bdiag.different_label_principal)}.
- Au moins un côté `multiple` : {int(bdiag.pairs_with_multiple)} ; différence entre côtés : {int(bdiag.multiple_differs_between_sides)}.
- Au moins un côté `chunk` : {int(bdiag.pairs_with_chunk)} ; différence entre côtés : {int(bdiag.chunk_differs_between_sides)}.
- Au moins un côté `Vide` : {int(bdiag.pairs_with_vide)} ; différence entre côtés : {int(bdiag.vide_differs_between_sides)}.

Les transitions détaillées Top → Bottom sont conservées dans `historical_tb_label_transitions.csv`.

## Structure de `sample_num`

```json
{json.dumps(sample_summary, ensure_ascii=False, indent=2)}
```

Les répétitions, ruptures et redémarrages sont détaillés par année/caméra/batch dans `historical_tb_sample_num_analysis.csv`. Elles montrent que `sample_num` est un identifiant local et largement réutilisé, insuffisant sans contexte.

## Classement et choix envisageable

Classement retenu : **B — clé fortement soutenue mais restant heuristique**.

La répétition des compteurs entre labels, leurs nombreux redémarrages à zéro et la cohérence littérale de 100 % des groupes B soutiennent fortement l'idée que `label_filename` appartient au namespace historique de `sample_num`. Toutefois, aucun générateur ni document métier disponible ne confirme cette règle et aucun identifiant physique de châtaigne ne permet de valider les groupes.

Dans ce rôle, `label_filename` n'est **ni la cible, ni une caractéristique ML**. Il sert uniquement à interpréter le namespace technique historique lors de l'audit T/B. `label_principal` demeure la cible d'entraînement et n'est jamais utilisé pour construire les groupes.

## Méthode retenue

La clé `year + batch_num + cam_num + sample_num + label_filename` est retenue pour construire les correspondances historiques lorsque le groupe contient exactement un Top et un Bottom. Elle identifie 16 176 groupes, couvre 32 352 images (91,77 %) et leurs noms sont tous strictement symétriques, seule la composante `Cam_T` / `Cam_B` différant.

### Choix

`label_filename` est conservé uniquement comme composante du namespace historique nécessaire à l'interprétation de `sample_num`. Il ne remplace jamais `label_principal`, n'est pas une cible ML et ne doit pas devenir une caractéristique d'entrée du modèle.

### Hypothèse et arguments

Le compteur `sample_num` semble local au contexte historique incluant le label du filename. Cette hypothèse est soutenue par la forte réutilisation des numéros entre labels, la symétrie de 100 % des groupes retenus, l'effet mesurable de `batch_num` en 2025 et l'absence d'ambiguïtés avec la clé B.

### Implication

Les lignes de `historical_tb_pairs.csv` peuvent servir à organiser et analyser conjointement les deux vues. Leur méthode est `historical_namespace` et leur statut `heuristic_pair`; elles ne doivent pas être présentées comme une vérité terrain physique garantie. Elles restent distinctes de `analysis/extracted_video/tb_pairs.csv`, fondé sur l'alignement temporel des vidéos.

### Critique

Le générateur historique et un identifiant physique de châtaigne restent absents. Des fichiers peuvent manquer, les comportements diffèrent entre 2025 et 2026 et une collision rare peut subsister sans être observable dans les métadonnées.

## Implications et critique

- Conserver `batch_num` évite des collisions 2025 ; l'ignorer fusionne les segments `_1`/`_2`.
- Ajouter `label_filename` améliore l'unicité, mais risque une séparation artificielle selon le diagnostic historique.
- L'absence d'identifiant physique de châtaigne empêche de prouver qu'un groupe 1T/1B représente le même objet.
- Les structures 2025 et 2026 diffèrent ; une règle unique n'a pas la même performance selon l'année.
- Des labels principaux différents entre faces sont possibles et ne doivent pas invalider automatiquement un groupe.
- Les indicateurs `multiple`, `chunk` et `Vide` signalent des groupes potentiellement moins fiables.
- Aucun offset temporel n'est supposé : la règle +7 est propre aux deux vidéos déjà traitées.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    before_hash = _sha256(manifest_path)
    df = _load_manifest(manifest_path)
    scopes = [("all", df), *[(str(year), df.loc[df.year.eq(year)]) for year in sorted(df.year.unique())]]

    comparison_rows = [
        _metrics(subset, key_id, scope)
        for scope, subset in scopes
        for key_id in KEYS
    ]
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output_dir / "historical_tb_key_comparison.csv", index=False)

    ambiguous_records: list[dict[str, object]] = []
    one_sided_records: list[dict[str, object]] = []
    for key_id in KEYS:
        groups = _group_record_table(df, key_id)
        ambiguous = groups.loc[groups.top_count.gt(1) | groups.bottom_count.gt(1)].copy()
        ambiguous["ambiguity"] = "multiple_bottom"
        ambiguous.loc[ambiguous.top_count.gt(1), "ambiguity"] = "multiple_top"
        ambiguous.loc[
            ambiguous.top_count.gt(1) & ambiguous.bottom_count.gt(1), "ambiguity"
        ] = "multiple_top_and_bottom"
        one_sided = groups.loc[
            groups.top_count.eq(0) | groups.bottom_count.eq(0)
        ].copy()
        one_sided["side"] = "bottom_only"
        one_sided.loc[one_sided.bottom_count.eq(0), "side"] = "top_only"
        ambiguous_records.extend(ambiguous.to_dict("records"))
        one_sided_records.extend(one_sided.to_dict("records"))
    pd.DataFrame(ambiguous_records).to_csv(
        output_dir / "historical_tb_ambiguous_groups.csv", index=False
    )
    pd.DataFrame(one_sided_records).to_csv(
        output_dir / "historical_tb_one_sided_groups.csv", index=False
    )

    diagnostics_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    for key_id in KEYS:
        pairs = _exact_pairs(df, key_id)
        for scope, subset in [("all", pairs), *[(str(year), pairs.loc[pairs.year.eq(year)]) for year in sorted(df.year.unique())]]:
            diagnostics_rows.append(_pair_diagnostics(subset, key_id, scope))
            transitions = subset.groupby(["top_label", "bottom_label"], dropna=False).size()
            for (top_label, bottom_label), count in transitions.items():
                transition_rows.append(
                    {"scope": scope, "key_id": key_id, "top_label_principal": top_label,
                     "bottom_label_principal": bottom_label, "count": int(count)}
                )
    diagnostics = pd.DataFrame(diagnostics_rows)
    diagnostics.to_csv(output_dir / "historical_tb_pair_diagnostics.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(
        output_dir / "historical_tb_label_transitions.csv", index=False
    )

    batch_impact = [
        _batch_impact(subset, with_batch, without_batch, scope)
        for scope, subset in scopes
        for with_batch, without_batch in [("A", "D"), ("B", "C")]
    ]
    label_effect = [
        _label_effect(subset, structural, with_label, scope)
        for scope, subset in scopes
        for structural, with_label in [("A", "B"), ("D", "C")]
    ]
    _, sample_summary = _sample_analysis(df)
    sequence_table, sequence_summary = _sequence_table(df)
    sequence_table.to_csv(output_dir / "historical_tb_sample_num_analysis.csv", index=False)
    cross_label_reuse = _cross_label_reuse(df)
    filename_consistency = _b_filename_consistency(df)
    split_cases = _a_pairs_split_by_label(df)

    documentation_audit = {
        "explicit_numbering_rule_found": False,
        "findings": [
            {
                "source": "README.md",
                "finding": "documents filename format and numEchantillon, but no counter/reset rule",
            },
            {
                "source": "labeling_tool/build_labels_csv.py",
                "finding": "parses sample_num from existing filenames; does not create or rename images",
            },
            {
                "source": "labeling_tool/app.py",
                "finding": "reads images by filename and updates labels only; no image renaming",
            },
            {
                "source": "analysis/build_dataset_manifest.py",
                "finding": "parses batch_num from existing filenames; does not assign sample_num",
            },
            {
                "source": "Rapport_Dataset_Chataignes.pdf",
                "finding": "repeats filename format; no sample counter/reset rule found",
            },
            {
                "source": ".git",
                "finding": "no commit history available in this repository copy",
            },
        ],
    }

    summary = {
        "purpose": "metadata_audit_only_no_definitive_pairing",
        "manifest": str(manifest_path),
        "manifest_sha256": before_hash,
        "images": len(df),
        "filename_unique": bool(df.filename.is_unique),
        "missing_batch_sentinel": NO_BATCH,
        "key_definitions": DISPLAY_KEYS,
        "key_comparison": comparison_rows,
        "batch_num_impact": batch_impact,
        "label_filename_effect": label_effect,
        "pair_diagnostics": diagnostics_rows,
        "sample_num": sample_summary,
        "sample_num_namespace_sequences": {
            "summary": sequence_summary,
            "details_file": "historical_tb_sample_num_analysis.csv",
        },
        "cross_label_sample_num_reuse": cross_label_reuse,
        "b_filename_consistency": filename_consistency,
        "a_exact_1t1b_split_by_label_filename": {
            "count": len(split_cases),
            "cases": split_cases,
        },
        "documentation_audit": documentation_audit,
        "conclusion": {
            "best_numerical_key": "B",
            "best_label_independent_structural_key": "A",
            "classification": "B_strongly_supported_but_still_heuristic",
            "recommendation": "use_label_filename_only_as_historical_sample_num_namespace_not_as_ml_target_or_feature",
            "ml_target": "label_principal",
        },
    }
    (output_dir / "historical_tb_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(
        output_dir / "historical_tb_audit_report.md",
        comparison,
        diagnostics,
        batch_impact,
        label_effect,
        sample_summary,
        sequence_summary,
        cross_label_reuse,
        filename_consistency,
        split_cases,
    )
    after_hash = _sha256(manifest_path)
    if before_hash != after_hash:
        raise RuntimeError("Le manifeste d'entrée a été modifié pendant l'audit")

    print(f"Audit historique T/B terminé: {len(df):,} images".replace(",", " "))
    print(comparison[["scope", "key_id", "exact_1t1b_groups", "top_only_groups",
                      "bottom_only_groups", "ambiguous_groups", "potential_pair_rate_pct"]].to_string(index=False))
    print(
        "Réutilisation inter-labels: "
        f"{cross_label_reuse['all']['slots_reused_between_labels']} / "
        f"{cross_label_reuse['all']['sample_slots']} slots "
        f"({cross_label_reuse['all']['slot_reuse_rate_pct']:.2f} %)"
    )
    print(
        "Cohérence littérale B (seul T/B diffère): "
        f"{filename_consistency['filenames_differing_only_by_cam_position']} / "
        f"{filename_consistency['potential_b_pairs']} ({filename_consistency['rate_pct']:.2f} %)"
    )
    print(f"Groupes A 1T/1B séparés par label_filename: {len(split_cases)}")
    print("Classement de B: fortement soutenue mais restant heuristique.")
    print(f"Sorties: {output_dir}")
    print("Aucune table de paires définitive n'a été créée.")


if __name__ == "__main__":
    main()
