"""Construit les correspondances historiques T/B validées comme heuristiques.

Seuls les groupes exactement 1 Top + 1 Bottom pour la clé historique retenue
sont exportés. Aucun label ni fichier source n'est modifié.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from historical_tb_audit import (
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    FILENAME_PATTERN,
    KEYS,
    NO_BATCH,
    _exact_members,
    _group_counts,
    _load_manifest,
    _masks,
    _sha256,
)


EXPECTED_PAIRS = 16_176
PAIRING_METHOD = "historical_namespace"
PAIRING_STATUS = "heuristic_pair"

PAIR_COLUMNS = [
    "pair_id",
    "top_filename",
    "bottom_filename",
    "year",
    "batch_num",
    "cam_num",
    "sample_num",
    "label_filename",
    "top_label_principal",
    "bottom_label_principal",
    "principal_label_agreement",
    "top_multiple",
    "bottom_multiple",
    "top_chunk",
    "bottom_chunk",
    "top_mixed_quality",
    "bottom_mixed_quality",
    "has_multiple",
    "has_chunk",
    "has_vide",
    "pairing_method",
    "pairing_status",
]


def _pair_id(row: pd.Series) -> str:
    canonical_key = "\x1f".join(
        [
            str(int(row.year)),
            str(row.batch_key),
            str(int(row.cam_num)),
            str(int(row.sample_num)),
            str(row.label_filename),
        ]
    )
    return "hist_" + hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:24]


def _validate_filename(filename: str, expected: dict[str, object], position: str) -> None:
    match = FILENAME_PATTERN.fullmatch(filename)
    if match is None:
        raise ValueError(f"Filename hors format dans une paire: {filename}")
    parts = match.groupdict()
    parsed_batch = parts["batch"] if parts["batch"] is not None else NO_BATCH
    observed = {
        "year": int(parts["year"]),
        "batch_key": parsed_batch,
        "cam_num": int(parts["camera"]),
        "sample_num": int(parts["sample"]),
        "label_filename": parts["label"].replace("_", " "),
        "cam_position": parts["position"],
    }
    wanted = {**expected, "cam_position": position}
    if observed != wanted:
        raise ValueError(
            f"Composantes incohérentes pour {filename}: observé={observed}, attendu={wanted}"
        )


def _build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    key_columns = KEYS["B"]
    members = _exact_members(df, "B")
    fields = key_columns + [
        "filename", "label_principal", "multiple", "chunk", "mixed_quality"
    ]
    top = members.loc[members.cam_position.eq("T"), fields].rename(
        columns={
            "filename": "top_filename",
            "label_principal": "top_label_principal",
            "multiple": "top_multiple",
            "chunk": "top_chunk",
            "mixed_quality": "top_mixed_quality",
        }
    )
    bottom = members.loc[members.cam_position.eq("B"), fields].rename(
        columns={
            "filename": "bottom_filename",
            "label_principal": "bottom_label_principal",
            "multiple": "bottom_multiple",
            "chunk": "bottom_chunk",
            "mixed_quality": "bottom_mixed_quality",
        }
    )
    pairs = top.merge(bottom, on=key_columns, how="inner", validate="one_to_one")
    if len(pairs) != EXPECTED_PAIRS:
        raise RuntimeError(
            f"Nombre de paires inattendu: {len(pairs)} au lieu de {EXPECTED_PAIRS}. "
            "Aucune sortie n'a été écrite."
        )

    pairs["pair_id"] = pairs.apply(_pair_id, axis=1)
    pairs["batch_num"] = pairs.batch_key.map(
        lambda value: pd.NA if value == NO_BATCH else int(value)
    ).astype("Int64")
    pairs["principal_label_agreement"] = pairs.top_label_principal.eq(
        pairs.bottom_label_principal
    )
    pairs["has_multiple"] = pairs.top_multiple | pairs.bottom_multiple
    pairs["has_chunk"] = pairs.top_chunk | pairs.bottom_chunk
    pairs["has_vide"] = (
        pairs.top_label_principal.astype("string").str.casefold().eq("vide")
        | pairs.bottom_label_principal.astype("string").str.casefold().eq("vide")
    )
    pairs["pairing_method"] = PAIRING_METHOD
    pairs["pairing_status"] = PAIRING_STATUS
    return pairs[PAIR_COLUMNS].sort_values(
        ["year", "batch_num", "cam_num", "sample_num", "label_filename"],
        na_position="first",
    ).reset_index(drop=True)


def _validate_pairs(df: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, bool]:
    validations = {
        "expected_pair_count": len(pairs) == EXPECTED_PAIRS,
        "pair_id_unique": bool(pairs.pair_id.is_unique),
        "top_filename_unique": bool(pairs.top_filename.is_unique),
        "bottom_filename_unique": bool(pairs.bottom_filename.is_unique),
        "no_image_used_twice": bool(pd.concat(
            [pairs.top_filename, pairs.bottom_filename], ignore_index=True
        ).is_unique),
        "all_top_in_manifest": bool(pairs.top_filename.isin(df.filename).all()),
        "all_bottom_in_manifest": bool(pairs.bottom_filename.isin(df.filename).all()),
    }
    if not all(validations.values()):
        failed = [name for name, passed in validations.items() if not passed]
        raise RuntimeError(f"Validations d'unicité/référence échouées: {', '.join(failed)}")

    indexed = df.set_index("filename", verify_integrity=True)
    for row in pairs.itertuples(index=False):
        if indexed.at[row.top_filename, "cam_position"] != "T":
            raise ValueError(f"La vue Top n'est pas T: {row.top_filename}")
        if indexed.at[row.bottom_filename, "cam_position"] != "B":
            raise ValueError(f"La vue Bottom n'est pas B: {row.bottom_filename}")
        batch_key = NO_BATCH if pd.isna(row.batch_num) else str(int(row.batch_num))
        expected = {
            "year": int(row.year),
            "batch_key": batch_key,
            "cam_num": int(row.cam_num),
            "sample_num": int(row.sample_num),
            "label_filename": row.label_filename,
        }
        _validate_filename(row.top_filename, expected, "T")
        _validate_filename(row.bottom_filename, expected, "B")

        top_normalized = row.top_filename.replace("_Cam_T_", "_Cam_{POSITION}_", 1)
        bottom_normalized = row.bottom_filename.replace("_Cam_B_", "_Cam_{POSITION}_", 1)
        if top_normalized != bottom_normalized:
            raise ValueError(
                "Les filenames ne diffèrent pas uniquement par T/B: "
                f"{row.top_filename} / {row.bottom_filename}"
            )

    validations.update(
        {
            "exactly_one_top_and_one_bottom": True,
            "historical_key_components_match": True,
            "filenames_differ_only_by_position": True,
        }
    )
    return validations


def _build_unpaired(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    paired_filenames = set(pairs.top_filename) | set(pairs.bottom_filename)
    unpaired = df.loc[~df.filename.isin(paired_filenames)].copy()
    group_counts = _group_counts(df, KEYS["B"]).reset_index().rename(
        columns={"T": "group_top_count", "B": "group_bottom_count"}
    )
    unpaired = unpaired.merge(group_counts, on=KEYS["B"], validate="many_to_one")
    unpaired["unpaired_reason"] = "ambiguous_group"
    unpaired.loc[unpaired.group_bottom_count.eq(0), "unpaired_reason"] = "top_only"
    unpaired.loc[unpaired.group_top_count.eq(0), "unpaired_reason"] = "bottom_only"
    unpaired["pairing_method"] = PAIRING_METHOD
    unpaired["pairing_status"] = "unpaired"
    columns = [
        "filename", "year", "batch_num", "cam_position", "cam_num", "sample_num",
        "label_filename", "label_principal", "multiple", "chunk", "mixed_quality",
        "group_top_count", "group_bottom_count", "unpaired_reason", "pairing_method",
        "pairing_status",
    ]
    return unpaired[columns].sort_values("filename").reset_index(drop=True)


def _summary(
    df: pd.DataFrame,
    pairs: pd.DataFrame,
    unpaired: pd.DataFrame,
    validations: dict[str, bool],
    manifest_hash: str,
) -> dict[str, object]:
    same = int(pairs.principal_label_agreement.sum())
    different = len(pairs) - same
    return {
        "pairing_method": PAIRING_METHOD,
        "pairing_status": PAIRING_STATUS,
        "historical_key": ["year", "batch_num", "cam_num", "sample_num", "label_filename"],
        "qualification": "strongly_supported_but_still_heuristic",
        "manifest_sha256": manifest_hash,
        "dataset_images": len(df),
        "pairs": len(pairs),
        "covered_images": 2 * len(pairs),
        "dataset_coverage_pct": round(100 * 2 * len(pairs) / len(df), 2),
        "same_label_principal_pairs": same,
        "different_label_principal_pairs": different,
        "label_principal_agreement_rate_pct": round(100 * same / len(pairs), 2),
        "pairs_with_multiple": int(pairs.has_multiple.sum()),
        "pairs_with_chunk": int(pairs.has_chunk.sum()),
        "pairs_with_vide": int(pairs.has_vide.sum()),
        "unpaired_top_images": int(unpaired.cam_position.eq("T").sum()),
        "unpaired_bottom_images": int(unpaired.cam_position.eq("B").sum()),
        "unpaired_images_total": len(unpaired),
        "unpaired_reason_distribution": {
            reason: int(count) for reason, count in unpaired.unpaired_reason.value_counts().items()
        },
        "validations": validations,
        "ml_semantics": {
            "label_filename": "historical_namespace_only_not_target_or_feature",
            "label_principal": "training_target_preserved_per_image",
            "pair_level_final_label_created": False,
        },
        "video_pairing_kept_separate": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()

    before_hash = _sha256(manifest_path)
    df = _load_manifest(manifest_path)
    pairs = _build_pairs(df)
    validations = _validate_pairs(df, pairs)
    unpaired = _build_unpaired(df, pairs)
    if len(pairs) * 2 + len(unpaired) != len(df):
        raise RuntimeError("La partition images appariées/non appariées est incohérente")
    summary = _summary(df, pairs, unpaired, validations, before_hash)
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)

    if _sha256(manifest_path) != before_hash:
        raise RuntimeError("Le manifeste a changé pendant le calcul; aucune sortie n'est écrite")

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(output_dir / "historical_tb_pairs.csv", index=False)
    unpaired.to_csv(output_dir / "historical_tb_unpaired.csv", index=False)
    (output_dir / "historical_tb_pairs_summary.json").write_text(
        summary_text, encoding="utf-8"
    )

    print(f"Paires historiques heuristiques: {len(pairs):,}".replace(",", " "))
    print(f"Images couvertes: {2 * len(pairs):,} / {len(df):,} ({summary['dataset_coverage_pct']:.2f} %)".replace(",", " "))
    print(f"Labels principaux identiques / différents: {summary['same_label_principal_pairs']:,} / {summary['different_label_principal_pairs']:,}".replace(",", " "))
    print(f"Avec multiple / chunk / Vide: {summary['pairs_with_multiple']:,} / {summary['pairs_with_chunk']:,} / {summary['pairs_with_vide']:,}".replace(",", " "))
    print(f"Top / Bottom non appariés: {summary['unpaired_top_images']:,} / {summary['unpaired_bottom_images']:,}".replace(",", " "))
    print("Toutes les validations 1T/1B, unicité, manifeste, clé et filename sont passées.")


if __name__ == "__main__":
    main()
