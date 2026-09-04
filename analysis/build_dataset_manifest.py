"""Construit un manifeste d'analyse sans modifier les CSV de labels sources."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "analysis" / "output" / "dataset_manifest.csv"
PRINCIPAL_CSV = ROOT / "labels_principal.csv"
MASKED_CSV = ROOT / "labels_masked.csv"

PRINCIPAL_COLUMNS = [
    "filename",
    "year",
    "cam_position",
    "cam_num",
    "sample_num",
    "label_filename",
    "label_principal",
    "multiple",
    "chunk",
    "mixed_quality",
    "reviewed",
    "labeled_by",
]

FILENAME_PATTERN = re.compile(
    r"^(?P<year>\d{4})_(?P<label>NON_Conforme|Conforme|PIETRA)"
    r"(?:_(?P<batch>\d+))?_Cam_(?P<position>[TB])_"
    r"(?P<camera>\d+)_(?P<sample>\d+)\.jpg$"
)

LABELER_CANONICAL_MAP = {
    "NicoG": "Nico",
    "Nico": "Nico",
    "nico": "Nico",
    "nico h": "Nico",
}


def _require_columns(df: pd.DataFrame, required: list[str], source: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {source.name}: {', '.join(missing)}")


def _as_bool(series: pd.Series, column: str) -> pd.Series:
    values = series.astype("string").str.strip().str.lower()
    invalid = values[~values.isin(["true", "false"]) & values.notna()].unique()
    if len(invalid):
        raise ValueError(f"Valeurs booléennes inattendues dans {column}: {invalid.tolist()}")
    return values.eq("true")


def _extract_batch_num(filenames: pd.Series) -> pd.Series:
    batches: list[object] = []
    invalid: list[str] = []
    for filename in filenames:
        match = FILENAME_PATTERN.fullmatch(str(filename))
        if match is None:
            invalid.append(str(filename))
            batches.append(pd.NA)
        else:
            batch = match.group("batch")
            batches.append(int(batch) if batch is not None else pd.NA)

    if invalid:
        examples = ", ".join(invalid[:5])
        raise ValueError(f"{len(invalid)} filename(s) non reconnus, exemples: {examples}")
    return pd.Series(batches, index=filenames.index, dtype="Int64")


def build_manifest(output_path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    principal = pd.read_csv(
        PRINCIPAL_CSV,
        dtype={"filename": "string", "labeled_by": "string"},
    )
    _require_columns(principal, PRINCIPAL_COLUMNS, PRINCIPAL_CSV)

    if principal["filename"].duplicated().any():
        raise ValueError("filename doit rester unique dans labels_principal.csv")

    for column in ("multiple", "chunk", "mixed_quality", "reviewed"):
        principal[column] = _as_bool(principal[column], column)
    principal["labeled_by"] = principal["labeled_by"].fillna("")

    manifest = principal[PRINCIPAL_COLUMNS].copy()
    manifest["labeled_by_canonical"] = manifest["labeled_by"].replace(LABELER_CANONICAL_MAP)
    manifest.insert(5, "batch_num", _extract_batch_num(manifest["filename"]))

    masked = pd.read_csv(MASKED_CSV, dtype={"filename": "string", "label": "string"})
    _require_columns(masked, ["filename", "label", "hot_frac", "max_blob"], MASKED_CSV)
    if masked["filename"].duplicated().any():
        raise ValueError("filename doit rester unique dans labels_masked.csv")

    masked = masked[["filename", "label", "hot_frac", "max_blob"]].rename(
        columns={"label": "masked_label"}
    )
    masked["hot_frac"] = pd.to_numeric(masked["hot_frac"], errors="raise").astype("Float64")
    masked["max_blob"] = pd.to_numeric(masked["max_blob"], errors="raise").astype("Int64")
    manifest = manifest.merge(masked, on="filename", how="left", validate="one_to_one")

    manifest["masked_covered"] = manifest["masked_label"].notna()
    principal_is_empty = manifest["label_principal"].astype("string").str.casefold().eq("vide")
    masked_is_empty = manifest["masked_label"].astype("string").str.casefold().eq("vide")
    manifest["masked_disagreement"] = manifest["masked_covered"] & (
        principal_is_empty != masked_is_empty
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Chemin du manifeste (défaut: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.output.resolve())
    print(f"Manifeste créé: {args.output.resolve()}")
    print(f"Images: {len(manifest):,}".replace(",", " "))
    print(f"filename uniques: {manifest['filename'].nunique():,}".replace(",", " "))
    print(f"Images avec batch_num: {manifest['batch_num'].notna().sum():,}".replace(",", " "))
    print(f"Images couvertes par labels_masked: {manifest['masked_covered'].sum():,}".replace(",", " "))


if __name__ == "__main__":
    main()
