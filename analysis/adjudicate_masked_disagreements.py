"""Dérive les actions de revue masked sans modifier les labels sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "analysis" / "output"
SOURCE_CSV = OUTPUT_DIR / "masked_disagreements.csv"
REVIEW_CSV = OUTPUT_DIR / "masked_disagreements_review.csv"
MULTICLASS_REVIEW_CSV = OUTPUT_DIR / "multiclass_conflicts_review.csv"
ADJUDICATED_CSV = OUTPUT_DIR / "masked_disagreements_adjudicated.csv"
SUMMARY_JSON = OUTPUT_DIR / "masked_disagreements_adjudication_summary.json"
EXPECTED_ROWS = 59

PRINCIPAL_BINARY_MAP = {
    "Vide": "Vide",
    "Conforme": "NonVide",
    "NON Conforme": "NonVide",
    "PIETRA": "NonVide",
}
MASKED_BINARY_MAP = {
    "vide": "Vide",
    "chataigne": "NonVide",
    "chunks": "NonVide",
    "multiple": "NonVide",
}
MULTICLASS_LABELS = {"Conforme", "NON Conforme", "PIETRA"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify(row: pd.Series) -> str:
    if row.human_review == "Incertain":
        return "EXCLUDE_UNCERTAIN"
    if row.human_review == row.principal_binary:
        return "KEEP_PRINCIPAL"
    if row.human_review == "Vide" and row.principal_binary == "NonVide":
        return "CORRECT_TO_VIDE"
    if row.human_review == "NonVide" and row.principal_binary == "Vide":
        return "NEED_MULTICLASS_REVIEW"
    raise ValueError(f"Combinaison de revue non classable pour {row.filename}")


def load_and_adjudicate(source_path: Path, review_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    hashes = {source_path.name: sha256(source_path), review_path.name: sha256(review_path)}
    source = pd.read_csv(source_path, low_memory=False)
    review = pd.read_csv(review_path, dtype="string", keep_default_na=False)
    if len(source) != EXPECTED_ROWS or len(review) != EXPECTED_ROWS:
        raise ValueError("Les deux fichiers doivent contenir exactement 59 lignes")
    if not source.filename.is_unique or not review.filename.is_unique:
        raise ValueError("filename doit être unique dans les fichiers de revue")
    if set(source.filename) != set(review.filename):
        raise ValueError("Les filenames source et revue ne correspondent pas")
    if not review.review_status.eq("reviewed").all() or review.human_review.eq("").any():
        raise ValueError("La première revue humaine n'est pas terminée")
    invalid_human = sorted(set(review.human_review) - {"Vide", "NonVide", "Incertain"})
    if invalid_human:
        raise ValueError(f"Décisions humaines inattendues: {invalid_human}")

    merged = source.merge(
        review[[
            "filename", "label_principal_before", "masked_label", "human_review",
            "review_status", "notes", "reviewed_at_utc",
        ]],
        on="filename", how="inner", suffixes=("", "_review"), validate="one_to_one",
    )
    if not merged.label_principal.eq(merged.label_principal_before).all():
        raise ValueError("label_principal_before ne correspond plus au fichier d'analyse")
    if not merged.masked_label.eq(merged.masked_label_review).all():
        raise ValueError("masked_label ne correspond plus au fichier d'analyse")
    if not merged.label_principal.isin(PRINCIPAL_BINARY_MAP).all():
        raise ValueError("label_principal inattendu")
    masked_normalized = merged.masked_label.str.casefold()
    if not masked_normalized.isin(MASKED_BINARY_MAP).all():
        raise ValueError("masked_label inattendu")

    merged["principal_binary"] = merged.label_principal.map(PRINCIPAL_BINARY_MAP)
    merged["masked_binary"] = masked_normalized.map(MASKED_BINARY_MAP)
    merged["human_agrees_principal"] = merged.human_review.eq(merged.principal_binary)
    merged["human_agrees_masked"] = merged.human_review.eq(merged.masked_binary)
    determinate = merged.human_review.isin(["Vide", "NonVide"])
    merged["human_disagrees_both"] = (
        determinate & ~merged.human_agrees_principal & ~merged.human_agrees_masked
    )
    merged["review_action"] = merged.apply(classify, axis=1)
    merged["effective_label"] = pd.NA
    keep = merged.review_action.eq("KEEP_PRINCIPAL")
    merged.loc[keep, "effective_label"] = merged.loc[keep, "label_principal"]
    merged.loc[merged.review_action.eq("CORRECT_TO_VIDE"), "effective_label"] = "Vide"
    merged["multiclass_human_review"] = ""
    merged["multiclass_review_status"] = "not_required"

    needs_multiclass = merged.review_action.eq("NEED_MULTICLASS_REVIEW")
    merged.loc[needs_multiclass, "multiclass_review_status"] = "pending"
    if MULTICLASS_REVIEW_CSV.exists():
        hashes[MULTICLASS_REVIEW_CSV.name] = sha256(MULTICLASS_REVIEW_CSV)
        multiclass = pd.read_csv(MULTICLASS_REVIEW_CSV, dtype="string", keep_default_na=False)
        required = {"filename", "human_review", "review_status"}
        if not required.issubset(multiclass.columns) or not multiclass.filename.is_unique:
            raise ValueError("Fichier de revue multiclasses invalide")
        if set(multiclass.filename) != set(merged.loc[needs_multiclass, "filename"]):
            raise ValueError("La revue multiclasses ne contient pas exactement les cas requis")
        merged = merged.merge(
            multiclass[["filename", "human_review", "review_status"]].rename(
                columns={
                    "human_review": "multiclass_decision",
                    "review_status": "multiclass_status_file",
                }
            ),
            on="filename", how="left", validate="one_to_one",
        )
        merged.loc[needs_multiclass, "multiclass_human_review"] = merged.loc[
            needs_multiclass, "multiclass_decision"
        ].fillna("")
        merged.loc[needs_multiclass, "multiclass_review_status"] = merged.loc[
            needs_multiclass, "multiclass_status_file"
        ].fillna("pending")
        determined = needs_multiclass & merged.multiclass_human_review.isin(MULTICLASS_LABELS)
        merged.loc[determined, "effective_label"] = merged.loc[
            determined, "multiclass_human_review"
        ]
        merged = merged.drop(columns=["multiclass_decision", "multiclass_status_file"])

    return merged, hashes


def build_summary(df: pd.DataFrame, hashes: dict[str, str]) -> dict[str, object]:
    action_counts = df.review_action.value_counts().reindex(
        [
            "KEEP_PRINCIPAL", "CORRECT_TO_VIDE", "NEED_MULTICLASS_REVIEW",
            "EXCLUDE_UNCERTAIN",
        ],
        fill_value=0,
    )
    reusable = df.effective_label.notna()
    unresolved_multiclass = (
        df.review_action.eq("NEED_MULTICLASS_REVIEW") & ~reusable
    )
    uncertain = df.review_action.eq("EXCLUDE_UNCERTAIN")
    return {
        "source_hashes": hashes,
        "reviewed_images": len(df),
        "human_review_distribution": {
            value: int(df.human_review.eq(value).sum())
            for value in ("Vide", "NonVide", "Incertain")
        },
        "agreement_with_label_principal": int(df.human_agrees_principal.sum()),
        "agreement_with_labels_masked": int(df.human_agrees_masked.sum()),
        "human_supports_principal_against_masked": int(
            (df.human_agrees_principal & ~df.human_agrees_masked).sum()
        ),
        "human_supports_masked_against_principal": int(
            (df.human_agrees_masked & ~df.human_agrees_principal).sum()
        ),
        "human_disagrees_with_both": int(df.human_disagrees_both.sum()),
        "action_counts": {action: int(count) for action, count in action_counts.items()},
        "multiclass_review_required": int(
            df.review_action.eq("NEED_MULTICLASS_REVIEW").sum()
        ),
        "multiclass_review_determined": int(
            df.multiclass_human_review.isin(MULTICLASS_LABELS).sum()
        ),
        "multiclass_review_still_pending": int(unresolved_multiclass.sum()),
        "s1_reviewed": {
            "status": "provisional_until_multiclass_review_is_complete"
            if unresolved_multiclass.any() else "ready_for_final_human_validation",
            "reintegrated_from_59": int(reusable.sum()),
            "excluded_from_59": int((~reusable).sum()),
            "excluded_uncertain": int(uncertain.sum()),
            "excluded_pending_multiclass": int(unresolved_multiclass.sum()),
            "dataset_images_if_applied_now": 35_254 - int((~reusable).sum()),
            "derived_corrections_to_vide": int(
                df.review_action.eq("CORRECT_TO_VIDE").sum()
            ),
            "source_labels_modified": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_CSV)
    parser.add_argument("--review", type=Path, default=REVIEW_CSV)
    parser.add_argument("--output", type=Path, default=ADJUDICATED_CSV)
    parser.add_argument("--summary", type=Path, default=SUMMARY_JSON)
    args = parser.parse_args()

    source_path = args.source.resolve()
    review_path = args.review.resolve()
    adjudicated, hashes = load_and_adjudicate(source_path, review_path)
    summary = build_summary(adjudicated, hashes)
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)

    paths_to_check = [source_path, review_path]
    if MULTICLASS_REVIEW_CSV.name in hashes:
        paths_to_check.append(MULTICLASS_REVIEW_CSV)
    if any(sha256(path) != hashes[path.name] for path in paths_to_check):
        raise RuntimeError("Une entrée a changé pendant l'adjudication")

    columns = [
        "filename", "year", "cam_position", "cam_num", "sample_num",
        "label_principal", "masked_label", "human_review", "review_action",
        "effective_label", "notes", "multiclass_human_review",
        "multiclass_review_status", "human_agrees_principal", "human_agrees_masked",
        "human_disagrees_both",
    ]
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    adjudicated[columns].to_csv(args.output.resolve(), index=False)
    args.summary.resolve().write_text(summary_text, encoding="utf-8")

    print(f"Décisions analysées: {len(adjudicated)}")
    print("Humain:", summary["human_review_distribution"])
    print("Actions:", summary["action_counts"])
    print(f"Seconde revue multiclasses requise: {summary['multiclass_review_required']}")
    print(
        "S1_REVIEWED provisoire: "
        f"{summary['s1_reviewed']['dataset_images_if_applied_now']} images; "
        f"{summary['s1_reviewed']['excluded_pending_multiclass']} cas multiclasses en attente"
    )


if __name__ == "__main__":
    main()
