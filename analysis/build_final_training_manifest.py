"""Construit le manifeste ML final dérivé après les deux revues humaines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from historical_tb_audit import DEFAULT_MANIFEST, DEFAULT_OUTPUT_DIR, _as_bool, _load_manifest, _sha256


ROOT = Path(__file__).resolve().parents[1]
ADJUDICATED_CSV = DEFAULT_OUTPUT_DIR / "masked_disagreements_adjudicated.csv"
PAIRS_CSV = DEFAULT_OUTPUT_DIR / "historical_tb_pairs.csv"
MULTICLASS_REVIEW_CSV = DEFAULT_OUTPUT_DIR / "multiclass_conflicts_review.csv"
OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "final_training_manifest.csv"
SUMMARY_JSON = DEFAULT_OUTPUT_DIR / "final_training_manifest_summary.json"
REPORT_MD = DEFAULT_OUTPUT_DIR / "final_training_manifest_report.md"
TRANSITIONS_CSV = DEFAULT_OUTPUT_DIR / "final_label_transitions.csv"

EXPECTED_IMAGES = 35_254
EXPECTED_PAIRS = 16_176
EXPECTED_PAIRED_IMAGES = 32_352
EXPECTED_UNPAIRED_IMAGES = 2_902
VALID_LABELS = ["Conforme", "NON Conforme", "PIETRA", "Vide"]
VALID_SOURCES = {
    "principal", "principal_confirmed_by_human", "human_binary_review",
    "human_multiclass_review",
}
VALID_ADJUDICATION_STATUSES = {
    "not_required", "confirmed", "corrected_to_vide", "corrected_multiclass"
}

FINAL_COLUMNS = [
    "filename", "year", "batch_num", "cam_position", "cam_num", "sample_num",
    "label_filename", "label_principal", "effective_label", "effective_label_source",
    "adjudication_status", "multiple", "chunk", "mixed_quality", "reviewed",
    "labeled_by", "labeled_by_canonical", "masked_covered", "masked_label",
    "hot_frac", "max_blob", "masked_disagreement", "historical_pair_id",
    "historical_pairing_status", "split_group_id",
]


def deterministic_unpaired_group(filename: str) -> str:
    return "image_" + hashlib.sha256(filename.encode("utf-8")).hexdigest()[:24]


def distribution(series: pd.Series) -> dict[str, dict[str, float | int]]:
    counts = series.value_counts(dropna=False)
    total = len(series)
    return {
        str(value): {
            "count": int(count),
            "percentage": round(100 * int(count) / total, 2) if total else 0.0,
        }
        for value, count in counts.items()
    }


def label_comparison(df: pd.DataFrame, group_column: str | None = None) -> dict[str, object]:
    groups = [("all", df)] if group_column is None else df.groupby(group_column, dropna=False)
    result: dict[str, object] = {}
    for group, subset in groups:
        rows: dict[str, object] = {}
        total = len(subset)
        for label in VALID_LABELS:
            original = int(subset.label_principal.eq(label).sum())
            effective = int(subset.effective_label.eq(label).sum())
            original_pct = round(100 * original / total, 4) if total else 0.0
            effective_pct = round(100 * effective / total, 4) if total else 0.0
            rows[label] = {
                "original_count": original,
                "effective_count": effective,
                "delta_count": effective - original,
                "original_percentage": original_pct,
                "effective_percentage": effective_pct,
                "delta_percentage_points": round(effective_pct - original_pct, 4),
            }
        result[str(group)] = rows
    return result


def load_inputs(
    manifest_path: Path, adjudicated_path: Path, pairs_path: Path, multiclass_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    paths = [manifest_path, adjudicated_path, pairs_path, multiclass_path]
    hashes = {path.name: _sha256(path) for path in paths}
    manifest = _load_manifest(manifest_path)
    for column in ("reviewed", "masked_covered", "masked_disagreement"):
        manifest[column] = _as_bool(manifest[column], column)
    adjudicated = pd.read_csv(adjudicated_path, dtype="string", keep_default_na=False)
    pairs = pd.read_csv(pairs_path, low_memory=False)
    multiclass = pd.read_csv(multiclass_path, dtype="string", keep_default_na=False)
    return manifest, adjudicated, pairs, hashes


def apply_adjudication(manifest: pd.DataFrame, adjudicated: pd.DataFrame) -> pd.DataFrame:
    required = {
        "filename", "label_principal", "human_review", "review_action", "effective_label",
        "multiclass_human_review", "multiclass_review_status",
    }
    if not required.issubset(adjudicated.columns) or not adjudicated.filename.is_unique:
        raise ValueError("Fichier d'adjudication invalide")
    if len(adjudicated) != 59 or not set(adjudicated.filename).issubset(set(manifest.filename)):
        raise ValueError("L'adjudication doit contenir les 59 conflits du manifeste")
    original_by_filename = manifest.set_index("filename").label_principal
    observed_original = adjudicated.filename.map(original_by_filename)
    if not observed_original.eq(adjudicated.label_principal).all():
        raise ValueError("Les labels principaux de l'adjudication ne correspondent pas au manifeste")
    if adjudicated.effective_label.eq("").any() or not adjudicated.effective_label.isin(VALID_LABELS).all():
        raise ValueError("Une adjudication reste sans effective_label valide")
    if adjudicated.human_review.eq("Incertain").any():
        raise ValueError("Une adjudication binaire reste Incertain")
    multiclass_cases = adjudicated.review_action.eq("NEED_MULTICLASS_REVIEW")
    if (
        adjudicated.loc[multiclass_cases, "multiclass_review_status"].ne("reviewed").any()
        or adjudicated.loc[multiclass_cases, "multiclass_human_review"].eq("Incertain").any()
        or adjudicated.loc[multiclass_cases, "multiclass_human_review"].eq("").any()
    ):
        raise ValueError("La seconde revue multiclasses n'est pas entièrement déterminée")

    result = manifest.copy()
    result["effective_label"] = result.label_principal
    result["effective_label_source"] = "principal"
    result["adjudication_status"] = "not_required"
    effective_map = adjudicated.set_index("filename").effective_label
    action_map = adjudicated.set_index("filename").review_action
    reviewed_names = result.filename.isin(adjudicated.filename)
    result.loc[reviewed_names, "effective_label"] = result.loc[
        reviewed_names, "filename"
    ].map(effective_map)

    source_by_action = {
        "KEEP_PRINCIPAL": "principal_confirmed_by_human",
        "CORRECT_TO_VIDE": "human_binary_review",
        "NEED_MULTICLASS_REVIEW": "human_multiclass_review",
    }
    status_by_action = {
        "KEEP_PRINCIPAL": "confirmed",
        "CORRECT_TO_VIDE": "corrected_to_vide",
        "NEED_MULTICLASS_REVIEW": "corrected_multiclass",
    }
    actions = result.loc[reviewed_names, "filename"].map(action_map)
    if not actions.isin(source_by_action).all():
        raise ValueError("Action d'adjudication finale inattendue")
    result.loc[reviewed_names, "effective_label_source"] = actions.map(source_by_action).values
    result.loc[reviewed_names, "adjudication_status"] = actions.map(status_by_action).values
    return result


def add_historical_groups(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    required = {"pair_id", "top_filename", "bottom_filename", "pairing_status"}
    if not required.issubset(pairs.columns):
        raise ValueError("Colonnes manquantes dans historical_tb_pairs.csv")
    if len(pairs) != EXPECTED_PAIRS or not pairs.pair_id.is_unique:
        raise ValueError("La table T/B ne contient pas 16 176 pair_id uniques")
    if not pairs.pairing_status.eq("heuristic_pair").all():
        raise ValueError("Statut T/B historique inattendu")
    members = pd.concat(
        [
            pairs[["pair_id", "top_filename"]].rename(columns={"top_filename": "filename"}),
            pairs[["pair_id", "bottom_filename"]].rename(columns={"bottom_filename": "filename"}),
        ],
        ignore_index=True,
    ).rename(columns={"pair_id": "historical_pair_id"})
    if len(members) != EXPECTED_PAIRED_IMAGES or not members.filename.is_unique:
        raise ValueError("Une image apparaît dans plusieurs paires historiques")
    result = df.merge(members, on="filename", how="left", validate="one_to_one")
    result["historical_pairing_status"] = "unpaired"
    result.loc[result.historical_pair_id.notna(), "historical_pairing_status"] = "heuristic_pair"
    result["split_group_id"] = result.historical_pair_id.astype("string")
    unpaired = result.historical_pair_id.isna()
    result.loc[unpaired, "split_group_id"] = result.loc[unpaired, "filename"].map(
        deterministic_unpaired_group
    )
    return result


def pair_label_metrics(df: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, int]:
    labels = df.set_index("filename").effective_label
    top = pairs.top_filename.map(labels)
    bottom = pairs.bottom_filename.map(labels)
    same = top.eq(bottom)
    return {
        "pairs": len(pairs),
        "pairs_same_effective_label": int(same.sum()),
        "pairs_different_effective_label": int((~same).sum()),
    }


def validate_final(df: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, bool]:
    paired = df.historical_pair_id.notna()
    unpaired = ~paired
    pair_group_sizes = df.loc[paired].groupby("historical_pair_id").size()
    unpaired_group_sizes = df.loc[unpaired].groupby("split_group_id").size()
    stripped_labels = df.effective_label.astype("string").str.strip()
    validations = {
        "row_count_35254": len(df) == EXPECTED_IMAGES,
        "filename_unique_35254": df.filename.nunique() == EXPECTED_IMAGES,
        "effective_label_populated_35254": int(df.effective_label.notna().sum()) == EXPECTED_IMAGES
        and stripped_labels.ne("").all(),
        "exactly_four_effective_labels": set(df.effective_label) == set(VALID_LABELS),
        "effective_label_exact_spelling": df.effective_label.isin(VALID_LABELS).all()
        and df.effective_label.eq(stripped_labels).all()
        and ~df.effective_label.eq("NON_Conforme").any(),
        "no_image_lost": len(df) == EXPECTED_IMAGES,
        "no_image_duplicated": df.filename.is_unique,
        "no_pending_adjudication": ~df.adjudication_status.astype("string").str.contains(
            "pending", case=False, na=False
        ).any(),
        "no_uncertain_adjudication": ~df.adjudication_status.astype("string").str.contains(
            "uncertain", case=False, na=False
        ).any(),
        "effective_label_sources_valid": set(df.effective_label_source).issubset(VALID_SOURCES),
        "adjudication_statuses_valid": set(df.adjudication_status).issubset(
            VALID_ADJUDICATION_STATUSES
        ),
        "historical_paired_images_32352": int(paired.sum()) == EXPECTED_PAIRED_IMAGES,
        "historical_unpaired_images_2902": int(unpaired.sum()) == EXPECTED_UNPAIRED_IMAGES,
        "historical_pair_ids_16176": df.historical_pair_id.nunique() == EXPECTED_PAIRS,
        "each_pair_has_two_members": len(pair_group_sizes) == EXPECTED_PAIRS
        and pair_group_sizes.eq(2).all(),
        "pair_members_share_split_group_id": df.loc[paired].groupby(
            "historical_pair_id"
        ).split_group_id.nunique().eq(1).all(),
        "no_pair_id_shared_between_pairs": pairs.pair_id.is_unique,
        "each_unpaired_image_has_own_split_group_id": len(unpaired_group_sizes)
        == EXPECTED_UNPAIRED_IMAGES and unpaired_group_sizes.eq(1).all(),
        "all_split_groups_are_disjoint": df.split_group_id.nunique()
        == EXPECTED_PAIRS + EXPECTED_UNPAIRED_IMAGES,
    }
    return {name: bool(value) for name, value in validations.items()}


def markdown_table(df: pd.DataFrame) -> str:
    printable = df.fillna("").astype(str)
    header = "| " + " | ".join(printable.columns) + " |"
    separator = "| " + " | ".join("---" for _ in printable.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in printable.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *body])


def distribution_table(values: dict[str, dict[str, object]], column: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{column: value, "count": data["count"], "percentage": data["percentage"]}
         for value, data in values.items()]
    )


def write_report(path: Path, summary: dict[str, object], transitions: pd.DataFrame) -> None:
    validations = pd.DataFrame(
        [{"validation": name, "result": "PASS" if passed else "FAIL"}
         for name, passed in summary["validations"].items()]
    )
    report = f"""# Manifeste final d'entraînement — §4.1

## Dataset final

Le manifeste final contient {summary['total_images']} images, toutes conservées. La distribution est calculée sur `effective_label` :

{markdown_table(distribution_table(summary['effective_label_distribution'], 'effective_label'))}

## Construction de effective_label

- Images hors des 59 conflits : reprise de `label_principal` (`principal`).
- `KEEP_PRINCIPAL` : reprise de `label_principal`, confirmée par la revue humaine.
- `CORRECT_TO_VIDE` : proposition dérivée `Vide` issue de la revue binaire.
- Cas NonVide ambigus : décision déterminée de la seconde revue multiclasses.

Les sources `labels_principal.csv` et `labels_masked.csv` restent intactes. `label_principal` est conservé à côté de `effective_label` pour assurer la traçabilité.

## Revue humaine

35 254 images → 59 conflits détectés → revue binaire en aveugle → 9 revues multiclasses en aveugle → 0 cas indéterminé. La revue multiclasses a produit 3 Conforme, 5 NON Conforme et 1 PIETRA.

Repères opérationnels utilisés pendant la revue humaine — et non définitions réglementaires officielles :

- **Conforme** : fruit clair, sain visuellement, sans moisissure ni défaut important.
- **NON Conforme** : fruit très noir, fortement dégradé, éclaté ou présentant des défauts importants.
- **PIETRA** : état intermédiaire entre Conforme et NON Conforme.

## Choix

Toutes les images sont conservées. Seuls les 16 cas explicitement corrigés par les revues humaines reçoivent un `effective_label` différent de `label_principal`.

- `multiple` : conservé.
- `chunk` : conservé.
- `mixed_quality` : conservé, même si aucune valeur True n'est présente actuellement.
- `missing labeled_by` : conservé ; l'absence de provenance est un problème de traçabilité, pas une preuve d'erreur.

Les simulations ont montré que retirer `chunk`, `multiple` ou les provenances manquantes introduisait des biais importants.

## Transitions de labels

{markdown_table(transitions)}

## Statistiques complémentaires

Années :

{markdown_table(distribution_table(summary['year_distribution'], 'year'))}

Positions :

{markdown_table(distribution_table(summary['position_distribution'], 'position'))}

Caméras :

{markdown_table(distribution_table(summary['camera_distribution'], 'camera'))}

Provenance de `effective_label` :

{markdown_table(distribution_table(summary['effective_label_source_distribution'], 'source'))}

## Appariement T/B historique et futur split

{summary['historical_tb']['paired_images']} images appartiennent à {summary['historical_tb']['pairs']} paires historiques heuristiques et {summary['historical_tb']['unpaired_images']} restent non appariées. Parmi les paires, {summary['historical_tb']['pairs_same_effective_label']} ont le même label effectif et {summary['historical_tb']['pairs_different_effective_label']} des labels différents.

`split_group_id` prépare le futur split sans le créer : les deux vues d'une paire partagent le même groupe ; chaque image non appariée possède un groupe déterministe propre. L'appariement historique reste heuristique et ne constitue pas une vérité physique certaine.

## Données vidéo

Les 52 crops de `analysis/extracted_video/` ne sont pas intégrés au manifeste ML historique. Ils n'ont pas nécessairement suivi le même protocole de labellisation quatre classes ; le mot « conforme » dans le nom des AVI ne constitue pas un label.

## Implication

Le dataset final conserve les situations difficiles observées en production tout en appliquant uniquement des corrections humaines ciblées et traçables.

## Point satisfaisant

- 100 % des images conservées ;
- corrections ciblées ;
- traçabilité complète dans les artefacts dérivés ;
- sources originales intactes.

## Critique

- certaines corrections reposent sur une revue humaine subjective ;
- l'appariement historique T/B est heuristique ;
- la provenance `labeled_by` reste incomplète ;
- 2025 et 2026 sont hétérogènes ;
- `mixed_quality` n'est actuellement pas utilisé ;
- `chunk` et `multiple` rendent le problème plus difficile, mais plus réaliste.

## Lien avec le cahier des charges

Pour préparer le §4.2, la classe Conforme devra atteindre un rappel ≥ 85 % et une précision ≥ 95 %. Une erreur NON Conforme / PIETRA → Conforme est plus critique métier qu'un Conforme → rejet. Aucun label, poids de classe, split ou modèle n'est modifié ici.

## Validations techniques

{markdown_table(validations)}
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--adjudicated", type=Path, default=ADJUDICATED_CSV)
    parser.add_argument("--pairs", type=Path, default=PAIRS_CSV)
    parser.add_argument("--multiclass-review", type=Path, default=MULTICLASS_REVIEW_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    args = parser.parse_args()

    input_paths = [
        args.manifest.resolve(), args.adjudicated.resolve(), args.pairs.resolve(),
        args.multiclass_review.resolve(),
    ]
    manifest, adjudicated, pairs, source_hashes = load_inputs(*input_paths)
    final = apply_adjudication(manifest, adjudicated)
    final = add_historical_groups(final, pairs)
    final = final[FINAL_COLUMNS].copy()

    transitions = (
        final.groupby(["label_principal", "effective_label"], dropna=False)
        .size().rename("count").reset_index()
        .rename(columns={"label_principal": "original_label"})
    )
    transitions["is_correction"] = transitions.original_label.ne(transitions.effective_label)
    transitions = transitions.sort_values(
        ["is_correction", "original_label", "effective_label"], ascending=[False, True, True]
    ).reset_index(drop=True)
    pair_metrics = pair_label_metrics(final, pairs)
    validations = validate_final(final, pairs)
    if not all(validations.values()):
        failed = [name for name, passed in validations.items() if not passed]
        raise RuntimeError(f"Validations finales échouées: {', '.join(failed)}")

    corrections = int(final.label_principal.ne(final.effective_label).sum())
    summary = {
        "purpose": "final_derived_training_manifest_for_section_4_2",
        "source_hashes": source_hashes,
        "total_images": len(final),
        "unique_filenames": int(final.filename.nunique()),
        "effective_label_distribution": distribution(final.effective_label),
        "year_distribution": distribution(final.year),
        "position_distribution": distribution(final.cam_position),
        "camera_distribution": distribution(final.cam_num),
        "difficult_cases": {
            "multiple": int(final.multiple.sum()),
            "chunk": int(final.chunk.sum()),
            "mixed_quality": int(final.mixed_quality.sum()),
        },
        "effective_label_source_distribution": distribution(final.effective_label_source),
        "adjudication_status_distribution": distribution(final.adjudication_status),
        "effective_corrections": corrections,
        "historical_tb": {
            "paired_images": int(final.historical_pair_id.notna().sum()),
            "unpaired_images": int(final.historical_pair_id.isna().sum()),
            **pair_metrics,
        },
        "before_after": {
            "global": label_comparison(final),
            "by_year": label_comparison(final, "year"),
            "by_position": label_comparison(final, "cam_position"),
            "by_camera": label_comparison(final, "cam_num"),
        },
        "validations": validations,
        "video_crops_integrated": False,
        "split_created": False,
    }
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)

    for path in input_paths:
        if _sha256(path) != source_hashes[path.name]:
            raise RuntimeError(f"La source {path.name} a changé pendant la construction")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)
    transitions.to_csv(TRANSITIONS_CSV, index=False)
    SUMMARY_JSON.write_text(summary_text, encoding="utf-8")
    write_report(REPORT_MD, summary, transitions)

    print(f"Manifeste final: {len(final):,} images, {final.filename.nunique():,} filenames uniques".replace(",", " "))
    print("Distribution effective_label:", final.effective_label.value_counts().to_dict())
    print(f"Corrections effectives: {corrections}")
    print(
        f"T/B historique: {summary['historical_tb']['paired_images']:,} paired, "
        f"{summary['historical_tb']['unpaired_images']:,} unpaired, "
        f"{pair_metrics['pairs_same_effective_label']:,} paires même label, "
        f"{pair_metrics['pairs_different_effective_label']:,} différentes".replace(",", " ")
    )
    print(f"Validations: {sum(validations.values())}/{len(validations)} PASS")


if __name__ == "__main__":
    main()
