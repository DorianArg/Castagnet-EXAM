"""Simule des stratégies de constitution du dataset final, sans créer ce dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from historical_tb_audit import DEFAULT_MANIFEST, DEFAULT_OUTPUT_DIR, _as_bool, _load_manifest, _sha256


VALID_LABELS = ["Conforme", "NON Conforme", "PIETRA", "Vide"]
PAIR_FILE = DEFAULT_OUTPUT_DIR / "historical_tb_pairs.csv"
UNPAIRED_FILE = DEFAULT_OUTPUT_DIR / "historical_tb_unpaired.csv"

STRATEGIES = {
    "S0": {
        "title": "Baseline large",
        "excluded_flags": [],
        "choice": "Conserver toutes les images dont label_principal est valide.",
        "hypothesis": "Faire confiance à la cible d'entraînement actuellement validée.",
        "positive": "Préserve toute la diversité observée en production.",
        "critique": "Conserve aussi les 59 conflits masked qui méritent une revue humaine.",
    },
    "S1": {
        "title": "Prudente",
        "excluded_flags": ["is_masked_disagreement"],
        "choice": "Exclure provisoirement les seuls conflits masked vide/non-vide.",
        "hypothesis": "Isoler un petit ensemble contradictoire avant revue humaine.",
        "positive": "Réduit l'incertitude explicite avec une perte très limitée.",
        "critique": "Le masque automatique n'est pas une vérité terrain et peut lui-même être erroné.",
    },
    "S2": {
        "title": "Contrôlée",
        "excluded_flags": [
            "is_masked_disagreement", "is_multiple", "is_chunk", "is_mixed_quality"
        ],
        "choice": "Conserver seulement les cas sans conflit masked, multiple, chunk ou qualité mixte.",
        "hypothesis": "Mesurer un dataset centré sur des images simples.",
        "positive": "Simplifie le signal visuel et l'analyse initiale.",
        "critique": "Peut retirer des situations normales de production et déplacer fortement les distributions.",
    },
    "S3a": {
        "title": "Stricte",
        "excluded_flags": [
            "is_masked_disagreement", "is_multiple", "is_chunk", "is_mixed_quality"
        ],
        "choice": "Appliquer les critères stricts sans exiger la provenance labeled_by.",
        "hypothesis": "Distinguer la qualité visuelle de la traçabilité d'annotation.",
        "positive": "Ne confond pas provenance manquante et label faux.",
        "critique": "Avec les règles fournies, cette stratégie est exactement identique à S2.",
    },
    "S3b": {
        "title": "Stricte avec provenance",
        "excluded_flags": [
            "is_masked_disagreement", "is_multiple", "is_chunk", "is_mixed_quality",
            "is_missing_labeled_by",
        ],
        "choice": "Ajouter aux critères stricts l'obligation d'un labeled_by renseigné.",
        "hypothesis": "Mesurer l'effet d'une exigence maximale de traçabilité.",
        "positive": "Toutes les images retenues possèdent une provenance explicite.",
        "critique": "L'absence de pseudo ne prouve aucune erreur et peut introduire un biais massif.",
    },
}

ATOMIC_FLAGS = [
    "is_masked_disagreement",
    "is_multiple",
    "is_chunk",
    "is_mixed_quality",
    "is_missing_labeled_by",
    "is_label_filename_changed",
    "is_historical_paired",
    "is_historical_unpaired",
    "has_pair_label_disagreement",
]

FILTER_FLAGS = [
    "is_masked_disagreement", "is_multiple", "is_chunk", "is_mixed_quality",
    "is_missing_labeled_by",
]


def _distribution(series: pd.Series, mask: pd.Series) -> dict[str, dict[str, float | int]]:
    counts = series.loc[mask].value_counts(dropna=False)
    total = int(mask.sum())
    return {
        str(category): {
            "count": int(count),
            "percentage": round(100 * int(count) / total, 2) if total else 0.0,
        }
        for category, count in counts.items()
    }


def _with_s0_delta(
    distribution: dict[str, dict[str, float | int]],
    baseline: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    categories = sorted(set(distribution) | set(baseline))
    return {
        category: {
            "count": int(distribution.get(category, {}).get("count", 0)),
            "percentage": float(distribution.get(category, {}).get("percentage", 0.0)),
            "s0_percentage": float(baseline.get(category, {}).get("percentage", 0.0)),
            "delta_percentage_points": round(
                float(distribution.get(category, {}).get("percentage", 0.0))
                - float(baseline.get(category, {}).get("percentage", 0.0)),
                2,
            ),
        }
        for category in categories
    }


def _load_inputs(manifest_path: Path, pair_path: Path, unpaired_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _load_manifest(manifest_path)
    for column in ("masked_disagreement", "masked_covered"):
        if column not in df:
            raise ValueError(f"Colonne absente du manifeste: {column}")
        df[column] = _as_bool(df[column], column)
    for column in ("labeled_by", "labeled_by_canonical"):
        if column not in df:
            raise ValueError(f"Colonne absente du manifeste: {column}")
        df[column] = df[column].astype("string").fillna("")

    pairs = pd.read_csv(pair_path, low_memory=False)
    required_pair_columns = {
        "pair_id", "top_filename", "bottom_filename", "principal_label_agreement",
        "pairing_method", "pairing_status",
    }
    missing = sorted(required_pair_columns.difference(pairs.columns))
    if missing:
        raise ValueError(f"Colonnes absentes de historical_tb_pairs.csv: {', '.join(missing)}")
    pairs["principal_label_agreement"] = _as_bool(
        pairs["principal_label_agreement"], "principal_label_agreement"
    )
    if len(pairs) != 16_176 or not pairs.pair_id.is_unique:
        raise ValueError("La table historique stabilisée ne contient pas 16 176 pair_id uniques")
    if not pairs.pairing_method.eq("historical_namespace").all() or not pairs.pairing_status.eq(
        "heuristic_pair"
    ).all():
        raise ValueError("Méthode ou statut historique inattendu")

    unpaired = pd.read_csv(unpaired_path, usecols=["filename"])
    paired_names = pd.concat(
        [pairs.top_filename.rename("filename"), pairs.bottom_filename.rename("filename")],
        ignore_index=True,
    )
    if not paired_names.is_unique or set(paired_names) & set(unpaired.filename):
        raise ValueError("Chevauchement dans la partition T/B historique")
    if set(paired_names) | set(unpaired.filename) != set(df.filename):
        raise ValueError("La partition T/B historique ne couvre pas exactement le manifeste")

    pair_members = pd.concat(
        [
            pairs[["pair_id", "top_filename", "principal_label_agreement"]].rename(
                columns={"top_filename": "filename"}
            ),
            pairs[["pair_id", "bottom_filename", "principal_label_agreement"]].rename(
                columns={"bottom_filename": "filename"}
            ),
        ],
        ignore_index=True,
    )
    pair_members["has_pair_label_disagreement"] = ~pair_members.principal_label_agreement
    df = df.merge(
        pair_members[["filename", "pair_id", "has_pair_label_disagreement"]],
        on="filename", how="left", validate="one_to_one",
    )
    return df, pairs


def _add_flags(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["is_masked_disagreement"] = result.masked_disagreement
    result["is_multiple"] = result.multiple
    result["is_chunk"] = result.chunk
    result["is_mixed_quality"] = result.mixed_quality
    result["is_missing_labeled_by"] = result.labeled_by.str.strip().eq("")
    result["is_label_filename_changed"] = result.label_filename.ne(result.label_principal)
    result["is_historical_paired"] = result.pair_id.notna()
    result["is_historical_unpaired"] = ~result.is_historical_paired
    result["has_pair_label_disagreement"] = (
        result.pair_id.notna() & result.has_pair_label_disagreement.eq(True)
    )
    return result


def _strategy_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    valid = df.label_principal.isin(VALID_LABELS)
    masks: dict[str, pd.Series] = {}
    for strategy, config in STRATEGIES.items():
        excluded = pd.Series(False, index=df.index)
        for flag in config["excluded_flags"]:
            excluded |= df[flag]
        masks[strategy] = valid & ~excluded
    return masks


def _pair_metrics(pairs: pd.DataFrame, included_filenames: set[str]) -> dict[str, int]:
    top_kept = pairs.top_filename.isin(included_filenames)
    bottom_kept = pairs.bottom_filename.isin(included_filenames)
    complete = top_kept & bottom_kept
    partial = top_kept ^ bottom_kept
    removed = ~top_kept & ~bottom_kept
    return {
        "complete_pairs": int(complete.sum()),
        "partial_pairs": int(partial.sum()),
        "fully_removed_pairs": int(removed.sum()),
        "complete_pairs_same_label_principal": int(
            (complete & pairs.principal_label_agreement).sum()
        ),
        "complete_pairs_different_label_principal": int(
            (complete & ~pairs.principal_label_agreement).sum()
        ),
    }


def _changed_vide_analysis(df: pd.DataFrame, mask: pd.Series) -> dict[str, object]:
    changed = mask & df.is_label_filename_changed & df.label_principal.eq("Vide")
    return {
        "count": int(changed.sum()),
        "by_year": _distribution(df.year, changed),
        "by_camera": _distribution(df.cam_num, changed),
        "by_position": _distribution(df.cam_position, changed),
        "by_labeled_by_canonical": _distribution(
            df.labeled_by_canonical.replace("", "(missing)"), changed
        ),
        "masked_covered": int((changed & df.masked_covered).sum()),
        "masked_not_covered": int((changed & ~df.masked_covered).sum()),
    }


def _strategy_summary(
    df: pd.DataFrame,
    pairs: pd.DataFrame,
    masks: dict[str, pd.Series],
) -> dict[str, object]:
    baseline_mask = masks["S0"]
    dimensions = {
        "classes": df.label_principal,
        "years": df.year,
        "positions": df.cam_position,
        "cameras": df.cam_num,
    }
    baseline_distributions = {
        name: _distribution(series, baseline_mask) for name, series in dimensions.items()
    }
    summaries: dict[str, object] = {}
    for strategy, mask in masks.items():
        kept = int(mask.sum())
        included = set(df.loc[mask, "filename"])
        distributions = {
            name: _with_s0_delta(
                _distribution(series, mask), baseline_distributions[name]
            )
            for name, series in dimensions.items()
        }
        pair_metrics = _pair_metrics(pairs, included)
        pair_metrics.update(
            {
                "paired_images_kept": int((mask & df.is_historical_paired).sum()),
                "unpaired_images_kept": int((mask & df.is_historical_unpaired).sum()),
                "images_in_disagreeing_pairs_kept": int(
                    (mask & df.has_pair_label_disagreement).sum()
                ),
            }
        )
        canonical = df.labeled_by_canonical.replace("", "(missing)")
        summaries[strategy] = {
            "title": STRATEGIES[strategy]["title"],
            "excluded_flags": STRATEGIES[strategy]["excluded_flags"],
            "kept_images": kept,
            "removed_images": len(df) - kept,
            "kept_pct": round(100 * kept / len(df), 2),
            "removed_pct": round(100 * (len(df) - kept) / len(df), 2),
            **distributions,
            "remaining_flags": {
                flag: int((mask & df[flag]).sum()) for flag in FILTER_FLAGS
            },
            "provenance": {
                "labeled_by_present": int((mask & ~df.is_missing_labeled_by).sum()),
                "labeled_by_missing": int((mask & df.is_missing_labeled_by).sum()),
                "labeled_by_canonical_distribution": _distribution(canonical, mask),
            },
            "historical_tb": pair_metrics,
            "changed_to_vide": _changed_vide_analysis(df, mask),
        }
    return summaries


def _filter_impacts(df: pd.DataFrame, baseline: pd.Series) -> dict[str, object]:
    impacts: dict[str, object] = {}
    for flag in FILTER_FLAGS:
        removed = baseline & df[flag]
        impacts[flag] = {
            "removed_images": int(removed.sum()),
            "by_class": _distribution(df.label_principal, removed),
            "by_year": _distribution(df.year, removed),
            "by_position": _distribution(df.cam_position, removed),
            "by_camera": _distribution(df.cam_num, removed),
        }
    return impacts


def _intersections(df: pd.DataFrame, baseline: pd.Series) -> dict[str, int]:
    combinations = {
        "multiple_and_chunk": ["is_multiple", "is_chunk"],
        "chunk_and_masked_disagreement": ["is_chunk", "is_masked_disagreement"],
        "multiple_and_masked_disagreement": ["is_multiple", "is_masked_disagreement"],
        "missing_labeled_by_and_vide": ["is_missing_labeled_by", "label_is_vide"],
        "missing_labeled_by_and_multiple": ["is_missing_labeled_by", "is_multiple"],
        "missing_labeled_by_and_chunk": ["is_missing_labeled_by", "is_chunk"],
        "missing_labeled_by_and_masked_disagreement": [
            "is_missing_labeled_by", "is_masked_disagreement"
        ],
        "multiple_and_chunk_and_masked_disagreement": [
            "is_multiple", "is_chunk", "is_masked_disagreement"
        ],
        "missing_labeled_by_and_multiple_and_chunk": [
            "is_missing_labeled_by", "is_multiple", "is_chunk"
        ],
    }
    masks = {flag: df[flag] for flag in FILTER_FLAGS}
    masks["label_is_vide"] = df.label_principal.eq("Vide")
    result: dict[str, int] = {}
    for name, flags in combinations.items():
        current = baseline.copy()
        for flag in flags:
            current &= masks[flag]
        result[name] = int(current.sum())
    quality_union = (
        df.is_masked_disagreement | df.is_multiple | df.is_chunk | df.is_mixed_quality
    )
    result["union_masked_multiple_chunk_mixed_quality"] = int(
        (baseline & quality_union).sum()
    )
    result["missing_labeled_by_and_any_quality_filter"] = int(
        (baseline & df.is_missing_labeled_by & quality_union).sum()
    )
    return result


def _s1_reviewed_metrics(df: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, object] | None:
    adjudicated_path = DEFAULT_OUTPUT_DIR / "masked_disagreements_adjudicated.csv"
    if not adjudicated_path.exists():
        return None
    adjudicated = pd.read_csv(adjudicated_path, dtype="string", keep_default_na=False)
    required = {"filename", "review_action", "effective_label"}
    if not required.issubset(adjudicated.columns) or not adjudicated.filename.is_unique:
        raise ValueError("masked_disagreements_adjudicated.csv est invalide")
    conflict_names = set(df.loc[df.is_masked_disagreement, "filename"])
    if set(adjudicated.filename) != conflict_names:
        raise ValueError("L'adjudication ne couvre pas exactement les conflits masked")
    effective_by_filename = adjudicated.set_index("filename").effective_label
    effective = df.label_principal.astype("string").copy()
    conflict = df.filename.isin(conflict_names)
    effective.loc[conflict] = df.loc[conflict, "filename"].map(effective_by_filename).fillna("")
    included = df.label_principal.isin(VALID_LABELS) & (~conflict | effective.isin(VALID_LABELS))
    baseline = df.label_principal.isin(VALID_LABELS)
    baseline_dimensions = {
        "classes": _distribution(df.label_principal, baseline),
        "years": _distribution(df.year, baseline),
        "positions": _distribution(df.cam_position, baseline),
        "cameras": _distribution(df.cam_num, baseline),
    }
    distributions = {
        "classes": _with_s0_delta(_distribution(effective, included), baseline_dimensions["classes"]),
        "years": _with_s0_delta(_distribution(df.year, included), baseline_dimensions["years"]),
        "positions": _with_s0_delta(_distribution(df.cam_position, included), baseline_dimensions["positions"]),
        "cameras": _with_s0_delta(_distribution(df.cam_num, included), baseline_dimensions["cameras"]),
    }
    included_names = set(df.loc[included, "filename"])
    return {
        "title": "S1 après revues humaines",
        "status": "provisional" if effective.loc[conflict].eq("").any() else "complete",
        "kept_images": int(included.sum()),
        "removed_images": int((~included).sum()),
        "kept_pct": round(100 * int(included.sum()) / len(df), 2),
        "removed_pct": round(100 * int((~included).sum()) / len(df), 2),
        **distributions,
        "historical_tb": _pair_metrics(pairs, included_names),
        "pending_or_uncertain_conflicts": int((conflict & effective.eq("")).sum()),
        "derived_labels_only": True,
    }


def _comparison_rows(summaries: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy, summary in summaries.items():
        for category, count, percentage in [
            ("kept", summary["kept_images"], summary["kept_pct"]),
            ("removed", summary["removed_images"], summary["removed_pct"]),
        ]:
            rows.append(
                {"strategy": strategy, "dimension": "size", "category": category,
                 "count": count, "percentage": percentage, "s0_percentage": "",
                 "delta_percentage_points": ""}
            )
        for dimension in ("classes", "years", "positions", "cameras"):
            for category, values in summary[dimension].items():
                rows.append(
                    {"strategy": strategy, "dimension": dimension, "category": category, **values}
                )
        for flag, count in summary["remaining_flags"].items():
            rows.append(
                {"strategy": strategy, "dimension": "remaining_flag", "category": flag,
                 "count": count, "percentage": round(100 * count / summary["kept_images"], 2)
                 if summary["kept_images"] else 0.0, "s0_percentage": "",
                 "delta_percentage_points": ""}
            )
        for metric, count in summary["historical_tb"].items():
            rows.append(
                {"strategy": strategy, "dimension": "historical_tb", "category": metric,
                 "count": count, "percentage": "", "s0_percentage": "",
                 "delta_percentage_points": ""}
            )
    return pd.DataFrame(rows)


def _append_s1_reviewed_rows(comparison: pd.DataFrame, reviewed: dict[str, object] | None) -> pd.DataFrame:
    if reviewed is None:
        return comparison
    rows: list[dict[str, object]] = [
        {"strategy": "S1_REVIEWED", "dimension": "size", "category": "kept",
         "count": reviewed["kept_images"], "percentage": reviewed["kept_pct"],
         "s0_percentage": "", "delta_percentage_points": ""},
        {"strategy": "S1_REVIEWED", "dimension": "size", "category": "removed",
         "count": reviewed["removed_images"], "percentage": reviewed["removed_pct"],
         "s0_percentage": "", "delta_percentage_points": ""},
    ]
    for dimension in ("classes", "years", "positions", "cameras"):
        for category, values in reviewed[dimension].items():
            rows.append(
                {"strategy": "S1_REVIEWED", "dimension": dimension,
                 "category": category, **values}
            )
    for metric, count in reviewed["historical_tb"].items():
        rows.append(
            {"strategy": "S1_REVIEWED", "dimension": "historical_tb", "category": metric,
             "count": count, "percentage": "", "s0_percentage": "",
             "delta_percentage_points": ""}
        )
    return pd.concat([comparison, pd.DataFrame(rows)], ignore_index=True)


def _markdown_table(df: pd.DataFrame) -> str:
    printable = df.fillna("").astype(str)
    header = "| " + " | ".join(printable.columns) + " |"
    separator = "| " + " | ".join("---" for _ in printable.columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in printable.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *body])


def _compact_distribution(values: dict[str, dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"category": category, "count": item["count"], "pct": item["percentage"],
             "delta_vs_S0_pp": item["delta_percentage_points"]}
            for category, item in values.items()
        ]
    )


def _write_report(
    path: Path,
    df: pd.DataFrame,
    summaries: dict[str, object],
    filter_impacts: dict[str, object],
    intersections: dict[str, int],
) -> None:
    headline = pd.DataFrame(
        [
            {
                "strategy": strategy,
                "kept": summary["kept_images"],
                "removed": summary["removed_images"],
                "kept_pct": summary["kept_pct"],
                "complete_pairs": summary["historical_tb"]["complete_pairs"],
                "partial_pairs": summary["historical_tb"]["partial_pairs"],
            }
            for strategy, summary in summaries.items()
        ]
    )
    impacts = pd.DataFrame(
        [
            {"filter": flag, "images_removed_alone": values["removed_images"]}
            for flag, values in filter_impacts.items()
        ]
    )
    overlap = pd.DataFrame(
        [{"intersection": name, "images": count} for name, count in intersections.items()]
    )
    adjudication_path = path.parent / "masked_disagreements_adjudication_summary.json"
    if adjudication_path.exists():
        adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
        actions = adjudication["action_counts"]
        human = adjudication["human_review_distribution"]
        s1_reviewed = adjudication["s1_reviewed"]
        pending_multiclass = s1_reviewed["excluded_pending_multiclass"]
        if pending_multiclass:
            reviewed_status = (
                f"La simulation provisoire `S1_REVIEWED` réintègre actuellement "
                f"{s1_reviewed['reintegrated_from_59']} des 59 cas et conserve "
                f"{s1_reviewed['dataset_images_if_applied_now']} images. Elle reste provisoire "
                f"tant que {pending_multiclass} décisions multiclasses sont en attente."
            )
        else:
            reviewed_status = (
                f"`S1_REVIEWED` réintègre les 59 cas et conserve "
                f"{s1_reviewed['dataset_images_if_applied_now']} images. Les 9 décisions "
                "multiclasses sont terminées : aucun cas pending ou uncertain ne subsiste."
            )
        human_review_section = f"""## Revue humaine des conflits masked/principal

59 conflits ont été détectés sur 35 254 images et revus manuellement en aveugle. La première revue portait uniquement sur `Vide` / `NonVide` : {human['Vide']} décisions Vide, {human['NonVide']} NonVide et {human['Incertain']} Incertain. Elle confirme directement {actions['KEEP_PRINCIPAL']} labels principaux et propose {actions['CORRECT_TO_VIDE']} corrections dérivées vers Vide.

Une décision NonVide ne suffit pas à reconstruire une classe parmi Conforme, NON Conforme et PIETRA. Une seconde revue multiclasses est donc nécessaire uniquement pour les {actions['NEED_MULTICLASS_REVIEW']} cas concernés. Aucune source existante — `label_principal`, `labels_masked`, `label_filename` ou la vue T/B associée — n'est privilégiée automatiquement.

{reviewed_status} Les labels effectifs restent dérivés et ne sont pas écrits dans les CSV sources.

## Pourquoi une revue manuelle ciblée ?

Le volume de 59 cas est faible. La revue humaine évite une exclusion automatique inutile, une confiance arbitraire dans `labels_masked` ou une confiance arbitraire dans `label_principal`. Elle reste subjective et les décisions `Incertain` sont conservées comme telles et exclues tant qu'aucune adjudication fiable n'existe.
"""
    else:
        human_review_section = """## Revue humaine des conflits masked/principal

Les 59 conflits doivent être revus en aveugle avant de mesurer `S1_REVIEWED`.
"""
    sections: list[str] = []
    for strategy, summary in summaries.items():
        config = STRATEGIES[strategy]
        tb = summary["historical_tb"]
        max_bias = max(
            (
                abs(float(values["delta_percentage_points"]))
                for dimension in ("classes", "years", "positions", "cameras")
                for values in summary[dimension].values()
            ),
            default=0.0,
        )
        sections.append(
            f"""## {strategy} — {config['title']}

### Choix

{config['choice']}

### Hypothèse

{config['hypothesis']}

### Conséquence data

- {summary['kept_images']} images conservées ({summary['kept_pct']:.2f} %), {summary['removed_images']} retirées.
- {tb['complete_pairs']} paires historiques complètes, {tb['partial_pairs']} paires partielles et {tb['fully_removed_pairs']} entièrement retirées.
- Parmi les paires complètes : {tb['complete_pairs_same_label_principal']} labels principaux identiques et {tb['complete_pairs_different_label_principal']} différents.
- Écart absolu maximal de proportion par rapport à S0 : {max_bias:.2f} point(s).

Classes :

{_markdown_table(_compact_distribution(summary['classes']))}

Années :

{_markdown_table(_compact_distribution(summary['years']))}

Positions :

{_markdown_table(_compact_distribution(summary['positions']))}

Caméras :

{_markdown_table(_compact_distribution(summary['cameras']))}

### Point positif

{config['positive']}

### Critique

{config['critique']}
"""
        )

    report = f"""# Simulation des stratégies de constitution du dataset final

Cette analyse ne supprime aucune image et ne crée ni dataset final ni split train/validation/test. `label_principal` reste la cible. Les désaccords T/B et les changements depuis `label_filename` sont mesurés, jamais utilisés comme filtres.

## Vue d'ensemble

{_markdown_table(headline)}

S2 et S3a sont identiques parce que les critères fournis sont identiques. Cette redondance est conservée explicitement plutôt que d'inventer une règle supplémentaire.

{human_review_section}

## Flags atomiques

{_markdown_table(pd.DataFrame([{"flag": flag, "images": int(df[flag].sum())} for flag in ATOMIC_FLAGS]))}

## Impact isolé de chaque filtre

{_markdown_table(impacts)}

Les répartitions par classe, année, position et caméra de chaque filtre sont conservées dans le résumé JSON.

## Chevauchements principaux

{_markdown_table(overlap)}

Les retraits des stratégies sont calculés sur l'union réelle des filenames et non par addition de ces compteurs.

{''.join(sections)}

## Analyse des changements vers Vide

Pour chaque stratégie, le JSON conserve la répartition des images `label_filename != label_principal` devenues `Vide` par année, caméra, position, labeliseur canonique et couverture masked. Ces images ne sont jamais exclues sur ce seul motif.

## Lecture quantitative

S1 constitue le compromis quantitatif le plus conservateur : elle isole uniquement les 59 conflits masked connus tout en conservant les cas `multiple`, `chunk`, `Vide`, les provenances manquantes et les désaccords entre vues. Cette observation ne vaut pas validation métier. S2/S3a mesurent le coût d'un dataset artificiellement simplifié. S3b mesure surtout une exigence de traçabilité et ne doit pas être interprétée comme une amélioration certaine des labels.

## Vigilance pour le futur split

Le split n'est pas créé ici. Lorsqu'il le sera, les deux vues d'une même paire historique devront probablement rester dans le même split afin de limiter les fuites de données.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--pairs", type=Path, default=PAIR_FILE)
    parser.add_argument("--unpaired", type=Path, default=UNPAIRED_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    source_hashes = {
        "dataset_manifest.csv": _sha256(manifest_path),
        "historical_tb_pairs.csv": _sha256(args.pairs.resolve()),
        "historical_tb_unpaired.csv": _sha256(args.unpaired.resolve()),
    }
    df, pairs = _load_inputs(manifest_path, args.pairs.resolve(), args.unpaired.resolve())
    df = _add_flags(df)
    masks = _strategy_masks(df)
    summaries = _strategy_summary(df, pairs, masks)
    filter_impacts = _filter_impacts(df, masks["S0"])
    intersections = _intersections(df, masks["S0"])

    diagnostics_columns = [
        "filename", "year", "batch_num", "cam_position", "cam_num", "sample_num",
        "label_filename", "label_principal", "labeled_by", "labeled_by_canonical",
        "masked_covered", "pair_id", *ATOMIC_FLAGS,
    ]
    diagnostics = df[diagnostics_columns].copy()
    for strategy, mask in masks.items():
        diagnostics[f"included_{strategy}"] = mask
        diagnostics[f"excluded_{strategy}"] = ~mask

    s1_reviewed = _s1_reviewed_metrics(df, pairs)
    comparison = _append_s1_reviewed_rows(_comparison_rows(summaries), s1_reviewed)
    summary = {
        "purpose": "simulation_only_no_dataset_created",
        "source_hashes": source_hashes,
        "dataset_images": len(df),
        "valid_label_images": int(masks["S0"].sum()),
        "atomic_flag_counts": {flag: int(df[flag].sum()) for flag in ATOMIC_FLAGS},
        "strategies": summaries,
        "S1_REVIEWED": s1_reviewed,
        "individual_filter_impacts": filter_impacts,
        "flag_intersections": intersections,
        "notes": {
            "S2_equals_S3a": bool(masks["S2"].equals(masks["S3a"])),
            "pair_label_disagreement_is_not_a_filter": True,
            "label_filename_change_is_not_a_filter": True,
            "recommended_for_human_review": "S1_quantitatively_conservative_not_finally_validated",
        },
    }
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2)

    for name, path in [
        ("dataset_manifest.csv", manifest_path),
        ("historical_tb_pairs.csv", args.pairs.resolve()),
        ("historical_tb_unpaired.csv", args.unpaired.resolve()),
    ]:
        if _sha256(path) != source_hashes[name]:
            raise RuntimeError(f"La source {name} a changé pendant l'analyse")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "final_dataset_strategy_comparison.csv", index=False)
    diagnostics.to_csv(output_dir / "final_dataset_exclusion_diagnostics.csv", index=False)
    (output_dir / "final_dataset_strategy_summary.json").write_text(
        summary_text, encoding="utf-8"
    )
    _write_report(
        output_dir / "final_dataset_strategy_report.md",
        df, summaries, filter_impacts, intersections,
    )

    print("Simulation terminée — aucun dataset final créé.")
    print(
        pd.DataFrame(
            [
                {"strategy": key, "kept": value["kept_images"],
                 "removed": value["removed_images"], "kept_pct": value["kept_pct"],
                 "complete_pairs": value["historical_tb"]["complete_pairs"]}
                for key, value in summaries.items()
            ]
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
