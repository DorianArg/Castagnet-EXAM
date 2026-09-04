"""Aligne temporellement les séquences de crops Top et Bottom.

L'alignement est monotone et 1-à-1. Une paire est admissible seulement pour un
décalage de +6, +7 ou +8 frames. Les associations qui ne sont pas présentes
dans toutes les solutions optimales sont conservées comme ambiguës.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTRACTION_DIR = ROOT / "analysis" / "extracted_video"
METADATA_CSV = EXTRACTION_DIR / "crops_metadata.csv"
PAIRS_CSV = EXTRACTION_DIR / "tb_pairs.csv"
SUMMARY_JSON = EXTRACTION_DIR / "tb_pairing_summary.json"

EXPECTED_DELTA = 7
TOLERANCE = 1

PAIR_COLUMNS = [
    "pair_id", "top_filename", "bottom_filename", "top_frame", "bottom_frame",
    "top_timestamp_ms", "bottom_timestamp_ms", "delta_frames", "delta_ms",
    "temporal_cost", "status", "notes",
]

Score = tuple[int, int]  # (nombre de paires, opposé du coût total)
Edge = tuple[int, int]


def crop_path(row: dict[str, str]) -> Path:
    folder = "top" if row["cam_position"] == "T" else "bottom"
    return EXTRACTION_DIR / folder / "crops" / row["crop_filename"]


def crop_sort_key(row: dict[str, str]) -> tuple[int, int, str]:
    return int(row["frame_index"]), int(row["crop_index"]), row["crop_filename"]


def read_sequences() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    with METADATA_CSV.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"crop_filename", "cam_position", "frame_index", "timestamp_ms", "crop_index"}
    if not rows:
        raise ValueError("crops_metadata.csv ne contient aucun crop")
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(f"Colonnes de métadonnées absentes : {sorted(missing_columns)}")
    if len({row["crop_filename"] for row in rows}) != len(rows):
        raise ValueError("crop_filename doit être unique dans crops_metadata.csv")
    missing_files = [row["crop_filename"] for row in rows if not crop_path(row).is_file()]
    if missing_files:
        raise ValueError(f"{len(missing_files)} crop(s) référencé(s) sont absents")
    invalid_positions = sorted({row["cam_position"] for row in rows} - {"T", "B"})
    if invalid_positions:
        raise ValueError(f"Positions caméra invalides : {invalid_positions}")
    tops = sorted((row for row in rows if row["cam_position"] == "T"), key=crop_sort_key)
    bottoms = sorted((row for row in rows if row["cam_position"] == "B"), key=crop_sort_key)
    return tops, bottoms


def temporal_cost(top: dict[str, str], bottom: dict[str, str]) -> int:
    delta = int(bottom["frame_index"]) - int(top["frame_index"])
    return abs(delta - EXPECTED_DELTA)


def is_admissible(top: dict[str, str], bottom: dict[str, str]) -> bool:
    return temporal_cost(top, bottom) <= TOLERANCE


def add_scores(*scores: Score) -> Score:
    return sum(score[0] for score in scores), sum(score[1] for score in scores)


def best_score(
    tops: list[dict[str, str]], bottoms: list[dict[str, str]],
    top_start: int = 0, top_end: int | None = None,
    bottom_start: int = 0, bottom_end: int | None = None,
    forbidden_edge: Edge | None = None,
) -> Score:
    """Maximise le nombre de paires puis minimise leur coût total."""
    top_end = len(tops) if top_end is None else top_end
    bottom_end = len(bottoms) if bottom_end is None else bottom_end
    top_count, bottom_count = top_end - top_start, bottom_end - bottom_start
    dp: list[list[Score]] = [
        [(0, 0) for _ in range(bottom_count + 1)] for _ in range(top_count + 1)
    ]
    for local_top in range(top_count - 1, -1, -1):
        for local_bottom in range(bottom_count - 1, -1, -1):
            top_index, bottom_index = top_start + local_top, bottom_start + local_bottom
            choices = [dp[local_top + 1][local_bottom], dp[local_top][local_bottom + 1]]
            edge = (top_index, bottom_index)
            if edge != forbidden_edge and is_admissible(tops[top_index], bottoms[bottom_index]):
                cost = temporal_cost(tops[top_index], bottoms[bottom_index])
                choices.append(add_scores((1, -cost), dp[local_top + 1][local_bottom + 1]))
            dp[local_top][local_bottom] = max(choices)
    return dp[0][0]


def score_with_required_edge(
    tops: list[dict[str, str]], bottoms: list[dict[str, str]], edge: Edge
) -> Score:
    top_index, bottom_index = edge
    prefix = best_score(tops, bottoms, 0, top_index, 0, bottom_index)
    suffix = best_score(
        tops, bottoms, top_index + 1, len(tops), bottom_index + 1, len(bottoms)
    )
    cost = temporal_cost(tops[top_index], bottoms[bottom_index])
    return add_scores(prefix, (1, -cost), suffix)


def classify_optimal_edges(
    tops: list[dict[str, str]], bottoms: list[dict[str, str]]
) -> tuple[Score, set[Edge], set[Edge]]:
    """Retourne le score optimal, les arêtes possibles et celles qui sont forcées."""
    optimum = best_score(tops, bottoms)
    admissible_edges = {
        (top_index, bottom_index)
        for top_index, top in enumerate(tops)
        for bottom_index, bottom in enumerate(bottoms)
        if is_admissible(top, bottom)
    }
    possible: set[Edge] = set()
    forced: set[Edge] = set()
    for edge in admissible_edges:
        if score_with_required_edge(tops, bottoms, edge) != optimum:
            continue
        possible.add(edge)
        if best_score(tops, bottoms, forbidden_edge=edge) != optimum:
            forced.add(edge)
    return optimum, possible, forced


def ambiguous_components(edges: set[Edge]) -> list[set[tuple[str, int]]]:
    graph: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    for top_index, bottom_index in edges:
        top_node, bottom_node = ("T", top_index), ("B", bottom_index)
        graph[top_node].add(bottom_node)
        graph[bottom_node].add(top_node)
    components: list[set[tuple[str, int]]] = []
    unseen = set(graph)
    while unseen:
        component: set[tuple[str, int]] = set()
        queue = deque([min(unseen)])
        while queue:
            node = queue.popleft()
            if node in component:
                continue
            component.add(node)
            unseen.discard(node)
            queue.extend(graph[node] - component)
        components.append(component)
    return components


def joined(rows: list[dict[str, str]], column: str) -> str:
    return "|".join(row[column] for row in rows)


def paired_row(top: dict[str, str], bottom: dict[str, str]) -> dict[str, object]:
    top_frame, bottom_frame = int(top["frame_index"]), int(bottom["frame_index"])
    top_timestamp, bottom_timestamp = float(top["timestamp_ms"]), float(bottom["timestamp_ms"])
    return {
        "top_filename": top["crop_filename"], "bottom_filename": bottom["crop_filename"],
        "top_frame": top_frame, "bottom_frame": bottom_frame,
        "top_timestamp_ms": top_timestamp, "bottom_timestamp_ms": bottom_timestamp,
        "delta_frames": bottom_frame - top_frame,
        "delta_ms": round(bottom_timestamp - top_timestamp, 3),
        "temporal_cost": temporal_cost(top, bottom), "status": "matched_temporal",
        "notes": "Paire forcée dans toutes les solutions optimales ; identité inférée temporellement.",
    }


def unmatched_row(crop: dict[str, str], position: str) -> dict[str, object]:
    is_top = position == "T"
    return {
        "top_filename": crop["crop_filename"] if is_top else "",
        "bottom_filename": "" if is_top else crop["crop_filename"],
        "top_frame": crop["frame_index"] if is_top else "",
        "bottom_frame": "" if is_top else crop["frame_index"],
        "top_timestamp_ms": crop["timestamp_ms"] if is_top else "",
        "bottom_timestamp_ms": "" if is_top else crop["timestamp_ms"],
        "delta_frames": "", "delta_ms": "", "temporal_cost": "",
        "status": "unmatched_top" if is_top else "unmatched_bottom",
        "notes": "Aucune association admissible forcée dans l'alignement temporel.",
    }


def ambiguous_row(
    component: set[tuple[str, int]], ambiguous_edges: set[Edge],
    tops: list[dict[str, str]], bottoms: list[dict[str, str]],
) -> dict[str, object]:
    top_indices = sorted(index for position, index in component if position == "T")
    bottom_indices = sorted(index for position, index in component if position == "B")
    component_tops = [tops[index] for index in top_indices]
    component_bottoms = [bottoms[index] for index in bottom_indices]
    component_edges = sorted(
        edge for edge in ambiguous_edges
        if ("T", edge[0]) in component and ("B", edge[1]) in component
    )
    candidates = []
    for top_index, bottom_index in component_edges:
        top, bottom = tops[top_index], bottoms[bottom_index]
        delta = int(bottom["frame_index"]) - int(top["frame_index"])
        candidates.append(
            f"{top['crop_filename']}->{bottom['crop_filename']} "
            f"(delta={delta}, cost={temporal_cost(top, bottom)})"
        )
    return {
        "top_filename": joined(component_tops, "crop_filename"),
        "bottom_filename": joined(component_bottoms, "crop_filename"),
        "top_frame": joined(component_tops, "frame_index"),
        "bottom_frame": joined(component_bottoms, "frame_index"),
        "top_timestamp_ms": joined(component_tops, "timestamp_ms"),
        "bottom_timestamp_ms": joined(component_bottoms, "timestamp_ms"),
        "delta_frames": "", "delta_ms": "", "temporal_cost": "",
        "status": "ambiguous",
        "notes": "Associations optimales concurrentes : " + "; ".join(candidates),
    }


def build_alignment(
    tops: list[dict[str, str]], bottoms: list[dict[str, str]]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    optimum, possible_edges, forced_edges = classify_optimal_edges(tops, bottoms)
    ambiguous_edges = possible_edges - forced_edges
    components = ambiguous_components(ambiguous_edges)
    forced_top = {top_index for top_index, _ in forced_edges}
    forced_bottom = {bottom_index for _, bottom_index in forced_edges}
    ambiguous_top = {
        index for component in components for position, index in component if position == "T"
    }
    ambiguous_bottom = {
        index for component in components for position, index in component if position == "B"
    }
    events: list[tuple[int, int, dict[str, object]]] = []
    for top_index, bottom_index in sorted(forced_edges):
        events.append((int(tops[top_index]["frame_index"]), 0, paired_row(tops[top_index], bottoms[bottom_index])))
    for component in components:
        row = ambiguous_row(component, ambiguous_edges, tops, bottoms)
        event_frames = [
            int(tops[index]["frame_index"]) for position, index in component if position == "T"
        ] + [
            int(bottoms[index]["frame_index"]) - EXPECTED_DELTA
            for position, index in component if position == "B"
        ]
        events.append((min(event_frames), 1, row))
    for top_index, top in enumerate(tops):
        if top_index not in forced_top and top_index not in ambiguous_top:
            events.append((int(top["frame_index"]), 2, unmatched_row(top, "T")))
    for bottom_index, bottom in enumerate(bottoms):
        if bottom_index not in forced_bottom and bottom_index not in ambiguous_bottom:
            events.append((int(bottom["frame_index"]) - EXPECTED_DELTA, 3, unmatched_row(bottom, "B")))
    rows = [
        {"pair_id": f"pair_{number:04d}", **row}
        for number, (_, _, row) in enumerate(sorted(events, key=lambda item: item[:2]), 1)
    ]
    diagnostics = {
        "optimal_match_count": optimum[0],
        "optimal_total_temporal_cost": -optimum[1],
        "admissible_edge_count": sum(is_admissible(top, bottom) for top in tops for bottom in bottoms),
        "possible_optimal_edge_count": len(possible_edges),
        "forced_edge_count": len(forced_edges),
        "ambiguous_edge_count": len(ambiguous_edges),
        "ambiguous_group_count": len(components),
    }
    return rows, diagnostics


def split_filenames(value: object) -> list[str]:
    return [name for name in str(value).split("|") if name]


def validate_alignment(
    rows: list[dict[str, object]], tops: list[dict[str, str]], bottoms: list[dict[str, str]]
) -> None:
    expected_top = {row["crop_filename"] for row in tops}
    expected_bottom = {row["crop_filename"] for row in bottoms}
    actual_top = [name for row in rows for name in split_filenames(row["top_filename"])]
    actual_bottom = [name for row in rows for name in split_filenames(row["bottom_filename"])]
    if Counter(actual_top) != Counter(expected_top):
        raise ValueError("Les crops Top ne sont pas représentés exactement une fois")
    if Counter(actual_bottom) != Counter(expected_bottom):
        raise ValueError("Les crops Bottom ne sont pas représentés exactement une fois")
    matched = [row for row in rows if row["status"] == "matched_temporal"]
    top_order = {row["crop_filename"]: index for index, row in enumerate(tops)}
    bottom_order = {row["crop_filename"]: index for index, row in enumerate(bottoms)}
    sequence_indices = [
        (top_order[str(row["top_filename"])], bottom_order[str(row["bottom_filename"])])
        for row in matched
    ]
    if any(
        left_top >= right_top or left_bottom >= right_bottom
        for (left_top, left_bottom), (right_top, right_bottom) in zip(
            sequence_indices, sequence_indices[1:]
        )
    ):
        raise ValueError("L'ordre temporel 1-à-1 des paires n'est pas respecté")
    if any(int(row["temporal_cost"]) > TOLERANCE for row in matched):
        raise ValueError("Une paire dépasse la tolérance temporelle")


def save_pairs(rows: list[dict[str, object]]) -> None:
    temporary = PAIRS_CSV.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(PAIRS_CSV)


def numeric_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "population_std": None, "min": None, "max": None}
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "population_std": round(statistics.pstdev(values), 3),
        "min": min(values), "max": max(values),
    }


def build_summary(
    rows: list[dict[str, object]], tops: list[dict[str, str]],
    bottoms: list[dict[str, str]], diagnostics: dict[str, object],
) -> dict[str, object]:
    counts = Counter(str(row["status"]) for row in rows)
    matched = [row for row in rows if row["status"] == "matched_temporal"]
    frame_deltas = [int(row["delta_frames"]) for row in matched]
    millisecond_deltas = [float(row["delta_ms"]) for row in matched]
    costs = [int(row["temporal_cost"]) for row in matched]
    ambiguous = [row for row in rows if row["status"] == "ambiguous"]
    return {
        "method": {
            "type": "monotone_sequence_alignment",
            "objective": "maximize_matches_then_minimize_total_temporal_cost",
            "cost": "abs((frame_B - frame_T) - 7)",
            "admissible_deltas": [6, 7, 8],
            "identity_claim": "temporal_inference_not_visual_ground_truth",
        },
        "total_crops_top": len(tops), "total_crops_bottom": len(bottoms),
        "status_counts": {
            "matched_temporal": counts["matched_temporal"],
            "unmatched_top": counts["unmatched_top"],
            "unmatched_bottom": counts["unmatched_bottom"],
            "ambiguous": counts["ambiguous"],
        },
        "ambiguous_crop_counts": {
            "top": sum(len(split_filenames(row["top_filename"])) for row in ambiguous),
            "bottom": sum(len(split_filenames(row["bottom_filename"])) for row in ambiguous),
        },
        "coverage": {
            "top_matched_percentage": round(100 * len(matched) / len(tops), 2),
            "bottom_matched_percentage": round(100 * len(matched) / len(bottoms), 2),
            "definition": "Seules les paires matched_temporal sont comptées.",
        },
        "matched_delta_frames": numeric_stats(frame_deltas),
        "matched_delta_ms": numeric_stats(millisecond_deltas),
        "delta_frames_distribution": {
            str(delta): count for delta, count in sorted(Counter(frame_deltas).items())
        },
        "delta_categories": {
            "plus_6": frame_deltas.count(6), "plus_7": frame_deltas.count(7),
            "plus_8": frame_deltas.count(8),
            "other": sum(delta not in {6, 7, 8} for delta in frame_deltas),
        },
        "temporal_cost": {
            "mean": round(statistics.mean(costs), 3) if costs else None,
            "median": round(statistics.median(costs), 3) if costs else None,
            "distribution": {str(cost): count for cost, count in sorted(Counter(costs).items())},
        },
        "ambiguous_cases": [
            {
                "pair_id": row["pair_id"],
                "top_filenames": split_filenames(row["top_filename"]),
                "bottom_filenames": split_filenames(row["bottom_filename"]),
                "notes": row["notes"],
            }
            for row in ambiguous
        ],
        "unmatched_top_filenames": [
            row["top_filename"] for row in rows if row["status"] == "unmatched_top"
        ],
        "unmatched_bottom_filenames": [
            row["bottom_filename"] for row in rows if row["status"] == "unmatched_bottom"
        ],
        "alignment_diagnostics": diagnostics,
    }


def save_summary(summary: dict[str, object]) -> None:
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_results(summary: dict[str, object]) -> None:
    statuses, deltas = summary["status_counts"], summary["matched_delta_frames"]
    print(f"Crops Top/Bottom: {summary['total_crops_top']}/{summary['total_crops_bottom']}")
    print(
        f"Statuts: matched_temporal={statuses['matched_temporal']}, "
        f"unmatched_top={statuses['unmatched_top']}, "
        f"unmatched_bottom={statuses['unmatched_bottom']}, ambiguous={statuses['ambiguous']}"
    )
    print(f"Distribution delta_frames: {summary['delta_frames_distribution']}")
    print(f"Delta moyen/médian: {deltas['mean']}/{deltas['median']} frames")
    print(
        f"Coût moyen/médian: {summary['temporal_cost']['mean']}/"
        f"{summary['temporal_cost']['median']}"
    )
    print(f"Table: {PAIRS_CSV}")
    print(f"Résumé: {SUMMARY_JSON}")


def compute_alignment() -> tuple[list[dict[str, object]], dict[str, object]]:
    tops, bottoms = read_sequences()
    rows, diagnostics = build_alignment(tops, bottoms)
    validate_alignment(rows, tops, bottoms)
    summary = build_summary(rows, tops, bottoms, diagnostics)
    return rows, summary


def run_alignment() -> dict[str, object]:
    rows, summary = compute_alignment()
    save_pairs(rows)
    save_summary(summary)
    return summary


def audit() -> None:
    tops, bottoms = read_sequences()
    optimum, possible, forced = classify_optimal_edges(tops, bottoms)
    ambiguous_edges = possible - forced
    print(f"Crops Top/Bottom: {len(tops)}/{len(bottoms)}")
    print(f"Score optimal: {optimum[0]} associations, coût total {-optimum[1]}")
    print(f"Arêtes optimales possibles/forcées/ambiguës: {len(possible)}/{len(forced)}/{len(ambiguous_edges)}")
    for component in ambiguous_components(ambiguous_edges):
        top_names = sorted(tops[index]["crop_filename"] for pos, index in component if pos == "T")
        bottom_names = sorted(bottoms[index]["crop_filename"] for pos, index in component if pos == "B")
        print(f"Ambiguïté: Top={top_names} Bottom={bottom_names}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--audit", action="store_true", help="Diagnostiquer sans réécrire les sorties")
    modes.add_argument("--stats", action="store_true", help="Recalculer les statistiques sans réécrire les sorties")
    args = parser.parse_args()
    if args.audit:
        audit()
    elif args.stats:
        _, summary = compute_alignment()
        print_results(summary)
    else:
        print_results(run_alignment())


if __name__ == "__main__":
    main()
