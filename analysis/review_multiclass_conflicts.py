"""Seconde revue aveugle des seuls conflits NonVide sans classe déterminée."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import pandas as pd

from review_masked_disagreements import IMAGES_DIR, put_lines, read_image


ROOT = Path(__file__).resolve().parents[1]
ADJUDICATED_CSV = ROOT / "analysis" / "output" / "masked_disagreements_adjudicated.csv"
REVIEW_CSV = ROOT / "analysis" / "output" / "multiclass_conflicts_review.csv"
CHOICES = {"c": "Conforme", "n": "NON Conforme", "p": "PIETRA", "i": "Incertain"}
REVIEW_COLUMNS = ["filename", "human_review", "review_status", "notes", "reviewed_at_utc"]


def load_cases() -> pd.DataFrame:
    adjudicated = pd.read_csv(ADJUDICATED_CSV, low_memory=False)
    required = {
        "filename", "year", "cam_position", "cam_num", "sample_num", "review_action"
    }
    missing = sorted(required.difference(adjudicated.columns))
    if missing:
        raise ValueError(f"Colonnes absentes du fichier d'adjudication: {', '.join(missing)}")
    cases = adjudicated.loc[
        adjudicated.review_action.eq("NEED_MULTICLASS_REVIEW"),
        ["filename", "year", "cam_position", "cam_num", "sample_num"],
    ].copy()
    if cases.empty:
        raise ValueError("Aucun cas NEED_MULTICLASS_REVIEW")
    if not cases.filename.is_unique:
        raise ValueError("filename n'est pas unique dans les cas multiclasses")
    return cases.reset_index(drop=True)


def save_review(review: pd.DataFrame) -> None:
    REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary = REVIEW_CSV.with_suffix(REVIEW_CSV.suffix + ".tmp")
    review[REVIEW_COLUMNS].to_csv(temporary, index=False)
    os.replace(temporary, REVIEW_CSV)


def initialize_review(cases: pd.DataFrame) -> pd.DataFrame:
    if REVIEW_CSV.exists():
        existing = pd.read_csv(REVIEW_CSV, dtype="string", keep_default_na=False)
        missing = sorted(set(REVIEW_COLUMNS).difference(existing.columns))
        if missing or not existing.filename.is_unique:
            raise ValueError(f"Fichier de revue multiclasses invalide: {missing}")
        if set(existing.filename) != set(cases.filename):
            raise ValueError("Le fichier de revue ne contient pas exactement les cas multiclasses requis")
        review = cases[["filename"]].merge(
            existing[REVIEW_COLUMNS], on="filename", how="left", validate="one_to_one"
        )
    else:
        review = cases[["filename"]].copy()
        review["human_review"] = ""
        review["review_status"] = "pending"
        review["notes"] = ""
        review["reviewed_at_utc"] = ""
    for column in REVIEW_COLUMNS[1:]:
        review[column] = review[column].astype("string").fillna("")
    invalid = sorted(set(review.human_review) - {"", *CHOICES.values()})
    if invalid:
        raise ValueError(f"Décisions multiclasses inattendues: {invalid}")
    review.loc[review.human_review.eq(""), "review_status"] = "pending"
    review.loc[review.human_review.ne(""), "review_status"] = "reviewed"
    save_review(review)
    return review[REVIEW_COLUMNS]


def print_summary(review: pd.DataFrame) -> None:
    counts = review.human_review.value_counts()
    reviewed = int(review.review_status.eq("reviewed").sum())
    print(
        f"Revue multiclasses: {reviewed}/{len(review)} — "
        f"Conforme={int(counts.get('Conforme', 0))}, "
        f"NON Conforme={int(counts.get('NON Conforme', 0))}, "
        f"PIETRA={int(counts.get('PIETRA', 0))}, "
        f"Incertain={int(counts.get('Incertain', 0))}, "
        f"en attente={len(review) - reviewed}"
    )


def render(case: pd.Series, review_row: pd.Series, index: int, total: int):
    image = read_image(IMAGES_DIR / case.filename)
    canvas_height, canvas_width = 900, 1500
    image_width = 1080
    import numpy as np

    canvas = np.full((canvas_height, canvas_width, 3), 28, dtype=np.uint8)
    height, width = image.shape[:2]
    scale = min((image_width - 30) / width, (canvas_height - 30) / height)
    resized = cv2.resize(
        image, (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
    )
    rh, rw = resized.shape[:2]
    x_image, y_image = (image_width - rw) // 2, (canvas_height - rh) // 2
    canvas[y_image:y_image + rh, x_image:x_image + rw] = resized
    cv2.line(canvas, (image_width, 0), (image_width, canvas_height), (90, 90, 90), 1)

    filename = str(case.filename)
    filename_lines = [filename[:42], filename[42:84]] if len(filename) > 42 else [filename]
    lines = [
        f"Image {index + 1}/{total}", "", *filename_lines, "",
        f"year: {case.year}",
        f"position: {case.cam_position}",
        f"camera: {case.cam_num}",
        f"sample_num: {case.sample_num}",
        "",
        "C = Conforme",
        "N = NON Conforme",
        "P = PIETRA",
        "I = Incertain",
        "B = image precedente",
        "T = saisir/modifier notes",
        "S = afficher le resume",
        "Q = sauvegarder et quitter",
    ]
    if review_row.review_status == "reviewed":
        reveal = ["", f"Decision deja validee: {review_row.human_review}"]
        lines[lines.index("C = Conforme"):lines.index("C = Conforme")] = reveal
    put_lines(canvas, lines, image_width + 22, 32)
    return canvas


def run_gui(cases: pd.DataFrame, review: pd.DataFrame) -> None:
    pending = review.index[review.review_status.ne("reviewed")].tolist()
    index = pending[0] if pending else 0
    window = "CastagNet - revue multiclasses aveugle"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1500, 900)
    while True:
        cv2.imshow(window, render(cases.iloc[index], review.iloc[index], index, len(cases)))
        key = cv2.waitKeyEx(0)
        char = chr(key & 0xFF).lower() if 0 <= (key & 0xFF) < 256 else ""
        if char in CHOICES:
            review.at[index, "human_review"] = CHOICES[char]
            review.at[index, "review_status"] = "reviewed"
            review.at[index, "reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
            save_review(review)
            print(f"Sauvegardé: {cases.iloc[index].filename} -> {CHOICES[char]}")
            if index < len(cases) - 1:
                index += 1
            else:
                print_summary(review)
        elif char == "b":
            index = max(0, index - 1)
        elif char == "t":
            cv2.destroyWindow(window)
            review.at[index, "notes"] = input(
                f"Notes pour {cases.iloc[index].filename} (vide pour effacer): "
            ).strip()
            save_review(review)
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window, 1500, 900)
        elif char == "s":
            print_summary(review)
        elif char == "q" or key == 27:
            save_review(review)
            print_summary(review)
            break
    cv2.destroyAllWindows()


def audit(cases: pd.DataFrame, review: pd.DataFrame) -> None:
    missing = [name for name in cases.filename if not (IMAGES_DIR / name).is_file()]
    unreadable: list[str] = []
    for filename in cases.filename:
        if filename in missing:
            continue
        try:
            read_image(IMAGES_DIR / filename)
        except ValueError:
            unreadable.append(filename)
    if missing or unreadable:
        raise ValueError(f"Images manquantes={len(missing)}, illisibles={len(unreadable)}")
    print(f"Cas multiclasses chargés: {len(cases)}; images présentes et lisibles: {len(cases)}")
    print(f"Fichier de décisions: {REVIEW_CSV}")
    print_summary(review)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", help="initialise et contrôle sans GUI")
    args = parser.parse_args()
    cases = load_cases()
    review = initialize_review(cases)
    if args.audit:
        audit(cases, review)
    else:
        run_gui(cases, review)


if __name__ == "__main__":
    main()
