"""Interface locale de revue humaine des 59 désaccords masked/principal.

Commandes:
    python analysis/review_masked_disagreements.py
    python analysis/review_masked_disagreements.py --audit
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "analysis" / "output" / "masked_disagreements.csv"
REVIEW_CSV = ROOT / "analysis" / "output" / "masked_disagreements_review.csv"
IMAGES_DIR = ROOT / "images"
EXPECTED_CONFLICTS = 59

REVIEW_COLUMNS = [
    "filename",
    "label_principal_before",
    "masked_label",
    "human_review",
    "review_status",
    "notes",
    "reviewed_at_utc",
]
CHOICES = {"v": "Vide", "n": "NonVide", "i": "Incertain"}
MASKED_BINARY_MAP = {
    "vide": "Vide",
    "chataigne": "NonVide",
    "chunks": "NonVide",
    "multiple": "NonVide",
}


def load_source(path: Path = SOURCE_CSV) -> pd.DataFrame:
    source = pd.read_csv(path, low_memory=False)
    required = {
        "filename", "label_principal", "masked_label", "hot_frac", "max_blob",
        "year", "cam_position", "cam_num", "sample_num", "chunk", "multiple",
        "masked_disagreement",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"Colonnes absentes de {path.name}: {', '.join(missing)}")
    if len(source) != EXPECTED_CONFLICTS:
        raise ValueError(
            f"Nombre de conflits inattendu: {len(source)} au lieu de {EXPECTED_CONFLICTS}"
        )
    if not source.filename.is_unique:
        raise ValueError("filename doit être unique dans masked_disagreements.csv")
    return source.reset_index(drop=True)


def initialize_review(source: pd.DataFrame, path: Path = REVIEW_CSV) -> pd.DataFrame:
    base = source[["filename", "label_principal", "masked_label"]].rename(
        columns={"label_principal": "label_principal_before"}
    )
    if path.exists():
        existing = pd.read_csv(path, dtype="string", keep_default_na=False)
        missing_columns = sorted(set(REVIEW_COLUMNS).difference(existing.columns))
        if missing_columns:
            raise ValueError(
                f"Colonnes absentes de {path.name}: {', '.join(missing_columns)}"
            )
        if not existing.filename.is_unique:
            raise ValueError(f"filename n'est pas unique dans {path.name}")
        unknown = sorted(set(existing.filename) - set(base.filename))
        if unknown:
            raise ValueError(f"Décisions orphelines dans {path.name}: {unknown[:5]}")
        review = base.merge(
            existing[["filename", "human_review", "review_status", "notes", "reviewed_at_utc"]],
            on="filename", how="left", validate="one_to_one",
        )
    else:
        review = base.copy()
        review["human_review"] = ""
        review["review_status"] = "pending"
        review["notes"] = ""
        review["reviewed_at_utc"] = ""

    for column in ("human_review", "review_status", "notes", "reviewed_at_utc"):
        review[column] = review[column].astype("string").fillna("")
    review.loc[review.human_review.eq(""), "review_status"] = "pending"
    review.loc[review.human_review.ne(""), "review_status"] = "reviewed"
    invalid = sorted(set(review.human_review) - {"", "Vide", "NonVide", "Incertain"})
    if invalid:
        raise ValueError(f"Décisions humaines inattendues: {invalid}")
    save_review(review, path)
    return review[REVIEW_COLUMNS]


def save_review(review: pd.DataFrame, path: Path = REVIEW_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    review[REVIEW_COLUMNS].to_csv(temporary, index=False)
    os.replace(temporary, path)


def binary_principal(value: object) -> str:
    return "Vide" if str(value).strip().casefold() == "vide" else "NonVide"


def binary_masked(value: object) -> str:
    normalized = str(value).strip().casefold()
    if normalized not in MASKED_BINARY_MAP:
        raise ValueError(f"masked_label inattendu pour la comparaison binaire: {value}")
    return MASKED_BINARY_MAP[normalized]


def review_summary(review: pd.DataFrame) -> dict[str, int]:
    reviewed = review.review_status.eq("reviewed")
    determinate = review.human_review.isin(["Vide", "NonVide"])
    principal_binary = review.label_principal_before.map(binary_principal)
    masked_binary = review.masked_label.map(binary_masked)
    agrees_principal = reviewed & determinate & review.human_review.eq(principal_binary)
    agrees_masked = reviewed & determinate & review.human_review.eq(masked_binary)
    disagrees_both = reviewed & determinate & ~agrees_principal & ~agrees_masked
    return {
        "total": len(review),
        "reviewed": int(reviewed.sum()),
        "pending": int((~reviewed).sum()),
        "Vide": int(review.human_review.eq("Vide").sum()),
        "NonVide": int(review.human_review.eq("NonVide").sum()),
        "Incertain": int(review.human_review.eq("Incertain").sum()),
        "agreement_with_label_principal": int(agrees_principal.sum()),
        "agreement_with_labels_masked": int(agrees_masked.sum()),
        "disagreement_with_both": int(disagrees_both.sum()),
    }


def print_summary(review: pd.DataFrame) -> None:
    summary = review_summary(review)
    print(
        "Revue: {reviewed}/{total} — Vide={Vide}, NonVide={NonVide}, "
        "Incertain={Incertain}, en attente={pending}".format(**summary)
    )
    print(
        "Accords: label_principal={agreement_with_label_principal}, "
        "labels_masked={agreement_with_labels_masked}, "
        "désaccord avec les deux={disagreement_with_both}".format(**summary)
    )


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image illisible: {path}")
    return image


def put_lines(
    canvas: np.ndarray,
    lines: list[str],
    x: int,
    y: int,
    color: tuple[int, int, int] = (235, 235, 235),
    scale: float = 0.56,
    spacing: int = 29,
) -> None:
    for offset, line in enumerate(lines):
        cv2.putText(
            canvas, line, (x, y + offset * spacing), cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, 1, cv2.LINE_AA,
        )


def render(source_row: pd.Series, review_row: pd.Series, index: int) -> np.ndarray:
    image = read_image(IMAGES_DIR / source_row.filename)
    canvas_height, canvas_width = 900, 1500
    image_width, panel_width = 1080, 420
    canvas = np.full((canvas_height, canvas_width, 3), 28, dtype=np.uint8)
    height, width = image.shape[:2]
    scale = min((image_width - 30) / width, (canvas_height - 30) / height)
    resized = cv2.resize(
        image, (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
    )
    resized_height, resized_width = resized.shape[:2]
    x_image = (image_width - resized_width) // 2
    y_image = (canvas_height - resized_height) // 2
    canvas[y_image:y_image + resized_height, x_image:x_image + resized_width] = resized
    cv2.line(canvas, (image_width, 0), (image_width, canvas_height), (90, 90, 90), 1)

    x = image_width + 22
    filename = str(source_row.filename)
    filename_lines = [filename[:42], filename[42:84]] if len(filename) > 42 else [filename]
    lines = [
        f"Image {index + 1}/{EXPECTED_CONFLICTS}",
        "",
        *filename_lines,
        "",
        f"year: {source_row.year}",
        f"position: {source_row.cam_position}",
        f"camera: {source_row.cam_num}",
        f"sample_num: {source_row.sample_num}",
        "",
        "V = Vide",
        "N = Non vide",
        "I = Incertain",
        "B = image precedente",
        "T = saisir/modifier notes",
        "S = afficher le resume",
        "Q = sauvegarder et quitter",
    ]
    if review_row.review_status == "reviewed":
        reveal = [
            "",
            "Decision deja validee:",
            f"human_review:   {review_row.human_review}",
            f"label_principal: {source_row.label_principal}",
            f"masked_label:    {source_row.masked_label}",
            f"Notes: {str(review_row.notes)[:42] or '-'}",
        ]
        controls_index = lines.index("V = Vide")
        lines[controls_index:controls_index] = reveal
    put_lines(canvas, lines, x, 32)
    return canvas


def run_gui(source: pd.DataFrame, review: pd.DataFrame) -> None:
    pending = review.index[review.review_status.ne("reviewed")].tolist()
    index = pending[0] if pending else 0
    window = "CastagNet - revue des conflits masked"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1500, 900)

    while True:
        canvas = render(source.iloc[index], review.iloc[index], index)
        cv2.imshow(window, canvas)
        key = cv2.waitKeyEx(0)
        char = chr(key & 0xFF).lower() if 0 <= (key & 0xFF) < 256 else ""

        if char in CHOICES:
            review.at[index, "human_review"] = CHOICES[char]
            review.at[index, "review_status"] = "reviewed"
            review.at[index, "reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
            save_review(review)
            print(f"Sauvegardé: {source.iloc[index].filename} -> {CHOICES[char]}")
            if index < len(source) - 1:
                index += 1
            else:
                print_summary(review)
        elif char == "b":
            index = max(0, index - 1)
        elif char == "t":
            cv2.destroyWindow(window)
            note = input(f"Notes pour {source.iloc[index].filename} (vide pour effacer): ").strip()
            review.at[index, "notes"] = note
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


def audit(source: pd.DataFrame, review: pd.DataFrame) -> None:
    missing = [name for name in source.filename if not (IMAGES_DIR / name).is_file()]
    unreadable: list[str] = []
    for filename in source.filename:
        if filename in missing:
            continue
        try:
            read_image(IMAGES_DIR / filename)
        except ValueError:
            unreadable.append(filename)
    if missing or unreadable:
        raise ValueError(
            f"Images manquantes: {len(missing)}; images illisibles: {len(unreadable)}"
        )
    print(f"Conflits chargés: {len(source)}; filenames uniques: {source.filename.nunique()}")
    print(f"Images présentes et lisibles: {len(source)}")
    print(f"Fichier de décisions: {REVIEW_CSV}")
    print_summary(review)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit", action="store_true",
        help="initialise/contrôle le fichier de revue sans ouvrir l'interface",
    )
    args = parser.parse_args()
    source = load_source()
    review = initialize_review(source)
    if args.audit:
        audit(source, review)
    else:
        run_gui(source, review)


if __name__ == "__main__":
    main()
