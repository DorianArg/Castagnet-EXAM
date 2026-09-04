"""Extraction manuelle de frames et de crops depuis les extraits vidéo CastagNet.

Exemples :
    python analysis/video_manual_crop.py extrait-cam-1-t-conforme.avi --cam-position T
    python analysis/video_manual_crop.py extrait-cam-1-b-conforme.avi --cam-position B
    python analysis/video_manual_crop.py --inspect
    python analysis/video_manual_crop.py --stats
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
EXTRACTION_DIR = ROOT / "analysis" / "extracted_video"
METADATA_CSV = EXTRACTION_DIR / "crops_metadata.csv"
SUMMARY_CSV = EXTRACTION_DIR / "extraction_summary.csv"

KNOWN_VIDEOS = {
    "T": ROOT / "extrait-cam-1-t-conforme.avi",
    "B": ROOT / "extrait-cam-1-b-conforme.avi",
}

METADATA_COLUMNS = [
    "crop_filename",
    "source_video",
    "cam_position",
    "frame_index",
    "timestamp_ms",
    "crop_index",
    "x",
    "y",
    "width",
    "height",
    "original_frame_width",
    "original_frame_height",
    "notes",
]

SUMMARY_COLUMNS = [
    "source_video",
    "cam_position",
    "total_video_frames",
    "frames_with_crops",
    "total_crops",
    "frames_with_multiple_crops",
    "mean_crops_per_selected_frame",
    "mean_crop_width",
    "median_crop_width",
    "mean_crop_height",
    "median_crop_height",
    "retained_frame_proportion",
]

WINDOW_NAME = "CastagNet - extraction manuelle"


def extraction_subdir(cam_position: str) -> Path:
    return EXTRACTION_DIR / ("top" if cam_position == "T" else "bottom")


def ensure_output_structure() -> None:
    for position in ("T", "B"):
        base = extraction_subdir(position)
        (base / "frames").mkdir(parents=True, exist_ok=True)
        (base / "crops").mkdir(parents=True, exist_ok=True)
    if not METADATA_CSV.exists():
        with METADATA_CSV.open("w", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=METADATA_COLUMNS).writeheader()


def inspect_video(path: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    capture.release()

    duration_ms = frame_count * 1000.0 / fps if fps > 0 else 0.0
    return {
        "path": path,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "codec": codec,
        "duration_ms": duration_ms,
    }


def print_video_info(info: dict[str, object]) -> None:
    print(info["path"].name)
    print(f"  Durée: {info['duration_ms'] / 1000:.3f} s")
    print(f"  Frames: {info['frame_count']}")
    print(f"  FPS: {info['fps']:.3f}")
    print(f"  Résolution: {info['width']} x {info['height']}")
    print(f"  Codec FourCC: {info['codec']}")


def read_metadata() -> list[dict[str, str]]:
    ensure_output_structure()
    with METADATA_CSV.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != METADATA_COLUMNS:
            raise ValueError("Schéma inattendu dans crops_metadata.csv")
        return list(reader)


def append_metadata(row: dict[str, object]) -> None:
    with METADATA_CSV.open("a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=METADATA_COLUMNS).writerow(row)


def next_crop_index(
    rows: list[dict[str, str]], source_video: str, cam_position: str, frame_index: int
) -> int:
    existing = [
        int(row["crop_index"])
        for row in rows
        if row["source_video"] == source_video
        and row["cam_position"] == cam_position
        and int(row["frame_index"]) == frame_index
    ]
    return max(existing, default=0) + 1


def read_frame(capture: cv2.VideoCapture, frame_index: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = capture.read()
    if not success:
        raise RuntimeError(f"Impossible de lire la frame {frame_index}")
    return frame


def draw_hud(frame, frame_index: int, total_frames: int, fps: float, crop_count: int):
    display = frame.copy()
    timestamp_ms = frame_index * 1000.0 / fps
    text = (
        f"Frame {frame_index}/{total_frames - 1} | {timestamp_ms:.1f} ms | "
        f"crops enregistres: {crop_count}"
    )
    cv2.rectangle(display, (0, 0), (display.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        display,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return display


def save_manual_crop(
    frame,
    video_path: Path,
    cam_position: str,
    frame_index: int,
    fps: float,
    rows: list[dict[str, str]],
    notes: str,
) -> bool:
    roi_window = "Dessiner puis ENTREE/ESPACE pour valider (C pour annuler)"
    roi = cv2.selectROI(roi_window, frame, False, False)
    cv2.destroyWindow(roi_window)
    x, y, width, height = (int(value) for value in roi)
    if width <= 0 or height <= 0:
        print("Crop annulé.")
        return False

    frame_height, frame_width = frame.shape[:2]
    if x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
        raise ValueError("Le crop sélectionné dépasse les limites de la frame")

    output_base = extraction_subdir(cam_position)
    frame_filename = f"{cam_position}_frame_{frame_index:06d}.jpg"
    frame_path = output_base / "frames" / frame_filename
    if not frame_path.exists() and not cv2.imwrite(str(frame_path), frame):
        raise RuntimeError(f"Échec de l'écriture de {frame_path}")

    crop_index = next_crop_index(rows, video_path.name, cam_position, frame_index)
    while True:
        crop_filename = (
            f"{cam_position}_frame_{frame_index:06d}_crop_{crop_index:02d}.jpg"
        )
        crop_path = output_base / "crops" / crop_filename
        if not crop_path.exists() and all(
            row["crop_filename"] != crop_filename for row in rows
        ):
            break
        crop_index += 1

    crop = frame[y : y + height, x : x + width]
    if not cv2.imwrite(str(crop_path), crop):
        raise RuntimeError(f"Échec de l'écriture de {crop_path}")

    row: dict[str, object] = {
        "crop_filename": crop_filename,
        "source_video": video_path.name,
        "cam_position": cam_position,
        "frame_index": frame_index,
        "timestamp_ms": round(frame_index * 1000.0 / fps, 3),
        "crop_index": crop_index,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "original_frame_width": frame_width,
        "original_frame_height": frame_height,
        "notes": notes,
    }
    append_metadata(row)
    rows.append({key: str(value) for key, value in row.items()})
    print(f"Crop enregistré : {crop_path}")
    return True


def run_manual_extraction(
    video_path: Path, cam_position: str, start_frame: int, notes: str
) -> None:
    ensure_output_structure()
    info = inspect_video(video_path)
    frame_count = int(info["frame_count"])
    fps = float(info["fps"])
    if frame_count <= 0 or fps <= 0:
        raise ValueError("La vidéo ne fournit pas un nombre de frames et un FPS valides")

    frame_index = max(0, min(start_frame, frame_count - 1))
    capture = cv2.VideoCapture(str(video_path))
    rows = read_metadata()

    print(
        "Touches : A/D ou gauche/droite = -/+1, S/Z ou bas/haut = -/+10, "
        "C = crop, Q ou ECHAP = quitter"
    )
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = read_frame(capture, frame_index)
            current_crops = sum(
                row["source_video"] == video_path.name
                and row["cam_position"] == cam_position
                and int(row["frame_index"]) == frame_index
                for row in rows
            )
            cv2.imshow(
                WINDOW_NAME,
                draw_hud(frame, frame_index, frame_count, fps, current_crops),
            )
            key = cv2.waitKeyEx(0)

            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("d"), ord("D"), 2555904, 65363):
                frame_index = min(frame_count - 1, frame_index + 1)
            elif key in (ord("a"), ord("A"), 2424832, 65361):
                frame_index = max(0, frame_index - 1)
            elif key in (ord("z"), ord("Z"), ord("]"), 2490368, 65362):
                frame_index = min(frame_count - 1, frame_index + 10)
            elif key in (ord("s"), ord("S"), ord("["), 2621440, 65364):
                frame_index = max(0, frame_index - 10)
            elif key in (ord("c"), ord("C")):
                save_manual_crop(
                    frame,
                    video_path,
                    cam_position,
                    frame_index,
                    fps,
                    rows,
                    notes,
                )
    finally:
        capture.release()
        cv2.destroyAllWindows()

    generate_statistics()


def validate_metadata(rows: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    filenames = [row["crop_filename"] for row in rows]
    duplicates = [name for name, count in Counter(filenames).items() if count > 1]
    if duplicates:
        errors.append(f"Noms de crops dupliqués : {', '.join(duplicates)}")

    for row in rows:
        try:
            position = row["cam_position"]
            crop_path = extraction_subdir(position) / "crops" / row["crop_filename"]
            if not crop_path.is_file():
                errors.append(f"Crop référencé absent : {crop_path}")

            x, y = int(row["x"]), int(row["y"])
            width, height = int(row["width"]), int(row["height"])
            frame_width = int(row["original_frame_width"])
            frame_height = int(row["original_frame_height"])
            if (
                x < 0
                or y < 0
                or width <= 0
                or height <= 0
                or x + width > frame_width
                or y + height > frame_height
            ):
                errors.append(f"Crop hors limites : {row['crop_filename']}")
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"Métadonnées invalides pour {row.get('crop_filename', '?')} : {error}")
    return errors


def generate_statistics() -> None:
    rows = read_metadata()
    errors = validate_metadata(rows)
    summaries: list[dict[str, object]] = []

    for position, video_path in KNOWN_VIDEOS.items():
        info = inspect_video(video_path)
        video_rows = [
            row
            for row in rows
            if row["source_video"] == video_path.name and row["cam_position"] == position
        ]
        per_frame = Counter(int(row["frame_index"]) for row in video_rows)
        widths = [int(row["width"]) for row in video_rows]
        heights = [int(row["height"]) for row in video_rows]
        selected_frames = len(per_frame)
        total_frames = int(info["frame_count"])
        summaries.append(
            {
                "source_video": video_path.name,
                "cam_position": position,
                "total_video_frames": total_frames,
                "frames_with_crops": selected_frames,
                "total_crops": len(video_rows),
                "frames_with_multiple_crops": sum(count > 1 for count in per_frame.values()),
                "mean_crops_per_selected_frame": round(len(video_rows) / selected_frames, 3)
                if selected_frames
                else 0.0,
                "mean_crop_width": round(statistics.mean(widths), 3) if widths else 0.0,
                "median_crop_width": round(statistics.median(widths), 3) if widths else 0.0,
                "mean_crop_height": round(statistics.mean(heights), 3) if heights else 0.0,
                "median_crop_height": round(statistics.median(heights), 3) if heights else 0.0,
                "retained_frame_proportion": round(selected_frames / total_frames, 6)
                if total_frames
                else 0.0,
            }
        )

    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summaries)

    for summary in summaries:
        print(f"{summary['cam_position']} - {summary['source_video']}")
        print(
            f"  Frames retenues: {summary['frames_with_crops']}/{summary['total_video_frames']} "
            f"({100 * float(summary['retained_frame_proportion']):.2f} %)"
        )
        print(f"  Crops: {summary['total_crops']}")
        print(f"  Frames avec plusieurs crops: {summary['frames_with_multiple_crops']}")
        print(f"  Moyenne crops/frame retenue: {summary['mean_crops_per_selected_frame']}")
        print(
            f"  Largeur moyenne/médiane: {summary['mean_crop_width']}/"
            f"{summary['median_crop_width']} px"
        )
        print(
            f"  Hauteur moyenne/médiane: {summary['mean_crop_height']}/"
            f"{summary['median_crop_height']} px"
        )

    if errors:
        print("Contrôles en erreur :")
        for error in errors:
            print(f"  - {error}")
        raise RuntimeError(f"{len(errors)} erreur(s) détectée(s) dans les extractions")
    print("Contrôles OK : noms uniques, fichiers présents et crops dans les limites.")
    print(f"Résumé enregistré : {SUMMARY_CSV}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path, help="Vidéo à parcourir")
    parser.add_argument("--cam-position", choices=["T", "B"], help="Position de la caméra")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame affichée au démarrage")
    parser.add_argument("--notes", default="", help="Note ajoutée aux crops de cette session")
    parser.add_argument("--inspect", action="store_true", help="Afficher les propriétés des deux vidéos")
    parser.add_argument("--stats", action="store_true", help="Contrôler les crops et produire les métriques")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inspect:
        for video_path in KNOWN_VIDEOS.values():
            print_video_info(inspect_video(video_path))
        return
    if args.stats:
        generate_statistics()
        return
    if args.video is None or args.cam_position is None:
        raise SystemExit("Une vidéo et --cam-position T|B sont requis pour l'extraction")

    video_path = args.video.resolve()
    if not video_path.is_file():
        raise SystemExit(f"Vidéo introuvable : {video_path}")
    expected_position = next(
        (
            position
            for position, known_path in KNOWN_VIDEOS.items()
            if known_path.name == video_path.name
        ),
        None,
    )
    if expected_position is not None and args.cam_position != expected_position:
        raise SystemExit(
            f"{video_path.name} correspond à la position {expected_position}, "
            f"pas {args.cam_position}"
        )
    run_manual_extraction(video_path, args.cam_position, args.start_frame, args.notes)


if __name__ == "__main__":
    main()
