"""Render the section 4.3 DOCX to a paginated PDF for layout control."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from zipfile import ZipFile
import xml.etree.ElementTree as ET
from pathlib import Path

import mammoth
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent.parent
DOCX = ROOT / "reports" / "Rapport_CastagNet_4_3_ONNX_Embarquement_FINAL.docx"
HTML = ROOT / "analysis" / "output" / "report_4_3_preview.html"
PDF = ROOT / "reports" / "Rapport_CastagNet_4_3_ONNX_Embarquement_FINAL.pdf"
SUMMARY = ROOT / "analysis" / "output" / "report_4_3_generation_summary.json"


def find_browser() -> Path:
    for executable in ("chrome", "chromium", "google-chrome", "msedge"):
        found = shutil.which(executable)
        if found:
            return Path(found)
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        for relative in (
            Path("Google/Chrome/Application/chrome.exe"),
            Path("Microsoft/Edge/Application/msedge.exe"),
        ):
            candidate = Path(program_files) / relative
            if candidate.is_file():
                return candidate
    raise FileNotFoundError("Chrome, Chromium ou Edge est requis pour le rendu PDF")

CSS = """
@page { size: A4; margin: 16mm 17mm 16mm 18mm; }
body { font-family: Arial, sans-serif; font-size: 9.3pt; line-height: 1.2; color: #263238; }
h1 { color: #17324d; font-size: 17pt; page-break-before: always; margin: 0 0 8pt; }
h2 { color: #245b78; font-size: 13.5pt; margin: 12pt 0 6pt; page-break-after: avoid; }
h3 { color: #2a7f7f; font-size: 11pt; margin: 9pt 0 4pt; page-break-after: avoid; }
p { margin: 0 0 5pt; orphans: 3; widows: 3; }
ul { margin: 2pt 0 6pt; }
table { width: 100%; border-collapse: collapse; margin: 5pt 0 7pt; font-size: 7.6pt; page-break-inside: avoid; }
th, td { border: 0.5pt solid #cfd8df; padding: 3.2pt 4pt; vertical-align: middle; }
tr:first-child td, th { background: #17324d; color: white; font-weight: bold; }
img { max-width: 100%; max-height: 104mm; object-fit: contain; display: block; margin: 4pt auto; }
"""


def main() -> None:
    with DOCX.open("rb") as stream:
        converted = mammoth.convert_to_html(stream)
    html = (
        "<!doctype html><html lang='fr'><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{converted.value}</body></html>"
    )
    HTML.write_text(html, encoding="utf-8")
    browser = find_browser()
    subprocess.run(
        [
            str(browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
            f"--print-to-pdf={PDF}", HTML.resolve().as_uri(),
        ],
        check=True,
    )
    reader = PdfReader(PDF)
    empty_pages = [index + 1 for index, page in enumerate(reader.pages) if not (page.extract_text() or "").strip()]
    with ZipFile(DOCX) as archive:
        properties = ET.fromstring(archive.read("docProps/app.xml"))
    namespace = {"p": "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"}
    word_page_count = int(properties.findtext("p:Pages", default="0", namespaces=namespace))
    if SUMMARY.is_file():
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
        summary["control_pdf"] = str(PDF.resolve())
        summary.pop("page_count", None)
        summary["word_page_count_document_property"] = word_page_count
        summary["control_pdf_page_count"] = len(reader.pages)
        summary["empty_pages"] = empty_pages
        summary["layout_control"] = "PDF de contrôle généré depuis le DOCX avec Mammoth et Chrome headless"
        SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"pages={len(reader.pages)}")
    print(f"word_pages={word_page_count}")
    print(f"empty_pages={empty_pages}")
    print(f"pdf={PDF}")
    if converted.messages:
        print("mammoth_messages=")
        for message in converted.messages:
            print(message)


if __name__ == "__main__":
    main()
