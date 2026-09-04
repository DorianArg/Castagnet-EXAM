"""Produit un aperçu PDF paginé du rapport DOCX pour contrôle local de mise en page."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import mammoth


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "reports" / "Rapport_CastagNet_4_1_Qualite_Dataset.docx"
HTML_OUT = ROOT / "analysis" / "report_assets" / "report_preview.html"
PDF_OUT = ROOT / "analysis" / "report_assets" / "report_preview.pdf"


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
@page { size: A4; margin: 17mm 18mm 17mm 19mm;
  @bottom-center { content: "CastagNet — §4.1  |  " counter(page); color: #6b7782; font-size: 8pt; }
}
body { font-family: Arial, sans-serif; font-size: 9.4pt; line-height: 1.22; color: #24313d; }
h1 { color: #17324d; font-size: 17pt; page-break-before: always; margin: 0 0 8pt 0; }
h2 { color: #087e8b; font-size: 13pt; margin: 12pt 0 5pt 0; page-break-after: avoid; }
h3 { color: #087e8b; font-size: 10.5pt; page-break-after: avoid; }
p { margin: 0 0 5pt 0; orphans: 3; widows: 3; }
table { width: 100%; border-collapse: collapse; margin: 5pt 0 8pt 0; font-size: 8pt; page-break-inside: avoid; }
th, td { border: 0.5pt solid #d6dee5; padding: 4pt 5pt; vertical-align: middle; }
tr:first-child td, th { background: #17324d; color: white; font-weight: bold; }
img { max-width: 100%; max-height: 112mm; object-fit: contain; display: block; margin: 4pt auto; }
p.caption { text-align: center; font-style: italic; color: #566573; font-size: 8pt; page-break-before: avoid; }
ul { margin-top: 2pt; margin-bottom: 5pt; }
"""


def main() -> None:
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    with DOCX.open("rb") as source:
        result = mammoth.convert_to_html(source)
    body = result.value.replace(
        "<p>Synthèse décisionnelle</p>", "<h1>Synthèse décisionnelle</h1>", 1
    ).replace("<p>Sommaire</p>", "<h1>Sommaire</h1>", 1)
    html = f"<!doctype html><html lang='fr'><meta charset='utf-8'><style>{CSS}</style><body>{body}</body></html>"
    HTML_OUT.write_text(html, encoding="utf-8")
    browser = find_browser()
    subprocess.run(
        [
            str(browser), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
            f"--print-to-pdf={PDF_OUT}", HTML_OUT.resolve().as_uri(),
        ],
        check=True,
    )
    print(PDF_OUT)
    if result.messages:
        print("Messages Mammoth:")
        for message in result.messages:
            print(message)


if __name__ == "__main__":
    main()
