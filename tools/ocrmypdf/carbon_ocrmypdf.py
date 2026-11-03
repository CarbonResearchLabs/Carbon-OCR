"""Carbon-OCR as OCRmyPDF's OCR engine: its plugin interface, on carbon-ocr.

usage:
  ocrmypdf --plugin tools/ocrmypdf/carbon_ocrmypdf.py --carbon-set models/fullerene-3M in.pdf out.pdf
  ocrmypdf --plugin ... --pdf-renderer sandwich ...    # our text-only PDFs under the pages

OCRmyPDF (17.x; tools/ocrmypdf/requirements.txt pins it) rasterizes each page and asks the
engine for it:

  generate_hocr   carbon-ocr ocr PAGE --format hocr, and the page's text beside it: the default
                  renderer (fpdf2) draws the invisible text from the hOCR
  generate_pdf    carbon-ocr ocr PAGE --format pdf --text-only at the image's own dpi: the text
                  layer alone, which the sandwich renderer grafts onto the original page

OCRmyPDF's built-in Tesseract plugin checks for tesseract on every run, so it must be installed
though it reads nothing. Ghostscript is needed only for PDF/A: OCRmyPDF rasterizes with
pypdfium2, and --output-type pdf goes without it.

Both run with --as-given: no quarter turns and no deskew, so the boxes are the page OCRmyPDF
gave (its --rotate-pages and --deskew turn and level pages before they reach the engine). The
engine's orientation is not reported (get_orientation's confidence is 0), so --rotate-pages
leaves pages as they are. Languages are the Latin-script ones rec-latin reads; -l takes
Tesseract's codes.

Options: --carbon-set DIR (a Fullerene set; else $CARBON_OCR_SET), --carbon-ocr PATH (the
carbon-ocr program; else $CARBON_OCR, else the one on PATH), --carbon-workers N (workers for a
page: OCRmyPDF runs pages in parallel itself, so 1 by default).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from ocrmypdf import hookimpl
from ocrmypdf.exceptions import MissingDependencyError
from ocrmypdf.pluginspec import OcrEngine, OrientationConfidence

LATIN = {"eng", "deu", "fra", "spa", "ita", "por", "nld", "swe", "dan", "nor", "fin", "isl", "pol", "ces", "slk",
         "slv", "hrv", "bos", "srp_latn", "ron", "hun", "tur", "est", "lav", "lit", "cat", "glg", "eus", "gle", "cym",
         "mlt", "sqi", "ind", "msa", "tgl", "afr", "swa", "lat", "fao", "ltz", "bre", "cos", "oci", "frk"}


def _find(options) -> str | None:
    return getattr(options, "carbon_ocr", None) or os.environ.get("CARBON_OCR") or shutil.which("carbon-ocr")


def _program(options) -> str:
    exe = _find(options)
    if not exe:
        raise MissingDependencyError("carbon-ocr: not on PATH; give --carbon-ocr PATH or set CARBON_OCR")
    return exe


def _version(exe: str | None) -> str:
    """The second word of `carbon-ocr version` ("carbon-ocr 0.1.0 (backends: ort, native)")."""
    if not exe:
        return "unknown"
    try:
        p = subprocess.run([exe, "version"], capture_output=True, text=True)
    except OSError:
        return "unknown"
    words = p.stdout.split()
    return words[1] if p.returncode == 0 and len(words) > 1 else "unknown"


def _set(options) -> str:
    s = getattr(options, "carbon_set", None) or os.environ.get("CARBON_OCR_SET")
    if not s:
        raise MissingDependencyError("a Fullerene set: give --carbon-set DIR or set CARBON_OCR_SET")
    return s


def _ocr(input_file: Path, options, *args: str) -> None:
    cmd = [_program(options), "ocr", str(input_file), "--set", _set(options), "--as-given", "--workers",
           str(getattr(options, "carbon_workers", 1) or 1), *args]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed ({p.returncode}): {p.stderr.strip()}")


def _dpi(image: Path) -> int:
    from PIL import Image

    with Image.open(image) as im:
        dpi = im.info.get("dpi", (300, 300))[0]
    return max(36, int(round(float(dpi))))


class CarbonOcrEngine(OcrEngine):
    @staticmethod
    def version() -> str:
        return _version(_find(None))

    @staticmethod
    def creator_tag(options) -> str:
        return f"Carbon-OCR {_version(_find(options))}"

    def __str__(self):
        return f"Carbon-OCR {self.version()}"

    @staticmethod
    def languages(options) -> set[str]:
        return set(LATIN)

    @staticmethod
    def get_orientation(input_file: Path, options) -> OrientationConfidence:
        return OrientationConfidence(angle=0, confidence=0.0)

    @staticmethod
    def get_deskew(input_file: Path, options) -> float:
        return 0.0

    @staticmethod
    def generate_hocr(input_file: Path, output_hocr: Path, output_text: Path, options) -> None:
        _ocr(input_file, options, "--format", "hocr", "--output", str(output_hocr), "--text-out", str(output_text))

    @staticmethod
    def generate_pdf(input_file: Path, output_pdf: Path, output_text: Path, options) -> None:
        _ocr(input_file, options, "--format", "pdf", "--text-only", "--image-dpi", str(_dpi(input_file)), "--output",
             str(output_pdf), "--text-out", str(output_text))


@hookimpl
def add_options(parser):
    g = parser.add_argument_group("Carbon-OCR", "Carbon-OCR as the OCR engine")
    g.add_argument("--carbon-set", metavar="DIR", help="a Fullerene set (default: $CARBON_OCR_SET)")
    g.add_argument("--carbon-ocr", metavar="PATH", help="the carbon-ocr program (default: $CARBON_OCR, then PATH)")
    g.add_argument("--carbon-workers", type=int, default=1, metavar="N",
                   help="carbon-ocr's workers for a page (OCRmyPDF runs pages in parallel; default 1)")


@hookimpl
def check_options(options):
    _program(options)
    _set(options)
    bad = set(getattr(options, "languages", None) or []) - LATIN
    if bad:
        raise MissingDependencyError(f"Carbon-OCR reads Latin-script languages; not {', '.join(sorted(bad))}")


@hookimpl
def get_ocr_engine():
    return CarbonOcrEngine()
