"""The OCRmyPDF plugin (tools/ocrmypdf/carbon_ocrmypdf.py) on the fake set's page: its hOCR, its
text-only PDF at the image's dpi, and the text beside each; then whole OCRmyPDF runs, with either
renderer. Needs ocrmypdf (tools/ocrmypdf/requirements.txt), and carbon-ocr (a build's, or
$CARBON_OCR); the runs need tesseract on PATH too, which OCRmyPDF checks for on every run."""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

pytest.importorskip("ocrmypdf")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data"
PLUGIN = ROOT / "tools" / "ocrmypdf" / "carbon_ocrmypdf.py"


def program():
    if os.environ.get("CARBON_OCR"):
        return str(Path(os.environ["CARBON_OCR"]).resolve())
    for p in sorted(ROOT.glob("build/*/cli/carbon-ocr*")):
        if p.suffix in ("", ".exe") and p.is_file():
            return str(p)
    pytest.skip("no carbon-ocr built")


def plugin():
    spec = importlib.util.spec_from_file_location("carbon_ocrmypdf", PLUGIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hocr_and_text_only_pdf(tmp_path):
    from PIL import Image

    mod = plugin()
    page = tmp_path / "page.png"
    Image.open(DATA / "fake-set" / "page.png").save(page, dpi=(150, 150))
    options = Namespace(carbon_set=str(DATA / "fake-set"), carbon_ocr=program(), carbon_workers=1, languages=["eng"])
    mod.check_options(options)
    engine = mod.get_ocr_engine()
    engine.generate_hocr(page, tmp_path / "p.hocr", tmp_path / "p.txt", options)
    assert "class=\"ocrx_word\"" in (tmp_path / "p.hocr").read_text(encoding="utf-8")
    assert (tmp_path / "p.txt").read_text(encoding="utf-8") == "aaa\naaaaa\naa\n"
    engine.generate_pdf(page, tmp_path / "p.pdf", tmp_path / "q.txt", options)
    pdf = (tmp_path / "p.pdf").read_bytes()
    assert b"/Subtype /Image" not in pdf and b"/MediaBox [0 0 384 491.52]" in pdf
    assert engine.get_orientation(page, options).confidence == 0.0
    assert re.fullmatch(r"Carbon-OCR \d+\.\d+\.\d+", engine.creator_tag(options))
    with pytest.raises(Exception):
        mod.check_options(Namespace(carbon_set=str(DATA / "fake-set"), carbon_ocr=program(), languages=["chi_sim"]))


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="OCRmyPDF checks for tesseract on every run")
@pytest.mark.parametrize("renderer", ["auto", "sandwich"])
def test_ocrmypdf_run(tmp_path, renderer):
    """OCRmyPDF itself on the fake set's page: its rasterizing, our engine's text, and its renderer,
    fpdf2 from our hOCR (auto) or our text-only PDF grafted under the page (sandwich)."""
    import pypdfium2 as pdfium

    exe, out = program(), tmp_path / "out.pdf"
    cmd = [sys.executable, "-X", "faulthandler", "-m", "ocrmypdf", "--plugin", str(PLUGIN), "--carbon-set",
           str(DATA / "fake-set"), "--carbon-ocr", exe, "--pdf-renderer", renderer, "--image-dpi", "150",
           "--output-type", "pdf", "--optimize", "0", "--jobs", "1", str(DATA / "fake-set" / "page.png"), str(out)]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, "CARBON_OCR": exe})
    assert p.returncode == 0, p.stderr
    doc = pdfium.PdfDocument(out)
    assert len(doc) == 1 and doc[0].get_size() == pytest.approx((384, 491.52), abs=0.01)
    assert "".join(doc[0].get_textpage().get_text_range().split()) == "a" * 10, p.stderr
