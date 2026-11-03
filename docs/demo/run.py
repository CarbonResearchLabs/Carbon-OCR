"""Run a Fullerene set on the demo pages and on a rough photo of one, and measure what it read.

    python docs/demo/run.py SET [OUT]

SET is a model set's folder (fullerene-3M from Hugging Face, for instance); OUT defaults to docs/demo/out.
The pages are invented, so their text is known word for word: page.html, a page of an annual report, and
tables.html, a page of two tables; headless Chrome prints each at twice its size (set CHROME to the browser if
it is not at the Windows default). The photo is page.html's page turned upside down and 2.5 degrees off, under
uneven light, noisy, soft and saved as JPEG. Writes to OUT:

    page.png, tables.png, photo.png   the inputs
    page.json, tables.json, photo.json
                          the engine's page for each: lines, blocks, reading order, cells, turn, skew, text, HTML
    detmap.png            the detector's line-core map of page.png, at the page's size
    metrics.json          words read right against the HTML's own text; for the tables, each cell against its
                          table's; the time of each stage (medians of seven runs, on one worker and on six)
"""
import collections
import html
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

import carbon_ocr as c

HERE = Path(__file__).resolve().parent
SET = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
CHROME = os.environ.get("CHROME", r"C:/Program Files/Google/Chrome/Application/chrome.exe")

# The pages: each at 850 x 1100 CSS pixels, printed at twice that, 1700 x 2200.
for name in ("page", "tables"):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                    "--window-size=850,1100", f"--screenshot={OUT / (name + '.png')}", (HERE / f"{name}.html").as_uri()],
                   check=True, capture_output=True)

# The photo: turned 182.5 degrees, on tinted paper under light brighter to the right, noisy, soft, JPEG.
page = Image.open(OUT / "page.png").convert("L")
rng = np.random.default_rng(7)
photo = page.rotate(182.5, resample=Image.BICUBIC, expand=True, fillcolor=255)
a = np.asarray(photo, np.float32)
h, w = a.shape
yy, xx = np.mgrid[0:h, 0:w]
a = 235 * (a / 255.0) * (0.80 + 0.17 * (xx / w) + 0.05 * (yy / h))
a += rng.normal(0, 7, a.shape)
photo = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
photo.save(OUT / "photo.jpg", quality=60)
Image.open(OUT / "photo.jpg").save(OUT / "photo.png")
(OUT / "photo.jpg").unlink()

# The truth: each page's own words, strictly: a superscript is a word of its own, so a footnote mark read
# into the word before it costs two (that word, and the mark). And the tables' cells, row by row.
words = lambda s: re.findall(r"\S+", s)  # noqa: E731
body = {n: (HERE / f"{n}.html").read_text(encoding="utf-8") for n in ("page", "tables")}
body = {n: b[b.index("<body>"):] for n, b in body.items()}
gt = {n: collections.Counter(words(html.unescape(re.sub(r"<[^>]+>", " ", b)))) for n, b in body.items()}
gt["photo"] = gt["page"]
tables = [[[html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[hd]>(.*?)</t[hd]>", r)]
           for r in re.findall(r"<tr[^>]*>(.*?)</tr>", t, re.S)]
          for t in re.findall(r"<table>(.*?)</table>", body["tables"], re.S)]


def cells(p):
    """The tables' cells as the engine wrote them: per table block, {(row, col): text}."""
    out = []
    for b in p["blocks"]:
        if b["label"] == "Table":
            t = {}
            for r, c, ls in b["cells"]:  # two cells at one place are one, as the HTML writes them
                t[(r, c)] = " ".join(([t[(r, c)]] if (r, c) in t else []) + [p["lines"][i]["text"] for i in ls])
            out.append(t)
    return out

metrics = {}
for workers in (1, 6):
    eng = c.Engine(str(SET), workers=workers, pin=False)
    for name in ("page", "photo", "tables"):
        img = c.load_image(str(OUT / f"{name}.png"))
        eng.ocr([img], html=True)  # warm-up
        runs, walls = [], []
        for _ in range(7):
            t = time.perf_counter()
            p = eng.ocr([img], html=True)[0]
            walls.append((time.perf_counter() - t) * 1000)
            runs.append(p["ms"])
        ms = {k: round(statistics.median(r[k] for r in runs), 1) for k in runs[0]}
        ms["wall"] = round(statistics.median(walls), 1)
        if workers == 1:
            (OUT / f"{name}.json").write_text(json.dumps(p, indent=1), encoding="utf-8")
            got = collections.Counter(w for l in p["lines"] for w in words(l["text"]))
            g = gt[name]
            hit = sum((g & got).values())
            metrics[name] = {
                "words_truth": sum(g.values()), "words_read": sum(got.values()), "words_right": hit,
                "recall": round(hit / sum(g.values()), 4), "precision": round(hit / max(1, sum(got.values())), 4),
                "missed": sorted((g - got).elements()), "extra": sorted((got - g).elements()),
                "lines": len(p["lines"]), "blocks": len(p["blocks"]), "order": p["order"],
                "rotation": p["rotation"], "skew": p["skew"],
                "confidence_mean": round(statistics.mean(l["confidence"] for l in p["lines"]), 4), "ms": {}}
        metrics[name]["ms"][str(workers)] = ms
        if workers == 1 and name == "tables":
            # Each truth cell against the engine's cell at the same row and column of the same table: exact
            # (character for character), or right once the column rule's "|" and stray spaces are taken out.
            read = cells(p)
            n = placed = exact = 0
            wrong = []
            for t, rows in enumerate(tables):
                got_t = read[t] if t < len(read) else {}
                for r, row in enumerate(rows):
                    for col, want in enumerate(row):
                        have = got_t.get((r, col), "")
                        n += 1
                        exact += have == want
                        placed += " ".join(have.replace("|", " ").split()) == want
                        if have != want:
                            wrong.append([t + 1, r, col, want, have])
            metrics[name]["cells"] = {"tables_truth": len(tables), "tables_read": len(read), "cells": n,
                                      "exact": exact, "right_but_rule": placed - exact,
                                      "rows_truth": [len(rows) for rows in tables],
                                      "rows_read": [len({r for r, _ in tb}) for tb in read], "wrong": wrong}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False), encoding="utf-8")

# The detector's view of the page: its line-core probability at half its input's scale, sized up to the page.
img = c.load_image(str(OUT / "page.png"))
hh, ww = img.shape
s = 1024 / max(hh, ww)
dw, dh = round(ww * s), round(hh * s)
small = np.asarray(Image.fromarray(img).resize((dw, dh), Image.BILINEAR), np.float32)
x = np.zeros((1, 1, -(-dh // 32) * 32, -(-dw // 32) * 32), np.float32)
x[0, 0, :dh, :dw] = 1 - small / 255.0
maps = c.Session(str(SET / "det.onnx"), threads=4).run([x])[0][0]
core = 1 / (1 + np.exp(-maps[0][: dh // 2, : dw // 2]))
Image.fromarray((core * 255).astype(np.uint8)).resize((ww, hh), Image.NEAREST).save(OUT / "detmap.png")

for name, m in metrics.items():
    print(f"{name}: {m['words_right']} of {m['words_truth']} words right, {m['lines']} lines, {m['blocks']} blocks, "
          f"turned {m['rotation']}, levelled {m['skew']:.2f}; {m['ms']['1']['wall']:.0f} ms on one worker, "
          f"{m['ms']['6']['wall']:.0f} ms on six")
