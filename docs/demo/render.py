"""The demo's images, from what run.py wrote, in the Carbon theme: dark (paper #0b0b0b, ink #efefed) and light
(paper #efefed, ink #1a1a1a), in the palette's colours and IBM Plex (from docs/design/fonts where the repository
has them, else the installed IBM Plex, else the browser's own), drawn by headless Chrome at twice their size:

    demo-read      the page, each line boxed in its block's colour and each block numbered in reading order,
                   beside the HTML the engine wrote
    demo-tables    the page of tables, each cell boxed in its column's colour, beside the HTML tables
    demo-photo     the rough photo, and the page as the engine read it: turned, levelled, every line found
    demo-detector  the detector's line map over the page, and the lines it became

    python docs/demo/render.py [--big DIR] [dark|light ...]

Each image is written as docs/assets/<name>-<theme>.png at its own size (for the documents and the README), and,
with --big, at twice that into DIR. Set CHROME to the browser if it is not at the Windows default.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
RUN = HERE / "out"
ASSETS = HERE.parent / "assets"
FONTS = HERE.parent / "design" / "fonts"
CHROME = os.environ.get("CHROME", r"C:/Program Files/Google/Chrome/Application/chrome.exe")
args = sys.argv[1:]
BIG = None
if "--big" in args:
    i = args.index("--big")
    BIG = Path(args[i + 1])
    BIG.mkdir(parents=True, exist_ok=True)
    del args[i:i + 2]
M = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
PAGE = json.loads((RUN / "page.json").read_text(encoding="utf-8"))
PHOTO = json.loads((RUN / "photo.json").read_text(encoding="utf-8"))
TABLES = json.loads((RUN / "tables.json").read_text(encoding="utf-8"))

THEMES = {
    "dark": dict(ink="#efefed", ink2="#a4a4a4", muted="#8c8c8c", paper="#0b0b0b", panel="#131313", rule="#2a2a2a",
                 head="#f25144", cell="#f5be41", cap="#75fbe1", on="#0b0b0b", heat=(117, 251, 225), veil=(11, 11, 11),
                 line="#75fbe1", columns=["#5ac5fa", "#75fbe1", "#75fb6d", "#f5be41", "#f25144", "#3987e5"],
                 label={"Text": "#5ac5fa", "Section-Header": "#f25144", "List-Group": "#75fb6d", "Table": "#f5be41",
                        "Caption": "#75fbe1", "Page-Header": "#a4a4a4", "Page-Footer": "#a4a4a4", "Footnote": "#c8c8c6",
                        "Formula": "#d95926", "Form": "#3987e5", "Picture": "#c8c8c6", "Other": "#8c8c8c",
                        "Table-Of-Contents": "#75fb6d"}),
    "light": dict(ink="#1a1a1a", ink2="#3d3d3d", muted="#555555", paper="#efefed", panel="#ffffff", rule="#d8d5ce",
                  head="#d6352a", cell="#1a1a1a", cap="#3d3d3d", on="#ffffff", heat=(42, 120, 214), veil=(239, 239, 237),
                  line="#2a78d6", columns=["#2a78d6", "#1f9e86", "#3fa63a", "#d68a00", "#d6352a", "#6b6b6b"],
                  label={"Text": "#2a78d6", "Section-Header": "#d6352a", "List-Group": "#3fa63a", "Table": "#d68a00",
                         "Caption": "#1f9e86", "Page-Header": "#6b6b6b", "Page-Footer": "#6b6b6b", "Footnote": "#555555",
                         "Formula": "#eb6834", "Form": "#2a78d6", "Picture": "#555555", "Other": "#6b6b6b",
                         "Table-Of-Contents": "#3fa63a"}),
}
NAMES = {"Text": "text", "Section-Header": "heading", "List-Group": "list", "Table": "table", "Caption": "caption",
         "Page-Header": "header", "Page-Footer": "footer", "Footnote": "footnote", "Formula": "formula", "Form": "form",
         "Picture": "picture", "Other": "other", "Table-Of-Contents": "contents"}
FACE = "".join(
    f"@font-face{{font-family:'{fam}';font-weight:{wt};src:url('{(FONTS / f).as_uri()}') format('woff2');}}"
    for fam, wt, f in (("Plex", 300, "IBMPlexSans-Light.woff2"), ("Plex", 400, "IBMPlexSans-Regular.woff2"),
                       ("Plex", 500, "IBMPlexSans-Medium.woff2"), ("Plex", 600, "IBMPlexSans-SemiBold.woff2"),
                       ("PlexMono", 400, "IBMPlexMono-Regular.woff2"), ("PlexMono", 500, "IBMPlexMono-Medium.woff2")))
FOOT = "github.com/karanvora2599/Carbon-OCR · huggingface.co/CarbonResearchLabs/fullerene-3M · getcarbonresearch.com"


def css(t):
    return re.sub(r"\b(PlexMono|Plex)\b(?=[;}])", lambda m: STACKS[m.group(1)], _css(t))


STACKS = {"Plex": "Plex,'IBM Plex Sans',Arial,sans-serif", "PlexMono": "PlexMono,'IBM Plex Mono',Consolas,monospace"}


def _css(t):
    return f"""{FACE}
html,body{{margin:0;background:{t["paper"]};color:{t["ink"]};font-family:Plex}}
.wrap{{padding:44px 56px}}
.kicker{{font:500 15px PlexMono;letter-spacing:2px;color:{t["ink2"]};text-transform:uppercase}}
h1{{font:400 44px Plex;letter-spacing:-.5px;margin:10px 0 6px}}
.stats{{font:400 18px Plex;color:{t["ink2"]};margin-bottom:26px}}
.stats b{{color:{t["ink"]};font-weight:500}}
.row{{display:flex;gap:36px;align-items:flex-start}}
.panel{{position:relative;border:1px solid {t["rule"]};background:#fff}}
.panel img{{display:block}}
.label{{font:500 14px PlexMono;letter-spacing:1.5px;color:{t["ink2"]};text-transform:uppercase;margin:0 0 10px}}
.arrow{{font:300 64px Plex;color:{t["ink2"]};align-self:center}}
.out{{background:{t["panel"]};border:1px solid {t["rule"]};padding:26px 30px;font:400 16.5px/1.55 Plex;color:{t["ink"]}}}
.out h2{{font:600 21px Plex;color:{t["head"]};margin:14px 0 6px}}
.out p{{margin:0 0 10px}}
.out ul{{margin:0 0 8px 20px;padding:0}}
.out header,.out footer{{font:400 13px PlexMono;color:{t["muted"]};margin:2px 0}}
.out header p,.out footer p{{margin:0}}
.out table{{border-collapse:collapse;margin:6px 0 12px;font-size:15px}}
.out td{{border:1px solid {t["rule"]};padding:3px 9px;color:{t["cell"]}}}
.out figcaption{{font-style:italic;color:{t["cap"]}}}
.out figure{{margin:0 0 10px}}
.legend{{display:flex;flex-wrap:wrap;gap:8px 20px;margin-top:14px;font:400 15px Plex;color:{t["ink2"]}}}
.legend i{{display:inline-block;width:13px;height:13px;border-radius:3px;margin-right:7px;vertical-align:-1px}}
.legend span i{{margin-right:3px}}
.foot{{font:400 15px Plex;color:{t["muted"]};margin-top:22px}}
"""


def overlay(t, page, w, h, scale, numbers=True, color=None, columns=False):
    """An SVG over a page image: each line's box, in its block's colour (or one colour; with columns, a table
    cell's in its column's), and each block's number in reading order at its first line."""
    lines = page["lines"]
    col, first = {}, {}
    for k, b in enumerate(page["blocks"]):
        for i in b["lines"]:
            col[i] = t["label"].get(b["label"], t["muted"])
        if columns and b["label"] == "Table":
            for _, c, ls in b["cells"]:
                for i in ls:
                    col[i] = t["columns"][c % len(t["columns"])]
        if b["lines"]:
            first[min(b["lines"], key=lambda i: (lines[i]["box"][1], lines[i]["box"][0]))] = k + 1
    parts = []
    for i, l in enumerate(lines):
        x0, y0, x1, y1 = (v * scale for v in l["box"])
        c = color or col.get(i, t["label"]["Text"])
        parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" rx="2" '
                     f'fill="{c}" fill-opacity="0.16" stroke="{c}" stroke-width="1.6"/>')
        if numbers and i in first:
            parts.append(f'<circle cx="{x0 - 2:.1f}" cy="{y0 - 2:.1f}" r="11" fill="{c}"/>'
                         f'<text x="{x0 - 2:.1f}" y="{y0 + 2.5:.1f}" text-anchor="middle" font-family="PlexMono, IBM Plex Mono, monospace" '
                         f'font-weight="500" font-size="12" fill="{t["on"]}">{first[i]}</text>')
    return (f'<svg style="position:absolute;left:0;top:0" width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            + "".join(parts) + "</svg>")


def legend(t, page):
    seen = []
    for b in page["blocks"]:
        c, name = t["label"].get(b["label"], t["muted"]), NAMES.get(b["label"], b["label"])
        if name not in [n for n, _ in seen]:
            seen.append((name, c))
    return '<div class="legend">' + "".join(f"<span><i style='background:{c}'></i>{n}</span>" for n, c in seen) + "</div>"


def render(t, name, body, w, h):
    """Draw at twice the size, trim the empty foot (keeping the 44 px margin), and save."""
    with tempfile.TemporaryDirectory() as tmp:
        page, png = Path(tmp) / "page.html", Path(tmp) / "shot.png"
        page.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{css(t)}</style></head>"
                        f"<body>{body}</body></html>", encoding="utf-8")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        f"--window-size={w},{h}", f"--screenshot={png}", page.as_uri()], check=True, capture_output=True)
        im = Image.open(png).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, im.getpixel((2, im.height - 2)))).getbbox()
    if box:
        im = im.crop((0, 0, im.width, min(im.height, box[3] + 88)))
    if BIG:
        im.save(BIG / f"{name}.png", optimize=True)
    im.resize((im.width // 2, im.height // 2), Image.LANCZOS).save(ASSETS / f"{name}.png", optimize=True)


def build(theme, t):
    m, mp = M["page"], M["photo"]
    one, six = m["ms"]["1"], m["ms"]["6"]
    kicker = "<div class='wrap'><div class='kicker'>Carbon Research Labs · Fullerene 3M</div>"

    # The page read: its lines in their blocks' colours, numbered in reading order, beside the HTML.
    pw = 1100
    s = pw / PAGE["width"]
    ph = round(PAGE["height"] * s)
    read = (
        f"{kicker}<h1>One page, read: lines, blocks, order, HTML</h1>"
        f"<div class='stats'><b>{m['words_right']} of {m['words_truth']}</b> words read right · <b>{m['lines']}</b> "
        f"lines in <b>{m['blocks']}</b> blocks, in reading order · <b>{one['wall']:.0f} ms</b> on one CPU core, "
        f"<b>{six['wall']:.0f} ms</b> on six</div>"
        "<div class='row'><div><div class='label'>The page, as Fullerene sees it</div>"
        f"<div class='panel' style='width:{pw}px;height:{ph}px'><img src='{(RUN / 'page.png').as_uri()}' "
        f"style='width:{pw}px;height:{ph}px'>{overlay(t, PAGE, pw, ph, s)}</div>{legend(t, PAGE)}</div>"
        "<div style='flex:1'><div class='label'>What it wrote: carbon-ocr --format html</div>"
        f"<div class='out'>{PAGE['html']}</div></div></div><div class='foot'>{FOOT}</div></div>")
    render(t, f"demo-read-{theme}", read, 2400, 2000)

    # The page of tables: each cell in its column's colour, beside the HTML tables.
    mt, cl = M["tables"], M["tables"]["cells"]
    tw = 1000
    ts = tw / TABLES["width"]
    th = round(TABLES["height"] * ts)
    rows = sum(cl["rows_truth"])
    swatch = "".join(f"<i style='background:{c}'></i>" for c in t["columns"])
    tv = (
        f"{kicker}<h1>Two tables, read cell by cell</h1>"
        f"<div class='stats'><b>{cl['tables_read']}</b> tables found, <b>{rows}</b> rows, <b>{cl['cells']}</b> cells: "
        f"<b>every cell in its row and column</b>, {cl['exact']} of them exact · <b>{mt['lines']}</b> lines in "
        f"<b>{mt['ms']['1']['wall']:.0f} ms</b> on one CPU core</div>"
        "<div class='row'><div><div class='label'>The page, each cell in its column's colour</div>"
        f"<div class='panel' style='width:{tw}px;height:{th}px'><img src='{(RUN / 'tables.png').as_uri()}' "
        f"style='width:{tw}px;height:{th}px'>{overlay(t, TABLES, tw, th, ts, numbers=False, columns=True)}</div>"
        f"<div class='legend'><span>{swatch}&nbsp;columns</span></div></div>"
        "<div style='flex:1'><div class='label'>What it wrote: carbon-ocr --format html</div>"
        f"<div class='out'>{TABLES['html']}</div></div></div><div class='foot'>{FOOT}</div></div>")
    render(t, f"demo-tables-{theme}", tv, 2400, 2000)

    # The photo, and the page as the engine levelled it: turned upright, then by its skew.
    photo = Image.open(RUN / "photo.png")
    photo.rotate(PHOTO["rotation"]).rotate(PHOTO["skew"], resample=Image.BILINEAR, fillcolor=235).save(
        RUN / "photo-levelled.png")
    w2 = 760
    s2 = w2 / photo.width
    h2 = round(photo.height * s2)
    pv = (
        f"{kicker}<h1>Upside down and tilted: read the right way up</h1>"
        f"<div class='stats'>A rough photo: turned <b>180°</b> and <b>2.5°</b> off, uneven light, noise, JPEG. "
        f"Fullerene turned it <b>{PHOTO['rotation']}°</b>, levelled it <b>{abs(PHOTO['skew']):.2f}°</b>, and read "
        f"<b>{mp['words_right']} of {mp['words_truth']}</b> words right in <b>{mp['ms']['1']['wall']:.0f} ms</b> on one "
        "core</div><div class='row'>"
        f"<div><div class='label'>In: the photo</div><div class='panel' style='width:{w2}px;height:{h2}px'>"
        f"<img src='{(RUN / 'photo.png').as_uri()}' style='width:{w2}px;height:{h2}px'></div></div>"
        "<div class='arrow'>→</div>"
        "<div><div class='label'>Out: turned, levelled, every line found</div>"
        f"<div class='panel' style='width:{w2}px;height:{h2}px'><img src='{(RUN / 'photo-levelled.png').as_uri()}' "
        f"style='width:{w2}px;height:{h2}px'>{overlay(t, PHOTO, w2, h2, s2)}</div>{legend(t, PHOTO)}</div></div>"
        "<div class='foot'>Orientation: 12 angle bins from a thumbnail, the page turned by quarter turns only when "
        "the model is sure. Deskew: a projection search, no model.</div></div>")
    render(t, f"demo-photo-{theme}", pv, 2000, 1420)

    # The detector's map over the page (veiled), and the line boxes the engine's decoder made of it.
    dm = Image.open(RUN / "detmap.png").convert("L")
    heat = Image.new("RGBA", dm.size, (*t["heat"], 0))
    heat.putalpha(dm.point(lambda v: int(v * 0.85)))
    base = Image.open(RUN / "page.png").convert("RGBA")
    veiled = Image.blend(base, Image.new("RGBA", base.size, (*t["veil"], 255)), 0.55)
    Image.alpha_composite(veiled, heat).convert("RGB").save(RUN / f"detmap-over-{theme}.png")
    w3 = 760
    s3 = w3 / PAGE["width"]
    h3 = round(PAGE["height"] * s3)
    dv = (
        f"{kicker}<h1>What the detector sees</h1>"
        f"<div class='stats'><b>det</b>, 0.44 M parameters, maps the page's line cores at half its input scale; the "
        f"engine's decoder turns the map into <b>{m['lines']}</b> line boxes, with no model in that step</div>"
        "<div class='row'>"
        f"<div><div class='label'>The line map</div><div class='panel' style='width:{w3}px;height:{h3}px'>"
        f"<img src='{(RUN / f'detmap-over-{theme}.png').as_uri()}' style='width:{w3}px;height:{h3}px'></div></div>"
        "<div class='arrow'>→</div>"
        f"<div><div class='label'>The lines</div><div class='panel' style='width:{w3}px;height:{h3}px'>"
        f"<img src='{(RUN / 'page.png').as_uri()}' style='width:{w3}px;height:{h3}px'>"
        f"{overlay(t, PAGE, w3, h3, s3, numbers=False, color=t['line'])}</div></div></div></div>")
    render(t, f"demo-detector-{theme}", dv, 2000, 1260)


for theme in args or ("dark", "light"):
    build(theme, THEMES[theme])
print(sorted(p.name for p in ASSETS.glob("demo-*.png")))
