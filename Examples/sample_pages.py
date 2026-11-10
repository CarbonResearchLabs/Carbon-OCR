"""Sample pages for the notebooks, drawn with Pillow in the DejaVu fonts that matplotlib ships.

Every page is drawn from known text, so each one comes with its truth: the text of every line
and the line's box in the page's pixels. The notebooks read these pages, measure the engine
against them and change the engine's options on them. The engine itself needs none of this.

    import sample_pages
    pages = sample_pages.make_all("data")       # {name: Path}, each with a .json beside it
    truth = sample_pages.truth(pages["letter"])  # {"width", "height", "lines": [{"text", "box"}]}
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def font(name: str = "DejaVuSerif", size: int = 30) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONTS / f"{name}.ttf"), size)


class Page:
    """A grayscale page and the lines drawn on it, each with its text and box."""

    def __init__(self, width: int, height: int, paper: int = 255):
        self.image = Image.new("L", (width, height), paper)
        self.draw = ImageDraw.Draw(self.image)
        self.lines: list[dict] = []
        self.pictures: list[list[int]] = []  # boxes of what is drawn as a picture
        self.tables: list[list[int]] = []

    def text(self, x, y, s, f, fill=0, record=True):
        """One line at (x, y), its top left; returns the box's bottom."""
        self.draw.text((x, y), s, font=f, fill=fill)
        box = list(self.draw.textbbox((x, y), s, font=f))
        if record:
            self.lines.append({"text": s, "box": box})
        return box[3]

    def right(self, x1, y, s, f, fill=0):
        """One line ending at x1."""
        return self.text(x1 - self.draw.textlength(s, font=f), y, s, f, fill)

    def wrap(self, s, f, width):
        lines, cur = [], []
        for w in s.split():
            if cur and self.draw.textlength(" ".join(cur + [w]), font=f) > width:
                lines.append(cur)
                cur = [w]
            else:
                cur.append(w)
        if cur:
            lines.append(cur)
        return lines

    def paragraph(self, x, y, width, s, f, leading=1.45, fill=0, justify=False, indent=0):
        """Words wrapped to `width`; justified lines spread their words to both edges. Returns the next y."""
        step = round(f.size * leading)
        lines = self.wrap(s, f, width - indent)
        for i, words in enumerate(lines):
            x0 = x + (indent if i == 0 else 0)
            w = width - (indent if i == 0 else 0)
            last = i == len(lines) - 1
            if justify and not last and len(words) > 1:
                ink = sum(self.draw.textlength(t, font=f) for t in words)
                gap = (w - ink) / (len(words) - 1)
                cx, top, bottom = x0, None, None
                for t in words:
                    self.draw.text((cx, y), t, font=f, fill=fill)
                    b = self.draw.textbbox((cx, y), t, font=f)
                    top = b[1] if top is None else min(top, b[1])
                    bottom = b[3] if bottom is None else max(bottom, b[3])
                    cx += self.draw.textlength(t, font=f) + gap
                x_end = self.draw.textbbox((cx - gap - self.draw.textlength(words[-1], font=f), y), words[-1],
                                           font=f)[2]
                self.lines.append({"text": " ".join(words), "box": [round(x0), top, round(x_end), bottom]})
            else:
                self.text(x0, y, " ".join(words), f, fill)
            y += step
        return y

    def rule(self, x0, y, x1, width=3, fill=0):
        self.draw.rectangle([x0, y, x1, y + width - 1], fill=fill)

    def save(self, path: Path, **kw):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path, **kw)
        meta = {"width": self.image.width, "height": self.image.height, "lines": self.lines,
                "pictures": self.pictures, "tables": self.tables}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
        return path


LETTER = (1700, 2200)  # US Letter at 200 dpi, the CLI's default for PDF pages

HARBOUR = [
    "The survey of the harbour began on the first calm morning of the month. Two boats took soundings "
    "along the channel every fifty metres, and a tide gauge on the north pier logged the water level "
    "every ten minutes. By the end of the week the crews had recorded 1,284 depths and 1,008 tide readings.",
    "The channel is shallower than the 2019 chart shows. Near the harbour mouth the least depth found "
    "was 4.6 m at low water, against 5.3 m on the chart; the silt has moved east, toward the old ferry "
    "slip. The inner basin has changed little (its depths are within 0.2 m of the chart's).",
    "Small boats can use the channel at any state of the tide. Vessels drawing more than 4 m should "
    "enter within two hours of high water until the channel is dredged.",
]
FINDINGS = [
    "The east bank of the channel has moved 12 m toward the fairway since 2019.",
    "Tide heights at the pier agree with the predictions to within 9 cm.",
    "The ferry slip needs a new fender line before the summer timetable.",
    "Dredging 18,000 cubic metres would restore the charted depth of 5.3 m.",
]
CLOSING = ("The full soundings, the tide record and the survey's charts are kept at the harbour office "
           "and may be read there on weekdays from 9:00 to 17:00. Questions about the survey may be sent "
           "to the office by post or e-mail (survey@harbour.example).")


def letter() -> Page:
    """A one-column report: a title, a date line, paragraphs, a heading and a bulleted list."""
    p = Page(*LETTER)
    x, w = 160, 1380
    y = 170
    y = p.text(x, y, "Notes on the Harbour Survey", font("DejaVuSerif-Bold", 52)) + 22
    y = p.text(x, y, "Prepared for the town council, 14 March 2026", font("DejaVuSerif-Italic", 30)) + 60
    body = font("DejaVuSerif", 31)
    for s in HARBOUR:
        y = p.paragraph(x, y, w, s, body) + 24
    y = p.text(x, y + 14, "Findings", font("DejaVuSans-Bold", 36)) + 34
    for s in FINDINGS:
        lines = p.wrap(s, body, w - 50)
        p.draw.text((x + 8, y), "•", font=body, fill=0)
        for i, words in enumerate(lines):
            t = " ".join(words)
            if i == 0:  # the bullet is part of the line's text, as it is printed
                p.draw.text((x + 50, y), t, font=body, fill=0)
                b0 = p.draw.textbbox((x + 8, y), "•", font=body)
                b1 = p.draw.textbbox((x + 50, y), t, font=body)
                p.lines.append({"text": "• " + t, "box": [b0[0], min(b0[1], b1[1]), b1[2], max(b0[3], b1[3])]})
            else:
                p.text(x + 50, y, t, body)
            y += round(body.size * 1.45)
        y += 10
    y = p.paragraph(x, y + 20, w, CLOSING, body) + 60
    p.text(x, y, "The Survey Office", font("DejaVuSerif-Italic", 31))
    return p


def ripples(w, h, seed=7):
    """A photograph-like picture: smooth noise with bright and dark bands."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    v = 128 + 50 * np.sin(xx / 23.0 + 3 * np.sin(yy / 41.0)) + 30 * np.cos(yy / 17.0 + xx / 61.0)
    v += rng.normal(0, 18, (h, w))
    img = Image.fromarray(np.clip(v, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))
    d = ImageDraw.Draw(img)
    for i in range(7):  # a pier's posts
        x = 40 + i * (w - 80) // 6
        d.rectangle([x, int(h * 0.45), x + 14, h - 10], fill=25)
    d.rectangle([20, int(h * 0.45) - 12, w - 20, int(h * 0.45)], fill=20)
    return img


TIDES = [
    "Every harbour keeps two clocks: the town's and the tide's. The tide's clock runs about fifty minutes "
    "slow each day, so high water comes later each morning, and a fisherman who learns the tide by the "
    "town clock will be wrong within a week.",
    "The gauge on the north pier is a float in a stilling well, a pipe open at the bottom that calms the "
    "waves so the float rides the true level of the sea. Its record shows two tides a day, the second "
    "a little lower than the first.",
]
METHOD = ("We compared the gauge's record for one week with the heights predicted for the pier, reading "
          "both every ten minutes. Where the record and the prediction differ, the difference is the "
          "weather's: wind and pressure push the water up or hold it down.")
FOUND = ("The record kept within 9 cm of the predictions on calm days. On the windy Thursday the evening "
         "tide came 21 cm higher than predicted, and 14 minutes early. A falling barometer adds about "
         "one centimetre of water for each hectopascal it falls, which accounts for most of that day's "
         "difference.")


def columns() -> Page:
    """Two justified columns, a title over both, a picture with its caption, a running header, rules,
    a footnote and a page number."""
    p = Page(*LETTER)
    m, gutter = 140, 60
    cw = (LETTER[0] - 2 * m - gutter) // 2
    small = font("DejaVuSans", 24)
    p.text(m, 90, "FIELD JOURNAL · AUTUMN 2026", small)
    p.right(LETTER[0] - m, 90, "Coastal Notes", small)
    p.rule(m, 132, LETTER[0] - m, 3)
    y = p.text(m, 190, "Reading the Tides", font("DejaVuSerif-Bold", 64)) + 24
    y = p.text(m, y, "What a week of gauge readings at the north pier shows", font("DejaVuSerif-Italic", 32)) + 60
    top = y
    body = font("DejaVuSerif", 29)
    x1, x2 = m, m + cw + gutter
    y = p.paragraph(x1, top, cw, TIDES[0], body, justify=True) + 30
    pic = ripples(cw, 520)
    p.image.paste(pic, (x1, y))
    picture = [x1, y, x1 + cw, y + 520]
    y += 520 + 18
    y = p.paragraph(x1, y, cw, "Figure 1. The north pier at low water; the gauge's well is "
                    "the third post from the left.", font("DejaVuSerif-Italic", 24), leading=1.35) + 30
    y = p.paragraph(x1, y, cw, TIDES[1], body, justify=True)
    y = p.text(x2, top, "Method", font("DejaVuSans-Bold", 34)) + 26
    y = p.paragraph(x2, y, cw, METHOD, body, justify=True) + 36
    y = p.text(x2, y, "What we found", font("DejaVuSans-Bold", 34)) + 26
    y = p.paragraph(x2, y, cw, FOUND, body, justify=True) + 40
    p.rule(x2, 1900, x2 + cw // 3, 2)
    p.paragraph(x2, 1920, cw, "* Heights are given against chart datum, the level of the lowest "
                "astronomical tide.", font("DejaVuSerif", 22), leading=1.35)
    p.rule(m, 2040, LETTER[0] - m, 2)
    p.text(LETTER[0] // 2 - 10, 2070, "7", font("DejaVuSans", 26))
    p.pictures.append(picture)
    return p


ROUTES = [("Harbour to Northgate", 41.2, 44.8, 58.3, 39.6), ("Harbour to Saltmarsh", 18.5, 20.1, 27.4, 17.9),
          ("Northgate to Saltmarsh", 9.7, 10.4, 15.2, 8.8), ("Island shuttle", 6.1, 7.9, 12.6, 5.4),
          ("Evening crossing", 3.3, 4.0, 6.8, 2.9)]


def table() -> Page:
    """A ruled table whose header spans four columns, between a heading, a paragraph and a note."""
    p = Page(*LETTER)
    m = 140
    y = p.text(m, 170, "Ferry passengers by quarter, 2025", font("DejaVuSans-Bold", 46)) + 40
    y = p.paragraph(m, y, LETTER[0] - 2 * m, "Passenger numbers rose in every quarter on every route. The "
                    "third quarter, with the summer timetable, carried a third of the year's passengers.",
                    font("DejaVuSerif", 30)) + 50
    cols = [m, m + 560, m + 720, m + 880, m + 1040, m + 1200, LETTER[0] - m]
    head, cell = font("DejaVuSans-Bold", 28), font("DejaVuSans", 28)
    row_h = 66
    top = y
    p.rule(m, y, cols[-1], 3)
    p.text(m + 16, y + 52, "Route", head)
    span = "Passengers (thousands)"
    sw = p.draw.textlength(span, font=head)
    p.text((cols[1] + cols[5]) / 2 - sw / 2, y + 14, span, head)
    p.rule(cols[1] + 10, y + row_h - 8, cols[5] - 10, 2)
    y += row_h
    for i, q in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        p.right(cols[i + 2] - 24, y + 12, q, head)
    p.right(cols[6] - 24, y - 20, "Year", head)
    y += row_h
    p.rule(m, y - 6, cols[-1], 2)
    totals = [0.0] * 5
    for name, *qs in ROUTES:
        p.text(m + 16, y + 12, name, cell)
        year = round(sum(qs), 1)
        for i, v in enumerate(qs + [year]):
            p.right(cols[i + 2] - 24, y + 12, f"{v:.1f}", cell)
            totals[i] += v
        y += row_h
    p.rule(m, y - 6, cols[-1], 2)
    p.text(m + 16, y + 12, "All routes", head)
    for i, v in enumerate(totals):
        p.right(cols[i + 2] - 24, y + 12, f"{v:.1f}", head)
    y += row_h
    p.rule(m, y - 4, cols[-1], 3)
    p.tables.append([m, top, cols[-1], y])
    y += 30
    p.paragraph(m, y, LETTER[0] - 2 * m, "Source: the harbour office's ticket counts. Figures are rounded to "
                "one decimal, so a year may differ from the sum of its quarters.", font("DejaVuSerif-Italic", 26))
    return p


TERMS = [
    "1. These terms apply to every ticket sold by the harbour office for the ferry routes it runs, whether "
    "the ticket is bought at the office, on board or through an agent.",
    "2. A ticket is valid only for the route, the date and the class of travel printed on it. A return "
    "ticket is valid for one month from the date of the outward journey.",
    "3. The office may change a timetable, a vessel or a route without notice where the weather, the tide "
    "or the state of a vessel makes it necessary for safety.",
    "4. A passenger whose crossing is cancelled may travel on the next crossing or have the fare refunded "
    "in full. No other costs are refunded.",
    "5. Children under five travel free with an adult. Children from five to fifteen travel at half the "
    "adult fare. A bicycle travels at the child's fare.",
    "6. Dogs may travel on the open deck only, on a lead, and at the owner's risk.",
    "7. Luggage is carried at the owner's risk. Each passenger may bring two pieces of luggage; more may be "
    "carried where there is room, at the crew's discretion.",
    "8. Passengers must follow the crew's instructions at all times, and must not enter the wheelhouse or "
    "the engine room.",
    "9. Smoking is not allowed on board, on the open deck or below it.",
    "10. A lost ticket is not replaced. A ticket found on board is kept at the harbour office for one month.",
    "11. These terms may be read at the harbour office, on board every vessel and on the office's notice "
    "board at each landing stage.",
]


def mixed() -> Page:
    """A form scanned at 300 dpi: ordinary print, then its conditions in fine print (4 pt), as forms print them."""
    W, H = 2550, 3300  # US Letter at 300 dpi
    p = Page(W, H)
    x, w = 240, 2070
    y = p.text(x, 250, "Season ticket application", font("DejaVuSerif-Bold", 72)) + 80
    body = font("DejaVuSerif", 46)
    for s in HARBOUR:
        y = p.paragraph(x, y, w, s, body) + 36
    y = p.text(x, y + 30, "Conditions", font("DejaVuSans-Bold", 46)) + 50
    f = font("DejaVuSans", 17)
    cw, top = (w - 80) // 2, y  # two columns of it
    for i, s in enumerate(TERMS):
        if i == 6:
            x, y = x + cw + 80, top
        y = p.paragraph(x, y, cw, s, f, leading=1.2) + 3
    return p


GAZETTE = ["Survey finds a shallower channel", "Tides run late", "Ferries carry more", "Council meets",
           "Weather", "Notices"]


def newspaper() -> Page:
    """A broadsheet page: a masthead over six justified columns of 24 px type on 3400 x 4800 px. At the
    detector's scale (its long side 1024 px) its lines are under 9 px high, so the engine detects it twice."""
    W, H = 3400, 4800
    p = Page(W, H)
    m, gut = 120, 50
    cw = (W - 2 * m - 5 * gut) // 6
    p.text(m, 80, "THE HARBOUR GAZETTE", font("DejaVuSerif-Bold", 150))
    p.rule(m, 300, W - m, 6)
    f = font("DejaVuSerif", 24)
    stories = [HARBOUR, TIDES, [METHOD, FOUND], [CLOSING] + HARBOUR[:1], TIDES[::-1], [FOUND, CLOSING]]
    for c in range(6):
        x = m + c * (cw + gut)
        y = 360
        for t in p.wrap(GAZETTE[c], font("DejaVuSans-Bold", 40), cw):
            y = p.text(x, y, " ".join(t), font("DejaVuSans-Bold", 40)) + 14
        y += 30
        k = 0
        while y < H - 250:
            y = p.paragraph(x, y, cw, stories[c][k % len(stories[c])], f, leading=1.25, justify=True) + 12
            k += 1
    return p


def banner() -> Page:
    """Light text on dark grounds: a banner, a box and a table's header row, above dark-on-light text."""
    p = Page(*LETTER)
    m = 140
    p.draw.rectangle([0, 120, LETTER[0], 330], fill=20)
    p.text(m, 160, "Harbour Open Day", font("DejaVuSans-Bold", 88), fill=245)
    p.draw.rectangle([m, 400, LETTER[0] - m, 490], fill=70)
    p.text(m + 30, 420, "Saturday 9 May · 10:00 to 16:00 · Free entry", font("DejaVuSans", 40), fill=255)
    y = 560
    body = font("DejaVuSerif", 31)
    y = p.paragraph(m, y, LETTER[0] - 2 * m, "The harbour office opens its doors for one day a year. Visitors "
                    "can climb the pier's tide gauge, take a short trip on the survey boat and see the "
                    "charts the office keeps.", body) + 50
    head, cell = font("DejaVuSans-Bold", 30), font("DejaVuSans", 30)
    p.draw.rectangle([m, y, LETTER[0] - m, y + 70], fill=30)
    p.text(m + 20, y + 16, "Time", head, fill=250)
    p.text(m + 300, y + 16, "Event", head, fill=250)
    y += 70
    for t, e in [("10:00", "The harbour master opens the day"), ("11:30", "Survey boat trips leave the slip"),
                 ("13:00", "Talk: how the channel is charted"), ("15:00", "Last boat trip of the day")]:
        p.text(m + 20, y + 16, t, cell)
        p.text(m + 300, y + 16, e, cell)
        p.rule(m, y + 68, LETTER[0] - m, 1, fill=150)
        y += 70
    return p


def receipt() -> Page:
    """A till receipt: a long narrow strip in a monospaced face."""
    w, h = 560, 3000
    p = Page(w, h)
    f, b = font("DejaVuSansMono", 26), font("DejaVuSansMono-Bold", 30)
    y = 60
    y = p.text(170, y, "HARBOUR CAFE", b) + 20
    y = p.text(130, y, "North Pier", f) + 14
    p.text(40, y, "14/03/2026", f)
    y = p.right(520, y, "10:42", f) + 50
    items = [("Tea", 2.20), ("Coffee, large", 3.40), ("Scone with jam", 2.95), ("Crab sandwich", 6.50),
             ("Soup of the day", 4.80), ("Fish and chips", 9.75), ("Apple cake", 3.10), ("Lemonade", 2.40),
             ("Ice cream", 2.60), ("Toast", 1.80), ("Hot chocolate", 3.20), ("Flapjack", 2.10)]
    for name, price in items * 2:  # an item and its price: two lines, in columns of their own
        p.text(40, y, name, f)
        p.right(520, y, f"{price:.2f}", f)
        y += 44
    total = sum(v for _, v in items) * 2
    y += 20
    p.text(40, y, "TOTAL", b)
    p.right(520, y, f"{total:.2f}", b)
    y += 70
    p.text(70, y, "Thank you, come again", f)
    return p


def slide() -> Page:
    """A presentation slide: a few lines of very large type on a wide page."""
    p = Page(1920, 1080)
    p.text(120, 90, "Channel depths, 2026", font("DejaVuSans-Bold", 88))
    f = font("DejaVuSans", 40)
    y = 300
    for s in ["Least depth at the mouth: 4.6 m", "Charted depth: 5.3 m", "Silt moved east, toward the slip",
              "Small boats: any state of the tide", "Deep draught: two hours of high water",
              "Dredging restores the charted depth"]:
        p.text(140, y, "• " + s, f)
        y += 110
    return p


def rotate(page: Page, quarter_turns: int = 1, skew: float = 0.0) -> Image.Image:
    """The page turned by quarter turns (clockwise) and then by `skew` degrees, on paper."""
    img = page.image
    if quarter_turns % 4:
        img = img.transpose([None, Image.Transpose.ROTATE_270, Image.Transpose.ROTATE_180,
                             Image.Transpose.ROTATE_90][quarter_turns % 4])
    if skew:
        img = img.rotate(-skew, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=255)
    return img


def scanned(img: Image.Image, blur=1.1, noise=10.0, paper=232, ink=45, quality=35, skew=0.6, seed=3) -> bytes:
    """The page as a poor scan: tinted paper, faded ink, blur, sensor noise, a slight skew, a hard JPEG."""
    a = np.asarray(img, np.float32)
    a = ink + (paper - ink) * (a / 255.0)
    rng = np.random.default_rng(seed)
    a += rng.normal(0, noise, a.shape)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))
    if skew:
        out = out.rotate(-skew, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=paper)
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def truth(path) -> dict:
    """The page's truth: its size and each line's text and box, in reading order."""
    return json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))


def truth_text(path) -> str:
    """The page's text as the engine writes it: each line followed by a newline."""
    return "".join(line["text"] + "\n" for line in truth(path)["lines"])


def make_all(folder="data") -> dict:
    """Draws every sample page into `folder` (PNG, and the scan as JPEG), each with its truth beside it."""
    folder = Path(folder)
    pages = {}
    for name, make in [("letter", letter), ("columns", columns), ("table", table), ("mixed", mixed),
                       ("newspaper", newspaper), ("banner", banner), ("receipt", receipt), ("slide", slide)]:
        pages[name] = make().save(folder / f"{name}.png")
    lt = letter()
    for name, turns, skew in [("rotated", 1, 0.0), ("upside-down", 2, 0.0), ("skewed", 0, 2.5)]:
        page = Page(1, 1)
        page.image, page.lines = rotate(lt, turns, skew), lt.lines
        pages[name] = page.save(folder / f"{name}.png")
    scan = folder / "scan.jpg"
    scan.write_bytes(scanned(lt.image))
    lt_meta = {"width": lt.image.width, "height": lt.image.height, "lines": lt.lines, "pictures": [], "tables": []}
    scan.with_suffix(".json").write_text(json.dumps(lt_meta, indent=1, ensure_ascii=False), encoding="utf-8")
    pages["scan"] = scan
    return pages


if __name__ == "__main__":
    for name, path in make_all().items():
        print(f"{name:12} {path}")
