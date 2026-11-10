"""What the notebooks share: where the model set and `carbon-ocr` are, pictures of pages and their
boxes, and lines matched to the truth they were drawn from."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parent
ROOT = EXAMPLES.parent  # the repository's root
SET_REPO = "CarbonResearchLabs/fullerene-3M"


def shown(path) -> str:
    """A path as the notebooks print it: from the repository's root when it is inside it."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.name


def model_set(download: bool = True) -> Path:
    """The Fullerene set's folder: $CARBON_OCR_SET where it is set, else models/fullerene-3M at the
    repository's root, downloaded there from Hugging Face the first time (about 20 MB: the ONNX files,
    their metadata, the vocabulary and the weights' license)."""
    env = os.environ.get("CARBON_OCR_SET")
    path = Path(env) if env else ROOT / "models" / "fullerene-3M"
    if not (path / "det.onnx").is_file():
        if env or not download:
            raise FileNotFoundError(f"no Fullerene set in {shown(path)}")
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
        snapshot_download(SET_REPO, local_dir=path,
                          allow_patterns=["*.onnx", "*.json", "*.bin", "LICENSE-WEIGHTS.md"])
    return path


def carbon_ocr_program() -> str | None:
    """The `carbon-ocr` program: $CARBON_OCR, else carbon-ocr on the PATH, else a build in build/<preset>/cli/."""
    if os.environ.get("CARBON_OCR"):
        return os.environ["CARBON_OCR"]
    if shutil.which("carbon-ocr"):
        return shutil.which("carbon-ocr")
    exe = "carbon-ocr.exe" if os.name == "nt" else "carbon-ocr"
    for pattern in (f"build/*/cli/{exe}", f"build/*/cli/*/{exe}", f"build/cli/{exe}"):
        found = sorted(ROOT.glob(pattern))
        if found:
            return str(found[0])
    return None


def show(image, lines=(), pictures=(), title=None, width=7.0, ax=None, color="tab:blue", label=False):
    """A page with its line boxes (and picture boxes in orange). `lines` are the engine's line dicts or boxes."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image = np.asarray(image)
    h, w = image.shape[:2]
    if ax is None:
        _, ax = plt.subplots(figsize=(width, width * h / w))
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="antialiased")
    for i, line in enumerate(lines):
        box = line["box"] if isinstance(line, dict) else line
        x0, y0, x1, y1 = box
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.0, ec=color))
        if label:
            ax.text(x0, y0, str(i), color=color, fontsize=6, va="bottom")
    for x0, y0, x1, y1 in pictures:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=1.5, ec="tab:orange", ls="--"))
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10)
    return ax


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def line_cer(truth_lines, lines) -> float:
    """The CER line by line, whatever the reading order: each truth line against the engine's line that
    overlaps it most, a line the engine missed all wrong, over the truth's characters."""
    import carbon_ocr

    pairs = match_lines(truth_lines, lines)
    errors = carbon_ocr.text_errors([t["text"] for t, _ in pairs], [m["text"] if m else "" for _, m in pairs])
    return float(errors[:, 0].sum() / max(1, errors[:, 1].sum()))


def match_lines(truth_lines, lines, least=0.3):
    """Each truth line paired with the engine's line that overlaps it most (IoU at least `least`), each line
    used once: a list of (truth line, engine line or None)."""
    pairs, used = [], set()
    for t in truth_lines:
        best, best_iou = None, least
        for i, line in enumerate(lines):
            if i in used:
                continue
            v = iou(t["box"], line["box"])
            if v >= best_iou:
                best, best_iou = i, v
        if best is not None:
            used.add(best)
        pairs.append((t, lines[best] if best is not None else None))
    return pairs
