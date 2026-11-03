# Using Carbon-OCR

This guide covers the `carbon-ocr` command line, the output formats, the Python module and the OCRmyPDF plugin. Every command that reads pages takes a Fullerene model set with `--set DIR`. A set is a folder holding its models and their metadata, laid out as described in [models/README.md](../models/README.md). The weights are not released yet.

**Contents**
1. Reading pages
2. Output formats
3. Timing and inspecting: `cpus`, `pages`, `bench`, `info`, `compare`
4. Python
5. OCRmyPDF

## <img src="assets/chip-01.svg" height="22" alt=""> Reading pages

```shell
carbon-ocr ocr page.png scan.jpg --set models/fullerene-3M          # plain text
carbon-ocr ocr page.png --set models/fullerene-3M --backend ort     # all on ONNX Runtime
carbon-ocr ocr report.pdf --set models/fullerene-3M --dpi 300       # each page of a PDF
carbon-ocr ocr page.png --set models/fullerene-3M --format json     # boxes, words, timings
carbon-ocr ocr scan.pdf --set models/fullerene-3M --format pdf --output searchable.pdf
carbon-ocr ocr scans/*.png --set models/fullerene-3M --device cuda --engine-cache .engines
```

`ocr` reads PNG, JPEG, BMP and PNM pages, and PDFs. A PDF's pages are rendered by PDFium in `carbon-pdf` helper processes. If a document crashes PDFium, only a helper dies, and your program keeps running.

A born-digital PDF page whose text layer passes the engine's checks is read from that layer instead of by OCR. Its pictures (the page's image objects) are still read by OCR, and their lines join the layer's. The JSON output gives each page's source (`ocr`, `text layer` or `hybrid`). For a PDF page read by OCR, it also says why the page's text layer was not used.

By default the pages run in parallel on the CPU, one worker per physical core. `--device cuda` reads them through the batched pipeline on the GPU instead: TensorRT runs the set's FP16 files, and each model is batched across the pages in flight. This needs a build with TensorRT and CUDA ([BUILD.md](BUILD.md)).

| option | default | what |
|---|---|---|
| `--set DIR` | required | the model set |
| `--rec NAME` | `rec-latin` | the recognizer: `DIR/NAME.onnx` and `DIR/NAME.vocab.json` |
| `--format F` | `text` | `text`, `json`, `hocr`, `alto`, `page`, `markdown` or `pdf` |
| `--output FILE` | standard output | the file to write; `pdf` needs one |
| `--out-dir DIR` | | `page` (PAGE XML) writes one file per page here |
| `--text-out FILE` | | also write the pages' text to this file, beside any format |
| `--text-only` | off | `pdf`: only the invisible text, without the page images |
| `--image-dpi N` | 300 | `pdf`: the dpi an image page is placed at |
| `--dpi N` | 200 | the dpi a PDF's pages are rendered at |
| `--text-layer auto\|off` | `auto` | read a PDF page from its own text layer where it passes the checks |
| `--hybrid auto\|off` | `auto` | read the pictures of a page that was read from its text layer by OCR |
| `--as-given` | off | no quarter turns and no deskew: the page as given |
| `--backend ort\|native` | per model | run every model on one backend; by default a model runs on the native int8 backend where the set has its `.int8.onnx`, else on ONNX Runtime |
| `--workers N` | one per physical core | CPU workers |
| `--no-pin` | pinned | do not pin workers to cores |
| `--device cpu\|cuda` | `cpu` | `cuda`: the batched pipeline on the GPU |
| `--batch N` | 8 | `cuda`: pages in a detection batch |
| `--in-flight N` | 32 | `cuda`: pages in the pipeline at once |
| `--engine-cache DIR` | | keep TensorRT's built engines here, for later runs |

The reading order comes from the set's `order.onnx` when the set has one; otherwise the engine orders lines by the XY-cut. The order model is also used only where it is sure of its result, and the XY-cut takes over elsewhere. The JSON output records which of the two ordered each page.

## <img src="assets/chip-02.svg" height="22" alt=""> Output formats

| format | what |
|---|---|
| `text` | each page's lines in reading order, one per line, with a form feed between pages |
| `json` | one object with every page: its size, rotation and source, each line's box, text, confidence and words, the pictures, the order model's blocks with their labels and table cells, and timings |
| `hocr` | hOCR 1.2 (XHTML), the format Tesseract writes and OCRmyPDF reads: blocks, lines (headings and captions marked) and words with their boxes and confidences |
| `alto` | ALTO XML 4.4: text blocks with their labels, lines, words with their confidences |
| `page` | PAGE XML (2019-07-15), one file per page in `--out-dir`: regions typed by their labels, lines, words |
| `markdown` | headings, paragraphs, lists, tables (as HTML, from their cells) and captions, from the order model's blocks; running headers and footers are left out |
| `pdf` | a searchable PDF: each page's image with its words as invisible text above it; `--text-only` writes the text layer alone |

Boxes are in the page image's pixels. For a PDF page they are in the pixels of the page as rendered at `--dpi`.

## <img src="assets/chip-03.svg" height="22" alt=""> Timing and inspecting

```shell
carbon-ocr version
carbon-ocr cpus                                                  # instruction sets and cores
carbon-ocr cpus --gemm                                           # the int8 GEMM timed per ISA
carbon-ocr pages models/fullerene-3M --workers 1,2,4,all         # a set timed page by page
carbon-ocr pages models/fullerene-3M --in-flight 1               # one page at a time: latency
carbon-ocr info  model.onnx                                      # inputs, outputs, metadata
carbon-ocr bench model.onnx --backend ort --shape x=64x1x32x640 --runs 200
carbon-ocr bench model.int8.onnx --backend native --shape x=8x1x32x512
carbon-ocr bench model.onnx --backend trt --shape x=64x1x32x640 --engine-cache .engines
carbon-ocr bench model.onnx --workers 1,2,4,all                  # a pinned worker per core
carbon-ocr compare model.onnx --shape x=8x1x32x512               # native against ONNX Runtime
```

- **`cpus`** prints the instruction sets the kernels use and the cores the workers run on. `--gemm` times the int8 GEMM on one thread with each instruction set, at the shapes of the models' convolutions.
- **`pages`** times a set the way the CPU pipeline runs a page: orientation beside detection, the line decoder, the lines in tasks any idle worker can take, then the order model. Its inputs are fixed at the models' design sizes. `--in-flight 1` measures one page at a time on every core (latency). By default every page is queued at once (throughput). `--det auto|one|wide|strips` picks how detection uses the cores.
- **`bench`** times one model with fixed inputs and reports the median, p90 and fastest run. On TensorRT it times two runs: from host memory with the copies included, and on device buffers, the way the GPU pipeline runs a model. `--opt-level 0-5` sets TensorRT's builder optimization level (default 3), and `--no-graphs` turns off CUDA graphs. `--isa NAME` holds the native backend to one instruction set. `--workers` runs a pool of pinned workers, each with its own session, and reports their scaling.
- **`compare`** runs a model on the native backend and on ONNX Runtime and reports how far apart their outputs are. `--against FILE` gives ONNX Runtime a different file, such as the FP32 model for an int8 one.

## <img src="assets/chip-04.svg" height="22" alt=""> Python

```shell
pip install ./python
```

```python
import numpy as np
import carbon_ocr

engine = carbon_ocr.Engine("models/fullerene-3M", workers=0)        # 0: one per core
pages = engine.ocr([carbon_ocr.load_image("a.png"), carbon_ocr.load_image("b.png")])
for page in pages:
    print(page["rotation"], page["order"])                         # "model" or "xy-cut"
    for line in page["lines"]:                                      # in reading order
        print(line["box"], round(line["confidence"], 3), line["text"])

pipeline = carbon_ocr.Pipeline("models/fullerene-3M", page_batch=8)  # batched; same results
pages = pipeline.ocr([carbon_ocr.load_image("a.png")])
print(pipeline.stats)

session = carbon_ocr.Session("models/fullerene-3M/rec-latin.onnx", backend="ort")
logits, = session.run([np.zeros((1, 1, 32, 512), np.float32)])     # [1, 800, 128]
```

The module also exposes the engine's building blocks: the crop rule (`crop`, `crop_lines`, `crop_region`), the CTC decoders (`ctc_greedy`, `ctc_collapse`), the detector's decoder (`decode_det`), the order model's features and decoder (`order_features`, `decode_order`, `reading_order`), and text scoring (`text_errors`, `char_confusions`). [python/README.md](../python/README.md) lists all of them.

## <img src="assets/chip-05.svg" height="22" alt=""> OCRmyPDF

`tools/ocrmypdf/carbon_ocrmypdf.py` plugs Carbon-OCR into OCRmyPDF 17 as its OCR engine. OCRmyPDF prepares each page itself, turning and deskewing it, and the plugin runs `carbon-ocr ocr --as-given` on the page. With OCRmyPDF's default renderer, the plugin returns hOCR. With `--pdf-renderer sandwich`, it returns a text-only PDF at the image's own dpi.

```shell
pip install -r tools/ocrmypdf/requirements.txt
ocrmypdf --plugin tools/ocrmypdf/carbon_ocrmypdf.py \
    --carbon-set models/fullerene-3M in.pdf out.pdf
```

The plugin's options:
- `--carbon-set DIR`: the model set. Without it, the plugin uses the set named by `CARBON_OCR_SET`.
- `--carbon-ocr PATH`: the program. Without it, the plugin uses `CARBON_OCR`, then the `carbon-ocr` on `PATH`.
- `--carbon-workers N`: the number of workers.

OCRmyPDF checks for `tesseract` on every run, so Tesseract must be installed, though the plugin never runs it.
