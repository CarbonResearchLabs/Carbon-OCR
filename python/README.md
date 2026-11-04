# python/

`carbon_ocr` is the Carbon-OCR engine for Python, a nanobind module built from the same C++ as `carbon-ocr`. It reads whole pages (`Engine`, and `Pipeline`, the batched one) and runs a single model from numpy (`Tensor`, `Session`). It also exposes the steps the engine is built from, so you can call any one of them on its own: the line crop rule, the CTC decoders, the detector's decoder and the reading-order decoder. These are written once, in `carbon/`, so the module returns the same bytes the engine produces.

```shell
pip install ./python                                        # CMake and a C++20 compiler build it
pip install ./python -C cmake.define.CARBON_WITH_CUDA=ON    # with the CUDA kernels
python -c "import carbon_ocr; print(carbon_ocr.__version__, carbon_ocr.HAS_CUDA)"
```

```python
import numpy as np, carbon_ocr

page = carbon_ocr.Engine("models/fullerene-3M").ocr([carbon_ocr.load_image("page.png")])[0]
print(page["text"])

s = carbon_ocr.Session("models/fullerene-3M/rec-latin.onnx")
logits, = s.run([np.zeros((1, 1, 32, 512), np.float32)])          # [1, 800, 128]
```

## Pages

| API | what |
|---|---|
| `Engine(set, workers=0, pin=True, rec="rec-latin", order_model=None, precision="auto")`, `.ocr(pages, html=False, markdown=False)` | the engine on the CPU: grayscale pages (uint8 [h, w]) in, read in parallel. Each page comes back as a dict: its lines (box, text, confidence, det_score, words) in reading order, which method made the order (`model` or `xy-cut`), its pictures, the order model's blocks (label, confidence, lines, table cells), its rotation, whether it took the second detection pass (`second_pass`), its timings and its text; with `html=True`, also `html`, the page as `carbon-ocr --format html` writes it; with `markdown=True`, also `markdown`, as `--format markdown` writes it |
| `Engine.submit(page)`, `.submit_file(path)`, `.next(html=False, markdown=False)` | the engine as a stream: `submit` gives a page and returns its number, from 0 in the order given, and each page starts as soon as a worker is free; `next` returns the next page's dict in that order, waiting for it, raising the error that page met, and `None` once every page given is out. The array is held, not copied, until `next` has returned its page; `submit` waits while two pages a worker are being read. `submit_file` leaves the file's read and decode to the worker that takes the page, its time in `ms["load"]`. One thread may submit while another takes |
| `Pipeline(set, device=None, page_batch=8, line_batch=256, in_flight=32, threads=0, model_threads=0, rec="rec-latin", order_model=None, engine_cache=None, precision="auto")`, `.ocr(pages, html=False, markdown=False)`, `.stats`, `.device` | the batched pipeline: each model runs on batches drawn from all the pages in flight. Each page's dict is the same as `Engine`'s, and on the CPU so are the results. `stats` counts the runs and what they held. `device="cuda"` needs TensorRT, which this module does not include yet (`carbon-ocr ocr --device cuda` has it). A page's error is raised once every page is through, and the pipeline takes the next call |
| `load_image(path)` | a PNG, JPEG, BMP or PNM page as grayscale uint8 [h, w], as the engine reads it: the decoder's own buffer, with Python's lock released while it reads |

## Models

| API | what |
|---|---|
| `Tensor(array)`, `.numpy()`, `.shape`, `.dtype` | a host tensor, copied from and to numpy: float32, float16, int32, int64, uint8 or bool |
| `Session(model, backend="ort", threads=1)`, `.inputs`, `.outputs`, `.run(inputs)` | one ONNX model, on `"ort"` (ONNX Runtime) or `"native"` (the engine's own CPU kernels). `run` takes arrays or Tensors, as a list in the model's input order or a dict by name, gives back numpy arrays, and releases the GIL while it runs |

## The engine's steps

| API | what |
|---|---|
| `crop_region(box)`, `ctc_frames(box)` | the page rectangle the crop rule takes for a line box, the crop's width, and the CTC frames that width gives |
| `crop(image, box)` | one line's crop: uint8 [32, width], paper 255 |
| `crop_lines(page, boxes, out_width, flip_margin=FLIP_MARGIN)` | many lines of one page, light-on-dark ones flipped, each padded to the batch width with its own ground |
| `light_on_dark(crop, margin)` | whether a crop is light text on a dark ground, which the crop rule flips |
| `resample(image, rect, w, h)` | the crop rule's resampling: Pillow's antialiased bilinear, in fixed point, byte for byte |
| `ground(crop)` | the value a batch pads a crop with |
| `width_bucket(crop_width)` | the batch width the engine runs a crop at: buckets 1.3x apart, from 32 to 2048 px |
| `HEIGHT`, `WIDTH_STEP`, `FRAME`, `PAD_V`, `PAD_H`, `FLIP_MARGIN` | the default crop rule's constants |
| `ctc_greedy(logits, lengths)` | CTC greedy decoding of [n, classes, frames] logits |
| `ctc_collapse(ids, probs, lengths)` | its second half, for a best class per frame found elsewhere (on a GPU) |
| `decode_det(maps, scale)` | the line decoder: line and picture boxes from the detector's maps, each `((x0, y0, x1, y1), score, pixels)` |
| `det_targets(lines, pictures, map_h, map_w)` | the decoder's inverse: the maps that decode to the given boxes, float32 [4, map_h, map_w] (core, top, bottom, picture) |
| `takes_second_pass(lines, below=9.0)` | whether a page takes the detector's second pass: its lines, as `decode_det` gives them at the detector's scale, their median height under `below` px (0: never) |
| `merge_passes(first, second, first_scale=1.0, below=9.0, iou=0.5)` | the two passes merged as the engine merges them: `first` and `second` each `(lines, pictures)` as `decode_det` gives them, in the same pixels; returns `(lines, pictures)` in reading order |
| `DET_SECOND_PASS_BELOW`, `DET_SECOND_PASS_SCALE`, `DET_MERGE_IOU` | the default second pass's constants |
| `cuda_decode_det(maps_ptr, n, map_h, map_w, stream, ...)` | the detector's maps thresholded on a GPU, with only the masks copied back, then decoded into `decode_det`'s boxes (a module built with CUDA) |
| `order_features(boxes, texts, width, height)`, `ORDER_FEATURES`, `BLOCK_LABELS` | the order model's input, float32 [n, 16] for a page's lines, and its block labels |
| `decode_order(successor, same_block, table_row, table_col, table_cell, labels, boxes)` | the order model's outputs decoded into `{order, blocks, sure, fallback}`: each block with its label, confidence, lines and table cells |
| `reading_order(boxes)` | the XY-cut's order, used where there is no order model or it is unsure |
| `text_errors(truths, predicted)`, `char_confusions(truths, predicted, top=50)`, `edit_distances(a, b)` | scoring: each line's edit distances in code points and in words (for CER and WER); the characters confused most over many lines; edit distances between class sequences |
| `HAS_CUDA` | whether the module was built with the CUDA kernels |

The module installs ONNX Runtime's library next to itself, where it finds it. Windows looks in the module's folder first, and on Linux the module's RPATH is its own folder.
