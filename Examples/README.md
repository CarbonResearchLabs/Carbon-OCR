# Examples

Jupyter notebooks on using the Carbon-OCR engine and tuning it. Each one runs on its own on a CPU, reading pages it draws itself, so every result can be measured against the text the page was drawn from.

| notebook | what it covers |
|---|---|
| [1. Quickstart](01-quickstart.ipynb) | the model set, a page read, everything a page's result holds, markdown and HTML, pages turned and skewed, many pages at once |
| [2. The engine's options](02-engine-options.ipynb) | every option of `carbon_ocr.Engine`, each tried on a page where it matters: workers, precisions, the gate, reading order, justified text, tables, the second detection pass, receipts and slides, the recognizer's buckets and kernels |
| [3. Streams and batches](03-streams-and-batches.ipynb) | pages as a stream (`submit`, `submit_file`, `next`), errors that leave the stream going, two threads, the batched `Pipeline`, throughput |
| [4. Inside the engine](04-inside-the-engine.ipynb) | each step by hand with the models' sessions: orientation, detection and the line decoder's thresholds, the second pass and its merge, the crop rule, recognition and CTC decoding, the order model and the XY-cut |
| [5. Measuring accuracy](05-measuring-accuracy.ipynb) | CER and WER, pages made harder by blur, noise, JPEG, low resolution and faded ink, the characters confused, a line's confidence against its errors, a folder of your own pages measured |
| [6. The command line and the server](06-command-line-and-server.ipynb) | `carbon-ocr ocr` and its eight formats, the command line's options, searchable PDFs and PDFs with text of their own, the timing tools, `carbon-ocr serve` over HTTP, the GPU and OCRmyPDF |

## Setup

From the repository's root:

```shell
pip install ./python                     # the carbon_ocr module: CMake and a C++20 compiler build it
pip install -r Examples/requirements.txt # Jupyter, matplotlib, Pillow, huggingface_hub
jupyter lab Examples
```

- **The model set.** The notebooks read pages with `fullerene-3M` from `models/fullerene-3M`, and download it there from [Hugging Face](https://huggingface.co/CarbonResearchLabs/fullerene-3M) the first time (about 20 MB). To use a set kept elsewhere, set `CARBON_OCR_SET` to its folder.
- **The command line.** Notebook 6 runs `carbon-ocr`, built with CMake as the [build guide](https://github.com/CarbonResearchLabs/Carbon-OCR/blob/main/docs/BUILD.md) describes. It finds the program in `build/<preset>/cli/`, on the `PATH`, or where `CARBON_OCR` points. Notebooks 1 to 5 need only the Python module.

## The files

| file | what |
|---|---|
| `sample_pages.py` | draws the sample pages with Pillow, in the DejaVu fonts matplotlib ships, each with its truth (its lines' text and boxes) beside it: a letter, two columns with a picture, a table, a form with fine print, a newspaper, light text on dark grounds, a receipt, a slide, and the letter turned, skewed and scanned badly |
| `helpers.py` | finds the model set and `carbon-ocr`, draws pages with their boxes, matches the engine's lines to the truth's |
| `data/`, `out/` | what the notebooks write as they run: the sample pages and their truths, and the files the command line writes |

The outputs saved in the notebooks come from one run on a laptop's CPU (an Intel Core i7-13620H, Windows 11, Python 3.12). Your times will differ, and so may a few characters on a different CPU's kernels. The pages are drawn, so they are cleaner than most scans: measure the engine on your own pages, as notebook 5 shows, before you rely on its numbers for them.
