// carbon-ocr: the command line.  It reads page images (ocr), and inspects and
// times models and model sets.
//
//   carbon-ocr ocr IMAGE|PDF... --set DIR [--rec rec-latin] [--backend ort|native]
//                           [--precision auto|fp32|fp16|int8]
//                           [--dpi 200|auto] [--text-layer auto|off] [--hybrid auto|detect|off]
//                           [--second-pass auto|off]
//                           [--layer-lines layer|detected]
//                           [--gate auto|off] [--block-order geometry|model]
//                           [--grid on|off] [--blocks-first on|off] [--turn auto|bin|quarters] [--max-skew D]
//                           [--wide-skew-gain F] [--turn-check on|off]
//                           [--format text|json|hocr|alto|page|markdown|html|pdf] [--out-dir DIR]
//                           [--output FILE] [--text-out FILE] [--text-only] [--image-dpi 300] [--as-given]
//                           [--workers N] [--no-pin] [--device cpu|cuda] [--batch 8] [--in-flight N]
//                           [--engine-cache DIR] [--head-accumulate fp32|fp16]
//   carbon-ocr serve --set DIR [--host 127.0.0.1] [--port 8080] [--max-body-mb 64] [--queue 64] [--quiet]
//                           [--decoders N] [--image-dpi 300] [and ocr's --rec --backend --precision
//                           --second-pass --as-given --workers --no-pin --device --batch --in-flight
//                           --engine-cache --head-accumulate --text-layer --hybrid --layer-lines]
//   carbon-ocr version
//   carbon-ocr cpus [--gemm]
//   carbon-ocr info  MODEL [--backend ort|trt|native] [--shape NAME=1x1x32x256 ...]
//   carbon-ocr bench MODEL [--backend ort|trt|native] [--shape NAME=1x1x32x256 ...] [--isa NAME]
//                          [--runs 100] [--warmup 10] [--threads 1] [--engine-cache DIR]
//                          [--opt-level 3] [--no-graphs] [--workers 1,2,4,all] [--no-pin]
//   carbon-ocr pages SETDIR [--rec rec-latin] [--lines 40] [--batch 8] [--pages 32] [--backend ort|native]
//                           [--precision auto|fp32|fp16|int8]
//                           [--in-flight N] [--workers 1,2,4,all] [--no-pin] [--threads 1]
//                           [--det auto|one|wide|strips]
//   carbon-ocr compare MODEL [--shape NAME=1x1x32x256 ...] [--isa NAME] [--against MODEL]
//
// --shape fixes a free dimension of an input (bench needs one for every
// input with free dimensions); for TensorRT it is also the engine's one
// profile, an exact one.  On TensorRT, bench times the run from host memory
// and the run on device buffers, the one the GPU pipeline makes.
//
// compare runs a model on the native backend (the engine's own executor) and
// on ONNX Runtime, on the same random input, and prints how far apart each
// output is.  --isa picks the native backend's instruction set; --against runs
// ONNX Runtime on another file (an int8 file natively against its FP32 one).
//
// cpus prints the instruction sets the kernels use, how their int8 products are
// made (a dot product, or widened to 16 bits: then FP32 is the one-worker latency
// choice, carbon::cpu::int8_widens) and the cores the workers
// use; --gemm times the int8 GEMM on one thread with each instruction set, at
// the shapes of Fullerene's convolutions.  bench --workers runs, for each
// count, a pool of that many workers pinned one per core, each with its own
// session and input, all at once: the CPU layout.  It reports
// the runs per second of the whole pool and its scaling against the first
// count's per-worker rate.
//
// ocr and pages run each model of a set on the native backend where the set has
// its int8 file (<model>.int8.onnx), else on ONNX Runtime (carbon::open_set_model);
// --backend runs every model on one (ocr's order model: order.native.bin natively
// unless --backend ort, else order.onnx on ONNX Runtime; pages', order.onnx on ONNX
// Runtime).  --precision
// picks the files instead (carbon::Precision): fp32 <model>.onnx (on ONNX Runtime
// unless --backend native), fp16 <model>.fp16.onnx on the native backend's FP16
// kernels, int8 <model>.int8.onnx (native unless --backend ort); a set without
// the file is an error.  auto, the default, is the choice above.
//
// ocr writes text, JSON, hOCR, ALTO XML, markdown, HTML, a searchable PDF (the pages'
// images under their words, --output FILE; carbon/output.h; --text-only: the
// words alone, an image's page at --image-dpi), or PAGE XML, a file a page in
// --out-dir; --text-out FILE writes the text beside any of them.  --as-given
// reads pages as given, without quarter turns or deskew (the OCRmyPDF plugin,
// tools/ocrmypdf, which turns and levels pages itself).  It reads PNG, JPEG, BMP, PNM, GIF or WebP pages, and the
// pages of PDFs rendered at --dpi, each file's format by its first bytes, not its name
// (carbon/pdf.h), with the Fullerene set in --set.  A PDF page whose own text
// passes the checks (carbon/textlayer.h) is read from it, not by OCR, unless
// --text-layer off; the JSON gives each page's source, and for a PDF page read
// by OCR why its text layer was not used.  A page whose layer runs sideways or
// upside down (most of its characters a quarter turn on the page as shown) is
// read turned upright, its rotation saying the turn (carbon::upright_turn); a page
// read by OCR is given as shown.  Such a page's pictures (its image
// objects), its ink the layer does not account for and its lines bad-code runs touch
// (set aside) are read by OCR, on the page with only them left, and their lines
// join the layer's (source "hybrid"), unless --hybrid off: where the detector found
// the page's lines for its layer, the lines it found there, those not under the
// layer's own (none: the pass is not run; a page with lines set aside takes the pass
// on its pictures' page); else, and with --hybrid detect, the lines the detector
// finds on that page.  A page whose lines are
// tiny (the first pass's median under 9 px at the detector's scale) is detected again at
// twice the scale and the passes merged (carbon/detect.h), unless --second-pass off
// (carbon::Engine, the pipeline v0 on the CPU; reading order by the set's order
// model, order.native.bin or order.onnx, where it has one, every page's, else by the
// XY-cut): text prints each page's lines,
// pages apart by a form feed; json one object with every page.  The pages run
// in parallel, a worker per physical core.  --device cuda reads them through the
// batched pipeline on the GPU instead (carbon/pipeline.h: TensorRT, from the set's
// files at --precision, carbon::gpu_set_model: auto and fp16 <model>.fp16.onnx and
// the recognizer's trunk and fused head, auto falling back to FP32 where a file is
// missing; fp32 <model>.onnx; int8 <model>.int8-sym.onnx; --batch pages a detection
// run, --in-flight pages at once, the built engines kept in --engine-cache).
//
// serve keeps the set's models loaded and reads pages sent over HTTP/1.1 (carbon/server.h): POST /ocr with a
// page's image or a PDF as the body (PNG, JPEG, BMP, PNM, GIF, WebP or PDF, by its first bytes; ?format= as ocr's,
// json by default: the page's object, a PDF's {"pages": [...]}; a PDF read as ocr reads it, by --text-layer,
// --hybrid and --layer-lines), POST /parse (a route for document-parsing benchmark clients: a JSON body of the
// page's file in base64, its suffix and its settings), GET /health, GET /served and GET /info. Every client's
// pages share the engine's workers (on the GPU, the pipeline's batches), each request answered as soon as its
// pages are read; --queue requests wait past the pages in flight, more are answered 503.
// It listens on --host (127.0.0.1: this machine only; 0.0.0.0 every interface) and --port, until Ctrl+C.
//
// pages times a model set's ONNX files (SETDIR/orient.onnx, det.onnx,
// script.onnx where the set has it, <rec>.onnx, order.onnx) the way the CPU
// pipeline runs a page: orientation and detection, the line
// decoder, then the page's lines in tasks any idle worker may take (the script
// classifier and the recognizer), then the order model when the last is done, each step planned
// for the cores free at the time (see pages() below).  The inputs are
// constant, at the budget's sizes: a 512 x 512 thumbnail, a 1024 x 800 page,
// 32 x 512 crops, 16 features a line; a constant page has no lines to find, so the
// decoder reads the maps of a rendered page of --lines lines.  --in-flight N keeps N pages going at a time (1: one
// page on every core, its latency); by default every page is queued at once
// (throughput).

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "carbon/carbon.h"
#include "carbon/document.h"
#include "carbon/output.h"
#include "carbon/pipeline.h"
#include "carbon/server.h"
#include "carbon/textlayer.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#ifdef CARBON_WITH_TENSORRT
#include <cuda_runtime_api.h>
#endif

namespace {

constexpr int USAGE = 2;

struct Args {
  std::string command, model;
  carbon::Backend backend = carbon::Backend::kOnnxRuntime;
  std::vector<std::pair<std::string, carbon::Shape>> shapes;
  int runs = 100, warmup = 10, threads = 1, opt_level = 3;
  bool graphs = true, pin = true, gemm = false;
  std::vector<int> workers;  // bench --workers, pages --workers
  std::string rec = "rec-latin";  // pages
  std::string det = "auto";
  std::string set, format = "text";  // ocr
  std::vector<std::string> images;
  int lines = 40, batch = 8, pages = 32, in_flight = 0;
  double dpi = 200;  // ocr: a PDF's pages
  std::string text_layer = "auto";  // ocr: auto, a page's text layer where it passes; off
  // ocr: auto, the pictures of a page read from its text layer read by OCR, from the lines the detector found on the
  // page where it ran; detect, its pictures detected again on their own; off
  std::string hybrid = "auto";
  std::string layer_lines = "detected";  // ocr: a text layer's lines the detector's, or "layer": PDFium's
  std::string text_pages = "gpu";  // ocr --device cuda: a text layer's pages' steps on the GPU's pipeline, or "cpu"
  std::string second_pass = "auto";  // ocr: auto, tiny text detected again at twice the scale; off
  std::string gate = "auto";         // ocr, serve: auto, lines that are not text dropped (carbon::GateRule); off
  std::string prefetch = "on";       // serve on the GPU: the next group's pages uploaded while the current is read
  std::string head_accumulate = "fp32";  // the GPU's fused head: its products summed in fp32 or fp16
  std::string table_spans = "on";       // ocr, serve: the order model's tables laid out again with spans
  std::string grid = "on";              // ocr, serve: the order model's tables by geometry (carbon::GridRule)
  std::string blocks_first = "off";     // ocr, serve: blocks as the same-block head's clusters (carbon::OrderRule)
  std::string turn = "auto";            // ocr, serve: orientation's turn by quarters where it is read, else its likeliest
                                        // bin (auto); by its bin, or by quarters, always
  std::string turn_check = "on";        // ocr, serve: turns read before they are made (carbon::TurnCheck)
  std::optional<double> max_skew, wide_skew_gain;  // ocr, serve: deskew's range and its gain past 5 degrees
  std::string spaces = "model";         // ocr, serve: spaces as the recognizer reads them, or borne out by the ink
  std::string block_order = "geometry";  // ocr, serve: the order model's blocks by geometry, or as the model gave them
  std::optional<double> second_pass_share, single_pixel;  // ocr, serve: carbon::DetRule's, where given (0: off)
  int beam = 0;  // ocr, serve: the CTC prefix beam's width for unsure lines (carbon::BeamRule); under 2: off
  int beam_topk = carbon::BeamRule{}.topk;
  float beam_gate = carbon::BeamRule{}.gate;
  std::string lm;  // ocr, serve: the beam's n-gram model (empty: the set's own where it has one; off: none)
  std::optional<float> lm_alpha, lm_beta;  // and its weights (none: the model file's)
  std::string tlm;  // ocr, serve: the transformer that rescores the beam's readings (empty: the set's; off: none)
  std::optional<float> tlm_alpha;
  std::optional<int> tlm_gpu;  // the transformer on this GPU (-1: the CPU); none: the pipeline's where it has one
  std::string lm_mode = "auto";  // ocr, serve: the language models as one switch (carbon::LmMode): none, ngram, ...
  std::optional<float> lm_gate;  // and the beam's gate under it (none: the models' tuned one)
  float beam_span = carbon::BeamRule{}.span;  // ocr, serve: span-level decoding's sure frames (carbon::BeamRule::span)
  std::array<carbon::EngineOptions::KindWeights, carbon::kLineKinds> lm_kinds{};  // --lm-kind: the beam by kind
  std::string out_dir;              // ocr --format page: a file a page here
  std::string output;                // ocr: the file to write (pdf needs one); empty: standard output
  std::string text_out;              // ocr: also the pages' text, to this file
  bool text_only = false;            // ocr --format pdf: the text layer alone, no images
  bool json_html = false, json_markdown = false;  // ocr --format json: each page's html body and markdown in it
  double image_dpi = 300;            // ocr --format pdf: an image's page at this dpi
  bool as_given = false;             // ocr: no quarter turns, no deskew: the boxes the given page's
  std::string device = "cpu";        // ocr: cpu, the engine's workers; cuda, the batched pipeline on the GPU
  std::string engine_cache;
  std::string isa;  // the native backend's
  std::string against;  // compare: ONNX Runtime runs this file
  bool backend_given = false;  // ocr and pages choose per model otherwise
  carbon::Precision precision = carbon::Precision::kAuto;  // ocr and pages: the set's files
  std::string host = "127.0.0.1";    // serve
  int port = 8080, queue = 64, max_body_mb = 64, decoders = 0;
  bool quiet = false;                // serve: no line a request
};

[[noreturn]] void usage(const std::string& why) {
  std::fprintf(stderr,
               "carbon-ocr: %s\n"
               "usage: carbon-ocr ocr IMAGE|PDF... --set DIR [--rec NAME] [--workers N]\n"
               "                        [--no-pin] [--backend ort|native] [--precision auto|fp32|fp16|int8]\n"
               "                        [--dpi N|auto] [--text-layer auto|off]\n"
               "                        [--hybrid auto|detect|off] [--layer-lines layer|detected]\n"
               "                        [--second-pass auto|off] [--gate auto|off]\n"
               "                        [--block-order geometry|model] [--second-pass-share F] [--single-pixel F]\n"
               "                        [--table-spans on|off] [--spaces model|ink]\n"
               "                        [--grid on|off] [--blocks-first on|off] [--turn auto|bin|quarters]\n"
               "                        [--max-skew D] [--wide-skew-gain F] [--turn-check on|off]\n"
               "                        [--lm-mode none|ngram|transformer|both] [--lm-gate F]\n"
               "                        [--beam N] [--beam-gate F] [--beam-topk N] [--beam-span P]\n"
               "                        [--lm FILE|off] [--lm-alpha F] [--lm-beta F] [--tlm FILE|off] [--tlm-alpha F]\n"
               "                        [--lm-kind prose|mixed|code=ALPHA,BETA,GATE]... [--tlm-gpu N|cpu]\n"
               "                        [--format text|json|hocr|alto|page|markdown|html|pdf] [--out-dir DIR]\n"
               "                        [--output FILE] [--text-out FILE] [--text-only] [--image-dpi N]\n"
               "                        [--as-given] [--device cpu|cuda] [--batch N] [--in-flight N]\n"
               "                        [--engine-cache DIR] [--head-accumulate fp32|fp16] [--text-pages gpu|cpu]\n"
               "                        [--json-html] [--json-markdown]\n"
               "       carbon-ocr serve --set DIR [--host ADDR] [--port N] [--max-body-mb N] [--queue N] [--quiet]\n"
               "                        [--decoders N] [--image-dpi N] [ocr's model, worker and device options]\n"
               "       carbon-ocr version\n"
               "       carbon-ocr cpus [--gemm]\n"
               "       carbon-ocr info  MODEL [--backend ort|trt|native] [--shape NAME=AxBxC ...]\n"
               "       carbon-ocr bench MODEL [--backend ort|trt|native] [--shape NAME=AxBxC ...] [--isa NAME]\n"
               "                        [--runs N] [--warmup N] [--threads N] [--engine-cache DIR]\n"
               "                        [--opt-level 0-5] [--no-graphs] [--workers N,N,...|all] [--no-pin]\n"
               "       carbon-ocr pages SETDIR [--rec NAME] [--lines N] [--batch N] [--pages N]\n"
               "                        [--backend ort|native] [--precision auto|fp32|fp16|int8]\n"
               "                        [--in-flight N] [--workers N,N,...|all] [--no-pin] [--threads N]\n"
               "                        [--det auto|one|wide|strips]\n"
               "       carbon-ocr compare MODEL [--shape NAME=AxBxC ...] [--isa NAME] [--against MODEL]\n",
               why.c_str());
  std::exit(USAGE);
}

int to_int(std::string_view flag, const std::string& v, int lo) {
  char* end = nullptr;
  long n = std::strtol(v.c_str(), &end, 10);
  if (end == v.c_str() || *end || n < lo) {
    usage(std::string(flag) + " takes a whole number >= " + std::to_string(lo) + ", got '" + v + "'");
  }
  return static_cast<int>(n);
}

double to_double(std::string_view flag, const std::string& v, double lo, double hi) {
  char* end = nullptr;
  const double x = std::strtod(v.c_str(), &end);
  if (end == v.c_str() || *end || !(x >= lo && x <= hi)) {
    usage(std::string(flag) + " takes a number from " + std::to_string(lo) + " to " + std::to_string(hi) + ", got '" +
          v + "'");
  }
  return x;
}

// "x=1x1x32x256" -> ("x", [1, 1, 32, 256])
std::pair<std::string, carbon::Shape> to_shape(const std::string& v) {
  size_t eq = v.find('=');
  if (eq == std::string::npos || eq == 0 || eq + 1 == v.size()) {
    usage("--shape takes NAME=AxBxC, got '" + v + "'");
  }
  carbon::Shape s;
  size_t i = eq + 1;
  while (i <= v.size()) {
    size_t x = v.find('x', i);
    if (x == std::string::npos) x = v.size();
    s.push_back(to_int("--shape", v.substr(i, x - i), 0));
    i = x + 1;
  }
  return {v.substr(0, eq), s};
}

// "1,2,4,all" -> [1, 2, 4, the default worker count]
std::vector<int> to_workers(const std::string& v) {
  std::vector<int> n;
  size_t i = 0;
  while (i <= v.size()) {
    size_t c = v.find(',', i);
    if (c == std::string::npos) c = v.size();
    std::string item = v.substr(i, c - i);
    n.push_back(item == "all" ? carbon::cpu_topology().default_workers() : to_int("--workers", item, 1));
    i = c + 1;
  }
  return n;
}

Args parse(int argc, char** argv) {
  Args a;
  if (argc < 2) usage("no command");
  a.command = argv[1];
  if (a.command == "--version" || a.command == "version") {
    a.command = "version";
    return a;
  }
  if (a.command == "cpus") {
    for (int i = 2; i < argc; ++i) {
      if (std::string(argv[i]) != "--gemm") usage(std::string("unknown option ") + argv[i]);
      a.gemm = true;
    }
    return a;
  }
  if (a.command != "info" && a.command != "bench" && a.command != "pages" && a.command != "ocr" &&
      a.command != "compare" && a.command != "serve") {
    usage("unknown command '" + a.command + "'");
  }
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) usage(arg + " needs a value");
      return argv[++i];
    };
    if (arg == "--backend") {
      std::string v = value();
      auto b = carbon::parse_backend(v);
      if (!b) usage("unknown backend '" + v + "' (ort, trt, native)");
      a.backend = *b;
      a.backend_given = true;
    } else if (arg == "--precision") {
      std::string v = value();
      auto p = carbon::parse_precision(v);
      if (!p) usage("unknown precision '" + v + "' (auto, fp32, fp16, int8)");
      a.precision = *p;
    } else if (arg == "--shape") {
      a.shapes.push_back(to_shape(value()));
    } else if (arg == "--runs") {
      a.runs = to_int(arg, value(), 1);
    } else if (arg == "--warmup") {
      a.warmup = to_int(arg, value(), 0);
    } else if (arg == "--threads") {
      a.threads = to_int(arg, value(), 1);
    } else if (arg == "--engine-cache") {
      a.engine_cache = value();
    } else if (arg == "--isa") {
      a.isa = value();
    } else if (arg == "--against") {
      a.against = value();
    } else if (arg == "--opt-level") {
      a.opt_level = to_int(arg, value(), 0);
      if (a.opt_level > 5) usage("--opt-level takes 0 to 5");
    } else if (arg == "--no-graphs") {
      a.graphs = false;
    } else if (arg == "--workers") {
      a.workers = to_workers(value());
    } else if (arg == "--no-pin") {
      a.pin = false;
    } else if (arg == "--rec") {
      a.rec = value();
    } else if (arg == "--lines") {
      a.lines = to_int(arg, value(), 1);
    } else if (arg == "--batch") {
      a.batch = to_int(arg, value(), 1);
    } else if (arg == "--pages") {
      a.pages = to_int(arg, value(), 1);
    } else if (arg == "--in-flight") {
      a.in_flight = to_int(arg, value(), 1);
    } else if (arg == "--set") {
      a.set = value();
    } else if (arg == "--device") {
      a.device = value();
      if (a.device != "cpu" && a.device != "cuda") usage("--device takes cpu or cuda");
    } else if (arg == "--text-layer") {
      a.text_layer = value();
      if (a.text_layer != "auto" && a.text_layer != "off") usage("--text-layer takes auto or off");
    } else if (arg == "--hybrid") {
      a.hybrid = value();
      if (a.hybrid != "auto" && a.hybrid != "detect" && a.hybrid != "off") usage("--hybrid takes auto, detect or off");
    } else if (arg == "--layer-lines") {
      a.layer_lines = value();
      if (a.layer_lines != "layer" && a.layer_lines != "detected") usage("--layer-lines takes layer or detected");
    } else if (arg == "--text-pages") {
      a.text_pages = value();
      if (a.text_pages != "gpu" && a.text_pages != "cpu") usage("--text-pages takes gpu or cpu");
    } else if (arg == "--second-pass") {
      a.second_pass = value();
      if (a.second_pass != "auto" && a.second_pass != "off") usage("--second-pass takes auto or off");
    } else if (arg == "--prefetch") {
      a.prefetch = value();
      if (a.prefetch != "on" && a.prefetch != "off") usage("--prefetch takes on or off");
    } else if (arg == "--head-accumulate") {
      a.head_accumulate = value();
      if (a.head_accumulate != "fp32" && a.head_accumulate != "fp16") usage("--head-accumulate takes fp32 or fp16");
    } else if (arg == "--gate") {
      a.gate = value();
      if (a.gate != "auto" && a.gate != "off") usage("--gate takes auto or off");
    } else if (arg == "--spaces") {
      a.spaces = value();
      if (a.spaces != "model" && a.spaces != "ink") usage("--spaces takes model or ink");
    } else if (arg == "--table-spans") {
      a.table_spans = value();
      if (a.table_spans != "on" && a.table_spans != "off") usage("--table-spans takes on or off");
    } else if (arg == "--block-order") {
      a.block_order = value();
      if (a.block_order != "geometry" && a.block_order != "model") usage("--block-order takes geometry or model");
    } else if (arg == "--grid") {
      a.grid = value();
      if (a.grid != "on" && a.grid != "off") usage("--grid takes on or off");
    } else if (arg == "--blocks-first") {
      a.blocks_first = value();
      if (a.blocks_first != "on" && a.blocks_first != "off") usage("--blocks-first takes on or off");
    } else if (arg == "--turn") {
      a.turn = value();
      if (a.turn != "auto" && a.turn != "bin" && a.turn != "quarters") usage("--turn takes auto, bin or quarters");
    } else if (arg == "--turn-check") {
      a.turn_check = value();
      if (a.turn_check != "on" && a.turn_check != "off") usage("--turn-check takes on or off");
    } else if (arg == "--max-skew") {
      a.max_skew = to_double(arg, value(), 0, 45);
    } else if (arg == "--wide-skew-gain") {
      a.wide_skew_gain = to_double(arg, value(), 1, 100);
    } else if (arg == "--beam") {
      a.beam = to_int(arg, value(), 0);
    } else if (arg == "--lm-mode") {
      a.lm_mode = value();
      if (!carbon::parse_lm_mode(a.lm_mode)) usage(arg + " takes auto, none, ngram, transformer or both");
    } else if (arg == "--lm-gate") {
      const std::string v = value();
      char* end = nullptr;
      const double x = std::strtod(v.c_str(), &end);
      if (end == v.c_str() || *end || !(x >= 0 && x <= 2)) usage(arg + " takes a number from 0 to 2");
      a.lm_gate = static_cast<float>(x);
    } else if (arg == "--tlm") {
      a.tlm = value();
    } else if (arg == "--tlm-gpu") {
      const std::string v = value();
      a.tlm_gpu = v == "cpu" ? -1 : to_int(arg, v, 0);
    } else if (arg == "--tlm-alpha") {
      const std::string v = value();
      char* end = nullptr;
      const double x = std::strtod(v.c_str(), &end);
      if (end == v.c_str() || *end) usage(arg + " takes a number");
      a.tlm_alpha = static_cast<float>(x);
    } else if (arg == "--lm") {
      a.lm = value();
    } else if (arg == "--lm-alpha" || arg == "--lm-beta") {
      const std::string v = value();
      char* end = nullptr;
      const double x = std::strtod(v.c_str(), &end);
      if (end == v.c_str() || *end) usage(arg + " takes a number");
      (arg == "--lm-alpha" ? a.lm_alpha : a.lm_beta) = static_cast<float>(x);
    } else if (arg == "--beam-topk") {
      a.beam_topk = to_int(arg, value(), 1);
    } else if (arg == "--beam-gate") {
      const std::string v = value();
      char* end = nullptr;
      const double x = std::strtod(v.c_str(), &end);
      if (end == v.c_str() || *end || !(x >= 0 && x <= 2)) usage(arg + " takes a number from 0 to 2");
      a.beam_gate = static_cast<float>(x);
    } else if (arg == "--beam-span") {
      const std::string v = value();
      char* end = nullptr;
      const double x = std::strtod(v.c_str(), &end);
      if (end == v.c_str() || *end || !(x >= 0 && x <= 1)) usage(arg + " takes a probability from 0 (off) to 1");
      a.beam_span = static_cast<float>(x);
    } else if (arg == "--lm-kind") {  // KIND=ALPHA,BETA,GATE, an empty field the beam's own
      const std::string v = value();
      const size_t eq = v.find('=');
      const auto kind = carbon::parse_line_kind(v.substr(0, eq));
      if (eq == std::string::npos || !kind) usage(arg + " takes prose|mixed|code=ALPHA,BETA,GATE");
      std::optional<float> f[3];
      size_t at = eq + 1;
      for (int i = 0; i < 3; ++i) {
        const size_t comma = i < 2 ? v.find(',', at) : v.size();
        if (comma == std::string::npos) usage(arg + " takes prose|mixed|code=ALPHA,BETA,GATE");
        const std::string s = v.substr(at, comma - at);
        if (!s.empty()) {
          char* end = nullptr;
          const double x = std::strtod(s.c_str(), &end);
          if (*end) usage(arg + ": '" + s + "' is not a number");
          f[i] = static_cast<float>(x);
        }
        at = comma + 1;
      }
      a.lm_kinds[static_cast<size_t>(*kind)] = {f[0], f[1], f[2]};
    } else if (arg == "--second-pass-share" || arg == "--single-pixel") {
      const std::string v = value();
      char* end = nullptr;
      const double x = std::strtod(v.c_str(), &end);
      if (end == v.c_str() || *end || !(x >= 0 && x <= 1)) usage(arg + " takes a number from 0 to 1");
      (arg == "--single-pixel" ? a.single_pixel : a.second_pass_share) = x;
    } else if (arg == "--dpi") {
      const std::string v = value();
      a.dpi = v == "auto" ? 0 : to_int(arg, v, 36);  // auto: each PDF page's own
    } else if (arg == "--format") {
      a.format = value();
      if (a.format != "text" && a.format != "json" && a.format != "hocr" && a.format != "alto" &&
          a.format != "page" && a.format != "markdown" && a.format != "html" && a.format != "pdf") {
        usage("--format takes text, json, hocr, alto, page, markdown, html or pdf");
      }
    } else if (arg == "--output") {
      a.output = value();
    } else if (arg == "--text-out") {
      a.text_out = value();
    } else if (arg == "--json-html") {
      a.json_html = true;
    } else if (arg == "--json-markdown") {
      a.json_markdown = true;
    } else if (arg == "--text-only") {
      a.text_only = true;
    } else if (arg == "--image-dpi") {
      a.image_dpi = to_int(arg, value(), 36);
    } else if (arg == "--as-given") {
      a.as_given = true;
    } else if (arg == "--out-dir") {
      a.out_dir = value();
    } else if (arg == "--det") {
      a.det = value();
      if (a.det != "auto" && a.det != "one" && a.det != "wide" && a.det != "strips") {
        usage("--det takes auto, one, wide or strips");
      }
    } else if (a.command == "serve" && arg == "--host") {
      a.host = value();
    } else if (a.command == "serve" && arg == "--port") {
      a.port = to_int(arg, value(), 0);
      if (a.port > 65535) usage("--port takes 0 to 65535");
    } else if (a.command == "serve" && arg == "--queue") {
      a.queue = to_int(arg, value(), 1);
    } else if (a.command == "serve" && arg == "--max-body-mb") {
      a.max_body_mb = to_int(arg, value(), 1);
    } else if (a.command == "serve" && arg == "--decoders") {
      a.decoders = to_int(arg, value(), 1);
    } else if (a.command == "serve" && arg == "--quiet") {
      a.quiet = true;
    } else if (arg.starts_with("--")) {
      usage("unknown option " + arg);
    } else if (a.command == "ocr") {
      a.images.push_back(arg);
    } else if (a.command == "serve") {
      usage("serve takes no files: clients send their pages ('" + arg + "')");
    } else if (a.model.empty()) {
      a.model = arg;
    } else {
      usage("more than one model: '" + a.model + "' and '" + arg + "'");
    }
  }
  if (a.command == "ocr") {
    if (a.images.empty()) usage("ocr needs at least one image");
    if (a.format == "page" && a.out_dir.empty()) usage("--format page writes a file a page: give --out-dir DIR");
    if (a.format == "pdf" && a.output.empty()) usage("--format pdf writes a file: give --output FILE");
    if (a.set.empty()) usage("ocr needs --set DIR, a Fullerene set");
    return a;
  }
  if (a.command == "serve") {
    if (a.set.empty()) usage("serve needs --set DIR, a Fullerene set");
    return a;
  }
  if (a.model.empty()) usage(a.command + (a.command == "pages" ? " needs a model set's directory" : " needs a model file"));
  if (!a.workers.empty() && a.backend == carbon::Backend::kTensorRT) {
    usage("--workers is for the CPU (--backend ort or native); the GPU runs many pages in one process");
  }
  return a;
}

const carbon::Shape* given_shape(const Args& a, const std::string& name) {
  for (const auto& [n, s] : a.shapes) {
    if (n == name) return &s;
  }
  return nullptr;
}

std::unique_ptr<carbon::Session> open(const Args& a) {
  carbon::SessionOptions o;
  o.threads = a.threads;
  o.engine_cache = a.engine_cache;
  o.optimization_level = a.opt_level;
  o.cuda_graphs = a.graphs;
  o.isa = a.isa;
  if (!a.shapes.empty()) {
    o.profiles.emplace_back();
    for (const auto& [name, s] : a.shapes) o.profiles[0].push_back(carbon::ShapeRange::exact(name, s));
  }
  return carbon::open_session(a.backend, a.model, o);
}

void print_infos(const char* what, const std::vector<carbon::TensorInfo>& infos) {
  for (const auto& t : infos) {
    std::printf("  %-7s %-24s %-8s %s\n", what, t.name.c_str(),
                std::string(carbon::dtype_name(t.dtype)).c_str(),
                carbon::shape_string(t.shape).c_str());
  }
}

// A fixed, non-zero fill: some models take shortcuts on all-zero inputs.
carbon::Tensor fill(carbon::DType dtype, carbon::Shape shape) {
  carbon::Tensor t(dtype, std::move(shape));
  switch (dtype) {
    case carbon::DType::kFloat32:
      for (float& v : t.data<float>()) v = 0.5f;
      break;
    case carbon::DType::kInt32:
      for (int32_t& v : t.data<int32_t>()) v = 1;
      break;
    case carbon::DType::kInt64:
      for (int64_t& v : t.data<int64_t>()) v = 1;
      break;
    default:
      break;  // zeros
  }
  return t;
}

struct Times {
  double median, p90, min;
};

template <class F>
Times time_runs(const Args& a, F&& run) {
  for (int i = 0; i < a.warmup; ++i) run();
  std::vector<double> ms;
  for (int i = 0; i < a.runs; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    run();
    auto t1 = std::chrono::steady_clock::now();
    ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  auto at = [&](double q) { return ms[static_cast<size_t>(q * (ms.size() - 1))]; };
  return {at(0.5), at(0.9), ms.front()};
}

void print_times(const Args& a, const std::string& what, const Times& t) {
  std::printf("%s, %d runs after %d warm-up: median %.3f ms, p90 %.3f ms, min %.3f ms\n",
              what.c_str(), a.runs, a.warmup, t.median, t.p90, t.min);
}

#ifdef CARBON_WITH_TENSORRT
void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw carbon::Error(std::string(what) + ": " + cudaGetErrorString(e));
}

// The inputs uploaded once and the outputs left on the GPU: the model alone,
// as the GPU pipeline runs it.
void bench_device(const Args& a, carbon::Session& s, const std::vector<carbon::Tensor>& inputs) {
  cudaStream_t stream = nullptr;
  cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
  std::vector<void*> buffers;
  auto alloc = [&](size_t bytes) {
    void* p = nullptr;
    cuda_check(cudaMalloc(&p, bytes), "cudaMalloc");
    buffers.push_back(p);
    return p;
  };
  std::vector<carbon::DeviceTensor> in, out;
  std::vector<carbon::Shape> shapes;
  for (const auto& t : inputs) {
    void* p = alloc(t.bytes());
    cuda_check(cudaMemcpy(p, t.raw(), t.bytes(), cudaMemcpyHostToDevice), "copy to device");
    in.push_back({p, t.dtype(), t.shape()});
    shapes.push_back(t.shape());
  }
  const auto out_shapes = s.output_shapes(shapes);
  for (size_t i = 0; i < out_shapes.size(); ++i) {
    const auto dtype = s.outputs()[i].dtype;
    out.push_back({alloc(static_cast<size_t>(carbon::element_count(out_shapes[i])) *
                         carbon::dtype_size(dtype)),
                   dtype, out_shapes[i]});
  }
  const Times t = time_runs(a, [&] {
    s.enqueue(in, out, stream);
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
  });
  print_times(a, "trt on device buffers", t);
  for (void* p : buffers) cudaFree(p);
  cudaStreamDestroy(stream);
}
#endif

std::vector<carbon::Tensor> make_inputs(const Args& a, const carbon::Session& s) {
  std::vector<carbon::Tensor> inputs;
  for (const auto& info : s.inputs()) {
    carbon::Shape shape = info.shape;
    if (const carbon::Shape* g = given_shape(a, info.name)) {
      shape = *g;
    } else if (std::any_of(shape.begin(), shape.end(), [](int64_t d) { return d < 0; })) {
      usage("input '" + info.name + "' has free dimensions " + carbon::shape_string(shape) +
            ": give it --shape " + info.name + "=...");
    }
    inputs.push_back(fill(info.dtype, shape));
  }
  return inputs;
}

// The native backend against ONNX Runtime on one random input in [0, 1): for
// each output, the largest difference, against the largest value.
int compare(Args a) {
  const std::string mine = a.model;
  if (!a.against.empty()) a.model = a.against;
  a.backend = carbon::Backend::kOnnxRuntime;
  auto ort = open(a);
  a.model = mine;
  a.backend = carbon::Backend::kNative;
  auto native = open(a);
  std::vector<carbon::Tensor> inputs = make_inputs(a, *ort);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> u(0, 1);
  for (carbon::Tensor& t : inputs) {
    if (t.dtype() == carbon::DType::kFloat32) {
      for (float& v : t.data<float>()) v = u(rng);
    }
  }
  const auto want = ort->run(inputs), got = native->run(inputs);
  std::printf("%s: native (%s) against ort%s%s\n", a.model.c_str(),
              a.isa.empty() ? carbon::cpu::isa_name(carbon::cpu::best_isa()) : a.isa.c_str(),
              a.against.empty() ? "" : " on ", a.against.c_str());
  for (size_t k = 0; k < want.size(); ++k) {
    if (want[k].shape() != got[k].shape()) {
      std::printf("  %-12s shapes differ: %s against %s\n", ort->outputs()[k].name.c_str(),
                  carbon::shape_string(got[k].shape()).c_str(), carbon::shape_string(want[k].shape()).c_str());
      return 1;
    }
    const auto w = want[k].data<float>(), g = got[k].data<float>();
    double diff = 0, mag = 0;
    for (size_t i = 0; i < w.size(); ++i) {
      diff = std::max(diff, static_cast<double>(std::abs(w[i] - g[i])));
      mag = std::max(mag, static_cast<double>(std::abs(w[i])));
    }
    std::printf("  %-12s %-18s largest difference %.3g, largest value %.3g (%.2g of it)\n",
                ort->outputs()[k].name.c_str(), carbon::shape_string(want[k].shape()).c_str(), diff, mag,
                mag > 0 ? diff / mag : 0.0);
  }
  return 0;
}

// The GEMMs' multiply-accumulates a second on one thread, for each instruction
// set, at layers of det (a 1024 x 800 page) and rec-latin (8 lines of 512 px) as
// convolutions read in place: m output positions, k = channels in x taps, n
// channels out.  int8 to uint8 (scaled, bias, relu), FP32 with bias and relu.
void bench_gemm() {
  namespace cpu = carbon::cpu;
  struct Shape {
    const char* layer;
    int m, taps, cin, n;
  };
  const Shape shapes[] = {{"det stem 3x3", 204800, 9, 1, 16},     {"det stride 4 3x3", 51200, 9, 24, 24},
                          {"det FPN 3x3", 51200, 9, 32, 32},     {"det maps 1x1", 204800, 1, 16, 4},
                          {"rec stage 1 3x3", 8192, 9, 48, 48},   {"rec stage 2 3x3", 2048, 9, 96, 96},
                          {"rec stage 3 3x3", 1024, 9, 160, 160}, {"rec mixing 1x1", 1024, 1, 256, 512},
                          {"rec CTC head", 1024, 1, 256, 800}};
  std::vector<cpu::Isa> isas;  // the scalar references are for tests, not for timing
  for (cpu::Isa isa : cpu::isas_available()) {
    if (isa != cpu::Isa::kScalar) isas.push_back(isa);
  }
  std::mt19937 rng(1);
  // The best of 5 rounds of 60 ms each, in GMAC/s.
  auto time = [](double macs, auto&& run) {
    run();
    double best = 0;
    for (int round = 0; round < 5; ++round) {
      int reps = 0;
      const auto t0 = std::chrono::steady_clock::now();
      double secs = 0;
      do {
        run();
        ++reps;
        secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      } while (secs < 0.06);
      best = std::max(best, macs * reps / secs / 1e9);
    }
    return best;
  };
  for (const bool int8 : {true, false}) {
    std::printf("%s GEMM, one thread, GMAC/s\n  %-18s  %-22s", int8 ? "int8" : "FP32", "layer", "m x k x n");
    for (cpu::Isa isa : isas) std::printf("  %9s", cpu::isa_name(isa));
    std::printf("\n");
    for (const Shape& s : shapes) {
      const int m = s.m, k = s.taps * s.cin, n = s.n;
      const std::string dims = std::to_string(m) + " x " + std::to_string(s.taps) + "*" + std::to_string(s.cin) +
                               " x " + std::to_string(n);
      std::printf("  %-18s  %-22s", s.layer, dims.c_str());
      // The rows' pieces: a pixel's channels, cin apart, as a convolution's taps
      // read them from its input.
      const int64_t pixels = m + 64;
      std::vector<float> scale(n, 1e-3f), bias(n, 0.5f);
      std::vector<const uint8_t*> ptrs8(static_cast<size_t>(m) * s.taps);
      std::vector<const float*> ptrs32(ptrs8.size());
      std::vector<uint8_t> in8(static_cast<size_t>(pixels) * s.cin + 64), out8(static_cast<size_t>(m) * n);
      std::vector<float> in32(in8.size()), out32(out8.size());
      for (auto& v : in8) v = static_cast<uint8_t>(rng());
      for (auto& v : in32) v = static_cast<float>(rng() % 1000) / 1000.0f;
      for (int i = 0; i < m; ++i) {
        for (int t = 0; t < s.taps; ++t) {
          const int64_t px = (i + t % 3 + t / 3 * 8) % pixels;
          ptrs8[static_cast<size_t>(i) * s.taps + t] = in8.data() + px * s.cin;
          ptrs32[static_cast<size_t>(i) * s.taps + t] = in32.data() + px * s.cin;
        }
      }
      cpu::Epilogue e;
      e.bias = bias.data();
      e.relu = true;
      e.out_ld = n;
      if (int8) {
        e.scale = scale.data();
        e.out_u8 = out8.data();
        e.out_inv_scale = 8.0f;
      } else {
        e.out_f32 = out32.data();
      }
      for (cpu::Isa isa : isas) {
        double gmacs;
        if (int8) {
          std::vector<int8_t> b(static_cast<size_t>(k) * n);
          for (auto& v : b) v = static_cast<int8_t>(rng());
          const cpu::PackedB pb = cpu::pack_b(b.data(), k, n, isa, s.taps);
          gmacs = time(static_cast<double>(m) * k * n, [&] { cpu::gemm_u8s8({ptrs8.data(), s.taps}, 128, pb, m, e); });
        } else {
          std::vector<float> b(static_cast<size_t>(k) * n);
          for (auto& v : b) v = static_cast<float>(static_cast<int>(rng() % 200) - 100) / 1000.0f;
          const cpu::PackedF32 pb = cpu::pack_f32(b.data(), k, n, isa);
          gmacs = time(static_cast<double>(m) * k * n, [&] { cpu::gemm_f32({ptrs32.data(), s.taps}, pb, m, e); });
        }
        std::printf("  %9.1f", gmacs);
      }
      std::printf("\n");
    }
  }
  // The depthwise convolutions: det's head (16 channels at stride 2) and
  // rec-latin's mixing blocks (256 channels, 1x9).
  const Shape dw[] = {{"det head dw 3x3", 204800, 9, 16, 16}, {"rec mixing dw 1x9", 1024, 9, 256, 256}};
  for (const bool int8 : {true, false}) {
    std::printf("%s depthwise, one thread, GMAC/s\n  %-18s  %-22s", int8 ? "int8" : "FP32", "layer", "m x taps x c");
    for (cpu::Isa isa : isas) std::printf("  %9s", cpu::isa_name(isa));
    std::printf("\n");
    for (const Shape& s : dw) {
      const int m = s.m, c = s.cin;
      std::printf("  %-18s  %-22s", s.layer,
                  (std::to_string(m) + " x " + std::to_string(s.taps) + " x " + std::to_string(c)).c_str());
      const int64_t pixels = m + 64;
      std::vector<float> bias(c, 0.5f), scale(c, 1e-3f);
      std::vector<uint8_t> in8(static_cast<size_t>(pixels) * c + 64), out8(static_cast<size_t>(m) * c);
      std::vector<float> in32(in8.size()), out32(out8.size());
      std::vector<int8_t> w8(static_cast<size_t>(s.taps) * c);
      std::vector<float> w32(w8.size());
      for (auto& v : in8) v = static_cast<uint8_t>(rng());
      for (auto& v : in32) v = static_cast<float>(rng() % 1000) / 1000.0f;
      for (auto& v : w8) v = static_cast<int8_t>(rng());
      for (auto& v : w32) v = static_cast<float>(static_cast<int>(rng() % 200) - 100) / 1000.0f;
      std::vector<const uint8_t*> ptrs8(static_cast<size_t>(m) * s.taps);
      std::vector<const float*> ptrs32(ptrs8.size());
      for (int i = 0; i < m; ++i) {
        for (int t = 0; t < s.taps; ++t) {
          const int64_t px = (i + t % 3 + t / 3 * 8) % pixels;
          ptrs8[static_cast<size_t>(i) * s.taps + t] = in8.data() + px * c;
          ptrs32[static_cast<size_t>(i) * s.taps + t] = in32.data() + px * c;
        }
      }
      cpu::Epilogue e;
      e.bias = bias.data();
      e.relu = true;
      e.out_ld = c;
      if (int8) {
        e.scale = scale.data();
        e.out_u8 = out8.data();
        e.out_inv_scale = 8.0f;
      } else {
        e.out_f32 = out32.data();
      }
      for (cpu::Isa isa : isas) {
        const double macs = static_cast<double>(m) * s.taps * c;
        const double gmacs =
            int8 ? time(macs, [&] { cpu::dwconv_u8s8({ptrs8.data(), s.taps}, 128, w8.data(), c, m, e, isa); })
                 : time(macs, [&] { cpu::dwconv_f32({ptrs32.data(), s.taps}, w32.data(), c, m, e, isa); });
        std::printf("  %9.1f", gmacs);
      }
      std::printf("\n");
    }
  }
}

int cpus(const Args& a) {
  const carbon::cpu::Features& f = carbon::cpu::features();
  std::string sets;
  for (carbon::cpu::Isa isa : carbon::cpu::isas_available()) {
    sets += (sets.empty() ? "" : ", ") + std::string(carbon::cpu::isa_name(isa));
  }
  std::printf("%s%s%s\n", f.vendor.c_str(), f.brand.empty() ? "" : ": ", f.brand.c_str());
  std::printf("kernels: %s (%s in use)\n", sets.c_str(), carbon::cpu::isa_name(carbon::cpu::best_isa()));
  std::printf("int8 products: %s\n",
              carbon::cpu::int8_widens()
                  ? "widened to 16 bits, no dot product (INT8 no faster than FP32 an instruction: one worker reads a "
                    "page sooner with --precision fp32, many workers more pages in int8, the default)"
                  : "a dot product (INT8 the faster at one worker and at many)");
  const carbon::CpuTopology& t = carbon::cpu_topology();
  std::printf("%zu physical cores, %d logical processors, %d processor group%s, %d NUMA node%s; ",
              t.cores.size(), t.logical, t.groups, t.groups == 1 ? "" : "s", t.numa_nodes,
              t.numa_nodes == 1 ? "" : "s");
  if (t.cpu_quota > 0) {
    std::printf("a CPU quota of %d cores\n", t.cpu_quota);
  } else {
    std::printf("no CPU quota\n");
  }
  std::printf("%d workers by default, one per physical core, in this order:\n", t.default_workers());
  std::printf("  worker  group  node  class  logical processors\n");
  for (size_t i = 0; i < t.cores.size(); ++i) {
    const carbon::CpuCore& c = t.cores[i];
    std::string list;
    for (int cpu : c.cpus) list += (list.empty() ? "" : " ") + std::to_string(cpu);
    std::printf("  %6zu  %5d  %4d  %5d  %s\n", i, c.group, c.numa_node, c.efficiency, list.c_str());
  }
  if (a.gemm) bench_gemm();
  return 0;
}

// For each worker count: a pool pinned one worker per core, each worker its
// own session and input (made on its own thread, so in its own NUMA node's
// memory).  Every run is a task of its own, workers x runs of them, so a fast
// core takes more of them than a slow one (P- and E-cores), as the pipeline's
// tasks do.
int bench_workers(const Args& a) {
  std::printf("%s on %s, %d thread%s a session, %d runs a worker after %d warm-up%s\n", a.model.c_str(),
              std::string(carbon::backend_name(a.backend)).c_str(), a.threads, a.threads == 1 ? "" : "s", a.runs,
              a.warmup, a.pin ? ", pinned" : "");
  std::printf("  workers     runs/s  per worker  median ms   scaling  runs by worker\n");
  double base = 0;  // the first count's runs/s per worker
  for (int n : a.workers) {
    carbon::WorkerPool pool({n, a.pin});
    std::vector<std::unique_ptr<carbon::Session>> sessions(n);
    std::vector<std::vector<carbon::Tensor>> inputs(n);
    std::vector<std::vector<double>> ms(n);
    pool.on_each_worker([&](int w) {
      sessions[w] = open(a);
      inputs[w] = make_inputs(a, *sessions[w]);
      for (int i = 0; i < a.warmup; ++i) sessions[w]->run(inputs[w]);
    });
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n * a.runs; ++i) {
      pool.submit([&](int w) {
        const auto r0 = std::chrono::steady_clock::now();
        sessions[w]->run(inputs[w]);
        const auto r1 = std::chrono::steady_clock::now();
        ms[w].push_back(std::chrono::duration<double, std::milli>(r1 - r0).count());
      });
    }
    pool.wait();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double rate = static_cast<double>(n) * a.runs / secs;
    if (base == 0) base = rate / n;
    std::vector<double> all;
    size_t fewest = ms[0].size(), most = 0;
    for (const auto& v : ms) {
      all.insert(all.end(), v.begin(), v.end());
      fewest = std::min(fewest, v.size());
      most = std::max(most, v.size());
    }
    std::sort(all.begin(), all.end());
    std::printf("  %7d  %9.1f  %10.2f  %9.3f  %7.1f%%  %zu to %zu%s\n", n, rate, rate / n, all[all.size() / 2],
                100.0 * rate / (n * base), fewest, most,
                !a.pin || pool.pinned() == n ? "" : "  (not every worker pinned)");
    pool.on_each_worker([&](int w) {
      sessions[w].reset();
      inputs[w].clear();
    });
  }
  return 0;
}

std::vector<carbon::Tensor> one_input(carbon::Shape shape) {
  std::vector<carbon::Tensor> v;
  v.push_back(fill(carbon::DType::kFloat32, std::move(shape)));
  return v;
}

double percentile(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  return v[static_cast<size_t>(q * (v.size() - 1))];
}

// A worker's sessions of one model set, and its inputs, made on its own thread:
// crops for each number of lines a task has given it.
struct SetWorker {
  std::unique_ptr<carbon::Session> orient, det, script, rec, order;
  std::vector<carbon::Tensor> thumb, page, features;
  std::vector<float> maps;  // a rendered page's detector maps, for the decoder
  std::map<int, std::vector<carbon::Tensor>> crops;

  const std::vector<carbon::Tensor>& lines(int k) {
    auto it = crops.find(k);
    if (it == crops.end()) it = crops.emplace(k, one_input({k, 1, 32, 512})).first;
    return it->second;
  }
};

// A page is planned for the cores that are free when it gets to them; with
// every core busy (a large job) the plan is the plain one: orientation, then
// detection, on one worker, the lines in tasks of --batch.
// - Orientation runs as a task of its own beside detection when a fast worker
//   is idle.  The page is taken as upright, as almost every page is; a turned
//   page detects again.
// - Detection, with fast workers idle, runs on a multi-threaded session (2, 4
//   or 6 threads, the widest the idle cores fit) or, from 8 idle, in strips, a
//   task each (where strips beat 6 threads); else on the worker's single-
//   threaded session.  The same bytes every way (--det one, wide or strips
//   forces one; auto chooses).
// - The lines split into a task for this worker and each idle fast one, or
//   tasks of --batch lines when none is idle: a recognizer reads each line on
//   its own, so the split changes no output.
int pages(const Args& a) {
  const std::filesystem::path dir = a.model;
  std::vector<int> counts = a.workers;
  if (counts.empty()) counts = {1, carbon::cpu_topology().default_workers()};
  const int fewest = (a.lines + a.batch - 1) / a.batch;  // line tasks with every core busy
  // the script classifier where the set has one: the engine reads a set without it
  const bool script = std::filesystem::is_regular_file(dir / "script.onnx") ||
                      std::filesystem::is_regular_file(dir / "script.int8.onnx");
  std::printf("%s: a page is orient and det, %d lines in %d to %d tasks (%s%s), then order; "
              "%d pages, %s\n", dir.string().c_str(), a.lines, fewest, a.lines, script ? "script, " : "",
              a.rec.c_str(), a.pages,
              a.in_flight ? ("at most " + std::to_string(a.in_flight) + " in flight").c_str() : "all queued");
  std::vector<std::string> names = {"orient", "det", a.rec};
  if (script) names.insert(names.begin() + 2, "script");
  std::string models;
  for (const std::string& name : names) {
    const carbon::SetModel m =
        carbon::choose_set_model(dir, name, a.backend_given ? std::optional(a.backend) : std::nullopt, a.precision);
    models += (models.empty() ? "" : ", ") + m.file.filename().string() + " on " +
              std::string(carbon::backend_name(m.backend));
  }
  std::printf("  %s, order.onnx on ort\n", models.c_str());
  std::printf("  workers    pages/s  per worker  ms a page (median, p90)   scaling  line tasks  det threads"
              "  det strips  to the lines  the lines\n");
  constexpr int kPageRows = 1024, kMostStrips = 16;
  std::vector<std::vector<carbon::Strip>> plans(kMostStrips + 1);
  for (int k = 1; k <= kMostStrips; ++k) plans[k] = carbon::plan_strips(kPageRows, k);
  const bool may_strip = a.det == "auto" || a.det == "strips";
  bool checked = false;
  using clock = std::chrono::steady_clock;
  double base = 0;
  for (int n : counts) {
    carbon::WorkerPool pool({n, a.pin});
    std::vector<SetWorker> m(n);
    pool.on_each_worker([&](int w) {
      carbon::SessionOptions o;
      o.threads = a.threads;
      auto load = [&](const std::string& name) {
        return carbon::open_set_model(dir, name, a.backend_given ? std::optional(a.backend) : std::nullopt, o,
                                      a.precision);
      };
      SetWorker& s = m[w];
      s.orient = load("orient");
      s.det = load("det");
      if (script) s.script = load("script");
      s.rec = load(a.rec);
      s.order = carbon::open_session(carbon::Backend::kOnnxRuntime, dir / "order.onnx", o);  // a transformer
      s.thumb = one_input({1, 1, 512, 512});
      s.page = one_input({1, 1, kPageRows, 800});
      s.features = one_input({1, a.lines, 16});
      // The maps of a page of a.lines lines, evenly spaced, as logits.
      std::vector<carbon::LineBox> boxes;
      const double pitch = 980.0 / a.lines;
      for (int i = 0; i < a.lines; ++i) boxes.push_back({40, 20 + pitch * i, 760, 20 + pitch * (i + 0.66)});
      const size_t plane = static_cast<size_t>(kPageRows / 2) * 400;
      s.maps.resize(4 * plane);
      carbon::det_targets(boxes, {}, kPageRows / 2, 400, s.maps.data());
      for (size_t k = 0; k < plane; ++k) {
        s.maps[k] = s.maps[k] > 0.5f ? 8.0f : -8.0f;
        s.maps[3 * plane + k] = -8.0f;
      }
      if (carbon::decode_det(s.maps.data(), kPageRows / 2, 400).lines.size() != static_cast<size_t>(a.lines)) {
        throw carbon::Error("the rendered page does not decode to its lines");
      }
      // one warm-up page, and every task size once
      s.orient->run(s.thumb);
      s.det->run(s.page);
      for (int k = 1; k <= std::min(a.batch, a.lines); ++k) {
        if (s.script) s.script->run(s.lines(k));
        s.rec->run(s.lines(k));
      }
      s.order->run(s.features);
      if (may_strip) {  // every strip height once
        std::set<int> heights;
        for (const auto& plan : plans) {
          for (const carbon::Strip& st : plan) heights.insert(st.y1 - st.y0);
        }
        for (int h : heights) s.det->run(one_input({1, 1, h, 800}));
      }
    });
    if (may_strip && !checked) {  // strips give the whole page's bytes, here too
      checked = true;
      std::vector<carbon::Tensor> page;
      page.push_back(m[0].page.front().clone());
      const carbon::Tensor full = std::move(m[0].det->run(page)[0]);
      for (int k : {4, 8, 16}) {
        carbon::Tensor out(full.dtype(), full.shape());
        for (const carbon::Strip& st : plans[k]) {
          std::vector<carbon::Tensor> in;
          in.push_back(carbon::strip_input(page.front(), st));
          carbon::stitch_strip(m[0].det->run(in)[0], st, out);
        }
        if (std::memcmp(out.raw(), full.raw(), full.bytes()) != 0) {
          std::printf("  %d strips do NOT give the whole page's bytes\n", k);
          return 1;
        }
      }
      std::printf("  4, 8 and 16 strips give the whole page's maps, byte for byte\n");
    }
    // Multi-threaded detectors, widest first, for a page that finds fast cores idle.
    struct Wide {
      int threads = 0;
      std::unique_ptr<carbon::Session> session;
      std::mutex m;
    };
    std::vector<std::unique_ptr<Wide>> wides;
    // det.onnx on ONNX Runtime: FP32, so only where the workers run det.onnx too, as the engine
    // (detail::wide_detector_allowed): at auto a set that ships det.int8.onnx runs that on every worker
    const bool fp32_det = (a.precision == carbon::Precision::kAuto || a.precision == carbon::Precision::kFp32) &&
                          carbon::choose_set_model(dir, "det", a.backend_given ? std::optional(a.backend) : std::nullopt,
                                                   a.precision).file == dir / "det.onnx";
    if ((a.det == "auto" || a.det == "wide") && fp32_det) {
      for (int t : {6, 4, 2}) {
        if (t > pool.fast()) continue;
        auto wd = std::make_unique<Wide>();
        wd->threads = t;
        carbon::SessionOptions o;
        o.threads = t;
        wd->session = carbon::open_session(carbon::Backend::kOnnxRuntime, dir / "det.onnx", o);
        wd->session->run(m[0].page);
        wides.push_back(std::move(wd));
      }
    }
    struct Page {
      std::atomic<int> before{0}, left{0}, strips_left{0};  // orientation and detection; line tasks; strips
      clock::time_point start;
      double ms = 0, to_lines = 0;
      int tasks = 0, det_threads = 1, det_strips = 1;
      carbon::Tensor maps;  // the stitched detector output
    };
    std::vector<Page> pg(a.pages);
    std::atomic<int> next{0};
    std::function<void(int)> start_page;
    auto finish = [&](int i, int v) {
      m[v].order->run(m[v].features);
      pg[i].ms = std::chrono::duration<double, std::milli>(clock::now() - pg[i].start).count();
      if (a.in_flight > 0) {
        const int k = next.fetch_add(1);
        if (k < a.pages) start_page(k);
      }
    };
    // The fast cores free for a page's next step: the idle fast workers, and this
    // one if it is fast (a page that ends up on a slow core hands its work on).
    auto free_fast = [&](int w) { return pool.idle() + (pool.is_fast(w) ? 1 : 0); };
    auto split_lines = [&](int i, int w) {
      pg[i].to_lines = std::chrono::duration<double, std::milli>(clock::now() - pg[i].start).count();
      // A task for each free fast core; --batch lines a task when none is idle.
      const int idle = pool.idle();
      const int tasks = std::min(a.lines, idle > 0 ? std::max(1, free_fast(w)) : fewest);
      pg[i].tasks = tasks;
      pg[i].left.store(tasks);
      for (int t = 0; t < tasks; ++t) {
        const int k = a.lines / tasks + (t < a.lines % tasks ? 1 : 0);
        pool.submit([&, i, k](int v) {
          const auto& x = m[v].lines(k);
          if (m[v].script) m[v].script->run(x);
          m[v].rec->run(x);
          if (pg[i].left.fetch_sub(1) == 1) finish(i, v);
        });
      }
    };
    auto ready = [&](int i, int w) {
      if (pg[i].before.fetch_sub(1) == 1) split_lines(i, w);
    };
    // Detection done: the line decoder, on the worker that finished it.
    auto detected = [&](int i, int w) {
      const carbon::Detection d = carbon::decode_det(m[w].maps.data(), kPageRows / 2, 400, 2200.0 / kPageRows);
      if (d.lines.empty()) throw carbon::Error("no lines");
      ready(i, w);
    };
    start_page = [&](int i) {
      pool.submit([&, i](int w) {
        pg[i].start = clock::now();
        const bool apart = pool.idle() > 0;
        pg[i].before.store(apart ? 2 : 1);
        if (apart) {
          pool.submit([&, i](int v) {
            m[v].orient->run(m[v].thumb);
            ready(i, v);
          });
        } else {
          m[w].orient->run(m[w].thumb);
        }
        // The fast cores free for detection: threads share them with orientation's
        // 4 ms; strips, a task each, leave orientation a core of its own.
        const int idle = pool.idle();
        const int cores = std::max(1, free_fast(w));
        const int strips = std::min(std::max(1, cores - (apart ? 1 : 0)), kMostStrips);
        if (may_strip && idle > 0 && (a.det == "strips" || strips >= 8)) {
          pg[i].det_strips = strips;
          pg[i].maps = carbon::Tensor::empty(carbon::DType::kFloat32, {1, 4, kPageRows / 2, 400});
          pg[i].strips_left.store(strips);
          for (int k = 1; k < strips; ++k) {
            pool.submit([&, i, k, strips](int v) {
              const carbon::Strip& st = plans[strips][k];
              std::vector<carbon::Tensor> in;
              in.push_back(carbon::strip_input(m[v].page.front(), st));
              carbon::stitch_strip(m[v].det->run(in)[0], st, pg[i].maps);
              if (pg[i].strips_left.fetch_sub(1) == 1) detected(i, v);
            });
          }
          const carbon::Strip& st = plans[strips][0];
          std::vector<carbon::Tensor> in;
          in.push_back(carbon::strip_input(m[w].page.front(), st));
          carbon::stitch_strip(m[w].det->run(in)[0], st, pg[i].maps);
          if (pg[i].strips_left.fetch_sub(1) == 1) detected(i, w);
          return;
        }
        bool on_threads = false;
        for (auto& wd : wides) {  // the widest that fits the idle cores and is free
          if (wd->threads > cores) continue;
          std::unique_lock lk(wd->m, std::try_to_lock);
          if (!lk) continue;
          pg[i].det_threads = wd->threads;
          wd->session->run(m[w].page);
          on_threads = true;
          break;
        }
        if (!on_threads) m[w].det->run(m[w].page);
        detected(i, w);
      });
    };
    const int first = a.in_flight > 0 ? std::min(a.in_flight, a.pages) : a.pages;
    next.store(first);
    const auto t0 = clock::now();
    for (int i = 0; i < first; ++i) start_page(i);
    pool.wait();
    const double secs = std::chrono::duration<double>(clock::now() - t0).count();
    const double rate = a.pages / secs;
    if (base == 0) base = rate / n;
    std::vector<double> ms, tasks, threads, strips, to_lines, lines;
    for (const Page& p : pg) {
      ms.push_back(p.ms);
      tasks.push_back(p.tasks);
      threads.push_back(p.det_threads);
      strips.push_back(p.det_strips);
      to_lines.push_back(p.to_lines);
      lines.push_back(p.ms - p.to_lines);
    }
    std::printf("  %7d  %9.2f  %10.2f  %10.1f  %10.1f  %10.1f%%  %10.0f  %11.0f  %10.0f  %12.1f  %9.1f%s\n", n,
                rate, rate / n, percentile(ms, 0.5), percentile(ms, 0.9), 100.0 * rate / (n * base),
                percentile(tasks, 0.5), percentile(threads, 0.5), percentile(strips, 0.5), percentile(to_lines, 0.5),
                percentile(lines, 0.5), !a.pin || pool.pinned() == n ? "" : "  (not every worker pinned)");
    pool.on_each_worker([&](int w) { m[w] = SetWorker{}; });
  }
  return 0;
}

std::string quoted(const std::string& v) {
  std::string q = "\"";
  for (char c : v) {
    if (c == '"' || c == '\\') q += '\\';
    q += c;
  }
  return q + "\"";
}

// A JSON string: quotes, backslashes and control characters escaped (html and markdown hold newlines).
std::string json_text(const std::string& v) {
  std::string q = "\"";
  for (const char ch : v) {
    const unsigned char c = static_cast<unsigned char>(ch);
    if (c == '"' || c == '\\') {
      q += '\\';
      q += ch;
    } else if (c == '\n') {
      q += "\\n";
    } else if (c < 0x20) {
      static constexpr char kHex[] = "0123456789abcdef";
      q += "\\u00";
      q += kHex[c >> 4];
      q += kHex[c & 15];
    } else {
      q += ch;
    }
  }
  return q + "\"";
}

// The engine's options as ocr and serve take them.
carbon::EngineOptions engine_options(const Args& a) {
  carbon::EngineOptions o;
  o.rec = a.rec;
  if (a.backend_given) o.backend = a.backend;
  o.precision = a.precision;
  o.pool.workers = a.workers.empty() ? 0 : a.workers.front();
  o.pool.pin = a.pin;
  if (a.second_pass == "off") o.det.second_pass_below = 0;  // one detection pass, whatever the lines' size
  if (a.gate == "off") o.gate.reset();                       // every line kept, text or not
  o.blocks_by_geometry = a.block_order == "geometry";
  o.table_spans = a.table_spans != "off";
  if (a.grid == "off") o.grid.reset();
  o.order.blocks_first = a.blocks_first == "on";
  o.turn_by_quarters = a.turn == "quarters";
  if (a.turn_check == "off") o.turn_check.reset();  // orientation's turn alone
  if (a.turn == "bin" && o.turn_check) o.turn_check->sum_bins = false;  // auto: quarters where the turn is read
  if (a.max_skew) o.max_skew = *a.max_skew;
  if (a.wide_skew_gain) o.wide_skew_gain = *a.wide_skew_gain;
  if (a.spaces == "ink") o.spaces = carbon::SpaceRule{};  // unsure spaces the ink does not bear out dropped
  o.beam.width = a.beam;  // under 2: off
  o.beam.topk = a.beam_topk;
  o.beam.gate = a.beam_gate;
  o.beam.span = a.beam_span;
  o.lm_kinds = a.lm_kinds;
  o.lm = a.lm;
  o.lm_alpha = a.lm_alpha;
  o.lm_beta = a.lm_beta;
  o.tlm = a.tlm;
  o.tlm_alpha = a.tlm_alpha;
  o.tlm_gpu = a.tlm_gpu;
  o.lm_mode = *carbon::parse_lm_mode(a.lm_mode);
  o.lm_gate = a.lm_gate;
  if (a.second_pass_share) o.det.second_pass_share = *a.second_pass_share;
  if (a.single_pixel) o.det.single_pixel = static_cast<float>(*a.single_pixel);
  if (a.as_given) {  // the page as given (OCRmyPDF turns and deskews pages itself, and grafts the text back)
    o.turn_probability = 2.0f;
    o.deskew = false;
  }
  return o;
}

// The document stream's options as ocr and serve take them (carbon/document.h).
carbon::DocumentOptions document_options(const Args& a) {
  carbon::DocumentOptions d;
  d.engine = engine_options(a);
  d.cuda = a.device == "cuda";
  d.pipeline.page_batch = a.batch;
  if (a.in_flight) d.pipeline.in_flight = a.in_flight;
  d.pipeline.engine_cache = a.engine_cache;
  d.pipeline.head_fp16_accumulate = a.head_accumulate == "fp16";
  d.text_layer = a.text_layer == "auto";
  d.pictures = a.hybrid == "off"      ? carbon::PictureRead::kOff
               : a.hybrid == "detect" ? carbon::PictureRead::kDetect
                                      : carbon::PictureRead::kFound;
  d.layer_lines = a.layer_lines == "detected";
  d.text_on_gpu = a.text_pages == "gpu";
  return d;
}

// The pages written in their order as they come: each page as soon as it and every page before it are out,
// in the format asked; hOCR, ALTO and the searchable PDF, documents written whole, keep the pages until the last.
// The same bytes as writing every page at the end.
class PageWriter {
 public:
  PageWriter(const Args& a, const std::vector<std::pair<std::string, int>>& sources) : a_(a), sources_(sources) {
    for (const auto& [file, number] : sources) names_.push_back(number ? file + "#page=" + std::to_string(number) : file);
    whole_ = a.format == "hocr" || a.format == "alto" || a.format == "pdf";
    // --output: the document formats' file (JSON and text go to standard output)
    if (!a.output.empty() && a.format != "json" && a.format != "text" && a.format != "page") {
      out_.open(std::filesystem::path(a.output), std::ios::binary);
    }
    if (!a.text_out.empty()) text_.open(std::filesystem::path(a.text_out), std::ios::binary);
    if (a.format == "page") std::filesystem::create_directories(a.out_dir);
    if (a.format == "json") put("{\"pages\": [");
    if (a.format == "html") put(carbon::html_head());
  }

  // Page `written()`, the next in order.
  void page(carbon::OcrPage p) {
    const size_t i = n_++;
    if (text_.is_open()) {  // the pages' text beside the format asked for, a form feed between pages
      const std::string t = (i ? "\f\n" : "") + p.text();
      text_.write(t.data(), static_cast<std::streamsize>(t.size()));
    }
    const auto& [file, number] = sources_[i];
    if (a_.format == "json") {
      std::string page = p.json();  // {"width": ...}: the file goes first
      // Its html body (as the server's html=1) and markdown, for a bench of the CLI's own path
      if (a_.json_html) page.insert(page.size() - 1, ", \"html\": " + json_text(carbon::to_html_body(p)));
      if (a_.json_markdown) {
        page.insert(page.size() - 1, ", \"markdown\": " + json_text(carbon::to_markdown(std::span(&p, 1))));
      }
      put(std::string(i ? ", " : "") + "{\"file\": " + quoted(file) + ", " +
          (number ? "\"page\": " + std::to_string(number) + ", " : std::string()) + (page.c_str() + 1));
    } else if (a_.format == "text") {
      put((i ? "\f\n" : "") + p.text());
    } else if (a_.format == "markdown") {
      put(carbon::markdown_page(p, i));
    } else if (a_.format == "html") {
      put(carbon::html_page(p, i));
    } else if (a_.format == "page") {
      std::string name = std::filesystem::path(file).stem().string();
      if (number) name += "-" + std::to_string(number);
      const std::string x = carbon::to_page_xml(p, names_[i]);
      std::ofstream(std::filesystem::path(a_.out_dir) / (name + ".xml"), std::ios::binary)
          .write(x.data(), static_cast<std::streamsize>(x.size()));
    } else {
      pages_.push_back(std::move(p));
    }
  }
  size_t written() const { return n_; }

  // Every page in: the document's end, or the documents written whole (pdf: each page's image at the dpi it was
  // read at, a PDF's --dpi, an image's --image-dpi).
  void finish(const std::vector<std::shared_ptr<const carbon::GrayImage>>& images, const std::vector<double>& dpis) {
    if (a_.format == "json") put("]}\n");
    if (a_.format == "html") put(carbon::html_tail());
    if (whole_) {
      if (a_.format == "pdf") {
        std::vector<carbon::GrayView> views;
        for (const auto& img : images) views.push_back(img->view());
        put(carbon::to_pdf(pages_, views, dpis, a_.text_only));
      } else {
        put(a_.format == "hocr" ? carbon::to_hocr(pages_, names_) : carbon::to_alto(pages_, names_));
      }
    }
    if (out_.is_open()) out_.flush();
    std::fflush(stdout);
  }

 private:
  void put(const std::string& s) {
    if (out_.is_open()) {
      out_.write(s.data(), static_cast<std::streamsize>(s.size()));
    } else {
      std::fwrite(s.data(), 1, s.size(), stdout);
    }
  }
  const Args& a_;
  const std::vector<std::pair<std::string, int>>& sources_;
  std::vector<std::string> names_;  // each page's image, as the formats name it: its file, a PDF's page file#page=N
  bool whole_ = false;
  std::ofstream out_, text_;
  std::vector<carbon::OcrPage> pages_;  // the documents written whole
  size_t n_ = 0;
};

int ocr(const Args& a) {
  // The pages stream. Producer threads decode the images and render the PDFs' pages (each page's image,
  // text layer and pictures from one request) and give them to the document stream (carbon/document.h), which
  // reads each page whole, its text layer's steps among them, while this thread takes them as they finish and
  // writes each in its order; so rendering, reading and writing run side by side, and only the pages in flight are
  // held (the pdf format, whose writer draws every page, keeps their images). The models are loaded before the
  // clock starts; on the GPU no CPU engine is made unless its text layers' pages need one.
  const carbon::DocumentOptions d = document_options(a);
  const bool cuda = d.cuda;
  carbon::DocumentStream stream(a.set, d);
  const int workers = cuda ? std::max(1, carbon::cpu_topology().default_workers()) : stream.workers();
  // Each image a page; each page of a PDF rendered at --dpi (carbon/pdf.h) in the
  // helper processes (carbon-pdf, beside this program), many at once.
  std::vector<std::pair<std::string, int>> sources;  // a page's file, and its page in a PDF (from 1; 0: an image)
  std::unique_ptr<carbon::PdfPool> pdf_pool;
  bool any_pdf = false;
  for (const std::string& f : a.images) {
    // a PDF by its first bytes (carbon::file_kind), whatever its name; by its name only where they say nothing
    std::string ext = std::filesystem::path(f).extension().string();
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const carbon::FileKind kind = carbon::file_kind(std::filesystem::path(f));
    if (kind != carbon::FileKind::kPdf && !(kind == carbon::FileKind::kUnknown && ext == ".pdf")) {
      sources.emplace_back(f, 0);
      continue;
    }
#if defined(CARBON_WITH_PDFIUM)
    if (!pdf_pool) pdf_pool = std::make_unique<carbon::PdfPool>(d.engine.pool.workers);
    const int n = pdf_pool->pages(f);  // UTF-8, as every path here (carbon-ocr.manifest)
    for (int p = 0; p < n; ++p) sources.emplace_back(f, p + 1);
    any_pdf = true;
#else
    throw carbon::Error(f + ": this build reads no PDFs (CMake option CARBON_WITH_PDFIUM)");
#endif
  }
  const size_t n_pages = sources.size();
  const bool keep_images = a.format == "pdf";
  std::vector<std::shared_ptr<const carbon::GrayImage>> kept(n_pages);  // the pdf format's images
  const double scale = a.dpi / 72.0;  // 0: each page's own (--dpi auto)
  std::vector<double> dpis(n_pages, any_pdf && a.dpi > 0 ? a.dpi : a.image_dpi);  // each page's, for the pdf format
  PageWriter writer(a, sources);
  // The run stops at the first page that cannot be read, rendered or OCR'd. A page's image is held while it is in
  // flight, and the stream's own submit waits at its in_flight (each page that holds an image goes through it), so
  // producers are bounded by it as before; they also go no further than `ahead` pages past the last written (the
  // finished pages waiting for one before them hold only their results), so a slow page holds back the others'
  // writing alone, not their reading.
  std::mutex m;
  std::condition_variable room;
  std::exception_ptr failed;
  std::atomic<bool> stop{false};
  size_t written = 0;  // under m
  auto fail = [&](std::exception_ptr e) {
    {
      std::lock_guard<std::mutex> g(m);
      if (!failed) failed = e;
      stop = true;
    }
    room.notify_all();
  };
  // Producers: a PDF's helpers each kept busy; image files decoded on as many threads as workers where this
  // thread decodes them (the GPU's, or the pdf format's), else handed to the engine's workers by one.
  const int threads = std::max({1, pdf_pool ? pdf_pool->processes() : 0, cuda || keep_images ? workers : 1});
  const size_t ahead = 32 * (static_cast<size_t>(stream.in_flight()) + static_cast<size_t>(threads));
  std::atomic<size_t> next_page{0};
  std::atomic<int> producing{threads};
  auto produce = [&] {
    for (size_t i; !stop && (i = next_page++) < n_pages;) {
      {
        std::unique_lock<std::mutex> g(m);
        room.wait(g, [&] { return stop || i < written + ahead; });
        if (stop) break;
      }
      try {
        const auto& [file, number] = sources[i];
        if (number == 0) {  // an image file: the engine's worker decodes it, unless this thread must
          if (!cuda && !keep_images) {
            stream.image_file(file, i);
          } else {
            carbon::GrayImage img = carbon::load_image(file);
            if (keep_images) kept[i] = std::make_shared<const carbon::GrayImage>(img);
            stream.image(std::move(img), i);
          }
          continue;
        }
#if defined(CARBON_WITH_PDFIUM)
        carbon::PdfPage pg = pdf_pool->page(file, number - 1, scale, stream.pdf_want());
        dpis[i] = a.dpi > 0 ? a.dpi : pg.scale * 72.0;  // the dpi asked as given (the same bytes), else auto's
        // the pdf format's image as shown: a page read turned upright, turned back, as the writer turns it by
        // the page's rotation; a turned page that goes to OCR rendered again as shown, and that kept
        if (keep_images) {
          kept[i] = std::make_shared<const carbon::GrayImage>(
              pg.turn ? carbon::rotate_quarter(pg.image.view(), 4 - pg.turn) : carbon::GrayImage(pg.image));
        }
        const std::string pdf = file;
        const int at = number - 1;
        const double rendered = pg.scale;
        stream.pdf_page(std::move(pg), i, [&, i, pdf, at, rendered] {
          carbon::GrayImage shown = pdf_pool->page(pdf, at, rendered, carbon::kPdfImage).image;
          if (keep_images) kept[i] = std::make_shared<const carbon::GrayImage>(shown);
          return shown;
        });
#endif
      } catch (...) {
        fail(std::current_exception());
      }
    }
    if (--producing == 0) stream.close();
  };
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> producers;
  for (int t = 0; t < threads; ++t) producers.emplace_back(produce);
  // The pages as they finish, each written once every page before it is.
  std::map<size_t, carbon::OcrPage> waiting;
  size_t from_text = 0, with_pictures = 0;
  try {
    while (std::optional<carbon::DocumentStream::Done> done = stream.next()) {
      if (done->error) {
        fail(done->error);
        continue;
      }
      from_text += done->from_text_layer;
      with_pictures += done->pictures_read;
      waiting.emplace(static_cast<size_t>(done->tag), std::move(done->page));
      while (!stop && !waiting.empty() && waiting.begin()->first == writer.written()) {
        writer.page(std::move(waiting.begin()->second));
        waiting.erase(waiting.begin());
        {
          std::lock_guard<std::mutex> g(m);
          written = writer.written();
        }
        room.notify_all();
      }
    }
  } catch (...) {
    fail(std::current_exception());
  }
  for (auto& p : producers) p.join();
  if (failed) std::rethrow_exception(failed);
  writer.finish(kept, dpis);
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const std::string on = cuda ? "the GPU" : std::to_string(workers) + " workers";
  std::fprintf(stderr,
               "%zu page%s (%zu from their text layers, %zu of those with pictures read) in %.2f s, %.2f pages a "
               "second, on %s\n",
               n_pages, n_pages == 1 ? "" : "s", from_text, with_pictures, secs, n_pages / secs, on.c_str());
  return 0;
}

carbon::Server* serving = nullptr;  // for Ctrl+C

int serve(const Args& a) {
  carbon::ServerOptions so;
  so.host = a.host;
  so.port = a.port;
  so.queue = a.queue;
  so.max_body = static_cast<size_t>(a.max_body_mb) << 20;
  so.decoders = a.decoders;
  so.image_dpi = a.image_dpi;
  so.engine = engine_options(a);
  const carbon::DocumentOptions d = document_options(a);  // a PDF's pages as ocr reads them
  so.text_layer = d.text_layer;
  so.pictures = d.pictures;
  so.layer_lines = d.layer_lines;
  if (a.in_flight) so.engine.in_flight = a.in_flight;
  so.cuda = a.device == "cuda";
  so.pipeline.page_batch = a.batch;
  so.pipeline.prefetch = a.prefetch != "off";
  so.pipeline.head_fp16_accumulate = a.head_accumulate == "fp16";
  if (a.in_flight) so.pipeline.in_flight = a.in_flight;
  so.pipeline.engine_cache = a.engine_cache;
  if (!a.quiet) {
    so.log = [](const std::string& line) {
      const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
      std::tm tm{};  // called on several threads: the reentrant forms
#if defined(_WIN32)
      localtime_s(&tm, &now);
#else
      localtime_r(&now, &tm);
#endif
      char t[16];
      std::strftime(t, sizeof t, "%H:%M:%S", &tm);
      std::fprintf(stderr, "[%s] %s\n", t, line.c_str());
    };
  }
  carbon::Server server(a.set, so);
  serving = &server;
  std::signal(SIGINT, [](int) { serving->stop(); });
  std::signal(SIGTERM, [](int) { serving->stop(); });
  std::fprintf(stderr, "carbon-ocr %s serving %s on http://%s:%d (%s): POST /ocr, POST /parse, GET /health, "
                       "GET /served, GET /info; Ctrl+C stops it\n",
               std::string(carbon::kVersion).c_str(), a.set.c_str(), a.host.c_str(), server.port(),
               so.cuda ? "the GPU" : "the CPU");
  server.run();
  serving = nullptr;
  std::fprintf(stderr, "carbon-ocr: stopped\n");
  return 0;
}

int bench(const Args& a, carbon::Session& s) {
  std::vector<carbon::Tensor> inputs = make_inputs(a, s);
  std::vector<carbon::Tensor> out;
  const Times t = time_runs(a, [&] { out = s.run(inputs); });
  const std::string name(carbon::backend_name(s.backend()));
  print_times(a, s.backend() == carbon::Backend::kTensorRT ? name + " from host memory" : name, t);
  for (size_t i = 0; i < out.size(); ++i) {
    std::printf("  output  %-24s %s\n", s.outputs()[i].name.c_str(),
                carbon::shape_string(out[i].shape()).c_str());
  }
#ifdef CARBON_WITH_TENSORRT
  if (s.backend() == carbon::Backend::kTensorRT) bench_device(a, s, inputs);
#endif
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
  // The arguments are UTF-8 (carbon-ocr.manifest); so is the text it prints.
  SetConsoleOutputCP(CP_UTF8);
#endif
  Args a = parse(argc, argv);
  if (a.command == "version") {
    std::string backends;
    for (carbon::Backend b : carbon::compiled_backends()) {
      backends += (backends.empty() ? "" : ", ") + std::string(carbon::backend_name(b));
    }
    std::printf("carbon-ocr %s (backends: %s)\n", std::string(carbon::kVersion).c_str(),
                backends.empty() ? "none" : backends.c_str());
    return 0;
  }
  if (a.command == "cpus") return cpus(a);
  try {
    if (a.command == "ocr") return ocr(a);
    if (a.command == "serve") return serve(a);
    if (a.command == "bench" && !a.workers.empty()) return bench_workers(a);
    if (a.command == "pages") return pages(a);
    if (a.command == "compare") return compare(a);
    auto s = open(a);
    if (a.command == "info") {
      std::printf("%s on %s\n", a.model.c_str(),
                  std::string(carbon::backend_name(s->backend())).c_str());
      print_infos("input", s->inputs());
      print_infos("output", s->outputs());
      return 0;
    }
    return bench(a, *s);
  } catch (const carbon::Error& e) {
    std::fprintf(stderr, "carbon-ocr: %s\n", e.what());
    return 1;
  }
}
