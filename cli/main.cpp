// carbon-ocr: the command line.  It reads page images (ocr), and inspects and
// times models and model sets.
//
//   carbon-ocr ocr IMAGE|PDF... --set DIR [--rec rec-latin] [--backend ort|native]
//                           [--precision auto|fp32|fp16|int8]
//                           [--dpi 200] [--text-layer auto|off] [--hybrid auto|off] [--second-pass auto|off]
//                           [--format text|json|hocr|alto|page|markdown|html|pdf] [--out-dir DIR]
//                           [--output FILE] [--text-out FILE] [--text-only] [--image-dpi 300] [--as-given]
//                           [--workers N] [--no-pin] [--device cpu|cuda] [--batch 8] [--in-flight N]
//                           [--engine-cache DIR]
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
// cpus prints the instruction sets the kernels use and the cores the workers
// use; --gemm times the int8 GEMM on one thread with each instruction set, at
// the shapes of Fullerene's convolutions.  bench --workers runs, for each
// count, a pool of that many workers pinned one per core, each with its own
// session and input, all at once: the CPU layout.  It reports
// the runs per second of the whole pool and its scaling against the first
// count's per-worker rate.
//
// ocr and pages run each model of a set on the native backend where the set has
// its int8 file (<model>.int8.onnx), else on ONNX Runtime (carbon::open_set_model);
// --backend runs every model on one (order stays on ONNX Runtime).  --precision
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
// tools/ocrmypdf, which turns and levels pages itself).  It reads PNG, JPEG, BMP or PNM pages, and the pages
// of PDFs rendered at --dpi
// (carbon/pdf.h), with the Fullerene set in --set.  A PDF page whose own text
// passes the checks (carbon/textlayer.h) is read from it, not by OCR, unless
// --text-layer off; the JSON gives each page's source, and for a PDF page read
// by OCR why its text layer was not used.  Such a page's pictures (its image
// objects) are read by OCR, on the page with only its pictures left, and their
// lines join the layer's (source "hybrid"), unless --hybrid off.  A page whose lines are
// tiny (the first pass's median under 9 px at the detector's scale) is detected again at
// twice the scale and the passes merged (carbon/detect.h), unless --second-pass off
// (carbon::Engine, the pipeline v0 on the CPU; reading order by the set's
// order.onnx where it has one, every page's, else by the XY-cut): text prints each page's lines,
// pages apart by a form feed; json one object with every page.  The pages run
// in parallel, a worker per physical core.  --device cuda reads them through the
// batched pipeline on the GPU instead (carbon/pipeline.h: TensorRT from the set's
// FP16 files, --batch pages a detection run, --in-flight pages at once, the built
// engines kept in --engine-cache).
//
// pages times a model set's ONNX files (SETDIR/orient.onnx, det.onnx,
// script.onnx where the set has it, <rec>.onnx, order.onnx) the way the CPU
// pipeline runs a page: orientation and detection, the line
// decoder, then the page's lines in tasks any idle worker may take (the script
// classifier and the recognizer), then the order model when the last is done, each step planned
// for the cores free at the time (see pages() below).  The inputs are
// constant, at the budget's sizes: a 512 x 512 thumbnail, a 1024 x 800 page,
// 32 x 512 crops, 16 features a line; random weights detect noise, so the
// decoder reads the maps of a rendered page of --lines lines.  --in-flight N keeps N pages going at a time (1: one
// page on every core, its latency); by default every page is queued at once
// (throughput).

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#include "carbon/pipeline.h"
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
  std::string hybrid = "auto";      // ocr: auto, the pictures of a page read from its text layer read by OCR; off
  std::string second_pass = "auto";  // ocr: auto, tiny text detected again at twice the scale; off
  std::string out_dir;               // ocr --format page: a file a page here
  std::string output;                // ocr: the file to write (pdf needs one); empty: standard output
  std::string text_out;              // ocr: also the pages' text, to this file
  bool text_only = false;            // ocr --format pdf: the text layer alone, no images
  double image_dpi = 300;            // ocr --format pdf: an image's page at this dpi
  bool as_given = false;             // ocr: no quarter turns, no deskew: the boxes the given page's
  std::string device = "cpu";        // ocr: cpu, the engine's workers; cuda, the batched pipeline on the GPU
  std::string engine_cache;
  std::string isa;  // the native backend's
  std::string against;  // compare: ONNX Runtime runs this file
  bool backend_given = false;  // ocr and pages choose per model otherwise
  carbon::Precision precision = carbon::Precision::kAuto;  // ocr and pages: the set's files
};

[[noreturn]] void usage(const std::string& why) {
  std::fprintf(stderr,
               "carbon-ocr: %s\n"
               "usage: carbon-ocr ocr IMAGE|PDF... --set DIR [--rec NAME] [--workers N]\n"
               "                        [--no-pin] [--backend ort|native] [--precision auto|fp32|fp16|int8]\n"
               "                        [--dpi N] [--text-layer auto|off]\n"
               "                        [--hybrid auto|off] [--second-pass auto|off]\n"
               "                        [--format text|json|hocr|alto|page|markdown|html|pdf] [--out-dir DIR]\n"
               "                        [--output FILE] [--text-out FILE] [--text-only] [--image-dpi N]\n"
               "                        [--as-given] [--device cpu|cuda] [--batch N] [--in-flight N]\n"
               "                        [--engine-cache DIR]\n"
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
      a.command != "compare") {
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
      if (a.hybrid != "auto" && a.hybrid != "off") usage("--hybrid takes auto or off");
    } else if (arg == "--second-pass") {
      a.second_pass = value();
      if (a.second_pass != "auto" && a.second_pass != "off") usage("--second-pass takes auto or off");
    } else if (arg == "--dpi") {
      a.dpi = to_int(arg, value(), 36);
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
    } else if (arg.starts_with("--")) {
      usage("unknown option " + arg);
    } else if (a.command == "ocr") {
      a.images.push_back(arg);
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
    const bool fp32_det = a.precision == carbon::Precision::kAuto || a.precision == carbon::Precision::kFp32;
    if ((a.det == "auto" || a.det == "wide") && fp32_det) {  // det.onnx on ONNX Runtime: FP32, as the engine
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

int ocr(const Args& a) {
  carbon::EngineOptions o;
  o.rec = a.rec;
  if (a.backend_given) o.backend = a.backend;
  o.precision = a.precision;
  o.pool.workers = a.workers.empty() ? 0 : a.workers.front();
  o.pool.pin = a.pin;
  if (a.second_pass == "off") o.det.second_pass_below = 0;  // one detection pass, whatever the lines' size
  if (a.as_given) {  // the page as given (OCRmyPDF turns and deskews pages itself, and grafts the text back)
    o.turn_probability = 2.0f;
    o.deskew = false;
  }
  carbon::Engine engine(a.set, o);
  // Each image a page; each page of a PDF rendered at --dpi (carbon/pdf.h) in the
  // helper processes (carbon-pdf, beside this program), many at once.
  std::vector<carbon::GrayImage> images;
  std::vector<std::pair<std::string, int>> sources;  // a page's file, and its page in a PDF (from 1; 0: an image)
  std::vector<size_t> pdf_pages;                     // the pages of `images` still to render
  std::unique_ptr<carbon::PdfPool> pdf_pool;
  // Each page's slot, made before the renderers start, so each writes its own pages'
  // alone: those read from their text layers, and for PDF pages read by OCR why not.
  std::vector<std::optional<carbon::OcrPage>> text_pages;
  std::vector<std::optional<carbon::GrayImage>> picture_pages;  // a text page's pictures alone, for OCR
  std::vector<std::string> notes;
  for (const std::string& f : a.images) {
    std::string ext = std::filesystem::path(f).extension().string();
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (ext != ".pdf") {
      images.push_back(carbon::load_image(f));
      sources.emplace_back(f, 0);
      continue;
    }
#if defined(CARBON_WITH_PDFIUM)
    if (!pdf_pool) pdf_pool = std::make_unique<carbon::PdfPool>(o.pool.workers);
    const int n = pdf_pool->pages(f);  // UTF-8, as every path here (carbon-ocr.manifest)
    for (int p = 0; p < n; ++p) {
      pdf_pages.push_back(images.size());
      images.emplace_back();
      sources.emplace_back(f, p + 1);
    }
#else
    throw carbon::Error(f + ": this build reads no PDFs (CMake option CARBON_WITH_PDFIUM)");
#endif
  }
  text_pages.resize(images.size());
  picture_pages.resize(images.size());
  notes.resize(images.size());
#if defined(CARBON_WITH_PDFIUM)
  if (!pdf_pages.empty()) {
    const double scale = a.dpi / 72.0;
    {
      std::atomic<size_t> next{0};
      std::vector<std::thread> renderers;
      std::exception_ptr failed;
      std::mutex failed_m;
      for (int t = 0; t < pdf_pool->processes(); ++t) {
        renderers.emplace_back([&] {
          for (size_t k; (k = next++) < pdf_pages.size();) {
            const size_t i = pdf_pages[k];
            try {
              const std::string& file = sources[i].first;
              const int page = sources[i].second - 1;
              images[i] = pdf_pool->render(file, page, scale);
              if (a.text_layer == "auto") {  // a page whose own text passes the checks is read from it
                const carbon::TextLayer layer =
                    carbon::read_text_layer(pdf_pool->chars(file, page, scale), images[i].view());
                if (layer.check.usable) {
                  text_pages[i] = carbon::page_from_text_layer(layer, images[i].width, images[i].height);
                  if (a.hybrid == "auto") {  // its pictures' own text, which the layer does not hold
                    text_pages[i]->pictures = pdf_pool->pictures(file, page, scale);
                    const std::vector<carbon::LineBox> read =
                        carbon::pictures_to_read(text_pages[i]->pictures, images[i].width, images[i].height);
                    if (!read.empty()) picture_pages[i] = carbon::pictures_only(images[i].view(), read);
                  }
                } else {
                  notes[i] = layer.check.reason;
                }
              }
            } catch (...) {
              std::lock_guard<std::mutex> g(failed_m);
              if (!failed) failed = std::current_exception();
            }
          }
        });
      }
      for (auto& r : renderers) r.join();
      if (failed) std::rethrow_exception(failed);
    }
  }
#endif
  // OCR for every page its text layer did not read, then for the pictures of those it did.
  std::vector<carbon::GrayView> views;
  std::vector<size_t> by_ocr, hybrid;
  for (size_t i = 0; i < images.size(); ++i) {
    if (text_pages[i]) continue;
    by_ocr.push_back(i);
    views.push_back(images[i].view());
  }
  for (size_t i = 0; i < images.size(); ++i) {
    if (!picture_pages[i]) continue;
    hybrid.push_back(i);
    views.push_back(picture_pages[i]->view());
  }
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<carbon::OcrPage> read;
  if (a.device == "cuda") {  // the batched pipeline on the GPU (carbon/pipeline.h)
    carbon::PipelineOptions po;
    po.engine = o;
    po.device = carbon::Device::kCuda;
    po.page_batch = a.batch;
    if (a.in_flight) po.in_flight = a.in_flight;
    po.engine_cache = a.engine_cache;
    carbon::Pipeline pipe(a.set, po);
    read = pipe.ocr(views);
  } else {
    read = engine.ocr(views);
  }
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::vector<carbon::OcrPage> pages(images.size());
  for (size_t k = 0; k < by_ocr.size(); ++k) {
    pages[by_ocr[k]] = std::move(read[k]);
    pages[by_ocr[k]].text_layer = notes[by_ocr[k]];
  }
  for (size_t k = 0; k < hybrid.size(); ++k) carbon::add_picture_lines(*text_pages[hybrid[k]], read[by_ocr.size() + k]);
  std::vector<size_t> from_text;
  for (size_t i = 0; i < images.size(); ++i) {
    if (!text_pages[i]) continue;
    pages[i] = std::move(*text_pages[i]);
    from_text.push_back(i);
  }
  if (!from_text.empty()) {  // the text layer's pages in the order model's order too, or their tables from geometry
    std::vector<carbon::OcrPage> ordered;
    for (size_t i : from_text) ordered.push_back(std::move(pages[i]));
    engine.order(ordered);
    for (size_t k = 0; k < from_text.size(); ++k) pages[from_text[k]] = std::move(ordered[k]);
  }
  if (!a.text_out.empty()) {  // the pages' text beside the format asked for, a form feed between pages
    std::string text;
    for (size_t i = 0; i < pages.size(); ++i) text += (i ? "\f\n" : "") + pages[i].text();
    std::ofstream(std::filesystem::path(a.text_out), std::ios::binary)
        .write(text.data(), static_cast<std::streamsize>(text.size()));
  }
  // Each page's image, as the formats name it: its file, a PDF's page as file#page=N.
  std::vector<std::string> names;
  for (const auto& [file, number] : sources) names.push_back(number ? file + "#page=" + std::to_string(number) : file);
  if (a.format == "hocr" || a.format == "alto" || a.format == "markdown" || a.format == "html" ||
      a.format == "pdf") {
    std::string out;
    if (a.format == "pdf") {  // each page's image at the dpi it was read at (a PDF's --dpi; an image's 300)
      std::vector<carbon::GrayView> views_all;
      for (const auto& img : images) views_all.push_back(img.view());
      out = carbon::to_pdf(pages, views_all, pdf_pages.empty() ? a.image_dpi : a.dpi, a.text_only);
    } else {
      out = a.format == "hocr" ? carbon::to_hocr(pages, names)
            : a.format == "alto" ? carbon::to_alto(pages, names)
            : a.format == "html" ? carbon::to_html(pages)
                                 : carbon::to_markdown(pages);
    }
    if (a.output.empty()) {
      std::fwrite(out.data(), 1, out.size(), stdout);
    } else {
      std::ofstream(std::filesystem::path(a.output), std::ios::binary)
          .write(out.data(), static_cast<std::streamsize>(out.size()));
    }
  } else if (a.format == "page") {
    std::filesystem::create_directories(a.out_dir);
    for (size_t i = 0; i < pages.size(); ++i) {
      const auto& [file, number] = sources[i];
      std::string name = std::filesystem::path(file).stem().string();
      if (number) name += "-" + std::to_string(number);
      const std::filesystem::path out = std::filesystem::path(a.out_dir) / (name + ".xml");
      const std::string x = carbon::to_page_xml(pages[i], names[i]);
      std::ofstream(out, std::ios::binary).write(x.data(), static_cast<std::streamsize>(x.size()));
    }
  } else if (a.format == "json") {
    std::printf("{\"pages\": [");
    for (size_t i = 0; i < pages.size(); ++i) {
      const std::string page = pages[i].json();  // {"width": ...}: the file goes first
      const auto& [file, number] = sources[i];
      std::printf("%s{\"file\": %s, %s%s", i ? ", " : "", quoted(file).c_str(),
                  number ? ("\"page\": " + std::to_string(number) + ", ").c_str() : "", page.c_str() + 1);
    }
    std::printf("]}\n");
  } else {
    for (size_t i = 0; i < pages.size(); ++i) std::printf("%s%s", i ? "\f\n" : "", pages[i].text().c_str());
  }
  const std::string on = a.device == "cuda" ? "the GPU" : std::to_string(engine.workers()) + " workers";
  std::fprintf(stderr,
               "%zu page%s (%zu from their text layers, %zu of those with pictures read) in %.2f s, %.2f pages a "
               "second, on %s\n",
               pages.size(), pages.size() == 1 ? "" : "s", images.size() - by_ocr.size(), hybrid.size(), secs,
               pages.size() / secs, on.c_str());
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
