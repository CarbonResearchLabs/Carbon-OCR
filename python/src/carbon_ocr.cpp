// carbon_ocr: Carbon-OCR's C++ for Python (nanobind): the crop rule, the decoders of
// the models' outputs (carbon/lines.h, carbon/detect.h, carbon/order.h), Tensor and
// Session to run a model from numpy, and Engine and Pipeline to read pages
// (carbon/engine.h, carbon/pipeline.h).

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include "carbon/detect.h"
#include "carbon/engine.h"
#include "carbon/error.h"
#include "carbon/image.h"
#include "carbon/lines.h"
#include "carbon/order.h"
#include "carbon/pipeline.h"
#include "carbon/score.h"
#include "carbon/session.h"
#include "carbon/tensor.h"
#include "carbon/version.h"
#if defined(CARBON_WITH_CUDA)
#include "carbon/cuda.h"
#endif

namespace nb = nanobind;
using namespace nb::literals;

namespace {

using Gray = nb::ndarray<const uint8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Box = std::tuple<double, double, double, double>;
using DecodedLine = std::tuple<std::vector<int32_t>, std::vector<int32_t>, std::vector<float>>;

carbon::CropRule rule(int height, double pad_v, double pad_h, int flip_margin = carbon::CropRule{}.flip_margin) {
  carbon::CropRule r;
  r.height = height;
  r.pad_v = pad_v;
  r.pad_h = pad_h;
  r.flip_margin = flip_margin;
  return r;
}

carbon::LineBox line_box(const Box& b) { return {std::get<0>(b), std::get<1>(b), std::get<2>(b), std::get<3>(b)}; }

carbon::GrayView view(const Gray& a) {
  return {a.data(), static_cast<int>(a.shape(1)), static_cast<int>(a.shape(0)), static_cast<int64_t>(a.shape(1))};
}

// A new uint8 numpy array of the given shape that owns its buffer.
template <size_t N>
nb::ndarray<nb::numpy, uint8_t> new_u8(const size_t (&shape)[N], uint8_t*& data) {
  size_t n = 1;
  for (size_t s : shape) n *= s;
  data = new uint8_t[n ? n : 1];
  nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<uint8_t*>(p); });
  return nb::ndarray<nb::numpy, uint8_t>(data, N, shape, owner);
}

// A numpy array that takes over a vector's buffer: no copy.
template <class T, size_t N>
nb::ndarray<nb::numpy, T> owned(std::vector<T>&& v, const size_t (&shape)[N]) {
  auto* held = new std::vector<T>(std::move(v));
  nb::capsule owner(held, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  return nb::ndarray<nb::numpy, T>(held->data(), N, shape, owner);
}

using Boxes = nb::ndarray<const double, nb::shape<-1, 4>, nb::c_contig, nb::device::cpu>;
using DetBoxes = std::vector<std::tuple<Box, float, int>>;

std::vector<carbon::LineBox> line_boxes(const Boxes& a) {
  std::vector<carbon::LineBox> b(a.shape(0));
  for (size_t i = 0; i < b.size(); ++i) b[i] = {a(i, 0), a(i, 1), a(i, 2), a(i, 3)};
  return b;
}

carbon::DetRule det_rule(int stride, double core_frac, double core_x_shrink, float core_threshold,
                         float picture_threshold, int min_core_pixels, int min_picture_pixels, double edge_quantile) {
  carbon::DetRule r;
  r.stride = stride;
  r.core_frac = core_frac;
  r.core_x_shrink = core_x_shrink;
  r.core_threshold = core_threshold;
  r.picture_threshold = picture_threshold;
  r.min_core_pixels = min_core_pixels;
  r.min_picture_pixels = min_picture_pixels;
  r.edge_quantile = edge_quantile;
  return r;
}

DetBoxes det_boxes(const std::vector<carbon::DetBox>& v) {
  DetBoxes out;
  out.reserve(v.size());
  for (const auto& b : v) out.emplace_back(Box{b.box.x0, b.box.y0, b.box.x1, b.box.y1}, b.score, b.pixels);
  return out;
}

// ---------------------------------------------------------------- tensors

using Array = nb::ndarray<nb::numpy, nb::c_contig, nb::device::cpu>;

carbon::DType dtype_of(const nb::dlpack::dtype& d) {
  using Code = nb::dlpack::dtype_code;
  if (d == nb::dtype<float>()) return carbon::DType::kFloat32;
  if (d.code == static_cast<uint8_t>(Code::Float) && d.bits == 16 && d.lanes == 1) return carbon::DType::kFloat16;
  if (d == nb::dtype<int32_t>()) return carbon::DType::kInt32;
  if (d == nb::dtype<int64_t>()) return carbon::DType::kInt64;
  if (d == nb::dtype<uint8_t>()) return carbon::DType::kUInt8;
  if (d == nb::dtype<bool>()) return carbon::DType::kBool;
  throw carbon::Error("carbon_ocr: arrays of float32, float16, int32, int64, uint8 or bool only");
}

nb::dlpack::dtype dlpack_of(carbon::DType t) {
  switch (t) {
    case carbon::DType::kFloat32: return nb::dtype<float>();
    case carbon::DType::kFloat16: return {static_cast<uint8_t>(nb::dlpack::dtype_code::Float), 16, 1};
    case carbon::DType::kInt32: return nb::dtype<int32_t>();
    case carbon::DType::kInt64: return nb::dtype<int64_t>();
    case carbon::DType::kUInt8: return nb::dtype<uint8_t>();
    case carbon::DType::kBool: return nb::dtype<bool>();
  }
  return nb::dtype<float>();
}

carbon::Tensor from_numpy(const Array& a) {
  carbon::Shape shape;
  for (size_t i = 0; i < a.ndim(); ++i) shape.push_back(static_cast<int64_t>(a.shape(i)));
  carbon::Tensor t = carbon::Tensor::empty(dtype_of(a.dtype()), shape);
  if (t.bytes()) std::memcpy(t.raw(), a.data(), t.bytes());
  return t;
}

// A numpy array that owns a copy of the tensor's elements.
nb::ndarray<nb::numpy> to_numpy(const carbon::Tensor& t) {
  std::vector<size_t> shape(t.shape().begin(), t.shape().end());
  auto* buf = new std::byte[t.bytes() ? t.bytes() : 1];
  if (t.bytes()) std::memcpy(buf, t.raw(), t.bytes());
  nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<std::byte*>(p); });
  return nb::ndarray<nb::numpy>(buf, shape.size(), shape.data(), owner, nullptr, dlpack_of(t.dtype()));
}

// A session's inputs from Python: Tensors or arrays, in a list in the model's
// order or in a dict by name.
std::vector<carbon::Tensor> inputs_of(const carbon::Session& s, nb::handle given) {
  std::vector<nb::handle> items;
  if (nb::isinstance<nb::dict>(given)) {
    nb::dict d = nb::borrow<nb::dict>(given);
    for (const auto& info : s.inputs()) {
      if (!d.contains(info.name.c_str())) throw carbon::Error("no input '" + info.name + "'");
      items.push_back(d[info.name.c_str()]);
    }
  } else {
    for (nb::handle h : given) items.push_back(h);
  }
  std::vector<carbon::Tensor> out;
  for (nb::handle h : items) {
    if (nb::isinstance<carbon::Tensor>(h)) {
      out.push_back(nb::cast<const carbon::Tensor&>(h).clone());
    } else {
      out.push_back(from_numpy(nb::cast<Array>(h)));
    }
  }
  return out;
}

nb::list infos(const std::vector<carbon::TensorInfo>& v) {
  nb::list l;
  for (const auto& t : v) {
    l.append(nb::make_tuple(t.name, std::string(carbon::dtype_name(t.dtype)), t.shape));
  }
  return l;
}

// ---------------------------------------------------------------- pages

nb::dict page_dict(const carbon::OcrPage& p) {
  nb::dict d;
  d["width"] = p.width;
  d["height"] = p.height;
  d["rotation"] = p.rotation;
  nb::list lines;
  for (const auto& l : p.lines) {
    nb::dict ld;
    ld["box"] = nb::make_tuple(l.box.x0, l.box.y0, l.box.x1, l.box.y1);
    ld["text"] = l.text;
    ld["confidence"] = l.confidence;
    ld["det_score"] = l.det_score;
    lines.append(ld);
  }
  d["lines"] = lines;
  nb::list pics;
  for (const auto& b : p.pictures) pics.append(nb::make_tuple(b.x0, b.y0, b.x1, b.y1));
  d["pictures"] = pics;
  d["order"] = p.order;
  nb::list blocks;
  for (const auto& b : p.blocks) {
    nb::dict bd;
    bd["label"] = b.label;
    bd["confidence"] = b.confidence;
    bd["lines"] = b.lines;
    nb::list cells;
    for (const auto& c : b.cells) cells.append(nb::make_tuple(c.row, c.col, c.lines));
    bd["cells"] = cells;
    blocks.append(bd);
  }
  d["blocks"] = blocks;
  nb::dict ms;
  ms["orient"] = p.ms.orient;
  ms["detect"] = p.ms.detect;
  ms["decode"] = p.ms.decode;
  ms["recognize"] = p.ms.recognize;
  ms["total"] = p.ms.total;
  d["ms"] = ms;
  d["text"] = p.text();
  return d;
}

std::vector<DecodedLine> to_python(std::vector<carbon::CtcLine>&& lines) {
  std::vector<DecodedLine> out;
  out.reserve(lines.size());
  for (auto& l : lines) out.emplace_back(std::move(l.ids), std::move(l.frames), std::move(l.probs));
  return out;
}

}  // namespace

NB_MODULE(carbon_ocr, m) {
  m.doc() = "Carbon-OCR's C++ for Python: the crop rule, detector targets and decoding, model sessions, the engine.";

  nb::class_<carbon::Tensor>(m, "Tensor", "A dense host tensor (carbon::Tensor), copied from and to numpy.")
      .def("__init__", [](carbon::Tensor* t, Array a) { new (t) carbon::Tensor(from_numpy(a)); }, "array"_a,
           "A copy of a C-contiguous float32, float16, int32, int64, uint8 or bool array.")
      .def("numpy", &to_numpy, "A copy of the elements as a numpy array.")
      .def_prop_ro("shape", [](const carbon::Tensor& t) { return t.shape(); })
      .def_prop_ro("dtype", [](const carbon::Tensor& t) { return std::string(carbon::dtype_name(t.dtype())); })
      .def("__repr__", [](const carbon::Tensor& t) {
        return "Tensor(" + std::string(carbon::dtype_name(t.dtype())) + ", " + carbon::shape_string(t.shape()) + ")";
      });

  nb::class_<carbon::Session>(m, "Session", "One ONNX model on one backend (carbon::Session).")
      .def(nb::new_([](const std::filesystem::path& model, const std::string& backend, int threads) {
             const auto b = carbon::parse_backend(backend);
             if (!b) throw carbon::Error("unknown backend '" + backend + "' (ort, trt)");
             carbon::SessionOptions o;
             o.threads = threads;
             return carbon::open_session(*b, model, o);
           }),
           "model"_a, "backend"_a = "ort", "threads"_a = 1,
           "Loads model.onnx; threads is ONNX Runtime's per run (1 suits a process per core).")
      .def_prop_ro("backend", [](const carbon::Session& s) { return std::string(carbon::backend_name(s.backend())); })
      .def_prop_ro("inputs", [](const carbon::Session& s) { return infos(s.inputs()); },
                   "(name, dtype, shape) for each input; -1 for a free dimension.")
      .def_prop_ro("outputs", [](const carbon::Session& s) { return infos(s.outputs()); })
      .def(
          "run",
          [](carbon::Session& s, nb::handle inputs) {
            std::vector<carbon::Tensor> in = inputs_of(s, inputs);
            std::vector<carbon::Tensor> out;
            {
              nb::gil_scoped_release unlocked;
              out = s.run(in);
            }
            nb::list result;
            for (const auto& t : out) result.append(to_numpy(t));
            return result;
          },
          "inputs"_a,
          "Runs the model: inputs as arrays or Tensors, a list in the model's order or a dict by name; the "
          "outputs as numpy arrays, in the model's order.");

  nb::class_<carbon::Engine>(m, "Engine", "The OCR pipeline on the CPU (carbon::Engine): page images in, lines out.")
      .def(
          "__init__",
          [](carbon::Engine* e, const std::filesystem::path& set, int workers, bool pin, const std::string& rec,
             std::optional<std::filesystem::path> order_model) {
            carbon::EngineOptions o;
            o.pool.workers = workers;
            o.pool.pin = pin;
            o.rec = rec;
            if (order_model) o.order_model = *order_model;
            new (e) carbon::Engine(set, o);
          },
          "set"_a, "workers"_a = 0, "pin"_a = true, "rec"_a = "rec-latin", "order_model"_a = nb::none(),
          "A Fullerene set's directory (orient.onnx, det.onnx, <rec>.onnx, <rec>.vocab.json, and order.onnx "
          "where it has one; order_model another); workers 0 is one per physical core.")
      .def_prop_ro("workers", &carbon::Engine::workers)
      .def(
          "ocr",
          [](carbon::Engine& e, std::vector<Gray> pages) {
            std::vector<carbon::GrayView> views;
            for (const Gray& g : pages) views.push_back(view(g));
            std::vector<carbon::OcrPage> out;
            {
              nb::gil_scoped_release unlocked;
              out = e.ocr(views);
            }
            nb::list result;
            for (const auto& p : out) result.append(page_dict(p));
            return result;
          },
          "pages"_a,
          "Reads grayscale pages (uint8 [h, w], paper 255) in parallel; for each a dict: width, height, rotation, "
          "lines (box, text, confidence, det_score) in reading order, pictures, order (\"model\" or \"xy-cut\"), "
          "blocks (label, confidence, lines, cells), ms (per stage), text.");

  nb::class_<carbon::Pipeline>(m, "Pipeline",
                               "The batched OCR pipeline (carbon::Pipeline, carbon/pipeline.h): every model run on "
                               "a batch from the pages in flight, on the CPU or a GPU.")
      .def(
          "__init__",
          [](carbon::Pipeline* p, const std::filesystem::path& set, std::optional<std::string> device,
             int page_batch, int line_batch, int in_flight, int threads, int model_threads, const std::string& rec,
             std::optional<std::filesystem::path> order_model, std::optional<std::filesystem::path> engine_cache) {
            carbon::PipelineOptions o;
            if (device) {
              if (*device != "cpu" && *device != "cuda") throw carbon::Error("device: cpu or cuda");
              o.device = *device == "cuda" ? carbon::Device::kCuda : carbon::Device::kCpu;
            }
            o.page_batch = page_batch;
            o.line_batch = line_batch;
            o.in_flight = in_flight;
            o.threads = threads;
            o.model_threads = model_threads;
            o.engine.rec = rec;
            if (order_model) o.engine.order_model = *order_model;
            if (engine_cache) o.engine_cache = *engine_cache;
            new (p) carbon::Pipeline(set, o);
          },
          "set"_a, "device"_a = nb::none(), "page_batch"_a = 8, "line_batch"_a = 256, "in_flight"_a = 32,
          "threads"_a = 0, "model_threads"_a = 0, "rec"_a = "rec-latin", "order_model"_a = nb::none(),
          "engine_cache"_a = nb::none(),
          "A Fullerene set's directory; device \"cpu\" or \"cuda\" (None: the GPU where this build has CUDA and "
          "TensorRT and a GPU is there); page_batch pages an orientation or detection run, line_batch lines a "
          "recognition run, in_flight pages at once; threads the CPU threads preparing and assembling pages and "
          "model_threads a CPU model run's (0: the physical cores).")
      .def_prop_ro("device",
                   [](const carbon::Pipeline& p) { return p.device() == carbon::Device::kCuda ? "cuda" : "cpu"; })
      .def(
          "ocr",
          [](carbon::Pipeline& p, std::vector<Gray> pages) {
            std::vector<carbon::GrayView> views;
            for (const Gray& g : pages) views.push_back(view(g));
            std::vector<carbon::OcrPage> out;
            {
              nb::gil_scoped_release unlocked;
              out = p.ocr(views);
            }
            nb::list result;
            for (const auto& page : out) result.append(page_dict(page));
            return result;
          },
          "pages"_a, "Reads grayscale pages (uint8 [h, w], paper 255), each a dict as Engine.ocr gives it, in order.")
      .def_prop_ro("stats", [](const carbon::Pipeline& p) {
        const carbon::PipelineStats s = p.stats();
        nb::dict d;
        d["pages"] = s.pages;
        d["lines"] = s.lines;
        d["orient_runs"] = s.orient_runs;
        d["detect_runs"] = s.detect_runs;
        d["recognize_runs"] = s.recognize_runs;
        return d;
      });

  m.def(
      "load_image",
      [](const std::filesystem::path& path) {
        carbon::GrayImage img = carbon::load_image(path);
        uint8_t* out = nullptr;
        auto a = new_u8({static_cast<size_t>(img.height), static_cast<size_t>(img.width)}, out);
        std::memcpy(out, img.pixels.data(), img.pixels.size());
        return a;
      },
      "path"_a, "A PNG, JPEG, BMP or PNM page as grayscale uint8 [h, w], as the engine reads it.");
  m.attr("__version__") = std::string(carbon::kVersion);
  const carbon::CropRule d;
  m.attr("HEIGHT") = d.height;
  m.attr("WIDTH_STEP") = d.width_step;
  m.attr("FRAME") = d.frame_px;
  m.attr("PAD_V") = d.pad_v;
  m.attr("PAD_H") = d.pad_h;
  m.attr("FLIP_MARGIN") = d.flip_margin;
#if defined(CARBON_WITH_CUDA)
  m.attr("HAS_CUDA") = true;
#else
  m.attr("HAS_CUDA") = false;
#endif

  m.def(
      "crop_region",
      [](const Box& box, int height, double pad_v, double pad_h) {
        const carbon::CropRegion r = carbon::crop_region(line_box(box), rule(height, pad_v, pad_h));
        return std::make_tuple(std::make_tuple(r.x0, r.y0, r.x1, r.y1), r.width);
      },
      "box"_a, "height"_a = d.height, "pad_v"_a = d.pad_v, "pad_h"_a = d.pad_h,
      "The page rectangle a box's crop resamples, and the crop's width: ((x0, y0, x1, y1), width).");

  m.def(
      "ctc_frames",
      [](const Box& box, int height, double pad_v, double pad_h) {
        return carbon::ctc_frames(line_box(box), rule(height, pad_v, pad_h));
      },
      "box"_a, "height"_a = d.height, "pad_v"_a = d.pad_v, "pad_h"_a = d.pad_h,
      "CTC frames the recognizer gives a box's crop.");

  m.def("width_bucket", &carbon::width_bucket, "crop_width"_a,
        "The batch width the engine runs a crop of this width at: buckets 1.3x apart, 32 to 2048 px.");

  m.def(
      "crop",
      [](Gray image, const Box& box, int height, double pad_v, double pad_h) {
        const carbon::CropRule r = rule(height, pad_v, pad_h);
        const carbon::CropRegion reg = carbon::crop_region(line_box(box), r);
        uint8_t* out = nullptr;
        auto a = new_u8({static_cast<size_t>(height), static_cast<size_t>(reg.width)}, out);
        const carbon::GrayView v = view(image);
        nb::gil_scoped_release unlocked;
        carbon::resample_gray(v, reg.x0, reg.y0, reg.x1, reg.y1, reg.width, height, out, reg.width);
        return a;
      },
      "image"_a, "box"_a, "height"_a = d.height, "pad_v"_a = d.pad_v, "pad_h"_a = d.pad_h,
      "The crop of one line: uint8 [height, width] (paper 255) of a grayscale image for a box in its pixels.");

  m.def(
      "crop_lines",
      [](Gray page, nb::ndarray<const double, nb::shape<-1, 4>, nb::c_contig, nb::device::cpu> boxes,
         int out_width, int height, double pad_v, double pad_h, int flip_margin) {
        const size_t n = boxes.shape(0);
        std::vector<carbon::LineBox> b(n);
        for (size_t i = 0; i < n; ++i) b[i] = {boxes(i, 0), boxes(i, 1), boxes(i, 2), boxes(i, 3)};
        uint8_t* out = nullptr;
        auto a = new_u8({n, static_cast<size_t>(height), static_cast<size_t>(out_width)}, out);
        std::vector<int32_t> widths(n);
        const carbon::GrayView v = view(page);
        {
          nb::gil_scoped_release unlocked;
          carbon::crop_lines(v, b, out_width, out, widths, rule(height, pad_v, pad_h, flip_margin));
        }
        return std::make_tuple(a, widths);
      },
      "page"_a, "boxes"_a, "out_width"_a, "height"_a = d.height, "pad_v"_a = d.pad_v, "pad_h"_a = d.pad_h,
      "flip_margin"_a = d.flip_margin,
      "Crops of many lines of one page, light-on-dark ones flipped, padded to out_width with each crop's ground: "
      "(uint8 [n, height, out_width], widths).");

  m.def(
      "light_on_dark",
      [](Gray crop, int margin) {
        return carbon::light_on_dark(crop.data(), static_cast<int>(crop.shape(1)), static_cast<int>(crop.shape(0)),
                                     static_cast<int64_t>(crop.shape(1)), margin);
      },
      "crop"_a, "margin"_a = d.flip_margin,
      "Whether a crop is light text on a dark ground (carbon::light_on_dark): the crop rule flips it.");

  m.def(
      "resample",
      [](Gray image, const Box& rect, int out_width, int out_height) {
        uint8_t* out = nullptr;
        auto a = new_u8({static_cast<size_t>(out_height), static_cast<size_t>(out_width)}, out);
        const carbon::GrayView v = view(image);
        nb::gil_scoped_release unlocked;
        carbon::resample_gray(v, std::get<0>(rect), std::get<1>(rect), std::get<2>(rect), std::get<3>(rect),
                              out_width, out_height, out, out_width);
        return a;
      },
      "image"_a, "rect"_a, "out_width"_a, "out_height"_a,
      "The crop rule's resampling of the rectangle (x0, y0, x1, y1) to out_width x out_height.");

  m.def(
      "ground", [](Gray crop) { return carbon::crop_ground(crop.data(), static_cast<int>(crop.shape(1)),
                                                           static_cast<int>(crop.shape(0)),
                                                           static_cast<int64_t>(crop.shape(1))); },
      "crop"_a, "The ground a batch pads a crop with: the median of its first row, last row and last column.");

  const carbon::DetRule dr;
  m.attr("DET_STRIDE") = dr.stride;
  m.attr("DET_CORE_FRAC") = dr.core_frac;
  m.attr("DET_CORE_X_SHRINK") = dr.core_x_shrink;

  m.def(
      "det_targets",
      [](Boxes lines, Boxes pictures, int map_h, int map_w, int stride, double core_frac, double core_x_shrink) {
        const auto l = line_boxes(lines), p = line_boxes(pictures);
        const carbon::DetRule r = det_rule(stride, core_frac, core_x_shrink, 0.5f, 0.5f, 2, 16, 0.05);
        const size_t shape[3] = {4, static_cast<size_t>(map_h), static_cast<size_t>(map_w)};
        float* out = new float[4 * shape[1] * shape[2] + 1];
        nb::capsule owner(out, [](void* q) noexcept { delete[] static_cast<float*>(q); });
        {
          nb::gil_scoped_release unlocked;
          carbon::det_targets(l, p, map_h, map_w, out, r);
        }
        return nb::ndarray<nb::numpy, float>(out, 3, shape, owner);
      },
      "lines"_a, "pictures"_a, "map_h"_a, "map_w"_a, "stride"_a = dr.stride, "core_frac"_a = dr.core_frac,
      "core_x_shrink"_a = dr.core_x_shrink,
      "The detector's targets for one page, float32 [4, map_h, map_w]: core 0/1, top and bottom (input px), "
      "picture 0/1, for line and picture boxes [n, 4] in input pixels.");

  m.def(
      "decode_det",
      [](nb::ndarray<const float, nb::shape<4, -1, -1>, nb::c_contig, nb::device::cpu> maps, double scale,
         int stride, double core_frac, double core_x_shrink, float core_threshold, float picture_threshold,
         int min_core_pixels, int min_picture_pixels, double edge_quantile) {
        const carbon::DetRule r = det_rule(stride, core_frac, core_x_shrink, core_threshold, picture_threshold,
                                           min_core_pixels, min_picture_pixels, edge_quantile);
        carbon::Detection d;
        {
          nb::gil_scoped_release unlocked;
          d = carbon::decode_det(maps.data(), static_cast<int>(maps.shape(1)), static_cast<int>(maps.shape(2)),
                                 scale, r);
        }
        return std::make_tuple(det_boxes(d.lines), det_boxes(d.pictures));
      },
      "maps"_a, "scale"_a = 1.0, "stride"_a = dr.stride, "core_frac"_a = dr.core_frac,
      "core_x_shrink"_a = dr.core_x_shrink, "core_threshold"_a = dr.core_threshold,
      "picture_threshold"_a = dr.picture_threshold, "min_core_pixels"_a = dr.min_core_pixels,
      "min_picture_pixels"_a = dr.min_picture_pixels, "edge_quantile"_a = dr.edge_quantile,
      "Line and picture boxes from the detector's maps [4, h, w] (core and picture as logits), in input pixels "
      "times scale: (lines, pictures), each a list of ((x0, y0, x1, y1), score, pixels).");

#if defined(CARBON_WITH_CUDA)
  m.def(
      "cuda_decode_det",
      [](uintptr_t maps, int n, int map_h, int map_w, uintptr_t stream, bool fp16, double scale, int stride,
         double core_frac, double core_x_shrink, float core_threshold, float picture_threshold, int min_core_pixels,
         int min_picture_pixels, double edge_quantile) {
        const carbon::DetRule r = det_rule(stride, core_frac, core_x_shrink, core_threshold, picture_threshold,
                                           min_core_pixels, min_picture_pixels, edge_quantile);
        std::vector<carbon::Detection> d(static_cast<size_t>(std::max(0, n)));
        {
          nb::gil_scoped_release unlocked;
          const std::vector<carbon::DetMasks> masks = carbon::cuda::threshold_det(
              reinterpret_cast<const void*>(maps), fp16, n, map_h, map_w, r, reinterpret_cast<void*>(stream));
          for (size_t i = 0; i < d.size(); ++i) d[i] = carbon::decode_det(masks[i], scale, r);
        }
        std::vector<std::tuple<DetBoxes, DetBoxes>> out;
        for (const carbon::Detection& x : d) out.emplace_back(det_boxes(x.lines), det_boxes(x.pictures));
        return out;
      },
      "maps"_a, "n"_a, "map_h"_a, "map_w"_a, "stream"_a = 0, "fp16"_a = false, "scale"_a = 1.0,
      "stride"_a = dr.stride, "core_frac"_a = dr.core_frac, "core_x_shrink"_a = dr.core_x_shrink,
      "core_threshold"_a = dr.core_threshold, "picture_threshold"_a = dr.picture_threshold,
      "min_core_pixels"_a = dr.min_core_pixels, "min_picture_pixels"_a = dr.min_picture_pixels,
      "edge_quantile"_a = dr.edge_quantile,
      "decode_det for n pages of maps on a GPU: maps the device address of [n, 4, map_h, map_w] float32 (or "
      "float16), a torch tensor's data_ptr(), after the work queued on the stream. They are thresholded there "
      "(carbon::cuda::threshold_det): only the masks and the set pixels' values come to the host. A list of "
      "(lines, pictures), a page each, the boxes decode_det gives.");
#endif

  {
    nb::list labels;
    for (int c = 0; c < carbon::kBlockLabels; ++c) labels.append(carbon::block_label_name(c));
    m.attr("BLOCK_LABELS") = labels;
  }
  m.attr("ORDER_FEATURES") = carbon::kOrderFeatures;

  m.def(
      "order_features",
      [](Boxes boxes, const std::vector<std::string>& texts, int width, int height) {
        std::vector<float> f = carbon::order_features(line_boxes(boxes), texts, width, height);
        return owned(std::move(f), {texts.size(), static_cast<size_t>(carbon::kOrderFeatures)});
      },
      "boxes"_a, "texts"_a, "width"_a, "height"_a,
      "The order model's features of a page's lines (carbon/order.h), float32 [n, 16].");

  m.def(
      "reading_order", [](Boxes boxes) { return carbon::reading_order(line_boxes(boxes)); }, "boxes"_a,
      "The XY-cut's reading order of boxes [n, 4]: indices.");

  using Square = nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
  m.def(
      "decode_order",
      [](Square successor, Square same_block, Square table_row, Square table_col, Square table_cell, Square labels,
         Boxes boxes, double min_sure, int sweeps, float join) {
        const size_t n = boxes.shape(0);
        auto span = [](const Square& a) { return std::span<const float>(a.data(), a.shape(0) * a.shape(1)); };
        carbon::OrderOutputs out{static_cast<int>(n), span(successor), span(same_block), span(table_row),
                                 span(table_col), span(table_cell), span(labels)};
        carbon::OrderRule rule;
        rule.min_sure = min_sure;
        rule.sweeps = sweeps;
        rule.join = join;
        const auto b = line_boxes(boxes);
        carbon::PageOrder po;
        {
          nb::gil_scoped_release unlocked;
          po = carbon::decode_order(out, b, rule);
        }
        nb::list blocks;
        for (const auto& blk : po.blocks) {
          nb::list cells;
          for (const auto& c : blk.cells) cells.append(std::make_tuple(c.row, c.col, c.lines));
          blocks.append(std::make_tuple(std::string(carbon::block_label_name(blk.label)), blk.confidence, blk.lines,
                                        cells));
        }
        nb::dict d;
        d["order"] = po.order;
        d["blocks"] = blocks;
        d["sure"] = po.sure;
        d["fallback"] = po.fallback;
        return d;
      },
      "successor"_a, "same_block"_a, "table_row"_a, "table_col"_a, "table_cell"_a, "labels"_a, "boxes"_a,
      "min_sure"_a = 0.5, "sweeps"_a = 4, "join"_a = 0.5f,
      "The order model's outputs for n lines decoded (carbon/order.h): {order, blocks: [(label, confidence, lines, "
      "cells: [(row, col, lines)])], sure, fallback}.");

  m.def(
      "ctc_greedy",
      [](nb::ndarray<const float, nb::ndim<3>, nb::c_contig, nb::device::cpu> logits,
         nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> lengths) {
        const int n = static_cast<int>(logits.shape(0)), c = static_cast<int>(logits.shape(1)),
                  t = static_cast<int>(logits.shape(2));
        std::vector<carbon::CtcLine> lines;
        {
          nb::gil_scoped_release unlocked;
          lines = carbon::ctc_greedy(logits.data(), n, c, t, {lengths.data(), lengths.shape(0)});
        }
        return to_python(std::move(lines));
      },
      "logits"_a, "lengths"_a,
      "CTC greedy decoding of logits [n, classes, frames]: per line (ids, start frames, probabilities).");

  m.def(
      "ctc_collapse",
      [](nb::ndarray<const int32_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> ids,
         nb::ndarray<const float, nb::ndim<2>, nb::c_contig, nb::device::cpu> probs,
         nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu> lengths) {
        const int n = static_cast<int>(ids.shape(0)), t = static_cast<int>(ids.shape(1));
        std::vector<carbon::CtcLine> lines;
        {
          nb::gil_scoped_release unlocked;
          lines = carbon::ctc_collapse(ids.data(), probs.data(), n, t, {lengths.data(), lengths.shape(0)});
        }
        return to_python(std::move(lines));
      },
      "ids"_a, "probs"_a, "lengths"_a,
      "The second half of ctc_greedy, for a best class per frame found elsewhere (a GPU): ids and their "
      "probabilities [n, frames].");

  m.def(
      "text_errors",
      [](const std::vector<std::string>& truths, const std::vector<std::string>& predicted) {
        if (truths.size() != predicted.size()) throw carbon::Error("text_errors: as many predictions as truths");
        const size_t n = truths.size(), shape[2] = {n, 4};
        auto* out = new int64_t[4 * n + 1];
        nb::capsule owner(out, [](void* q) noexcept { delete[] static_cast<int64_t*>(q); });
        {
          nb::gil_scoped_release unlocked;
          for (size_t i = 0; i < n; ++i) {
            const carbon::TextErrors e = carbon::text_errors(truths[i], predicted[i]);
            out[4 * i] = e.char_errors;
            out[4 * i + 1] = e.chars;
            out[4 * i + 2] = e.word_errors;
            out[4 * i + 3] = e.words;
          }
        }
        return nb::ndarray<nb::numpy, int64_t>(out, 2, shape, owner);
      },
      "truths"_a, "predicted"_a,
      "Each line's errors against its truth (carbon/score.h), int64 [n, 4]: the edit distance in code points and "
      "the truth's code points (the CER's), then in words and the truth's words (the WER's), words split as "
      "str.split() splits them.");

  m.def(
      "char_confusions",
      [](const std::vector<std::string>& truths, const std::vector<std::string>& predicted, size_t top) {
        if (truths.size() != predicted.size()) throw carbon::Error("char_confusions: as many predictions as truths");
        carbon::Confusions c;
        {
          nb::gil_scoped_release unlocked;
          c = carbon::char_confusions(truths, predicted);
        }
        nb::list edits;
        for (size_t i = 0; i < c.edits.size() && i < top; ++i) {
          edits.append(nb::make_tuple(carbon::utf8_of(c.edits[i].truth), carbon::utf8_of(c.edits[i].predicted),
                                      c.edits[i].count));
        }
        nb::dict d;
        d["edits"] = edits;
        d["distinct"] = c.edits.size();
        d["substitutions"] = c.substitutions;
        d["insertions"] = c.insertions;
        d["deletions"] = c.deletions;
        d["chars"] = c.chars;
        return d;
      },
      "truths"_a, "predicted"_a, "top"_a = 50,
      "The characters a recognizer confuses, over many lines (carbon/score.h): an alignment of the fewest edits "
      "per line, and a dict of `edits`, the `top` most frequent as (truth, predicted, count), \"\" for none (an "
      "insertion has no truth, a deletion no prediction), the most frequent first; `distinct` edits in all; the "
      "`substitutions`, `insertions` and `deletions`; and the truths' `chars`.");

  m.def(
      "edit_distances",
      [](const std::vector<std::vector<int32_t>>& a, const std::vector<std::vector<int32_t>>& b) {
        if (a.size() != b.size()) throw carbon::Error("edit_distances: as many sequences in each");
        const size_t n = a.size(), shape[1] = {n};
        auto* out = new int64_t[n + 1];
        nb::capsule owner(out, [](void* q) noexcept { delete[] static_cast<int64_t*>(q); });
        {
          nb::gil_scoped_release unlocked;
          for (size_t i = 0; i < n; ++i) out[i] = carbon::edit_distance(std::span<const int32_t>(a[i]), b[i]);
        }
        return nb::ndarray<nb::numpy, int64_t>(out, 1, shape, owner);
      },
      "a"_a, "b"_a,
      "The edit distance between each pair of class sequences (lists of ints), int64 [n]: the classes a "
      "precision changes against another's, the parity gate's.");
}
