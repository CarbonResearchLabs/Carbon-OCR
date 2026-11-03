// The batched pipeline's models on a GPU (carbon/pipeline.h, Device::kCuda):
// TensorRT from the set's FP16 files, a CUDA stream for each stage that calls
// them, and the cuda:: kernels between, so a page crosses to the device once and
// only masks, ids and probabilities come back.
//
//   orient     the thumbnails (float32, from the CPU threads) uploaded and made
//              float16 on the device; orient.fp16.onnx at an exact profile for
//              each batch size (1, 2, 4, ... page_batch): a CUDA graph each
//   detect     the pages at the detector's scale, uint8, padded with paper into a
//              pinned buffer at one of the fixed sizes, uploaded, made an ink map
//              (cuda::ink_map), det.fp16.onnx, and cuda::threshold_det: only the
//              masks come back for carbon::decode_det
//   recognize  the upright pages uploaded once a group (the same views across the
//              group's runs), every line cropped by cuda::crop_lines, made an ink
//              map, the trunk (<rec>.features.fp16.onnx) and the head fused with
//              greedy decoding (cuda::ctc_head_argmax): ids and probabilities back,
//              collapsed by carbon::ctc_collapse.  Without the trunk and head files,
//              <rec>.onnx in float32 and cuda::ctc_greedy.
//
// Every buffer a model reads or writes is allocated at open at its largest, so
// its address stays fixed and the CUDA graphs replay.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "carbon/cuda.h"
#include "carbon/engine.h"
#include "carbon/error.h"
#include "carbon/pipeline.h"
#include "carbon/session.h"

namespace carbon {

namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw Error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e));
}

int round_up(int v, int step) { return (v + step - 1) / step * step; }

float half_to_float(uint16_t h) {
  const bool negative = h & 0x8000u;
  const uint32_t exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
  float v;
  if (exp == 0) {
    v = std::ldexp(static_cast<float>(man), -24);  // zero or subnormal
  } else if (exp == 31) {
    v = man ? NAN : INFINITY;
  } else {
    uint32_t bits = ((exp + 112u) << 23) | (man << 13);
    std::memcpy(&v, &bits, sizeof v);
  }
  return negative ? -v : v;
}

// Device memory, allocated once at its largest.
struct DeviceMem {
  void* p = nullptr;
  size_t bytes = 0;
  void reserve(size_t n) {
    if (n <= bytes) return;
    if (p) cudaFree(p);
    p = nullptr;
    check(cudaMalloc(&p, n), "cudaMalloc");
    bytes = n;
  }
  template <class T>
  T* as() const {
    return static_cast<T*>(p);
  }
  ~DeviceMem() {
    if (p) cudaFree(p);
  }
};

// Pinned host memory, for copies the stream does not wait for.
struct Pinned {
  void* p = nullptr;
  size_t bytes = 0;
  void reserve(size_t n) {
    if (n <= bytes) return;
    if (p) cudaFreeHost(p);
    p = nullptr;
    check(cudaHostAlloc(&p, n, cudaHostAllocDefault), "cudaHostAlloc");
    bytes = n;
  }
  template <class T>
  T* as() const {
    return static_cast<T*>(p);
  }
  ~Pinned() {
    if (p) cudaFreeHost(p);
  }
};

struct Stream {
  cudaStream_t s = nullptr;
  Stream() { check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate"); }
  ~Stream() {
    if (s) cudaStreamDestroy(s);
  }
  void sync() const { check(cudaStreamSynchronize(s), "cudaStreamSynchronize"); }
};

// The trainer's width buckets up to the widest a batch holds (carbon::width_bucket).
std::vector<int> buckets_to(int widest) {
  std::vector<int> out;
  for (int w = 32; w <= widest; w = width_bucket(w + 1)) out.push_back(w);
  return out;
}

class CudaModels final : public PipelineModels {
 public:
  CudaModels(const std::filesystem::path& set, const EngineOptions& o, const PipelineOptions& p)
      : gpu_(p.gpu), side_(o.orient_side), long_(o.det_long_side), lines_max_(p.line_batch), crop_(o.crop) {
    check(cudaSetDevice(gpu_), "cudaSetDevice");
    for (int b = 1; b < p.page_batch; b *= 2) batches_.push_back(b);
    batches_.push_back(p.page_batch);
    const int short_side = round_up(static_cast<int>(std::lround(800.0 * long_ / 1024)), 32);
    sizes_ = {{round_up(long_, 32), short_side}, {short_side, round_up(long_, 32)},
              {round_up(long_, 32), round_up(long_, 32)}};
    const int maxb = batches_.back();

    SessionOptions so;
    so.device = gpu_;
    so.engine_cache = p.engine_cache;
    SessionOptions orient_o = so;
    for (int b : batches_) orient_o.profiles.push_back({ShapeRange::exact("x", {b, 1, side_, side_})});
    orient_ = open_session(Backend::kTensorRT, set / "orient.fp16.onnx", orient_o);
    SessionOptions det_o = so;
    for (const auto& [h, w] : sizes_) {
      for (int b : batches_) det_o.profiles.push_back({ShapeRange::exact("x", {b, 1, h, w})});
    }
    det_ = open_session(Backend::kTensorRT, set / "det.fp16.onnx", det_o);

    // the recognizer: a profile for each width bucket, batches of 1 to line_batch
    const int h = crop_.height;
    widths_ = buckets_to(2048);
    SessionOptions rec_o = so;
    for (int bw : widths_) {
      rec_o.profiles.push_back({{"x", {1, 1, h, bw}, {lines_max_, 1, h, bw}, {lines_max_, 1, h, bw}}});
    }
    const std::filesystem::path trunk = set / (o.rec + ".features.fp16.onnx"), head = set / (o.rec + ".head.bin");
    fused_ = p.fused_head && std::filesystem::is_regular_file(trunk) && std::filesystem::is_regular_file(head);
    const size_t classes_vocab = load_vocab(set / (o.rec + ".vocab.json")).size();
    if (fused_) {
      rec_ = open_session(Backend::kTensorRT, trunk, rec_o);
      const CtcHead hd = load_ctc_head(head);
      classes_ = hd.classes;
      k_ = hd.k;
      head_w_.reserve(hd.weight.size() * sizeof(uint16_t));
      head_b_.reserve(hd.bias.size() * sizeof(float));
      check(cudaMemcpy(head_w_.p, hd.weight.data(), hd.weight.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
            "head weight");
      check(cudaMemcpy(head_b_.p, hd.bias.data(), hd.bias.size() * sizeof(float), cudaMemcpyHostToDevice),
            "head bias");
    } else {
      rec_ = open_session(Backend::kTensorRT, set / (o.rec + ".onnx"), rec_o);
      const Shape& logits = rec_->outputs().at(0).shape;
      classes_ = static_cast<int>(logits.at(1));
    }
    if (static_cast<size_t>(classes_) != classes_vocab) {
      throw Error(o.rec + " has " + std::to_string(classes_) + " classes but its vocabulary " +
                  std::to_string(classes_vocab));
    }

    // every buffer at its largest, once
    const size_t thumbs = static_cast<size_t>(maxb) * side_ * side_;
    orient_f32_.reserve(thumbs * sizeof(float));
    orient_in_.reserve(thumbs * sizeof(uint16_t));
    orient_out_.reserve(static_cast<size_t>(maxb) * 13 * sizeof(uint16_t));
    orient_host_.reserve(thumbs * sizeof(float));
    orient_logits_.reserve(static_cast<size_t>(maxb) * 13 * sizeof(uint16_t));
    size_t page_px = 0;
    for (const auto& [ph, pw] : sizes_) page_px = std::max(page_px, static_cast<size_t>(ph) * pw);
    det_host_.reserve(maxb * page_px);
    det_u8_.reserve(maxb * page_px);
    det_in_.reserve(maxb * page_px * sizeof(uint16_t));
    det_maps_.reserve(maxb * page_px / 4 * 4 * sizeof(uint16_t));  // [b][4][h/2][w/2]
    const size_t widest = static_cast<size_t>(widths_.back()), frames = widest / crop_.frame_px;
    crops_.reserve(static_cast<size_t>(lines_max_) * h * widest);
    rec_in_.reserve(static_cast<size_t>(lines_max_) * h * widest * (fused_ ? 2 : 4));
    crop_widths_.reserve(static_cast<size_t>(lines_max_) * sizeof(int32_t));
    if (fused_) {
      features_.reserve(static_cast<size_t>(lines_max_) * k_ * frames * sizeof(uint16_t));
      ids_.reserve(static_cast<size_t>(lines_max_) * frames * sizeof(int32_t));
      probs_.reserve(static_cast<size_t>(lines_max_) * frames * sizeof(float));
      ids_host_.reserve(ids_.bytes);
      probs_host_.reserve(probs_.bytes);
    } else {
      logits_.reserve(static_cast<size_t>(lines_max_) * classes_ * frames * sizeof(float));
    }
    widths_host_.reserve(crop_widths_.bytes);
  }

  Device device() const override { return Device::kCuda; }

  std::vector<float> orient(const float* thumbs, int n, int side) override {
    check(cudaSetDevice(gpu_), "cudaSetDevice");
    const int b = batch_for(n);
    const size_t per = static_cast<size_t>(side) * side;
    float* host = orient_host_.as<float>();
    std::memcpy(host, thumbs, n * per * sizeof(float));
    std::fill(host + n * per, host + b * per, 0.0f);  // paper
    const cudaStream_t s = orient_s_.s;
    check(cudaMemcpyAsync(orient_f32_.p, host, b * per * sizeof(float), cudaMemcpyHostToDevice, s), "thumbs");
    cuda::to_half(orient_f32_.as<float>(), orient_in_.p, static_cast<int64_t>(b * per), s);
    const Shape in{b, 1, side, side};
    const std::vector<DeviceTensor> x = {{orient_in_.p, DType::kFloat16, in}};
    const std::vector<DeviceTensor> y = {{orient_out_.p, DType::kFloat16, orient_->output_shapes({&in, 1})[0]}};
    orient_->enqueue(x, y, s);
    check(cudaMemcpyAsync(orient_logits_.p, orient_out_.p, static_cast<size_t>(b) * 13 * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost, s),
          "orient logits");
    orient_s_.sync();
    std::vector<float> out(static_cast<size_t>(n) * 13);
    const uint16_t* z = orient_logits_.as<uint16_t>();
    for (size_t i = 0; i < out.size(); ++i) out[i] = half_to_float(z[i]);
    return out;
  }

  std::pair<int, int> detect_size(int width, int height) const override {
    for (const auto& [h, w] : sizes_) {
      if (height <= h && width <= w) return {h, w};
    }
    return {round_up(height, 32), round_up(width, 32)};  // larger than every fixed size: no profile holds it
  }

  std::vector<Detection> detect(std::span<const GrayView> pages, int ph, int pw, const DetRule& rule) override {
    check(cudaSetDevice(gpu_), "cudaSetDevice");
    const int n = static_cast<int>(pages.size()), b = batch_for(n);
    const size_t per = static_cast<size_t>(ph) * pw;
    uint8_t* host = det_host_.as<uint8_t>();
    std::memset(host, 255, b * per);  // paper
    for (int i = 0; i < n; ++i) {
      const GrayView& g = pages[i];
      uint8_t* to = host + i * per;
      for (int y = 0; y < g.height; ++y) std::memcpy(to + static_cast<size_t>(y) * pw, g.data + y * g.stride, g.width);
    }
    const cudaStream_t s = det_s_.s;
    check(cudaMemcpyAsync(det_u8_.p, host, b * per, cudaMemcpyHostToDevice, s), "pages");
    cuda::ink_map(det_u8_.as<uint8_t>(), static_cast<int64_t>(b * per), det_in_.p, true, s);
    const Shape in{b, 1, ph, pw};
    const Shape out_shape = det_->output_shapes({&in, 1})[0];  // [b, 4, ph/2, pw/2]
    const std::vector<DeviceTensor> x = {{det_in_.p, DType::kFloat16, in}};
    const std::vector<DeviceTensor> y = {{det_maps_.p, DType::kFloat16, out_shape}};
    det_->enqueue(x, y, s);
    const std::vector<DetMasks> masks = cuda::threshold_det(det_maps_.p, true, n, static_cast<int>(out_shape[2]),
                                                            static_cast<int>(out_shape[3]), rule, s);
    std::vector<Detection> found;
    for (const DetMasks& m : masks) found.push_back(decode_det(m, 1.0, rule));
    return found;
  }

  // The group's upright pages on the device, once for all its runs.
  void set_pages(std::span<const GrayView> pages) override {
    check(cudaSetDevice(gpu_), "cudaSetDevice");
    size_t total = 0;
    for (const GrayView& g : pages) total += static_cast<size_t>(g.width) * g.height;
    rec_s_.sync();  // the last group's crops are done with the buffer
    pages_.reserve(total);
    device_pages_.clear();
    size_t at = 0;
    for (const GrayView& g : pages) {
      uint8_t* d = pages_.as<uint8_t>() + at;
      check(cudaMemcpy2DAsync(d, g.width, g.data, g.stride, g.width, g.height, cudaMemcpyHostToDevice, rec_s_.s),
            "page");
      device_pages_.push_back({d, g.width, g.height, g.width});
      at += static_cast<size_t>(g.width) * g.height;
    }
  }

  void recognize(std::span<const int32_t> page_of, std::span<const LineBox> boxes, int bw, const CropRule& crop,
                 std::vector<CtcLine>& lines, std::vector<int32_t>& lengths) override {
    check(cudaSetDevice(gpu_), "cudaSetDevice");
    const cudaStream_t s = rec_s_.s;
    const int m = static_cast<int>(boxes.size()), h = crop.height, frames = bw / crop.frame_px;
    cuda::crop_lines(device_pages_, page_of, boxes, bw, crops_.as<uint8_t>(), crop_widths_.as<int32_t>(), crop, s);
    const int64_t px = static_cast<int64_t>(m) * h * bw;
    cuda::ink_map(crops_.as<uint8_t>(), px, rec_in_.p, fused_, s);
    const Shape in{m, 1, h, bw};
    const std::vector<DeviceTensor> x = {{rec_in_.p, fused_ ? DType::kFloat16 : DType::kFloat32, in}};
    check(cudaMemcpyAsync(widths_host_.p, crop_widths_.p, m * sizeof(int32_t), cudaMemcpyDeviceToHost, s), "widths");
    if (fused_) {
      const std::vector<DeviceTensor> y = {{features_.p, DType::kFloat16, rec_->output_shapes({&in, 1})[0]}};
      rec_->enqueue(x, y, s);
      cuda::ctc_head_argmax(features_.p, head_w_.p, head_b_.as<float>(), m, k_, classes_, frames, ids_.as<int32_t>(),
                            probs_.as<float>(), s);
      check(cudaMemcpyAsync(ids_host_.p, ids_.p, static_cast<size_t>(m) * frames * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, s),
            "ids");
      check(cudaMemcpyAsync(probs_host_.p, probs_.p, static_cast<size_t>(m) * frames * sizeof(float),
                            cudaMemcpyDeviceToHost, s),
            "probs");
      rec_s_.sync();
      set_lengths(m, frames, crop, lengths);
      lines = ctc_collapse(ids_host_.as<int32_t>(), probs_host_.as<float>(), m, frames, lengths);
    } else {
      const Shape out_shape = rec_->output_shapes({&in, 1})[0];  // [m, classes, frames]
      const std::vector<DeviceTensor> y = {{logits_.p, DType::kFloat32, out_shape}};
      rec_->enqueue(x, y, s);
      rec_s_.sync();
      set_lengths(m, static_cast<int>(out_shape[2]), crop, lengths);
      lines = cuda::ctc_greedy(logits_.as<float>(), m, classes_, static_cast<int>(out_shape[2]), lengths, s);
    }
  }

 private:
  int batch_for(int n) const {
    for (int b : batches_) {
      if (b >= n) return b;
    }
    throw Error("a batch of " + std::to_string(n) + " pages is more than page_batch");
  }

  void set_lengths(int m, int frames, const CropRule& crop, std::vector<int32_t>& lengths) const {
    const int32_t* w = widths_host_.as<int32_t>();
    lengths.resize(m);
    for (int i = 0; i < m; ++i) lengths[i] = std::min<int32_t>(frames, w[i] / crop.frame_px);
  }

  int gpu_, side_, long_, lines_max_;
  CropRule crop_;
  std::vector<int> batches_, widths_;
  std::vector<std::pair<int, int>> sizes_;
  std::unique_ptr<Session> orient_, det_, rec_;
  bool fused_ = false;
  int classes_ = 0, k_ = 0;
  Stream orient_s_, det_s_, rec_s_;
  DeviceMem head_w_, head_b_;
  DeviceMem orient_f32_, orient_in_, orient_out_;
  Pinned orient_host_, orient_logits_;
  Pinned det_host_;
  DeviceMem det_u8_, det_in_, det_maps_;
  DeviceMem pages_, crops_, rec_in_, crop_widths_, features_, ids_, probs_, logits_;
  Pinned ids_host_, probs_host_, widths_host_;
  std::vector<cuda::DevicePage> device_pages_;
};

}  // namespace

bool cuda_pipeline_available() {
  const auto built = compiled_backends();
  return std::find(built.begin(), built.end(), Backend::kTensorRT) != built.end() && cuda::available();
}

std::unique_ptr<PipelineModels> cuda_pipeline_models(const std::filesystem::path& set, const EngineOptions& o,
                                                     const PipelineOptions& p) {
  return std::make_unique<CudaModels>(set, o, p);
}

}  // namespace carbon
