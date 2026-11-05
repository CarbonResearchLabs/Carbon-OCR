// The batched pipeline's models on a GPU (carbon/pipeline.h, Device::kCuda):
// TensorRT from the set's files at engine.precision (gpu_set_model: FP16, FP32 or
// symmetric INT8; TensorRT's networks are strongly typed, so the file is the
// precision), a CUDA stream for each stage that calls them, and the cuda:: kernels
// between, so a page crosses to the device once and only masks, ids and
// probabilities come back.  Each model's input and output are float16 or float32 as
// its file has them (an FP16 file's float16, an FP32 or Q/DQ file's float32), and
// its buffers and kernels follow.
//
//   orient     the thumbnails (float32, from the CPU threads) uploaded (made
//              float16 on the device for a float16 input); orient's file at an
//              exact profile for each batch size (1, 2, 4, ... page_batch): a CUDA
//              graph each
//   detect     the pages at the detector's scale, uint8, padded with paper into a
//              pinned buffer at one of the fixed sizes, uploaded, made an ink map
//              (cuda::ink_map), det's file, and cuda::threshold_det: only the
//              masks come back for carbon::decode_det.  The second pass (a page of
//              tiny text, as the CPU runs it: carbon/detect.h) gives detect_one each
//              strip's ink map, any size up to second_pass_long_side square: det's
//              file again, at one profile that holds them all, its maps back as
//              float32, decoded and merged on the CPU as the CPU's are
//   recognize  the upright pages uploaded once a group (the same views across the
//              group's runs), every line cropped by cuda::crop_lines, made an ink
//              map; at fp16 and auto the trunk (<rec>.features.fp16.onnx) and the
//              head fused with greedy decoding (cuda::ctc_head_argmax): ids and
//              probabilities back, collapsed by carbon::ctc_collapse; else (no trunk
//              and head files, fused_head off, fp32, int8) the recognizer whole,
//              <rec>.onnx or <rec>.int8-sym.onnx, its float32 logits through
//              cuda::ctc_greedy.  A line wider than every bucket (past 2048 px) is
//              read on the CPU, by the CPU device's models, as the CPU reads it.
//
// Every buffer a model reads or writes is allocated at open at its largest, so
// its address stays fixed and the CUDA graphs replay.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <string>
#include <utility>
#include <vector>

#include "carbon/cuda.h"
#include "carbon/engine.h"
#include "carbon/error.h"
#include "carbon/parallel.h"
#include "carbon/pipeline.h"
#include "carbon/session.h"

namespace carbon {

namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw Error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e));
}

int round_up(int v, int step) { return (v + step - 1) / step * step; }

// Whether a model's tensor is float16 (else float32, the two the pipeline's buffers and kernels take).
bool is_half(const TensorInfo& t, const std::filesystem::path& file) {
  if (t.dtype == DType::kFloat16) return true;
  if (t.dtype == DType::kFloat32) return false;
  throw Error(file.string() + ": " + t.name + " is " + std::string(dtype_name(t.dtype)) +
              ", not float32 or float16");
}

DType dtype(bool half) { return half ? DType::kFloat16 : DType::kFloat32; }

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
    const Precision pr = o.precision;
    SessionOptions orient_o = so;
    for (int b : batches_) orient_o.profiles.push_back({ShapeRange::exact("x", {b, 1, side_, side_})});
    const std::filesystem::path orient_file = gpu_set_model(set, "orient", pr);
    orient_ = open_session(Backend::kTensorRT, orient_file, orient_o);
    orient_half_in_ = is_half(orient_->inputs().at(0), orient_file);
    orient_half_out_ = is_half(orient_->outputs().at(0), orient_file);
    SessionOptions det_o = so;
    for (const auto& [h, w] : sizes_) {
      for (int b : batches_) det_o.profiles.push_back({ShapeRange::exact("x", {b, 1, h, w})});
    }
    const std::filesystem::path det_file = gpu_set_model(set, "det", pr);
    det_ = open_session(Backend::kTensorRT, det_file, det_o);
    det_half_in_ = is_half(det_->inputs().at(0), det_file);
    det_half_out_ = is_half(det_->outputs().at(0), det_file);
    // The second detection pass: the upright page again at twice the detector's scale, its long side at most
    // second_pass_long_side, whole or in strips of about a first pass's pixels (engine_steps.h second_pass), so
    // any size of multiples of 32 up to that side square. A session of det's file of its own, so the first
    // pass's exact profiles (and their cached engines) stay as they are. None where the pass is off.
    if (o.det.second_pass_below > 0) {
      second_side_ = round_up(o.second_pass_long_side, 32);
      const int first = round_up(long_, 32);
      SessionOptions det2_o = so;
      det2_o.profiles.push_back({{"x", {1, 1, 32, 32}, {1, 1, first, first}, {1, 1, second_side_, second_side_}}});
      det2_ = open_session(Backend::kTensorRT, det_file, det2_o);
      det2_half_in_ = is_half(det2_->inputs().at(0), det_file);
      det2_half_out_ = is_half(det2_->outputs().at(0), det_file);
      const size_t px = static_cast<size_t>(second_side_) * second_side_;
      det2_f32_.reserve(px * sizeof(float));
      if (det2_half_in_) det2_in_.reserve(px * sizeof(uint16_t));
      det2_maps_.reserve(px / 4 * 4 * (det2_half_out_ ? 2 : 4));  // [1][4][h/2][w/2]
      det2_host_.reserve(det2_maps_.bytes);
    }

    // the recognizer: a profile for each width bucket, batches of 1 to line_batch
    const int h = crop_.height;
    widths_ = buckets_to(2048);
    SessionOptions rec_o = so;
    for (int bw : widths_) {
      rec_o.profiles.push_back({{"x", {1, 1, h, bw}, {lines_max_, 1, h, bw}, {lines_max_, 1, h, bw}}});
    }
    const std::filesystem::path trunk = set / (o.rec + ".features.fp16.onnx"), head = set / (o.rec + ".head.bin");
    const bool fused_files = std::filesystem::is_regular_file(trunk) && std::filesystem::is_regular_file(head);
    if (pr == Precision::kFp16 && p.fused_head && !fused_files) {
      throw Error(o.rec + ": FP16 on the GPU reads it through its trunk and fused head, " + trunk.string() + " and " +
                  head.string() + ", which the set does not have");
    }
    fused_ = p.fused_head && (pr == Precision::kAuto || pr == Precision::kFp16) && fused_files;
    const size_t classes_vocab = load_vocab(set / (o.rec + ".vocab.json")).size();
    if (fused_) {
      rec_ = open_session(Backend::kTensorRT, trunk, rec_o);
      rec_half_in_ = is_half(rec_->inputs().at(0), trunk);
      const CtcHead hd = load_ctc_head(head);
      classes_ = hd.classes;
      k_ = hd.k;
      head_w_.reserve(hd.weight.size() * sizeof(uint16_t));
      head_b_.reserve(hd.bias.size() * sizeof(float));
      check(cudaMemcpy(head_w_.p, hd.weight.data(), hd.weight.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
            "head weight");
      check(cudaMemcpy(head_b_.p, hd.bias.data(), hd.bias.size() * sizeof(float), cudaMemcpyHostToDevice),
            "head bias");
    } else {  // the recognizer whole: INT8's file, or FP32 (fp32; auto and fp16 without the fused head)
      const std::filesystem::path whole = gpu_set_model(set, o.rec, pr == Precision::kInt8 ? pr : Precision::kFp32);
      rec_ = open_session(Backend::kTensorRT, whole, rec_o);
      rec_half_in_ = is_half(rec_->inputs().at(0), whole);
      if (is_half(rec_->outputs().at(0), whole)) {
        throw Error(whole.string() + ": its logits are float16; cuda::ctc_greedy reads float32");
      }
      const Shape& logits = rec_->outputs().at(0).shape;
      classes_ = static_cast<int>(logits.at(1));
    }
    if (static_cast<size_t>(classes_) != classes_vocab) {
      throw Error(o.rec + " has " + std::to_string(classes_) + " classes but its vocabulary " +
                  std::to_string(classes_vocab));
    }
    // A line wider than every bucket (a crop past 2048 px, which carbon::width_bucket gives its own width rounded
    // up to 8) has no profile here: recognize gives such lines to the CPU device's models, opened at the first.
    set_ = set;
    cpu_o_ = o;
    cpu_o_.precision = Precision::kAuto;  // the CPU's own files (the GPU's int8-sym file is not the CPU's int8)

    // every buffer at its largest, once
    const size_t thumbs = static_cast<size_t>(maxb) * side_ * side_;
    const size_t orient_out = static_cast<size_t>(maxb) * 13 * (orient_half_out_ ? 2 : 4);
    orient_f32_.reserve(thumbs * sizeof(float));
    if (orient_half_in_) orient_in_.reserve(thumbs * sizeof(uint16_t));
    orient_out_.reserve(orient_out);
    orient_host_.reserve(thumbs * sizeof(float));
    orient_logits_.reserve(orient_out);
    size_t page_px = 0;
    for (const auto& [ph, pw] : sizes_) page_px = std::max(page_px, static_cast<size_t>(ph) * pw);
    det_host_.reserve(maxb * page_px);
    det_u8_.reserve(maxb * page_px);
    det_in_.reserve(maxb * page_px * (det_half_in_ ? 2 : 4));
    det_maps_.reserve(maxb * page_px / 4 * 4 * (det_half_out_ ? 2 : 4));  // [b][4][h/2][w/2]
    const size_t widest = static_cast<size_t>(widths_.back()), frames = widest / crop_.frame_px;
    crops_.reserve(static_cast<size_t>(lines_max_) * h * widest);
    rec_in_.reserve(static_cast<size_t>(lines_max_) * h * widest * (rec_half_in_ ? 2 : 4));
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
    uploader_ = std::thread([this] { uploader(); });
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
    void* thumbs_in = orient_f32_.p;
    if (orient_half_in_) {
      cuda::to_half(orient_f32_.as<float>(), orient_in_.p, static_cast<int64_t>(b * per), s);
      thumbs_in = orient_in_.p;
    }
    const Shape in{b, 1, side, side};
    const std::vector<DeviceTensor> x = {{thumbs_in, dtype(orient_half_in_), in}};
    const std::vector<DeviceTensor> y = {{orient_out_.p, dtype(orient_half_out_), orient_->output_shapes({&in, 1})[0]}};
    orient_->enqueue(x, y, s);
    check(cudaMemcpyAsync(orient_logits_.p, orient_out_.p, static_cast<size_t>(b) * 13 * (orient_half_out_ ? 2 : 4),
                          cudaMemcpyDeviceToHost, s),
          "orient logits");
    orient_s_.sync();
    std::vector<float> out(static_cast<size_t>(n) * 13);
    if (orient_half_out_) {
      const uint16_t* z = orient_logits_.as<uint16_t>();
      for (size_t i = 0; i < out.size(); ++i) out[i] = half_to_float(z[i]);
    } else {
      std::memcpy(out.data(), orient_logits_.p, out.size() * sizeof(float));
    }
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
    cuda::ink_map(det_u8_.as<uint8_t>(), static_cast<int64_t>(b * per), det_in_.p, det_half_in_, s);
    const Shape in{b, 1, ph, pw};
    const Shape out_shape = det_->output_shapes({&in, 1})[0];  // [b, 4, ph/2, pw/2]
    const std::vector<DeviceTensor> x = {{det_in_.p, dtype(det_half_in_), in}};
    const std::vector<DeviceTensor> y = {{det_maps_.p, dtype(det_half_out_), out_shape}};
    det_->enqueue(x, y, s);
    const std::vector<DetMasks> masks = cuda::threshold_det(det_maps_.p, det_half_out_, n,
                                                            static_cast<int>(out_shape[2]),
                                                            static_cast<int>(out_shape[3]), rule, s);
    std::vector<Detection> found;
    for (const DetMasks& m : masks) found.push_back(decode_det(m, 1.0, rule));
    return found;
  }

  bool second_pass() const override { return det2_ != nullptr; }
  // A sticky error: the context can run nothing more, every later call fails with it (CUDA's own list of them).
  bool broken() override {
    cudaSetDevice(gpu_);
    switch (cudaDeviceSynchronize()) {
      case cudaErrorIllegalAddress:
      case cudaErrorLaunchFailure:
      case cudaErrorHardwareStackError:
      case cudaErrorIllegalInstruction:
      case cudaErrorMisalignedAddress:
      case cudaErrorInvalidAddressSpace:
      case cudaErrorInvalidPc:
      case cudaErrorLaunchTimeout:
      case cudaErrorECCUncorrectable:
      case cudaErrorDevicesUnavailable:
      case cudaErrorContextIsDestroyed:
      case cudaErrorUnknown:
        return true;
      default:
        return false;
    }
  }

  // A second pass's run (the detect stage's thread, on its stream): the page's or strip's ink map, [1, 1, h, w]
  // float32 from the CPU, uploaded (made float16 for a float16 input), det's file, and the maps back as float32,
  // [1, 4, h/2, w/2], which the CPU decodes as it decodes its own.
  Tensor detect_one(const Tensor& input) override {
    check(cudaSetDevice(gpu_), "cudaSetDevice");
    const Shape& s = input.shape();
    if (s.size() != 4 || s[0] != 1 || s[1] != 1 || s[2] < 32 || s[3] < 32 || s[2] > second_side_ ||
        s[3] > second_side_ || s[2] % 32 || s[3] % 32 || input.dtype() != DType::kFloat32) {
      throw Error("the second detection pass on the GPU takes [1, 1, h, w] float32, h and w multiples of 32 to " +
                  std::to_string(second_side_) + ", not " + shape_string(s) + " " +
                  std::string(dtype_name(input.dtype())));
    }
    const size_t px = static_cast<size_t>(s[2]) * s[3];
    const cudaStream_t st = det_s_.s;
    check(cudaMemcpyAsync(det2_f32_.p, input.data<float>().data(), px * sizeof(float), cudaMemcpyHostToDevice, st),
          "second pass input");
    void* x_in = det2_f32_.p;
    if (det2_half_in_) {
      cuda::to_half(det2_f32_.as<float>(), det2_in_.p, static_cast<int64_t>(px), st);
      x_in = det2_in_.p;
    }
    const Shape in{1, 1, s[2], s[3]};
    const Shape out_shape = det2_->output_shapes({&in, 1})[0];
    const std::vector<DeviceTensor> x = {{x_in, dtype(det2_half_in_), in}};
    const std::vector<DeviceTensor> y = {{det2_maps_.p, dtype(det2_half_out_), out_shape}};
    det2_->enqueue(x, y, st);
    size_t n = 1;
    for (const int64_t d : out_shape) n *= static_cast<size_t>(d);
    check(cudaMemcpyAsync(det2_host_.p, det2_maps_.p, n * (det2_half_out_ ? 2 : 4), cudaMemcpyDeviceToHost, st),
          "second pass maps");
    det_s_.sync();
    Tensor out = Tensor::empty(DType::kFloat32, out_shape);
    float* to = out.data<float>().data();
    if (det2_half_out_) {
      const uint16_t* z = det2_host_.as<uint16_t>();
      for (size_t i = 0; i < n; ++i) to[i] = half_to_float(z[i]);
    } else {
      std::memcpy(to, det2_host_.p, n * sizeof(float));
    }
    return out;
  }

  // The group's upright pages on the device, once for all its runs: the pages prefetch_pages is uploading or
  // uploaded already where they are these (waited for on the host only until their copies are queued), else
  // uploaded now. Either way the recognize stream waits for its slot's upload alone.
  void set_pages(std::span<const GrayView> pages, std::span<const RowRuns> rows) override {
    check(cudaSetDevice(gpu_), "cudaSetDevice");
    PageSlot& next = slots_[1 - cur_];
    wait_uploader();
    if (!(next.ready && same_pages(next.views, pages))) upload(next, pages, rows);
    next.ready = false;
    cur_ = 1 - cur_;
    check(cudaStreamWaitEvent(rec_s_.s, next.done, 0), "cudaStreamWaitEvent");
    host_pages_ = next.views;  // for lines wider than every bucket, read on the CPU
    device_pages_ = next.device;
  }

  // The next group's pages into the slot the current group does not use, by the uploader thread on the copy
  // stream: the recognize thread goes straight on to the current group's runs, and the bus works beside them.
  void prefetch_pages(std::span<const GrayView> pages, std::span<const RowRuns> rows) override {
    wait_uploader();
    PageSlot& next = slots_[1 - cur_];
    next.views.assign(pages.begin(), pages.end());  // what set_pages compares; the upload writes the same
    next.ready = true;
    {
      std::lock_guard lk(up_m_);
      up_job_ = {&next, std::vector<GrayView>(pages.begin(), pages.end()),
                 std::vector<RowRuns>(rows.begin(), rows.end())};
      up_busy_ = true;
    }
    up_cv_.notify_all();
  }

  ~CudaModels() override {
    {
      std::lock_guard lk(up_m_);
      up_stop_ = true;
    }
    up_cv_.notify_all();
    if (uploader_.joinable()) uploader_.join();
  }

  void recognize(std::span<const int32_t> page_of, std::span<const LineBox> boxes, int bw, const CropRule& crop,
                 std::vector<CtcLine>& lines, std::vector<int32_t>& lengths) override {
    if (bw > widths_.back()) {  // wider than every bucket: the CPU device's crop, recognizer and decoding
      if (!cpu_) cpu_ = cpu_pipeline_models(set_, cpu_o_, std::clamp(cpu_topology().default_workers(), 1, 4));
      cpu_->set_pages(host_pages_, {});
      cpu_->recognize(page_of, boxes, bw, crop, lines, lengths);
      return;
    }
    check(cudaSetDevice(gpu_), "cudaSetDevice");
    const cudaStream_t s = rec_s_.s;
    const int m = static_cast<int>(boxes.size()), h = crop.height, frames = bw / crop.frame_px;
    cuda::crop_lines(device_pages_, page_of, boxes, bw, crops_.as<uint8_t>(), crop_widths_.as<int32_t>(), crop, s);
    const int64_t px = static_cast<int64_t>(m) * h * bw;
    cuda::ink_map(crops_.as<uint8_t>(), px, rec_in_.p, rec_half_in_, s);
    const Shape in{m, 1, h, bw};
    const std::vector<DeviceTensor> x = {{rec_in_.p, dtype(rec_half_in_), in}};
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

  // A group's pages on the device: two of them, the current group's and the next's, each its
  // pages' bytes in pinned memory (copied at memory speed, then sent by the copy engine while the recognize stream
  // reads the other slot's) and an event its upload records.
  struct PageSlot {
    DeviceMem dev;
    Pinned host;
    std::vector<GrayView> views;
    std::vector<cuda::DevicePage> device;
    cudaEvent_t done = nullptr;
    bool ready = false;  // a prefetch_pages' upload, not yet taken by set_pages
    PageSlot() { check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "cudaEventCreate"); }
    ~PageSlot() {
      if (done) cudaEventDestroy(done);
    }
  };

  static bool same_pages(const std::vector<GrayView>& a, std::span<const GrayView> b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i].data != b[i].data || a[i].width != b[i].width || a[i].height != b[i].height ||
          a[i].stride != b[i].stride) {
        return false;
      }
    }
    return true;
  }

  // The slot was last read two groups ago (every run of it synchronized since) or by a prefetch not taken: its
  // last upload is waited for before its pinned bytes are written again. Each page keeps its whole layout on the
  // device, but only the rows its crops read (rows[i], where given) are copied and sent, in chunks of kChunk,
  // each sent (asynchronously, from pinned memory) as soon as it is copied, so the copy engine sends one chunk while
  // the host copies the next, as CUDA's own staging of pageable memory overlaps them.
  void upload(PageSlot& slot, std::span<const GrayView> pages, std::span<const RowRuns> rows) {
    constexpr size_t kChunk = size_t{1} << 20;
    check(cudaEventSynchronize(slot.done), "cudaEventSynchronize");
    size_t total = 0;
    for (const GrayView& g : pages) total += static_cast<size_t>(g.width) * g.height;
    if (total > slot.dev.bytes) {  // grown by half again, so a few larger groups do not each reallocate
      const size_t grown = std::max(total, slot.dev.bytes + slot.dev.bytes / 2);
      slot.dev.reserve(grown);
      slot.host.reserve(grown);
    }
    slot.views.assign(pages.begin(), pages.end());
    slot.device.clear();
    uint8_t* h = slot.host.as<uint8_t>();
    uint8_t* d = slot.dev.as<uint8_t>();
    size_t from = 0, to = 0;  // the bytes copied and not yet sent, [from, to)
    auto flush = [&] {
      if (to > from) check(cudaMemcpyAsync(d + from, h + from, to - from, cudaMemcpyHostToDevice, copy_s_.s), "pages");
      from = to;
    };
    size_t base = 0;
    auto copy_rows = [&](const GrayView& g, int y0, int y1) {
      const size_t w = static_cast<size_t>(g.width);
      const size_t at = base + static_cast<size_t>(y0) * w;
      if (at != to) {  // a gap: what is copied so far goes on its own
        flush();
        from = to = at;
      }
      for (int y = y0; y < y1; ++y) {
        std::memcpy(h + to, g.data + static_cast<int64_t>(y) * g.stride, w);
        to += w;
        if (to - from >= kChunk) flush();
      }
    };
    for (size_t i = 0; i < pages.size(); ++i) {
      const GrayView& g = pages[i];
      slot.device.push_back({d + base, g.width, g.height, g.width});
      if (i < rows.size() && !rows[i].empty()) {
        for (const auto& [y0, y1] : rows[i]) copy_rows(g, std::max(0, y0), std::min(g.height, y1));
      } else {
        copy_rows(g, 0, g.height);
      }
      base += static_cast<size_t>(g.width) * g.height;
    }
    flush();
    check(cudaEventRecord(slot.done, copy_s_.s), "cudaEventRecord");
  }

  // The uploader thread: a prefetch's upload at a time.
  void uploader() {
    cudaSetDevice(gpu_);
    std::unique_lock lk(up_m_);
    for (;;) {
      up_cv_.wait(lk, [&] { return up_stop_ || (up_busy_ && up_job_.slot); });
      if (up_stop_) return;
      UpJob job = std::move(up_job_);
      up_job_ = {};
      lk.unlock();
      try {
        upload(*job.slot, job.pages, job.rows);
      } catch (...) {  // the group's set_pages uploads it again, and meets the error itself if it lasts
        job.slot->ready = false;
      }
      lk.lock();
      up_busy_ = false;
      up_cv_.notify_all();
    }
  }
  // Until the uploader holds no prefetch: its slot's copies queued (its event recorded), or its error met.
  void wait_uploader() {
    std::unique_lock lk(up_m_);
    up_cv_.wait(lk, [&] { return !up_busy_; });
  }

  int gpu_, side_, long_, lines_max_;
  CropRule crop_;
  std::vector<int> batches_, widths_;
  std::vector<std::pair<int, int>> sizes_;
  std::unique_ptr<Session> orient_, det_, rec_;
  std::filesystem::path set_;
  EngineOptions cpu_o_;                  // lines wider than every bucket: the CPU device's models,
  std::unique_ptr<PipelineModels> cpu_;  // opened at the first such line
  std::vector<GrayView> host_pages_;     // set_pages', for them
  std::unique_ptr<Session> det2_;  // the second detection pass; none where it is off
  int second_side_ = 0;            // its input's largest side
  bool det2_half_in_ = true, det2_half_out_ = true;
  DeviceMem det2_f32_, det2_in_, det2_maps_;
  Pinned det2_host_;
  bool fused_ = false;
  // each model's input and output: float16, else float32
  bool orient_half_in_ = true, orient_half_out_ = true, det_half_in_ = true, det_half_out_ = true;
  bool rec_half_in_ = true;
  int classes_ = 0, k_ = 0;
  Stream orient_s_, det_s_, rec_s_;
  DeviceMem head_w_, head_b_;
  DeviceMem orient_f32_, orient_in_, orient_out_;
  Pinned orient_host_, orient_logits_;
  Pinned det_host_;
  DeviceMem det_u8_, det_in_, det_maps_;
  DeviceMem crops_, rec_in_, crop_widths_, features_, ids_, probs_, logits_;
  Stream copy_s_;          // the pages' uploads
  PageSlot slots_[2];
  int cur_ = 1;            // the current group's slot (set_pages takes the other, 0, first)
  struct UpJob {
    PageSlot* slot = nullptr;
    std::vector<GrayView> pages;
    std::vector<RowRuns> rows;
  };
  std::mutex up_m_;
  std::condition_variable up_cv_;
  UpJob up_job_;
  bool up_busy_ = false, up_stop_ = false;
  std::thread uploader_;  // started last in the constructor, so a constructor that throws has none to join
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
