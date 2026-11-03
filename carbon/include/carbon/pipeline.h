// The OCR pipeline for many pages at once: pages go in and their
// results come out in the order given, while the stages work on the pages in
// flight, each model batched across them.
//
//   prepare     CPU threads: each page at the detector's scale, and orientation's
//               thumbnail of it
//   orient      the thumbnails of up to page_batch pages in one run; the turn each
//               asks for (only a quarter turn, only when sure)
//   upright     CPU threads: the page turned and levelled (deskew), its detector input
//   detect      up to page_batch pages of one padded size in one run, the maps
//               thresholded (on the GPU: cuda::threshold_det, only the masks come
//               back) and decoded into lines on the upright page
//   recognize   the lines of every page waiting, from any of them, by width bucket,
//               up to line_batch a run: cropped (on the GPU: cuda::crop_lines, every
//               line of every page in one set of launches), read, CTC-decoded (on the
//               GPU fused with the head's last convolution: cuda::ctc_head_argmax,
//               only ids and probabilities come back)
//   assemble    CPU threads: lines split at wide gaps, reading order by the order
//               model or the XY-cut, blocks and tables
//
// Each model stage is a thread of its own (on the GPU a CUDA stream each), joined
// by bounded queues, so orientation, detection and recognition of different
// pages run at once; at most in_flight pages are in the pipeline, and submit
// waits for room.  On the CPU (Device::kCpu) the models run on the set's files as
// the CPU engine opens them, multi-threaded, and a page's result is the CPU
// engine's (carbon/engine.h; the same steps, engine_steps.h).  On a GPU
// (Device::kCuda: CARBON_WITH_CUDA and TensorRT) they run on TensorRT from the
// set's FP16 files, detection padded to a few fixed sizes (1024 x 800, 800 x 1024,
// 1024 x 1024 at a long side of 1024) and batched in powers of two up to
// page_batch, each an exact profile (a CUDA graph).
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "carbon/engine.h"
#include "carbon/image.h"

namespace carbon {

enum class Device : uint8_t { kCpu, kCuda };

struct PipelineOptions {
  EngineOptions engine;          // the recognizer, the crop rule, orientation, deskew, order, splitting, tables;
                                 // engine.backend the CPU device's models (none: the fastest the set has)
  std::optional<Device> device;  // none: CUDA where it is built in with TensorRT and a GPU is there, else the CPU
  int page_batch = 8;            // pages an orientation or detection run holds at most
  int line_batch = 256;          // lines a recognition run holds at most
  int in_flight = 32;            // pages in the pipeline at once: submit waits past it
  int threads = 0;               // CPU threads preparing and assembling pages; 0: the physical cores
  int model_threads = 0;         // the CPU device: threads a model run uses; 0: the physical cores
  // CUDA
  int gpu = 0;
  std::filesystem::path engine_cache;  // TensorRT's built engines; empty: built at every open (minutes)
  bool fused_head = true;  // the recognizer's trunk and cuda::ctc_head_argmax where the set has
                           // <rec>.features.fp16.onnx and <rec>.head.bin; else <rec>.onnx and cuda::ctc_greedy
};

// What the pipeline has done: its runs and what they held.
struct PipelineStats {
  int64_t pages = 0, lines = 0;
  int64_t orient_runs = 0, detect_runs = 0, recognize_runs = 0;
  double pages_per_detect_run() const { return detect_runs ? 1.0 * pages / detect_runs : 0.0; }
  double lines_per_recognize_run() const { return recognize_runs ? 1.0 * lines / recognize_runs : 0.0; }
};

// The pages' models, on one device.  The pipeline's stages call each from one
// thread of their own; implementations hold the sessions and buffers.
class PipelineModels {
 public:
  virtual ~PipelineModels() = default;
  virtual Device device() const = 0;
  // Orientation logits, [n][13], for n thumbnails ([n][side][side] ink maps).
  virtual std::vector<float> orient(const float* thumbs, int n, int side) = 0;
  // The size, (height, width), a page at the detector's scale is padded to for a
  // detection run (pages batch together only at the same size): multiples of 32.
  virtual std::pair<int, int> detect_size(int width, int height) const = 0;
  // The detections of pages at the detector's scale, all of detect_size (ph x pw),
  // at that scale.
  virtual std::vector<Detection> detect(std::span<const GrayView> pages, int ph, int pw, const DetRule& rule) = 0;
  // The pages the recognize runs that follow crop from (a group of upright pages,
  // alive until the next call): on a GPU, uploaded once for all of them.
  virtual void set_pages(std::span<const GrayView> pages) = 0;
  // The lines boxes[i] of page page_of[i] (set_pages'), cropped by `crop` at bucket
  // width bw: their CTC decodings and the frames read (each crop's own width /
  // frame_px, at most the model's frames).
  virtual void recognize(std::span<const int32_t> page_of, std::span<const LineBox> boxes, int bw,
                         const CropRule& crop, std::vector<CtcLine>& lines, std::vector<int32_t>& lengths) = 0;
};

class Pipeline {
 public:
  explicit Pipeline(const std::filesystem::path& set, PipelineOptions options = {});
  ~Pipeline();
  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  // Gives the pipeline a page, its number from 0 in the order given; waits while
  // in_flight pages are in it.  Throws Error after close, or once a stage failed.
  size_t submit(GrayImage page);
  // No more pages: next() returns nullopt once every page given is out.
  void close();
  // The next page's result, in the order given; waits for it.  Throws the first
  // stage's error.
  std::optional<OcrPage> next();
  // Every page through, and their results in their order (copies the pages).
  std::vector<OcrPage> ocr(std::span<const GrayView> pages);

  Device device() const;
  PipelineStats stats() const;
  const CropRule& crop() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The CPU device's models (Device::kCpu): the set's files as the CPU engine opens
// them, each run on `threads` threads.
std::unique_ptr<PipelineModels> cpu_pipeline_models(const std::filesystem::path& set, const EngineOptions& o,
                                                    int threads);
// The GPU's (Device::kCuda): TensorRT, CUDA streams and the cuda:: kernels; throws
// Error where CUDA or TensorRT is not built in.
std::unique_ptr<PipelineModels> cuda_pipeline_models(const std::filesystem::path& set, const EngineOptions& o,
                                                     const PipelineOptions& p);
// Whether this build can run the pipeline on a GPU (CUDA and TensorRT in, a device there).
bool cuda_pipeline_available();

}  // namespace carbon
