// The crop rule, CTC greedy decoding and the detector's threshold on an NVIDIA GPU:
// the same bytes as the CPU code of carbon/lines.h and carbon/detect.h, for pages,
// batches and tensors already in device memory.  Lines from many pages are cropped by one
// set of kernel launches.
//
// Built when CARBON_WITH_CUDA is on.  Streams are passed as void* (a
// cudaStream_t), so this header needs no CUDA headers; nullptr is the default
// stream.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "carbon/detect.h"
#include "carbon/lines.h"

namespace carbon::cuda {

// Whether a CUDA device is there to run on.
bool available();

// A grayscale page in device memory, paper 255, rows `stride` bytes apart.
struct DevicePage {
  const uint8_t* data;
  int width, height;
  int64_t stride;
};

// Crops boxes[i] from pages[page_of[i]] into out[n][rule.height][out_width]
// (device memory, uint8, paper 255), each finished as carbon::finish_crop does
// (light on dark flipped, then padded past its own width with its ground),
// exactly as carbon::crop_lines.  widths (device, int32[n]) receives each crop's
// own width.  out_width and rule.width_step are multiples of 4.  Throws Error if
// a crop is wider than out_width.
void crop_lines(std::span<const DevicePage> pages, std::span<const int32_t> page_of,
                std::span<const LineBox> boxes, int out_width, uint8_t* out, int32_t* widths,
                const CropRule& rule = {}, void* stream = nullptr);

// carbon::finish_crop for each crop of out[n][rule.height][out_width] in device
// memory, crop i widths[i] wide (host; multiples of 4, as out_width).  Asynchronous
// on `stream`.
void finish_crops(uint8_t* out, int n, int out_width, std::span<const int32_t> widths, const CropRule& rule = {},
                  void* stream = nullptr);

// The detector's maps for n pages in device memory, [n][4][map_h][map_w], float32
// or float16 (TensorRT's det.fp16.onnx), thresholded as carbon::threshold_det
// does: the same masks, but only the bits and the set pixels' values cross to the
// host.  Waits for the stream.
std::vector<DetMasks> threshold_det(const void* maps, bool fp16, int n, int map_h, int map_w, const DetRule& rule = {},
                                    void* stream = nullptr);

// CTC greedy decoding of device logits [n][classes][frames] (float32), as
// carbon::ctc_greedy.  The ids and frames are identical; the probabilities
// agree to the last few bits (the GPU's exp is not the CPU's).
std::vector<CtcLine> ctc_greedy(const float* logits, int n, int classes, int frames,
                                std::span<const int32_t> lengths, void* stream = nullptr);

// The CTC head fused with greedy decoding: logits = weight x features + bias on
// the tensor cores, reduced to each frame's best class and softmax probability
// as they are made, so the logits never leave the chip.  features [n][k][frames]
// (fp16, the mixing blocks' output), weight [classes][k] (fp16), bias [classes]
// (float32); k is 128, 192, 256 or 320.  Writes ids [n][frames] (int32) and
// probs [n][frames] (float32), all in device memory; carbon::ctc_collapse turns
// them into lines.  Asynchronous on `stream`.
//
// fp16_accumulate: each 64 products are summed in fp16 by the tensor cores
// (twice their fp32-accumulating rate on GeForce GPUs) and the chunks in fp32.
// The logits move by about 1e-3, so it is for a model whose
// text it leaves the same.
void ctc_head_argmax(const void* features, const void* weight, const float* bias, int n, int k, int classes,
                     int frames, int32_t* ids, float* probs, void* stream = nullptr,
                     bool fp16_accumulate = false);

// The beam's candidates (carbon/beam.h) on the GPU, for the lines it reads again (src/cuda/beam.cu).  ctc_topk:
// of device logits [n][classes][frames] (float32), lines rows[j] (device int32, n_rows of them), each its first
// lengths[j] frames (device int32): ids [n_rows][frames][k] (int32), logp [n_rows][frames][k] and blank
// [n_rows][frames] (float32), all device memory, as cpu::ctc_topk gives them but for the GPU's exp (the last few
// bits); k at most 16.  ctc_head_logits: the fused head's logits (ctc_head_argmax's features, weight and bias) of
// lines rows[j]: logits [n_rows][classes][frames] (float32), each summed in float32 in k's order, within about 1e-3
// of the tensor cores' sums.  Both asynchronous on `stream`.
void ctc_topk(const float* logits, int classes, int frames, const int32_t* rows, const int32_t* lengths, int n_rows,
              int k, int32_t* ids, float* logp, float* blank, void* stream = nullptr);
void ctc_head_logits(const void* features, const void* weight, const float* bias, const int32_t* rows, int n_rows,
                     int k, int classes, int frames, float* logits, void* stream = nullptr);

// A model's input from gray pixels in device memory: n bytes in, n ink values out
// (1 - v / 255: ink 1, paper 0), float16 when fp16 (TensorRT's .fp16.onnx inputs)
// else float32; the CPU's carbon::ink_input values, rounded to float16 once.
// Asynchronous on `stream`.
void ink_map(const uint8_t* in, int64_t n, void* out, bool fp16, void* stream = nullptr);

// n float32 values to float16 (round to nearest even), in device memory.
// Asynchronous on `stream`.
void to_half(const float* in, void* out, int64_t n, void* stream = nullptr);

}  // namespace carbon::cuda
