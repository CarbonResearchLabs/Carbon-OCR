// The two seams between the pages and the recognizers:
// the line crop rule and CTC greedy decoding.  Training crops and decodes
// through this same code (the Python bindings, python/), so what a recognizer
// reads in training is byte for byte what it reads in the engine.
//
// Each has a CPU implementation here and a CUDA one (carbon/cuda.h) that gives
// the same bytes: the resampling is integer arithmetic on coefficients computed
// in IEEE double precision without fused multiply-adds.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "carbon/cpu.h"

namespace carbon {

// A line box of height h becomes a crop `height` px high: h * pad_v of padding
// above and below, h * pad_h at each end, scaled to `height`, then widened on
// both sides (into the page) to a multiple of `width_step`.
struct CropRule {
  int height = 32;      // px
  int width_step = 8;   // the backbone's stride
  int frame_px = 4;     // px of crop width per CTC frame
  double pad_v = 0.2;
  double pad_h = 0.3;
  // Light text on a dark ground is flipped (255 - v), so a recognizer always reads
  // dark ink on light paper: a crop with more pixels at least flip_margin lighter
  // than its ground than pixels as much darker (carbon::light_on_dark).  0: never.
  int flip_margin = 48;
  // A crop is flipped only where over half its pixels are at most 127 as well: where both padding rows land
  // on a table's rules, the ground reads dark though the crop is dark text on paper. On by default; false: the
  // ground's test alone.
  bool flip_needs_dark = true;
};

// A box in page pixels, x right, y down.
struct LineBox {
  double x0, y0, x1, y1;
};

// The page rectangle a box's crop resamples, and the crop's width.
struct CropRegion {
  double x0, y0, x1, y1;
  int width;
};

CropRegion crop_region(const LineBox& box, const CropRule& rule = {});
int ctc_frames(const LineBox& box, const CropRule& rule = {});

// A grayscale image in host memory, paper 255 and ink 0, rows `stride` bytes apart.
struct GrayView {
  const uint8_t* data;
  int width, height;
  int64_t stride;
};

// The resampling of the crop rule: the rectangle [x0, x1) x [y0, y1) of `image`
// to out_width x out_height.  A triangle filter whose support widens with the
// downscale (PIL's BILINEAR, so the crops training has always used), with
// coefficients in 22-bit fixed point: a horizontal pass into an 8-bit
// intermediate, then a vertical pass.  Pixels outside the image read as paper.
// `out` is row-major, rows `out_stride` bytes apart.  The bytes equal Pillow's
// resize(box=...) whenever Pillow runs its horizontal pass first, which it does
// unless the whole image is much taller than it is wide relative to the output
// (in_h - out_h > 2 (in_w - out_w)): never for a line canvas with margins, so
// training has always read these bytes.  The rule itself never depends on the
// size of the image around the box.
//
// The integer passes run on the CPU's vectors (carbon/cpu.h: avx2 for every x86-64
// set, neon for Arm), which give the scalar reference's bytes: the sums are exact
// in 32 bits whatever their order.  isa: any available one, kScalar for the
// reference itself (tests, benchmarks).
void resample_gray(const GrayView& image, double x0, double y0, double x1, double y1, int out_width,
                   int out_height, uint8_t* out, int64_t out_stride, cpu::Isa isa = cpu::best_isa());
// Its output rows [row0, row1) alone (a band of a page at another scale), out's first row the output's row0:
// the same bytes as those rows of the whole output.
void resample_gray_rows(const GrayView& image, double x0, double y0, double x1, double y1, int out_width,
                        int out_height, int row0, int row1, uint8_t* out, int64_t out_stride,
                        cpu::Isa isa = cpu::best_isa());

// Crops `boxes` from `page` into out[n][rule.height][out_width] (uint8, paper
// 255), each finished by finish_crop: flipped if light on dark, then padded past
// its own width with its own ground.  widths[i] receives crop i's own width.
// Throws Error if a crop is wider than out_width.
void crop_lines(const GrayView& page, std::span<const LineBox> boxes, int out_width, uint8_t* out,
                std::span<int32_t> widths, const CropRule& rule = {});

// The ground the batch padding uses: the median of a crop's first row, last row
// and last column (for an even count, the two middle values averaged, rounded
// down).
uint8_t crop_ground(const uint8_t* crop, int width, int height, int64_t stride);

// Whether a crop is light text on a dark ground: more of its pixels at least
// `margin` lighter than its ground (crop_ground) than at least as much darker.
// Never with a margin of 0.
bool light_on_dark(const uint8_t* crop, int width, int height, int64_t stride, int margin);

// The crop rule's last step, for a crop `width` wide in rows `out_width` apart:
// flipped (255 - v) if it is light on dark (rule.flip_margin), then padded to
// out_width with its ground.  crop_lines finishes each crop with it.
void finish_crop(uint8_t* crop, int width, int out_width, const CropRule& rule = {});

// One decoded line: its class ids (blanks dropped, repeats collapsed), the
// frame each character starts at, and its probability (the highest softmax
// probability over the frames of its run).
struct CtcLine {
  std::vector<int32_t> ids, frames;
  std::vector<float> probs;
};

// CTC greedy decoding of logits [n][classes][frames] (row-major float32, the
// recognizers' output layout).  Line i reads its first lengths[i] frames.
// Class 0 is the blank.  Ties go to the lower class id.
std::vector<CtcLine> ctc_greedy(const float* logits, int n, int classes, int frames,
                                std::span<const int32_t> lengths);

// The second half of ctc_greedy, for a best class per frame already found (on
// a GPU, say): ids [n][frames], and their probabilities (or null: all 1).
std::vector<CtcLine> ctc_collapse(const int32_t* ids, const float* probs, int n, int frames,
                                  std::span<const int32_t> lengths);

// A recognizer's last 1x1 convolution, which the GPU runs fused with the
// greedy decode (carbon::cuda::ctc_head_argmax): <model>.head.bin, written by
// the export as "CTCHEAD1", int32 classes, int32 k, the weight, the bias.
struct CtcHead {
  int classes = 0, k = 0;
  std::vector<uint16_t> weight;  // float16 bits, [classes][k]
  std::vector<float> bias;       // [classes]
};

// Throws Error if the file is missing, is not a head, or has the wrong size.
CtcHead load_ctc_head(const std::filesystem::path& file);

}  // namespace carbon
