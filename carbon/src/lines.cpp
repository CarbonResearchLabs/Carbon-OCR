// The crop rule and CTC greedy decoding on the CPU: the reference the CUDA code
// matches byte for byte (carbon/lines.h).  The resampling's coefficients are
// computed here, in double precision without contraction; its integer passes run
// on the CPU's vectors (carbon/src/cpu/kernels.h), the scalar loops kept as the
// reference they match byte for byte.

#include "carbon/lines.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "carbon/cpu.h"
#include "carbon/error.h"
#include "cpu/kernels.h"
#include "resample_math.h"

namespace carbon {

using detail::clip8;
using detail::coefficients;
using detail::filter_for;
using detail::kHalf;
using detail::max_taps;

CropRegion crop_region(const LineBox& b, const CropRule& r) {
  // The same expressions, in the same order, as the original Python crop rule.
  const double h = b.y1 - b.y0;
  const double sy0 = b.y0 - r.pad_v * h, sy1 = b.y1 + r.pad_v * h;
  const double scale = r.height / (sy1 - sy0);
  const double w_src = (b.x1 - b.x0) + 2 * r.pad_h * h;
  const int width = std::max(
      r.width_step,
      static_cast<int>(std::ceil(w_src * scale / r.width_step - 1e-6)) * r.width_step);
  const double extra = (width / scale - w_src) / 2;
  return {b.x0 - r.pad_h * h - extra, sy0, b.x1 + r.pad_h * h + extra, sy1, width};
}

int ctc_frames(const LineBox& box, const CropRule& rule) {
  return crop_region(box, rule).width / rule.frame_px;
}

namespace {

// Coefficients of one axis: per output sample, its first tap and count, and
// `taps` fixed-point coefficients.
struct Axis {
  int taps = 0;
  std::vector<int32_t> first, count, k;

  Axis(double in0, double in1, int out) : first(out), count(out) {
    const detail::Filter f = filter_for(in0, in1, out);
    taps = max_taps(f);
    k.assign(static_cast<size_t>(out) * taps, 0);
    for (int i = 0; i < out; ++i) {
      first[i] = coefficients(f, in0, i, &k[static_cast<size_t>(i) * taps], &count[i]);
    }
  }

  int most_taps() const {  // the most any sample has, at most `taps`
    int n = 1;
    for (int c : count) n = std::max(n, c);
    return n;
  }

  // Whether the windows move only forward (both ends nondecreasing), as they do for
  // any in1 >= in0; a reversed range is left to the scalar reference.
  bool forward() const {
    for (size_t i = 1; i < first.size(); ++i) {
      if (first[i] < first[i - 1] || first[i] + count[i] < first[i - 1] + count[i - 1]) return false;
    }
    return true;
  }
};

// The vector kernels of the two passes (carbon/src/cpu/kernels.h): avx2 for every
// x86-64 set, neon for Arm; none for the scalar reference.
struct ResampleKernels {
  void (*layout)(const int32_t*, const int32_t*, int, int, int, void*) = nullptr;
  void (*h)(const uint8_t*, int, const int32_t*, const void*, int, int, int, uint8_t*) = nullptr;
  void (*v)(const uint8_t* const*, const int32_t*, int, int, uint8_t*) = nullptr;
};

ResampleKernels resample_kernels(cpu::Isa isa) {
  namespace kd = cpu::detail;
  switch (isa) {
#if defined(CARBON_KERNEL_AVX2)
    case cpu::Isa::kAvx2:
    case cpu::Isa::kAvxVnni:
    case cpu::Isa::kAvx512: return {kd::resample_hk_avx2, kd::resample_h_avx2, kd::resample_v_avx2};
#endif
#if defined(CARBON_KERNEL_NEON)
    case cpu::Isa::kNeon:
    case cpu::Isa::kNeonDot: return {kd::resample_hk_neon, kd::resample_h_neon, kd::resample_v_neon};
#endif
    default: return {};
  }
}

// Columns [a, b) of an image row `width` wide into dst, paper outside the row (all
// of it for a null row: outside the image).
void paper_copy(const uint8_t* row, int width, int a, int b, uint8_t* dst) {
  int in0 = std::max(a, 0), in1 = std::min(b, width);
  if (!row || in1 <= in0) in0 = in1 = b;
  std::memset(dst, 255, static_cast<size_t>(in0 - a));
  if (in1 > in0) std::memcpy(dst + (in0 - a), row + in0, static_cast<size_t>(in1 - in0));
  std::memset(dst + (in1 - a), 255, static_cast<size_t>(b - in1));
}

// The scalar passes, the reference: the horizontal one over every row the vertical
// one reads, into an 8-bit intermediate, as in PIL.
void resample_scalar(const GrayView& im, const Axis& ax, const Axis& ay, int ow, int oh, uint8_t* out, int64_t os) {
  const int top = ay.first[0];
  const int rows = ay.first[oh - 1] + ay.count[oh - 1] - top;
  std::vector<uint8_t> tmp(static_cast<size_t>(rows) * ow);
  for (int r = 0; r < rows; ++r) {
    const int y = top + r;
    const uint8_t* row = (y >= 0 && y < im.height) ? im.data + y * im.stride : nullptr;
    uint8_t* t = &tmp[static_cast<size_t>(r) * ow];
    for (int xx = 0; xx < ow; ++xx) {
      const int32_t* k = &ax.k[static_cast<size_t>(xx) * ax.taps];
      const int xmin = ax.first[xx], n = ax.count[xx];
      int32_t ss = kHalf;
      for (int i = 0; i < n; ++i) {
        const int x = xmin + i;
        const int v = (row && x >= 0 && x < im.width) ? row[x] : 255;
        ss += v * k[i];
      }
      t[xx] = clip8(ss);
    }
  }
  for (int yy = 0; yy < oh; ++yy) {
    const int32_t* k = &ay.k[static_cast<size_t>(yy) * ay.taps];
    const uint8_t* src = &tmp[static_cast<size_t>(ay.first[yy] - top) * ow];
    const int n = ay.count[yy];
    uint8_t* o = out + yy * os;
    for (int xx = 0; xx < ow; ++xx) {
      int32_t ss = kHalf;
      for (int i = 0; i < n; ++i) ss += src[static_cast<size_t>(i) * ow + xx] * k[i];
      o[xx] = clip8(ss);
    }
  }
}

// The same passes on vectors.  Horizontal: each output sample reads 8 taps a chunk,
// as many chunks as the widest sample needs (its weights past its own taps are
// zeros), in groups of 8 samples, the last group padded with copies of the last
// sample's window.  The groups whose windows lie inside the image read its rows where
// they are; the groups at the ends, a copy of the columns they read with paper
// outside the image.  A row wholly outside the image is the same paper row each time,
// made once.  Vertical: an output row at a time, from the rows of its taps, which
// the horizontal pass keeps in a ring as tall as the tallest window: the windows only
// move down, so each row goes through the horizontal pass once, in order.
void resample_vector(const GrayView& im, const Axis& ax, const Axis& ay, int ow, int oh, uint8_t* out, int64_t os,
                     const ResampleKernels& kern) {
  const int chunks = (ax.most_taps() + 7) / 8, reach = 8 * chunks, groups = (ow + 7) / 8;
  std::vector<int32_t> first(static_cast<size_t>(groups) * 8);
  for (size_t i = 0; i < first.size(); ++i) first[i] = ax.first[std::min(i, static_cast<size_t>(ow - 1))];
  std::vector<int32_t> k(first.size() * reach);  // 32 bytes a sample a chunk
  kern.layout(ax.k.data(), ax.count.data(), ax.taps, ow, chunks, k.data());

  // Groups [0, ga) read columns [span0, left1), groups [gb, groups) columns [right0,
  // span1); those between, only columns of the image (first is nondecreasing).
  const int span0 = first.front(), span1 = first.back() + reach;
  int ga = 0, gb = 0;
  while (ga < groups && first[8 * ga] < 0) ++ga;
  for (gb = ga; gb < groups && first[8 * gb + 7] + reach <= im.width;) ++gb;
  const int left1 = ga > 0 ? first[8 * ga - 1] + reach : span0;
  const int right0 = gb < groups ? first[8 * gb] : span1;
  std::vector<uint8_t> edge(static_cast<size_t>(span1 - span0));  // column c at edge[c - span0]

  // The ring's rows, and the paper row after them, `stride` bytes apart: the vertical
  // kernel reads them to a multiple of 32.
  const int tall = ay.most_taps(), top = ay.first[0];
  const size_t stride = static_cast<size_t>((groups * 8 + 31) / 32 * 32);
  std::vector<uint8_t> ring(static_cast<size_t>(tall + 1) * stride);
  uint8_t* const paper = ring.data() + static_cast<size_t>(tall) * stride;
  auto inside = [&](int y) { return y >= 0 && y < im.height; };
  auto slot = [&](int y) { return inside(y) ? ring.data() + static_cast<size_t>((y - top) % tall) * stride : paper; };
  auto horizontal = [&](const uint8_t* row, uint8_t* t) {
    if (!row) {
      paper_copy(nullptr, 0, span0, span1, edge.data());
      kern.h(edge.data(), span0, first.data(), k.data(), chunks, 0, groups, t);
      return;
    }
    if (ga > 0) {
      paper_copy(row, im.width, span0, left1, edge.data());
      kern.h(edge.data(), span0, first.data(), k.data(), chunks, 0, ga, t);
    }
    if (gb > ga) kern.h(row, 0, first.data(), k.data(), chunks, ga, gb, t);
    if (gb < groups) {
      paper_copy(row, im.width, right0, span1, edge.data() + (right0 - span0));
      kern.h(edge.data(), span0, first.data(), k.data(), chunks, gb, groups, t);
    }
  };

  std::vector<const uint8_t*> rows(static_cast<size_t>(tall));
  bool have_paper = false;
  int next = top;  // the first row not through the horizontal pass yet
  for (int yy = 0; yy < oh; ++yy) {
    const int f = ay.first[yy], n = ay.count[yy];
    for (int y = std::max(next, f); y < f + n; ++y) {
      if (inside(y)) {
        horizontal(im.data + y * im.stride, slot(y));
      } else if (!have_paper) {
        horizontal(nullptr, paper);
        have_paper = true;
      }
    }
    next = std::max(next, f + n);
    for (int i = 0; i < n; ++i) rows[i] = slot(f + i);
    kern.v(rows.data(), &ay.k[static_cast<size_t>(yy) * ay.taps], n, ow, out + yy * os);
  }
}

}  // namespace

void resample_gray(const GrayView& im, double x0, double y0, double x1, double y1, int ow, int oh,
                   uint8_t* out, int64_t os, cpu::Isa isa) {
  if (ow <= 0 || oh <= 0) throw Error("resample_gray: empty output");
  if (!cpu::isa_available(isa)) {
    throw Error(std::string("resample_gray: ") + cpu::isa_name(isa) + " is not available on this CPU");
  }
  const Axis ax(x0, x1, ow), ay(y0, y1, oh);
  const ResampleKernels kern = resample_kernels(isa);
  if (kern.h && ax.forward() && ay.forward()) {
    resample_vector(im, ax, ay, ow, oh, out, os, kern);
  } else {
    resample_scalar(im, ax, ay, ow, oh, out, os);
  }
}

namespace {

// The bytes crop_ground takes the median of: the top and bottom rows and the right column (the corners twice).
std::array<int, 256> ground_histogram(const uint8_t* c, int w, int h, int64_t stride) {
  std::array<int, 256> hist{};
  for (int x = 0; x < w; ++x) {
    ++hist[c[x]];
    ++hist[c[(h - 1) * stride + x]];
  }
  for (int y = 0; y < h; ++y) ++hist[c[y * stride + w - 1]];
  return hist;
}

// Their median (the mean of the middle two, rounded down, of an even count); of the crop flipped (255 - each byte),
// read from the same histogram backwards.
uint8_t ground_of(const std::array<int, 256>& hist, int count, bool flipped) {
  auto nth = [&](int n) {  // the n-th smallest, from 0
    for (int v = 0, seen = 0; v < 256; ++v) {
      seen += hist[flipped ? 255 - v : v];
      if (seen > n) return v;
    }
    return 255;
  };
  return static_cast<uint8_t>(count % 2 ? nth(count / 2) : (nth(count / 2 - 1) + nth(count / 2)) / 2);
}

}  // namespace

uint8_t crop_ground(const uint8_t* c, int w, int h, int64_t stride) {
  return ground_of(ground_histogram(c, w, h, stride), 2 * w + h, false);
}

void crop_lines(const GrayView& page, std::span<const LineBox> boxes, int out_width, uint8_t* out,
                std::span<int32_t> widths, const CropRule& rule) {
  if (widths.size() < boxes.size()) throw Error("crop_lines: widths is shorter than boxes");
  const size_t plane = static_cast<size_t>(rule.height) * out_width;
  for (size_t i = 0; i < boxes.size(); ++i) {
    const CropRegion r = crop_region(boxes[i], rule);
    if (r.width > out_width) {
      throw Error("crop_lines: line " + std::to_string(i) + " is " + std::to_string(r.width) +
                  " px wide, the batch " + std::to_string(out_width));
    }
    uint8_t* o = out + i * plane;
    resample_gray(page, r.x0, r.y0, r.x1, r.y1, r.width, rule.height, o, out_width);
    finish_crop(o, r.width, out_width, rule);
    widths[i] = r.width;
  }
}

namespace {

// light_on_dark's test on the crop's ground g: the CPU's vectors count its bytes (cpu::count_beyond).
bool lighter_than_darker(const uint8_t* c, int w, int h, int64_t stride, int g, int margin) {
  const cpu::ByteCounts n = cpu::count_beyond(c, w, h, stride, g - margin, g + margin);
  return n.above > n.below;
}

}  // namespace

bool light_on_dark(const uint8_t* c, int w, int h, int64_t stride, int margin) {
  if (margin <= 0 || w <= 0 || h <= 0) return false;
  return lighter_than_darker(c, w, h, stride, crop_ground(c, w, h, stride), margin);
}

void finish_crop(uint8_t* c, int w, int out_width, const CropRule& rule) {
  const int h = rule.height;
  if (w <= 0 || h <= 0) return;
  // The ground's histogram once: the test's ground, and the padding's (the flipped crop's, the histogram backwards).
  const std::array<int, 256> hist = ground_histogram(c, w, h, out_width);
  const bool flip = rule.flip_margin > 0 &&
                    lighter_than_darker(c, w, h, out_width, ground_of(hist, 2 * w + h, false), rule.flip_margin);
  if (flip) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) c[y * out_width + x] = static_cast<uint8_t>(255 - c[y * out_width + x]);
  }
  if (w < out_width) {
    const uint8_t ground = ground_of(hist, 2 * w + h, flip);
    for (int y = 0; y < h; ++y) std::fill(c + y * out_width + w, c + (y + 1) * out_width, ground);
  }
}

std::vector<CtcLine> ctc_collapse(const int32_t* ids, const float* probs, int n, int frames,
                                  std::span<const int32_t> lengths) {
  if (static_cast<int>(lengths.size()) < n) throw Error("ctc_collapse: lengths is shorter than the batch");
  std::vector<CtcLine> out(n);
  for (int i = 0; i < n; ++i) {
    const int t_n = std::clamp(lengths[i], 0, frames);
    const int32_t* a = ids + static_cast<size_t>(i) * frames;
    const float* q = probs ? probs + static_cast<size_t>(i) * frames : nullptr;
    CtcLine& l = out[i];
    int prev = 0;
    for (int t = 0; t < t_n; ++t) {
      const int id = a[t];
      const float p = q ? q[t] : 1.0f;
      if (id != 0 && id != prev) {
        l.ids.push_back(id);
        l.frames.push_back(t);
        l.probs.push_back(p);
      } else if (id != 0) {
        l.probs.back() = std::max(l.probs.back(), p);
      }
      prev = id;
    }
  }
  return out;
}

std::vector<CtcLine> ctc_greedy(const float* logits, int n, int classes, int frames,
                                std::span<const int32_t> lengths) {
  if (static_cast<int>(lengths.size()) < n) throw Error("ctc_greedy: lengths is shorter than the batch");
  std::vector<int32_t> arg(static_cast<size_t>(n) * frames, 0);
  std::vector<float> prob(static_cast<size_t>(n) * frames, 0.0f);
  // Class by class, so the reads run along the frames (the layout's fast axis), on the
  // CPU's vectors (carbon/cpu.h): the best class of each frame (ties to the lower id),
  // then the softmax normalizer only where a character won; most frames are blank.
  for (int i = 0; i < n; ++i) {
    cpu::ctc_argmax(logits + static_cast<size_t>(i) * classes * frames, classes, frames, std::clamp(lengths[i], 0, frames),
                    &arg[static_cast<size_t>(i) * frames], &prob[static_cast<size_t>(i) * frames]);
  }
  return ctc_collapse(arg.data(), prob.data(), n, frames, lengths);
}

CtcHead load_ctc_head(const std::filesystem::path& file) {
  static_assert(std::endian::native == std::endian::little, "the head file is little-endian");
  std::ifstream f(file, std::ios::binary);
  if (!f) throw Error("cannot read " + file.string());
  const std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  constexpr size_t kHeader = 16;
  if (bytes.size() < kHeader || std::memcmp(bytes.data(), "CTCHEAD1", 8) != 0) {
    throw Error(file.string() + " is not a CTC head (the export writes <model>.head.bin)");
  }
  CtcHead h;
  int32_t dims[2];
  std::memcpy(dims, bytes.data() + 8, sizeof dims);
  h.classes = dims[0];
  h.k = dims[1];
  const auto cells = static_cast<size_t>(std::max(h.classes, 0)) * static_cast<size_t>(std::max(h.k, 0));
  const size_t want = kHeader + cells * 2 + static_cast<size_t>(std::max(h.classes, 0)) * 4;
  if (h.classes <= 0 || h.k <= 0 || bytes.size() != want) {
    throw Error(file.string() + ": a head of " + std::to_string(h.classes) + " classes by " +
                std::to_string(h.k) + " takes " + std::to_string(want) + " bytes, the file has " +
                std::to_string(bytes.size()));
  }
  h.weight.resize(cells);
  h.bias.resize(static_cast<size_t>(h.classes));
  std::memcpy(h.weight.data(), bytes.data() + kHeader, cells * 2);
  std::memcpy(h.bias.data(), bytes.data() + kHeader + cells * 2, h.bias.size() * 4);
  return h;
}

}  // namespace carbon
