// The crop rule and CTC greedy decoding on the CPU: the reference the CUDA code
// matches byte for byte (carbon/lines.h).

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
};

}  // namespace

void resample_gray(const GrayView& im, double x0, double y0, double x1, double y1, int ow, int oh,
                   uint8_t* out, int64_t os) {
  if (ow <= 0 || oh <= 0) throw Error("resample_gray: empty output");
  const Axis ax(x0, x1, ow), ay(y0, y1, oh);
  // The horizontal pass covers the rows the vertical one reads, as in PIL.
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

uint8_t crop_ground(const uint8_t* c, int w, int h, int64_t stride) {
  std::array<int, 256> hist{};
  for (int x = 0; x < w; ++x) {
    ++hist[c[x]];
    ++hist[c[(h - 1) * stride + x]];
  }
  for (int y = 0; y < h; ++y) ++hist[c[y * stride + w - 1]];
  const int m = 2 * w + h;
  auto nth = [&](int n) {  // the n-th smallest, from 0
    for (int v = 0, seen = 0; v < 256; ++v) {
      seen += hist[v];
      if (seen > n) return v;
    }
    return 255;
  };
  return static_cast<uint8_t>(m % 2 ? nth(m / 2) : (nth(m / 2 - 1) + nth(m / 2)) / 2);
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

bool light_on_dark(const uint8_t* c, int w, int h, int64_t stride, int margin) {
  if (margin <= 0 || w <= 0 || h <= 0) return false;
  const int g = crop_ground(c, w, h, stride);
  int64_t lighter = 0, darker = 0;
  for (int y = 0; y < h; ++y) {
    const uint8_t* row = c + y * stride;
    for (int x = 0; x < w; ++x) {
      lighter += row[x] >= g + margin;
      darker += row[x] <= g - margin;
    }
  }
  return lighter > darker;
}

void finish_crop(uint8_t* c, int w, int out_width, const CropRule& rule) {
  const int h = rule.height;
  if (light_on_dark(c, w, h, out_width, rule.flip_margin)) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) c[y * out_width + x] = static_cast<uint8_t>(255 - c[y * out_width + x]);
  }
  if (w < out_width) {
    const uint8_t ground = crop_ground(c, w, h, out_width);
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
