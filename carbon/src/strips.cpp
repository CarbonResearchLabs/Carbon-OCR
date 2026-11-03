// Detection in strips (carbon/strips.h).

#include "carbon/strips.h"

#include <algorithm>
#include <cstring>
#include <string>

#include "carbon/error.h"

namespace carbon {

namespace {

// Copies rows [r0, r0 + rows) of every [N, C] plane of `from` (H_from rows) to
// rows [d0, d0 + rows) of `to` (H_to rows); both [N, C, H, W] of one dtype.
void copy_rows(const Tensor& from, int64_t r0, Tensor& to, int64_t d0, int64_t rows) {
  const Shape& a = from.shape();
  const Shape& b = to.shape();
  const size_t row = static_cast<size_t>(a[3]) * dtype_size(from.dtype());
  const auto* src = static_cast<const std::byte*>(from.raw());
  auto* dst = static_cast<std::byte*>(to.raw());
  for (int64_t plane = 0; plane < a[0] * a[1]; ++plane) {
    std::memcpy(dst + (plane * b[2] + d0) * row, src + (plane * a[2] + r0) * row, rows * row);
  }
}

}  // namespace

std::vector<Strip> plan_strips(int height, int strips, const StripRule& rule) {
  if (rule.align <= 0 || rule.out_stride <= 0 || rule.align % rule.out_stride || rule.halo < 0 ||
      rule.halo % rule.align) {
    throw Error("plan_strips: the halo must be a multiple of align, and align of out_stride");
  }
  if (height <= 0 || height % rule.align) {
    throw Error("plan_strips: the page's height (" + std::to_string(height) + ") must be a multiple of " +
                std::to_string(rule.align));
  }
  const int blocks = height / rule.align;
  const int n = std::clamp(strips, 1, blocks);
  std::vector<Strip> plan;
  for (int k = 0; k < n; ++k) {
    // Cuts on whole blocks, as even as they go.
    const int c0 = k * blocks / n * rule.align;
    const int c1 = (k + 1) * blocks / n * rule.align;
    Strip s;
    s.y0 = std::max(0, c0 - rule.halo);
    s.y1 = std::min(height, c1 + rule.halo);
    s.keep0 = (c0 - s.y0) / rule.out_stride;
    s.keep1 = s.keep0 + (c1 - c0) / rule.out_stride;
    s.out0 = c0 / rule.out_stride;
    plan.push_back(s);
  }
  return plan;
}

Tensor strip_input(const Tensor& page, const Strip& strip) {
  const Shape& s = page.shape();
  if (s.size() != 4 || strip.y0 < 0 || strip.y1 > s[2] || strip.y0 >= strip.y1) {
    throw Error("strip_input: rows [" + std::to_string(strip.y0) + ", " + std::to_string(strip.y1) +
                ") of a page " + shape_string(s));
  }
  Tensor t = Tensor::empty(page.dtype(), {s[0], s[1], strip.y1 - strip.y0, s[3]});
  copy_rows(page, strip.y0, t, 0, strip.y1 - strip.y0);
  return t;
}

void stitch_strip(const Tensor& strip_out, const Strip& strip, Tensor& page_out) {
  const Shape& a = strip_out.shape();
  const Shape& b = page_out.shape();
  const int rows = strip.keep1 - strip.keep0;
  if (a.size() != 4 || b.size() != 4 || a[0] != b[0] || a[1] != b[1] || a[3] != b[3] ||
      strip_out.dtype() != page_out.dtype() || strip.keep0 < 0 || strip.keep1 > a[2] ||
      strip.out0 < 0 || strip.out0 + rows > b[2]) {
    throw Error("stitch_strip: a strip's output " + shape_string(a) + " does not fit the page's " +
                shape_string(b));
  }
  copy_rows(strip_out, strip.keep0, page_out, strip.out0, rows);
}

Tensor run_in_strips(const Tensor& page, int strips, const std::function<Tensor(const Tensor&)>& run,
                     const StripRule& rule) {
  const Shape& s = page.shape();
  if (s.size() != 4) throw Error("run_in_strips: a page " + shape_string(s) + ", not [N, C, H, W]");
  if (strips <= 1) return run(page);
  Tensor out;
  for (const Strip& st : plan_strips(static_cast<int>(s[2]), strips, rule)) {
    const Tensor y = run(strip_input(page, st));
    if (out.shape().empty()) {
      const Shape& ys = y.shape();
      if (ys.size() != 4) throw Error("run_in_strips: an output " + shape_string(ys) + ", not [N, C, H, W]");
      out = Tensor::empty(y.dtype(), {ys[0], ys[1], s[2] / rule.out_stride, ys[3]});
    }
    stitch_strip(y, st, out);
  }
  return out;
}

}  // namespace carbon
