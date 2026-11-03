// The page turns (carbon/cpu.h): the scalar references, the columns' terms the
// vector kernels read, dispatch.  Built without fused multiply-adds, as the
// kernels are, so the reference rounds as they do on every CPU.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "carbon/error.h"
#include "kernels.h"

namespace carbon::cpu {
namespace {

// The kernels: avx2 for every x86-64 set, neon for Arm.
Isa turn_kernel(Isa isa) {
  switch (isa) {
    case Isa::kAvx2:
    case Isa::kAvxVnni:
    case Isa::kAvx512: return Isa::kAvx2;
    case Isa::kNeon:
    case Isa::kNeonDot: return Isa::kNeon;
    default: return Isa::kScalar;
  }
}

void check(Isa isa, const char* what) {
  if (!isa_available(isa)) throw Error(std::string(what) + ": " + isa_name(isa) + " is not available on this CPU");
}

// The reference, a pixel at a time: where it was on the page, turned back
// clockwise, and its four taps there.
void rotate_bilinear_scalar(const uint8_t* src, int width, int height, int64_t stride, double s, double c,
                            uint8_t* out, int row0, int row1) {
  const double cx = (width - 1) / 2.0, cy = (height - 1) / 2.0;
  for (int y = row0; y < row1; ++y) {
    uint8_t* dst = out + static_cast<int64_t>(y) * width;
    const double dy = y - cy;
    for (int x = 0; x < width; ++x) {
      const double dx = x - cx;
      const double sx = dx * c - dy * s + cx, sy = dx * s + dy * c + cy;
      const int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
      if (x0 < -1 || y0 < -1 || x0 >= width || y0 >= height) continue;
      const double fx = sx - x0, fy = sy - y0;
      auto at = [&](int xx, int yy) -> double {
        if (xx < 0 || yy < 0 || xx >= width || yy >= height) return 255.0;
        return src[yy * stride + xx];
      };
      const double v = (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) +
                       (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
      dst[x] = static_cast<uint8_t>(std::lround(std::clamp(v, 0.0, 255.0)));
    }
  }
}

// The reference, a byte at a time to where it lands.
void rotate_quarters_scalar(const uint8_t* src, int w, int h, int64_t stride, int q, uint8_t* out) {
  const int out_w = q % 2 ? h : w;
  for (int y = 0; y < h; ++y) {
    const uint8_t* row = src + y * stride;
    for (int x = 0; x < w; ++x) {
      int nx = x, ny = y;  // where (x, y) lands, turned clockwise q times
      if (q == 1) {
        nx = h - 1 - y;
        ny = x;
      } else if (q == 2) {
        nx = w - 1 - x;
        ny = h - 1 - y;
      } else if (q == 3) {
        nx = y;
        ny = w - 1 - x;
      }
      out[static_cast<size_t>(ny) * out_w + nx] = row[x];
    }
  }
}

}  // namespace

void rotate_bilinear(const uint8_t* src, int width, int height, int64_t stride, double s, double c, uint8_t* dst,
                     Isa isa) {
  rotate_bilinear_rows(src, width, height, stride, s, c, dst, 0, height, isa);
}

void rotate_bilinear_rows(const uint8_t* src, int width, int height, int64_t stride, double s, double c, uint8_t* dst,
                          int row0, int row1, Isa isa) {
  row0 = std::max(row0, 0);
  row1 = std::min(row1, height);
  if (width <= 0 || height <= 0 || row0 >= row1) return;
  check(isa, "rotate_bilinear");
  const Isa kern = turn_kernel(isa);
  if (kern == Isa::kScalar) {
    rotate_bilinear_scalar(src, width, height, stride, s, c, dst, row0, row1);
    return;
  }
  // Each column's dx c and dx s, the products the reference makes for every pixel.
  const double cx = (width - 1) / 2.0, cy = (height - 1) / 2.0;
  std::vector<double> terms(2 * static_cast<size_t>(width));
  for (int x = 0; x < width; ++x) {
    const double dx = x - cx;
    terms[x] = dx * c;
    terms[width + x] = dx * s;
  }
  const detail::TurnView t{src, width, height, stride, s, c, cx, cy, terms.data(), terms.data() + width, row0, row1};
  switch (kern) {
#if defined(CARBON_KERNEL_AVX2)
    case Isa::kAvx2: detail::rotate_bilinear_avx2(t, dst); return;
#endif
#if defined(CARBON_KERNEL_NEON)
    case Isa::kNeon: detail::rotate_bilinear_neon(t, dst); return;
#endif
    default: throw Error(std::string("rotate_bilinear: no kernel for ") + isa_name(isa) + " in this build");
  }
}

void rotate_quarters(const uint8_t* src, int width, int height, int64_t stride, int q, uint8_t* dst, Isa isa) {
  if (width <= 0 || height <= 0) return;
  if (q < 0 || q > 3) throw Error("rotate_quarters: q must be 0 to 3");
  check(isa, "rotate_quarters");
  switch (turn_kernel(isa)) {
    case Isa::kScalar: rotate_quarters_scalar(src, width, height, stride, q, dst); return;
#if defined(CARBON_KERNEL_AVX2)
    case Isa::kAvx2: detail::rotate_quarters_avx2(src, width, height, stride, q, dst); return;
#endif
#if defined(CARBON_KERNEL_NEON)
    case Isa::kNeon: detail::rotate_quarters_neon(src, width, height, stride, q, dst); return;
#endif
    default: throw Error(std::string("rotate_quarters: no kernel for ") + isa_name(isa) + " in this build");
  }
}

}  // namespace carbon::cpu
