// The resampling coefficients of the crop rule, one source for the CPU and the
// CUDA code.  This is PIL's precompute_coeffs and normalize_coeffs_8bpc for a
// bilinear (triangle) filter, on an image with no edges (outside is paper), so
// the crops are the ones training has always read.
//
// The CPU and the GPU agree to the bit only if every double operation rounds
// the same way: IEEE-754 does that for + - * / and floor, as long as nothing is
// fused into an FMA.  The CUDA file is compiled with -fmad=false, and the C++
// one with contraction off.
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define CARBON_HD __host__ __device__
#else
#define CARBON_HD
#endif

namespace carbon::detail {

constexpr int kPrecisionBits = 32 - 8 - 2;  // PIL's PRECISION_BITS
constexpr int32_t kHalf = int32_t{1} << (kPrecisionBits - 1);

struct Filter {
  double scale, support, ss;
};

// The filter for resampling [in0, in1) to out_size samples: its support widens
// with the downscale, so shrinking averages instead of skipping pixels.
CARBON_HD inline Filter filter_for(double in0, double in1, int out_size) {
  double scale = (in1 - in0) / out_size;
  double filterscale = scale < 1.0 ? 1.0 : scale;
  return {scale, 1.0 * filterscale, 1.0 / filterscale};
}

// The most taps any output sample has: PIL's ksize.
CARBON_HD inline int max_taps(const Filter& f) {
  return static_cast<int>(ceil(f.support)) * 2 + 1;
}

CARBON_HD inline double triangle(double x) {
  if (x < 0.0) x = -x;
  return x < 1.0 ? 1.0 - x : 0.0;
}

// floor(v) as an int, for v in an int's range: the truncation, one less where that is above v. The same value as
// static_cast<int>(floor(v)), without the library's call.
CARBON_HD inline int floor_int(double v) {
  const int i = static_cast<int>(v);
  return i - (v < i ? 1 : 0);
}

// Output sample `xx`: its first source index (returned), its tap count, and its
// coefficients in fixed point (k[0 .. count)).  PIL truncates the window ends
// toward zero and clamps them to the image; with paper outside the image, floor
// is the same rounding on a page shifted to positive coordinates.
CARBON_HD inline int coefficients(const Filter& f, double in0, int xx, int32_t* k, int* count) {
  const double center = in0 + (xx + 0.5) * f.scale;
  const int xmin = floor_int(center - f.support + 0.5);
  const int n = floor_int(center + f.support + 0.5) - xmin;
  double ww = 0.0;
  for (int x = 0; x < n; ++x) ww += triangle((x + xmin - center + 0.5) * f.ss);
  for (int x = 0; x < n; ++x) {
    double w = triangle((x + xmin - center + 0.5) * f.ss);
    if (ww != 0.0) w /= ww;
    k[x] = static_cast<int32_t>(w < 0 ? -0.5 + w * (1 << kPrecisionBits) : 0.5 + w * (1 << kPrecisionBits));
  }
  *count = n;
  return xmin;
}

CARBON_HD inline uint8_t clip8(int32_t v) {
  v >>= kPrecisionBits;
  return static_cast<uint8_t>(v < 0 ? 0 : v > 255 ? 255 : v);
}

}  // namespace carbon::detail
