// PNG's rows (carbon/src/png.cpp): the filters undone and RGB turned to gray, a version
// per instruction set, each in its own file built for that set (png_avx2.cpp,
// png_neon.cpp) under the rules of kernels.h: everything such a file defines but the
// functions declared here has internal linkage, and it calls no inline function of the
// standard library.  The scalar code below is each file's own copy (an anonymous
// namespace), so it is built with that file's flags; png.cpp's reference is the same
// code a row at a time.  Every version gives the same bytes: those of stb_image.
//
// Unfiltering is serial for Sub, Average and Paeth: each byte needs the one a pixel to
// its left, just made.  The scalar code keeps that pixel in registers (1, 3 and 4 bytes
// a pixel; a load of the byte just stored would put a store-to-load forwarding on the
// chain), and does two Paeth rows of 1 byte a pixel at once, the second a pixel behind,
// so that the processor runs two chains side by side.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace carbon::cpu::detail {

// Rows to unfilter: row k's n bytes at raw + k * raw_stride, its filter byte (0 None,
// 1 Sub, 2 Up, 3 Average, 4 Paeth; checked by the caller) just before them, unfiltered
// to cur + k * cur_stride: in place (cur == raw, the same stride) or into rows apart from
// raw's.  prior is the row above row 0 as unfiltered (zeros above an image's first
// row).  bpp is the bytes a pixel takes, 1 to 8 (1 below 8 bits a pixel), and n a
// multiple of it.
struct PngRows {
  const uint8_t* raw;
  size_t raw_stride;
  uint8_t* cur;
  size_t cur_stride;
  const uint8_t* prior;
  size_t rows;  // at least 1
  size_t n;
  int bpp;
};

// Unfilters the first rows of `r`, as many as the kernel takes at once (1, a pair of
// Paeth rows, or a block of 16), and returns how many.
size_t png_unfilter_avx2(const PngRows& r);
size_t png_unfilter_neon(const PngRows& r);

// w RGB pixels (step 3 bytes a pixel) or RGBA (step 4), 8 bits a channel, to gray as
// stb_image computes it: (77 r + 150 g + 29 b) >> 8, alpha dropped.  Reads only the
// row's 3 w or 4 w bytes.
void png_gray_rgb8_avx2(const uint8_t* px, uint8_t* out, size_t w, int step);
void png_gray_rgb8_neon(const uint8_t* px, uint8_t* out, size_t w, int step);

namespace {

// stb_image's Paeth predictor (stbi__paeth): the PNG specification's choice of a (left),
// b (above) or c (above left), without branches.
inline int png_paeth(int a, int b, int c) {
  const int thresh = c * 3 - (a + b);
  const int lo = a < b ? a : b, hi = a < b ? b : a;
  const int t0 = hi <= thresh ? lo : c;
  return thresh <= lo ? hi : t0;
}

// Sub, Average and Paeth from byte i0 (a whole pixel, at least one in) to n, the pixel
// left of i0 in registers: B bytes a pixel.
template <int B>
inline void png_sub_from(const uint8_t* raw, uint8_t* cur, size_t i0, size_t n) {
  unsigned a[B];
  for (int k = 0; k < B; ++k) a[k] = cur[i0 - B + k];
  for (size_t i = i0; i < n; i += B) {
    for (int k = 0; k < B; ++k) {
      a[k] = static_cast<uint8_t>(raw[i + k] + a[k]);
      cur[i + k] = static_cast<uint8_t>(a[k]);
    }
  }
}

template <int B>
inline void png_average_from(const uint8_t* raw, const uint8_t* prior, uint8_t* cur, size_t i0, size_t n) {
  unsigned a[B];
  for (int k = 0; k < B; ++k) a[k] = cur[i0 - B + k];
  for (size_t i = i0; i < n; i += B) {
    for (int k = 0; k < B; ++k) {
      a[k] = static_cast<uint8_t>(raw[i + k] + ((a[k] + prior[i + k]) >> 1));
      cur[i + k] = static_cast<uint8_t>(a[k]);
    }
  }
}

template <int B>
inline void png_paeth_from(const uint8_t* raw, const uint8_t* prior, uint8_t* cur, size_t i0, size_t n) {
  unsigned a[B];
  for (int k = 0; k < B; ++k) a[k] = cur[i0 - B + k];
  for (size_t i = i0; i < n; i += B) {
    for (int k = 0; k < B; ++k) {
      a[k] = static_cast<uint8_t>(raw[i + k] + png_paeth(static_cast<int>(a[k]), prior[i + k], prior[i + k - B]));
      cur[i + k] = static_cast<uint8_t>(a[k]);
    }
  }
}

// The first pixel of a row (bpp bytes): Sub and Paeth take it as it is plus its byte
// above (Paeth with a and c zero is b), Average plus half that.
inline void png_first_pixel(int filter, const uint8_t* raw, const uint8_t* prior, uint8_t* cur, size_t bpp) {
  for (size_t k = 0; k < bpp; ++k) {
    const int up = filter == 1 ? 0 : filter == 3 ? prior[k] >> 1 : prior[k];
    cur[k] = static_cast<uint8_t>(raw[k] + up);
  }
}

// A row of Paeth, B bytes a pixel, from its first pixel.
template <int B>
inline void png_paeth_row(const uint8_t* raw, const uint8_t* prior, uint8_t* cur, size_t n) {
  png_first_pixel(4, raw, prior, cur, B);
  png_paeth_from<B>(raw, prior, cur, B, n);
}

// One row, in scalar code: the reference, and the instruction sets' kernels' code for
// the rows they have nothing faster for.
inline void png_unfilter_row(int filter, const uint8_t* raw, const uint8_t* prior, uint8_t* cur, size_t n, int bpp) {
  const size_t b = static_cast<size_t>(bpp);
  if (filter == 0) {
    if (cur != raw) std::memcpy(cur, raw, n);
    return;
  }
  if (filter == 2) {
    for (size_t i = 0; i < n; ++i) cur[i] = static_cast<uint8_t>(raw[i] + prior[i]);
    return;
  }
  png_first_pixel(filter, raw, prior, cur, b);
  if (bpp == 1 || bpp == 3 || bpp == 4) {
    switch (filter * 8 + bpp) {
      case 1 * 8 + 1: png_sub_from<1>(raw, cur, 1, n); return;
      case 1 * 8 + 3: png_sub_from<3>(raw, cur, 3, n); return;
      case 1 * 8 + 4: png_sub_from<4>(raw, cur, 4, n); return;
      case 3 * 8 + 1: png_average_from<1>(raw, prior, cur, 1, n); return;
      case 3 * 8 + 3: png_average_from<3>(raw, prior, cur, 3, n); return;
      case 3 * 8 + 4: png_average_from<4>(raw, prior, cur, 4, n); return;
      case 4 * 8 + 1: png_paeth_from<1>(raw, prior, cur, 1, n); return;
      case 4 * 8 + 3: png_paeth_from<3>(raw, prior, cur, 3, n); return;
      default: png_paeth_from<4>(raw, prior, cur, 4, n); return;
    }
  }
  // 2, 6 and 8 bytes a pixel (16-bit samples): the byte a pixel back from memory
  switch (filter) {
    case 1:
      for (size_t i = b; i < n; ++i) cur[i] = static_cast<uint8_t>(raw[i] + cur[i - b]);
      return;
    case 3:
      for (size_t i = b; i < n; ++i) cur[i] = static_cast<uint8_t>(raw[i] + ((cur[i - b] + prior[i]) >> 1));
      return;
    default:
      for (size_t i = b; i < n; ++i) {
        cur[i] = static_cast<uint8_t>(raw[i] + png_paeth(cur[i - b], prior[i], prior[i - b]));
      }
      return;
  }
}

// Two Paeth rows of 1 byte a pixel (n at least 2), the second (whose row above is the
// first) a pixel behind: two independent chains a step.
inline void png_paeth_pair(const uint8_t* raw0, const uint8_t* raw1, const uint8_t* prior, uint8_t* cur0, uint8_t* cur1,
                           size_t n) {
  unsigned a0 = static_cast<uint8_t>(raw0[0] + prior[0]);
  cur0[0] = static_cast<uint8_t>(a0);
  unsigned a1 = static_cast<uint8_t>(raw1[0] + a0);
  cur1[0] = static_cast<uint8_t>(a1);
  unsigned l0 = a0;  // the first row's byte left of a0
  a0 = static_cast<uint8_t>(raw0[1] + png_paeth(static_cast<int>(a0), prior[1], prior[0]));
  cur0[1] = static_cast<uint8_t>(a0);
  for (size_t i = 2; i < n; ++i) {
    const unsigned v1 = static_cast<uint8_t>(raw1[i - 1] + png_paeth(static_cast<int>(a1), a0, l0));
    const unsigned v0 = static_cast<uint8_t>(raw0[i] + png_paeth(static_cast<int>(a0), prior[i], prior[i - 1]));
    cur1[i - 1] = static_cast<uint8_t>(v1);
    cur0[i] = static_cast<uint8_t>(v0);
    a1 = v1;
    l0 = a0;
    a0 = v0;
  }
  cur1[n - 1] = static_cast<uint8_t>(raw1[n - 1] + png_paeth(static_cast<int>(a1), a0, l0));
}

// The rows of `r` the instruction sets' kernels do in scalar code: a pair of Paeth rows
// of 1 byte a pixel, or a row; how many.
inline size_t png_unfilter_scalar_rows(const PngRows& r) {
  const int filter = r.raw[-1];
  if (filter == 4 && r.bpp == 1 && r.rows >= 2 && r.n >= 2 && r.raw[r.raw_stride - 1] == 4) {
    png_paeth_pair(r.raw, r.raw + r.raw_stride, r.prior, r.cur, r.cur + r.cur_stride, r.n);
    return 2;
  }
  png_unfilter_row(filter, r.raw, r.prior, r.cur, r.n, r.bpp);
  return 1;
}

// How many rows from the first of `r` are Paeth, at most `most`.
inline size_t png_paeth_run(const PngRows& r, size_t most) {
  size_t k = 0;
  while (k < most && k < r.rows && (r.raw + k * r.raw_stride)[-1] == 4) ++k;
  return k;
}

}  // namespace
}  // namespace carbon::cpu::detail
