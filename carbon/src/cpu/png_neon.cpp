// PNG's rows on Armv8-A Advanced SIMD (carbon/src/cpu/png_kernels.h).
//
// As png_avx2.cpp, in 128-bit vectors: Up 16 bytes an add; Sub of 1, 2, 4 or 8 bytes a
// pixel as a prefix sum of 16 bytes in log steps; runs of 16 Paeth rows of 1, 3 or 4
// bytes a pixel as a wavefront, the rows in the byte lanes (vabdq_u8 for the distances,
// zips for the transposes); the rest png_kernels.h's scalar code.  RGB to gray: vld3q_u8
// or vld4q_u8 splits 16 pixels into their channels, and widening multiply-adds
// (77 r + 150 g + 29 b, at most 65280) give the sums in 16 bits.

#include <arm_neon.h>

#include "png_kernels.h"

namespace carbon::cpu::detail {
namespace {

// x shifted up by K bytes, zeros in
template <int K>
inline uint8x16_t shift_up(uint8x16_t x) {
  return vextq_u8(vdupq_n_u8(0), x, 16 - K);
}

// Sub of B bytes a pixel (1, 2, 4 or 8), 16 bytes a step.
template <int B>
void sub(const uint8_t* raw, uint8_t* cur, size_t n) {
  uint8x16_t carry = vdupq_n_u8(0);  // the last pixel made, in every pixel's place
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    uint8x16_t x = vld1q_u8(raw + i);
    x = vaddq_u8(x, shift_up<B>(x));
    if constexpr (B <= 4) x = vaddq_u8(x, shift_up<2 * B>(x));
    if constexpr (B <= 2) x = vaddq_u8(x, shift_up<4 * B>(x));
    if constexpr (B == 1) x = vaddq_u8(x, shift_up<8>(x));
    x = vaddq_u8(x, carry);
    vst1q_u8(cur + i, x);
    if constexpr (B == 1) {
      carry = vdupq_laneq_u8(x, 15);
    } else if constexpr (B == 2) {
      carry = vreinterpretq_u8_u16(vdupq_laneq_u16(vreinterpretq_u16_u8(x), 7));
    } else if constexpr (B == 4) {
      carry = vreinterpretq_u8_u32(vdupq_laneq_u32(vreinterpretq_u32_u8(x), 3));
    } else {
      carry = vreinterpretq_u8_u64(vdupq_laneq_u64(vreinterpretq_u64_u8(x), 1));
    }
  }
  for (; i < n; ++i) cur[i] = static_cast<uint8_t>(raw[i] + (i >= B ? cur[i - B] : 0));
}

void up(const uint8_t* raw, const uint8_t* prior, uint8_t* cur, size_t n) {
  size_t i = 0;
  for (; i + 16 <= n; i += 16) vst1q_u8(cur + i, vaddq_u8(vld1q_u8(raw + i), vld1q_u8(prior + i)));
  for (; i < n; ++i) cur[i] = static_cast<uint8_t>(raw[i] + prior[i]);
}

// Paeth's predictor of 16 bytes (png_avx2.cpp).
inline uint8x16_t paeth16(uint8x16_t a, uint8x16_t b, uint8x16_t c) {
  const uint8x16_t pa = vabdq_u8(b, c), pb = vabdq_u8(a, c);
  const uint8x16_t same = vceqq_u8(vcleq_u8(a, c), vcleq_u8(b, c));
  const uint8x16_t a_side = vcleq_u8(pa, pb);
  const uint8x16_t a_near = vcleq_u8(pa, vqsubq_u8(pb, pa));  // 2 pa <= pb
  const uint8x16_t b_near = vcleq_u8(pb, vqsubq_u8(pa, pb));  // 2 pb <= pa
  const uint8x16_t take_a = vandq_u8(a_side, vorrq_u8(same, a_near));
  const uint8x16_t take_b = vbicq_u8(vorrq_u8(same, b_near), a_side);
  return vbslq_u8(take_a, a, vbslq_u8(take_b, b, c));
}

// The 16 x 16 bytes of v transposed (png_avx2.cpp).
inline void transpose(uint8x16_t v[16]) {
  for (int round = 0; round < 4; ++round) {
    uint8x16_t t[16];
    for (int j = 0; j < 8; ++j) {
      t[2 * j] = vzip1q_u8(v[j], v[j + 8]);
      t[2 * j + 1] = vzip2q_u8(v[j], v[j + 8]);
    }
    for (int j = 0; j < 16; ++j) v[j] = t[j];
  }
}

// 16 Paeth rows of B bytes a pixel as a wavefront (png_avx2.cpp), at least 32 pixels a row.
template <int B>
void paeth_wave(const PngRows& r) {
  const size_t n = r.n, w = n / B;
  const uint8_t* raw[16];
  uint8_t* cur[16];
  const uint8_t* up[16];
  for (int k = 0; k < 16; ++k) {
    raw[k] = r.raw + k * r.raw_stride;
    cur[k] = r.cur + k * r.cur_stride;
    up[k] = k ? cur[k - 1] : r.prior;
  }
  for (int k = 0; k < 15; ++k) png_paeth_row<B>(raw[k], up[k], cur[k], static_cast<size_t>(B) * (15 - k));
  uint8x16_t a[B], bp[B];
  for (int ch = 0; ch < B; ++ch) {
    uint8_t av[16], bv[16];
    for (int k = 0; k < 16; ++k) {
      av[k] = k < 15 ? cur[k][B * (14 - k) + ch] : 0;
      bv[k] = k == 0 ? r.prior[B * 14 + ch] : k < 15 ? cur[k - 1][B * (14 - k) + ch] : 0;
    }
    a[ch] = vld1q_u8(av);
    bp[ch] = vld1q_u8(bv);
  }
  size_t s0 = 15;
  for (; s0 + 16 <= w; s0 += 16) {
    uint8x16_t v[B][16];
    for (int j = 0; j < B; ++j) {
      for (int k = 0; k < 16; ++k) v[j][k] = vld1q_u8(raw[k] + B * (s0 - k) + 16 * j);
      transpose(v[j]);
    }
    for (int i = 0; i < 16; ++i) {
      const uint8_t* above = r.prior + B * (s0 + i) - 15;
      for (int ch = 0; ch < B; ++ch) {
        const int q = B * i + ch;
        const uint8x16_t b = vextq_u8(vld1q_u8(above + ch), a[ch], 15);
        const uint8x16_t x = vaddq_u8(v[q / 16][q % 16], paeth16(a[ch], b, bp[ch]));
        v[q / 16][q % 16] = x;
        bp[ch] = b;
        a[ch] = x;
      }
    }
    for (int j = 0; j < B; ++j) {
      transpose(v[j]);
      for (int k = 0; k < 16; ++k) vst1q_u8(cur[k] + B * (s0 - k) + 16 * j, v[j][k]);
    }
  }
  for (int k = 0; k < 16; ++k) png_paeth_from<B>(raw[k], up[k], cur[k], B * (s0 - k), n);
}

// 16 pixels' channels to their gray
inline uint8x16_t gray16(uint8x16_t r, uint8x16_t g, uint8x16_t b) {
  uint16x8_t lo = vmull_u8(vget_low_u8(r), vdup_n_u8(77));
  lo = vmlal_u8(lo, vget_low_u8(g), vdup_n_u8(150));
  lo = vmlal_u8(lo, vget_low_u8(b), vdup_n_u8(29));
  uint16x8_t hi = vmull_u8(vget_high_u8(r), vdup_n_u8(77));
  hi = vmlal_u8(hi, vget_high_u8(g), vdup_n_u8(150));
  hi = vmlal_u8(hi, vget_high_u8(b), vdup_n_u8(29));
  return vcombine_u8(vshrn_n_u16(lo, 8), vshrn_n_u16(hi, 8));
}

}  // namespace

size_t png_unfilter_neon(const PngRows& r) {
  const int filter = r.raw[-1], bpp = r.bpp;
  if (filter == 4 && (bpp == 1 || bpp == 3 || bpp == 4) && r.n / bpp >= 32 && png_paeth_run(r, 16) == 16) {
    if (bpp == 1) {
      paeth_wave<1>(r);
    } else if (bpp == 3) {
      paeth_wave<3>(r);
    } else {
      paeth_wave<4>(r);
    }
    return 16;
  }
  if (filter == 2) {
    up(r.raw, r.prior, r.cur, r.n);
    return 1;
  }
  if (filter == 1) {
    switch (bpp) {
      case 1: sub<1>(r.raw, r.cur, r.n); return 1;
      case 2: sub<2>(r.raw, r.cur, r.n); return 1;
      case 4: sub<4>(r.raw, r.cur, r.n); return 1;
      case 8: sub<8>(r.raw, r.cur, r.n); return 1;
      default: break;
    }
  }
  return png_unfilter_scalar_rows(r);
}

void png_gray_rgb8_neon(const uint8_t* px, uint8_t* out, size_t w, int step) {
  size_t x = 0;
  if (step == 4) {
    for (; x + 16 <= w; x += 16) {
      const uint8x16x4_t v = vld4q_u8(px + x * 4);
      vst1q_u8(out + x, gray16(v.val[0], v.val[1], v.val[2]));
    }
  } else {
    for (; x + 16 <= w; x += 16) {
      const uint8x16x3_t v = vld3q_u8(px + x * 3);
      vst1q_u8(out + x, gray16(v.val[0], v.val[1], v.val[2]));
    }
  }
  const size_t s = static_cast<size_t>(step);
  for (const uint8_t* p = px + x * s; x < w; ++x, p += s) {
    out[x] = static_cast<uint8_t>((p[0] * 77 + p[1] * 150 + p[2] * 29) >> 8);
  }
}

}  // namespace carbon::cpu::detail
