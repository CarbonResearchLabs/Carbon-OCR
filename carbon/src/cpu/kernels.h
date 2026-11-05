// The GEMM kernels, one set per instruction set, each in its own source file
// compiled for that set (carbon/cpu.h dispatches).  B is packed by pack_b or
// pack_f32 for the same set, a panel of columns at a time, k within it:
//
//   int8  avx2      int16 [n/16][taps][segp/2][16][2]   vpmaddwd: exact 16-bit products
//         avx-vnni  int8  [n/16][taps][segp/4][16][4]   vpdpbusd, 256 bits
//         avx512    int8  [n/32][taps][segp/4][32][4]   vpdpbusd, 512 bits
//         neon-dot  int8  [n/16][taps][segp/4][16][4]   sdot by element, A shifted to signed
//   fp32  avx2      [n/16][k][16]                       vfmadd231ps, 256 bits
//         avx512    [n/32][k][32]                       vfmadd231ps, 512 bits
//         neon      [n/16][k][16]                       fmla
//   fp16  avx2      [n/16][k][16] FP16                  widened to FP32 (vcvtph2ps) as a call reads
//                                                       it, then fp32's tiles
//         avx512    [n/32][k][32] FP16                  the same on 512 bits
//         neon      [n/16][k][16] FP16                  the same with fcvtl
//         avx512-fp16  [n/32][k][32] FP16               vfmadd231ph, two panels a tile
//         neon-fp16    [n/16][k][16] FP16               fmla on 8 halves
//
// Each kernel sums a tile of rows and columns into a small buffer, then applies
// the output stage (carbon::cpu::Epilogue) to it a row at a time: vectors where a
// whole vector of columns is there, epilogue_scalar for the rest, the same float
// operations in the same order, so the bits are the reference's.  (The FP16
// arithmetic kernels' sums are their own; their output stage is still these.)
// After them, the depthwise convolutions, CTC's argmax, a crop's counts, a
// threshold's bits and the crop rule's resampling, one version a set too, and the
// detector's head on AVX2.
//
// The page turns (turns.cpp has the references): rotate_bilinear's kernels run
// the reference's double arithmetic on a vector of pixels whose four taps are all
// on the page, and turn_pixels on the others; rotate_quarters' kernels copy rows
// as they are or reversed, and for the odd turns transpose tiles of 16 rows in
// registers, the bytes no tile covers by quarter_bytes.
//
// The files built for an instruction set share no code with the others.  A
// function two files both emit (an inline one, a template's instance, a member of
// a class in a header) is one copy the linker keeps for all of them, and the copy
// it keeps could be avx512.cpp's, run on a CPU without AVX-512.  So everything
// defined below has internal linkage, each file its own copy built with its own
// flags; the kernels read their rows through piece() here, not Rows::piece, and
// the packed matrices through views; their working memory is scratch(), in
// kernels.cpp; and they call no inline function of the standard library: min_of
// for std::min, the C functions for std::nearbyint and std::fma, INFINITY for
// numeric_limits.  tests/isa_objects_test.cmake checks that those files define
// no code another file could share.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdint>

#include "carbon/cpu.h"

namespace carbon::cpu::detail {

// PackedB and PackedF32 as the kernels read them, through plain pointers: the
// kernels call none of std::vector's functions.
struct BView {
  const int8_t* data;
  const int32_t* col_sums;
  int k, n, taps, seg, segp, np;
};
struct F32View {
  const float* data;
  int k, n, np;
};
struct F16View {
  const uint16_t* data;  // FP16
  int k, n, np;
  uint64_t id;  // PackedF16::id
};

void gemm_u8s8_avx2(Rows<uint8_t>, int32_t, const BView&, int, const Epilogue&);
void gemm_u8s8_avxvnni(Rows<uint8_t>, int32_t, const BView&, int, const Epilogue&);
void gemm_u8s8_avx512(Rows<uint8_t>, int32_t, const BView&, int, const Epilogue&);
void gemm_u8s8_neondot(Rows<uint8_t>, int32_t, const BView&, int, const Epilogue&);
void gemm_f32_avx2(Rows<float>, const F32View&, int, const Epilogue&);
void gemm_f32_avx512(Rows<float>, const F32View&, int, const Epilogue&);
void gemm_f32_neon(Rows<float>, const F32View&, int, const Epilogue&);
void dwconv_f32_avx2(Rows<float>, const float*, int, int, const Epilogue&);
void dwconv_u8s8_avx2(Rows<uint8_t>, int32_t, const int8_t*, int, int, const Epilogue&);
void dwconv_f32_neon(Rows<float>, const float*, int, int, const Epilogue&);
void dwconv_u8s8_neon(Rows<uint8_t>, int32_t, const int8_t*, int, int, const Epilogue&);
void ctc_argmax_avx2(const float*, int, int, int, int32_t*, float*);
ByteCounts count_beyond_avx2(const uint8_t*, int, int, int64_t, int, int);
void bgrx_to_gray_avx2(const uint8_t*, int64_t, int, int, uint8_t*, int64_t);
void softmax_rows_avx2(float* rows, int n_rows, int n, float scale);
void det_stem_avx2(const float* x, int h, int w, const DetStemWeights& sw, float* out, int row0, int row1);
void dot_chains_avx2(const float* a, const float* b, const int32_t* ia, const int32_t* ib, int pairs, int d,
                     float* out);
void gelu_avx2(float* v, size_t n);
// erf(x) as gelu takes it: Eigen's float rational, x clamped to [-4, 4], each multiply and add on its own
constexpr float kErfA1 = -1.60960333262415e-02f, kErfA3 = -2.95459980854025e-03f, kErfA5 = -7.34990630326855e-04f,
                kErfA7 = -5.69250639462346e-05f, kErfA9 = -2.10102402082508e-06f, kErfA11 = 2.77068142495902e-08f,
                kErfA13 = -2.72614225801306e-10f;
constexpr float kErfB0 = -1.42647390514189e-02f, kErfB2 = -7.37332916720468e-03f, kErfB4 = -1.68282697438203e-03f,
                kErfB6 = -2.13374055278905e-04f, kErfB8 = -1.45660718464996e-05f;
inline float erf_of(float x) {
  x = x < 4.0f ? x : 4.0f;
  x = x > -4.0f ? x : -4.0f;
  const float x2 = x * x;
  float p = x2 * kErfA13 + kErfA11;
  p = x2 * p + kErfA9;
  p = x2 * p + kErfA7;
  p = x2 * p + kErfA5;
  p = x2 * p + kErfA3;
  p = x2 * p + kErfA1;
  p = x * p;
  float q = x2 * kErfB8 + kErfB6;
  q = x2 * q + kErfB4;
  q = x2 * q + kErfB2;
  q = x2 * q + kErfB0;
  return p / q;
}
// a tile of kHeadFrames frames from t0 (all within `frames`): its logits [classes][kHeadFrames] into `tile`
void head_tile_avx2(const float* features, int k, int frames, int t0, const float* w, const float* b, int classes,
                    float* tile);
void skew_rows_avx2(const float*, const float*, int64_t, double, double, double, int, int32_t*);
void above_bits_avx2(const float*, int, float, uint32_t*);
// det_head's band: its working memory (scratch kA) the head's row, three upsampled rows and a row of zeros.
void det_head_avx2(const float* x, int h4, int w4, const DetHeadWeights& w, float* maps, int row0, int row1);
void ctc_argmax_neon(const float*, int, int, int, int32_t*, float*);
ByteCounts count_beyond_neon(const uint8_t*, int, int, int64_t, int, int);
void above_bits_neon(const float*, int, float, uint32_t*);
// FP16 (cpu.h): A in FP16 (uint16_t) or FP32, B and the depthwise weights in FP16.
void gemm_f16_avx2(Rows<uint16_t>, const F16View&, int, const Epilogue&);
void gemm_f16_avx2(Rows<float>, const F16View&, int, const Epilogue&);
void gemm_f16_avx512(Rows<uint16_t>, const F16View&, int, const Epilogue&);
void gemm_f16_avx512(Rows<float>, const F16View&, int, const Epilogue&);
void gemm_f16_neon(Rows<uint16_t>, const F16View&, int, const Epilogue&);
void gemm_f16_neon(Rows<float>, const F16View&, int, const Epilogue&);
void gemm_f16_avx512fp16(Rows<uint16_t>, const F16View&, int, const Epilogue&);
void gemm_f16_neonfp16(Rows<uint16_t>, const F16View&, int, const Epilogue&);
void dwconv_f16_avx2(Rows<uint16_t>, const uint16_t*, int, int, const Epilogue&);
void dwconv_f16_avx2(Rows<float>, const uint16_t*, int, int, const Epilogue&);
void dwconv_f16_neon(Rows<uint16_t>, const uint16_t*, int, int, const Epilogue&);
void dwconv_f16_neon(Rows<float>, const uint16_t*, int, int, const Epilogue&);
void dwconv_f16_avx512fp16(Rows<uint16_t>, const uint16_t*, int, int, const Epilogue&);
void dwconv_f16_neonfp16(Rows<uint16_t>, const uint16_t*, int, int, const Epilogue&);
void to_f16_avx2(const float*, uint16_t*, size_t);
void to_f32_avx2(const uint16_t*, float*, size_t);
void to_f16_neon(const float*, uint16_t*, size_t);
void to_f32_neon(const uint16_t*, float*, size_t);

// The crop rule's resampling (carbon::resample_gray; carbon/src/lines.cpp drives it a
// row at a time), the integer arithmetic of its two passes on vectors.  The weights
// are 22-bit fixed point, so each product and sum is exact in 32 bits.
//
// Horizontal: output sample x reads 8 * chunks source pixels from first[x] on (its
// weights past its own taps are zeros), in groups of 8 samples.  resample_hk_*
// lays out the weights of `width` samples (k[x * taps + t] for t < count[x]) for
// the kernel: 32 bytes a sample a chunk, the groups padded to 8 samples, into
// `out`, which must hold zeros.  resample_h_* writes groups [g0, g1) of a row,
// out[8 g0 .. 8 g1), its pixels read from src, which holds the source columns from
// col0 on: every column a group reads must be there (the image's own row, or a copy
// padded with paper).
//
//   avx2   int16, k = 2048 * (k >> 11) + (k & 2047): two vpmaddwd for exact
//          32-bit products, two samples a register, [group][chunk][4][lo 16][hi 16]
//   neon   int32, mla on 32-bit lanes, [sample][chunk][8]
//
// Vertical: out[x] for x < width, from n rows (their pixel x times k[i]); each row
// readable to width rounded up to 32.
void resample_hk_avx2(const int32_t* k, const int32_t* count, int taps, int width, int chunks, void* out);
void resample_hk_neon(const int32_t* k, const int32_t* count, int taps, int width, int chunks, void* out);
void resample_h_avx2(const uint8_t* src, int col0, const int32_t* first, const void* k, int chunks, int g0, int g1,
                     uint8_t* out);
void resample_h_neon(const uint8_t* src, int col0, const int32_t* first, const void* k, int chunks, int g0, int g1,
                     uint8_t* out);
void resample_v_avx2(const uint8_t* const* rows, const int32_t* k, int n, int width, uint8_t* out);
void resample_v_neon(const uint8_t* const* rows, const int32_t* k, int n, int width, uint8_t* out);

// rotate_bilinear as the kernels read it: the page, its centre, the angle, and each
// column's terms of the turn, xc[x] = (x - cx) c and xs[x] = (x - cx) s, the same in
// every row (turns.cpp makes them).
struct TurnView {
  const uint8_t* src;
  int w, h;
  int64_t stride;
  double s, c, cx, cy;
  const double* xc;
  const double* xs;
  int row0, row1;  // the rows of dst written, row0 to row1 - 1 (each row's bytes do not depend on the others)
};

void rotate_bilinear_avx2(const TurnView&, uint8_t*);
void rotate_bilinear_neon(const TurnView&, uint8_t*);
void rotate_quarters_avx2(const uint8_t*, int, int, int64_t, int, uint8_t*);
void rotate_quarters_neon(const uint8_t*, int, int, int64_t, int, uint8_t*);

// A kernel's working memory: at least `bytes` bytes from a cache line's start, its
// thread's own, kept from call to call.  A buffer a slot, so one kernel's buffers do
// not overlap.  kB: B widened to FP32, or depthwise weights.
enum class Scratch { kFix, kRows, kA, kB };
void* scratch(Scratch slot, size_t bytes);
// The same buffer, and whether it still holds what was written there under `tag`
// (not 0) by the last call on this slot, at this size or more: a call that takes
// it without a tag, or makes it grow, forgets the tag.
void* scratch_kept(Scratch slot, size_t bytes, uint64_t tag, bool& kept);

namespace {

template <typename T>
inline T* scratch_of(Scratch slot, size_t n) {
  return static_cast<T*>(scratch(slot, n * sizeof(T)));
}

inline int min_of(int a, int b) { return a < b ? a : b; }

// The FP32 arithmetic FP16 kernels keep B widened from call to call (scratch_kept)
// up to this size: a layer's whole weights within L2, a widening a layer and run, not
// one an output row; past it, a panel at a time.
constexpr size_t kKeepB = size_t{2} << 20;

// Rows::piece: piece t of row i.
template <typename T>
inline const T* piece(const Rows<T>& a, int64_t i, int t) {
  return a.ptrs ? a.ptrs[i * a.taps + t] : a.base + i * a.stride + (a.offsets ? a.offsets[t] : 0);
}

// exp(x) for x <= 0, as every version of ctc_argmax computes it: x clamped to -87 (so
// 2^n stays normal), n = round(x log2 e) (ties to even), r = x - n ln 2 in two parts
// (Cody and Waite), Cephes's degree-5 polynomial, times 2^n through the exponent
// bits.  Each multiply and add on its own, in this order, in every version.
constexpr float kExpLo = -87.0f, kLog2e = 1.44269504088896341f, kLn2Hi = 0.693359375f, kLn2Lo = -2.12194440e-4f;
constexpr float kExpP0 = 1.9875691500e-4f, kExpP1 = 1.3981999507e-3f, kExpP2 = 8.3334519073e-3f,
                kExpP3 = 4.1665795894e-2f, kExpP4 = 1.6666665459e-1f, kExpP5 = 5.0000001201e-1f;

inline float exp_neg(float x) {
  x = x > kExpLo ? x : kExpLo;
  const float n = nearbyintf(x * kLog2e);
  float r = x - n * kLn2Hi;
  r = r - n * kLn2Lo;
  float p = kExpP0;
  p = p * r + kExpP1;
  p = p * r + kExpP2;
  p = p * r + kExpP3;
  p = p * r + kExpP4;
  p = p * r + kExpP5;
  const float y = p * (r * r) + r + 1.0f;
  const int32_t bits = (static_cast<int32_t>(n) + 127) << 23;
  float scale;
  std::memcpy(&scale, &bits, sizeof scale);
  return y * scale;
}

// A tap of rotate_bilinear: the page's byte at (x, y), paper (255) off the page.
inline double turn_tap(const TurnView& t, int x, int y) {
  if (x < 0 || y < 0 || x >= t.w || y >= t.h) return 255.0;
  return t.src[y * t.stride + x];
}

// Pixels [x0, x1) of an output row d of rotate_bilinear the reference's way, the
// row's terms ds = dy s and dc = dy c given: where each was on the page, its taps,
// their blend, rounded half away from zero.  For the pixels near the page's edges,
// whose taps are not all on it.
inline void turn_pixels(const TurnView& t, double ds, double dc, int x0, int x1, uint8_t* d) {
  for (int x = x0; x < x1; ++x) {
    const double sx = t.xc[x] - ds + t.cx, sy = t.xs[x] + dc + t.cy;
    const int ix = static_cast<int>(floor(sx)), iy = static_cast<int>(floor(sy));
    if (ix < -1 || iy < -1 || ix >= t.w || iy >= t.h) continue;
    const double fx = sx - ix, fy = sy - iy;
    const double v = (turn_tap(t, ix, iy) * (1 - fx) + turn_tap(t, ix + 1, iy) * fx) * (1 - fy) +
                     (turn_tap(t, ix, iy + 1) * (1 - fx) + turn_tap(t, ix + 1, iy + 1) * fx) * fy;
    d[x] = static_cast<uint8_t>(lround(v < 0.0 ? 0.0 : v > 255.0 ? 255.0 : v));
  }
}

// The tile of 16 rows from (x, y) in an odd quarter turn q: its rows in the order
// the transpose takes them (q = 1: bottom up) into r, and where its column 0
// lands in dst (its column i is i * (q = 1 ? h : -h) bytes on).
inline uint8_t* quarter_tile(const uint8_t* src, int64_t stride, int w, int h, int q, int x, int y, uint8_t* dst,
                             const uint8_t** r) {
  const uint8_t* first = src + (q == 1 ? y + 15 : y) * stride + x;
  const int64_t step = q == 1 ? -stride : stride;
  for (int j = 0; j < 16; ++j) r[j] = first + j * step;
  return q == 1 ? dst + static_cast<int64_t>(x) * h + (h - 16 - y) : dst + static_cast<int64_t>(w - 1 - x) * h + y;
}

// The bytes of the odd quarter turn q no tile covered, one at a time: columns
// x0 and on of rows [y0, y1).
inline void quarter_bytes(const uint8_t* src, int64_t stride, int w, int h, int q, int x0, int y0, int y1,
                          uint8_t* dst) {
  for (int y = y0; y < y1; ++y) {
    const uint8_t* r = src + y * stride;
    if (q == 1) {
      for (int x = x0; x < w; ++x) dst[static_cast<int64_t>(x) * h + (h - 1 - y)] = r[x];
    } else {
      for (int x = x0; x < w; ++x) dst[static_cast<int64_t>(w - 1 - x) * h + y] = r[x];
    }
  }
}

// The pieces of rows [m0, m0 + mr) as a list, mr * taps pointers: the list
// itself, or made from the grid in the kRows scratch.
template <typename T>
inline const T* const* block(const Rows<T>& a, int64_t m0, int mr) {
  if (a.ptrs) return a.ptrs + m0 * a.taps;
  const T** out = scratch_of<const T*>(Scratch::kRows, static_cast<size_t>(mr) * a.taps);
  for (int r = 0; r < mr; ++r) {
    for (int t = 0; t < a.taps; ++t) out[static_cast<size_t>(r) * a.taps + t] = piece(a, m0 + r, t);
  }
  return out;
}

// FP16 bits to FP32, exactly, as vcvtph2ps and fcvtl do it: a NaN keeps its payload
// and becomes quiet.
inline float half_to_float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16, exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0x1f) {
    bits = sign | 0x7f800000u | (man << 13) | (man ? 0x400000u : 0u);
  } else if (exp) {
    bits = sign | ((exp + 112) << 23) | (man << 13);
  } else {
    const float sub = static_cast<float>(man) * 5.9604644775390625e-8f;  // man * 2^-24, exact
    std::memcpy(&bits, &sub, 4);
    bits |= sign;
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// FP32 to FP16 bits, rounded to nearest with ties to even, as vcvtps2ph (immediate
// 0) and fcvtn do it: past 65504 by half a step or more infinity, a NaN quiet with
// its payload's top bits, under 2^-14 a subnormal.
inline uint16_t float_to_half(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000u);
  const uint32_t ax = x & 0x7fffffffu;
  if (ax > 0x7f800000u) return static_cast<uint16_t>(sign | 0x7e00u | ((ax >> 13) & 0x3ffu));
  if (ax >= 0x477ff000u) return static_cast<uint16_t>(sign | 0x7c00u);  // 65520 and up (and infinity)
  if (ax >= 0x38800000u) {  // a normal half: the exponent rebased, 13 bits rounded away
    uint32_t r = ax - 0x38000000u;
    r += 0xfffu + ((r >> 13) & 1u);
    return static_cast<uint16_t>(sign | (r >> 13));
  }
  if (ax <= 0x33000000u) return sign;  // 2^-25 and under: zero (2^-25 a tie, to the even 0)
  const int shift = 126 - static_cast<int>(ax >> 23);  // 14 .. 24: the value in steps of 2^-24
  const uint32_t m = (ax & 0x7fffffu) | 0x800000u, half = 1u << (shift - 1), rest = m & ((1u << shift) - 1);
  uint32_t q = m >> shift;
  if (rest > half || (rest == half && (q & 1u))) ++q;
  return static_cast<uint16_t>(sign | q);
}

// The output stage of columns [j0, j0 + w) of row i, their sums in s[0 .. w).
template <typename S>
inline void epilogue_scalar(const S* s, int64_t i, int j0, int w, const Epilogue& e) {
  const float lo = static_cast<float>(-e.out_zero), hi = static_cast<float>(255 - e.out_zero);
  for (int t = 0; t < w; ++t) {
    const int j = j0 + t;
    float v = static_cast<float>(s[t]);
    if (e.scale) v = v * e.scale[j];
    if (e.bias) v = v + e.bias[j];
    const int64_t ri = (i >> e.residual_shift) * e.residual_ld + j;
    if (e.residual_f32) v = v + e.residual_f32[ri];
    if (e.residual_u8) {
      v = v + static_cast<float>(static_cast<int32_t>(e.residual_u8[ri]) - e.residual_zero) * e.residual_scale;
    }
    if (e.residual_f16) v = v + half_to_float(e.residual_f16[ri]);
    if (e.relu) v = v > 0 ? v : 0.0f;
    if (e.out_u8) {
      float q = v * e.out_inv_scale;
      q = q > lo ? q : lo;  // as maxps and minps do it
      q = q < hi ? q : hi;
      e.out_u8[i * e.out_ld + j] = static_cast<uint8_t>(static_cast<int32_t>(nearbyintf(q)) + e.out_zero);
    } else if (e.out_f16) {
      e.out_f16[i * e.out_ld + j] = float_to_half(v);
    } else {
      e.out_f32[i * e.out_ld + j] = v;
    }
  }
}

}  // namespace
}  // namespace carbon::cpu::detail
