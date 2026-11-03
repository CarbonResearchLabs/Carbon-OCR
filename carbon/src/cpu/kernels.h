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
//
// Each kernel sums a tile of rows and columns into a small buffer, then applies
// the output stage (carbon::cpu::Epilogue) to it a row at a time: vectors where a
// whole vector of columns is there, epilogue_scalar for the rest, the same float
// operations in the same order, so the bits are the reference's.
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
void ctc_argmax_neon(const float*, int, int, int, int32_t*, float*);

// A kernel's working memory: at least `bytes` bytes from a cache line's start, its
// thread's own, kept from call to call.  A buffer a slot, so one kernel's buffers do
// not overlap.
enum class Scratch { kFix, kRows, kA };
void* scratch(Scratch slot, size_t bytes);

namespace {

template <typename T>
inline T* scratch_of(Scratch slot, size_t n) {
  return static_cast<T*>(scratch(slot, n * sizeof(T)));
}

inline int min_of(int a, int b) { return a < b ? a : b; }

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

// The output stage of columns [j0, j0 + w) of row i, their sums in s[0 .. w).
template <typename S>
inline void epilogue_scalar(const S* s, int64_t i, int j0, int w, const Epilogue& e) {
  const float lo = static_cast<float>(-e.out_zero), hi = static_cast<float>(255 - e.out_zero);
  for (int t = 0; t < w; ++t) {
    const int j = j0 + t;
    float v = static_cast<float>(s[t]);
    if (e.scale) v = v * e.scale[j];
    if (e.bias) v = v + e.bias[j];
    if (e.residual_f32) v = v + e.residual_f32[i * e.residual_ld + j];
    if (e.residual_u8) {
      v = v + static_cast<float>(static_cast<int32_t>(e.residual_u8[i * e.residual_ld + j]) - e.residual_zero) *
                  e.residual_scale;
    }
    if (e.relu) v = v > 0 ? v : 0.0f;
    if (e.out_u8) {
      float q = v * e.out_inv_scale;
      q = q > lo ? q : lo;  // as maxps and minps do it
      q = q < hi ? q : hi;
      e.out_u8[i * e.out_ld + j] = static_cast<uint8_t>(static_cast<int32_t>(nearbyintf(q)) + e.out_zero);
    } else {
      e.out_f32[i * e.out_ld + j] = v;
    }
  }
}

}  // namespace
}  // namespace carbon::cpu::detail
