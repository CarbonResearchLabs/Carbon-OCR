// The output stage on AVX-512, 16 columns at a time, masked at the end of a row
// (kernels.h), for the files built with AVX-512: avx512.cpp and avx512fp16.cpp.
// FP16 residuals and outputs through AVX-512 F's vcvtph2ps and vcvtps2ph
// (rounding to nearest even), for the FP16 kernels.
#pragma once

#include <immintrin.h>

#include <type_traits>

#include "kernels.h"

namespace carbon::cpu::detail {
namespace {  // internal linkage: each file its own copy (kernels.h)

// A tile's output stage, as epilogue_avx2 (epilogue_avx2.h).
template <typename S>
void epilogue_avx512(const S* s, int ld, int rows, int64_t i0, int j0, int w, const Epilogue& e) {
  const float* const scale = e.scale;
  const float* const bias = e.bias;
  const float* const res_f32 = e.residual_f32;
  const uint8_t* const res_u8 = e.residual_u8;
  const uint16_t* const res_f16 = e.residual_f16;
  const int64_t res_ld = e.residual_ld, out_ld = e.out_ld;
  const int rshift = e.residual_shift;
  const bool relu = e.relu;
  float* const out_f32 = e.out_f32;
  uint8_t* const out_u8 = e.out_u8;
  uint16_t* const out_f16 = e.out_f16;
  const __m512 lo = _mm512_set1_ps(static_cast<float>(-e.out_zero));
  const __m512 hi = _mm512_set1_ps(static_cast<float>(255 - e.out_zero));
  const __m512 inv = _mm512_set1_ps(e.out_inv_scale), rscale = _mm512_set1_ps(e.residual_scale);
  const __m512i ozero = _mm512_set1_epi32(e.out_zero), rzero = _mm512_set1_epi32(e.residual_zero);
  for (int t = 0; t < w; t += 16) {
    const int64_t j = j0 + t;
    const __mmask16 k = w - t >= 16 ? static_cast<__mmask16>(0xffff) : static_cast<__mmask16>((1u << (w - t)) - 1);
    const __m512 sc = scale ? _mm512_maskz_loadu_ps(k, scale + j) : _mm512_setzero_ps();
    const __m512 bi = bias ? _mm512_maskz_loadu_ps(k, bias + j) : _mm512_setzero_ps();
    for (int r = 0; r < rows; ++r) {
      const int64_t i = i0 + r;
      __m512 v;
      if constexpr (std::is_same_v<S, int32_t>) {
        v = _mm512_cvtepi32_ps(_mm512_maskz_loadu_epi32(k, s + r * ld + t));
      } else {
        v = _mm512_maskz_loadu_ps(k, s + r * ld + t);
      }
      if (scale) v = _mm512_mul_ps(v, sc);
      if (bias) v = _mm512_add_ps(v, bi);
      if (res_f32) v = _mm512_add_ps(v, _mm512_maskz_loadu_ps(k, res_f32 + (i >> rshift) * res_ld + j));
      if (res_u8) {
        const __m128i r8 = _mm_maskz_loadu_epi8(k, res_u8 + (i >> rshift) * res_ld + j);
        const __m512i rv = _mm512_sub_epi32(_mm512_cvtepu8_epi32(r8), rzero);
        v = _mm512_add_ps(v, _mm512_mul_ps(_mm512_cvtepi32_ps(rv), rscale));
      }
      if (res_f16) v = _mm512_add_ps(v, _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(k, res_f16 + (i >> rshift) * res_ld + j)));
      if (relu) v = _mm512_max_ps(v, _mm512_setzero_ps());
      if (out_u8) {
        const __m512 q = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(v, inv), lo), hi);
        _mm512_mask_cvtusepi32_storeu_epi8(out_u8 + i * out_ld + j, k, _mm512_add_epi32(_mm512_cvtps_epi32(q), ozero));
      } else if (out_f16) {
        _mm256_mask_storeu_epi16(out_f16 + i * out_ld + j, k, _mm512_cvtps_ph(v, 0));
      } else {
        _mm512_mask_storeu_ps(out_f32 + i * out_ld + j, k, v);
      }
    }
  }
}

}  // namespace
}  // namespace carbon::cpu::detail
