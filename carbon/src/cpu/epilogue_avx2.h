// The output stage on AVX2, 8 columns at a time (kernels.h), for the files built
// with AVX2: avx2.cpp and avxvnni.cpp.
#pragma once

#include <immintrin.h>

#include <cstring>
#include <type_traits>

#include "kernels.h"

namespace carbon::cpu::detail {
namespace {  // internal linkage: each file its own copy (kernels.h)

// A tile's output stage: rows [i0, i0 + rows) and columns [j0, j0 + w) of the
// output, their sums in s, rows ld apart.  The stage's fields are read once
// (stores through the outputs could alias them), each column vector's scale and
// bias once for all the rows.
template <typename S>
inline void epilogue_avx2(const S* s, int ld, int rows, int64_t i0, int j0, int w, const Epilogue& e) {
  const float* const scale = e.scale;
  const float* const bias = e.bias;
  const float* const res_f32 = e.residual_f32;
  const uint8_t* const res_u8 = e.residual_u8;
  const int64_t res_ld = e.residual_ld, out_ld = e.out_ld;
  const bool relu = e.relu;
  float* const out_f32 = e.out_f32;
  uint8_t* const out_u8 = e.out_u8;
  const __m256 lo = _mm256_set1_ps(static_cast<float>(-e.out_zero));
  const __m256 hi = _mm256_set1_ps(static_cast<float>(255 - e.out_zero));
  const __m256 inv = _mm256_set1_ps(e.out_inv_scale), rscale = _mm256_set1_ps(e.residual_scale);
  const __m256i ozero = _mm256_set1_epi32(e.out_zero), rzero = _mm256_set1_epi32(e.residual_zero);
  int t = 0;
  for (; t + 8 <= w; t += 8) {
    const int64_t j = j0 + t;
    const __m256 sc = scale ? _mm256_loadu_ps(scale + j) : _mm256_setzero_ps();
    const __m256 bi = bias ? _mm256_loadu_ps(bias + j) : _mm256_setzero_ps();
    for (int r = 0; r < rows; ++r) {
      const int64_t i = i0 + r;
      __m256 v;
      if constexpr (std::is_same_v<S, int32_t>) {
        v = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + r * ld + t)));
      } else {
        v = _mm256_loadu_ps(s + r * ld + t);
      }
      if (scale) v = _mm256_mul_ps(v, sc);
      if (bias) v = _mm256_add_ps(v, bi);
      if (res_f32) v = _mm256_add_ps(v, _mm256_loadu_ps(res_f32 + i * res_ld + j));
      if (res_u8) {
        const __m128i r8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(res_u8 + i * res_ld + j));
        const __m256i rv = _mm256_sub_epi32(_mm256_cvtepu8_epi32(r8), rzero);
        v = _mm256_add_ps(v, _mm256_mul_ps(_mm256_cvtepi32_ps(rv), rscale));
      }
      if (relu) v = _mm256_max_ps(v, _mm256_setzero_ps());
      if (out_u8) {
        const __m256 q = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v, inv), lo), hi);
        const __m256i qi = _mm256_add_epi32(_mm256_cvtps_epi32(q), ozero);
        const __m128i p16 = _mm_packus_epi32(_mm256_castsi256_si128(qi), _mm256_extracti128_si256(qi, 1));
        _mm_storel_epi64(reinterpret_cast<__m128i*>(out_u8 + i * out_ld + j), _mm_packus_epi16(p16, p16));
      } else {
        _mm256_storeu_ps(out_f32 + i * out_ld + j, v);
      }
    }
  }
  if (t == w) return;
  // The last columns, fewer than 8: the same operations on a vector whose
  // lanes past w are loaded as 0 and not stored (s has 8 values from t on).
  const int left = w - t;
  const int64_t j = j0 + t;
  alignas(32) static constexpr int32_t kLanes[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  const __m256i lanes = _mm256_load_si256(reinterpret_cast<const __m256i*>(kLanes));
  const __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(left), lanes);
  const __m256 sc = scale ? _mm256_maskload_ps(scale + j, mask) : _mm256_setzero_ps();
  const __m256 bi = bias ? _mm256_maskload_ps(bias + j, mask) : _mm256_setzero_ps();
  for (int r = 0; r < rows; ++r) {
    const int64_t i = i0 + r;
    __m256 v;
    if constexpr (std::is_same_v<S, int32_t>) {
      v = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + r * ld + t)));
    } else {
      v = _mm256_loadu_ps(s + r * ld + t);
    }
    if (scale) v = _mm256_mul_ps(v, sc);
    if (bias) v = _mm256_add_ps(v, bi);
    if (res_f32) v = _mm256_add_ps(v, _mm256_maskload_ps(res_f32 + i * res_ld + j, mask));
    if (res_u8) {
      uint8_t r8[8] = {};
      std::memcpy(r8, res_u8 + i * res_ld + j, left);
      const __m128i r8v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(r8));
      const __m256i rv = _mm256_sub_epi32(_mm256_cvtepu8_epi32(r8v), rzero);
      v = _mm256_add_ps(v, _mm256_mul_ps(_mm256_cvtepi32_ps(rv), rscale));
    }
    if (relu) v = _mm256_max_ps(v, _mm256_setzero_ps());
    if (out_u8) {
      const __m256 q = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(v, inv), lo), hi);
      const __m256i qi = _mm256_add_epi32(_mm256_cvtps_epi32(q), ozero);
      const __m128i p16 = _mm_packus_epi32(_mm256_castsi256_si128(qi), _mm256_extracti128_si256(qi, 1));
      uint8_t q8[16];
      _mm_storeu_si128(reinterpret_cast<__m128i*>(q8), _mm_packus_epi16(p16, p16));
      std::memcpy(out_u8 + i * out_ld + j, q8, left);
    } else {
      _mm256_maskstore_ps(out_f32 + i * out_ld + j, mask, v);
    }
  }
}

}  // namespace
}  // namespace carbon::cpu::detail
