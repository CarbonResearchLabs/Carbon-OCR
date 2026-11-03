// The GEMMs on AVX-512 (carbon/src/cpu/kernels.h), 32 columns a tile, A read in
// place through its rows' pointers.
//
// int8: vpdpbusd on 512 bits, 12 rows x 32 columns of int32 sums in 24 registers,
// B streamed 128 bytes per 4 k.
// FP32: vfmadd231ps on 512 bits, 12 rows x 32 columns.
// The output stage is 16 columns at a time, masked at the end of a row.
// Built with -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni (/arch:AVX512).

#include <immintrin.h>

#include <cstring>
#include <type_traits>

#include "kernels.h"

namespace carbon::cpu::detail {
namespace {

// A tile's output stage, as epilogue_avx2 (epilogue_avx2.h), 16 columns at a
// time, masked at the end of a row.
template <typename S>
void epilogue(const S* s, int ld, int rows, int64_t i0, int j0, int w, const Epilogue& e) {
  const float* const scale = e.scale;
  const float* const bias = e.bias;
  const float* const res_f32 = e.residual_f32;
  const uint8_t* const res_u8 = e.residual_u8;
  const int64_t res_ld = e.residual_ld, out_ld = e.out_ld;
  const bool relu = e.relu;
  float* const out_f32 = e.out_f32;
  uint8_t* const out_u8 = e.out_u8;
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
      if (res_f32) v = _mm512_add_ps(v, _mm512_maskz_loadu_ps(k, res_f32 + i * res_ld + j));
      if (res_u8) {
        const __m128i r8 = _mm_maskz_loadu_epi8(k, res_u8 + i * res_ld + j);
        const __m512i rv = _mm512_sub_epi32(_mm512_cvtepu8_epi32(r8), rzero);
        v = _mm512_add_ps(v, _mm512_mul_ps(_mm512_cvtepi32_ps(rv), rscale));
      }
      if (relu) v = _mm512_max_ps(v, _mm512_setzero_ps());
      if (out_u8) {
        const __m512 q = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(v, inv), lo), hi);
        _mm512_mask_cvtusepi32_storeu_epi8(out_u8 + i * out_ld + j, k, _mm512_add_epi32(_mm512_cvtps_epi32(q), ozero));
      } else {
        _mm512_mask_storeu_ps(out_f32 + i * out_ld + j, k, v);
      }
    }
  }
}

template <int MR>
void tile_u8(const uint8_t* const* rows, int taps, int seg4, const int8_t* bp, const int32_t* fix, int32_t* buf) {
  __m512i acc[MR][2];
  for (int r = 0; r < MR; ++r) acc[r][0] = acc[r][1] = _mm512_setzero_si512();
  for (int t = 0; t < taps; ++t) {
    const uint8_t* p[MR];
    for (int r = 0; r < MR; ++r) p[r] = rows[r * taps + t];
    for (int kk = 0; kk < seg4; ++kk, bp += 128) {
      const __m512i b0 = _mm512_loadu_si512(bp), b1 = _mm512_loadu_si512(bp + 64);
      for (int r = 0; r < MR; ++r) {
        int32_t a4;
        std::memcpy(&a4, p[r] + kk * 4, 4);
        const __m512i av = _mm512_set1_epi32(a4);
        acc[r][0] = _mm512_dpbusd_epi32(acc[r][0], av, b0);
        acc[r][1] = _mm512_dpbusd_epi32(acc[r][1], av, b1);
      }
    }
  }
  const __m512i f0 = _mm512_loadu_si512(fix), f1 = _mm512_loadu_si512(fix + 16);
  for (int r = 0; r < MR; ++r) {
    _mm512_store_si512(buf + r * 32, _mm512_add_epi32(acc[r][0], f0));
    _mm512_store_si512(buf + r * 32 + 16, _mm512_add_epi32(acc[r][1], f1));
  }
}

template <int MR>
void tile_f32(const float* const* rows, int taps, int seg, const float* bp, float* buf) {
  __m512 acc[MR][2];
  for (int r = 0; r < MR; ++r) acc[r][0] = acc[r][1] = _mm512_setzero_ps();
  for (int t = 0; t < taps; ++t) {
    const float* p[MR];
    for (int r = 0; r < MR; ++r) p[r] = rows[r * taps + t];
    for (int kk = 0; kk < seg; ++kk, bp += 32) {
      const __m512 b0 = _mm512_loadu_ps(bp), b1 = _mm512_loadu_ps(bp + 16);
      for (int r = 0; r < MR; ++r) {
        const __m512 av = _mm512_set1_ps(p[r][kk]);
        acc[r][0] = _mm512_fmadd_ps(av, b0, acc[r][0]);
        acc[r][1] = _mm512_fmadd_ps(av, b1, acc[r][1]);
      }
    }
  }
  for (int r = 0; r < MR; ++r) {
    _mm512_store_ps(buf + r * 32, acc[r][0]);
    _mm512_store_ps(buf + r * 32 + 16, acc[r][1]);
  }
}

// Calls tile with a block's rows, 1 to 12, as a constant.
template <typename F>
void by_rows(int mr, F&& tile) {
  switch (mr) {
    case 12: tile(std::integral_constant<int, 12>{}); break;
    case 11: tile(std::integral_constant<int, 11>{}); break;
    case 10: tile(std::integral_constant<int, 10>{}); break;
    case 9: tile(std::integral_constant<int, 9>{}); break;
    case 8: tile(std::integral_constant<int, 8>{}); break;
    case 7: tile(std::integral_constant<int, 7>{}); break;
    case 6: tile(std::integral_constant<int, 6>{}); break;
    case 5: tile(std::integral_constant<int, 5>{}); break;
    case 4: tile(std::integral_constant<int, 4>{}); break;
    case 3: tile(std::integral_constant<int, 3>{}); break;
    case 2: tile(std::integral_constant<int, 2>{}); break;
    default: tile(std::integral_constant<int, 1>{}); break;
  }
}

}  // namespace

void gemm_u8s8_avx512(Rows<uint8_t> a, int32_t a_zero, const BView& b, int m, const Epilogue& e) {
  const int taps = b.taps, seg4 = b.segp / 4, panels = b.np / 32;
  int32_t* fix = scratch_of<int32_t>(Scratch::kFix, b.np);  // A's zero point, after the products
  for (int j = 0; j < b.np; ++j) fix[j] = j < b.n ? -a_zero * b.col_sums[j] : 0;
  alignas(64) int32_t buf[12 * 32];
  for (int m0 = 0; m0 < m; m0 += 12) {
    const int mr = min_of(12, m - m0);
    const uint8_t* const* rows = block(a, m0, mr);
    for (int pj = 0; pj < panels; ++pj) {
      const int8_t* bp = b.data + static_cast<size_t>(pj) * taps * seg4 * 128;
      const int32_t* f = fix + pj * 32;
      by_rows(mr, [&](auto rows_c) { tile_u8<decltype(rows_c)::value>(rows, taps, seg4, bp, f, buf); });
      const int w = min_of(32, b.n - pj * 32);
      epilogue(buf, 32, mr, m0, pj * 32, w, e);
    }
  }
}

void gemm_f32_avx512(Rows<float> a, const F32View& b, int m, const Epilogue& e) {
  const int taps = a.taps, seg = b.k / taps, panels = b.np / 32;
  alignas(64) float buf[12 * 32];
  for (int m0 = 0; m0 < m; m0 += 12) {
    const int mr = min_of(12, m - m0);
    const float* const* rows = block(a, m0, mr);
    for (int pj = 0; pj < panels; ++pj) {
      const float* bp = b.data + static_cast<size_t>(pj) * b.k * 32;
      by_rows(mr, [&](auto rows_c) { tile_f32<decltype(rows_c)::value>(rows, taps, seg, bp, buf); });
      const int w = min_of(32, b.n - pj * 32);
      epilogue(buf, 32, mr, m0, pj * 32, w, e);
    }
  }
}

}  // namespace carbon::cpu::detail
