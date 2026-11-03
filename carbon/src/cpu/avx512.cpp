// The GEMMs on AVX-512 (carbon/src/cpu/kernels.h), 32 columns a tile, A read in
// place through its rows' pointers.
//
// int8: vpdpbusd on 512 bits, 12 rows x 32 columns of int32 sums in 24 registers,
// B streamed 128 bytes per 4 k.
// FP32: vfmadd231ps on 512 bits, 12 rows x 32 columns.
// FP16 without FP16 arithmetic (AMD Zen 4 and 5, Intel before Sapphire Rapids): as
// avx2.cpp does it, on 512 bits (AVX-512 F's vcvtph2ps).
// The output stage is 16 columns at a time, masked at the end of a row.
// Built with -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni (/arch:AVX512).

#include <immintrin.h>

#include <cstring>
#include <type_traits>

#include "epilogue_avx512.h"
#include "kernels.h"

namespace carbon::cpu::detail {
namespace {

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
      epilogue_avx512(buf, 32, mr, m0, pj * 32, w, e);
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
      epilogue_avx512(buf, 32, mr, m0, pj * 32, w, e);
    }
  }
}

// ---------------------------------------------------------------- FP16
//
// FP16 stored, FP32 arithmetic, as avx2.cpp does it on 512 bits: B widened whole
// and kept from call to call up to kKeepB (past it a panel at a time); FP16 A a
// block of 12 rows at a time widened into the kA scratch, a piece of k values a
// row.  The reference's bits.  (Depthwise and the conversions are avx2's.)

namespace {

// n FP16 values widened: 16 an instruction, the rest one at a time (the same bits).
inline void widen_f16(const uint16_t* src, float* dst, int64_t n) {
  int64_t i = 0;
  for (; i + 16 <= n; i += 16) {
    _mm512_storeu_ps(dst + i, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + i))));
  }
  for (; i < n; ++i) dst[i] = half_to_float(src[i]);
}

// All of B widened (k x np floats, its panels as packed), as the last call on the
// same packing left it where it fits under kKeepB.
const float* widened(const F16View& b) {
  const size_t n = static_cast<size_t>(b.k) * b.np;
  bool kept = false;
  float* const bw = static_cast<float*>(
      n * sizeof(float) <= kKeepB ? scratch_kept(Scratch::kB, n * sizeof(float), b.id, kept)
                                  : scratch(Scratch::kB, n * sizeof(float)));
  if (!kept) widen_f16(b.data, bw, static_cast<int64_t>(n));
  return bw;
}

void gemm_f16_wide(const Rows<float>& a, const F16View& b, int m, const Epilogue& e) {
  const int taps = a.taps, k = b.k, seg = k / taps, panels = b.np / 32;
  const size_t panel = static_cast<size_t>(k) * 32;
  const bool whole = panel * panels * sizeof(float) <= kKeepB;
  const float* const bw = whole ? widened(b) : nullptr;
  float* const bp = whole ? nullptr : scratch_of<float>(Scratch::kB, panel);
  const float* const* all = block(a, 0, m);
  alignas(64) float buf[12 * 32];
  for (int pj = 0; pj < panels; ++pj) {
    if (!whole) widen_f16(b.data + pj * panel, bp, static_cast<int64_t>(panel));
    const float* const p = whole ? bw + pj * panel : bp;
    for (int m0 = 0; m0 < m; m0 += 12) {
      const int mr = min_of(12, m - m0);
      const float* const* rows = all + static_cast<size_t>(m0) * taps;
      by_rows(mr, [&](auto rows_c) { tile_f32<decltype(rows_c)::value>(rows, taps, seg, p, buf); });
      epilogue_avx512(buf, 32, mr, m0, pj * 32, min_of(32, b.n - pj * 32), e);
    }
  }
}

void gemm_f16_half(const Rows<uint16_t>& a, const F16View& b, int m, const Epilogue& e) {
  const int taps = a.taps, k = b.k, seg = k / taps, panels = b.np / 32;
  const float* const bw = widened(b);
  float* const a32 = scratch_of<float>(Scratch::kA, static_cast<size_t>(12) * k);
  const float* rows[12];
  alignas(64) float buf[12 * 32];
  for (int m0 = 0; m0 < m; m0 += 12) {
    const int mr = min_of(12, m - m0);
    for (int r = 0; r < mr; ++r) {
      float* row = a32 + static_cast<size_t>(r) * k;
      for (int t = 0; t < taps; ++t) widen_f16(piece(a, m0 + r, t), row + static_cast<size_t>(t) * seg, seg);
      rows[r] = row;
    }
    for (int pj = 0; pj < panels; ++pj) {
      const float* bp = bw + static_cast<size_t>(pj) * k * 32;
      by_rows(mr, [&](auto rows_c) { tile_f32<decltype(rows_c)::value>(rows, 1, k, bp, buf); });
      epilogue_avx512(buf, 32, mr, m0, pj * 32, min_of(32, b.n - pj * 32), e);
    }
  }
}

}  // namespace

void gemm_f16_avx512(Rows<uint16_t> a, const F16View& b, int m, const Epilogue& e) { gemm_f16_half(a, b, m, e); }
void gemm_f16_avx512(Rows<float> a, const F16View& b, int m, const Epilogue& e) { gemm_f16_wide(a, b, m, e); }

}  // namespace carbon::cpu::detail
