// The FP16 GEMM and depthwise convolution on AVX-512 FP16 (carbon/src/cpu/kernels.h):
// Intel Sapphire Rapids and later servers, the x86-64 CPUs with FP16 arithmetic.
// vfmadd231ph makes 32 FP16 products and sums an instruction, twice vfmadd231ps's
// 16.  The products are summed in FP16 for runs of at most kF16Run values of k
// (carbon/cpu.h), and each run's sums widened (vcvtph2ps) and added to FP32 ones,
// so an FP16 sum rounds over at most 64 products, not over a layer's whole k.
//
// GEMM: 12 rows x 64 columns (two panels of 32), 24 accumulators, A broadcast a
// value at a time, B streamed 128 bytes a k.  Depthwise: 4 pixels x 32 channels a
// tile, the channels past c masked off.  The output stage is avx512's on the FP32
// sums (epilogue_avx512.h).  FP32 activations run on avx512.cpp's FP16 kernels.
// Built with -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512fp16 (/arch:AVX512).
//
// No machine of this project's has AVX-512 FP16: the file is compiled in every x86-64
// build and held to the reference (tests/cpp/cpu_test.cpp, net_test.cpp) only on
// a CPU that has it.  Its logic is the reference's loops with FP16 vectors.

#include <immintrin.h>

#include <type_traits>

#include "epilogue_avx512.h"
#include "kernels.h"

namespace carbon::cpu::detail {
namespace {

inline __m512h bcast(uint16_t h) { return _mm512_castsi512_ph(_mm512_set1_epi16(static_cast<short>(h))); }
inline __m512h load_ph(const uint16_t* p) { return _mm512_castsi512_ph(_mm512_loadu_si512(p)); }

// The 32 FP16 sums of acc added to the 32 FP32 sums at s.
inline void add_run(__m512h acc, float* s) {
  const __m512i bits = _mm512_castph_si512(acc);
  const __m512 lo = _mm512_cvtph_ps(_mm512_castsi512_si256(bits));
  const __m512 hi = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(bits, 1));
  _mm512_storeu_ps(s, _mm512_add_ps(_mm512_loadu_ps(s), lo));
  _mm512_storeu_ps(s + 16, _mm512_add_ps(_mm512_loadu_ps(s + 16), hi));
}

// MR rows x NP panels of 32 columns into buf ([MR][64] FP32 sums), the products of
// each run of kF16Run values of k (in the order of k, across the taps) summed in
// FP16 first.  b0 and b1: the panels, [k][32] each (b1 read only when NP is 2).
template <int MR, int NP>
void tile(const uint16_t* const* rows, int taps, int seg, const uint16_t* b0, [[maybe_unused]] const uint16_t* b1,
          float* buf) {
  __m512h acc[MR][NP];
  for (int r = 0; r < MR; ++r) {
    for (int q = 0; q < 4; ++q) _mm512_storeu_ps(buf + r * 64 + 16 * q, _mm512_setzero_ps());
    for (int p = 0; p < NP; ++p) acc[r][p] = _mm512_setzero_ph();
  }
  auto flush = [&] {
    for (int r = 0; r < MR; ++r) {
      for (int p = 0; p < NP; ++p) {
        add_run(acc[r][p], buf + r * 64 + 32 * p);
        acc[r][p] = _mm512_setzero_ph();
      }
    }
  };
  int run = 0;
  size_t at = 0;  // B's offset: k * 32
  for (int t = 0; t < taps; ++t) {
    const uint16_t* p[MR];
    for (int r = 0; r < MR; ++r) p[r] = rows[r * taps + t];
    for (int kk = 0; kk < seg;) {
      const int end = kk + min_of(seg - kk, kF16Run - run);
      run += end - kk;
      for (; kk < end; ++kk, at += 32) {
        const __m512h v0 = load_ph(b0 + at);
        if constexpr (NP == 2) {
          const __m512h v1 = load_ph(b1 + at);
          for (int r = 0; r < MR; ++r) {
            const __m512h av = bcast(p[r][kk]);
            acc[r][0] = _mm512_fmadd_ph(av, v0, acc[r][0]);
            acc[r][1] = _mm512_fmadd_ph(av, v1, acc[r][1]);
          }
        } else {
          for (int r = 0; r < MR; ++r) acc[r][0] = _mm512_fmadd_ph(bcast(p[r][kk]), v0, acc[r][0]);
        }
      }
      if (run == kF16Run) {
        flush();
        run = 0;
      }
    }
  }
  if (run) flush();
}

// Calls tile with a block's rows, 1 to 12, as a constant.
template <typename F>
void by_rows(int mr, F&& f) {
  switch (mr) {
    case 12: f(std::integral_constant<int, 12>{}); break;
    case 11: f(std::integral_constant<int, 11>{}); break;
    case 10: f(std::integral_constant<int, 10>{}); break;
    case 9: f(std::integral_constant<int, 9>{}); break;
    case 8: f(std::integral_constant<int, 8>{}); break;
    case 7: f(std::integral_constant<int, 7>{}); break;
    case 6: f(std::integral_constant<int, 6>{}); break;
    case 5: f(std::integral_constant<int, 5>{}); break;
    case 4: f(std::integral_constant<int, 4>{}); break;
    case 3: f(std::integral_constant<int, 3>{}); break;
    case 2: f(std::integral_constant<int, 2>{}); break;
    default: f(std::integral_constant<int, 1>{}); break;
  }
}

// PX pixels x 32 channels (those of `mask`) into buf (a row of 64 floats a pixel):
// FP16 sums over runs of kF16Run taps, widened and added to FP32 ones.
template <int PX>
void dw_tile(const Rows<uint16_t>& a, int64_t i0, int taps, const uint16_t* w, int c, int off, __mmask32 mask,
             float* buf) {
  __m512 s[PX][2];
  for (int r = 0; r < PX; ++r) s[r][0] = s[r][1] = _mm512_setzero_ps();
  for (int t0 = 0; t0 < taps; t0 += kF16Run) {
    const int te = min_of(taps, t0 + kF16Run);
    __m512h acc[PX];
    for (int r = 0; r < PX; ++r) acc[r] = _mm512_setzero_ph();
    for (int t = t0; t < te; ++t) {
      const __m512h wv = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, w + static_cast<int64_t>(t) * c + off));
      for (int r = 0; r < PX; ++r) {
        const __m512h x = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(mask, piece(a, i0 + r, t) + off));
        acc[r] = _mm512_fmadd_ph(x, wv, acc[r]);
      }
    }
    for (int r = 0; r < PX; ++r) {
      const __m512i bits = _mm512_castph_si512(acc[r]);
      s[r][0] = _mm512_add_ps(s[r][0], _mm512_cvtph_ps(_mm512_castsi512_si256(bits)));
      s[r][1] = _mm512_add_ps(s[r][1], _mm512_cvtph_ps(_mm512_extracti64x4_epi64(bits, 1)));
    }
  }
  for (int r = 0; r < PX; ++r) {
    _mm512_storeu_ps(buf + r * 64, s[r][0]);
    _mm512_storeu_ps(buf + r * 64 + 16, s[r][1]);
  }
}

}  // namespace

void gemm_f16_avx512fp16(Rows<uint16_t> a, const F16View& b, int m, const Epilogue& e) {
  const int taps = a.taps, seg = b.k / taps, panels = b.np / 32;
  alignas(64) float buf[12 * 64];
  for (int m0 = 0; m0 < m; m0 += 12) {
    const int mr = min_of(12, m - m0);
    const uint16_t* const* rows = block(a, m0, mr);
    for (int pj = 0; pj < panels; pj += 2) {
      const uint16_t* b0 = b.data + static_cast<size_t>(pj) * b.k * 32;
      if (pj + 1 < panels) {
        const uint16_t* b1 = b0 + static_cast<size_t>(b.k) * 32;
        by_rows(mr, [&](auto rows_c) { tile<decltype(rows_c)::value, 2>(rows, taps, seg, b0, b1, buf); });
      } else {
        by_rows(mr, [&](auto rows_c) { tile<decltype(rows_c)::value, 1>(rows, taps, seg, b0, b0, buf); });
      }
      epilogue_avx512(buf, 64, mr, m0, pj * 32, min_of(64, b.n - pj * 32), e);
    }
  }
}

void dwconv_f16_avx512fp16(Rows<uint16_t> a, const uint16_t* w, int c, int m, const Epilogue& e) {
  const int taps = a.taps;
  alignas(64) float buf[4 * 64];
  for (int i0 = 0; i0 < m; i0 += 4) {
    const int px = min_of(4, m - i0);
    for (int c0 = 0; c0 < c; c0 += 64) {
      const int cw = min_of(64, c - c0);
      for (int cc = 0; cc < cw; cc += 32) {
        const int n = min_of(32, cw - cc);
        const __mmask32 mask = n == 32 ? static_cast<__mmask32>(0xffffffffu) : static_cast<__mmask32>((1u << n) - 1);
        float* out = buf + cc;
        switch (px) {
          case 4: dw_tile<4>(a, i0, taps, w, c, c0 + cc, mask, out); break;
          case 3: dw_tile<3>(a, i0, taps, w, c, c0 + cc, mask, out); break;
          case 2: dw_tile<2>(a, i0, taps, w, c, c0 + cc, mask, out); break;
          default: dw_tile<1>(a, i0, taps, w, c, c0 + cc, mask, out); break;
        }
      }
      epilogue_avx512(buf, 64, px, i0, c0, cw, e);
    }
  }
}

}  // namespace carbon::cpu::detail
