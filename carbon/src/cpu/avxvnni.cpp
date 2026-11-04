// The int8 GEMM on AVX-VNNI (carbon/src/cpu/kernels.h): vpdpbusd on 256 bits, 6 rows
// x 16 columns of int32 sums in 12 registers, B streamed 64 bytes per 4 k, A read
// in place through its rows' pointers.  Built with -mavx2 -mfma -mf16c
// -mavxvnni (/arch:AVX2 with MSVC).

#include <immintrin.h>

#include <cstring>

#include "epilogue_avx2.h"

namespace carbon::cpu::detail {
namespace {

template <int MR>
void tile(const uint8_t* const* rows, int taps, int seg4, const int8_t* bp, const int32_t* fix, int32_t* buf) {
  __m256i acc[MR][2];
  for (int r = 0; r < MR; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_si256();
  for (int t = 0; t < taps; ++t) {
    const uint8_t* p[MR];
    for (int r = 0; r < MR; ++r) p[r] = rows[r * taps + t];
    for (int kk = 0; kk < seg4; ++kk, bp += 64) {
      const __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bp));
      const __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bp + 32));
      for (int r = 0; r < MR; ++r) {
        int32_t a4;
        std::memcpy(&a4, p[r] + kk * 4, 4);
        const __m256i av = _mm256_set1_epi32(a4);
        acc[r][0] = _mm256_dpbusd_avx_epi32(acc[r][0], av, b0);
        acc[r][1] = _mm256_dpbusd_avx_epi32(acc[r][1], av, b1);
      }
    }
  }
  const __m256i f0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fix));
  const __m256i f1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fix + 8));
  for (int r = 0; r < MR; ++r) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(buf + r * 16), _mm256_add_epi32(acc[r][0], f0));
    _mm256_store_si256(reinterpret_cast<__m256i*>(buf + r * 16 + 8), _mm256_add_epi32(acc[r][1], f1));
  }
}

}  // namespace

void gemm_u8s8_avxvnni(Rows<uint8_t> a, int32_t a_zero, const BView& b, int m, const Epilogue& e) {
  const int taps = b.taps, seg4 = b.segp / 4, panels = b.np / 16;
  int32_t* fix = scratch_of<int32_t>(Scratch::kFix, b.np);  // A's zero point, after the products
  for (int j = 0; j < b.np; ++j) fix[j] = j < b.n ? -a_zero * b.col_sums[j] : 0;
  alignas(32) int32_t buf[6 * 16];
  for (int m0 = 0; m0 < m; m0 += 6) {
    const int mr = min_of(6, m - m0);
    const uint8_t* const* rows = block(a, m0, mr);
    for (int pj = 0; pj < panels; ++pj) {
      const int8_t* bp = b.data + static_cast<size_t>(pj) * taps * seg4 * 64;
      const int32_t* f = fix + pj * 16;
      switch (mr) {
        case 6: tile<6>(rows, taps, seg4, bp, f, buf); break;
        case 5: tile<5>(rows, taps, seg4, bp, f, buf); break;
        case 4: tile<4>(rows, taps, seg4, bp, f, buf); break;
        case 3: tile<3>(rows, taps, seg4, bp, f, buf); break;
        case 2: tile<2>(rows, taps, seg4, bp, f, buf); break;
        default: tile<1>(rows, taps, seg4, bp, f, buf); break;
      }
      const int w = min_of(16, b.n - pj * 16);
      epilogue_avx2(buf, 16, mr, m0, pj * 16, w, e);
    }
  }
}

}  // namespace carbon::cpu::detail
