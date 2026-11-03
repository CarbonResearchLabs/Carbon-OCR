// The int8 GEMM on Arm64 with the dot-product instructions (carbon/src/cpu/kernels.h):
// Armv8.2-A dotprod and later (Graviton 2 to 4, Apple M, Neoverse, Cortex-A76 on).
// sdot by element: 4 rows x 16 columns of int32 sums in 16 registers, B 64 bytes
// per 4 k, A gathered from its rows 4 at a time and shifted to signed (a ^ 0x80 =
// a - 128; the shift comes back as 128 times B's column sums).  Built with
// -march=armv8.2-a+dotprod.

#include <arm_neon.h>

#include <cstring>

#include "epilogue_neon.h"

namespace carbon::cpu::detail {
namespace {

template <int MR>
void tile(const int8_t* ap, const int8_t* bp, int k4, const int32_t* fix, int32_t* buf) {
  int32x4_t acc[4][4];
  for (int r = 0; r < 4; ++r) {
    for (int g = 0; g < 4; ++g) acc[r][g] = vdupq_n_s32(0);
  }
  for (int kk = 0; kk < k4; ++kk) {
    const int8x16_t a = vld1q_s8(ap + kk * 16);
    const int8x16_t b0 = vld1q_s8(bp + kk * 64), b1 = vld1q_s8(bp + kk * 64 + 16);
    const int8x16_t b2 = vld1q_s8(bp + kk * 64 + 32), b3 = vld1q_s8(bp + kk * 64 + 48);
    acc[0][0] = vdotq_laneq_s32(acc[0][0], b0, a, 0);
    acc[0][1] = vdotq_laneq_s32(acc[0][1], b1, a, 0);
    acc[0][2] = vdotq_laneq_s32(acc[0][2], b2, a, 0);
    acc[0][3] = vdotq_laneq_s32(acc[0][3], b3, a, 0);
    if (MR > 1) {
      acc[1][0] = vdotq_laneq_s32(acc[1][0], b0, a, 1);
      acc[1][1] = vdotq_laneq_s32(acc[1][1], b1, a, 1);
      acc[1][2] = vdotq_laneq_s32(acc[1][2], b2, a, 1);
      acc[1][3] = vdotq_laneq_s32(acc[1][3], b3, a, 1);
    }
    if (MR > 2) {
      acc[2][0] = vdotq_laneq_s32(acc[2][0], b0, a, 2);
      acc[2][1] = vdotq_laneq_s32(acc[2][1], b1, a, 2);
      acc[2][2] = vdotq_laneq_s32(acc[2][2], b2, a, 2);
      acc[2][3] = vdotq_laneq_s32(acc[2][3], b3, a, 2);
    }
    if (MR > 3) {
      acc[3][0] = vdotq_laneq_s32(acc[3][0], b0, a, 3);
      acc[3][1] = vdotq_laneq_s32(acc[3][1], b1, a, 3);
      acc[3][2] = vdotq_laneq_s32(acc[3][2], b2, a, 3);
      acc[3][3] = vdotq_laneq_s32(acc[3][3], b3, a, 3);
    }
  }
  for (int r = 0; r < MR; ++r) {
    for (int g = 0; g < 4; ++g) vst1q_s32(buf + r * 16 + 4 * g, vaddq_s32(acc[r][g], vld1q_s32(fix + 4 * g)));
  }
}

}  // namespace

void gemm_u8s8_neondot(Rows<uint8_t> a, int32_t a_zero, const BView& b, int m, const Epilogue& e) {
  const int taps = b.taps, seg4 = b.segp / 4, k4 = taps * seg4, panels = b.np / 16;
  int32_t* fix = scratch_of<int32_t>(Scratch::kFix, b.np);  // the shift to signed and A's zero point
  for (int j = 0; j < b.np; ++j) fix[j] = j < b.n ? (128 - a_zero) * b.col_sums[j] : 0;
  const size_t ap_bytes = static_cast<size_t>(16) * k4;
  int8_t* ap = scratch_of<int8_t>(Scratch::kA, ap_bytes);
  alignas(16) int32_t buf[4 * 16];
  for (int m0 = 0; m0 < m; m0 += 4) {
    const int mr = min_of(4, m - m0);
    // A's 4 rows interleaved by 4 bytes, [k/4][4 rows][4], shifted to signed; rows
    // past m are zeros (their sums are dropped)
    std::memset(ap, 0, ap_bytes);
    for (int r = 0; r < mr; ++r) {
      for (int t = 0; t < taps; ++t) {
        const uint8_t* src = piece(a, m0 + r, t);
        for (int kk = 0; kk < seg4; ++kk) {
          uint32_t v;
          std::memcpy(&v, src + kk * 4, 4);
          v ^= 0x80808080u;
          std::memcpy(ap + ((t * seg4 + kk) * 4 + r) * 4, &v, 4);
        }
      }
    }
    for (int pj = 0; pj < panels; ++pj) {
      const int8_t* bp = b.data + static_cast<size_t>(pj) * k4 * 64;
      const int32_t* f = fix + pj * 16;
      switch (mr) {
        case 4: tile<4>(ap, bp, k4, f, buf); break;
        case 3: tile<3>(ap, bp, k4, f, buf); break;
        case 2: tile<2>(ap, bp, k4, f, buf); break;
        default: tile<1>(ap, bp, k4, f, buf); break;
      }
      const int w = min_of(16, b.n - pj * 16);
      epilogue_neon(buf, 16, mr, m0, pj * 16, w, e);
    }
  }
}

}  // namespace carbon::cpu::detail
