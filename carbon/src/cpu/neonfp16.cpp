// The FP16 GEMM and depthwise convolution on Arm's FP16 arithmetic (Armv8.2-A
// FEAT_FP16; carbon/src/cpu/kernels.h): fmla on 8 halves, twice the lanes of FP32,
// on the Arm servers and cores of the last years (Graviton 2 to 4, Neoverse N1 on,
// Apple M, Cortex-A55 and A75 on).  The products are summed in FP16 for runs of at
// most kF16Run values of k (carbon/cpu.h), and each run's sums widened (fcvtl) and
// added to FP32 ones, so an FP16 sum rounds over at most 64 products, not over a
// layer's whole k.
//
// GEMM: 8 rows x 16 columns, 16 accumulators of 8 halves, A broadcast from memory
// (ld1r) a value at a time.  Depthwise: 4 pixels x 16 channels a tile; a row's last
// channels (fewer than 16) as the reference sums them, in FP32.  The output stage
// is neon's on the FP32 sums (epilogue_neon.h).  FP32 activations run on neon.cpp's
// FP16 kernels.  Built with -march=armv8.2-a+fp16+dotprod.

#include <arm_neon.h>

#include <math.h>

#include "epilogue_neon.h"
#include "kernels.h"

namespace carbon::cpu::detail {
namespace {

inline float16x8_t zero8() { return vreinterpretq_f16_u16(vdupq_n_u16(0)); }
inline float16x8_t load8(const uint16_t* p) { return vreinterpretq_f16_u16(vld1q_u16(p)); }

// The 8 FP16 sums of acc added to the 8 FP32 sums at s.
inline void add_run(float16x8_t acc, float* s) {
  vst1q_f32(s, vaddq_f32(vld1q_f32(s), vcvt_f32_f16(vget_low_f16(acc))));
  vst1q_f32(s + 4, vaddq_f32(vld1q_f32(s + 4), vcvt_high_f32_f16(acc)));
}

// MR rows x 16 columns into buf ([MR][16] FP32 sums), the products of each run of
// kF16Run values of k (in the order of k, across the taps) summed in FP16 first.
template <int MR>
void tile(const uint16_t* const* rows, int taps, int seg, const uint16_t* bp, float* buf) {
  float16x8_t acc[MR][2];
  for (int r = 0; r < MR; ++r) {
    for (int g = 0; g < 4; ++g) vst1q_f32(buf + r * 16 + 4 * g, vdupq_n_f32(0.0f));
    acc[r][0] = acc[r][1] = zero8();
  }
  auto flush = [&] {
    for (int r = 0; r < MR; ++r) {
      add_run(acc[r][0], buf + r * 16);
      add_run(acc[r][1], buf + r * 16 + 8);
      acc[r][0] = acc[r][1] = zero8();
    }
  };
  int run = 0;
  for (int t = 0; t < taps; ++t) {
    const uint16_t* p[MR];
    for (int r = 0; r < MR; ++r) p[r] = rows[r * taps + t];
    for (int kk = 0; kk < seg;) {
      const int end = kk + min_of(seg - kk, kF16Run - run);
      run += end - kk;
      for (; kk < end; ++kk, bp += 16) {
        const float16x8_t b0 = load8(bp), b1 = load8(bp + 8);
        for (int r = 0; r < MR; ++r) {
          const float16x8_t av = vreinterpretq_f16_u16(vld1q_dup_u16(p[r] + kk));
          acc[r][0] = vfmaq_f16(acc[r][0], av, b0);
          acc[r][1] = vfmaq_f16(acc[r][1], av, b1);
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

// PX pixels x 16 channels into buf (a row of 64 floats a pixel): FP16 sums over
// runs of kF16Run taps, widened and added to FP32 ones.
template <int PX>
void dw_tile(const Rows<uint16_t>& a, int64_t i0, int taps, const uint16_t* w, int c, int off, float* buf) {
  float32x4_t s[PX][4];
  for (int r = 0; r < PX; ++r) {
    for (int g = 0; g < 4; ++g) s[r][g] = vdupq_n_f32(0.0f);
  }
  for (int t0 = 0; t0 < taps; t0 += kF16Run) {
    const int te = min_of(taps, t0 + kF16Run);
    float16x8_t acc[PX][2];
    for (int r = 0; r < PX; ++r) acc[r][0] = acc[r][1] = zero8();
    for (int t = t0; t < te; ++t) {
      const uint16_t* wt = w + static_cast<int64_t>(t) * c + off;
      const float16x8_t w0 = load8(wt), w1 = load8(wt + 8);
      for (int r = 0; r < PX; ++r) {
        const uint16_t* x = piece(a, i0 + r, t) + off;
        acc[r][0] = vfmaq_f16(acc[r][0], load8(x), w0);
        acc[r][1] = vfmaq_f16(acc[r][1], load8(x + 8), w1);
      }
    }
    for (int r = 0; r < PX; ++r) {
      s[r][0] = vaddq_f32(s[r][0], vcvt_f32_f16(vget_low_f16(acc[r][0])));
      s[r][1] = vaddq_f32(s[r][1], vcvt_high_f32_f16(acc[r][0]));
      s[r][2] = vaddq_f32(s[r][2], vcvt_f32_f16(vget_low_f16(acc[r][1])));
      s[r][3] = vaddq_f32(s[r][3], vcvt_high_f32_f16(acc[r][1]));
    }
  }
  for (int r = 0; r < PX; ++r) {
    for (int g = 0; g < 4; ++g) vst1q_f32(buf + r * 64 + 4 * g, s[r][g]);
  }
}

}  // namespace

void gemm_f16_neonfp16(Rows<uint16_t> a, const F16View& b, int m, const Epilogue& e) {
  const int taps = a.taps, seg = b.k / taps, panels = b.np / 16;
  alignas(16) float buf[8 * 16];
  for (int m0 = 0; m0 < m; m0 += 8) {
    const int mr = min_of(8, m - m0);
    const uint16_t* const* rows = block(a, m0, mr);
    for (int pj = 0; pj < panels; ++pj) {
      const uint16_t* bp = b.data + static_cast<size_t>(pj) * b.k * 16;
      switch (mr) {
        case 8: tile<8>(rows, taps, seg, bp, buf); break;
        case 7: tile<7>(rows, taps, seg, bp, buf); break;
        case 6: tile<6>(rows, taps, seg, bp, buf); break;
        case 5: tile<5>(rows, taps, seg, bp, buf); break;
        case 4: tile<4>(rows, taps, seg, bp, buf); break;
        case 3: tile<3>(rows, taps, seg, bp, buf); break;
        case 2: tile<2>(rows, taps, seg, bp, buf); break;
        default: tile<1>(rows, taps, seg, bp, buf); break;
      }
      epilogue_neon(buf, 16, mr, m0, pj * 16, min_of(16, b.n - pj * 16), e);
    }
  }
}

void dwconv_f16_neonfp16(Rows<uint16_t> a, const uint16_t* w, int c, int m, const Epilogue& e) {
  const int taps = a.taps;
  alignas(16) float buf[4 * 64];
  for (int i0 = 0; i0 < m; i0 += 4) {
    const int px = min_of(4, m - i0);
    for (int c0 = 0; c0 < c; c0 += 64) {
      const int cw = min_of(64, c - c0);
      int cc = 0;
      for (; cc + 16 <= cw; cc += 16) {
        switch (px) {
          case 4: dw_tile<4>(a, i0, taps, w, c, c0 + cc, buf + cc); break;
          case 3: dw_tile<3>(a, i0, taps, w, c, c0 + cc, buf + cc); break;
          case 2: dw_tile<2>(a, i0, taps, w, c, c0 + cc, buf + cc); break;
          default: dw_tile<1>(a, i0, taps, w, c, c0 + cc, buf + cc); break;
        }
      }
      for (int r = 0; r < px; ++r) {
        for (int ct = cc; ct < cw; ++ct) {
          float s = 0.0f;
          for (int t = 0; t < taps; ++t) {
            s = fmaf(half_to_float(piece(a, i0 + r, t)[c0 + ct]), half_to_float(w[t * c + c0 + ct]), s);
          }
          buf[r * 64 + ct] = s;
        }
      }
      epilogue_neon(buf, 64, px, i0, c0, cw, e);
    }
  }
}

}  // namespace carbon::cpu::detail
