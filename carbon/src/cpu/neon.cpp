// The FP32 GEMM on Arm64 NEON (carbon/src/cpu/kernels.h), every Armv8-A CPU: fmla
// with A broadcast from memory (ld1r), 6 rows x 16 columns of sums in 24
// registers, A read in place through its rows' pointers.  No flags beyond the
// baseline.

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <limits>

#include "epilogue_neon.h"
#include "kernels.h"

namespace carbon::cpu::detail {
namespace {

template <int MR>
void tile(const float* const* rows, int taps, int seg, const float* bp, float* buf) {
  float32x4_t acc[MR][4];
  for (int r = 0; r < MR; ++r) {
    for (int g = 0; g < 4; ++g) acc[r][g] = vdupq_n_f32(0.0f);
  }
  for (int t = 0; t < taps; ++t) {
    const float* p[MR];
    for (int r = 0; r < MR; ++r) p[r] = rows[r * taps + t];
    for (int kk = 0; kk < seg; ++kk, bp += 16) {
      const float32x4_t b0 = vld1q_f32(bp), b1 = vld1q_f32(bp + 4), b2 = vld1q_f32(bp + 8), b3 = vld1q_f32(bp + 12);
      for (int r = 0; r < MR; ++r) {
        const float32x4_t av = vld1q_dup_f32(p[r] + kk);
        acc[r][0] = vfmaq_f32(acc[r][0], av, b0);
        acc[r][1] = vfmaq_f32(acc[r][1], av, b1);
        acc[r][2] = vfmaq_f32(acc[r][2], av, b2);
        acc[r][3] = vfmaq_f32(acc[r][3], av, b3);
      }
    }
  }
  for (int r = 0; r < MR; ++r) {
    for (int g = 0; g < 4; ++g) vst1q_f32(buf + r * 16 + 4 * g, acc[r][g]);
  }
}

}  // namespace

void gemm_f32_neon(Rows<float> a, const F32View& b, int m, const Epilogue& e) {
  const int taps = a.taps, seg = b.k / taps, panels = b.np / 16;
  alignas(16) float buf[6 * 16];
  for (int m0 = 0; m0 < m; m0 += 6) {
    const int mr = std::min(6, m - m0);
    const float* const* rows = block(a, m0, mr);
    for (int pj = 0; pj < panels; ++pj) {
      const float* bp = b.data + static_cast<size_t>(pj) * b.k * 16;
      switch (mr) {
        case 6: tile<6>(rows, taps, seg, bp, buf); break;
        case 5: tile<5>(rows, taps, seg, bp, buf); break;
        case 4: tile<4>(rows, taps, seg, bp, buf); break;
        case 3: tile<3>(rows, taps, seg, bp, buf); break;
        case 2: tile<2>(rows, taps, seg, bp, buf); break;
        default: tile<1>(rows, taps, seg, bp, buf); break;
      }
      const int w = std::min(16, b.n - pj * 16);
      epilogue_neon(buf, 16, mr, m0, pj * 16, w, e);
    }
  }
}

// ---------------------------------------------------------------- depthwise
//
// 4 pixels x 16 channels a tile (16 chains of sums), the channels in chunks of 64
// for the output stage.  int8 in 16 bits: (a - zero) and w both fit, and smlal
// widens their products into int32 exactly.

namespace {

template <int PX>
void dw_tile_f32(const Rows<float>& a, int64_t i0, int taps, const float* w, int c, int off, float* buf) {
  float32x4_t acc[PX][4];
  for (int r = 0; r < PX; ++r) {
    for (int g = 0; g < 4; ++g) acc[r][g] = vdupq_n_f32(0.0f);
  }
  for (int t = 0; t < taps; ++t) {
    float32x4_t wv[4];
    for (int g = 0; g < 4; ++g) wv[g] = vld1q_f32(w + t * c + 4 * g);
    for (int r = 0; r < PX; ++r) {
      const float* x = piece(a, i0 + r, t) + off;
      for (int g = 0; g < 4; ++g) acc[r][g] = vfmaq_f32(acc[r][g], vld1q_f32(x + 4 * g), wv[g]);
    }
  }
  for (int r = 0; r < PX; ++r) {
    for (int g = 0; g < 4; ++g) vst1q_f32(buf + r * 64 + 4 * g, acc[r][g]);
  }
}

template <int PX>
void dw_tile_u8(const Rows<uint8_t>& a, int64_t i0, int taps, int16x8_t zero, const int8_t* w, int c, int off,
                int32_t* buf) {
  int32x4_t acc[PX][4];
  for (int r = 0; r < PX; ++r) {
    for (int g = 0; g < 4; ++g) acc[r][g] = vdupq_n_s32(0);
  }
  for (int t = 0; t < taps; ++t) {
    const int8x16_t wv = vld1q_s8(w + t * c);
    const int16x8_t w0 = vmovl_s8(vget_low_s8(wv)), w1 = vmovl_high_s8(wv);
    for (int r = 0; r < PX; ++r) {
      const uint8x16_t xv = vld1q_u8(piece(a, i0 + r, t) + off);
      const int16x8_t x0 = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(xv))), zero);
      const int16x8_t x1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_high_u8(xv)), zero);
      acc[r][0] = vmlal_s16(acc[r][0], vget_low_s16(x0), vget_low_s16(w0));
      acc[r][1] = vmlal_high_s16(acc[r][1], x0, w0);
      acc[r][2] = vmlal_s16(acc[r][2], vget_low_s16(x1), vget_low_s16(w1));
      acc[r][3] = vmlal_high_s16(acc[r][3], x1, w1);
    }
  }
  for (int r = 0; r < PX; ++r) {
    for (int g = 0; g < 4; ++g) vst1q_s32(buf + r * 64 + 4 * g, acc[r][g]);
  }
}

}  // namespace

void dwconv_f32_neon(Rows<float> a, const float* w, int c, int m, const Epilogue& e) {
  const int taps = a.taps;
  alignas(16) float buf[4 * 64];
  for (int i0 = 0; i0 < m; i0 += 4) {
    const int px = std::min(4, m - i0);
    for (int c0 = 0; c0 < c; c0 += 64) {
      const int cw = std::min(64, c - c0);
      int cc = 0;
      for (; cc + 16 <= cw; cc += 16) {
        const float* wc = w + c0 + cc;
        switch (px) {
          case 4: dw_tile_f32<4>(a, i0, taps, wc, c, c0 + cc, buf + cc); break;
          case 3: dw_tile_f32<3>(a, i0, taps, wc, c, c0 + cc, buf + cc); break;
          case 2: dw_tile_f32<2>(a, i0, taps, wc, c, c0 + cc, buf + cc); break;
          default: dw_tile_f32<1>(a, i0, taps, wc, c, c0 + cc, buf + cc); break;
        }
      }
      for (int r = 0; r < px; ++r) {
        int ct = cc;
        for (; ct + 4 <= cw; ct += 4) {
          float32x4_t acc = vdupq_n_f32(0.0f);
          for (int t = 0; t < taps; ++t) {
            acc = vfmaq_f32(acc, vld1q_f32(piece(a, i0 + r, t) + c0 + ct), vld1q_f32(w + t * c + c0 + ct));
          }
          vst1q_f32(buf + r * 64 + ct, acc);
        }
        for (; ct < cw; ++ct) {
          float s = 0.0f;
          for (int t = 0; t < taps; ++t) s = std::fma(piece(a, i0 + r, t)[c0 + ct], w[t * c + c0 + ct], s);
          buf[r * 64 + ct] = s;
        }
      }
      epilogue_neon(buf, 64, px, i0, c0, cw, e);
    }
  }
}

void dwconv_u8s8_neon(Rows<uint8_t> a, int32_t a_zero, const int8_t* w, int c, int m, const Epilogue& e) {
  const int taps = a.taps;
  const int16x8_t zero = vdupq_n_s16(static_cast<int16_t>(a_zero));
  alignas(16) int32_t buf[4 * 64];
  for (int i0 = 0; i0 < m; i0 += 4) {
    const int px = std::min(4, m - i0);
    for (int c0 = 0; c0 < c; c0 += 64) {
      const int cw = std::min(64, c - c0);
      int cc = 0;
      for (; cc + 16 <= cw; cc += 16) {
        const int8_t* wc = w + c0 + cc;
        switch (px) {
          case 4: dw_tile_u8<4>(a, i0, taps, zero, wc, c, c0 + cc, buf + cc); break;
          case 3: dw_tile_u8<3>(a, i0, taps, zero, wc, c, c0 + cc, buf + cc); break;
          case 2: dw_tile_u8<2>(a, i0, taps, zero, wc, c, c0 + cc, buf + cc); break;
          default: dw_tile_u8<1>(a, i0, taps, zero, wc, c, c0 + cc, buf + cc); break;
        }
      }
      for (int r = 0; r < px; ++r) {
        for (int ct = cc; ct < cw; ++ct) {
          int32_t s = 0;
          for (int t = 0; t < taps; ++t) {
            s += (static_cast<int32_t>(piece(a, i0 + r, t)[c0 + ct]) - a_zero) * w[t * c + c0 + ct];
          }
          buf[r * 64 + ct] = s;
        }
      }
      epilogue_neon(buf, 64, px, i0, c0, cw, e);
    }
  }
}

// ---------------------------------------------------------------- CTC greedy

namespace {

// exp_neg (kernels.h) on 4 lanes: the same operations in the same order.
inline float32x4_t exp_neg4(float32x4_t x) {
  const float32x4_t lo = vdupq_n_f32(kExpLo);
  x = vbslq_f32(vcgtq_f32(x, lo), x, lo);  // (x > lo) ? x : lo, as the reference
  const float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(kLog2e)));
  float32x4_t r = vsubq_f32(x, vmulq_f32(n, vdupq_n_f32(kLn2Hi)));
  r = vsubq_f32(r, vmulq_f32(n, vdupq_n_f32(kLn2Lo)));
  float32x4_t p = vdupq_n_f32(kExpP0);
  p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(kExpP1));
  p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(kExpP2));
  p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(kExpP3));
  p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(kExpP4));
  p = vaddq_f32(vmulq_f32(p, r), vdupq_n_f32(kExpP5));
  const float32x4_t y = vaddq_f32(vaddq_f32(vmulq_f32(p, vmulq_f32(r, r)), r), vdupq_n_f32(1.0f));
  const int32x4_t bits = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
  return vmulq_f32(y, vreinterpretq_f32_s32(bits));
}

}  // namespace

void ctc_argmax_neon(const float* logits, int classes, int frames, int t_n, int32_t* ids, float* probs) {
  int t0 = 0;
  for (; t0 + 4 <= t_n; t0 += 4) {  // 4 frames at a time: each class's 4 logits are contiguous
    float32x4_t best = vdupq_n_f32(-std::numeric_limits<float>::infinity());
    int32x4_t arg = vdupq_n_s32(0);
    for (int c = 0; c < classes; ++c) {
      const float32x4_t v = vld1q_f32(logits + static_cast<size_t>(c) * frames + t0);
      const uint32x4_t more = vcgtq_f32(v, best);  // strictly: ties keep the lower id
      best = vbslq_f32(more, v, best);
      arg = vbslq_s32(more, vdupq_n_s32(c), arg);
    }
    vst1q_s32(ids + t0, arg);
    if (vmaxvq_u32(vreinterpretq_u32_s32(arg)) == 0) continue;  // four blank frames
    float32x4_t sum = vdupq_n_f32(0.0f);
    for (int c = 0; c < classes; ++c) {
      sum = vaddq_f32(sum, exp_neg4(vsubq_f32(vld1q_f32(logits + static_cast<size_t>(c) * frames + t0), best)));
    }
    float prob[4];
    vst1q_f32(prob, vdivq_f32(vdupq_n_f32(1.0f), sum));
    int32_t a[4];
    vst1q_s32(a, arg);
    for (int k = 0; k < 4; ++k) {
      if (a[k] != 0) probs[t0 + k] = prob[k];
    }
  }
  for (int t = t0; t < t_n; ++t) {  // the last frames as the reference does them
    float best = -std::numeric_limits<float>::infinity();
    int32_t a = 0;
    for (int c = 0; c < classes; ++c) {
      const float v = logits[static_cast<size_t>(c) * frames + t];
      if (v > best) {
        best = v;
        a = c;
      }
    }
    ids[t] = a;
    if (a == 0) continue;
    float sum = 0.0f;
    for (int c = 0; c < classes; ++c) sum = sum + exp_neg(logits[static_cast<size_t>(c) * frames + t] - best);
    probs[t] = 1.0f / sum;
  }
}

}  // namespace carbon::cpu::detail
