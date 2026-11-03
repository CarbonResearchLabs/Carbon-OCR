// The FP32 GEMM on Arm64 NEON (carbon/src/cpu/kernels.h), every Armv8-A CPU: fmla
// with A broadcast from memory (ld1r), 6 rows x 16 columns of sums in 24
// registers, A read in place through its rows' pointers.  Also the depthwise
// convolutions, CTC greedy decoding, the crop rule's resampling and the page
// turns.  FP16 without FP16 arithmetic: widened by fcvtl (Armv8-A's) and summed
// as FP32.  No flags beyond the baseline.

#include <arm_neon.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "epilogue_neon.h"
#include "kernels.h"
#include "resample_math.h"

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

// 4 activations as FP32: FP32 loaded, FP16 widened.
inline float32x4_t load4(const float* x) { return vld1q_f32(x); }
inline float32x4_t load4(const uint16_t* x) { return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(x))); }
inline float load1(const float* x) { return *x; }
inline float load1(const uint16_t* x) { return half_to_float(*x); }

template <int PX, typename A>
void dw_tile_f32(const Rows<A>& a, int64_t i0, int taps, const float* w, int c, int off, float* buf) {
  float32x4_t acc[PX][4];
  for (int r = 0; r < PX; ++r) {
    for (int g = 0; g < 4; ++g) acc[r][g] = vdupq_n_f32(0.0f);
  }
  for (int t = 0; t < taps; ++t) {
    float32x4_t wv[4];
    for (int g = 0; g < 4; ++g) wv[g] = vld1q_f32(w + t * c + 4 * g);
    for (int r = 0; r < PX; ++r) {
      const A* x = piece(a, i0 + r, t) + off;
      for (int g = 0; g < 4; ++g) acc[r][g] = vfmaq_f32(acc[r][g], load4(x + 4 * g), wv[g]);
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

// FP32 weights; FP32 or FP16 activations.
template <typename A>
void dwconv_neon(const Rows<A>& a, const float* w, int c, int m, const Epilogue& e) {
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
            acc = vfmaq_f32(acc, load4(piece(a, i0 + r, t) + c0 + ct), vld1q_f32(w + t * c + c0 + ct));
          }
          vst1q_f32(buf + r * 64 + ct, acc);
        }
        for (; ct < cw; ++ct) {
          float s = 0.0f;
          for (int t = 0; t < taps; ++t) s = std::fma(load1(piece(a, i0 + r, t) + c0 + ct), w[t * c + c0 + ct], s);
          buf[r * 64 + ct] = s;
        }
      }
      epilogue_neon(buf, 64, px, i0, c0, cw, e);
    }
  }
}

}  // namespace

void dwconv_f32_neon(Rows<float> a, const float* w, int c, int m, const Epilogue& e) { dwconv_neon(a, w, c, m, e); }

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

// ---------------------------------------------------------------- FP16 (fcvtl)
//
// FP16 stored, FP32 arithmetic, as avx2.cpp does it: B widened whole and kept
// from call to call up to kKeepB (past it a panel at a time); FP16 A a block of 6
// rows at a time widened into the kA scratch, a piece of k values a row.  The
// reference's bits.

namespace {

// n FP16 values widened: 8 at a time, the rest one at a time (the same bits).
inline void widen_f16(const uint16_t* src, float* dst, int64_t n) {
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(src + i));
    vst1q_f32(dst + i, vcvt_f32_f16(vget_low_f16(h)));
    vst1q_f32(dst + i + 4, vcvt_high_f32_f16(h));
  }
  for (; i < n; ++i) dst[i] = half_to_float(src[i]);
}

// tile on mr rows, 1 to 6.
inline void tile_rows(int mr, const float* const* rows, int taps, int seg, const float* bp, float* buf) {
  switch (mr) {
    case 6: tile<6>(rows, taps, seg, bp, buf); break;
    case 5: tile<5>(rows, taps, seg, bp, buf); break;
    case 4: tile<4>(rows, taps, seg, bp, buf); break;
    case 3: tile<3>(rows, taps, seg, bp, buf); break;
    case 2: tile<2>(rows, taps, seg, bp, buf); break;
    default: tile<1>(rows, taps, seg, bp, buf); break;
  }
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
  const int taps = a.taps, k = b.k, seg = k / taps, panels = b.np / 16;
  const size_t panel = static_cast<size_t>(k) * 16;
  const bool whole = panel * panels * sizeof(float) <= kKeepB;
  const float* const bw = whole ? widened(b) : nullptr;
  float* const bp = whole ? nullptr : scratch_of<float>(Scratch::kB, panel);
  const float* const* all = block(a, 0, m);
  alignas(16) float buf[6 * 16];
  for (int pj = 0; pj < panels; ++pj) {
    if (!whole) widen_f16(b.data + pj * panel, bp, static_cast<int64_t>(panel));
    const float* const p = whole ? bw + pj * panel : bp;
    for (int m0 = 0; m0 < m; m0 += 6) {
      const int mr = min_of(6, m - m0);
      tile_rows(mr, all + static_cast<size_t>(m0) * taps, taps, seg, p, buf);
      epilogue_neon(buf, 16, mr, m0, pj * 16, min_of(16, b.n - pj * 16), e);
    }
  }
}

void gemm_f16_half(const Rows<uint16_t>& a, const F16View& b, int m, const Epilogue& e) {
  const int taps = a.taps, k = b.k, seg = k / taps, panels = b.np / 16;
  const float* const bw = widened(b);
  float* const a32 = scratch_of<float>(Scratch::kA, static_cast<size_t>(6) * k);
  const float* rows[6];
  alignas(16) float buf[6 * 16];
  for (int m0 = 0; m0 < m; m0 += 6) {
    const int mr = min_of(6, m - m0);
    for (int r = 0; r < mr; ++r) {
      float* row = a32 + static_cast<size_t>(r) * k;
      for (int t = 0; t < taps; ++t) widen_f16(piece(a, m0 + r, t), row + static_cast<size_t>(t) * seg, seg);
      rows[r] = row;
    }
    for (int pj = 0; pj < panels; ++pj) {
      tile_rows(mr, rows, 1, k, bw + static_cast<size_t>(pj) * k * 16, buf);
      epilogue_neon(buf, 16, mr, m0, pj * 16, min_of(16, b.n - pj * 16), e);
    }
  }
}

template <typename A>
void dwconv_f16_impl(const Rows<A>& a, const uint16_t* w, int c, int m, const Epilogue& e) {
  const int64_t n = static_cast<int64_t>(a.taps) * c;
  float* const wf = scratch_of<float>(Scratch::kB, static_cast<size_t>(n));
  widen_f16(w, wf, n);
  dwconv_neon(a, wf, c, m, e);
}

}  // namespace

void gemm_f16_neon(Rows<uint16_t> a, const F16View& b, int m, const Epilogue& e) { gemm_f16_half(a, b, m, e); }
void gemm_f16_neon(Rows<float> a, const F16View& b, int m, const Epilogue& e) { gemm_f16_wide(a, b, m, e); }

void dwconv_f16_neon(Rows<uint16_t> a, const uint16_t* w, int c, int m, const Epilogue& e) {
  dwconv_f16_impl(a, w, c, m, e);
}
void dwconv_f16_neon(Rows<float> a, const uint16_t* w, int c, int m, const Epilogue& e) {
  dwconv_f16_impl(a, w, c, m, e);
}

void to_f32_neon(const uint16_t* in, float* out, size_t n) { widen_f16(in, out, static_cast<int64_t>(n)); }

void to_f16_neon(const float* in, uint16_t* out, size_t n) {
  size_t i = 0;
  for (; i + 4 <= n; i += 4) vst1_u16(out + i, vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(in + i))));
  for (; i < n; ++i) out[i] = float_to_half(in[i]);
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

// 32 values a word, 4 a compare: each lane's mask and its bit (1, 2, 4, 8) summed across the vector; the last
// word's values one at a time.
void above_bits_neon(const float* v, int n, float t, uint32_t* bits) {
  const float32x4_t tv = vdupq_n_f32(t);
  const uint32_t lane_bits[4] = {1, 2, 4, 8};
  const uint32x4_t weight = vld1q_u32(lane_bits);
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    uint32_t word = 0;
    for (int k = 0; k < 8; ++k) {
      const uint32x4_t more = vcgtq_f32(vld1q_f32(v + i + 4 * k), tv);  // a NaN is never above
      word |= vaddvq_u32(vandq_u32(more, weight)) << (4 * k);
    }
    bits[i / 32] = word;
  }
  if (i < n) {
    uint32_t word = 0;
    for (int k = 0; i + k < n; ++k) word |= (v[i + k] > t ? 1u : 0u) << k;
    bits[i / 32] = word;
  }
}

// 16 bytes a step: the compares' ones summed in 8-bit lanes, widened into 64 bits every 255 steps (and at a row's
// end). The row's last bytes one at a time.
ByteCounts count_beyond_neon(const uint8_t* c, int w, int h, int64_t stride, int lo, int hi) {
  const bool any_below = lo >= 0, any_above = hi <= 255;
  const uint8x16_t lo8 = vdupq_n_u8(static_cast<uint8_t>(any_below ? lo : 0));
  const uint8x16_t hi8 = vdupq_n_u8(static_cast<uint8_t>(any_above ? hi : 255));
  const uint8x16_t one = vdupq_n_u8(1);
  ByteCounts n{0, 0};
  int64_t below = 0, above = 0;
  for (int y = 0; y < h; ++y) {
    const uint8_t* row = c + y * stride;
    int x = 0;
    while (x + 16 <= w) {
      uint8x16_t b = vdupq_n_u8(0), a = vdupq_n_u8(0);
      for (int k = 0; k < 255 && x + 16 <= w; ++k, x += 16) {
        const uint8x16_t v = vld1q_u8(row + x);
        b = vaddq_u8(b, vandq_u8(vcleq_u8(v, lo8), one));
        a = vaddq_u8(a, vandq_u8(vcgeq_u8(v, hi8), one));
      }
      below += vaddlvq_u8(b);
      above += vaddlvq_u8(a);
    }
    for (; x < w; ++x) {
      n.below += row[x] <= lo;
      n.above += row[x] >= hi;
    }
  }
  if (any_below) n.below += below;
  if (any_above) n.above += above;
  return n;
}

void ctc_argmax_neon(const float* logits, int classes, int frames, int t_n, int32_t* ids, float* probs) {
  // As ctc_argmax_avx2: in chunks of up to 256 frames, the logits read in their order; each frame's best class 4 frames
  // a vector (the last frames one at a time), the chunk's running bests in L1; then the softmax sums of only the frames
  // where a character won, 4 to a vector, each lane loaded from its own frame; past the group's highest best class, a
  // class whose 4 terms are all of logits more than 17 below their bests passed over (it cannot change a sum of 1 or more).
  constexpr int kChunk = 256;
  alignas(16) float top[kChunk + 4];
  alignas(16) int32_t arg[kChunk + 4];
  alignas(16) int32_t hit[kChunk + 4];
  const float inf = std::numeric_limits<float>::infinity();
  const float32x4_t far = vdupq_n_f32(-17.0f);
  for (int c0 = 0; c0 < t_n; c0 += kChunk) {
    const int len = t_n - c0 < kChunk ? t_n - c0 : kChunk;
    const int nv = len / 4;
    for (int t = 0; t < len; ++t) {
      top[t] = -inf;
      arg[t] = 0;
    }
    for (int c = 0; c < classes; ++c) {
      const float* row = logits + static_cast<size_t>(c) * frames + c0;
      const int32x4_t cv = vdupq_n_s32(c);
      for (int v = 0; v < nv; ++v) {
        const float32x4_t x = vld1q_f32(row + 4 * v), b = vld1q_f32(top + 4 * v);
        const uint32x4_t more = vcgtq_f32(x, b);  // strictly: ties keep the lower id
        vst1q_f32(top + 4 * v, vbslq_f32(more, x, b));
        vst1q_s32(arg + 4 * v, vbslq_s32(more, cv, vld1q_s32(arg + 4 * v)));
      }
      for (int t = 4 * nv; t < len; ++t) {
        if (row[t] > top[t]) {
          top[t] = row[t];
          arg[t] = c;
        }
      }
    }
    std::memcpy(ids + c0, arg, static_cast<size_t>(len) * sizeof(int32_t));
    int n = 0;
    for (int t = 0; t < len; ++t) {
      hit[n] = t;
      n += arg[t] != 0;
    }
    if (n == 0) continue;  // a blank chunk: no probability is read
    for (int k = n; k < n + 4; ++k) hit[k] = hit[n - 1];  // a short last group repeats its last frame
    const float* chunk = logits + c0;
    for (int k = 0; k < n; k += 4) {
      const int h0 = hit[k], h1 = hit[k + 1], h2 = hit[k + 2], h3 = hit[k + 3];
      float32x4_t best = vld1q_dup_f32(top + h0);
      best = vld1q_lane_f32(top + h1, best, 1);
      best = vld1q_lane_f32(top + h2, best, 2);
      best = vld1q_lane_f32(top + h3, best, 3);
      const int last = std::max(std::max(arg[h0], arg[h1]), std::max(arg[h2], arg[h3]));
      float32x4_t sum = vdupq_n_f32(0.0f);
      for (int c = 0; c < classes; ++c) {
        const float* row = chunk + static_cast<size_t>(c) * frames;
        float32x4_t v = vld1q_dup_f32(row + h0);
        v = vld1q_lane_f32(row + h1, v, 1);
        v = vld1q_lane_f32(row + h2, v, 2);
        v = vld1q_lane_f32(row + h3, v, 3);
        const float32x4_t d = vsubq_f32(v, best);
        if (c > last && vminvq_u32(vcltq_f32(d, far)) != 0) continue;  // NaN is added, as the reference
        sum = vaddq_f32(sum, exp_neg4(d));
      }
      float prob[4];
      vst1q_f32(prob, vdivq_f32(vdupq_n_f32(1.0f), sum));
      const int m = n - k < 4 ? n - k : 4;
      for (int q = 0; q < m; ++q) probs[c0 + hit[k + q]] = prob[q];
    }
  }
}

// ---------------------------------------------------------------- resampling
//
// The pixels widened to 32 bits and multiplied by the 22-bit weights there (mla on
// 4 lanes): every product and sum is exact (255 * 2^22 < 2^31).

void resample_hk_neon(const int32_t* k, const int32_t* count, int taps, int width, int chunks, void* out) {
  auto* o = static_cast<int32_t*>(out);
  for (int x = 0; x < width; ++x) {
    int32_t* ox = o + static_cast<size_t>(x) * 8 * chunks;
    for (int t = 0; t < count[x]; ++t) ox[t] = k[static_cast<size_t>(x) * taps + t];
  }
}

namespace {

// 8 taps of a sample from p, times its weights kx, added to acc.
inline int32x4_t h_chunk(int32x4_t acc, const uint8_t* p, const int32_t* kx) {
  const uint16x8_t v = vmovl_u8(vld1_u8(p));
  acc = vmlaq_s32(acc, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v))), vld1q_s32(kx));
  return vmlaq_s32(acc, vreinterpretq_s32_u32(vmovl_high_u16(v)), vld1q_s32(kx + 4));
}

// clip8 of 4 sums (kHalf added): the shifted sums fit 16 bits, and the unsigned
// narrowing clamps them to 0 .. 255.
inline int16x4_t shifted(int32x4_t s) { return vqmovn_s32(vshrq_n_s32(s, carbon::detail::kPrecisionBits)); }

}  // namespace

void resample_h_neon(const uint8_t* src, int col0, const int32_t* first, const void* k, int chunks, int g0, int g1,
                     uint8_t* out) {
  const auto* kk = static_cast<const int32_t*>(k);
  const int32x4_t half = vdupq_n_s32(carbon::detail::kHalf);
  for (int g = g0; g < g1; ++g) {
    int32x4_t s[8];  // per sample, 4 partial sums
    for (int p = 0; p < 8; ++p) {
      const int x = 8 * g + p;
      const uint8_t* a = src + (first[x] - col0);
      const int32_t* kx = kk + static_cast<size_t>(x) * chunks * 8;
      int32x4_t acc = vdupq_n_s32(0);
      for (int c = 0; c < chunks; ++c) acc = h_chunk(acc, a + 8 * c, kx + 8 * c);
      s[p] = acc;
    }
    // pairwise adds to one sum a sample, in order
    const int32x4_t lo = vpaddq_s32(vpaddq_s32(s[0], s[1]), vpaddq_s32(s[2], s[3]));
    const int32x4_t hi = vpaddq_s32(vpaddq_s32(s[4], s[5]), vpaddq_s32(s[6], s[7]));
    const int16x8_t w = vcombine_s16(shifted(vaddq_s32(lo, half)), shifted(vaddq_s32(hi, half)));
    vst1_u8(out + 8 * g, vqmovun_s16(w));
  }
}

// 16 pixels of the row at a time, in four registers of 4 int32 sums.
void resample_v_neon(const uint8_t* const* rows, const int32_t* k, int n, int width, uint8_t* out) {
  const int32x4_t half = vdupq_n_s32(carbon::detail::kHalf);
  for (int x = 0; x < width; x += 16) {
    int32x4_t a0 = half, a1 = half, a2 = half, a3 = half;
    for (int i = 0; i < n; ++i) {
      const uint8x16_t v = vld1q_u8(rows[i] + x);
      const uint16x8_t l = vmovl_u8(vget_low_u8(v)), h = vmovl_high_u8(v);
      const int32_t w = k[i];
      a0 = vmlaq_n_s32(a0, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(l))), w);
      a1 = vmlaq_n_s32(a1, vreinterpretq_s32_u32(vmovl_high_u16(l)), w);
      a2 = vmlaq_n_s32(a2, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(h))), w);
      a3 = vmlaq_n_s32(a3, vreinterpretq_s32_u32(vmovl_high_u16(h)), w);
    }
    const uint8x16_t b = vcombine_u8(vqmovun_s16(vcombine_s16(shifted(a0), shifted(a1))),
                                     vqmovun_s16(vcombine_s16(shifted(a2), shifted(a3))));
    if (x + 16 <= width) {
      vst1q_u8(out + x, b);
    } else {
      uint8_t last[16];
      vst1q_u8(last, b);
      for (int j = 0; j < width - x; ++j) out[x + j] = last[j];
    }
  }
}

// ---------------------------------------------------------------- page turns

namespace {

// Two bytes as the two lanes of a vector of doubles.
inline float64x2_t pair_f64(uint8_t a, uint8_t b) { return vcvtq_f64_u64(vsetq_lane_u64(b, vdupq_n_u64(a), 1)); }

}  // namespace

// Four pixels at a time, two to a vector: where they were on the page in doubles
// (frintm for the floor), then, when all their taps are on the page, the taps
// loaded one by one, the blend in doubles as the reference orders it, and fcvtau
// (round half away from zero, as lround).  Pixels with a tap off the page go to
// turn_pixels.
void rotate_bilinear_neon(const TurnView& t, uint8_t* dst) {
  const int w = t.w, h = t.h;
  const bool wide = w >= 4 && h >= 2;  // so that w - 2 and h - 2 are not negative
  const float64x2_t cx = vdupq_n_f64(t.cx), cy = vdupq_n_f64(t.cy), one = vdupq_n_f64(1.0);
  const float64x2_t lo = vdupq_n_f64(0.0), hi = vdupq_n_f64(255.0);
  const uint64x2_t x_last = vdupq_n_u64(static_cast<uint64_t>(w - 2));
  const uint64x2_t y_last = vdupq_n_u64(static_cast<uint64_t>(h - 2));
  const int64_t stride = t.stride;
  for (int y = t.row0; y < t.row1; ++y) {
    const double dy = y - t.cy, ds = dy * t.s, dc = dy * t.c;
    const float64x2_t vds = vdupq_n_f64(ds), vdc = vdupq_n_f64(dc);
    uint8_t* d = dst + static_cast<int64_t>(y) * w;
    int x = 0;
    for (; wide && x + 4 <= w; x += 4) {
      float64x2_t fx[2], fy[2];
      int64x2_t ix[2], iy[2];
      uint64x2_t inside = vdupq_n_u64(~uint64_t{0});
      for (int k = 0; k < 2; ++k) {
        const float64x2_t sx = vaddq_f64(vsubq_f64(vld1q_f64(t.xc + x + 2 * k), vds), cx);
        const float64x2_t sy = vaddq_f64(vaddq_f64(vld1q_f64(t.xs + x + 2 * k), vdc), cy);
        const float64x2_t x0 = vrndmq_f64(sx), y0 = vrndmq_f64(sy);
        fx[k] = vsubq_f64(sx, x0);
        fy[k] = vsubq_f64(sy, y0);
        ix[k] = vcvtq_s64_f64(x0);
        iy[k] = vcvtq_s64_f64(y0);
        // x0 <= w - 2 and y0 <= h - 2 as unsigned: a negative one is past them
        inside = vandq_u64(inside, vandq_u64(vcleq_u64(vreinterpretq_u64_s64(ix[k]), x_last),
                                             vcleq_u64(vreinterpretq_u64_s64(iy[k]), y_last)));
      }
      if (vminvq_u32(vreinterpretq_u32_u64(inside)) == 0) {
        turn_pixels(t, ds, dc, x, x + 4, d);
        continue;
      }
      for (int k = 0; k < 2; ++k) {
        // the two pixels' first taps
        const uint8_t* a = t.src + vgetq_lane_s64(iy[k], 0) * stride + vgetq_lane_s64(ix[k], 0);
        const uint8_t* b = t.src + vgetq_lane_s64(iy[k], 1) * stride + vgetq_lane_s64(ix[k], 1);
        const float64x2_t gx = vsubq_f64(one, fx[k]), gy = vsubq_f64(one, fy[k]);
        const float64x2_t upper =
            vaddq_f64(vmulq_f64(pair_f64(a[0], b[0]), gx), vmulq_f64(pair_f64(a[1], b[1]), fx[k]));
        const float64x2_t lower = vaddq_f64(vmulq_f64(pair_f64(a[stride], b[stride]), gx),
                                            vmulq_f64(pair_f64(a[stride + 1], b[stride + 1]), fx[k]));
        const float64x2_t blend = vaddq_f64(vmulq_f64(upper, gy), vmulq_f64(lower, fy[k]));
        const uint64x2_t v = vcvtaq_u64_f64(vmaxq_f64(vminq_f64(blend, hi), lo));
        d[x + 2 * k] = static_cast<uint8_t>(vgetq_lane_u64(v, 0));
        d[x + 2 * k + 1] = static_cast<uint8_t>(vgetq_lane_u64(v, 1));
      }
    }
    turn_pixels(t, ds, dc, x, w, d);
  }
}

namespace {

// zip1 and zip2 on lanes of 16, 32 and 64 bits, for the transpose below.
inline uint8x16_t zip1_16(uint8x16_t a, uint8x16_t b) {
  return vreinterpretq_u8_u16(vzip1q_u16(vreinterpretq_u16_u8(a), vreinterpretq_u16_u8(b)));
}
inline uint8x16_t zip2_16(uint8x16_t a, uint8x16_t b) {
  return vreinterpretq_u8_u16(vzip2q_u16(vreinterpretq_u16_u8(a), vreinterpretq_u16_u8(b)));
}
inline uint8x16_t zip1_32(uint8x16_t a, uint8x16_t b) {
  return vreinterpretq_u8_u32(vzip1q_u32(vreinterpretq_u32_u8(a), vreinterpretq_u32_u8(b)));
}
inline uint8x16_t zip2_32(uint8x16_t a, uint8x16_t b) {
  return vreinterpretq_u8_u32(vzip2q_u32(vreinterpretq_u32_u8(a), vreinterpretq_u32_u8(b)));
}
inline uint8x16_t zip1_64(uint8x16_t a, uint8x16_t b) {
  return vreinterpretq_u8_u64(vzip1q_u64(vreinterpretq_u64_u8(a), vreinterpretq_u64_u8(b)));
}
inline uint8x16_t zip2_64(uint8x16_t a, uint8x16_t b) {
  return vreinterpretq_u8_u64(vzip2q_u64(vreinterpretq_u64_u8(a), vreinterpretq_u64_u8(b)));
}

// 16 rows of 16 bytes transposed: column i of the rows (r[0] first) to o + i * os.
// Interleaving bytes, then pairs, fours and eights of them (zip1, zip2).
inline void transpose16(const uint8_t* const* r, uint8_t* o, int64_t os) {
  uint8x16_t a[16], b[16];
  for (int i = 0; i < 16; ++i) a[i] = vld1q_u8(r[i]);
  for (int i = 0; i < 8; ++i) {  // b[2i]: columns 0-7 of rows 2i and 2i + 1, b[2i + 1]: columns 8-15
    b[2 * i] = vzip1q_u8(a[2 * i], a[2 * i + 1]);
    b[2 * i + 1] = vzip2q_u8(a[2 * i], a[2 * i + 1]);
  }
  for (int i = 0; i < 4; ++i) {  // a[4i + k]: columns 4k to 4k + 3 of rows 4i to 4i + 3
    a[4 * i] = zip1_16(b[4 * i], b[4 * i + 2]);
    a[4 * i + 1] = zip2_16(b[4 * i], b[4 * i + 2]);
    a[4 * i + 2] = zip1_16(b[4 * i + 1], b[4 * i + 3]);
    a[4 * i + 3] = zip2_16(b[4 * i + 1], b[4 * i + 3]);
  }
  for (int i = 0; i < 2; ++i) {  // b[8i + m]: columns 2m and 2m + 1 of rows 8i to 8i + 7
    for (int k = 0; k < 4; ++k) {
      b[8 * i + 2 * k] = zip1_32(a[8 * i + k], a[8 * i + 4 + k]);
      b[8 * i + 2 * k + 1] = zip2_32(a[8 * i + k], a[8 * i + 4 + k]);
    }
  }
  for (int m = 0; m < 8; ++m) {  // columns 2m and 2m + 1, all 16 rows
    vst1q_u8(o + 2 * m * os, zip1_64(b[m], b[8 + m]));
    vst1q_u8(o + (2 * m + 1) * os, zip2_64(b[m], b[8 + m]));
  }
}

}  // namespace

// As rotate_quarters_avx2: q = 0 the rows copied, q = 2 each row reversed 16
// bytes at a time (rev64, then ext swaps the halves), q odd 16 x 16 tiles
// transposed in registers in blocks of 64 columns by 256 rows, the bytes left
// over one at a time.
void rotate_quarters_neon(const uint8_t* src, int w, int h, int64_t stride, int q, uint8_t* dst) {
  if (q == 0) {
    for (int y = 0; y < h; ++y) std::memcpy(dst + static_cast<int64_t>(y) * w, src + y * stride, w);
    return;
  }
  if (q == 2) {
    for (int y = 0; y < h; ++y) {
      const uint8_t* r = src + y * stride;
      uint8_t* o = dst + static_cast<int64_t>(h - 1 - y) * w;
      int x = 0;
      for (; x + 16 <= w; x += 16) {
        const uint8x16_t v = vrev64q_u8(vld1q_u8(r + x));
        vst1q_u8(o + w - 16 - x, vextq_u8(v, v, 8));
      }
      for (; x < w; ++x) o[w - 1 - x] = r[x];
    }
    return;
  }
  const int w16 = w / 16 * 16, h16 = h / 16 * 16;
  const int64_t os = q == 1 ? h : -h;
  const uint8_t* rows[16];
  for (int by = 0; by < h16; by += 256) {
    const int by1 = min_of(h16, by + 256);
    for (int bx = 0; bx < w16; bx += 64) {
      const int bx1 = min_of(w16, bx + 64);
      for (int x = bx; x < bx1; x += 16) {
        for (int y = by; y < by1; y += 16) {
          uint8_t* o = quarter_tile(src, stride, w, h, q, x, y, dst, rows);
          transpose16(rows, o, os);
        }
      }
    }
  }
  quarter_bytes(src, stride, w, h, q, w16, 0, h16, dst);  // the columns past the tiles
  quarter_bytes(src, stride, w, h, q, 0, h16, h, dst);    // the rows below them
}

}  // namespace carbon::cpu::detail
