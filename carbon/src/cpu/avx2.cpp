// The GEMMs on AVX2 + FMA (carbon/src/cpu/kernels.h), 6 rows x 16 columns a tile.
//
// int8 without VNNI (AMD Zen 2 and 3, Intel before Ice Lake): vpmaddwd on 16-bit
// values, A widened to uint16 and B packed as int16 pairs of k, so every product
// is exact (vpmaddubsw would saturate at 255 * 127 * 2).
// FP32: vfmadd231ps, the sums in 12 registers.
// Built with -mavx2 -mfma (/arch:AVX2).

#include <immintrin.h>

#include <cmath>
#include <cstring>

#include "epilogue_avx2.h"
#include "kernels.h"

namespace carbon::cpu::detail {
namespace {

template <int MR>
void tile_u8(const uint16_t* a16, int kp, const int16_t* bp, const int32_t* fix, int32_t* buf) {
  __m256i acc[MR][2];
  for (int r = 0; r < MR; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_si256();
  for (int kk = 0; kk < kp / 2; ++kk) {
    const __m256i b0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bp + kk * 32));
    const __m256i b1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bp + kk * 32 + 16));
    for (int r = 0; r < MR; ++r) {
      int32_t pair;
      std::memcpy(&pair, a16 + r * kp + kk * 2, 4);
      const __m256i av = _mm256_set1_epi32(pair);
      acc[r][0] = _mm256_add_epi32(acc[r][0], _mm256_madd_epi16(av, b0));
      acc[r][1] = _mm256_add_epi32(acc[r][1], _mm256_madd_epi16(av, b1));
    }
  }
  const __m256i f0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fix));
  const __m256i f1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(fix + 8));
  for (int r = 0; r < MR; ++r) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(buf + r * 16), _mm256_add_epi32(acc[r][0], f0));
    _mm256_store_si256(reinterpret_cast<__m256i*>(buf + r * 16 + 8), _mm256_add_epi32(acc[r][1], f1));
  }
}

// A piece widened to uint16, 16 bytes at a time.
void widen(const uint8_t* src, uint16_t* dst, int n) {
  int k = 0;
  for (; k + 16 <= n; k += 16) {
    const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + k));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + k), _mm256_cvtepu8_epi16(v));
  }
  for (; k < n; ++k) dst[k] = src[k];
}

template <int MR>
void tile_f32(const float* const* rows, int taps, int seg, const float* bp, float* buf) {
  __m256 acc[MR][2];
  for (int r = 0; r < MR; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_ps();
  for (int t = 0; t < taps; ++t) {
    const float* p[MR];
    for (int r = 0; r < MR; ++r) p[r] = rows[r * taps + t];
    for (int kk = 0; kk < seg; ++kk, bp += 16) {
      const __m256 b0 = _mm256_loadu_ps(bp), b1 = _mm256_loadu_ps(bp + 8);
      for (int r = 0; r < MR; ++r) {
        const __m256 av = _mm256_broadcast_ss(p[r] + kk);
        acc[r][0] = _mm256_fmadd_ps(av, b0, acc[r][0]);
        acc[r][1] = _mm256_fmadd_ps(av, b1, acc[r][1]);
      }
    }
  }
  for (int r = 0; r < MR; ++r) {
    _mm256_store_ps(buf + r * 16, acc[r][0]);
    _mm256_store_ps(buf + r * 16 + 8, acc[r][1]);
  }
}

}  // namespace

void gemm_u8s8_avx2(Rows<uint8_t> a, int32_t a_zero, const BView& b, int m, const Epilogue& e) {
  const int taps = b.taps, segp = b.segp, kp = taps * segp, panels = b.np / 16;
  int32_t* fix = scratch_of<int32_t>(Scratch::kFix, b.np);  // A's zero point, after the products
  for (int j = 0; j < b.np; ++j) fix[j] = j < b.n ? -a_zero * b.col_sums[j] : 0;
  uint16_t* a16 = scratch_of<uint16_t>(Scratch::kA, static_cast<size_t>(6) * kp);
  const auto* bdata = reinterpret_cast<const int16_t*>(b.data);
  alignas(32) int32_t buf[6 * 16];
  for (int m0 = 0; m0 < m; m0 += 6) {
    const int mr = min_of(6, m - m0);
    for (int r = 0; r < mr; ++r) {
      for (int t = 0; t < taps; ++t) {
        widen(piece(a, m0 + r, t), a16 + r * kp + t * segp, segp);
      }
    }
    for (int pj = 0; pj < panels; ++pj) {
      const int16_t* bp = bdata + static_cast<size_t>(pj) * kp * 16;
      const int32_t* f = fix + pj * 16;
      switch (mr) {
        case 6: tile_u8<6>(a16, kp, bp, f, buf); break;
        case 5: tile_u8<5>(a16, kp, bp, f, buf); break;
        case 4: tile_u8<4>(a16, kp, bp, f, buf); break;
        case 3: tile_u8<3>(a16, kp, bp, f, buf); break;
        case 2: tile_u8<2>(a16, kp, bp, f, buf); break;
        default: tile_u8<1>(a16, kp, bp, f, buf); break;
      }
      const int w = min_of(16, b.n - pj * 16);
      epilogue_avx2(buf, 16, mr, m0, pj * 16, w, e);
    }
  }
}

void gemm_f32_avx2(Rows<float> a, const F32View& b, int m, const Epilogue& e) {
  const int taps = a.taps, seg = b.k / taps, panels = b.np / 16;
  alignas(32) float buf[6 * 16];
  // A panel of B at a time, every row of A past it: the panel stays in L1.
  const float* const* all = block(a, 0, m);
  for (int pj = 0; pj < panels; ++pj) {
    const float* bp = b.data + static_cast<size_t>(pj) * b.k * 16;
    for (int m0 = 0; m0 < m; m0 += 6) {
      const int mr = min_of(6, m - m0);
      const float* const* rows = all + static_cast<size_t>(m0) * taps;
      switch (mr) {
        case 6: tile_f32<6>(rows, taps, seg, bp, buf); break;
        case 5: tile_f32<5>(rows, taps, seg, bp, buf); break;
        case 4: tile_f32<4>(rows, taps, seg, bp, buf); break;
        case 3: tile_f32<3>(rows, taps, seg, bp, buf); break;
        case 2: tile_f32<2>(rows, taps, seg, bp, buf); break;
        default: tile_f32<1>(rows, taps, seg, bp, buf); break;
      }
      const int w = min_of(16, b.n - pj * 16);
      epilogue_avx2(buf, 16, mr, m0, pj * 16, w, e);
    }
  }
}

// ---------------------------------------------------------------- depthwise
//
// 4 pixels x 16 channels a tile, 8 chains of sums at once (one chain a tap would
// wait on each sum's latency), the channels in chunks of 64 for the output stage.

namespace {

template <int PX>
void dw_tile_f32(const Rows<float>& a, int64_t i0, int taps, const float* w, int c, int off, float* buf) {
  __m256 acc[PX][2];
  for (int r = 0; r < PX; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_ps();
  for (int t = 0; t < taps; ++t) {
    const __m256 w0 = _mm256_loadu_ps(w + t * c), w1 = _mm256_loadu_ps(w + t * c + 8);
    for (int r = 0; r < PX; ++r) {
      const float* x = piece(a, i0 + r, t) + off;
      acc[r][0] = _mm256_fmadd_ps(_mm256_loadu_ps(x), w0, acc[r][0]);
      acc[r][1] = _mm256_fmadd_ps(_mm256_loadu_ps(x + 8), w1, acc[r][1]);
    }
  }
  for (int r = 0; r < PX; ++r) {
    _mm256_storeu_ps(buf + r * 64, acc[r][0]);
    _mm256_storeu_ps(buf + r * 64 + 8, acc[r][1]);
  }
}

// fma(x, y, s), fused as the reference's std::fma
inline float fma1(float x, float y, float s) {
  return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(x), _mm_set_ss(y), _mm_set_ss(s)));
}

inline __m256i widen_u8(const uint8_t* x, __m256i zero) {
  return _mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(x))), zero);
}
inline __m256i widen_s8(const int8_t* w) {
  return _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w)));
}

// Two taps at a time: both taps' (a - zero) for 16 channels in 16 bits,
// interleaved, against their weights interleaved alike, so one vpmaddwd makes
// 16 exact products and adds them in pairs (a lone last tap pairs with zeros).
// The interleave works inside each 128-bit lane, so the sums come out as
// channels 0-3 and 8-11 (lo), 4-7 and 12-15 (hi), and are put back in order
// at the end.  kGrid: the rows are a grid (no branch on the form per piece).
template <int PX, bool kGrid>
void dw_tile_u8(const Rows<uint8_t>& a, int64_t i0, int taps, __m256i zero, const int8_t* w, int c, int off,
                int32_t* buf) {
  const uint8_t* const* ptrs = a.ptrs;
  const uint8_t* const base = a.base;
  const int64_t stride = a.stride;
  const int64_t* const offsets = a.offsets;
  auto piece = [&](int r, int t) -> const __m128i* {
    if constexpr (kGrid) {
      return reinterpret_cast<const __m128i*>(base + (i0 + r) * stride + (offsets ? offsets[t] : 0) + off);
    } else {
      return reinterpret_cast<const __m128i*>(ptrs[(i0 + r) * taps + t] + off);
    }
  };
  __m256i lo[PX], hi[PX];
  for (int r = 0; r < PX; ++r) lo[r] = hi[r] = _mm256_setzero_si256();
  for (int t = 0; t < taps; t += 2) {
    const bool pair = t + 1 < taps;
    const __m256i wa = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + t * c)));
    const __m256i wb = pair ? _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + (t + 1) * c)))
                            : _mm256_setzero_si256();
    const __m256i wlo = _mm256_unpacklo_epi16(wa, wb), whi = _mm256_unpackhi_epi16(wa, wb);
    for (int r = 0; r < PX; ++r) {
      const __m256i xa = _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(piece(r, t))), zero);
      const __m256i xb = pair ? _mm256_sub_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128(piece(r, t + 1))), zero)
                              : _mm256_setzero_si256();
      lo[r] = _mm256_add_epi32(lo[r], _mm256_madd_epi16(_mm256_unpacklo_epi16(xa, xb), wlo));
      hi[r] = _mm256_add_epi32(hi[r], _mm256_madd_epi16(_mm256_unpackhi_epi16(xa, xb), whi));
    }
  }
  for (int r = 0; r < PX; ++r) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(buf + r * 64), _mm256_permute2x128_si256(lo[r], hi[r], 0x20));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(buf + r * 64 + 8), _mm256_permute2x128_si256(lo[r], hi[r], 0x31));
  }
}

}  // namespace

void dwconv_f32_avx2(Rows<float> a, const float* w, int c, int m, const Epilogue& e) {
  const int taps = a.taps;
  alignas(32) float buf[4 * 64];
  for (int i0 = 0; i0 < m; i0 += 4) {
    const int px = min_of(4, m - i0);
    for (int c0 = 0; c0 < c; c0 += 64) {
      const int cw = min_of(64, c - c0);
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
        for (; ct + 8 <= cw; ct += 8) {
          __m256 acc = _mm256_setzero_ps();
          for (int t = 0; t < taps; ++t) {
            const __m256 x = _mm256_loadu_ps(piece(a, i0 + r, t) + c0 + ct);
            acc = _mm256_fmadd_ps(x, _mm256_loadu_ps(w + t * c + c0 + ct), acc);
          }
          _mm256_storeu_ps(buf + r * 64 + ct, acc);
        }
        for (; ct < cw; ++ct) {
          float s = 0.0f;
          for (int t = 0; t < taps; ++t) s = fma1(piece(a, i0 + r, t)[c0 + ct], w[t * c + c0 + ct], s);
          buf[r * 64 + ct] = s;
        }
      }
      epilogue_avx2(buf, 64, px, i0, c0, cw, e);
    }
  }
}

void dwconv_u8s8_avx2(Rows<uint8_t> a, int32_t a_zero, const int8_t* w, int c, int m, const Epilogue& e) {
  const int taps = a.taps;
  const __m256i zero = _mm256_set1_epi32(a_zero), zero16 = _mm256_set1_epi16(static_cast<int16_t>(a_zero));
  const bool grid = !a.ptrs;
  alignas(32) int32_t buf[4 * 64];
  for (int i0 = 0; i0 < m; i0 += 4) {
    const int px = min_of(4, m - i0);
    for (int c0 = 0; c0 < c; c0 += 64) {
      const int cw = min_of(64, c - c0);
      int cc = 0;
      for (; cc + 16 <= cw; cc += 16) {
        const int8_t* wc = w + c0 + cc;
        int32_t* b = buf + cc;
        const int o = c0 + cc;
        switch (px * 2 + grid) {
          case 9: dw_tile_u8<4, true>(a, i0, taps, zero16, wc, c, o, b); break;
          case 8: dw_tile_u8<4, false>(a, i0, taps, zero16, wc, c, o, b); break;
          case 7: dw_tile_u8<3, true>(a, i0, taps, zero16, wc, c, o, b); break;
          case 6: dw_tile_u8<3, false>(a, i0, taps, zero16, wc, c, o, b); break;
          case 5: dw_tile_u8<2, true>(a, i0, taps, zero16, wc, c, o, b); break;
          case 4: dw_tile_u8<2, false>(a, i0, taps, zero16, wc, c, o, b); break;
          case 3: dw_tile_u8<1, true>(a, i0, taps, zero16, wc, c, o, b); break;
          default: dw_tile_u8<1, false>(a, i0, taps, zero16, wc, c, o, b); break;
        }
      }
      for (int r = 0; r < px; ++r) {
        int ct = cc;
        for (; ct + 8 <= cw; ct += 8) {
          __m256i acc = _mm256_setzero_si256();
          for (int t = 0; t < taps; ++t) {
            acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(widen_u8(piece(a, i0 + r, t) + c0 + ct, zero),
                                                           widen_s8(w + t * c + c0 + ct)));
          }
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(buf + r * 64 + ct), acc);
        }
        for (; ct < cw; ++ct) {
          int32_t s = 0;
          for (int t = 0; t < taps; ++t) {
            s += (static_cast<int32_t>(piece(a, i0 + r, t)[c0 + ct]) - a_zero) * w[t * c + c0 + ct];
          }
          buf[r * 64 + ct] = s;
        }
      }
      epilogue_avx2(buf, 64, px, i0, c0, cw, e);
    }
  }
}

// ---------------------------------------------------------------- CTC greedy

namespace {

// exp_neg (kernels.h) on 8 lanes: the same operations in the same order.
inline __m256 exp_neg8(__m256 x) {
  x = _mm256_max_ps(x, _mm256_set1_ps(kExpLo));  // (x > lo) ? x : lo, as the reference
  const __m256 n =
      _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(kLog2e)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n, _mm256_set1_ps(kLn2Hi)));
  r = _mm256_sub_ps(r, _mm256_mul_ps(n, _mm256_set1_ps(kLn2Lo)));
  __m256 p = _mm256_set1_ps(kExpP0);
  p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(kExpP1));
  p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(kExpP2));
  p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(kExpP3));
  p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(kExpP4));
  p = _mm256_add_ps(_mm256_mul_ps(p, r), _mm256_set1_ps(kExpP5));
  const __m256 y = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(p, _mm256_mul_ps(r, r)), r), _mm256_set1_ps(1.0f));
  const __m256i bits = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(bits));
}

}  // namespace

void ctc_argmax_avx2(const float* logits, int classes, int frames, int t_n, int32_t* ids, float* probs) {
  int t0 = 0;
  for (; t0 + 8 <= t_n; t0 += 8) {  // 8 frames at a time: each class's 8 logits are contiguous
    __m256 best = _mm256_set1_ps(-INFINITY);
    __m256i arg = _mm256_setzero_si256();
    for (int c = 0; c < classes; ++c) {
      const __m256 v = _mm256_loadu_ps(logits + static_cast<size_t>(c) * frames + t0);
      const __m256 more = _mm256_cmp_ps(v, best, _CMP_GT_OQ);  // strictly: ties keep the lower id
      best = _mm256_blendv_ps(best, v, more);
      arg = _mm256_blendv_epi8(arg, _mm256_set1_epi32(c), _mm256_castps_si256(more));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(ids + t0), arg);
    if (_mm256_testz_si256(arg, arg)) continue;  // eight blank frames: no probability is read
    __m256 sum = _mm256_setzero_ps();
    for (int c = 0; c < classes; ++c) {
      const __m256 v = _mm256_loadu_ps(logits + static_cast<size_t>(c) * frames + t0);
      sum = _mm256_add_ps(sum, exp_neg8(_mm256_sub_ps(v, best)));
    }
    const __m256 prob = _mm256_div_ps(_mm256_set1_ps(1.0f), sum);
    const __m256i blank = _mm256_cmpeq_epi32(arg, _mm256_setzero_si256());
    _mm256_maskstore_ps(probs + t0, _mm256_xor_si256(blank, _mm256_set1_epi32(-1)), prob);
  }
  for (int t = t0; t < t_n; ++t) {  // the last frames as the reference does them
    float best = -INFINITY;
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
