// The GEMMs on AVX2 + FMA (carbon/src/cpu/kernels.h), 6 rows x 16 columns a tile.
//
// int8 without VNNI (AMD Zen 2 and 3, Intel before Ice Lake): vpmaddwd on 16-bit
// values, A widened to uint16 and B packed as int16 pairs of k, so every product
// is exact (vpmaddubsw would saturate at 255 * 127 * 2).
// FP32: vfmadd231ps, the sums in 12 registers.
// Also the depthwise convolutions, CTC greedy decoding, a crop's counts, a
// threshold's bits, the detector's head, the crop rule's resampling and the
// page turns (bilinear in doubles with vpgatherdd for the
// taps; 16 x 16 byte transposes, two to a register), which every x86-64 set runs
// from here.
// FP16 (F16C, no FP16 arithmetic): the weights stay FP16 and a panel is widened
// (vcvtph2ps) into L1 as a call starts on it, then the FP32 tiles run on it.
// Built with -mavx2 -mfma -mf16c (/arch:AVX2).

#include <immintrin.h>

#include <cmath>
#include <cstring>

#include "epilogue_avx2.h"
#include "kernels.h"
#include "resample_math.h"

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

// 8 activations as FP32: FP32 loaded, FP16 widened.
inline __m256 load8(const float* x) { return _mm256_loadu_ps(x); }
inline __m256 load8(const uint16_t* x) { return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x))); }
inline float load1(const float* x) { return *x; }
inline float load1(const uint16_t* x) { return half_to_float(*x); }

template <int PX, typename A>
void dw_tile_f32(const Rows<A>& a, int64_t i0, int taps, const float* w, int c, int off, float* buf) {
  __m256 acc[PX][2];
  for (int r = 0; r < PX; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_ps();
  for (int t = 0; t < taps; ++t) {
    const __m256 w0 = _mm256_loadu_ps(w + t * c), w1 = _mm256_loadu_ps(w + t * c + 8);
    for (int r = 0; r < PX; ++r) {
      const A* x = piece(a, i0 + r, t) + off;
      acc[r][0] = _mm256_fmadd_ps(load8(x), w0, acc[r][0]);
      acc[r][1] = _mm256_fmadd_ps(load8(x + 8), w1, acc[r][1]);
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

// FP32 weights; FP32 or FP16 activations.
template <typename A>
void dwconv_avx2(const Rows<A>& a, const float* w, int c, int m, const Epilogue& e) {
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
            const __m256 x = load8(piece(a, i0 + r, t) + c0 + ct);
            acc = _mm256_fmadd_ps(x, _mm256_loadu_ps(w + t * c + c0 + ct), acc);
          }
          _mm256_storeu_ps(buf + r * 64 + ct, acc);
        }
        for (; ct < cw; ++ct) {
          float s = 0.0f;
          for (int t = 0; t < taps; ++t) s = fma1(load1(piece(a, i0 + r, t) + c0 + ct), w[t * c + c0 + ct], s);
          buf[r * 64 + ct] = s;
        }
      }
      epilogue_avx2(buf, 64, px, i0, c0, cw, e);
    }
  }
}

}  // namespace

void dwconv_f32_avx2(Rows<float> a, const float* w, int c, int m, const Epilogue& e) { dwconv_avx2(a, w, c, m, e); }

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

// ---------------------------------------------------------------- FP16 (F16C)
//
// FP16 stored, FP32 arithmetic (net.cpp says why the activations stay FP32 here).
// B widened whole into the kB scratch and kept there for the next call on the same
// packing (a layer's next output row), up to kKeepB; past it a panel at a time.
// FP32 A: a panel of B at a time, every row of A past it, as gemm_f32_avx2.  FP16
// A: a block of 6 rows at a time widened into the kA scratch, once for every
// panel, each row's pieces end to end, so tile_f32 reads it as one piece of k
// values.  Either way the same sums in the same order as over the taps: the
// reference's bits.  Depthwise: the weights widened once a call, FP16 activations
// as they are loaded.

namespace {

// n FP16 values widened: 8 an instruction, the rest one at a time (the same bits).
inline void widen_f16(const uint16_t* src, float* dst, int64_t n) {
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i))));
  }
  for (; i < n; ++i) dst[i] = half_to_float(src[i]);
}

// tile_f32 on mr rows, 1 to 6.
inline void tile_rows(int mr, const float* const* rows, int taps, int seg, const float* bp, float* buf) {
  switch (mr) {
    case 6: tile_f32<6>(rows, taps, seg, bp, buf); break;
    case 5: tile_f32<5>(rows, taps, seg, bp, buf); break;
    case 4: tile_f32<4>(rows, taps, seg, bp, buf); break;
    case 3: tile_f32<3>(rows, taps, seg, bp, buf); break;
    case 2: tile_f32<2>(rows, taps, seg, bp, buf); break;
    default: tile_f32<1>(rows, taps, seg, bp, buf); break;
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
  alignas(32) float buf[6 * 16];
  for (int pj = 0; pj < panels; ++pj) {
    if (!whole) widen_f16(b.data + pj * panel, bp, static_cast<int64_t>(panel));
    const float* const p = whole ? bw + pj * panel : bp;
    for (int m0 = 0; m0 < m; m0 += 6) {
      const int mr = min_of(6, m - m0);
      tile_rows(mr, all + static_cast<size_t>(m0) * taps, taps, seg, p, buf);
      epilogue_avx2(buf, 16, mr, m0, pj * 16, min_of(16, b.n - pj * 16), e);
    }
  }
}

void gemm_f16_half(const Rows<uint16_t>& a, const F16View& b, int m, const Epilogue& e) {
  const int taps = a.taps, k = b.k, seg = k / taps, panels = b.np / 16;
  const float* const bw = widened(b);
  float* const a32 = scratch_of<float>(Scratch::kA, static_cast<size_t>(6) * k);
  const float* rows[6];
  alignas(32) float buf[6 * 16];
  for (int m0 = 0; m0 < m; m0 += 6) {
    const int mr = min_of(6, m - m0);
    for (int r = 0; r < mr; ++r) {
      float* row = a32 + static_cast<size_t>(r) * k;
      for (int t = 0; t < taps; ++t) widen_f16(piece(a, m0 + r, t), row + static_cast<size_t>(t) * seg, seg);
      rows[r] = row;
    }
    for (int pj = 0; pj < panels; ++pj) {
      tile_rows(mr, rows, 1, k, bw + static_cast<size_t>(pj) * k * 16, buf);
      epilogue_avx2(buf, 16, mr, m0, pj * 16, min_of(16, b.n - pj * 16), e);
    }
  }
}

template <typename A>
void dwconv_f16_impl(const Rows<A>& a, const uint16_t* w, int c, int m, const Epilogue& e) {
  const int64_t n = static_cast<int64_t>(a.taps) * c;
  float* const wf = scratch_of<float>(Scratch::kB, static_cast<size_t>(n));
  widen_f16(w, wf, n);
  dwconv_avx2(a, wf, c, m, e);
}

}  // namespace

void gemm_f16_avx2(Rows<uint16_t> a, const F16View& b, int m, const Epilogue& e) { gemm_f16_half(a, b, m, e); }
void gemm_f16_avx2(Rows<float> a, const F16View& b, int m, const Epilogue& e) { gemm_f16_wide(a, b, m, e); }

void dwconv_f16_avx2(Rows<uint16_t> a, const uint16_t* w, int c, int m, const Epilogue& e) {
  dwconv_f16_impl(a, w, c, m, e);
}
void dwconv_f16_avx2(Rows<float> a, const uint16_t* w, int c, int m, const Epilogue& e) {
  dwconv_f16_impl(a, w, c, m, e);
}

void to_f32_avx2(const uint16_t* in, float* out, size_t n) { widen_f16(in, out, static_cast<int64_t>(n)); }

void to_f16_avx2(const float* in, uint16_t* out, size_t n) {
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), _mm256_cvtps_ph(_mm256_loadu_ps(in + i), 0));
  }
  for (; i < n; ++i) out[i] = float_to_half(in[i]);
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

namespace {

// det_head's row of the head at stride 4: channel c's row in head[c][w4], an FMA chain a pixel, bias first.  Four
// channels by 16 pixels a pass (8 accumulators, each input's two vectors loaded once for the four, 6 loads to
// 8 FMAs), then each channel's pixels left 8 and 1 at a time; every pixel's chain the same.
void det_head_row(const float* x, int h4, int w4, const DetHeadWeights& w, int p, float* head) {
  const __m256 zero = _mm256_setzero_ps();
  const int64_t plane = static_cast<int64_t>(h4) * w4;
  const float* xr = x + static_cast<int64_t>(p) * w4;
  const int wide = w4 & ~15, quads = w.mid & ~3;
  for (int c = 0; c < quads; c += 4) {
    const float* wc = w.w1 + static_cast<size_t>(c) * w.in;
    for (int px = 0; px < wide; px += 16) {
      __m256 a[4][2];
      for (int j = 0; j < 4; ++j) a[j][0] = a[j][1] = _mm256_set1_ps(w.b1[c + j]);
      for (int i = 0; i < w.in; ++i) {
        const float* xi = xr + i * plane + px;
        const __m256 x0 = _mm256_loadu_ps(xi), x1 = _mm256_loadu_ps(xi + 8);
        for (int j = 0; j < 4; ++j) {
          const __m256 wv = _mm256_set1_ps(wc[static_cast<size_t>(j) * w.in + i]);
          a[j][0] = _mm256_fmadd_ps(wv, x0, a[j][0]);
          a[j][1] = _mm256_fmadd_ps(wv, x1, a[j][1]);
        }
      }
      for (int j = 0; j < 4; ++j) {
        float* hc = head + static_cast<size_t>(c + j) * w4 + px;
        _mm256_storeu_ps(hc, _mm256_max_ps(a[j][0], zero));
        _mm256_storeu_ps(hc + 8, _mm256_max_ps(a[j][1], zero));
      }
    }
  }
  for (int c = 0; c < w.mid; ++c) {
    const float* wc = w.w1 + static_cast<size_t>(c) * w.in;
    float* hc = head + static_cast<size_t>(c) * w4;
    int px = c < quads ? wide : 0;
    for (; px + 8 <= w4; px += 8) {
      __m256 acc = _mm256_set1_ps(w.b1[c]);
      for (int i = 0; i < w.in; ++i) {
        acc = _mm256_fmadd_ps(_mm256_set1_ps(wc[i]), _mm256_loadu_ps(x + (static_cast<int64_t>(i) * h4 + p) * w4 + px),
                              acc);
      }
      _mm256_storeu_ps(hc + px, _mm256_max_ps(acc, zero));
    }
    for (; px < w4; ++px) {
      float acc = w.b1[c];
      for (int i = 0; i < w.in; ++i) acc = fmaf(wc[i], x[(static_cast<int64_t>(i) * h4 + p) * w4 + px], acc);
      hc[px] = acc > 0.0f ? acc : 0.0f;
    }
  }
}

// The head's row upsampled 2x (nearest) into u[c][P], a zero either side.
void det_head_up(const float* head, int mid, int w4, int P, float* u) {
  for (int c = 0; c < mid; ++c) {
    const float* hc = head + static_cast<size_t>(c) * w4;
    float* uc = u + static_cast<size_t>(c) * P;
    uc[0] = 0.0f;
    uc[P - 1] = 0.0f;
    int px = 0;
    for (; px + 8 <= w4; px += 8) {
      const __m256 v = _mm256_loadu_ps(hc + px);  // a b c d | e f g h
      const __m256 lo = _mm256_unpacklo_ps(v, v);  // a a b b | e e f f
      const __m256 hi = _mm256_unpackhi_ps(v, v);  // c c d d | g g h h
      _mm256_storeu_ps(uc + 1 + 2 * px, _mm256_permute2f128_ps(lo, hi, 0x20));
      _mm256_storeu_ps(uc + 1 + 2 * px + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
    }
    for (; px < w4; ++px) {
      uc[1 + 2 * px] = hc[px];
      uc[2 + 2 * px] = hc[px];
    }
  }
}

}  // namespace

namespace {

// 16 output pixels of a row from ox, OUT maps: per channel 9 weight broadcasts and 18 loads for 18 FMAs, then
// 4 broadcasts for 8 (each broadcast twice), 13 accumulators and temporaries in the 16 ymm; every pixel's chains as
// the 8-pixel pass's.  row: the maps' row oy, planes H x W apart.
template <int OUT>
void head_pass16(const float* const* z, const DetHeadWeights& w, int P, int ox, float* row, int H, int W) {
  const __m256 zero = _mm256_setzero_ps();
  __m256 m[OUT][2];
  for (int o = 0; o < OUT; ++o) m[o][0] = m[o][1] = _mm256_set1_ps(w.b3[o]);
  for (int c = 0; c < w.mid; ++c) {
    const float* wc = w.w2 + static_cast<size_t>(c) * 9;
    __m256 a0 = _mm256_set1_ps(w.b2[c]), a1 = a0;
    for (int ky = 0; ky < 3; ++ky) {
      const float* zr = z[ky] + static_cast<size_t>(c) * P + ox;
      for (int kx = 0; kx < 3; ++kx) {
        const __m256 wv = _mm256_set1_ps(wc[ky * 3 + kx]);
        a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(zr + kx), a0);
        a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(zr + kx + 8), a1);
      }
    }
    a0 = _mm256_max_ps(a0, zero);
    a1 = _mm256_max_ps(a1, zero);
    for (int o = 0; o < OUT; ++o) {
      const __m256 wv = _mm256_set1_ps(w.w3[static_cast<size_t>(o) * w.mid + c]);
      m[o][0] = _mm256_fmadd_ps(wv, a0, m[o][0]);
      m[o][1] = _mm256_fmadd_ps(wv, a1, m[o][1]);
    }
  }
  for (int o = 0; o < OUT; ++o) {
    float* r = row + static_cast<int64_t>(o) * H * W + ox;
    _mm256_storeu_ps(r, m[o][0]);
    _mm256_storeu_ps(r + 8, m[o][1]);
  }
}

}  // namespace

void det_head_avx2(const float* x, int h4, int w4, const DetHeadWeights& w, float* maps, int row0, int row1) {
  const int H = 2 * h4, W = 2 * w4, P = W + 2, mid = w.mid, out = w.out;
  float* head = scratch_of<float>(Scratch::kA, static_cast<size_t>(mid) * (w4 + 4 * P));
  float* up = head + static_cast<size_t>(mid) * w4;
  float* zero_row = up + static_cast<size_t>(3) * mid * P;
  memset(zero_row, 0, sizeof(float) * static_cast<size_t>(mid) * P);
  int held[3] = {-1, -1, -1};
  const __m256 zero = _mm256_setzero_ps();
  for (int oy = row0; oy < row1; ++oy) {
    const float* z[3];
    for (int ky = 0; ky < 3; ++ky) {
      const int zy = oy + ky - 1;
      if (zy < 0 || zy >= H) {
        z[ky] = zero_row;
        continue;
      }
      const int p = zy >> 1;
      float* u = up + static_cast<size_t>(p % 3) * mid * P;
      if (held[p % 3] != p) {
        det_head_row(x, h4, w4, w, p, head);
        det_head_up(head, mid, w4, P, u);
        held[p % 3] = p;
      }
      z[ky] = u;
    }
    int ox = 0;
    if (out == 4) {  // 16 pixels a pass, each broadcast used twice (fullerene's head: 4 maps)
      for (; ox + 16 <= W; ox += 16) head_pass16<4>(z, w, P, ox, maps + static_cast<int64_t>(oy) * W, H, W);
    }
    for (; ox + 8 <= W; ox += 8) {
      __m256 m[8];
      for (int o = 0; o < out; ++o) m[o] = _mm256_set1_ps(w.b3[o]);
      for (int c = 0; c < mid; ++c) {
        const float* wc = w.w2 + static_cast<size_t>(c) * 9;
        __m256 acc = _mm256_set1_ps(w.b2[c]);
        for (int ky = 0; ky < 3; ++ky) {
          const float* zr = z[ky] + static_cast<size_t>(c) * P + ox;
          acc = _mm256_fmadd_ps(_mm256_set1_ps(wc[ky * 3]), _mm256_loadu_ps(zr), acc);
          acc = _mm256_fmadd_ps(_mm256_set1_ps(wc[ky * 3 + 1]), _mm256_loadu_ps(zr + 1), acc);
          acc = _mm256_fmadd_ps(_mm256_set1_ps(wc[ky * 3 + 2]), _mm256_loadu_ps(zr + 2), acc);
        }
        const __m256 d = _mm256_max_ps(acc, zero);
        for (int o = 0; o < out; ++o) m[o] = _mm256_fmadd_ps(_mm256_set1_ps(w.w3[static_cast<size_t>(o) * mid + c]), d, m[o]);
      }
      for (int o = 0; o < out; ++o) _mm256_storeu_ps(maps + (static_cast<int64_t>(o) * H + oy) * W + ox, m[o]);
    }
    for (; ox < W; ++ox) {  // the row's tail, as the reference
      float m[8];
      for (int o = 0; o < out; ++o) m[o] = w.b3[o];
      for (int c = 0; c < mid; ++c) {
        float acc = w.b2[c];
        for (int ky = 0; ky < 3; ++ky) {
          for (int kx = 0; kx < 3; ++kx) {
            acc = fmaf(w.w2[static_cast<size_t>(c) * 9 + ky * 3 + kx], z[ky][static_cast<size_t>(c) * P + ox + kx], acc);
          }
        }
        const float d = acc > 0.0f ? acc : 0.0f;
        for (int o = 0; o < out; ++o) m[o] = fmaf(w.w3[static_cast<size_t>(o) * mid + c], d, m[o]);
      }
      for (int o = 0; o < out; ++o) maps[(static_cast<int64_t>(o) * H + oy) * W + ox] = m[o];
    }
  }
}

// 32 values a word, 8 a compare: vcmpps (ordered, so a NaN is never above) and vmovmskps; the last word's values one
// at a time.
void above_bits_avx2(const float* v, int n, float t, uint32_t* bits) {
  const __m256 tv = _mm256_set1_ps(t);
  int i = 0;
  for (; i + 32 <= n; i += 32) {
    uint32_t word = 0;
    for (int k = 0; k < 4; ++k) {
      const __m256 more = _mm256_cmp_ps(_mm256_loadu_ps(v + i + 8 * k), tv, _CMP_GT_OQ);
      word |= static_cast<uint32_t>(_mm256_movemask_ps(more)) << (8 * k);
    }
    bits[i / 32] = word;
  }
  if (i < n) {
    uint32_t word = 0;
    for (int k = 0; i + k < n; ++k) word |= (v[i + k] > t ? 1u : 0u) << k;
    bits[i / 32] = word;
  }
}

// 32 bytes a step: a byte at most lo where min(v, lo) is v, at least hi where max(v, hi) is v; the masks' ones
// summed by vpsadbw into four 64-bit lanes. The row's last bytes one at a time.
// Four points a step: x and y widened by vcvtps2pd (exact), r = y c - x s + mid by vmulpd, vmulpd, vsubpd, vaddpd
// (the reference's operations in its order; no FMA), truncated by vcvttpd2dq as the reference's cast, clamped by
// vpmaxsd and vpminsd, each lane's row and its histogram (lane i & 3) as one index, then the four counts.
void skew_rows_avx2(const float* x, const float* y, int64_t n, double c, double s, double mid, int last,
                    int32_t* count) {
  const __m256d vc = _mm256_set1_pd(c), vs = _mm256_set1_pd(s), vm = _mm256_set1_pd(mid);
  const __m128i zero = _mm_setzero_si128(), top = _mm_set1_epi32(last), lane = _mm_setr_epi32(0, 1, 2, 3);
  alignas(16) int32_t at[4];
  int64_t i = 0;
  for (; i + 4 <= n; i += 4) {
    const __m256d xd = _mm256_cvtps_pd(_mm_loadu_ps(x + i)), yd = _mm256_cvtps_pd(_mm_loadu_ps(y + i));
    const __m256d r = _mm256_add_pd(_mm256_sub_pd(_mm256_mul_pd(yd, vc), _mm256_mul_pd(xd, vs)), vm);
    __m128i v = _mm_min_epi32(_mm_max_epi32(_mm256_cvttpd_epi32(r), zero), top);
    v = _mm_add_epi32(_mm_slli_epi32(v, 2), lane);
    _mm_store_si128(reinterpret_cast<__m128i*>(at), v);
    ++count[at[0]];
    ++count[at[1]];
    ++count[at[2]];
    ++count[at[3]];
  }
  for (; i < n; ++i) {
    const double r = static_cast<double>(y[i]) * c - static_cast<double>(x[i]) * s + mid;
    int v = static_cast<int>(r);
    v = v < 0 ? 0 : v > last ? last : v;
    ++count[4 * v + static_cast<int>(i & 3)];
  }
}

ByteCounts count_beyond_avx2(const uint8_t* c, int w, int h, int64_t stride, int lo, int hi) {
  const bool any_below = lo >= 0, any_above = hi <= 255;
  const __m256i lo8 = _mm256_set1_epi8(static_cast<char>(any_below ? lo : 0));
  const __m256i hi8 = _mm256_set1_epi8(static_cast<char>(any_above ? hi : 255));
  const __m256i one = _mm256_set1_epi8(1), zero = _mm256_setzero_si256();
  __m256i below = zero, above = zero;
  ByteCounts n{0, 0};
  for (int y = 0; y < h; ++y) {
    const uint8_t* row = c + y * stride;
    int x = 0;
    for (; x + 32 <= w; x += 32) {
      const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + x));
      const __m256i le = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_min_epu8(v, lo8), v), one);
      const __m256i ge = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_max_epu8(v, hi8), v), one);
      below = _mm256_add_epi64(below, _mm256_sad_epu8(le, zero));
      above = _mm256_add_epi64(above, _mm256_sad_epu8(ge, zero));
    }
    for (; x < w; ++x) {
      n.below += row[x] <= lo;
      n.above += row[x] >= hi;
    }
  }
  alignas(32) int64_t lanes[4];
  _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), below);
  if (any_below) n.below += lanes[0] + lanes[1] + lanes[2] + lanes[3];
  _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), above);
  if (any_above) n.above += lanes[0] + lanes[1] + lanes[2] + lanes[3];
  return n;
}

// The softmax: the scale and the max by vmulps and vmaxps (the max is the same whatever the order); exp_neg8 of
// each less the max into 8 lanes, lane j mod 8 the reference's partial sum j, a tail's lanes past the row 0; the
// lanes joined as the reference joins them (high half to low, then pairs, then the two); times 1 / the sum.
void softmax_rows_avx2(float* rows, int n_rows, int n, float scale) {
  const __m256 sc = _mm256_set1_ps(scale);
  const int whole = n & ~7, tail = n - whole;
  alignas(32) static const int32_t kLanes[16] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};
  const __m256i mask = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(kLanes + 8 - tail));
  for (int i = 0; i < n_rows; ++i) {
    float* row = rows + static_cast<size_t>(i) * n;
    __m256 hi8 = _mm256_set1_ps(-INFINITY);
    for (int j = 0; j < whole; j += 8) {
      const __m256 x = _mm256_mul_ps(_mm256_loadu_ps(row + j), sc);
      _mm256_storeu_ps(row + j, x);
      hi8 = _mm256_max_ps(hi8, x);
    }
    float hi = -INFINITY;
    for (int j = whole; j < n; ++j) {
      row[j] *= scale;
      hi = row[j] > hi ? row[j] : hi;
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, hi8);
    for (int k = 0; k < 8; ++k) hi = lanes[k] > hi ? lanes[k] : hi;
    const __m256 h = _mm256_set1_ps(hi);
    __m256 acc = _mm256_setzero_ps();
    for (int j = 0; j < whole; j += 8) {
      const __m256 e = exp_neg8(_mm256_sub_ps(_mm256_loadu_ps(row + j), h));
      _mm256_storeu_ps(row + j, e);
      acc = _mm256_add_ps(acc, e);
    }
    if (tail) {
      const __m256 e = exp_neg8(_mm256_sub_ps(_mm256_maskload_ps(row + whole, mask), h));
      _mm256_maskstore_ps(row + whole, mask, e);
      acc = _mm256_add_ps(acc, _mm256_and_ps(e, _mm256_castsi256_ps(mask)));
    }
    const __m128 q = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));  // s_k + s_k+4
    const __m128 pr = _mm_add_ps(q, _mm_movehl_ps(q, q));                                   // (0+4)+(2+6), (1+5)+(3+7)
    const float sum = _mm_cvtss_f32(_mm_add_ss(pr, _mm_shuffle_ps(pr, pr, 1)));
    const __m256 inv = _mm256_set1_ps(1.0f / sum);
    for (int j = 0; j < whole; j += 8) _mm256_storeu_ps(row + j, _mm256_mul_ps(_mm256_loadu_ps(row + j), inv));
    const float inv1 = 1.0f / sum;
    for (int j = whole; j < n; ++j) row[j] *= inv1;
  }
}

// The stem: the stride-2 rows in a ring of three, each [w2 + 2 pixels][16 channels] with a zero pixel either side
// (the second convolution's padding), a zero row for the rows past the map; 4 stride-2 pixels a step, per tap 4
// broadcasts of the input and 2 weight vectors for 8 FMAs (the input rows copied with a zero either side). The
// downsampling: each output pixel's 144 inputs gathered from the ring into L1 (9 taps of 16 contiguous channels,
// in the reference's order), a 4-pixel by c2-channel tile (up to 12 accumulators) over them, per k c2 / 8 weight
// vectors and 4 broadcasts; ReLU, then each channel's 4 pixels stored to the NCHW output.
namespace {

// The 4-pixel tile over the 144 inputs, NV vectors of 8 channels (c2 = 8 NV), ReLU'd into tile[pixel][channel].
template <int NV>
[[maybe_unused]] void stem_tile(const float* const* cols, const float* w2p, const float* b2, float (*tile)[24]) {
  constexpr int kK = 144, kC2 = 8 * NV;
  __m256 acc[4][NV];
  for (int j = 0; j < NV; ++j) {
    const __m256 bj = _mm256_loadu_ps(b2 + 8 * j);
    for (int p = 0; p < 4; ++p) acc[p][j] = bj;
  }
  for (int k = 0; k < kK; ++k) {
    __m256 b[NV];
    for (int j = 0; j < NV; ++j) b[j] = _mm256_loadu_ps(w2p + static_cast<size_t>(k) * kC2 + 8 * j);
    for (int p = 0; p < 4; ++p) {
      const __m256 a = _mm256_broadcast_ss(cols[p] + k);
      for (int j = 0; j < NV; ++j) acc[p][j] = _mm256_fmadd_ps(b[j], a, acc[p][j]);
    }
  }
  const __m256 zero = _mm256_setzero_ps();
  for (int p = 0; p < 4; ++p) {
    for (int j = 0; j < NV; ++j) _mm256_store_ps(&tile[p][8 * j], _mm256_max_ps(acc[p][j], zero));
  }
}

// fullerene's 24 channels: the 12 accumulators named (MSVC keeps an array of them in memory), 3 weight vectors and
// a broadcast: the 16 ymm registers.
void stem_tile24(const float* const* cols, const float* w2p, const float* b2, float (*tile)[24]) {
  const float *p0 = cols[0], *p1 = cols[1], *p2 = cols[2], *p3 = cols[3];
  __m256 a00 = _mm256_loadu_ps(b2), a01 = _mm256_loadu_ps(b2 + 8), a02 = _mm256_loadu_ps(b2 + 16);
  __m256 a10 = a00, a11 = a01, a12 = a02, a20 = a00, a21 = a01, a22 = a02, a30 = a00, a31 = a01, a32 = a02;
  for (int k = 0; k < 144; ++k) {
    const float* wk = w2p + static_cast<size_t>(k) * 24;
    const __m256 b0 = _mm256_loadu_ps(wk), b1 = _mm256_loadu_ps(wk + 8), b2v = _mm256_loadu_ps(wk + 16);
    __m256 v = _mm256_broadcast_ss(p0 + k);
    a00 = _mm256_fmadd_ps(b0, v, a00);
    a01 = _mm256_fmadd_ps(b1, v, a01);
    a02 = _mm256_fmadd_ps(b2v, v, a02);
    v = _mm256_broadcast_ss(p1 + k);
    a10 = _mm256_fmadd_ps(b0, v, a10);
    a11 = _mm256_fmadd_ps(b1, v, a11);
    a12 = _mm256_fmadd_ps(b2v, v, a12);
    v = _mm256_broadcast_ss(p2 + k);
    a20 = _mm256_fmadd_ps(b0, v, a20);
    a21 = _mm256_fmadd_ps(b1, v, a21);
    a22 = _mm256_fmadd_ps(b2v, v, a22);
    v = _mm256_broadcast_ss(p3 + k);
    a30 = _mm256_fmadd_ps(b0, v, a30);
    a31 = _mm256_fmadd_ps(b1, v, a31);
    a32 = _mm256_fmadd_ps(b2v, v, a32);
  }
  const __m256 z = _mm256_setzero_ps();
  _mm256_store_ps(&tile[0][0], _mm256_max_ps(a00, z));
  _mm256_store_ps(&tile[0][8], _mm256_max_ps(a01, z));
  _mm256_store_ps(&tile[0][16], _mm256_max_ps(a02, z));
  _mm256_store_ps(&tile[1][0], _mm256_max_ps(a10, z));
  _mm256_store_ps(&tile[1][8], _mm256_max_ps(a11, z));
  _mm256_store_ps(&tile[1][16], _mm256_max_ps(a12, z));
  _mm256_store_ps(&tile[2][0], _mm256_max_ps(a20, z));
  _mm256_store_ps(&tile[2][8], _mm256_max_ps(a21, z));
  _mm256_store_ps(&tile[2][16], _mm256_max_ps(a22, z));
  _mm256_store_ps(&tile[3][0], _mm256_max_ps(a30, z));
  _mm256_store_ps(&tile[3][8], _mm256_max_ps(a31, z));
  _mm256_store_ps(&tile[3][16], _mm256_max_ps(a32, z));
}

// The stem's 4 stride-2 pixels from q0 of a row: the 8 accumulators named; per tap 2 weight vectors and 4
// broadcasts. in: the 3 padded input rows (w + 2 apart), at the pixels' first column.
void stem_four(const float* in, int64_t pitch, const float* w1p, const float* b1, float* r) {
  __m256 a0 = _mm256_loadu_ps(b1), b0 = _mm256_loadu_ps(b1 + 8);
  __m256 a1 = a0, b1v = b0, a2 = a0, b2 = b0, a3 = a0, b3 = b0;
  for (int ky = 0; ky < 3; ++ky) {
    const float* p = in + ky * pitch;
    for (int kx = 0; kx < 3; ++kx) {
      const float* wt = w1p + (ky * 3 + kx) * 16;
      const __m256 wa = _mm256_loadu_ps(wt), wb = _mm256_loadu_ps(wt + 8);
      __m256 v = _mm256_broadcast_ss(p + kx);
      a0 = _mm256_fmadd_ps(wa, v, a0);
      b0 = _mm256_fmadd_ps(wb, v, b0);
      v = _mm256_broadcast_ss(p + 2 + kx);
      a1 = _mm256_fmadd_ps(wa, v, a1);
      b1v = _mm256_fmadd_ps(wb, v, b1v);
      v = _mm256_broadcast_ss(p + 4 + kx);
      a2 = _mm256_fmadd_ps(wa, v, a2);
      b2 = _mm256_fmadd_ps(wb, v, b2);
      v = _mm256_broadcast_ss(p + 6 + kx);
      a3 = _mm256_fmadd_ps(wa, v, a3);
      b3 = _mm256_fmadd_ps(wb, v, b3);
    }
  }
  const __m256 z = _mm256_setzero_ps();
  _mm256_storeu_ps(r, _mm256_max_ps(a0, z));
  _mm256_storeu_ps(r + 8, _mm256_max_ps(b0, z));
  _mm256_storeu_ps(r + 16, _mm256_max_ps(a1, z));
  _mm256_storeu_ps(r + 24, _mm256_max_ps(b1v, z));
  _mm256_storeu_ps(r + 32, _mm256_max_ps(a2, z));
  _mm256_storeu_ps(r + 40, _mm256_max_ps(b2, z));
  _mm256_storeu_ps(r + 48, _mm256_max_ps(a3, z));
  _mm256_storeu_ps(r + 56, _mm256_max_ps(b3, z));
}

}  // namespace

void det_stem_avx2(const float* x, int h, int w, const DetStemWeights& sw, float* out, int row0, int row1) {
  constexpr int kC1 = 16, kK = 9 * kC1;
  const int h2 = h / 2, w2 = w / 2, h4 = h / 4, w4 = w / 4, c2 = sw.c2, nv = c2 / 8;
  const size_t pitch = static_cast<size_t>(w2 + 2) * kC1;  // a ring row's floats
  // One scratch block: the ring's 3 rows and a zero row, 3 padded input rows, W2 packed [k][c2] (k = tap * 16 + c),
  // W1 packed [tap][16], and 4 pixels' 144 inputs.
  const size_t ring_n = 4 * pitch, in_n = 3 * static_cast<size_t>(w + 2), w2_n = static_cast<size_t>(kK) * 24;
  float* ring = scratch_of<float>(Scratch::kA, ring_n + in_n + w2_n + 144 + 4 * kK + 64);
  float* zero_row = ring + 3 * pitch;
  float* in = ring + ring_n;
  float* w2p = in + in_n;
  float* w1p = w2p + w2_n;
  float* col = w1p + 144;
  memset(zero_row, 0, sizeof(float) * pitch);
  for (int d = 0; d < c2; ++d) {
    for (int c = 0; c < kC1; ++c) {
      for (int t = 0; t < 9; ++t) w2p[static_cast<size_t>(t * kC1 + c) * c2 + d] = sw.w2[(static_cast<size_t>(d) * kC1 + c) * 9 + t];
    }
  }
  for (int c = 0; c < kC1; ++c) {
    for (int t = 0; t < 9; ++t) w1p[t * 16 + c] = sw.w1[c * 9 + t];
  }
  int held[3] = {-1, -1, -1};
  const __m256 zero = _mm256_setzero_ps();
  const __m256 b1a = _mm256_loadu_ps(sw.b1), b1b = _mm256_loadu_ps(sw.b1 + 8);
  auto stem_row = [&](int y2) -> const float* {
    float* r = ring + static_cast<size_t>(y2 % 3) * pitch;
    if (held[y2 % 3] == y2) return r;
    for (int ky = 0; ky < 3; ++ky) {  // the input rows, a zero either side (and zero rows past the map)
      float* p = in + static_cast<size_t>(ky) * (w + 2);
      const int iy = 2 * y2 + ky - 1;
      p[0] = 0.0f;
      p[w + 1] = 0.0f;
      if (iy < 0 || iy >= h) {
        memset(p + 1, 0, sizeof(float) * w);
      } else {
        memcpy(p + 1, x + static_cast<size_t>(iy) * w, sizeof(float) * w);
      }
    }
    memset(r, 0, sizeof(float) * kC1);  // the zero pixels either side
    memset(r + pitch - kC1, 0, sizeof(float) * kC1);
    int x2 = 0;
    for (; x2 + 4 <= w2; x2 += 4) {  // padded index 2 x2 + kx; the ring's pixels from x2 + 1
      stem_four(in + 2 * x2, w + 2, w1p, sw.b1, r + static_cast<size_t>(x2 + 1) * kC1);
    }
    for (; x2 < w2; ++x2) {
      __m256 a0 = b1a, a1 = b1b;
      for (int ky = 0; ky < 3; ++ky) {
        const float* p = in + static_cast<size_t>(ky) * (w + 2) + 2 * x2;
        for (int kx = 0; kx < 3; ++kx) {
          const int t = ky * 3 + kx;
          const __m256 v = _mm256_broadcast_ss(p + kx);
          a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w1p + t * 16), v, a0);
          a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w1p + t * 16 + 8), v, a1);
        }
      }
      float* o = r + static_cast<size_t>(x2 + 1) * kC1;
      _mm256_storeu_ps(o, _mm256_max_ps(a0, zero));
      _mm256_storeu_ps(o + 8, _mm256_max_ps(a1, zero));
    }
    held[y2 % 3] = y2;
    return r;
  };
  alignas(32) float tile[4][24];
  for (int y4 = row0; y4 < row1; ++y4) {
    const float* rows[3];
    for (int ky = 0; ky < 3; ++ky) {
      const int y2 = 2 * y4 + ky - 1;
      rows[ky] = y2 < 0 || y2 >= h2 ? zero_row : stem_row(y2);
    }
    for (int x4 = 0; x4 < w4; x4 += 4) {
      const int np = w4 - x4 < 4 ? w4 - x4 : 4;
      for (int p = 0; p < np; ++p) {  // the pixel's 144 inputs: tap by tap, 16 channels each (ring pixel 2 x4 + kx)
        float* cp = col + static_cast<size_t>(p) * kK;
        for (int ky = 0; ky < 3; ++ky) {
          for (int kx = 0; kx < 3; ++kx) {
            memcpy(cp + (ky * 3 + kx) * kC1, rows[ky] + static_cast<size_t>(2 * (x4 + p) + kx) * kC1,
                        sizeof(float) * kC1);
          }
        }
      }
      const float* cols[4];
      for (int p = 0; p < 4; ++p) cols[p] = col + static_cast<size_t>(p < np ? p : 0) * kK;
      switch (nv) {  // the accumulators in registers: the vector count a constant
        case 1: stem_tile<1>(cols, w2p, sw.b2, tile); break;
        case 2: stem_tile<2>(cols, w2p, sw.b2, tile); break;
        default: stem_tile24(cols, w2p, sw.b2, tile); break;
      }
      for (int d = 0; d < c2; ++d) {
        float* o = out + (static_cast<size_t>(d) * h4 + y4) * w4 + x4;
        for (int p = 0; p < np; ++p) o[p] = tile[p][d];
      }
    }
  }
}

// The dot products: 8 pairs in the lanes, each lane's chain an FMA a k from 0 (the rows by vgatherdps, the
// indices kept under 2^31 by the caller's check), the pairs past a multiple of 8 one chain each by vfmadd231ss.
void dot_chains_avx2(const float* a, const float* b, const int32_t* ia, const int32_t* ib, int pairs, int d,
                     float* out) {
  int p = 0;
  const __m256i dv = _mm256_set1_epi32(d), one = _mm256_set1_epi32(1);
  for (; p + 8 <= pairs; p += 8) {
    __m256i xa = _mm256_mullo_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ia + p)), dv);
    __m256i xb = _mm256_mullo_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(ib + p)), dv);
    __m256 acc = _mm256_setzero_ps();
    for (int k = 0; k < d; ++k) {
      acc = _mm256_fmadd_ps(_mm256_i32gather_ps(a, xa, 4), _mm256_i32gather_ps(b, xb, 4), acc);
      xa = _mm256_add_epi32(xa, one);
      xb = _mm256_add_epi32(xb, one);
    }
    _mm256_storeu_ps(out + p, acc);
  }
  for (; p < pairs; ++p) {
    const float* x = a + static_cast<int64_t>(ia[p]) * d;
    const float* y = b + static_cast<int64_t>(ib[p]) * d;
    __m128 s = _mm_setzero_ps();
    for (int k = 0; k < d; ++k) s = _mm_fmadd_ss(_mm_set_ss(x[k]), _mm_set_ss(y[k]), s);
    out[p] = _mm_cvtss_f32(s);
  }
}

// The GELU: erf_of on 8 lanes, the same operations in the same order (vminps/vmaxps, vmulps and vaddps apart,
// one vdivps).
void gelu_avx2(float* v, size_t n) {
  const __m256 four = _mm256_set1_ps(4.0f), mfour = _mm256_set1_ps(-4.0f), half = _mm256_set1_ps(0.5f),
               one = _mm256_set1_ps(1.0f), r2 = _mm256_set1_ps(0.70710678118654752f);
  auto madd = [](__m256 a, __m256 b, float c) { return _mm256_add_ps(_mm256_mul_ps(a, b), _mm256_set1_ps(c)); };
  alignas(32) float pad[8];
  for (size_t i = 0; i < n; i += 8) {
    const size_t m = n - i < 8 ? n - i : 8;  // a tail through the same lanes, from a copy (no inline erf_of here)
    float* at = v + i;
    if (m < 8) {
      for (size_t k = 0; k < 8; ++k) pad[k] = k < m ? at[k] : 0.0f;
      at = pad;
    }
    const __m256 x0 = _mm256_loadu_ps(at);
    __m256 x = _mm256_mul_ps(x0, r2);
    x = _mm256_max_ps(_mm256_min_ps(x, four), mfour);
    const __m256 x2 = _mm256_mul_ps(x, x);
    __m256 p = madd(x2, _mm256_set1_ps(kErfA13), kErfA11);
    p = madd(x2, p, kErfA9);
    p = madd(x2, p, kErfA7);
    p = madd(x2, p, kErfA5);
    p = madd(x2, p, kErfA3);
    p = madd(x2, p, kErfA1);
    p = _mm256_mul_ps(x, p);
    __m256 q = madd(x2, _mm256_set1_ps(kErfB8), kErfB6);
    q = madd(x2, q, kErfB4);
    q = madd(x2, q, kErfB2);
    q = madd(x2, q, kErfB0);
    const __m256 erf = _mm256_div_ps(p, q);
    _mm256_storeu_ps(at, _mm256_mul_ps(_mm256_mul_ps(half, x0), _mm256_add_ps(one, erf)));
    if (m < 8) {
      for (size_t k = 0; k < m; ++k) v[i + k] = pad[k];
    }
  }
}

// The tile: 8 frames in the lanes, kHeadBlock (10) classes a block in 10 accumulators; per channel j one load of
// the 8 frames' feature, 10 broadcasts of the block's weights (contiguous: packed [block][k][10]) and 10
// vfmadd231ps, from the biases, j in order: the reference's FMA chains. 13 ymm of 16. FMA-bound.
void head_tile_avx2(const float* features, int k, int frames, int t0, const float* w, const float* b, int classes,
                    float* tile) {
  const int blocks = (classes + 9) / 10;
  for (int cb = 0; cb < blocks; ++cb) {
    const int c0 = cb * 10, nc = classes - c0 < 10 ? classes - c0 : 10;  // no std::min: no template here
    __m256 acc[10];
    for (int q = 0; q < 10; ++q) acc[q] = _mm256_set1_ps(q < nc ? b[c0 + q] : 0.0f);
    const float* wb = w + static_cast<size_t>(cb) * k * 10;
    const float* f = features + t0;
    for (int j = 0; j < k; ++j) {
      const __m256 fv = _mm256_loadu_ps(f + static_cast<size_t>(j) * frames);
      const float* wj = wb + static_cast<size_t>(j) * 10;
      for (int q = 0; q < 10; ++q) acc[q] = _mm256_fmadd_ps(_mm256_broadcast_ss(wj + q), fv, acc[q]);
    }
    for (int q = 0; q < nc; ++q) _mm256_storeu_ps(tile + static_cast<size_t>(c0 + q) * 8, acc[q]);
  }
}

// 8 pixels a step: the BGRx bytes widened to 16 bits (vpunpck*bw), each pixel's (b, g, r, x) by (114, 587, 299, 0)
// in two vpmaddwd sums, joined by vphaddd (which keeps the pixels in order, 0-3 in the low lane, 4-7 in the high);
// + 500, and the division by 1000 exact as ((s >> 3) * 33555) >> 22 (vpsrld, vpmulld, vpsrld); packed to bytes.
void bgrx_to_gray_avx2(const uint8_t* src, int64_t src_stride, int w, int h, uint8_t* dst, int64_t dst_stride) {
  const __m256i weights = _mm256_setr_epi16(114, 587, 299, 0, 114, 587, 299, 0, 114, 587, 299, 0, 114, 587, 299, 0);
  const __m256i zero = _mm256_setzero_si256(), half = _mm256_set1_epi32(500), magic = _mm256_set1_epi32(33555);
  for (int y = 0; y < h; ++y) {
    const uint8_t* row = src + y * src_stride;
    uint8_t* out = dst + y * dst_stride;
    int x = 0;
    for (; x + 8 <= w; x += 8) {
      const __m256i px = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + 4 * x));
      const __m256i lo = _mm256_madd_epi16(_mm256_unpacklo_epi8(px, zero), weights);
      const __m256i hi = _mm256_madd_epi16(_mm256_unpackhi_epi8(px, zero), weights);
      const __m256i s = _mm256_add_epi32(_mm256_hadd_epi32(lo, hi), half);
      const __m256i q = _mm256_srli_epi32(_mm256_mullo_epi32(_mm256_srli_epi32(s, 3), magic), 22);
      const __m256i b = _mm256_packus_epi16(_mm256_packus_epi32(q, zero), zero);
      const uint32_t first = static_cast<uint32_t>(_mm_cvtsi128_si32(_mm256_castsi256_si128(b)));
      const uint32_t second = static_cast<uint32_t>(_mm_cvtsi128_si32(_mm256_extracti128_si256(b, 1)));
      std::memcpy(out + x, &first, 4);
      std::memcpy(out + x + 4, &second, 4);
    }
    for (; x < w; ++x) {
      const int bl = row[4 * x], g = row[4 * x + 1], r = row[4 * x + 2];
      out[x] = static_cast<uint8_t>((299 * r + 587 * g + 114 * bl + 500) / 1000);
    }
  }
}

void ctc_argmax_avx2(const float* logits, int classes, int frames, int t_n, int32_t* ids, float* probs) {
  // In chunks of up to 256 frames, the logits read in their order, a class's frames after another's:
  //  - each frame's best class, 8 frames a vector, the chunk's running bests and their classes kept in L1 (each
  //    frame's comparisons the reference's, in class order; the last frames' vector a masked load);
  //  - the softmax sums of only the frames where a character won (most frames are blank), gathered 8 to a vector, each
  //    lane its own frame's operations in the reference's order, so the same bits whichever lane a frame takes;
  //  - past the group's highest best class every lane's sum is at least 1 (its best's term, exp(0), is exactly 1), so a
  //    term below 2^-24 cannot change it: a class whose 8 terms are all of logits more than 17 below their bests
  //    (exp_neg(-17) is 4.1e-8; tests/cpp/cpu_test.cpp checks every float below) adds nothing and is passed over.
  constexpr int kChunk = 256;
  alignas(32) float top[kChunk + 8];
  alignas(32) int32_t arg[kChunk + 8];
  alignas(32) int32_t hit[kChunk + 8];
  const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  const __m256 far = _mm256_set1_ps(-17.0f);
  for (int c0 = 0; c0 < t_n; c0 += kChunk) {
    const int len = t_n - c0 < kChunk ? t_n - c0 : kChunk;
    const int nv = len / 8, rem = len % 8;
    const __m256i tail = _mm256_cmpgt_epi32(_mm256_set1_epi32(rem), lane);
    for (int v = 0; v <= nv; ++v) {
      _mm256_store_ps(top + 8 * v, _mm256_set1_ps(-INFINITY));
      _mm256_store_si256(reinterpret_cast<__m256i*>(arg + 8 * v), _mm256_setzero_si256());
    }
    for (int c = 0; c < classes; ++c) {
      const float* row = logits + static_cast<size_t>(c) * frames + c0;
      const __m256i cv = _mm256_set1_epi32(c);
      for (int v = 0; v < nv; ++v) {
        const __m256 x = _mm256_loadu_ps(row + 8 * v), b = _mm256_load_ps(top + 8 * v);
        const __m256 more = _mm256_cmp_ps(x, b, _CMP_GT_OQ);  // strictly: ties keep the lower id
        _mm256_store_ps(top + 8 * v, _mm256_blendv_ps(b, x, more));
        __m256i* a = reinterpret_cast<__m256i*>(arg + 8 * v);
        _mm256_store_si256(a, _mm256_blendv_epi8(_mm256_load_si256(a), cv, _mm256_castps_si256(more)));
      }
      if (rem != 0) {  // the lanes past the chunk read as 0 and are never used
        const __m256 x = _mm256_maskload_ps(row + 8 * nv, tail), b = _mm256_load_ps(top + 8 * nv);
        const __m256 more = _mm256_cmp_ps(x, b, _CMP_GT_OQ);
        _mm256_store_ps(top + 8 * nv, _mm256_blendv_ps(b, x, more));
        __m256i* a = reinterpret_cast<__m256i*>(arg + 8 * nv);
        _mm256_store_si256(a, _mm256_blendv_epi8(_mm256_load_si256(a), cv, _mm256_castps_si256(more)));
      }
    }
    std::memcpy(ids + c0, arg, static_cast<size_t>(len) * sizeof(int32_t));
    int n = 0;
    for (int t = 0; t < len; ++t) {
      hit[n] = t;
      n += arg[t] != 0;
    }
    if (n == 0) continue;  // a blank chunk: no probability is read
    for (int k = n; k < n + 8; ++k) hit[k] = hit[n - 1];  // a short last group repeats its last frame
    const float* chunk = logits + c0;
    for (int k = 0; k < n; k += 8) {
      const __m256i at = _mm256_load_si256(reinterpret_cast<const __m256i*>(hit + k));
      const __m256 best = _mm256_i32gather_ps(top, at, 4);
      int last = 0;
      for (int q = 0; q < 8; ++q) last = arg[hit[k + q]] > last ? arg[hit[k + q]] : last;
      __m256 sum = _mm256_setzero_ps();
      int c = 0;
      for (; c <= last; ++c) {
        const __m256 v = _mm256_i32gather_ps(chunk + static_cast<size_t>(c) * frames, at, 4);
        sum = _mm256_add_ps(sum, exp_neg8(_mm256_sub_ps(v, best)));
      }
      for (; c < classes; ++c) {
        const __m256 d = _mm256_sub_ps(_mm256_i32gather_ps(chunk + static_cast<size_t>(c) * frames, at, 4), best);
        if (_mm256_movemask_ps(_mm256_cmp_ps(d, far, _CMP_NLT_UQ)) == 0) continue;  // NaN is added, as the reference
        sum = _mm256_add_ps(sum, exp_neg8(d));
      }
      alignas(32) float prob[8];
      _mm256_store_ps(prob, _mm256_div_ps(_mm256_set1_ps(1.0f), sum));
      const int m = n - k < 8 ? n - k : 8;
      for (int q = 0; q < m; ++q) probs[c0 + hit[k + q]] = prob[q];
    }
  }
}

// ---------------------------------------------------------------- resampling
//
// The weights are 22-bit fixed point (up to 2^22), too wide for vpmaddwd, so each is
// split into two 16-bit halves, k = 2048 * hi + lo; the pixel times each half is
// exact, and (sum of the hi products << 11) + (sum of the lo products) is the sum of
// the whole products, below 2^31 (255 times the weights' sum, about 2^22).

void resample_hk_avx2(const int32_t* k, const int32_t* count, int taps, int width, int chunks, void* out) {
  auto* o = static_cast<int16_t*>(out);
  for (int x = 0; x < width; ++x) {
    // sample x is sample q of its group's pair q (the low lane) or q + 4 (the high one)
    const int g = x / 8, q = x % 4, lane = x % 8 / 4;
    for (int t = 0; t < count[x]; ++t) {
      int16_t* at = o + ((static_cast<size_t>(g) * chunks + t / 8) * 4 + q) * 32 + lane * 8 + t % 8;
      const int32_t v = k[static_cast<size_t>(x) * taps + t];
      at[0] = static_cast<int16_t>(v & 2047);
      at[16] = static_cast<int16_t>(v >> 11);
    }
  }
}

namespace {

// Samples q and q + 4 of a group, which share a register, 8 taps of each in 16 bits,
// one in each 128-bit lane: vpmaddwd leaves 4 sums of pairs a sample, (hi << 11) +
// lo.  a and b: their first pixels; kq: their weights, a chunk every 128 values.
// kChunks: the chunks known when compiled (0: `chunks`).
template <int kChunks>
inline __m256i h_pair(const uint8_t* a, const uint8_t* b, const int16_t* kq, int chunks) {
  const int n = kChunks ? kChunks : chunks;
  __m256i lo = _mm256_setzero_si256(), hi = _mm256_setzero_si256();
  for (int c = 0; c < n; ++c, a += 8, b += 8, kq += 128) {
    const __m256i v = _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(a)),
                                                              _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b))));
    const __m256i l = _mm256_madd_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(kq)));
    const __m256i h = _mm256_madd_epi16(v, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(kq + 16)));
    lo = c ? _mm256_add_epi32(lo, l) : l;
    hi = c ? _mm256_add_epi32(hi, h) : h;
  }
  return _mm256_add_epi32(_mm256_slli_epi32(hi, 11), lo);
}

// Groups [g0, g1): three vphaddd make each sample's 4 sums one, in order.
template <int kChunks>
void h_groups(const uint8_t* src, int col0, const int32_t* first, const int16_t* k, int chunks, int g0, int g1,
              uint8_t* out) {
  const __m256i half = _mm256_set1_epi32(carbon::detail::kHalf);
  for (int g = g0; g < g1; ++g) {
    const int32_t* f = first + 8 * g;
    const int16_t* kg = k + static_cast<size_t>(g) * chunks * 128;
    const __m256i s0 = h_pair<kChunks>(src + (f[0] - col0), src + (f[4] - col0), kg, chunks);
    const __m256i s1 = h_pair<kChunks>(src + (f[1] - col0), src + (f[5] - col0), kg + 32, chunks);
    const __m256i s2 = h_pair<kChunks>(src + (f[2] - col0), src + (f[6] - col0), kg + 64, chunks);
    const __m256i s3 = h_pair<kChunks>(src + (f[3] - col0), src + (f[7] - col0), kg + 96, chunks);
    __m256i sum = _mm256_hadd_epi32(_mm256_hadd_epi32(s0, s1), _mm256_hadd_epi32(s2, s3));
    sum = _mm256_srai_epi32(_mm256_add_epi32(sum, half), carbon::detail::kPrecisionBits);
    // clip8: the shifted sums fit 16 bits, and packus clamps them to 0 .. 255
    const __m128i w = _mm_packs_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    _mm_storel_epi64(reinterpret_cast<__m128i*>(out + 8 * g), _mm_packus_epi16(w, w));
  }
}

}  // namespace

void resample_h_avx2(const uint8_t* src, int col0, const int32_t* first, const void* k, int chunks, int g0, int g1,
                     uint8_t* out) {
  const auto* kk = static_cast<const int16_t*>(k);
  switch (chunks) {
    case 1: h_groups<1>(src, col0, first, kk, chunks, g0, g1, out); break;
    case 2: h_groups<2>(src, col0, first, kk, chunks, g0, g1, out); break;
    default: h_groups<0>(src, col0, first, kk, chunks, g0, g1, out); break;
  }
}

namespace {

inline __m256i load8_u32(const uint8_t* p) {
  return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
}

}  // namespace

// 32 pixels of the row at a time, in four registers of 8 int32 sums: vpmulld is
// exact (255 * 2^22 < 2^31).
void resample_v_avx2(const uint8_t* const* rows, const int32_t* k, int n, int width, uint8_t* out) {
  const __m256i half = _mm256_set1_epi32(carbon::detail::kHalf);
  const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);  // undoes the packs' lane interleave
  constexpr int bits = carbon::detail::kPrecisionBits;
  for (int x = 0; x < width; x += 32) {
    __m256i a0 = half, a1 = half, a2 = half, a3 = half;
    for (int i = 0; i < n; ++i) {
      const uint8_t* r = rows[i] + x;
      const __m256i w = _mm256_set1_epi32(k[i]);
      a0 = _mm256_add_epi32(a0, _mm256_mullo_epi32(load8_u32(r), w));
      a1 = _mm256_add_epi32(a1, _mm256_mullo_epi32(load8_u32(r + 8), w));
      a2 = _mm256_add_epi32(a2, _mm256_mullo_epi32(load8_u32(r + 16), w));
      a3 = _mm256_add_epi32(a3, _mm256_mullo_epi32(load8_u32(r + 24), w));
    }
    const __m256i lo = _mm256_packs_epi32(_mm256_srai_epi32(a0, bits), _mm256_srai_epi32(a1, bits));
    const __m256i hi = _mm256_packs_epi32(_mm256_srai_epi32(a2, bits), _mm256_srai_epi32(a3, bits));
    const __m256i b = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(lo, hi), order);
    if (x + 32 <= width) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + x), b);
    } else {
      alignas(32) uint8_t last[32];
      _mm256_store_si256(reinterpret_cast<__m256i*>(last), b);
      for (int j = 0; j < width - x; ++j) out[x + j] = last[j];
    }
  }
}

// ---------------------------------------------------------------- page turns

namespace {

// The double just below 0.5: for 0 <= v < 256, v plus it truncated is v rounded
// half away from zero (lround), since the sum rounds up to the next integer only
// when v's fraction is at least one half.
constexpr double kBelowHalf = 0.49999999999999994;

// Lanes 0-3 (k = 0) or 4-7 of eight int32 as doubles.
inline __m256d lanes_pd(__m256i v, int k) {
  return _mm256_cvtepi32_pd(k ? _mm256_extracti128_si256(v, 1) : _mm256_castsi256_si128(v));
}

// a <= b for each lane as unsigned int32: a negative a is past any b.
inline __m256i at_most(__m256i a, __m256i b) { return _mm256_cmpeq_epi32(_mm256_max_epu32(a, b), b); }

}  // namespace

// Eight pixels at a time: where they were on the page in doubles (vroundpd for the
// floor), then, when all their taps are on the page, the taps by two gathers of
// four bytes (a row's two taps and the two past them; x0 <= w - 4 keeps those on
// the row), the blend in doubles as the reference orders it, and rounding by
// kBelowHalf and truncation.  Pixels with a tap off the page, or x0 past w - 4,
// go to turn_pixels, unless none of their taps is on it.
void rotate_bilinear_avx2(const TurnView& t, uint8_t* dst) {
  const int w = t.w, h = t.h;
  // w - 4 and h - 2 not negative, and every tap's offset an int32 (the gathers')
  const bool wide = w >= 8 && h >= 2 && t.stride <= INT32_MAX && (h - 1) * t.stride + w <= INT32_MAX;
  const __m256d cx = _mm256_set1_pd(t.cx), cy = _mm256_set1_pd(t.cy), one = _mm256_set1_pd(1.0);
  const __m256d lo = _mm256_setzero_pd(), hi = _mm256_set1_pd(255.0), half = _mm256_set1_pd(kBelowHalf);
  const __m256i x_last = _mm256_set1_epi32(w - 4), y_last = _mm256_set1_epi32(h - 2);
  const __m256i x_seen = _mm256_set1_epi32(w), y_seen = _mm256_set1_epi32(h), one_i = _mm256_set1_epi32(1);
  const __m256i stride = _mm256_set1_epi32(wide ? static_cast<int32_t>(t.stride) : 0);
  const __m256i byte = _mm256_set1_epi32(0xff);
  const int* top_row = reinterpret_cast<const int*>(t.src);
  const int* next_row = reinterpret_cast<const int*>(t.src + (wide ? t.stride : 0));
  for (int y = t.row0; y < t.row1; ++y) {
    const double dy = y - t.cy, ds = dy * t.s, dc = dy * t.c;
    const __m256d vds = _mm256_set1_pd(ds), vdc = _mm256_set1_pd(dc);
    uint8_t* d = dst + static_cast<int64_t>(y) * w;
    int x = 0;
    for (; wide && x + 8 <= w; x += 8) {
      __m256d fx[2], fy[2];
      __m128i ix[2], iy[2];
      for (int k = 0; k < 2; ++k) {
        const __m256d sx = _mm256_add_pd(_mm256_sub_pd(_mm256_loadu_pd(t.xc + x + 4 * k), vds), cx);
        const __m256d sy = _mm256_add_pd(_mm256_add_pd(_mm256_loadu_pd(t.xs + x + 4 * k), vdc), cy);
        const __m256d x0 = _mm256_floor_pd(sx), y0 = _mm256_floor_pd(sy);
        fx[k] = _mm256_sub_pd(sx, x0);
        fy[k] = _mm256_sub_pd(sy, y0);
        ix[k] = _mm256_cvttpd_epi32(x0);
        iy[k] = _mm256_cvttpd_epi32(y0);
      }
      const __m256i x0 = _mm256_inserti128_si256(_mm256_castsi128_si256(ix[0]), ix[1], 1);
      const __m256i y0 = _mm256_inserti128_si256(_mm256_castsi128_si256(iy[0]), iy[1], 1);
      const __m256i inside = _mm256_and_si256(at_most(x0, x_last), at_most(y0, y_last));
      if (_mm256_movemask_ps(_mm256_castsi256_ps(inside)) != 0xff) {
        // a tap off the page: x0 + 1 in [0, w] and y0 + 1 in [0, h] when any is on it
        const __m256i seen = _mm256_and_si256(at_most(_mm256_add_epi32(x0, one_i), x_seen),
                                              at_most(_mm256_add_epi32(y0, one_i), y_seen));
        if (_mm256_movemask_ps(_mm256_castsi256_ps(seen))) turn_pixels(t, ds, dc, x, x + 8, d);
        continue;
      }
      const __m256i at = _mm256_add_epi32(_mm256_mullo_epi32(y0, stride), x0);
      const __m256i top = _mm256_i32gather_epi32(top_row, at, 1), bottom = _mm256_i32gather_epi32(next_row, at, 1);
      const __m256i p00 = _mm256_and_si256(top, byte), p10 = _mm256_and_si256(_mm256_srli_epi32(top, 8), byte);
      const __m256i p01 = _mm256_and_si256(bottom, byte), p11 = _mm256_and_si256(_mm256_srli_epi32(bottom, 8), byte);
      __m128i v[2];
      for (int k = 0; k < 2; ++k) {
        const __m256d gx = _mm256_sub_pd(one, fx[k]), gy = _mm256_sub_pd(one, fy[k]);
        const __m256d upper =
            _mm256_add_pd(_mm256_mul_pd(lanes_pd(p00, k), gx), _mm256_mul_pd(lanes_pd(p10, k), fx[k]));
        const __m256d lower =
            _mm256_add_pd(_mm256_mul_pd(lanes_pd(p01, k), gx), _mm256_mul_pd(lanes_pd(p11, k), fx[k]));
        const __m256d blend = _mm256_add_pd(_mm256_mul_pd(upper, gy), _mm256_mul_pd(lower, fy[k]));
        v[k] = _mm256_cvttpd_epi32(_mm256_add_pd(_mm256_max_pd(_mm256_min_pd(blend, hi), lo), half));
      }
      const __m128i words = _mm_packs_epi32(v[0], v[1]);
      _mm_storel_epi64(reinterpret_cast<__m128i*>(d + x), _mm_packus_epi16(words, words));
    }
    turn_pixels(t, ds, dc, x, w, d);
  }
}

namespace {

// 16 rows of 16 bytes transposed: column i of the rows (r[0] first) to o + i * os.
// Interleaving bytes, then pairs, fours and eights of them, as a 16 x 16 transpose
// is done in SSE2.
inline void transpose16(const uint8_t* const* r, uint8_t* o, int64_t os) {
  __m128i a[16], b[16];
  for (int i = 0; i < 16; ++i) a[i] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(r[i]));
  for (int i = 0; i < 8; ++i) {  // b[2i]: columns 0-7 of rows 2i and 2i + 1, b[2i + 1]: columns 8-15
    b[2 * i] = _mm_unpacklo_epi8(a[2 * i], a[2 * i + 1]);
    b[2 * i + 1] = _mm_unpackhi_epi8(a[2 * i], a[2 * i + 1]);
  }
  for (int i = 0; i < 4; ++i) {  // a[4i + k]: columns 4k to 4k + 3 of rows 4i to 4i + 3
    a[4 * i] = _mm_unpacklo_epi16(b[4 * i], b[4 * i + 2]);
    a[4 * i + 1] = _mm_unpackhi_epi16(b[4 * i], b[4 * i + 2]);
    a[4 * i + 2] = _mm_unpacklo_epi16(b[4 * i + 1], b[4 * i + 3]);
    a[4 * i + 3] = _mm_unpackhi_epi16(b[4 * i + 1], b[4 * i + 3]);
  }
  for (int i = 0; i < 2; ++i) {  // b[8i + m]: columns 2m and 2m + 1 of rows 8i to 8i + 7
    for (int k = 0; k < 4; ++k) {
      b[8 * i + 2 * k] = _mm_unpacklo_epi32(a[8 * i + k], a[8 * i + 4 + k]);
      b[8 * i + 2 * k + 1] = _mm_unpackhi_epi32(a[8 * i + k], a[8 * i + 4 + k]);
    }
  }
  for (int m = 0; m < 8; ++m) {  // columns 2m and 2m + 1, all 16 rows
    _mm_storeu_si128(reinterpret_cast<__m128i*>(o + 2 * m * os), _mm_unpacklo_epi64(b[m], b[8 + m]));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(o + (2 * m + 1) * os), _mm_unpackhi_epi64(b[m], b[8 + m]));
  }
}

// 16 rows of 32 bytes transposed, as two 16 x 16 tiles side by side in the halves
// of the registers (vpunpck* works within each half): the same steps as
// transpose16 for half the instructions.  Column i to o + i * os, i < 32.
inline void transpose32(const uint8_t* const* r, uint8_t* o, int64_t os) {
  __m256i a[16], b[16];
  for (int i = 0; i < 16; ++i) a[i] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r[i]));
  for (int i = 0; i < 8; ++i) {
    b[2 * i] = _mm256_unpacklo_epi8(a[2 * i], a[2 * i + 1]);
    b[2 * i + 1] = _mm256_unpackhi_epi8(a[2 * i], a[2 * i + 1]);
  }
  for (int i = 0; i < 4; ++i) {
    a[4 * i] = _mm256_unpacklo_epi16(b[4 * i], b[4 * i + 2]);
    a[4 * i + 1] = _mm256_unpackhi_epi16(b[4 * i], b[4 * i + 2]);
    a[4 * i + 2] = _mm256_unpacklo_epi16(b[4 * i + 1], b[4 * i + 3]);
    a[4 * i + 3] = _mm256_unpackhi_epi16(b[4 * i + 1], b[4 * i + 3]);
  }
  for (int i = 0; i < 2; ++i) {
    for (int k = 0; k < 4; ++k) {
      b[8 * i + 2 * k] = _mm256_unpacklo_epi32(a[8 * i + k], a[8 * i + 4 + k]);
      b[8 * i + 2 * k + 1] = _mm256_unpackhi_epi32(a[8 * i + k], a[8 * i + 4 + k]);
    }
  }
  for (int m = 0; m < 8; ++m) {  // columns 2m and 2m + 1 in the low halves, 16 + 2m and 17 + 2m in the high
    const __m256i even = _mm256_unpacklo_epi64(b[m], b[8 + m]), odd = _mm256_unpackhi_epi64(b[m], b[8 + m]);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(o + 2 * m * os), _mm256_castsi256_si128(even));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(o + (2 * m + 1) * os), _mm256_castsi256_si128(odd));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(o + (2 * m + 16) * os), _mm256_extracti128_si256(even, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(o + (2 * m + 17) * os), _mm256_extracti128_si256(odd, 1));
  }
}

}  // namespace

// q = 0: the rows copied; q = 2: each row reversed, 32 bytes at a time (vpshufb
// reverses each half, vpermq swaps them), into the row opposite; q odd: tiles
// of 16 rows transposed in registers, 32 columns at a time (then 16), in blocks
// of 64 columns by 256 rows (16 KB read, 16 KB written: they stay in L1, and each
// output row takes 256 bytes in a row), the bytes left over one at a time.
//
//   q = 1: (x, y) to row x, column h - 1 - y: a tile's rows taken bottom up
//   q = 3: (x, y) to row w - 1 - x, column y: its columns stored bottom up
void rotate_quarters_avx2(const uint8_t* src, int w, int h, int64_t stride, int q, uint8_t* dst) {
  if (q == 0) {
    for (int y = 0; y < h; ++y) std::memcpy(dst + static_cast<int64_t>(y) * w, src + y * stride, w);
    return;
  }
  if (q == 2) {
    const __m256i reverse = _mm256_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,  //
                                             15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    for (int y = 0; y < h; ++y) {
      const uint8_t* r = src + y * stride;
      uint8_t* o = dst + static_cast<int64_t>(h - 1 - y) * w;
      int x = 0;
      for (; x + 32 <= w; x += 32) {
        const __m256i v = _mm256_shuffle_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(r + x)), reverse);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + w - 32 - x), _mm256_permute4x64_epi64(v, 0x4e));
      }
      for (; x < w; ++x) o[w - 1 - x] = r[x];
    }
    return;
  }
  const int w32 = w / 32 * 32, w16 = w / 16 * 16, h16 = h / 16 * 16;
  const int64_t os = q == 1 ? h : -h;
  const uint8_t* rows[16];
  for (int by = 0; by < h16; by += 256) {
    const int by1 = min_of(h16, by + 256);
    for (int bx = 0; bx < w32; bx += 64) {
      const int bx1 = min_of(w32, bx + 64);
      for (int x = bx; x < bx1; x += 32) {
        for (int y = by; y < by1; y += 16) {
          uint8_t* o = quarter_tile(src, stride, w, h, q, x, y, dst, rows);
          transpose32(rows, o, os);
        }
      }
    }
    if (w16 > w32) {
      for (int y = by; y < by1; y += 16) {  // the next 16 columns
        uint8_t* o = quarter_tile(src, stride, w, h, q, w32, y, dst, rows);
        transpose16(rows, o, os);
      }
    }
  }
  quarter_bytes(src, stride, w, h, q, w16, 0, h16, dst);  // the columns past the tiles
  quarter_bytes(src, stride, w, h, q, 0, h16, h, dst);    // the rows below them
}

}  // namespace carbon::cpu::detail
