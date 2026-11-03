// PNG's rows on AVX2 (carbon/src/cpu/png_kernels.h).
//
// Up: 32 bytes an add.  Sub of 1, 2, 4 or 8 bytes a pixel: a prefix sum of 16 bytes in
// log steps (each byte plus the one a pixel before it, then two and four pixels before),
// then the last pixel of the bytes before added to all of them.
//
// Paeth, 16 rows at once: 16 rows of 1, 3 or 4 bytes a pixel of which at least 6 are Paeth
// or Average (the serial filters a row's scalar code makes slowly) go as a wavefront, row k
// a pixel behind row k - 1, the rows in the 16 byte lanes of a vector (a vector a channel);
// each lane applies its own row's filter (None 0, Sub a, Up b, Average (a + b) / 2, Paeth),
// chosen by masks made once for the 16 rows.  Each step makes a pixel of every row: its left (a) is the
// lane's last result, its above (b) the last result moved up a lane (lane 0 from the
// row above the run), its above left (c) the b of the step before.  The raw bytes come
// into lanes by 16-byte loads from each row, each a pixel further left than the row
// before, and a 16 x 16 transpose; the results go back the same way.  What the
// wavefront cannot reach with all 16 rows (a triangle of the first 15 pixels, and the
// ends) is scalar code.  Paeth on bytes, with pa = |b - c| and pb = |a - c|: pc =
// |(a - c) + (b - c)| is pa + pb when a - c and b - c have the same sign, so the choice
// is a if pa <= pb, else b; and |pa - pb| when not, so it is a if pa <= pb - pa, b if
// pb <= pa - pb, else c.  (Checked against stb_image's formula for all 2^24 a, b, c.)
//
// Average and the other Paeth rows are png_kernels.h's scalar code (Paeth rows of 1 byte
// a pixel in pairs).  RGB to gray: 8 pixels a 256-bit vector, r and b in 16-bit lanes
// for one vpmaddwd and g in another.
// Built with -mavx2 -mfma (/arch:AVX2).

#include <immintrin.h>

#include "png_kernels.h"

namespace carbon::cpu::detail {
namespace {

// Sub of B bytes a pixel (1, 2, 4 or 8), 16 bytes a step.
template <int B>
void sub(const uint8_t* raw, uint8_t* cur, size_t n) {
  __m128i carry = _mm_setzero_si128();  // the last pixel made, in every pixel's place
  size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(raw + i));
    x = _mm_add_epi8(x, _mm_slli_si128(x, B));
    if constexpr (B <= 4) x = _mm_add_epi8(x, _mm_slli_si128(x, 2 * B));
    if constexpr (B <= 2) x = _mm_add_epi8(x, _mm_slli_si128(x, 4 * B));
    if constexpr (B == 1) x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
    x = _mm_add_epi8(x, carry);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(cur + i), x);
    if constexpr (B == 1) {
      carry = _mm_shuffle_epi8(x, _mm_set1_epi8(15));
    } else if constexpr (B == 2) {
      carry = _mm_shuffle_epi8(x, _mm_set1_epi16(0x0f0e));
    } else if constexpr (B == 4) {
      carry = _mm_shuffle_epi32(x, 0xff);
    } else {
      carry = _mm_unpackhi_epi64(x, x);
    }
  }
  for (; i < n; ++i) cur[i] = static_cast<uint8_t>(raw[i] + (i >= B ? cur[i - B] : 0));
}

void up(const uint8_t* raw, const uint8_t* prior, uint8_t* cur, size_t n) {
  size_t i = 0;
  for (; i + 32 <= n; i += 32) {
    const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(raw + i));
    const __m256i p = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(prior + i));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(cur + i), _mm256_add_epi8(x, p));
  }
  for (; i < n; ++i) cur[i] = static_cast<uint8_t>(raw[i] + prior[i]);
}

// |x - y| of unsigned bytes
inline __m128i absdiff(__m128i x, __m128i y) { return _mm_or_si128(_mm_subs_epu8(x, y), _mm_subs_epu8(y, x)); }

// x <= y, unsigned bytes
inline __m128i le(__m128i x, __m128i y) { return _mm_cmpeq_epi8(_mm_min_epu8(x, y), x); }

// Paeth's predictor of 16 bytes (above).
inline __m128i paeth16(__m128i a, __m128i b, __m128i c) {
  const __m128i pa = absdiff(b, c), pb = absdiff(a, c);
  const __m128i same = _mm_cmpeq_epi8(le(a, c), le(b, c));
  const __m128i a_side = le(pa, pb);
  const __m128i a_near = le(pa, _mm_subs_epu8(pb, pa));  // 2 pa <= pb
  const __m128i b_near = le(pb, _mm_subs_epu8(pa, pb));  // 2 pb <= pa
  const __m128i take_a = _mm_and_si128(a_side, _mm_or_si128(same, a_near));
  const __m128i take_b = _mm_andnot_si128(a_side, _mm_or_si128(same, b_near));
  const __m128i ac = _mm_and_si128(take_a, _mm_xor_si128(a, c)), bc = _mm_and_si128(take_b, _mm_xor_si128(b, c));
  return _mm_xor_si128(c, _mm_or_si128(ac, bc));
}

// The 16 x 16 bytes of v transposed: four rounds of interleaving each vector with the one
// 8 on, each of which turns a byte's place (vector, byte: 8 bits) one bit to the left.
inline void transpose(__m128i v[16]) {
  for (int round = 0; round < 4; ++round) {
    __m128i t[16];
    for (int j = 0; j < 8; ++j) {
      t[2 * j] = _mm_unpacklo_epi8(v[j], v[j + 8]);
      t[2 * j + 1] = _mm_unpackhi_epi8(v[j], v[j + 8]);
    }
    for (int j = 0; j < 16; ++j) v[j] = t[j];
  }
}

// A row of B bytes a pixel from byte i0 (a whole pixel, at least one in) to n, by its filter.
template <int B>
void row_from(int filter, const uint8_t* raw, const uint8_t* prior, uint8_t* cur, size_t i0, size_t n) {
  switch (filter) {
    case 0:
      for (size_t i = i0; i < n; ++i) cur[i] = raw[i];
      return;
    case 1: png_sub_from<B>(raw, cur, i0, n); return;
    case 2:
      for (size_t i = i0; i < n; ++i) cur[i] = static_cast<uint8_t>(raw[i] + prior[i]);
      return;
    case 3: png_average_from<B>(raw, prior, cur, i0, n); return;
    default: png_paeth_from<B>(raw, prior, cur, i0, n); return;
  }
}

// 16 rows of B bytes a pixel (above), at least 32 pixels a row, each by its own filter.
template <int B>
void paeth_wave(const PngRows& r) {
  const size_t n = r.n, w = n / B;
  const uint8_t* raw[16];
  uint8_t* cur[16];
  const uint8_t* up[16];
  alignas(16) uint8_t m_sub[16], m_up[16], m_avg[16], m_paeth[16];
  int filter[16];
  for (int k = 0; k < 16; ++k) {
    raw[k] = r.raw + k * r.raw_stride;
    cur[k] = r.cur + k * r.cur_stride;
    up[k] = k ? cur[k - 1] : r.prior;
    filter[k] = raw[k][-1];
    m_sub[k] = filter[k] == 1 ? 0xff : 0;
    m_up[k] = filter[k] == 2 ? 0xff : 0;
    m_avg[k] = filter[k] == 3 ? 0xff : 0;
    m_paeth[k] = filter[k] == 4 ? 0xff : 0;
  }
  const __m128i is_sub = _mm_load_si128(reinterpret_cast<const __m128i*>(m_sub));
  const __m128i is_up = _mm_load_si128(reinterpret_cast<const __m128i*>(m_up));
  const __m128i is_avg = _mm_load_si128(reinterpret_cast<const __m128i*>(m_avg));
  const __m128i is_paeth = _mm_load_si128(reinterpret_cast<const __m128i*>(m_paeth));
  const __m128i one = _mm_set1_epi8(1);
  // The triangle the wavefront starts from: row k's first 15 - k pixels, each by its filter.
  for (int k = 0; k < 15; ++k) png_unfilter_row(filter[k], raw[k], up[k], cur[k], static_cast<size_t>(B) * (15 - k), B);
  // Lane k at step 15 makes row k's pixel 15 - k: its left is pixel 14 - k (zero for
  // row 15's first), the b of step 14 row k - 1's pixel 14 - k.
  __m128i a[B], bp[B];
  for (int ch = 0; ch < B; ++ch) {
    alignas(16) uint8_t av[16], bv[16];
    for (int k = 0; k < 16; ++k) {
      av[k] = k < 15 ? cur[k][B * (14 - k) + ch] : 0;
      bv[k] = k == 0 ? r.prior[B * 14 + ch] : k < 15 ? cur[k - 1][B * (14 - k) + ch] : 0;
    }
    a[ch] = _mm_load_si128(reinterpret_cast<const __m128i*>(av));
    bp[ch] = _mm_load_si128(reinterpret_cast<const __m128i*>(bv));
  }
  size_t s0 = 15;  // a group's first step: 16 steps, 16 B bytes of each row
  for (; s0 + 16 <= w; s0 += 16) {
    __m128i v[B][16];  // v[j][i]: byte 16 j + i of the group in each row's lane
    for (int j = 0; j < B; ++j) {
      for (int k = 0; k < 16; ++k) {
        v[j][k] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(raw[k] + B * (s0 - k) + 16 * j));
      }
      transpose(v[j]);
    }
    for (int i = 0; i < 16; ++i) {
      const uint8_t* above = r.prior + B * (s0 + i) - 15;  // byte 15 of a load: lane 0's above
      for (int ch = 0; ch < B; ++ch) {
        const int q = B * i + ch;
        const __m128i b = _mm_alignr_epi8(a[ch], _mm_loadu_si128(reinterpret_cast<const __m128i*>(above + ch)), 15);
        // Average's (a + b) >> 1: the rounding up avg less the odd bit of a + b
        const __m128i avg = _mm_sub_epi8(_mm_avg_epu8(a[ch], b), _mm_and_si128(_mm_xor_si128(a[ch], b), one));
        const __m128i pred = _mm_or_si128(
            _mm_or_si128(_mm_and_si128(is_sub, a[ch]), _mm_and_si128(is_up, b)),
            _mm_or_si128(_mm_and_si128(is_avg, avg), _mm_and_si128(is_paeth, paeth16(a[ch], b, bp[ch]))));
        const __m128i x = _mm_add_epi8(v[q / 16][q % 16], pred);
        v[q / 16][q % 16] = x;  // in the raw bytes' place
        bp[ch] = b;
        a[ch] = x;
      }
    }
    for (int j = 0; j < B; ++j) {
      transpose(v[j]);
      for (int k = 0; k < 16; ++k) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(cur[k] + B * (s0 - k) + 16 * j), v[j][k]);
      }
    }
  }
  // The rest of each row, from the first pixel the groups left.
  for (int k = 0; k < 16; ++k) row_from<B>(filter[k], raw[k], up[k], cur[k], B * (s0 - k), n);
}

// Of the 16 rows from the first, how many are Paeth or Average; 0 where there are fewer than 16.
size_t serial_rows(const PngRows& r) {
  if (r.rows < 16) return 0;
  size_t k = 0;
  for (size_t i = 0; i < 16; ++i) {
    const int f = (r.raw + i * r.raw_stride)[-1];
    k += f == 3 || f == 4;
  }
  return k;
}

}  // namespace

size_t png_unfilter_avx2(const PngRows& r) {
  const int filter = r.raw[-1], bpp = r.bpp;
  if ((filter == 3 || filter == 4) && (bpp == 1 || bpp == 3 || bpp == 4) && r.n / bpp >= 32 && serial_rows(r) >= 6) {
    if (bpp == 1) {
      paeth_wave<1>(r);
    } else if (bpp == 3) {
      paeth_wave<3>(r);
    } else {
      paeth_wave<4>(r);
    }
    return 16;
  }
  if (filter == 2) {
    up(r.raw, r.prior, r.cur, r.n);
    return 1;
  }
  if (filter == 1) {
    switch (bpp) {
      case 1: sub<1>(r.raw, r.cur, r.n); return 1;
      case 2: sub<2>(r.raw, r.cur, r.n); return 1;
      case 4: sub<4>(r.raw, r.cur, r.n); return 1;
      case 8: sub<8>(r.raw, r.cur, r.n); return 1;
      default: break;
    }
  }
  return png_unfilter_scalar_rows(r);
}

void png_gray_rgb8_avx2(const uint8_t* px, uint8_t* out, size_t w, int step) {
  const __m256i lo_bytes = _mm256_set1_epi32(0x00ff00ff);
  const __m256i w_rb = _mm256_set1_epi32(29 << 16 | 77), w_g = _mm256_set1_epi32(150);
  // RGB: pixels 0-3 from the first 12 of 16 bytes, 4-7 from the last 12 of 16 bytes 8 on.
  const __m256i spread = _mm256_setr_epi8(0, 1, 2, -1, 3, 4, 5, -1, 6, 7, 8, -1, 9, 10, 11, -1,  //
                                          4, 5, 6, -1, 7, 8, 9, -1, 10, 11, 12, -1, 13, 14, 15, -1);
  // 8 pixels as 32-bit r g b x, to their gray in 32-bit lanes
  auto gray8 = [&](__m256i v) {
    const __m256i rb = _mm256_and_si256(v, lo_bytes), gx = _mm256_and_si256(_mm256_srli_epi32(v, 8), lo_bytes);
    const __m256i s = _mm256_add_epi32(_mm256_madd_epi16(rb, w_rb), _mm256_madd_epi16(gx, w_g));
    return _mm256_srli_epi32(s, 8);
  };
  auto load8 = [&](const uint8_t* p) {
    if (step == 4) return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + 8));
    return _mm256_shuffle_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1), spread);
  };
  const size_t s = static_cast<size_t>(step);
  const __m256i order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
  size_t x = 0;
  for (; x + 32 <= w; x += 32) {
    const uint8_t* p = px + x * s;
    const __m256i a = gray8(load8(p)), b = gray8(load8(p + 8 * s));
    const __m256i c = gray8(load8(p + 16 * s)), d = gray8(load8(p + 24 * s));
    // packs work within 128-bit halves: a0-3 b0-3 c0-3 d0-3 | a4-7 b4-7 c4-7 d4-7, then in order
    const __m256i bytes = _mm256_packus_epi16(_mm256_packus_epi32(a, b), _mm256_packus_epi32(c, d));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + x), _mm256_permutevar8x32_epi32(bytes, order));
  }
  for (const uint8_t* p = px + x * s; x < w; ++x, p += s) {
    out[x] = static_cast<uint8_t>((p[0] * 77 + p[1] * 150 + p[2] * 29) >> 8);
  }
}

}  // namespace carbon::cpu::detail
