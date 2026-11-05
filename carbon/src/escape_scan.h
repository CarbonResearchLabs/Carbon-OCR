// The writers' escapes a run at a time: the first byte of a string, at or past a place, that JSON or XML writes
// otherwise than as itself. The run before it is copied whole.
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>

#if defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
#include <emmintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace carbon::detail {

// A control byte (under 0x20), or " ' \ & < >. 16 bytes a step: SSE2's pcmpeqb against each, a control byte where
// its unsigned minimum with 0x1F is itself (pminub), por, pmovmskb and the first set bit (tzcnt); NEON vceqq_u8 and
// vcleq_u8, the mask's first lane by vshrn_n_u16. The tail byte by byte. Every ISA finds the same byte.
inline size_t special_from(const std::string& s, size_t i) {
  const size_t n = s.size();
  const char* d = s.data();
#if defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
  const __m128i q = _mm_set1_epi8('"'), ap = _mm_set1_epi8('\''), bs = _mm_set1_epi8('\\'), amp = _mm_set1_epi8('&'),
                lt = _mm_set1_epi8('<'), gt = _mm_set1_epi8('>'), ctl = _mm_set1_epi8(0x1F);
  for (; i + 16 <= n; i += 16) {
    const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(d + i));
    __m128i m = _mm_or_si128(_mm_cmpeq_epi8(v, q), _mm_cmpeq_epi8(v, ap));
    m = _mm_or_si128(m, _mm_or_si128(_mm_cmpeq_epi8(v, bs), _mm_cmpeq_epi8(v, amp)));
    m = _mm_or_si128(m, _mm_or_si128(_mm_cmpeq_epi8(v, lt), _mm_cmpeq_epi8(v, gt)));
    m = _mm_or_si128(m, _mm_cmpeq_epi8(_mm_min_epu8(v, ctl), v));
    const int bits = _mm_movemask_epi8(m);
    if (bits) return i + static_cast<size_t>(std::countr_zero(static_cast<unsigned>(bits)));
  }
#elif defined(__ARM_NEON)
  for (; i + 16 <= n; i += 16) {
    const uint8x16_t v = vld1q_u8(reinterpret_cast<const uint8_t*>(d + i));
    uint8x16_t m = vorrq_u8(vceqq_u8(v, vdupq_n_u8('"')), vceqq_u8(v, vdupq_n_u8('\'')));
    m = vorrq_u8(m, vorrq_u8(vceqq_u8(v, vdupq_n_u8('\\')), vceqq_u8(v, vdupq_n_u8('&'))));
    m = vorrq_u8(m, vorrq_u8(vceqq_u8(v, vdupq_n_u8('<')), vceqq_u8(v, vdupq_n_u8('>'))));
    m = vorrq_u8(m, vcleq_u8(v, vdupq_n_u8(0x1F)));
    const uint64_t bits = vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(m), 4)), 0);
    if (bits) return i + static_cast<size_t>(std::countr_zero(bits) / 4);
  }
#endif
  for (; i < n; ++i) {
    const unsigned char c = static_cast<unsigned char>(d[i]);
    if (c < 0x20 || c == '"' || c == '\'' || c == '\\' || c == '&' || c == '<' || c == '>') return i;
  }
  return n;
}

}  // namespace carbon::detail
