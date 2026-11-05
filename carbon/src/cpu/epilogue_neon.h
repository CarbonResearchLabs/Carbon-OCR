// The output stage on Arm64 NEON, 4 columns at a time (kernels.h), for neon.cpp,
// neondot.cpp and neonfp16.cpp.  FP16 residuals and outputs through fcvtl and fcvtn
// (rounding to nearest even: FPCR's default), for the FP16 kernels.
#pragma once

#include <arm_neon.h>

#include <cstring>
#include <type_traits>

#include "kernels.h"

namespace carbon::cpu::detail {
namespace {  // internal linkage: each file its own copy (kernels.h)

// A tile's output stage, as epilogue_avx2 (epilogue_avx2.h): the stage's fields
// read once, each column vector's scale and bias once for all the rows.
template <typename S>
inline void epilogue_neon(const S* s, int ld, int rows, int64_t i0, int j0, int w, const Epilogue& e) {
  const float* const scale = e.scale;
  const float* const bias = e.bias;
  const float* const res_f32 = e.residual_f32;
  const uint8_t* const res_u8 = e.residual_u8;
  const uint16_t* const res_f16 = e.residual_f16;
  const int64_t res_ld = e.residual_ld, out_ld = e.out_ld;
  const int rshift = e.residual_shift;
  const bool relu = e.relu;
  float* const out_f32 = e.out_f32;
  uint8_t* const out_u8 = e.out_u8;
  uint16_t* const out_f16 = e.out_f16;
  const float32x4_t lo = vdupq_n_f32(static_cast<float>(-e.out_zero));
  const float32x4_t hi = vdupq_n_f32(static_cast<float>(255 - e.out_zero));
  const float32x4_t inv = vdupq_n_f32(e.out_inv_scale), rscale = vdupq_n_f32(e.residual_scale);
  const int32x4_t ozero = vdupq_n_s32(e.out_zero), rzero = vdupq_n_s32(e.residual_zero);
  int t = 0;
  for (; t + 4 <= w; t += 4) {
    const int64_t j = j0 + t;
    const float32x4_t sc = scale ? vld1q_f32(scale + j) : vdupq_n_f32(0.0f);
    const float32x4_t bi = bias ? vld1q_f32(bias + j) : vdupq_n_f32(0.0f);
    for (int r = 0; r < rows; ++r) {
      const int64_t i = i0 + r;
      float32x4_t v;
      if constexpr (std::is_same_v<S, int32_t>) {
        v = vcvtq_f32_s32(vld1q_s32(s + r * ld + t));
      } else {
        v = vld1q_f32(s + r * ld + t);
      }
      if (scale) v = vmulq_f32(v, sc);
      if (bias) v = vaddq_f32(v, bi);
      if (res_f32) v = vaddq_f32(v, vld1q_f32(res_f32 + (i >> rshift) * res_ld + j));
      if (res_u8) {
        uint32_t r4;
        std::memcpy(&r4, res_u8 + (i >> rshift) * res_ld + j, 4);
        const uint16x4_t r16 = vget_low_u16(vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(r4))));
        const int32x4_t rv = vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(r16)), rzero);
        v = vaddq_f32(v, vmulq_f32(vcvtq_f32_s32(rv), rscale));
      }
      if (res_f16) v = vaddq_f32(v, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(res_f16 + (i >> rshift) * res_ld + j))));
      if (relu) v = vmaxq_f32(v, vdupq_n_f32(0.0f));
      if (out_u8) {
        const float32x4_t q = vminq_f32(vmaxq_f32(vmulq_f32(v, inv), lo), hi);
        const int32x4_t qi = vaddq_s32(vcvtnq_s32_f32(q), ozero);
        const uint8x8_t q8 = vqmovn_u16(vcombine_u16(vqmovun_s32(qi), vdup_n_u16(0)));
        const uint32_t out4 = vget_lane_u32(vreinterpret_u32_u8(q8), 0);
        std::memcpy(out_u8 + i * out_ld + j, &out4, 4);
      } else if (out_f16) {
        vst1_u16(out_f16 + i * out_ld + j, vreinterpret_u16_f16(vcvt_f16_f32(v)));
      } else {
        vst1q_f32(out_f32 + i * out_ld + j, v);
      }
    }
  }
  if (t == w) return;
  // The last columns, fewer than 4: the same operations on 4 lanes, copied in
  // and out (s has 4 values from t on).
  const int left = w - t;
  const int64_t j = j0 + t;
  float sc4[4] = {}, bi4[4] = {};
  if (scale) std::memcpy(sc4, scale + j, 4 * left);
  if (bias) std::memcpy(bi4, bias + j, 4 * left);
  const float32x4_t sc = vld1q_f32(sc4), bi = vld1q_f32(bi4);
  for (int r = 0; r < rows; ++r) {
    const int64_t i = i0 + r;
    float32x4_t v;
    if constexpr (std::is_same_v<S, int32_t>) {
      v = vcvtq_f32_s32(vld1q_s32(s + r * ld + t));
    } else {
      v = vld1q_f32(s + r * ld + t);
    }
    if (scale) v = vmulq_f32(v, sc);
    if (bias) v = vaddq_f32(v, bi);
    if (res_f32) {
      float r4[4] = {};
      std::memcpy(r4, res_f32 + (i >> rshift) * res_ld + j, 4 * left);
      v = vaddq_f32(v, vld1q_f32(r4));
    }
    if (res_u8) {
      uint32_t r4 = 0;
      std::memcpy(&r4, res_u8 + (i >> rshift) * res_ld + j, left);
      const uint16x4_t r16 = vget_low_u16(vmovl_u8(vreinterpret_u8_u32(vdup_n_u32(r4))));
      const int32x4_t rv = vsubq_s32(vreinterpretq_s32_u32(vmovl_u16(r16)), rzero);
      v = vaddq_f32(v, vmulq_f32(vcvtq_f32_s32(rv), rscale));
    }
    if (res_f16) {
      uint16_t r4[4] = {};
      std::memcpy(r4, res_f16 + (i >> rshift) * res_ld + j, 2 * left);
      v = vaddq_f32(v, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(r4))));
    }
    if (relu) v = vmaxq_f32(v, vdupq_n_f32(0.0f));
    if (out_u8) {
      const float32x4_t q = vminq_f32(vmaxq_f32(vmulq_f32(v, inv), lo), hi);
      const int32x4_t qi = vaddq_s32(vcvtnq_s32_f32(q), ozero);
      const uint8x8_t q8 = vqmovn_u16(vcombine_u16(vqmovun_s32(qi), vdup_n_u16(0)));
      const uint32_t out4 = vget_lane_u32(vreinterpret_u32_u8(q8), 0);
      std::memcpy(out_u8 + i * out_ld + j, &out4, left);
    } else if (out_f16) {
      uint16_t h4[4];
      vst1_u16(h4, vreinterpret_u16_f16(vcvt_f16_f32(v)));
      std::memcpy(out_f16 + i * out_ld + j, h4, 2 * left);
    } else {
      float o4[4];
      vst1q_f32(o4, v);
      std::memcpy(out_f32 + i * out_ld + j, o4, 4 * left);
    }
  }
}

}  // namespace
}  // namespace carbon::cpu::detail
