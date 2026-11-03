// The GEMMs and the depthwise convolution (carbon/cpu.h): packing, the scalar
// references, dispatch.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include "carbon/error.h"
#include "kernels.h"

namespace carbon::cpu {

// Here, built for the baseline, so the instruction sets' files make no vector code (kernels.h).
void* detail::scratch(Scratch slot, size_t bytes) {
  thread_local std::array<std::vector<std::byte>, 3> buffers;
  std::vector<std::byte>& b = buffers[static_cast<size_t>(slot)];
  if (b.size() < bytes + 64) b.resize(bytes + 64);
  const auto at = reinterpret_cast<uintptr_t>(b.data());
  return b.data() + ((64 - at % 64) % 64);  // a cache line's start
}

namespace {

// The kernel each instruction set runs: Arm without dotprod has no int8 kernel
// of its own, and AVX-VNNI and dotprod add nothing to FP32.
Isa int8_kernel(Isa isa) { return isa == Isa::kNeon ? Isa::kScalar : isa; }
Isa f32_kernel(Isa isa) {
  return isa == Isa::kAvxVnni ? Isa::kAvx2 : isa == Isa::kNeonDot ? Isa::kNeon : isa;
}

// The depthwise kernels: avx2 for every x86-64 set, neon for Arm.
Isa dw_kernel(Isa isa) {
  switch (isa) {
    case Isa::kAvx2:
    case Isa::kAvxVnni:
    case Isa::kAvx512: return Isa::kAvx2;
    case Isa::kNeon:
    case Isa::kNeonDot: return Isa::kNeon;
    default: return Isa::kScalar;
  }
}

int k_group(Isa isa) {
  isa = int8_kernel(isa);
  return isa == Isa::kScalar ? 1 : isa == Isa::kAvx2 ? 2 : 4;
}
int panel(Isa isa) { return isa == Isa::kScalar ? 1 : isa == Isa::kAvx512 ? 32 : 16; }
int round_up(int v, int to) { return (v + to - 1) / to * to; }

detail::BView view(const PackedB& b) {
  return {b.data.data(), b.col_sums.data(), b.k, b.n, b.taps, b.seg, b.segp, b.np};
}
detail::F32View view(const PackedF32& b) { return {b.data.data(), b.k, b.n, b.np}; }

void check(Isa isa, const char* what) {
  if (!isa_available(isa)) throw Error(std::string(what) + ": " + isa_name(isa) + " is not available on this CPU");
}

void gemm_u8s8_scalar(Rows<uint8_t> a, int32_t a_zero, const PackedB& b, int m, const Epilogue& e) {
  std::vector<int32_t> sum(b.n);
  for (int i = 0; i < m; ++i) {
    std::fill(sum.begin(), sum.end(), 0);
    for (int t = 0; t < a.taps; ++t) {
      const uint8_t* piece = a.piece(i, t);
      for (int kk = 0; kk < b.seg; ++kk) {
        const int32_t av = static_cast<int32_t>(piece[kk]) - a_zero;
        const int8_t* br = &b.data[(static_cast<size_t>(t) * b.seg + kk) * b.n];
        for (int j = 0; j < b.n; ++j) sum[j] += av * br[j];
      }
    }
    detail::epilogue_scalar(sum.data(), i, 0, b.n, e);
  }
}

void gemm_f32_scalar(Rows<float> a, const PackedF32& b, int m, const Epilogue& e) {
  const int seg = b.k / a.taps;
  std::vector<float> sum(b.n);
  for (int i = 0; i < m; ++i) {
    std::fill(sum.begin(), sum.end(), 0.0f);
    for (int t = 0; t < a.taps; ++t) {
      const float* piece = a.piece(i, t);
      for (int kk = 0; kk < seg; ++kk) {
        const float* br = &b.data[(static_cast<size_t>(t) * seg + kk) * b.n];
        for (int j = 0; j < b.n; ++j) sum[j] = std::fma(piece[kk], br[j], sum[j]);
      }
    }
    detail::epilogue_scalar(sum.data(), i, 0, b.n, e);
  }
}

void dwconv_f32_scalar(Rows<float> a, const float* w, int c, int m, const Epilogue& e) {
  std::vector<float> sum(c);
  for (int i = 0; i < m; ++i) {
    std::fill(sum.begin(), sum.end(), 0.0f);
    for (int t = 0; t < a.taps; ++t) {
      const float* x = a.piece(i, t);
      for (int j = 0; j < c; ++j) sum[j] = std::fma(x[j], w[static_cast<size_t>(t) * c + j], sum[j]);
    }
    detail::epilogue_scalar(sum.data(), i, 0, c, e);
  }
}

void dwconv_u8s8_scalar(Rows<uint8_t> a, int32_t a_zero, const int8_t* w, int c, int m, const Epilogue& e) {
  std::vector<int32_t> sum(c);
  for (int i = 0; i < m; ++i) {
    std::fill(sum.begin(), sum.end(), 0);
    for (int t = 0; t < a.taps; ++t) {
      const uint8_t* x = a.piece(i, t);
      for (int j = 0; j < c; ++j) sum[j] += (static_cast<int32_t>(x[j]) - a_zero) * w[static_cast<size_t>(t) * c + j];
    }
    detail::epilogue_scalar(sum.data(), i, 0, c, e);
  }
}

}  // namespace

int k_padded(int k, Isa isa) { return round_up(k, k_group(isa)); }

PackedB pack_b(const int8_t* b, int k, int n, Isa isa, int taps) {
  check(isa, "pack_b");
  if (taps < 1 || k % taps) throw Error("pack_b: k must be a whole number of taps");
  PackedB p;
  p.isa = isa;
  p.k = k;
  p.n = n;
  p.taps = taps;
  p.seg = k / taps;
  p.segp = k_padded(p.seg, isa);
  const Isa kern = int8_kernel(isa);
  p.np = round_up(n, panel(kern));
  p.col_sums.assign(n, 0);
  for (int kk = 0; kk < k; ++kk) {
    for (int j = 0; j < n; ++j) p.col_sums[j] += b[static_cast<size_t>(kk) * n + j];
  }
  if (kern == Isa::kScalar) {
    p.data.assign(b, b + static_cast<size_t>(k) * n);
    return p;
  }
  const int g = k_group(isa), w = panel(kern), groups = p.segp / g, panels = p.np / w;
  auto at = [&](int t, int kk, int j) -> int8_t {
    return kk < p.seg && j < n ? b[(static_cast<size_t>(t) * p.seg + kk) * n + j] : 0;
  };
  const size_t values = static_cast<size_t>(panels) * taps * groups * w * g;
  if (kern == Isa::kAvx2) {  // int16 pairs: [panel][tap][segp/2][16][2]
    p.data.assign(values * 2, 0);
    auto* out = reinterpret_cast<int16_t*>(p.data.data());
    for (int pj = 0; pj < panels; ++pj) {
      for (int t = 0; t < taps; ++t) {
        for (int kg = 0; kg < groups; ++kg) {
          for (int l = 0; l < w; ++l) {
            for (int u = 0; u < g; ++u) *out++ = at(t, kg * g + u, pj * w + l);
          }
        }
      }
    }
    return p;
  }
  p.data.assign(values, 0);  // [panel][tap][segp/4][w][4]
  int8_t* out = p.data.data();
  for (int pj = 0; pj < panels; ++pj) {
    for (int t = 0; t < taps; ++t) {
      for (int kg = 0; kg < groups; ++kg) {
        for (int l = 0; l < w; ++l) {
          for (int u = 0; u < g; ++u) *out++ = at(t, kg * g + u, pj * w + l);
        }
      }
    }
  }
  return p;
}

void gemm_u8s8(Rows<uint8_t> a, int32_t a_zero, const PackedB& b, int m, const Epilogue& out) {
  if (m <= 0) return;
  if (a.taps != b.taps) throw Error("gemm_u8s8: A and B have different taps");
  switch (int8_kernel(b.isa)) {
    case Isa::kScalar: gemm_u8s8_scalar(a, a_zero, b, m, out); return;
#if defined(CARBON_KERNEL_AVX2)
    case Isa::kAvx2: detail::gemm_u8s8_avx2(a, a_zero, view(b), m, out); return;
#endif
#if defined(CARBON_KERNEL_AVXVNNI)
    case Isa::kAvxVnni: detail::gemm_u8s8_avxvnni(a, a_zero, view(b), m, out); return;
#endif
#if defined(CARBON_KERNEL_AVX512)
    case Isa::kAvx512: detail::gemm_u8s8_avx512(a, a_zero, view(b), m, out); return;
#endif
#if defined(CARBON_KERNEL_NEONDOT)
    case Isa::kNeonDot: detail::gemm_u8s8_neondot(a, a_zero, view(b), m, out); return;
#endif
    default: throw Error(std::string("gemm_u8s8: no kernel for ") + isa_name(b.isa) + " in this build");
  }
}

PackedF32 pack_f32(const float* b, int k, int n, Isa isa) {
  check(isa, "pack_f32");
  PackedF32 p;
  p.isa = isa;
  p.k = k;
  p.n = n;
  const Isa kern = f32_kernel(isa);
  const int w = panel(kern);
  p.np = round_up(n, w);
  if (kern == Isa::kScalar) {
    p.data.assign(b, b + static_cast<size_t>(k) * n);
    return p;
  }
  p.data.assign(static_cast<size_t>(p.np) * k, 0.0f);  // [panel][k][w]
  float* out = p.data.data();
  for (int pj = 0; pj < p.np / w; ++pj) {
    for (int kk = 0; kk < k; ++kk) {
      for (int l = 0; l < w; ++l) {
        const int j = pj * w + l;
        *out++ = j < n ? b[static_cast<size_t>(kk) * n + j] : 0.0f;
      }
    }
  }
  return p;
}

void gemm_f32(Rows<float> a, const PackedF32& b, int m, const Epilogue& out) {
  if (m <= 0) return;
  if (a.taps < 1 || b.k % a.taps) throw Error("gemm_f32: k must be a whole number of taps");
  switch (f32_kernel(b.isa)) {
    case Isa::kScalar: gemm_f32_scalar(a, b, m, out); return;
#if defined(CARBON_KERNEL_AVX2)
    case Isa::kAvx2: detail::gemm_f32_avx2(a, view(b), m, out); return;
#endif
#if defined(CARBON_KERNEL_AVX512)
    case Isa::kAvx512: detail::gemm_f32_avx512(a, view(b), m, out); return;
#endif
#if defined(CARBON_KERNEL_NEON)
    case Isa::kNeon: detail::gemm_f32_neon(a, view(b), m, out); return;
#endif
    default: throw Error(std::string("gemm_f32: no kernel for ") + isa_name(b.isa) + " in this build");
  }
}

void dwconv_f32(Rows<float> a, const float* w, int c, int m, const Epilogue& out, Isa isa) {
  if (m <= 0) return;
  check(isa, "dwconv_f32");
  switch (dw_kernel(isa)) {
    case Isa::kScalar: dwconv_f32_scalar(a, w, c, m, out); return;
#if defined(CARBON_KERNEL_AVX2)
    case Isa::kAvx2: detail::dwconv_f32_avx2(a, w, c, m, out); return;
#endif
#if defined(CARBON_KERNEL_NEON)
    case Isa::kNeon: detail::dwconv_f32_neon(a, w, c, m, out); return;
#endif
    default: throw Error(std::string("dwconv_f32: no kernel for ") + isa_name(isa) + " in this build");
  }
}

void ctc_argmax(const float* logits, int classes, int frames, int t_n, int32_t* ids, float* probs, Isa isa) {
  if (t_n <= 0 || classes <= 0) return;
  check(isa, "ctc_argmax");
  switch (dw_kernel(isa)) {
#if defined(CARBON_KERNEL_AVX2)
    case Isa::kAvx2: detail::ctc_argmax_avx2(logits, classes, frames, t_n, ids, probs); return;
#endif
#if defined(CARBON_KERNEL_NEON)
    case Isa::kNeon: detail::ctc_argmax_neon(logits, classes, frames, t_n, ids, probs); return;
#endif
    default: break;
  }
  // The reference, a frame at a time.
  for (int t = 0; t < t_n; ++t) {
    float best = -std::numeric_limits<float>::infinity();
    int32_t arg = 0;
    for (int c = 0; c < classes; ++c) {
      const float v = logits[static_cast<size_t>(c) * frames + t];
      if (v > best) {
        best = v;
        arg = c;
      }
    }
    ids[t] = arg;
    if (arg == 0) continue;
    float sum = 0.0f;
    for (int c = 0; c < classes; ++c) sum = sum + detail::exp_neg(logits[static_cast<size_t>(c) * frames + t] - best);
    probs[t] = 1.0f / sum;
  }
}

void dwconv_u8s8(Rows<uint8_t> a, int32_t a_zero, const int8_t* w, int c, int m, const Epilogue& out, Isa isa) {
  if (m <= 0) return;
  check(isa, "dwconv_u8s8");
  switch (dw_kernel(isa)) {
    case Isa::kScalar: dwconv_u8s8_scalar(a, a_zero, w, c, m, out); return;
#if defined(CARBON_KERNEL_AVX2)
    case Isa::kAvx2: detail::dwconv_u8s8_avx2(a, a_zero, w, c, m, out); return;
#endif
#if defined(CARBON_KERNEL_NEON)
    case Isa::kNeon: detail::dwconv_u8s8_neon(a, a_zero, w, c, m, out); return;
#endif
    default: throw Error(std::string("dwconv_u8s8: no kernel for ") + isa_name(isa) + " in this build");
  }
}

}  // namespace carbon::cpu
