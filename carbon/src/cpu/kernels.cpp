// The GEMMs, the depthwise convolutions, a threshold's bits, a crop's counts, CTC's
// argmax and the detector's head (carbon/cpu.h): packing, the scalar references, dispatch.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "carbon/error.h"
#include "kernels.h"

namespace carbon::cpu {

// Here, built for the baseline, so the instruction sets' files make no vector code (kernels.h).
namespace {

struct ScratchSlot {
  std::vector<std::byte> buffer;
  uint64_t tag = 0;  // what it holds (scratch_kept), 0: nothing to keep
};

ScratchSlot& slot_of(detail::Scratch slot, size_t bytes, bool& grew) {
  thread_local std::array<ScratchSlot, 4> slots;
  ScratchSlot& s = slots[static_cast<size_t>(slot)];
  grew = s.buffer.size() < bytes + 64;
  if (grew) s.buffer.resize(bytes + 64);
  return s;
}

void* line_start(std::vector<std::byte>& b) {
  const auto at = reinterpret_cast<uintptr_t>(b.data());
  return b.data() + ((64 - at % 64) % 64);
}

}  // namespace

void* detail::scratch(Scratch slot, size_t bytes) {
  bool grew;
  ScratchSlot& s = slot_of(slot, bytes, grew);
  s.tag = 0;
  return line_start(s.buffer);
}

void* detail::scratch_kept(Scratch slot, size_t bytes, uint64_t tag, bool& kept) {
  bool grew;
  ScratchSlot& s = slot_of(slot, bytes, grew);
  kept = tag != 0 && s.tag == tag && !grew;  // a tag's contents are the same size each time
  s.tag = tag;
  return line_start(s.buffer);
}

namespace {

// The kernel each instruction set runs: Arm without dotprod has no int8 kernel
// of its own, AVX-VNNI and dotprod add nothing to FP32, and FP16 arithmetic
// nothing to int8 and FP32.
Isa int8_kernel(Isa isa) {
  switch (isa) {
    case Isa::kNeon: return Isa::kScalar;
    case Isa::kAvx512Fp16: return Isa::kAvx512;
    case Isa::kNeonFp16: return Isa::kNeonDot;
    default: return isa;
  }
}
Isa f32_kernel(Isa isa) {
  switch (isa) {
    case Isa::kAvxVnni: return Isa::kAvx2;
    case Isa::kNeonDot:
    case Isa::kNeonFp16: return Isa::kNeon;
    case Isa::kAvx512Fp16: return Isa::kAvx512;
    default: return isa;
  }
}

// The depthwise kernels: avx2 for every x86-64 set, neon for Arm.
Isa dw_kernel(Isa isa) {
  switch (isa) {
    case Isa::kAvx2:
    case Isa::kAvxVnni:
    case Isa::kAvx512:
    case Isa::kAvx512Fp16: return Isa::kAvx2;
    case Isa::kNeon:
    case Isa::kNeonDot:
    case Isa::kNeonFp16: return Isa::kNeon;
    default: return Isa::kScalar;
  }
}

// The FP16 kernels.  On x86-64 the FP32 arithmetic ones widen FP16 with F16C (avx2)
// or AVX-512 F (avx512): F16C came before AVX2 on every CPU there is, but a
// virtual machine could hide it, so without it they are the reference's.  FP32 A
// on an FP16 arithmetic set runs on the same CPU's FP32 arithmetic (the `wide` kernels).
Isa f16_kernel(Isa isa) {
  switch (isa) {
    case Isa::kAvx2:
    case Isa::kAvxVnni: return features().f16c ? Isa::kAvx2 : Isa::kScalar;
    case Isa::kNeonDot: return Isa::kNeon;
    default: return isa;
  }
}
Isa f16_wide_kernel(Isa isa) {
  return isa == Isa::kAvx512Fp16 ? Isa::kAvx512 : isa == Isa::kNeonFp16 ? Isa::kNeon : f16_kernel(isa);
}
// The FP16 depthwise kernels: the FP16 arithmetic sets' own, else dwconv_f32's
// (avx2 for every x86-64 set, with F16C; neon for Arm).
Isa dw16_wide_kernel(Isa isa) {
  const Isa k = dw_kernel(isa);
  return k == Isa::kAvx2 && !features().f16c ? Isa::kScalar : k;
}
Isa dw16_kernel(Isa isa) {
  return isa == Isa::kAvx512Fp16 || isa == Isa::kNeonFp16 ? isa : dw16_wide_kernel(isa);
}

int k_group(Isa isa) {
  isa = int8_kernel(isa);
  return isa == Isa::kScalar ? 1 : isa == Isa::kAvx2 ? 2 : 4;
}
int panel(Isa isa) {
  return isa == Isa::kScalar ? 1 : isa == Isa::kAvx512 || isa == Isa::kAvx512Fp16 ? 32 : 16;
}
int round_up(int v, int to) { return (v + to - 1) / to * to; }

detail::BView view(const PackedB& b) {
  return {b.data.data(), b.col_sums.data(), b.k, b.n, b.taps, b.seg, b.segp, b.np};
}
detail::F32View view(const PackedF32& b) { return {b.data.data(), b.k, b.n, b.np}; }
detail::F16View view(const PackedF16& b) { return {b.data.data(), b.k, b.n, b.np, b.id}; }

// An activation as FP32: an FP16 one widened.
inline float value_of(float v) { return v; }
inline float value_of(uint16_t h) { return detail::half_to_float(h); }

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

// The FP16 references: every value widened to FP32, then gemm_f32's and
// dwconv_f32's sums.
template <typename A>
void gemm_f16_scalar(Rows<A> a, const PackedF16& b, int m, const Epilogue& e) {
  const int seg = b.k / a.taps;
  std::vector<float> sum(b.n), bf(b.data.size());
  for (size_t i = 0; i < bf.size(); ++i) bf[i] = detail::half_to_float(b.data[i]);
  for (int i = 0; i < m; ++i) {
    std::fill(sum.begin(), sum.end(), 0.0f);
    for (int t = 0; t < a.taps; ++t) {
      const A* piece = a.piece(i, t);
      for (int kk = 0; kk < seg; ++kk) {
        const float av = value_of(piece[kk]);
        const float* br = &bf[(static_cast<size_t>(t) * seg + kk) * b.n];
        for (int j = 0; j < b.n; ++j) sum[j] = std::fma(av, br[j], sum[j]);
      }
    }
    detail::epilogue_scalar(sum.data(), i, 0, b.n, e);
  }
}

template <typename A>
void dwconv_f16_scalar(Rows<A> a, const uint16_t* w, int c, int m, const Epilogue& e) {
  std::vector<float> sum(c), wf(static_cast<size_t>(a.taps) * c);
  for (size_t i = 0; i < wf.size(); ++i) wf[i] = detail::half_to_float(w[i]);
  for (int i = 0; i < m; ++i) {
    std::fill(sum.begin(), sum.end(), 0.0f);
    for (int t = 0; t < a.taps; ++t) {
      const A* x = a.piece(i, t);
      for (int j = 0; j < c; ++j) sum[j] = std::fma(value_of(x[j]), wf[static_cast<size_t>(t) * c + j], sum[j]);
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

void above_bits(const float* v, int n, float t, uint32_t* bits, Isa isa) {
  if (n <= 0) return;
  check(isa, "above_bits");
  switch (dw_kernel(isa)) {
#if defined(CARBON_KERNEL_AVX2)
    case Isa::kAvx2: detail::above_bits_avx2(v, n, t, bits); return;
#endif
#if defined(CARBON_KERNEL_NEON)
    case Isa::kNeon: detail::above_bits_neon(v, n, t, bits); return;
#endif
    default: break;
  }
  for (int w = 0; w < (n + 31) / 32; ++w) bits[w] = 0;  // the reference, a value at a time
  for (int i = 0; i < n; ++i) {
    if (v[i] > t) bits[i / 32] |= 1u << (i % 32);
  }
}

ByteCounts count_beyond(const uint8_t* c, int w, int h, int64_t stride, int lo, int hi, Isa isa) {
  ByteCounts n{0, 0};
  if (w <= 0 || h <= 0 || (lo < 0 && hi > 255)) return n;
  check(isa, "count_beyond");
  switch (dw_kernel(isa)) {
#if defined(CARBON_KERNEL_AVX2)
    case Isa::kAvx2: return detail::count_beyond_avx2(c, w, h, stride, lo, hi);
#endif
#if defined(CARBON_KERNEL_NEON)
    case Isa::kNeon: return detail::count_beyond_neon(c, w, h, stride, lo, hi);
#endif
    default: break;
  }
  for (int y = 0; y < h; ++y) {  // the reference, a byte at a time
    const uint8_t* row = c + y * stride;
    for (int x = 0; x < w; ++x) {
      n.below += row[x] <= lo;
      n.above += row[x] >= hi;
    }
  }
  return n;
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

namespace {

// The reference, as the kernels run it: the head at stride 4 a row at a time, upsampled into a padded row (a
// zero either side), three of them kept (slot p % 3 for the head's row p), and a row of zeros above and below.
void det_head_scalar(const float* x, int h4, int w4, const DetHeadWeights& w, float* maps, int row0, int row1) {
  const int H = 2 * h4, W = 2 * w4, P = W + 2, mid = w.mid;
  float* head = detail::scratch_of<float>(detail::Scratch::kA, static_cast<size_t>(mid) * (w4 + 4 * P));
  float* up = head + static_cast<size_t>(mid) * w4;
  float* zero = up + static_cast<size_t>(3) * mid * P;
  std::fill(zero, zero + static_cast<size_t>(mid) * P, 0.0f);
  int held[3] = {-1, -1, -1};
  const auto row_of = [&](int p) -> const float* {  // the head's row p, upsampled
    float* u = up + static_cast<size_t>(p % 3) * mid * P;
    if (held[p % 3] == p) return u;
    for (int c = 0; c < mid; ++c) {
      for (int px = 0; px < w4; ++px) {
        float acc = w.b1[c];
        for (int i = 0; i < w.in; ++i) {
          acc = std::fma(w.w1[static_cast<size_t>(c) * w.in + i], x[(static_cast<int64_t>(i) * h4 + p) * w4 + px], acc);
        }
        head[static_cast<size_t>(c) * w4 + px] = acc > 0.0f ? acc : 0.0f;
      }
      float* uc = u + static_cast<size_t>(c) * P;
      uc[0] = 0.0f;
      uc[P - 1] = 0.0f;
      for (int zx = 0; zx < W; ++zx) uc[1 + zx] = head[static_cast<size_t>(c) * w4 + (zx >> 1)];
    }
    held[p % 3] = p;
    return u;
  };
  for (int oy = row0; oy < row1; ++oy) {
    const float* z[3];
    for (int ky = 0; ky < 3; ++ky) {
      const int zy = oy + ky - 1;
      z[ky] = zy < 0 || zy >= H ? zero : row_of(zy >> 1);
    }
    for (int ox = 0; ox < W; ++ox) {
      float m[8];
      for (int o = 0; o < w.out; ++o) m[o] = w.b3[o];
      for (int c = 0; c < mid; ++c) {
        float acc = w.b2[c];
        for (int ky = 0; ky < 3; ++ky) {
          for (int kx = 0; kx < 3; ++kx) {
            acc = std::fma(w.w2[static_cast<size_t>(c) * 9 + ky * 3 + kx], z[ky][static_cast<size_t>(c) * P + ox + kx], acc);
          }
        }
        const float d = acc > 0.0f ? acc : 0.0f;
        for (int o = 0; o < w.out; ++o) m[o] = std::fma(w.w3[static_cast<size_t>(o) * mid + c], d, m[o]);
      }
      for (int o = 0; o < w.out; ++o) maps[(static_cast<int64_t>(o) * H + oy) * W + ox] = m[o];
    }
  }
}

}  // namespace

void det_head(const float* x, int h4, int w4, const DetHeadWeights& w, float* maps, int row0, int row1, Isa isa) {
  if (w.out < 1 || w.out > 8 || w.mid < 1 || w.in < 1) throw Error("det_head: 1 to 8 maps, and channels");
  row0 = std::max(row0, 0);
  row1 = std::min(row1, 2 * h4);
  if (h4 <= 0 || w4 <= 0 || row0 >= row1) return;
  check(isa, "det_head");
  switch (dw_kernel(isa)) {
#if defined(CARBON_KERNEL_AVX2)
    case Isa::kAvx2: detail::det_head_avx2(x, h4, w4, w, maps, row0, row1); return;
#endif
    default: det_head_scalar(x, h4, w4, w, maps, row0, row1); return;
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

// ---------------------------------------------------------------- FP16

float f16_to_f32(uint16_t h) { return detail::half_to_float(h); }
uint16_t f32_to_f16(float v) { return detail::float_to_half(v); }
bool f16_arithmetic(Isa isa) { return isa == Isa::kAvx512Fp16 || isa == Isa::kNeonFp16; }

void to_f16(const float* in, uint16_t* out, size_t n, Isa isa) {
  check(isa, "to_f16");
  switch (dw16_wide_kernel(isa)) {
#if defined(CARBON_KERNEL_AVX2)
    case Isa::kAvx2: detail::to_f16_avx2(in, out, n); return;
#endif
#if defined(CARBON_KERNEL_NEON)
    case Isa::kNeon: detail::to_f16_neon(in, out, n); return;
#endif
    default:
      for (size_t i = 0; i < n; ++i) out[i] = detail::float_to_half(in[i]);
  }
}

void to_f32(const uint16_t* in, float* out, size_t n, Isa isa) {
  check(isa, "to_f32");
  switch (dw16_wide_kernel(isa)) {
#if defined(CARBON_KERNEL_AVX2)
    case Isa::kAvx2: detail::to_f32_avx2(in, out, n); return;
#endif
#if defined(CARBON_KERNEL_NEON)
    case Isa::kNeon: detail::to_f32_neon(in, out, n); return;
#endif
    default:
      for (size_t i = 0; i < n; ++i) out[i] = detail::half_to_float(in[i]);
  }
}

PackedF16 pack_f16(const uint16_t* b, int k, int n, Isa isa) {
  check(isa, "pack_f16");
  static std::atomic<uint64_t> packings{0};
  PackedF16 p;
  p.isa = isa;
  p.k = k;
  p.n = n;
  p.id = ++packings;
  const int w = panel(f16_kernel(isa));  // f16_wide_kernel's is the same
  p.np = round_up(n, w);
  if (w == 1) {
    p.data.assign(b, b + static_cast<size_t>(k) * n);
    return p;
  }
  p.data.assign(static_cast<size_t>(p.np) * k, 0);  // [panel][k][w], zeros past n
  uint16_t* out = p.data.data();
  for (int pj = 0; pj < p.np / w; ++pj) {
    for (int kk = 0; kk < k; ++kk) {
      for (int l = 0; l < w; ++l) {
        const int j = pj * w + l;
        *out++ = j < n ? b[static_cast<size_t>(kk) * n + j] : uint16_t{0};
      }
    }
  }
  return p;
}

void gemm_f16(Rows<uint16_t> a, const PackedF16& b, int m, const Epilogue& out) {
  if (m <= 0) return;
  if (a.taps < 1 || b.k % a.taps) throw Error("gemm_f16: k must be a whole number of taps");
  switch (f16_kernel(b.isa)) {
    case Isa::kScalar: gemm_f16_scalar(a, b, m, out); return;
#if defined(CARBON_KERNEL_AVX2)
    case Isa::kAvx2: detail::gemm_f16_avx2(a, view(b), m, out); return;
#endif
#if defined(CARBON_KERNEL_AVX512)
    case Isa::kAvx512: detail::gemm_f16_avx512(a, view(b), m, out); return;
#endif
#if defined(CARBON_KERNEL_AVX512FP16)
    case Isa::kAvx512Fp16: detail::gemm_f16_avx512fp16(a, view(b), m, out); return;
#endif
#if defined(CARBON_KERNEL_NEON)
    case Isa::kNeon: detail::gemm_f16_neon(a, view(b), m, out); return;
#endif
#if defined(CARBON_KERNEL_NEONFP16)
    case Isa::kNeonFp16: detail::gemm_f16_neonfp16(a, view(b), m, out); return;
#endif
    default: throw Error(std::string("gemm_f16: no kernel for ") + isa_name(b.isa) + " in this build");
  }
}

void gemm_f16(Rows<float> a, const PackedF16& b, int m, const Epilogue& out) {
  if (m <= 0) return;
  if (a.taps < 1 || b.k % a.taps) throw Error("gemm_f16: k must be a whole number of taps");
  switch (f16_wide_kernel(b.isa)) {
    case Isa::kScalar: gemm_f16_scalar(a, b, m, out); return;
#if defined(CARBON_KERNEL_AVX2)
    case Isa::kAvx2: detail::gemm_f16_avx2(a, view(b), m, out); return;
#endif
#if defined(CARBON_KERNEL_AVX512)
    case Isa::kAvx512: detail::gemm_f16_avx512(a, view(b), m, out); return;
#endif
#if defined(CARBON_KERNEL_NEON)
    case Isa::kNeon: detail::gemm_f16_neon(a, view(b), m, out); return;
#endif
    default: throw Error(std::string("gemm_f16: no kernel for ") + isa_name(b.isa) + " in this build");
  }
}

void dwconv_f16(Rows<uint16_t> a, const uint16_t* w, int c, int m, const Epilogue& out, Isa isa) {
  if (m <= 0) return;
  check(isa, "dwconv_f16");
  switch (dw16_kernel(isa)) {
    case Isa::kScalar: dwconv_f16_scalar(a, w, c, m, out); return;
#if defined(CARBON_KERNEL_AVX2)
    case Isa::kAvx2: detail::dwconv_f16_avx2(a, w, c, m, out); return;
#endif
#if defined(CARBON_KERNEL_AVX512FP16)
    case Isa::kAvx512Fp16: detail::dwconv_f16_avx512fp16(a, w, c, m, out); return;
#endif
#if defined(CARBON_KERNEL_NEON)
    case Isa::kNeon: detail::dwconv_f16_neon(a, w, c, m, out); return;
#endif
#if defined(CARBON_KERNEL_NEONFP16)
    case Isa::kNeonFp16: detail::dwconv_f16_neonfp16(a, w, c, m, out); return;
#endif
    default: throw Error(std::string("dwconv_f16: no kernel for ") + isa_name(isa) + " in this build");
  }
}

void dwconv_f16(Rows<float> a, const uint16_t* w, int c, int m, const Epilogue& out, Isa isa) {
  if (m <= 0) return;
  check(isa, "dwconv_f16");
  switch (dw16_wide_kernel(isa)) {
    case Isa::kScalar: dwconv_f16_scalar(a, w, c, m, out); return;
#if defined(CARBON_KERNEL_AVX2)
    case Isa::kAvx2: detail::dwconv_f16_avx2(a, w, c, m, out); return;
#endif
#if defined(CARBON_KERNEL_NEON)
    case Isa::kNeon: detail::dwconv_f16_neon(a, w, c, m, out); return;
#endif
    default: throw Error(std::string("dwconv_f16: no kernel for ") + isa_name(isa) + " in this build");
  }
}

}  // namespace carbon::cpu
