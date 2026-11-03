// The CPU's instruction sets and the kernels built on them: what the engine's own
// layers run on Intel, AMD and Arm.  Each kernel has a scalar reference and a
// version per instruction set, chosen at run time from what the CPU reports, and
// all of them give the same bits, so a test can hold every version to the
// reference on whatever machine runs it: integer sums are exact, a float sum is
// one fused multiply-add after another in the order of k, and the output stage
// is the same float operations in the same order everywhere.
//
//   x86-64   avx2      AVX2 + FMA: FP32, and int8 through exact 16-bit products
//                      (vpmaddwd) for CPUs without VNNI (AMD Zen 2 and 3, Intel
//                      before Ice Lake)
//            avx-vnni  AVX-VNNI (vpdpbusd on 256 bits): Intel Alder Lake and later
//                      clients, AMD Zen 5; FP32 as avx2
//            avx512    AVX-512 F/BW/VL/DQ + VNNI (vpdpbusd on 512 bits): Intel Ice
//                      Lake and later servers, AMD Zen 4 and 5
//   arm64    neon      Armv8-A Advanced SIMD: FP32; int8 by the reference
//            neon-dot  dotprod (sdot): Armv8.2-A and later (Graviton 2 to 4, Apple
//                      M, Neoverse, Cortex-A76 on); i8mm (usmmla) is reported for later
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace carbon::cpu {

struct Features {
  std::string vendor, brand;
  // x86-64
  bool sse41 = false, avx2 = false, fma = false, f16c = false, avx_vnni = false;
  bool avx512f = false, avx512bw = false, avx512vl = false, avx512dq = false, avx512vnni = false;
  bool avx512bf16 = false, avx512fp16 = false, amx_int8 = false, amx_bf16 = false;
  // arm64
  bool neon = false, dotprod = false, i8mm = false, fp16 = false, bf16 = false, sve = false, sve2 = false;
};

// Read once from cpuid and xgetbv (x86: the OS must save the wide registers too),
// or from the OS (Arm: getauxval, sysctl, IsProcessorFeaturePresent).
const Features& features();

enum class Isa { kScalar, kNeon, kAvx2, kAvxVnni, kAvx512, kNeonDot };

const char* isa_name(Isa isa);
bool isa_available(Isa isa);       // compiled in and supported by this CPU
std::vector<Isa> isas_available();  // the scalar reference first, the best last
Isa best_isa();                     // CARBON_ISA=<name> in the environment chooses another (tests, benchmarks)

// ---------------------------------------------------------------- A's rows
//
// A GEMM's A, m rows of k values, each row given as `taps` pieces end to end,
// each piece k / taps values: a convolution's row is an output pixel, a piece
// the channels under one tap of its kernel, read where they are, so nothing is
// copied into columns first (im2col).  Two forms:
//
//   a list    piece t of row i at ptrs[i * taps + t]
//   a grid    piece t of row i at base + i * stride + offsets[t] (no offsets: 0),
//             an output row of a convolution on an input with a zero border:
//             stride the pixels' step, offsets[t] the tap's place
template <typename T>
struct Rows {
  const T* const* ptrs = nullptr;
  int taps = 1;
  const T* base = nullptr;
  int64_t stride = 0;
  const int64_t* offsets = nullptr;

  const T* piece(int64_t i, int t) const {
    return ptrs ? ptrs[i * taps + t] : base + i * stride + (offsets ? offsets[t] : 0);
  }
};

// A matrix: rows lda values apart, one piece each.
template <typename T>
Rows<T> rows_of(const T* a, int64_t lda) {
  Rows<T> r;
  r.base = a;
  r.stride = lda;
  return r;
}

// A grid of `taps` pieces a row.
template <typename T>
Rows<T> grid(const T* base, int64_t stride, int taps, const int64_t* offsets) {
  Rows<T> r;
  r.taps = taps;
  r.base = base;
  r.stride = stride;
  r.offsets = offsets;
  return r;
}

// ---------------------------------------------------------------- the output stage
//
// What the sums become, in the kernel, per row i and column j of the output:
//
//   v = sum * scale[j]           int8 sums only, the product's scale (A's times B's
//                                column j); none: 1
//     + bias[j]
//     + residual[i][j]           float, or uint8 dequantized: (r - zero) * scale
//   v = v > 0 ? v : 0            relu
//
// then stored as float, or quantized: round(v * out_inv_scale) + out_zero,
// clamped to 0 .. 255 (rounding half to even).
struct Epilogue {
  const float* scale = nullptr;
  const float* bias = nullptr;
  const float* residual_f32 = nullptr;
  const uint8_t* residual_u8 = nullptr;
  int64_t residual_ld = 0;
  float residual_scale = 1;
  int32_t residual_zero = 0;
  bool relu = false;
  float* out_f32 = nullptr;  // one of the two
  uint8_t* out_u8 = nullptr;
  int64_t out_ld = 0;
  float out_inv_scale = 1;
  int32_t out_zero = 0;
};

// ---------------------------------------------------------------- int8 GEMM
//
// sum[i][j] = sum over k of (A[i][k] - a_zero) * B[k][j], A uint8 (an activation,
// with its zero point), B int8 (weights, symmetric).  B is packed once for an ISA,
// in `taps` pieces of k / taps rows each, matching A's; each piece of A must have
// k_padded(k / taps) bytes readable (the bytes past the piece are any value: they
// meet B's zero padding).

struct PackedB {
  Isa isa = Isa::kScalar;
  int k = 0, n = 0, taps = 1;  // the logical shape
  int seg = 0, segp = 0;       // a piece's rows, and padded to the ISA's group of k
  int np = 0;                  // n padded to the ISA's panel width
  std::vector<int8_t> data;
  std::vector<int32_t> col_sums;  // sum over k of B[k][j]: for A's zero point
};

int k_padded(int k, Isa isa);  // a piece's bytes the kernel reads
PackedB pack_b(const int8_t* b, int k, int n, Isa isa, int taps = 1);  // b: [k][n], row-major

void gemm_u8s8(Rows<uint8_t> a, int32_t a_zero, const PackedB& b, int m, const Epilogue& out);

// ---------------------------------------------------------------- FP32 GEMM
//
// sum[i][j] starts at 0 and takes each k in turn, fused: sum = fma(A[i][k],
// B[k][j], sum), so the bits do not depend on the instruction set.  The output
// stage takes no scale.

struct PackedF32 {
  Isa isa = Isa::kScalar;
  int k = 0, n = 0, np = 0;
  std::vector<float> data;
};

PackedF32 pack_f32(const float* b, int k, int n, Isa isa);  // b: [k][n], row-major

void gemm_f32(Rows<float> a, const PackedF32& b, int m, const Epilogue& out);

// ---------------------------------------------------------------- depthwise convolution
//
// Each channel on its own: sum[i][c] = sum over taps t of A's piece t of row i,
// channel c, times w[t][c] (int8: (a - a_zero) * w), each piece c values, then
// the output stage.  FP32 sums are fused multiply-adds in the order of the taps.
// isa: any available, for tests (its vector kernel: avx2 for every x86-64 set,
// neon for Arm).

void dwconv_f32(Rows<float> a, const float* w, int c, int m, const Epilogue& out, Isa isa = best_isa());
void dwconv_u8s8(Rows<uint8_t> a, int32_t a_zero, const int8_t* w, int c, int m, const Epilogue& out,
                 Isa isa = best_isa());

// ---------------------------------------------------------------- CTC greedy
//
// One line of a recognizer's logits [classes][frames] (class-major, the models'
// layout), its first t_n frames: ids[t] the best class (ties to the lower id), and
// for a frame whose best is not the blank (class 0), probs[t] its softmax
// probability, 1 / the sum over the classes of exp(logit - best), in class order.
// A blank frame's probability is not written.  The exp is Cephes's expf polynomial
// in separate multiplies and adds, so every version gives the reference's bits.
// Vectors run along the frames: avx2 for every x86-64 set, neon for Arm.
void ctc_argmax(const float* logits, int classes, int frames, int t_n, int32_t* ids, float* probs,
                Isa isa = best_isa());

}  // namespace carbon::cpu
