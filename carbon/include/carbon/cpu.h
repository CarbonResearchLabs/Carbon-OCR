// The CPU's instruction sets and the kernels built on them: what the engine's own
// layers run on Intel, AMD and Arm.  Each kernel has a scalar reference and a
// version per instruction set, chosen at run time from what the CPU reports, and
// all of them give the same bits, so a test can hold every version to the
// reference on whatever machine runs it: integer sums are exact, a float sum is
// one fused multiply-add after another in the order of k, and the output stage
// is the same float operations in the same order everywhere.  The exception is
// FP16 arithmetic (the FP16 section below): its sums round to FP16, so those
// kernels are held to the reference within a bound.
//
//   x86-64   avx2         AVX2 + FMA: FP32, and int8 through exact 16-bit products
//                         (vpmaddwd) for CPUs without VNNI (AMD Zen 2 and 3, Intel
//                         before Ice Lake); FP16 stored, widened by F16C (vcvtph2ps)
//            avx-vnni     AVX-VNNI (vpdpbusd on 256 bits): Intel Alder Lake and later
//                         clients, AMD Zen 5; FP32 and FP16 as avx2
//            avx512       AVX-512 F/BW/VL/DQ + VNNI (vpdpbusd on 512 bits): Intel Ice
//                         Lake and later servers, AMD Zen 4 and 5
//            avx512-fp16  AVX-512 FP16 (vfmadd231ph, 32 halves a vector): Intel Sapphire
//                         Rapids and later servers; int8 and FP32 as avx512
//   arm64    neon         Armv8-A Advanced SIMD: FP32, FP16 widened (fcvtl); int8 by the
//                         reference
//            neon-dot     dotprod (sdot): Armv8.2-A and later (Graviton 2 to 4, Apple
//                         M, Neoverse, Cortex-A76 on); i8mm (usmmla) is reported for later
//            neon-fp16    dotprod and FP16 arithmetic (fmla on 8 halves): the same
//                         CPUs but a few; int8 as neon-dot, FP32 as neon
//
// The page turns (rotate_bilinear, rotate_quarters) are kernels too: copies of
// bytes, and bilinear samples in doubles, each operation rounded on its own.
#pragma once

#include <cstddef>
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

enum class Isa { kScalar, kNeon, kAvx2, kAvxVnni, kAvx512, kNeonDot, kAvx512Fp16, kNeonFp16 };

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
// clamped to 0 .. 255 (rounding half to even).  The FP16 kernels alone take an
// FP16 residual and store FP16 (rounded to nearest, ties to even).
struct Epilogue {
  const float* scale = nullptr;
  const float* bias = nullptr;
  const float* residual_f32 = nullptr;  // at most one of the three
  const uint8_t* residual_u8 = nullptr;
  const uint16_t* residual_f16 = nullptr;
  int64_t residual_ld = 0;
  float residual_scale = 1;
  int32_t residual_zero = 0;
  bool relu = false;
  float* out_f32 = nullptr;  // one of the three
  uint8_t* out_u8 = nullptr;
  uint16_t* out_f16 = nullptr;
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

// ---------------------------------------------------------------- FP16
//
// IEEE binary16 values held as their bits (uint16_t): the weights in FP16, the
// activations (A) in FP16 or FP32.  Two kinds of kernel, by instruction set:
//
//   FP32 arithmetic  scalar, avx2 and avx-vnni (F16C's vcvtph2ps), avx512, neon and
//                    neon-dot (fcvtl): each value widened to FP32 as it is read and
//                    summed as gemm_f32 and dwconv_f32 sum (fused, in the order of k),
//                    so every version gives the reference's bits.  B is kept in FP16;
//                    a call widens it, and the widened copy (2 MiB at most) stays
//                    for the next call on the same packing, a layer's next row.
//   FP16 arithmetic  avx512-fp16 (vfmadd231ph), neon-fp16 (fmla on 8 halves): FP16 A
//                    times FP16 B summed in FP16, twice the lanes of FP32, for at most
//                    kF16Run values of k at a time; each run's sums are then added to
//                    FP32 ones (FP16 sums over a 3x3 layer's whole k would be off by
//                    a percent).  Held to the reference within that rounding.  FP32 A
//                    runs on the FP32 arithmetic of the same CPU (avx512, neon).
//
// The output stage is the float one above on the sums (no scale), stored as FP32
// or FP16.  B is packed as pack_f32 packs it, in FP16.

constexpr int kF16Run = 64;

float f16_to_f32(uint16_t h);  // exact
uint16_t f32_to_f16(float v);  // to nearest, ties to even; past the range infinity, NaN a NaN
bool f16_arithmetic(Isa isa);  // whether the FP16 kernels of isa sum in FP16 (avx512-fp16, neon-fp16)

// n values converted, a vector at a time where the instruction set has one; the
// same bits as the functions above.
void to_f16(const float* in, uint16_t* out, size_t n, Isa isa = best_isa());
void to_f32(const uint16_t* in, float* out, size_t n, Isa isa = best_isa());

// id: the packing's own number (pack_f16 gives each a new one): a kernel that
// widens B keeps it widened for the next call on the same packing, as a layer
// makes one call an output row.
struct PackedF16 {
  Isa isa = Isa::kScalar;
  int k = 0, n = 0, np = 0;
  uint64_t id = 0;
  std::vector<uint16_t> data;
};

PackedF16 pack_f16(const uint16_t* b, int k, int n, Isa isa);  // b: [k][n], FP16, row-major

void gemm_f16(Rows<uint16_t> a, const PackedF16& b, int m, const Epilogue& out);
void gemm_f16(Rows<float> a, const PackedF16& b, int m, const Epilogue& out);

// isa: any available (its vector kernel: avx2's FP32 arithmetic for the x86-64 sets
// but avx512-fp16, neon's for Arm but neon-fp16, as dwconv_f32 does).
void dwconv_f16(Rows<uint16_t> a, const uint16_t* w, int c, int m, const Epilogue& out, Isa isa = best_isa());
void dwconv_f16(Rows<float> a, const uint16_t* w, int c, int m, const Epilogue& out, Isa isa = best_isa());

// ---------------------------------------------------------------- a threshold's bits
//
// Bit i % 32 of bits[i / 32] set where v[i] > t (a NaN never), for i in [0, n); the
// last word's bits past n clear. The detector's masks (carbon/detect.h). Vectors:
// avx2 for every x86-64 set, neon for Arm; the scalar reference elsewhere.
void above_bits(const float* v, int n, float t, uint32_t* bits, Isa isa = best_isa());

// ---------------------------------------------------------------- a crop's counts
//
// Of a crop's w x h bytes (rows `stride` apart), how many are at most `lo` and how
// many at least `hi` (light_on_dark's counts; none where lo < 0 or hi > 255).
// Vectors: avx2 for every x86-64 set, neon for Arm; the scalar reference elsewhere.
struct ByteCounts {  // an aggregate, so no constructor of its is built for an instruction set (made {0, 0})
  int64_t below, above;
};
ByteCounts count_beyond(const uint8_t* c, int w, int h, int64_t stride, int lo, int hi, Isa isa = best_isa());

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

// det_head: the line detector's head (fullerene det) in one pass.  From x[in][h4][w4] (the FPN's stride-4 features):
// a 1x1 convolution to `mid` channels and ReLU, a nearest 2x upsample, a depthwise 3x3 (padding 1) and ReLU, and a
// 1x1 convolution to the `out` maps; rows row0 to row1 - 1 of maps[out][2 h4][2 w4].  The stride-2 tensors between
// them are never written (two of 12.5 MB for a Letter page): each band keeps three upsampled rows.  Every output an
// FMA chain, the bias first and then its inputs in order (the 1x1's channels; the 3x3's taps row by row), so every
// version gives the reference's bits however the rows are cut.  Vectors run along the rows: avx2 for every x86-64
// set; the reference elsewhere.
struct DetHeadWeights {
  int in = 0, mid = 0, out = 0;  // channels: x's, the head's, the maps' (out at most 8)
  const float* w1 = nullptr;     // [mid][in]
  const float* b1 = nullptr;     // [mid]
  const float* w2 = nullptr;     // [mid][3][3]
  const float* b2 = nullptr;     // [mid]
  const float* w3 = nullptr;     // [out][mid]
  const float* b3 = nullptr;     // [out]
};
void det_head(const float* x, int h4, int w4, const DetHeadWeights& w, float* maps, int row0, int row1,
              Isa isa = best_isa());

// ---------------------------------------------------------------- page turns
//
// The loops of carbon::rotate_page and carbon::rotate_quarter, on a page of
// width x height bytes, rows `stride` apart, into dst, its rows packed.  Vectors:
// avx2 for every x86-64 set, neon for Arm; the scalar reference elsewhere.  isa:
// any available, for tests.
//
// rotate_bilinear: the page turned counterclockwise about its centre, the angle
// given by its sine and cosine.  Each pixel of dst is the bilinear sample of the
// page at (dx c - dy s + cx, dx s + dy c + cy), (dx, dy) its place from the centre
// (cx, cy) = ((width - 1) / 2, (height - 1) / 2), taps off the page paper (255),
// rounded half away from zero; a pixel none of whose taps is on the page is not
// written (dst comes in paper).  Doubles throughout, each product and sum rounded
// on its own in that order, so every version gives the reference's bytes.
void rotate_bilinear(const uint8_t* src, int width, int height, int64_t stride, double s, double c, uint8_t* dst,
                     Isa isa = best_isa());
// The same for dst's rows row0 to row1 - 1 alone (dst the whole page's): each row's bytes are what the whole turn
// writes there, so the rows can be turned by several threads at once.
void rotate_bilinear_rows(const uint8_t* src, int width, int height, int64_t stride, double s, double c, uint8_t* dst,
                          int row0, int row1, Isa isa = best_isa());

// rotate_quarters: the page turned clockwise q times 90 degrees (q 0 to 3), so
// dst's rows are height bytes long when q is odd.
void rotate_quarters(const uint8_t* src, int width, int height, int64_t stride, int q, uint8_t* dst,
                     Isa isa = best_isa());

}  // namespace carbon::cpu
