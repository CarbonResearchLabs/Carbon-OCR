// The small transformer language model on a GPU (carbon/tlm.h, cuda_transformer_lm): a page's unsure lines
// rescored in one run.
//
// A run has two passes over rows of tokens, both enqueued on one stream, and one wait at the end:
//   the shared pass  every line's context and its readings' common prefix, a line's rows causal among themselves,
//                    every layer's q, k, v kept (the cache);
//   the rest pass    every reading's own tokens, each row attending to its line's shared rows (the cache) and its
//                    reading's rows so far.
// A layer is the same chain of kernels in both:
//   layer_norm   a warp a row, FP32 in, FP16 out (the next product's A)
//   gemm         C = A B on the tensor cores (WMMA 16x16x16, FP16 in, FP32 sums): tiles of 64 x 64 by four warps,
//                32 of K a step, the next step's tiles loaded into registers while the tensor cores run this one;
//                the epilogue fused (the bias, GELU, the residual), four columns a thread; FP32 out (the stream and
//                q, k, v) or FP16 (the next product's A)
//   attend       a warp a (row, head): its lanes take 32 keys at once (each lane a key's dot product with the
//                query, which every lane holds), the softmax online by chunks, then the chunk's values summed a
//                lane a dimension, each key's weight broadcast
// In the shared pass's last layer only the rows whose next token is asked (from each line's last context token
// on) go past q, k, v: they are gathered, and only they run the attention, the feed-forward and the head; the
// context's other rows are only keys and values there. The head's columns are padded to 8 (the vector loads);
// `pick` takes each target's log-softmax over the real ones, a warp a target. The tokens, the row descriptors
// and the picked log-probabilities are all that cross the bus, one copy each way a pass, from pinned memory.
//
// A line with a reading longer than the window runs as the CPU's model scores it, in pieces that each fit: its
// readings that fit, each alone after the line's context, and each longer one in windows, each window after the one
// before, less the line's end after every window but the last. The pieces are queries of the same run as the page's
// other lines, so nothing of a page waits on the CPU.
//
// Each caller runs on a state of its own (a stream, the passes it lays out, their device and pinned memory), taken
// from a pool of up to kRuns and given back when its run ends; only the weights are shared. Workers' pages run on
// their own streams at once, not one after another. A page's lines go in batches of at most kBatchRows rows a pass,
// one after another on its run, so a run's memory is a batch's, not its largest page's.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "carbon/error.h"
#include "carbon/tlm.h"
#include "../tlm_file.h"

namespace carbon {

namespace {

using namespace nvcuda;

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw Error(std::string("CUDA: ") + what + ": " + cudaGetErrorString(e));
}

constexpr int kWarps = 4, kThreads = kWarps * 32;
constexpr int kTile = 64, kStep = 32;  // the GEMM's tile of C and its step along K

__inline__ __device__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__inline__ __device__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// x[r] = emb[tok[r]] + pos[at[r]]
__global__ void embed_k(const float* __restrict__ emb, const float* __restrict__ pos, const int32_t* __restrict__ tok,
                        const int32_t* __restrict__ at, int rows, int d, float* __restrict__ x) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const float4* e = reinterpret_cast<const float4*>(emb + static_cast<size_t>(tok[r]) * d);
  const float4* p = reinterpret_cast<const float4*>(pos + static_cast<size_t>(at[r]) * d);
  float4* o = reinterpret_cast<float4*>(x + static_cast<size_t>(r) * d);
  for (int j = threadIdx.x; j < d / 4; j += blockDim.x) {
    const float4 a = e[j], b = p[j];
    o[j] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
  }
}

// out[r] = x[rows[r]]: the last layer's rows
__global__ void gather_k(const float* __restrict__ x, const int32_t* __restrict__ rows, int n, int d,
                         float* __restrict__ out) {
  const int r = blockIdx.x;
  if (r >= n) return;
  const float4* s = reinterpret_cast<const float4*>(x + static_cast<size_t>(rows[r]) * d);
  float4* o = reinterpret_cast<float4*>(out + static_cast<size_t>(r) * d);
  for (int j = threadIdx.x; j < d / 4; j += blockDim.x) o[j] = s[j];
}

// v quantized to uint8 by (qs, qz) as the CPU's INT8 layers quantize their inputs (QuantizeLinear's IEEE division,
// rounding half to even), less the zero point: q - qz, a whole number of -255 .. 255 that FP16 holds exactly. The
// next product, on int8 weights held the same way, sums these exactly in FP32 (|sum| < 255 * 127 * K < 2^24 for K up
// to 512) and its epilogue scales the sums by the input's and each column's scales: the CPU's int8 sums exactly.
// qs 0: v as it is (an FP32 or FP16 file)
__device__ __forceinline__ float fq(float v, float qs, float qz) {
  if (qs <= 0.0f) return v;
  return fminf(255.0f, fmaxf(0.0f, rintf(__fdiv_rn(v, qs)) + qz)) - qz;
}

// out[r] = LayerNorm(x[r]) w + b in FP16 (fq'd by qs, qz: INT8's input steps), eps 1e-5 (PyTorch's and the CPU's), a warp a row
__global__ void layer_norm_k(const float* __restrict__ x, int rows, int d, const float* __restrict__ w,
                             const float* __restrict__ b, float qs, float qz, __half* __restrict__ out) {
  const int r = blockIdx.x * kWarps + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (r >= rows) return;
  const float* v = x + static_cast<size_t>(r) * d;
  float s = 0;
  for (int j = lane; j < d; j += 32) s += v[j];
  const float mean = warp_sum(s) / d;
  float q = 0;
  for (int j = lane; j < d; j += 32) q += (v[j] - mean) * (v[j] - mean);
  const float rs = rsqrtf(warp_sum(q) / d + 1e-5f);
  __half* o = out + static_cast<size_t>(r) * d;
  for (int j = lane; j < d; j += 32) o[j] = __float2half(fq((v[j] - mean) * rs * w[j] + b[j], qs, qz));
}

// C[M][N] = A[M][K] B[K][N], A and B FP16 row-major, K a multiple of 32 and N of 8; then per element * scale[n]
// (INT8's sums), + bias[n], GELU (exact, erf), + residual[m][n]; into out32 and/or out16 (fq'd by qs, qz).
__global__ void __launch_bounds__(kThreads)
    gemm_k(const __half* __restrict__ A, const __half* __restrict__ B, int M, int N, int K,
           const float* __restrict__ scale, const float* __restrict__ bias, const float* __restrict__ residual,
           bool gelu, float* __restrict__ out32, float qs, float qz, __half* __restrict__ out16) {
  __shared__ __align__(32) __half As[kTile][kStep + 8];
  __shared__ __align__(32) __half Bs[kStep][kTile + 8];
  __shared__ __align__(32) float Cs[kTile][kTile + 4];
  const int tid = threadIdx.x, warp = tid / 32, wm = warp / 2, wn = warp % 2;
  const int m0 = blockIdx.y * kTile, n0 = blockIdx.x * kTile;
  // each thread's two 8-half pieces of A's tile (64 x 32) and of B's (32 x 64)
  int ar[2], ac[2], br[2], bc[2];
  for (int i = 0; i < 2; ++i) {
    const int e = tid + i * kThreads;
    ar[i] = e / 4, ac[i] = (e % 4) * 8;
    br[i] = e / 8, bc[i] = (e % 8) * 8;
  }
  uint4 ra[2], rb[2];
  const uint4 zero = make_uint4(0, 0, 0, 0);
  auto load = [&](int k0) {
    for (int i = 0; i < 2; ++i) {
      const int gm = m0 + ar[i], gn = n0 + bc[i];
      ra[i] = gm < M ? *reinterpret_cast<const uint4*>(A + static_cast<size_t>(gm) * K + k0 + ac[i]) : zero;
      rb[i] = gn < N ? *reinterpret_cast<const uint4*>(B + static_cast<size_t>(k0 + br[i]) * N + gn) : zero;
    }
  };
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);
  }
  load(0);
  for (int k0 = 0; k0 < K; k0 += kStep) {
    for (int i = 0; i < 2; ++i) {
      *reinterpret_cast<uint4*>(&As[ar[i]][ac[i]]) = ra[i];
      *reinterpret_cast<uint4*>(&Bs[br[i]][bc[i]]) = rb[i];
    }
    __syncthreads();
    if (k0 + kStep < K) load(k0 + kStep);  // the next step's tiles, while this one runs
    for (int kk = 0; kk < kStep; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b[2];
      for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &As[wm * 32 + i * 16][kk], kStep + 8);
      for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &Bs[kk][wn * 32 + j * 16], kTile + 8);
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      wmma::store_matrix_sync(&Cs[wm * 32 + i * 16][wn * 32 + j * 16], acc[i][j], kTile + 4, wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int e = tid; e < kTile * kTile / 4; e += kThreads) {
    const int r = e / (kTile / 4), c = (e % (kTile / 4)) * 4, gm = m0 + r, gn = n0 + c;
    if (gm >= M || gn >= N) continue;
    float v[4] = {Cs[r][c], Cs[r][c + 1], Cs[r][c + 2], Cs[r][c + 3]};
    const size_t at = static_cast<size_t>(gm) * N + gn;
    if (scale) {
      const float4 ss = *reinterpret_cast<const float4*>(scale + gn);
      v[0] *= ss.x, v[1] *= ss.y, v[2] *= ss.z, v[3] *= ss.w;
    }
    if (bias) {
      const float4 bb = *reinterpret_cast<const float4*>(bias + gn);
      v[0] += bb.x, v[1] += bb.y, v[2] += bb.z, v[3] += bb.w;
    }
    if (gelu) {
      for (float& t : v) t = 0.5f * t * (1.0f + erff(t * 0.70710678118654752f));
    }
    if (residual) {
      const float4 rr = *reinterpret_cast<const float4*>(residual + at);
      v[0] += rr.x, v[1] += rr.y, v[2] += rr.z, v[3] += rr.w;
    }
    if (out32) *reinterpret_cast<float4*>(out32 + at) = make_float4(v[0], v[1], v[2], v[3]);
    if (out16) {
      __half2* o2 = reinterpret_cast<__half2*>(out16 + at);
      o2[0] = __floats2half2_rn(fq(v[0], qs, qz), fq(v[1], qs, qz));
      o2[1] = __floats2half2_rn(fq(v[2], qs, qz), fq(v[3], qs, qz));
    }
  }
}

// A row's keys: rows [s1, s1 + n1) of buf1 and [s2, s2 + n2) of buf2, each row q, k, v ([3d]).
struct Runs {
  int32_t s1, n1, s2, n2;
};

// out[r] = the attention of row qrow[r] of qkv (row r where qrow is null) over its runs' keys, head by head (fq'd by
// qs, qz); DH the head's width, a multiple of 32: a lane holds DH / 32 of the query's and of the output's dimensions.
template <int DH>
__global__ void __launch_bounds__(kThreads)
    attend_k(const float* __restrict__ qkv, const int32_t* __restrict__ qrow, const float* __restrict__ buf1,
             const float* __restrict__ buf2, const Runs* __restrict__ runs, int rows, int d, int heads, float scale,
             float qs, float qz, __half* __restrict__ out) {
  constexpr int E = DH / 32;
  const int g = blockIdx.x * kWarps + threadIdx.x / 32, lane = threadIdx.x % 32;
  const int r = g / heads, h = g % heads;
  if (r >= rows) return;  // the whole warp
  const int src = qrow ? qrow[r] : r, D3 = 3 * d, col = h * DH;
  // the query, all of it in every lane's registers, scaled
  float mine[E];
  for (int e = 0; e < E; ++e) mine[e] = qkv[static_cast<size_t>(src) * D3 + col + e * 32 + lane] * scale;
  float q[DH];
  for (int e = 0; e < E; ++e) {
    for (int t = 0; t < 32; ++t) q[e * 32 + t] = __shfl_sync(0xffffffffu, mine[e], t);
  }
  const Runs ru = runs[src];
  float m = -INFINITY, s = 0, acc[E] = {};
  for (int run = 0; run < 2; ++run) {
    const float* base = run ? buf2 : buf1;
    const int st = run ? ru.s2 : ru.s1, n = run ? ru.n2 : ru.n1;
    for (int c0 = 0; c0 < n; c0 += 32) {
      const int j = c0 + lane;
      float score = -INFINITY;
      if (j < n) {  // this lane's key
        const float4* k = reinterpret_cast<const float4*>(base + static_cast<size_t>(st + j) * D3 + d + col);
        float a = 0;
        for (int t = 0; t < DH / 4; ++t) {
          const float4 kv = k[t];
          a = fmaf(q[4 * t], kv.x, a);
          a = fmaf(q[4 * t + 1], kv.y, a);
          a = fmaf(q[4 * t + 2], kv.z, a);
          a = fmaf(q[4 * t + 3], kv.w, a);
        }
        score = a;
      }
      const float m2 = fmaxf(m, warp_max(score));
      const float c = __expf(m - m2), p = j < n ? __expf(score - m2) : 0.0f;
      s = s * c + warp_sum(p);
      for (int e = 0; e < E; ++e) acc[e] *= c;
      const int cnt = min(32, n - c0);
      const float* v = base + static_cast<size_t>(st + c0) * D3 + 2 * d + col + lane;
      for (int t = 0; t < cnt; ++t) {
        const float pt = __shfl_sync(0xffffffffu, p, t);
        for (int e = 0; e < E; ++e) acc[e] = fmaf(pt, v[static_cast<size_t>(t) * D3 + e * 32], acc[e]);
      }
      m = m2;
    }
  }
  const float inv = 1.0f / s;
  for (int e = 0; e < E; ++e) {
    out[static_cast<size_t>(r) * d + col + e * 32 + lane] = __float2half(fq(acc[e] * inv, qs, qz));
  }
}

// out[i] = ln softmax(logits[row[i]][0, n))[target[i]], the rows ld apart; a warp a pick
__global__ void pick_k(const float* __restrict__ logits, int n, int ld, const int32_t* __restrict__ row,
                       const int32_t* __restrict__ target, int picks, float* __restrict__ out) {
  const int i = blockIdx.x * kWarps + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (i >= picks) return;
  const float* v = logits + static_cast<size_t>(row[i]) * ld;
  float m = -INFINITY;
  for (int j = lane; j < n; j += 32) m = fmaxf(m, v[j]);
  m = warp_max(m);
  float s = 0;
  for (int j = lane; j < n; j += 32) s += __expf(v[j] - m);
  s = warp_sum(s);
  if (lane == 0) out[i] = v[target[i]] - m - logf(s);
}

// Device memory, grown when asked and never shrunk; moved, never copied.
template <class T>
struct Buf {
  T* p = nullptr;
  size_t n = 0;
  Buf() = default;
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  Buf(Buf&& o) noexcept : p(std::exchange(o.p, nullptr)), n(std::exchange(o.n, 0)) {}
  ~Buf() {
    if (p) cudaFree(p);
  }
  T* reserve(size_t k) {
    if (k > n) {
      if (p) cudaFree(p);
      p = nullptr;
      n = 0;
      ck(cudaMalloc(reinterpret_cast<void**>(&p), std::max<size_t>(k, 1) * sizeof(T)), "cudaMalloc");
      n = k;
    }
    return p;
  }
};

// Pinned host memory, the same.
template <class T>
struct Pinned {
  T* p = nullptr;
  size_t n = 0;
  Pinned() = default;
  Pinned(const Pinned&) = delete;
  Pinned& operator=(const Pinned&) = delete;
  ~Pinned() {
    if (p) cudaFreeHost(p);
  }
  T* reserve(size_t k) {
    if (k > n) {
      if (p) cudaFreeHost(p);
      p = nullptr;
      n = 0;
      ck(cudaMallocHost(reinterpret_cast<void**>(&p), std::max<size_t>(k, 1) * sizeof(T)), "cudaMallocHost");
      n = k;
    }
    return p;
  }
};

// The caller's device, set back when the call ends.
struct OnDevice {
  int prev = 0;
  explicit OnDevice(int gpu) {
    ck(cudaGetDevice(&prev), "cudaGetDevice");
    if (prev != gpu) ck(cudaSetDevice(gpu), "cudaSetDevice");
  }
  ~OnDevice() {
    int now = 0;
    if (cudaGetDevice(&now) == cudaSuccess && now != prev) cudaSetDevice(prev);
  }
};

// A layer's weights; an INT8 file's: w* its int8 values (exact in FP16), s* each column's scale times its input's
// (the sums' scale), q* its input's quantization (scale, zero point; 0, 0 and no s* otherwise)
struct LayerW {
  Buf<float> ln1w, ln1b, bqkv, bo, ln2w, ln2b, b1, b2;
  Buf<__half> wqkv, wo, w1, w2;
  Buf<float> sqkv, so, sw1, sw2;
  float qqkv[2] = {}, qo[2] = {}, qw1[2] = {}, qw2[2] = {};
};

// One pass's rows, as the host lays them out.
struct Pass {
  std::vector<int32_t> tok, at;
  std::vector<Runs> runs;
  std::vector<int32_t> last;  // the rows the last layer takes past q, k, v (their next token asked), in order
  bool all = true;            // every row is (last unused)
  std::vector<int32_t> prow;  // a pick's row of the head's (of `last`, or of every row)
  std::vector<int32_t> ptgt;  // its target
  void clear() {
    tok.clear(), at.clear(), runs.clear(), last.clear(), prow.clear(), ptgt.clear();
    all = true;
  }
  int head_rows() const { return all ? static_cast<int>(tok.size()) : static_cast<int>(last.size()); }
};

// Where a pick's log-probability goes: a query of the run and its reading (-1: every reading of the query).
struct To {
  int32_t line, reading;
};

// Where a reading of a query of the run goes: added to out[line][reading], times sign (-1: a line's end taken back
// out of a window that is not the line's last).
struct Part {
  int32_t line, reading;
  double sign;
};

// One caller's run: its stream, the passes it lays out, and the device and pinned memory they take, grown when asked
// (to a batch's, kBatchRows) and never shrunk.
struct Run {
  cudaStream_t s = nullptr;
  Pass shared, rest;
  std::vector<To> sto, rto;
  Buf<float> cache, x, xa, xb, xg, qkv, logits, picks_s, picks_r;
  Buf<__half> h16, att, ff;
  Buf<int32_t> ints_s, ints_r;
  Pinned<int32_t> stage_s, stage_r;
  Pinned<float> back_s, back_r;
  Run() = default;
  Run(const Run&) = delete;
  Run& operator=(const Run&) = delete;
  ~Run() {
    if (s) cudaStreamDestroy(s);
  }
};

// The runs callers can hold at once; a caller past them waits for one.
constexpr size_t kRuns = 8;

// The rows a pass of one batch takes at most: a page with more lines runs as several batches, one after another on
// its run. A run keeps device memory for the largest batch it has run, so its memory is bounded whatever the pages:
// for Latin's model (d 128, 4 layers, a feed-forward of 512, 800 tokens) about 120 MB, and about 1 GB for all kRuns.
// Kept for the largest page instead, 8 workers' runs grew to 15.9 GB of a 16 GB card, and pages failed with CUDA's
// "out of memory" (10 of 17,898 in 12 minutes).
constexpr size_t kBatchRows = 8192;

class CudaTlm final : public LineRescorer {
 public:
  CudaTlm(const std::filesystem::path& file, int gpu) : gpu_(gpu), cpu_(TransformerLM::open(file)) {
    h_ = cpu_->header();
    d_ = static_cast<int>(h_.d);
    dh_ = d_ / static_cast<int>(h_.heads);
    if (dh_ != 32 && dh_ != 64 && dh_ != 128) {
      throw Error(file.string() + ": the GPU's attention takes heads 32, 64 or 128 wide, not " + std::to_string(dh_));
    }
    if (d_ % kStep || h_.ffn % kStep) {
      throw Error(file.string() + ": the GPU's products take a width and a feed-forward that are multiples of 32");
    }
    tp_ = (static_cast<int>(h_.tokens) + 7) / 8 * 8;
    OnDevice on(gpu_);
    // the weights from the file again (the CPU's model keeps its own, packed for its kernels): FP32's and FP16's as they
    // compute (FP16's widened, so exactly the file's halves again here); INT8's int8 values, with the sums' scales and
    // the inputs' quantization
    const tlm::File f = tlm::read(file);
    const bool i8 = h_.precision == TlmPrecision::kInt8;
    const int d = d_, T = static_cast<int>(h_.tokens);
    auto f32 = [&](Buf<float>& b, const std::vector<float>& v) {
      ck(cudaMemcpy(b.reserve(v.size()), v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice),
         "the weights' upload");
    };
    auto f16 = [&](Buf<__half>& b, Buf<float>& sums, const tlm::Matrix& m, float* q) {
      std::vector<__half> h(m.f32.size());
      for (size_t i = 0; i < h.size(); ++i) h[i] = __float2half(i8 ? static_cast<float>(m.i8[i]) : m.f32[i]);
      ck(cudaMemcpy(b.reserve(h.size()), h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice),
         "the weights' upload");
      q[0] = i8 ? m.in_scale : 0.0f;
      q[1] = i8 ? m.in_zero : 0.0f;
      if (i8) {
        std::vector<float> s(m.scale.size());
        for (size_t j = 0; j < s.size(); ++j) s[j] = m.in_scale * m.scale[j];  // the CPU's (tlm.cpp's pack)
        f32(sums, s);
      }
    };
    f32(emb_, f.emb);
    f32(pos_, f.pos);
    layers_.resize(h_.layers);
    for (size_t l = 0; l < layers_.size(); ++l) {
      LayerW& L = layers_[l];
      const tlm::Layer& F = f.layers[l];
      f32(L.ln1w, F.ln1w);
      f32(L.ln1b, F.ln1b);
      f16(L.wqkv, L.sqkv, F.qkv, L.qqkv);
      f32(L.bqkv, F.bqkv);
      f16(L.wo, L.so, F.o, L.qo);
      f32(L.bo, F.bo);
      f32(L.ln2w, F.ln2w);
      f32(L.ln2b, F.ln2b);
      f16(L.w1, L.sw1, F.w1, L.qw1);
      f32(L.b1, F.b1);
      f16(L.w2, L.sw2, F.w2, L.qw2);
      f32(L.b2, F.b2);
    }
    f32(lnfw_, f.lnfw);
    f32(lnfb_, f.lnfb);
    std::vector<__half> head(static_cast<size_t>(d) * tp_, __float2half(0.0f));  // [d][tokens], padded to tp_
    for (int j = 0; j < d; ++j) {
      for (int i = 0; i < T; ++i) {
        const size_t at = static_cast<size_t>(j) * T + i;
        head[static_cast<size_t>(j) * tp_ + i] = __float2half(i8 ? static_cast<float>(f.head.i8[at]) : f.head.f32[at]);
      }
    }
    ck(cudaMemcpy(head_.reserve(head.size()), head.data(), head.size() * sizeof(__half), cudaMemcpyHostToDevice),
       "the weights' upload");
    qhead_[0] = i8 ? f.head.in_scale : 0.0f;
    qhead_[1] = i8 ? f.head.in_zero : 0.0f;
    if (i8) {
      std::vector<float> s(static_cast<size_t>(tp_), 0.0f);
      for (int i = 0; i < T; ++i) s[static_cast<size_t>(i)] = f.head.in_scale * f.head.scale[static_cast<size_t>(i)];
      f32(shead_, s);
    }
  }
  int boundary() const override { return static_cast<int>(h_.classes); }

  void score(std::span<const int32_t> context, std::span<const std::vector<int32_t>> readings,
             std::span<double> out) const override {
    const RescoreQuery q{context, readings};
    std::vector<std::vector<double>> o(1);
    score_lines(std::span<const RescoreQuery>(&q, 1), o);
    std::copy(o[0].begin(), o[0].end(), out.begin());
  }

  void score_lines(std::span<const RescoreQuery> queries, std::span<std::vector<double>> out) const override {
    const int C = static_cast<int>(h_.ctx), B = static_cast<int>(h_.classes);
    for (const RescoreQuery& q : queries) {  // the CPU's checks, before anything runs
      if (q.readings.empty()) continue;
      if (q.context.empty()) throw Error("TransformerLM::score: a context of at least one token (the boundary)");
      for (const int32_t t : q.context) {
        if (t < 1 || t > B) throw Error("TransformerLM::score: context token " + std::to_string(t));
      }
      for (const auto& rd : q.readings) {
        for (const int32_t t : rd) {
          if (t < 1 || t > B) throw Error("TransformerLM::score: token " + std::to_string(t));
        }
      }
    }
    // the run's queries, each of whose readings fits the window, and where each reading's log-probability goes
    std::vector<RescoreQuery> qs;
    std::vector<std::vector<Part>> parts;
    std::deque<std::vector<int32_t>> pieces;  // the windows' tokens and contexts (a deque: they stay where they are)
    static const std::vector<int32_t> none;  // the reading a line's end alone is scored as
    for (size_t l = 0; l < queries.size(); ++l) {
      const RescoreQuery& q = queries[l];
      out[l].assign(q.readings.size(), 0.0);
      if (q.readings.empty()) continue;
      size_t longest = 0;
      for (const auto& r : q.readings) longest = std::max(longest, r.size());
      const int32_t line = static_cast<int32_t>(l);
      if (static_cast<int>(longest) + 1 <= C) {  // the line as it is
        qs.push_back(q);
        parts.emplace_back();
        for (size_t r = 0; r < q.readings.size(); ++r) parts.back().push_back({line, static_cast<int32_t>(r), 1.0});
        continue;
      }
      // longer than the window: as the CPU's model scores it (TransformerLM::score)
      for (size_t r = 0; r < q.readings.size(); ++r) {
        const auto& rd = q.readings[r];
        const int32_t reading = static_cast<int32_t>(r);
        if (static_cast<int>(rd.size()) + 1 <= C) {  // a reading that fits: alone, after the whole context
          qs.push_back({q.context, std::span<const std::vector<int32_t>>(&rd, 1)});
          parts.push_back({{line, reading, 1.0}});
          continue;
        }
        // windows of ctx / 2 tokens, each after the one before (the first after the context's last token); every
        // window but the last less the line's end after it
        const size_t step = static_cast<size_t>(C / 2);
        const std::vector<int32_t>* ctx = &pieces.emplace_back(q.context.end() - 1, q.context.end());
        for (size_t a = 0; a < rd.size(); a += step) {
          const size_t b = std::min(rd.size(), a + step);
          const std::vector<int32_t>& piece = pieces.emplace_back(rd.begin() + static_cast<std::ptrdiff_t>(a),
                                                                  rd.begin() + static_cast<std::ptrdiff_t>(b));
          qs.push_back({*ctx, std::span<const std::vector<int32_t>>(&piece, 1)});
          parts.push_back({{line, reading, 1.0}});
          if (b < rd.size()) {
            std::vector<int32_t>& both = pieces.emplace_back(*ctx);
            both.insert(both.end(), piece.begin(), piece.end());
            qs.push_back({both, std::span<const std::vector<int32_t>>(&none, 1)});
            parts.push_back({{line, reading, -1.0}});
          }
          ctx = &piece;
        }
      }
    }
    if (qs.empty()) return;
    OnDevice on(gpu_);
    Lease lease(*this);
    Run& job = *lease.run;
    Pass& S = job.shared;
    Pass& R = job.rest;
    auto fresh = [&] {
      S.clear();
      R.clear();
      S.all = false;
      job.sto.clear();
      job.rto.clear();
    };
    auto add = [&](const Part& p, double v) {
      out[static_cast<size_t>(p.line)][static_cast<size_t>(p.reading)] += p.sign * v;
    };
    // the batch laid out so far: its two passes run, their picks added where they go, and a new batch begun
    auto run = [&] {
      reserve(job);  // everything both passes take, before either is enqueued: nothing is freed under a running kernel
      const float* sp = enqueue(job, S, job.ints_s, job.stage_s, job.picks_s, job.back_s, job.cache.p, nullptr);
      const float* rp = R.tok.empty()
                            ? nullptr
                            : enqueue(job, R, job.ints_r, job.stage_r, job.picks_r, job.back_r, nullptr, job.cache.p);
      ck(cudaStreamSynchronize(job.s), "the rescore");
      for (size_t i = 0; i < job.sto.size(); ++i) {
        const auto& to = parts[static_cast<size_t>(job.sto[i].line)];
        if (job.sto[i].reading < 0) {
          for (const Part& p : to) add(p, sp[i]);
        } else {
          add(to[static_cast<size_t>(job.sto[i].reading)], sp[i]);
        }
      }
      for (size_t i = 0; i < job.rto.size(); ++i) {
        add(parts[static_cast<size_t>(job.rto[i].line)][static_cast<size_t>(job.rto[i].reading)], rp[i]);
      }
      fresh();
    };
    fresh();
    for (size_t l = 0; l < qs.size(); ++l) {
      const RescoreQuery& q = qs[l];
      size_t longest = 0;
      for (const auto& r : q.readings) longest = std::max(longest, r.size());
      // the context, its oldest tokens dropped to fit the window with the longest reading (as the CPU's)
      const size_t keep = std::min(q.context.size(), static_cast<size_t>(C) - longest);
      const std::span<const int32_t> ctx = q.context.subspan(q.context.size() - keep);
      size_t P = q.readings[0].size();
      for (const auto& r : q.readings) {
        size_t k = 0;
        while (k < P && k < r.size() && r[k] == q.readings[0][k]) ++k;
        P = k;
      }
      // a batch takes at most kBatchRows rows a pass (a query larger than that runs alone): the page's lines past that
      // go in the next, on the same run, so a run's memory is a batch's however large the page
      size_t rest = 0;
      for (const auto& r : q.readings) rest += r.size() - P;
      if (!S.tok.empty() && (S.tok.size() + ctx.size() + P > kBatchRows || R.tok.size() + rest > kBatchRows)) run();
      const int seg = static_cast<int>(S.tok.size()), Ls = static_cast<int>(ctx.size() + P);
      for (int i = 0; i < Ls; ++i) {
        S.tok.push_back(i < static_cast<int>(ctx.size()) ? ctx[static_cast<size_t>(i)]
                                                          : q.readings[0][static_cast<size_t>(i) - ctx.size()]);
        S.at.push_back(i);
        S.runs.push_back({seg, i + 1, 0, 0});
      }
      // the rows from the context's last on: each predicts the next token, the last every reading's rest's first
      const int first = static_cast<int>(S.last.size());
      for (int i = static_cast<int>(ctx.size()) - 1; i < Ls; ++i) S.last.push_back(seg + i);
      for (size_t i = 0; i < P; ++i) {
        S.prow.push_back(first + static_cast<int>(i));
        S.ptgt.push_back(q.readings[0][i]);
        job.sto.push_back({static_cast<int32_t>(l), -1});
      }
      const int lastrow = static_cast<int>(S.last.size()) - 1;
      for (size_t r = 0; r < q.readings.size(); ++r) {
        const auto& rd = q.readings[r];
        S.prow.push_back(lastrow);
        S.ptgt.push_back(rd.size() > P ? rd[P] : B);
        job.sto.push_back({static_cast<int32_t>(l), static_cast<int32_t>(r)});
        const int r0 = static_cast<int>(R.tok.size());
        for (size_t j = P; j < rd.size(); ++j) {  // the reading's rest: each row predicts the next, the last the end
          const int row = static_cast<int>(R.tok.size());
          R.tok.push_back(rd[j]);
          R.at.push_back(Ls + static_cast<int>(j - P));
          R.runs.push_back({seg, Ls, r0, static_cast<int>(j - P) + 1});
          R.prow.push_back(row);
          R.ptgt.push_back(j + 1 < rd.size() ? rd[j + 1] : B);
          job.rto.push_back({static_cast<int32_t>(l), static_cast<int32_t>(r)});
        }
      }
    }
    run();
  }

 private:
  // A caller's hold on a run, given back (its stream drained) when it ends.
  struct Lease {
    const CudaTlm& t;
    std::unique_ptr<Run> run;
    explicit Lease(const CudaTlm& owner) : t(owner), run(owner.acquire()) {}
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;
    ~Lease() { t.release(std::move(run)); }
  };

  // An idle run, or a new one while fewer than kRuns are made; otherwise the wait for one. On the caller's device.
  std::unique_ptr<Run> acquire() const {
    std::unique_lock lk(pool_m_);
    pool_cv_.wait(lk, [&] { return !idle_.empty() || made_ < kRuns; });
    if (!idle_.empty()) {
      std::unique_ptr<Run> r = std::move(idle_.back());
      idle_.pop_back();
      return r;
    }
    ++made_;
    lk.unlock();
    auto r = std::make_unique<Run>();
    const cudaError_t e = cudaStreamCreateWithFlags(&r->s, cudaStreamNonBlocking);
    if (e != cudaSuccess) {
      {
        std::lock_guard g(pool_m_);
        --made_;
      }
      pool_cv_.notify_one();
      ck(e, "cudaStreamCreate");
    }
    return r;
  }

  // A run back to the pool, nothing of it still in flight (a run an error cut short may have left copies queued on
  // its pinned memory).
  void release(std::unique_ptr<Run> r) const {
    cudaStreamSynchronize(r->s);
    {
      std::lock_guard lk(pool_m_);
      idle_.push_back(std::move(r));
    }
    pool_cv_.notify_one();
  }

  static size_t ints_of(const Pass& P) {
    return 6 * P.tok.size() + P.last.size() + 2 * P.prow.size();  // tok, at, runs (4 each), last, prow, ptgt
  }

  void reserve(Run& j) const {
    const size_t d = static_cast<size_t>(d_), rows = std::max(j.shared.tok.size(), j.rest.tok.size());
    const size_t head = static_cast<size_t>(std::max(j.shared.head_rows(), j.rest.head_rows()));
    j.cache.reserve(j.shared.tok.size() * 3 * d * layers_.size());
    j.x.reserve(rows * d);
    j.xa.reserve(rows * d);
    j.xb.reserve(rows * d);
    j.xg.reserve(rows * d);
    j.h16.reserve(rows * d);
    j.att.reserve(rows * d);
    j.ff.reserve(rows * h_.ffn);
    j.qkv.reserve(rows * 3 * d);
    j.logits.reserve(head * static_cast<size_t>(tp_));
    j.ints_s.reserve(ints_of(j.shared));
    j.ints_r.reserve(ints_of(j.rest));
    j.stage_s.reserve(ints_of(j.shared));
    j.stage_r.reserve(ints_of(j.rest));
    j.picks_s.reserve(j.shared.prow.size());
    j.picks_r.reserve(j.rest.prow.size());
    j.back_s.reserve(j.shared.prow.size());
    j.back_r.reserve(j.rest.prow.size());
  }

  // Enqueues one pass of run j on its stream. keep: where its q, k, v go, a layer after another (the shared pass's,
  // the cache); cache: the shared pass's, each row's first run of keys. Gives the pinned memory its picks will be in
  // after the wait.
  const float* enqueue(Run& j, const Pass& P, Buf<int32_t>& ints, Pinned<int32_t>& stage, Buf<float>& picks,
                       Pinned<float>& back, float* keep, const float* cache) const {
    const int rows = static_cast<int>(P.tok.size()), d = d_, heads = static_cast<int>(h_.heads);
    const int ffn = static_cast<int>(h_.ffn), T = static_cast<int>(h_.tokens);
    const int n = P.head_rows(), np = static_cast<int>(P.prow.size());
    // the tokens, positions, runs, last rows and picks: one copy
    int32_t* h = stage.p;
    int32_t* dv = ints.p;
    size_t o = 0;
    auto put = [&](const void* src, size_t count) {
      if (count) std::memcpy(h + o, src, count * sizeof(int32_t));
      const int32_t* at = dv + o;
      o += count;
      return at;
    };
    const int32_t* tok = put(P.tok.data(), P.tok.size());
    const int32_t* at = put(P.at.data(), P.at.size());
    const Runs* runs = reinterpret_cast<const Runs*>(put(P.runs.data(), 4 * P.runs.size()));
    const int32_t* last = put(P.last.data(), P.last.size());
    const int32_t* prow = put(P.prow.data(), P.prow.size());
    const int32_t* ptgt = put(P.ptgt.data(), P.ptgt.size());
    ck(cudaMemcpyAsync(dv, h, o * sizeof(int32_t), cudaMemcpyHostToDevice, j.s), "the rows' upload");
    const size_t layer = static_cast<size_t>(rows) * 3 * d;
    const size_t cache_layer = j.shared.tok.size() * 3 * static_cast<size_t>(d);
    // q: the quantization of what the kernel writes in FP16, the next product's input (INT8's; 0, 0 otherwise)
    // sc: INT8's sums' scales (null otherwise)
    auto gemm = [&](int M, const __half* A, const __half* Bw, const float* sc, int N, int K, const float* bias,
                    const float* residual, bool gelu, float* o32, __half* o16, const float* q = nullptr) {
      const dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
      gemm_k<<<grid, kThreads, 0, j.s>>>(A, Bw, M, N, K, sc, bias, residual, gelu, o32, q ? q[0] : 0.0f,
                                         q ? q[1] : 0.0f, o16);
    };
    auto attend = [&](int M, const float* qkv, const int32_t* qrow, const float* buf1, const float* buf2,
                      const float* q) {
      const int blocks = (M * heads + kWarps - 1) / kWarps;
      const float scale = 1.0f / sqrtf(static_cast<float>(dh_));
      if (dh_ == 32) {
        attend_k<32><<<blocks, kThreads, 0, j.s>>>(qkv, qrow, buf1, buf2, runs, M, d, heads, scale, q[0], q[1],
                                                   j.att.p);
      } else if (dh_ == 64) {
        attend_k<64><<<blocks, kThreads, 0, j.s>>>(qkv, qrow, buf1, buf2, runs, M, d, heads, scale, q[0], q[1],
                                                   j.att.p);
      } else {
        attend_k<128><<<blocks, kThreads, 0, j.s>>>(qkv, qrow, buf1, buf2, runs, M, d, heads, scale, q[0], q[1],
                                                   j.att.p);
      }
    };
    auto warps = [](int M) { return (M + kWarps - 1) / kWarps; };
    embed_k<<<rows, 32, 0, j.s>>>(emb_.p, pos_.p, tok, at, rows, d, j.x.p);
    float* cur = j.x.p;
    auto other = [&](const float* c) { return c == j.xb.p ? j.xa.p : j.xb.p; };
    for (size_t l = 0; l < layers_.size(); ++l) {
      const LayerW& L = layers_[l];
      float* qkv = keep ? keep + l * layer : j.qkv.p;
      layer_norm_k<<<warps(rows), kThreads, 0, j.s>>>(cur, rows, d, L.ln1w.p, L.ln1b.p, L.qqkv[0], L.qqkv[1], j.h16.p);
      gemm(rows, j.h16.p, L.wqkv.p, L.sqkv.p, 3 * d, d, L.bqkv.p, nullptr, false, qkv, nullptr);
      // a row's keys: the shared pass's, among its own rows; the rest pass's, the cache's and then its own
      const float* buf1 = cache ? cache + l * cache_layer : qkv;
      int M = rows;
      const int32_t* qrow = nullptr;
      if (l + 1 == layers_.size() && !P.all) {  // the last layer: the asked rows alone past q, k, v
        gather_k<<<n, 32, 0, j.s>>>(cur, last, n, d, j.xg.p);
        cur = j.xg.p;
        M = n;
        qrow = last;
      }
      attend(M, qkv, qrow, buf1, qkv, L.qo);
      float* nxt = other(cur);
      gemm(M, j.att.p, L.wo.p, L.so.p, d, d, L.bo.p, cur, false, nxt, nullptr);
      cur = nxt;
      layer_norm_k<<<warps(M), kThreads, 0, j.s>>>(cur, M, d, L.ln2w.p, L.ln2b.p, L.qw1[0], L.qw1[1], j.h16.p);
      gemm(M, j.h16.p, L.w1.p, L.sw1.p, ffn, d, L.b1.p, nullptr, true, nullptr, j.ff.p, L.qw2);
      nxt = other(cur);
      gemm(M, j.ff.p, L.w2.p, L.sw2.p, d, ffn, L.b2.p, cur, false, nxt, nullptr);
      cur = nxt;
    }
    layer_norm_k<<<warps(n), kThreads, 0, j.s>>>(cur, n, d, lnfw_.p, lnfb_.p, qhead_[0], qhead_[1], j.h16.p);
    gemm(n, j.h16.p, head_.p, shead_.p, tp_, d, nullptr, nullptr, false, j.logits.p, nullptr);
    pick_k<<<warps(np), kThreads, 0, j.s>>>(j.logits.p, T, tp_, prow, ptgt, np, picks.p);
    ck(cudaGetLastError(), "the transformer's kernels");
    ck(cudaMemcpyAsync(back.p, picks.p, static_cast<size_t>(np) * sizeof(float), cudaMemcpyDeviceToHost, j.s),
       "the picks");
    return back.p;
  }

  int gpu_ = 0, d_ = 0, dh_ = 0, tp_ = 0;
  std::shared_ptr<TransformerLM> cpu_;  // the file's header
  TlmHeader h_{};
  Buf<float> emb_, pos_, lnfw_, lnfb_;
  Buf<__half> head_;
  Buf<float> shead_;     // INT8's: the head's sums' scales, padded to tp_
  float qhead_[2] = {};  // the head's input's quantization (INT8's)
  std::vector<LayerW> layers_;
  // the runs: idle ones, and how many are made
  mutable std::mutex pool_m_;
  mutable std::condition_variable pool_cv_;
  mutable std::vector<std::unique_ptr<Run>> idle_;
  mutable size_t made_ = 0;
};

}  // namespace

std::shared_ptr<LineRescorer> cuda_transformer_lm(const std::filesystem::path& file, int gpu) {
  return std::make_shared<CudaTlm>(file, gpu);
}

}  // namespace carbon
