// The CTC head fused with greedy decoding (carbon/cuda.h): logits = W x F + b on
// the tensor cores, reduced to each frame's best class and its softmax
// probability as they are made.  The [n, classes, frames] logits, the largest
// tensor of a recognizer (0.73 GB for a batch of 256 rec-cjk lines), are never
// written.
//
// One block is 8 warps; each warp owns 16 frames of one line, taken from one list
// of every line's 16-frame groups, so a block's warps may serve several short
// lines and every warp of every block has work.  A warp loads its
// slice of the features once, into registers, as the B operands of mma.sync
// m16n8k16 (f16 in, f32 accumulate), then walks every class: the weights stream
// through shared memory 32 classes at a time (cp.async, double-buffered, read
// with ldmatrix), and each thread keeps an online (max, argmax, sum of exp) for
// the frames its accumulators hold.  Warp shuffles merge the threads' partial
// results at the end.  Classes are visited in increasing order and merges
// prefer the lower class, so ties go to the lower id, as in carbon::ctc_greedy.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <string>

#include "carbon/cuda.h"
#include "carbon/error.h"

namespace carbon::cuda {

namespace {

constexpr int kWarps = 8;
constexpr int kFramesPerWarp = 16;  // two n8 tiles
constexpr int kClassTile = 32;      // two m16 tiles per shared-memory stage

__device__ __forceinline__ uint32_t pack(__half lo, __half hi) {
  return static_cast<uint32_t>(__half_as_ushort(lo)) | (static_cast<uint32_t>(__half_as_ushort(hi)) << 16);
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const __half* p) {
  const auto a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The same product with fp16 accumulation: twice the tensor-core rate of fp32
// accumulation on GeForce GPUs.  Used for short chunks of K only (below).
__device__ __forceinline__ void mma_16816_f16(uint32_t (&d)[2], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%0, %1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void cp_async16(__half* smem, const __half* gmem) {
  const auto s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// A running softmax over the classes seen: the max, the sum of exp(v - max), the argmax.
struct Best {
  float m, s;
  int a;
};

__device__ __forceinline__ void see(Best& b, float v, int c) {
  if (v > b.m) {
    b.s = b.s * __expf(b.m - v) + 1.0f;
    b.m = v;
    b.a = c;
  } else {
    b.s += __expf(v - b.m);
  }
}

// Order-free: both lanes of a shuffle pair get the same result.
__device__ __forceinline__ Best merge(Best x, Best y) {
  if (y.m > x.m || (y.m == x.m && y.a < x.a)) {
    const Best t = x;
    x = y;
    y = t;
  }
  if (y.s > 0.0f) x.s += y.s * __expf(y.m - x.m);
  return x;
}

// kHalfAcc: each 64 products (4 k steps) are summed in fp16 by the tensor cores,
// and the chunks in fp32; otherwise every product is accumulated in fp32.
template <int K, bool kHalfAcc>
__global__ void __launch_bounds__(kWarps * 32)
    ctc_head_kernel(const __half* __restrict__ features, const __half* __restrict__ weight,
                    const float* __restrict__ bias, int classes, int frames, int groups_per_line,
                    int groups, int32_t* __restrict__ ids, float* __restrict__ probs) {
  constexpr int kSteps = K / 16;
  constexpr int kRow = K + 8;  // halves; the padding keeps ldmatrix free of bank conflicts
  __shared__ __align__(16) __half tile[2][kClassTile * kRow];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int group = blockIdx.x * kWarps + warp;
  const bool active = group < groups;  // idle warps still load their share of the weights
  const int line = active ? group / groups_per_line : 0;
  const int f0 = active ? (group % groups_per_line) * kFramesPerWarp : frames;
  const __half* f_line = features + static_cast<size_t>(line) * K * frames;

  // This warp's B operands for all of K, loaded once: thread lane holds frame
  // f0 + nt * 8 + lane / 4, features k and k + 1 (k = 16 ks + 2 (lane % 4)), and k + 8, k + 9.
  uint32_t b[kSteps][2][2];
#pragma unroll
  for (int ks = 0; ks < kSteps; ++ks) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int f = f0 + nt * 8 + lane / 4;
      const int k = ks * 16 + (lane % 4) * 2;
      const __half z = __float2half(0.0f);
      const bool in = f < frames;
      const __half* p = f_line + static_cast<size_t>(k) * frames + f;
      b[ks][nt][0] = pack(in ? p[0] : z, in ? p[frames] : z);
      b[ks][nt][1] = pack(in ? p[8 * static_cast<size_t>(frames)] : z, in ? p[9 * static_cast<size_t>(frames)] : z);
    }
  }

  auto load = [&](int stage, int c0) {
    __half* dst = tile[stage];
    constexpr int kChunks = kClassTile * (K / 8);  // 16-byte chunks
    for (int i = threadIdx.x; i < kChunks; i += kWarps * 32) {
      const int r = i / (K / 8), ch = i % (K / 8);
      __half* d = dst + r * kRow + ch * 8;
      if (c0 + r < classes) {
        cp_async16(d, weight + static_cast<size_t>(c0 + r) * K + ch * 8);
      } else {
        *reinterpret_cast<uint4*>(d) = make_uint4(0, 0, 0, 0);
      }
    }
    cp_async_commit();
  };

  Best best[2][2];  // [n tile][the thread's two frames in it]
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) best[nt][0] = best[nt][1] = {-INFINITY, 0.0f, 0};

  const int tiles = (classes + kClassTile - 1) / kClassTile;
  load(0, 0);
  for (int t = 0; t < tiles; ++t) {
    if (t + 1 < tiles) {
      load((t + 1) & 1, (t + 1) * kClassTile);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    const __half* w = tile[t & 1];
#pragma unroll
    for (int mt = 0; mt < kClassTile / 16; ++mt) {
      float d[2][4] = {};
      if constexpr (kHalfAcc) {
        constexpr int kChunk = 4;
        static_assert(kSteps % kChunk == 0, "K must be a multiple of 64 for fp16 accumulation");
#pragma unroll
        for (int kc = 0; kc < kSteps; kc += kChunk) {
          uint32_t h[2][2] = {};
#pragma unroll
          for (int ks = kc; ks < kc + kChunk; ++ks) {
            uint32_t a[4];
            ldmatrix_x4(a, w + (mt * 16 + lane % 16) * kRow + ks * 16 + (lane / 16) * 8);
            mma_16816_f16(h[0], a, b[ks][0]);
            mma_16816_f16(h[1], a, b[ks][1]);
          }
#pragma unroll
          for (int nt = 0; nt < 2; ++nt) {  // h[nt][0]: the class c_lo pair, h[nt][1]: c_lo + 8
            const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&h[nt][0]));
            const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&h[nt][1]));
            d[nt][0] += lo.x;
            d[nt][1] += lo.y;
            d[nt][2] += hi.x;
            d[nt][3] += hi.y;
          }
        }
      } else {
#pragma unroll
        for (int ks = 0; ks < kSteps; ++ks) {
          uint32_t a[4];
          ldmatrix_x4(a, w + (mt * 16 + lane % 16) * kRow + ks * 16 + (lane / 16) * 8);
          mma_16816(d[0], a, b[ks][0]);
          mma_16816(d[1], a, b[ks][1]);
        }
      }
      // d[nt][0, 1]: class c_lo, frames 2 (lane % 4) + {0, 1}; d[nt][2, 3]: class c_lo + 8.
      const int c_lo = t * kClassTile + mt * 16 + lane / 4, c_hi = c_lo + 8;
      if (c_lo < classes) {
        const float bl = __ldg(bias + c_lo);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          see(best[nt][0], d[nt][0] + bl, c_lo);
          see(best[nt][1], d[nt][1] + bl, c_lo);
        }
      }
      if (c_hi < classes) {
        const float bh = __ldg(bias + c_hi);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          see(best[nt][0], d[nt][2] + bh, c_hi);
          see(best[nt][1], d[nt][3] + bh, c_hi);
        }
      }
    }
    __syncthreads();  // this stage is reloaded two tiles on
  }

  // The 8 lanes that share lane % 4 hold the same frames: merge them.
#pragma unroll
  for (int nt = 0; nt < 2; ++nt) {
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      Best x = best[nt][j];
#pragma unroll
      for (int off = 4; off < 32; off <<= 1) {
        const Best y{__shfl_xor_sync(0xffffffffu, x.m, off), __shfl_xor_sync(0xffffffffu, x.s, off),
                     __shfl_xor_sync(0xffffffffu, x.a, off)};
        x = merge(x, y);
      }
      const int f = f0 + nt * 8 + (lane % 4) * 2 + j;
      if (lane < 4 && f < frames) {
        ids[static_cast<size_t>(line) * frames + f] = x.a;
        probs[static_cast<size_t>(line) * frames + f] = 1.0f / x.s;
      }
    }
  }
}

template <int K, bool kHalfAcc>
void launch(const void* features, const void* weight, const float* bias, int n, int classes, int frames,
            int32_t* ids, float* probs, cudaStream_t s) {
  const int per_line = (frames + kFramesPerWarp - 1) / kFramesPerWarp;
  const int groups = n * per_line;
  const int blocks = (groups + kWarps - 1) / kWarps;
  ctc_head_kernel<K, kHalfAcc><<<blocks, kWarps * 32, 0, s>>>(static_cast<const __half*>(features),
                                                    static_cast<const __half*>(weight), bias, classes, frames,
                                                    per_line, groups, ids, probs);
}

}  // namespace

void ctc_head_argmax(const void* features, const void* weight, const float* bias, int n, int k, int classes,
                     int frames, int32_t* ids, float* probs, void* stream, bool fp16_accumulate) {
  if (n <= 0 || frames <= 0) return;
  const auto s = static_cast<cudaStream_t>(stream);
#define CARBON_HEAD(K)                                                                      (fp16_accumulate ? launch<K, true>(features, weight, bias, n, classes, frames, ids, probs, s)                    : launch<K, false>(features, weight, bias, n, classes, frames, ids, probs, s))
  switch (k) {
    case 128: CARBON_HEAD(128); break;
    case 192: CARBON_HEAD(192); break;
    case 256: CARBON_HEAD(256); break;
    case 320: CARBON_HEAD(320); break;
    default: throw Error("cuda::ctc_head_argmax: k = " + std::to_string(k) + " (128, 192, 256 or 320)");
  }
#undef CARBON_HEAD
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw Error(std::string("CUDA: ctc_head_argmax: ") + cudaGetErrorString(e));
}

}  // namespace carbon::cuda
