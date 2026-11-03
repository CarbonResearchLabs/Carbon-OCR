// The GPU pipeline's own kernels (carbon/cuda.h): gray pixels to a model's ink map,
// and float32 to float16.  Four pixels a thread from a 32-bit load where the
// buffers are aligned, a grid-stride loop, so any n runs in one launch.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "carbon/cuda.h"
#include "carbon/error.h"

namespace carbon::cuda {

namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw Error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e));
}

int blocks_for(int64_t n, int threads) {
  const int64_t b = (n + threads - 1) / threads;
  return static_cast<int>(b < 4096 ? (b < 1 ? 1 : b) : 4096);
}

__device__ __forceinline__ float ink(uint8_t v) { return 1.0f - static_cast<float>(v) / 255.0f; }

// four pixels a thread: one 32-bit load, two half2 stores
__global__ void ink_half4(const uchar4* __restrict__ in, __half2* __restrict__ out, int64_t n4) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n4;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const uchar4 v = in[i];
    out[2 * i] = __floats2half2_rn(ink(v.x), ink(v.y));
    out[2 * i + 1] = __floats2half2_rn(ink(v.z), ink(v.w));
  }
}

__global__ void ink_float4(const uchar4* __restrict__ in, float4* __restrict__ out, int64_t n4) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n4;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const uchar4 v = in[i];
    out[i] = make_float4(ink(v.x), ink(v.y), ink(v.z), ink(v.w));
  }
}

// the rest (or an unaligned buffer), a pixel a thread
__global__ void ink_one(const uint8_t* __restrict__ in, void* __restrict__ out, int64_t from, int64_t n, bool fp16) {
  for (int64_t i = from + blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    if (fp16) {
      static_cast<__half*>(out)[i] = __float2half_rn(ink(in[i]));
    } else {
      static_cast<float*>(out)[i] = ink(in[i]);
    }
  }
}

__global__ void half_of(const float* __restrict__ in, __half* __restrict__ out, int64_t n) {
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = __float2half_rn(in[i]);
  }
}

}  // namespace

void ink_map(const uint8_t* in, int64_t n, void* out, bool fp16, void* stream) {
  if (n <= 0) return;
  const auto s = static_cast<cudaStream_t>(stream);
  constexpr int kThreads = 256;
  const bool aligned = reinterpret_cast<uintptr_t>(in) % 4 == 0 &&
                       reinterpret_cast<uintptr_t>(out) % (fp16 ? 8 : 16) == 0;
  int64_t done = 0;
  if (aligned) {
    const int64_t n4 = n / 4;
    if (n4) {
      if (fp16) {
        ink_half4<<<blocks_for(n4, kThreads), kThreads, 0, s>>>(reinterpret_cast<const uchar4*>(in),
                                                                static_cast<__half2*>(out), n4);
      } else {
        ink_float4<<<blocks_for(n4, kThreads), kThreads, 0, s>>>(reinterpret_cast<const uchar4*>(in),
                                                                 static_cast<float4*>(out), n4);
      }
    }
    done = n4 * 4;
  }
  if (done < n) ink_one<<<blocks_for(n - done, kThreads), kThreads, 0, s>>>(in, out, done, n, fp16);
  check(cudaGetLastError(), "ink_map");
}

void to_half(const float* in, void* out, int64_t n, void* stream) {
  if (n <= 0) return;
  constexpr int kThreads = 256;
  half_of<<<blocks_for(n, kThreads), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(in, static_cast<__half*>(out),
                                                                                       n);
  check(cudaGetLastError(), "to_half");
}

}  // namespace carbon::cuda
