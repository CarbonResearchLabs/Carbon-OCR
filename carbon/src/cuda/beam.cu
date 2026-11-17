// The beam's candidates on the GPU (carbon/cuda.h, carbon/beam.h): for the few lines of a recognition run the
// beam reads again, each frame's likeliest classes and the blank, so only those come back to the host
// (about 5.5 KB a 128-frame line, against 400 KB of its logits).
//
// ctc_topk: one thread a frame of a line, its logits read class by class (the warp's threads on neighbouring
// frames, so each class's read is coalesced): the frame's best, its sum of exp(logit - best), then its top k but
// the blank by insertion (a later class must beat a kept one, so ties keep the lower id), as cpu::ctc_topk.  The
// GPU's exp is not the CPU's, so the log-probabilities agree to the last few bits and the ids but for near ties.
//
// ctc_head_logits: the fused head's features of those lines (ctc_head_argmax's) to their logits, one thread a
// logit, its sum over k in float32 in k's order: within about 1e-3 of the tensor cores' sums (which the fused
// argmax reads), so a frame's candidates may differ from its greedy class at a near tie.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <string>

#include "carbon/cuda.h"
#include "carbon/error.h"

namespace carbon::cuda {

namespace {

constexpr int kMaxK = 16;
constexpr int kThreads = 128;

__global__ void topk_kernel(const float* __restrict__ logits, int classes, int frames,
                            const int32_t* __restrict__ rows, const int32_t* __restrict__ lengths, int n_rows, int k,
                            int32_t* __restrict__ ids, float* __restrict__ logp, float* __restrict__ blank) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y;
  if (j >= n_rows || t >= lengths[j] || t >= frames) return;
  const float* l = logits + static_cast<size_t>(rows[j]) * classes * frames + t;
  float best = -INFINITY;
  for (int c = 0; c < classes; ++c) best = fmaxf(best, l[static_cast<size_t>(c) * frames]);
  float sum = 0.0f;
  for (int c = 0; c < classes; ++c) sum += expf(l[static_cast<size_t>(c) * frames] - best);
  float v[kMaxK];
  int32_t id[kMaxK];
  for (int q = 0; q < k; ++q) {
    v[q] = -INFINITY;
    id[q] = 0;
  }
  if (k > 0) {
    for (int c = 1; c < classes; ++c) {
      const float x = l[static_cast<size_t>(c) * frames];
      if (!(x > v[k - 1])) continue;
      int q = k - 1;
      for (; q > 0 && v[q - 1] < x; --q) {
        v[q] = v[q - 1];
        id[q] = id[q - 1];
      }
      v[q] = x;
      id[q] = c;
    }
  }
  const float off = best + logf(sum);
  const size_t o = (static_cast<size_t>(j) * frames + t) * k;
  for (int q = 0; q < k; ++q) {
    ids[o + q] = id[q];
    logp[o + q] = v[q] - off;
  }
  blank[static_cast<size_t>(j) * frames + t] = l[0] - off;
}

__global__ void head_logits_kernel(const __half* __restrict__ features, const __half* __restrict__ weight,
                                   const float* __restrict__ bias, const int32_t* __restrict__ rows, int k,
                                   int classes, int frames, float* __restrict__ logits) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.y, j = blockIdx.z;
  if (t >= frames) return;
  const __half* f = features + static_cast<size_t>(rows[j]) * k * frames + t;
  const __half* w = weight + static_cast<size_t>(c) * k;
  float acc = 0.0f;
  for (int q = 0; q < k; ++q) acc = fmaf(__half2float(w[q]), __half2float(f[static_cast<size_t>(q) * frames]), acc);
  logits[(static_cast<size_t>(j) * classes + c) * frames + t] = acc + bias[c];
}

void check_launch(const char* what) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) throw Error(std::string("CUDA: ") + what + ": " + cudaGetErrorString(e));
}

}  // namespace

void ctc_topk(const float* logits, int classes, int frames, const int32_t* rows, const int32_t* lengths, int n_rows,
              int k, int32_t* ids, float* logp, float* blank, void* stream) {
  if (n_rows <= 0 || frames <= 0) return;
  if (k < 0 || k > kMaxK || k > classes - 1) {
    throw Error("cuda::ctc_topk: k = " + std::to_string(k) + " (0 to " + std::to_string(kMaxK) + ", under " +
                std::to_string(classes) + " classes)");
  }
  if (n_rows > 65535) throw Error("cuda::ctc_topk: more than 65535 lines a call");
  const dim3 grid((frames + kThreads - 1) / kThreads, n_rows);
  topk_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(logits, classes, frames, rows, lengths,
                                                                        n_rows, k, ids, logp, blank);
  check_launch("ctc_topk");
}

void ctc_head_logits(const void* features, const void* weight, const float* bias, const int32_t* rows, int n_rows,
                     int k, int classes, int frames, float* logits, void* stream) {
  if (n_rows <= 0 || frames <= 0 || classes <= 0) return;
  if (classes > 65535 || n_rows > 65535) throw Error("cuda::ctc_head_logits: more than 65535 classes or lines");
  const dim3 grid((frames + kThreads - 1) / kThreads, classes, n_rows);
  head_logits_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __half*>(features), static_cast<const __half*>(weight), bias, rows, k, classes, frames,
      logits);
  check_launch("ctc_head_logits");
}

}  // namespace carbon::cuda
