// The fused CTC head gives what the plain computation gives: the logits
// W x F + b in double from the same fp16 values, then argmax and softmax.
// Skipped without a GPU.

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "carbon/cuda.h"

namespace {

template <class T>
struct Device {
  T* p = nullptr;
  explicit Device(size_t n) { EXPECT_EQ(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T)), cudaSuccess); }
  explicit Device(const std::vector<T>& h) : Device(h.size()) {
    EXPECT_EQ(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
  }
  ~Device() { cudaFree(p); }
  std::vector<T> get(size_t n) const {
    std::vector<T> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
  }
};

struct Case {
  int n, k, classes, frames;
};

// fp32 accumulation must agree to 1e-3; fp16 accumulation (chunks of 64
// products) moves the logits by about that much, so it is held to 1e-2.
class CudaCtcHead : public ::testing::TestWithParam<bool> {};

TEST_P(CudaCtcHead, MatchesTheUnfusedHead) {
  if (!carbon::cuda::available()) GTEST_SKIP() << "no CUDA device";
  const bool half_acc = GetParam();
  const double tol = half_acc ? 1e-2 : 1e-3;
  std::mt19937 rng(5);
  // k of every recognizer; classes that are and are not multiples of 32; a line shorter than a warp's 16 frames
  for (const Case c : {Case{5, 256, 800, 77}, Case{3, 192, 1500, 40}, Case{2, 128, 33, 200}, Case{4, 320, 400, 9}}) {
    std::normal_distribution<float> gf(0.0f, 1.0f), gw(0.0f, 0.06f), gb(0.0f, 0.5f);
    std::vector<__half> f(static_cast<size_t>(c.n) * c.k * c.frames), w(static_cast<size_t>(c.classes) * c.k);
    std::vector<float> bias(c.classes);
    for (auto& v : f) v = __float2half(gf(rng));
    for (auto& v : w) v = __float2half(gw(rng));
    for (auto& v : bias) v = gb(rng);

    Device<__half> df(f), dw(w);
    Device<float> db(bias), dp(static_cast<size_t>(c.n) * c.frames);
    Device<int32_t> di(static_cast<size_t>(c.n) * c.frames);
    carbon::cuda::ctc_head_argmax(df.p, dw.p, db.p, c.n, c.k, c.classes, c.frames, di.p, dp.p, nullptr, half_acc);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto ids = di.get(static_cast<size_t>(c.n) * c.frames);
    const auto probs = dp.get(static_cast<size_t>(c.n) * c.frames);

    int near_ties = 0;
    std::vector<double> logit(c.classes);
    for (int i = 0; i < c.n; ++i) {
      for (int t = 0; t < c.frames; ++t) {
        int best = 0;
        for (int cl = 0; cl < c.classes; ++cl) {
          double acc = bias[cl];
          for (int k = 0; k < c.k; ++k) {
            acc += static_cast<double>(__half2float(w[static_cast<size_t>(cl) * c.k + k])) *
                   __half2float(f[(static_cast<size_t>(i) * c.k + k) * c.frames + t]);
          }
          logit[cl] = acc;
          if (acc > logit[best]) best = cl;
        }
        double sum = 0;
        for (double v : logit) sum += std::exp(v - logit[best]);
        const size_t at = static_cast<size_t>(i) * c.frames + t;
        if (ids[at] != best) {
          // fp32 accumulation in another order may flip a near tie, and only a near tie
          ASSERT_GE(ids[at], 0);
          ASSERT_LT(ids[at], c.classes);
          EXPECT_LT(logit[best] - logit[ids[at]], tol) << "k " << c.k << " line " << i << " frame " << t;
          ++near_ties;
        } else {
          EXPECT_NEAR(probs[at], 1.0 / sum, tol / sum) << "k " << c.k << " line " << i << " frame " << t;
        }
      }
    }
    EXPECT_LE(near_ties, c.n * c.frames / (half_acc ? 20 : 100) + 1) << "k " << c.k;
  }
}

INSTANTIATE_TEST_SUITE_P(Accumulate, CudaCtcHead, ::testing::Values(false, true),
                         [](const auto& p) { return p.param ? "Fp16Chunks" : "Fp32"; });

}  // namespace
