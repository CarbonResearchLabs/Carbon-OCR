// The beam's candidates on the GPU (carbon/cuda.h, src/cuda/beam.cu) against the CPU's (cpu::ctc_topk): the same
// classes but at near ties, their log-probabilities to float32's last bits; the fused head's logits against the
// plain sum in double. Skipped without a GPU.

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "carbon/cpu.h"
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

TEST(CudaBeam, CandidatesAreTheCpus) {
  if (!carbon::cuda::available()) GTEST_SKIP() << "no CUDA device";
  std::mt19937 rng(41);
  const int n = 6, classes = 801, frames = 70;
  std::normal_distribution<float> g(0.0f, 3.0f);
  std::vector<float> logits(static_cast<size_t>(n) * classes * frames);
  for (float& v : logits) v = g(rng);
  const std::vector<int32_t> rows = {4, 1, 5}, lengths = {70, 33, 1};
  for (const int k : {0, 1, 5, 16}) {
    Device<float> dl(logits), dlp(static_cast<size_t>(rows.size()) * frames * std::max(k, 1)),
        db(rows.size() * static_cast<size_t>(frames));
    Device<int32_t> dr(rows), dn(lengths), di(static_cast<size_t>(rows.size()) * frames * std::max(k, 1));
    carbon::cuda::ctc_topk(dl.p, classes, frames, dr.p, dn.p, static_cast<int>(rows.size()), k, di.p, dlp.p, db.p);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto ids = di.get(rows.size() * frames * k);
    const auto lp = dlp.get(rows.size() * frames * k);
    const auto bl = db.get(rows.size() * frames);
    int near_ties = 0;
    for (size_t j = 0; j < rows.size(); ++j) {
      const int t_n = lengths[j];
      std::vector<int32_t> want_ids(static_cast<size_t>(t_n) * k);
      std::vector<float> want_lp(want_ids.size()), want_bl(t_n);
      carbon::cpu::ctc_topk(logits.data() + static_cast<size_t>(rows[j]) * classes * frames, classes, frames, t_n, k,
                            want_ids.data(), want_lp.data(), want_bl.data());
      for (int t = 0; t < t_n; ++t) {
        EXPECT_NEAR(bl[j * frames + t], want_bl[t], 1e-4) << "line " << j << " frame " << t;
        for (int q = 0; q < k; ++q) {
          const size_t got = (j * frames + t) * k + q, want = static_cast<size_t>(t) * k + q;
          EXPECT_NEAR(lp[got], want_lp[want], 1e-4) << "line " << j << " frame " << t << " rank " << q;
          if (ids[got] != want_ids[want]) ++near_ties;  // only where two logits are as good as equal
        }
      }
    }
    EXPECT_EQ(near_ties, 0) << "k " << k;
  }
}

TEST(CudaBeam, HeadLogitsAreTheHeadsSums) {
  if (!carbon::cuda::available()) GTEST_SKIP() << "no CUDA device";
  std::mt19937 rng(43);
  const int n = 4, k = 192, classes = 513, frames = 45;
  std::normal_distribution<float> gf(0.0f, 1.0f), gw(0.0f, 0.06f), gb(0.0f, 0.5f);
  std::vector<__half> f(static_cast<size_t>(n) * k * frames), w(static_cast<size_t>(classes) * k);
  std::vector<float> bias(classes);
  for (auto& v : f) v = __float2half(gf(rng));
  for (auto& v : w) v = __float2half(gw(rng));
  for (auto& v : bias) v = gb(rng);
  const std::vector<int32_t> rows = {3, 0};
  Device<__half> df(f), dw(w);
  Device<float> dbias(bias), dl(rows.size() * classes * static_cast<size_t>(frames));
  Device<int32_t> dr(rows);
  carbon::cuda::ctc_head_logits(df.p, dw.p, dbias.p, dr.p, static_cast<int>(rows.size()), k, classes, frames, dl.p);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  const auto got = dl.get(rows.size() * classes * frames);
  for (size_t j = 0; j < rows.size(); ++j) {
    for (int c = 0; c < classes; c += 7) {
      for (int t = 0; t < frames; ++t) {
        double want = bias[c];
        for (int q = 0; q < k; ++q) {
          want += double(__half2float(w[static_cast<size_t>(c) * k + q])) *
                  __half2float(f[(static_cast<size_t>(rows[j]) * k + q) * frames + t]);
        }
        EXPECT_NEAR(got[(j * classes + c) * frames + t], want, 1e-3) << "line " << j << " class " << c;
      }
    }
  }
}

}  // namespace
