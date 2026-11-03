// The CUDA crop rule and CTC decoding give what the CPU reference gives: the
// same crop bytes, the same character ids and frames.  Skipped without a GPU.

#include "carbon/cuda.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "carbon/lines.h"

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

// A page with structure: a gradient, noise, and dark bars, so every filter tap matters.
std::vector<uint8_t> page(int w, int h, std::mt19937& rng) {
  std::vector<uint8_t> px(static_cast<size_t>(w) * h);
  std::uniform_int_distribution<int> noise(-30, 30);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      int v = 200 + (x * 37 + y * 11) % 56 + noise(rng);
      if ((y / 9) % 3 == 1 && (x / 5) % 4 != 0) v = (x * 13 + y * 7) % 90;
      px[static_cast<size_t>(y) * w + x] = static_cast<uint8_t>(std::clamp(v, 0, 255));
    }
  return px;
}

TEST(CudaLines, CropsAreTheCpuBytes) {
  if (!carbon::cuda::available()) GTEST_SKIP() << "no CUDA device";
  std::mt19937 rng(7);
  const int pw[2] = {1700, 900}, ph[2] = {2200, 640};
  std::vector<std::vector<uint8_t>> pages = {page(pw[0], ph[0], rng), page(pw[1], ph[1], rng)};
  for (uint8_t& v : pages[1]) v = static_cast<uint8_t>(255 - v);  // light on dark: its crops are flipped
  std::uniform_real_distribution<double> u(0, 1);
  std::vector<carbon::LineBox> boxes;
  std::vector<int32_t> page_of;
  for (int i = 0; i < 300; ++i) {
    const int p = i % 2;
    const double h = 6 + 140 * u(rng) * u(rng);                     // mostly small, some headings
    const double x0 = -40 + (pw[p] + 40) * u(rng), y0 = -30 + (ph[p] + 30) * u(rng);
    const double w = std::min(1400.0, 4 + 30 * h * u(rng));          // some run off the page
    boxes.push_back({x0, y0, x0 + w, y0 + h});
    page_of.push_back(p);
  }
  int out_w = 8;
  for (const auto& b : boxes) out_w = std::max(out_w, carbon::crop_region(b).width);
  const int n = static_cast<int>(boxes.size()), height = 32;
  const size_t plane = static_cast<size_t>(height) * out_w;

  std::vector<uint8_t> cpu(n * plane), kept(n * plane);
  std::vector<int32_t> cpu_w(n);
  carbon::CropRule keep;
  keep.flip_margin = 0;
  int flipped = 0;
  for (int i = 0; i < n; ++i) {
    const int p = page_of[i];
    const carbon::GrayView v{pages[p].data(), pw[p], ph[p], pw[p]};
    carbon::crop_lines(v, {&boxes[i], 1}, out_w, cpu.data() + i * plane, {&cpu_w[i], 1});
    carbon::crop_lines(v, {&boxes[i], 1}, out_w, kept.data() + i * plane, {&cpu_w[i], 1}, keep);
    flipped += !std::equal(cpu.begin() + i * plane, cpu.begin() + (i + 1) * plane, kept.begin() + i * plane);
  }
  EXPECT_GT(flipped, n / 4);  // the inverted page's lines

  Device<uint8_t> d0(pages[0]), d1(pages[1]), out(n * plane);
  Device<int32_t> widths(n);
  const std::vector<carbon::cuda::DevicePage> dev = {{d0.p, pw[0], ph[0], pw[0]}, {d1.p, pw[1], ph[1], pw[1]}};
  carbon::cuda::crop_lines(dev, page_of, boxes, out_w, out.p, widths.p);
  EXPECT_EQ(widths.get(n), cpu_w);
  const std::vector<uint8_t> gpu = out.get(n * plane);
  size_t diff = 0;
  for (size_t i = 0; i < gpu.size(); ++i) diff += gpu[i] != cpu[i];
  EXPECT_EQ(diff, 0u) << "bytes that differ from the CPU crop";
}

TEST(CudaLines, CtcMatchesTheCpu) {
  if (!carbon::cuda::available()) GTEST_SKIP() << "no CUDA device";
  std::mt19937 rng(11);
  const int n = 37, classes = 800, frames = 131;
  std::normal_distribution<float> g(0.0f, 3.0f);
  std::vector<float> logits(static_cast<size_t>(n) * classes * frames);
  for (auto& v : logits) v = g(rng);
  for (int i = 0; i < n; ++i)  // a blank-heavy line, as a trained model gives
    for (int t = 0; t < frames; t += 2) logits[(static_cast<size_t>(i) * classes) * frames + t] += 8.0f;
  std::vector<int32_t> lengths(n);
  for (int i = 0; i < n; ++i) lengths[i] = std::uniform_int_distribution<int>(0, frames)(rng);

  const auto cpu = carbon::ctc_greedy(logits.data(), n, classes, frames, lengths);
  Device<float> d(logits);
  const auto gpu = carbon::cuda::ctc_greedy(d.p, n, classes, frames, lengths);
  ASSERT_EQ(gpu.size(), cpu.size());
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(gpu[i].ids, cpu[i].ids) << "line " << i;
    EXPECT_EQ(gpu[i].frames, cpu[i].frames) << "line " << i;
    ASSERT_EQ(gpu[i].probs.size(), cpu[i].probs.size());
    for (size_t k = 0; k < cpu[i].probs.size(); ++k) EXPECT_NEAR(gpu[i].probs[k], cpu[i].probs[k], 1e-5f);
  }
}

}  // namespace
