// The transformer language model on a GPU (cuda_transformer_lm, src/cuda/tlm.cu) against the CPU's
// (TransformerLM): the same model file, random weights, the same scores to FP16's rounding; a page's lines in one
// run the lines one by one; the readings longer than the window the CPU's exactly.
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "carbon/error.h"
#include "carbon/tlm.h"
#include "random_tlm.h"

namespace carbon {
namespace {

bool have_gpu() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

std::vector<int32_t> random_tokens(std::mt19937& g, int classes, size_t n) {
  std::uniform_int_distribution<int32_t> t(1, classes - 1);
  std::vector<int32_t> v(n);
  for (auto& x : v) x = t(g);
  return v;
}

// A beam's readings of a line: one, and the rest of them a few substitutions, insertions and cuts away.
std::vector<std::vector<int32_t>> readings_of(std::mt19937& g, int classes, size_t len, int n) {
  std::vector<std::vector<int32_t>> r{random_tokens(g, classes, len)};
  std::uniform_int_distribution<int32_t> t(1, classes - 1);
  for (int i = 1; i < n; ++i) {
    std::vector<int32_t> v = r[0];
    if (v.empty()) {
      r.push_back(v);
      continue;
    }
    std::uniform_int_distribution<size_t> at(0, v.size() - 1);
    switch (i % 3) {
      case 0: v[at(g)] = t(g); break;
      case 1: v.insert(v.begin() + static_cast<std::ptrdiff_t>(at(g)), t(g)); break;
      default: v.resize(v.size() - 1); break;
    }
    r.push_back(std::move(v));
  }
  return r;
}

// A reading's tolerance: FP16 products, its tokens' log-probabilities each a few thousandths of a nat off.
double tolerance(size_t tokens) { return 0.02 + 0.004 * static_cast<double>(tokens); }

TEST(CudaTlm, ScoresAsTheCpuDoes) {
  if (!have_gpu()) GTEST_SKIP() << "no GPU";
  for (const int heads : {4, 2, 1}) {  // heads 32, 64 and 128 wide
    const int classes = 61, d = 128;
    const auto file = random_model("gpu-" + std::to_string(heads), d, 3, heads, 256, 64, classes, 7u + heads);
    const auto cpu = TransformerLM::open(file);
    const auto gpu = cuda_transformer_lm(file, 0);
    EXPECT_EQ(gpu->boundary(), classes);
    std::mt19937 g(11u + heads);
    const int B = classes;
    struct Case {
      std::vector<int32_t> context;
      std::vector<std::vector<int32_t>> readings;
    };
    std::vector<Case> cases;
    cases.push_back({{B}, readings_of(g, classes, 20, 8)});                 // a line's start alone
    std::vector<int32_t> above = random_tokens(g, classes, 30);            // the line above, then the boundary
    above.insert(above.begin(), B);
    above.push_back(B);
    cases.push_back({above, readings_of(g, classes, 25, 8)});
    cases.push_back({random_tokens(g, classes, 60), readings_of(g, classes, 30, 5)});  // a context cut to fit
    cases.push_back({{B}, {{}, {3}, {3, 4}}});                              // an empty reading; one a prefix of another
    cases.push_back({{B}, readings_of(g, classes, 40, 1)});                 // one reading
    cases.push_back({{B}, {random_tokens(g, classes, 63)}});                // the longest the window holds
    for (size_t c = 0; c < cases.size(); ++c) {
      const Case& k = cases[c];
      std::vector<double> want(k.readings.size()), got(k.readings.size());
      cpu->score(k.context, k.readings, want);
      gpu->score(k.context, k.readings, got);
      for (size_t r = 0; r < want.size(); ++r) {
        EXPECT_NEAR(got[r], want[r], tolerance(k.readings[r].size() + 1))
            << "heads " << heads << ", case " << c << ", reading " << r;
      }
    }
  }
}

// FP16's and INT8's files on the GPU as on the CPU: FP16's halves are the GPU's own; INT8's inputs quantized where they
// are made and its weights kept as int8 values, both whole numbers FP16 holds exactly, so the tensor cores' FP32 sums
// are the CPU's int8 sums, scaled the same way: within FP16's tolerance of the CPU's INT8, and far nearer it than
// the FP32 file's scores (the GPU's file is INT8's, not a dequantized FP16 one).
TEST(CudaTlm, ScoresAsTheCpuDoesInEachPrecision) {
  if (!have_gpu()) GTEST_SKIP() << "no GPU";
  const int classes = 61, B = classes;
  const auto f32 = TransformerLM::open(random_model("gpu-prec", 128, 3, 4, 256, 64, classes, 21u));
  for (const TlmPrecision p : {TlmPrecision::kF16, TlmPrecision::kInt8}) {
    const auto file = random_model("gpu-prec", 128, 3, 4, 256, 64, classes, 21u, 0, p);
    const auto cpu = TransformerLM::open(file);
    const auto gpu = cuda_transformer_lm(file, 0);
    std::mt19937 g(17u);
    std::vector<int32_t> above = random_tokens(g, classes, 30);
    above.push_back(B);
    double to_cpu = 0, to_f32 = 0;
    for (const auto& context : {std::vector<int32_t>{B}, above}) {
      const auto readings = readings_of(g, classes, 24, 8);
      std::vector<double> want(readings.size()), got(readings.size()), full(readings.size());
      cpu->score(context, readings, want);
      gpu->score(context, readings, got);
      f32->score(context, readings, full);
      for (size_t r = 0; r < want.size(); ++r) {
        EXPECT_NEAR(got[r], want[r], tolerance(readings[r].size() + 1)) << precision_name(p) << ", reading " << r;
        to_cpu += std::abs(got[r] - want[r]);
        to_f32 += std::abs(got[r] - full[r]);
      }
    }
    if (p == TlmPrecision::kInt8) EXPECT_LT(to_cpu, 0.2 * to_f32) << "int8: the GPU's file is not the CPU's INT8";
  }
}

TEST(CudaTlm, APagesLinesInOneRunAreEachLinesAlone) {
  if (!have_gpu()) GTEST_SKIP() << "no GPU";
  const int classes = 97;
  const auto file = random_model("gpu-page", 128, 4, 4, 512, 128, classes, 3u);
  const auto cpu = TransformerLM::open(file);
  const auto gpu = cuda_transformer_lm(file, 0);
  std::mt19937 g(5u);
  std::vector<std::vector<int32_t>> contexts;
  std::vector<std::vector<std::vector<int32_t>>> readings;
  for (int l = 0; l < 40; ++l) {
    std::vector<int32_t> ctx{classes};
    if (l % 3) {  // most with the line above
      const auto above = random_tokens(g, classes, 10 + static_cast<size_t>(l) * 2);
      ctx.insert(ctx.end(), above.begin(), above.end());
      ctx.push_back(classes);
    }
    contexts.push_back(ctx);
    readings.push_back(readings_of(g, classes, 5 + static_cast<size_t>(l) * 3 % 60, 1 + l % 8));
  }
  readings.push_back(readings_of(g, classes, 200, 3));  // longer than the window: the CPU's, in its windows
  contexts.push_back({classes});
  readings.push_back({});  // none
  contexts.push_back({classes});
  std::vector<RescoreQuery> q;
  for (size_t l = 0; l < contexts.size(); ++l) q.push_back({contexts[l], readings[l]});
  std::vector<std::vector<double>> page(q.size());
  gpu->score_lines(q, page);
  for (size_t l = 0; l < q.size(); ++l) {
    ASSERT_EQ(page[l].size(), readings[l].size());
    std::vector<double> alone(readings[l].size()), want(readings[l].size());
    gpu->score(contexts[l], readings[l], alone);
    cpu->score(contexts[l], readings[l], want);
    for (size_t r = 0; r < alone.size(); ++r) {
      EXPECT_NEAR(page[l][r], alone[r], 1e-3) << "line " << l << ", reading " << r;
      EXPECT_NEAR(page[l][r], want[r], tolerance(readings[l][r].size() + contexts[l].size()))
          << "line " << l << ", reading " << r;
    }
  }
  EXPECT_EQ(page[40], [&] {  // the long line: the CPU's model itself
    std::vector<double> v(readings[40].size());
    cpu->score(contexts[40], readings[40], v);
    return v;
  }());
}

TEST(CudaTlm, RefusesWhatItDoesNotRun) {
  if (!have_gpu()) GTEST_SKIP() << "no GPU";
  const auto narrow = random_model("gpu-narrow", 64, 1, 4, 128, 16, 20, 1u);  // heads 16 wide
  EXPECT_THROW(cuda_transformer_lm(narrow, 0), Error);
  const auto file = random_model("gpu-ok", 64, 1, 2, 128, 16, 20, 1u);
  const auto gpu = cuda_transformer_lm(file, 0);
  std::vector<double> out(1);
  const std::vector<std::vector<int32_t>> bad{{25}};
  EXPECT_THROW(gpu->score(std::vector<int32_t>{20}, bad, out), Error);
  const std::vector<std::vector<int32_t>> ok{{1, 2}};
  EXPECT_THROW(gpu->score(std::vector<int32_t>{}, ok, out), Error);
}

}  // namespace
}  // namespace carbon
