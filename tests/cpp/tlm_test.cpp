// The transformer language model's kernels (carbon/cpu.h): every instruction set's against the reference, and the
// reference against the plain formula in double; its files in each precision read and checked.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "carbon/cpu.h"
#include "carbon/error.h"
#include "carbon/tlm.h"
#include "random_tlm.h"

namespace {

using carbon::cpu::Isa;

std::vector<float> normal(std::mt19937& rng, size_t n, float sd) {
  std::normal_distribution<float> d(0.0f, sd);
  std::vector<float> v(n);
  for (float& x : v) x = d(rng);
  return v;
}

TEST(TlmKernels, LayerNorm) {
  std::mt19937 rng(3);
  const int rows = 7, n = 128;
  const auto x = normal(rng, rows * 130, 2.0f), w = normal(rng, n, 1.0f), b = normal(rng, n, 0.5f);
  for (const Isa isa : {Isa::kScalar, carbon::cpu::best_isa()}) {
    std::vector<float> out(rows * n);
    carbon::cpu::layer_norm_rows(x.data(), 130, rows, n, w.data(), b.data(), 1e-5f, out.data(), n, isa);
    for (int r = 0; r < rows; ++r) {
      double mean = 0, var = 0;
      for (int j = 0; j < n; ++j) mean += x[r * 130 + j];
      mean /= n;
      for (int j = 0; j < n; ++j) var += (x[r * 130 + j] - mean) * (x[r * 130 + j] - mean);
      var /= n;
      for (int j = 0; j < n; ++j) {
        const double want = (x[r * 130 + j] - mean) / std::sqrt(var + 1e-5) * w[j] + b[j];
        ASSERT_NEAR(out[r * n + j], want, 1e-4) << "row " << r << " col " << j;
      }
    }
  }
}

TEST(TlmKernels, LogSoftmaxPick) {
  std::mt19937 rng(5);
  for (const int n : {801, 64, 13}) {
    const int rows = 9;
    const auto x = normal(rng, rows * n, 4.0f);
    std::vector<int32_t> t(rows);
    for (int r = 0; r < rows; ++r) t[r] = static_cast<int32_t>(rng() % n);
    for (const Isa isa : {Isa::kScalar, carbon::cpu::best_isa()}) {
      std::vector<float> out(rows);
      carbon::cpu::log_softmax_pick(x.data(), n, rows, n, t.data(), out.data(), isa);
      for (int r = 0; r < rows; ++r) {
        double m = -1e300, s = 0;
        for (int j = 0; j < n; ++j) m = std::max(m, static_cast<double>(x[r * n + j]));
        for (int j = 0; j < n; ++j) s += std::exp(x[r * n + j] - m);
        ASSERT_NEAR(out[r], x[r * n + t[r]] - m - std::log(s), 1e-4) << n << " row " << r;
      }
    }
    // ld 0: one row for every target
    std::vector<float> one(3);
    const int32_t tt[3] = {0, 1, n - 1};
    carbon::cpu::log_softmax_pick(x.data(), 0, 3, n, tt, one.data());
    std::vector<float> each(1);
    for (int i = 0; i < 3; ++i) {
      carbon::cpu::log_softmax_pick(x.data(), n, 1, n, tt + i, each.data());
      EXPECT_EQ(one[i], each[0]);
    }
  }
}

TEST(TlmKernels, AttentionOverTwoRuns) {
  std::mt19937 rng(7);
  for (const int dh : {32, 64, 8}) {
    const int n1 = 37, n2 = 5, ld = 3 * dh;
    const auto q = normal(rng, dh, 1.0f), a = normal(rng, n1 * ld, 1.0f), b = normal(rng, n2 * ld, 1.0f);
    const carbon::cpu::KeyRun runs[2] = {{a.data(), a.data() + dh, ld, n1}, {b.data(), b.data() + dh, ld, n2}};
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    for (const Isa isa : {Isa::kScalar, carbon::cpu::best_isa()}) {
      std::vector<float> out(dh), sc(n1 + n2);
      carbon::cpu::attend(q.data(), dh, runs, 2, scale, out.data(), sc.data(), isa);
      std::vector<double> s;
      for (int r = 0; r < 2; ++r) {
        for (int i = 0; i < runs[r].n; ++i) {
          double d = 0;
          for (int j = 0; j < dh; ++j) d += q[j] * runs[r].k[i * ld + j];
          s.push_back(d * scale);
        }
      }
      const double m = *std::max_element(s.begin(), s.end());
      double z = 0;
      for (double& v : s) z += (v = std::exp(v - m));
      for (int j = 0; j < dh; ++j) {
        double want = 0;
        int at = 0;
        for (int r = 0; r < 2; ++r) {
          for (int i = 0; i < runs[r].n; ++i) want += s[at++] / z * runs[r].v[i * ld + j];
        }
        ASSERT_NEAR(out[j], want, 1e-5) << "dh " << dh << " col " << j;
      }
    }
  }
}

// The same random model in FP16 and INT8 scores as in FP32, within each one's rounding: FP16's weights (a few
// thousandths of a nat a token), INT8's weights and inputs (a few hundredths, more where a sharp distribution's
// token moves); each file says its precision.
TEST(Tlm, EachPrecisionScoresAsFp32Does) {
  using carbon::TlmPrecision;
  const int classes = 61, B = classes;
  const auto f32 = carbon::TransformerLM::open(carbon::random_model("prec", 64, 2, 2, 128, 48, classes, 5u));
  std::mt19937 g(3);
  std::uniform_int_distribution<int32_t> tok(1, classes - 1);
  std::vector<std::vector<int32_t>> readings(6);
  for (size_t r = 0; r < readings.size(); ++r) {
    for (size_t i = 0; i < 10 + 4 * r; ++i) readings[r].push_back(tok(g));
  }
  std::vector<int32_t> context{B};
  for (int i = 0; i < 12; ++i) context.push_back(tok(g));
  context.push_back(B);
  std::vector<double> want(readings.size());
  f32->score(context, readings, want);
  EXPECT_EQ(f32->precision(), TlmPrecision::kF32);
  const std::tuple<TlmPrecision, double, double> cases[] = {{TlmPrecision::kF16, 0.01, 0.002},
                                                           {TlmPrecision::kInt8, 0.3, 0.1}};
  for (const auto& [p, base, per] : cases) {
    const auto m = carbon::TransformerLM::open(carbon::random_model("prec", 64, 2, 2, 128, 48, classes, 5u, 0, p));
    EXPECT_EQ(m->precision(), p);
    EXPECT_EQ(m->header().version, 2u);
    std::vector<double> got(readings.size());
    m->score(context, readings, got);
    double mean = 0;
    for (size_t r = 0; r < got.size(); ++r) {
      const double tokens = static_cast<double>(readings[r].size() + 1);
      EXPECT_NEAR(got[r], want[r], base + per * tokens) << carbon::precision_name(p) << ", reading " << r;
      EXPECT_NE(got[r], want[r]) << carbon::precision_name(p) << ": the file's own weights, not FP32's";
      mean += std::abs(got[r] - want[r]) / tokens / static_cast<double>(got.size());
    }
    EXPECT_LT(mean, p == TlmPrecision::kF16 ? 0.002 : 0.05) << carbon::precision_name(p);
  }
}

// A file whose size is not its precision's, or of a version or precision the engine does not know, is refused.
TEST(Tlm, RefusesAFileThatIsNotItsPrecisions) {
  using carbon::TlmPrecision;
  for (const TlmPrecision p : {TlmPrecision::kF32, TlmPrecision::kF16, TlmPrecision::kInt8}) {
    const auto file = carbon::random_model("refuse", 32, 1, 1, 64, 16, 20, 1u, 0, p);
    std::string bytes;
    {
      std::ifstream f(file, std::ios::binary);
      bytes.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    }
    auto write = [&](const std::string& b) {
      std::ofstream f(file, std::ios::binary | std::ios::trunc);
      f.write(b.data(), static_cast<std::streamsize>(b.size()));
    };
    write(bytes.substr(0, bytes.size() - 4));
    EXPECT_THROW(carbon::TransformerLM::open(file), carbon::Error) << carbon::precision_name(p) << ": short";
    std::string other = bytes;  // another precision's word on these bytes
    const uint32_t q = p == TlmPrecision::kF32 ? 1u : 0u;
    std::memcpy(other.data() + offsetof(carbon::TlmHeader, precision), &q, 4);
    const uint32_t v2 = 2;
    std::memcpy(other.data() + offsetof(carbon::TlmHeader, version), &v2, 4);
    write(other);
    EXPECT_THROW(carbon::TransformerLM::open(file), carbon::Error) << carbon::precision_name(p) << ": as another";
    std::string unknown = bytes;
    const uint32_t nine = 9;
    std::memcpy(unknown.data() + offsetof(carbon::TlmHeader, precision), &nine, 4);
    write(unknown);
    EXPECT_THROW(carbon::TransformerLM::open(file), carbon::Error) << carbon::precision_name(p) << ": unknown";
    write(bytes);
    EXPECT_NO_THROW(carbon::TransformerLM::open(file)) << carbon::precision_name(p);
  }
}

}  // namespace
