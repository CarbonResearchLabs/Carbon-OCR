// The order model on the engine's own CPU kernels (carbon/src/net/order_net.cpp,
// Backend::kNative from an order.native.bin): against a plain forward in double
// precision written here from order.py, on every instruction set this CPU has,
// for a small random network written to a file the test makes.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "carbon/cpu.h"
#include "carbon/error.h"
#include "carbon/order.h"
#include "carbon/session.h"

namespace carbon {
namespace {

constexpr int kD = 16, kLayers = 2, kHeads = 2, kFfn = 24;

struct Lin {
  int out, in;
  std::vector<float> w, b;  // [out][in], [out] (empty: no bias)
};

struct Net {
  Lin embed;
  std::vector<Lin> layer;  // qkv, out, ff1, ff2 for each
  Lin forms[5], labels;
};

Lin random_lin(std::mt19937& g, int out, int in, bool bias) {
  std::normal_distribution<float> n(0.0f, 1.0f / std::sqrt(static_cast<float>(in)));
  Lin l{out, in, std::vector<float>(static_cast<size_t>(out) * in), {}};
  for (float& v : l.w) v = n(g);
  if (bias) {
    l.b.resize(out);
    for (float& v : l.b) v = 0.1f * n(g);
  }
  return l;
}

Net random_net(uint32_t seed) {
  std::mt19937 g(seed);
  Net net;
  net.embed = random_lin(g, kD, kOrderFeatures, true);
  for (int k = 0; k < kLayers; ++k) {
    net.layer.push_back(random_lin(g, 3 * kD, kD, true));
    net.layer.push_back(random_lin(g, kD, kD, true));
    net.layer.push_back(random_lin(g, kFfn, kD, true));
    net.layer.push_back(random_lin(g, kD, kFfn, true));
  }
  for (Lin& f : net.forms) f = random_lin(g, kD, kD, false);
  net.labels = random_lin(g, kBlockLabels, kD, true);
  return net;
}

void write(const Net& net, const std::filesystem::path& file) {
  std::ofstream f(file, std::ios::binary);
  f.write("ORDERNT1", 8);
  const int32_t dims[6] = {kD, kLayers, kHeads, kFfn, kOrderFeatures, kBlockLabels};
  f.write(reinterpret_cast<const char*>(dims), sizeof dims);
  auto put = [&](const Lin& l) {
    f.write(reinterpret_cast<const char*>(l.w.data()), static_cast<std::streamsize>(l.w.size() * 4));
    f.write(reinterpret_cast<const char*>(l.b.data()), static_cast<std::streamsize>(l.b.size() * 4));
  };
  put(net.embed);
  for (const Lin& l : net.layer) put(l);
  for (const Lin& l : net.forms) put(l);
  put(net.labels);
}

using M = std::vector<std::vector<double>>;

M affine(const Lin& l, const M& x) {
  M y(x.size(), std::vector<double>(l.out));
  for (size_t r = 0; r < x.size(); ++r) {
    for (int o = 0; o < l.out; ++o) {
      double s = l.b.empty() ? 0.0 : l.b[o];
      for (int i = 0; i < l.in; ++i) s += x[r][i] * l.w[static_cast<size_t>(o) * l.in + i];
      y[r][o] = s;
    }
  }
  return y;
}

M norm(const M& x) {
  M y = x;
  for (auto& r : y) {
    double mean = 0, var = 0;
    for (double v : r) mean += v;
    mean /= r.size();
    for (double v : r) var += (v - mean) * (v - mean);
    var /= r.size();
    for (double& v : r) v = (v - mean) / std::sqrt(var + 1e-5);
  }
  return y;
}

// order.py's forward in doubles: the six outputs, flattened
std::vector<std::vector<double>> reference(const Net& net, const std::vector<float>& x, int n) {
  const int t = n + 1, dh = kD / kHeads;
  M in(t, std::vector<double>(kOrderFeatures, -1.0));
  for (int i = 0; i < n; ++i) {
    for (int c = 0; c < kOrderFeatures; ++c) in[i + 1][c] = x[static_cast<size_t>(i) * kOrderFeatures + c];
  }
  M h = affine(net.embed, in);
  for (int k = 0; k < kLayers; ++k) {
    const Lin &qkv = net.layer[4 * k], &out = net.layer[4 * k + 1], &ff1 = net.layer[4 * k + 2],
              &ff2 = net.layer[4 * k + 3];
    const M y = affine(qkv, norm(h));
    M att(t, std::vector<double>(kD, 0.0));
    for (int hd = 0; hd < kHeads; ++hd) {
      for (int i = 0; i < t; ++i) {
        std::vector<double> p(t);
        double hi = -1e300, sum = 0;
        for (int j = 0; j < t; ++j) {
          double s = 0;
          for (int c = 0; c < dh; ++c) s += y[i][hd * dh + c] * y[j][kD + hd * dh + c];
          p[j] = s / std::sqrt(static_cast<double>(dh));
          hi = std::max(hi, p[j]);
        }
        for (double& v : p) sum += (v = std::exp(v - hi));
        for (int c = 0; c < dh; ++c) {
          double s = 0;
          for (int j = 0; j < t; ++j) s += p[j] / sum * y[j][2 * kD + hd * dh + c];
          att[i][hd * dh + c] = s;
        }
      }
    }
    const M o = affine(out, att);
    for (int i = 0; i < t; ++i) {
      for (int c = 0; c < kD; ++c) h[i][c] += o[i][c];
    }
    M f = affine(ff1, norm(h));
    for (auto& r : f) {
      for (double& v : r) v = 0.5 * v * (1 + std::erf(v / std::sqrt(2.0)));
    }
    const M f2 = affine(ff2, f);
    for (int i = 0; i < t; ++i) {
      for (int c = 0; c < kD; ++c) h[i][c] += f2[i][c];
    }
  }
  h = norm(h);
  const M lines(h.begin() + 1, h.end());
  const double scale = 1 / std::sqrt(static_cast<double>(kD));
  std::vector<std::vector<double>> out;
  auto pair = [&](const Lin& form, const M& a, bool symmetric) {
    const M w = affine(form, a);
    const size_t m = a.size();
    std::vector<double> s(m * m);
    for (size_t i = 0; i < m; ++i) {
      for (size_t j = 0; j < m; ++j) {
        double v = 0;
        for (int c = 0; c < kD; ++c) v += w[i][c] * a[j][c];
        s[i * m + j] = v * scale;
      }
    }
    if (symmetric) {
      std::vector<double> t2 = s;
      for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) t2[i * m + j] = (s[i * m + j] + s[j * m + i]) / 2;
      }
      s = t2;
    }
    out.push_back(s);
  };
  pair(net.forms[0], h, false);
  for (int f = 1; f < 5; ++f) pair(net.forms[f], lines, true);
  std::vector<double> labels;
  for (const auto& r : affine(net.labels, lines)) labels.insert(labels.end(), r.begin(), r.end());
  out.push_back(labels);
  return out;
}

TEST(OrderNet, IsOrderPysForwardOnEveryInstructionSet) {
  const Net net = random_net(7);
  const auto file = std::filesystem::temp_directory_path() / "carbon-test-order.native.bin";
  write(net, file);
  std::mt19937 g(3);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for (cpu::Isa isa : cpu::isas_available()) {
    SessionOptions o;
    o.isa = cpu::isa_name(isa);
    auto s = open_session(Backend::kNative, file, o);
    EXPECT_EQ(s->backend(), Backend::kNative);
    ASSERT_EQ(s->outputs().size(), 6u);
    for (int n : {1, 5, 37}) {
      std::vector<float> x(static_cast<size_t>(n) * kOrderFeatures);
      for (float& v : x) v = u(g);
      Tensor in = Tensor::empty(DType::kFloat32, {1, n, kOrderFeatures});
      std::copy(x.begin(), x.end(), in.data<float>().begin());
      const std::vector<Tensor> y = s->run(std::span<const Tensor>(&in, 1));
      const auto want = reference(net, x, n);
      ASSERT_EQ(y.size(), 6u);
      EXPECT_EQ(y[0].shape(), (Shape{1, n + 1, n + 1}));
      EXPECT_EQ(y[5].shape(), (Shape{1, n, kBlockLabels}));
      for (int k = 0; k < 6; ++k) {
        const auto got = y[k].data<float>();
        ASSERT_EQ(got.size(), want[k].size()) << "output " << k;
        double worst = 0, top = 1e-6;
        for (size_t i = 0; i < got.size(); ++i) {
          worst = std::max(worst, std::abs(got[i] - want[k][i]));
          top = std::max(top, std::abs(want[k][i]));
        }
        EXPECT_LT(worst / top, 1e-4) << cpu::isa_name(isa) << ", " << n << " lines, output " << k;
      }
    }
  }
  // not an order model: refused
  const auto bad = std::filesystem::temp_directory_path() / "carbon-test-bad.native.bin";
  std::ofstream(bad, std::ios::binary).write("ORDERNT1\x01\x00\x00\x00", 12);
  EXPECT_THROW(open_session(Backend::kNative, bad), Error);
  std::filesystem::remove(file);
  std::filesystem::remove(bad);
}

}  // namespace
}  // namespace carbon
