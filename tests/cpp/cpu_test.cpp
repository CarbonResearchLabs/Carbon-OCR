#include "carbon/cpu.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "../../carbon/src/cpu/kernels.h"

namespace carbon::cpu {
namespace {

TEST(Cpu, ReportsTheInstructionSets) {
  const auto all = isas_available();
  ASSERT_FALSE(all.empty());
  EXPECT_EQ(all.front(), Isa::kScalar);
  std::set<std::string> names;
  for (Isa i : all) {
    EXPECT_TRUE(isa_available(i));
    EXPECT_TRUE(names.insert(isa_name(i)).second) << isa_name(i) << " twice";
  }
  bool listed = false;
  for (Isa i : all) listed = listed || i == best_isa();
  EXPECT_TRUE(listed) << isa_name(best_isa());
#if defined(_M_X64) || defined(__x86_64__)
  EXPECT_FALSE(features().vendor.empty());
#endif
#if defined(_M_ARM64) || defined(__aarch64__)
  EXPECT_TRUE(isa_available(Isa::kNeon));
#endif
}

// A's rows as a convolution's: each row `taps` pieces of `seg` values at random
// places in one buffer, with room past each piece for the bytes a kernel reads.
template <typename T>
struct Pieces {
  std::vector<T> data;
  std::vector<const T*> ptrs;
  std::vector<int64_t> offsets;
};

template <typename T, typename Gen>
Pieces<T> pieces(int m, int taps, int readable, std::mt19937& rng, Gen&& value) {
  Pieces<T> p;
  p.data.resize(static_cast<size_t>(readable) * 24 + 512);
  for (auto& v : p.data) v = value();
  std::uniform_int_distribution<int64_t> at(0, static_cast<int64_t>(p.data.size()) - readable);
  p.offsets.resize(static_cast<size_t>(m) * taps);
  for (auto& o : p.offsets) o = at(rng);
  p.ptrs.resize(p.offsets.size());
  for (size_t i = 0; i < p.ptrs.size(); ++i) p.ptrs[i] = p.data.data() + p.offsets[i];
  return p;
}

// The output stages the tests run: float out (with a scale, a bias), and uint8
// out (relu, a residual of either kind); FP16 out and residuals for the FP16 kernels.
struct Stage {
  const char* name;
  bool scale, bias, relu, residual_u8, residual_f32, out_u8;
  bool residual_f16 = false, out_f16 = false;
};
constexpr Stage kStages[] = {
    {"sums", false, false, false, false, false, false},
    {"scale bias", true, true, false, false, false, false},
    {"bias relu residual", false, true, true, false, true, false},
    {"u8 relu", true, true, true, false, false, true},
    {"u8 residual u8", true, true, false, true, false, true},
    {"u8 residual f32 relu", true, false, true, false, true, true},
};
constexpr Stage kF16Stages[] = {
    {"sums", false, false, false, false, false, false},
    {"bias relu residual", false, true, true, false, true, false},
    {"bias residual f16", false, true, false, false, false, false, true, false},
    {"f16 bias relu", false, true, true, false, false, false, false, true},
    {"f16 residual f16 relu", false, true, true, false, false, false, true, true},
    {"f16 residual f32", false, false, false, false, true, false, false, true},
};

struct Outputs {
  std::vector<float> scale, bias, res_f32, out_f32;
  std::vector<uint8_t> res_u8, out_u8;
  std::vector<uint16_t> res_f16, out_f16;
  int64_t ld;
};

Epilogue make(const Stage& s, Outputs& o, int m, int n, std::mt19937& rng, bool int8) {
  std::uniform_real_distribution<float> u(-1, 1);
  o.ld = n + 5;
  o.scale.resize(n);
  o.bias.resize(n);
  for (auto& v : o.scale) v = (int8 ? 1e-4f : 1.0f) * (0.5f + u(rng));
  for (auto& v : o.bias) v = u(rng);
  o.res_f32.resize(static_cast<size_t>(m) * o.ld);
  o.res_u8.resize(o.res_f32.size());
  o.res_f16.resize(o.res_f32.size());
  for (auto& v : o.res_f32) v = u(rng);
  for (auto& v : o.res_u8) v = static_cast<uint8_t>(rng());
  for (auto& v : o.res_f16) v = f32_to_f16(u(rng));
  o.out_f32.assign(static_cast<size_t>(m) * o.ld, -7.0f);
  o.out_u8.assign(static_cast<size_t>(m) * o.ld, 0x5a);
  o.out_f16.assign(static_cast<size_t>(m) * o.ld, 0x5a5a);
  Epilogue e;
  if (s.scale && int8) e.scale = o.scale.data();
  if (s.bias) e.bias = o.bias.data();
  e.relu = s.relu;
  e.residual_ld = o.ld;
  if (s.residual_f32) e.residual_f32 = o.res_f32.data();
  if (s.residual_u8) {
    e.residual_u8 = o.res_u8.data();
    e.residual_scale = 0.01f;
    e.residual_zero = 120;
  }
  if (s.residual_f16) e.residual_f16 = o.res_f16.data();
  e.out_ld = o.ld;
  if (s.out_u8) {
    e.out_u8 = o.out_u8.data();
    e.out_inv_scale = 40.0f;
    e.out_zero = s.relu ? 0 : 128;
  } else if (s.out_f16) {
    e.out_f16 = o.out_f16.data();
  } else {
    e.out_f32 = o.out_f32.data();
  }
  return e;
}

// Every value of the two outputs alike, bit for bit, the columns past n untouched.
void expect_same(const Outputs& want, const Outputs& got, int m, int n, const std::string& where) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < want.ld; ++j) {
      const size_t at = static_cast<size_t>(i) * want.ld + j;
      uint32_t a, b;
      std::memcpy(&a, &want.out_f32[at], 4);
      std::memcpy(&b, &got.out_f32[at], 4);
      ASSERT_EQ(a, b) << where << " float at " << i << "," << j << ": " << want.out_f32[at] << " vs "
                      << got.out_f32[at];
      ASSERT_EQ(want.out_u8[at], got.out_u8[at]) << where << " uint8 at " << i << "," << j;
      ASSERT_EQ(want.out_f16[at], got.out_f16[at]) << where << " fp16 at " << i << "," << j;
      if (j >= n) {
        ASSERT_EQ(got.out_f32[at], -7.0f) << where << " wrote past n at " << i << "," << j;
        ASSERT_EQ(got.out_u8[at], 0x5a) << where << " wrote past n at " << i << "," << j;
        ASSERT_EQ(got.out_f16[at], 0x5a5a) << where << " wrote past n at " << i << "," << j;
      }
    }
  }
}

// The outputs of a kernel that sums in FP16 against the reference's: within
// `tol` of each output (FP16 out: one rounding of FP16 more), nothing written past n.
void expect_near(const Outputs& want, const Outputs& got, int m, int n, const std::vector<double>& tol,
                 const std::string& where) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < want.ld; ++j) {
      const size_t at = static_cast<size_t>(i) * want.ld + j;
      if (j >= n) {
        ASSERT_EQ(got.out_f32[at], -7.0f) << where << " wrote past n at " << i << "," << j;
        ASSERT_EQ(got.out_f16[at], 0x5a5a) << where << " wrote past n at " << i << "," << j;
        continue;
      }
      const double t = tol[static_cast<size_t>(i) * n + j];
      const double w32 = want.out_f32[at], g32 = got.out_f32[at];
      ASSERT_NEAR(g32, w32, t) << where << " float at " << i << "," << j;
      const double w16 = f16_to_f32(want.out_f16[at]), g16 = f16_to_f32(got.out_f16[at]);
      ASSERT_NEAR(g16, w16, t + std::abs(w16) / 1024) << where << " fp16 at " << i << "," << j;
    }
  }
}

struct Shape {
  int m, taps, seg, n;
};

std::vector<Shape> shapes() {
  std::vector<Shape> out;
  for (int m : {1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 25}) {
    for (int n : {1, 3, 8, 15, 16, 17, 24, 31, 32, 33, 97}) {
      for (int k : {1, 2, 3, 4, 5, 7, 27, 64, 99, 288}) out.push_back({m, 1, k, n});
      for (int seg : {1, 3, 4, 5, 16, 48}) out.push_back({m, 9, seg, n});
    }
  }
  return out;
}

// The scalar kernel's sums are the int64 reference's; every other kernel this
// CPU runs gives the scalar's bits in every output stage: every tail of m (the
// row blocks are 4, 6 and 12), of n (panels of 16 and 32, vectors of 4, 8 and
// 16) and of a piece (groups of 2 and 4), pieces at any offset with the bytes past
// them set to anything, and nothing written past n.
TEST(CpuGemm, Int8EveryKernelGivesTheReferenceBits) {
  std::mt19937 rng(7);
  const auto isas = isas_available();
  int checked = 0;
  for (const Shape& s : shapes()) {
    const int k = s.taps * s.seg;
    std::vector<int8_t> b(static_cast<size_t>(k) * s.n);
    for (auto& v : b) v = static_cast<int8_t>(rng());
    const auto a = pieces<uint8_t>(s.m, s.taps, k_padded(s.seg, Isa::kAvx512), rng,
                                   [&] { return static_cast<uint8_t>(rng()); });
    const Rows<uint8_t> rows{a.ptrs.data(), s.taps};
    for (int a_zero : {0, 128, 37}) {
      // the sums, against int64
      {
        Outputs o;
        const Epilogue e = make(kStages[0], o, s.m, s.n, rng, true);
        gemm_u8s8(rows, a_zero, pack_b(b.data(), k, s.n, Isa::kScalar, s.taps), s.m, e);
        for (int i = 0; i < s.m; ++i) {
          for (int j = 0; j < s.n; ++j) {
            int64_t want = 0;
            for (int t = 0; t < s.taps; ++t) {
              for (int kk = 0; kk < s.seg; ++kk) {
                want += (static_cast<int64_t>(a.ptrs[static_cast<size_t>(i) * s.taps + t][kk]) - a_zero) *
                        b[(static_cast<size_t>(t) * s.seg + kk) * s.n + j];
              }
            }
            ASSERT_EQ(o.out_f32[static_cast<size_t>(i) * o.ld + j], static_cast<float>(want));
          }
        }
      }
      for (const Stage& st : kStages) {
        std::mt19937 stage_rng(static_cast<unsigned>(checked));
        Outputs want;
        const Epilogue ew = make(st, want, s.m, s.n, stage_rng, true);
        gemm_u8s8(rows, a_zero, pack_b(b.data(), k, s.n, Isa::kScalar, s.taps), s.m, ew);
        for (Isa isa : isas) {
          if (isa == Isa::kScalar) continue;
          std::mt19937 same_rng(static_cast<unsigned>(checked));
          Outputs got;
          const Epilogue eg = make(st, got, s.m, s.n, same_rng, true);
          gemm_u8s8(rows, a_zero, pack_b(b.data(), k, s.n, isa, s.taps), s.m, eg);
          expect_same(want, got, s.m, s.n,
                      std::string(isa_name(isa)) + " " + st.name + ": m " + std::to_string(s.m) + " taps " +
                          std::to_string(s.taps) + " seg " + std::to_string(s.seg) + " n " + std::to_string(s.n) +
                          " a_zero " + std::to_string(a_zero));
          if (::testing::Test::HasFatalFailure()) return;
        }
        ++checked;
      }
    }
  }
  EXPECT_GT(checked, 0);
}

// The FP32 GEMM: the scalar kernel within rounding of a double reference, and
// every other kernel the scalar's bits, in every output stage.
TEST(CpuGemm, F32EveryKernelGivesTheReferenceBits) {
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> u(-1, 1);
  const auto isas = isas_available();
  int checked = 0;
  for (const Shape& s : shapes()) {
    const int k = s.taps * s.seg;
    std::vector<float> b(static_cast<size_t>(k) * s.n);
    for (auto& v : b) v = u(rng);
    const auto a = pieces<float>(s.m, s.taps, s.seg, rng, [&] { return u(rng); });
    const Rows<float> rows{a.ptrs.data(), s.taps};
    {
      Outputs o;
      const Epilogue e = make(kStages[0], o, s.m, s.n, rng, false);
      gemm_f32(rows, pack_f32(b.data(), k, s.n, Isa::kScalar), s.m, e);
      for (int i = 0; i < s.m; ++i) {
        for (int j = 0; j < s.n; ++j) {
          double want = 0, mag = 0;
          for (int t = 0; t < s.taps; ++t) {
            for (int kk = 0; kk < s.seg; ++kk) {
              const double p = static_cast<double>(a.ptrs[static_cast<size_t>(i) * s.taps + t][kk]) *
                               b[(static_cast<size_t>(t) * s.seg + kk) * s.n + j];
              want += p;
              mag += std::abs(p);
            }
          }
          ASSERT_NEAR(o.out_f32[static_cast<size_t>(i) * o.ld + j], want, 1e-6 * k * mag + 1e-30);
        }
      }
    }
    for (const Stage& st : kStages) {
      std::mt19937 stage_rng(static_cast<unsigned>(checked));
      Outputs want;
      const Epilogue ew = make(st, want, s.m, s.n, stage_rng, false);
      gemm_f32(rows, pack_f32(b.data(), k, s.n, Isa::kScalar), s.m, ew);
      for (Isa isa : isas) {
        if (isa == Isa::kScalar) continue;
        std::mt19937 same_rng(static_cast<unsigned>(checked));
        Outputs got;
        const Epilogue eg = make(st, got, s.m, s.n, same_rng, false);
        gemm_f32(rows, pack_f32(b.data(), k, s.n, isa), s.m, eg);
        expect_same(want, got, s.m, s.n,
                    std::string(isa_name(isa)) + " " + st.name + ": m " + std::to_string(s.m) + " taps " +
                        std::to_string(s.taps) + " seg " + std::to_string(s.seg) + " n " + std::to_string(s.n));
        if (::testing::Test::HasFatalFailure()) return;
      }
      ++checked;
    }
  }
  EXPECT_GT(checked, 0);
}

// FP16 bits and FP32: every half widens exactly and rounds back to itself, a
// float rounds to the nearest half (ties to even, past 65504 to infinity, under
// 2^-14 to a subnormal), and the vector conversions give the same bits.
TEST(CpuF16, ConvertsAsTheHardwareDoes) {
  EXPECT_EQ(f32_to_f16(1.0f), 0x3c00);
  EXPECT_EQ(f32_to_f16(-2.0f), 0xc000);
  EXPECT_EQ(f32_to_f16(0.1f), 0x2e66);
  EXPECT_EQ(f32_to_f16(65504.0f), 0x7bff);
  EXPECT_EQ(f32_to_f16(65519.996f), 0x7bff);
  EXPECT_EQ(f32_to_f16(65520.0f), 0x7c00);  // the tie past the largest half: to the even, infinity
  EXPECT_EQ(f32_to_f16(std::ldexp(1.0f, -24)), 0x0001);
  EXPECT_EQ(f32_to_f16(std::ldexp(1.0f, -25)), 0x0000);  // a tie: to the even 0
  EXPECT_EQ(f32_to_f16(std::ldexp(1.5f, -25)), 0x0001);
  EXPECT_EQ(f32_to_f16(std::ldexp(3.0f, -25)), 0x0002);  // 1.5 steps: a tie, to the even 2
  EXPECT_EQ(f32_to_f16(1.0f + std::ldexp(1.0f, -11)), 0x3c00);  // halfway to the next: to the even
  EXPECT_EQ(f32_to_f16(1.0f + std::ldexp(3.0f, -11)), 0x3c02);
  EXPECT_EQ(f16_to_f32(0x0001), std::ldexp(1.0f, -24));
  EXPECT_EQ(f16_to_f32(0x7bff), 65504.0f);
  EXPECT_TRUE(std::isinf(f16_to_f32(0xfc00)) && f16_to_f32(0xfc00) < 0);
  std::vector<uint16_t> halves(65536);
  std::vector<float> floats(65536);
  for (int h = 0; h < 65536; ++h) {
    halves[h] = static_cast<uint16_t>(h);
    floats[h] = f16_to_f32(halves[h]);
    if (std::isnan(floats[h])) {
      EXPECT_TRUE((h & 0x7c00) == 0x7c00 && (h & 0x3ff)) << h;
      EXPECT_EQ(f32_to_f16(floats[h]), h | 0x200) << h;  // quiet, its payload kept
    } else {
      ASSERT_EQ(f32_to_f16(floats[h]), h) << h;
    }
  }
  // floats on both sides of every rounding boundary, and random ones of every size
  std::vector<float> probe;
  for (int h = 0; h < 0x7c00; ++h) {
    const float lo = f16_to_f32(static_cast<uint16_t>(h)), hi = f16_to_f32(static_cast<uint16_t>(h + 1));
    const float mid = lo + (hi - lo) / 2;
    for (const float v : {mid, std::nextafter(mid, 0.0f), std::nextafter(mid, 1e9f), lo}) {
      probe.push_back(v);
      probe.push_back(-v);
    }
  }
  std::mt19937 rng(1);
  for (int i = 0; i < 100000; ++i) {
    const uint32_t bits = static_cast<uint32_t>(rng());
    float v;
    std::memcpy(&v, &bits, 4);
    probe.push_back(v);
  }
  for (const float v : probe) {
    if (std::isnan(v)) continue;
    const double r = f16_to_f32(f32_to_f16(v)), d = std::abs(r - v);
    if (std::isinf(r)) {
      ASSERT_GE(std::abs(v), 65520.0f) << v;
      continue;
    }
    // no half is nearer, and a tie went to the even
    const uint16_t h = f32_to_f16(v);
    for (const int step : {-1, 1}) {
      const uint16_t other = static_cast<uint16_t>(h + step);
      if ((other & 0x7c00) == 0x7c00 || ((h & 0x7fff) == 0 && step < 0)) continue;
      const double od = std::abs(static_cast<double>(f16_to_f32(other)) - v);
      ASSERT_TRUE(d < od || (d == od && (h & 1) == 0)) << v << " -> " << h;
    }
  }
  for (Isa isa : isas_available()) {
    std::vector<float> wide(halves.size());
    to_f32(halves.data(), wide.data(), halves.size(), isa);
    ASSERT_EQ(std::memcmp(wide.data(), floats.data(), 4 * floats.size()), 0) << isa_name(isa);
    std::vector<uint16_t> narrow(probe.size()), want(probe.size());
    to_f16(probe.data(), narrow.data(), probe.size(), isa);
    for (size_t i = 0; i < probe.size(); ++i) want[i] = f32_to_f16(probe[i]);
    for (size_t i = 0; i < probe.size(); ++i) ASSERT_EQ(narrow[i], want[i]) << isa_name(isa) << " " << probe[i];
  }
}

// The FP16 GEMM: the reference within rounding of a double sum of its values,
// every FP32 arithmetic kernel the reference's bits with A in FP16 or FP32 in every
// output stage (FP16 out and residuals too), and an FP16 arithmetic kernel
// (avx512-fp16, neon-fp16, on a CPU with one) within the rounding of its FP16
// runs: each of at most kF16Run products rounds by at most 2^-11 of the run's
// sum of |products|, so the whole by at most kF16Run / 2048 of the sum's.
TEST(CpuGemm, F16EveryKernelGivesTheReferenceBits) {
  std::mt19937 rng(13);
  std::uniform_real_distribution<float> u(-1, 1);
  const auto isas = isas_available();
  int checked = 0;
  for (const Shape& s : shapes()) {
    const int k = s.taps * s.seg;
    std::vector<uint16_t> b(static_cast<size_t>(k) * s.n);
    for (auto& v : b) v = f32_to_f16(u(rng));
    const auto a16 = pieces<uint16_t>(s.m, s.taps, s.seg, rng, [&] { return f32_to_f16(u(rng)); });
    const auto a32 = pieces<float>(s.m, s.taps, s.seg, rng, [&] { return u(rng); });
    for (const bool half : {true, false}) {
      auto a_at = [&](int i, int t, int kk) -> double {
        const size_t at = static_cast<size_t>(i) * s.taps + t;
        return half ? f16_to_f32(a16.ptrs[at][kk]) : a32.ptrs[at][kk];
      };
      auto gemm = [&](Isa isa, const Epilogue& e) {
        const PackedF16 pb = pack_f16(b.data(), k, s.n, isa);
        if (half) {
          gemm_f16(Rows<uint16_t>{a16.ptrs.data(), s.taps}, pb, s.m, e);
        } else {
          gemm_f16(Rows<float>{a32.ptrs.data(), s.taps}, pb, s.m, e);
        }
      };
      std::vector<double> tol(static_cast<size_t>(s.m) * s.n);
      {
        Outputs o;
        gemm(Isa::kScalar, make(kF16Stages[0], o, s.m, s.n, rng, false));
        for (int i = 0; i < s.m; ++i) {
          for (int j = 0; j < s.n; ++j) {
            double want = 0, mag = 0;
            for (int t = 0; t < s.taps; ++t) {
              for (int kk = 0; kk < s.seg; ++kk) {
                const double p = a_at(i, t, kk) * f16_to_f32(b[(static_cast<size_t>(t) * s.seg + kk) * s.n + j]);
                want += p;
                mag += std::abs(p);
              }
            }
            ASSERT_NEAR(o.out_f32[static_cast<size_t>(i) * o.ld + j], want, 1e-6 * k * mag + 1e-30);
            tol[static_cast<size_t>(i) * s.n + j] = mag * (kF16Run / 2048.0 + 1e-4) + 1e-5;
          }
        }
      }
      for (const Stage& st : kF16Stages) {
        std::mt19937 stage_rng(static_cast<unsigned>(checked));
        Outputs want;
        gemm(Isa::kScalar, make(st, want, s.m, s.n, stage_rng, false));
        for (Isa isa : isas) {
          if (isa == Isa::kScalar) continue;
          std::mt19937 same_rng(static_cast<unsigned>(checked));
          Outputs got;
          gemm(isa, make(st, got, s.m, s.n, same_rng, false));
          const std::string where = std::string(isa_name(isa)) + (half ? " A fp16 " : " A fp32 ") + st.name +
                                    ": m " + std::to_string(s.m) + " taps " + std::to_string(s.taps) + " seg " +
                                    std::to_string(s.seg) + " n " + std::to_string(s.n);
          if (half && f16_arithmetic(isa)) {
            expect_near(want, got, s.m, s.n, tol, where);
          } else {
            expect_same(want, got, s.m, s.n, where);
          }
          if (::testing::Test::HasFatalFailure()) return;
        }
        ++checked;
      }
    }
  }
  EXPECT_GT(checked, 0);
}

// The uint8 output: round(v * inv_scale) + zero, clamped, against double.
TEST(CpuGemm, QuantizesTheOutput) {
  const std::vector<int8_t> b = {1, -1, 2, 3};  // k 2, n 2
  const uint8_t row[2] = {10, 200};
  const uint8_t* ptrs[] = {row};
  const float scale[] = {0.5f, 0.25f}, bias[] = {0.25f, -1.0f};
  uint8_t out[2] = {};
  for (Isa isa : isas_available()) {
    Epilogue e;
    e.scale = scale;
    e.bias = bias;
    e.out_u8 = out;
    e.out_ld = 2;
    e.out_inv_scale = 2.0f;
    e.out_zero = 100;
    gemm_u8s8({ptrs, 1}, 0, pack_b(b.data(), 2, 2, isa), 1, e);
    // column 0: 10 + 400 = 410, * 0.5 + 0.25 = 205.25, * 2 = 410.5: past 255 - 100, so 255
    // column 1: -10 + 600 = 590, * 0.25 - 1 = 146.5, * 2 = 293: 255
    EXPECT_EQ(out[0], 255) << isa_name(isa);
    EXPECT_EQ(out[1], 255) << isa_name(isa);
    e.out_inv_scale = 0.01f;
    gemm_u8s8({ptrs, 1}, 0, pack_b(b.data(), 2, 2, isa), 1, e);
    EXPECT_EQ(out[0], static_cast<uint8_t>(100 + std::nearbyint(205.25f * 0.01f))) << isa_name(isa);
    EXPECT_EQ(out[1], static_cast<uint8_t>(100 + std::nearbyint(146.5f * 0.01f))) << isa_name(isa);
    // halves to even: A = (1, 0) sums to 1 and -1, scaled to 2.5 and -1.5
    const uint8_t one[2] = {1, 0};
    const uint8_t* one_ptrs[] = {one};
    const float halves[] = {2.5f, 1.5f}, zeros[] = {0.0f, 0.0f};
    e.scale = halves;
    e.bias = zeros;
    e.out_inv_scale = 1.0f;
    gemm_u8s8({one_ptrs, 1}, 0, pack_b(b.data(), 2, 2, isa), 1, e);
    EXPECT_EQ(out[0], 102) << isa_name(isa);
    EXPECT_EQ(out[1], 98) << isa_name(isa);
  }
}

// The depthwise convolution: the scalar kernel's int8 sums are int64's, its FP32
// sums within rounding of double, and every other kernel the scalar's bits in
// every output stage, on every tail of pixels (tiles of 4) and channels (16, 8, 4).
TEST(CpuDwconv, EveryKernelGivesTheReferenceBits) {
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> u(-1, 1);
  const auto isas = isas_available();
  int checked = 0;
  for (int c : {1, 3, 4, 8, 15, 16, 17, 24, 31, 64, 65, 100, 256}) {
    for (int taps : {1, 9}) {
      for (int m : {1, 2, 3, 4, 5, 7, 8, 13}) {
        std::vector<int8_t> w8(static_cast<size_t>(taps) * c);
        std::vector<float> w32(w8.size());
        for (auto& v : w8) v = static_cast<int8_t>(rng());
        for (auto& v : w32) v = u(rng);
        const auto a8 = pieces<uint8_t>(m, taps, c, rng, [&] { return static_cast<uint8_t>(rng()); });
        const auto a32 = pieces<float>(m, taps, c, rng, [&] { return u(rng); });
        const Rows<uint8_t> r8{a8.ptrs.data(), taps};
        const Rows<float> r32{a32.ptrs.data(), taps};
        {
          Outputs o8, o32;
          const Epilogue e8 = make(kStages[0], o8, m, c, rng, true), e32 = make(kStages[0], o32, m, c, rng, false);
          dwconv_u8s8(r8, 37, w8.data(), c, m, e8, Isa::kScalar);
          dwconv_f32(r32, w32.data(), c, m, e32, Isa::kScalar);
          for (int i = 0; i < m; ++i) {
            for (int j = 0; j < c; ++j) {
              int64_t want8 = 0;
              double want32 = 0, mag = 0;
              for (int t = 0; t < taps; ++t) {
                const size_t at = static_cast<size_t>(i) * taps + t;
                want8 += (static_cast<int64_t>(a8.ptrs[at][j]) - 37) * w8[static_cast<size_t>(t) * c + j];
                const double prod = static_cast<double>(a32.ptrs[at][j]) * w32[static_cast<size_t>(t) * c + j];
                want32 += prod;
                mag += std::abs(prod);
              }
              ASSERT_EQ(o8.out_f32[static_cast<size_t>(i) * o8.ld + j], static_cast<float>(want8));
              ASSERT_NEAR(o32.out_f32[static_cast<size_t>(i) * o32.ld + j], want32, 1e-6 * taps * mag + 1e-30);
            }
          }
        }
        for (const Stage& st : kStages) {
          const std::string where = std::string(st.name) + ": c " + std::to_string(c) + " taps " +
                                    std::to_string(taps) + " m " + std::to_string(m);
          for (const bool int8 : {true, false}) {
            std::mt19937 stage_rng(static_cast<unsigned>(checked));
            Outputs want;
            const Epilogue ew = make(st, want, m, c, stage_rng, int8);
            if (int8) {
              dwconv_u8s8(r8, 37, w8.data(), c, m, ew, Isa::kScalar);
            } else {
              dwconv_f32(r32, w32.data(), c, m, ew, Isa::kScalar);
            }
            for (Isa isa : isas) {
              if (isa == Isa::kScalar) continue;
              std::mt19937 same_rng(static_cast<unsigned>(checked));
              Outputs got;
              const Epilogue eg = make(st, got, m, c, same_rng, int8);
              if (int8) {
                dwconv_u8s8(r8, 37, w8.data(), c, m, eg, isa);
              } else {
                dwconv_f32(r32, w32.data(), c, m, eg, isa);
              }
              expect_same(want, got, m, c, std::string(isa_name(isa)) + (int8 ? " int8 " : " fp32 ") + where);
              if (::testing::Test::HasFatalFailure()) return;
            }
            ++checked;
          }
        }
      }
    }
  }
  EXPECT_GT(checked, 0);
}

// The FP16 depthwise convolution as the FP16 GEMM is held: the reference within
// rounding of double, the FP32 arithmetic kernels its bits, the FP16 arithmetic
// ones within their runs' rounding; every tail of pixels and channels.
TEST(CpuDwconv, F16EveryKernelGivesTheReferenceBits) {
  std::mt19937 rng(9);
  std::uniform_real_distribution<float> u(-1, 1);
  const auto isas = isas_available();
  int checked = 0;
  for (int c : {1, 3, 4, 8, 15, 16, 17, 24, 31, 32, 33, 64, 65, 100, 256}) {
    for (int taps : {1, 9}) {
      for (int m : {1, 2, 3, 4, 5, 7, 8, 13}) {
        std::vector<uint16_t> w(static_cast<size_t>(taps) * c);
        for (auto& v : w) v = f32_to_f16(u(rng));
        const auto a16 = pieces<uint16_t>(m, taps, c, rng, [&] { return f32_to_f16(u(rng)); });
        const auto a32 = pieces<float>(m, taps, c, rng, [&] { return u(rng); });
        for (const bool half : {true, false}) {
          auto dw = [&](Isa isa, const Epilogue& e) {
            if (half) {
              dwconv_f16(Rows<uint16_t>{a16.ptrs.data(), taps}, w.data(), c, m, e, isa);
            } else {
              dwconv_f16(Rows<float>{a32.ptrs.data(), taps}, w.data(), c, m, e, isa);
            }
          };
          std::vector<double> tol(static_cast<size_t>(m) * c);
          {
            Outputs o;
            dw(Isa::kScalar, make(kF16Stages[0], o, m, c, rng, false));
            for (int i = 0; i < m; ++i) {
              for (int j = 0; j < c; ++j) {
                double want = 0, mag = 0;
                for (int t = 0; t < taps; ++t) {
                  const size_t at = static_cast<size_t>(i) * taps + t;
                  const double x = half ? f16_to_f32(a16.ptrs[at][j]) : a32.ptrs[at][j];
                  const double p = x * f16_to_f32(w[static_cast<size_t>(t) * c + j]);
                  want += p;
                  mag += std::abs(p);
                }
                ASSERT_NEAR(o.out_f32[static_cast<size_t>(i) * o.ld + j], want, 1e-6 * taps * mag + 1e-30);
                tol[static_cast<size_t>(i) * c + j] = mag * (kF16Run / 2048.0 + 1e-4) + 1e-5;
              }
            }
          }
          for (const Stage& st : kF16Stages) {
            std::mt19937 stage_rng(static_cast<unsigned>(checked));
            Outputs want;
            dw(Isa::kScalar, make(st, want, m, c, stage_rng, false));
            for (Isa isa : isas) {
              if (isa == Isa::kScalar) continue;
              std::mt19937 same_rng(static_cast<unsigned>(checked));
              Outputs got;
              dw(isa, make(st, got, m, c, same_rng, false));
              const std::string where = std::string(isa_name(isa)) + (half ? " A fp16 " : " A fp32 ") + st.name +
                                        ": c " + std::to_string(c) + " taps " + std::to_string(taps) + " m " +
                                        std::to_string(m);
              if (half && f16_arithmetic(isa)) {
                expect_near(want, got, m, c, tol, where);
              } else {
                expect_same(want, got, m, c, where);
              }
              if (::testing::Test::HasFatalFailure()) return;
            }
            ++checked;
          }
        }
      }
    }
  }
  EXPECT_GT(checked, 0);
}

// A grid of rows (a convolution over an input with a zero border) gives the
// same bits as the same rows listed one pointer at a time.
TEST(CpuGemm, GridRowsAreTheirList) {
  std::mt19937 rng(3);
  const int cin = 20, cout = 24, width = 30, pw = width + 2, stride = 2, m = 14;  // 3x3, stride 2
  std::vector<uint8_t> in8(static_cast<size_t>(3) * pw * cin + 64);
  std::vector<float> in32(in8.size());
  for (auto& v : in8) v = static_cast<uint8_t>(rng());
  for (auto& v : in32) v = static_cast<float>(static_cast<int>(rng() % 200) - 100) / 64.0f;
  std::vector<int64_t> offsets;
  for (int ky = 0; ky < 3; ++ky) {
    for (int kx = 0; kx < 3; ++kx) offsets.push_back((static_cast<int64_t>(ky) * pw + kx) * cin);
  }
  std::vector<const uint8_t*> list8;
  std::vector<const float*> list32;
  for (int i = 0; i < m; ++i) {
    for (int t = 0; t < 9; ++t) {
      list8.push_back(in8.data() + i * stride * cin + offsets[t]);
      list32.push_back(in32.data() + i * stride * cin + offsets[t]);
    }
  }
  std::vector<int8_t> b8(static_cast<size_t>(9) * cin * cout);
  std::vector<float> b32(b8.size());
  for (auto& v : b8) v = static_cast<int8_t>(rng());
  for (auto& v : b32) v = static_cast<float>(static_cast<int>(rng() % 200) - 100) / 128.0f;
  for (Isa isa : isas_available()) {
    for (const bool int8 : {true, false}) {
      Outputs want, got;
      std::mt19937 r1(1), r2(1);
      const Epilogue ew = make(kStages[4], want, m, cout, r1, int8), eg = make(kStages[4], got, m, cout, r2, int8);
      if (int8) {
        const PackedB pb = pack_b(b8.data(), 9 * cin, cout, isa, 9);
        gemm_u8s8({list8.data(), 9}, 5, pb, m, ew);
        gemm_u8s8(grid(in8.data(), int64_t{stride} * cin, 9, offsets.data()), 5, pb, m, eg);
        dwconv_u8s8(grid(in8.data(), int64_t{stride} * cin, 9, offsets.data()), 5, b8.data(), cin, m, eg, isa);
        dwconv_u8s8({list8.data(), 9}, 5, b8.data(), cin, m, ew, isa);
      } else {
        const PackedF32 pb = pack_f32(b32.data(), 9 * cin, cout, isa);
        gemm_f32({list32.data(), 9}, pb, m, ew);
        gemm_f32(grid(in32.data(), int64_t{stride} * cin, 9, offsets.data()), pb, m, eg);
      }
      expect_same(want, got, m, cout, std::string(isa_name(isa)) + (int8 ? " int8" : " fp32"));
    }
  }
}

// A kernel keeps B widened for the next call on the same packing (a layer's rows):
// calls on one packing, another packing of the same shape (the same buffer, other
// weights), the first again, a depthwise call between (it widens its weights into
// the same buffer), and B past the size kept, all with the reference's results.
TEST(CpuGemm, F16KeepsBWidenedForTheSamePacking) {
  std::mt19937 rng(6);
  std::uniform_real_distribution<float> u(-1, 1);
  for (Isa isa : isas_available()) {
    for (const int n : {40, 1100}) {  // 1100 x 576 floats: past the 2 MiB kept
      const int k = 576, m = 9, c = 32;
      std::vector<uint16_t> b1(static_cast<size_t>(k) * n), b2(b1.size()), w(9 * c);
      for (auto& v : b1) v = f32_to_f16(u(rng));
      for (auto& v : b2) v = f32_to_f16(u(rng));
      for (auto& v : w) v = f32_to_f16(u(rng));
      std::vector<float> a32(static_cast<size_t>(m) * k);
      std::vector<uint16_t> a16(a32.size());
      for (size_t i = 0; i < a32.size(); ++i) a16[i] = f32_to_f16(a32[i] = u(rng));
      const PackedF16 p1 = pack_f16(b1.data(), k, n, isa), p2 = pack_f16(b2.data(), k, n, isa);
      const PackedF16 r1 = pack_f16(b1.data(), k, n, Isa::kScalar), r2 = pack_f16(b2.data(), k, n, Isa::kScalar);
      std::vector<const uint16_t*> dw_rows(static_cast<size_t>(m) * 9);
      for (size_t i = 0; i < dw_rows.size(); ++i) dw_rows[i] = a16.data() + (i % 9) * c;
      for (const bool half : {false, true}) {
        for (const int step : {0, 1, 2, 3, 4}) {  // p1, p1, p2, depthwise then p1, p1
          const bool second = step == 2;
          Outputs want, got;
          std::mt19937 r_1(2), r_2(2);
          const Epilogue ew = make(kF16Stages[1], want, m, n, r_1, false);
          const Epilogue eg = make(kF16Stages[1], got, m, n, r_2, false);
          if (step == 3) {
            Outputs dw_out;
            std::mt19937 r_3(3);
            dwconv_f16(Rows<uint16_t>{dw_rows.data(), 9}, w.data(), c, m, make(kF16Stages[0], dw_out, m, c, r_3, false),
                       isa);
          }
          if (half) {
            gemm_f16(rows_of(a16.data(), k), second ? r2 : r1, m, ew);
            gemm_f16(rows_of(a16.data(), k), second ? p2 : p1, m, eg);
          } else {
            gemm_f16(rows_of(a32.data(), k), second ? r2 : r1, m, ew);
            gemm_f16(rows_of(a32.data(), k), second ? p2 : p1, m, eg);
          }
          const std::string where = std::string(isa_name(isa)) + (half ? " A fp16" : " A fp32") + " n " +
                                    std::to_string(n) + " call " + std::to_string(step);
          if (half && f16_arithmetic(isa)) {
            std::vector<double> tol(static_cast<size_t>(m) * n, k * (kF16Run / 2048.0 + 1e-4) + 1e-5);
            expect_near(want, got, m, n, tol, where);
          } else {
            expect_same(want, got, m, n, where);
          }
          if (::testing::Test::HasFatalFailure()) return;
        }
      }
    }
  }
}

// The same for FP16, A in FP16 and in FP32, the GEMM and the depthwise convolution.
TEST(CpuGemm, F16GridRowsAreTheirList) {
  std::mt19937 rng(4);
  const int cin = 20, cout = 24, width = 30, pw = width + 2, stride = 2, m = 14;  // 3x3, stride 2
  std::vector<uint16_t> in16(static_cast<size_t>(3) * pw * cin + 64);
  std::vector<float> in32(in16.size());
  for (size_t i = 0; i < in16.size(); ++i) {
    in32[i] = static_cast<float>(static_cast<int>(rng() % 200) - 100) / 64.0f;
    in16[i] = f32_to_f16(in32[i]);
  }
  std::vector<int64_t> offsets;
  for (int ky = 0; ky < 3; ++ky) {
    for (int kx = 0; kx < 3; ++kx) offsets.push_back((static_cast<int64_t>(ky) * pw + kx) * cin);
  }
  std::vector<const uint16_t*> list16;
  std::vector<const float*> list32;
  for (int i = 0; i < m; ++i) {
    for (int t = 0; t < 9; ++t) {
      list16.push_back(in16.data() + i * stride * cin + offsets[t]);
      list32.push_back(in32.data() + i * stride * cin + offsets[t]);
    }
  }
  std::vector<uint16_t> b(static_cast<size_t>(9) * cin * cout);
  for (auto& v : b) v = f32_to_f16(static_cast<float>(static_cast<int>(rng() % 200) - 100) / 128.0f);
  for (Isa isa : isas_available()) {
    for (const bool half : {true, false}) {
      for (const bool depthwise : {false, true}) {
        Outputs want, got;
        std::mt19937 r1(1), r2(1);
        const int n = depthwise ? cin : cout;
        const Epilogue ew = make(kF16Stages[4], want, m, n, r1, false), eg = make(kF16Stages[4], got, m, n, r2, false);
        const PackedF16 pb = pack_f16(b.data(), 9 * cin, cout, isa);
        const int64_t step = int64_t{stride} * cin;
        if (half && depthwise) {
          dwconv_f16(Rows<uint16_t>{list16.data(), 9}, b.data(), cin, m, ew, isa);
          dwconv_f16(grid(in16.data(), step, 9, offsets.data()), b.data(), cin, m, eg, isa);
        } else if (half) {
          gemm_f16(Rows<uint16_t>{list16.data(), 9}, pb, m, ew);
          gemm_f16(grid(in16.data(), step, 9, offsets.data()), pb, m, eg);
        } else if (depthwise) {
          dwconv_f16(Rows<float>{list32.data(), 9}, b.data(), cin, m, ew, isa);
          dwconv_f16(grid(in32.data(), step, 9, offsets.data()), b.data(), cin, m, eg, isa);
        } else {
          gemm_f16(Rows<float>{list32.data(), 9}, pb, m, ew);
          gemm_f16(grid(in32.data(), step, 9, offsets.data()), pb, m, eg);
        }
        expect_same(want, got, m, n,
                    std::string(isa_name(isa)) + (half ? " A fp16" : " A fp32") + (depthwise ? " depthwise" : ""));
      }
    }
  }
}

TEST(CpuGemm, PackingNeedsTheInstructionSet) {
  const std::vector<int8_t> b(16, 1);
  const std::vector<float> bf(16, 1.0f);
  const std::vector<uint16_t> bh(16, 0x3c00);
  for (Isa isa : {Isa::kNeon, Isa::kAvx2, Isa::kAvxVnni, Isa::kAvx512, Isa::kNeonDot, Isa::kAvx512Fp16,
                  Isa::kNeonFp16}) {
    if (!isa_available(isa)) {
      EXPECT_ANY_THROW(pack_b(b.data(), 4, 4, isa)) << isa_name(isa);
      EXPECT_ANY_THROW(pack_f32(bf.data(), 4, 4, isa)) << isa_name(isa);
      EXPECT_ANY_THROW(pack_f16(bh.data(), 4, 4, isa)) << isa_name(isa);
    }
  }
  EXPECT_ANY_THROW(pack_b(b.data(), 4, 4, Isa::kScalar, 3));  // 4 rows are not 3 taps
  EXPECT_TRUE(f16_arithmetic(Isa::kAvx512Fp16) && f16_arithmetic(Isa::kNeonFp16));
  EXPECT_FALSE(f16_arithmetic(Isa::kAvx2) || f16_arithmetic(Isa::kAvx512) || f16_arithmetic(Isa::kNeon));
}

// The detector's threshold bits on every instruction set, as the reference sets them: values at the threshold (not
// above), NaNs (never above), infinities, lengths that leave a word part full; the last word's high bits clear.
TEST(CpuThreshold, AboveBitsAsTheReference) {
  std::mt19937 rng(29);
  std::uniform_int_distribution<int> pick(0, 9);
  for (const int n : {1, 7, 8, 31, 32, 33, 64, 100, 362}) {
    std::vector<float> v(n);
    for (auto& x : v) {
      const int k = pick(rng);
      x = k == 0 ? std::numeric_limits<float>::quiet_NaN() : k == 1 ? 0.25f : k == 2 ? INFINITY
          : k == 3 ? -INFINITY : 0.1f * static_cast<float>(pick(rng)) - 0.2f;
    }
    const int words = (n + 31) / 32;
    std::vector<uint32_t> want(words, 0xdeadbeef);
    above_bits(v.data(), n, 0.25f, want.data(), Isa::kScalar);
    for (int i = 0; i < n; ++i) ASSERT_EQ((want[i / 32] >> (i % 32)) & 1u, v[i] > 0.25f ? 1u : 0u) << i;
    if (n % 32) ASSERT_EQ(want.back() >> (n % 32), 0u);
    for (Isa isa : isas_available()) {
      std::vector<uint32_t> got(words, 0xdeadbeef);
      above_bits(v.data(), n, 0.25f, got.data(), isa);
      EXPECT_EQ(got, want) << isa_name(isa) << " n " << n;
    }
  }
}

// A crop's bytes at most lo and at least hi on every instruction set, as the reference counts them: widths that leave
// each vector a tail, rows a stride apart, thresholds at and past the bytes' ends.
TEST(CpuCrops, CountBeyondAsTheReference) {
  std::mt19937 rng(23);
  std::uniform_int_distribution<int> byte(0, 255);
  for (const int w : {1, 15, 16, 31, 32, 33, 100, 517}) {
    const int h = 32;
    const int64_t stride = w + 7;
    std::vector<uint8_t> c(static_cast<size_t>(stride) * h);
    for (auto& v : c) v = static_cast<uint8_t>(byte(rng) % 3 == 0 ? 255 : byte(rng));
    for (const auto [lo, hi] : std::vector<std::pair<int, int>>{{100, 160}, {-1, 256}, {0, 255}, {-5, 200}, {40, 300},
                                                                {255, 0}, {128, 128}}) {
      const cpu::ByteCounts want = count_beyond(c.data(), w, h, stride, lo, hi, Isa::kScalar);
      int64_t below = 0, above = 0;
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          below += c[static_cast<size_t>(y) * stride + x] <= lo;
          above += c[static_cast<size_t>(y) * stride + x] >= hi;
        }
      ASSERT_EQ(want.below, below);
      ASSERT_EQ(want.above, above);
      for (Isa isa : isas_available()) {
        const cpu::ByteCounts got = count_beyond(c.data(), w, h, stride, lo, hi, isa);
        EXPECT_EQ(got.below, below) << isa_name(isa) << " w " << w << " lo " << lo;
        EXPECT_EQ(got.above, above) << isa_name(isa) << " w " << w << " hi " << hi;
      }
    }
  }
}

TEST(CpuCtc, ATermOfALogitMoreThan17BelowItsBestCannotMoveASumOfOne) {
  // Past a group's highest best class, every frame's sum is at least 1 (its best's term, exp_neg(0), is exactly 1), and
  // the kernels pass over a class whose terms are all of logits more than 17 below their bests: such a term must be
  // under half an ulp of 1, 2^-24. Every float from -17 to -87, where exp_neg clamps, and the clamp's own value.
  EXPECT_EQ(carbon::cpu::detail::exp_neg(0.0f), 1.0f);
  float worst = 0.0f;
  uint32_t lo, hi;
  const float from = -17.0f, to = -87.0f;
  std::memcpy(&lo, &from, 4);
  std::memcpy(&hi, &to, 4);
  for (uint32_t b = lo + 1; b <= hi + 1; ++b) {  // negative floats grow in magnitude with their bits
    float x;
    std::memcpy(&x, &b, 4);
    const float e = carbon::cpu::detail::exp_neg(x);
    worst = e > worst ? e : worst;
  }
  for (const float x : {-1e30f, -std::numeric_limits<float>::infinity()}) {
    const float e = carbon::cpu::detail::exp_neg(x);
    worst = e > worst ? e : worst;
  }
  EXPECT_LT(worst, std::ldexp(1.0f, -24));
  volatile float one = 1.0f;
  EXPECT_EQ(one + worst, 1.0f);
}

TEST(CpuCtc, EveryKernelGivesTheReferenceBits) {
  // Logits on a coarse grid (ties between classes), narrow and wide (most terms then too small to change a sum, which
  // the kernels pass over), blank-heavy frames, frame counts that leave every vector width a tail, and lines longer
  // than the kernels' chunks of 256 frames, one chunk of them all blank.
  std::mt19937 rng(17);
  std::uniform_int_distribution<int> grid(-24, 24);
  for (const float spread : {0.5f, 3.0f}) {
    for (const int classes : {1, 7, 800}) {
      for (const int frames : {1, 13, 131, 263, 700}) {
        std::vector<float> logits(static_cast<size_t>(classes) * frames);
        for (auto& v : logits) v = spread * static_cast<float>(grid(rng));
        for (int t = 0; t < frames; t += 3) logits[t] += 30.0f;  // class 0, the blank, raised in a third of them
        for (int t = 256; t < 512 && t < frames; ++t) logits[t] = 100.0f;  // every frame of the second chunk
        const int t_n = frames - frames / 5;
        std::vector<int32_t> ref_ids(frames, -1);
        std::vector<float> ref_probs(frames, -1.0f);
        ctc_argmax(logits.data(), classes, frames, t_n, ref_ids.data(), ref_probs.data(), Isa::kScalar);
        for (int t = 0; t < t_n; ++t) {  // the reference against the definition
          int best = 0;
          for (int c = 1; c < classes; ++c)
            if (logits[static_cast<size_t>(c) * frames + t] > logits[static_cast<size_t>(best) * frames + t]) best = c;
          ASSERT_EQ(ref_ids[t], best);
          if (best == 0) continue;
          double sum = 0;
          for (int c = 0; c < classes; ++c)
            sum += std::exp(double(logits[static_cast<size_t>(c) * frames + t]) - logits[static_cast<size_t>(best) * frames + t]);
          EXPECT_NEAR(ref_probs[t], 1.0 / sum, 2e-5 / sum);  // a float sum of up to 800 terms
        }
        for (Isa isa : isas_available()) {
          std::vector<int32_t> ids(frames, -1);
          std::vector<float> probs(frames, -1.0f);
          ctc_argmax(logits.data(), classes, frames, t_n, ids.data(), probs.data(), isa);
          EXPECT_EQ(ids, ref_ids) << isa_name(isa) << " " << classes << " x " << frames;
          for (int t = 0; t < t_n; ++t) {
            if (ref_ids[t] == 0) continue;  // a blank frame's probability is not written
            uint32_t a, b;
            std::memcpy(&a, &probs[t], 4);
            std::memcpy(&b, &ref_probs[t], 4);
            EXPECT_EQ(a, b) << isa_name(isa) << " frame " << t;
          }
        }
      }
    }
  }
}


// The detector's head in one pass (det_head) gives, bit for bit, what its four steps give one after another over
// whole tensors (a 1x1 and ReLU, a nearest 2x upsample, a depthwise 3x3 with a zero border and ReLU, a 1x1), each an
// FMA chain from its bias in the same order; on every instruction set, and in row bands cut anywhere.
TEST(Cpu, DetectorHeadInOnePassAsItsStepsGiveIt) {
  std::mt19937 rng(5);
  std::normal_distribution<float> n(0.0f, 0.5f);
  for (const auto& [in, mid, out, h4, w4] : std::vector<std::array<int, 5>>{{5, 3, 4, 7, 13}, {8, 8, 4, 4, 6},
                                                                             {32, 16, 4, 3, 11}, {2, 2, 1, 1, 1}}) {
    std::vector<float> x(static_cast<size_t>(in) * h4 * w4), w1(mid * in), b1(mid), w2(mid * 9), b2(mid), w3(out * mid),
        b3(out);
    for (auto* v : {&x, &w1, &b1, &w2, &b2, &w3, &b3}) {
      for (float& e : *v) e = n(rng);
    }
    const DetHeadWeights w{in, mid, out, w1.data(), b1.data(), w2.data(), b2.data(), w3.data(), b3.data()};
    const int H = 2 * h4, W = 2 * w4;
    std::vector<float> head(static_cast<size_t>(mid) * h4 * w4), up(static_cast<size_t>(mid) * H * W),
        dw(static_cast<size_t>(mid) * H * W), want(static_cast<size_t>(out) * H * W);
    for (int c = 0; c < mid; ++c) {
      for (int p = 0; p < h4 * w4; ++p) {
        float acc = b1[c];
        for (int i = 0; i < in; ++i) acc = std::fma(w1[c * in + i], x[static_cast<size_t>(i) * h4 * w4 + p], acc);
        head[static_cast<size_t>(c) * h4 * w4 + p] = acc > 0.0f ? acc : 0.0f;
      }
      for (int y = 0; y < H; ++y) {
        for (int xx = 0; xx < W; ++xx) {
          up[(static_cast<size_t>(c) * H + y) * W + xx] = head[(static_cast<size_t>(c) * h4 + y / 2) * w4 + xx / 2];
        }
      }
      for (int y = 0; y < H; ++y) {
        for (int xx = 0; xx < W; ++xx) {
          float acc = b2[c];
          for (int ky = 0; ky < 3; ++ky) {
            for (int kx = 0; kx < 3; ++kx) {
              const int yy = y + ky - 1, xk = xx + kx - 1;
              const float v = yy < 0 || yy >= H || xk < 0 || xk >= W ? 0.0f : up[(static_cast<size_t>(c) * H + yy) * W + xk];
              acc = std::fma(w2[c * 9 + ky * 3 + kx], v, acc);
            }
          }
          dw[(static_cast<size_t>(c) * H + y) * W + xx] = acc > 0.0f ? acc : 0.0f;
        }
      }
    }
    for (int o = 0; o < out; ++o) {
      for (int p = 0; p < H * W; ++p) {
        float acc = b3[o];
        for (int c = 0; c < mid; ++c) acc = std::fma(w3[o * mid + c], dw[static_cast<size_t>(c) * H * W + p], acc);
        want[static_cast<size_t>(o) * H * W + p] = acc;
      }
    }
    for (Isa isa : isas_available()) {
      std::vector<float> got(want.size(), -1.0f);
      det_head(x.data(), h4, w4, w, got.data(), 0, H, isa);
      EXPECT_EQ(std::memcmp(got.data(), want.data(), want.size() * sizeof(float)), 0) << isa_name(isa) << " " << in << "x" << h4 << "x" << w4;
      std::vector<float> bands(want.size(), -1.0f);
      const int cuts[] = {0, 1, H / 2, H / 2, H};
      for (int b = 0; b + 1 < 5; ++b) det_head(x.data(), h4, w4, w, bands.data(), cuts[b], cuts[b + 1], isa);
      EXPECT_EQ(std::memcmp(bands.data(), want.data(), want.size() * sizeof(float)), 0) << isa_name(isa) << " in bands";
    }
  }
}

}  // namespace
}  // namespace carbon::cpu
