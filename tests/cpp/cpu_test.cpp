#include "carbon/cpu.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

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
// out (relu, a residual of either kind).
struct Stage {
  const char* name;
  bool scale, bias, relu, residual_u8, residual_f32, out_u8;
};
constexpr Stage kStages[] = {
    {"sums", false, false, false, false, false, false},
    {"scale bias", true, true, false, false, false, false},
    {"bias relu residual", false, true, true, false, true, false},
    {"u8 relu", true, true, true, false, false, true},
    {"u8 residual u8", true, true, false, true, false, true},
    {"u8 residual f32 relu", true, false, true, false, true, true},
};

struct Outputs {
  std::vector<float> scale, bias, res_f32, out_f32;
  std::vector<uint8_t> res_u8, out_u8;
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
  for (auto& v : o.res_f32) v = u(rng);
  for (auto& v : o.res_u8) v = static_cast<uint8_t>(rng());
  o.out_f32.assign(static_cast<size_t>(m) * o.ld, -7.0f);
  o.out_u8.assign(static_cast<size_t>(m) * o.ld, 0x5a);
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
  e.out_ld = o.ld;
  if (s.out_u8) {
    e.out_u8 = o.out_u8.data();
    e.out_inv_scale = 40.0f;
    e.out_zero = s.relu ? 0 : 128;
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
      if (j >= n) {
        ASSERT_EQ(got.out_f32[at], -7.0f) << where << " wrote past n at " << i << "," << j;
        ASSERT_EQ(got.out_u8[at], 0x5a) << where << " wrote past n at " << i << "," << j;
      }
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

TEST(CpuGemm, PackingNeedsTheInstructionSet) {
  const std::vector<int8_t> b(16, 1);
  const std::vector<float> bf(16, 1.0f);
  for (Isa isa : {Isa::kNeon, Isa::kAvx2, Isa::kAvxVnni, Isa::kAvx512, Isa::kNeonDot}) {
    if (!isa_available(isa)) {
      EXPECT_ANY_THROW(pack_b(b.data(), 4, 4, isa)) << isa_name(isa);
      EXPECT_ANY_THROW(pack_f32(bf.data(), 4, 4, isa)) << isa_name(isa);
    }
  }
  EXPECT_ANY_THROW(pack_b(b.data(), 4, 4, Isa::kScalar, 3));  // 4 rows are not 3 taps
}

TEST(CpuCtc, EveryKernelGivesTheReferenceBits) {
  // Logits on a coarse grid (ties between classes), blank-heavy frames, and frame counts
  // that leave every vector width a tail.
  std::mt19937 rng(17);
  std::uniform_int_distribution<int> grid(-24, 24);
  for (const int classes : {1, 7, 800}) {
    for (const int frames : {1, 13, 131}) {
      std::vector<float> logits(static_cast<size_t>(classes) * frames);
      for (auto& v : logits) v = 0.5f * static_cast<float>(grid(rng));
      for (int t = 0; t < frames; t += 3) logits[t] += 30.0f;  // class 0, the blank, wins a third of them
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

}  // namespace
}  // namespace carbon::cpu
