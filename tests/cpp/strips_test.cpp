#include "carbon/strips.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "carbon/error.h"

namespace carbon {
namespace {

// The strips' kept rows cover the page's output once, in order; every cut is on
// a multiple of align; each strip has its halo where the page has rows for it.
TEST(Strips, CoverThePageOnceWithTheirHalos) {
  const StripRule rule;  // halo 128, align 32, maps at stride 2
  for (int n : {1, 2, 3, 4, 8, 16}) {
    const auto plan = plan_strips(1024, n, rule);
    ASSERT_EQ(plan.size(), static_cast<size_t>(n));
    int next = 0;
    for (const Strip& s : plan) {
      EXPECT_EQ(s.out0, next);
      const int c0 = s.out0 * rule.out_stride;
      const int c1 = c0 + (s.keep1 - s.keep0) * rule.out_stride;
      EXPECT_EQ(c0 % rule.align, 0);
      EXPECT_EQ(s.y0 % rule.align, 0);
      EXPECT_EQ(s.y0, std::max(0, c0 - rule.halo));
      EXPECT_EQ(s.y1, std::min(1024, c1 + rule.halo));
      EXPECT_EQ(s.keep0, (c0 - s.y0) / rule.out_stride);
      EXPECT_LE(s.keep1, (s.y1 - s.y0) / rule.out_stride);
      next += s.keep1 - s.keep0;
    }
    EXPECT_EQ(next, 512);
  }
  EXPECT_EQ(plan_strips(64, 8, rule).size(), 2u);  // no more strips than 32-row blocks
}

TEST(Strips, RejectPagesAndRulesThatDoNotFit) {
  EXPECT_THROW(plan_strips(1000, 4), Error);                // not a multiple of 32
  EXPECT_THROW(plan_strips(1024, 4, {100, 32, 2}), Error);  // halo off the stride grid
  EXPECT_THROW(plan_strips(1024, 4, {128, 32, 3}), Error);
}

TEST(Strips, InputAndStitchMoveWholeRows) {
  std::vector<float> v(2 * 8 * 3);
  for (size_t i = 0; i < v.size(); ++i) v[i] = static_cast<float>(i);
  const Tensor page = Tensor::from<float>({1, 2, 8, 3}, v);  // two planes of 8 rows
  const Strip s{2, 6, 1, 3, 4};
  const Tensor in = strip_input(page, s);
  ASSERT_EQ(in.shape(), (Shape{1, 2, 4, 3}));
  EXPECT_EQ(in.data<float>()[0], 6.0f);   // plane 0, row 2
  EXPECT_EQ(in.data<float>()[12], 30.0f); // plane 1, row 2
  Tensor out(DType::kFloat32, {1, 2, 8, 3});
  stitch_strip(in, s, out);  // its rows 1 and 2 to rows 4 and 5
  EXPECT_EQ(out.data<float>()[4 * 3], 9.0f);
  EXPECT_EQ(out.data<float>()[5 * 3 + 2], 14.0f);
  EXPECT_EQ(out.data<float>()[24 + 4 * 3], 33.0f);
  EXPECT_EQ(out.data<float>()[3 * 3], 0.0f);  // rows outside the strip untouched
  Tensor narrow(DType::kFloat32, {1, 2, 8, 2});
  EXPECT_THROW(stitch_strip(in, s, narrow), Error);
  EXPECT_THROW(strip_input(page, {6, 10, 0, 1, 0}), Error);
}

// A model in miniature, [N, 1, H, W] to maps [N, 2, H / 2, W]: each map row the
// sum of the 8 input rows around its own (zero past the input's edges, as a
// convolution's padding) and its row doubled, so its receptive field reaches 4
// rows.
Tensor rows_model(const Tensor& x) {
  const Shape& s = x.shape();
  const int64_t n = s[0], h = s[2], w = s[3];
  Tensor y(DType::kFloat32, {n, 2, h / 2, w});
  const auto in = x.data<float>();
  auto out = y.data<float>();
  for (int64_t b = 0; b < n; ++b) {
    for (int64_t r = 0; r < h / 2; ++r) {
      for (int64_t c = 0; c < w; ++c) {
        float sum = 0;
        for (int64_t k = 2 * r - 3; k <= 2 * r + 4; ++k) sum += k >= 0 && k < h ? in[(b * h + k) * w + c] : 0.0f;
        out[((b * 2) * (h / 2) + r) * w + c] = sum;
        out[((b * 2 + 1) * (h / 2) + r) * w + c] = 2 * in[(b * h + 2 * r) * w + c];
      }
    }
  }
  return y;
}

// A model run in strips gives a whole run's bytes where the halo covers its
// receptive field, one strip being the whole run; with no halo it does not.
TEST(Strips, RunInStripsGivesAWholeRunsBytes) {
  std::vector<float> v(2 * 128 * 5);
  uint32_t r = 7;
  for (float& x : v) {
    r = r * 1664525u + 1013904223u;
    x = static_cast<float>(r >> 8) / 16777216.0f;
  }
  const Tensor page = Tensor::from<float>({2, 1, 128, 5}, v);
  const Tensor whole = rows_model(page);
  auto same = [&](const Tensor& t) {
    return t.shape() == whole.shape() && std::memcmp(t.raw(), whole.raw(), whole.bytes()) == 0;
  };
  int runs = 0;
  auto counted = [&](const Tensor& x) {
    ++runs;
    return rows_model(x);
  };
  for (int n : {1, 3, 4, 16}) {
    runs = 0;
    EXPECT_TRUE(same(run_in_strips(page, n, counted, {8, 8, 2}))) << n << " strips";
    EXPECT_EQ(runs, n);
  }
  EXPECT_FALSE(same(run_in_strips(page, 4, rows_model, {0, 8, 2})));
  EXPECT_THROW(run_in_strips(Tensor(DType::kFloat32, {1, 128, 5}), 2, rows_model), Error);
}

}  // namespace
}  // namespace carbon
