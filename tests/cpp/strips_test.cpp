#include "carbon/strips.h"

#include <gtest/gtest.h>

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

}  // namespace
}  // namespace carbon
