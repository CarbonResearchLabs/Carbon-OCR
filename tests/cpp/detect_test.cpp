#include "carbon/detect.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

namespace carbon {
namespace {

constexpr int kH = 512, kW = 400;  // the maps of a 1024 x 800 page
constexpr size_t kPlane = static_cast<size_t>(kH) * kW;

// Targets as a detector would ideally give them: the 0/1 maps as confident logits.
std::vector<float> ideal_maps(const std::vector<LineBox>& lines, const std::vector<LineBox>& pictures = {}) {
  std::vector<float> m(4 * kPlane);
  det_targets(lines, pictures, kH, kW, m.data());
  for (size_t i = 0; i < kPlane; ++i) {
    m[i] = m[i] > 0.5f ? 8.0f : -8.0f;
    m[3 * kPlane + i] = m[3 * kPlane + i] > 0.5f ? 8.0f : -8.0f;
  }
  return m;
}

// A page of two columns of lines, 10 to 30 px high, set 1.2 to 1.6 times their
// height apart, of random length.
std::vector<LineBox> random_page(std::mt19937& rng) {
  std::uniform_real_distribution<double> height(10, 30), leading(1.2, 1.6), u(0, 1);
  std::vector<LineBox> lines;
  for (double col : {20.0, 420.0}) {
    for (double y = 30; ; ) {
      const double h = height(rng), x0 = col + 30 * u(rng), x1 = col + 120 + 230 * u(rng);
      if (y + h > 1000) break;
      lines.push_back({x0, y, x1, y + h});
      y += h * leading(rng);
    }
  }
  return lines;
}

// decode_det(det_targets(boxes)) gives the boxes back: the top and bottom
// exactly (the distance maps carry them), the ends within the map's rounding.
TEST(Detect, DecodingTheTargetsGivesTheBoxesBack) {
  std::mt19937 rng(7);
  for (int page = 0; page < 20; ++page) {
    const std::vector<LineBox> lines = random_page(rng);
    const Detection d = decode_det(ideal_maps(lines).data(), kH, kW);
    ASSERT_EQ(d.lines.size(), lines.size()) << "page " << page;
    std::vector<LineBox> want = lines;
    std::sort(want.begin(), want.end(),
              [](const LineBox& a, const LineBox& b) { return a.y0 != b.y0 ? a.y0 < b.y0 : a.x0 < b.x0; });
    for (size_t i = 0; i < want.size(); ++i) {
      const LineBox& got = d.lines[i].box;
      EXPECT_NEAR(got.y0, want[i].y0, 1e-4) << "page " << page << " line " << i;
      EXPECT_NEAR(got.y1, want[i].y1, 1e-4);
      EXPECT_LE(std::abs(got.x0 - want[i].x0), 1.0);
      EXPECT_LE(std::abs(got.x1 - want[i].x1), 1.0);
      EXPECT_GT(d.lines[i].score, 0.99f);
    }
  }
}

TEST(Detect, BoxesScaleToThePage) {
  const std::vector<LineBox> lines = {{100, 200, 500, 220}};
  const Detection a = decode_det(ideal_maps(lines).data(), kH, kW);
  const Detection b = decode_det(ideal_maps(lines).data(), kH, kW, 2.125);
  ASSERT_EQ(a.lines.size(), 1u);
  ASSERT_EQ(b.lines.size(), 1u);
  EXPECT_NEAR(b.lines[0].box.y0, 2.125 * a.lines[0].box.y0, 1e-9);
  EXPECT_NEAR(b.lines[0].box.x1, 2.125 * a.lines[0].box.x1, 1e-9);
  EXPECT_NEAR(a.median_line_height(), 20, 1e-4);
}

// Two lines set so close that their bands overlap: the shared pixels go to
// neither, so they still decode as two.
TEST(Detect, LinesWhoseBandsWouldTouchStayApart) {
  // bands: rows 112 to 128, and 117 to 133
  const std::vector<LineBox> lines = {{100, 100, 400, 140}, {100, 105, 400, 145}};
  const Detection d = decode_det(ideal_maps(lines).data(), kH, kW);
  ASSERT_EQ(d.lines.size(), 2u);
  EXPECT_NEAR(d.lines[0].box.y0, 100, 1e-4);
  EXPECT_NEAR(d.lines[1].box.y1, 145, 1e-4);
}

// A line slanted by 3 px over its length: its box holds both ends.
TEST(Detect, ASlantedLinesBoxHoldsItsEnds) {
  std::vector<float> m(4 * kPlane, 0.0f);
  for (size_t i = 0; i < kPlane; ++i) m[i] = m[3 * kPlane + i] = -8.0f;
  for (int c = 50; c < 250; ++c) {
    const double line_top = 200 + 3.0 * (c - 50) / 200, line_bottom = line_top + 20;  // input px
    for (int r = 0; r < kH; ++r) {
      const double y = 2.0 * r + 1;
      if (y < line_top + 6 || y > line_bottom - 6) continue;  // the middle 40%
      const size_t k = static_cast<size_t>(r) * kW + c;
      m[k] = 8.0f;
      m[kPlane + k] = static_cast<float>(y - line_top);
      m[2 * kPlane + k] = static_cast<float>(line_bottom - y);
    }
  }
  const Detection d = decode_det(m.data(), kH, kW);
  ASSERT_EQ(d.lines.size(), 1u);
  EXPECT_LE(d.lines[0].box.y0, 200 + 0.3);   // within the top 10% of its tops
  EXPECT_GE(d.lines[0].box.y1, 223 - 0.3);   // and of its bottoms
  EXPECT_NEAR(d.lines[0].box.x0, 100 - 0.15 * (d.lines[0].box.y1 - d.lines[0].box.y0), 1e-6);
}

TEST(Detect, PicturesAndNoise) {
  const std::vector<LineBox> lines = {{100, 600, 500, 620}};
  const std::vector<LineBox> pictures = {{200, 100, 600, 400}};
  std::vector<float> m = ideal_maps(lines, pictures);
  m[static_cast<size_t>(10) * kW + 10] = 8.0f;  // one stray core pixel: noise
  const Detection d = decode_det(m.data(), kH, kW);
  ASSERT_EQ(d.lines.size(), 1u);
  ASSERT_EQ(d.pictures.size(), 1u);
  const LineBox& p = d.pictures[0].box;
  EXPECT_LE(std::abs(p.x0 - 200), 1.0);
  EXPECT_LE(std::abs(p.y0 - 100), 1.0);
  EXPECT_LE(std::abs(p.x1 - 600), 1.0);
  EXPECT_LE(std::abs(p.y1 - 400), 1.0);
  EXPECT_EQ(d.pictures[0].pixels, 200 * 150);
}

// Every line gets a band, even one too small to hold a pixel centre.
TEST(Detect, TinyLinesStillGetABand) {
  std::vector<float> m(4 * kPlane);
  const std::vector<LineBox> lines = {{100, 100, 103, 104}};
  det_targets(lines, {}, kH, kW, m.data());
  float sum = 0;
  for (size_t i = 0; i < kPlane; ++i) sum += m[i];
  EXPECT_EQ(sum, 1.0f);
}

TEST(Detect, TheMasksHoldTheSetPixelsAndTheirValues) {
  // A few set pixels across word boundaries; their values in row-major order.
  const int h = 3, w = 70;
  const size_t plane = static_cast<size_t>(h) * w;
  std::vector<float> m(4 * plane, -8.0f);
  const int core[] = {0 * w + 0, 0 * w + 31, 0 * w + 32, 1 * w + 69, 2 * w + 40};
  for (int i : core) {
    m[i] = 3.0f + i;
    m[plane + i] = 1.0f + i;
    m[2 * plane + i] = 2.0f + i;
  }
  m[3 * plane + 2 * w + 5] = 4.0f;
  const DetMasks k = threshold_det(m.data(), h, w);
  ASSERT_EQ(k.words, 3);
  EXPECT_EQ(k.core[0], 0x80000001u);  // pixels 0 and 31
  EXPECT_EQ(k.core[1], 1u);           // pixel 32
  EXPECT_EQ(k.core[1 * 3 + 2], 1u << 5);  // pixel 69 = 64 + 5
  EXPECT_EQ(k.core[2 * 3 + 1], 1u << 8);  // pixel 40
  EXPECT_EQ(k.picture[2 * 3 + 0], 1u << 5);
  ASSERT_EQ(k.core_values.size(), 15u);
  for (size_t j = 0; j < 5; ++j) {
    EXPECT_EQ(k.core_values[3 * j], 3.0f + core[j]);
    EXPECT_EQ(k.core_values[3 * j + 1], 1.0f + core[j]);
    EXPECT_EQ(k.core_values[3 * j + 2], 2.0f + core[j]);
  }
  EXPECT_EQ(k.picture_values, std::vector<float>{4.0f});
}

TEST(Detect, DecodingTheMasksIsDecodingTheMaps) {
  std::mt19937 rng(11);
  const std::vector<float> m = ideal_maps(random_page(rng), {{100, 600, 300, 800}});
  const Detection a = decode_det(m.data(), kH, kW, 1.5), b = decode_det(threshold_det(m.data(), kH, kW), 1.5);
  ASSERT_EQ(a.lines.size(), b.lines.size());
  ASSERT_EQ(a.pictures.size(), b.pictures.size());
  for (size_t i = 0; i < a.lines.size(); ++i) {
    EXPECT_EQ(a.lines[i].box.x0, b.lines[i].box.x0);
    EXPECT_EQ(a.lines[i].box.y1, b.lines[i].box.y1);
    EXPECT_EQ(a.lines[i].score, b.lines[i].score);
  }
}

TEST(Detect, NothingToFind) {
  std::vector<float> m(4 * kPlane, -8.0f);
  const Detection d = decode_det(m.data(), kH, kW);
  EXPECT_TRUE(d.lines.empty());
  EXPECT_TRUE(d.pictures.empty());
  EXPECT_EQ(d.median_line_height(), 0);
  EXPECT_FALSE(takes_second_pass(d));
}

Detection lines_of(const std::vector<LineBox>& boxes, float score = 0.5f) {
  Detection d;
  for (const LineBox& b : boxes) d.lines.push_back({b, score, 10});
  return d;
}

// A page takes the second pass where its median line, at the detector's scale, is
// under 9 px: a page of tiny lines with a heading does, a page of print with a
// footnote does not, and neither does an empty one.
TEST(Detect, OnlyAPageOfTinyLinesTakesTheSecondPass) {
  std::vector<LineBox> tiny, print;
  for (int i = 0; i < 9; ++i) {
    tiny.push_back({40, 100.0 + 12 * i, 700, 107.0 + 12 * i});    // 7 px
    print.push_back({40, 100.0 + 24 * i, 700, 116.0 + 24 * i});   // 16 px
  }
  tiny.push_back({40, 40, 600, 70});        // a heading
  print.push_back({40, 900, 600, 906});     // a footnote
  EXPECT_TRUE(takes_second_pass(decode_det(ideal_maps(tiny).data(), kH, kW)));
  EXPECT_FALSE(takes_second_pass(decode_det(ideal_maps(print).data(), kH, kW)));
  EXPECT_TRUE(takes_second_pass(lines_of({{0, 0, 100, 8.9}})));
  EXPECT_FALSE(takes_second_pass(lines_of({{0, 0, 100, 9.0}})));
  EXPECT_FALSE(takes_second_pass(Detection{}));
  DetRule never;
  never.second_pass_below = 0;
  EXPECT_FALSE(takes_second_pass(lines_of(tiny), never));
  // the boxes at the detector's scale: decoded at the page's (scale 2), the
  // tiny page's lines are 14 px
  EXPECT_FALSE(takes_second_pass(decode_det(ideal_maps(tiny).data(), kH, kW, 2.0)));
}

// Merging the passes (page pixels, the first pass at half the page's scale): a line
// both found is kept from the pass whose scale suits it (the second's for a line
// under 9 px at the first's scale, 18 here), every other line of either kept, each
// line in one pair at most; pictures likewise; all in reading order.
TEST(Detect, MergingKeepsEachLineFromThePassThatSuitsIt) {
  constexpr float kFirst = 0.25f, kSecond = 0.75f;  // which pass a box came from
  Detection first = lines_of({{100, 100, 500, 112},    // tiny, in both
                              {600, 100, 900, 112},    // tiny, the first's alone
                              {100, 200, 900, 240},    // print, in both
                              {100, 400, 300, 412},    // under the IoU with the second's
                              {100, 500, 500, 512}},   // over it with two of the second's
                             kFirst);
  Detection second = lines_of({{102, 99, 498, 111},
                               {104, 202, 896, 238},
                               {100, 300, 400, 311},   // the second's alone
                               {250, 400, 450, 412},
                               {110, 500, 500, 512},
                               {100, 500, 500, 512}},
                              kSecond);
  first.pictures.push_back({{50, 600, 950, 900}, kFirst, 100});
  second.pictures.push_back({{52, 602, 948, 898}, kSecond, 400});
  second.pictures.push_back({{50, 950, 300, 1000}, kSecond, 50});
  const Detection m = merge_passes(first, second, 2.0);
  struct Want {
    LineBox box;
    float score;
  };
  const std::vector<Want> want = {{{102, 99, 498, 111}, kSecond}, {{600, 100, 900, 112}, kFirst},
                                   {{100, 200, 900, 240}, kFirst}, {{100, 300, 400, 311}, kSecond},
                                   {{100, 400, 300, 412}, kFirst}, {{250, 400, 450, 412}, kSecond},
                                   {{100, 500, 500, 512}, kSecond}, {{110, 500, 500, 512}, kSecond}};
  ASSERT_EQ(m.lines.size(), want.size());
  for (size_t i = 0; i < want.size(); ++i) {
    EXPECT_EQ(m.lines[i].box.x0, want[i].box.x0) << "line " << i;
    EXPECT_EQ(m.lines[i].box.y0, want[i].box.y0) << "line " << i;
    EXPECT_EQ(m.lines[i].box.x1, want[i].box.x1) << "line " << i;
    EXPECT_EQ(m.lines[i].box.y1, want[i].box.y1) << "line " << i;
    EXPECT_EQ(m.lines[i].score, want[i].score) << "line " << i;
  }
  ASSERT_EQ(m.pictures.size(), 2u);
  EXPECT_EQ(m.pictures[0].score, kFirst);
  EXPECT_EQ(m.pictures[0].box.y0, 600);
  EXPECT_EQ(m.pictures[1].box.y0, 950);

  // At the page's own scale the 12-px line is not tiny: the first's box
  const Detection same = merge_passes(first, second, 1.0);
  ASSERT_EQ(same.lines.size(), want.size());
  EXPECT_EQ(same.lines[0].box.y0, 100);
  EXPECT_EQ(same.lines[0].score, kFirst);
  // nothing found twice: everything kept
  EXPECT_EQ(merge_passes(first, Detection{}, 2.0).lines.size(), first.lines.size());
  EXPECT_EQ(merge_passes(Detection{}, second, 2.0).lines.size(), second.lines.size());
}

// A first box over text the second pass read as tiny lines goes, so nothing is read
// twice: two tiny lines the first pass ran together, or one it boxed loosely. A first
// box over lines the second pass found large (a heading the larger scale split) stays.
TEST(Detect, MergingDropsAFirstBoxTheSecondPassReadAsTinyLines) {
  constexpr float kFirst = 0.25f, kSecond = 0.75f;
  const Detection first = lines_of({{100, 700, 500, 724},    // two tiny lines run together
                                    {600, 700, 900, 724},    // one tiny line, boxed loosely
                                    {100, 800, 900, 860}},   // a heading
                                   kFirst);
  const Detection second = lines_of({{100, 700, 500, 711},
                                     {100, 713, 500, 724},
                                     {600, 705, 900, 716},
                                     {100, 800, 500, 860},
                                     {510, 800, 900, 860}},
                                    kSecond);
  const Detection m = merge_passes(first, second, 2.0);  // tiny: under 18 page pixels
  std::vector<std::pair<double, float>> got;  // (top, pass)
  for (const DetBox& l : m.lines) got.emplace_back(l.box.y0, l.score);
  const std::vector<std::pair<double, float>> want = {
      {700, kSecond}, {705, kSecond}, {713, kSecond}, {800, kFirst}, {800, kSecond}, {800, kSecond}};
  std::sort(got.begin(), got.end());
  EXPECT_EQ(got, want);
}

}  // namespace
}  // namespace carbon
