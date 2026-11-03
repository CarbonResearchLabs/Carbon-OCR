// The crop rule and CTC greedy decoding on the CPU.

#include "carbon/lines.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "carbon/error.h"

namespace {

using carbon::CropRegion;
using carbon::CropRule;
using carbon::GrayView;
using carbon::LineBox;

struct Page {
  int w, h;
  std::vector<uint8_t> px;
  Page(int width, int height, uint8_t v = 255) : w(width), h(height), px(static_cast<size_t>(width) * height, v) {}
  GrayView view() const { return {px.data(), w, h, w}; }
  void fill(int x0, int y0, int x1, int y1, uint8_t v) {
    for (int y = y0; y < y1; ++y)
      for (int x = x0; x < x1; ++x) px[static_cast<size_t>(y) * w + x] = v;
  }
};

TEST(CropRule, RegionPadsScalesAndRoundsTheWidth) {
  const CropRegion r = carbon::crop_region({100, 50, 400, 80});  // h = 30
  EXPECT_DOUBLE_EQ(r.y0, 44.0);
  EXPECT_DOUBLE_EQ(r.y1, 86.0);
  EXPECT_EQ(r.width, 248);  // (300 + 2 x 9) x 32 / 42 = 242.3, up to a multiple of 8
  EXPECT_LE(r.x0, 100 - 0.3 * 30);
  EXPECT_GE(r.x1, 400 + 0.3 * 30);
  EXPECT_NEAR((r.x1 - r.x0) / (r.y1 - r.y0), r.width / 32.0, 1e-9);  // no distortion
  EXPECT_EQ(carbon::ctc_frames({100, 50, 400, 80}), 62);
  CropRule tall;
  tall.height = 48;
  EXPECT_GT(carbon::crop_region({100, 50, 400, 80}, tall).width, r.width);
}

TEST(CropRule, IdentityResampleKeepsEveryByte) {
  Page p(40, 32);
  for (int i = 0; i < 40 * 32; ++i) p.px[i] = static_cast<uint8_t>((i * 37) % 256);
  std::vector<uint8_t> out(40 * 32);
  carbon::resample_gray(p.view(), 0, 0, 40, 32, 40, 32, out.data(), 40);
  EXPECT_EQ(out, p.px);
}

TEST(CropRule, PaperStaysPaperAndOutsideThePageIsPaper) {
  Page p(200, 100);
  std::vector<uint8_t> out(32 * 512);
  std::vector<int32_t> w(1);
  const LineBox box{-50, -20, 300, 140};  // mostly off the page
  carbon::crop_lines(p.view(), {&box, 1}, 512, out.data(), w);
  for (uint8_t v : out) ASSERT_EQ(v, 255);
}

TEST(CropRule, InkLandsWhereTheBoxSaysAndThePadIsTheGround) {
  Page p(600, 200);
  p.fill(150, 70, 450, 100, 0);  // a solid "line" of ink
  const LineBox box{150, 70, 450, 100};
  const int out_w = 400;
  std::vector<uint8_t> out(32 * out_w);
  std::vector<int32_t> w(1);
  carbon::crop_lines(p.view(), {&box, 1}, out_w, out.data(), w);
  ASSERT_EQ(w[0], 248);
  auto at = [&](int x, int y) { return out[static_cast<size_t>(y) * out_w + x]; };
  EXPECT_EQ(at(w[0] / 2, 16), 0);     // the middle is ink
  EXPECT_EQ(at(1, 1), 255);           // the corners are paper
  EXPECT_EQ(at(w[0] - 2, 30), 255);
  for (int y = 0; y < 32; ++y)        // past the width: the ground, here paper
    for (int x = w[0]; x < out_w; ++x) ASSERT_EQ(at(x, y), 255);
  std::vector<uint8_t> narrow(32 * 200);
  EXPECT_THROW(carbon::crop_lines(p.view(), {&box, 1}, 200, narrow.data(), w), carbon::Error);
}

TEST(CropRule, GroundIsTheMedianOfTheBorder) {
  // 4 x 3 crop: first row 10 10 10 10, last row 20 20 20 20, last column 10 30 20
  std::vector<uint8_t> c = {10, 10, 10, 10, 0, 0, 0, 30, 20, 20, 20, 20};
  // the border values: 10 x4, 20 x4, 10, 30, 20 -> 11 values, median 20
  EXPECT_EQ(carbon::crop_ground(c.data(), 4, 3, 4), 20);
  // 2 x 2 crops: six border values, so the two middle ones are averaged
  std::vector<uint8_t> d = {10, 30, 10, 30};  // 10 30 | 10 30 | 30 30 -> 10 10 30 30 30 30
  EXPECT_EQ(carbon::crop_ground(d.data(), 2, 2, 2), 30);
  std::vector<uint8_t> f = {11, 11, 20, 20};  // 11 11 | 20 20 | 11 20 -> 11 11 11 20 20 20
  EXPECT_EQ(carbon::crop_ground(f.data(), 2, 2, 2), 15);  // (11 + 20) / 2, rounded down
}

TEST(CropRule, LightOnDarkIsFlippedAndThenPaddedWithItsGround) {
  // A 16 x 32 crop of white strokes on a dark ground, in rows 24 wide.
  const CropRule rule;
  std::vector<uint8_t> c(32 * 24, 7);
  for (int y = 10; y < 22; ++y)
    for (int x = 3; x < 13; x += 3) c[y * 24 + x] = 230;
  EXPECT_TRUE(carbon::light_on_dark(c.data(), 16, 32, 24, rule.flip_margin));
  EXPECT_FALSE(carbon::light_on_dark(c.data(), 16, 32, 24, 0));  // a margin of 0 never flips
  carbon::finish_crop(c.data(), 16, 24, rule);
  EXPECT_EQ(c[0], 248);          // the ground, flipped
  EXPECT_EQ(c[10 * 24 + 3], 25);  // a stroke, now dark
  EXPECT_EQ(c[5 * 24 + 20], 248);  // the padding: the flipped crop's ground
  EXPECT_FALSE(carbon::light_on_dark(c.data(), 16, 32, 24, rule.flip_margin));
  // Dark ink on paper, and paper alone, are left as they are.
  std::vector<uint8_t> d(32 * 16, 250);
  d[12 * 16 + 4] = 10;
  EXPECT_FALSE(carbon::light_on_dark(d.data(), 16, 32, 16, rule.flip_margin));
  std::vector<uint8_t> e(32 * 16, 120);
  EXPECT_FALSE(carbon::light_on_dark(e.data(), 16, 32, 16, rule.flip_margin));
}

TEST(CropRule, CropLinesFlipsALightOnDarkLine) {
  // A dark band with light text, and a dark-on-light line below it: only the first flips.
  Page page(400, 200, 255);
  page.fill(0, 20, 400, 80, 20);
  for (int x = 40; x < 300; x += 12) page.fill(x, 40, x + 6, 60, 235);
  for (int x = 40; x < 300; x += 12) page.fill(x, 130, x + 6, 150, 15);
  const std::vector<LineBox> boxes = {{36, 40, 304, 60}, {36, 130, 304, 150}};
  const int width = 512;
  std::vector<uint8_t> out(2 * 32 * width);
  std::vector<int32_t> widths(2);
  carbon::crop_lines(page.view(), boxes, width, out.data(), widths);
  const uint8_t* a = out.data();
  const uint8_t* b = out.data() + 32 * width;
  EXPECT_GT(carbon::crop_ground(a, widths[0], 32, width), 200);  // the dark band read as paper
  EXPECT_GT(carbon::crop_ground(b, widths[1], 32, width), 200);
  EXPECT_FALSE(carbon::light_on_dark(a, widths[0], 32, width, 48));
  CropRule keep;
  keep.flip_margin = 0;
  carbon::crop_lines(page.view(), boxes, width, out.data(), widths, keep);
  EXPECT_LT(carbon::crop_ground(a, widths[0], 32, width), 60);  // without the flip, still dark
}

TEST(Ctc, GreedyCollapsesRepeatsAndDropsBlanks) {
  const int classes = 4, frames = 6;
  const std::vector<int> best = {0, 2, 2, 0, 2, 3};  // -> 2 2 3, starting at frames 1 4 5
  std::vector<float> logits(classes * frames, 0.0f);
  for (int t = 0; t < frames; ++t) logits[best[t] * frames + t] = 5.0f;
  const std::vector<int32_t> lengths = {frames};
  const auto lines = carbon::ctc_greedy(logits.data(), 1, classes, frames, lengths);
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(lines[0].ids, (std::vector<int32_t>{2, 2, 3}));
  EXPECT_EQ(lines[0].frames, (std::vector<int32_t>{1, 4, 5}));
  for (float p : lines[0].probs) EXPECT_NEAR(p, std::exp(5.0f) / (std::exp(5.0f) + 3), 1e-6);
  const std::vector<int32_t> short_len = {3};  // only the first three frames: "2"
  EXPECT_EQ(carbon::ctc_greedy(logits.data(), 1, classes, frames, short_len)[0].ids,
            (std::vector<int32_t>{2}));
  std::vector<float> flat(classes * frames, 1.0f);  // ties go to the lowest class: all blank
  EXPECT_TRUE(carbon::ctc_greedy(flat.data(), 1, classes, frames, lengths)[0].ids.empty());
}

// The layout the export writes: "CTCHEAD1", int32 classes, int32 k, the
// float16 weight, the float32 bias.
TEST(Ctc, HeadFileLoads) {
  const auto path = std::filesystem::temp_directory_path() / "carbon-test.head.bin";
  auto write = [&](const std::string& bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  };
  const int32_t dims[2] = {3, 2};
  const uint16_t w[6] = {0x3c00, 0xbc00, 0x0000, 0x4000, 0x3800, 0x7bff};  // 1 -1 0 2 0.5 65504
  const float b[3] = {0.25f, -1.0f, 3.0f};
  std::string bytes = "CTCHEAD1";
  bytes.append(reinterpret_cast<const char*>(dims), sizeof dims);
  bytes.append(reinterpret_cast<const char*>(w), sizeof w);
  bytes.append(reinterpret_cast<const char*>(b), sizeof b);
  write(bytes);
  const carbon::CtcHead h = carbon::load_ctc_head(path);
  EXPECT_EQ(h.classes, 3);
  EXPECT_EQ(h.k, 2);
  EXPECT_EQ(h.weight, (std::vector<uint16_t>(w, w + 6)));
  EXPECT_EQ(h.bias, (std::vector<float>(b, b + 3)));

  write(bytes.substr(0, bytes.size() - 1));  // one byte short
  EXPECT_THROW(carbon::load_ctc_head(path), carbon::Error);
  write("CTCHEAD2" + bytes.substr(8));
  EXPECT_THROW(carbon::load_ctc_head(path), carbon::Error);
  std::filesystem::remove(path);
  EXPECT_THROW(carbon::load_ctc_head(path), carbon::Error);
}

}  // namespace
