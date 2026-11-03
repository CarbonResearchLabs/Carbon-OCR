#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "carbon/engine.h"
#include "carbon/error.h"
#include "carbon/image.h"

namespace carbon {
namespace {

const std::filesystem::path kData = CARBON_TEST_DATA;

// The recognizers' training buckets, so the engine batches
// lines at the widths the recognizer learned on.
TEST(Pipeline, WidthBucketsAreTheTrainers) {
  const std::vector<int> buckets = {32, 48, 64, 88, 120, 160, 208, 272, 360, 472, 616, 808, 1056, 1376, 1792, 2048};
  int prev = 0;
  for (int b : buckets) {
    EXPECT_EQ(width_bucket(b), b);
    EXPECT_EQ(width_bucket(prev + 1), b) << "a crop " << prev + 1 << " px wide";
    prev = b;
  }
  EXPECT_EQ(width_bucket(8), 32);
  EXPECT_EQ(width_bucket(2049), 2056);  // past the last: its own width, rounded up to 8
}

// A title, then two columns: the title, the left column down, the right column down.
TEST(Pipeline, ReadingOrderTakesColumnsWhole) {
  const std::vector<LineBox> boxes = {
      {420, 100, 760, 120},  // 0 right column, first line
      {40, 40, 760, 70},     // 1 the title, across both
      {40, 130, 380, 150},   // 2 left, second
      {40, 100, 380, 120},   // 3 left, first
      {420, 130, 700, 150},  // 4 right, second
  };
  EXPECT_EQ(reading_order(boxes), (std::vector<size_t>{1, 3, 2, 0, 4}));
  EXPECT_TRUE(reading_order({}).empty());
}

TEST(Pipeline, VocabularyReadsJsonStrings) {
  const auto file = std::filesystem::temp_directory_path() / "carbon_vocab_test.json";
  {
    std::ofstream f(file, std::ios::binary);
    f << R"({"model": "x", "blank": 0, "chars": ["", "a", "\"", "\\", "é", "😀", "中", " "]})";
  }
  const auto v = load_vocab(file);
  ASSERT_EQ(v.size(), 8u);
  EXPECT_EQ(v[0], "");
  EXPECT_EQ(v[2], "\"");
  EXPECT_EQ(v[3], "\\");
  EXPECT_EQ(v[4], "\xc3\xa9");          // é
  EXPECT_EQ(v[5], "\xf0\x9f\x98\x80");  // a surrogate pair
  EXPECT_EQ(v[6], "\xe4\xb8\xad");
  EXPECT_EQ(v[7], " ");
  std::filesystem::remove(file);
  EXPECT_THROW(load_vocab(kData / "no-such.json"), Error);
  EXPECT_EQ(load_vocab(kData / "fake-set" / "rec-latin.vocab.json").size(), 3u);
}

TEST(Image, QuarterTurnsAndInputs) {
  // 3 x 2:  1 2 3 / 4 5 6
  GrayImage img{3, 2, {1, 2, 3, 4, 5, 6}};
  const GrayImage cw = rotate_quarter(img.view(), 1);  // 2 x 3:  4 1 / 5 2 / 6 3
  EXPECT_EQ(cw.width, 2);
  EXPECT_EQ(cw.pixels, (std::vector<uint8_t>{4, 1, 5, 2, 6, 3}));
  EXPECT_EQ(rotate_quarter(img.view(), 2).pixels, (std::vector<uint8_t>{6, 5, 4, 3, 2, 1}));
  EXPECT_EQ(rotate_quarter(cw.view(), 3).pixels, img.pixels);
  EXPECT_EQ(rotate_quarter(img.view(), 4).pixels, img.pixels);

  GrayImage white{4, 4, std::vector<uint8_t>(16, 255)};
  white.pixels[0] = 0;
  std::vector<float> in(8 * 8, -1.0f);
  ink_input(white.view(), 4, 4, 8, 8, in.data());
  EXPECT_FLOAT_EQ(in[0], 1.0f);    // ink
  EXPECT_FLOAT_EQ(in[1], 0.0f);    // paper
  EXPECT_FLOAT_EQ(in[8 * 8 - 1], 0.0f);  // the padding is paper
  EXPECT_THROW(ink_input(white.view(), 9, 4, 8, 8, in.data()), Error);
}

TEST(Image, DecodesFiles) {
  const std::string pgm = std::string("P5\n2 2\n255\n") + std::string("\x00\x40\x80\xff", 4);
  const GrayImage g = decode_image({reinterpret_cast<const uint8_t*>(pgm.data()), pgm.size()});
  EXPECT_EQ(g.width, 2);
  EXPECT_EQ(g.pixels, (std::vector<uint8_t>{0, 0x40, 0x80, 0xff}));
  const GrayImage page = load_image(kData / "fake-set" / "page.png");
  EXPECT_EQ(page.width, 800);
  EXPECT_EQ(page.height, 1024);
  EXPECT_EQ(page.pixels[200 * 800 + 100], 20);  // the first dash
  EXPECT_THROW(load_image(kData / "no-such.png"), Error);
  const std::string junk = "not an image";
  EXPECT_THROW(decode_image({reinterpret_cast<const uint8_t*>(junk.data()), junk.size()}), Error);
}

}  // namespace
}  // namespace carbon
