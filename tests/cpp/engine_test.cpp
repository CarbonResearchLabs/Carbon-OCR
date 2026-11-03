// carbon::Engine end to end on tests/data/fake-set (tools/make_test_models.py):
// a page of three dashed bars, which the fake set reads as "aaa", "aaaaa" and "aa".
#include "carbon/engine.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "carbon/deskew.h"
#include "carbon/error.h"
#include "carbon/image.h"
#include "carbon/textlayer.h"

namespace carbon {
namespace {

const std::filesystem::path kSet = std::filesystem::path(CARBON_TEST_DATA) / "fake-set";

TEST(Engine, ReadsAPageEndToEnd) {
  const GrayImage page = load_image(kSet / "page.png");
  for (int workers : {1, 2, 4}) {
    EngineOptions o;
    o.pool.workers = workers;
    Engine engine(kSet, o);
    EXPECT_EQ(engine.workers(), workers);
    const OcrPage p = engine.ocr(page.view());
    EXPECT_EQ(p.width, 800);
    EXPECT_EQ(p.height, 1024);
    EXPECT_EQ(p.rotation, 0);
    EXPECT_EQ(p.skew, 0.0);
    ASSERT_EQ(p.lines.size(), 3u) << workers << " workers";
    const char* want[] = {"aaa", "aaaaa", "aa"};
    for (int i = 0; i < 3; ++i) {
      const OcrLine& l = p.lines[i];
      EXPECT_EQ(l.text, want[i]) << "line " << i;
      EXPECT_NEAR(l.box.y0, 200 + 120 * i, 2.0);  // the bar's own rows
      EXPECT_NEAR(l.box.y1, 224 + 120 * i, 2.0);
      EXPECT_GT(l.confidence, 0.9f);
      EXPECT_GT(l.det_score, 0.9f);
    }
    EXPECT_EQ(p.text(), "aaa\naaaaa\naa\n");
    EXPECT_GT(p.ms.total, 0.0);
    EXPECT_GE(p.ms.total, p.ms.detect);
  }
}

// A page of tiny text for the fake set, at the detector's own scale (800 x 1024):
// six bars 5 px high of 3, 4, 2, 3, 4 and 2 dashes, 24 px long and 8 apart, the
// first at an even row, the next at an odd one, and so on.  The fake detector's
// 5 x 5 window at stride 2 fits in such a bar only where it starts on an even row,
// so its first pass finds every other bar, 4 px high; at twice the scale its
// window fits in every bar.
GrayImage tiny_page() {
  GrayImage p{800, 1024, std::vector<uint8_t>(800 * 1024, 255)};
  const int dashes[] = {3, 4, 2, 3, 4, 2};
  for (int i = 0; i < 6; ++i) {
    const int y = 200 + 41 * i;
    for (int d = 0; d < dashes[i]; ++d) {
      for (int r = y; r < y + 5; ++r) std::fill_n(p.pixels.begin() + r * 800 + 100 + 32 * d, 24, uint8_t{20});
    }
  }
  return p;
}

// The page of tiny text takes the second pass (its median line 4 px, under 9),
// whose page at twice the scale (1600 x 2048) runs in 4 strips: every bar is
// found, once, its box the second pass's.  A page of print takes none.
TEST(Engine, DetectsAPageOfTinyTextAgainAtTwiceTheScale) {
  const GrayImage tiny = tiny_page();
  EngineOptions o;
  o.pool.workers = 2;
  o.crop.flip_margin = 0;  // a tiny bar's box is all bar: its crop's ground is its ink, not paper
  Engine engine(kSet, o);
  const OcrPage p = engine.ocr(tiny.view());
  EXPECT_EQ(p.skew, 0.0);
  EXPECT_TRUE(p.second_pass);
  EXPECT_GT(p.ms.second_pass, 0.0);
  EXPECT_GE(p.ms.total, p.ms.detect + p.ms.second_pass);
  ASSERT_EQ(p.lines.size(), 6u);
  const char* want[] = {"aaa", "aaaa", "aa", "aaa", "aaaa", "aa"};
  for (int i = 0; i < 6; ++i) {
    const OcrLine& l = p.lines[i];
    EXPECT_EQ(l.text, want[i]) << "line " << i;
    EXPECT_NEAR(l.box.y0, 200 + 41 * i + 0.5, 0.25) << "line " << i;  // the second pass's, at either row
    EXPECT_NEAR(l.box.y1, 200 + 41 * i + 4.5, 0.25) << "line " << i;
    EXPECT_NEAR(l.box.x0, 96, 2.0) << "line " << i;
  }
  EXPECT_NE(p.json().find("\"second_pass\": true"), std::string::npos);

  // Without the second pass, or with no larger scale to take it at, the first's
  // lines alone: the bars that start on even rows.
  for (int k = 0; k < 2; ++k) {
    EngineOptions first = o;
    if (k == 0) first.det.second_pass_below = 0;
    if (k == 1) first.second_pass_long_side = first.det_long_side;
    const OcrPage q = Engine(kSet, first).ocr(tiny.view());
    EXPECT_FALSE(q.second_pass);
    EXPECT_EQ(q.ms.second_pass, 0.0);
    ASSERT_EQ(q.lines.size(), 3u);
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(q.lines[i].box.y0, 200 + 82 * i + 1, 0.25) << "line " << i;
  }

  const OcrPage print = engine.ocr(load_image(kSet / "page.png").view());
  EXPECT_FALSE(print.second_pass);
  EXPECT_EQ(print.ms.second_pass, 0.0);
  EXPECT_EQ(print.text(), "aaa\naaaaa\naa\n");
  EXPECT_NE(print.json().find("\"second_pass\": false"), std::string::npos);
}

// The page turned 2.5 degrees: levelled before detection, read as it was.
TEST(Engine, LevelsASkewedPage) {
  const GrayImage level = load_image(kSet / "page.png");
  const GrayImage page = rotate_page(level.view(), 2.5);
  EngineOptions o;
  o.pool.workers = 1;
  Engine engine(kSet, o);
  const OcrPage p = engine.ocr(page.view());
  EXPECT_NEAR(p.skew, -2.5, 0.15);
  ASSERT_EQ(p.lines.size(), 3u);
  const char* want[] = {"aaa", "aaaaa", "aa"};
  for (int i = 0; i < 3; ++i) EXPECT_EQ(p.lines[i].text, want[i]) << "line " << i;
  o.deskew = false;  // without it, the lines' boxes hold the slant
  const OcrPage slanted = Engine(kSet, o).ocr(page.view());
  EXPECT_EQ(slanted.skew, 0.0);
}

TEST(Engine, ManyPagesComeBackInOrder) {
  const GrayImage page = load_image(kSet / "page.png");
  GrayImage blank{400, 300, std::vector<uint8_t>(400 * 300, 255)};
  EngineOptions o;
  o.pool.workers = 3;
  Engine engine(kSet, o);
  std::vector<GrayView> pages;
  for (int i = 0; i < 9; ++i) pages.push_back(i % 3 == 1 ? blank.view() : page.view());
  const auto out = engine.ocr(pages);
  ASSERT_EQ(out.size(), 9u);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(out[i].lines.size(), i % 3 == 1 ? 0u : 3u) << "page " << i;
    EXPECT_EQ(out[i].width, i % 3 == 1 ? 400 : 800);
  }
  const std::string json = out[0].json();
  EXPECT_NE(json.find("\"text\": \"aaaaa\""), std::string::npos);
  EXPECT_NE(json.find("\"width\": 800"), std::string::npos);
}

// A hybrid page: the fake page's middle bar inside a picture of a page otherwise
// read from its text layer, found on the page with only its pictures, at the bar's
// own place, and put in reading order with the layer's line.
TEST(Engine, ReadsAPagesPictures) {
  const GrayImage page = load_image(kSet / "page.png");
  OcrPage p;
  p.width = page.width;
  p.height = page.height;
  p.source = "text layer";
  OcrLine own;
  own.box = {100, 700, 500, 730};
  own.text = "Hello PDF";
  own.confidence = 1;
  p.lines = {own};
  const std::vector<LineBox> pictures = {{80, 300, 520, 370}};
  Engine engine(kSet);
  const GrayImage only = pictures_only(page.view(), pictures_to_read(pictures, page.width, page.height));
  const OcrPage read = engine.ocr(only.view());
  add_picture_lines(p, read);
  EXPECT_EQ(p.source, "hybrid");
  ASSERT_EQ(p.lines.size(), 2u);
  EXPECT_EQ(p.lines[0].text, "aaaaa");
  EXPECT_NEAR(p.lines[0].box.y0, 320, 2.0);
  EXPECT_NEAR(p.lines[0].box.y1, 344, 2.0);
  EXPECT_EQ(p.lines[1].text, "Hello PDF");
}

// The order model's reading order, not the XY-cut's: tests/data/fake-order.onnx
// reads lines bottom to top, each a block of its own, a Section-Header.  Unsure, the
// XY-cut's order, the blocks still the model's.
TEST(Engine, PutsLinesInTheOrderModelsOrder) {
  const GrayImage page = load_image(kSet / "page.png");
  EngineOptions o;
  o.order_model = std::filesystem::path(CARBON_TEST_DATA) / "fake-order.onnx";
  o.order.min_sure = 0;  // the fake model's links are not sure ones
  Engine engine(kSet, o);
  ASSERT_TRUE(engine.has_order_model());
  const OcrPage p = engine.ocr(page.view());
  ASSERT_EQ(p.lines.size(), 3u);
  EXPECT_EQ(p.order, "model");
  EXPECT_EQ(p.lines[0].text, "aa");
  EXPECT_EQ(p.lines[1].text, "aaaaa");
  EXPECT_EQ(p.lines[2].text, "aaa");
  ASSERT_EQ(p.blocks.size(), 3u);
  for (int b = 0; b < 3; ++b) {
    EXPECT_EQ(p.blocks[b].label, "Section-Header");
    EXPECT_EQ(p.blocks[b].lines, std::vector<int>{b});
  }
  EXPECT_NE(p.json().find("\"order\": \"model\""), std::string::npos);
  EXPECT_NE(p.json().find("\"blocks\": [{\"label\": \"Section-Header\""), std::string::npos);
  // a page read otherwise (from a text layer), in the same order
  std::vector<OcrPage> pages = {p};
  std::reverse(pages[0].lines.begin(), pages[0].lines.end());
  engine.order(pages);
  EXPECT_EQ(pages[0].lines[0].text, "aa");
  EXPECT_EQ(pages[0].blocks.size(), 3u);

  EngineOptions unsure = o;
  unsure.order = OrderRule{};
  Engine e(kSet, unsure);
  const OcrPage q = e.ocr(page.view());
  EXPECT_EQ(q.order, "xy-cut");
  ASSERT_EQ(q.lines.size(), 3u);
  EXPECT_EQ(q.lines[0].text, "aaa");
  EXPECT_EQ(q.blocks.size(), 3u);

  EngineOptions none;
  none.use_order = false;
  EXPECT_FALSE(Engine(kSet, none).has_order_model());
  EXPECT_FALSE(Engine(kSet).has_order_model()) << "the fake set has no order.onnx";
}

// Appends metadata_props entries (ModelProto field 14) to an ONNX file, as the export's
// onnx.helper.set_model_props writes them: a field may come anywhere in a message.
void add_metadata(const std::filesystem::path& file, const std::vector<std::pair<std::string, std::string>>& props) {
  auto varint = [](std::string& s, uint64_t v) {
    for (; v >= 0x80; v >>= 7) s += static_cast<char>((v & 0x7F) | 0x80);
    s += static_cast<char>(v);
  };
  std::string out;
  for (const auto& [key, value] : props) {
    std::string entry = "\x0A";
    varint(entry, key.size());
    entry += key + "\x12";
    varint(entry, value.size());
    entry += value;
    out += "\x72";
    varint(out, entry.size());
    out += entry;
  }
  std::ofstream(file, std::ios::binary | std::ios::app) << out;
}

// A recognizer that carries its crop rule (its height and padding) is
// read by it; one without reads by the options'.
TEST(Engine, ReadsLinesByTheRecognizersOwnCropRule) {
  const std::filesystem::path set = std::filesystem::temp_directory_path() / "carbon-engine-crop-set";
  auto fresh = [&] {
    std::filesystem::remove_all(set);
    std::filesystem::copy(kSet, set, std::filesystem::copy_options::recursive);
  };
  fresh();
  EXPECT_EQ(Engine(set).crop().pad_h, CropRule{}.pad_h);
  add_metadata(set / "rec-latin.onnx",
               {{"carbon.crop.height", "32"}, {"carbon.crop.pad_v", "0.25"}, {"carbon.crop.pad_h", "0.4"}});
  {
    Engine engine(set);
    EXPECT_EQ(engine.crop().height, 32);
    EXPECT_EQ(engine.crop().pad_v, 0.25);
    EXPECT_EQ(engine.crop().pad_h, 0.4);
    EXPECT_EQ(engine.ocr(load_image(set / "page.png").view()).text(), "aaa\naaaaa\naa\n");
  }
  fresh();
  add_metadata(set / "rec-latin.onnx", {{"carbon.crop.pad_h", "lots"}});
  EXPECT_THROW(Engine{set}, Error);
  std::filesystem::remove_all(set);
}

// An orientation model that carries its turn threshold (calibrated on
// held-out pages at export) sets it; options that give one keep theirs (carbon-ocr --as-given's 2).
TEST(Engine, TurnsPagesByTheOrientationModelsOwnThreshold) {
  const std::filesystem::path set = std::filesystem::temp_directory_path() / "carbon-engine-turn-set";
  auto fresh = [&] {
    std::filesystem::remove_all(set);
    std::filesystem::copy(kSet, set, std::filesystem::copy_options::recursive);
  };
  fresh();
  EXPECT_EQ(Engine(set).turn_probability(), 0.95f);
  add_metadata(set / "orient.onnx", {{"carbon.orient.turn_probability", "0.62"}});
  EXPECT_EQ(Engine(set).turn_probability(), 0.62f);
  EngineOptions given;
  given.turn_probability = 2.0f;
  EXPECT_EQ(Engine(set, given).turn_probability(), 2.0f);
  fresh();
  add_metadata(set / "orient.onnx", {{"carbon.orient.turn_probability", "0.2"}});  // under even odds
  EXPECT_THROW(Engine{set}, Error);
  std::filesystem::remove_all(set);
}

TEST(Engine, ASetMustBeWhole) {
  EXPECT_THROW(Engine(std::filesystem::path(CARBON_TEST_DATA) / "no-such-set"), Error);
  EngineOptions o;
  o.rec = "rec-cjk";  // not in the fake set
  EXPECT_THROW(Engine(kSet, o), Error);
}

}  // namespace
}  // namespace carbon
