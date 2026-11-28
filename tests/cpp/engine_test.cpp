// carbon::Engine end to end on tests/data/fake-set (tools/make_test_models.py):
// a page of three dashed bars, which the fake set reads as "aaa", "aaaaa" and "aa".
#include "carbon/engine.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "carbon/beam.h"
#include "carbon/deskew.h"
#include "carbon/error.h"
#include "carbon/image.h"
#include "carbon/output.h"
#include "carbon/textlayer.h"
#include "carbon/tlm.h"
#include "../../carbon/src/engine_steps.h"
#include "../../carbon/src/escape_scan.h"

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

// The page of tiny text takes the second pass (its median line 4 px, under 9), at
// twice the scale on the band of rows its tiny bars hold: every bar is found,
// once, its box the second pass's.  A page of print takes none.
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

// A page of print with a share of tiny text (the six tiny bars, then eight bars 16 px high): its median line is
// not tiny, so the pass by share takes it on the tiny lines' bands alone, and finds what the whole page's pass finds,
// every bar once.
TEST(Engine, ThePassByShareRunsOnTheTinyLinesBands) {
  GrayImage page = tiny_page();
  for (int i = 0; i < 8; ++i) {
    const int y = 560 + 40 * i;
    for (int d = 0; d < 3; ++d) {
      for (int r = y; r < y + 16; ++r) std::fill_n(page.pixels.begin() + r * 800 + 100 + 32 * d, 24, uint8_t{20});
    }
  }
  EngineOptions o;
  o.pool.workers = 2;
  o.crop.flip_margin = 0;
  o.det.second_pass_share = 0.15;
  o.det.second_pass_lines = 3;
  EngineOptions whole = o;
  whole.det.second_pass_bands = false;
  const OcrPage a = Engine(kSet, o).ocr(page.view()), b = Engine(kSet, whole).ocr(page.view());
  EXPECT_TRUE(a.second_pass);
  EXPECT_TRUE(b.second_pass);
  ASSERT_EQ(a.lines.size(), 14u);
  ASSERT_EQ(b.lines.size(), a.lines.size());
  for (size_t i = 0; i < a.lines.size(); ++i) {
    EXPECT_EQ(a.lines[i].text, b.lines[i].text) << "line " << i;
    EXPECT_NEAR(a.lines[i].box.y0, b.lines[i].box.y0, 0.25) << "line " << i;
    EXPECT_NEAR(a.lines[i].box.y1, b.lines[i].box.y1, 0.25) << "line " << i;
    EXPECT_NEAR(a.lines[i].box.x0, b.lines[i].box.x0, 0.25) << "line " << i;
  }
}

// A page all of tiny print in two blocks far apart (the six bars, and six more 600 rows below): its median line
// is tiny, so it takes the pass by the median, on two bands, and finds what the whole page's pass (in 4 strips of
// the page at twice the scale) finds, every bar once.
TEST(Engine, ThePassByTheMedianRunsOnTheTinyLinesBands) {
  GrayImage page = tiny_page();
  const int dashes[] = {2, 3, 4, 2, 3, 4};
  for (int i = 0; i < 6; ++i) {
    const int y = 800 + 31 * i;
    for (int d = 0; d < dashes[i]; ++d) {
      for (int r = y; r < y + 5; ++r) std::fill_n(page.pixels.begin() + r * 800 + 300 + 32 * d, 24, uint8_t{20});
    }
  }
  EngineOptions o;
  o.pool.workers = 2;
  o.crop.flip_margin = 0;
  EngineOptions whole = o;
  whole.det.second_pass_bands = false;
  const OcrPage a = Engine(kSet, o).ocr(page.view()), b = Engine(kSet, whole).ocr(page.view());
  EXPECT_TRUE(a.second_pass);
  EXPECT_TRUE(b.second_pass);
  ASSERT_EQ(a.lines.size(), 12u);
  ASSERT_EQ(b.lines.size(), a.lines.size());
  for (size_t i = 0; i < a.lines.size(); ++i) {
    EXPECT_EQ(a.lines[i].text, b.lines[i].text) << "line " << i;
    EXPECT_NEAR(a.lines[i].box.y0, b.lines[i].box.y0, 0.25) << "line " << i;
    EXPECT_NEAR(a.lines[i].box.y1, b.lines[i].box.y1, 0.25) << "line " << i;
    EXPECT_NEAR(a.lines[i].box.x0, b.lines[i].box.x0, 0.25) << "line " << i;
  }
}

// The line pitch of pages of lines 30, 17 and 50 rows apart, with ragged lengths; none for a blank page or
// one of noise; and a page of large text detected at a smaller scale reads as it does at the full one.
TEST(Engine, ALinesPitchFromItsRowsInk) {
  std::mt19937 rng(8);
  for (int pitch : {30, 17, 50}) {
    GrayImage p{600, 800, std::vector<uint8_t>(600 * 800, 255)};
    for (int y = 20; y + 10 < 800; y += pitch) {
      const int len = 200 + static_cast<int>(rng() % 300), hgt = std::max(4, pitch / 3);
      for (int r = y; r < y + hgt; ++r) std::fill_n(p.pixels.begin() + r * 600 + 40, len, uint8_t{30});
    }
    EXPECT_EQ(detail::line_pitch(p.view()), pitch) << pitch;
  }
  GrayImage blank{600, 800, std::vector<uint8_t>(600 * 800, 255)};
  EXPECT_EQ(detail::line_pitch(blank.view()), 0);
  GrayImage noise{600, 800, std::vector<uint8_t>(600 * 800)};
  for (uint8_t& v : noise.pixels) v = static_cast<uint8_t>(rng());
  EXPECT_EQ(detail::line_pitch(noise.view()), 0);
}

// The wide detector (det.onnx in FP32 on ONNX Runtime's threads, for a page that finds fast cores idle) only where the
// workers' detector is det.onnx too: at auto a set that ships det.int8.onnx runs that on every worker, and a page read
// alone would otherwise meet another detector than in a batch, and read differently.
TEST(Engine, AWideDetectorOnlyWhereTheWorkersDetectorIsDetOnnx) {
  const std::filesystem::path set = "set";
  EXPECT_TRUE(detail::wide_detector_allowed(Precision::kAuto, set, set / "det.onnx"));
  EXPECT_TRUE(detail::wide_detector_allowed(Precision::kFp32, set, set / "det.onnx"));
  EXPECT_FALSE(detail::wide_detector_allowed(Precision::kAuto, set, set / "det.int8.onnx"));
  EXPECT_FALSE(detail::wide_detector_allowed(Precision::kInt8, set, set / "det.int8.onnx"));
  EXPECT_FALSE(detail::wide_detector_allowed(Precision::kFp16, set, set / "det.fp16.onnx"));
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
  // On 4 workers, the page alone: its skew searched beside orientation, its rows levelled in bands; the same page.
  EngineOptions four = o;
  four.pool.workers = 4;
  const OcrPage q = Engine(kSet, four).ocr(page.view());
  EXPECT_EQ(q.skew, p.skew);
  ASSERT_EQ(q.lines.size(), p.lines.size());
  for (size_t i = 0; i < p.lines.size(); ++i) {
    EXPECT_EQ(q.lines[i].text, p.lines[i].text);
    EXPECT_EQ(q.lines[i].box.x0, p.lines[i].box.x0);
    EXPECT_EQ(q.lines[i].box.y0, p.lines[i].box.y0);
    EXPECT_EQ(q.lines[i].confidence, p.lines[i].confidence);
  }
  o.deskew = false;  // without it, the lines' boxes hold the slant
  const OcrPage slanted = Engine(kSet, o).ocr(page.view());
  EXPECT_EQ(slanted.skew, 0.0);
}

// A page turned 15 degrees, past the narrow search: levelled (max_skew 45, the default), and read as it was;
// with max_skew 5 it is not levelled by 15. A page within 5 degrees finds the same angle either way.
TEST(Engine, LevelsAPageSkewedPastFiveDegrees) {
  const GrayImage level = load_image(kSet / "page.png");
  const GrayImage page = rotate_page(level.view(), 15);
  EngineOptions o;
  o.pool.workers = 1;
  EXPECT_EQ(o.max_skew, 45.0);
  const OcrPage p = Engine(kSet, o).ocr(page.view());
  EXPECT_NEAR(p.skew, -15, 0.15);
  ASSERT_EQ(p.lines.size(), 3u);
  const char* want[] = {"aaa", "aaaaa", "aa"};
  for (int i = 0; i < 3; ++i) EXPECT_EQ(p.lines[i].text, want[i]) << "line " << i;
  EngineOptions narrow;
  narrow.pool.workers = 1;
  narrow.max_skew = 5;
  EXPECT_LE(std::abs(Engine(kSet, narrow).ocr(page.view()).skew), 5.0);
  const GrayImage small = rotate_page(level.view(), 2.5);
  EXPECT_EQ(find_skew(small.view(), 45).degrees, find_skew(small.view(), 5).degrees);
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

// The stream: pages given one at a time, more than in_flight of them, come back in
// their order with the same lines, boxes and confidences as ocr() gives, whether
// the engine holds a view of them (and what keeps it) or the pages themselves.
TEST(Engine, StreamsPagesInOrderAsOcrReadsThem) {
  const GrayImage page = load_image(kSet / "page.png");
  GrayImage blank{400, 300, std::vector<uint8_t>(400 * 300, 255)};
  EngineOptions o;
  o.pool.workers = 3;
  o.in_flight = 2;
  Engine engine(kSet, o);
  const OcrPage want = engine.ocr(page.view());
  EXPECT_FALSE(engine.next().has_value());  // nothing given
  auto held = std::make_shared<GrayImage>(page);
  std::vector<size_t> numbers;
  for (int i = 0; i < 9; ++i) {
    if (i % 3 == 1) numbers.push_back(engine.submit(blank));
    else if (i % 3 == 0) numbers.push_back(engine.submit(held->view(), held));
    else numbers.push_back(engine.submit(GrayImage(page)));
    if (i == 4) {  // taking one while others are still being read
      const auto first = engine.next();
      ASSERT_TRUE(first.has_value());
      EXPECT_EQ(first->lines.size(), 3u);
    }
  }
  for (int i = 0; i < 9; ++i) EXPECT_EQ(numbers[i], static_cast<size_t>(i));
  held.reset();  // the engine's own reference keeps the view's pixels
  for (int i = 1; i < 9; ++i) {
    const auto p = engine.next();
    ASSERT_TRUE(p.has_value()) << "page " << i;
    if (i % 3 == 1) {
      EXPECT_TRUE(p->lines.empty());
      EXPECT_EQ(p->width, 400);
      continue;
    }
    ASSERT_EQ(p->lines.size(), want.lines.size()) << "page " << i;
    for (size_t k = 0; k < want.lines.size(); ++k) {
      EXPECT_EQ(p->lines[k].text, want.lines[k].text);
      EXPECT_EQ(p->lines[k].box.x0, want.lines[k].box.x0);
      EXPECT_EQ(p->lines[k].box.y1, want.lines[k].box.y1);
      EXPECT_EQ(p->lines[k].confidence, want.lines[k].confidence);
    }
  }
  EXPECT_FALSE(engine.next().has_value());  // every page out
  EXPECT_THROW(engine.submit(GrayImage{}), Error);

  // Files, decoded by the worker that takes them; one that cannot be read fails its own page alone.
  engine.submit(kSet / "page.png");
  engine.submit(kSet / "no-such-page.png");
  engine.submit(kSet / "page.png");
  const auto a = engine.next();
  ASSERT_TRUE(a.has_value());
  EXPECT_EQ(a->lines.size(), 3u);
  EXPECT_EQ(a->lines[1].text, want.lines[1].text);
  EXPECT_GT(a->ms.load, 0.0);
  EXPECT_THROW(engine.next(), Error);
  const auto c = engine.next();
  ASSERT_TRUE(c.has_value());
  EXPECT_EQ(c->lines.size(), 3u);
  EXPECT_FALSE(engine.next().has_value());
  EXPECT_EQ(engine.ocr(page.view()).lines.size(), 3u);  // ocr() still reads after a stream
}

// A page's task (PageTask): read as given, it is what an engine whose options never turn or level a page reads
// (orientation is not run: it could not turn it), whatever the engine's own options; read from lines found already
// (LineFinder::detect), detection is skipped and those lines are read, as the engine reads the lines it finds.
TEST(Engine, ReadsAPageAsGivenOrFromTheLinesFoundOnIt) {
  const GrayImage level = load_image(kSet / "page.png");
  const GrayImage page = rotate_page(level.view(), 2.5);  // deskew levels it, unless read as given
  EngineOptions o;
  o.pool.workers = 2;
  Engine engine(kSet, o);
  EngineOptions given = o;
  given.turn_probability = 2.0f;
  given.deskew = false;
  const OcrPage want = Engine(kSet, given).ocr(page.view());
  EXPECT_EQ(want.skew, 0.0);
  ASSERT_GT(std::abs(engine.ocr(page.view()).skew), 1.0) << "levelled when read by the engine's own options";
  PageTask as_given;
  as_given.as_given = true;
  engine.submit(GrayImage(page), as_given);
  const OcrPage got = *engine.next();
  EXPECT_EQ(got.rotation, 0);
  EXPECT_EQ(got.skew, 0.0);
  ASSERT_EQ(got.lines.size(), want.lines.size());
  for (size_t i = 0; i < want.lines.size(); ++i) {
    EXPECT_EQ(got.lines[i].text, want.lines[i].text);
    EXPECT_EQ(got.lines[i].box.x0, want.lines[i].box.x0);
    EXPECT_EQ(got.lines[i].box.y1, want.lines[i].box.y1);
    EXPECT_EQ(got.lines[i].confidence, want.lines[i].confidence);
  }

  // From the lines found: the level page's three bars, or the first two of them alone.
  LineFinder finder(kSet, o);
  const OcrPage found = finder.detect(level.view());
  ASSERT_EQ(found.lines.size(), 3u);
  EXPECT_EQ(found.width, level.width);
  EXPECT_TRUE(found.lines[0].text.empty());
  const OcrPage read = engine.ocr(level.view());
  PageTask from;
  from.found = std::make_shared<const OcrPage>(found);
  engine.submit(GrayImage(level), from);
  OcrPage two = found;
  two.lines.pop_back();
  from.found = std::make_shared<const OcrPage>(two);
  engine.submit(GrayImage(level), from);
  OcrPage small = found;
  small.width = 10;
  from.found = std::make_shared<const OcrPage>(small);
  engine.submit(GrayImage(level), from);
  const OcrPage all = *engine.next();
  ASSERT_EQ(all.lines.size(), 3u);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(all.lines[i].text, read.lines[i].text);
    EXPECT_EQ(all.lines[i].det_score, found.lines[i].det_score);
    EXPECT_NEAR(all.lines[i].box.y0, read.lines[i].box.y0, 1.0);
  }
  const OcrPage first_two = *engine.next();
  ASSERT_EQ(first_two.lines.size(), 2u);
  EXPECT_EQ(first_two.lines[1].text, read.lines[1].text);
  EXPECT_THROW(engine.next(), Error) << "lines found on a page of another size";
  EXPECT_FALSE(engine.next().has_value());
}

// The stream taken as pages finish (a server's): images as their encoded bytes, decoded by the workers, each
// result with its number, the same lines as ocr() reads; bytes that are no image, and a file not there, their
// own pages' errors, marked unreadable, the others read; next() refused once next_finished has taken a page.
TEST(Engine, StreamsEncodedPagesAsTheyFinish) {
  const GrayImage page = load_image(kSet / "page.png");
  std::ifstream f(kSet / "page.png", std::ios::binary);
  const auto png = std::make_shared<const std::string>(std::istreambuf_iterator<char>(f),
                                                       std::istreambuf_iterator<char>());
  auto bytes = [](const std::shared_ptr<const std::string>& b) {
    return std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(b->data()), b->size());
  };
  EngineOptions o;
  o.pool.workers = 3;
  o.in_flight = 2;
  Engine engine(kSet, o);
  const OcrPage want = engine.ocr(page.view());
  EXPECT_FALSE(engine.next_finished().has_value());  // nothing given
  const auto junk = std::make_shared<const std::string>("not an image at all");
  std::vector<size_t> numbers;
  for (int i = 0; i < 7; ++i) {
    if (i == 2) numbers.push_back(engine.submit_encoded(bytes(junk), junk));
    else if (i == 5) numbers.push_back(engine.submit(kSet / "no-such-page.png"));
    else numbers.push_back(engine.submit_encoded(bytes(png), png));
  }
  for (int i = 0; i < 7; ++i) EXPECT_EQ(numbers[i], static_cast<size_t>(i));
  std::vector<int> seen(7, 0);
  for (int i = 0; i < 7; ++i) {
    std::optional<Engine::Finished> r = engine.next_finished();
    ASSERT_TRUE(r.has_value());
    ASSERT_LT(r->number, 7u);
    ++seen[r->number];
    if (r->number == 2 || r->number == 5) {
      EXPECT_TRUE(r->error) << r->number;
      EXPECT_TRUE(r->unreadable) << r->number;
      continue;
    }
    EXPECT_FALSE(r->error);
    EXPECT_FALSE(r->unreadable);
    EXPECT_GT(r->page.ms.load, 0.0);
    ASSERT_EQ(r->page.lines.size(), want.lines.size());
    for (size_t k = 0; k < want.lines.size(); ++k) {
      EXPECT_EQ(r->page.lines[k].text, want.lines[k].text);
      EXPECT_EQ(r->page.lines[k].box.x0, want.lines[k].box.x0);
      EXPECT_EQ(r->page.lines[k].confidence, want.lines[k].confidence);
    }
  }
  for (int v : seen) EXPECT_EQ(v, 1);
  EXPECT_FALSE(engine.next_finished().has_value());  // every page out
  EXPECT_THROW(engine.next(), Error);                  // a stream taken one way only
  EXPECT_THROW(engine.submit_encoded({}, nullptr), Error);
  EXPECT_EQ(engine.ocr(page.view()).lines.size(), 3u);  // ocr() still reads after a stream
}

// A page's lines over idle cores: tasks of about equal cost (a line costs its width bucket), every line once, each
// task's lines by width so a bucket's lines are one batch; with every core busy, tasks of line_batch lines.
TEST(Engine, SplitsAPagesLinesByTheirCost) {
  std::vector<OcrLine> lines;
  for (int i = 0; i < 52; ++i) {  // 12 short lines and 40 long ones, as on a page of body text and headings
    OcrLine l;
    const double w = i % 13 == 0 ? 60 + 7 * i : 700 + (i % 5) * 40;
    l.box = {50.0, 40.0 * i, 50.0 + w, 40.0 * i + 32};
    lines.push_back(l);
  }
  const CropRule crop;
  for (int parts : {2, 3, 6}) {
    const detail::LineTasks plan = detail::line_tasks(lines, crop, parts, 8);
    ASSERT_EQ(plan.starts.size(), static_cast<size_t>(parts + 1));
    std::vector<int> seen(lines.size(), 0);
    std::vector<double> cost;
    for (int t = 0; t < parts; ++t) {
      double c = 0;
      int last = 0;
      for (size_t a = plan.starts[t]; a < plan.starts[t + 1]; ++a) {
        const int b = width_bucket(crop_region(lines[plan.lines[a]].box, crop).width);
        EXPECT_GE(b, last) << "task " << t;  // by width inside the task
        last = b;
        c += b;
        ++seen[plan.lines[a]];
      }
      cost.push_back(c);
    }
    for (int v : seen) EXPECT_EQ(v, 1);
    const double mean = std::accumulate(cost.begin(), cost.end(), 0.0) / parts;
    EXPECT_LT(*std::max_element(cost.begin(), cost.end()), 1.1 * mean) << parts << " parts";
  }
  const detail::LineTasks busy = detail::line_tasks(lines, crop, 0, 8);
  EXPECT_EQ(busy.starts.size(), 8u);  // 52 lines in tasks of 8, and the end
  EXPECT_EQ(busy.starts[1], 8u);
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
// XY-cut's order, the blocks still the model's.  By geometry (the default), the
// model's blocks top to bottom, their labels the model's.
TEST(Engine, PutsLinesInTheOrderModelsOrder) {
  const GrayImage page = load_image(kSet / "page.png");
  EngineOptions o;
  o.order_model = std::filesystem::path(CARBON_TEST_DATA) / "fake-order.onnx";
  o.order.min_sure = 0;  // the fake model's links are not sure ones
  {
    const OcrPage g = Engine(kSet, o).ocr(page.view());
    ASSERT_EQ(g.lines.size(), 3u);
    EXPECT_EQ(g.order, "model");
    EXPECT_EQ(g.lines[0].text, "aaa");  // the top line first
    ASSERT_EQ(g.blocks.size(), 3u);
    for (const OcrBlock& b : g.blocks) EXPECT_EQ(b.label, "Section-Header");
  }
  o.blocks_by_geometry = false;  // the model's own order between its blocks, from here
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
  // The file the engine runs gives its own: FP16's its own threshold, the FP32 file's none of its business
  fresh();
  add_metadata(set / "orient.onnx", {{"carbon.orient.turn_probability", "0.62"}});
  add_metadata(set / "orient.fp16.onnx", {{"carbon.orient.turn_probability", "0.7"}});
  EXPECT_EQ(detail::model_turn_probability_of(set / "orient.fp16.onnx"), 0.7f);  // (its fake is TensorRT's alone)
  EXPECT_EQ(Engine(set).turn_probability(), 0.62f);
  // and the detector's thresholds from its file, where stamped; a bad one named
  add_metadata(set / "det.onnx", {{"carbon.det.core_threshold", "0.4"}, {"carbon.det.min_core_pixels", "3"}});
  const DetRule stamped = detail::model_det_rule(set / "det.onnx", DetRule{});
  EXPECT_FLOAT_EQ(stamped.core_threshold, 0.4f);
  EXPECT_EQ(stamped.min_core_pixels, 3);
  EXPECT_FLOAT_EQ(stamped.picture_threshold, DetRule{}.picture_threshold);
  EXPECT_FLOAT_EQ(detail::model_det_rule(set / "det.fp16.onnx", DetRule{}).core_threshold, DetRule{}.core_threshold);
  add_metadata(set / "det.fp16.onnx", {{"carbon.det.merge_iou", "1.5"}});
  EXPECT_THROW(detail::model_det_rule(set / "det.fp16.onnx", DetRule{}), Error);
  std::filesystem::remove_all(set);
}

TEST(Engine, ASetMustBeWhole) {
  EXPECT_THROW(Engine(std::filesystem::path(CARBON_TEST_DATA) / "no-such-set"), Error);
  EngineOptions o;
  o.rec = "rec-cjk";  // not in the fake set
  EXPECT_THROW(Engine(kSet, o), Error);
}

OcrLine gate_line(LineBox b, const char* text, float conf, float least) {
  OcrLine l;
  l.box = b;
  l.text = text;
  l.confidence = conf;
  l.least = least;
  return l;
}

std::vector<std::string> texts_of(const std::vector<OcrLine>& lines) {
  std::vector<std::string> t;
  for (const OcrLine& l : lines) t.push_back(l.text);
  return t;
}

// The gate on whole lines: nothing read, rules and marks alone, a sliver of two characters at most, dropped with
// their reasons; text, a tall box that read a word, and a rule beside text kept.
TEST(Gate, DropsWholeLinesThatAreNotText) {
  OcrPage p;
  p.width = p.height = 1000;
  p.lines = {gate_line({10, 10, 500, 30}, "A real line of text", 0.99f, 0.9f),
             gate_line({10, 40, 500, 60}, "  ", 0, 0),
             gate_line({10, 70, 500, 90}, "___ | __ — •", 0.8f, 0.5f),
             gate_line({600, 10, 610, 60}, "|1", 0.3f, 0.3f),      // 10 wide, 50 high: a sliver
             gate_line({600, 100, 620, 160}, "Abc", 0.9f, 0.9f),  // tall, but three characters
             gate_line({10, 200, 500, 220}, "Total ____ 12", 0.9f, 0.6f)};
  detail::gate_whole_lines(p, GateRule{});
  EXPECT_EQ(texts_of(p.lines), (std::vector<std::string>{"A real line of text", "Abc", "Total ____ 12"}));
  ASSERT_EQ(p.dropped.size(), 3u);
  EXPECT_EQ(p.dropped[0].why, "empty");
  EXPECT_EQ(p.dropped[1].why, "rules");
  EXPECT_EQ(p.dropped[2].why, "sliver");
  EXPECT_NE(p.json().find("\"dropped\": [{\"box\": [10.0, 40.0, 500.0, 60.0]"), std::string::npos) << p.json();
}

// The gate on pieces: read out of the picture mask unsurely, or a few unsure characters alone on their row,
// dropped; a sure line on the picture, a few sure characters alone, and a few unsure ones beside another kept.
TEST(Gate, DropsPiecesReadOutOfPicturesAndLoneMarks) {
  OcrPage p;
  p.width = p.height = 1000;
  p.picture_mask.height = 20;
  p.picture_mask.width = 64;
  p.picture_mask.words = 2;
  p.picture_mask.px_x = p.picture_mask.px_y = 10;  // a map pixel covers 10 x 10 page pixels
  p.picture_mask.bits.assign(20 * 2, 0);
  for (int r = 0; r < 10; ++r) p.picture_mask.bits[static_cast<size_t>(r) * 2] = ~0u;  // x 0-320, y 0-100
  EXPECT_DOUBLE_EQ(p.picture_mask.share({0, 0, 320, 100}), 1.0);
  EXPECT_DOUBLE_EQ(p.picture_mask.share({160, 0, 480, 100}), 0.5);
  p.lines = {gate_line({20, 20, 300, 40}, "qur 51R9", 0.7f, 0.3f),             // on the picture, unsure
             gate_line({20, 60, 300, 80}, "A caption in the photo", 0.97f, 0.9f),  // on it, sure
             gate_line({20, 300, 40, 320}, "4", 0.4f, 0.4f),                      // alone, unsure
             gate_line({20, 400, 40, 420}, "7", 0.4f, 0.4f),                      // unsure, beside a word
             gate_line({50, 400, 300, 420}, "seven items", 0.95f, 0.9f),
             gate_line({20, 500, 50, 520}, "12", 0.9f, 0.9f)};                    // alone, sure
  detail::gate_pieces(p, GateRule{});
  EXPECT_EQ(texts_of(p.lines),
            (std::vector<std::string>{"A caption in the photo", "7", "seven items", "12"}));
  ASSERT_EQ(p.dropped.size(), 2u);
  EXPECT_EQ(p.dropped[0].why, "picture");
  EXPECT_EQ(p.dropped[1].why, "lone");
}

// A table's column rules, as the demo's page of tables read them (docs/DEMO.md): a cell's line that took in the
// rule beside it ("Homes served|", "6,420 |") loses the bar, its words and box with it; a piece of the rule the
// detector found apart, over the cell's line, read as "d", is dropped; a one-letter cell clear of its neighbours,
// and a bar inside a line, stay.
TEST(Gate, ColumnRulesTakenOffCells) {
  OcrPage p;
  p.width = p.height = 1700;
  OcrLine served = gate_line({403, 385, 577, 422}, "Homes served|", 0.96f, 0.5f);
  served.words = {{{403, 385, 480, 422}, "Homes", 0.99f}, {{490, 385, 577, 422}, "served|", 0.93f}};
  OcrLine count = gate_line({511, 438, 585, 471}, "6,420 |", 0.998f, 0.9f);
  count.words = {{{511, 438, 570, 471}, "6,420", 0.999f}, {{580, 438, 585, 471}, "|", 0.99f}};
  p.lines = {served,
             gate_line({571, 388, 585, 419}, "d", 0.956f, 0.9f),     // the rule's piece, over "served|"
             count,
             gate_line({700, 438, 714, 471}, "P", 0.9f, 0.9f),       // a cell of one letter, clear of the others
             gate_line({800, 438, 1000, 471}, "a | b", 0.95f, 0.9f)};  // a bar between words: the text's
  detail::gate_pieces(p, GateRule{});
  EXPECT_EQ(texts_of(p.lines), (std::vector<std::string>{"Homes served", "6,420", "P", "a | b"}));
  ASSERT_EQ(p.dropped.size(), 1u);
  EXPECT_EQ(p.dropped[0].why, "overlap");
  EXPECT_EQ(p.dropped[0].line.text, "d");
  EXPECT_EQ(p.lines[0].words.back().text, "served");
  ASSERT_EQ(p.lines[1].words.size(), 1u);  // the bar's word gone, the box to the number's edge
  EXPECT_EQ(p.lines[1].box.x1, 570);
  GateRule keep;
  keep.edge_bars = false;
  keep.overlap_chars = 0;
  OcrPage q;
  q.width = q.height = 1700;
  q.lines = {gate_line({403, 385, 577, 422}, "Homes served|", 0.96f, 0.5f),
             gate_line({571, 388, 585, 419}, "d", 0.956f, 0.9f)};
  detail::gate_pieces(q, keep);
  EXPECT_EQ(texts_of(q.lines), (std::vector<std::string>{"Homes served|", "d"}));
}

// The order model's blocks in two columns, the model reading across them: by geometry, the left column's blocks
// come first. The gap between a column's paragraphs (60 px) is wider than the gutter (50 px) in pixels, but the
// gutter is 5 character widths and the gap 3 line heights, so the columns are cut first.
TEST(Gate, BlocksInColumnsReadColumnByColumn) {
  std::vector<LineBox> boxes;
  std::vector<std::string> texts;
  for (double x0 : {0.0, 450.0}) {  // the left column's four lines, then the right's
    for (double y0 : {0.0, 30.0, 110.0, 140.0}) {
      boxes.push_back({x0, y0, x0 + 400, y0 + 20});
      texts.push_back(std::string(40, 'x'));
    }
  }
  PageOrder po;
  po.order = {0, 1, 4, 5, 2, 3, 6, 7};  // the model's: across the columns
  po.blocks.resize(4);
  po.blocks[0].lines = {0, 1};  // left, top
  po.blocks[1].lines = {4, 5};  // right, top
  po.blocks[2].lines = {2, 3};  // left, bottom
  po.blocks[3].lines = {6, 7};  // right, bottom
  po.blocks[1].label = 5;       // the labels go with their blocks
  detail::blocks_by_geometry(po, boxes, texts);
  EXPECT_EQ(po.order, (std::vector<size_t>{0, 1, 2, 3, 4, 5, 6, 7}));
  ASSERT_EQ(po.blocks.size(), 4u);
  EXPECT_EQ(po.blocks[2].lines, (std::vector<size_t>{4, 5}));
  EXPECT_EQ(po.blocks[2].label, 5);
}

// A table whose header spans three columns, the model's columns shifted by it (the header a column of its own): laid
// out again from the cells' boxes, the header spans the three, each cell in its own column, and the HTML says so.
TEST(Gate, TablesLaidOutWithTheirSpans) {
  OcrPage p;
  p.width = p.height = 1000;
  const char* texts[] = {"Results", "A", "B", "C", "1", "2", "3"};
  const LineBox boxes[] = {{100, 10, 700, 30},   {100, 40, 250, 60},  {300, 40, 450, 60}, {550, 40, 700, 60},
                           {100, 70, 250, 90},   {300, 70, 450, 90},  {550, 70, 700, 90}};
  for (int i = 0; i < 7; ++i) p.lines.push_back(gate_line(boxes[i], texts[i], 1, 1));
  p.blocks = {{"Table", 0.9f, {0, 1, 2, 3, 4, 5, 6},
               {{0, 1, {0}}, {1, 0, {1}}, {1, 2, {2}}, {1, 3, {3}}, {2, 0, {4}}, {2, 2, {5}}, {2, 3, {6}}}}};
  detail::table_spans(p);
  const std::vector<OcrCell>& c = p.blocks[0].cells;
  ASSERT_EQ(c.size(), 7u);
  EXPECT_EQ(c[0].lines, std::vector<int>{0});
  EXPECT_EQ(c[0].row, 0);
  EXPECT_EQ(c[0].col, 0);
  EXPECT_EQ(c[0].col_span, 3);
  for (int k = 1; k < 7; ++k) {
    EXPECT_EQ(c[k].row, 1 + (k - 1) / 3) << k;
    EXPECT_EQ(c[k].col, (k - 1) % 3) << k;
    EXPECT_EQ(c[k].col_span, 1) << k;
  }
  EXPECT_NE(p.json().find("\"col_span\": 3"), std::string::npos);
}

// The width buckets at a ratio of their own: at the trainer's 1.3, every width's bucket as width_bucket's; at 1.15
// a finer ladder, each bucket holding its width, past 2048 a width rounded up to 8 as at 1.3.
TEST(Gate, WidthBucketsByRatio) {
  for (int w = 1; w <= 4096; ++w) ASSERT_EQ(width_bucket(w, 1.3), width_bucket(w)) << w;
  int buckets_13 = 0, buckets_115 = 0, last_13 = 0, last_115 = 0;
  for (int w = 1; w <= 2048; ++w) {
    const int b = width_bucket(w, 1.15);
    ASSERT_GE(b, w);
    ASSERT_EQ(b % 8, 0);
    if (w >= 32) ASSERT_LE(b - w, 0.16 * w + 8) << w;  // past the first bucket, about 15% of padding at most
    buckets_115 += b != last_115;
    buckets_13 += width_bucket(w) != last_13;
    last_115 = b;
    last_13 = width_bucket(w);
  }
  EXPECT_GT(buckets_115, buckets_13);
  EXPECT_EQ(width_bucket(3001, 1.15), 3008);
}

// The order model's table whose median cell is a line of text is text; one of short cells stays a table.
TEST(Gate, TablesOfTextAreText) {
  OcrPage p;
  p.width = p.height = 1000;
  const char* texts[] = {"This is a whole line of body text, far longer than a cell", "and here the next one goes on",
                         "Name", "Age"};
  for (int i = 0; i < 4; ++i) p.lines.push_back(gate_line({10, 10.0 + 30 * i, 900, 30.0 + 30 * i}, texts[i], 1, 1));
  p.blocks = {{"Table", 0.6f, {0, 1}, {{0, 0, {0}}, {1, 0, {1}}}},
              {"Table", 0.6f, {2, 3}, {{0, 0, {2}}, {0, 1, {3}}}}};
  detail::text_tables_as_text(p, TableRule{}.max_cell_chars);
  EXPECT_EQ(p.blocks[0].label, "Text");
  EXPECT_TRUE(p.blocks[0].cells.empty());
  EXPECT_EQ(p.blocks[1].label, "Table");
  EXPECT_EQ(p.blocks[1].cells.size(), 2u);
}

// The gate's marks: a bullet the detector found apart from its item joins it as its first word; a dash with
// nothing before it and its item close after it is a list's marker too; a dash between a table's cells, far from
// them, is kept as an empty cell's; a dash alone on its row, or at a word's end (a hyphen the detector cut off), is
// a rule, dropped as before. A letter's tall box (a table's "P")
// is kept, a rule's sliver (an "l", "1" or "|"; or anything over four times taller than wide) dropped.
TEST(Gate, KeepsTheMarksOfListsAndEmptyCells) {
  OcrPage p;
  p.width = p.height = 1000;
  p.lines = {gate_line({10, 10, 22, 30}, "•", 0.9f, 0.9f),            // 0: a bullet, its item just after
             gate_line({34, 10, 400, 30}, "First item", 0.99f, 0.9f),
             gate_line({10, 50, 22, 70}, "-", 0.9f, 0.9f),            // 2: a dash, its item just after
             gate_line({36, 50, 400, 70}, "Second item", 0.99f, 0.9f),
             gate_line({10, 100, 120, 120}, "Gender", 0.99f, 0.9f),   // 4: a table row, its middle cell a dash
             gate_line({300, 100, 312, 120}, "–", 0.9f, 0.9f),
             gate_line({500, 100, 560, 120}, "0.48", 0.99f, 0.9f),
             gate_line({10, 200, 22, 220}, "-", 0.9f, 0.9f),          // 7: a dash alone on its row
             gate_line({700, 300, 712, 328}, "P", 0.85f, 0.8f),       // 8: a letter, 2.3 times taller than wide
             gate_line({700, 400, 708, 428}, "l", 0.85f, 0.8f),       // 9: a rule's sliver, read as an l
             gate_line({800, 400, 805, 440}, "W", 0.85f, 0.8f),       // 10: eight times taller than wide
             gate_line({10, 500, 200, 520}, "develop", 0.99f, 0.9f),  // 11: a word, and the hyphen after it
             gate_line({202, 500, 212, 520}, "-", 0.9f, 0.9f),        //     a line's end, not a cell
             gate_line({600, 500, 900, 520}, "the next column", 0.99f, 0.9f)};
  detail::gate_whole_lines(p, GateRule{});
  EXPECT_EQ(texts_of(p.lines), (std::vector<std::string>{"• First item", "- Second item", "Gender", "–", "0.48", "P",
                                                          "develop", "the next column"}));
  ASSERT_EQ(p.lines[0].words.size(), 0u);  // no words before: none made
  EXPECT_EQ(p.lines[0].box.x0, 10);
  ASSERT_EQ(p.dropped.size(), 4u);
  EXPECT_EQ(p.dropped[0].why, "rules");
  EXPECT_EQ(p.dropped[0].line.box.y0, 200);
  EXPECT_EQ(p.dropped[1].why, "sliver");
  EXPECT_EQ(p.dropped[2].why, "sliver");
  EXPECT_EQ(p.dropped[3].why, "rules");
  EXPECT_EQ(p.dropped[3].line.box.x0, 202);
  // a line with words gets the marker as its first word; and each mark is off at 0, as before
  OcrPage q;
  q.width = q.height = 1000;
  q.lines = {gate_line({10, 10, 22, 30}, "•", 0.9f, 0.9f), gate_line({34, 10, 400, 30}, "Item", 0.99f, 0.9f),
             gate_line({10, 100, 120, 120}, "Gender", 0.99f, 0.9f), gate_line({300, 100, 312, 120}, "–", 0.9f, 0.9f)};
  q.lines[1].words = {{{34, 10, 400, 30}, "Item", 0.99f}};
  OcrPage off = q;
  detail::gate_whole_lines(q, GateRule{});
  ASSERT_EQ(q.lines[0].words.size(), 2u);
  EXPECT_EQ(q.lines[0].words[0].text, "•");
  EXPECT_EQ(q.lines[0].words[0].box.x0, 10);
  GateRule none;
  none.bullet_join = none.dash_cell = 0;
  detail::gate_whole_lines(off, none);
  EXPECT_EQ(texts_of(off.lines), (std::vector<std::string>{"Item", "Gender"}));
  // after the split, a kept dash is not a lone mark to drop either
  OcrPage r;
  r.width = r.height = 1000;
  r.lines = {gate_line({10, 100, 120, 120}, "Gender", 0.99f, 0.9f), gate_line({300, 100, 312, 120}, "–", 0.3f, 0.3f)};
  detail::gate_pieces(r, GateRule{});
  EXPECT_EQ(r.lines.size(), 2u);
}

// Tables by geometry: the model's table in two blocks (its header, then its body) is one; its third heading,
// which the model took for a heading of the page, is inside its box and joins it; the second heading set on two
// lines is one cell, its lines a line each in the HTML; the lines before and after it stay where they were.
TEST(Gate, TablesByGeometry) {
  OcrPage p;
  p.width = p.height = 1000;
  const std::vector<std::pair<LineBox, const char*>> lines = {
      {{100, 0, 300, 20}, "Results"},                                                         // 0  Text
      {{100, 40, 200, 60}, "Name"},     {{300, 40, 360, 60}, "Age"},                          // 1, 2  Table A
      {{480, 40, 560, 60}, "Total"},                                                          // 3  Section-Header
      {{300, 61, 370, 79}, "(years)"},                                                        // 4  Table A
      {{100, 90, 180, 110}, "Ann"},     {{300, 90, 330, 110}, "31"}, {{500, 90, 530, 110}, "12"},   // 5-7  Table B
      {{100, 120, 180, 140}, "Bob"},    {{300, 120, 330, 140}, "40"}, {{500, 120, 530, 140}, "7"},  // 8-10 Table B
      {{100, 200, 400, 220}, "The end."}};                                                    // 11 Text
  for (const auto& [b, t] : lines) p.lines.push_back(gate_line(b, t, 1, 1));
  p.blocks = {{"Text", 0.9f, {0}, {}},
              {"Table", 0.7f, {1, 2, 4}, {{0, 0, {1}}, {0, 1, {2, 4}}}},
              {"Section-Header", 0.8f, {3}, {}},
              {"Table", 0.8f, {5, 6, 7, 8, 9, 10}, {}},
              {"Text", 0.9f, {11}, {}}};
  detail::tables_by_geometry(p, GridRule{});
  ASSERT_EQ(p.blocks.size(), 3u);
  EXPECT_EQ(p.blocks[0].label, "Text");
  EXPECT_EQ(p.blocks[1].label, "Table");
  EXPECT_FLOAT_EQ(p.blocks[1].confidence, 0.8f);
  EXPECT_EQ(p.blocks[2].label, "Text");
  EXPECT_EQ(texts_of(p.lines), (std::vector<std::string>{"Results", "Name", "Age", "(years)", "Total", "Ann", "31",
                                                          "12", "Bob", "40", "7", "The end."}));
  EXPECT_EQ(p.blocks[0].lines, std::vector<int>{0});
  EXPECT_EQ(p.blocks[2].lines, std::vector<int>{11});
  const std::vector<OcrCell>& c = p.blocks[1].cells;
  ASSERT_EQ(c.size(), 9u);
  const int want[9][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}};
  for (int k = 0; k < 9; ++k) {
    EXPECT_EQ(c[k].row, want[k][0]) << k;
    EXPECT_EQ(c[k].col, want[k][1]) << k;
    EXPECT_EQ(c[k].col_span, 1) << k;
  }
  EXPECT_EQ(c[1].lines, (std::vector<int>{2, 3}));  // "Age", then "(years)"
  EXPECT_EQ(c[2].lines, std::vector<int>{4});       // "Total"
  const std::string html = to_html_body(p);
  EXPECT_NE(html.find("<tr><td>Name</td><td>Age<br>(years)</td><td>Total</td></tr>"), std::string::npos) << html;
  EXPECT_NE(html.find("<tr><td>Bob</td><td>40</td><td>7</td></tr>"), std::string::npos) << html;

  // Side by side: a wide table the model cut down the middle, its rows in line across the cut, is one; two tables
  // beside each other whose rows do not line up stay two.
  auto two = [&](double shift) {
    OcrPage s;
    s.width = s.height = 1000;
    for (int r = 0; r < 4; ++r) {
      const double y = 100 + 30.0 * r, z = y + (r % 2 ? shift : 0);
      s.lines.push_back(gate_line({100, y, 180, y + 20}, "a", 1, 1));
      s.lines.push_back(gate_line({250, y, 300, y + 20}, "1", 1, 1));
      s.lines.push_back(gate_line({600, z, 650, z + 20}, "2", 1, 1));
      s.lines.push_back(gate_line({750, z, 800, z + 20}, "3", 1, 1));
    }
    s.blocks = {{"Table", 0.8f, {0, 1, 4, 5, 8, 9, 12, 13}, {}}, {"Table", 0.8f, {2, 3, 6, 7, 10, 11, 14, 15}, {}}};
    detail::tables_by_geometry(s, GridRule{});
    return s;
  };
  const OcrPage one = two(0), apart = two(13);
  ASSERT_EQ(one.blocks.size(), 1u);
  EXPECT_EQ(one.blocks[0].cells.size(), 16u);
  EXPECT_EQ(one.blocks[0].cells.back().col, 3);
  ASSERT_EQ(apart.blocks.size(), 2u);
  EXPECT_EQ(apart.blocks[0].cells.back().col, 1);

  // Two tables one above the other with a caption between them stay two, however close.
  OcrPage cap;
  cap.width = cap.height = 1000;
  for (double y : {100.0, 130.0}) {
    cap.lines.push_back(gate_line({100, y, 180, y + 20}, "x", 1, 1));
    cap.lines.push_back(gate_line({300, y, 340, y + 20}, "1", 1, 1));
  }
  cap.lines.push_back(gate_line({100, 160, 400, 178}, "Table 2. The next one", 1, 1));
  for (double y : {182.0, 212.0}) {
    cap.lines.push_back(gate_line({100, y, 180, y + 20}, "y", 1, 1));
    cap.lines.push_back(gate_line({300, y, 340, y + 20}, "2", 1, 1));
  }
  cap.blocks = {{"Table", 0.8f, {0, 1, 2, 3}, {}}, {"Text", 0.8f, {4}, {}}, {"Table", 0.8f, {5, 6, 7, 8}, {}}};
  detail::tables_by_geometry(cap, GridRule{});
  ASSERT_EQ(cap.blocks.size(), 3u);
  EXPECT_EQ(cap.blocks[0].label, "Table");
  EXPECT_EQ(cap.blocks[1].label, "Text");
  EXPECT_EQ(cap.lines[static_cast<size_t>(cap.blocks[1].lines[0])].text, "Table 2. The next one");
  EXPECT_EQ(cap.blocks[2].label, "Table");

  // A table of one column by geometry keeps the model's cells and blocks.
  OcrPage list;
  list.width = list.height = 1000;
  for (int r = 0; r < 3; ++r) list.lines.push_back(gate_line({100, 100.0 + 30 * r, 300, 120.0 + 30 * r}, "item", 1, 1));
  list.blocks = {{"Table", 0.6f, {0, 1, 2}, {{0, 0, {0}}, {0, 1, {1}}, {1, 0, {2}}}}};
  const OcrPage before = list;
  detail::tables_by_geometry(list, GridRule{});
  ASSERT_EQ(list.blocks.size(), 1u);
  EXPECT_EQ(list.blocks[0].cells.size(), 3u);
  EXPECT_EQ(list.blocks[0].cells[1].col, 1);
  EXPECT_EQ(list.json(), before.json());
}

// Tables by geometry, row labels: each set in the middle of its group of rows (the first group's first row has none), it spans its
// rows, so each row's value has its label to its left; a row of numbers alone is never a cell's next line.
TEST(Gate, TablesByGeometrySpanTheirRowLabels) {
  OcrPage p;
  p.width = p.height = 1000;
  const char* items[] = {"a", "b", "c", "d", "e", "f"};
  p.lines.push_back(gate_line({100, 0, 180, 20}, "Group", 1, 1));
  p.lines.push_back(gate_line({250, 0, 320, 20}, "Item", 1, 1));
  p.lines.push_back(gate_line({450, 0, 520, 20}, "Value", 1, 1));
  for (int r = 0; r < 6; ++r) {
    const double y = 30.0 + 30 * r;
    if (r == 1) p.lines.push_back(gate_line({100, y, 170, y + 20}, "Fruit", 1, 1));
    if (r == 4) p.lines.push_back(gate_line({100, y, 170, y + 20}, "Nuts", 1, 1));
    p.lines.push_back(gate_line({250, y, 280, y + 20}, items[r], 1, 1));
    p.lines.push_back(gate_line({450, y, 470, y + 20}, std::to_string(r + 1).c_str(), 1, 1));
  }
  std::vector<int> all(p.lines.size());
  std::iota(all.begin(), all.end(), 0);
  p.blocks = {{"Table", 0.8f, all, {}}};
  detail::tables_by_geometry(p, GridRule{});
  ASSERT_EQ(p.blocks.size(), 1u);
  const std::string html = to_html_body(p);
  EXPECT_NE(html.find("<tr><td rowspan=\"3\">Fruit</td><td>a</td><td>1</td></tr>\n<tr><td>b</td><td>2</td></tr>\n"
                      "<tr><td>c</td><td>3</td></tr>\n<tr><td rowspan=\"3\">Nuts</td><td>d</td><td>4</td></tr>"),
            std::string::npos)
      << html;
  EXPECT_NE(html.find("<tr><td>Group</td><td>Item</td><td>Value</td></tr>"), std::string::npos) << html;
}

// Tables by geometry, a two-row header: "Route" and "Year" are set between the heading that spans the quarters and
// the quarters under it, level with neither; each spans both header rows (GridRule 7), so the quarters are the
// second row alone, under the spanning heading. A heading's second line in its own column stays its cell's line.
TEST(Gate, TablesByGeometryHeaderCellsSpanTwoHeaderRows) {
  OcrPage p;
  p.width = p.height = 1000;
  const double qx[4] = {300, 420, 540, 660};
  p.lines.push_back(gate_line({300, 0, 700, 20}, "Passengers (thousands)", 1, 1));
  p.lines.push_back(gate_line({100, 20, 180, 40}, "Route", 1, 1));
  p.lines.push_back(gate_line({820, 18, 880, 38}, "Year", 1, 1));
  for (int q = 0; q < 4; ++q) p.lines.push_back(gate_line({qx[q], 40, qx[q] + 40, 60}, ("Q" + std::to_string(q + 1)).c_str(), 1, 1));
  const char* routes[] = {"Harbour", "Island", "Evening"};
  for (int r = 0; r < 3; ++r) {
    const double y = 75.0 + 30 * r;
    p.lines.push_back(gate_line({100, y, 200, y + 20}, routes[r], 1, 1));
    for (int q = 0; q < 4; ++q) p.lines.push_back(gate_line({qx[q], y, qx[q] + 40, y + 20}, std::to_string(10 + q + r).c_str(), 1, 1));
    p.lines.push_back(gate_line({820, y, 870, y + 20}, std::to_string(50 + r).c_str(), 1, 1));
  }
  std::vector<int> all(p.lines.size());
  std::iota(all.begin(), all.end(), 0);
  p.blocks = {{"Table", 0.8f, all, {}}};
  detail::tables_by_geometry(p, GridRule{});
  ASSERT_EQ(p.blocks.size(), 1u);
  const std::vector<OcrCell>& c = p.blocks[0].cells;
  int route = -1, year = -1, q1 = -1;
  for (size_t k = 0; k < c.size(); ++k) {
    const std::string& t = p.lines[static_cast<size_t>(c[k].lines[0])].text;
    if (t == "Route") route = static_cast<int>(k);
    if (t == "Year") year = static_cast<int>(k);
    if (t == "Q1") q1 = static_cast<int>(k);
  }
  ASSERT_GE(route, 0);
  ASSERT_GE(year, 0);
  ASSERT_GE(q1, 0);
  EXPECT_EQ(c[static_cast<size_t>(route)].row, 0);
  EXPECT_EQ(c[static_cast<size_t>(route)].row_span, 2);
  EXPECT_EQ(c[static_cast<size_t>(year)].row, 0);
  EXPECT_EQ(c[static_cast<size_t>(year)].row_span, 2);
  EXPECT_EQ(c[static_cast<size_t>(q1)].row, 1);
  int rows = 0;
  for (const OcrCell& x : c) rows = std::max(rows, x.row + x.row_span);
  EXPECT_EQ(rows, 5);  // two header rows and three of routes
  const std::string html = to_html_body(p);
  EXPECT_NE(html.find("<td rowspan=\"2\">Route</td>"), std::string::npos) << html;
  EXPECT_NE(html.find("<tr><td>Q1</td><td>Q2</td><td>Q3</td><td>Q4</td></tr>"), std::string::npos) << html;
}

namespace {
// A page of one Table block holding every line, each (x0, y, x1, text) 20 px tall, laid out by geometry.
OcrPage grid_page(const std::vector<std::tuple<double, double, double, const char*>>& lines, const GridRule& rule) {
  OcrPage p;
  p.width = p.height = 1000;
  for (const auto& [x0, y, x1, t] : lines) p.lines.push_back(gate_line({x0, y, x1, y + 20}, t, 1, 1));
  std::vector<int> all(p.lines.size());
  std::iota(all.begin(), all.end(), 0);
  p.blocks = {{"Table", 0.8f, all, {}}};
  detail::tables_by_geometry(p, rule);
  return p;
}
}  // namespace

// Tables by geometry, a body cell's next line (GridRule 5, body_wrap): a row at the table's own pitch, of fewer
// lines than the row above and under half its columns, that reads as going on (lowercase first, or under a line
// that ends open) joins the cell above; a capitalised line under one that ends closed is a row of its own.
TEST(Gate, TablesByGeometryJoinABodyCellsNextLine) {
  const std::vector<std::tuple<double, double, double, const char*>> lines = {
      {100, 0, 180, "Drug"},       {300, 0, 350, "Dose"}, {450, 0, 520, "Effect"},
      {100, 25, 180, "Aspirin"},   {300, 25, 330, "100"}, {450, 25, 600, "Pain relief and"},
      {450, 50, 500, "fever"},
      {100, 75, 190, "Ibuprofen"}, {300, 75, 330, "200"}, {450, 75, 560, "Swelling"},
      {100, 100, 190, "Naproxen"}, {300, 100, 330, "250"}, {450, 100, 560, "Long acting"},
      {450, 125, 500, "Daily"}};
  const OcrPage on = grid_page(lines, GridRule{});
  const std::string html = to_html_body(on);
  EXPECT_NE(html.find("<tr><td>Aspirin</td><td>100</td><td>Pain relief and<br>fever</td></tr>\n<tr><td>Ibuprofen"),
            std::string::npos) << html;
  EXPECT_NE(html.find("<tr><td></td><td></td><td>Daily</td></tr>"), std::string::npos) << html;
  GridRule off;
  off.body_wrap = false;
  const std::string apart = to_html_body(grid_page(lines, off));
  EXPECT_NE(apart.find("<tr><td></td><td></td><td>fever</td></tr>"), std::string::npos) << apart;
  // a hyphen at the end of the line above carries a capitalised line on; a value never joins
  const std::vector<std::tuple<double, double, double, const char*>> more = {
      {100, 0, 180, "Site"},   {300, 0, 350, "n"},   {450, 0, 520, "Note"},
      {100, 25, 180, "North"}, {300, 25, 330, "12"}, {450, 25, 600, "Anti-"},
      {450, 50, 520, "Inflammatory"},
      {100, 75, 180, "South"}, {300, 75, 330, "14"}, {450, 75, 560, "Mild and"},
      {450, 100, 480, "12"}};
  const std::string h2 = to_html_body(grid_page(more, GridRule{}));
  EXPECT_NE(h2.find("<td>Anti-<br>Inflammatory</td>"), std::string::npos) << h2;
  EXPECT_NE(h2.find("<td>Mild and</td></tr>"), std::string::npos) << h2;
}

// Tables by geometry, a section row (GridRule 8): a lone body cell centred in a table of three columns spans it;
// one set off centre stays in its column, and with section_rows off the centred one does too; a wrapped cell's line,
// set where the line above it in its column starts, is no section either, centred or not.
TEST(Gate, TablesByGeometrySpanACentredSectionRow) {
  const std::vector<std::tuple<double, double, double, const char*>> lines = {
      {100, 0, 200, "Gene"},     {300, 0, 360, "Cases"}, {450, 0, 560, "Controls"},
      {100, 25, 180, "BRCA1"},   {300, 25, 330, "12"},   {450, 25, 480, "30"},
      {100, 50, 180, "BRCA2"},   {300, 50, 315, "8"},    {450, 50, 480, "25"},
      {320, 75, 400, "Women"},
      {100, 100, 180, "BRCA1"},  {300, 100, 330, "15"},  {450, 100, 480, "28"},
      {450, 125, 530, "Others"},
      {100, 150, 180, "BRCA2"},  {300, 150, 315, "9"},   {450, 150, 480, "22"}};
  const std::string html = to_html_body(grid_page(lines, GridRule{}));
  EXPECT_NE(html.find("<tr><td colspan=\"3\">Women</td></tr>"), std::string::npos) << html;
  EXPECT_NE(html.find("<tr><td></td><td></td><td>Others</td></tr>"), std::string::npos) << html;
  GridRule off;
  off.section_rows = false;
  const std::string kept = to_html_body(grid_page(lines, off));
  EXPECT_NE(kept.find("<tr><td></td><td>Women</td><td></td></tr>"), std::string::npos) << kept;
  const std::vector<std::tuple<double, double, double, const char*>> wrapped = {
      {100, 0, 140, "No."},    {300, 0, 420, "Description"},     {600, 0, 660, "Date"},
      {100, 25, 110, "1"},     {300, 25, 450, "Amendment No. 1"}, {600, 25, 660, "3-30-95"},
      {300, 50, 470, "Agreement (PVNGS Unit"},
      {100, 75, 110, "2"},     {300, 75, 440, "Amendment No. 2"}, {600, 75, 660, "3-28-97"}};
  const std::string w = to_html_body(grid_page(wrapped, GridRule{}));
  EXPECT_NE(w.find("<tr><td></td><td>Agreement (PVNGS Unit</td><td></td></tr>"), std::string::npos) << w;
}

// Tables by geometry, a heading over columns (GridRule 4, header_cover): a header line spans each column it lies
// over by more than header_cover of the column's width, though it covers under half of the last; a body line keeps
// `cover`; with header_cover 0 the heading takes the one column it covers half of.
TEST(Gate, TablesByGeometryHeadingSpansTheColumnsItOverlaps) {
  const std::vector<std::tuple<double, double, double, const char*>> lines = {
      {100, 0, 180, "Drug"},  {310, 0, 430, "Treatment group"},
      {300, 25, 340, "A"},    {420, 25, 460, "B"},
      {100, 50, 180, "X"},    {300, 50, 340, "1"},  {420, 50, 460, "2"},
      {100, 75, 180, "Y"},    {300, 75, 340, "3"},  {420, 75, 460, "4"},
      {100, 100, 180, "Z"},   {310, 100, 430, "5 to 6"}};
  const OcrPage on = grid_page(lines, GridRule{});
  const std::string html = to_html_body(on);
  EXPECT_NE(html.find("<tr><td>Drug</td><td colspan=\"2\">Treatment group</td></tr>"), std::string::npos) << html;
  EXPECT_NE(html.find("<tr><td>Z</td><td>5 to 6</td><td></td></tr>"), std::string::npos) << html;
  GridRule off;
  off.header_cover = 0;
  const std::string one = to_html_body(grid_page(lines, off));
  EXPECT_NE(one.find("<tr><td>Drug</td><td>Treatment group</td><td></td></tr>"), std::string::npos) << one;
}

// Tables by geometry, a column of group labels (GridRule 4, label_band): labels set left of the items, between
// their groups' rows, are no column of the fullest rows; they get one of their own and span their groups
// (GridRule 6), the items in the next. Without it each label is a row of its own in the items' column. A stub set
// out left of its totals (each such line its row's first, the row's values beside it) gets no column of its own.
TEST(Gate, TablesByGeometryGiveGroupLabelsAColumn) {
  const std::vector<std::tuple<double, double, double, const char*>> lines = {
      {150, 0, 220, "Crop"},     {300, 0, 340, "Yield"}, {420, 0, 460, "Area"},
      {150, 30, 210, "Wheat"},   {300, 30, 330, "12"},  {420, 30, 450, "14"},
      {20, 45, 90, "North"},
      {150, 60, 210, "Barley"},  {300, 60, 315, "8"},   {420, 60, 435, "9"},
      {150, 90, 210, "Wheat"},   {300, 90, 315, "5"},   {420, 90, 435, "6"},
      {20, 105, 90, "South"},
      {150, 120, 210, "Barley"}, {300, 120, 315, "7"},  {420, 120, 435, "3"}};
  const OcrPage on = grid_page(lines, GridRule{});
  const std::string html = to_html_body(on);
  EXPECT_NE(html.find("<tr><td></td><td>Crop</td><td>Yield</td><td>Area</td></tr>"), std::string::npos) << html;
  EXPECT_NE(html.find("<tr><td rowspan=\"3\">North</td><td>Wheat</td><td>12</td><td>14</td></tr>"), std::string::npos)
      << html;
  EXPECT_NE(html.find("<tr><td rowspan=\"3\">South</td><td>Wheat</td><td>5</td><td>6</td></tr>"), std::string::npos)
      << html;
  GridRule off;
  off.label_band = false;
  const std::string same = to_html_body(grid_page(lines, off));
  EXPECT_NE(same.find("<tr><td>North</td><td></td><td></td></tr>"), std::string::npos) << same;
  const std::vector<std::tuple<double, double, double, const char*>> stub = {
      {60, 0, 120, "Total"},  {200, 0, 210, "$"}, {250, 0, 280, "530"},
      {10, 25, 55, "JBI"},    {250, 25, 280, "375"},
      {10, 50, 55, "DCS"},    {250, 50, 280, "116"},
      {10, 75, 55, "ICS"},    {250, 75, 280, "16"},
      {60, 100, 150, "Sum"},  {200, 100, 210, "$"}, {250, 100, 280, "900"}};
  const std::string s = to_html_body(grid_page(stub, GridRule{}));
  EXPECT_NE(s.find("<tr><td>JBI</td><td></td><td>375</td></tr>"), std::string::npos) << s;
}

namespace {
// A page of lines (x0, y, x1, text, block), each 20 px tall, its blocks labelled as given, laid out by geometry.
OcrPage blocks_page(const std::vector<std::tuple<double, double, double, const char*, int>>& lines,
                    const std::vector<std::string>& labels, const GridRule& rule) {
  OcrPage p;
  p.width = p.height = 1000;
  for (const std::string& l : labels) p.blocks.push_back({l, 0.8f, {}, {}});
  for (const auto& [x0, y, x1, t, b] : lines) {
    p.blocks[static_cast<size_t>(b)].lines.push_back(static_cast<int>(p.lines.size()));
    p.lines.push_back(gate_line({x0, y, x1, y + 20}, t, 1, 1));
  }
  detail::tables_by_geometry(p, rule);
  return p;
}

std::vector<std::string> labels_of(const OcrPage& p) {
  std::vector<std::string> out;
  for (const OcrBlock& b : p.blocks) out.push_back(b.label);
  return out;
}
}  // namespace

// Tables by geometry, a heading above the table (GridRule 2, heading_up): a block of one line just above the
// table, over its value columns, joins it as its first header row, spanning the columns it lies over; a
// paragraph's line above it, over its first column too, does not.
TEST(Gate, TablesByGeometryTakeAHeadingOverTheirColumns) {
  const std::vector<std::tuple<double, double, double, const char*, int>> lines = {
      {300, 0, 560, "Mean error, trials 2-8", 0},
      {100, 50, 180, "Group", 1}, {300, 50, 370, "Stage 1", 1}, {420, 50, 490, "Stage 2", 1}, {540, 50, 610, "Stage 3", 1},
      {100, 75, 180, "A", 1},     {300, 75, 340, "1.2", 1},     {420, 75, 460, "3.4", 1},     {540, 75, 580, "5.6", 1},
      {100, 100, 180, "B", 1},    {300, 100, 340, "2.2", 1},    {420, 100, 460, "4.4", 1},    {540, 100, 580, "6.6", 1}};
  const OcrPage on = blocks_page(lines, {"Caption", "Table"}, GridRule{});
  EXPECT_EQ(labels_of(on), std::vector<std::string>{"Table"});
  const std::string html = to_html_body(on);
  EXPECT_NE(html.find("<table>\n<tr><td></td><td colspan=\"3\">Mean error, trials 2-8</td></tr>\n<tr><td>Group</td>"),
            std::string::npos) << html;
  GridRule off;
  off.heading_up = false;
  EXPECT_EQ(labels_of(blocks_page(lines, {"Caption", "Table"}, off)), (std::vector<std::string>{"Caption", "Table"}));
  auto para = lines;
  std::get<0>(para[0]) = 100;
  std::get<3>(para[0]) = "the last line of a paragraph above it";
  EXPECT_EQ(labels_of(blocks_page(para, {"Text", "Table"}, GridRule{})), (std::vector<std::string>{"Text", "Table"}));
  // over the value columns too, but a caption's next line (lowercase first) or a sentence's end is no heading
  for (const char* t : {"with the electron donor", "Means of three trials."}) {
    auto tail = lines;
    std::get<3>(tail[0]) = t;
    EXPECT_EQ(labels_of(blocks_page(tail, {"Caption", "Table"}, GridRule{})),
              (std::vector<std::string>{"Caption", "Table"})) << t;
  }
}

// Tables by geometry, captions (caption_out) and a title row (GridRule 9, title_out): the table's own "Table 2."
// line above its others leaves the grid as a Caption block before it; a lone first row over its columns, from the
// first, is its title, a Caption block before it, and the grid is laid out without it.
TEST(Gate, TablesByGeometryLeaveTheirCaptionAndTitleOut) {
  const std::vector<std::tuple<double, double, double, const char*, int>> cap = {
      {100, 0, 400, "Table 2. Results", 0},
      {100, 30, 180, "Name", 0}, {300, 30, 340, "Age", 0},
      {100, 55, 180, "Ann", 0},  {300, 55, 330, "31", 0},
      {100, 80, 180, "Bob", 0},  {300, 80, 330, "40", 0}};
  const OcrPage on = blocks_page(cap, {"Table"}, GridRule{});
  ASSERT_EQ(labels_of(on), (std::vector<std::string>{"Caption", "Table"}));
  EXPECT_EQ(on.lines[static_cast<size_t>(on.blocks[0].lines[0])].text, "Table 2. Results");
  EXPECT_EQ(on.blocks[1].cells.size(), 6u);
  GridRule off;
  off.caption_out = false;
  off.title_out = false;
  const OcrPage kept = blocks_page(cap, {"Table"}, off);
  EXPECT_EQ(labels_of(kept), std::vector<std::string>{"Table"});
  EXPECT_EQ(kept.blocks[0].cells.size(), 7u);
  const std::vector<std::tuple<double, double, double, const char*, int>> title = {
      {100, 0, 450, "STANDARD RESISTANCE VALUES", 0},
      {100, 30, 180, "Code", 0}, {250, 30, 300, "Ohms", 0}, {400, 30, 450, "Watts", 0},
      {100, 55, 180, "R1", 0},   {250, 55, 290, "100", 0},  {400, 55, 430, "0.5", 0},
      {100, 80, 180, "R2", 0},   {250, 80, 290, "220", 0},  {400, 80, 430, "1", 0}};
  const OcrPage t = blocks_page(title, {"Table"}, GridRule{});
  ASSERT_EQ(labels_of(t), (std::vector<std::string>{"Caption", "Table"}));
  EXPECT_EQ(t.lines[static_cast<size_t>(t.blocks[0].lines[0])].text, "STANDARD RESISTANCE VALUES");
  const std::string html = to_html_body(t);
  EXPECT_NE(html.find("<table>\n<tr><td>Code</td><td>Ohms</td><td>Watts</td></tr>"), std::string::npos) << html;
  GridRule stay;
  stay.title_out = false;
  const std::string with = to_html_body(blocks_page(title, {"Table"}, stay));
  EXPECT_NE(with.find("<tr><td colspan=\"3\">STANDARD RESISTANCE VALUES</td></tr>"), std::string::npos) << with;
}

// A page's ruled lines (find_rules): a rule across and one down, thin, are found and scaled to the page's pixels;
// a line of text's strokes, short, and a filled box, thick, are not.
TEST(Gate, FindsAPagesRuledLines) {
  GrayImage g;
  g.width = 400;
  g.height = 300;
  g.pixels.assign(static_cast<size_t>(g.width) * g.height, 255);
  auto fill = [&](int x0, int y0, int x1, int y1, uint8_t v) {
    for (int y = y0; y < y1; ++y) {
      for (int x = x0; x < x1; ++x) g.pixels[static_cast<size_t>(y) * g.width + x] = v;
    }
  };
  fill(50, 100, 350, 102, 0);    // a rule across, 2 px
  fill(200, 20, 201, 280, 120);  // a grey rule down, 1 px
  for (int k = 0; k < 20; ++k) fill(40 + 12 * k, 40, 48 + 12 * k, 52, 0);  // letters: strokes 8 px wide
  fill(60, 200, 160, 240, 0);    // a filled box
  const std::vector<LineBox> r = detail::find_rules(g.view(), 800, 600, detail::kRuleDarkFull);
  ASSERT_EQ(r.size(), 2u);
  const LineBox& across = r[0].x1 - r[0].x0 > r[0].y1 - r[0].y0 ? r[0] : r[1];
  const LineBox& down = &across == &r[0] ? r[1] : r[0];
  EXPECT_DOUBLE_EQ(across.x0, 100);
  EXPECT_DOUBLE_EQ(across.x1, 700);
  EXPECT_DOUBLE_EQ(across.y0, 200);
  EXPECT_DOUBLE_EQ(across.y1, 204);
  EXPECT_DOUBLE_EQ(down.x0, 400);
  EXPECT_DOUBLE_EQ(down.y0, 40);
  EXPECT_DOUBLE_EQ(down.y1, 560);
}

// The false-table check's features: a grid of numbers under a header, its rules across, and the order model's
// probabilities as given; and a table the check finds none (check_keep over every probability) is written as text,
// its block relabelled, in the model's order, its cells gone, the lines it took given back.
TEST(Gate, TablesByGeometryCheckForFalseTables) {
  const std::vector<std::tuple<double, double, double, const char*, int>> lines = {
      {100, 0, 180, "Name", 1},  {300, 0, 340, "Age", 1},  {450, 0, 500, "Score", 1},
      {100, 25, 180, "Ann", 1},  {300, 25, 330, "31", 1},  {450, 25, 480, "12", 1},
      {100, 50, 180, "Bob", 1},  {300, 50, 330, "40", 1},  {450, 50, 480, "7", 1},
      {320, 12, 330, "x", 0}};  // a line of another block inside its box: the table takes it
  OcrPage p;
  p.width = p.height = 1000;
  p.blocks = {{"Text", 0.9f, {}, {}}, {"Table", 0.8f, {}, {}}};
  for (const auto& [x0, y, x1, t, b] : lines) {
    p.blocks[static_cast<size_t>(b)].lines.push_back(static_cast<int>(p.lines.size()));
    p.lines.push_back(gate_line({x0, y, x1, y + 20}, t, 1, 1));
  }
  p.rules = {{95, -3, 505, -1}, {95, 22, 505, 23}, {95, 72, 505, 74}, {98, -3, 99, 74}};
  std::vector<int> table = {0, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> probs(static_cast<size_t>(kBlockLabels), 0.0f);
  probs[static_cast<size_t>(block_label("Table"))] = 0.7f;
  probs[static_cast<size_t>(block_label("Text"))] = 0.2f;
  const std::vector<OcrCell> none;
  const std::vector<float> f = detail::table_features(p, table, none, probs, GridRule{});
  const std::vector<std::string>& names = detail::table_feature_names();
  ASSERT_EQ(f.size(), names.size());
  auto at = [&](const char* name) { return f[static_cast<size_t>(std::find(names.begin(), names.end(), name) - names.begin())]; };
  EXPECT_FLOAT_EQ(at("no_grid"), 1.0f);
  EXPECT_NEAR(at("numbers"), 4.0 / 9, 1e-6);
  EXPECT_FLOAT_EQ(at("p_table"), 0.7f);
  EXPECT_FLOAT_EQ(at("p_text"), 0.2f);
  EXPECT_FLOAT_EQ(at("h_rules"), 0.75f);  // three rules across over three rows, four row edges
  EXPECT_NEAR(at("v_rules"), 1.0 / 2, 1e-6);  // one down over one column (no grid)
  EXPECT_FLOAT_EQ(at("sentences"), 0.0f);
  OcrPage kept = p, text = p;
  GridRule on;
  on.check = true;
  on.check_keep = 0.0f;  // every probability over it: kept
  detail::tables_by_geometry(kept, on);
  EXPECT_EQ(labels_of(kept), (std::vector<std::string>{"Table"}));
  on.check_keep = 1.01f;
  detail::tables_by_geometry(text, on);
  EXPECT_EQ(labels_of(text), (std::vector<std::string>{"Text", "Text"}));
  EXPECT_TRUE(text.blocks[1].cells.empty());
  EXPECT_EQ(text.blocks[0].lines, std::vector<int>{9});  // "x" its own block's again
  EXPECT_EQ(texts_of(text.lines), texts_of(p.lines));
}

// The false-table check's fitted weights, for pages read by OCR and from their text layer: a ruled grid of numbers
// under a header, sure the model's Table, is a table; two columns of sentences the model was unsure of, over a
// picture, are not.
TEST(Gate, TableCheckWeightsTellTablesFromText) {
  auto page_of = [](bool table) {
    OcrPage p;
    p.width = 1000;
    p.height = 1400;
    if (table) {
      const char* head[] = {"Group", "Mean", "SD", "n"};
      for (int c = 0; c < 4; ++c) p.lines.push_back(gate_line({100.0 + 200 * c, 100, 160.0 + 200 * c, 120}, head[c], 1, 1));
      for (int r = 0; r < 8; ++r) {
        const double y = 130.0 + 25 * r;
        p.lines.push_back(gate_line({100, y, 170, y + 20}, ("Site " + std::to_string(r + 1)).c_str(), 1, 1));
        for (int c = 1; c < 4; ++c) {
          p.lines.push_back(gate_line({100.0 + 200 * c, y, 150.0 + 200 * c, y + 20}, std::to_string(10 + r * c).c_str(), 1, 1));
        }
      }
      p.rules = {{90, 95, 760, 97}, {90, 125, 760, 126}, {90, 330, 760, 332}};
    } else {
      for (int r = 0; r < 10; ++r) {
        const double y = 100.0 + 25 * r;
        p.lines.push_back(gate_line({100, y, 480, y + 20}, "the results of the trial were read again by two.", 1, 1));
        p.lines.push_back(gate_line({520, y, 900, y + 20}, "and the samples were kept for a week at four degrees.", 1, 1));
      }
      p.pictures = {{80, 80, 920, 360}};
    }
    for (int k = 0; k < 30; ++k) {  // the rest of the page
      p.lines.push_back(gate_line({100, 500.0 + 25 * k, 900, 520.0 + 25 * k}, "A line of the page's text.", 1, 1));
    }
    return p;
  };
  for (const bool text_layer : {false, true}) {
    for (const bool table : {true, false}) {
      OcrPage p = page_of(table);
      std::vector<int> lines;
      for (int i = 0; i < (table ? 36 : 20); ++i) lines.push_back(i);
      p.blocks = {{"Table", 0.9f, lines, {}}};
      std::vector<float> probs(static_cast<size_t>(kBlockLabels), 0.0f);
      probs[static_cast<size_t>(block_label("Table"))] = table ? 0.95f : 0.55f;
      probs[static_cast<size_t>(block_label("Text"))] = table ? 0.03f : 0.4f;
      const std::vector<OcrCell> grid = detail::table_grid(p, lines, GridRule{});
      const float prob = detail::table_probability(detail::table_features(p, lines, grid, probs, GridRule{}), text_layer);
      if (table) {
        EXPECT_GT(prob, 0.5f) << "text layer " << text_layer;
      } else {
        EXPECT_LT(prob, 0.5f) << "text layer " << text_layer;
      }
    }
  }
}

// The detector alone (a text layer's page) finds the lines the engine reads, where it reads them.
TEST(Engine, FindsALinesPlacesAlone) {
  const GrayImage page = load_image(kSet / "page.png");
  EngineOptions o;
  o.pool.workers = 1;
  const OcrPage p = Engine(kSet, o).ocr(page.view());
  std::vector<LineBox> found = LineFinder(kSet, o).find(page.view());
  std::vector<LineBox> read;
  for (const OcrLine& l : p.lines) read.push_back(l.box);
  auto by_y = [](const LineBox& a, const LineBox& b) { return a.y0 < b.y0; };
  std::sort(found.begin(), found.end(), by_y);
  std::sort(read.begin(), read.end(), by_y);
  ASSERT_EQ(found.size(), 3u);
  ASSERT_EQ(found.size(), read.size());
  for (size_t i = 0; i < found.size(); ++i) {
    EXPECT_NEAR(found[i].x0, read[i].x0, 1.0) << i;
    EXPECT_NEAR(found[i].y0, read[i].y0, 1.0) << i;
    EXPECT_NEAR(found[i].x1, read[i].x1, 1.0) << i;
    EXPECT_NEAR(found[i].y1, read[i].y1, 1.0) << i;
  }
}

// A quarter turn by its three bins' sum (60, 90 and 120 degrees for 90), where no bin alone is sure; the
// upright quarter's bins (330, 0 and 30) never turn the page.
TEST(Engine, TurnsByQuartersOfThreeBins) {
  std::vector<float> z(13, 0.0f);
  z[2] = z[3] = z[4] = 2.0f;  // each bin 0.23, the quarter 0.69
  EXPECT_EQ(detail::rotation_of(z, 0.65f), 0);
  EXPECT_EQ(detail::rotation_of(z, 0.65f, true), 90);
  EXPECT_EQ(detail::rotation_of(z, 0.7f, true), 0);
  std::vector<float> up(13, 0.0f);
  up[11] = up[0] = up[1] = 4.0f;
  EXPECT_EQ(detail::rotation_of(up, 0.5f, true), 0);
  std::vector<float> half(13, 0.0f);
  half[7] = 9.0f;  // 210 degrees: the 180 quarter's
  EXPECT_EQ(detail::rotation_of(half, 0.65f, true), 180);
  EXPECT_EQ(detail::rotation_of(half, 0.65f), 0) << "the bin alone is not a quarter turn's";
}

// The turn a page's readings bear out, judged only where a reading reaches the floor (no veto without evidence).
// A half turn stands unless the page reads surer as given; where neither reading is sure (a script the recognizer
// cannot read either way up), orientation's turn as made without the check stands. A quarter turn goes the other way
// round on the surer reading; an unsure sideways page turns to its surer side where that is surer than the page as
// given and reaches the floor.
TEST(Engine, TurnsTheReadingsBearOut) {
  const TurnCheck c;  // floor 0.8
  EXPECT_EQ(detail::settle_turn(180, 0.95, 0.60, c, 180), 180);
  EXPECT_EQ(detail::settle_turn(180, 0.60, 0.95, c, 180), 0) << "upright text, taken for upside down";
  EXPECT_EQ(detail::settle_turn(180, 0.85, 0.82, c, 180), 180);
  EXPECT_EQ(detail::settle_turn(180, 0.82, 0.85, c, 180), 0);
  EXPECT_EQ(detail::settle_turn(180, 0.90, 0.90, c, 180), 180) << "a tie vetoes nothing";
  EXPECT_EQ(detail::settle_turn(180, 0.70, 0.50, c, 180), 180) << "no reading sure: orientation's turn";
  EXPECT_EQ(detail::settle_turn(180, 0.50, 0.70, c, 180), 180) << "no reading sure: orientation's turn";
  EXPECT_EQ(detail::settle_turn(180, 0.70, 0.50, c, 0), 0) << "the sums' half turn, unread: the bins' turn";
  EXPECT_EQ(detail::settle_turn(90, 0.55, 0.93, c, 0), 270);
  EXPECT_EQ(detail::settle_turn(270, 0.55, 0.93, c, 270), 90);
  EXPECT_EQ(detail::settle_turn(90, 0.93, 0.55, c, 90), 90);
  EXPECT_EQ(detail::settle_turn(90, 0.90, 0.90, c, 90), 90);
  EXPECT_EQ(detail::settle_turn(90, 0.40, 0.70, c, 0), 90) << "neither reading sure: orientation's turn";
  TurnCheck no_quarters = c;
  no_quarters.quarters = false;
  EXPECT_EQ(detail::settle_turn(90, 0.55, 0.93, no_quarters, 90), 90);
  EXPECT_EQ(detail::settle_turn(0, 0.2, 0.99, c, 0), 0);
  EXPECT_EQ(detail::settle_sideways(90, 0.30, 0.92, 0.50, c), 90);
  EXPECT_EQ(detail::settle_sideways(90, 0.30, 0.50, 0.92, c), 270);
  EXPECT_EQ(detail::settle_sideways(270, 0.30, 0.50, 0.92, c), 90);
  EXPECT_EQ(detail::settle_sideways(90, 0.95, 0.92, 0.50, c), 0) << "the page reads best as given";
  EXPECT_EQ(detail::settle_sideways(90, 0.30, 0.70, 0.60, c), 0) << "no sideways reading reaches the floor";
  std::vector<float> z(13, 0.0f);
  z[2] = z[3] = z[4] = 10.0f;
  const detail::SidewaysMass right = detail::sideways_mass(z);
  EXPECT_GT(right.right, 0.99);
  EXPECT_LT(right.left, 0.01);
  std::fill(z.begin(), z.end(), 0.0f);
  z[9] = 10.0f;
  EXPECT_GT(detail::sideways_mass(z).left, 0.99);
  EXPECT_NEAR(detail::sideways_mass(std::vector<float>(13, 0.0f)).right, 3.0 / 13, 1e-9);
}

// The fake set's orientation model with other biases: its 13 logits' constant (`was`; the file's own 10 on upright),
// found in the file by its bytes and written over.
void set_orient_bias(const std::filesystem::path& file, const std::array<float, 13>& was,
                     const std::array<float, 13>& bias) {
  std::ifstream in(file, std::ios::binary);
  std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  in.close();
  const size_t at = bytes.find(std::string(reinterpret_cast<const char*>(was.data()), sizeof was));
  ASSERT_NE(at, std::string::npos);
  std::memcpy(bytes.data() + at, bias.data(), sizeof bias);
  std::ofstream(file, std::ios::binary | std::ios::trunc) << bytes;
}

// The checks end to end, on the fake set, whose recognizer reads its bars as surely either way up: a half turn the
// orientation model is sure of is read, both ways, and made, since the page reads no surer as given (a tie vetoes
// nothing); with the check off, it is made unread. A quarter turn sure by its three bins' sum alone is made where
// turns are read (and read both ways, and kept), not by a bin's probability. An unsure page whose sideways bins hold its probability is
// read as given and turned both ways, and stays as given. Nothing is read where nothing may turn (turn_probability 2,
// --as-given's).
TEST(Engine, ReadsTurnsBeforeMakingThem) {
  const std::filesystem::path set = std::filesystem::temp_directory_path() / "carbon-engine-turn-check-set";
  std::filesystem::remove_all(set);
  std::filesystem::copy(kSet, set, std::filesystem::copy_options::recursive);
  const GrayImage page = load_image(set / "page.png");
  std::array<float, 13> upright{}, half{}, quarter{}, sideways{};
  upright[0] = 10.0f;
  half[6] = 10.0f;
  quarter[2] = quarter[3] = quarter[4] = 10.0f;  // a third on each bin: no bin sure, the quarter all of it
  sideways = quarter;
  sideways[8] = sideways[9] = sideways[10] = 10.0f;  // each side half: no quarter sure, the sideways bins all of it
  set_orient_bias(set / "orient.onnx", upright, half);
  {
    const OcrPage p = Engine(set).ocr(page.view());
    EXPECT_EQ(p.rotation, 180);
    ASSERT_EQ(p.turn_reads.size(), 2u);
    EXPECT_EQ(p.turn_reads[0].rotation, 180);
    EXPECT_EQ(p.turn_reads[1].rotation, 0);
    EXPECT_GT(p.turn_reads[0].sure, 0.9);
    EXPECT_NEAR(p.turn_reads[0].sure, p.turn_reads[1].sure, 1e-6);
    EXPECT_NE(p.json().find("\"turn_reads\": [{\"rotation\": 180, \"sure\": "), std::string::npos);
    EngineOptions off;
    off.turn_check.reset();
    const OcrPage q = Engine(set, off).ocr(page.view());
    EXPECT_EQ(q.rotation, 180);
    EXPECT_TRUE(q.turn_reads.empty());
    EXPECT_EQ(q.json().find("turn_reads"), std::string::npos);
    EngineOptions never;
    never.turn_probability = 2.0f;
    const OcrPage r = Engine(set, never).ocr(page.view());
    EXPECT_EQ(r.rotation, 0);
    EXPECT_TRUE(r.turn_reads.empty());
  }
  set_orient_bias(set / "orient.onnx", half, quarter);
  {
    const OcrPage p = Engine(set).ocr(page.view());
    EXPECT_EQ(p.rotation, 90);
    ASSERT_EQ(p.turn_reads.size(), 2u);
    EXPECT_EQ(p.turn_reads[1].rotation, 270);
    EngineOptions off;
    off.turn_check.reset();
    EXPECT_EQ(Engine(set, off).ocr(page.view()).rotation, 0) << "no bin alone is sure";
    EngineOptions bins;
    bins.turn_check->sum_bins = false;
    const OcrPage b = Engine(set, bins).ocr(page.view());
    EXPECT_EQ(b.rotation, 0);
    EXPECT_EQ(b.turn_reads.size(), 3u) << "unsure, and sideways: read three ways";
  }
  set_orient_bias(set / "orient.onnx", quarter, sideways);
  {
    const OcrPage p = Engine(set).ocr(page.view());
    EXPECT_EQ(p.rotation, 0);
    ASSERT_EQ(p.turn_reads.size(), 3u);
    EXPECT_EQ(p.turn_reads[0].rotation, 0);
    EXPECT_EQ(p.turn_reads[1].rotation, 90);
    EXPECT_EQ(p.turn_reads[2].rotation, 270);
    EXPECT_EQ(p.text(), "aaa\naaaaa\naa\n");
  }
  std::filesystem::remove_all(set);
}

// Five graded options (the characters' confidence on by default, the rest off): a tall page detected by its
// area (letter pages as before); the second pass at the tiny lines' own scale; a justified line of one wide gap
// kept whole; a line's confidence over its characters; a word's box at its ink.
TEST(Gate, TheGradedOptions) {
  GrayImage tall;
  tall.width = 600;
  tall.height = 8000;
  tall.pixels.assign(600 * 8000, 255);
  const GrayImage by_side = detail::det_scale(tall.view(), 1024);
  EXPECT_EQ(by_side.height, 1024);
  EXPECT_LT(by_side.width, 80);
  const GrayImage by_area = detail::det_scale(tall.view(), 1024, 1024 * 800);
  EXPECT_NEAR(1.0 * by_area.width * by_area.height, 1024.0 * 800, 1024.0 * 800 * 0.01);
  GrayImage letter;
  letter.width = 1700;
  letter.height = 2200;
  letter.pixels.assign(1700 * 2200, 255);
  const GrayImage l1 = detail::det_scale(letter.view(), 1024), l2 = detail::det_scale(letter.view(), 1024, 1024 * 800);
  EXPECT_EQ(l1.width, l2.width);
  EXPECT_EQ(l1.height, l2.height);

  const GrayImage tiny = tiny_page();
  EngineOptions o;
  o.pool.workers = 2;
  o.crop.flip_margin = 0;
  o.det.second_pass_target = 18;
  const OcrPage p = Engine(kSet, o).ocr(tiny.view());
  EXPECT_TRUE(p.second_pass);
  ASSERT_EQ(p.lines.size(), 6u) << "the tiny lines found at their own scale";

  OcrLine two;
  two.box = {0, 0, 300, 20};
  two.words = {{{0, 0, 100, 20}, "Total", 0.9f}, {{114, 0, 300, 20}, "amount", 0.9f}};  // one gap, 0.7 heights
  two.text = "Total amount";
  SplitRule rule;
  EXPECT_EQ(split_lines({two}, rule).size(), 2u);
  rule.justified = true;
  EXPECT_EQ(split_lines({two}, rule).size(), 1u);

  const CropRule crop;
  const std::vector<std::string> vocab = {"", "a", "b", " "};
  CtcLine d;
  d.ids = {1, 3, 2};
  d.frames = {2, 5, 9};
  d.probs = {0.9f, 0.3f, 0.7f};
  OcrLine all, chars;
  all.box = chars.box = {0, 0, 100, 20};
  detail::read_line(all, d, 16, crop, vocab);
  detail::read_line(chars, d, 16, crop, vocab, nullptr, 0, nullptr, true);
  EXPECT_FLOAT_EQ(all.confidence, (0.9f + 0.3f + 0.7f) / 3);
  EXPECT_FLOAT_EQ(chars.confidence, (0.9f + 0.7f) / 2);

  std::vector<uint8_t> ink(64, 0);
  for (int x = 4; x < 14; ++x) ink[x] = 1;   // the first word's ink: columns 4-13
  for (int x = 30; x < 44; ++x) ink[x] = 1;  // the second's: 30-43
  OcrLine boxed;
  boxed.box = {0, 0, 64, 20};
  CtcLine e;
  e.ids = {1, 3, 2};
  e.frames = {2, 5, 9};  // a at crop x 8 (inside its ink), the space at 20, b at 36
  e.probs = {0.9f, 0.9f, 0.9f};
  detail::read_line(boxed, e, 16, crop, vocab, ink.data(), 64, nullptr, false, true);
  ASSERT_EQ(boxed.words.size(), 2u);
  const double per_px = (crop_region(boxed.box, crop).x1 - crop_region(boxed.box, crop).x0) /
                        crop_region(boxed.box, crop).width;
  const double x0 = crop_region(boxed.box, crop).x0;
  EXPECT_NEAR(boxed.words[0].box.x0, std::max(0.0, x0 + 4 * per_px), 1e-9) << "out of the glyph to its blank";
  EXPECT_NEAR(boxed.words[0].box.x1, x0 + 14 * per_px, 1e-9) << "back from the gap to the ink";
}

// A rescorer that keeps each context it is given, in the order asked, and the calls; a shorter reading scores higher.
class ContextLog : public LineRescorer {
 public:
  void score(std::span<const int32_t> context, std::span<const std::vector<int32_t>> readings,
             std::span<double> out) const override {
    contexts.emplace_back(context.begin(), context.end());
    for (size_t i = 0; i < readings.size(); ++i) out[i] = -static_cast<double>(readings[i].size());
  }
  void score_lines(std::span<const RescoreQuery> q, std::span<std::vector<double>> out) const override {
    ++calls;
    LineRescorer::score_lines(q, out);
  }
  int boundary() const override { return 999; }
  mutable std::vector<std::vector<int32_t>> contexts;
  mutable int calls = 0;
};

// detail::rescore_page: each waiting line after the line above it, that line's text in the recognizer's classes by
// the longest class at each place (a code point no class starts with skipped), its last 96 alone where it is longer
// (mid-line, as a training window starts: no boundary before them); a waiting line above another its best reading;
// a line with none above the boundary alone; every line in one call, each read as its best reading after the rescore.
TEST(Engine, RescoresAPagesWaitingLinesAfterTheLineAbove) {
  const std::vector<std::string> vocab = {"", "a", "b", " ", "ch", "c", "h"};
  auto reading = [](std::vector<int32_t> ids, double score) {
    BeamLine r;
    for (size_t i = 0; i < ids.size(); ++i) {
      r.line.frames.push_back(static_cast<int32_t>(2 + 2 * i));
      r.line.probs.push_back(0.9f);
    }
    r.line.ids = std::move(ids);
    r.score = score;
    return r;
  };
  const std::string first = std::string(100, 'a') + "chx\xc3\xa9" "b";  // 100 a, ch, x and é (no class), b
  OcrPage page;
  page.width = 1400;
  page.height = 300;
  page.lines.resize(4);
  page.lines[0].box = {0, 0, 300, 20};
  page.lines[0].text = first;                  // read already
  page.lines[1].box = {0, 30, 300, 50};        // under line 0, 10 px down
  page.lines[2].box = {0, 60, 300, 80};        // under line 1 (line 0 is 40 px up: more than 1.5 of its heights)
  page.lines[3].box = {1000, 200, 1300, 220};  // under none: no line overlaps it across
  std::vector<detail::UnsureLine> waiting(3);
  waiting[0].line = 1;
  waiting[0].nbest = {reading({4, 5, 6, 1}, -1.0), reading({2}, -1.5)};  // "chcha" the beam's; "b" after the rescore
  waiting[1].line = 2;
  waiting[1].nbest = {reading({2, 3, 1}, 0.0)};
  waiting[2].line = 3;
  waiting[2].nbest = {reading({1, 1, 1}, 0.0), reading({1}, -0.5)};
  for (detail::UnsureLine& u : waiting) u.length = 16;
  EngineOptions o;
  o.beam.rescore = 1.0f;
  auto log = std::make_shared<ContextLog>();
  o.rescorer = log;
  detail::rescore_page(page, waiting, o, vocab);

  EXPECT_EQ(log->calls, 1);
  ASSERT_EQ(log->contexts.size(), 3u);
  std::vector<int32_t> cut(94, 1);  // line 0's 102 classes, its last 96: 94 a, ch, b; then the boundary
  cut.insert(cut.end(), {4, 2, 999});
  EXPECT_EQ(log->contexts[0], cut);
  EXPECT_EQ(log->contexts[1], (std::vector<int32_t>{999, 4, 4, 1, 999}))  // "chcha" by the longest: ch, ch, a
      << "line 1's best reading as the beam read it, not as it was rescored";
  EXPECT_EQ(log->contexts[2], (std::vector<int32_t>{999}));
  EXPECT_EQ(page.lines[1].text, "b");  // -1.5 - 1 over -1.0 - 4
  EXPECT_EQ(page.lines[2].text, "b a");
  EXPECT_EQ(page.lines[3].text, "a");  // -0.5 - 1 over 0 - 3
  EXPECT_EQ(page.lines[0].text, first) << "a line read already is left as it was";
}

// The JSON writer's numbers (std::to_chars) as printf writes them on this platform, ties included, and its
// escapes as a byte-by-byte scan finds them.
TEST(Gate, TheWritersNumbersAndEscapes) {
  OcrPage p;
  p.width = p.height = 100;
  const double edges[] = {0.25, 0.75, 12.25, 1.05, 2.5, 1e-5, 3.0, 99.95, 1234.45, 0.125};
  for (double x0 : edges) {
    OcrLine l;
    l.box = {x0, x0 / 2, x0 + 0.35, x0 * 3};
    l.text = "a";
    l.confidence = static_cast<float>(x0 / 2000);
    l.det_score = 0.03125f;  // a tie at the fifth decimal
    p.lines.push_back(l);
  }
  p.skew = -0.125;
  p.ms.total = 2.675;
  const std::string json = p.json();
  for (const OcrLine& l : p.lines) {
    char want[160];
    std::snprintf(want, sizeof want, "{\"box\": [%.1f, %.1f, %.1f, %.1f], \"text\": \"a\", \"confidence\": %.4f, "
                  "\"det_score\": %.4f", l.box.x0, l.box.y0, l.box.x1, l.box.y1, l.confidence, l.det_score);
    EXPECT_NE(json.find(want), std::string::npos) << want << "\n" << json;
  }
  char skew[48], total[48];
  std::snprintf(skew, sizeof skew, "\"skew\": %.2f", p.skew);
  std::snprintf(total, sizeof total, "\"total\": %.2f}}", p.ms.total);
  EXPECT_NE(json.find(skew), std::string::npos) << json;
  EXPECT_NE(json.find(total), std::string::npos) << json;

  std::string s;
  uint32_t r = 99;
  const char pool[] = "ab\"'\\&<>\t\n\x01\x1f e\xc3\xa9";
  for (int i = 0; i < 4000; ++i) {
    r = r * 1664525u + 1013904223u;
    s += pool[(r >> 16) % (sizeof pool - 1)];
  }
  for (size_t i = 0; i <= s.size(); i += 7) {
    size_t want = i;
    while (want < s.size()) {
      const unsigned char c = static_cast<unsigned char>(s[want]);
      if (c < 0x20 || c == '"' || c == '\'' || c == '\\' || c == '&' || c == '<' || c == '>') break;
      ++want;
    }
    ASSERT_EQ(detail::special_from(s, i), want) << i;
  }
  OcrPage q;
  OcrLine l;
  l.text = "say \"hi\" \\ <b>&\x01\t'";
  q.lines.push_back(l);
  const std::string escaped = "\"text\": \"say \\\"hi\\\" \\\\ <b>&\\u0001\\t'\"";
  EXPECT_NE(q.json().find(escaped), std::string::npos) << q.json();
}

// Spaces by the ink (SpaceRule): of a crop of four characters, a and b a pixel apart, b and c 9 apart, c and d a
// pixel apart, the unsure space between a and b dropped, the one between b and c kept, and the sure one between c
// and d kept though the ink does not part them. Letter-spaced, every gap 9 wide, the unsure spaces dropped. A crop
// of no contrast has no columns of ink.
TEST(Gate, SpacesTheInkBearsOut) {
  const CropRule crop;
  const int h = crop.height, w = 64;
  std::vector<uint8_t> c(static_cast<size_t>(h) * w, 255);
  for (auto [x0, x1] : {std::pair{0, 11}, {12, 23}, {32, 43}, {44, 55}}) {
    for (int y = 8; y < 25; ++y) std::fill(c.begin() + y * w + x0, c.begin() + y * w + x1, uint8_t{0});
  }
  std::vector<uint8_t> ink(w);
  ASSERT_TRUE(detail::ink_columns(c.data(), w, h, w, crop, ink.data()));
  EXPECT_EQ(ink[5], 1);
  EXPECT_EQ(ink[11], 0);
  EXPECT_EQ(ink[27], 0);
  const std::vector<std::string> vocab = {"", "a", "b", "c", "d", " "};
  CtcLine d;
  d.ids = {1, 5, 2, 5, 3, 5, 4};
  d.frames = {1, 3, 4, 7, 9, 11, 12};
  d.probs = {0.99f, 0.5f, 0.99f, 0.5f, 0.99f, 0.9f, 0.99f};
  const SpaceRule rule;
  OcrLine by_ink;
  by_ink.box = {100, 100, 300, 130};
  OcrLine as_read = by_ink;
  detail::read_line(by_ink, d, w / crop.frame_px, crop, vocab, ink.data(), w, &rule);
  detail::read_line(as_read, d, w / crop.frame_px, crop, vocab);
  EXPECT_EQ(by_ink.text, "ab c d");
  ASSERT_EQ(by_ink.words.size(), 3u);
  EXPECT_EQ(by_ink.words[0].text, "ab");
  EXPECT_EQ(as_read.text, "a b c d");
  std::fill(c.begin(), c.end(), uint8_t{255});  // letter-spaced: a, b, c 9 apart
  for (auto [x0, x1] : {std::pair{0, 11}, {20, 31}, {40, 51}}) {
    for (int y = 8; y < 25; ++y) std::fill(c.begin() + y * w + x0, c.begin() + y * w + x1, uint8_t{0});
  }
  ASSERT_TRUE(detail::ink_columns(c.data(), w, h, w, crop, ink.data()));
  CtcLine spaced;
  spaced.ids = {1, 5, 2, 5, 3};
  spaced.frames = {1, 4, 6, 9, 11};
  spaced.probs = {0.99f, 0.5f, 0.99f, 0.5f, 0.99f};
  OcrLine heading;
  heading.box = by_ink.box;
  detail::read_line(heading, spaced, w / crop.frame_px, crop, vocab, ink.data(), w, &rule);
  EXPECT_EQ(heading.text, "abc");
  const std::vector<uint8_t> flat(static_cast<size_t>(h) * w, 200);
  EXPECT_FALSE(detail::ink_columns(flat.data(), w, h, w, crop, ink.data()));
}

}  // namespace
}  // namespace carbon
