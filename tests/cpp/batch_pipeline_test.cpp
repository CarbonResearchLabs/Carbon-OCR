// carbon::Pipeline (carbon/pipeline.h) on the CPU device, on tests/data/fake-set: page
// for page the CPU engine's results, its models batched across the pages in flight,
// the results in the order given.
#include "carbon/pipeline.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

#include "carbon/deskew.h"
#include "carbon/error.h"
#include "carbon/image.h"

namespace carbon {
namespace {

const std::filesystem::path kSet = std::filesystem::path(CARBON_TEST_DATA) / "fake-set";

PipelineOptions cpu_options(int page_batch = 4) {
  PipelineOptions o;
  o.device = Device::kCpu;
  o.page_batch = page_batch;
  o.threads = 3;
  o.model_threads = 2;
  return o;
}

GrayImage paper(int w, int h) { return {w, h, std::vector<uint8_t>(static_cast<size_t>(w) * h, 255)}; }

void expect_same(const OcrPage& a, const OcrPage& b, int page) {
  EXPECT_EQ(a.width, b.width) << "page " << page;
  EXPECT_EQ(a.height, b.height) << "page " << page;
  EXPECT_EQ(a.rotation, b.rotation) << "page " << page;
  EXPECT_DOUBLE_EQ(a.skew, b.skew) << "page " << page;
  EXPECT_EQ(a.order, b.order) << "page " << page;
  EXPECT_EQ(a.second_pass, b.second_pass) << "page " << page;
  ASSERT_EQ(a.lines.size(), b.lines.size()) << "page " << page;
  for (size_t i = 0; i < a.lines.size(); ++i) {
    EXPECT_EQ(a.lines[i].text, b.lines[i].text) << "page " << page << " line " << i;
    EXPECT_DOUBLE_EQ(a.lines[i].box.x0, b.lines[i].box.x0);
    EXPECT_DOUBLE_EQ(a.lines[i].box.y0, b.lines[i].box.y0);
    EXPECT_DOUBLE_EQ(a.lines[i].box.x1, b.lines[i].box.x1);
    EXPECT_DOUBLE_EQ(a.lines[i].box.y1, b.lines[i].box.y1);
    EXPECT_NEAR(a.lines[i].confidence, b.lines[i].confidence, 1e-6);
    EXPECT_EQ(a.lines[i].words.size(), b.lines[i].words.size());
  }
  EXPECT_EQ(a.pictures.size(), b.pictures.size()) << "page " << page;
}

// A page of tiny text at the detector's scale: bars 5 px high, every other one
// starting on an odd row, which the fake detector finds only at twice the scale
// (engine_test.cpp's tiny_page).
GrayImage tiny_page() {
  GrayImage p = paper(800, 1024);
  const int dashes[] = {3, 4, 2, 3, 4, 2};
  for (int i = 0; i < 6; ++i) {
    const int y = 200 + 41 * i;
    for (int d = 0; d < dashes[i]; ++d) {
      for (int r = y; r < y + 5; ++r) std::fill_n(p.pixels.begin() + r * 800 + 100 + 32 * d, 24, uint8_t{20});
    }
  }
  return p;
}

// Pages of three sizes (the fake page, a blank one, the page skewed) and a page of
// tiny text, which takes the second detection pass, through the pipeline and
// through the CPU engine: the same results.
TEST(BatchPipeline, ReadsPagesAsTheEngineDoes) {
  const GrayImage page = load_image(kSet / "page.png");
  const GrayImage skewed = rotate_page(page.view(), 2.5);
  const GrayImage blank = paper(400, 300);
  const GrayImage tiny = tiny_page();
  const GrayView kinds[] = {page.view(), blank.view(), skewed.view(), tiny.view()};
  std::vector<GrayView> pages;
  for (int i = 0; i < 10; ++i) pages.push_back(kinds[i % 4]);
  EngineOptions eo;
  eo.pool.workers = 2;
  const std::vector<OcrPage> want = Engine(kSet, eo).ocr(pages);
  Pipeline pipe(kSet, cpu_options());
  EXPECT_EQ(pipe.device(), Device::kCpu);
  const std::vector<OcrPage> got = pipe.ocr(pages);
  ASSERT_EQ(got.size(), pages.size());
  for (size_t i = 0; i < got.size(); ++i) expect_same(got[i], want[i], static_cast<int>(i));
  EXPECT_EQ(got[0].text(), "aaa\naaaaa\naa\n");
  EXPECT_NEAR(got[2].skew, -2.5, 0.15);
  EXPECT_GT(got[0].ms.total, 0.0);
  EXPECT_FALSE(got[0].second_pass);
  EXPECT_TRUE(got[3].second_pass);
  EXPECT_EQ(got[3].lines.size(), 6u);
  EXPECT_GT(got[3].ms.second_pass, 0.0);
}

// Many pages in flight: a detection run holds several pages of a size, a
// recognition run the lines of several pages.
TEST(BatchPipeline, BatchesAcrossThePagesInFlight) {
  const GrayImage page = load_image(kSet / "page.png");
  PipelineOptions o = cpu_options(4);
  o.in_flight = 16;
  Pipeline pipe(kSet, o);
  // given before any is taken out, so they wait for each other in the stages
  for (int i = 0; i < 16; ++i) pipe.submit(GrayImage{page});
  pipe.close();
  int n = 0;
  while (auto p = pipe.next()) {
    EXPECT_EQ(p->text(), "aaa\naaaaa\naa\n") << "page " << n;
    ++n;
  }
  EXPECT_EQ(n, 16);
  const PipelineStats s = pipe.stats();
  EXPECT_EQ(s.pages, 16);
  EXPECT_EQ(s.lines, 48);
  EXPECT_GE(s.detect_runs, 4);  // page_batch 4: at least 16 / 4
  EXPECT_LE(s.detect_runs, 16);
  EXPECT_GE(s.orient_runs, 4);
  EXPECT_GE(s.recognize_runs, 1);
  EXPECT_GE(s.pages_per_detect_run(), 1.0);
  EXPECT_GE(s.lines_per_recognize_run(), 1.0);
}

// The results come out in the order given, whatever order the stages finish them in,
// and submit waits for room past in_flight.
TEST(BatchPipeline, ResultsInTheOrderGiven) {
  const GrayImage page = load_image(kSet / "page.png");
  PipelineOptions o = cpu_options(2);
  o.in_flight = 2;
  Pipeline pipe(kSet, o);
  std::vector<size_t> numbers;
  std::vector<OcrPage> out;
  for (int i = 0; i < 7; ++i) {
    numbers.push_back(i % 2 ? pipe.submit(paper(300 + 10 * i, 200)) : pipe.submit(GrayImage{page}));
    if (i >= 2) out.push_back(*pipe.next());  // two in flight at most: take one out to give the next
  }
  pipe.close();
  while (auto p = pipe.next()) out.push_back(std::move(*p));
  ASSERT_EQ(out.size(), 7u);
  for (int i = 0; i < 7; ++i) {
    EXPECT_EQ(numbers[i], static_cast<size_t>(i));
    EXPECT_EQ(out[i].width, i % 2 ? 300 + 10 * i : 800) << "page " << i;
    EXPECT_EQ(out[i].lines.size(), i % 2 ? 0u : 3u) << "page " << i;
  }
  EXPECT_FALSE(pipe.next().has_value());
  EXPECT_THROW(pipe.submit(GrayImage{page}), Error);  // closed
}

TEST(BatchPipeline, AnEmptyPageIsAnError) {
  Pipeline pipe(kSet, cpu_options());
  EXPECT_THROW(pipe.submit(GrayImage{}), Error);
  const GrayView none{};
  EXPECT_THROW(pipe.ocr(std::span<const GrayView>(&none, 1)), Error);
  PipelineOptions bad = cpu_options();
  bad.page_batch = 0;
  EXPECT_THROW(Pipeline(kSet, bad), Error);
  EXPECT_THROW(Pipeline(std::filesystem::path(CARBON_TEST_DATA) / "no-such-set", cpu_options()), Error);
}

}  // namespace
}  // namespace carbon
