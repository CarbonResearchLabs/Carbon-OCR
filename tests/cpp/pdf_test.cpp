// PDF documents through PDFium (carbon/pdf.h), on tests/data/text.pdf
// (tools/make_test_pdfs.py): "Hello PDF" and a black bar on a 300 x 150 pt page,
// and the same page shown turned a quarter (/Rotate 90).

#include "carbon/pdf.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "carbon/error.h"
#include "carbon/output.h"
#include "carbon/textlayer.h"

namespace carbon {
namespace {

std::filesystem::path data(const char* name) { return std::filesystem::path(CARBON_TEST_DATA) / name; }

// The darkest pixel inside a box.
int darkest(const GrayImage& img, float x0, float y0, float x1, float y1) {
  int lo = 255;
  for (int y = std::max(0, static_cast<int>(y0)); y < std::min(img.height, static_cast<int>(std::ceil(y1))); ++y) {
    for (int x = std::max(0, static_cast<int>(x0)); x < std::min(img.width, static_cast<int>(std::ceil(x1))); ++x) {
      lo = std::min<int>(lo, img.pixels[static_cast<size_t>(y) * img.width + x]);
    }
  }
  return lo;
}

TEST(Pdf, ReadsPagesAsShown) {
  PdfDocument doc(data("text.pdf"));
  ASSERT_EQ(doc.pages(), 2);
  EXPECT_DOUBLE_EQ(doc.page_size(0).width, 300);
  EXPECT_DOUBLE_EQ(doc.page_size(0).height, 150);
  EXPECT_DOUBLE_EQ(doc.page_size(1).width, 150);  // turned a quarter
  EXPECT_DOUBLE_EQ(doc.page_size(1).height, 300);
  const GrayImage img = doc.render(0, 2.0);
  ASSERT_EQ(img.width, 600);
  ASSERT_EQ(img.height, 300);
  EXPECT_EQ(darkest(img, 0, 0, 30, 30), 255) << "paper is white";
  // the bar: x 200..260, y 100..130 in PDF space (origin at the bottom) is
  // x 400..520, y 40..100 in pixels at 2 px a point
  EXPECT_LT(darkest(img, 410, 50, 510, 90), 10);
  EXPECT_EQ(img.pixels[static_cast<size_t>(70) * img.width + 460], 0);
  const GrayImage turned = doc.render(1, 2.0);
  ASSERT_EQ(turned.width, 300);
  ASSERT_EQ(turned.height, 600);
  // turned clockwise: the upright page's pixel (X, Y) shows at (150 - Y, X) points, and
  // the upright page's (X, Y) is PDF (x, 150 - y), so PDF (x, y) shows at (y, x): the bar
  // is x 100..130, y 200..260 points, x 200..260, y 400..520 pixels
  EXPECT_EQ(turned.pixels[static_cast<size_t>(460) * turned.width + 230], 0);
  EXPECT_EQ(turned.pixels[static_cast<size_t>(460) * turned.width + 70], 255);
  EXPECT_TRUE(doc.pictures(0, 2.0).empty()) << "the bar is a path, not a picture";
}

TEST(Pdf, ReadsTheTextLayerOverItsInk) {
  PdfDocument doc(data("text.pdf"));
  for (int page = 0; page < 2; ++page) {
    SCOPED_TRACE("page " + std::to_string(page));
    const GrayImage img = doc.render(page, 2.0);
    const std::vector<PdfChar> chars = doc.chars(page, 2.0);
    std::u32string text;
    for (const PdfChar& c : chars) {
      if (c.generated) continue;
      text += c.code;
      EXPECT_FALSE(c.bad_map);
      EXPECT_FALSE(c.invisible);
      EXPECT_NEAR(c.size, 24.0f, 0.01f);
      // the loose box holds the tight one
      EXPECT_LE(c.lx0, c.x0 + 0.5f);
      EXPECT_LE(c.ly0, c.y0 + 0.5f);
      EXPECT_GE(c.lx1, c.x1 - 0.5f);
      EXPECT_GE(c.ly1, c.y1 - 0.5f);
      EXPECT_GE(c.x0, 0);
      EXPECT_GE(c.y0, 0);
      EXPECT_LE(c.x1, img.width);
      EXPECT_LE(c.y1, img.height);
      EXPECT_LT(c.x0, c.x1);
      EXPECT_LT(c.y0, c.y1);
      if (c.code != U' ') EXPECT_LT(darkest(img, c.x0, c.y0, c.x1, c.y1), 100) << "no ink under " << char(c.code);
    }
    EXPECT_EQ(text, U"Hello PDF");
    // in reading order: along x on the upright page, along y on the turned one
    const PdfChar &h = chars.front(), &f = chars.back();
    if (page == 0) {
      EXPECT_LT(h.x0, f.x0);
      EXPECT_NEAR(h.x0, 40, 4);  // the text starts at x = 20 pt
    } else {
      EXPECT_LT(h.y0, f.y0);
    }
  }
}

// figure.pdf's picture: its image object's box, (180, 30) to (280, 110) in PDF space,
// x 360..560, y 80..240 pixels at 2 px a point.
TEST(Pdf, FindsPictures) {
  PdfDocument doc(data("figure.pdf"));
  const std::vector<LineBox> pictures = doc.pictures(0, 2.0);
  ASSERT_EQ(pictures.size(), 1u);
  EXPECT_NEAR(pictures[0].x0, 360, 0.5);
  EXPECT_NEAR(pictures[0].y0, 80, 0.5);
  EXPECT_NEAR(pictures[0].x1, 560, 0.5);
  EXPECT_NEAR(pictures[0].y1, 240, 0.5);
}

TEST(Pdf, RefusesWhatIsNotAPdf) {
  EXPECT_THROW(PdfDocument(data("no-such.pdf")), Error);
  EXPECT_THROW(PdfDocument(data("colsum_argmax.onnx")), Error);
  PdfDocument doc(data("text.pdf"));
  EXPECT_THROW(doc.render(2, 1.0), Error);
  EXPECT_THROW(doc.chars(-1, 1.0), Error);
}

// A searchable PDF (carbon/output.h) read back by PDFium: the page its image,
// the size the dpi says, a picture over the whole page, the words its text,
// invisible, each in its box; a page orientation turned, upright.
TEST(Pdf, ReadsBackTheSearchablePdfItWrites) {
  GrayImage img;
  img.width = 400;
  img.height = 200;
  img.pixels.assign(400 * 200, 255);
  for (int y = 30; y < 50; ++y) {
    for (int x = 20; x < 180; x += 3) img.pixels[static_cast<size_t>(y) * 400 + x] = 30;  // "ink"
  }
  OcrPage page;
  page.width = 400;
  page.height = 200;
  OcrLine line;
  line.box = {20, 30, 180, 50};
  line.text = "Hello wörld";
  line.confidence = 0.9f;
  line.words = {{{20, 30, 80, 50}, "Hello", 0.9f}, {{100, 30, 180, 50}, "wörld", 0.9f}};
  OcrLine second;
  second.box = {20, 120, 300, 150};
  second.text = "Second";  // no words: written as one
  page.lines = {line, second};
  const std::vector<OcrPage> pages = {page};
  const std::vector<GrayView> views = {img.view()};
  const std::string pdf = to_pdf(pages, views, 144);
  const auto path = std::filesystem::temp_directory_path() / "carbon-searchable.pdf";
  std::ofstream(path, std::ios::binary).write(pdf.data(), static_cast<std::streamsize>(pdf.size()));
  {
    PdfDocument doc(path);
    ASSERT_EQ(doc.pages(), 1);
    EXPECT_DOUBLE_EQ(doc.page_size(0).width, 200);  // 400 px at 144 dpi
    EXPECT_DOUBLE_EQ(doc.page_size(0).height, 100);
    const GrayImage back = doc.render(0, 2.0);  // the image's own pixels
    ASSERT_EQ(back.width, 400);
    ASSERT_EQ(back.height, 200);
    int worst = 0;
    for (size_t i = 0; i < back.pixels.size(); ++i) worst = std::max(worst, std::abs(back.pixels[i] - img.pixels[i]));
    EXPECT_LE(worst, 2) << "the image under the text, and nothing of the text drawn";
    const std::vector<LineBox> pictures = doc.pictures(0, 2.0);
    ASSERT_EQ(pictures.size(), 1u);
    EXPECT_NEAR(pictures[0].x0, 0, 0.5);
    EXPECT_NEAR(pictures[0].y0, 0, 0.5);
    EXPECT_NEAR(pictures[0].x1, 400, 0.5);
    EXPECT_NEAR(pictures[0].y1, 200, 0.5);
#if defined(CARBON_PDF_HELPER)
    PdfPool pool(1, CARBON_PDF_HELPER);
    const std::vector<LineBox> through = pool.pictures(path, 0, 2.0);
    ASSERT_EQ(through.size(), 1u);
    EXPECT_EQ(through[0].x0, pictures[0].x0);
    EXPECT_EQ(through[0].y1, pictures[0].y1);
#endif
    std::u32string text;
    for (const PdfChar& c : doc.chars(0, 2.0)) {
      if (c.generated) {
        if (c.code == U' ' && !text.empty() && text.back() != U' ') text += U' ';
        continue;
      }
      EXPECT_TRUE(c.invisible);
      text += c.code;
      // inside its line's box, give or take a pixel
      const LineBox& box = c.ly1 < 100 ? line.box : second.box;
      EXPECT_GE(c.lx0, box.x0 - 1);
      EXPECT_LE(c.lx1, box.x1 + 1);
      EXPECT_GE(c.ly0, box.y0 - 1);
      EXPECT_LE(c.ly1, box.y1 + 1);
    }
    EXPECT_NE(text.find(U"Hello"), std::u32string::npos);
    EXPECT_NE(text.find(U"w\u00f6rld"), std::u32string::npos);
    EXPECT_NE(text.find(U"Second"), std::u32string::npos);
  }
  // the text layer alone (OCRmyPDF grafts it onto the page): white, its words still there
  const std::string text_only = to_pdf(pages, views, 144, true);
  std::ofstream(path, std::ios::binary).write(text_only.data(), static_cast<std::streamsize>(text_only.size()));
  {
    PdfDocument doc(path);
    EXPECT_DOUBLE_EQ(doc.page_size(0).width, 200);
    const GrayImage back = doc.render(0, 1.0);
    EXPECT_EQ(*std::min_element(back.pixels.begin(), back.pixels.end()), 255) << "no image, no ink";
    EXPECT_TRUE(doc.pictures(0, 1.0).empty());
    std::u32string text;
    for (const PdfChar& c : doc.chars(0, 1.0)) text += c.code;
    EXPECT_NE(text.find(U"Hello"), std::u32string::npos);
  }
  // turned: the image embedded upright, as the boxes are
  std::vector<OcrPage> turned_pages = pages;
  turned_pages[0].rotation = 90;
  const std::string turned = to_pdf(turned_pages, views, 72);
  std::ofstream(path, std::ios::binary).write(turned.data(), static_cast<std::streamsize>(turned.size()));
  {
    PdfDocument doc(path);
    EXPECT_DOUBLE_EQ(doc.page_size(0).width, 200);  // 200 x 400 upright, at 72 dpi
    EXPECT_DOUBLE_EQ(doc.page_size(0).height, 400);
    const std::vector<LineBox> pictures = doc.pictures(0, 1.0);
    ASSERT_EQ(pictures.size(), 1u);
    EXPECT_NEAR(pictures[0].x1, 200, 0.5);
    EXPECT_NEAR(pictures[0].y1, 400, 0.5);
  }
  std::filesystem::remove(path);
}

#if defined(CARBON_PDF_HELPER)
// Pages in helper processes: the bytes of the process's own render, from many
// threads at once; a file that is not a PDF an error, not a crash; a helper that
// dies replaced, and the pool still working.
TEST(PdfPool, RendersInHelperProcesses) {
  PdfPool pool(2, CARBON_PDF_HELPER);
  ASSERT_EQ(pool.processes(), 2);
  EXPECT_EQ(pool.pages(data("text.pdf")), 2);
  PdfDocument doc(data("text.pdf"));
  const GrayImage want[2] = {doc.render(0, 1.5), doc.render(1, 1.5)};
  std::vector<std::thread> threads;
  std::atomic<int> same{0};
  for (int t = 0; t < 4; ++t) {
    threads.emplace_back([&, t] {
      for (int k = 0; k < 6; ++k) {
        const int page = (t + k) % 2;
        const GrayImage got = pool.render(data("text.pdf"), page, 1.5);
        if (got.width == want[page].width && got.height == want[page].height && got.pixels == want[page].pixels) ++same;
      }
    });
  }
  for (auto& th : threads) th.join();
  EXPECT_EQ(same.load(), 24);
  EXPECT_THROW(pool.render(data("colsum_argmax.onnx"), 0, 1.0), Error);
  EXPECT_THROW(pool.render(data("text.pdf"), 5, 1.0), Error);
  try {
    pool.render(data("text.pdf"), -1, 1.0);  // the helper exits, as if it crashed
    FAIL() << "rendered";
  } catch (const Error& e) {
    EXPECT_NE(std::string(e.what()).find("died"), std::string::npos) << e.what();
  }
  for (int k = 0; k < 4; ++k) EXPECT_EQ(pool.render(data("text.pdf"), k % 2, 1.5).pixels, want[k % 2].pixels);
  EXPECT_THROW(PdfPool(1, data("no-such-helper.exe")), Error);
}

// The pixels through each helper's shared region, through the pipe where a page is
// larger than the region, and with no region: the same bytes as the process's own.
TEST(PdfPool, RendersThroughSharedMemoryOrThePipe) {
  PdfDocument doc(data("text.pdf"));
  const GrayImage want = doc.render(1, 2.0);
  for (size_t bytes : {PdfPool::kSharedBytes, want.pixels.size(), want.pixels.size() - 1, size_t{0}}) {
    PdfPool pool(1, CARBON_PDF_HELPER, bytes);
    for (int k = 0; k < 3; ++k) {
      const GrayImage got = pool.render(data("text.pdf"), 1, 2.0);
      EXPECT_EQ(got.width, want.width) << bytes;
      EXPECT_EQ(got.height, want.height) << bytes;
      EXPECT_TRUE(got.pixels == want.pixels) << bytes << " bytes, render " << k;
      EXPECT_EQ(pool.render(data("text.pdf"), 0, 1.0).pixels, doc.render(0, 1.0).pixels) << bytes;
    }
    try {
      pool.render(data("text.pdf"), -1, 1.0);  // the helper dies; its replacement maps the region again
    } catch (const Error&) {
    }
    EXPECT_TRUE(pool.render(data("text.pdf"), 1, 2.0).pixels == want.pixels) << bytes << " bytes, after a restart";
  }
}

// The text layer through the helpers: their characters are the process's own,
// and make the page's one line, trusted, upright and turned.
TEST(PdfPool, ReadsTextLayers) {
  PdfPool pool(1, CARBON_PDF_HELPER);
  PdfDocument doc(data("text.pdf"));
  for (int page = 0; page < 2; ++page) {
    SCOPED_TRACE("page " + std::to_string(page));
    const std::vector<PdfChar> got = pool.chars(data("text.pdf"), page, 2.0), want = doc.chars(page, 2.0);
    ASSERT_EQ(got.size(), want.size());
    for (size_t i = 0; i < got.size(); ++i) {
      EXPECT_EQ(got[i].code, want[i].code);
      EXPECT_EQ(got[i].lx0, want[i].lx0);
      EXPECT_EQ(got[i].y1, want[i].y1);
      EXPECT_EQ(got[i].generated, want[i].generated);
    }
    const GrayImage img = pool.render(data("text.pdf"), page, 2.0);
    const TextLayer t = read_text_layer(got, img.view());
    EXPECT_TRUE(t.check.usable) << t.check.reason;
    ASSERT_EQ(t.lines.size(), 1u);
    EXPECT_EQ(t.lines[0].text, "Hello PDF");
  }
}
#endif

}  // namespace
}  // namespace carbon
