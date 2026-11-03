// A born-digital page's text layer (carbon/textlayer.h): characters into lines,
// and the checks, on characters made here.

#include "carbon/textlayer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "carbon/deskew.h"
#include "carbon/image.h"

namespace carbon {
namespace {

// A line of characters 10 px wide and 20 px high (loose), the ink 6 x 12 inside,
// from x, with the baseline of the row at y; spaces drawn as spaces.
void put_line(std::vector<PdfChar>& out, const std::u32string& text, float x, float y) {
  for (char32_t c : text) {
    PdfChar p;
    p.code = c;
    p.lx0 = x;
    p.lx1 = x + 10;
    p.ly0 = y - 16;
    p.ly1 = y + 4;
    p.x0 = x + 2;
    p.x1 = x + 8;
    p.y0 = y - 12;
    p.y1 = y;
    p.size = 12;
    out.push_back(p);
    x += 10;
  }
}

PdfChar generated(char32_t c) {
  PdfChar p;
  p.code = c;
  p.generated = true;
  return p;
}

// A page with ink under every drawn character but spaces.
GrayImage inked(const std::vector<PdfChar>& chars, int w, int h) {
  GrayImage img;
  img.width = w;
  img.height = h;
  img.pixels.assign(static_cast<size_t>(w) * h, 255);
  for (const PdfChar& c : chars) {
    if (c.generated || c.code == U' ') continue;
    img.pixels[static_cast<size_t>((c.y0 + c.y1) / 2) * w + static_cast<size_t>((c.x0 + c.x1) / 2)] = 0;
  }
  return img;
}

TEST(TextLayer, MakesLinesOfItsCharacters) {
  std::vector<PdfChar> chars;
  put_line(chars, U"Hi there", 10, 40);
  chars.push_back(generated(U'\r'));
  chars.push_back(generated(U'\n'));
  put_line(chars, U"Line", 10, 80);
  chars.push_back(generated(U' '));  // PDFium's space between words it sees apart
  put_line(chars, U"two", 70, 80);
  put_line(chars, U"aside", 300, 200);  // no line break, but off the line's band: a line of its own
  const GrayImage page = inked(chars, 400, 240);
  const TextLayer t = read_text_layer(chars, page.view());
  EXPECT_TRUE(t.check.usable) << t.check.reason;
  EXPECT_EQ(t.check.chars, 20);
  ASSERT_EQ(t.lines.size(), 3u);
  EXPECT_EQ(t.lines[0].text, "Hi there");
  EXPECT_EQ(t.lines[1].text, "Line two");
  EXPECT_EQ(t.lines[2].text, "aside");
  EXPECT_DOUBLE_EQ(t.lines[0].box.x0, 10);
  EXPECT_DOUBLE_EQ(t.lines[0].box.x1, 90);
  EXPECT_DOUBLE_EQ(t.lines[0].box.y0, 24);
  EXPECT_DOUBLE_EQ(t.lines[0].box.y1, 44);
  EXPECT_DOUBLE_EQ(t.lines[1].box.x1, 100);
  EXPECT_FLOAT_EQ(t.lines[0].confidence, 1.0f);
  // words, at drawn spaces and PDFium's own, their boxes the characters'
  ASSERT_EQ(t.lines[0].words.size(), 2u);
  EXPECT_EQ(t.lines[0].words[0].text, "Hi");
  EXPECT_EQ(t.lines[0].words[1].text, "there");
  EXPECT_DOUBLE_EQ(t.lines[0].words[0].box.x0, 10);
  EXPECT_DOUBLE_EQ(t.lines[0].words[0].box.x1, 30);
  EXPECT_DOUBLE_EQ(t.lines[0].words[1].box.x0, 40);
  ASSERT_EQ(t.lines[1].words.size(), 2u);
  EXPECT_EQ(t.lines[1].words[1].text, "two");
  EXPECT_DOUBLE_EQ(t.lines[1].words[1].box.x0, 70);
}

TEST(TextLayer, ChecksWhatItWouldTrust) {
  std::vector<PdfChar> good;
  put_line(good, U"A sentence of plain text", 0, 20);
  const GrayImage page = inked(good, 300, 40);
  EXPECT_TRUE(read_text_layer(good, page.view()).check.usable);

  auto with = [&](auto change) {
    std::vector<PdfChar> c = good;
    change(c);
    return read_text_layer(c, page.view()).check;
  };
  const TextLayerCheck bad = with([](auto& c) { c[3].code = 0xFFFD; });
  EXPECT_FALSE(bad.usable);
  EXPECT_EQ(bad.bad, 1);
  EXPECT_NE(bad.reason.find("not readable"), std::string::npos) << bad.reason;
  const TextLayerCheck hidden = with([](auto& c) {
    for (auto& p : c) p.invisible = true;
  });
  EXPECT_FALSE(hidden.usable);
  EXPECT_NE(hidden.reason.find("invisible"), std::string::npos) << hidden.reason;
  const GrayImage blank = inked({}, 300, 40);
  const TextLayerCheck no_ink = read_text_layer(good, blank.view()).check;
  EXPECT_FALSE(no_ink.usable);
  EXPECT_NE(no_ink.reason.find("no ink"), std::string::npos) << no_ink.reason;
  EXPECT_TRUE(read_text_layer(good, GrayView{nullptr, 0, 0, 0}).check.usable);  // no page: no ink check
  std::vector<PdfChar> few;
  put_line(few, U"ok", 0, 20);
  EXPECT_FALSE(read_text_layer(few, page.view()).check.usable);
}

// A table row the layer reads as one line, split into its cells.
TEST(TextLayer, APageSplitsItsLinesAtWideGaps) {
  std::vector<PdfChar> chars;
  put_line(chars, U"Name ", 10, 50);
  put_line(chars, U"Age ", 200, 50);  // the gap from "Name": 140 px, 7 line heights
  put_line(chars, U"City", 300, 50);
  const TextLayer layer = read_text_layer(chars, GrayView{nullptr, 0, 0, 0});
  ASSERT_EQ(layer.lines.size(), 1u);
  const OcrPage page = page_from_text_layer(layer, 400, 100);
  ASSERT_EQ(page.lines.size(), 3u);
  EXPECT_EQ(page.lines[0].text, "Name");
  EXPECT_EQ(page.lines[1].text, "Age");
  EXPECT_EQ(page.lines[2].text, "City");
  EXPECT_EQ(page.lines[1].box.x0, 200);
}

// The bounds of an image's dark pixels.
LineBox dark_bounds(const GrayImage& img) {
  LineBox b{1e9, 1e9, -1, -1};
  for (int y = 0; y < img.height; ++y) {
    for (int x = 0; x < img.width; ++x) {
      if (img.pixels[static_cast<size_t>(y) * img.width + x] >= 128) continue;
      b = {std::min<double>(b.x0, x), std::min<double>(b.y0, y), std::max<double>(b.x1, x + 1),
           std::max<double>(b.y1, y + 1)};
    }
  }
  return b;
}

// Hybrid pages: the pictures worth reading, the page with only them, and the lines
// OCR finds there taken back to the page from the frame the engine read it in
// (turned upright as orientation turns it, levelled as deskew does), the layer's
// own lines winning where both have one.
TEST(TextLayer, HybridPagesAddTheirPicturesLines) {
  const std::vector<LineBox> pictures = {{-5, 50, 120, 150}, {300, 10, 320, 300}, {200, 150, 260, 230}};
  const std::vector<LineBox> read = pictures_to_read(pictures, 400, 200);
  ASSERT_EQ(read.size(), 2u) << "the narrow one left out";
  EXPECT_EQ(read[0].x0, 0);  // clipped to the page
  EXPECT_EQ(read[1].y1, 200);

  GrayImage page;
  page.width = 400;
  page.height = 200;
  page.pixels.assign(400 * 200, 0);
  const GrayImage only = pictures_only(page.view(), read);
  EXPECT_EQ(only.pixels[static_cast<size_t>(100) * 400 + 60], 0);    // in the first picture
  EXPECT_EQ(only.pixels[static_cast<size_t>(180) * 400 + 230], 0);   // in the second
  EXPECT_EQ(only.pixels[static_cast<size_t>(20) * 400 + 60], 255);   // above the first: paper
  EXPECT_EQ(only.pixels[static_cast<size_t>(100) * 400 + 150], 255);

  // the frames: a mark on the page as given, found where the engine reads it, taken back
  GrayImage given;
  given.width = 400;
  given.height = 200;
  given.pixels.assign(400 * 200, 255);
  const LineBox mark{300, 20, 380, 50};
  for (int y = 20; y < 50; ++y) std::fill_n(given.pixels.begin() + y * 400 + 300, 80, 0);
  for (int rotation : {0, 90, 180, 270}) {
    for (double skew : {0.0, 3.0}) {
      SCOPED_TRACE("rotation " + std::to_string(rotation) + ", skew " + std::to_string(skew));
      GrayImage view = rotate_quarter(given.view(), 4 - rotation / 90);  // as the engine turns it
      if (skew != 0) view = rotate_page(view.view(), skew);
      OcrPage r;
      r.width = 400;
      r.height = 200;
      r.rotation = rotation;
      r.skew = skew;
      OcrLine l;
      l.box = dark_bounds(view);
      l.text = "aaa";
      l.words = {{l.box, "aaa", 1.0f}};
      r.lines = {l};
      OcrPage p;
      p.width = 400;
      p.height = 200;
      add_picture_lines(p, r);
      ASSERT_EQ(p.lines.size(), 1u);
      EXPECT_EQ(p.source, "hybrid");
      const LineBox& b = p.lines[0].box;
      const double slack = skew == 0 ? 0.01 : 6;  // a levelled box's bounds turned back hold more than the mark
      EXPECT_LE(b.x0, mark.x0 + 1);
      EXPECT_GE(b.x0, mark.x0 - slack);
      EXPECT_LE(b.y0, mark.y0 + 1);
      EXPECT_GE(b.y0, mark.y0 - slack);
      EXPECT_GE(b.x1, mark.x1 - 1);
      EXPECT_LE(b.x1, mark.x1 + slack);
      EXPECT_GE(b.y1, mark.y1 - 1);
      EXPECT_LE(b.y1, mark.y1 + slack);
      EXPECT_EQ(p.lines[0].words[0].box.x0, b.x0);
    }
  }

  // the layer's lines win: a line found under one is left out, as is one with no text
  OcrPage p;
  p.width = 400;
  p.height = 200;
  p.source = "text layer";
  OcrLine own;
  own.box = {20, 120, 200, 140};
  own.text = "Caption";
  own.confidence = 1;
  p.lines = {own};
  OcrPage r;
  r.width = 400;
  r.height = 200;
  OcrLine under, inside, empty;
  under.box = {22, 121, 190, 139};
  under.text = "Captlon";
  inside.box = {210, 30, 330, 50};
  inside.text = "aaa";
  empty.box = {210, 60, 330, 80};
  r.lines = {under, inside, empty};
  add_picture_lines(p, r);
  ASSERT_EQ(p.lines.size(), 2u);
  EXPECT_EQ(p.lines[0].text, "aaa");  // in reading order: it is above the caption
  EXPECT_EQ(p.lines[1].text, "Caption");
  EXPECT_EQ(p.source, "hybrid");
  OcrPage none;
  none.source = "text layer";
  none.lines = {own};
  r.lines = {under};
  add_picture_lines(none, r);
  EXPECT_EQ(none.source, "text layer") << "nothing added";
}

}  // namespace
}  // namespace carbon
