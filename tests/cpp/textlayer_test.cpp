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

// Ink by contrast: gray text (#AAAAAA, luma 170) and light blue (164) on white pass; white on white, and a
// box of one tone however dark, does not.
TEST(TextLayer, InkIsContrastNotALevel) {
  std::vector<PdfChar> good;
  put_line(good, U"A sentence of plain text", 0, 20);
  for (uint8_t tone : {uint8_t{170}, uint8_t{164}}) {
    GrayImage page = inked(good, 300, 40);
    for (uint8_t& v : page.pixels) v = v == 0 ? tone : v;
    EXPECT_TRUE(read_text_layer(good, page.view()).check.usable) << int{tone};
  }
  GrayImage flat = inked({}, 300, 40);
  for (uint8_t& v : flat.pixels) v = 40;  // dark, but one tone: nothing drawn there
  EXPECT_FALSE(read_text_layer(good, flat.view()).check.usable);
}

// Coverage: ink outside the layer's characters and the pictures, two line heights tall or more, is found (a
// chart's labels drawn as paths); ink under the characters, and a short speck, is not.
TEST(TextLayer, InkTheLayerDoesNotHoldIsFound) {
  std::vector<PdfChar> chars;
  put_line(chars, U"A line of the layer", 10, 30);
  put_line(chars, U"and another of it", 10, 60);
  GrayImage page = inked(chars, 400, 300);
  for (int y = 150; y < 200; ++y) {  // a drawn label, 50 rows tall (lines 20 high): outside every character
    for (int x = 200; x < 260; x += 3) page.pixels[static_cast<size_t>(y) * 400 + x] = 0;
  }
  for (int x = 300; x < 320; ++x) page.pixels[static_cast<size_t>(250) * 400 + x] = 0;  // a rule, one row
  const TextLayer layer = read_text_layer(chars, page.view());
  ASSERT_TRUE(layer.check.usable);
  const std::vector<LineBox> found = uncovered_ink(page.view(), layer, chars, {});
  ASSERT_EQ(found.size(), 1u);
  EXPECT_NEAR(found[0].x0, 200, 1);
  EXPECT_NEAR(found[0].y0, 150, 1);
  EXPECT_NEAR(found[0].y1, 200, 1);
  const LineBox picture{190, 140, 270, 210};  // under a picture: the hybrid path reads it already
  EXPECT_TRUE(uncovered_ink(page.view(), layer, chars, std::span<const LineBox>(&picture, 1)).empty());
}

// A character past U+FFFF given as its UTF-16 surrogates reads as itself, in four bytes of UTF-8; a lone half
// is unreadable; a page with a few invisible characters (a watermark) is still read from its layer.
TEST(TextLayer, SurrogatesPairAndAFewInvisibleCharactersPass) {
  std::vector<PdfChar> chars;
  put_line(chars, U"Bold x", 0, 20);
  chars[5].code = 0xD835;  // U+1D431, a mathematical bold x, as two UTF-16 units
  put_line(chars, U"!", 60, 20);
  chars.back().code = 0xDC31;
  const GrayImage page = inked(chars, 300, 40);
  const TextLayer t = read_text_layer(chars, page.view());
  EXPECT_EQ(t.check.chars, 6);
  EXPECT_EQ(t.check.bad, 0);
  ASSERT_EQ(t.lines.size(), 1u);
  EXPECT_EQ(t.lines[0].text, "Bold \xF0\x9D\x90\xB1");
  std::vector<PdfChar> lone = chars;
  lone.pop_back();  // the high half alone
  EXPECT_EQ(read_text_layer(lone, page.view()).check.bad, 1);

  std::vector<PdfChar> marked;
  for (int r = 0; r < 10; ++r) put_line(marked, U"twenty characters ok", 0, 20 + 30.0f * r);
  marked[7].invisible = true;  // one of 200
  const GrayImage mp = inked(marked, 300, 340);
  const TextLayerCheck k = read_text_layer(marked, mp.view()).check;
  EXPECT_TRUE(k.usable) << k.reason;
  EXPECT_EQ(k.invisible, 1);
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
  // a justified line: every space stretched to about 0.7 of its height (14 px of 20), none under the split's
  // least gap; one line still, not a row of cells
  std::vector<PdfChar> wide;
  float x = 10;
  for (const std::u32string w : {U"justified ", U"text ", U"stretched ", U"across ", U"its ", U"column"}) {
    put_line(wide, w, x, 50);  // each word's space drawn, 10 px, then 4 more: gaps of 14 px
    x += 10 * static_cast<float>(w.size()) + 4;
  }
  const OcrPage just = page_from_text_layer(read_text_layer(wide, GrayView{nullptr, 0, 0, 0}), 800, 100);
  ASSERT_EQ(just.lines.size(), 1u);
  EXPECT_EQ(just.lines[0].text, "justified text stretched across its column");
}

// A tagged page reads in its structure tree's order, not the page's geometry, its blocks the elements' roles;
// an untagged or barely tagged one as before.
TEST(TextLayer, ATaggedPageTakesItsTreesOrder) {
  std::vector<PdfChar> chars;
  put_line(chars, U"Second", 10, 30);  // drawn first, top of the page, but the tree reads it second
  for (PdfChar& c : chars) c.mcid = 2;
  std::vector<PdfChar> head;
  put_line(head, U"Heading", 10, 80);
  for (PdfChar& c : head) c.mcid = 1;
  chars.insert(chars.end(), head.begin(), head.end());
  const TextLayer layer = read_text_layer(chars, GrayView{nullptr, 0, 0, 0});
  ASSERT_EQ(layer.lines.size(), 2u);
  EXPECT_EQ(layer.mcids, (std::vector<int>{2, 1}));
  PdfStructure tree;
  tree.elements = {{"H1", -1}, {"P", -1}};
  tree.runs = {{0, 1}, {1, 2}};  // the heading's content, then the paragraph's
  const OcrPage tagged = page_from_tagged_layer(layer, tree, 400, 100);
  EXPECT_EQ(tagged.order, "tagged");
  ASSERT_EQ(tagged.lines.size(), 2u);
  EXPECT_EQ(tagged.lines[0].text, "Heading");
  EXPECT_EQ(tagged.lines[1].text, "Second");
  ASSERT_EQ(tagged.blocks.size(), 2u);
  EXPECT_EQ(tagged.blocks[0].label, "Section-Header");
  EXPECT_EQ(tagged.blocks[1].label, "Text");
  const OcrPage plain = page_from_tagged_layer(layer, PdfStructure{}, 400, 100);
  EXPECT_NE(plain.order, "tagged");
  EXPECT_EQ(plain.lines[0].text, "Second");  // the XY-cut's: top first
  PdfStructure half;
  half.elements = {{"P", -1}};
  half.runs = {{0, 1}};  // the heading alone: 7 of 13 characters, under 75%
  EXPECT_NE(page_from_tagged_layer(layer, half, 400, 100).order, "tagged");
}

// A word broken at a line's end: PDFium's U+0002 (and a soft hyphen, U+00AD) reads as the hyphen the page prints,
// not a control character (it counted as unreadable text and left "gov ernment" once the writers joined the lines).
TEST(TextLayer, ALineEndHyphenIsAHyphen) {
  std::vector<PdfChar> chars;
  put_line(chars, U"However, gov\x02", 10, 30);
  put_line(chars, U"ernment spends", 10, 60);
  chars.insert(chars.begin() + 13, generated(U'\n'));
  const TextLayer layer = read_text_layer(chars, GrayView{nullptr, 0, 0, 0});
  EXPECT_EQ(layer.check.bad, 0);
  ASSERT_EQ(layer.lines.size(), 2u);
  EXPECT_EQ(layer.lines[0].text, "However, gov-");
  std::vector<PdfChar> soft;
  put_line(soft, U"glob\u00ADally", 10, 30);
  EXPECT_EQ(read_text_layer(soft, GrayView{nullptr, 0, 0, 0}).lines[0].text, "glob-ally");
}

// The bounds of an image's dark pixels.// The bounds of an image's dark pixels.
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
