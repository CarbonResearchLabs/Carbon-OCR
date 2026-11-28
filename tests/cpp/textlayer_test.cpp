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

// Characters set at `scale` of put_line's size (12 pt), their baseline `raise` px above the row's (below where
// negative): a script, as a PDF sets one.
void put_script(std::vector<PdfChar>& out, const std::u32string& text, float x, float y, float raise,
                float scale = 0.7f) {
  const float base = y - raise, w = 10 * scale;
  for (char32_t c : text) {
    PdfChar p;
    p.code = c;
    p.lx0 = x;
    p.lx1 = x + w;
    p.ly0 = base - 16 * scale;
    p.ly1 = base + 4 * scale;
    p.x0 = x + 2 * scale;
    p.x1 = x + 8 * scale;
    p.y0 = base - 12 * scale;
    p.y1 = base;
    p.size = 12 * scale;
    out.push_back(p);
    x += w;
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

// A usable page's ruled lines, for the false-table check, only where TextLayerRule::rules asks for them; on the
// page's own pixels, in them.
TEST(TextLayer, FindsItsRuledLinesWhereAsked) {
  std::vector<PdfChar> chars;
  put_line(chars, U"A table under a rule", 10, 40);
  GrayImage page = inked(chars, 400, 240);
  for (int x = 10; x < 390; ++x) page.pixels[static_cast<size_t>(60) * page.width + x] = 0;  // a rule, 1 px
  EXPECT_TRUE(read_text_layer(chars, page.view()).rules.empty());
  TextLayerRule asked;
  asked.rules = true;
  const TextLayer t = read_text_layer(chars, page.view(), asked);
  ASSERT_EQ(t.rules.size(), 1u);
  EXPECT_DOUBLE_EQ(t.rules[0].x0, 10);
  EXPECT_DOUBLE_EQ(t.rules[0].x1, 390);
  EXPECT_DOUBLE_EQ(t.rules[0].y0, 60);
  EXPECT_EQ(page_from_text_layer(t, page.width, page.height).rules.size(), 1u);
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

// The layer's line across two table cells (one space between them, too narrow to split) made the detector's two
// lines, each word into the detected line that holds it; a line no detected box takes, and a box far taller than
// the words, leave the layer's lines as they were.
TEST(TextLayer, LinesFromTheDetectorsLines) {
  std::vector<PdfChar> chars;
  put_line(chars, U"Total cost 12", 10, 50);  // words at x 10-60, 70-110 and 120-140, y 34-54
  chars.push_back(generated(U'\n'));
  put_line(chars, U"Note", 10, 120);
  TextLayer layer = read_text_layer(chars, GrayView{nullptr, 0, 0, 0});
  ASSERT_EQ(layer.lines.size(), 2u);
  EXPECT_EQ(layer.lines[0].text, "Total cost 12");
  const std::vector<LineBox> detected = {{5, 30, 112, 58}, {115, 30, 145, 58}, {0, 90, 400, 170}};
  EXPECT_TRUE(reads_across(layer));
  EXPECT_TRUE(lines_from_detection(layer, detected));
  ASSERT_EQ(layer.lines.size(), 3u);
  EXPECT_EQ(layer.lines[0].text, "Total cost");
  EXPECT_EQ(layer.lines[0].box.x0, 10);
  EXPECT_EQ(layer.lines[0].box.x1, 110);
  EXPECT_EQ(layer.lines[0].words.size(), 2u);
  EXPECT_EQ(layer.lines[1].text, "12");
  EXPECT_EQ(layer.lines[2].text, "Note");  // the box over it is four of its heights tall: not a line of its
  EXPECT_EQ(layer.mcids, (std::vector<int>{-1, -1, -1}));
  const OcrPage page = page_from_text_layer(layer, 400, 200);
  ASSERT_EQ(page.lines.size(), 3u);
  EXPECT_EQ(page.source, "text layer");

  // A superscript set smaller and raised well off the line (its baseline 10 px up, its centre 8 px over the
  // line's: the text after it is out of its band) stays in the layer's line with the text after it; as a word of its
  // own in no detected line (its centre over the detected line's top), it goes with its line's words.
  std::vector<PdfChar> sup;
  put_line(sup, U"area m ", 10, 50);
  put_script(sup, U"2", 80, 50, 10);
  put_line(sup, U" is", 87, 50);
  TextLayer s = read_text_layer(sup, GrayView{nullptr, 0, 0, 0});
  ASSERT_EQ(s.lines.size(), 1u);
  ASSERT_EQ(s.lines[0].words.size(), 4u);
  EXPECT_EQ(s.lines[0].text, "area m 2 is");
  EXPECT_EQ(s.lines[0].words[2].scripts, (std::vector<TextScript>{{0, 1, false}}));
  EXPECT_TRUE(lines_from_detection(s, std::vector<LineBox>{{5, 36, 120, 56}}));
  ASSERT_EQ(s.lines.size(), 1u);
  EXPECT_EQ(s.lines[0].text, "area m 2 is");
  EXPECT_EQ(s.lines[0].scripts, (std::vector<TextScript>{{7, 8, false}}));
  // the line below (24 px down, its loose boxes 20 high), no break between them, is a line of its own still
  std::vector<PdfChar> two = sup;
  put_line(two, U"next", 10, 74);
  EXPECT_EQ(read_text_layer(two, GrayView{nullptr, 0, 0, 0}).lines.size(), 2u);
  // a table's cell under the one before, its first character under the line's but not its last (a step down
  // the page, across the way the line runs: held to the last character's band), its own line
  std::vector<PdfChar> cells;
  put_line(cells, U"Nam", 10, 50);
  put_line(cells, U"Age", 20, 74);  // its A 10 px left of the m and 24 px down: a step down, under the line
  const TextLayer stacked = read_text_layer(cells, GrayView{nullptr, 0, 0, 0});
  ASSERT_EQ(stacked.lines.size(), 2u);
  EXPECT_EQ(stacked.lines[1].text, "Age");

  // A page turned upside down (each line's words right to left on the page) keeps the layer's lines.
  TextLayer turned;
  OcrLine l;
  l.text = "upside down";
  l.box = {10, 30, 120, 50};
  l.words = {{{70, 30, 120, 50}, "upside", 1}, {{10, 30, 60, 50}, "down", 1}};
  turned.lines = {l};
  turned.mcids = {-1};
  EXPECT_FALSE(reads_across(turned));
  EXPECT_FALSE(lines_from_detection(turned, std::vector<LineBox>{{5, 25, 125, 55}}));
  ASSERT_EQ(turned.lines.size(), 1u);
  EXPECT_EQ(turned.lines[0].text, "upside down");
}

// Characters set smaller (at most 0.8 of the line's median size as drawn) and raised or lowered (their centre
// more than 0.15 of the body's height off the body's) are a line's scripts, marked in its words and the line, a run
// of them one; one raised but not smaller, smaller but on the baseline, in a line too short to have a body, or in
// one that runs down the page, is not. The size as drawn: a PDF that sets its fonts at 1 and scales the text matrix
// marks the same. The marks follow the words into the pieces a line is split into and the detector's lines.
TEST(TextLayer, MarksScriptsSetSmallerAndRaisedOrLowered) {
  using S = std::vector<TextScript>;
  auto one_line = [](const std::vector<PdfChar>& chars) {
    const TextLayer t = read_text_layer(chars, GrayView{nullptr, 0, 0, 0});
    EXPECT_EQ(t.lines.size(), 1u);
    return t.lines.empty() ? OcrLine{} : t.lines[0];
  };
  std::vector<PdfChar> water;
  put_line(water, U"H", 10, 50);
  put_script(water, U"2", 20, 50, -4);  // lowered 4 px at 0.7 of the size: its centre 5.8 px below the body's
  put_line(water, U"O is wet", 27, 50);
  OcrLine l = one_line(water);
  EXPECT_EQ(l.text, "H2O is wet");
  ASSERT_EQ(l.words.size(), 3u);
  EXPECT_EQ(l.words[0].scripts, (S{{1, 2, true}}));
  EXPECT_TRUE(l.words[1].scripts.empty());
  EXPECT_EQ(l.scripts, (S{{1, 2, true}}));

  std::vector<PdfChar> area;
  put_line(area, U"area m", 10, 50);
  put_script(area, U"2", 70, 50, 5);  // raised 5 px: its centre 3.2 px (0.16 of the body) above the body's
  put_line(area, U" is", 77, 50);
  l = one_line(area);
  EXPECT_EQ(l.text, "area m2 is");
  EXPECT_EQ(l.words[1].scripts, (S{{1, 2, false}}));
  EXPECT_EQ(l.scripts, (S{{6, 7, false}}));

  std::vector<PdfChar> ion;
  put_line(ion, U"[Ca", 10, 50);
  put_script(ion, U"2+", 40, 50, 5);
  put_line(ion, U"]", 54, 50);
  put_script(ion, U"i", 64, 50, -4);
  put_line(ion, U" (nM)", 71, 50);
  l = one_line(ion);
  EXPECT_EQ(l.text, "[Ca2+]i (nM)");
  EXPECT_EQ(l.words[0].scripts, (S{{3, 5, false}, {6, 7, true}}));  // "2+" one run

  std::vector<PdfChar> unit = area;  // set at 1, scaled by the matrix: the same size as drawn
  for (PdfChar& c : unit) {
    c.matrix_scale = c.size;
    c.size = 1;
  }
  EXPECT_EQ(one_line(unit).scripts, (S{{6, 7, false}}));

  std::vector<PdfChar> raised;  // raised 6 px, at the line's own size: no script
  put_line(raised, U"area m", 10, 50);
  put_line(raised, U"2", 70, 44);
  put_line(raised, U" is", 80, 50);
  EXPECT_TRUE(one_line(raised).scripts.empty());
  std::vector<PdfChar> small;  // smaller, on the baseline (small capitals): no script
  put_line(small, U"ab", 10, 50);
  put_script(small, U"cd", 30, 50, 0);
  put_line(small, U"ef", 44, 50);
  EXPECT_TRUE(one_line(small).scripts.empty());
  std::vector<PdfChar> shrt;  // three characters: no body to measure against
  put_line(shrt, U"m", 10, 50);
  put_script(shrt, U"2", 20, 50, 5);
  put_line(shrt, U"s", 27, 50);
  EXPECT_TRUE(one_line(shrt).scripts.empty());
  std::vector<PdfChar> down = area;  // down the page
  for (PdfChar& c : down) c.angle = 90;
  EXPECT_TRUE(read_text_layer(down, GrayView{nullptr, 0, 0, 0}).lines[0].scripts.empty());

  // split at a table's gap: each piece its own words' scripts
  std::vector<PdfChar> row = area;
  put_line(row, U" 12", 290, 50);
  const TextLayer layer = read_text_layer(row, GrayView{nullptr, 0, 0, 0});
  ASSERT_EQ(layer.lines.size(), 1u);
  const OcrPage page = page_from_text_layer(layer, 400, 100);
  ASSERT_EQ(page.lines.size(), 2u);
  EXPECT_EQ(page.lines[0].text, "area m2 is");
  EXPECT_EQ(page.lines[0].scripts, (S{{6, 7, false}}));
  EXPECT_EQ(page.lines[1].text, "12");
  EXPECT_TRUE(page.lines[1].scripts.empty());
  // the detector's lines: "area" in one, "m2 is" in another
  TextLayer d = read_text_layer(area, GrayView{nullptr, 0, 0, 0});
  ASSERT_TRUE(lines_from_detection(d, std::vector<LineBox>{{5, 30, 52, 58}, {55, 30, 120, 58}}));
  ASSERT_EQ(d.lines.size(), 2u);
  EXPECT_EQ(d.lines[1].text, "m2 is");
  EXPECT_EQ(d.lines[1].scripts, (S{{1, 2, false}}));
  EXPECT_TRUE(d.lines[0].scripts.empty());
}

// A layer refused for its bad codes, all of them in runs (text objects) of their own that hold no letter (a
// math extension font's pieces), has those runs set aside when asked: the lines they touch left out, their boxes to
// read by OCR, the checks on the rest. A bad run that holds a letter (a broken encoding), runs over half the lines,
// or the rule left off: the page refused as before.
TEST(TextLayer, BadCodeRunsAreSetAsideOneRunAtATime) {
  auto page = [](bool letter_in_run, int touched) {
    std::vector<PdfChar> chars;
    for (int r = 0; r < 4; ++r) {
      put_line(chars, U"a line of sound text", 10, 30.0f + 30 * r);
      for (size_t i = chars.size() - 20; i < chars.size(); ++i) chars[i].object = r;
      chars.push_back(generated(U'\n'));
      if (r < touched) {  // a bracket's pieces in a font of their own, mapped to private use, then its closing
        put_line(chars, letter_in_run ? U"x" : U")", 10, 45.0f + 30 * r);
        for (size_t i = chars.size() - 3; i < chars.size(); ++i) chars[i].object = 10 + r;
        chars.push_back(generated(U'\n'));
      }
    }
    return chars;
  };
  TextLayerRule rule;
  rule.set_aside_bad_runs = true;
  const std::vector<PdfChar> one = page(false, 1);  // 2 bad of 83: 2.4%
  EXPECT_FALSE(read_text_layer(one, GrayView{nullptr, 0, 0, 0}).check.usable) << "the rule off: refused";
  const TextLayer t = read_text_layer(one, GrayView{nullptr, 0, 0, 0}, rule);
  EXPECT_TRUE(t.check.usable) << t.check.reason;
  EXPECT_EQ(t.check.set_aside, 3);
  EXPECT_EQ(t.check.bad, 0);
  EXPECT_EQ(t.check.chars, 80);
  ASSERT_EQ(t.lines.size(), 4u);
  EXPECT_EQ(t.mcids.size(), 4u);
  for (const OcrLine& l : t.lines) EXPECT_EQ(l.text, "a line of sound text");
  ASSERT_EQ(t.set_aside.size(), 1u);
  EXPECT_DOUBLE_EQ(t.set_aside[0].x0, 10);
  EXPECT_DOUBLE_EQ(t.set_aside[0].x1, 40);
  EXPECT_DOUBLE_EQ(t.set_aside[0].y0, 29);
  EXPECT_DOUBLE_EQ(t.set_aside[0].y1, 49);
  const std::vector<LineBox> read = set_aside_to_read(t, 400, 48);  // 2 px larger about it, clipped to the page
  ASSERT_EQ(read.size(), 1u);
  EXPECT_DOUBLE_EQ(read[0].x0, 8);
  EXPECT_DOUBLE_EQ(read[0].y0, 27);
  EXPECT_DOUBLE_EQ(read[0].y1, 48);

  const TextLayer mixed = read_text_layer(page(true, 1), GrayView{nullptr, 0, 0, 0}, rule);
  EXPECT_FALSE(mixed.check.usable) << "a letter in a bad run";
  EXPECT_NE(mixed.check.reason.find("not readable"), std::string::npos);
  EXPECT_TRUE(mixed.set_aside.empty());
  const TextLayer most = read_text_layer(page(false, 4), GrayView{nullptr, 0, 0, 0}, rule);
  EXPECT_TRUE(most.check.usable) << "four of eight lines: half, not over";
  std::vector<PdfChar> over = page(false, 4);
  over.erase(over.begin(), over.begin() + 21);  // the first sound line gone: four of seven lines touched
  const TextLayer refused = read_text_layer(over, GrayView{nullptr, 0, 0, 0}, rule);
  EXPECT_FALSE(refused.check.usable);
  EXPECT_EQ(refused.check.reason, read_text_layer(over, GrayView{nullptr, 0, 0, 0}).check.reason);
  EXPECT_TRUE(refused.set_aside.empty());
  std::vector<PdfChar> loose = one;  // a bad code in no text object: refused
  for (PdfChar& c : loose) {
    if (c.object == 10) c.object = -1;
  }
  EXPECT_FALSE(read_text_layer(loose, GrayView{nullptr, 0, 0, 0}, rule).check.usable);
}

// A page whose text runs sideways or upside down as shown is turned upright when over half the characters it
// draws run one quarter turn: back by that turn; its generated characters are not counted, a page half turned is
// not turned, and an upright page never is.
TEST(TextLayer, APageWhoseTextRunsSidewaysIsTurnedUpright) {
  auto page = [](int turned, int upright, float other = -2) {
    std::vector<PdfChar> chars;
    for (int i = 0; i < turned; ++i) {
      PdfChar c;
      c.code = U'a';
      c.angle = other > -2 ? other : 0;
      chars.push_back(c);
    }
    for (int i = 0; i < upright; ++i) {
      PdfChar c;
      c.code = U'b';
      chars.push_back(c);
    }
    return chars;
  };
  auto at = [](std::vector<PdfChar> chars, float angle) {
    for (PdfChar& c : chars) {
      if (c.code == U'a') c.angle = angle;
    }
    return chars;
  };
  const std::vector<PdfChar> most = page(6, 4), half = page(5, 5);
  EXPECT_EQ(upright_turn(at(most, 270)), 1);  // up the page: a quarter clockwise brings it upright
  EXPECT_EQ(upright_turn(at(most, 90)), 3);   // down the page
  EXPECT_EQ(upright_turn(at(most, 180)), 2);
  EXPECT_EQ(upright_turn(at(most, 268)), 1);  // within 4.5 degrees of the turn
  EXPECT_EQ(upright_turn(at(most, 260)), 0);  // set at a slant: no quarter turn
  EXPECT_EQ(upright_turn(at(most, 359.5f)), 0);
  EXPECT_EQ(upright_turn(at(half, 270)), 0);  // half, not over half
  EXPECT_EQ(upright_turn(at(most, -1)), 0);   // no angle read
  EXPECT_EQ(upright_turn(page(0, 10)), 0);
  EXPECT_EQ(upright_turn({}), 0);
  std::vector<PdfChar> spaced = at(half, 90);
  for (int i = 0; i < 4; ++i) {  // PDFium's own spaces, down the page too, are not the page's characters
    PdfChar g;
    g.code = U' ';
    g.generated = true;
    g.angle = 90;
    spaced.push_back(g);
  }
  EXPECT_EQ(upright_turn(spaced), 0);

  // read turned, the page is the page as shown: its rotation the text's turn there, its boxes mapped back
  OcrPage p;
  p.width = 400;  // the upright page: 400 x 200
  p.height = 200;
  OcrLine l;
  l.box = {10, 20, 110, 40};
  l.text = "x";
  p.lines = {l};
  read_turned(p, 0);
  EXPECT_EQ(p.rotation, 0);
  EXPECT_EQ(p.width, 400);
  read_turned(p, 1);  // shown 200 x 400, its text up the page: turned 270 degrees clockwise
  EXPECT_EQ(p.rotation, 270);
  EXPECT_EQ(p.width, 200);
  EXPECT_EQ(p.height, 400);
  const LineBox g = box_as_given(p.lines[0].box, p);  // the upright (x, y) shows at (y, 400 - x)
  EXPECT_DOUBLE_EQ(g.x0, 20);
  EXPECT_DOUBLE_EQ(g.x1, 40);
  EXPECT_DOUBLE_EQ(g.y0, 290);
  EXPECT_DOUBLE_EQ(g.y1, 390);
  OcrPage down;
  down.width = 400;
  down.height = 200;
  read_turned(down, 3);
  EXPECT_EQ(down.rotation, 90);
  EXPECT_EQ(down.width, 200);
  OcrPage over;
  over.width = 400;
  over.height = 200;
  read_turned(over, 2);
  EXPECT_EQ(over.rotation, 180);
  EXPECT_EQ(over.width, 400);
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

// Of the lines the detector found on a whole page, its pictures pass reads those at least half inside the
// regions to read and less than half under the layer's own lines; none such, nothing to read.
TEST(TextLayer, APicturesPassReadsTheLinesFoundInItsPictures) {
  OcrPage page;  // read from its layer: one line, across the top
  page.width = 600;
  page.height = 800;
  OcrLine own;
  own.box = {50, 50, 550, 70};
  page.lines.push_back(own);
  OcrPage found;
  found.width = 600;
  found.height = 800;
  found.pictures = {{100, 200, 400, 500}};
  found.second_pass = true;
  auto line = [](double x0, double y0, double x1, double y1, float score) {
    OcrLine l;
    l.box = {x0, y0, x1, y1};
    l.det_score = score;
    return l;
  };
  found.lines = {line(50, 50, 550, 70, 0.9f),     // the layer's own line: under it
                 line(120, 250, 300, 270, 0.8f),  // in the picture
                 line(300, 300, 500, 320, 0.7f),  // half in it
                 line(350, 350, 550, 370, 0.6f),  // a quarter in it
                 line(50, 600, 300, 620, 0.5f),   // in the uncovered ink
                 line(400, 700, 500, 720, 0.4f)};  // in neither
  const std::vector<LineBox> to_read = {{100, 200, 400, 500}, {40, 590, 310, 630}};
  const OcrPage picked = picture_lines(found, page, to_read);
  ASSERT_EQ(picked.lines.size(), 3u);
  EXPECT_EQ(picked.lines[0].det_score, 0.8f);
  EXPECT_EQ(picked.lines[1].det_score, 0.7f);
  EXPECT_EQ(picked.lines[2].det_score, 0.5f);
  EXPECT_EQ(picked.width, 600);
  EXPECT_EQ(picked.pictures.size(), 1u);
  EXPECT_TRUE(picked.second_pass);
  // nothing found in the regions: nothing to read
  EXPECT_TRUE(picture_lines(found, page, std::vector<LineBox>{{0, 750, 50, 800}}).lines.empty());
  EXPECT_TRUE(picture_lines(found, page, {}).lines.empty());
}

}  // namespace
}  // namespace carbon
