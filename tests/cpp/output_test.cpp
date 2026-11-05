// The engine's pages as hOCR, ALTO, PAGE XML, markdown and HTML (carbon/output.h).

#include "carbon/output.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace carbon {
namespace {

int count(const std::string& s, const std::string& what) {
  int n = 0;
  for (size_t at = s.find(what); at != std::string::npos; at = s.find(what, at + 1)) ++n;
  return n;
}

// Two lines, the second's text with characters XML and markdown take as their own.
OcrPage page() {
  OcrPage p;
  p.width = 400;
  p.height = 200;
  OcrLine a;
  a.box = {10.2, 20.5, 190.7, 40};
  a.text = "Hello world";
  a.confidence = 0.9f;
  a.words = {{{10.2, 20.5, 80, 40}, "Hello", 0.95f}, {{95, 20.5, 190.7, 40}, "world", 0.85f}};
  OcrLine b;
  b.box = {10, 120, 300, 140};  // a gap of four lines above: a paragraph of its own
  b.text = "# R&D <x> \"q\"";
  b.confidence = 0.5f;
  p.lines = {a, b};  // b has no words: it is written as one
  return p;
}

TEST(Output, WritesHocr) {
  const std::vector<OcrPage> pages = {page(), OcrPage{}};
  const std::vector<std::string> images = {"scan \"1\".png", "empty.png"};
  const std::string h = to_hocr(pages, images);
  EXPECT_EQ(count(h, "class=\"ocr_page\""), 2);
  EXPECT_NE(h.find("title=\"image &quot;scan &quot;1&quot;.png&quot;; bbox 0 0 400 200; ppageno 0\""),
            std::string::npos)
      << h;
  EXPECT_EQ(count(h, "class=\"ocr_line\""), 2);
  EXPECT_EQ(count(h, "class=\"ocrx_word\""), 3);
  EXPECT_NE(h.find("title=\"bbox 10 20 80 40; x_wconf 95\">Hello</span>"), std::string::npos) << h;
  EXPECT_NE(h.find("bbox 10 20 191 40; baseline"), std::string::npos) << h;  // corners rounded outward
  EXPECT_NE(h.find("# R&amp;D &lt;x&gt; &quot;q&quot;"), std::string::npos) << h;
  EXPECT_EQ(count(h, "<span"), count(h, "</span>"));
  EXPECT_EQ(count(h, "<div"), count(h, "</div>"));
}

TEST(Output, WritesAlto) {
  const std::vector<OcrPage> pages = {page()};
  const std::vector<std::string> images = {"scan.png"};
  const std::string a = to_alto(pages, images);
  EXPECT_NE(a.find("xmlns=\"http://www.loc.gov/standards/alto/ns-v4#\""), std::string::npos);
  EXPECT_NE(a.find("<Page ID=\"page_0\" PHYSICAL_IMG_NR=\"1\" WIDTH=\"400\" HEIGHT=\"200\">"), std::string::npos);
  EXPECT_EQ(count(a, "<TextLine "), 2);
  EXPECT_EQ(count(a, "<String "), 3);
  EXPECT_NE(a.find("HPOS=\"10\" VPOS=\"20\" WIDTH=\"70\" HEIGHT=\"20\" WC=\"0.95\" CONTENT=\"Hello\""),
            std::string::npos)
      << a;
  EXPECT_NE(a.find("<SP WIDTH=\"15\" HPOS=\"80\""), std::string::npos) << a;
  EXPECT_NE(a.find("CONTENT=\"# R&amp;D &lt;x&gt; &quot;q&quot;\""), std::string::npos) << a;
  EXPECT_EQ(count(a, "<TextLine"), count(a, "</TextLine>"));
}

TEST(Output, WritesPageXml) {
  const std::string x = to_page_xml(page(), "scan.png");
  EXPECT_NE(x.find("<Page imageFilename=\"scan.png\" imageWidth=\"400\" imageHeight=\"200\">"), std::string::npos);
  EXPECT_NE(x.find("<TextLine id=\"r0l0\">"), std::string::npos);
  EXPECT_NE(x.find("<Coords points=\"10,20 80,20 80,40 10,40\"/>"), std::string::npos) << x;
  EXPECT_NE(x.find("<TextEquiv conf=\"0.9500\"><Unicode>Hello</Unicode></TextEquiv>"), std::string::npos) << x;
  EXPECT_EQ(count(x, "<Word "), 3);
  EXPECT_NE(x.find("<Created>"), std::string::npos);
  EXPECT_EQ(count(x, "<TextLine"), count(x, "</TextLine>"));
}

TEST(Output, WritesMarkdown) {
  const std::vector<OcrPage> pages = {page(), page()};
  const std::string m = to_markdown(pages);
  // a paragraph break at the wide gap, the heading's # escaped, a rule between pages
  EXPECT_EQ(m, "Hello world\n\n\\# R&D <x> \"q\"\n\n---\n\nHello world\n\n\\# R&D <x> \"q\"\n");
}

// A page with the order model's blocks: a heading, a paragraph, a list whose second
// item runs over two lines, a 2 x 2 table, a running footer.
OcrPage laid_out() {
  OcrPage p;
  p.width = 600;
  p.height = 800;
  const char* texts[] = {"Report", "First line of text", "second line.", "\u2022 apples", "\u2022 pears and",
                         "plums", "Name", "Age", "Ann & Bo", "31", "Page 3"};
  for (int i = 0; i < 11; ++i) {
    OcrLine l;
    l.box = {50.0, 40.0 + 50 * i, 400.0, 60.0 + 50 * i};
    l.text = texts[i];
    l.confidence = 1;
    p.lines.push_back(l);
  }
  p.order = "model";
  p.blocks = {{"Section-Header", 0.9f, {0}, {}},
              {"Text", 0.9f, {1, 2}, {}},
              {"List-Group", 0.8f, {3, 4, 5}, {}},
              {"Table", 0.7f, {6, 7, 8, 9}, {{0, 0, {6}}, {0, 1, {7}}, {1, 0, {8}}, {1, 1, {9}}}},
              {"Page-Footer", 0.9f, {10}, {}}};
  return p;
}

TEST(Output, WritesTheOrderModelsBlocks) {
  const std::vector<OcrPage> pages = {laid_out()};
  EXPECT_EQ(to_markdown(pages),
            "## Report\n\nFirst line of text second line.\n\n- apples\n- pears and plums\n\n<table>\n"
            "<tr><td>Name</td><td>Age</td></tr>\n<tr><td>Ann &amp; Bo</td><td>31</td></tr>\n</table>\n");
  const std::vector<std::string> images = {"scan.png"};
  const std::string h = to_hocr(pages, images);
  EXPECT_EQ(count(h, "class=\"ocr_carea\""), 5);
  EXPECT_EQ(count(h, "class=\"ocr_header\""), 1);
  EXPECT_EQ(count(h, "class=\"ocr_line\""), 10);
  EXPECT_NE(h.find("id=\"block_1_5\" title=\"bbox 50 540 400 560\""), std::string::npos) << h;
  const std::string a = to_alto(pages, images);
  EXPECT_EQ(count(a, "<TextBlock "), 5);
  EXPECT_NE(a.find("<LayoutTag ID=\"layout_1\" LABEL=\"Section-Header\"/>"), std::string::npos) << a;
  EXPECT_NE(a.find("<TextBlock ID=\"block_0_0\" HPOS=\"50\" VPOS=\"40\" WIDTH=\"350\" HEIGHT=\"20\" "
                   "TAGREFS=\"layout_1\">"),
            std::string::npos)
      << a;
  EXPECT_LT(a.find("</Tags>"), a.find("<Layout>"));
  EXPECT_EQ(to_alto(std::vector<OcrPage>{page()}, images).find("<Tags>"), std::string::npos) << "no blocks, no tags";
  const std::string x = to_page_xml(laid_out(), "scan.png");
  EXPECT_EQ(count(x, "<TextRegion "), 5);
  EXPECT_NE(x.find("<TextRegion id=\"r0\" type=\"heading\" custom=\"structure {type:Section-Header;}\">"),
            std::string::npos)
      << x;
  EXPECT_NE(x.find("type=\"footer\""), std::string::npos);
  EXPECT_NE(x.find("<TextLine id=\"r2l2\">"), std::string::npos);  // "plums", the list's third
  EXPECT_EQ(count(x, "<TextRegion"), count(x, "</TextRegion>"));
}

TEST(Output, WritesTheBlocksAsHtml) {
  OcrPage p = laid_out();
  p.lines[2].text = "line, then a hyphen-";  // "hyphen-" + "ated" on the next line: one word again
  p.lines.push_back({{50, 600, 400, 620}, "ated word", 1, 0, {}});
  p.blocks[1].lines = {1, 2, 11};
  EXPECT_EQ(to_html_body(p),
            "<h2>Report</h2>\n<p>First line of text line, then a hyphenated word</p>\n"
            "<ul><li>apples</li><li>pears and plums</li></ul>\n<table>\n<tr><td>Name</td><td>Age</td></tr>\n"
            "<tr><td>Ann &amp; Bo</td><td>31</td></tr>\n</table>\n<footer><p>Page 3</p></footer>\n");
  const std::string doc = to_html(std::vector<OcrPage>{p, page()});
  EXPECT_EQ(doc.rfind("<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">", 0), 0u) << doc;
  EXPECT_EQ(count(doc, "<section class=\"page\""), 2);
  EXPECT_NE(doc.find("<section class=\"page\" id=\"page-2\">\n<p>Hello world</p>\n"
                     "<p># R&amp;D &lt;x&gt; \"q\"</p>\n</section>"),
            std::string::npos)
      << doc;  // no blocks: a paragraph at the wide gap, the text escaped and never read as markup
}

TEST(Output, WritesEachLabelAsItsHtml) {
  OcrPage p;
  p.width = 600;
  p.height = 600;
  const char* texts[] = {"3. third", "4. fourth", "• dot", "Fig. 2: a plot", "1 A note.", "E = mc2",
                         "Name: ____", "Contents", "Chapter 1 ..... 5", "in the figure", "Running head",
                         "Hyphen-", "Ated"};  // a capital after the hyphen: not a broken word
  for (int i = 0; i < 13; ++i) p.lines.push_back({{10, 10.0 + 40 * i, 300, 30.0 + 40 * i}, texts[i], 1, 0, {}});
  p.blocks = {{"List-Group", 1, {0, 1, 2}, {}}, {"Caption", 1, {3}, {}},        {"Footnote", 1, {4}, {}},
              {"Formula", 1, {5}, {}},          {"Form", 1, {6}, {}},           {"Table-Of-Contents", 1, {7, 8}, {}},
              {"Picture", 1, {9}, {}},          {"Page-Header", 1, {10}, {}},   {"Text", 1, {11, 12}, {}}};
  EXPECT_EQ(to_html_body(p),
            "<ol start=\"3\"><li>third</li><li>fourth</li></ol>\n<ul><li>dot</li></ul>\n"
            "<figure><figcaption>Fig. 2: a plot</figcaption></figure>\n"
            "<aside class=\"footnote\"><p>1 A note.</p></aside>\n<p class=\"formula\">E = mc2</p>\n"
            "<div class=\"form\"><p>Name: ____</p></div>\n"
            "<nav><ul><li>Contents</li><li>Chapter 1 ..... 5</li></ul></nav>\n<figure><p>in the figure</p></figure>\n"
            "<header><p>Running head</p></header>\n<p>Hyphen- Ated</p>\n");
}

// A Text block's lines, one paragraph: in markdown as in HTML, joined by spaces, and a word broken at a line's end
// joined again after any letter (ß is two bytes), at a soft hyphen too, before any lowercase (ł is lowercase).
TEST(Output, JoinsAParagraphsLinesInMarkdownAsInHtml) {
  OcrPage p;
  p.width = p.height = 600;
  const char* texts[] = {"Ein groß-", "artiger Satz mit Wort\u00AD", "teilen und abc-", "\u0142\u00F3d\u017A."};
  for (int i = 0; i < 4; ++i) p.lines.push_back({{10, 10.0 + 30 * i, 300, 30.0 + 30 * i}, texts[i], 1, 0, {}});
  p.blocks = {{"Text", 1, {0, 1, 2, 3}, {}}};
  const std::string want = "Ein großartiger Satz mit Wortteilen und abc\u0142\u00F3d\u017A.";
  EXPECT_EQ(to_markdown(std::vector<OcrPage>{p}), want + "\n");
  EXPECT_EQ(to_html_body(p), "<p>" + want + "</p>\n");
  p.blocks.clear();  // no blocks: the lines are close, one paragraph
  EXPECT_EQ(to_markdown(std::vector<OcrPage>{p}), want + "\n");
}

// A paragraph's start that markdown would read as its own: a fence, an HTML block, a thematic break.
TEST(Output, EscapesWhatMarkdownWouldReadAsBlocks) {
  OcrPage p;
  p.width = p.height = 600;
  const char* texts[] = {"```code", "~~~", "<div> a", "____", "= = =", "a < b"};
  for (int i = 0; i < 6; ++i) p.lines.push_back({{10, 10.0 + 60 * i, 300, 30.0 + 60 * i}, texts[i], 1, 0, {}});
  for (int i = 0; i < 6; ++i) p.blocks.push_back({"Text", 1, {i}, {}});
  EXPECT_EQ(to_markdown(std::vector<OcrPage>{p}),
            "\\```code\n\n\\~~~\n\n\\<div> a\n\n\\____\n\n\\= = =\n\na < b\n");
}

// A picture's unsure lines (what a photograph read as) are left out of markdown and HTML; its sure ones written.
// A heading that is not one (a formula read as bars and digits, or read unsurely) is written as a paragraph.
TEST(Output, WritesAPicturesSureLinesAndHeadingsThatReadAsHeadings) {
  OcrPage p;
  p.width = p.height = 600;
  p.lines = {{{10, 10, 300, 30}, "qur 51R9", 0.6f, 0, {}},
             {{10, 40, 300, 60}, "Figure legend text", 0.97f, 0, {}},
             {{10, 100, 300, 120}, "| ? 3785 2 17 |", 0.95f, 0, {}},
             {{10, 140, 300, 160}, "Methods", 0.5f, 0, {}},
             {{10, 180, 300, 200}, "Results", 0.99f, 0, {}}};
  p.blocks = {{"Picture", 1, {0, 1}, {}},
              {"Section-Header", 1, {2}, {}},
              {"Section-Header", 1, {3}, {}},
              {"Section-Header", 1, {4}, {}}};
  EXPECT_EQ(to_markdown(std::vector<OcrPage>{p}),
            "Figure legend text\n\n\\| ? 3785 2 17 |\n\nMethods\n\n## Results\n");
  EXPECT_EQ(to_html_body(p),
            "<figure><p>Figure legend text</p></figure>\n<p>| ? 3785 2 17 |</p>\n<p>Methods</p>\n<h2>Results</h2>\n");
}

// A table's spans: a header over three columns one <td colspan>, a cell over two rows one <td rowspan>, the
// places they cover left out; markdown's table as HTML's.
TEST(Output, WritesATablesSpans) {
  OcrPage p;
  p.width = p.height = 600;
  const char* texts[] = {"Results", "A", "B", "C", "x", "1", "2"};
  for (int i = 0; i < 7; ++i) p.lines.push_back({{10, 10.0 + 30 * i, 300, 30.0 + 30 * i}, texts[i], 1, 0, {}});
  OcrBlock t{"Table", 1, {0, 1, 2, 3, 4, 5, 6}, {}};
  t.cells = {{0, 0, {0}, 1, 3}, {1, 0, {1}, 1, 1}, {1, 1, {2}, 1, 1}, {1, 2, {3}, 1, 1},
             {2, 0, {4}, 2, 1}, {2, 1, {5}, 1, 1}, {3, 1, {6}, 1, 1}};
  p.blocks = {t};
  const std::string want =
      "<table>\n<tr><td colspan=\"3\">Results</td></tr>\n<tr><td>A</td><td>B</td><td>C</td></tr>\n"
      "<tr><td rowspan=\"2\">x</td><td>1</td><td></td></tr>\n<tr><td>2</td><td></td></tr>\n</table>";
  EXPECT_EQ(to_html_body(p), want + "\n");
  EXPECT_EQ(to_markdown(std::vector<OcrPage>{p}), want + "\n");
}

// Boxes on the page as given: a page read turned a quarter (its boxes the upright page's) and one levelled.
TEST(Output, PlacesBoxesOnThePageAsGiven) {
  OcrPage p;
  p.width = 200;   // as given: 200 wide, 400 high, its text running down
  p.height = 400;
  p.rotation = 90;  // read upright, 400 wide and 200 high
  p.lines = {{{10, 20, 190, 40}, "Hello", 0.9f, 0, {{{10, 20, 190, 40}, "Hello", 0.9f}}}};
  const LineBox g = box_as_given(p.lines[0].box, p);
  EXPECT_EQ(g.x0, 160);
  EXPECT_EQ(g.y0, 10);
  EXPECT_EQ(g.x1, 180);
  EXPECT_EQ(g.y1, 190);
  const std::vector<std::string> images = {"turned.png"};
  const std::string h = to_hocr(std::vector<OcrPage>{p}, images);
  EXPECT_NE(h.find("bbox 160 10 180 190; baseline"), std::string::npos) << h;
  EXPECT_NE(h.find("x_size 20;"), std::string::npos) << h;  // the text's own height, as read
  const std::string a = to_alto(std::vector<OcrPage>{p}, images);
  EXPECT_NE(a.find("HPOS=\"160\" VPOS=\"10\" WIDTH=\"20\" HEIGHT=\"180\" WC=\"0.90\" CONTENT=\"Hello\""),
            std::string::npos)
      << a;

  OcrPage lv;  // levelled by 2 degrees: a polygon in PAGE XML, not a box
  lv.width = 400;
  lv.height = 200;
  lv.skew = 2.0;
  lv.lines = {{{100, 90, 300, 110}, "Hello", 0.9f, 0, {}}};
  const auto c = corners_as_given(lv.lines[0].box, lv);
  EXPECT_NE(c[0][1], c[1][1]);  // the top edge is not level on the page as given
  const std::string x = to_page_xml(lv, "levelled.png");
  EXPECT_EQ(x.find("<Coords points=\"100,90 300,90 300,110 100,110\"/>"), std::string::npos) << x;
  OcrPage given = lv;
  given.skew = 0;  // read as given: the box itself, rounded outward as before
  EXPECT_NE(to_page_xml(given, "g.png").find("<Coords points=\"100,90 300,90 300,110 100,110\"/>"),
            std::string::npos);
  const LineBox same = box_as_given(given.lines[0].box, given);
  EXPECT_EQ(same.x0, 100);
  EXPECT_EQ(same.y1, 110);
}

}  // namespace
}  // namespace carbon
