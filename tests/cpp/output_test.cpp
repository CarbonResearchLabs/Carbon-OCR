// The engine's pages as hOCR, ALTO, PAGE XML and markdown (carbon/output.h).

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
            "## Report\n\nFirst line of text\nsecond line.\n\n- apples\n- pears and plums\n\n<table>\n"
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

}  // namespace
}  // namespace carbon
