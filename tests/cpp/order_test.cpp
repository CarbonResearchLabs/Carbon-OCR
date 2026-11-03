// Reading order and layout (carbon/order.h): the order model's features, and the
// decoding of outputs made here: a chain, its repair, the XY-cut when unsure,
// blocks and their labels, a table's cells.

#include "carbon/order.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "carbon/engine.h"
#include "carbon/error.h"

namespace carbon {
namespace {

// Outputs for n lines, every logit 0 (the model sure of nothing).
struct Outputs {
  int n;
  std::vector<float> successor, same_block, row, col, cell, labels;
  explicit Outputs(int lines)
      : n(lines),
        successor(static_cast<size_t>(lines + 1) * (lines + 1), 0.0f),
        same_block(static_cast<size_t>(lines) * lines, 0.0f),
        row(same_block),
        col(same_block),
        cell(same_block),
        labels(static_cast<size_t>(lines) * kBlockLabels, 0.0f) {}
  // a chain the successor is sure of: start, the lines in `order`, end
  void chain(const std::vector<int>& order, float logit = 8.0f) {
    int from = 0;
    for (int i : order) {
      successor[static_cast<size_t>(from) * (n + 1) + i + 1] = logit;
      from = i + 1;
    }
    successor[static_cast<size_t>(from) * (n + 1)] = logit;
  }
  void pair(std::vector<float>& m, int a, int b, float v) {
    m[static_cast<size_t>(a) * n + b] = v;
    m[static_cast<size_t>(b) * n + a] = v;
  }
  void label(int line, const char* name) { labels[static_cast<size_t>(line) * kBlockLabels + block_label(name)] = 8; }
  OrderOutputs view() const { return {n, successor, same_block, row, col, cell, labels}; }
};

TEST(Order, Features) {
  const std::vector<LineBox> boxes = {{100, 50, 500, 70}, {100, 80, 300, 100}, {100, 110, 400, 150}};
  const std::vector<std::string> texts = {"• Item one", "Results 2024 12", "ends with a hyphen-"};
  const std::vector<float> f = order_features(boxes, texts, 1000, 2000);
  ASSERT_EQ(f.size(), 3u * kOrderFeatures);
  EXPECT_FLOAT_EQ(f[0], 0.1f);   // x0 / W
  EXPECT_FLOAT_EQ(f[1], 0.025f); // y0 / H
  EXPECT_FLOAT_EQ(f[4], 0.3f);   // centre x
  EXPECT_FLOAT_EQ(f[6], 0.4f);   // width
  EXPECT_NEAR(f[7], 0.0f, 1e-6);  // the median height: 20
  EXPECT_NEAR(f[2 * kOrderFeatures + 7], std::log(2.0f), 1e-6);
  EXPECT_NEAR(f[8], std::log(20.0f), 1e-5);  // 400 / 20
  EXPECT_FLOAT_EQ(f[9], 0.01f);  // 20 / 2000
  EXPECT_NEAR(f[10], std::log1p(8.0f), 1e-6);  // "•Itemone": 8 characters, spaces aside
  EXPECT_EQ(f[12], 1.0f);  // a bullet
  EXPECT_EQ(f[15], 0.0f);  // "•" is no capital
  EXPECT_NEAR(f[kOrderFeatures + 11], 6.0f / 13, 1e-6);  // digits
  EXPECT_EQ(f[kOrderFeatures + 12], 0.0f);
  EXPECT_EQ(f[kOrderFeatures + 15], 1.0f);
  EXPECT_EQ(f[2 * kOrderFeatures + 13], 1.0f);  // the hyphen
  EXPECT_EQ(f[2 * kOrderFeatures + 14], 0.0f);
  for (const char* marker : {"1. First", "(iv) Fourth", "a) one", "– dash", "12) twelve"}) {
    EXPECT_EQ(order_features(std::vector<LineBox>{{0, 0, 10, 10}}, std::vector<std::string>{marker}, 10, 10)[12], 1.0f)
        << marker;
  }
  for (const char* plain : {"1999 was", "Mr. Smith", "(see below)", "e.g. this"}) {
    EXPECT_EQ(order_features(std::vector<LineBox>{{0, 0, 10, 10}}, std::vector<std::string>{plain}, 10, 10)[12], 0.0f)
        << plain;
  }
  for (const char* end : {"The end.", "Really?”", "He said (so).", "完。"}) {
    EXPECT_EQ(order_features(std::vector<LineBox>{{0, 0, 10, 10}}, std::vector<std::string>{end}, 10, 10)[14], 1.0f)
        << end;
  }
  EXPECT_EQ(order_features(std::vector<LineBox>{{0, 0, 10, 10}}, std::vector<std::string>{"Жuk"}, 10, 10)[15], 1.0f);
  EXPECT_THROW(order_features(boxes, std::vector<std::string>{"one"}, 10, 10), Error);
  EXPECT_TRUE(order_features({}, {}, 10, 10).empty());
}

// Two columns of two lines; the model reads them across, which the XY-cut would not.
TEST(Order, DecodesTheChainItIsSureOf) {
  const std::vector<LineBox> boxes = {{0, 0, 90, 10}, {0, 20, 90, 30}, {110, 0, 200, 10}, {110, 20, 200, 30}};
  Outputs o(4);
  o.chain({0, 2, 1, 3});
  const PageOrder p = decode_order(o.view(), boxes);
  EXPECT_FALSE(p.fallback);
  EXPECT_GT(p.sure, 0.99f);
  EXPECT_EQ(p.order, (std::vector<size_t>{0, 2, 1, 3}));
}

// Greedy goes wrong at the start (line 1 a little likelier than line 0), and the
// repair moves line 0 back to the front.
TEST(Order, RepairsTheGreedyChain) {
  const std::vector<LineBox> boxes = {{0, 0, 90, 10}, {0, 20, 90, 30}, {0, 40, 90, 50}};
  Outputs o(3);
  auto s = [&](int from, int to, float v) { o.successor[static_cast<size_t>(from) * 4 + to] = v; };
  s(0, 2, 3.0f);  // start -> line 1
  s(0, 1, 2.9f);  // start -> line 0
  s(1, 2, 8.0f);  // 0 -> 1
  s(2, 3, 8.0f);  // 1 -> 2
  s(3, 0, 8.0f);  // 2 -> end
  OrderRule rule;
  rule.min_sure = 0;
  rule.sweeps = 0;
  EXPECT_EQ(decode_order(o.view(), boxes, rule).order, (std::vector<size_t>{1, 2, 0})) << "greedy alone";
  rule.sweeps = 4;
  EXPECT_EQ(decode_order(o.view(), boxes, rule).order, (std::vector<size_t>{0, 1, 2}));
}

TEST(Order, FallsBackToTheXyCut) {
  const std::vector<LineBox> boxes = {{110, 0, 200, 10}, {0, 20, 90, 30}, {0, 0, 90, 10}, {110, 20, 200, 30}};
  Outputs o(4);  // sure of nothing
  const PageOrder p = decode_order(o.view(), boxes);
  EXPECT_TRUE(p.fallback);
  EXPECT_LT(p.sure, 0.5f);
  EXPECT_EQ(p.order, reading_order(boxes));
  EXPECT_EQ(p.order, (std::vector<size_t>{2, 1, 0, 3}));  // the left column, then the right
}

TEST(Order, BlocksAndTheirLabels) {
  const std::vector<LineBox> boxes = {{0, 0, 90, 10}, {0, 20, 90, 30}, {0, 40, 90, 50}, {0, 60, 90, 70}};
  Outputs o(4);
  o.chain({0, 1, 2, 3});
  o.pair(o.same_block, 0, 1, -6);  // a heading, then a paragraph of three lines
  o.pair(o.same_block, 1, 2, 6);
  o.pair(o.same_block, 2, 3, 6);
  o.label(0, "Section-Header");
  for (int i : {1, 2, 3}) o.label(i, "Text");
  o.labels[2 * kBlockLabels + block_label("Text")] = 1;  // one line less sure
  const PageOrder p = decode_order(o.view(), boxes);
  ASSERT_EQ(p.blocks.size(), 2u);
  EXPECT_STREQ(block_label_name(p.blocks[0].label), "Section-Header");
  EXPECT_EQ(p.blocks[0].lines, (std::vector<size_t>{0}));
  EXPECT_STREQ(block_label_name(p.blocks[1].label), "Text");
  EXPECT_EQ(p.blocks[1].lines, (std::vector<size_t>{1, 2, 3}));
  EXPECT_GT(p.blocks[0].confidence, 0.99f);
  EXPECT_LT(p.blocks[1].confidence, 0.99f);
  EXPECT_GT(p.blocks[1].confidence, 0.6f);
  EXPECT_TRUE(p.blocks[1].cells.empty());
  EXPECT_EQ(block_label("Nothing"), -1);
  EXPECT_STREQ(block_label_name(99), "Other");
}

// A 2 x 3 table, one cell of two lines; rows and columns numbered by place.
TEST(Order, ATablesCells) {
  //   [0]      [1]    [2]
  //   [3 + 4]  [5]    [6]
  const std::vector<LineBox> boxes = {{0, 0, 50, 10},  {100, 0, 150, 10}, {200, 0, 250, 10}, {0, 20, 50, 30},
                                      {0, 32, 50, 42}, {100, 20, 150, 30}, {200, 20, 250, 30}};
  Outputs o(7);
  o.chain({0, 1, 2, 3, 4, 5, 6});
  for (int a = 0; a < 7; ++a) {
    o.label(a, "Table");
    for (int b = a + 1; b < 7; ++b) {
      o.pair(o.same_block, a, b, 6);
      o.pair(o.cell, a, b, -6);
      const bool top_a = a < 3, top_b = b < 3;
      o.pair(o.row, a, b, top_a == top_b ? 6.0f : -6.0f);
      const int ca = a < 3 ? a : a == 3 || a == 4 ? 0 : a - 4, cb = b < 3 ? b : b == 3 || b == 4 ? 0 : b - 4;
      o.pair(o.col, a, b, ca == cb ? 6.0f : -6.0f);
    }
  }
  o.pair(o.cell, 3, 4, 6);
  const PageOrder p = decode_order(o.view(), boxes);
  ASSERT_EQ(p.blocks.size(), 1u);
  EXPECT_STREQ(block_label_name(p.blocks[0].label), "Table");
  const auto& cells = p.blocks[0].cells;
  ASSERT_EQ(cells.size(), 6u);
  const int want[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}};
  for (int c = 0; c < 6; ++c) {
    EXPECT_EQ(cells[c].row, want[c][0]) << c;
    EXPECT_EQ(cells[c].col, want[c][1]) << c;
  }
  EXPECT_EQ(cells[3].lines, (std::vector<size_t>{3, 4}));
  EXPECT_EQ(cells[5].lines, (std::vector<size_t>{6}));
}

// A line of words at x0..x1 (h 20 px), each word's confidence its index's tenth.
OcrLine words_line(const std::vector<std::pair<double, double>>& spans) {
  OcrLine l;
  for (size_t i = 0; i < spans.size(); ++i) {
    l.words.push_back({{spans[i].first, 100, spans[i].second, 120}, std::string(1, static_cast<char>('a' + i)),
                       0.1f * static_cast<float>(i + 1)});
    l.text += (i ? " " : "") + l.words.back().text;
  }
  l.box = {spans.front().first, 100, spans.back().second, 120};
  l.confidence = 0.5f;
  l.det_score = 0.9f;
  return l;
}

// Split at a column gutter and between a table row's cells; loose justified text,
// and a line without words, kept whole.
TEST(Split, LinesAtWideGaps) {
  // word spaces of 5 px (0.25 h), then a gutter of 40 (2 h)
  const std::vector<OcrLine> gutter = split_lines({words_line({{0, 30}, {35, 60}, {100, 140}})});
  ASSERT_EQ(gutter.size(), 2u);
  EXPECT_EQ(gutter[0].text, "a b");
  EXPECT_EQ(gutter[1].text, "c");
  EXPECT_EQ(gutter[0].box.x1, 60);
  EXPECT_EQ(gutter[1].box.x0, 100);
  EXPECT_EQ(gutter[1].box.y0, 100);
  EXPECT_EQ(gutter[1].words.size(), 1u);
  EXPECT_NEAR(gutter[0].confidence, 0.15f, 1e-6);  // its words', by characters
  EXPECT_EQ(gutter[1].det_score, 0.9f);
  // justified loosely: every gap 11 px (0.55 h), under min_gap: its own spaces
  EXPECT_EQ(split_lines({words_line({{0, 30}, {41, 70}, {81, 120}, {131, 160}})}).size(), 1u);
  // ... and a gap of 1.2 h among them, twice its spaces: not enough
  EXPECT_EQ(split_lines({words_line({{0, 30}, {41, 70}, {94, 120}})}).size(), 1u);
  // a table row: one word a cell, every gap 14 px (0.7 h): no spaces, so 0.25 h each
  EXPECT_EQ(split_lines({words_line({{0, 30}, {44, 70}, {84, 120}})}).size(), 3u);
  // a gap of 1.5 h splits whatever the spaces
  EXPECT_EQ(split_lines({words_line({{0, 30}, {41, 70}, {100, 130}})}).size(), 2u);
  OcrLine bare;
  bare.box = {0, 0, 500, 20};
  bare.text = "no words";
  const std::vector<OcrLine> kept = split_lines({bare});
  ASSERT_EQ(kept.size(), 1u);
  EXPECT_EQ(kept[0].text, "no words");
}

OcrLine at(double x0, double y0, double x1, const char* text) {
  OcrLine l;
  l.box = {x0, y0, x1, y0 + 20};
  l.text = text;
  return l;
}

// A 3 x 3 table between two paragraphs, its lines in the XY-cut's order (column by
// column): found, read row by row, its cells numbered; the paragraphs Text blocks.
TEST(Tables, FromGeometry) {
  OcrPage p;
  p.width = 600;
  p.height = 300;
  p.lines = {at(50, 20, 550, "A paragraph above the table, long enough to be one."),
             at(50, 60, 120, "Name"), at(50, 90, 110, "Ann"), at(50, 120, 110, "Bob"),
             at(200, 60, 240, "Age"), at(200, 90, 225, "31"), at(200, 120, 215, "4"),
             at(350, 60, 400, "City"), at(350, 90, 410, "Oslo"), at(350, 120, 405, "Rome"),
             at(50, 200, 300, "After the table.")};
  ASSERT_EQ(find_tables(p), 1);
  ASSERT_EQ(p.blocks.size(), 3u);
  EXPECT_EQ(p.blocks[0].label, "Text");
  EXPECT_EQ(p.blocks[0].lines, std::vector<int>{0});
  const OcrBlock& t = p.blocks[1];
  EXPECT_EQ(t.label, "Table");
  EXPECT_EQ(t.confidence, 0.0f) << "no model's";
  ASSERT_EQ(t.cells.size(), 9u);
  std::vector<std::string> read;
  for (int i : t.lines) read.push_back(p.lines[static_cast<size_t>(i)].text);
  EXPECT_EQ(read, (std::vector<std::string>{"Name", "Age", "City", "Ann", "31", "Oslo", "Bob", "4", "Rome"}));
  EXPECT_EQ(t.cells[4].row, 1);
  EXPECT_EQ(t.cells[4].col, 1);
  EXPECT_EQ(p.lines[static_cast<size_t>(t.cells[4].lines[0])].text, "31");
  EXPECT_EQ(p.blocks[2].lines, std::vector<int>{10});
  EXPECT_EQ(find_tables(p), 0) << "a page with blocks is left as it is";
}

// Two columns of body text line up as rows of two, but their lines are long: no table.
TEST(Tables, NotColumnsOfText) {
  OcrPage p;
  p.width = 600;
  p.height = 400;
  for (int r = 0; r < 6; ++r) {
    p.lines.push_back(at(40, 40 + 25.0 * r, 280, "the left column's body text, a long line of it"));
    p.lines.push_back(at(320, 40 + 25.0 * r, 560, "and the right column's, as long as the left's"));
  }
  const std::vector<OcrLine> before = p.lines;
  EXPECT_EQ(find_tables(p), 0);
  EXPECT_TRUE(p.blocks.empty());
  EXPECT_EQ(p.lines.size(), before.size());
  OcrPage one;  // a single row of cells is no table
  one.lines = {at(0, 0, 40, "a"), at(100, 0, 140, "b"), at(200, 0, 240, "c"), at(0, 200, 400, "text")};
  EXPECT_EQ(find_tables(one), 0);
}

TEST(Order, OutputsMustFitTheLines) {
  Outputs o(3);
  const std::vector<LineBox> two = {{0, 0, 1, 1}, {0, 2, 1, 3}};
  EXPECT_THROW(decode_order(o.view(), two), Error);
  const PageOrder none = decode_order(Outputs(0).view(), {});
  EXPECT_TRUE(none.order.empty());
  EXPECT_TRUE(none.blocks.empty());
}

}  // namespace
}  // namespace carbon
