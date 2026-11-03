// Reading order and layout: the `order` model's input, 16
// features a line, and the decoding of its outputs into the lines' order, blocks
// with labels, and tables' cells.  The XY-cut (carbon::reading_order) stands in
// where the model is not sure.
//
// The features of a page's lines (boxes in page pixels, and their text):
//
//   0-3   the box, over the page's width and height
//   4-5   its centre, likewise
//   6     its width over the page's
//   7     log of its height over the page's median line height
//   8     log of its aspect, width over height
//   9     the page's median line height over the page's height
//   10    log(1 + its characters)
//   11    the share of its characters that are digits
//   12    1 if it starts with a list marker: a bullet or dash, or 1-3 digits, a
//         letter or a roman numeral then "." or ")" (or in brackets)
//   13    1 if it ends with a hyphen
//   14    1 if it ends a sentence: . ! ? : ; or an ellipsis (CJK's too), before
//         any closing quotes or brackets
//   15    1 if it starts with a capital: Latin, Greek or Cyrillic
//
// The model's outputs for n lines (token 0 the page's start, 1..n its lines):
//
//   successor        [n+1][n+1] logits: row i (0 the start) the line after line
//                    i, column 0 the page's end
//   same block       [n][n] logits, symmetric
//   table row, table column, table cell   [n][n] logits each, symmetric
//   labels           [n][13] logits: the label of the line's block
//
// Decoding: a greedy chain from the start over the successor's probabilities, then
// repair sweeps that move a line wherever the chain's log probability rises; the
// XY-cut's order instead when the chain's mean link probability is low.  Blocks
// are the runs of that order whose neighbours are in the same block; a block's
// label the mean of its lines'.  A table's lines join into cells where both are
// in one, cells into rows and columns likewise, rows numbered top to bottom and
// columns left to right.
#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "carbon/lines.h"

namespace carbon {

inline constexpr int kOrderFeatures = 16;
inline constexpr int kBlockLabels = 13;

// Text, Section-Header, Page-Header, Page-Footer, List-Group, Table, Caption,
// Footnote, Table-Of-Contents, Form, Picture, Formula, Other.
const char* block_label_name(int label);
int block_label(std::string_view name);  // -1 for none of them

// [n][kOrderFeatures], row-major.
std::vector<float> order_features(std::span<const LineBox> boxes, std::span<const std::string> texts, int width,
                                  int height);

struct OrderOutputs {
  int n = 0;
  std::span<const float> successor;                                 // (n+1) * (n+1)
  std::span<const float> same_block, table_row, table_col, table_cell;  // n * n
  std::span<const float> labels;                                    // n * kBlockLabels
};

struct OrderRule {
  double min_sure = 0.5;  // the chain's mean link probability below which the XY-cut's order is used
  int sweeps = 4;         // repair sweeps at most
  float join = 0.5f;      // the pairwise probability that puts two lines in one block, cell, row or column
};

struct TableCell {
  int row = 0, col = 0;
  std::vector<size_t> lines;  // in reading order
};

struct OrderBlock {
  int label = 0;              // block_label_name
  float confidence = 0;       // the mean probability of that label over its lines
  std::vector<size_t> lines;  // a run of PageOrder::order
  std::vector<TableCell> cells;  // a table's, by row then column
};

struct PageOrder {
  std::vector<size_t> order;  // the lines in reading order
  std::vector<OrderBlock> blocks;
  float sure = 0;             // the chain's mean link probability
  bool fallback = false;      // the order is the XY-cut's
};

PageOrder decode_order(const OrderOutputs& out, std::span<const LineBox> boxes, const OrderRule& rule = {});

}  // namespace carbon
