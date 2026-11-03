// Reading order and layout (carbon/order.h): the order model's features, and the
// decoding of its outputs.

#include "carbon/order.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "carbon/engine.h"
#include "carbon/error.h"
#include "carbon/score.h"  // code_points

namespace carbon {
namespace {

constexpr const char* kLabels[kBlockLabels] = {
    "Text",     "Section-Header",    "Page-Header", "Page-Footer", "List-Group", "Table", "Caption",
    "Footnote", "Table-Of-Contents", "Form",        "Picture",     "Formula",    "Other"};

bool is_capital(char32_t c) {
  return (c >= U'A' && c <= U'Z') || (c >= 0xC0 && c <= 0xDE && c != 0xD7) || (c >= 0x391 && c <= 0x3A9) ||
         (c >= 0x400 && c <= 0x42F);
}

bool is_roman(char32_t c) { return std::u32string_view(U"ivxlcIVXLC").find(c) != std::u32string_view::npos; }

// A list marker: the line's first word a bullet or dash, or 1-3 digits, a letter or
// a roman numeral (up to 5) followed by "." or ")", or the same in brackets.
bool list_marker(const std::u32string& t) {
  size_t a = 0;
  while (a < t.size() && t[a] == U' ') ++a;
  size_t b = a;
  while (b < t.size() && t[b] != U' ') ++b;
  const std::u32string_view w(t.data() + a, b - a);
  if (w.empty()) return false;
  if (w.size() == 1 && std::u32string_view(U"•◦▪▫‣●○■□-–—*·►▸➢✓").find(w[0]) != std::u32string_view::npos) {
    return true;
  }
  std::u32string_view core = w;
  if (core.size() >= 3 && core.front() == U'(' && core.back() == U')') {
    core = core.substr(1, core.size() - 2);
  } else if (core.size() >= 2 && (core.back() == U'.' || core.back() == U')')) {
    core = core.substr(0, core.size() - 1);
  } else {
    return false;
  }
  if (core.empty()) return false;
  const bool digits = core.size() <= 3 && std::all_of(core.begin(), core.end(), [](char32_t c) {
    return c >= U'0' && c <= U'9';
  });
  const bool letter =
      core.size() == 1 && ((core[0] >= U'a' && core[0] <= U'z') || (core[0] >= U'A' && core[0] <= U'Z'));
  const bool roman = core.size() <= 5 && std::all_of(core.begin(), core.end(), is_roman);
  return digits || letter || roman;
}

bool sentence_end(const std::u32string& t) {
  size_t e = t.size();
  while (e > 0 && std::u32string_view(U" \"'”’»)]}」』").find(t[e - 1]) != std::u32string_view::npos) --e;
  return e > 0 && std::u32string_view(U".!?:;…。！？").find(t[e - 1]) != std::u32string_view::npos;
}

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Groups of `items` joined where the mean pairwise probability between two groups'
// members reaches `join` (average linkage, merged greedily, best pair first).
std::vector<std::vector<size_t>> cluster(const std::vector<size_t>& items, std::span<const float> logits, size_t n,
                                         float join) {
  const size_t m = items.size();
  std::vector<std::vector<size_t>> g;
  for (size_t i : items) g.push_back({i});
  std::vector<double> s(m * m, 0.0);  // the summed probabilities between two groups' members
  for (size_t a = 0; a < m; ++a) {
    for (size_t b = 0; b < m; ++b) {
      if (a == b) continue;
      s[a * m + b] = (sigmoid(logits[items[a] * n + items[b]]) + sigmoid(logits[items[b] * n + items[a]])) / 2;
    }
  }
  std::vector<bool> alive(m, true);
  for (;;) {
    double best = join;
    size_t ba = m, bb = m;
    for (size_t a = 0; a < m; ++a) {
      for (size_t b = a + 1; alive[a] && b < m; ++b) {
        if (!alive[b]) continue;
        const double mean = s[a * m + b] / static_cast<double>(g[a].size() * g[b].size());
        if (mean >= best) {
          best = mean;
          ba = a;
          bb = b;
        }
      }
    }
    if (ba == m) break;
    for (size_t c = 0; c < m; ++c) {
      if (!alive[c] || c == ba || c == bb) continue;
      s[ba * m + c] += s[bb * m + c];
      s[c * m + ba] = s[ba * m + c];
    }
    g[ba].insert(g[ba].end(), g[bb].begin(), g[bb].end());
    alive[bb] = false;
  }
  std::vector<std::vector<size_t>> out;
  for (size_t a = 0; a < m; ++a) {
    if (alive[a]) out.push_back(std::move(g[a]));
  }
  return out;
}

}  // namespace

const char* block_label_name(int label) { return label >= 0 && label < kBlockLabels ? kLabels[label] : "Other"; }

int block_label(std::string_view name) {
  for (int i = 0; i < kBlockLabels; ++i) {
    if (name == kLabels[i]) return i;
  }
  return -1;
}

std::vector<float> order_features(std::span<const LineBox> boxes, std::span<const std::string> texts, int width,
                                  int height) {
  if (boxes.size() != texts.size()) throw Error("order_features: a text for every box");
  const size_t n = boxes.size();
  std::vector<float> f(n * kOrderFeatures, 0.0f);
  if (n == 0) return f;
  const double W = std::max(1, width), H = std::max(1, height);
  std::vector<double> heights;
  for (const LineBox& b : boxes) heights.push_back(std::max(1.0, b.y1 - b.y0));
  std::vector<double> sorted = heights;
  std::nth_element(sorted.begin(), sorted.begin() + static_cast<int64_t>(n / 2), sorted.end());
  const double median = sorted[n / 2];
  for (size_t i = 0; i < n; ++i) {
    const LineBox& b = boxes[i];
    float* r = &f[i * kOrderFeatures];
    const double w = std::max(1.0, b.x1 - b.x0);
    r[0] = static_cast<float>(b.x0 / W);
    r[1] = static_cast<float>(b.y0 / H);
    r[2] = static_cast<float>(b.x1 / W);
    r[3] = static_cast<float>(b.y1 / H);
    r[4] = static_cast<float>((b.x0 + b.x1) / 2 / W);
    r[5] = static_cast<float>((b.y0 + b.y1) / 2 / H);
    r[6] = static_cast<float>(w / W);
    r[7] = static_cast<float>(std::log(heights[i] / median));
    r[8] = static_cast<float>(std::log(w / heights[i]));
    r[9] = static_cast<float>(median / H);
    const std::u32string t = code_points(texts[i]);
    size_t chars = 0, digits = 0;
    for (char32_t c : t) {
      if (c == U' ') continue;
      ++chars;
      digits += c >= U'0' && c <= U'9';
    }
    r[10] = static_cast<float>(std::log1p(static_cast<double>(chars)));
    r[11] = chars ? static_cast<float>(digits) / static_cast<float>(chars) : 0.0f;
    r[12] = list_marker(t);
    size_t e = t.size();
    while (e > 0 && t[e - 1] == U' ') --e;
    r[13] = e > 0 && (t[e - 1] == U'-' || t[e - 1] == 0xAD || t[e - 1] == 0x2010);
    r[14] = sentence_end(t);
    size_t s = 0;
    while (s < t.size() && t[s] == U' ') ++s;
    r[15] = s < t.size() && is_capital(t[s]);
  }
  return f;
}

PageOrder decode_order(const OrderOutputs& out, std::span<const LineBox> boxes, const OrderRule& rule) {
  const int n = out.n;
  const size_t N = static_cast<size_t>(n), T = N + 1;
  if (boxes.size() != N || out.successor.size() != T * T || out.same_block.size() != N * N ||
      out.table_row.size() != N * N || out.table_col.size() != N * N || out.table_cell.size() != N * N ||
      out.labels.size() != N * kBlockLabels) {
    throw Error("decode_order: outputs not the size of " + std::to_string(n) + " lines");
  }
  PageOrder po;
  if (n == 0) {
    po.sure = 1;
    return po;
  }
  // log probabilities of the successor, each row a softmax over the other tokens (column 0 the end)
  std::vector<double> lp(T * T, -1e30);
  for (size_t i = 0; i < T; ++i) {
    double m = -1e30;
    for (size_t j = 0; j < T; ++j) {
      if (j != i) m = std::max(m, static_cast<double>(out.successor[i * T + j]));
    }
    double z = 0;
    for (size_t j = 0; j < T; ++j) {
      if (j != i) z += std::exp(out.successor[i * T + j] - m);
    }
    for (size_t j = 0; j < T; ++j) {
      if (j != i) lp[i * T + j] = out.successor[i * T + j] - m - std::log(z);
    }
  }
  // the chain, as tokens: start, then lines (token = line + 1), then the end (0)
  std::vector<size_t> chain = {0};
  std::vector<bool> used(T, false);
  for (int k = 0; k < n; ++k) {
    const size_t cur = chain.back();
    size_t best = 0;
    double bv = -1e300;
    for (size_t j = 1; j < T; ++j) {
      if (!used[j] && lp[cur * T + j] > bv) {
        bv = lp[cur * T + j];
        best = j;
      }
    }
    used[best] = true;
    chain.push_back(best);
  }
  chain.push_back(0);
  // repair: move one line to where the chain's log probability rises most, while it does
  for (int sweep = 0; sweep < rule.sweeps; ++sweep) {
    bool moved = false;
    for (size_t p = 1; p + 1 < chain.size(); ++p) {
      const size_t x = chain[p], a = chain[p - 1], b = chain[p + 1];
      const double out_gain = lp[a * T + b] - lp[a * T + x] - lp[x * T + b];
      double best = 1e-9;
      size_t to = 0;
      for (size_t q = 0; q + 1 < chain.size(); ++q) {  // between chain[q] and chain[q + 1], x taken out
        if (q == p - 1 || q == p) continue;
        const size_t c = chain[q], d = chain[q + 1];
        const double gain = out_gain + lp[c * T + x] + lp[x * T + d] - lp[c * T + d];
        if (gain > best) {
          best = gain;
          to = q + 1;
        }
      }
      if (to) {
        chain.erase(chain.begin() + static_cast<int64_t>(p));
        chain.insert(chain.begin() + static_cast<int64_t>(to > p ? to - 1 : to), x);
        moved = true;
      }
    }
    if (!moved) break;
  }
  double sum = 0;
  for (size_t k = 0; k + 1 < chain.size(); ++k) sum += std::exp(lp[chain[k] * T + chain[k + 1]]);
  po.sure = static_cast<float>(sum / static_cast<double>(chain.size() - 1));
  if (po.sure < rule.min_sure) {
    po.fallback = true;
    po.order = reading_order(boxes);
  } else {
    for (size_t k = 1; k + 1 < chain.size(); ++k) po.order.push_back(chain[k] - 1);
  }
  // blocks: runs of the order whose neighbours share a block
  for (size_t k = 0; k < N; ++k) {
    const size_t i = po.order[k];
    if (k == 0 || sigmoid(out.same_block[po.order[k - 1] * N + i]) < rule.join) po.blocks.emplace_back();
    po.blocks.back().lines.push_back(i);
  }
  for (OrderBlock& b : po.blocks) {
    std::vector<double> mean(kBlockLabels, 0.0);
    for (size_t i : b.lines) {
      const float* z = &out.labels[i * kBlockLabels];
      const float m = *std::max_element(z, z + kBlockLabels);
      double s = 0;
      for (int c = 0; c < kBlockLabels; ++c) s += std::exp(z[c] - m);
      for (int c = 0; c < kBlockLabels; ++c) mean[c] += std::exp(z[c] - m) / s / static_cast<double>(b.lines.size());
    }
    b.label = static_cast<int>(std::max_element(mean.begin(), mean.end()) - mean.begin());
    b.confidence = static_cast<float>(mean[b.label]);
    if (b.label != block_label("Table")) continue;
    // cells, then rows and columns of cells
    std::vector<std::vector<size_t>> cells = cluster(b.lines, out.table_cell, N, rule.join);
    for (auto& c : cells) {  // a cell's lines in reading order
      std::sort(c.begin(), c.end(), [&](size_t x, size_t y) {
        return std::find(b.lines.begin(), b.lines.end(), x) < std::find(b.lines.begin(), b.lines.end(), y);
      });
    }
    std::vector<size_t> heads(cells.size());  // a cell stands for its lines by its first
    for (size_t c = 0; c < cells.size(); ++c) heads[c] = cells[c].front();
    auto groups = [&](std::span<const float> logits, bool rows) {
      std::vector<std::vector<size_t>> g = cluster(heads, logits, N, rule.join);
      auto at = [&](const std::vector<size_t>& v) {  // the group's mean centre across it
        double s = 0;
        for (size_t i : v) s += rows ? boxes[i].y0 + boxes[i].y1 : boxes[i].x0 + boxes[i].x1;
        return s / static_cast<double>(v.size());
      };
      std::sort(g.begin(), g.end(), [&](const auto& x, const auto& y) { return at(x) < at(y); });
      std::vector<int> of(N, 0);
      for (size_t k = 0; k < g.size(); ++k) {
        for (size_t i : g[k]) of[i] = static_cast<int>(k);
      }
      return of;
    };
    const std::vector<int> row = groups(out.table_row, true), col = groups(out.table_col, false);
    for (auto& c : cells) b.cells.push_back({row[c.front()], col[c.front()], std::move(c)});
    std::sort(b.cells.begin(), b.cells.end(),
              [](const TableCell& x, const TableCell& y) { return x.row != y.row ? x.row < y.row : x.col < y.col; });
  }
  return po;
}

}  // namespace carbon
