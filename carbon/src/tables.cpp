// Tables by geometry inside the order model's Table blocks (carbon/engine.h: GridRule, EngineOptions::grid).

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

#include "carbon/engine.h"
#include "engine_steps.h"

namespace carbon::detail {
namespace {

double overlap(double a0, double a1, double b0, double b1) { return std::min(a1, b1) - std::max(a0, b0); }
double centre_y(const LineBox& b) { return (b.y0 + b.y1) / 2; }
double centre_x(const LineBox& b) { return (b.x0 + b.x1) / 2; }

int chars_of(const std::string& s) {
  int n = 0;
  for (unsigned char c : s) n += (c & 0xC0) != 0x80;
  return n;
}

double median_of(std::vector<double> v, double none) {
  if (v.empty()) return none;
  std::nth_element(v.begin(), v.begin() + static_cast<int64_t>(v.size() / 2), v.end());
  return v[v.size() / 2];
}

using Bands = std::vector<std::pair<double, double>>;

// The steps' view of a page's lines.
struct Lines {
  const std::vector<OcrLine>& lines;
  const GridRule& rule;

  const LineBox& box(int i) const { return lines[static_cast<size_t>(i)].box; }

  double height(const std::vector<int>& items) const {
    std::vector<double> h;
    h.reserve(items.size());
    for (int i : items) h.push_back(std::max(1.0, box(i).y1 - box(i).y0));
    return std::max(1.0, median_of(std::move(h), 1.0));
  }

  LineBox extent(const std::vector<int>& items) const {
    LineBox u{1e300, 1e300, -1e300, -1e300};
    for (int i : items) {
      u = {std::min(u.x0, box(i).x0), std::min(u.y0, box(i).y0), std::max(u.x1, box(i).x1), std::max(u.y1, box(i).y1)};
    }
    return u;
  }

  // GridRule 3: rows by centre, each left to right.
  std::vector<std::vector<int>> rows(std::vector<int> items, double h) const {
    std::stable_sort(items.begin(), items.end(), [&](int a, int b) { return centre_y(box(a)) < centre_y(box(b)); });
    std::vector<std::vector<int>> out;
    double sum = 0;
    for (int i : items) {
      if (!out.empty() &&
          std::abs(centre_y(box(i)) - sum / static_cast<double>(out.back().size())) <= rule.row_tol * h) {
        out.back().push_back(i);
        sum += centre_y(box(i));
        continue;
      }
      out.push_back({i});
      sum = centre_y(box(i));
    }
    for (auto& r : out) std::stable_sort(r.begin(), r.end(), [&](int a, int b) { return box(a).x0 < box(b).x0; });
    return out;
  }

  // GridRule 4: the columns of the fullest rows, those that overlap joined.
  Bands columns(const std::vector<std::vector<int>>& rows) const {
    size_t m = 0;
    for (const auto& r : rows) m = std::max(m, r.size());
    Bands bands;
    for (size_t k = 0; k < m; ++k) {
      std::vector<double> x0, x1;
      for (const auto& r : rows) {
        if (r.size() != m) continue;
        x0.push_back(box(r[k]).x0);
        x1.push_back(box(r[k]).x1);
      }
      bands.push_back({median_of(std::move(x0), 0), median_of(std::move(x1), 0)});
    }
    std::sort(bands.begin(), bands.end());
    Bands out;
    for (const auto& b : bands) {
      if (!out.empty() && b.first < out.back().second - 1) {
        out.back().second = std::max(out.back().second, b.second);
      } else {
        out.push_back(b);
      }
    }
    return out;
  }

  // A line's first column and how many it spans.
  std::pair<int, int> column(int i, const Bands& bands) const {
    const LineBox& b = box(i);
    int first = -1, last = -1;
    for (size_t k = 0; k < bands.size(); ++k) {
      const double w = std::max(1.0, bands[k].second - bands[k].first);
      if (overlap(b.x0, b.x1, bands[k].first, bands[k].second) > rule.cover * w) {
        if (first < 0) first = static_cast<int>(k);
        last = static_cast<int>(k);
      }
    }
    if (first >= 0) return {first, last - first + 1};
    const double cx = centre_x(b);
    double best = 1e300;
    for (size_t k = 0; k < bands.size(); ++k) {
      const double d = cx < bands[k].first ? bands[k].first - cx : cx > bands[k].second ? cx - bands[k].second : 0;
      if (d < best) {
        best = d;
        first = static_cast<int>(k);
      }
    }
    return {first, 1};
  }

  // GridRule 1: two tables' lines one table.
  bool one_table(const std::vector<int>& a, const std::vector<int>& b) const {
    const LineBox A = extent(a), B = extent(b);
    const double h = std::min(height(a), height(b));
    const double across = overlap(A.x0, A.x1, B.x0, B.x1);
    if (across > 0.5 * std::min(A.x1 - A.x0, B.x1 - B.x0)) {
      return std::max(B.y0 - A.y1, A.y0 - B.y1) <= rule.merge_gap * h;
    }
    if (across > 0 || overlap(A.y0, A.y1, B.y0, B.y1) <= 0.5 * std::min(A.y1 - A.y0, B.y1 - B.y0)) return false;
    const std::vector<int>& small = a.size() <= b.size() ? a : b;
    const std::vector<int>& big = a.size() <= b.size() ? b : a;
    size_t hits = 0;
    for (int i : small) {
      for (int j : big) {
        if (std::abs(centre_y(box(i)) - centre_y(box(j))) < rule.side_tol * h) {
          ++hits;
          break;
        }
      }
    }
    return static_cast<double>(hits) >= rule.side_rows * static_cast<double>(small.size());
  }

  // GridRule 5 and the cells: rows joined where a row is a cell's next lines, each row's lines in its columns (two
  // in one column one cell), a cell's lines down the page. Empty where the lines make no two columns.
  std::vector<OcrCell> grid(const std::vector<int>& items) const {
    const double h = height(items);
    const std::vector<std::vector<int>> rows = this->rows(items, h);
    const Bands bands = columns(rows);
    if (bands.size() < 2) return {};
    std::vector<double> cy;
    for (const auto& r : rows) {
      double s = 0;
      for (int i : r) s += centre_y(box(i));
      cy.push_back(s / static_cast<double>(r.size()));
    }
    std::vector<double> pitch;
    for (size_t k = 1; k < rows.size(); ++k) pitch.push_back(cy[k] - cy[k - 1]);
    const double mp = median_of(std::move(pitch), h);
    size_t first_full = rows.size();
    for (size_t k = 0; k < rows.size(); ++k) {
      if (rows[k].size() >= bands.size()) {
        first_full = k;
        break;
      }
    }
    // a line of a number alone (digits, no letter) is a value of its own row, never a cell's next line
    auto number = [&](int i) { return this->number(i); };
    const bool header = first_full <= static_cast<size_t>(rule.header_rows);  // a header of a few lines at most
    std::vector<std::vector<std::vector<int>>> joined(1);
    for (int i : rows[0]) joined.back().push_back({i});
    size_t prev = 0;  // the row the last joined row's cells began with
    for (size_t k = 1; k < rows.size(); ++k) {
      const std::vector<int>& r = rows[k];
      bool join = r.size() <= rows[prev].size() && std::none_of(r.begin(), r.end(), number) &&
                  ((header && k < first_full) || cy[k] - cy[k - 1] < rule.wrap_pitch * mp);
      std::vector<size_t> hit;
      for (size_t a = 0; join && a < r.size(); ++a) {  // each over a different line of that row
        size_t best = 0;
        double most = -1e300;
        for (size_t q = 0; q < rows[prev].size(); ++q) {
          const double v = overlap(box(r[a]).x0, box(r[a]).x1, box(rows[prev][q]).x0, box(rows[prev][q]).x1);
          if (v > most) {
            most = v;
            best = q;
          }
        }
        join = most > 0 && std::find(hit.begin(), hit.end(), best) == hit.end();
        hit.push_back(best);
      }
      if (join) {
        for (size_t a = 0; a < r.size(); ++a) joined.back()[hit[a]].push_back(r[a]);
        continue;
      }
      joined.emplace_back();
      for (int i : r) joined.back().push_back({i});
      prev = k;
    }
    std::vector<OcrCell> cells;
    for (size_t r = 0; r < joined.size(); ++r) {
      std::map<int, size_t> at;  // the cell of each column of this row
      for (const std::vector<int>& lines : joined[r]) {
        const auto [c, span] = column(lines.front(), bands);
        if (const auto f = at.find(c); f != at.end()) {
          std::vector<int>& to = cells[f->second].lines;
          to.insert(to.end(), lines.begin(), lines.end());
          continue;
        }
        at[c] = cells.size();
        OcrCell cell;
        cell.row = static_cast<int>(r);
        cell.col = c;
        cell.col_span = std::max(1, std::min(span, static_cast<int>(bands.size()) - c));
        cell.lines = lines;
        cells.push_back(std::move(cell));
      }
    }
    row_labels(cells, static_cast<int>(joined.size()));
    for (OcrCell& c : cells) {
      std::stable_sort(c.lines.begin(), c.lines.end(), [&](int a, int b) {
        return centre_y(box(a)) != centre_y(box(b)) ? centre_y(box(a)) < centre_y(box(b)) : box(a).x0 < box(b).x0;
      });
    }
    std::stable_sort(cells.begin(), cells.end(),
                     [](const OcrCell& x, const OcrCell& y) { return x.row != y.row ? x.row < y.row : x.col < y.col; });
    return cells;
  }

  // GridRule 6: a first-column label over the data rows next to it whose first cells are empty (rows that hold a
  // number: a value's row, not a header's) spans them. Where every label has such rows only below it, a label heads
  // its group (the rows down to the next); where most have them above and below, it is set in the group's middle
  // (each row the nearer label's, the one above's at a tie; the first label's the rows above it, the last's those
  // below). A row whose first cell spans columns (a section's) bounds the groups and spans nothing. Else no spans.
  void row_labels(std::vector<OcrCell>& cells, int nrows) const {
    const size_t R = static_cast<size_t>(nrows);
    std::vector<int> first(R, -1);
    std::vector<char> any(R, 0), value(R, 0);
    for (size_t k = 0; k < cells.size(); ++k) {
      const size_t r = static_cast<size_t>(cells[k].row);
      any[r] = 1;
      if (cells[k].col == 0) first[r] = static_cast<int>(k);
      for (int i : cells[k].lines) value[r] |= number(i);
    }
    int data = 0;  // the first row with a value: the rows above are the header's, its first cell no row's label
    while (data < nrows && !value[static_cast<size_t>(data)]) ++data;
    auto open = [&](int r) {
      return r >= data && r < nrows && first[static_cast<size_t>(r)] < 0 && any[static_cast<size_t>(r)] &&
             value[static_cast<size_t>(r)];
    };
    struct Label {
      int row, above, below;  // its row, and the open rows next to it either side
    };
    std::vector<Label> labels;
    for (int r = data; r < nrows; ++r) {
      if (first[static_cast<size_t>(r)] < 0 || cells[static_cast<size_t>(first[static_cast<size_t>(r)])].col_span != 1) continue;
      int a = 0, b = 0;
      while (open(r - a - 1)) ++a;
      while (open(r + b + 1)) ++b;
      labels.push_back({r, a, b});
    }
    if (labels.empty()) return;
    // the open rows between two labels are below the one and above the other: the first label tells the style
    const bool top = labels.front().above == 0;
    int both = 0;
    for (const Label& l : labels) both += l.above > 0 && l.below > 0;
    const bool middle = !top && 2 * both > static_cast<int>(labels.size());
    if (!top && !middle) return;
    for (size_t j = 0; j < labels.size(); ++j) {
      const Label& l = labels[j];
      int from = l.row, to = l.row + l.below;
      if (middle) {
        // the open rows between it and the label above split between them, as those below it and the next
        if (j && labels[j - 1].row + labels[j - 1].below >= l.row - l.above - 1) {
          from = (labels[j - 1].row + l.row) / 2 + 1;
        } else {
          from = l.row - l.above;
        }
        if (j + 1 < labels.size() && l.row + l.below >= labels[j + 1].row - labels[j + 1].above - 1) {
          to = (l.row + labels[j + 1].row) / 2;
        }
      }
      if (from == l.row && to == l.row) continue;
      OcrCell& c = cells[static_cast<size_t>(first[static_cast<size_t>(l.row)])];
      c.row = from;
      c.row_span = to - from + 1;
    }
  }

  // A line of a number alone (digits, no letter): a value.
  bool number(int i) const {
    bool digit = false;
    for (unsigned char c : lines[static_cast<size_t>(i)].text) {
      if (c >= 0x80 || std::isalpha(c)) return false;
      digit |= c >= '0' && c <= '9';
    }
    return digit;
  }
};

}  // namespace

void tables_by_geometry(OcrPage& page, const GridRule& rule) {
  const size_t n = page.lines.size();
  const Lines g{page.lines, rule};
  // 1. the Table blocks that are one table
  std::vector<std::vector<size_t>> members;  // each table's blocks
  std::vector<std::vector<int>> items;       // and their lines
  for (size_t b = 0; b < page.blocks.size(); ++b) {
    if (page.blocks[b].label != "Table" || page.blocks[b].lines.empty()) continue;
    members.push_back({b});
    items.push_back(page.blocks[b].lines);
  }
  if (items.empty()) return;
  // Whether another line lies between two tables (a caption, a paragraph, another table: two tables, not one): its
  // centre in the gap between their boxes, down the page within the span across they share, or across the page
  // within the span down they share.
  std::vector<char> mine(n, 0);
  auto between = [&](const std::vector<int>& a, const std::vector<int>& b) {
    const LineBox A = g.extent(a), B = g.extent(b);
    for (int i : a) mine[static_cast<size_t>(i)] = 1;
    for (int i : b) mine[static_cast<size_t>(i)] = 1;
    const double gy0 = std::min(A.y1, B.y1), gy1 = std::max(A.y0, B.y0);  // the gap down the page, if any
    const double gx0 = std::min(A.x1, B.x1), gx1 = std::max(A.x0, B.x0);  // the gap across it, if any
    const double sx0 = std::max(A.x0, B.x0), sx1 = std::min(A.x1, B.x1);  // the span across they share
    const double sy0 = std::max(A.y0, B.y0), sy1 = std::min(A.y1, B.y1);  // the span down they share
    bool found = false;
    for (size_t i = 0; i < n && !found; ++i) {
      if (mine[i]) continue;
      const LineBox& l = page.lines[i].box;
      const double cx = centre_x(l), cy = centre_y(l);
      found = (gy1 > gy0 && cy > gy0 && cy < gy1 && cx >= sx0 && cx <= sx1) ||
              (gx1 > gx0 && cx > gx0 && cx < gx1 && cy >= sy0 && cy <= sy1);
    }
    for (int i : a) mine[static_cast<size_t>(i)] = 0;
    for (int i : b) mine[static_cast<size_t>(i)] = 0;
    return found;
  };
  for (bool joined = true; joined;) {
    joined = false;
    for (size_t a = 0; a < items.size() && !joined; ++a) {
      for (size_t b = a + 1; b < items.size(); ++b) {
        if (!g.one_table(items[a], items[b]) || between(items[a], items[b])) continue;
        items[a].insert(items[a].end(), items[b].begin(), items[b].end());
        members[a].insert(members[a].end(), members[b].begin(), members[b].end());
        items.erase(items.begin() + static_cast<int64_t>(b));
        members.erase(members.begin() + static_cast<int64_t>(b));
        joined = true;
        break;
      }
    }
  }
  std::vector<char> taken(n, 0);
  for (const auto& t : items) {
    for (int i : t) taken[static_cast<size_t>(i)] = 1;
  }
  std::vector<std::vector<OcrCell>> grids(items.size());
  for (size_t t = 0; t < items.size(); ++t) {
    std::vector<int>& it = items[t];
    std::vector<int> added;  // the other blocks' lines it takes, given back where it finds no grid
    auto take = [&](int i) {
      it.push_back(i);
      added.push_back(i);
      taken[static_cast<size_t>(i)] = 1;
    };
    // 2. its extent: the lines inside its box, then the rows beside it that fall in its columns
    const LineBox u = g.extent(it);
    for (size_t i = 0; i < n; ++i) {
      const LineBox& b = page.lines[i].box;
      const bool within = b.x0 >= u.x0 && b.x1 <= u.x1 && b.y0 >= u.y0 && b.y1 <= u.y1;  // wholly in its box
      if (!taken[i] && centre_x(b) >= u.x0 && centre_x(b) <= u.x1 && centre_y(b) >= u.y0 && centre_y(b) <= u.y1 &&
          (within || chars_of(page.lines[i].text) <= rule.inside_chars)) {
        take(static_cast<int>(i));
      }
    }
    const double h = g.height(it);
    const Bands bands = g.columns(g.rows(it, h));
    if (bands.size() >= 2 && rule.grow_gap > 0) {
      std::vector<int> others;
      for (size_t i = 0; i < n; ++i) {
        const LineBox& b = page.lines[i].box;
        if (!taken[i] && b.x0 >= u.x0 - h && b.x1 <= u.x1 + h) others.push_back(static_cast<int>(i));
      }
      const std::vector<std::vector<int>> rows = g.rows(others, h);
      for (const int dir : {1, -1}) {  // down from its bottom, then up from its top
        double edge = dir > 0 ? g.extent(it).y1 : g.extent(it).y0;
        std::vector<size_t> by(rows.size());
        std::iota(by.begin(), by.end(), size_t{0});
        if (dir < 0) std::reverse(by.begin(), by.end());
        for (size_t k : by) {
          const LineBox r = g.extent(rows[k]);
          if ((dir > 0 && r.y0 < edge - 0.5 * h) || (dir < 0 && r.y1 > edge + 0.5 * h)) continue;  // not past it
          if ((dir > 0 ? r.y0 - edge : edge - r.y1) > rule.grow_gap * h) break;
          std::vector<int> cols;
          std::vector<double> chars;
          for (int i : rows[k]) {
            cols.push_back(g.column(i, bands).first);
            chars.push_back(chars_of(page.lines[static_cast<size_t>(i)].text));
          }
          std::sort(cols.begin(), cols.end());
          if (rows[k].size() < 2 || std::adjacent_find(cols.begin(), cols.end()) != cols.end() ||
              median_of(chars, 0) > rule.grow_chars) {
            break;
          }
          for (int i : rows[k]) {
            if (!taken[static_cast<size_t>(i)]) take(i);
          }
          edge = dir > 0 ? r.y1 : r.y0;
        }
      }
    }
    grids[t] = g.grid(it);
    if (grids[t].empty()) {  // no two columns: the model's cells, its blocks as they were
      for (int i : added) taken[static_cast<size_t>(i)] = 0;
    }
  }
  // The page again: each table where its first block was, its lines cell by cell; the other blocks without the
  // lines the tables took; the lines in the blocks' order.
  std::vector<int> table_of_block(page.blocks.size(), -1);
  std::vector<char> in_grid(n, 0);
  bool any = false;
  for (size_t t = 0; t < items.size(); ++t) {
    if (grids[t].empty()) continue;
    any = true;
    for (size_t b : members[t]) table_of_block[b] = static_cast<int>(t);
    for (const OcrCell& c : grids[t]) {
      for (int i : c.lines) in_grid[static_cast<size_t>(i)] = 1;
    }
  }
  if (!any) return;
  std::vector<int> order;  // the page's lines (their indices as they were) in their new order
  order.reserve(n);
  std::vector<OcrBlock> blocks;
  std::vector<char> placed(items.size(), 0), emitted(n, 0);
  std::vector<int> remap(n, -1);
  for (size_t b = 0; b < page.blocks.size(); ++b) {
    const int t = table_of_block[b];
    if (t >= 0) {
      if (placed[static_cast<size_t>(t)]) continue;
      placed[static_cast<size_t>(t)] = 1;
      OcrBlock tb;
      tb.label = "Table";
      for (size_t m : members[static_cast<size_t>(t)]) tb.confidence = std::max(tb.confidence, page.blocks[m].confidence);
      for (const OcrCell& c : grids[static_cast<size_t>(t)]) {
        OcrCell cell = c;
        cell.lines.clear();
        for (int i : c.lines) {
          cell.lines.push_back(static_cast<int>(order.size()));
          tb.lines.push_back(static_cast<int>(order.size()));
          order.push_back(i);
          emitted[static_cast<size_t>(i)] = 1;
        }
        tb.cells.push_back(std::move(cell));
      }
      blocks.push_back(std::move(tb));
      continue;
    }
    OcrBlock ob = page.blocks[b];
    ob.lines.clear();
    for (int i : page.blocks[b].lines) {
      if (in_grid[static_cast<size_t>(i)] || emitted[static_cast<size_t>(i)]) continue;
      remap[static_cast<size_t>(i)] = static_cast<int>(order.size());
      ob.lines.push_back(static_cast<int>(order.size()));
      order.push_back(i);
      emitted[static_cast<size_t>(i)] = 1;
    }
    if (ob.lines.empty()) continue;
    for (OcrCell& c : ob.cells) {  // a table that kept the model's cells
      std::vector<int> kept;
      for (int i : c.lines) {
        if (remap[static_cast<size_t>(i)] >= 0) kept.push_back(remap[static_cast<size_t>(i)]);
      }
      c.lines = std::move(kept);
    }
    std::erase_if(ob.cells, [](const OcrCell& c) { return c.lines.empty(); });
    blocks.push_back(std::move(ob));
  }
  for (size_t i = 0; i < n; ++i) {  // a line in no block (where the model gave blocks, none): kept, at the end
    if (emitted[i]) continue;
    blocks.push_back({"Text", 0.0f, {static_cast<int>(order.size())}, {}});
    order.push_back(static_cast<int>(i));
  }
  std::vector<OcrLine> lines;
  lines.reserve(n);
  for (int i : order) lines.push_back(std::move(page.lines[static_cast<size_t>(i)]));
  page.lines = std::move(lines);
  page.blocks = std::move(blocks);
}

}  // namespace carbon::detail
