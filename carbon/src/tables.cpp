// Tables by geometry inside the order model's Table blocks (carbon/engine.h: GridRule, EngineOptions::grid).

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <iterator>
#include <map>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "carbon/engine.h"
#include "engine_steps.h"
#include "table_check_weights.h"

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

// ------------------------------------------------------------------ a cell's text: where it goes on

// The code point starting at s[i] and its length in bytes (a byte that starts no well-formed sequence: itself).
std::pair<uint32_t, size_t> code_at(std::string_view s, size_t i) {
  const unsigned char c = static_cast<unsigned char>(s[i]);
  const size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
  if (n == 1 || i + n > s.size()) return {c, 1};
  uint32_t cp = c & (0x7Fu >> n);
  for (size_t k = 1; k < n; ++k) {
    const unsigned char d = static_cast<unsigned char>(s[i + k]);
    if ((d & 0xC0) != 0x80) return {c, 1};
    cp = cp << 6 | (d & 0x3Fu);
  }
  return {cp, n};
}

// A lowercase letter, as the writers' (output.cpp): ASCII; Latin-1; Latin Extended-A and the regular pairs of
// Extended-B and Extended Additional; IPA; Greek and Cyrillic.
bool is_lower(uint32_t c) {
  if (c < 0x80) return c >= 'a' && c <= 'z';
  if (c < 0x100) return c >= 0xDF && c != 0xF7;
  if (c <= 0x137) return c % 2 == 1;
  if (c == 0x138 || c == 0x149 || c == 0x17F) return true;
  if (c <= 0x148) return c % 2 == 0;
  if (c <= 0x177) return c % 2 == 1;
  if (c <= 0x17E) return c >= 0x179 && c % 2 == 0;
  if (c <= 0x24F) {
    if (c >= 0x1CD && c <= 0x1DC) return c % 2 == 0;
    return ((c >= 0x1DE && c <= 0x1EF) || c >= 0x1F8) && c % 2 == 1;
  }
  if (c <= 0x2AF) return true;
  if (c >= 0x3AC && c <= 0x3CE) return true;
  if (c >= 0x430 && c <= 0x45F) return true;
  if ((c >= 0x460 && c <= 0x481) || (c >= 0x48A && c <= 0x4BF) || (c >= 0x4D0 && c <= 0x52F)) return c % 2 == 1;
  if (c >= 0x1E00 && c <= 0x1EFF) return c % 2 == 1;
  return false;
}

// A letter, as the writers' (output.cpp): ASCII's; above, any code point outside the punctuation, symbols, spacing
// modifiers and combining marks.
bool is_letter(uint32_t c) {
  if (c < 0x80) return (c | 0x20) >= 'a' && (c | 0x20) <= 'z';
  if (c < 0xC0) return c == 0xAA || c == 0xB5 || c == 0xBA;
  if (c == 0xD7 || c == 0xF7) return false;
  if (c >= 0x2B0 && c <= 0x36F) return false;
  if (c >= 0x2000 && c <= 0x2BFF) return false;
  if (c >= 0x3000 && c <= 0x303F) return false;
  if (c >= 0xFF00 && c <= 0xFF20) return false;
  return true;
}

bool has_letter(std::string_view t) {
  for (size_t i = 0; i < t.size();) {
    const auto [c, n] = code_at(t, i);
    if (is_letter(c)) return true;
    i += n;
  }
  return false;
}

std::string_view trimmed(std::string_view t) {
  while (!t.empty() && (t.front() == ' ' || t.front() == '\t')) t.remove_prefix(1);
  while (!t.empty() && (t.back() == ' ' || t.back() == '\t')) t.remove_suffix(1);
  return t;
}

// A cell's line that goes on onto the next: a bracket left open, or a hyphen or dash, a comma, a slash, a plus, an
// ampersand or an opening bracket at its end, or a word that joins what follows it ("and", "of", "vs.").
bool ends_open(std::string_view t) {
  t = trimmed(t);
  if (t.empty()) return false;
  if (std::count(t.begin(), t.end(), '(') > std::count(t.begin(), t.end(), ')') ||
      std::count(t.begin(), t.end(), '[') > std::count(t.begin(), t.end(), ']')) {
    return true;
  }
  size_t last = t.size() - 1;
  while (last > 0 && (static_cast<unsigned char>(t[last]) & 0xC0) == 0x80) --last;
  const uint32_t end = code_at(t, last).first;
  if (end == '-' || end == ',' || end == '/' || end == '+' || end == '&' || end == '(' || end == 0x2010 ||
      end == 0x2011 || end == 0x2013) {
    return true;
  }
  const size_t space = t.find_last_of(" \t");
  std::string word(space == std::string_view::npos ? t : t.substr(space + 1));
  for (char& c : word) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  static constexpr std::string_view kJoins[] = {"and", "or",  "of", "the", "in",  "on",   "for", "with",
                                                "to",  "a",   "an", "by",  "at",  "from", "vs",  "vs.",
                                                "per", "than", "as", "not", "nor"};
  return std::find(std::begin(kJoins), std::end(kJoins), word) != std::end(kJoins);
}

// A line that carries on the cell above it: a lowercase letter, a closing bracket or a percent sign first, or a
// bracket that opens on a lowercase letter (a unit or a qualifier: "(months)", "(mg/l)").
bool starts_on(std::string_view t) {
  t = trimmed(t);
  if (t.empty()) return false;
  const auto [c, n] = code_at(t, 0);
  if (is_lower(c) || c == ')' || c == ']' || c == '%') return true;
  return c == '(' && t.size() > 1 && is_lower(code_at(t, 1).first);
}

// A table's caption line: "Table", "Tab.", "Tableau", "Tabelle", "Tabla" or "Tabela" in any case, then a number,
// a roman numeral, or a letter and a number ("Table 3", "TABLE II.", "Table S1", "Table1."), then the line's end or
// a mark that is no letter ("Table 2: ...", "Table 4 – ...").
bool is_caption(std::string_view t) {
  t = trimmed(t);
  std::string low(t.substr(0, 8));
  for (char& c : low) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  static constexpr std::string_view kWords[] = {"tableau", "tabelle", "tabela", "tabla", "table", "tab."};
  size_t k = std::string_view::npos;
  for (std::string_view w : kWords) {
    if (std::string_view(low).starts_with(w)) {
      k = w.size();
      break;
    }
  }
  if (k == std::string_view::npos) return false;
  while (k < t.size() && t[k] == ' ') ++k;
  auto digit = [&](size_t p) { return p < t.size() && t[p] >= '0' && t[p] <= '9'; };
  auto alpha = [&](size_t p) { return p < t.size() && std::isalpha(static_cast<unsigned char>(t[p])); };
  size_t j = k;
  if (digit(j) || (alpha(j) && digit(j + 1))) {  // a number (3, 1.4), or a letter and a number (S1)
    if (!digit(j)) ++j;
    while (digit(j) || (j < t.size() && t[j] == '.' && digit(j + 1))) ++j;
  } else {
    while (j < t.size() && std::string_view("ivxlcIVXLC").find(t[j]) != std::string_view::npos) ++j;
    if (j == k) return false;
  }
  return !alpha(j);
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

  // GridRule 4: the columns of the fullest rows, those that overlap joined; with label_band (the grid's, not the
  // extent's), a column before them for the lines wholly left of the first, where they are group labels: two or more,
  // and most of them on a row that has a line in the first column too, or alone on theirs (a label beside its group's
  // rows, or between them). A line left of the first column on a row whose first column it would be (its values on
  // the row, no other first-column line) is that column's, set out or in (a total, an indented stub), not a label.
  Bands columns(const std::vector<std::vector<int>>& rows, bool label_band = false) const {
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
    if (label_band && !bands.empty()) {
      const auto [b0, b1] = bands.front();
      const double w0 = std::max(1.0, b1 - b0);
      double l0 = 1e300, l1 = -1e300;
      int left = 0, labels = 0;
      for (const auto& r : rows) {
        int here = 0;
        bool first = false;  // a line of the row in the first column
        for (int i : r) {
          if (box(i).x1 < b0) {
            l0 = std::min(l0, box(i).x0);
            l1 = std::max(l1, box(i).x1);
            ++here;
          } else {
            first = first || overlap(box(i).x0, box(i).x1, b0, b1) > rule.cover * w0;
          }
        }
        left += here;
        if (first || here == static_cast<int>(r.size())) labels += here;
      }
      if (labels >= 2 && 2 * labels >= left) bands.insert(bands.begin(), {l0, l1});
    }
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

  // A line's first column and how many it spans: those it covers more than `cover` of (GridRule::cover, or a header
  // line's header_cover).
  std::pair<int, int> column(int i, const Bands& bands) const { return column(i, bands, rule.cover); }
  std::pair<int, int> column(int i, const Bands& bands, double cover) const {
    const LineBox& b = box(i);
    int first = -1, last = -1;
    for (size_t k = 0; k < bands.size(); ++k) {
      const double w = std::max(1.0, bands[k].second - bands[k].first);
      if (overlap(b.x0, b.x1, bands[k].first, bands[k].second) > cover * w) {
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

  // The line of `row` that a overlaps most across, and by how much (the first of the most).
  std::pair<size_t, double> most_over(int a, const std::vector<int>& row) const {
    size_t best = 0;
    double most = -1e300;
    for (size_t q = 0; q < row.size(); ++q) {
      const double v = overlap(box(a).x0, box(a).x1, box(row[q]).x0, box(row[q]).x1);
      if (v > most) {
        most = v;
        best = q;
      }
    }
    return {best, most};
  }

  // GridRule 5's body continuation: row r goes on the cells of the row above it. Fewer lines than that row, under
  // half the columns, none a number alone; each line under a different line of that row, within wrap_gap line
  // heights of white of it, and read as going on from it (it starts on a lowercase letter, a closing bracket or a
  // unit, or the line above it ends open: a bracket, a hyphen, a comma, a joining word).
  bool continues_body(const std::vector<int>& above, const std::vector<int>& r, size_t columns, double h) const {
    if (r.size() >= above.size() || 2 * r.size() >= columns) return false;
    std::vector<size_t> hit;
    for (int a : r) {
      if (number(a)) return false;
      const auto [q, most] = most_over(a, above);
      if (most <= 0 || std::find(hit.begin(), hit.end(), q) != hit.end()) return false;
      hit.push_back(q);
      const int j = above[q];
      if (box(a).y0 - box(j).y1 > rule.wrap_gap * h) return false;
      if (!starts_on(lines[static_cast<size_t>(a)].text) && !ends_open(lines[static_cast<size_t>(j)].text)) return false;
    }
    return true;
  }

  // GridRule 5 and the cells: rows joined where a row is a cell's next lines, each row's lines in its columns (two
  // in one column one cell), a cell's lines down the page. Empty where the lines make no two columns.
  std::vector<OcrCell> grid(const std::vector<int>& items) const {
    const double h = height(items);
    const std::vector<std::vector<int>> rows = this->rows(items, h);
    const Bands bands = columns(rows, rule.label_band);
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
    size_t first_value = rows.size();  // the first row that holds one: the rows above are the header's
    for (size_t k = 0; k < rows.size() && first_value == rows.size(); ++k) {
      if (std::any_of(rows[k].begin(), rows[k].end(), number)) first_value = k;
    }
    const bool header = first_full <= static_cast<size_t>(rule.header_rows);  // a header of a few lines at most
    std::vector<std::vector<std::vector<int>>> joined(1);
    for (int i : rows[0]) joined.back().push_back({i});
    std::vector<size_t> began{0};  // the row each joined row's cells began with
    size_t prev = 0;               // the last joined row's
    for (size_t k = 1; k < rows.size(); ++k) {
      const std::vector<int>& r = rows[k];
      const bool in_header = header && k < first_full;
      bool join = r.size() <= rows[prev].size() && std::none_of(r.begin(), r.end(), number) &&
                  (in_header || cy[k] - cy[k - 1] < rule.wrap_pitch * mp);
      std::vector<size_t> hit;
      for (size_t a = 0; join && a < r.size(); ++a) {  // each over a different line of that row
        const auto [best, most] = most_over(r[a], rows[prev]);
        join = most > 0 && std::find(hit.begin(), hit.end(), best) == hit.end();
        hit.push_back(best);
      }
      if (join) {
        for (size_t a = 0; a < r.size(); ++a) joined.back()[hit[a]].push_back(r[a]);
        continue;
      }
      if (rule.body_wrap && !in_header && continues_body(rows[k - 1], r, bands.size(), h)) {
        // each line into the cell of the last joined row that holds the line it lies under
        std::vector<size_t> to;
        for (int a : r) {
          const int j = rows[k - 1][most_over(a, rows[k - 1]).first];
          for (size_t g = 0; g < joined.back().size(); ++g) {
            const std::vector<int>& cell = joined.back()[g];
            if (std::find(cell.begin(), cell.end(), j) == cell.end()) continue;
            if (std::find(to.begin(), to.end(), g) == to.end()) to.push_back(g);
            break;
          }
        }
        if (to.size() == r.size()) {
          for (size_t a = 0; a < r.size(); ++a) joined.back()[to[a]].push_back(r[a]);
          continue;
        }
      }
      joined.emplace_back();
      for (int i : r) joined.back().push_back({i});
      began.push_back(k);
      prev = k;
    }
    // the header's joined rows: those begun above the first row with a value
    size_t header_joined = 0;
    while (header_joined < joined.size() && began[header_joined] < first_value) ++header_joined;
    std::vector<OcrCell> cells;
    for (size_t r = 0; r < joined.size(); ++r) {
      std::map<int, size_t> at;  // the cell of each column of this row
      // a header line spans the columns it covers header_cover of (a heading over the columns it heads)
      const double cover = r < header_joined && rule.header_cover > 0 ? rule.header_cover : rule.cover;
      for (const std::vector<int>& lines : joined[r]) {
        const auto [c, span] = column(lines.front(), bands, cover);
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
    const int nrows = header_spans(cells, static_cast<int>(joined.size()));
    row_labels(cells, nrows);
    for (OcrCell& c : cells) {
      std::stable_sort(c.lines.begin(), c.lines.end(), [&](int a, int b) {
        return centre_y(box(a)) != centre_y(box(b)) ? centre_y(box(a)) < centre_y(box(b)) : box(a).x0 < box(b).x0;
      });
    }
    if (rule.section_rows) section_rows(cells, nrows, static_cast<int>(bands.size()));
    std::stable_sort(cells.begin(), cells.end(),
                     [](const OcrCell& x, const OcrCell& y) { return x.row != y.row ? x.row < y.row : x.col < y.col; });
    return cells;
  }

  // The first row that holds a number (a value): the rows above it are the header's.
  int first_value_row(const std::vector<OcrCell>& cells, int nrows) const {
    std::vector<char> value(static_cast<size_t>(nrows), 0);
    for (const OcrCell& c : cells) {
      for (int i : c.lines) value[static_cast<size_t>(c.row)] |= number(i);
    }
    int data = 0;
    while (data < nrows && !value[static_cast<size_t>(data)]) ++data;
    return data;
  }

  // GridRule 8: a section row spans the table. In a table of three columns or more, a body row (at or below the
  // first that holds a number) of a single cell, not in the first column and under no cell that spans rows, whose
  // lines hold a letter, are centred in the table (their middle within section_centre of the table's width of its
  // middle) and do not begin as a cell's next line (lowercase first, or set where the line above it in its column
  // starts, within half a line height: a wrapped cell's), is a heading of the rows below it: it spans every column.
  void section_rows(std::vector<OcrCell>& cells, int nrows, int ncols) const {
    if (ncols < 3 || cells.empty()) return;
    const int data = first_value_row(cells, nrows);
    double t0 = 1e300, t1 = -1e300;  // the table across
    std::vector<int> all;
    for (const OcrCell& c : cells) {
      for (int i : c.lines) {
        t0 = std::min(t0, box(i).x0);
        t1 = std::max(t1, box(i).x1);
        all.push_back(i);
      }
    }
    const double h = height(all);
    auto under_its_column = [&](int i) {  // set where the line just above it, across from it, starts
      int up = -1;
      for (int j : all) {
        if (j == i || box(j).y1 > box(i).y0 + 0.5 * h || overlap(box(i).x0, box(i).x1, box(j).x0, box(j).x1) <= 0) {
          continue;
        }
        if (up < 0 || box(j).y1 > box(up).y1) up = j;
      }
      return up >= 0 && std::abs(box(up).x0 - box(i).x0) <= 0.5 * h;
    };
    std::map<int, std::vector<size_t>> by_row;  // the cells each row begins
    for (size_t k = 0; k < cells.size(); ++k) by_row[cells[k].row].push_back(k);
    for (const auto& [r, ks] : by_row) {
      if (r < data || ks.size() != 1) continue;
      OcrCell& c = cells[ks.front()];
      if (c.col == 0 || c.row_span != 1 || c.lines.empty()) continue;
      double x0 = 1e300, x1 = -1e300;
      bool letters = false;
      for (int i : c.lines) {
        x0 = std::min(x0, box(i).x0);
        x1 = std::max(x1, box(i).x1);
        letters = letters || has_letter(lines[static_cast<size_t>(i)].text);
      }
      if (!letters || std::abs((x0 + x1) / 2 - (t0 + t1) / 2) > rule.section_centre * (t1 - t0)) continue;
      if (starts_on(lines[static_cast<size_t>(c.lines.front())].text) || under_its_column(c.lines.front())) continue;
      const bool under = std::any_of(cells.begin(), cells.end(), [&](const OcrCell& o) {
        return &o != &c && o.row < r && r < o.row + o.row_span;
      });
      if (under) continue;
      c.col = 0;
      c.col_span = ncols;
    }
  }

  // GridRule 7: header cells set between two header rows span both: a heading such as "Route" or "Year" beside a
  // heading that spans columns, level with neither it nor the headings under it. A header row (one above the first
  // row that holds a number) each of whose cells takes only columns that neither the row above nor the row below
  // has, and whose lines' middle lies between theirs, joins the row above, each of its cells two rows tall. The
  // rows after it move up one. Returns the rows left.
  int header_spans(std::vector<OcrCell>& cells, int nrows) const {
    std::vector<char> value(static_cast<size_t>(nrows), 0);
    for (const OcrCell& c : cells) {
      for (int i : c.lines) value[static_cast<size_t>(c.row)] |= number(i);
    }
    int data = 0;
    while (data < nrows && !value[static_cast<size_t>(data)]) ++data;
    auto covers = [&](int r, int col) {  // a cell of row r, or one spanning down into it, over this column
      for (const OcrCell& c : cells) {
        if (c.row <= r && r < c.row + c.row_span && col >= c.col && col < c.col + c.col_span) return true;
      }
      return false;
    };
    auto middle = [&](int r) {  // the mean middle of the lines of row r's own cells
      double s = 0;
      int n = 0;
      for (const OcrCell& c : cells) {
        if (c.row != r) continue;
        for (int i : c.lines) {
          s += centre_y(box(i));
          ++n;
        }
      }
      return n ? s / n : std::numeric_limits<double>::quiet_NaN();
    };
    for (int r = 1; r + 1 < data;) {
      bool apart = true, any = false;
      for (const OcrCell& c : cells) {
        if (c.row != r) continue;
        any = true;
        for (int k = c.col; k < c.col + c.col_span && apart; ++k) apart = !covers(r - 1, k) && !covers(r + 1, k);
      }
      const double above = middle(r - 1), here = middle(r), below = middle(r + 1);
      if (!any || !apart || !(above < here && here < below)) {
        ++r;
        continue;
      }
      for (OcrCell& c : cells) {
        if (c.row == r) {
          c.row = r - 1;
          c.row_span = 2;
        } else if (c.row > r) {
          --c.row;
        }
      }
      --nrows;
      --data;
    }
    return nrows;
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

  // GridRule 2's heading above a table: whether `row`, the row of lines just above the table `table`, heads its
  // columns. Each of its lines inside the table across (within a line height), with a letter, not a number alone
  // or a caption, a phrase (not begun as a line's continuation, a lowercase letter first, nor ended as a sentence,
  // a full stop after a word: a caption's or a paragraph's line), the whole of its block (a heading of one line),
  // each in a column of its own; and one of them over two columns or more (by header_cover), not over the first,
  // its middle over them.
  bool heads_columns(const std::vector<int>& row, const Bands& bands, const LineBox& table, double h,
                     const std::vector<int>& block_of, const std::vector<OcrBlock>& blocks,
                     const std::vector<char>& caption) const {
    const double cover = rule.header_cover > 0 ? rule.header_cover : rule.cover;
    std::vector<int> cols;
    bool heading = false;
    for (int i : row) {
      const LineBox& b = box(i);
      const std::string& t = lines[static_cast<size_t>(i)].text;
      if (b.x0 < table.x0 - h || b.x1 > table.x1 + h) return false;
      if (number(i) || caption[static_cast<size_t>(i)] || !has_letter(t) || starts_on(t)) return false;
      const std::string_view s = trimmed(t);
      if (s.size() >= 2 && s.back() == '.' &&
          (std::isalnum(static_cast<unsigned char>(s[s.size() - 2])) || s[s.size() - 2] == ')' || s[s.size() - 2] == ']')) {
        return false;
      }
      if (const int k = block_of[static_cast<size_t>(i)]; k >= 0) {
        for (int j : blocks[static_cast<size_t>(k)].lines) {
          if (std::find(row.begin(), row.end(), j) == row.end()) return false;
        }
      }
      cols.push_back(column(i, bands, cover).first);
      int first = -1, last = -1;
      for (size_t c = 0; c < bands.size(); ++c) {
        const double w = std::max(1.0, bands[c].second - bands[c].first);
        if (overlap(b.x0, b.x1, bands[c].first, bands[c].second) > cover * w) {
          if (first < 0) first = static_cast<int>(c);
          last = static_cast<int>(c);
        }
      }
      heading = heading || (first >= 1 && last > first && centre_x(b) >= bands[static_cast<size_t>(first)].first &&
                            centre_x(b) <= bands[static_cast<size_t>(last)].second);
    }
    std::sort(cols.begin(), cols.end());
    return heading && std::adjacent_find(cols.begin(), cols.end()) == cols.end();
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

// A mathematical sign: = + < > ^ _ ± × ÷ − and the operators, relations and arrows; a Greek letter; a letter of
// the mathematical alphabets.
bool math_sign(uint32_t c) {
  if (c < 0x80) return c == '=' || c == '+' || c == '<' || c == '>' || c == '^' || c == '_';
  if (c == 0xB1 || c == 0xD7 || c == 0xF7 || c == 0x2212) return true;
  if (c >= 0x391 && c <= 0x3C9) return true;
  if (c >= 0x2190 && c <= 0x22FF) return true;
  return c >= 0x1D400 && c <= 0x1D7FF;
}

}  // namespace

const std::vector<std::string>& table_feature_names() {
  static const std::vector<std::string> names = {
      "cols",  "rows",   "fill",      "multi_rows", "no_grid",   "cell_chars", "numbers", "digits",   "sentences",
      "words", "math",   "lower_first", "lines",    "width",     "height_cv",  "gutter",  "pitch_mad", "p_table",
      "p_text", "p_formula", "p_picture", "p_list", "h_rules",   "v_rules",    "pictures", "page_share"};
  return names;
}

std::vector<float> table_features(const OcrPage& page, const std::vector<int>& lines, const std::vector<OcrCell>& cells,
                                  std::span<const float> probs, const GridRule& rule) {
  std::vector<float> f(static_cast<size_t>(kTableFeatures), 0.0f);
  if (lines.empty()) return f;
  const Lines g{page.lines, rule};
  auto text = [&](int i) -> const std::string& { return page.lines[static_cast<size_t>(i)].text; };
  const double h = g.height(lines);
  const LineBox u = g.extent(lines);
  const std::vector<std::vector<int>> rows = g.rows(lines, h);
  const double n = static_cast<double>(lines.size());
  // the grid: its columns and rows, how full, how many of its rows have two cells or more
  int R = static_cast<int>(rows.size()), C = 1;
  if (!cells.empty()) {
    R = C = 0;
    for (const OcrCell& c : cells) {
      R = std::max(R, c.row + c.row_span);
      C = std::max(C, c.col + c.col_span);
    }
  }
  f[0] = static_cast<float>(std::log1p(C));
  f[1] = static_cast<float>(std::log1p(R));
  double area = 0;
  std::map<int, int> per_row;
  for (const OcrCell& c : cells) {
    area += static_cast<double>(c.row_span) * c.col_span;
    ++per_row[c.row];
  }
  f[2] = cells.empty() ? 1.0f : static_cast<float>(std::min(1.0, area / std::max(1, R * C)));
  int multi = 0;
  for (const auto& [r, k] : per_row) multi += k >= 2;
  f[3] = cells.empty() ? 0.0f : static_cast<float>(multi) / static_cast<float>(std::max(1, R));
  f[4] = cells.empty() ? 1.0f : 0.0f;
  // the text
  std::vector<double> chars;
  if (cells.empty()) {
    for (int i : lines) chars.push_back(chars_of(text(i)));
  } else {
    for (const OcrCell& c : cells) {
      int k = 0;
      for (int i : c.lines) k += chars_of(text(i));
      chars.push_back(k);
    }
  }
  f[5] = static_cast<float>(std::log1p(median_of(chars, 0)));
  int numbers = 0, sentences = 0, lower = 0, words = 0;
  double digits = 0, maths = 0, visible = 0;
  for (int i : lines) {
    const std::string& t = text(i);
    numbers += g.number(i);
    const std::string_view first = trimmed(t);
    lower += !first.empty() && is_lower(code_at(first, 0).first);
    int w = 0;
    bool in_word = false;
    for (size_t k = 0; k < t.size();) {
      const auto [c, len] = code_at(t, k);
      k += len;
      const bool space = c == ' ' || c == '\t';
      if (!space) {
        ++visible;
        digits += c >= '0' && c <= '9';
        maths += math_sign(c);
      }
      w += !space && !in_word;
      in_word = !space;
    }
    words += w;
    const std::string_view s = trimmed(t);
    if (w >= 4 && s.size() >= 2 && (s.back() == '.' || s.back() == '?' || s.back() == '!') &&
        std::isalpha(static_cast<unsigned char>(s[s.size() - 2]))) {
      ++sentences;
    }
  }
  f[6] = static_cast<float>(numbers / n);
  f[7] = static_cast<float>(visible > 0 ? digits / visible : 0);
  f[8] = static_cast<float>(sentences / n);
  f[9] = static_cast<float>(std::log1p(words / n));
  f[10] = static_cast<float>(visible > 0 ? maths / visible : 0);
  f[11] = static_cast<float>(lower / n);
  // the lines' geometry: how many, the table's width on the page, their heights' spread, the gutters between the
  // lines of a row, the rows' pitch
  f[12] = static_cast<float>(std::log1p(n));
  const double pw = page.rotation == 90 || page.rotation == 270 ? page.height : page.width;
  f[13] = pw > 0 ? static_cast<float>(std::min(1.0, (u.x1 - u.x0) / pw)) : 0.0f;
  double hs = 0, hs2 = 0;
  for (int i : lines) {
    const double v = g.box(i).y1 - g.box(i).y0;
    hs += v;
    hs2 += v * v;
  }
  const double hm = hs / n;
  f[14] = hm > 0 ? static_cast<float>(std::sqrt(std::max(0.0, hs2 / n - hm * hm)) / hm) : 0.0f;
  std::vector<double> gutters, centres;
  for (const std::vector<int>& r : rows) {
    double s = 0;
    for (int i : r) s += centre_y(g.box(i));
    centres.push_back(s / static_cast<double>(r.size()));
    if (r.size() < 2) continue;
    std::vector<double> gaps;
    for (size_t k = 1; k < r.size(); ++k) gaps.push_back(g.box(r[k]).x0 - g.box(r[k - 1]).x1);
    gutters.push_back(median_of(std::move(gaps), 0));
  }
  f[15] = static_cast<float>(std::max(0.0, median_of(std::move(gutters), 0)) / h);
  if (centres.size() >= 3) {
    std::vector<double> pitch;
    for (size_t k = 1; k < centres.size(); ++k) pitch.push_back(centres[k] - centres[k - 1]);
    const double mp = median_of(pitch, 0);
    std::vector<double> dev;
    for (double p : pitch) dev.push_back(std::abs(p - mp));
    f[16] = mp > 0 ? static_cast<float>(median_of(std::move(dev), 0) / mp) : 0.0f;
  }
  // the order model's label probabilities over its lines
  if (probs.size() == static_cast<size_t>(kBlockLabels)) {
    f[17] = probs[static_cast<size_t>(block_label("Table"))];
    f[18] = probs[static_cast<size_t>(block_label("Text"))];
    f[19] = probs[static_cast<size_t>(block_label("Formula"))];
    f[20] = probs[static_cast<size_t>(block_label("Picture"))];
    f[21] = probs[static_cast<size_t>(block_label("List-Group"))];
  } else {
    f[17] = 1.0f;
  }
  // the page's ruled lines over its rows (across half its width at least, within two line heights of it) and over
  // its columns (down half its height at least)
  int across = 0, down = 0;
  for (const LineBox& r : page.rules) {
    if (r.x1 - r.x0 >= r.y1 - r.y0) {
      across += centre_y(r) >= u.y0 - 2 * h && centre_y(r) <= u.y1 + 2 * h &&
                overlap(r.x0, r.x1, u.x0, u.x1) >= 0.5 * (u.x1 - u.x0);
    } else {
      down += centre_x(r) >= u.x0 - 2 * h && centre_x(r) <= u.x1 + 2 * h &&
              overlap(r.y0, r.y1, u.y0, u.y1) >= 0.5 * (u.y1 - u.y0);
    }
  }
  f[22] = static_cast<float>(std::min(1.0, across / (R + 1.0)));
  f[23] = static_cast<float>(std::min(1.0, down / (C + 1.0)));
  double pict = 0;
  const double ua = std::max(1.0, (u.x1 - u.x0) * (u.y1 - u.y0));
  for (const LineBox& p : page.pictures) {
    pict += std::max(0.0, overlap(p.x0, p.x1, u.x0, u.x1)) * std::max(0.0, overlap(p.y0, p.y1, u.y0, u.y1));
  }
  f[24] = static_cast<float>(std::min(1.0, pict / ua));
  f[25] = static_cast<float>(n / static_cast<double>(page.lines.size()));
  return f;
}

std::vector<OcrCell> table_grid(const OcrPage& page, const std::vector<int>& lines, const GridRule& rule) {
  return Lines{page.lines, rule}.grid(lines);
}

float table_probability(std::span<const float> features, bool text_layer) {
  const TableCheckWeights& w = text_layer ? kTableCheckTextLayer : kTableCheckOcr;
  double z = w.bias;
  for (int i = 0; i < kTableFeatures && i < static_cast<int>(features.size()); ++i) {
    z += w.weight[i] * (features[static_cast<size_t>(i)] - w.mean[i]) / w.scale[i];
  }
  return static_cast<float>(1.0 / (1.0 + std::exp(-z)));
}

std::vector<LineBox> find_rules(const GrayView& image, int page_width, int page_height, int dark) {
  std::vector<LineBox> out;
  const int W = image.width, H = image.height;
  if (!image.data || W <= 0 || H <= 0) return out;
  const double sx = static_cast<double>(page_width) / W, sy = static_cast<double>(page_height) / H;
  const int across = std::max(24, W / 30), down = std::max(24, H / 30), thick = std::max(3, std::min(W, H) / 200);
  struct Run {
    int a, b, y0, y1;  // columns [a, b), rows y0..y1
  };
  std::vector<Run> open, next, flat;
  std::vector<int> col(static_cast<size_t>(W), 0);
  std::vector<Run> tall;
  for (int y = 0; y < H; ++y) {
    const uint8_t* p = image.data + static_cast<size_t>(y) * image.stride;
    next.clear();
    for (int x = 0; x < W;) {
      if (p[x] > dark) {
        ++x;
        continue;
      }
      const int a = x;
      while (x < W && p[x] <= dark) ++x;
      if (x - a < across) continue;
      Run r{a, x, y, y};
      for (Run& o : open) {  // the same rule a row down: their spans across overlap by half the shorter's
        if (o.y1 == y - 1 && std::min(o.b, x) - std::max(o.a, a) > std::min(o.b - o.a, x - a) / 2) {
          r = {std::min(o.a, a), std::max(o.b, x), o.y0, y};
          o.y1 = -2;  // taken on
          break;
        }
      }
      next.push_back(r);
    }
    for (const Run& o : open) {
      if (o.y1 != -2) flat.push_back(o);
    }
    std::swap(open, next);
    for (int x = 0; x < W; ++x) {  // runs down each column
      int& k = col[static_cast<size_t>(x)];
      if (p[x] <= dark) {
        ++k;
      } else {
        if (k >= down) tall.push_back({x, x + 1, y - k, y - 1});
        k = 0;
      }
    }
  }
  flat.insert(flat.end(), open.begin(), open.end());
  for (int x = 0; x < W; ++x) {
    if (col[static_cast<size_t>(x)] >= down) tall.push_back({x, x + 1, H - col[static_cast<size_t>(x)], H - 1});
  }
  for (const Run& r : flat) {
    if (r.y1 - r.y0 + 1 <= thick) out.push_back({r.a * sx, r.y0 * sy, r.b * sx, (r.y1 + 1) * sy});
  }
  // the columns' runs side by side, overlapping down by half the shorter's, one rule
  std::stable_sort(tall.begin(), tall.end(), [](const Run& a, const Run& b) { return a.a < b.a; });
  std::vector<Run> joined;
  for (const Run& r : tall) {
    bool on = false;
    for (Run& j : joined) {
      if (j.b == r.a && std::min(j.y1, r.y1) - std::max(j.y0, r.y0) > std::min(j.y1 - j.y0, r.y1 - r.y0) / 2) {
        j = {j.a, r.b, std::min(j.y0, r.y0), std::max(j.y1, r.y1)};
        on = true;
        break;
      }
    }
    if (!on) joined.push_back(r);
  }
  for (const Run& r : joined) {
    if (r.b - r.a <= thick) out.push_back({r.a * sx, r.y0 * sy, r.b * sx, (r.y1 + 1) * sy});
  }
  return out;
}

void tables_by_geometry(OcrPage& page, const GridRule& rule, std::span<const std::vector<float>> block_probs) {
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
  std::vector<int> block_of(n, -1);  // each line's block, as the model gave them
  for (size_t b = 0; b < page.blocks.size(); ++b) {
    for (int i : page.blocks[b].lines) block_of[static_cast<size_t>(i)] = static_cast<int>(b);
  }
  std::vector<char> caption(n, 0);  // a table's caption ("Table 3", "TABLE II."): never a heading over its columns,
  for (size_t i = 0; i < n; ++i) caption[i] = is_caption(page.lines[i].text);  // nor, with caption_out, a cell
  std::vector<std::vector<OcrCell>> grids(items.size());
  std::vector<std::vector<int>> before(items.size()), after(items.size());  // its own lines out of it, as captions
  std::vector<char> rejected(items.size(), 0);                              // the false-table check's
  for (size_t t = 0; t < items.size(); ++t) {
    std::vector<int>& it = items[t];
    std::vector<int> added;  // the other blocks' lines it takes, given back where it finds no grid
    auto take = [&](int i) {
      it.push_back(i);
      added.push_back(i);
      taken[static_cast<size_t>(i)] = 1;
    };
    // 2. its extent: its own caption lines at its top or bottom out of it (caption_out); the lines inside its box,
    // then the rows beside it that fall in its columns, then a heading over its columns just above it (heading_up)
    if (rule.caption_out) {
      for (size_t k = 0; k < it.size();) {
        const int i = it[k];
        const LineBox& b = page.lines[static_cast<size_t>(i)].box;
        bool top = true, bottom = true;  // above (below) every other line of it
        for (int j : it) {
          if (j == i) continue;
          top = top && centre_y(b) < page.lines[static_cast<size_t>(j)].box.y0;
          bottom = bottom && centre_y(b) > page.lines[static_cast<size_t>(j)].box.y1;
        }
        if (!caption[static_cast<size_t>(i)] || it.size() < 3 || (!top && !bottom)) {
          ++k;
          continue;
        }
        (top ? before : after)[t].push_back(i);
        it.erase(it.begin() + static_cast<int64_t>(k));
      }
    }
    const LineBox u = g.extent(it);
    auto takeable = [&](size_t i) { return !taken[i] && !(rule.caption_out && caption[i]); };  // a line it may take
    for (size_t i = 0; i < n; ++i) {
      const LineBox& b = page.lines[i].box;
      const bool within = b.x0 >= u.x0 && b.x1 <= u.x1 && b.y0 >= u.y0 && b.y1 <= u.y1;  // wholly in its box
      if (takeable(i) && centre_x(b) >= u.x0 && centre_x(b) <= u.x1 && centre_y(b) >= u.y0 && centre_y(b) <= u.y1 &&
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
        if (takeable(i) && b.x0 >= u.x0 - h && b.x1 <= u.x1 + h) others.push_back(static_cast<int>(i));
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
    if (rule.heading_up && rule.head_gap > 0 && bands.size() >= 2) {
      // the row just above it, inside it across, with nothing between: a heading of its columns where one of its
      // lines lies over two of them or more, not over the first (a heading over the values, not the stub), its
      // middle over them, and every line of the row the whole of its block (a heading of one line, not a
      // paragraph's last), none a number alone or a caption, each in a column of its own
      const LineBox v = g.extent(it);
      const Bands now = g.columns(g.rows(it, h));
      std::vector<int> above;
      for (size_t i = 0; i < n; ++i) {
        const LineBox& b = page.lines[i].box;
        if (!taken[i] && b.y1 <= v.y0 + 0.5 * h && overlap(b.x0, b.x1, v.x0, v.x1) > 0) above.push_back(static_cast<int>(i));
      }
      const std::vector<std::vector<int>> rows = g.rows(above, h);
      if (!rows.empty() && g.heads_columns(rows.back(), now, v, h, block_of, page.blocks, caption)) {
        const LineBox r = g.extent(rows.back());
        if (v.y0 - r.y1 <= rule.head_gap * h) {
          for (int i : rows.back()) take(i);
        }
      }
    }
    grids[t] = g.grid(it);
    if (!grids[t].empty() && rule.title_out) {
      // 9. a title row: the grid's first row one cell, from its first column over two or more, its text with a
      // letter and no line a number alone, over a row of two cells or more and one row more at least, leaves the
      // table: its lines the table took from other blocks go back to them, its own as a caption before it
      const std::vector<OcrCell>& c = grids[t];
      int first = 0, second = 0, rows = 0;
      const OcrCell* lone = nullptr;
      for (const OcrCell& x : c) {
        if (x.row == 0) {
          ++first;
          lone = &x;
        }
        second += x.row == 1;
        rows = std::max(rows, x.row + x.row_span);
      }
      if (first == 1 && second >= 2 && rows >= 3 && lone->col == 0 && lone->col_span >= 2 && !lone->lines.empty() &&
          std::all_of(lone->lines.begin(), lone->lines.end(), [&](int i) {
            return has_letter(page.lines[static_cast<size_t>(i)].text) && !g.number(i);
          })) {
        const std::vector<int> title = lone->lines;
        std::vector<int> rest;
        for (int i : it) {
          if (std::find(title.begin(), title.end(), i) == title.end()) rest.push_back(i);
        }
        std::vector<OcrCell> again = g.grid(rest);
        if (!again.empty()) {
          for (int i : title) {
            if (std::find(added.begin(), added.end(), i) != added.end()) {
              taken[static_cast<size_t>(i)] = 0;
            } else {
              before[t].push_back(i);
            }
          }
          std::erase_if(added, [&](int i) { return std::find(title.begin(), title.end(), i) != title.end(); });
          it = std::move(rest);
          grids[t] = std::move(again);
        }
      }
    }
    if (rule.check) {
      // 10. the false-table check, over its grid's lines, or over its own blocks' lines where it has no grid; their
      // label probabilities the line-weighted mean of its blocks'
      std::vector<int> own;
      std::vector<float> probs(static_cast<size_t>(kBlockLabels), 0.0f);
      size_t weighed = 0;
      for (size_t b : members[t]) {
        own.insert(own.end(), page.blocks[b].lines.begin(), page.blocks[b].lines.end());
        if (b < block_probs.size() && block_probs[b].size() == probs.size()) {
          for (size_t c = 0; c < probs.size(); ++c) {
            probs[c] += block_probs[b][c] * static_cast<float>(page.blocks[b].lines.size());
          }
          weighed += page.blocks[b].lines.size();
        }
      }
      for (float& p : probs) p /= static_cast<float>(std::max<size_t>(1, weighed));
      if (!weighed) probs.clear();
      const std::vector<int>& lines = grids[t].empty() ? own : it;
      const std::vector<float> f = table_features(page, lines, grids[t], probs, rule);
      const float p = table_probability(f, page.source != "ocr");
      if (p < rule.check_keep) {
        grids[t].clear();
        rejected[t] = 1;
      }
    }
    if (grids[t].empty()) {  // no two columns: the model's cells, its blocks as they were
      for (int i : added) taken[static_cast<size_t>(i)] = 0;
      before[t].clear();
      after[t].clear();
    }
  }
  for (size_t t = 0; t < items.size(); ++t) {  // a false table: its blocks text, in the model's order
    if (!rejected[t]) continue;
    for (size_t b : members[t]) {
      page.blocks[b].label = "Text";
      page.blocks[b].cells.clear();
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
      const float confidence = tb.confidence;
      auto captions = [&](std::vector<int> at) {  // its own lines it left out, as a caption, down the page
        if (at.empty()) return;
        std::stable_sort(at.begin(), at.end(), [&](int a, int c) {
          return centre_y(page.lines[static_cast<size_t>(a)].box) < centre_y(page.lines[static_cast<size_t>(c)].box);
        });
        OcrBlock cb;
        cb.label = "Caption";
        cb.confidence = confidence;
        for (int i : at) {
          cb.lines.push_back(static_cast<int>(order.size()));
          order.push_back(i);
          emitted[static_cast<size_t>(i)] = 1;
        }
        blocks.push_back(std::move(cb));
      };
      captions(before[static_cast<size_t>(t)]);
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
      captions(after[static_cast<size_t>(t)]);
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
