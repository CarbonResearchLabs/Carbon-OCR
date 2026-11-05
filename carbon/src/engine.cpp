// The OCR pipeline v0 on the CPU (carbon/engine.h).

#include "carbon/engine.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <latch>
#include <exception>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <numeric>

#include "carbon/deskew.h"
#include "carbon/error.h"
#include "carbon/image.h"
#include "carbon/session.h"
#include "carbon/strips.h"
#include "det_head.h"
#include "engine_steps.h"
#include "net/onnx.h"

namespace carbon {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// ---------------------------------------------------------------- the vocabulary

void put_utf8(std::string& out, uint32_t c) {
  if (c < 0x80) {
    out += static_cast<char>(c);
  } else if (c < 0x800) {
    out += static_cast<char>(0xC0 | (c >> 6));
    out += static_cast<char>(0x80 | (c & 0x3F));
  } else if (c < 0x10000) {
    out += static_cast<char>(0xE0 | (c >> 12));
    out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (c & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (c >> 18));
    out += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (c & 0x3F));
  }
}

// A JSON string at s[i] (at its opening quote); i ends past its closing quote.
std::string json_string(const std::string& s, size_t& i) {
  auto bad = [] { throw Error("load_vocab: a malformed string"); };
  if (i >= s.size() || s[i] != '"') bad();
  std::string out;
  auto hex4 = [&](size_t at) {
    if (at + 4 > s.size()) bad();
    return static_cast<uint32_t>(std::stoul(s.substr(at, 4), nullptr, 16));
  };
  for (++i; i < s.size() && s[i] != '"'; ++i) {
    if (s[i] != '\\') {
      out += s[i];
      continue;
    }
    if (++i >= s.size()) bad();
    switch (s[i]) {
      case '"': out += '"'; break;
      case '\\': out += '\\'; break;
      case '/': out += '/'; break;
      case 'b': out += '\b'; break;
      case 'f': out += '\f'; break;
      case 'n': out += '\n'; break;
      case 'r': out += '\r'; break;
      case 't': out += '\t'; break;
      case 'u': {
        uint32_t c = hex4(i + 1);
        i += 4;
        if (c >= 0xD800 && c < 0xDC00 && i + 6 < s.size() && s[i + 1] == '\\' && s[i + 2] == 'u') {
          const uint32_t lo = hex4(i + 3);
          c = 0x10000 + ((c - 0xD800) << 10) + (lo - 0xDC00);
          i += 6;
        }
        put_utf8(out, c);
        break;
      }
      default: bad();
    }
  }
  if (i >= s.size()) bad();
  ++i;
  return out;
}

// ---------------------------------------------------------------- reading order

double lo(const LineBox& b, int axis) { return axis ? b.y0 : b.x0; }
double hi(const LineBox& b, int axis) { return axis ? b.y1 : b.x1; }

// The widest gap along an axis (0: x, 1: y) that no box crosses, and where it is.
std::pair<double, double> widest_gap(std::vector<size_t>& idx, std::span<const LineBox> b, int axis) {
  std::sort(idx.begin(), idx.end(), [&](size_t p, size_t q) { return lo(b[p], axis) < lo(b[q], axis); });
  double end = hi(b[idx[0]], axis), best = 0, at = 0;
  for (size_t k = 1; k < idx.size(); ++k) {
    const double start = lo(b[idx[k]], axis);
    if (start > end && start - end > best) {
      best = start - end;
      at = (start + end) / 2;
    }
    end = std::max(end, hi(b[idx[k]], axis));
  }
  return {best, at};
}

void xy_cut(std::vector<size_t> idx, std::span<const LineBox> b, std::vector<size_t>& out) {
  if (idx.size() <= 1) {
    out.insert(out.end(), idx.begin(), idx.end());
    return;
  }
  const auto [gx, atx] = widest_gap(idx, b, 0);
  const auto [gy, aty] = widest_gap(idx, b, 1);
  if (gx > 0 || gy > 0) {
    const int axis = gx > gy ? 0 : 1;  // a column gutter before the space between lines
    const double at = axis ? aty : atx;
    std::vector<size_t> before, after;
    for (size_t i : idx) (hi(b[i], axis) <= at ? before : after).push_back(i);
    xy_cut(std::move(before), b, out);
    xy_cut(std::move(after), b, out);
    return;
  }
  std::sort(idx.begin(), idx.end(), [&](size_t p, size_t q) {
    return b[p].y0 != b[q].y0 ? b[p].y0 < b[q].y0 : b[p].x0 < b[q].x0;
  });
  out.insert(out.end(), idx.begin(), idx.end());
}

// The XY-cut over blocks, its gaps in the page's own units: a gap across the page (a gutter) in median character
// widths `cw`, a gap down it in median line heights `lh`. The gutter wins where it is at least as wide in its units,
// so a page's columns are cut before the space between its paragraphs, which in raw pixels is often the wider.
void unit_xy_cut(std::vector<size_t> idx, std::span<const LineBox> b, double cw, double lh, std::vector<size_t>& out) {
  if (idx.size() <= 1) {
    out.insert(out.end(), idx.begin(), idx.end());
    return;
  }
  const auto [gx, atx] = widest_gap(idx, b, 0);
  const auto [gy, aty] = widest_gap(idx, b, 1);
  if (gx > 0 || gy > 0) {
    const int axis = gx > 0 && gx / cw >= gy / lh ? 0 : 1;
    const double at = axis ? aty : atx;
    std::vector<size_t> before, after;
    for (size_t i : idx) (hi(b[i], axis) <= at ? before : after).push_back(i);
    unit_xy_cut(std::move(before), b, cw, lh, out);
    unit_xy_cut(std::move(after), b, cw, lh, out);
    return;
  }
  std::sort(idx.begin(), idx.end(), [&](size_t p, size_t q) {
    return b[p].y0 != b[q].y0 ? b[p].y0 < b[q].y0 : b[p].x0 < b[q].x0;
  });
  out.insert(out.end(), idx.begin(), idx.end());
}

// ---------------------------------------------------------------- output

void json_escape(std::string& out, const std::string& s) {
  out += '"';
  for (unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          out += buf;
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  out += '"';
}

std::string box_json(const LineBox& b) {
  char buf[96];
  std::snprintf(buf, sizeof buf, "[%.1f, %.1f, %.1f, %.1f]", b.x0, b.y0, b.x1, b.y1);
  return buf;
}

std::vector<Tensor> one(Tensor t) {
  std::vector<Tensor> v;
  v.push_back(std::move(t));
  return v;
}

int round_up(int v, int step) { return (v + step - 1) / step * step; }

// ---------------------------------------------------------------- the gate's reading of a text

// A text's characters, spaces aside, by kind: letters and digits; rules and marks (_ - — – . | = ~ · • / \); the
// rest (punctuation, symbols).
struct TextKinds {
  int alnum = 0, marks = 0, other = 0;
  int chars() const { return alnum + marks + other; }
};

TextKinds kinds_of(const std::string& t) {
  TextKinds k;
  for (size_t i = 0; i < t.size();) {
    const unsigned char c = static_cast<unsigned char>(t[i]);
    size_t n = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 1;
    if (i + n > t.size()) n = 1;
    uint32_t cp = n == 1 ? c : c & (0x7Fu >> n);
    for (size_t j = 1; j < n; ++j) cp = cp << 6 | (static_cast<unsigned char>(t[i + j]) & 0x3Fu);
    i += n;
    if (cp == ' ' || cp == '\t') continue;
    switch (cp) {
      case '_': case '-': case '.': case '|': case '=': case '~': case '/': case '\\':
      case 0x2014: case 0x2013: case 0xB7: case 0x2022:
        ++k.marks;
        continue;
      default: break;
    }
    const bool ascii_alnum = cp < 0x80 && std::isalnum(static_cast<int>(cp));
    const bool letter = cp >= 0xC0 && cp != 0xD7 && cp != 0xF7 && !(cp >= 0x2B0 && cp <= 0x36F) &&
                        !(cp >= 0x2000 && cp <= 0x2BFF) && !(cp >= 0x3000 && cp <= 0x303F) &&
                        !(cp >= 0xFF00 && cp <= 0xFF20);
    ++(ascii_alnum || letter ? k.alnum : k.other);
  }
  return k;
}

// The characters of a text, by code points.
int code_points(const std::string& s) {
  int n = 0;
  for (unsigned char c : s) n += (c & 0xC0) != 0x80;
  return n;
}

}  // namespace

namespace detail {

// The crop rule a recognizer was trained with: its ONNX file's carbon.crop.* metadata
// (written at export) over `rule`. A file without them reads lines by `rule`.
CropRule model_crop(const std::filesystem::path& set, const std::string& rec, CropRule rule) {
  for (const char* kind : {"", ".fp16", ".int8", ".int8-sym"}) {
    const std::filesystem::path file = set / (rec + kind + ".onnx");
    if (!std::filesystem::is_regular_file(file)) continue;
    const std::map<std::string, std::string> meta = onnx::metadata(file);
    auto number = [&](const char* key, double lo, double hi, double& v) {
      const auto it = meta.find(key);
      if (it == meta.end()) return;
      const char* const end = it->second.data() + it->second.size();
      double x = 0;
      const auto [at, ec] = std::from_chars(it->second.data(), end, x);
      if (ec != std::errc() || at != end || !(x >= lo && x <= hi)) {
        throw Error(file.string() + ": " + key + " is \"" + it->second + "\"");
      }
      v = x;
    };
    double height = rule.height;
    number("carbon.crop.height", 8, 512, height);
    number("carbon.crop.pad_v", 0, 2, rule.pad_v);
    number("carbon.crop.pad_h", 0, 2, rule.pad_h);
    if (height != std::floor(height)) throw Error(file.string() + ": carbon.crop.height is not whole");
    rule.height = static_cast<int>(height);
    return rule;
  }
  return rule;
}

// How sure the set's orientation model must be to turn a page: its ONNX file's
// carbon.orient.turn_probability (calibrated on held-out pages at export), else 0.95.
float model_turn_probability(const std::filesystem::path& set) {
  for (const char* kind : {"", ".fp16", ".int8", ".int8-sym"}) {
    const std::filesystem::path file = set / (std::string("orient") + kind + ".onnx");
    if (!std::filesystem::is_regular_file(file)) continue;
    const std::map<std::string, std::string> meta = onnx::metadata(file);
    const auto it = meta.find("carbon.orient.turn_probability");
    if (it == meta.end()) return 0.95f;
    const char* const end = it->second.data() + it->second.size();
    double x = 0;
    const auto [at, ec] = std::from_chars(it->second.data(), end, x);
    if (ec != std::errc() || at != end || !(x >= 0.5 && x <= 1.0)) {
      throw Error(file.string() + ": carbon.orient.turn_probability is \"" + it->second + "\"");
    }
    return static_cast<float>(x);
  }
  return 0.95f;
}

}  // namespace detail

std::vector<std::string> load_vocab(const std::filesystem::path& file) {
  std::ifstream f(file, std::ios::binary);
  if (!f) throw Error("cannot open " + file.string());
  const std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  size_t i = s.find("\"chars\"");
  if (i == std::string::npos) throw Error(file.string() + ": no \"chars\"");
  i = s.find('[', i);
  if (i == std::string::npos) throw Error(file.string() + ": \"chars\" is not a list");
  std::vector<std::string> chars;
  for (++i; i < s.size();) {
    const char c = s[i];
    if (c == ']') return chars;
    if (c == '"') {
      chars.push_back(json_string(s, i));
    } else if (c == ',' || std::isspace(static_cast<unsigned char>(c))) {
      ++i;
    } else {
      throw Error(file.string() + ": \"chars\" holds something other than strings");
    }
  }
  throw Error(file.string() + ": \"chars\" is not closed");
}

std::vector<size_t> reading_order(std::span<const LineBox> boxes) {
  std::vector<size_t> idx(boxes.size()), out;
  std::iota(idx.begin(), idx.end(), size_t{0});
  out.reserve(boxes.size());
  xy_cut(std::move(idx), boxes, out);
  return out;
}

std::vector<OcrLine> split_lines(std::vector<OcrLine> lines, const SplitRule& rule) {
  std::vector<OcrLine> out;
  out.reserve(lines.size());
  auto chars = [](const std::string& s) {
    int64_t n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
  };
  for (OcrLine& l : lines) {
    const double h = std::max(1.0, l.box.y1 - l.box.y0);
    if (l.words.size() < 2) {
      out.push_back(std::move(l));
      continue;
    }
    std::vector<double> gaps, small;
    for (size_t i = 1; i < l.words.size(); ++i) {
      gaps.push_back((l.words[i].box.x0 - l.words[i - 1].box.x1) / h);
      if (gaps.back() < rule.min_gap) small.push_back(gaps.back());
    }
    double space = rule.space;
    if (!small.empty()) {
      std::nth_element(small.begin(), small.begin() + static_cast<int64_t>(small.size() / 2), small.end());
      space = small[small.size() / 2];
    }
    const double cut = std::max(rule.min_gap, rule.ratio * space);
    size_t from = 0;
    auto piece = [&](size_t a, size_t b) {  // words [a, b)
      if (a == 0 && b == l.words.size()) {
        out.push_back(std::move(l));
        return;
      }
      OcrLine p;
      p.box = {l.words[a].box.x0, l.box.y0, l.words[b - 1].box.x1, l.box.y1};
      p.det_score = l.det_score;
      p.least = l.least;  // the whole line's: its words keep no least of their own
      double conf = 0;
      int64_t n = 0;
      for (size_t i = a; i < b; ++i) {
        if (i > a) p.text += ' ';
        p.text += l.words[i].text;
        const int64_t c = chars(l.words[i].text);
        conf += static_cast<double>(l.words[i].confidence) * static_cast<double>(c);
        n += c;
        p.words.push_back(l.words[i]);
      }
      p.confidence = n ? static_cast<float>(conf / static_cast<double>(n)) : 0.0f;
      out.push_back(std::move(p));
    };
    for (size_t i = 1; i < l.words.size(); ++i) {
      if (gaps[i - 1] >= rule.always || gaps[i - 1] >= cut) {
        piece(from, i);
        from = i;
      }
    }
    piece(from, l.words.size());
  }
  return out;
}

int find_tables(OcrPage& page, const TableRule& rule) {
  const size_t n = page.lines.size();
  if (!page.blocks.empty() || n < static_cast<size_t>(rule.min_rows * rule.min_cols)) return 0;
  auto height = [&](size_t i) { return std::max(1.0, page.lines[i].box.y1 - page.lines[i].box.y0); };
  std::vector<double> hs;
  for (size_t i = 0; i < n; ++i) hs.push_back(height(i));
  std::nth_element(hs.begin(), hs.begin() + static_cast<int64_t>(n / 2), hs.end());
  const double h = hs[n / 2];
  // rows: the lines by their centres down the page, each joining the row of the one above where they overlap
  std::vector<size_t> by_y(n);
  std::iota(by_y.begin(), by_y.end(), size_t{0});
  auto cy = [&](size_t i) { return (page.lines[i].box.y0 + page.lines[i].box.y1) / 2; };
  std::sort(by_y.begin(), by_y.end(), [&](size_t a, size_t b) { return cy(a) < cy(b); });
  std::vector<std::vector<size_t>> rows;
  for (size_t i : by_y) {
    if (!rows.empty()) {
      const LineBox& f = page.lines[rows.back().front()].box;
      const LineBox& l = page.lines[i].box;
      const double ov = std::min(f.y1, l.y1) - std::max(f.y0, l.y0);
      if (ov >= rule.row_overlap * std::min(height(i), height(rows.back().front()))) {
        rows.back().push_back(i);
        continue;
      }
    }
    rows.push_back({i});
  }
  for (auto& r : rows) {
    std::sort(r.begin(), r.end(), [&](size_t a, size_t b) { return page.lines[a].box.x0 < page.lines[b].box.x0; });
  }
  auto cells_apart = [&](const std::vector<size_t>& r) {  // two cells or more, none over another
    if (r.size() < 2) return false;
    for (size_t k = 1; k < r.size(); ++k) {
      if (page.lines[r[k]].box.x0 < page.lines[r[k - 1]].box.x1) return false;
    }
    return true;
  };
  auto chars = [](const std::string& s) {
    int n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
  };
  struct Table {
    std::vector<std::vector<size_t>> rows;
    std::vector<std::pair<double, double>> cols;
  };
  std::vector<Table> tables;
  auto try_run = [&](size_t a, size_t b) {  // rows [a, b)
    if (b - a < static_cast<size_t>(rule.min_rows)) return;
    std::vector<std::pair<double, double>> spans, cols;
    std::vector<int> lens;
    for (size_t r = a; r < b; ++r) {
      for (size_t i : rows[r]) {
        spans.push_back({page.lines[i].box.x0, page.lines[i].box.x1});
        lens.push_back(chars(page.lines[i].text));
      }
    }
    std::nth_element(lens.begin(), lens.begin() + static_cast<int64_t>(lens.size() / 2), lens.end());
    if (lens[lens.size() / 2] > rule.max_cell_chars) return;  // columns of text
    std::sort(spans.begin(), spans.end());
    for (const auto& s : spans) {
      if (!cols.empty() && s.first < cols.back().second) {
        cols.back().second = std::max(cols.back().second, s.second);
      } else {
        cols.push_back(s);
      }
    }
    if (cols.size() < static_cast<size_t>(rule.min_cols)) return;
    auto col_of = [&](size_t i) {
      for (size_t c = 0; c < cols.size(); ++c) {
        if (page.lines[i].box.x0 >= cols[c].first && page.lines[i].box.x0 < cols[c].second) return c;
      }
      return cols.size();
    };
    for (size_t r = a; r < b; ++r) {  // each row's cells in columns of their own
      for (size_t k = 1; k < rows[r].size(); ++k) {
        if (col_of(rows[r][k]) == col_of(rows[r][k - 1])) return;
      }
    }
    tables.push_back({std::vector<std::vector<size_t>>(rows.begin() + static_cast<int64_t>(a),
                                                       rows.begin() + static_cast<int64_t>(b)),
                      cols});
  };
  size_t run = 0;
  for (size_t r = 0; r <= rows.size(); ++r) {
    const bool ok = r < rows.size() && cells_apart(rows[r]);
    const bool joins = ok && r > run && [&] {
      double top = 1e300, bottom = -1e300;
      for (size_t i : rows[r]) top = std::min(top, page.lines[i].box.y0);
      for (size_t i : rows[r - 1]) bottom = std::max(bottom, page.lines[i].box.y1);
      return top - bottom <= rule.max_row_gap * h;
    }();
    if (!joins) {
      try_run(run, r);
      run = ok ? r : r + 1;
    }
  }
  if (tables.empty()) return 0;

  // the page's new order: a table where its first line was, row by row; the rest as it was
  std::vector<int> table_of(n, -1);
  for (size_t t = 0; t < tables.size(); ++t) {
    for (const auto& r : tables[t].rows) {
      for (size_t i : r) table_of[i] = static_cast<int>(t);
    }
  }
  std::vector<OcrLine> lines;
  std::vector<OcrBlock> blocks;
  std::vector<bool> placed(tables.size(), false);
  for (size_t i = 0; i < n; ++i) {
    const int t = table_of[i];
    if (t >= 0) {
      if (placed[static_cast<size_t>(t)]) continue;
      placed[static_cast<size_t>(t)] = true;
      OcrBlock b;
      b.label = "Table";
      const Table& tb = tables[static_cast<size_t>(t)];
      for (size_t r = 0; r < tb.rows.size(); ++r) {
        for (size_t k : tb.rows[r]) {
          size_t c = 0;
          while (c + 1 < tb.cols.size() && page.lines[k].box.x0 >= tb.cols[c].second) ++c;
          b.lines.push_back(static_cast<int>(lines.size()));
          b.cells.push_back({static_cast<int>(r), static_cast<int>(c), {static_cast<int>(lines.size())}});
          lines.push_back(page.lines[k]);
        }
      }
      blocks.push_back(std::move(b));
      continue;
    }
    const OcrLine& l = page.lines[i];
    const bool paragraph = blocks.empty() || blocks.back().label != "Text" || [&] {
      const OcrLine& prev = lines.back();
      return l.box.y0 - prev.box.y1 > h || l.box.y1 < prev.box.y0;
    }();
    if (paragraph) blocks.push_back({"Text", 0.0f, {}, {}});
    blocks.back().lines.push_back(static_cast<int>(lines.size()));
    lines.push_back(l);
  }
  page.lines = std::move(lines);
  page.blocks = std::move(blocks);
  return static_cast<int>(tables.size());
}

int width_bucket(int crop_width, double ratio) {
  for (int b = 32;;) {
    if (crop_width <= b) return b;
    if (b >= 2048) return round_up(crop_width, 8);
    b = std::min(2048, std::max(b + 8, round_up(static_cast<int>(std::ceil(b * ratio)), 8)));
  }
}

int width_bucket(int crop_width) {
  for (int b = 32;;) {
    if (crop_width <= b) return b;
    if (b >= 2048) return round_up(crop_width, 8);
    b = std::min(2048, round_up(static_cast<int>(std::ceil(b * 1.3)), 8));
  }
}

std::string OcrPage::text() const {
  std::string s;
  for (const OcrLine& l : lines) {
    s += l.text;
    s += '\n';
  }
  return s;
}

std::array<std::array<double, 2>, 4> corners_as_given(const LineBox& b, const OcrPage& read) {
  const bool across = read.rotation == 90 || read.rotation == 270;
  const double w = across ? read.height : read.width, h = across ? read.width : read.height;  // upright
  std::array<std::array<double, 2>, 4> p = {{{b.x0, b.y0}, {b.x1, b.y0}, {b.x1, b.y1}, {b.x0, b.y1}}};
  if (read.skew != 0) {  // levelled by rotate_page: where each corner was before, inside the page
    constexpr double kPi = 3.14159265358979323846;
    const double a = read.skew * kPi / 180.0, s = std::sin(a), c = std::cos(a);
    for (auto& q : p) {
      const double dx = q[0] - w / 2, dy = q[1] - h / 2;
      q = {std::clamp(dx * c - dy * s + w / 2, 0.0, w), std::clamp(dx * s + dy * c + h / 2, 0.0, h)};
    }
  }
  for (auto& q : p) {  // turned back clockwise: the upright page's (x, y) on the page as given
    const double x = q[0], y = q[1];
    switch (read.rotation) {
      case 90: q = {h - y, x}; break;
      case 180: q = {w - x, h - y}; break;
      case 270: q = {y, w - x}; break;
      default: break;
    }
  }
  return p;
}

LineBox box_as_given(const LineBox& b, const OcrPage& read) {
  if (read.rotation == 0 && read.skew == 0) return b;
  const auto p = corners_as_given(b, read);
  LineBox u{p[0][0], p[0][1], p[0][0], p[0][1]};
  for (const auto& q : p) u = {std::min(u.x0, q[0]), std::min(u.y0, q[1]), std::max(u.x1, q[0]), std::max(u.y1, q[1])};
  return u;
}

std::string skew_text(double degrees) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.2f", degrees);
  return buf;
}

std::string OcrPage::json() const {
  std::string s = "{\"width\": " + std::to_string(width) + ", \"height\": " + std::to_string(height) +
                  ", \"rotation\": " + std::to_string(rotation) + ", \"skew\": " + skew_text(skew) +
                  ", \"source\": ";
  json_escape(s, source);
  s += ", \"order\": ";
  json_escape(s, order);
  s += second_pass ? ", \"second_pass\": true" : ", \"second_pass\": false";
  if (!text_layer.empty()) {
    s += ", \"text_layer\": ";
    json_escape(s, text_layer);
  }
  s += ", \"lines\": [";
  char buf[256];
  for (size_t i = 0; i < lines.size(); ++i) {
    const OcrLine& l = lines[i];
    s += i ? ", {\"box\": " : "{\"box\": ";
    s += box_json(l.box);
    s += ", \"text\": ";
    json_escape(s, l.text);
    std::snprintf(buf, sizeof buf, ", \"confidence\": %.4f, \"det_score\": %.4f, \"words\": [", l.confidence,
                  l.det_score);
    s += buf;
    for (size_t w = 0; w < l.words.size(); ++w) {
      s += w ? ", {\"box\": " : "{\"box\": ";
      s += box_json(l.words[w].box);
      s += ", \"text\": ";
      json_escape(s, l.words[w].text);
      std::snprintf(buf, sizeof buf, ", \"confidence\": %.4f}", l.words[w].confidence);
      s += buf;
    }
    s += "]}";
  }
  s += "], \"pictures\": [";
  for (size_t i = 0; i < pictures.size(); ++i) s += (i ? ", " : "") + box_json(pictures[i]);
  s += "]";
  if (!dropped.empty()) {  // the lines the gate took for something other than text, and why
    s += ", \"dropped\": [";
    for (size_t i = 0; i < dropped.size(); ++i) {
      s += i ? ", {\"box\": " : "{\"box\": ";
      s += box_json(dropped[i].line.box);
      s += ", \"text\": ";
      json_escape(s, dropped[i].line.text);
      std::snprintf(buf, sizeof buf, ", \"confidence\": %.4f, \"why\": ", dropped[i].line.confidence);
      s += buf;
      json_escape(s, dropped[i].why);
      s += "}";
    }
    s += "]";
  }
  auto ints = [](const std::vector<int>& v) {
    std::string r = "[";
    for (size_t i = 0; i < v.size(); ++i) r += (i ? ", " : "") + std::to_string(v[i]);
    return r + "]";
  };
  if (!blocks.empty()) {
    s += ", \"blocks\": [";
    for (size_t b = 0; b < blocks.size(); ++b) {
      s += b ? ", {\"label\": " : "{\"label\": ";
      json_escape(s, blocks[b].label);
      std::snprintf(buf, sizeof buf, ", \"confidence\": %.4f, \"lines\": ", blocks[b].confidence);
      s += buf + ints(blocks[b].lines);
      if (!blocks[b].cells.empty()) {
        s += ", \"cells\": [";
        for (size_t c = 0; c < blocks[b].cells.size(); ++c) {
          const OcrCell& cell = blocks[b].cells[c];
          s += (c ? ", {\"row\": " : "{\"row\": ") + std::to_string(cell.row) + ", \"col\": " +
               std::to_string(cell.col) + ", \"lines\": " + ints(cell.lines);
          if (cell.row_span != 1) s += ", \"row_span\": " + std::to_string(cell.row_span);
          if (cell.col_span != 1) s += ", \"col_span\": " + std::to_string(cell.col_span);
          s += "}";
        }
        s += "]";
      }
      s += "}";
    }
    s += "]";
  }
  std::snprintf(buf, sizeof buf,
                ", \"ms\": {\"orient\": %.2f, \"detect\": %.2f, \"decode\": %.2f, \"second_pass\": %.2f, "
                "\"recognize\": %.2f, \"total\": %.2f}}",
                ms.orient, ms.detect, ms.decode, ms.second_pass, ms.recognize, ms.total);
  s += buf;
  return s;
}

// ---------------------------------------------------------------- the engine

std::string_view precision_name(Precision precision) {
  switch (precision) {
    case Precision::kAuto: return "auto";
    case Precision::kFp32: return "fp32";
    case Precision::kFp16: return "fp16";
    case Precision::kInt8: return "int8";
  }
  return "?";
}

std::optional<Precision> parse_precision(std::string_view name) {
  for (Precision p : {Precision::kAuto, Precision::kFp32, Precision::kFp16, Precision::kInt8}) {
    if (name == precision_name(p)) return p;
  }
  return std::nullopt;
}

SetModel choose_set_model(const std::filesystem::path& set, const std::string& name, std::optional<Backend> backend,
                          Precision precision) {
  const std::filesystem::path int8 = set / (name + ".int8.onnx"), fp32 = set / (name + ".onnx");
  const auto built = compiled_backends();
  const bool ort = std::find(built.begin(), built.end(), Backend::kOnnxRuntime) != built.end();
  // A precision asked for: its file, which the set must have.
  auto file = [&](const char* kind) {
    const std::filesystem::path f = set / (name + kind + ".onnx");
    if (!std::filesystem::is_regular_file(f)) {
      throw Error(f.string() + ": the set has no " + std::string(precision_name(precision)) + " file for " + name);
    }
    return f;
  };
  switch (precision) {
    case Precision::kFp32: return {backend.value_or(ort ? Backend::kOnnxRuntime : Backend::kNative), file("")};
    case Precision::kFp16:
      if (backend && backend != Backend::kNative) {
        throw Error(name + ": FP16 on the CPU runs on the native backend, not " +
                    std::string(backend_name(*backend)));
      }
      return {Backend::kNative, file(".fp16")};
    case Precision::kInt8: return {backend.value_or(Backend::kNative), file(".int8")};
    case Precision::kAuto: break;
  }
  const bool has_int8 = std::filesystem::is_regular_file(int8);
  if (backend == Backend::kNative || (!backend && has_int8)) return {Backend::kNative, has_int8 ? int8 : fp32};
  if (backend) return {*backend, fp32};
  return {ort ? Backend::kOnnxRuntime : Backend::kNative, fp32};
}

std::unique_ptr<Session> open_set_model(const std::filesystem::path& set, const std::string& name,
                                        std::optional<Backend> backend, const SessionOptions& options,
                                        Precision precision) {
  const SetModel m = choose_set_model(set, name, backend, precision);
  return open_session(m.backend, m.file, options);
}

struct Engine::Impl {
  struct Worker {
    std::shared_ptr<Session> orient, det, rec, order;  // a worker's own, or shared (share_sessions)
  };
  struct Wide {  // a multi-threaded detector, for a page that finds fast cores idle
    int threads = 0;
    std::unique_ptr<Session> session;
    std::mutex m;
  };
  // The detector's head, where its file has fullerene's (det_head.h): the workers' detectors and the wide ones
  // then run their file cut before it, and the engine runs the head (cpu::det_head).
  std::optional<detail::FusedHead> head, wide_head;
  struct Job {
    GrayView page{};
    std::shared_ptr<const void> keep;  // what holds the page's pixels, for a page of the stream
    GrayImage own;                     // or its own pixels
    std::filesystem::path file;        // or its file, decoded by the worker that takes the page
    std::span<const uint8_t> bytes;    // or its encoded image (held by keep), decoded so too
    bool unreadable = false;           // its file or bytes did not give an image
    detail::Upright up;  // the page turned upright and levelled, and its copy at the detector's scale
    OcrPage out;
    std::atomic<int> left{0};
    Clock::time_point start, lines_start;
    bool streamed = false;
    bool complete = false;  // under Stream::m
    std::mutex error_m;
    std::exception_ptr error;  // the first a task of the page met

    void fail(std::exception_ptr e) {
      std::lock_guard lk(error_m);
      if (!error) error = std::move(e);
    }
  };
  // The stream's pages (Engine::submit, next): given and not yet taken, by number.
  struct Stream {
    std::mutex m;
    std::condition_variable cv;
    std::map<size_t, std::unique_ptr<Job>> jobs;
    size_t submitted = 0, completed = 0, taken = 0;
    bool unordered = false;  // a page taken by next_finished: next would skip pages
  };

  EngineOptions o;
  std::vector<std::string> vocab;
  WorkerPool pool;
  std::vector<Worker> w;
  std::vector<std::unique_ptr<Wide>> wides;
  Stream stream;

  // The engine's pool: a page starts on a fast core, and slow cores take the lines of pages already started
  // (an E-core detects a page 1.5 times slower than a P-core, sharing its cluster's L2: 10 workers read fewer pages
  // a second than the 6 P-cores alone when E-cores took whole pages).
  static PoolOptions engine_pool(PoolOptions p) {
    p.slow_takes_new = false;
    return p;
  }

  Impl(const std::filesystem::path& set, EngineOptions options)
      : o(std::move(options)), pool(engine_pool(o.pool)) {
    vocab = load_vocab(set / (o.rec + ".vocab.json"));
    o.crop = detail::model_crop(set, o.rec, o.crop);
    if (!o.turn_probability) o.turn_probability = detail::model_turn_probability(set);
    w.resize(pool.size());
    SessionOptions single;
    single.threads = 1;
    const detail::OrderModel order_model = detail::choose_order(set, o);
    choose_set_model(set, "orient", o.backend, o.precision);  // its file first: a missing one is named first, as before
    const SetModel det = choose_set_model(set, "det", o.backend, o.precision);
    head = detail::fused_det_head(det.file);
    SessionOptions det_options = single;
    if (head) det_options.model_bytes = head->cut;
    // With share_sessions an ONNX Runtime model is opened by worker 0 alone and its session given to every worker
    // (each run is still the caller's, on its own thread); a native one stays each worker's own. Each file is
    // chosen just before it is opened, so a missing or unopenable one is named in that order.
    auto own = [&](int k, Backend b) { return k == 0 || !o.share_sessions || b != Backend::kOnnxRuntime; };
    pool.on_each_worker([&](int k) {
      const SetModel orient_m = choose_set_model(set, "orient", o.backend, o.precision);
      if (own(k, orient_m.backend)) w[k].orient = open_session(orient_m.backend, orient_m.file, single);
      if (own(k, det.backend)) w[k].det = open_session(det.backend, det.file, det_options);
      const SetModel rec_m = choose_set_model(set, o.rec, o.backend, o.precision);
      if (own(k, rec_m.backend)) w[k].rec = open_session(rec_m.backend, rec_m.file, single);
      if (!order_model.file.empty() && own(k, order_model.backend)) {
        w[k].order = open_session(order_model.backend, order_model.file, single);
      }
    });
    for (size_t k = 1; k < w.size(); ++k) {
      if (!w[k].orient) w[k].orient = w[0].orient;
      if (!w[k].det) w[k].det = w[0].det;
      if (!w[k].rec) w[k].rec = w[0].rec;
      if (!w[k].order) w[k].order = w[0].order;
    }
    const Shape& lines_in = w[0].rec->inputs().at(0).shape;  // [N, 1, height, W]
    if (lines_in.size() == 4 && lines_in[2] > 0 && lines_in[2] != o.crop.height) {
      throw Error(o.rec + " reads lines " + std::to_string(lines_in[2]) + " px high, but its crop rule makes them " +
                  std::to_string(o.crop.height));
    }
    const Shape& logits = w[0].rec->outputs().at(0).shape;
    if (logits.size() != 3 || (logits[1] > 0 && logits[1] != static_cast<int64_t>(vocab.size()))) {
      throw Error(o.rec + " gives " + shape_string(logits) + " but its vocabulary has " +
                  std::to_string(vocab.size()) + " classes");
    }
    const auto built = compiled_backends();
    const bool ort = std::find(built.begin(), built.end(), Backend::kOnnxRuntime) != built.end();
    // (ONNX Runtime on det.onnx: FP32, so not where another precision is asked for)
    const bool fp32_det = o.precision == Precision::kAuto || o.precision == Precision::kFp32;
    // A wide detector's threads on the fast cores: on a hybrid CPU the OS may otherwise put them on an E-core,
    // which the whole run then waits for (processor group 0 alone, where ONNX Runtime's numbers are these).
    std::vector<int> fast_cpus;
    const CpuTopology& topo = cpu_topology();
    if (o.pool.pin && topo.groups == 1 && !topo.cores.empty() &&
        topo.cores.front().efficiency != topo.cores.back().efficiency) {
      for (const CpuCore& c : topo.cores) {
        if (c.efficiency == topo.cores.front().efficiency) fast_cpus.insert(fast_cpus.end(), c.cpus.begin(), c.cpus.end());
      }
    }
    if (ort && o.backend != Backend::kNative && fp32_det && pool.fast() >= 2) {
      wide_head = det.file == set / "det.onnx" ? head : detail::fused_det_head(set / "det.onnx");
    }
    for (int t : {6, 4, 2}) {
      if (t > pool.fast() || !ort || o.backend == Backend::kNative || !fp32_det) continue;
      auto wd = std::make_unique<Wide>();
      wd->threads = t;
      SessionOptions so;
      so.threads = t;
      so.thread_cpus = fast_cpus;
      if (wide_head) so.model_bytes = wide_head->cut;
      wd->session = open_session(Backend::kOnnxRuntime, set / "det.onnx", so);
      wides.push_back(std::move(wd));
    }
  }

  ~Impl() {
    pool.wait();
    pool.on_each_worker([&](int k) { w[k] = Worker{}; });  // freed on the threads that made them
  }

  int free_fast(int k) const { return pool.idle() + (pool.is_fast(k) ? 1 : 0); }

  // A page's first task: its plan, and an error it meets kept with the page (a page with lines ends in
  // the last of its line tasks instead).
  void start(Job& j, int k) {
    try {
      if (!j.file.empty() || !j.bytes.empty()) {
        const auto t = Clock::now();
        try {
          j.own = j.file.empty() ? decode_image(j.bytes) : load_image(j.file);
        } catch (...) {
          j.unreadable = true;
          throw;
        }
        if (!j.bytes.empty()) {
          j.bytes = {};
          j.keep.reset();  // the bytes, decoded
        }
        j.page = j.own.view();
        j.out.ms.load = ms_since(t);
      }
      page(j, k);
    } catch (...) {
      j.fail(std::current_exception());
      done(j);
    }
  }
  void page(Job& j, int k);
  void recognize(Job& j, std::span<const size_t> lines, int k);
  void read_lines(Job& j, std::span<const size_t> lines, int k);
  // The page's last task: its lines split and ordered, unless a task failed.
  void end(Job& j, int k) {
    if (!j.error) {
      try {
        finish(j, k);
      } catch (...) {
        j.fail(std::current_exception());
      }
    }
    done(j);
  }
  void done(Job& j) {
    if (!j.streamed) return;
    {
      std::lock_guard lk(stream.m);
      j.complete = true;
      ++stream.completed;
    }
    stream.cv.notify_all();
  }
  size_t give(std::unique_ptr<Job> job);
  void finish(Job& j, int k);

  // Runs fs side by side on the idle fast workers, the first here: each one nobody has taken by the time this
  // worker gets to it runs here too. Returns when every one has run; rethrows the first error.
  void together(std::vector<std::function<void()>> fs) {
    if (fs.size() <= 1 || pool.idle() == 0) {
      for (auto& f : fs) f();
      return;
    }
    struct Shared {
      explicit Shared(std::vector<std::function<void()>> f) : fs(std::move(f)), state(fs.size()), left(fs.size()) {}
      std::vector<std::function<void()>> fs;
      std::vector<std::atomic<int>> state;  // 0 waiting, 1 taken
      std::latch left;
      std::mutex m;
      std::exception_ptr error;
    };
    auto sh = std::make_shared<Shared>(std::move(fs));
    const auto run = [](Shared& s, size_t i) {
      if (s.state[i].exchange(1) != 0) return;
      try {
        s.fs[i]();
      } catch (...) {
        std::lock_guard lk(s.m);
        if (!s.error) s.error = std::current_exception();
      }
      s.left.count_down();
    };
    for (size_t i = 1; i < sh->fs.size(); ++i) pool.submit([sh, i, run](int) { run(*sh, i); });
    for (size_t i = 0; i < sh->fs.size(); ++i) run(*sh, i);
    sh->left.wait();
    if (sh->error) std::rethrow_exception(sh->error);
  }
  // The detector's maps from its head's input: [1, in, h/4, w/4] to [1, out, h/2, w/2], in row bands on the idle
  // fast workers (each row's maps the same however the rows are cut).
  Tensor head_maps(const Session::OutputView& y, const detail::FusedHead& h) {
    const Shape& s = y.shape;
    if (s.size() != 4 || s[0] != 1 || s[1] != h.in || y.dtype != DType::kFloat32) {
      throw Error("the detector cut before its head gives " + shape_string(s));
    }
    const int h4 = static_cast<int>(s[2]), w4 = static_cast<int>(s[3]);
    Tensor maps = Tensor::empty(DType::kFloat32, {1, h.out, 2 * h4, 2 * w4});
    const cpu::DetHeadWeights wt = h.weights();
    const float* x = y.as<float>();
    float* out = maps.data<float>().data();
    split_rows(2 * h4, [&](int r0, int r1) { cpu::det_head(x, h4, w4, wt, out, r0, r1); });
    return maps;
  }
  // A page's rows in a band for this worker and each idle fast one (RowSplit).
  void split_rows(int rows, const std::function<void(int, int)>& band) {
    const int parts = std::min(rows, 1 + pool.idle());
    if (parts <= 1) {
      band(0, rows);
      return;
    }
    std::vector<std::function<void()>> fs;
    for (int p = 0; p < parts; ++p) {
      const int r0 = static_cast<int>(static_cast<int64_t>(rows) * p / parts);
      const int r1 = static_cast<int>(static_cast<int64_t>(rows) * (p + 1) / parts);
      fs.push_back([&band, r0, r1] { band(r0, r1); });
    }
    together(std::move(fs));
  }
  void order_page(OcrPage& page, int k) { detail::order_page(page, w[k].order.get(), o); }
};

void Engine::Impl::page(Job& j, int k) {
  j.start = Clock::now();
  OcrPage& out = j.out;
  out.width = j.page.width;
  out.height = j.page.height;

  // The page at the detector's scale, once; orientation reads a thumbnail of it,
  // and turns the page only by a quarter turn, only when sure.
  GrayImage small = detail::det_scale(j.page, o.det_long_side);
  const int side = o.orient_side;
  const auto orient = [&] {
    Tensor thumb = Tensor::empty(DType::kFloat32, {1, 1, side, side});
    detail::thumbnail_input(small.view(), j.page.width, j.page.height, side, thumb.data<float>().data());
    const Tensor logits = std::move(w[k].orient->run(one(std::move(thumb)))[0]);
    out.rotation = detail::rotation_of(logits.data<float>(), *o.turn_probability);
  };
  // With fast cores idle (a page alone), the skew's search on the page as given runs beside orientation: a page
  // orientation does not turn keeps it, a turned one searches again (the same bytes either way).
  std::optional<Skew> skew;
  if (o.deskew && pool.idle() > 0) {
    together({orient, [&] { skew = find_skew(small.view(), o.max_skew); }});
  } else {
    orient();
  }
  // What is left of a turn, a few degrees: the page and its copy at the
  // detector's scale levelled, when the level angle clearly wins.
  detail::make_upright(j.page, out.rotation, std::move(small), o, j.up, skew,
                       [this](int rows, const std::function<void(int, int)>& band) { split_rows(rows, band); });
  out.skew = j.up.skew;
  out.ms.orient = ms_since(j.start);

  // Detection: the page at its scale, padded to a multiple of 32, on threads
  // when fast cores are idle (the same bytes as on one).
  auto t = Clock::now();
  const GrayView sv = j.up.small.view();
  const int dw = sv.width, dh = sv.height;
  const int pw = round_up(dw, 32), ph = round_up(dh, 32);
  Tensor input = Tensor::empty(DType::kFloat32, {1, 1, ph, pw});
  ink_input(sv, dw, dh, pw, ph, input.data<float>().data());
  const detail::DetectRun run = [&](const Tensor& x) {
    const std::span<const Tensor> in(&x, 1);
    for (auto& wd : wides) {
      if (wd->threads > free_fast(k)) continue;
      std::unique_lock lk(wd->m, std::try_to_lock);
      if (!lk) continue;
      if (!wide_head) return std::move(wd->session->run(in)[0]);
      Tensor maps;
      wd->session->run(in, [&](std::span<const Session::OutputView> y) { maps = head_maps(y[0], *wide_head); });
      return maps;
    }
    if (!head) return std::move(w[k].det->run(in)[0]);
    Tensor maps;
    w[k].det->run(in, [&](std::span<const Session::OutputView> y) { maps = head_maps(y[0], *head); });
    return maps;
  };
  const Tensor maps = run(input);
  out.ms.detect = ms_since(t);

  // The line decoder, its boxes back in the upright page's pixels; a page of tiny
  // text detected again at a larger scale (its strips by the same run) and the
  // two passes merged.
  t = Clock::now();
  const Shape& shape = maps.shape();
  const Detection d =
      decode_det(maps.data<float>().data(), static_cast<int>(shape[2]), static_cast<int>(shape[3]), 1.0, o.det);
  detail::place_detection(d, j.up.view, dw, dh, o, run, out);
  out.ms.decode = ms_since(t) - out.ms.second_pass;

  // Recognition: the lines by their width buckets, in a task for each free fast
  // core, or in tasks of line_batch when every core is busy.
  j.lines_start = Clock::now();
  const size_t n = out.lines.size();
  if (n == 0) {
    end(j, k);
    return;
  }
  const detail::LineTasks plan = detail::line_tasks(out.lines, o.crop, pool.idle() > 0 ? free_fast(k) : 0,
                                                    o.line_batch, o.bucket_ratio);
  auto lines = std::make_shared<std::vector<size_t>>(std::move(plan.lines));
  const int tasks = static_cast<int>(plan.starts.size()) - 1;
  j.left.store(tasks);
  for (int t2 = 1; t2 < tasks; ++t2) {
    const size_t a = plan.starts[t2], b = plan.starts[t2 + 1];
    pool.submit([this, &j, lines, a, b](int v) { recognize(j, std::span<const size_t>(lines->data() + a, b - a), v); });
  }
  recognize(j, std::span<const size_t>(lines->data(), plan.starts[1]), k);
}

void Engine::Impl::recognize(Job& j, std::span<const size_t> lines, int k) {
  try {
    read_lines(j, lines, k);
  } catch (...) {
    j.fail(std::current_exception());
  }
  if (j.left.fetch_sub(1) == 1) end(j, k);
}

void Engine::Impl::read_lines(Job& j, std::span<const size_t> lines, int k) {
  const int h = o.crop.height;
  for (size_t a = 0; a < lines.size();) {
    // one batch for each width bucket the task's lines fall in
    const int bw = width_bucket(crop_region(j.out.lines[lines[a]].box, o.crop).width, o.bucket_ratio);
    size_t b = a;
    std::vector<LineBox> boxes;
    while (b < lines.size() &&
           width_bucket(crop_region(j.out.lines[lines[b]].box, o.crop).width, o.bucket_ratio) == bw) {
      boxes.push_back(j.out.lines[lines[b]].box);
      ++b;
    }
    const int m = static_cast<int>(boxes.size());
    std::vector<uint8_t> crops(static_cast<size_t>(m) * h * bw);
    std::vector<int32_t> widths(m);
    crop_lines(j.up.view, boxes, bw, crops.data(), widths, o.crop);
    Tensor x = Tensor::empty(DType::kFloat32, {m, 1, h, bw});
    float* px = x.data<float>().data();
    for (size_t i = 0; i < crops.size(); ++i) px[i] = 1.0f - crops[i] / 255.0f;
    // The logits decoded where the recognizer wrote them (about 20 MB a page, read once): no copy.
    std::vector<int32_t> lengths(m);
    std::vector<CtcLine> decoded;
    const std::vector<Tensor> in = one(std::move(x));
    w[k].rec->run(in, [&](std::span<const Session::OutputView> out) {
      const Session::OutputView& v = out[0];
      if (v.dtype != DType::kFloat32 || v.shape.size() != 3 || v.shape[0] != m) {
        throw Error(o.rec + " gives logits " + shape_string(v.shape) + " for " + std::to_string(m) + " lines");
      }
      for (int i = 0; i < m; ++i) {
        lengths[i] = std::min<int32_t>(static_cast<int32_t>(v.shape[2]), widths[i] / o.crop.frame_px);
      }
      decoded = ctc_greedy(v.as<float>(), m, static_cast<int>(v.shape[1]), static_cast<int>(v.shape[2]), lengths);
    });
    if (o.spaces) {  // each crop's columns of ink, for its spaces (SpaceRule)
      std::vector<uint8_t> ink(static_cast<size_t>(bw));
      for (int i = 0; i < m; ++i) {
        const uint8_t* c = crops.data() + static_cast<size_t>(i) * h * bw;
        const bool has = detail::ink_columns(c, widths[i], h, bw, o.crop, ink.data());
        detail::read_line(j.out.lines[lines[a + i]], decoded[i], lengths[i], o.crop, vocab,
                          has ? ink.data() : nullptr, widths[i], &*o.spaces);
      }
    } else {
      for (int i = 0; i < m; ++i) detail::read_line(j.out.lines[lines[a + i]], decoded[i], lengths[i], o.crop, vocab);
    }
    a = b;
  }
}

void Engine::Impl::finish(Job& j, int k) {
  OcrPage& out = j.out;
  out.ms.recognize = ms_since(j.lines_start);
  detail::finish_page(out, w[k].order.get(), o);
  out.ms.total = ms_since(j.start);
}

// ---------------------------------------------------------------- the steps the pipeline shares

namespace detail {

LineTasks line_tasks(std::span<const OcrLine> lines, const CropRule& crop, int parts, int line_batch, double ratio) {
  const size_t n = lines.size();
  std::vector<int> bucket(n);
  for (size_t i = 0; i < n; ++i) bucket[i] = width_bucket(crop_region(lines[i].box, crop).width, ratio);
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return bucket[a] < bucket[b]; });
  LineTasks plan;
  if (parts <= 0) {  // every core busy: tasks of line_batch width-sorted lines, which other pages' workers steal
    const size_t chunk = static_cast<size_t>(std::max(1, line_batch));
    plan.lines = std::move(order);
    for (size_t a = 0; a < n; a += chunk) plan.starts.push_back(a);
    plan.starts.push_back(n);
    return plan;
  }
  // Idle cores: a task each, split by cost (a line costs its bucket's width): the widest first, each into the least
  // loaded task (equal counts of width-sorted lines left the widest task 2.3 times the mean on the median page);
  // inside each task the lines by width again, so a bucket's lines stay one batch.
  const int p = static_cast<int>(std::min<size_t>(static_cast<size_t>(parts), n));
  std::vector<std::vector<size_t>> part(p);
  std::vector<int64_t> load(p, 0);
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    const int t = static_cast<int>(std::min_element(load.begin(), load.end()) - load.begin());
    load[t] += bucket[*it];
    part[t].push_back(*it);
  }
  for (auto& t : part) {
    std::stable_sort(t.begin(), t.end(), [&](size_t a, size_t b) { return bucket[a] < bucket[b]; });
    plan.starts.push_back(plan.lines.size());
    plan.lines.insert(plan.lines.end(), t.begin(), t.end());
  }
  plan.starts.push_back(n);
  return plan;
}

OrderModel choose_order(const std::filesystem::path& set, const EngineOptions& o) {
  OrderModel m;
  if (!o.use_order) return m;
  const auto built = compiled_backends();
  const bool has_ort = std::find(built.begin(), built.end(), Backend::kOnnxRuntime) != built.end();
  if (!o.order_model.empty()) {
    m.file = o.order_model;
    m.backend = o.order_model.extension() == ".bin" ? Backend::kNative : Backend::kOnnxRuntime;
    if (m.backend == Backend::kOnnxRuntime && !has_ort) {
      throw Error(o.order_model.string() + ": an ONNX order model runs on ONNX Runtime, not built in");
    }
  } else if (o.backend != Backend::kOnnxRuntime && std::filesystem::is_regular_file(set / "order.native.bin")) {
    m.file = set / "order.native.bin";
    m.backend = Backend::kNative;
  } else if (has_ort && std::filesystem::is_regular_file(set / "order.onnx")) {
    m.file = set / "order.onnx";
    m.backend = Backend::kOnnxRuntime;
  }
  return m;
}

int fit_side(int side, int longest, int to) {
  return std::max(1, static_cast<int>(std::lround(1.0 * side * to / longest)));
}

GrayImage det_scale(const GrayView& page, int long_side) {
  const int longest = std::max(page.width, page.height);
  return resize_gray(page, fit_side(page.width, longest, long_side), fit_side(page.height, longest, long_side));
}

void thumbnail_input(const GrayView& small, int page_w, int page_h, int side, float* out) {
  const int longest = std::max(page_w, page_h);
  ink_input(small, fit_side(page_w, longest, side), fit_side(page_h, longest, side), side, side, out);
}

int rotation_of(std::span<const float> z, float turn_probability) {
  const size_t best = static_cast<size_t>(std::max_element(z.begin(), z.end()) - z.begin());
  double sum = 0;
  for (float v : z) sum += std::exp(v - z[best]);
  const double p = 1.0 / sum;
  return p >= turn_probability && (best == 3 || best == 6 || best == 9) ? static_cast<int>(best) * 30 : 0;
}

void make_upright(const GrayView& page, int rotation, GrayImage small, const EngineOptions& o, Upright& u,
                  std::optional<Skew> skew, const RowSplit& split) {
  u.view = page;
  u.skew = 0;
  u.small = std::move(small);
  if (rotation) {
    const int q = 4 - rotation / 90;  // its text turned back counterclockwise
    u.turned = rotate_quarter(page, q);
    u.view = u.turned.view();
    u.small = rotate_quarter(u.small.view(), q);
  }
  if (o.deskew) {
    const Skew s = skew && !rotation ? *skew : find_skew(u.small.view(), o.max_skew);
    if (std::abs(s.degrees) >= o.min_skew && s.gain >= o.min_skew_gain) {
      u.skew = s.degrees;
      u.levelled = split ? rotate_page(u.view, s.degrees, split) : rotate_page(u.view, s.degrees);
      u.view = u.levelled.view();
      u.small = rotate_page(u.small.view(), s.degrees);
    }
  }
}

Detection on_page(const Detection& d, int vw, int vh, int dw, int dh) {
  const double sx = 1.0 * vw / dw, sy = 1.0 * vh / dh;
  auto to_page = [&](DetBox b) {
    b.box = {std::clamp(b.box.x0 * sx, 0.0, 1.0 * vw), std::clamp(b.box.y0 * sy, 0.0, 1.0 * vh),
             std::clamp(b.box.x1 * sx, 0.0, 1.0 * vw), std::clamp(b.box.y1 * sy, 0.0, 1.0 * vh)};
    return b;
  };
  Detection p;
  p.lines.reserve(d.lines.size());
  for (const DetBox& l : d.lines) p.lines.push_back(to_page(l));
  for (const DetBox& b : d.pictures) p.pictures.push_back(to_page(b));
  p.picture_mask = d.picture_mask;  // its map pixels now cover the page's
  p.picture_mask.px_x *= sx;
  p.picture_mask.px_y *= sy;
  return p;
}

std::optional<Detection> second_pass(const GrayView& upright, int dw, int dh, const EngineOptions& o,
                                     const DetectRun& run) {
  const double f = std::min(o.det.second_pass_scale, 1.0 * o.second_pass_long_side / std::max(dw, dh));
  const int w = std::max(1, static_cast<int>(std::lround(dw * f)));
  const int h = std::max(1, static_cast<int>(std::lround(dh * f)));
  if (w <= dw && h <= dh) return std::nullopt;
  const GrayImage page = resize_gray(upright, w, h);
  const int pw = round_up(w, 32), ph = round_up(h, 32);
  Tensor x = Tensor::empty(DType::kFloat32, {1, 1, ph, pw});
  ink_input(page.view(), w, h, pw, ph, x.data<float>().data());
  // Strips of about a first pass's pixels (the detector's long side squared), so
  // the detector's memory, which keeps its largest run's, stays about a first
  // pass's; their cores at least two halos high, so the halos at most double
  // their rows.
  const StripRule rule{StripRule{}.halo, StripRule{}.align, o.det.stride};
  const int64_t side = round_up(o.det_long_side, 32);
  const int64_t by_pixels = (static_cast<int64_t>(ph) * pw + side * side - 1) / (side * side);
  const int strips = static_cast<int>(std::clamp<int64_t>(by_pixels, 1, std::max(1, ph / (2 * rule.halo))));
  const Tensor maps = run_in_strips(x, strips, run, rule);
  const Shape& s = maps.shape();
  return on_page(decode_det(maps.data<float>().data(), static_cast<int>(s[2]), static_cast<int>(s[3]), 1.0, o.det),
                 upright.width, upright.height, w, h);
}

void place_detection(const Detection& first, const GrayView& upright, int dw, int dh, const EngineOptions& o,
                     const DetectRun& run, OcrPage& out) {
  Detection d = on_page(first, upright.width, upright.height, dw, dh);
  if (run && takes_second_pass(first, o.det)) {
    const auto t = Clock::now();
    if (const std::optional<Detection> second = second_pass(upright, dw, dh, o, run)) {
      d = merge_passes(d, *second, 1.0 * upright.height / dh, o.det);
      out.second_pass = true;
      out.ms.second_pass = ms_since(t);
    }
  }
  for (const DetBox& l : d.lines) {
    const LineBox& b = l.box;
    if (b.x1 - b.x0 >= 1 && b.y1 - b.y0 >= 1) out.lines.push_back({b, "", 0.0f, l.score});
  }
  for (const DetBox& p : d.pictures) out.pictures.push_back(p.box);
  out.picture_mask = std::move(d.picture_mask);
}

bool ink_columns(const uint8_t* crop, int width, int h, int64_t stride, const CropRule& rule, uint8_t* out) {
  if (width <= 0 || h <= 0) return false;
  // the text's rows: the crop less its padding above and below (pad_v of the text's height each)
  const int pad = static_cast<int>(std::lround(h * rule.pad_v / (1 + 2 * rule.pad_v)));
  const int r0 = std::min(pad, h - 1), r1 = std::max(r0 + 1, h - pad);
  std::array<int32_t, 256> hist{};
  std::vector<uint8_t> low(static_cast<size_t>(width), 255);  // each column's darkest, in the text's rows
  for (int y = r0; y < r1; ++y) {
    const uint8_t* row = crop + y * stride;
    for (int x = 0; x < width; ++x) {  // a loop the compiler takes to pminub
      low[static_cast<size_t>(x)] = std::min(low[static_cast<size_t>(x)], row[x]);
      ++hist[row[x]];
    }
  }
  const int64_t n = static_cast<int64_t>(r1 - r0) * width;
  int darkest = 0;
  while (darkest < 255 && hist[static_cast<size_t>(darkest)] == 0) ++darkest;
  int64_t seen = 0;
  int p90 = 255;
  for (int v = 0; v < 256; ++v) {
    seen += hist[static_cast<size_t>(v)];
    if (seen * 10 >= n * 9) {
      p90 = v;
      break;
    }
  }
  if (p90 - darkest < 32) return false;
  const int t = (darkest + p90) / 2;
  for (int x = 0; x < width; ++x) out[x] = low[static_cast<size_t>(x)] < t;
  return true;
}

void read_line(OcrLine& line, const CtcLine& d, int32_t length, const CropRule& crop,
               const std::vector<std::string>& vocab, const uint8_t* ink, int ink_width, const SpaceRule* spaces) {
  auto is_space = [&](size_t c) {
    const int32_t id = d.ids[c];
    return id > 0 && static_cast<size_t>(id) < vocab.size() &&
           vocab[static_cast<size_t>(id)].find_first_not_of(" \t") == std::string::npos;
  };
  auto is_char = [&](size_t c) {
    const int32_t id = d.ids[c];
    return id > 0 && static_cast<size_t>(id) < vocab.size() && !is_space(c);
  };
  // The line's median blank run between inks, in crop columns (SpaceRule::wider), found at the first unsure space.
  double median_gap = -1;
  auto line_gap = [&] {
    if (median_gap >= 0) return median_gap;
    std::vector<int> runs;
    int run = 0;
    bool inked = false;
    for (int x = 0; x < ink_width; ++x) {
      if (!ink[x]) {
        ++run;
        continue;
      }
      if (inked && run > 0) runs.push_back(run);
      run = 0;
      inked = true;
    }
    median_gap = 0;
    if (!runs.empty()) {
      std::nth_element(runs.begin(), runs.begin() + runs.size() / 2, runs.end());
      median_gap = runs[runs.size() / 2];
    }
    return median_gap;
  };
  // Whether the ink bears out the space at c: a blank run of the crop's columns, between its two neighbouring
  // characters' frames, `wider` times the line's median and min_gap of the text's height (no neighbour on a side:
  // kept).
  auto parted = [&](size_t c) {
    size_t a = c, b = c;
    while (a > 0 && !is_char(a - 1)) --a;
    while (b + 1 < d.ids.size() && !is_char(b + 1)) ++b;
    if (a == 0 || b + 1 >= d.ids.size()) return true;
    const int x0 = std::clamp(d.frames[a - 1] * crop.frame_px, 0, ink_width);
    const int x1 = std::clamp(d.frames[b + 1] * crop.frame_px, 0, ink_width);
    const double text_h = crop.height / (1 + 2 * crop.pad_v);
    int run = 0, longest = 0;
    for (int x = x0; x < x1; ++x) {
      run = ink[x] ? 0 : run + 1;
      longest = std::max(longest, run);
    }
    return longest >= std::max(spaces->min_gap * text_h, spaces->wider * line_gap());
  };
  // A frame's place on the page: frame f starts at crop x f * frame_px, and the
  // crop is its region of the page resampled to the crop's own width.
  const CropRegion r = crop_region(line.box, crop);
  const double per_px = (r.x1 - r.x0) / std::max(1, r.width);
  auto page_x = [&](int frame) { return std::clamp(r.x0 + frame * crop.frame_px * per_px, line.box.x0, line.box.x1); };
  double sum = 0, word_sum = 0;
  int word_chars = 0;
  float least = 1;
  bool any = false;
  OcrWord word;
  auto end_word = [&](int frame) {
    if (word.text.empty()) return;
    word.box.x1 = std::max(word.box.x0, page_x(frame));
    word.confidence = static_cast<float>(word_sum / std::max(1, word_chars));
    line.words.push_back(std::move(word));
    word = OcrWord{};
    word_sum = 0;
    word_chars = 0;
  };
  for (size_t c = 0; c < d.ids.size(); ++c) {
    const int32_t id = d.ids[c];
    sum += d.probs[c];
    if (id <= 0 || static_cast<size_t>(id) >= vocab.size()) continue;
    const std::string& ch = vocab[id];
    if (ch.find_first_not_of(" \t") == std::string::npos) {  // a space ends a word, unless the ink belies it
      if (ink && spaces && d.probs[c] < spaces->sure && !parted(c)) continue;
      line.text += ch;
      end_word(d.frames[c]);
      continue;
    }
    line.text += ch;
    if (word.text.empty()) word.box = {page_x(d.frames[c]), line.box.y0, line.box.x1, line.box.y1};
    word.text += ch;
    word_sum += d.probs[c];
    ++word_chars;
    least = std::min(least, d.probs[c]);
    any = true;
  }
  end_word(length);
  line.confidence = d.ids.empty() ? 0.0f : static_cast<float>(sum / d.ids.size());
  line.least = any ? least : 0.0f;
}

// ---------------------------------------------------------------- the gate (GateRule)

void gate_whole_lines(OcrPage& page, const GateRule& rule) {
  std::vector<OcrLine> kept;
  kept.reserve(page.lines.size());
  for (OcrLine& l : page.lines) {
    const TextKinds k = kinds_of(l.text);
    const double w = l.box.x1 - l.box.x0, h = l.box.y1 - l.box.y0;
    const char* why = k.chars() == 0                                                   ? "empty"
                      : k.alnum == 0 && k.other == 0                                   ? "rules"
                      : h > rule.sliver * w && k.chars() <= rule.sliver_chars           ? "sliver"
                                                                                        : nullptr;
    if (why) {
      page.dropped.push_back({std::move(l), why});
    } else {
      kept.push_back(std::move(l));
    }
  }
  page.lines = std::move(kept);
}

void gate_pieces(OcrPage& page, const GateRule& rule) {
  const size_t n = page.lines.size();
  std::vector<const char*> why(n, nullptr);
  for (size_t i = 0; i < n; ++i) {  // read out of a picture, unsurely
    const OcrLine& l = page.lines[i];
    if ((l.confidence < rule.picture_sure || l.least < rule.picture_least) &&
        page.picture_mask.share(l.box) >= rule.picture_share) {
      why[i] = "picture";
    }
  }
  for (size_t i = 0; i < n; ++i) {  // a few unsure characters with nothing beside them on their row
    const OcrLine& l = page.lines[i];
    if (why[i] || l.confidence >= rule.lone_sure || kinds_of(l.text).alnum > rule.lone_chars) continue;
    const double h = std::max(1.0, l.box.y1 - l.box.y0);
    bool beside = false;
    for (size_t j = 0; j < n && !beside; ++j) {
      if (j == i || why[j]) continue;
      const LineBox& b = page.lines[j].box;
      const double over = std::min(l.box.y1, b.y1) - std::max(l.box.y0, b.y0);
      const double gap = std::max(b.x0, l.box.x0) - std::min(b.x1, l.box.x1);
      beside = over >= 0.5 * std::min(h, b.y1 - b.y0) && gap <= h;
    }
    if (!beside) why[i] = "lone";
  }
  std::vector<OcrLine> kept;
  kept.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    if (why[i]) {
      page.dropped.push_back({std::move(page.lines[i]), why[i]});
    } else {
      kept.push_back(std::move(page.lines[i]));
    }
  }
  page.lines = std::move(kept);
}

void blocks_by_geometry(PageOrder& po, std::span<const LineBox> boxes, std::span<const std::string> texts) {
  if (po.blocks.size() <= 1) return;
  std::vector<double> hs, cws;
  for (size_t i = 0; i < boxes.size(); ++i) {
    const double h = boxes[i].y1 - boxes[i].y0, w = boxes[i].x1 - boxes[i].x0;
    const int c = code_points(texts[i]);
    if (h > 0) hs.push_back(h);
    if (c > 0 && w > 0) cws.push_back(w / c);
  }
  auto median = [](std::vector<double>& v, double none) {
    if (v.empty()) return none;
    std::nth_element(v.begin(), v.begin() + static_cast<int64_t>(v.size() / 2), v.end());
    return v[v.size() / 2];
  };
  const double lh = std::max(1.0, median(hs, 1.0));
  const double cw = std::max(1.0, median(cws, lh / 2));
  std::vector<LineBox> bb;
  for (const OrderBlock& b : po.blocks) {
    LineBox u = boxes[b.lines.front()];
    for (size_t i : b.lines) {
      u = {std::min(u.x0, boxes[i].x0), std::min(u.y0, boxes[i].y0), std::max(u.x1, boxes[i].x1),
           std::max(u.y1, boxes[i].y1)};
    }
    bb.push_back(u);
  }
  std::vector<size_t> idx(bb.size()), order;
  std::iota(idx.begin(), idx.end(), size_t{0});
  unit_xy_cut(std::move(idx), bb, cw, lh, order);
  std::vector<OrderBlock> blocks;
  po.order.clear();
  for (size_t k : order) {
    po.order.insert(po.order.end(), po.blocks[k].lines.begin(), po.blocks[k].lines.end());
    blocks.push_back(std::move(po.blocks[k]));
  }
  po.blocks = std::move(blocks);
}

void table_spans(OcrPage& page) {
  for (OcrBlock& b : page.blocks) {
    const size_t m = b.cells.size();
    if (b.label != "Table" || m < 2) continue;
    std::vector<LineBox> box(m);
    for (size_t k = 0; k < m; ++k) {
      LineBox u{1e300, 1e300, -1e300, -1e300};
      for (int i : b.cells[k].lines) {
        const LineBox& l = page.lines[static_cast<size_t>(i)].box;
        u = {std::min(u.x0, l.x0), std::min(u.y0, l.y0), std::max(u.x1, l.x1), std::max(u.y1, l.y1)};
      }
      box[k] = u;
    }
    auto over = [](double a0, double a1, double b0, double b1) { return std::min(a1, b1) - std::max(a0, b0); };
    // A cell is wide where it covers more than half of two cells that lie apart across the page (a header over
    // columns); every other cell keeps the model's row and column.
    std::vector<bool> wide(m, false);
    std::vector<size_t> covered;
    for (size_t a = 0; a < m; ++a) {
      covered.clear();
      for (size_t c = 0; c < m; ++c) {
        if (c != a && over(box[a].x0, box[a].x1, box[c].x0, box[c].x1) > 0.5 * (box[c].x1 - box[c].x0)) {
          covered.push_back(c);
        }
      }
      for (size_t x = 0; x < covered.size() && !wide[a]; ++x) {
        for (size_t y = x + 1; y < covered.size(); ++y) {
          const LineBox &p = box[covered[x]], &q = box[covered[y]];
          if (over(p.x0, p.x1, q.x0, q.x1) <= 0) {
            wide[a] = true;
            break;
          }
        }
      }
    }
    if (std::find(wide.begin(), wide.end(), true) == wide.end()) continue;  // no span: the model's grid as it is
    // The model's columns by their narrow cells' extents; a column of wide cells alone (the model gave a header a
    // column of its own) goes, the rest numbered left to right.
    int cols = 0;
    for (const OcrCell& c : b.cells) cols = std::max(cols, c.col + 1);
    std::vector<std::pair<double, double>> extent(static_cast<size_t>(cols), {1e300, -1e300});
    for (size_t k = 0; k < m; ++k) {
      if (wide[k]) continue;
      auto& e = extent[static_cast<size_t>(b.cells[k].col)];
      e = {std::min(e.first, box[k].x0), std::max(e.second, box[k].x1)};
    }
    std::vector<int> kept;  // the model's columns that hold a narrow cell, by their left edge
    for (int c = 0; c < cols; ++c) {
      if (extent[static_cast<size_t>(c)].first <= extent[static_cast<size_t>(c)].second) kept.push_back(c);
    }
    if (kept.empty()) continue;
    std::stable_sort(kept.begin(), kept.end(),
                     [&](int x, int y) { return extent[static_cast<size_t>(x)].first < extent[static_cast<size_t>(y)].first; });
    std::vector<int> renumber(static_cast<size_t>(cols), -1);
    for (size_t t = 0; t < kept.size(); ++t) renumber[static_cast<size_t>(kept[t])] = static_cast<int>(t);
    for (size_t k = 0; k < m; ++k) {
      OcrCell& c = b.cells[k];
      if (!wide[k]) {
        c.col = renumber[static_cast<size_t>(c.col)];
        continue;
      }
      int f = -1, n = 0;  // the kept columns it covers more than half of
      for (size_t t = 0; t < kept.size(); ++t) {
        const auto& e = extent[static_cast<size_t>(kept[t])];
        if (over(box[k].x0, box[k].x1, e.first, e.second) > 0.5 * (e.second - e.first)) {
          if (f < 0) f = static_cast<int>(t);
          ++n;
        }
      }
      if (f < 0) {  // none by half: the column its centre is nearest
        const double x = (box[k].x0 + box[k].x1) / 2;
        double best = 1e300;
        for (size_t t = 0; t < kept.size(); ++t) {
          const auto& e = extent[static_cast<size_t>(kept[t])];
          const double d = x < e.first ? e.first - x : x > e.second ? x - e.second : 0;
          if (d < best) {
            best = d;
            f = static_cast<int>(t);
          }
        }
        n = 1;
      }
      c.col = f;
      c.col_span = std::max(1, n);
    }
    std::stable_sort(b.cells.begin(), b.cells.end(),
                     [](const OcrCell& x, const OcrCell& y) { return x.row != y.row ? x.row < y.row : x.col < y.col; });
  }
}

void text_tables_as_text(OcrPage& page, int max_cell_chars) {
  for (OcrBlock& b : page.blocks) {
    if (b.label != "Table" || b.cells.empty()) continue;
    std::vector<int> chars;
    for (const OcrCell& c : b.cells) {
      int n = 0;
      for (int i : c.lines) n += code_points(page.lines[static_cast<size_t>(i)].text);
      chars.push_back(n);
    }
    std::nth_element(chars.begin(), chars.begin() + static_cast<int64_t>(chars.size() / 2), chars.end());
    if (chars[chars.size() / 2] > max_cell_chars) {
      b.label = "Text";
      b.cells.clear();
    }
  }
}

void finish_page(OcrPage& page, Session* order, const EngineOptions& o) {
  if (o.gate) gate_whole_lines(page, *o.gate);
  if (o.split) page.lines = split_lines(std::move(page.lines), *o.split);
  if (o.gate) gate_pieces(page, *o.gate);
  page.picture_mask = PictureMask{};
  order_page(page, order, o);
}

void order_page(OcrPage& page, Session* model, const EngineOptions& o) {
  const size_t n = page.lines.size();
  std::vector<LineBox> boxes;
  std::vector<std::string> texts;
  for (const OcrLine& l : page.lines) {
    boxes.push_back(l.box);
    texts.push_back(l.text);
  }
  page.blocks.clear();
  page.order = "xy-cut";
  std::vector<size_t> order;
  if (!model || n == 0) {
    order = reading_order(boxes);
  } else {
    // the boxes are the upright page's
    const bool across = page.rotation == 90 || page.rotation == 270;
    const int pw = across ? page.height : page.width, ph = across ? page.width : page.height;
    Tensor x = Tensor::empty(DType::kFloat32, {1, static_cast<int64_t>(n), kOrderFeatures});
    const std::vector<float> f = order_features(boxes, texts, pw, ph);
    std::copy(f.begin(), f.end(), x.data<float>().begin());
    const std::vector<Tensor> y = model->run(one(std::move(x)));
    if (y.size() != 6) throw Error("the order model gives " + std::to_string(y.size()) + " outputs, not 6");
    auto at = [&](size_t i) { return std::span<const float>(y[i].data<float>()); };
    const OrderOutputs outs{static_cast<int>(n), at(0), at(1), at(2), at(3), at(4), at(5)};
    PageOrder po = decode_order(outs, boxes, o.order);
    if (!po.fallback && o.blocks_by_geometry) blocks_by_geometry(po, boxes, texts);
    order = po.order;
    page.order = po.fallback ? "xy-cut" : "model";
    std::vector<int> pos(n);  // where each line goes
    for (size_t p = 0; p < n; ++p) pos[order[p]] = static_cast<int>(p);
    for (const OrderBlock& b : po.blocks) {
      OcrBlock ob;
      ob.label = block_label_name(b.label);
      ob.confidence = b.confidence;
      for (size_t i : b.lines) ob.lines.push_back(pos[i]);
      for (const TableCell& c : b.cells) {
        OcrCell cell{c.row, c.col, {}};
        for (size_t i : c.lines) cell.lines.push_back(pos[i]);
        ob.cells.push_back(std::move(cell));
      }
      page.blocks.push_back(std::move(ob));
    }
  }
  std::vector<OcrLine> ordered;
  for (size_t i : order) ordered.push_back(std::move(page.lines[i]));
  page.lines = std::move(ordered);
  text_tables_as_text(page, o.tables ? o.tables->max_cell_chars : TableRule{}.max_cell_chars);
  if (o.table_spans) table_spans(page);
  if (page.blocks.empty() && o.tables) find_tables(page, *o.tables);  // no model's blocks: tables from geometry
}

}  // namespace detail

Engine::Engine(const std::filesystem::path& set, EngineOptions options)
    : impl_(std::make_unique<Impl>(set, std::move(options))) {}

Engine::~Engine() = default;

int Engine::workers() const { return impl_->pool.size(); }

std::vector<OcrPage> Engine::ocr(std::span<const GrayView> pages) {
  std::vector<std::unique_ptr<Impl::Job>> jobs;
  for (const GrayView& p : pages) {
    if (!p.data || p.width <= 0 || p.height <= 0) throw Error("Engine::ocr: an empty page");
    jobs.push_back(std::make_unique<Impl::Job>());
    jobs.back()->page = p;
  }
  for (auto& j : jobs) {
    Impl::Job* job = j.get();
    impl_->pool.submit([this, job](int k) { impl_->start(*job, k); });
  }
  impl_->pool.wait();
  for (auto& j : jobs) {
    if (j->error) std::rethrow_exception(j->error);
  }
  std::vector<OcrPage> out;
  for (auto& j : jobs) out.push_back(std::move(j->out));
  return out;
}

size_t Engine::Impl::give(std::unique_ptr<Job> job) {
  job->streamed = true;
  const size_t cap = static_cast<size_t>(o.in_flight > 0 ? o.in_flight : 2 * pool.size());
  Job* j = job.get();
  {
    std::unique_lock lk(stream.m);
    stream.cv.wait(lk, [&] { return stream.submitted - stream.completed < cap; });
    stream.jobs.emplace(stream.submitted, std::move(job));
    ++stream.submitted;
  }
  pool.submit([this, j](int k) { start(*j, k); });
  return stream.submitted - 1;
}

size_t Engine::submit(const GrayView& page, std::shared_ptr<const void> keep) {
  if (!page.data || page.width <= 0 || page.height <= 0) throw Error("Engine::submit: an empty page");
  auto job = std::make_unique<Impl::Job>();
  job->page = page;
  job->keep = std::move(keep);
  return impl_->give(std::move(job));
}

size_t Engine::submit(GrayImage page) {
  if (page.pixels.empty() || page.width <= 0 || page.height <= 0) throw Error("Engine::submit: an empty page");
  auto job = std::make_unique<Impl::Job>();
  job->own = std::move(page);
  job->page = job->own.view();
  return impl_->give(std::move(job));
}

size_t Engine::submit(const std::filesystem::path& file) {
  if (file.empty()) throw Error("Engine::submit: no file");
  auto job = std::make_unique<Impl::Job>();
  job->file = file;
  return impl_->give(std::move(job));
}

size_t Engine::submit_encoded(std::span<const uint8_t> bytes, std::shared_ptr<const void> keep) {
  if (bytes.empty()) throw Error("Engine::submit_encoded: no bytes");
  auto job = std::make_unique<Impl::Job>();
  job->bytes = bytes;
  job->keep = std::move(keep);
  return impl_->give(std::move(job));
}

std::optional<OcrPage> Engine::next() {
  Impl::Stream& s = impl_->stream;
  std::unique_ptr<Impl::Job> job;
  {
    std::unique_lock lk(s.m);
    if (s.unordered) throw Error("Engine::next: this stream's pages are taken by next_finished");
    if (s.taken == s.submitted) return std::nullopt;
    s.cv.wait(lk, [&] { return s.jobs.at(s.taken)->complete; });
    auto it = s.jobs.find(s.taken);
    job = std::move(it->second);
    s.jobs.erase(it);
    ++s.taken;
  }
  if (job->error) std::rethrow_exception(job->error);
  return std::move(job->out);
}

std::optional<Engine::Finished> Engine::next_finished() {
  Impl::Stream& s = impl_->stream;
  std::unique_ptr<Impl::Job> job;
  Finished f;
  {
    std::unique_lock lk(s.m);
    s.unordered = true;
    if (s.taken == s.submitted) return std::nullopt;
    auto it = s.jobs.begin();
    s.cv.wait(lk, [&] {  // the pages not taken: at most in_flight being read, and those done
      for (it = s.jobs.begin(); it != s.jobs.end(); ++it) {
        if (it->second->complete) return true;
      }
      return false;
    });
    f.number = it->first;
    job = std::move(it->second);
    s.jobs.erase(it);
    ++s.taken;
  }
  f.error = job->error;
  f.unreadable = job->unreadable;
  if (!f.error) f.page = std::move(job->out);
  return f;
}

OcrPage Engine::ocr(const GrayView& page) { return std::move(ocr(std::span<const GrayView>(&page, 1))[0]); }

void Engine::order(std::span<OcrPage> pages) {
  for (OcrPage& p : pages) {
    OcrPage* page = &p;
    impl_->pool.submit([this, page](int k) { impl_->order_page(*page, k); });
  }
  impl_->pool.wait();
}

bool Engine::has_order_model() const { return !impl_->w.empty() && impl_->w[0].order != nullptr; }

const CropRule& Engine::crop() const { return impl_->o.crop; }

float Engine::turn_probability() const { return *impl_->o.turn_probability; }

}  // namespace carbon
