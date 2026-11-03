// The OCR pipeline v0 on the CPU (carbon/engine.h).

#include "carbon/engine.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
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
               std::to_string(cell.col) + ", \"lines\": " + ints(cell.lines) + "}";
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
    std::unique_ptr<Session> orient, det, rec, order;
  };
  struct Wide {  // a multi-threaded detector, for a page that finds fast cores idle
    int threads = 0;
    std::unique_ptr<Session> session;
    std::mutex m;
  };
  struct Job {
    GrayView page{};
    detail::Upright up;  // the page turned upright and levelled, and its copy at the detector's scale
    OcrPage out;
    std::atomic<int> left{0};
    Clock::time_point start, lines_start;
  };

  EngineOptions o;
  std::vector<std::string> vocab;
  WorkerPool pool;
  std::vector<Worker> w;
  std::vector<std::unique_ptr<Wide>> wides;

  Impl(const std::filesystem::path& set, EngineOptions options) : o(std::move(options)), pool(o.pool) {
    vocab = load_vocab(set / (o.rec + ".vocab.json"));
    o.crop = detail::model_crop(set, o.rec, o.crop);
    if (!o.turn_probability) o.turn_probability = detail::model_turn_probability(set);
    w.resize(pool.size());
    SessionOptions single;
    single.threads = 1;
    const detail::OrderModel order_model = detail::choose_order(set, o);
    pool.on_each_worker([&](int k) {
      w[k].orient = open_set_model(set, "orient", o.backend, single, o.precision);
      w[k].det = open_set_model(set, "det", o.backend, single, o.precision);
      w[k].rec = open_set_model(set, o.rec, o.backend, single, o.precision);
      if (!order_model.file.empty()) w[k].order = open_session(order_model.backend, order_model.file, single);
    });
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
    for (int t : {6, 4, 2}) {
      if (t > pool.fast() || !ort || o.backend == Backend::kNative || !fp32_det) continue;
      auto wd = std::make_unique<Wide>();
      wd->threads = t;
      SessionOptions so;
      so.threads = t;
      wd->session = open_session(Backend::kOnnxRuntime, set / "det.onnx", so);
      wides.push_back(std::move(wd));
    }
  }

  ~Impl() {
    pool.wait();
    pool.on_each_worker([&](int k) { w[k] = Worker{}; });  // freed on the threads that made them
  }

  int free_fast(int k) const { return pool.idle() + (pool.is_fast(k) ? 1 : 0); }

  void page(Job& j, int k);
  void recognize(Job& j, std::span<const size_t> lines, int k);
  void finish(Job& j, int k);
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
  {
    Tensor thumb = Tensor::empty(DType::kFloat32, {1, 1, side, side});
    detail::thumbnail_input(small.view(), j.page.width, j.page.height, side, thumb.data<float>().data());
    const Tensor logits = std::move(w[k].orient->run(one(std::move(thumb)))[0]);
    out.rotation = detail::rotation_of(logits.data<float>(), *o.turn_probability);
  }
  // What is left of a turn, a few degrees: the page and its copy at the
  // detector's scale levelled, when the level angle clearly wins.
  detail::make_upright(j.page, out.rotation, std::move(small), o, j.up);
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
      return std::move(wd->session->run(in)[0]);
    }
    return std::move(w[k].det->run(in)[0]);
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
    finish(j, k);
    return;
  }
  std::vector<int> bucket(n);
  for (size_t i = 0; i < n; ++i) bucket[i] = width_bucket(crop_region(out.lines[i].box, o.crop).width);
  auto order = std::make_shared<std::vector<size_t>>(n);
  std::iota(order->begin(), order->end(), size_t{0});
  std::stable_sort(order->begin(), order->end(), [&](size_t a, size_t b) { return bucket[a] < bucket[b]; });
  const int idle = pool.idle();
  const size_t chunk = idle > 0 ? (n + std::max(1, free_fast(k)) - 1) / std::max(1, free_fast(k))
                                : static_cast<size_t>(std::max(1, o.line_batch));
  const int tasks = static_cast<int>((n + chunk - 1) / chunk);
  j.left.store(tasks);
  for (int t2 = 1; t2 < tasks; ++t2) {
    pool.submit([this, &j, order, chunk, t2, n](int v) {
      const size_t a = t2 * chunk, b = std::min(n, a + chunk);
      recognize(j, std::span<const size_t>(order->data() + a, b - a), v);
    });
  }
  recognize(j, std::span<const size_t>(order->data(), std::min(n, chunk)), k);
}

void Engine::Impl::recognize(Job& j, std::span<const size_t> lines, int k) {
  const int h = o.crop.height;
  for (size_t a = 0; a < lines.size();) {
    // one batch for each width bucket the task's lines fall in
    const int bw = width_bucket(crop_region(j.out.lines[lines[a]].box, o.crop).width);
    size_t b = a;
    std::vector<LineBox> boxes;
    while (b < lines.size() && width_bucket(crop_region(j.out.lines[lines[b]].box, o.crop).width) == bw) {
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
    const Tensor logits = std::move(w[k].rec->run(one(std::move(x)))[0]);
    const Shape& s = logits.shape();
    std::vector<int32_t> lengths(m);
    for (int i = 0; i < m; ++i) lengths[i] = std::min<int32_t>(static_cast<int32_t>(s[2]), widths[i] / o.crop.frame_px);
    const auto decoded = ctc_greedy(logits.data<float>().data(), m, static_cast<int>(s[1]), static_cast<int>(s[2]),
                                    lengths);
    for (int i = 0; i < m; ++i) detail::read_line(j.out.lines[lines[a + i]], decoded[i], lengths[i], o.crop, vocab);
    a = b;
  }
  if (j.left.fetch_sub(1) == 1) finish(j, k);
}

void Engine::Impl::finish(Job& j, int k) {
  OcrPage& out = j.out;
  out.ms.recognize = ms_since(j.lines_start);
  if (o.split) out.lines = split_lines(std::move(out.lines), *o.split);
  order_page(out, k);
  out.ms.total = ms_since(j.start);
}

// ---------------------------------------------------------------- the steps the pipeline shares

namespace detail {

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

void make_upright(const GrayView& page, int rotation, GrayImage small, const EngineOptions& o, Upright& u) {
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
    const Skew s = find_skew(u.small.view(), o.max_skew);
    if (std::abs(s.degrees) >= o.min_skew && s.gain >= o.min_skew_gain) {
      u.skew = s.degrees;
      u.levelled = rotate_page(u.view, s.degrees);
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
}

void read_line(OcrLine& line, const CtcLine& d, int32_t length, const CropRule& crop,
               const std::vector<std::string>& vocab) {
  // A frame's place on the page: frame f starts at crop x f * frame_px, and the
  // crop is its region of the page resampled to the crop's own width.
  const CropRegion r = crop_region(line.box, crop);
  const double per_px = (r.x1 - r.x0) / std::max(1, r.width);
  auto page_x = [&](int frame) { return std::clamp(r.x0 + frame * crop.frame_px * per_px, line.box.x0, line.box.x1); };
  double sum = 0, word_sum = 0;
  int word_chars = 0;
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
    line.text += ch;
    if (ch.find_first_not_of(" \t") == std::string::npos) {  // a space ends a word
      end_word(d.frames[c]);
      continue;
    }
    if (word.text.empty()) word.box = {page_x(d.frames[c]), line.box.y0, line.box.x1, line.box.y1};
    word.text += ch;
    word_sum += d.probs[c];
    ++word_chars;
  }
  end_word(length);
  line.confidence = d.ids.empty() ? 0.0f : static_cast<float>(sum / d.ids.size());
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
    const PageOrder po = decode_order(outs, boxes, o.order);
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
    impl_->pool.submit([this, job](int k) { impl_->page(*job, k); });
  }
  impl_->pool.wait();
  std::vector<OcrPage> out;
  for (auto& j : jobs) out.push_back(std::move(j->out));
  return out;
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
