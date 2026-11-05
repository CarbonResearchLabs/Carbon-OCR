// A born-digital page's text layer (carbon/textlayer.h): lines and checks.

#include "carbon/textlayer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace carbon {
namespace {

bool bad_code(char32_t c) {
  if (c == 0 || c == 0xFFFD) return true;
  if (c < 0x20 && c != U'\t' && c != U'\n' && c != U'\r') return true;
  if (c >= 0x7F && c < 0xA0) return true;                          // C1 controls
  if ((c >= 0xE000 && c <= 0xF8FF) || c >= 0xF0000) return true;   // private use
  if (c >= 0xD800 && c <= 0xDFFF) return true;                     // a surrogate left alone
  return false;
}

// UTF-16's surrogate pairs as their code points: PDFium gives a character's UTF-16 units where wchar_t is 16
// bits, so a character past U+FFFF comes as two. The pair's box their union; a lone half stays, to be counted bad.
std::vector<PdfChar> paired(std::span<const PdfChar> chars) {
  std::vector<PdfChar> out;
  out.reserve(chars.size());
  for (size_t i = 0; i < chars.size(); ++i) {
    const PdfChar& c = chars[i];
    if (c.code >= 0xD800 && c.code <= 0xDBFF && i + 1 < chars.size() && chars[i + 1].code >= 0xDC00 &&
        chars[i + 1].code <= 0xDFFF) {
      const PdfChar& lo = chars[i + 1];
      PdfChar p = c;
      p.code = 0x10000 + ((c.code - 0xD800) << 10) + (lo.code - 0xDC00);
      p.x0 = std::min(c.x0, lo.x0);
      p.y0 = std::min(c.y0, lo.y0);
      p.x1 = std::max(c.x1, lo.x1);
      p.y1 = std::max(c.y1, lo.y1);
      p.lx0 = std::min(c.lx0, lo.lx0);
      p.ly0 = std::min(c.ly0, lo.ly0);
      p.lx1 = std::max(c.lx1, lo.lx1);
      p.ly1 = std::max(c.ly1, lo.ly1);
      out.push_back(p);
      ++i;
      continue;
    }
    out.push_back(c);
    // the hyphen a word is broken at, at a line's end: PDFium gives it as U+0002 (a hyphen it found ends the
    // line), a font may give U+00AD; it is the hyphen the page prints, so the writers join the word as OCR's
    if (c.code == 0x02 || c.code == 0xAD) out.back().code = U'-';
  }
  return out;
}

bool space(char32_t c) { return c == U' ' || c == U'\t' || c == 0xA0 || c == U'\r' || c == U'\n'; }

void put_utf8(std::string& s, char32_t c) {
  if (c < 0x80) {
    s += static_cast<char>(c);
  } else if (c < 0x800) {
    s += static_cast<char>(0xC0 | (c >> 6));
    s += static_cast<char>(0x80 | (c & 0x3F));
  } else if (c < 0x10000) {
    s += static_cast<char>(0xE0 | (c >> 12));
    s += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
    s += static_cast<char>(0x80 | (c & 0x3F));
  } else {
    s += static_cast<char>(0xF0 | (c >> 18));
    s += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
    s += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
    s += static_cast<char>(0x80 | (c & 0x3F));
  }
}

// The darkest pixel under a box (255 for a box off the page).
// The lightest less the darkest pixel under a box (0 for a box off the page).
int spread(const GrayView& page, float x0, float y0, float x1, float y1) {
  int lo = 255, hi = 0;
  const int ya = std::max(0, static_cast<int>(std::floor(y0)));
  const int yb = std::min(page.height, static_cast<int>(std::ceil(y1)));
  const int xa = std::max(0, static_cast<int>(std::floor(x0)));
  const int xb = std::min(page.width, static_cast<int>(std::ceil(x1)));
  for (int y = ya; y < yb; ++y) {
    const uint8_t* row = page.data + y * page.stride;
    for (int x = xa; x < xb; ++x) {
      lo = std::min<int>(lo, row[x]);
      hi = std::max<int>(hi, row[x]);
    }
  }
  return hi >= lo ? hi - lo : 0;
}

[[maybe_unused]] int darkest(const GrayView& page, float x0, float y0, float x1, float y1) {
  int lo = 255;
  const int ya = std::max(0, static_cast<int>(std::floor(y0)));
  const int yb = std::min(page.height, static_cast<int>(std::ceil(y1)));
  const int xa = std::max(0, static_cast<int>(std::floor(x0)));
  const int xb = std::min(page.width, static_cast<int>(std::ceil(x1)));
  for (int y = ya; y < yb; ++y) {
    const uint8_t* row = page.data + y * page.stride;
    for (int x = xa; x < xb; ++x) lo = std::min<int>(lo, row[x]);
  }
  return lo;
}

double area(const LineBox& b) { return std::max(0.0, b.x1 - b.x0) * std::max(0.0, b.y1 - b.y0); }

double overlap(const LineBox& a, const LineBox& b) {
  return std::max(0.0, std::min(a.x1, b.x1) - std::max(a.x0, b.x0)) *
         std::max(0.0, std::min(a.y1, b.y1) - std::max(a.y0, b.y0));
}

// A box of a read page's (the page turned upright, then levelled), on the page as given.
LineBox as_given(const LineBox& b, const OcrPage& read) {
  const bool across = read.rotation == 90 || read.rotation == 270;
  const double w = across ? read.height : read.width, h = across ? read.width : read.height;  // upright
  LineBox u = b;
  if (read.skew != 0) {  // levelled by rotate_page: where its corners were before, their bounds
    constexpr double kPi = 3.14159265358979323846;
    const double a = read.skew * kPi / 180.0, s = std::sin(a), c = std::cos(a);
    constexpr double kInf = std::numeric_limits<double>::infinity();
    u = {kInf, kInf, -kInf, -kInf};
    for (const double x : {b.x0, b.x1}) {
      for (const double y : {b.y0, b.y1}) {
        const double dx = x - w / 2, dy = y - h / 2;
        const double sx = dx * c - dy * s + w / 2, sy = dx * s + dy * c + h / 2;
        u = {std::min(u.x0, sx), std::min(u.y0, sy), std::max(u.x1, sx), std::max(u.y1, sy)};
      }
    }
    u = {std::max(0.0, u.x0), std::max(0.0, u.y0), std::min(w, u.x1), std::min(h, u.y1)};
  }
  switch (read.rotation) {  // turned back clockwise: the upright page's (x, y) on the page as given
    case 90:
      return {h - u.y1, u.x0, h - u.y0, u.x1};
    case 180:
      return {w - u.x1, h - u.y1, w - u.x0, h - u.y0};
    case 270:
      return {u.y0, w - u.x1, u.y1, w - u.x0};
    default:
      return u;
  }
}

// Whether `c` carries on the line `prev` is on: its centre inside prev's band
// across the direction from prev to it.
bool carries_on(const PdfChar& prev, const PdfChar& c) {
  const float cx = (c.lx0 + c.lx1) / 2, cy = (c.ly0 + c.ly1) / 2;
  const float px = (prev.lx0 + prev.lx1) / 2, py = (prev.ly0 + prev.ly1) / 2;
  if (std::abs(cx - px) >= std::abs(cy - py)) return cy >= prev.ly0 && cy <= prev.ly1;  // along x
  return cx >= prev.lx0 && cx <= prev.lx1;                                               // along y
}

}  // namespace

TextLayer read_text_layer(std::span<const PdfChar> drawn, const GrayView& page, const TextLayerRule& rule) {
  const std::vector<PdfChar> whole = paired(drawn);
  const std::span<const PdfChar> chars(whole);
  TextLayer out;
  TextLayerCheck& k = out.check;
  for (const PdfChar& c : chars) {
    if (c.generated) continue;
    ++k.chars;
    out.marked += c.mcid >= 0;
    if (c.bad_map || bad_code(c.code)) ++k.bad;
    if (c.invisible) ++k.invisible;
    // ink: two tones over the box and its surround (a quarter of its height around it, a pixel at least), so a
    // glyph that fills its box shows against the paper about it, and white on white shows nothing
    const float m = std::max(1.0f, 0.25f * (c.y1 - c.y0));
    if (page.data && !space(c.code) && c.x1 > c.x0 && c.y1 > c.y0 &&
        spread(page, c.x0 - m, c.y0 - m, c.x1 + m, c.y1 + m) < rule.contrast) {
      ++k.no_ink;
    }
  }
  const double n = std::max(1, k.chars);
  if (k.chars < rule.min_chars) {
    k.reason = "too few characters (" + std::to_string(k.chars) + ")";
  } else if (k.bad / n > rule.max_bad) {
    k.reason = std::to_string(k.bad) + " of " + std::to_string(k.chars) + " characters are not readable text";
  } else if (k.invisible / n > rule.max_invisible) {
    k.reason = std::to_string(k.invisible) + " of " + std::to_string(k.chars) + " characters are invisible";
  } else if (k.no_ink / n > rule.max_no_ink) {
    k.reason = std::to_string(k.no_ink) + " of " + std::to_string(k.chars) + " characters have no ink";
  }
  k.usable = k.reason.empty();

  // The lines, and their words.
  OcrLine line;
  OcrWord word;
  bool open = false, pending_space = false;
  const PdfChar* prev = nullptr;
  auto end_word = [&] {
    if (word.text.empty()) return;
    word.confidence = 1.0f;
    line.words.push_back(std::move(word));
    word = OcrWord{};
  };
  int line_mcid = -1;
  auto close = [&] {
    end_word();
    if (open && !line.text.empty()) {
      line.confidence = 1.0f;
      out.lines.push_back(line);
      out.mcids.push_back(line_mcid);
    }
    line = OcrLine{};
    open = pending_space = false;
    prev = nullptr;
  };
  for (const PdfChar& c : chars) {
    if (c.generated) {
      if (c.code == U'\r' || c.code == U'\n') {
        close();
      } else if (space(c.code)) {
        pending_space = open;
        end_word();
      }
      continue;
    }
    if (prev && !carries_on(*prev, c)) close();
    if (space(c.code)) {
      pending_space = open;
      end_word();
      continue;
    }
    if (c.lx1 <= c.lx0 || c.ly1 <= c.ly0) continue;  // nothing on the page
    if (!open) {
      line.box = {c.lx0, c.ly0, c.lx1, c.ly1};
      line_mcid = c.mcid;
      open = true;
    } else {
      line.box.x0 = std::min<double>(line.box.x0, c.lx0);
      line.box.y0 = std::min<double>(line.box.y0, c.ly0);
      line.box.x1 = std::max<double>(line.box.x1, c.lx1);
      line.box.y1 = std::max<double>(line.box.y1, c.ly1);
    }
    if (pending_space) line.text += ' ';
    pending_space = false;
    put_utf8(line.text, c.code);
    if (word.text.empty()) {
      word.box = {c.lx0, c.ly0, c.lx1, c.ly1};
    } else {
      word.box.x0 = std::min<double>(word.box.x0, c.lx0);
      word.box.y0 = std::min<double>(word.box.y0, c.ly0);
      word.box.x1 = std::max<double>(word.box.x1, c.lx1);
      word.box.y1 = std::max<double>(word.box.y1, c.ly1);
    }
    put_utf8(word.text, c.code);
    prev = &c;
  }
  close();
  return out;
}

OcrPage page_from_text_layer(const TextLayer& layer, int width, int height) {
  OcrPage page;
  page.width = width;
  page.height = height;
  page.source = "text layer";
  // a table row's cells, a gutter the layer ran across; a justified line's stretched spaces are its own, not
  // splits (the justified rule: the layer's spaces are exact, so a justified column's every gap is wide)
  SplitRule split;
  split.justified = true;
  std::vector<OcrLine> lines = split_lines(layer.lines, split);
  std::vector<LineBox> boxes;
  for (const OcrLine& l : lines) boxes.push_back(l.box);
  for (size_t i : reading_order(boxes)) page.lines.push_back(std::move(lines[i]));
  return page;
}

namespace {

// A structure element's role as a block's label (carbon/order.h's names).
const char* role_label(const std::string& t) {
  if (t == "P" || t == "Span" || t == "Quote" || t == "BlockQuote" || t == "Code") return "Text";
  if (t == "H" || (t.size() == 2 && t[0] == 'H' && t[1] >= '1' && t[1] <= '6') || t == "Title") return "Section-Header";
  if (t == "L" || t == "LI" || t == "Lbl" || t == "LBody") return "List-Group";
  if (t == "Table" || t == "TR" || t == "TH" || t == "TD" || t == "THead" || t == "TBody" || t == "TFoot") return "Table";
  if (t == "Caption") return "Caption";
  if (t == "Note" || t == "FENote") return "Footnote";
  if (t == "TOC" || t == "TOCI") return "Table-Of-Contents";
  if (t == "Figure") return "Picture";
  if (t == "Formula") return "Formula";
  if (t == "Form") return "Form";
  return "Text";
}

}  // namespace

OcrPage page_from_tagged_layer(const TextLayer& layer, const PdfStructure& tree, int width, int height,
                               double cover) {
  // Each MCID's place in the tree's reading order and its element: its first run.
  std::map<int, std::pair<size_t, int>> where;
  for (size_t r = 0; r < tree.runs.size(); ++r) where.emplace(tree.runs[r].second, std::make_pair(r, tree.runs[r].first));
  size_t chars = 0, held = 0;  // the lines' characters, and those of lines whose marked content the tree holds
  const bool marked = layer.mcids.size() == layer.lines.size();
  for (size_t i = 0; i < layer.lines.size(); ++i) {
    const size_t n = layer.lines[i].text.size();
    chars += n;
    if (marked && where.count(layer.mcids[i])) held += n;
  }
  if (where.empty() || held == 0 || static_cast<double>(held) < cover * static_cast<double>(chars)) {
    return page_from_text_layer(layer, width, height);
  }
  // The lines by their marked content's place (a line the tree does not hold after the rest), the layer's order
  // breaking ties; each split at wide gaps as page_from_text_layer splits, its pieces kept together.
  std::vector<size_t> idx(layer.lines.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
  auto place = [&](size_t i) {
    const auto it = where.find(layer.mcids[i]);
    return it == where.end() ? tree.runs.size() : it->second.first;
  };
  std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return place(a) < place(b); });
  OcrPage page;
  page.width = width;
  page.height = height;
  page.source = "text layer";
  page.order = "tagged";
  SplitRule split;
  split.justified = true;
  int last_element = -2;
  for (size_t i : idx) {
    const auto it = where.find(layer.mcids[i]);
    const int element = it == where.end() ? -1 : it->second.second;
    if (element != last_element || page.blocks.empty()) {  // a run of lines under one element: a block
      OcrBlock b;
      if (element >= 0 && element < static_cast<int>(tree.elements.size())) {
        b.label = role_label(tree.elements[static_cast<size_t>(element)].type);
      } else {  // outside the tree: an artifact, page furniture where it sits at the top or the bottom
        const LineBox& lb = layer.lines[i].box;
        b.label = lb.y1 <= 0.15 * height ? "Page-Header" : lb.y0 >= 0.85 * height ? "Page-Footer" : "Other";
      }
      b.confidence = 1.0f;
      page.blocks.push_back(std::move(b));
      last_element = element;
    }
    for (OcrLine& piece : split_lines({layer.lines[i]}, split)) {
      page.blocks.back().lines.push_back(static_cast<int>(page.lines.size()));
      page.lines.push_back(std::move(piece));
    }
  }
  return page;
}

std::vector<LineBox> uncovered_ink(const GrayView& page, const TextLayer& layer, std::span<const PdfChar> chars,
                                   std::span<const LineBox> pictures, const TextLayerRule& rule, int min_ink) {
  std::vector<LineBox> out;
  if (!page.data || layer.lines.empty() || page.width <= 0 || page.height <= 0) return out;
  std::vector<double> heights;
  for (const OcrLine& l : layer.lines) heights.push_back(l.box.y1 - l.box.y0);
  std::nth_element(heights.begin(), heights.begin() + static_cast<std::ptrdiff_t>(heights.size() / 2), heights.end());
  const double lh = std::max(1.0, heights[heights.size() / 2]);
  // Each row's covered spans: the characters' loose boxes and the pictures, as column ranges.
  std::vector<std::vector<std::pair<int, int>>> covered(static_cast<size_t>(page.height));
  auto cover = [&](double x0, double y0, double x1, double y1) {
    const int ya = std::max(0, static_cast<int>(std::floor(y0))), yb = std::min(page.height, static_cast<int>(std::ceil(y1)));
    const int xa = std::max(0, static_cast<int>(std::floor(x0))), xb = std::min(page.width, static_cast<int>(std::ceil(x1)));
    if (xb <= xa) return;
    for (int y = ya; y < yb; ++y) covered[static_cast<size_t>(y)].emplace_back(xa, xb);
  };
  for (const PdfChar& c : chars) {
    if (!c.generated && c.lx1 > c.lx0 && c.ly1 > c.ly0) cover(c.lx0, c.ly0, c.lx1, c.ly1);
  }
  for (const LineBox& p : pictures) cover(p.x0, p.y0, p.x1, p.y1);
  // Each row's uncovered ink: its count and its columns' extent.
  struct Row {
    int n = 0, x0 = 0, x1 = 0;
  };
  std::vector<Row> rows(static_cast<size_t>(page.height));
  std::vector<uint8_t> mask(static_cast<size_t>(page.width));
  for (int y = 0; y < page.height; ++y) {
    std::fill(mask.begin(), mask.end(), uint8_t{0});
    for (const auto& [a, b] : covered[static_cast<size_t>(y)]) std::fill(mask.begin() + a, mask.begin() + b, uint8_t{1});
    const uint8_t* r = page.data + y * page.stride;
    Row& row = rows[static_cast<size_t>(y)];
    for (int x = 0; x < page.width; ++x) {
      if (mask[static_cast<size_t>(x)] || r[x] > rule.ink) continue;
      if (row.n++ == 0) row.x0 = x;
      row.x1 = x + 1;
    }
  }
  // Runs of rows with uncovered ink, a line height of rows without joined, kept at two line heights or taller.
  const int gap = static_cast<int>(std::ceil(lh)), tall = static_cast<int>(std::ceil(2 * lh));
  int y = 0;
  while (y < page.height) {
    if (rows[static_cast<size_t>(y)].n < min_ink) {
      ++y;
      continue;
    }
    int y0 = y, y1 = y + 1, last = y, x0 = rows[static_cast<size_t>(y)].x0, x1 = rows[static_cast<size_t>(y)].x1;
    for (int k = y + 1; k < page.height && k - last <= gap; ++k) {
      const Row& row = rows[static_cast<size_t>(k)];
      if (row.n < min_ink) continue;
      last = k;
      y1 = k + 1;
      x0 = std::min(x0, row.x0);
      x1 = std::max(x1, row.x1);
    }
    if (y1 - y0 >= tall) out.push_back({static_cast<double>(x0), static_cast<double>(y0), static_cast<double>(x1),
                                        static_cast<double>(y1)});
    y = y1 + 1;
  }
  return out;
}

std::vector<LineBox> pictures_to_read(std::span<const LineBox> pictures, int width, int height,
                                      const HybridRule& rule) {
  std::vector<LineBox> out;
  for (const LineBox& p : pictures) {
    const LineBox c{std::max(0.0, p.x0), std::max(0.0, p.y0), std::min<double>(width, p.x1),
                    std::min<double>(height, p.y1)};
    if (c.x1 - c.x0 >= rule.min_side && c.y1 - c.y0 >= rule.min_side) out.push_back(c);
  }
  return out;
}

GrayImage pictures_only(const GrayView& page, std::span<const LineBox> pictures) {
  GrayImage out;
  out.width = page.width;
  out.height = page.height;
  out.pixels.assign(static_cast<size_t>(page.width) * page.height, 255);
  for (const LineBox& p : pictures) {
    const int x0 = std::clamp(static_cast<int>(std::floor(p.x0)), 0, page.width);
    const int x1 = std::clamp(static_cast<int>(std::ceil(p.x1)), x0, page.width);
    const int y0 = std::clamp(static_cast<int>(std::floor(p.y0)), 0, page.height);
    const int y1 = std::clamp(static_cast<int>(std::ceil(p.y1)), y0, page.height);
    for (int y = y0; y < y1; ++y) {
      std::copy_n(page.data + y * page.stride + x0, x1 - x0,
                  out.pixels.begin() + static_cast<int64_t>(y) * page.width + x0);
    }
  }
  return out;
}

void add_picture_lines(OcrPage& page, const OcrPage& read, const HybridRule& rule) {
  const size_t own = page.lines.size();
  std::vector<OcrLine> lines = page.lines;
  for (const OcrLine& l : read.lines) {
    if (l.text.empty()) continue;
    OcrLine g = l;
    g.box = as_given(l.box, read);
    for (OcrWord& w : g.words) w.box = as_given(w.box, read);
    double under = 0;  // the layer's lines do not overlap one another, so their overlaps add up
    for (size_t i = 0; i < own; ++i) under += overlap(g.box, page.lines[i].box);
    if (area(g.box) <= 0 || under >= rule.max_covered * area(g.box)) continue;
    lines.push_back(std::move(g));
  }
  if (lines.size() == own) return;
  std::vector<LineBox> boxes;
  for (const OcrLine& l : lines) boxes.push_back(l.box);
  page.lines.clear();
  for (size_t i : reading_order(boxes)) page.lines.push_back(std::move(lines[i]));
  page.source = "hybrid";
}

}  // namespace carbon
