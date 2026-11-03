// A born-digital page's text layer (carbon/textlayer.h): lines and checks.

#include "carbon/textlayer.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace carbon {
namespace {

bool bad_code(char32_t c) {
  if (c == 0 || c == 0xFFFD) return true;
  if (c < 0x20 && c != U'\t' && c != U'\n' && c != U'\r') return true;
  if (c >= 0x7F && c < 0xA0) return true;                          // C1 controls
  if ((c >= 0xE000 && c <= 0xF8FF) || c >= 0xF0000) return true;   // private use
  return false;
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
int darkest(const GrayView& page, float x0, float y0, float x1, float y1) {
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

TextLayer read_text_layer(std::span<const PdfChar> chars, const GrayView& page, const TextLayerRule& rule) {
  TextLayer out;
  TextLayerCheck& k = out.check;
  for (const PdfChar& c : chars) {
    if (c.generated) continue;
    ++k.chars;
    if (c.bad_map || bad_code(c.code)) ++k.bad;
    if (c.invisible) ++k.invisible;
    if (page.data && !space(c.code) && c.x1 > c.x0 && c.y1 > c.y0 &&
        darkest(page, c.x0, c.y0, c.x1, c.y1) > rule.ink) {
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
  auto close = [&] {
    end_word();
    if (open && !line.text.empty()) {
      line.confidence = 1.0f;
      out.lines.push_back(line);
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
  std::vector<OcrLine> lines = split_lines(layer.lines);  // a table row's cells, a gutter the layer ran across
  std::vector<LineBox> boxes;
  for (const OcrLine& l : lines) boxes.push_back(l.box);
  for (size_t i : reading_order(boxes)) page.lines.push_back(std::move(lines[i]));
  return page;
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
