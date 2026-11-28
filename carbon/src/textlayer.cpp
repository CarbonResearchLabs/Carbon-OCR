// A born-digital page's text layer (carbon/textlayer.h): lines and checks.

#include "carbon/textlayer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <utility>

#include "engine_steps.h"

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

// A letter, as a run of a broken encoding would hold one: ASCII's; past Latin-1's symbols, any code point
// outside the spacing modifiers and combining marks, the general punctuation, symbol and operator blocks
// (U+2000-2BFF), CJK punctuation and the full-width forms' symbols (a mathematical alphanumeric is a letter).
bool letter(char32_t c) {
  if (c < 0x80) return (c | 0x20) >= 'a' && (c | 0x20) <= 'z';
  if (c < 0xC0) return c == 0xAA || c == 0xB5 || c == 0xBA;
  if (c == 0xD7 || c == 0xF7) return false;
  if (c >= 0x2B0 && c <= 0x36F) return false;
  if (c >= 0x2000 && c <= 0x2BFF) return false;
  if (c >= 0x3000 && c <= 0x303F) return false;
  if (c >= 0xFF00 && c <= 0xFF20) return false;
  return true;
}

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

// The way a line runs: by the step from its first character to its second.
enum class Run { kUnknown, kAlongX, kAlongY };

// Whether `c` carries on the line whose last character is `prev`: its centre inside prev's band across the
// direction from prev to it (its height range along x, its width range along y); or, a step the way the line runs,
// inside the line's running band across it (the line's box so far, the union of its characters' loose boxes), so
// the text after a raised or lowered script, back on the line's band but off the script's, carries on. A step
// across the way the line runs (to the row below, in a table PDFium breaks no line in) is held to prev's band.
bool carries_on(const PdfChar& prev, const PdfChar& c, const LineBox& band, Run run) {
  const float cx = (c.lx0 + c.lx1) / 2, cy = (c.ly0 + c.ly1) / 2;
  const float px = (prev.lx0 + prev.lx1) / 2, py = (prev.ly0 + prev.ly1) / 2;
  if (std::abs(cx - px) >= std::abs(cy - py)) {  // along x
    return run == Run::kAlongX ? cy >= band.y0 && cy <= band.y1 : cy >= prev.ly0 && cy <= prev.ly1;
  }
  return run == Run::kAlongY ? cx >= band.x0 && cx <= band.x1 : cx >= prev.lx0 && cx <= prev.lx1;  // along y
}

// The way the step from prev to c runs.
Run step(const PdfChar& prev, const PdfChar& c) {
  const float dx = (c.lx0 + c.lx1 - prev.lx0 - prev.lx1) / 2, dy = (c.ly0 + c.ly1 - prev.ly0 - prev.ly1) / 2;
  return std::abs(dx) >= std::abs(dy) ? Run::kAlongX : Run::kAlongY;
}

// A character of a line as read_text_layer puts it there: its word (an index into the line's words) and its bytes
// in that word's text.
struct Placed {
  const PdfChar* c;
  size_t word;
  uint32_t begin, end;
};

float median(std::vector<float> v) {
  const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
  std::nth_element(v.begin(), mid, v.end());
  return *mid;
}

// A line's scripts, marked in its words (OcrWord::scripts) and in the line (scripts_of_words). Its characters
// that run across, upright (their angle within 45 degrees of none): their median size as drawn; its body those at
// 0.9 of that or more, at least 3 of 4 or more, their loose boxes' median centre and height; a character at most
// rule.script_size of the median, its centre more than rule.script_shift of the height above the body's centre a
// superscript, below it a subscript.
void mark_scripts(OcrLine& line, std::span<const Placed> placed, const TextLayerRule& rule) {
  auto across = [](const PdfChar& c) {
    return c.angle < 0 || std::fmod(c.angle + 45.0f, 360.0f) < 90.0f;  // within 45 degrees of upright
  };
  auto drawn = [](const PdfChar& c) { return c.size * c.matrix_scale; };
  std::vector<float> sizes;
  for (const Placed& p : placed) {
    if (across(*p.c)) sizes.push_back(drawn(*p.c));
  }
  if (sizes.size() < 4) return;
  const float ms = median(sizes);
  if (!(ms > 0)) return;
  std::vector<float> centres, heights;
  for (const Placed& p : placed) {
    if (!across(*p.c) || drawn(*p.c) < 0.9f * ms) continue;
    centres.push_back((p.c->ly0 + p.c->ly1) / 2);
    heights.push_back(p.c->ly1 - p.c->ly0);
  }
  if (centres.size() < 3) return;
  const float mcy = median(centres), shift = static_cast<float>(rule.script_shift) * median(heights);
  bool any = false;
  for (const Placed& p : placed) {
    const PdfChar& c = *p.c;
    if (!across(c) || drawn(c) > static_cast<float>(rule.script_size) * ms) continue;
    const float cy = (c.ly0 + c.ly1) / 2;
    if (std::abs(cy - mcy) <= shift) continue;
    const bool sub = cy > mcy;  // y runs down the page
    std::vector<TextScript>& s = line.words[p.word].scripts;
    if (!s.empty() && s.back().end == p.begin && s.back().sub == sub) {
      s.back().end = p.end;  // the run goes on
    } else {
      s.push_back({p.begin, p.end, sub});
    }
    any = true;
  }
  if (any) line.scripts = scripts_of_words(line.words);
}

}  // namespace

TextLayer read_text_layer(std::span<const PdfChar> drawn, const GrayView& page, const TextLayerRule& rule) {
  const std::vector<PdfChar> whole = paired(drawn);
  const std::span<const PdfChar> chars(whole);
  TextLayer out;
  TextLayerCheck& k = out.check;
  std::vector<uint8_t> aside(chars.size(), 0);  // in a bad-code run set aside
  auto bad = [](const PdfChar& c) { return c.bad_map || bad_code(c.code); };
  auto count = [&] {  // the checks' counts over the drawn characters not set aside
    TextLayerCheck t;
    for (size_t i = 0; i < chars.size(); ++i) {
      const PdfChar& c = chars[i];
      if (c.generated) continue;
      if (aside[i]) {
        ++t.set_aside;
        continue;
      }
      ++t.chars;
      if (bad(c)) ++t.bad;
      if (c.invisible) ++t.invisible;
      // ink: two tones over the box and its surround (a quarter of its height around it, a pixel at least), so a
      // glyph that fills its box shows against the paper about it, and white on white shows nothing
      const float m = std::max(1.0f, 0.25f * (c.y1 - c.y0));
      if (page.data && !space(c.code) && c.x1 > c.x0 && c.y1 > c.y0 &&
          spread(page, c.x0 - m, c.y0 - m, c.x1 + m, c.y1 + m) < rule.contrast) {
        ++t.no_ink;
      }
    }
    return t;
  };
  auto decide = [&](TextLayerCheck& t) {
    const double n = std::max(1, t.chars);
    if (t.chars < rule.min_chars) {
      t.reason = "too few characters (" + std::to_string(t.chars) + ")";
    } else if (t.bad / n > rule.max_bad) {
      t.reason = std::to_string(t.bad) + " of " + std::to_string(t.chars) + " characters are not readable text";
    } else if (t.invisible / n > rule.max_invisible) {
      t.reason = std::to_string(t.invisible) + " of " + std::to_string(t.chars) + " characters are invisible";
    } else if (t.no_ink / n > rule.max_no_ink) {
      t.reason = std::to_string(t.no_ink) + " of " + std::to_string(t.chars) + " characters have no ink";
    }
    t.usable = t.reason.empty();
  };
  for (const PdfChar& c : chars) out.marked += !c.generated && c.mcid >= 0;
  k = count();
  decide(k);
  const TextLayerCheck whole_check = k;
  // Refused for its bad codes, but all of them in runs of their own that hold no letter: those runs set aside,
  // the checks on the rest
  bool setting_aside = false;
  if (rule.set_aside_bad_runs && !k.usable && k.chars >= rule.min_chars && k.bad > rule.max_bad * k.chars) {
    std::map<int, bool> runs;  // the text objects with bad codes, and whether one holds a letter
    for (const PdfChar& c : chars) {
      if (!c.generated && bad(c)) runs.emplace(c.object, false);
    }
    for (const PdfChar& c : chars) {
      const auto it = runs.find(c.object);
      if (!c.generated && it != runs.end() && !bad(c) && letter(c.code)) it->second = true;
    }
    if (!runs.count(-1) && std::none_of(runs.begin(), runs.end(), [](const auto& r) { return r.second; })) {
      for (size_t i = 0; i < chars.size(); ++i) aside[i] = !chars[i].generated && runs.count(chars[i].object);
      TextLayerCheck rest = count();
      decide(rest);
      if (rest.usable) {
        k = rest;
        setting_aside = true;
      } else {
        std::fill(aside.begin(), aside.end(), uint8_t{0});
      }
    }
  }

  // The lines, and their words.
  OcrLine line;
  OcrWord word;
  bool open = false, pending_space = false;
  const PdfChar* prev = nullptr;
  Run run = Run::kUnknown;  // the open line's
  auto end_word = [&] {
    if (word.text.empty()) return;
    word.confidence = 1.0f;
    line.words.push_back(std::move(word));
    word = OcrWord{};
  };
  int line_mcid = -1;
  std::vector<Placed> placed;  // the line's characters, for its scripts
  bool touched = false;        // the line holds a character set aside: its box to OCR, not the line
  auto close = [&] {
    end_word();
    if (open && !line.text.empty()) {
      if (touched) {
        out.set_aside.push_back(line.box);
      } else {
        line.confidence = 1.0f;
        mark_scripts(line, placed, rule);
        out.lines.push_back(line);
        out.mcids.push_back(line_mcid);
      }
    }
    line = OcrLine{};
    placed.clear();
    open = pending_space = touched = false;
    prev = nullptr;
    run = Run::kUnknown;
  };
  for (size_t i = 0; i < chars.size(); ++i) {
    const PdfChar& c = chars[i];
    if (c.generated) {
      if (c.code == U'\r' || c.code == U'\n') {
        close();
      } else if (space(c.code)) {
        pending_space = open;
        end_word();
      }
      continue;
    }
    if (prev && !carries_on(*prev, c, line.box, run)) close();
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
    const auto begin = static_cast<uint32_t>(word.text.size());
    put_utf8(word.text, c.code);
    placed.push_back({&c, line.words.size(), begin, static_cast<uint32_t>(word.text.size())});
    touched |= aside[i] != 0;
    if (prev && run == Run::kUnknown) run = step(*prev, c);
    prev = &c;
  }
  close();
  // the runs set aside touch over half the layer's lines: the page refused as a whole, as before
  if (setting_aside && 2 * out.set_aside.size() > out.lines.size() + out.set_aside.size()) {
    k = whole_check;
    out.set_aside.clear();
  }
  // the page's ruled lines, for the false-table check (GridRule::check), on the render as drawn
  if (rule.rules && k.usable && page.data) {
    out.rules = detail::find_rules(page, page.width, page.height, detail::kRuleDarkFull);
  }
  return out;
}

bool reads_across(const TextLayer& layer) {
  int lines = 0, across = 0, words = 0, wide = 0;
  for (const OcrLine& l : layer.lines) {
    for (const OcrWord& w : l.words) {
      ++words;
      wide += w.box.x1 - w.box.x0 >= w.box.y1 - w.box.y0;
    }
    if (l.words.size() < 2) continue;
    ++lines;
    bool right = l.box.x1 - l.box.x0 >= l.box.y1 - l.box.y0;
    for (size_t i = 1; right && i < l.words.size(); ++i) {
      const LineBox &a = l.words[i - 1].box, &b = l.words[i].box;
      right = b.x0 >= a.x0 + 0.25 * (a.x1 - a.x0);  // the next word starts further right
    }
    across += right;
  }
  if (lines > 0) return across >= 0.9 * lines;
  return words > 0 && 2 * wide >= words;  // single words only: most of them wider than tall
}

int upright_turn(std::span<const PdfChar> chars) {
  int drawn = 0;
  int quarter[4] = {0, 0, 0, 0};
  for (const PdfChar& c : chars) {
    if (c.generated) continue;
    ++drawn;
    if (c.angle < 0) continue;
    const double k = c.angle / 90.0, q = std::round(k);
    if (std::abs(k - q) < 0.05) ++quarter[static_cast<int>(q) % 4];
  }
  int most = 1;
  for (int q = 2; q < 4; ++q) {
    if (quarter[q] > quarter[most]) most = q;
  }
  return 2 * quarter[most] > drawn ? (4 - most) % 4 : 0;  // run `most` quarters clockwise: turned back
}

void read_turned(OcrPage& page, int turn) {
  turn = ((turn % 4) + 4) % 4;
  if (turn == 0) return;
  page.rotation = 90 * ((4 - turn) % 4);
  if (turn % 2) std::swap(page.width, page.height);
}

bool lines_from_detection(TextLayer& layer, std::span<const LineBox> detected, double tall) {
  if (!reads_across(layer)) return false;  // a page turned, or set down the page: the layer's own lines
  const size_t m = detected.size();
  std::vector<std::vector<OcrWord>> into(m);  // each detected line's words
  std::vector<OcrLine> rest;                  // the layer's lines, with the words no detected line took
  for (const OcrLine& l : layer.lines) {
    std::vector<size_t> at(l.words.size(), m);  // the detected line each word goes into
    for (size_t j = 0; j < l.words.size(); ++j) {
      const LineBox& w = l.words[j].box;
      const double cx = (w.x0 + w.x1) / 2, cy = (w.y0 + w.y1) / 2;
      const double h = std::max(1.0, w.y1 - w.y0);
      double least = 1e300;
      for (size_t k = 0; k < m; ++k) {
        const LineBox& d = detected[k];
        if (cx < d.x0 || cx > d.x1 || cy < d.y0 || cy > d.y1 || d.y1 - d.y0 > tall * h) continue;
        if (area(d) < least) {
          least = area(d);
          at[j] = k;
        }
      }
    }
    // a word no detected line holds (a superscript, a subscript above or below the line's band) stays with its
    // line's words: in the detected line of its nearest neighbour in the layer's line that went into one
    const std::vector<size_t> found = at;
    for (size_t j = 0; j < at.size(); ++j) {
      if (found[j] < m) continue;
      for (size_t d = 1; d < at.size(); ++d) {  // the nearest before it first, at each distance
        if (j >= d && found[j - d] < m) {
          at[j] = found[j - d];
          break;
        }
        if (j + d < at.size() && found[j + d] < m) {
          at[j] = found[j + d];
          break;
        }
      }
    }
    OcrLine left;
    left.confidence = 1.0f;
    for (size_t j = 0; j < l.words.size(); ++j) {
      const OcrWord& w = l.words[j];
      if (at[j] < m) {
        into[at[j]].push_back(w);
        continue;
      }
      if (left.words.empty()) {
        left.box = w.box;
      } else {
        left.box = {std::min(left.box.x0, w.box.x0), std::min(left.box.y0, w.box.y0), std::max(left.box.x1, w.box.x1),
                    std::max(left.box.y1, w.box.y1)};
        left.text += ' ';
      }
      left.text += w.text;
      left.words.push_back(w);
    }
    if (!left.words.empty()) {
      left.scripts = scripts_of_words(left.words);
      rest.push_back(std::move(left));
    }
  }
  std::vector<OcrLine> lines;
  for (size_t k = 0; k < m; ++k) {
    std::vector<OcrWord>& ws = into[k];
    if (ws.empty()) continue;
    std::stable_sort(ws.begin(), ws.end(), [](const OcrWord& a, const OcrWord& b) { return a.box.x0 < b.box.x0; });
    OcrLine l;
    l.confidence = 1.0f;
    l.box = ws.front().box;
    for (size_t i = 0; i < ws.size(); ++i) {
      const LineBox& b = ws[i].box;
      l.box = {std::min(l.box.x0, b.x0), std::min(l.box.y0, b.y0), std::max(l.box.x1, b.x1), std::max(l.box.y1, b.y1)};
      if (i) l.text += ' ';
      l.text += ws[i].text;
    }
    l.words = std::move(ws);
    l.scripts = scripts_of_words(l.words);
    lines.push_back(std::move(l));
  }
  for (OcrLine& l : rest) lines.push_back(std::move(l));
  layer.lines = std::move(lines);
  layer.mcids.assign(layer.lines.size(), -1);
  return true;
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
  page.rules = layer.rules;
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

std::vector<LineBox> set_aside_to_read(const TextLayer& layer, int width, int height) {
  std::vector<LineBox> out;
  for (const LineBox& b : layer.set_aside) {
    const double m = std::max(1.0, 0.1 * (b.y1 - b.y0));
    const LineBox c{std::max(0.0, b.x0 - m), std::max(0.0, b.y0 - m), std::min<double>(width, b.x1 + m),
                    std::min<double>(height, b.y1 + m)};
    if (c.x1 > c.x0 && c.y1 > c.y0) out.push_back(c);
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

OcrPage picture_lines(const OcrPage& found, const OcrPage& page, std::span<const LineBox> to_read,
                      const HybridRule& rule) {
  OcrPage out;
  out.width = found.width;
  out.height = found.height;
  out.pictures = found.pictures;
  out.picture_mask = found.picture_mask;
  out.second_pass = found.second_pass;
  for (const OcrLine& l : found.lines) {
    const double a = area(l.box);
    if (a <= 0) continue;
    double inside = 0, under = 0;  // the regions to read do not overlap one another (pictures may: capped)
    for (const LineBox& r : to_read) inside += overlap(l.box, r);
    for (const OcrLine& own : page.lines) under += overlap(l.box, own.box);
    if (std::min(inside, a) < 0.5 * a || under >= rule.max_covered * a) continue;
    out.lines.push_back(l);
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
