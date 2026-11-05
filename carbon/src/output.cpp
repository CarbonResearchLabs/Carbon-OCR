// The engine's pages as hOCR, ALTO, PAGE XML, markdown and HTML (carbon/output.h).

#include "carbon/output.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <optional>
#include <string>
#include <vector>

#include "carbon/error.h"
#include "carbon/version.h"
#include "escape_scan.h"

namespace carbon {
namespace {

// Text for XML: & < > " ' escaped, control characters (not XML) left out.
std::string xml(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    const size_t j = detail::special_from(s, i);  // the run written as itself, copied whole
    out.append(s, i, j - i);
    if (j == s.size()) break;
    i = j + 1;
    const char ch = s[j];
    switch (ch) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '"': out += "&quot;"; break;
      case '\'': out += "&apos;"; break;
      case '\\': out += ch; break;
      default:
        if (ch == '\t' || ch == '\n') out += ch;  // other control bytes are not XML's
    }
  }
  return out;
}

struct Px {
  long x0, y0, x1, y1;
};

// A box to whole pixels: its corners rounded outward.
Px px(const LineBox& b) {
  return {std::lround(std::floor(b.x0)), std::lround(std::floor(b.y0)), std::lround(std::ceil(b.x1)),
          std::lround(std::ceil(b.y1))};
}

std::string fmt(const char* f, ...) {
  char buf[512];
  va_list args;
  va_start(args, f);
  std::vsnprintf(buf, sizeof buf, f, args);
  va_end(args);
  return buf;
}

// A page's blocks as written: the order model's, or one of every line.
struct Region {
  std::string label;  // empty: no model's
  std::vector<int> lines;
  const OcrBlock* block = nullptr;
};

std::vector<Region> regions(const OcrPage& page) {
  std::vector<Region> r;
  if (page.blocks.empty()) {
    if (page.lines.empty()) return r;
    Region all;
    for (size_t i = 0; i < page.lines.size(); ++i) all.lines.push_back(static_cast<int>(i));
    r.push_back(std::move(all));
    return r;
  }
  for (const OcrBlock& b : page.blocks) {
    if (!b.lines.empty()) r.push_back({b.label, b.lines, &b});
  }
  return r;
}

// The box around some of a page's lines.
LineBox lines_box(const OcrPage& page, const std::vector<int>& lines) {
  LineBox b{1e30, 1e30, -1e30, -1e30};
  for (int i : lines) {
    const LineBox& l = page.lines[static_cast<size_t>(i)].box;
    b = {std::min(b.x0, l.x0), std::min(b.y0, l.y0), std::max(b.x1, l.x1), std::max(b.y1, l.y1)};
  }
  return b;
}

// A line's words; a line without words (none were found) is one word of its text.
std::vector<OcrWord> words_of(const OcrLine& l) {
  if (!l.words.empty()) return l.words;
  if (l.text.empty()) return {};
  return {OcrWord{l.box, l.text, l.confidence}};
}

void check(std::span<const OcrPage> pages, std::span<const std::string> images) {
  if (pages.size() != images.size()) throw Error("output: " + std::to_string(pages.size()) + " pages, " +
                                                 std::to_string(images.size()) + " image names");
}

std::string now_utc() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
#if defined(_WIN32)
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  char buf[32];
  std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

}  // namespace

std::string to_hocr(std::span<const OcrPage> pages, std::span<const std::string> images) {
  check(pages, images);
  std::string s =
      "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
      "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 Transitional//EN\"\n"
      "    \"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd\">\n"
      "<html xmlns=\"http://www.w3.org/1999/xhtml\" xml:lang=\"en\" lang=\"en\">\n"
      " <head>\n"
      "  <title></title>\n"
      "  <meta http-equiv=\"Content-Type\" content=\"text/html;charset=utf-8\"/>\n"
      "  <meta name=\"ocr-system\" content=\"carbon-ocr " +
      std::string(kVersion) +
      "\"/>\n"
      "  <meta name=\"ocr-capabilities\" content=\"ocr_page ocr_carea ocr_par ocr_line ocr_header ocr_caption "
      "ocrx_word ocrp_wconf\"/>\n"
      " </head>\n"
      " <body>\n";
  int line_id = 0, word_id = 0;
  for (size_t p = 0; p < pages.size(); ++p) {
    const OcrPage& page = pages[p];
    const int n = static_cast<int>(p) + 1;
    s += fmt("  <div class=\"ocr_page\" id=\"page_%d\" title=\"image &quot;", n) + xml(images[p]) +
         fmt("&quot;; bbox 0 0 %d %d; ppageno %zu\">\n", page.width, page.height, p);
    int k = 0;
    for (const Region& r : regions(page)) {
      const Px b = px(box_as_given(lines_box(page, r.lines), page));
      const std::string bbox = fmt("bbox %ld %ld %ld %ld", b.x0, b.y0, b.x1, b.y1);
      ++k;
      s += fmt("   <div class=\"ocr_carea\" id=\"block_%d_%d\" title=\"", n, k) + bbox + "\">\n";
      s += fmt("    <p class=\"ocr_par\" id=\"par_%d_%d\" title=\"", n, k) + bbox + "\">\n";
      const char* kind = r.label == "Section-Header" ? "ocr_header" : r.label == "Caption" ? "ocr_caption" : "ocr_line";
      for (int li : r.lines) {
        const OcrLine& l = page.lines[static_cast<size_t>(li)];
        const Px lb = px(box_as_given(l.box, page)), read = px(l.box);  // x_size: the text's height, as read
        s += fmt("     <span class=\"%s\" id=\"line_%d\" title=\"bbox %ld %ld %ld %ld; baseline 0 0; "
                 "x_size %ld; x_descenders 0; x_ascenders 0\">",
                 kind, ++line_id, lb.x0, lb.y0, lb.x1, lb.y1, read.y1 - read.y0);
        bool first = true;
        for (const OcrWord& w : words_of(l)) {
          const Px wb = px(box_as_given(w.box, page));
          if (!first) s += ' ';
          first = false;
          s += fmt("<span class=\"ocrx_word\" id=\"word_%d\" title=\"bbox %ld %ld %ld %ld; x_wconf %d\">", ++word_id,
                   wb.x0, wb.y0, wb.x1, wb.y1, static_cast<int>(std::lround(100.0 * w.confidence))) +
               xml(w.text) + "</span>";
        }
        s += "</span>\n";
      }
      s += "    </p>\n   </div>\n";
    }
    s += "  </div>\n";
  }
  s += " </body>\n</html>\n";
  return s;
}

std::string to_alto(std::span<const OcrPage> pages, std::span<const std::string> images) {
  check(pages, images);
  std::string s =
      "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
      "<alto xmlns=\"http://www.loc.gov/standards/alto/ns-v4#\" "
      "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" "
      "xsi:schemaLocation=\"http://www.loc.gov/standards/alto/ns-v4# "
      "http://www.loc.gov/alto/v4/alto-4-4.xsd\">\n"
      " <Description>\n"
      "  <MeasurementUnit>pixel</MeasurementUnit>\n"
      "  <sourceImageInformation>\n";
  s += "   <fileName>" + xml(images.empty() ? std::string() : images[0]) + "</fileName>\n";
  s += "  </sourceImageInformation>\n"
       "  <OCRProcessing ID=\"OCR_0\">\n"
       "   <ocrProcessingStep>\n"
       "    <processingSoftware>\n"
       "     <softwareName>carbon-ocr</softwareName>\n"
       "     <softwareVersion>" +
       std::string(kVersion) +
       "</softwareVersion>\n"
       "    </processingSoftware>\n"
       "   </ocrProcessingStep>\n"
       "  </OCRProcessing>\n"
       " </Description>\n";
  std::vector<bool> used(kBlockLabels, false);
  for (const OcrPage& page : pages) {
    for (const OcrBlock& b : page.blocks) {
      if (block_label(b.label) >= 0) used[static_cast<size_t>(block_label(b.label))] = true;
    }
  }
  if (std::find(used.begin(), used.end(), true) != used.end()) {
    s += " <Tags>\n";
    for (int c = 0; c < kBlockLabels; ++c) {
      if (!used[static_cast<size_t>(c)]) continue;
      s += fmt("  <LayoutTag ID=\"layout_%d\" LABEL=\"", c) + block_label_name(c) + "\"/>\n";
    }
    s += " </Tags>\n";
  }
  s += " <Layout>\n";
  auto pos = [](const Px& b) {
    return fmt("HPOS=\"%ld\" VPOS=\"%ld\" WIDTH=\"%ld\" HEIGHT=\"%ld\"", b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0);
  };
  int line_id = 0, word_id = 0;
  for (size_t p = 0; p < pages.size(); ++p) {
    const OcrPage& page = pages[p];
    s += fmt("  <Page ID=\"page_%zu\" PHYSICAL_IMG_NR=\"%zu\" WIDTH=\"%d\" HEIGHT=\"%d\">\n", p, p + 1, page.width,
             page.height);
    s += fmt("   <PrintSpace HPOS=\"0\" VPOS=\"0\" WIDTH=\"%d\" HEIGHT=\"%d\">\n", page.width, page.height);
    int k = 0;
    for (const Region& r : regions(page)) {
      const int tag = block_label(r.label);
      s += fmt("    <TextBlock ID=\"block_%zu_%d\" ", p, k++) + pos(px(box_as_given(lines_box(page, r.lines), page))) +
           (tag >= 0 ? fmt(" TAGREFS=\"layout_%d\"", tag) : std::string()) + ">\n";
      for (int li : r.lines) {
        const OcrLine& l = page.lines[static_cast<size_t>(li)];
        s += fmt("     <TextLine ID=\"line_%d\" ", line_id++) + pos(px(box_as_given(l.box, page))) + ">\n";
        const std::vector<OcrWord> ws = words_of(l);
        for (size_t w = 0; w < ws.size(); ++w) {
          const Px wb = px(box_as_given(ws[w].box, page));
          if (w) {  // the space between this word and the last
            const Px prev = px(box_as_given(ws[w - 1].box, page));
            const long gap = std::max(0L, wb.x0 - prev.x1);
            s += fmt("      <SP WIDTH=\"%ld\" HPOS=\"%ld\" VPOS=\"%ld\"/>\n", gap, prev.x1, wb.y0);
          }
          s += fmt("      <String ID=\"string_%d\" ", word_id++) + pos(wb) +
               fmt(" WC=\"%.2f\" CONTENT=\"", ws[w].confidence) + xml(ws[w].text) + "\"/>\n";
        }
        s += "     </TextLine>\n";
      }
      s += "    </TextBlock>\n";
    }
    s += "   </PrintSpace>\n  </Page>\n";
  }
  s += " </Layout>\n</alto>\n";
  return s;
}

std::string to_page_xml(const OcrPage& page, const std::string& image) {
  const std::string when = now_utc();
  // A box's four corners on the page as given (its polygon holds a levelled page's box exactly), rounded.
  auto coords = [&page](const LineBox& box) {
    if (page.rotation == 0 && page.skew == 0) {  // the box as given: its corners rounded outward, as hOCR's
      const Px b = px(box);
      return fmt("<Coords points=\"%ld,%ld %ld,%ld %ld,%ld %ld,%ld\"/>", b.x0, b.y0, b.x1, b.y0, b.x1, b.y1, b.x0,
                 b.y1);
    }
    const auto q = corners_as_given(box, page);
    auto r = [](double v) { return std::lround(v); };
    return fmt("<Coords points=\"%ld,%ld %ld,%ld %ld,%ld %ld,%ld\"/>", r(q[0][0]), r(q[0][1]), r(q[1][0]), r(q[1][1]),
               r(q[2][0]), r(q[2][1]), r(q[3][0]), r(q[3][1]));
  };
  auto text_equiv = [](const std::string& text, float conf) {
    return fmt("<TextEquiv conf=\"%.4f\"><Unicode>", conf) + xml(text) + "</Unicode></TextEquiv>";
  };
  std::string s =
      "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
      "<PcGts xmlns=\"http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15\" "
      "xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" "
      "xsi:schemaLocation=\"http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15 "
      "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15/pagecontent.xsd\">\n"
      " <Metadata>\n"
      "  <Creator>carbon-ocr " +
      std::string(kVersion) + "</Creator>\n  <Created>" + when + "</Created>\n  <LastChange>" + when +
      "</LastChange>\n </Metadata>\n";
  s += " <Page imageFilename=\"" + xml(image) + fmt("\" imageWidth=\"%d\" imageHeight=\"%d\">\n", page.width,
                                                       page.height);
  // PAGE's region types for the labels; the label itself in custom
  auto type_of = [](const std::string& label) -> const char* {
    if (label.empty() || label == "Text" || label == "List-Group") return "paragraph";
    if (label == "Section-Header") return "heading";
    if (label == "Page-Header") return "header";
    if (label == "Page-Footer") return "footer";
    if (label == "Caption") return "caption";
    if (label == "Footnote") return "footnote";
    if (label == "Table-Of-Contents") return "TOC-entry";
    return "other";
  };
  size_t k = 0;
  for (const Region& r : regions(page)) {
    s += fmt("  <TextRegion id=\"r%zu\" type=\"%s\"", k, type_of(r.label)) +
         (r.label.empty() ? std::string() : " custom=\"structure {type:" + xml(r.label) + ";}\"") + ">\n   " +
         coords(lines_box(page, r.lines)) + "\n";
    std::string region_text;
    for (size_t i = 0; i < r.lines.size(); ++i) {
      const OcrLine& l = page.lines[static_cast<size_t>(r.lines[i])];
      s += fmt("   <TextLine id=\"r%zul%zu\">\n    ", k, i) + coords(l.box) + "\n";
      const std::vector<OcrWord> ws = words_of(l);
      for (size_t w = 0; w < ws.size(); ++w) {
        s += fmt("    <Word id=\"r%zul%zuw%zu\">", k, i, w) + coords(ws[w].box) +
             text_equiv(ws[w].text, ws[w].confidence) + "</Word>\n";
      }
      s += "    " + text_equiv(l.text, l.confidence) + "\n   </TextLine>\n";
      region_text += (i ? "\n" : "") + l.text;
    }
    s += "   <TextEquiv><Unicode>" + xml(region_text) + "</Unicode></TextEquiv>\n  </TextRegion>\n";
    ++k;
  }
  s += " </Page>\n</PcGts>\n";
  return s;
}

namespace {

// ------------------------------------------------------------------ UTF-8, letters and paragraphs

// The code point starting at s[i] and its length in bytes; a byte that does not start a well-formed sequence is
// itself, one byte long.
std::pair<uint32_t, size_t> code_at(const std::string& s, size_t i) {
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

// Where the code point that ends at s[end) starts (end > 0).
size_t code_before(const std::string& s, size_t end) {
  size_t i = end - 1;
  while (i > 0 && end - i < 4 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80) --i;
  return i;
}

// A lowercase letter: ASCII; Latin-1; Latin Extended-A and the regular pairs of Extended-B and Extended
// Additional; IPA; Greek and Cyrillic.
bool is_lower(uint32_t c) {
  if (c < 0x80) return c >= 'a' && c <= 'z';
  if (c < 0x100) return c >= 0xDF && c != 0xF7;
  if (c <= 0x137) return c % 2 == 1;
  if (c == 0x138 || c == 0x149 || c == 0x17F) return true;  // ĸ, ŉ, ſ
  if (c <= 0x148) return c % 2 == 0;                         // Ĺ ĺ .. Ň ň
  if (c <= 0x177) return c % 2 == 1;                         // Ŋ ŋ .. Ŷ ŷ
  if (c <= 0x17E) return c >= 0x179 && c % 2 == 0;           // Ÿ; Ź ź .. Ž ž
  if (c <= 0x24F) {
    if (c >= 0x1CD && c <= 0x1DC) return c % 2 == 0;
    return ((c >= 0x1DE && c <= 0x1EF) || c >= 0x1F8) && c % 2 == 1;
  }
  if (c <= 0x2AF) return true;  // IPA
  if (c >= 0x3AC && c <= 0x3CE) return true;
  if (c >= 0x430 && c <= 0x45F) return true;
  if ((c >= 0x460 && c <= 0x481) || (c >= 0x48A && c <= 0x4BF) || (c >= 0x4D0 && c <= 0x52F)) return c % 2 == 1;
  if (c >= 0x1E00 && c <= 0x1EFF) return c % 2 == 1;
  return false;
}

// A letter, as far as a hyphen's join needs: ASCII's; above, any code point outside the punctuation, symbols,
// spacing modifiers and combining marks.
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

bool starts_lowercase(const std::string& t) { return !t.empty() && is_lower(code_at(t, 0).first); }

// The letters of a text, and its characters that are not spaces.
std::pair<int, int> letters_of(const std::string& t) {
  int letters = 0, chars = 0;
  for (size_t i = 0; i < t.size();) {
    const auto [c, n] = code_at(t, i);
    i += n;
    if (c == ' ' || c == '\t') continue;
    ++chars;
    letters += is_letter(c);
  }
  return {letters, chars};
}

// t joined to a paragraph's text so far: by a space, or, where a hyphen broke a word at the line's end (a letter,
// then -, U+00AD, U+2010 or ¬, then a lowercase letter on the next line), by nothing, the hyphen dropped.
void join_into(std::string& out, const std::string& t) {
  if (t.empty()) return;
  if (out.empty()) {
    out = t;
    return;
  }
  const size_t h = code_before(out, out.size());
  const uint32_t hy = code_at(out, h).first;
  const bool broken = (hy == '-' || hy == 0xAD || hy == 0x2010 || hy == 0xAC) && h > 0 &&
                      is_letter(code_at(out, code_before(out, h)).first) && starts_lowercase(t);
  if (broken) {
    out.resize(h);
  } else {
    out += ' ';
  }
  out += t;
}

// A block's lines as one paragraph's text: joined by spaces, and a word that a hyphen broke at a line's end
// joined again, its hyphen dropped (join_into).
std::string paragraph(const OcrPage& page, const std::vector<int>& lines) {
  std::string out;
  for (int i : lines) join_into(out, page.lines[static_cast<size_t>(i)].text);
  return out;
}

// A Section-Header block that reads as a heading: at least half its characters letters, its lines read surely
// (their mean confidence 0.8 or more). Else it is text the order model took for one (a formula read as letters
// and bars), written as a paragraph.
bool heading_like(const OcrPage& page, const std::vector<int>& lines) {
  int letters = 0, chars = 0;
  double conf = 0;
  for (int i : lines) {
    const auto [l, c] = letters_of(page.lines[static_cast<size_t>(i)].text);
    letters += l;
    chars += c;
    conf += page.lines[static_cast<size_t>(i)].confidence;
  }
  return chars > 0 && 2 * letters >= chars && conf >= 0.8 * static_cast<double>(lines.size());
}

// A picture's lines worth writing: read surely (confidence 0.9 or more), four letters at least. The rest are what
// the recognizer made of a photograph's or a chart's marks; the JSON keeps every line.
std::vector<int> sure_lines(const OcrPage& page, const std::vector<int>& lines) {
  std::vector<int> out;
  for (int i : lines) {
    const OcrLine& l = page.lines[static_cast<size_t>(i)];
    if (l.confidence >= 0.9f && letters_of(l.text).first >= 4) out.push_back(i);
  }
  return out;
}

// ------------------------------------------------------------------ markdown

// A paragraph's text as markdown's: its own characters at a line's start would make it a heading, a list, a quote,
// a table (# > - + * |), a fence (``` ~~~), an HTML block (< then a letter, / or !), or a thematic break (a line of
// _ or = alone).
std::string md_line(const std::string& text) {
  if (text.empty()) return text;
  const char c = text[0];
  bool escape = std::string("#>-+*|").find(c) != std::string::npos;
  escape |= text.rfind("```", 0) == 0 || text.rfind("~~~", 0) == 0;
  escape |= c == '<' && text.size() > 1 && (std::isalpha(static_cast<unsigned char>(text[1])) || text[1] == '/' ||
                                            text[1] == '!' || text[1] == '?');
  if (c == '_' || c == '=') escape |= text.find_first_not_of(std::string(1, c) + " \t") == std::string::npos;
  return escape ? "\\" + text : text;
}

std::string html(const std::string& t) {
  std::string out;
  for (char c : t) out += c == '&' ? "&amp;" : c == '<' ? "&lt;" : c == '>' ? "&gt;" : std::string(1, c);
  return out;
}

// A list item's text without its bullet, or nullopt where the line has no bullet.
std::optional<std::string> bulleted(const std::string& t) {
  static const char* const kBullets[] = {"•", "◦", "▪", "▫", "‣", "●", "○", "■", "□", "-", "–", "—", "*", "·", "►",
                                         "▸", "➢", "✓"};
  for (const char* b : kBullets) {
    const std::string mark = std::string(b) + " ";
    if (t.rfind(mark, 0) == 0) return t.substr(mark.size());
  }
  return std::nullopt;
}

// Whether a line starts with a number markdown reads as an ordered list's: 1-3 digits then ". " or ") ".
bool numbered(const std::string& t) {
  size_t i = 0;
  while (i < t.size() && i < 3 && t[i] >= '0' && t[i] <= '9') ++i;
  return i > 0 && i + 1 < t.size() && (t[i] == '.' || t[i] == ')') && t[i + 1] == ' ';
}

// A table's cells as HTML, markdown's and HTML's alike: a <td> a cell at its row and column, colspan and rowspan
// where it covers more (OcrCell's spans), the places a span covers left out; a cell's lines one paragraph. A cell at
// a place another's span covers gives its lines to that one, so no text is lost.
std::string table_html(const OcrPage& page, const OcrBlock& b) {
  int rows = 0, cols = 0;
  for (const OcrCell& c : b.cells) {
    rows = std::max(rows, c.row + std::max(1, c.row_span));
    cols = std::max(cols, c.col + std::max(1, c.col_span));
  }
  struct Slot {
    std::vector<int> lines;
    int row_span = 1, col_span = 1;
    int owner = -1;  // the slot whose span covers this one; -1: its own
  };
  std::vector<Slot> g(static_cast<size_t>(rows) * cols);
  auto at = [&](int r, int c) -> Slot& { return g[static_cast<size_t>(r) * cols + c]; };
  for (const OcrCell& c : b.cells) {
    Slot* s = &at(c.row, c.col);
    if (s->owner >= 0) {
      s = &g[static_cast<size_t>(s->owner)];
    } else if (s->lines.empty() && s->row_span == 1 && s->col_span == 1) {
      s->row_span = std::max(1, c.row_span);
      s->col_span = std::max(1, c.col_span);
      for (int r = c.row; r < c.row + s->row_span; ++r) {
        for (int k = c.col; k < c.col + s->col_span; ++k) {
          if ((r != c.row || k != c.col) && at(r, k).owner < 0) at(r, k).owner = c.row * cols + c.col;
        }
      }
    }
    s->lines.insert(s->lines.end(), c.lines.begin(), c.lines.end());
  }
  std::string out = "<table>\n";
  for (int r = 0; r < rows; ++r) {
    out += "<tr>";
    for (int k = 0; k < cols; ++k) {
      const Slot& s = at(r, k);
      if (s.owner >= 0) continue;
      out += "<td";
      if (s.col_span > 1) out += " colspan=\"" + std::to_string(s.col_span) + "\"";
      if (s.row_span > 1) out += " rowspan=\"" + std::to_string(s.row_span) + "\"";
      out += ">" + html(paragraph(page, s.lines)) + "</td>";
    }
    out += "</tr>\n";
  }
  return out + "</table>";
}

// A block as markdown; empty for one left out.
std::string md_block(const OcrPage& page, const Region& r) {
  auto text = [&](int i) { return page.lines[static_cast<size_t>(i)].text; };
  if (r.label == "Page-Header" || r.label == "Page-Footer") return "";
  if (r.label == "Section-Header") {
    const std::string h = paragraph(page, r.lines);
    return heading_like(page, r.lines) ? "## " + h : md_line(h);
  }
  if (r.label == "Caption") return "*" + paragraph(page, r.lines) + "*";
  if (r.label == "Picture") return md_line(paragraph(page, sure_lines(page, r.lines)));
  if (r.label == "List-Group") {
    std::vector<std::string> items;
    for (int i : r.lines) {
      const std::string t = text(i);
      if (const auto item = bulleted(t)) {
        items.push_back("- " + *item);
      } else if (numbered(t)) {
        items.push_back(t);
      } else if (!items.empty()) {
        join_into(items.back(), t);  // the item before, carried on
      } else {
        items.push_back("- " + t);
      }
    }
    std::string out;
    for (size_t k = 0; k < items.size(); ++k) out += (k ? "\n" : "") + items[k];
    return out;
  }
  if (r.label == "Table" && r.block && !r.block->cells.empty()) return table_html(page, *r.block);
  return md_line(paragraph(page, r.lines));  // a paragraph: its lines joined, as HTML's <p>
}

// ------------------------------------------------------------------ HTML

// A list block as HTML: runs of numbered items an <ol> (from the first's number,
// the numbers dropped from the text, as a browser draws them back), the rest a
// <ul>, their bullets dropped; a line without a marker carries on the item before.
std::string html_list(const OcrPage& page, const std::vector<int>& lines) {
  struct Item {
    bool ordered;
    int number;
    std::string text;
  };
  std::vector<Item> items;
  for (int i : lines) {
    const std::string& t = page.lines[static_cast<size_t>(i)].text;
    if (const auto b = bulleted(t)) {
      items.push_back({false, 0, *b});
    } else if (numbered(t)) {
      const size_t dot = t.find_first_of(".)");
      items.push_back({true, std::atoi(t.substr(0, dot).c_str()), t.substr(dot + 2)});
    } else if (!items.empty()) {
      join_into(items.back().text, t);  // the item before, carried on
    } else {
      items.push_back({false, 0, t});
    }
  }
  std::string out;
  for (size_t k = 0; k < items.size();) {
    const bool ordered = items[k].ordered;
    out += ordered ? (items[k].number == 1 ? "<ol>" : "<ol start=\"" + std::to_string(items[k].number) + "\">")
                   : "<ul>";
    for (; k < items.size() && items[k].ordered == ordered; ++k) out += "<li>" + html(items[k].text) + "</li>";
    out += ordered ? "</ol>\n" : "</ul>\n";
  }
  return out;
}

// A block as HTML, by its label.
std::string html_block(const OcrPage& page, const Region& r) {
  auto text = [&](int i) { return html(page.lines[static_cast<size_t>(i)].text); };
  auto each = [&](const char* tag) {
    std::string out;
    for (int i : r.lines) out += std::string("<") + tag + ">" + text(i) + "</" + tag + ">";
    return out;
  };
  const std::string para = html(paragraph(page, r.lines));
  if (r.label == "Section-Header") {
    return heading_like(page, r.lines) ? "<h2>" + para + "</h2>\n" : "<p>" + para + "</p>\n";
  }
  if (r.label == "Page-Header") return "<header>" + each("p") + "</header>\n";
  if (r.label == "Page-Footer") return "<footer>" + each("p") + "</footer>\n";
  if (r.label == "Caption") return "<figure><figcaption>" + para + "</figcaption></figure>\n";
  if (r.label == "Footnote") return "<aside class=\"footnote\"><p>" + para + "</p></aside>\n";
  if (r.label == "Formula") return "<p class=\"formula\">" + para + "</p>\n";
  if (r.label == "List-Group") return html_list(page, r.lines);
  if (r.label == "Table-Of-Contents") return "<nav><ul>" + each("li") + "</ul></nav>\n";
  if (r.label == "Form") return "<div class=\"form\">" + each("p") + "</div>\n";
  if (r.label == "Picture") {  // the picture's sure lines only (sure_lines)
    std::string out;
    for (int i : sure_lines(page, r.lines)) out += "<p>" + text(i) + "</p>";
    return out.empty() ? std::string() : "<figure>" + out + "</figure>\n";
  }
  if (r.label == "Table" && r.block && !r.block->cells.empty()) return table_html(page, *r.block) + "\n";
  if (r.label == "Table") return "<div class=\"table\">" + each("p") + "</div>\n";  // no cells found
  return "<p>" + para + "</p>\n";
}

}  // namespace

std::string to_html_body(const OcrPage& page) {
  std::string s;
  if (!page.blocks.empty()) {
    for (const Region& r : regions(page)) s += html_block(page, r);
    return s;
  }
  // No blocks: a paragraph where the gap between lines is wider than the median line.
  std::vector<double> heights;
  for (const OcrLine& l : page.lines) heights.push_back(l.box.y1 - l.box.y0);
  std::nth_element(heights.begin(), heights.begin() + heights.size() / 2, heights.end());
  const double h = heights.empty() ? 0 : heights[heights.size() / 2];
  std::vector<int> run;
  for (size_t i = 0; i < page.lines.size(); ++i) {
    if (i) {
      const LineBox& a = page.lines[i - 1].box;
      const LineBox& b = page.lines[i].box;
      if (b.y0 - a.y1 > h || b.y1 < a.y0) {
        s += "<p>" + html(paragraph(page, run)) + "</p>\n";
        run.clear();
      }
    }
    run.push_back(static_cast<int>(i));
  }
  if (!run.empty()) s += "<p>" + html(paragraph(page, run)) + "</p>\n";
  return s;
}

std::string to_html(std::span<const OcrPage> pages) {
  std::string s = "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"utf-8\">\n<meta name=\"generator\" content=\"" +
                  std::string("Carbon-OCR ") + std::string(kVersion) + "\">\n<title>Carbon-OCR</title>\n</head>\n<body>\n";
  for (size_t p = 0; p < pages.size(); ++p) {
    s += "<section class=\"page\" id=\"page-" + std::to_string(p + 1) + "\">\n" + to_html_body(pages[p]) +
         "</section>\n";
  }
  return s + "</body>\n</html>\n";
}

std::string to_markdown(std::span<const OcrPage> pages) {
  std::string s;
  for (size_t p = 0; p < pages.size(); ++p) {
    const OcrPage& page = pages[p];
    if (p) s += "\n---\n\n";
    if (!page.blocks.empty()) {  // the order model's blocks
      bool first = true;
      for (const Region& r : regions(page)) {
        const std::string b = md_block(page, r);
        if (b.empty()) continue;
        s += (first ? "" : "\n\n") + b;
        first = false;
      }
      if (!first) s += '\n';
      continue;
    }
    // the median line height: a gap wider than it starts a paragraph, its lines joined as a block's
    std::vector<double> heights;
    for (const OcrLine& l : page.lines) heights.push_back(l.box.y1 - l.box.y0);
    std::nth_element(heights.begin(), heights.begin() + heights.size() / 2, heights.end());
    const double h = heights.empty() ? 0 : heights[heights.size() / 2];
    std::vector<int> run;
    bool first = true;
    auto flush = [&] {
      const std::string para = md_line(paragraph(page, run));
      run.clear();
      if (para.empty()) return;
      s += (first ? "" : "\n\n") + para;
      first = false;
    };
    for (size_t i = 0; i < page.lines.size(); ++i) {
      if (i) {
        const LineBox& a = page.lines[i - 1].box;
        const LineBox& b = page.lines[i].box;
        if (b.y0 - a.y1 > h || b.y1 < a.y0) flush();
      }
      run.push_back(static_cast<int>(i));
    }
    flush();
    if (!first) s += '\n';
  }
  return s;
}

}  // namespace carbon
