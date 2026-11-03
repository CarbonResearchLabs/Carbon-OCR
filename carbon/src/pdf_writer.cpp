// A searchable PDF (carbon/output.h): each page's image under its words, the
// words invisible text in a font with no glyphs.
//
// The font is a TrueType font made here: two empty glyphs, 500 units wide on a
// 1000-unit em.  The text is two-byte codes (UTF-16 code units, Identity-H); every
// code shows glyph 1 (CIDToGIDMap) and reads back as itself (ToUnicode), so any
// text can be set in it, nothing of it is drawn, and a viewer selects and
// searches it.  Each word is set at its box: the font size the box's height, its
// baseline the em's descent above the box's bottom, and stretched (Tz) to the
// box's width.  A page orientation turned is embedded turned upright, as its
// boxes are.

#include <libdeflate.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "carbon/error.h"
#include "carbon/image.h"
#include "carbon/output.h"

namespace carbon {
namespace {

// ---------------------------------------------------------------- the font

void u16(std::string& s, uint32_t v) {
  s += static_cast<char>((v >> 8) & 0xFF);
  s += static_cast<char>(v & 0xFF);
}
void u32(std::string& s, uint32_t v) {
  u16(s, v >> 16);
  u16(s, v & 0xFFFF);
}
void i16(std::string& s, int v) { u16(s, static_cast<uint16_t>(static_cast<int16_t>(v))); }

uint32_t checksum(const std::string& t) {
  uint32_t sum = 0;
  for (size_t i = 0; i < t.size(); i += 4) {
    uint32_t w = 0;
    for (size_t k = 0; k < 4; ++k) w = (w << 8) | (i + k < t.size() ? static_cast<uint8_t>(t[i + k]) : 0u);
    sum += w;
  }
  return sum;
}

constexpr int kAdvance = 500, kAscent = 800, kDescent = -200;  // font units of a 1000-unit em

std::string glyphless_ttf() {
  std::map<std::string, std::string> t;  // by tag, the order the table directory wants
  std::string& head = t["head"];
  u32(head, 0x00010000);  // version
  u32(head, 0x00010000);  // fontRevision
  u32(head, 0);           // checkSumAdjustment, set last
  u32(head, 0x5F0F3CF5);  // magicNumber
  u16(head, 0x000B);      // flags
  u16(head, 1000);        // unitsPerEm
  for (int i = 0; i < 4; ++i) u32(head, 0);  // created, modified
  i16(head, 0);
  i16(head, kDescent);
  i16(head, kAdvance);
  i16(head, kAscent);  // the bounding box
  u16(head, 0);        // macStyle
  u16(head, 3);        // lowestRecPPEM
  i16(head, 2);        // fontDirectionHint
  i16(head, 0);        // indexToLocFormat: short offsets
  i16(head, 0);        // glyphDataFormat

  std::string& hhea = t["hhea"];
  u32(hhea, 0x00010000);
  i16(hhea, kAscent);
  i16(hhea, kDescent);
  i16(hhea, 0);         // lineGap
  u16(hhea, kAdvance);  // advanceWidthMax
  i16(hhea, 0);         // minLeftSideBearing
  i16(hhea, 0);         // minRightSideBearing
  i16(hhea, 0);         // xMaxExtent
  i16(hhea, 1);         // caretSlopeRise
  i16(hhea, 0);         // caretSlopeRun
  i16(hhea, 0);         // caretOffset
  for (int i = 0; i < 4; ++i) i16(hhea, 0);
  i16(hhea, 0);  // metricDataFormat
  u16(hhea, 2);  // numberOfHMetrics

  std::string& maxp = t["maxp"];
  u32(maxp, 0x00010000);
  u16(maxp, 2);  // numGlyphs
  for (int i = 0; i < 13; ++i) u16(maxp, i == 4 ? 2 : 0);  // maxZones 2, the rest none

  std::string& hmtx = t["hmtx"];
  for (int g = 0; g < 2; ++g) {
    u16(hmtx, kAdvance);
    i16(hmtx, 0);
  }

  std::string& loca = t["loca"];
  for (int g = 0; g < 3; ++g) u16(loca, 0);  // two glyphs, no outlines
  t["glyf"];

  std::string& cmap = t["cmap"];
  u16(cmap, 0);  // version
  u16(cmap, 1);  // one subtable: Windows, Unicode BMP, format 4, mapping nothing
  u16(cmap, 3);
  u16(cmap, 1);
  u32(cmap, 12);
  u16(cmap, 4);
  u16(cmap, 24);
  u16(cmap, 0);
  u16(cmap, 2);  // segCountX2
  u16(cmap, 2);  // searchRange
  u16(cmap, 0);  // entrySelector
  u16(cmap, 0);  // rangeShift
  u16(cmap, 0xFFFF);
  u16(cmap, 0);
  u16(cmap, 0xFFFF);
  i16(cmap, 1);
  u16(cmap, 0);

  std::string& name = t["name"];
  const std::pair<int, std::string> names[] = {
      {1, "GlyphLessFont"}, {2, "Regular"}, {4, "GlyphLessFont"}, {6, "GlyphLessFont"}};
  std::string strings;
  u16(name, 0);
  u16(name, 4);
  u16(name, 6 + 12 * 4);
  for (const auto& [id, text] : names) {
    u16(name, 3);
    u16(name, 1);
    u16(name, 0x0409);
    u16(name, id);
    u16(name, static_cast<uint32_t>(2 * text.size()));
    u16(name, static_cast<uint32_t>(strings.size()));
    for (char c : text) u16(strings, static_cast<unsigned char>(c));  // UTF-16BE
  }
  name += strings;

  std::string& post = t["post"];
  u32(post, 0x00030000);  // format 3: no glyph names
  u32(post, 0);           // italicAngle
  i16(post, -100);        // underlinePosition
  i16(post, 50);          // underlineThickness
  u32(post, 0);           // isFixedPitch
  for (int i = 0; i < 4; ++i) u32(post, 0);

  // the table directory, then the tables, each on four bytes
  const uint32_t n = static_cast<uint32_t>(t.size());
  uint32_t power = 1, log2 = 0;
  while (power * 2 <= n) {
    power *= 2;
    ++log2;
  }
  std::string font;
  u32(font, 0x00010000);
  u16(font, n);
  u16(font, power * 16);
  u16(font, log2);
  u16(font, n * 16 - power * 16);
  uint32_t offset = 12 + 16 * n, head_at = 0;
  std::string body;
  for (const auto& [tag, table] : t) {
    font += tag;
    u32(font, checksum(table));
    u32(font, offset);
    u32(font, static_cast<uint32_t>(table.size()));
    if (tag == "head") head_at = offset;
    std::string padded = table;
    padded.resize((table.size() + 3) / 4 * 4, '\0');
    body += padded;
    offset += static_cast<uint32_t>(padded.size());
  }
  font += body;
  const uint32_t adjust = 0xB1B0AFBA - checksum(font);
  for (int k = 0; k < 4; ++k) font[head_at + 8 + k] = static_cast<char>((adjust >> (24 - 8 * k)) & 0xFF);
  return font;
}

// ---------------------------------------------------------------- the PDF

std::string deflated(const void* data, size_t n) {
  libdeflate_compressor* c = libdeflate_alloc_compressor(6);
  if (!c) throw Error("to_pdf: no memory for the compressor");
  std::string out(libdeflate_zlib_compress_bound(c, n), '\0');
  const size_t got = libdeflate_zlib_compress(c, data, n, out.data(), out.size());
  libdeflate_free_compressor(c);
  if (!got) throw Error("to_pdf: deflate failed");
  out.resize(got);
  return out;
}

std::string num(double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.4f", v);
  std::string s = buf;
  s.erase(s.find_last_not_of('0') + 1);
  if (s.back() == '.') s.pop_back();
  return s == "-0" ? "0" : s;
}

std::string stream(const std::string& dict_rest, const std::string& data) {
  return "<< " + dict_rest + " /Length " + std::to_string(data.size()) + " >>\nstream\n" + data + "\nendstream";
}

// A word's text as hex UTF-16 code units; how many.
std::pair<std::string, int> utf16_hex(const std::string& utf8) {
  std::string hex;
  int units = 0;
  auto unit = [&](uint32_t u) {
    char b[5];
    std::snprintf(b, sizeof b, "%04X", u);
    hex += b;
    ++units;
  };
  for (size_t i = 0; i < utf8.size();) {
    const unsigned char c = static_cast<unsigned char>(utf8[i]);
    uint32_t cp = c;
    int more = 0;
    if (c >= 0xF0) {
      cp = c & 0x07;
      more = 3;
    } else if (c >= 0xE0) {
      cp = c & 0x0F;
      more = 2;
    } else if (c >= 0xC0) {
      cp = c & 0x1F;
      more = 1;
    }
    ++i;
    for (int k = 0; k < more && i < utf8.size(); ++k, ++i) {
      cp = (cp << 6) | (static_cast<unsigned char>(utf8[i]) & 0x3F);
    }
    if (cp >= 0x10000) {
      cp -= 0x10000;
      unit(0xD800 + (cp >> 10));
      unit(0xDC00 + (cp & 0x3FF));
    } else {
      unit(cp);
    }
  }
  return {hex, units};
}

}  // namespace

std::string to_pdf(std::span<const OcrPage> pages, std::span<const GrayView> images, double dpi, bool text_only) {
  if (pages.size() != images.size()) throw Error("to_pdf: as many images as pages");
  if (!(dpi > 0)) throw Error("to_pdf: dpi must be positive");
  const double k = 72.0 / dpi;  // points a pixel
  // 1 catalog, 2 pages, 3 the font, 4 its CID font, 5 descriptor, 6 font file, 7 CID to glyph, 8 to Unicode
  std::vector<std::string> obj(7);
  obj[0] = "<< /Type /Catalog /Pages 2 0 R >>";
  obj[2] = "<< /Type /Font /Subtype /Type0 /BaseFont /GlyphLessFont /Encoding /Identity-H "
           "/DescendantFonts [4 0 R] /ToUnicode 8 0 R >>";
  obj[3] = "<< /Type /Font /Subtype /CIDFontType2 /BaseFont /GlyphLessFont "
           "/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
           "/FontDescriptor 5 0 R /DW " + std::to_string(kAdvance) + " /CIDToGIDMap 7 0 R >>";
  obj[4] = "<< /Type /FontDescriptor /FontName /GlyphLessFont /Flags 5 /FontBBox [0 " + std::to_string(kDescent) +
           " " + std::to_string(kAdvance) + " " + std::to_string(kAscent) + "] /ItalicAngle 0 /Ascent " +
           std::to_string(kAscent) + " /Descent " + std::to_string(kDescent) + " /CapHeight " +
           std::to_string(kAscent) + " /StemV 80 /FontFile2 6 0 R >>";
  const std::string ttf = glyphless_ttf();
  obj[5] = stream("/Filter /FlateDecode /Length1 " + std::to_string(ttf.size()), deflated(ttf.data(), ttf.size()));
  std::string gids;  // every code: glyph 1
  for (int i = 0; i < 65536; ++i) u16(gids, 1);
  obj[6] = stream("/Filter /FlateDecode", deflated(gids.data(), gids.size()));
  const std::string cmap =
      "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
      "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
      "/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
      "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
      "1 beginbfrange\n<0000> <FFFF> <0000>\nendbfrange\n"
      "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n";
  obj.push_back(stream("", cmap));  // 8

  std::string kids;
  for (size_t p = 0; p < pages.size(); ++p) {
    const OcrPage& page = pages[p];
    GrayImage turned;
    GrayView img = images[p];
    if (page.rotation) {  // the boxes are the upright page's
      turned = rotate_quarter(img, 4 - page.rotation / 90);
      img = turned.view();
    }
    const double w = img.width * k, h = img.height * k;
    const size_t image_obj = text_only ? 0 : obj.size() + 1;
    const size_t content_obj = text_only ? obj.size() + 1 : image_obj + 1, page_obj = content_obj + 1;
    std::string c = "BT\n3 Tr\n";
    if (!text_only) {
      std::string pixels(static_cast<size_t>(img.width) * img.height, '\0');
      for (int y = 0; y < img.height; ++y) {
        std::copy_n(img.data + y * img.stride, img.width, pixels.begin() + static_cast<int64_t>(y) * img.width);
      }
      obj.push_back(stream("/Type /XObject /Subtype /Image /Width " + std::to_string(img.width) + " /Height " +
                               std::to_string(img.height) +
                               " /ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode",
                           deflated(pixels.data(), pixels.size())));
      c = "q\n" + num(w) + " 0 0 " + num(h) + " 0 0 cm\n/Im0 Do\nQ\n" + c;
    }
    for (const OcrLine& l : page.lines) {
      const std::vector<OcrWord> words = l.words.empty() && !l.text.empty()
                                             ? std::vector<OcrWord>{{l.box, l.text, l.confidence}}
                                             : l.words;
      for (const OcrWord& word : words) {
        const auto [hex, units] = utf16_hex(word.text);
        const double size = (word.box.y1 - word.box.y0) * k, width = (word.box.x1 - word.box.x0) * k;
        if (!units || size <= 0 || width <= 0) continue;
        const double stretch = 100.0 * width / (units * size * kAdvance / 1000.0);
        const double x = word.box.x0 * k, base = h - word.box.y1 * k - size * kDescent / 1000.0;
        c += "/F0 " + num(size) + " Tf " + num(stretch) + " Tz 1 0 0 1 " + num(x) + " " + num(base) + " Tm <" + hex +
             "> Tj\n";
      }
    }
    c += "ET\n";
    obj.push_back(stream("/Filter /FlateDecode", deflated(c.data(), c.size())));
    const std::string xobject = text_only ? "" : "/XObject << /Im0 " + std::to_string(image_obj) + " 0 R >> ";
    obj.push_back("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 " + num(w) + " " + num(h) + "] /Resources << " +
                  xobject + "/Font << /F0 3 0 R >> >> /Contents " + std::to_string(content_obj) + " 0 R >>");
    kids += (kids.empty() ? "" : " ") + std::to_string(page_obj) + " 0 R";
  }
  obj[1] = "<< /Type /Pages /Kids [" + kids + "] /Count " + std::to_string(pages.size()) + " >>";

  std::string out = "%PDF-1.5\n%\xE2\xE3\xCF\xD3\n";
  std::vector<size_t> at;
  for (size_t i = 0; i < obj.size(); ++i) {
    at.push_back(out.size());
    out += std::to_string(i + 1) + " 0 obj\n" + obj[i] + "\nendobj\n";
  }
  const size_t xref = out.size();
  out += "xref\n0 " + std::to_string(obj.size() + 1) + "\n0000000000 65535 f \n";
  char line[24];
  for (size_t o : at) {
    std::snprintf(line, sizeof line, "%010zu 00000 n \n", o);
    out += line;
  }
  out += "trailer\n<< /Size " + std::to_string(obj.size() + 1) + " /Root 1 0 R >>\nstartxref\n" +
         std::to_string(xref) + "\n%%EOF\n";
  return out;
}

}  // namespace carbon
