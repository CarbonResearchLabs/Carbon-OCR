// A born-digital page's text layer: the characters PDFium
// reads (carbon/pdf.h) made into lines, and the checks that say whether to trust
// them or read the page by OCR.  A page whose layer passes skips detection and
// recognition.
//
// The checks, on the characters the page draws (PDFium's own spaces and line
// breaks aside):
//
//   enough     at least min_chars of them: fewer, and the page is pictures or a scan
//   bad        broken Unicode maps, U+FFFD, controls and private-use code points:
//              text that would read as garbage (legacy font codes, symbol fonts)
//   invisible  text drawn neither filled nor stroked: an OCR layer over a scan,
//              read again rather than trusted
//   no ink     characters with no dark pixel under their box on the rendered page:
//              white text, clipped text, glyphs that are not the codes they claim
//
// Lines: a line ends at PDFium's line breaks, and where the next character leaves
// the band of the one before (its height range, or its width range on a page
// turned a quarter, where the text runs down).  A line's box is the union of its
// characters' loose boxes (the font's whole height), its text their codes, a
// space where PDFium puts one.
#pragma once

#include <span>
#include <string>
#include <vector>

#include "carbon/engine.h"
#include "carbon/lines.h"
#include "carbon/pdf.h"

namespace carbon {

struct TextLayerRule {
  int min_chars = 4;
  double max_bad = 0.02;      // of the drawn characters
  double max_invisible = 0.0;
  double max_no_ink = 0.05;
  int ink = 160;              // a pixel this dark or darker is ink
};

struct TextLayerCheck {
  bool usable = false;
  std::string reason;  // why not; empty when usable
  int chars = 0;       // drawn characters
  int bad = 0, invisible = 0, no_ink = 0;
};

struct TextLayer {
  std::vector<OcrLine> lines;  // in the layer's order, boxes in the page's pixels, confidence 1
  TextLayerCheck check;
};

// page: the page rendered at the scale the characters were read at, for the ink
// check (an empty view, data null: no ink check).
TextLayer read_text_layer(std::span<const PdfChar> chars, const GrayView& page, const TextLayerRule& rule = {});

// The page its text layer reads as: its lines split at wide gaps (split_lines: a
// table row's cells, a column gutter), in reading order (the XY-cut the engine puts
// OCR's lines in), source "text layer".
OcrPage page_from_text_layer(const TextLayer& layer, int width, int height);

// Hybrid pages: a page read from its text layer whose pictures hold text of their
// own, which the layer does not (a chart's labels, a scanned table, a screenshot),
// has that text read by OCR.  The page with all but its pictures made paper is
// read whole, so the pictures' text is found at the page's own scale (the detector
// sees every page at one), and the lines found join the layer's.
struct HybridRule {
  int min_side = 32;         // px: a smaller picture (an icon, a bullet) holds no line worth reading
  double max_covered = 0.5;  // a line found this much under the layer's lines is theirs: text drawn over a picture
};

// The pictures to read of a page's (PdfDocument::pictures): clipped to the page,
// at least min_side pixels on both sides.
std::vector<LineBox> pictures_to_read(std::span<const LineBox> pictures, int width, int height,
                                      const HybridRule& rule = {});

// The page with only its pictures left, paper (255) elsewhere.
GrayImage pictures_only(const GrayView& page, std::span<const LineBox> pictures);

// Puts into `page` (read from its text layer) the lines OCR found on its
// pictures_only page, `read`: their boxes and their words' taken back from the
// read page's frame (turned upright and levelled) to the page's, leaving out a
// line with no text or mostly under the layer's own lines; then every line in
// reading order again, and the page's source "hybrid" if any was added.
void add_picture_lines(OcrPage& page, const OcrPage& read, const HybridRule& rule = {});

}  // namespace carbon
