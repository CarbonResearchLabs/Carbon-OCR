// The pages the engine reads (carbon/engine.h), in the formats other tools take
//.  Boxes are the page image's pixels; `images` names each page's
// image (a file, or "file.pdf, page 3"), one for each page.
//
//   hocr      hOCR 1.2 (XHTML): Tesseract's format, which OCRmyPDF and search
//             engines read; ocr_page, ocr_carea and ocr_par a block, ocr_line
//             (ocr_header for a heading's, ocr_caption for a caption's),
//             ocrx_word with x_wconf
//   alto      ALTO XML 4.4: Page, PrintSpace, TextBlock (its label a LayoutTag),
//             TextLine, String (WC, the word's confidence) and SP
//   page_xml  PAGE XML (2019-07-15), one document a page: TextRegion (its type
//             from the label, the label in custom), TextLine, Word, each with
//             its Coords and TextEquiv
//   pdf       a searchable PDF: the page images under their words, invisible
//   markdown  headings (##), paragraphs, lists (- items, a line without a
//             marker carrying on the item before), tables as HTML from their
//             cells, captions in italics; running headers and footers left out;
//             pages apart by a rule.  Without blocks, a paragraph where the gap
//             between lines is wider than a line.
//   html      an HTML5 document, a <section class="page"> a page: headings <h2>;
//             paragraphs <p>, a block's lines joined and a word a hyphen broke at
//             a line's end joined again; lists <ul> and <ol start> of <li> (their
//             bullets and numbers dropped from the text, a browser draws them);
//             tables <table> of <tr> and <td> from their cells; captions
//             <figure><figcaption>; running headers and footers <header> and
//             <footer>, kept; footnotes <aside class="footnote">; formulas
//             <p class="formula">; forms <div class="form">; a contents page
//             <nav><ul>; a picture's text in a <figure>.  Without blocks, a
//             paragraph where the gap between lines is wider than a line.
//
// The blocks are the order model's (OcrPage::blocks); a page without them is one
// block of all its lines.
#pragma once

#include <span>
#include <string>

#include "carbon/engine.h"
#include "carbon/lines.h"

namespace carbon {

std::string to_hocr(std::span<const OcrPage> pages, std::span<const std::string> images);
std::string to_alto(std::span<const OcrPage> pages, std::span<const std::string> images);
std::string to_page_xml(const OcrPage& page, const std::string& image);
std::string to_markdown(std::span<const OcrPage> pages);
std::string to_html(std::span<const OcrPage> pages);
// One page as to_html writes it inside its <section>: its blocks, no document.
std::string to_html_body(const OcrPage& page);

// A searchable PDF: each page its image (8-bit gray, deflated) at `dpi`, and over
// it its words as invisible text (render mode 3), each stretched to its box, in a
// font with no glyphs (made here, every code its one empty glyph, a ToUnicode map
// giving the text back), so a viewer selects and searches the text and shows the
// image.  images[i] is page i's image, the one its boxes are in.  text_only: the
// text alone on each page, no image (what OCRmyPDF's sandwich renderer grafts onto
// the original page).
std::string to_pdf(std::span<const OcrPage> pages, std::span<const GrayView> images, double dpi,
                   bool text_only = false);

}  // namespace carbon
