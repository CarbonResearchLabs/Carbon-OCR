// PDF documents: their pages as images and their text layers, read
// through PDFium (a prebuilt build, pinned; CARBON_WITH_PDFIUM).
//
// PDFium lives in the helper processes only: PdfPool (in carbon) asks carbon-pdf
// helpers for pages and text layers, each helper with its own PDFium, many pages
// at once, and a PDF that crashes PDFium takes down a helper, not the caller.
// PdfDocument, PDFium in the process, is in the library carbon_pdf
// (carbon::pdf), which the helper and the tests link: no process of the engine
// loads PDFium.  PDFium is not
// thread-safe: a PdfDocument's calls hold one lock.
#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "carbon/image.h"
#include "carbon/lines.h"

namespace carbon {

// A page's size as it is shown (its /Rotate applied), in points (1/72 inch).
struct PdfPageSize {
  double width = 0, height = 0;
};

// A character of a page's text layer, as PDFium reads it.
struct PdfChar {
  char32_t code = 0;  // Unicode; 0 where the font maps the glyph to none
  // Its box in the page's pixels at the scale asked for, the page as shown,
  // the origin at the top left: [x0, x1) x [y0, y1).
  float x0 = 0, y0 = 0, x1 = 0, y1 = 0;
  // Its loose box: the font's whole height, whatever the glyph (the box a line is
  // made of), in the same pixels.
  float lx0 = 0, ly0 = 0, lx1 = 0, ly1 = 0;
  float size = 0;          // the font size, points
  bool generated = false;  // made by PDFium (a space or a line break it infers), not drawn
  bool bad_map = false;    // the font's Unicode map for it is broken
  bool invisible = false;  // drawn neither filled nor stroked (text render mode 3): an OCR layer
  int mcid = -1;           // its marked content's ID (a tagged PDF's), -1 for none; a generated
                           // space or break takes the character's before it
};

// A tagged page's structure tree, as PDFium reads it for the page (a document whose
// MarkInfo says Marked, with a ParentTree): its elements in the tree's order, and the
// marked content they hold, in reading order.  Empty for an untagged page.
struct PdfStructElement {
  std::string type;  // its role, through the role map: P, H1, L, LI, Table, TR, TD, Figure, ...
  int parent = -1;   // an index into elements; -1 at the top
};

struct PdfStructure {
  std::vector<PdfStructElement> elements;
  std::vector<std::pair<int, int>> runs;  // (element, MCID): the page's marked content in reading order
};

class PdfDocument {
 public:
  // Throws Error if the file cannot be read as a PDF (or the password is wrong).
  explicit PdfDocument(const std::filesystem::path& file, const std::string& password = "");
  ~PdfDocument();
  PdfDocument(PdfDocument&&) noexcept;
  PdfDocument& operator=(PdfDocument&&) noexcept;

  int pages() const;
  PdfPageSize page_size(int page) const;

  // The page as shown, in gray (paper 255), at `scale` pixels a point (dpi / 72):
  // white paper under it, its annotations drawn, as printed.
  GrayImage render(int page, double scale) const;

  // Its text layer: every character, in the order PDFium reads them, boxes at `scale`.
  std::vector<PdfChar> chars(int page, double scale) const;

  // Its pictures: the boxes of its image objects at `scale`, on the page as shown
  // (an image inside a form XObject is not seen; nor is a picture drawn as paths).
  std::vector<LineBox> pictures(int page, double scale) const;
  // Its structure tree (a tagged PDF's).
  PdfStructure structure(int page) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// PDF pages rendered in helper processes: carbon-pdf (built beside carbon-ocr),
// each answering one request at a time on its standard input and output, the
// pixels in a shared memory region of the helper's (shared_bytes; a page larger
// comes back down the pipe, as every page does with shared_bytes 0).  Thread-safe: each call takes a free helper
// (one with the file open already, if it can) or waits for one.  A helper that
// dies is started again and the page it was on throws Error.
class PdfPool {
 public:
  // processes: 0 for one per physical core.  helper: the carbon-pdf program;
  // empty for the one beside the running program.  Throws Error if it is missing.
  // shared_bytes: each helper's region (48 MiB holds a Letter or A4 page at 600 dpi).
  static constexpr size_t kSharedBytes = size_t{48} << 20;
  explicit PdfPool(int processes = 0, std::filesystem::path helper = {}, size_t shared_bytes = kSharedBytes);
  ~PdfPool();
  PdfPool(const PdfPool&) = delete;
  PdfPool& operator=(const PdfPool&) = delete;

  int processes() const;
  int pages(const std::filesystem::path& file);
  // A negative page makes the helper exit at once, as a crash would (tests).
  GrayImage render(const std::filesystem::path& file, int page, double scale);
  // The page's text layer, as PdfDocument::chars reads it.
  std::vector<PdfChar> chars(const std::filesystem::path& file, int page, double scale);
  // Its pictures, as PdfDocument::pictures finds them.
  std::vector<LineBox> pictures(const std::filesystem::path& file, int page, double scale);
  // Its structure tree, as PdfDocument::structure reads it.
  PdfStructure structure(const std::filesystem::path& file, int page);

  struct Helper;  // a process and its pipes

 private:
  template <typename F>
  auto with_helper(const std::filesystem::path& file, F&& f);
  Helper& acquire(const std::string& file);
  void release(Helper& h);
  std::filesystem::path helper_;
  std::vector<std::unique_ptr<Helper>> helpers_;
  std::mutex m_;
  struct Wait;
  std::unique_ptr<Wait> wait_;
};

// The program the running process was started from.
std::filesystem::path running_program();

// A helper's loop (carbon-pdf's main; in carbon_pdf): requests on its standard
// input, answers on its standard output, until the input ends.  Returns the exit
// code.
int run_pdf_helper();

// Whether this build reads PDFs (CARBON_WITH_PDFIUM).
constexpr bool pdf_supported() {
#if defined(CARBON_WITH_PDFIUM)
  return true;
#else
  return false;
#endif
}

}  // namespace carbon
