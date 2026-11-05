// PDF documents through PDFium (carbon/pdf.h): one lock for every call, the
// library set up once.

#include "carbon/pdf.h"

// fpdfview.h takes in <windows.h> on Windows: without its min and max macros
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif
#include <fpdf_edit.h>
#include <fpdf_structtree.h>
#include <fpdf_text.h>
#include <fpdfview.h>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <vector>

#include "carbon/cpu.h"
#include "carbon/error.h"

namespace carbon {
namespace {

std::mutex& lock() {
  static std::mutex m;
  return m;
}

// FPDF_InitLibraryWithConfig once, under the lock; PDFium stays up for the process.
void init() {
  static bool done = false;
  if (done) return;
  FPDF_LIBRARY_CONFIG config{};
  config.version = 2;
  FPDF_InitLibraryWithConfig(&config);
  done = true;
}

std::string load_error() {
  switch (FPDF_GetLastError()) {
    case FPDF_ERR_FILE: return "cannot be opened";
    case FPDF_ERR_FORMAT: return "is not a PDF, or is damaged";
    case FPDF_ERR_PASSWORD: return "needs another password";
    case FPDF_ERR_SECURITY: return "uses an unsupported security scheme";
    case FPDF_ERR_PAGE: return "has a page that cannot be read";
    default: return "cannot be read";
  }
}

struct Page {
  FPDF_PAGE p = nullptr;
  Page(FPDF_DOCUMENT doc, int i) : p(FPDF_LoadPage(doc, i)) {}
  ~Page() {
    if (p) FPDF_ClosePage(p);
  }
  Page(const Page&) = delete;
  Page& operator=(const Page&) = delete;
};

}  // namespace

struct PdfDocument::Impl {
  FPDF_DOCUMENT doc = nullptr;
  std::string path;

  void check_page(int i) const {
    const int n = FPDF_GetPageCount(doc);
    if (i < 0 || i >= n) throw Error(path + ": no page " + std::to_string(i) + " (it has " + std::to_string(n) + ")");
  }
};

PdfDocument::PdfDocument(const std::filesystem::path& file, const std::string& password) : impl_(new Impl) {
  std::lock_guard<std::mutex> g(lock());
  init();
  impl_->path = file.string();
  // FPDF_LoadDocument takes UTF-8 on every platform.
  const std::u8string u8 = file.u8string();
  const std::string name(u8.begin(), u8.end());
  impl_->doc = FPDF_LoadDocument(name.c_str(), password.empty() ? nullptr : password.c_str());
  if (!impl_->doc) throw Error(impl_->path + " " + load_error());
}

PdfDocument::~PdfDocument() {
  if (!impl_ || !impl_->doc) return;
  std::lock_guard<std::mutex> g(lock());
  FPDF_CloseDocument(impl_->doc);
}

PdfDocument::PdfDocument(PdfDocument&&) noexcept = default;
PdfDocument& PdfDocument::operator=(PdfDocument&&) noexcept = default;

int PdfDocument::pages() const {
  std::lock_guard<std::mutex> g(lock());
  return FPDF_GetPageCount(impl_->doc);
}

PdfPageSize PdfDocument::page_size(int page) const {
  std::lock_guard<std::mutex> g(lock());
  impl_->check_page(page);
  FS_SIZEF size{};
  if (!FPDF_GetPageSizeByIndexF(impl_->doc, page, &size)) {
    throw Error(impl_->path + ": page " + std::to_string(page) + " has no size");
  }
  return {size.width, size.height};
}

namespace {

struct PageRef {  // a loaded page, as Page names it
  FPDF_PAGE p;
};

// The page's gray at scale, as render gives it: into `into` when it fits into_bytes (out.pixels then empty).
void render_page(FPDF_PAGE pg, const std::string& path, int page, double scale, GrayImage& out, uint8_t* into,
                 size_t into_bytes) {
  const PageRef p{pg};
  const int w = std::max(1, static_cast<int>(std::lround(FPDF_GetPageWidthF(p.p) * scale)));
  const int h = std::max(1, static_cast<int>(std::lround(FPDF_GetPageHeightF(p.p) * scale)));
  if (static_cast<int64_t>(w) * h > (int64_t{1} << 30)) {
    throw Error(path + ": page " + std::to_string(page) + " at this scale is " + std::to_string(w) + " x " +
                std::to_string(h) + " px");
  }
  FPDF_BITMAP bmp = FPDFBitmap_Create(w, h, 0);  // BGRx
  if (!bmp) throw Error("PDFium: no memory for a " + std::to_string(w) + " x " + std::to_string(h) + " page");
  FPDFBitmap_FillRect(bmp, 0, 0, w, h, 0xFFFFFFFF);
  FPDF_RenderPageBitmap(bmp, p.p, 0, 0, w, h, 0, FPDF_ANNOT | FPDF_PRINTING);
  out.width = w;
  out.height = h;
  const size_t bytes = static_cast<size_t>(w) * h;
  uint8_t* gray = into && bytes <= into_bytes ? into : nullptr;
  if (!gray) {
    out.pixels.resize(bytes);
    gray = out.pixels.data();
  }
  // BT.601 luma, rounded: 0.299 R + 0.587 G + 0.114 B, on the CPU's vectors, the same bytes
  cpu::bgrx_to_gray(static_cast<const uint8_t*>(FPDFBitmap_GetBuffer(bmp)), FPDFBitmap_GetStride(bmp), w, h, gray, w);
  FPDFBitmap_Destroy(bmp);
}

std::vector<LineBox> page_pictures(FPDF_PAGE pg, double scale) {
  const PageRef p{pg};
  constexpr int kFine = 64;  // as in chars(): FPDF_PageToDevice rounds to whole pixels
  const int dw = std::max(1, static_cast<int>(std::lround(FPDF_GetPageWidthF(p.p) * scale * kFine)));
  const int dh = std::max(1, static_cast<int>(std::lround(FPDF_GetPageHeightF(p.p) * scale * kFine)));
  std::vector<LineBox> out;
  const int n = FPDFPage_CountObjects(p.p);
  for (int i = 0; i < n; ++i) {
    FPDF_PAGEOBJECT obj = FPDFPage_GetObject(p.p, i);
    if (!obj || FPDFPageObj_GetType(obj) != FPDF_PAGEOBJ_IMAGE) continue;
    float l = 0, b = 0, r = 0, t = 0;
    if (!FPDFPageObj_GetBounds(obj, &l, &b, &r, &t)) continue;
    int x0, y0, x1, y1;
    FPDF_PageToDevice(p.p, 0, 0, dw, dh, 0, l, t, &x0, &y0);
    FPDF_PageToDevice(p.p, 0, 0, dw, dh, 0, r, b, &x1, &y1);
    LineBox box{std::min(x0, x1) / static_cast<double>(kFine), std::min(y0, y1) / static_cast<double>(kFine),
                std::max(x0, x1) / static_cast<double>(kFine), std::max(y0, y1) / static_cast<double>(kFine)};
    if (box.x1 > box.x0 && box.y1 > box.y0) out.push_back(box);
  }
  return out;
}

std::vector<PdfChar> page_chars(FPDF_PAGE pg, double scale) {
  const PageRef p{pg};
  FPDF_TEXTPAGE text = FPDFText_LoadPage(p.p);
  if (!text) return {};
  // The page's own space (unrotated, its box's origin anywhere) to the shown page's
  // pixels at `scale`: FPDF_PageToDevice, on a device 64 times as large, since it
  // rounds to whole pixels.
  constexpr int kFine = 64;
  const double fw = FPDF_GetPageWidthF(p.p) * scale * kFine, fh = FPDF_GetPageHeightF(p.p) * scale * kFine;
  const int dw = std::max(1, static_cast<int>(std::lround(fw))), dh = std::max(1, static_cast<int>(std::lround(fh)));
  auto device = [&](double px, double py, float& dx, float& dy) {
    int ix = 0, iy = 0;
    FPDF_PageToDevice(p.p, 0, 0, dw, dh, 0, px, py, &ix, &iy);
    dx = static_cast<float>(ix) / kFine;
    dy = static_cast<float>(iy) / kFine;
  };
  std::vector<PdfChar> out;
  const int n = FPDFText_CountChars(text);
  out.reserve(std::max(0, n));
  int mcid = -1;
  for (int i = 0; i < n; ++i) {
    PdfChar c;
    c.code = static_cast<char32_t>(FPDFText_GetUnicode(text, i));
    c.generated = FPDFText_IsGenerated(text, i) == 1;
    c.bad_map = FPDFText_HasUnicodeMapError(text, i) == 1;
    c.size = static_cast<float>(FPDFText_GetFontSize(text, i));
    if (FPDF_PAGEOBJECT obj = FPDFText_GetTextObject(text, i)) {
      c.invisible = FPDFTextObj_GetTextRenderMode(obj) == FPDF_TEXTRENDERMODE_INVISIBLE;
      mcid = FPDFPageObj_GetMarkedContentID(obj);
    } else if (!c.generated) {
      mcid = -1;
    }
    c.mcid = mcid;  // a generated space or break: the character's before it
    FS_RECTF loose{};
    if (FPDFText_GetLooseCharBox(text, i, &loose)) {
      float x0, y0, x1, y1;
      device(loose.left, loose.top, x0, y0);
      device(loose.right, loose.bottom, x1, y1);
      c.lx0 = std::min(x0, x1);
      c.lx1 = std::max(x0, x1);
      c.ly0 = std::min(y0, y1);
      c.ly1 = std::max(y0, y1);
    }
    double l = 0, r = 0, b = 0, t = 0;
    if (FPDFText_GetCharBox(text, i, &l, &r, &b, &t)) {
      float x0, y0, x1, y1;  // two opposite corners: any rotation keeps them opposite
      device(l, t, x0, y0);
      device(r, b, x1, y1);
      c.x0 = std::min(x0, x1);
      c.x1 = std::max(x0, x1);
      c.y0 = std::min(y0, y1);
      c.y1 = std::max(y0, y1);
    }
    out.push_back(c);
  }
  FPDFText_ClosePage(text);
  return out;
}

// A structure element's type (UTF-16LE from PDFium), as UTF-8.
std::string element_type(FPDF_STRUCTELEMENT e) {
  const unsigned long bytes = FPDF_StructElement_GetType(e, nullptr, 0);
  if (bytes < 2) return {};
  std::vector<uint16_t> w(bytes / 2);
  FPDF_StructElement_GetType(e, w.data(), bytes);
  std::string out;
  for (const uint16_t u : w) {
    if (u == 0) break;
    if (u < 0x80) {
      out += static_cast<char>(u);
    } else if (u < 0x800) {
      out += static_cast<char>(0xC0 | (u >> 6));
      out += static_cast<char>(0x80 | (u & 0x3F));
    } else {
      out += static_cast<char>(0xE0 | (u >> 12));
      out += static_cast<char>(0x80 | ((u >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (u & 0x3F));
    }
  }
  return out;
}

PdfStructure page_structure(FPDF_PAGE pg) {
  const PageRef p{pg};
  PdfStructure out;
  FPDF_STRUCTTREE tree = FPDF_StructTree_GetForPage(p.p);
  if (!tree) return out;  // untagged, or its MarkInfo does not say Marked
  constexpr size_t kMost = 200000;  // elements: a damaged tree stops here
  // Pre-order, each element's kids in their order, its marked content and its child elements
  // interleaved as they come: the reading order. A stack of work keeps that order: an element
  // is expanded when popped, its kids pushed in reverse, a marked-content kid as an item of its
  // own. A kid that is neither (another page's content, an object) is passed over.
  struct Item {
    FPDF_STRUCTELEMENT e;
    int parent;
    int mcid;  // >= 0: a marked-content kid of `parent`, not an element
  };
  std::vector<Item> work;
  for (int k = FPDF_StructTree_CountChildren(tree) - 1; k >= 0; --k) {
    if (FPDF_STRUCTELEMENT e = FPDF_StructTree_GetChildAtIndex(tree, k)) work.push_back({e, -1, -1});
  }
  while (!work.empty() && out.elements.size() < kMost) {
    const Item it = work.back();
    work.pop_back();
    if (it.mcid >= 0) {
      out.runs.emplace_back(it.parent, it.mcid);
      continue;
    }
    const int me = static_cast<int>(out.elements.size());
    out.elements.push_back({element_type(it.e), it.parent});
    const int n = FPDF_StructElement_CountChildren(it.e);
    if (n <= 0) {
      const int m = FPDF_StructElement_GetMarkedContentID(it.e);
      if (m >= 0) out.runs.emplace_back(me, m);
      continue;
    }
    for (int j = n - 1; j >= 0; --j) {
      if (FPDF_STRUCTELEMENT c = FPDF_StructElement_GetChildAtIndex(it.e, j)) {
        work.push_back({c, me, -1});
      } else {
        const int m = FPDF_StructElement_GetChildMarkedContentID(it.e, j);
        if (m >= 0) work.push_back({nullptr, me, m});
      }
    }
  }
  FPDF_StructTree_Close(tree);
  return out;
}

}  // namespace

namespace {

// The scale for a loaded page (PdfDocument::auto_scale).
double page_auto_scale(FPDF_PAGE pg) {
  constexpr double kLow = 200.0 / 72, kHigh = 400.0 / 72, kPx = 25;  // the small print's font size in pixels
  std::vector<float> sizes;
  for (const PdfChar& c : page_chars(pg, 1.0)) {
    if (!c.generated && c.size > 0 && c.code != U' ') sizes.push_back(c.size);
  }
  if (sizes.size() >= 20) {
    const auto at = sizes.begin() + static_cast<std::ptrdiff_t>(sizes.size() / 10);
    std::nth_element(sizes.begin(), at, sizes.end());
    return std::clamp(kPx / static_cast<double>(*at), kLow, kHigh);
  }
  double best = 0, area = 0;  // no text to speak of: the largest image's own resolution
  const int n = FPDFPage_CountObjects(pg);
  for (int i = 0; i < n; ++i) {
    FPDF_PAGEOBJECT obj = FPDFPage_GetObject(pg, i);
    if (!obj || FPDFPageObj_GetType(obj) != FPDF_PAGEOBJ_IMAGE) continue;
    FPDF_IMAGEOBJ_METADATA m{};
    float l = 0, b = 0, r = 0, t = 0;
    if (!FPDFImageObj_GetImageMetadata(obj, pg, &m) || !FPDFPageObj_GetBounds(obj, &l, &b, &r, &t)) continue;
    const double a = static_cast<double>(r - l) * (t - b);
    if (a > area && m.horizontal_dpi > 0) {
      area = a;
      best = m.horizontal_dpi / 72.0;
    }
  }
  return best > 0 ? std::clamp(best, kLow, kHigh) : kLow;
}

}  // namespace

double PdfDocument::auto_scale(int page) const {
  std::lock_guard<std::mutex> g(lock());
  impl_->check_page(page);
  Page p(impl_->doc, page);
  if (!p.p) throw Error(impl_->path + ": page " + std::to_string(page) + " cannot be read");
  return page_auto_scale(p.p);
}

PdfPage PdfDocument::page(int page, double scale, unsigned want, uint8_t* into, size_t into_bytes) const {
  std::lock_guard<std::mutex> g(lock());
  impl_->check_page(page);
  Page p(impl_->doc, page);
  if (!p.p) throw Error(impl_->path + ": page " + std::to_string(page) + " cannot be read");
  PdfPage out;
  if (!(scale > 0)) scale = page_auto_scale(p.p);
  out.scale = scale;
  if (want & kPdfImage) render_page(p.p, impl_->path, page, scale, out.image, into, into_bytes);
  if (want & kPdfChars) out.chars = page_chars(p.p, scale);
  if (want & kPdfPictures) out.pictures = page_pictures(p.p, scale);
  if (want & kPdfStructure) out.structure = page_structure(p.p);
  return out;
}

GrayImage PdfDocument::render(int page, double scale) const { return this->page(page, scale, kPdfImage).image; }
std::vector<PdfChar> PdfDocument::chars(int page, double scale) const {
  return this->page(page, scale, kPdfChars).chars;
}
std::vector<LineBox> PdfDocument::pictures(int page, double scale) const {
  return this->page(page, scale, kPdfPictures).pictures;
}
PdfStructure PdfDocument::structure(int page) const { return this->page(page, 1.0, kPdfStructure).structure; }

}  // namespace carbon
