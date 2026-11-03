// A PdfPool helper's loop (carbon/pdf.h, protocol.h): the carbon-pdf program,
// with PDFium in it, one request at a time.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "carbon/error.h"
#include "carbon/pdf.h"
#include "protocol.h"

#if !defined(_WIN32)
#include <signal.h>
#include <sys/mman.h>
#endif

namespace carbon {

int run_pdf_helper() {
  using namespace pdf_wire;
#if defined(_WIN32)
  const Handle in = GetStdHandle(STD_INPUT_HANDLE), out = GetStdHandle(STD_OUTPUT_HANDLE);
#else
  signal(SIGPIPE, SIG_DFL);  // the pool gone: end
  const Handle in = 0, out = 1;
#endif
  auto get = [&](void* p, size_t n) { return read_all(in, p, n); };
  auto put = [&](const void* p, size_t n) { return write_all(out, p, n); };
  std::unique_ptr<PdfDocument> doc;
  std::string open;
  // The region the pool reads rendered pages from (op 8), mapped once.
  uint8_t* shared = nullptr;
  size_t shared_bytes = 0;
  for (;;) {
    uint32_t op = 0, len = 0;
    int32_t page = 0;
    double scale = 0;
    if (!get(&op, 4)) return 0;  // the pool closed its end
    if (!get(&page, 4) || !get(&scale, 8) || !get(&len, 4)) return 1;
    std::string file(len, '\0');
    if (!get(file.data(), len)) return 1;
    if (op == kExit) std::_Exit(3);
    if (op == kShared) {
      const auto bytes = static_cast<size_t>(scale);
      void* p = nullptr;
#if defined(_WIN32)
      const auto mapping = reinterpret_cast<HANDLE>(static_cast<uintptr_t>(static_cast<uint32_t>(page)));
      p = MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, bytes);
#else
      p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, page, 0);
      if (p == MAP_FAILED) p = nullptr;
#endif
      const uint32_t status = p ? 0 : 1;
      if (p) {
        shared = static_cast<uint8_t*>(p);
        shared_bytes = bytes;
        if (!put(&status, 4)) return 1;
      } else {
        const std::string msg = "carbon-pdf: cannot map the shared region";
        const uint32_t n = static_cast<uint32_t>(msg.size());
        if (!put(&status, 4) || !put(&n, 4) || !put(msg.data(), msg.size())) return 1;
      }
      continue;
    }
    try {
      if (!doc || file != open) {
        doc.reset();
        open.clear();
        doc = std::make_unique<PdfDocument>(std::filesystem::path(std::u8string(file.begin(), file.end())));
        open = file;
      }
      const uint32_t ok = 0;
      if (op == kPages) {
        const int32_t n = doc->pages();
        if (!put(&ok, 4) || !put(&n, 4)) return 1;
      } else if (op == kRender) {
        const GrayImage img = doc->render(page, scale);
        const int32_t wh[2] = {img.width, img.height};
        if (!put(&ok, 4) || !put(wh, 8) || !put(img.pixels.data(), img.pixels.size())) return 1;
      } else if (op == kRenderShared) {  // the pixels into the region, only the size down the pipe
        const GrayImage img = doc->render(page, scale);
        const int32_t wh[2] = {img.width, img.height};
        const uint32_t there = shared && img.pixels.size() <= shared_bytes ? 1 : 0;
        if (there) std::memcpy(shared, img.pixels.data(), img.pixels.size());
        if (!put(&ok, 4) || !put(wh, 8) || !put(&there, 4)) return 1;
        if (!there && !put(img.pixels.data(), img.pixels.size())) return 1;
      } else if (op == kChars) {
        const std::vector<PdfChar> chars = doc->chars(page, scale);
        std::vector<WireChar> wire(chars.size());
        for (size_t i = 0; i < chars.size(); ++i) {
          const PdfChar& c = chars[i];
          wire[i] = {static_cast<uint32_t>(c.code), c.x0, c.y0, c.x1, c.y1, c.lx0, c.ly0, c.lx1, c.ly1, c.size,
                     c.mcid, static_cast<uint8_t>((c.generated ? 1 : 0) | (c.bad_map ? 2 : 0) | (c.invisible ? 4 : 0))};
        }
        const uint32_t n = static_cast<uint32_t>(wire.size());
        if (!put(&ok, 4) || !put(&n, 4) || !put(wire.data(), wire.size() * sizeof(WireChar))) return 1;
      } else if (op == kPictures) {
        const std::vector<LineBox> boxes = doc->pictures(page, scale);
        const uint32_t n = static_cast<uint32_t>(boxes.size());
        if (!put(&ok, 4) || !put(&n, 4)) return 1;
        for (const LineBox& b : boxes) {
          const double v[4] = {b.x0, b.y0, b.x1, b.y1};
          if (!put(v, sizeof v)) return 1;
        }
      } else if (op == kStructure) {
        const PdfStructure st = doc->structure(page);
        const uint32_t ne = static_cast<uint32_t>(st.elements.size());
        if (!put(&ok, 4) || !put(&ne, 4)) return 1;
        for (const PdfStructElement& e : st.elements) {
          const int32_t parent = e.parent;
          const uint32_t bytes = static_cast<uint32_t>(e.type.size());
          if (!put(&parent, 4) || !put(&bytes, 4) || !put(e.type.data(), e.type.size())) return 1;
        }
        const uint32_t nr = static_cast<uint32_t>(st.runs.size());
        if (!put(&nr, 4)) return 1;
        for (const auto& [el, mcid] : st.runs) {
          const int32_t v[2] = {el, mcid};
          if (!put(v, sizeof v)) return 1;
        }
      } else {
        throw Error("carbon-pdf: unknown request " + std::to_string(op));
      }
    } catch (const std::exception& e) {
      doc.reset();
      open.clear();
      const uint32_t bad = 1;
      const std::string msg = e.what();
      const uint32_t n = static_cast<uint32_t>(msg.size());
      if (!put(&bad, 4) || !put(&n, 4) || !put(msg.data(), msg.size())) return 1;
    }
  }
}

}  // namespace carbon
