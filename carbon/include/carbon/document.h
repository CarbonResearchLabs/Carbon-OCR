// Documents' pages read as one stream: page images, encoded pages and rendered PDF pages go in from any
// number of threads, and each page comes out as soon as it is finished whole, with the caller's tag. carbon-ocr ocr
// and carbon-ocr serve both read through it, so a page reads the same either way.
//
// A page's stages:
//
//   an image      OCR: the CPU engine's stream (Engine::submit), or on a GPU the batched pipeline's
//                 (carbon/pipeline.h)
//   a PDF page    its text layer checked (carbon/textlayer.h, on the thread that gives the page; a layer that
//                 runs sideways or upside down read upright), its bad-code runs set aside with the pictures
//                 pass on; a page whose layer fails goes to OCR as shown, with why. A page read from its layer:
//                 a tagged page in its tree's order, else its words in the lines the detector finds on it
//                 (LineFinder, where its text reads across); then its pictures pass, where it has pictures, ink
//                 the layer does not hold or lines set aside (the page with only them left read as given, from
//                 the lines the detector found there where it ran and no line was set aside: PictureRead), the
//                 lines found joining the layer's; then, its rotation set where it was read turned, the order
//                 model's reading order, but for a tagged page. Each step after the layer is a task of the engine's stream as it comes up,
//                 so a page finishes inside the stream, and only the pages in flight are held.
//
// On a GPU the text layer's pages' steps are tasks of the pipeline's stream too: their lines found in the
// detect stage's batches (Pipeline::submit_detect), their pictures read as given or from the lines found
// (Pipeline::submit with a PageTask), their order on its CPU threads (Pipeline::submit_order); with text_on_gpu
// off, on a CPU engine of their own, made at the first such page, by the CPU's models as on the CPU.
#pragma once

#include <cstdint>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "carbon/engine.h"
#include "carbon/pdf.h"
#include "carbon/pipeline.h"

namespace carbon {

// A text layer's page's pictures pass (carbon/textlayer.h's hybrid pages).
enum class PictureRead : uint8_t {
  kOff,     // none: the layer's lines alone
  kDetect,  // the page with only its pictures (and uncovered ink) left, its lines detected there
  kFound,   // the lines the detector found on the whole page in them, where it ran for the layer (picture_lines;
            // none: no pass); else as kDetect (the default)
};

struct DocumentOptions {
  EngineOptions engine;        // the CPU engine's, and the pipeline's engine options
  bool cuda = false;           // OCR pages on the GPU's batched pipeline, with `pipeline`'s options
  PipelineOptions pipeline;    // its engine options are `engine`
  bool text_layer = true;      // a PDF page whose text layer passes the checks read from it; false: every page OCR'd
  PictureRead pictures = PictureRead::kFound;
  bool layer_lines = true;     // a layer's words in the detector's lines (its text reading across); false: its own
  // On a GPU: a text layer's page's steps (its lines found, its pictures read, its order) as tasks of the
  // pipeline's stream, beside its OCR pages: detection and recognition on the GPU at the pipeline's precision, the
  // order on its CPU threads. False: on a CPU engine of their own, as on the CPU.
  bool text_on_gpu = true;
};

class DocumentStream {
 public:
  // Loads the set's models (the pipeline's on a GPU); throws Error where that fails.
  DocumentStream(const std::filesystem::path& set, DocumentOptions options);
  ~DocumentStream();
  DocumentStream(const DocumentStream&) = delete;
  DocumentStream& operator=(const DocumentStream&) = delete;

  // The PdfWant bits a PDF page given to pdf_page needs: its image, and with the text layer on, its characters,
  // structure tree and (with the pictures pass) pictures, read upright where its layer runs sideways.
  unsigned pdf_want() const;

  // A page in, with the caller's tag, which comes back with it; each waits while the stream is full, and throws
  // Error once close() was called. From any thread.
  void image(GrayImage page, uint64_t tag);
  // An image file, read and decoded by the worker that takes the page (on a GPU, here).
  void image_file(const std::filesystem::path& file, uint64_t tag);
  // An encoded image (PNG, JPEG, BMP, PNM), decoded by a worker (on a GPU, here); bytes that do not decode are the
  // page's error, `unreadable`.
  void encoded(std::shared_ptr<const std::string> bytes, uint64_t tag);
  // A rendered PDF page, with what pdf_want asks: read from its text layer where it passes, else by OCR. The text
  // layer's checks and the detector's lines for it run on this thread. A page whose layer runs sideways or upside
  // down comes rendered upright (kPdfUpright, PdfPage::turn): read from its layer so, its rotation set to the turn
  // (read_turned); refused for OCR, it is read as shown, as_shown's render of it (on this thread; none given: the
  // upright render turned back).
  void pdf_page(PdfPage page, uint64_t tag, const std::function<GrayImage()>& as_shown = {});

  struct Done {
    uint64_t tag = 0;
    OcrPage page;
    std::exception_ptr error;
    bool unreadable = false;      // its file or bytes did not give an image
    bool from_text_layer = false;  // read from its text layer
    bool pictures_read = false;    // and its pictures pass run
  };
  // The next page finished, of every page given (in the order they finish); nullopt once close() was called and
  // every page given is out. An error outside every page's work (a GPU that can run nothing more) throws.
  std::optional<Done> next();
  // No more pages: next() ends once every page given is out.
  void close();

  bool cuda() const;
  int workers() const;        // the CPU engine's workers (0 on a GPU before a text page made one)
  int in_flight() const;      // the OCR stream's pages in flight at most
  const Pipeline* pipeline() const;  // the GPU's, else null

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace carbon
