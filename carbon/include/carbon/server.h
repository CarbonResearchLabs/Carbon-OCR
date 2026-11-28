// carbon-ocr serve: the engine behind HTTP/1.1 (carbon/http.h). Its models load once, and every client's pages
// keep its workers busy together, through the document stream carbon-ocr ocr reads by (carbon/document.h): on the
// CPU the engine's stream, each image decoded by the worker that reads it and answered as soon as it is done; on a
// GPU the batched pipeline (carbon/pipeline.h), pages from every client in its batches; a PDF's pages rendered by
// the helper processes (PdfPool) and read as carbon-ocr ocr reads them (from their text layer where it passes, its
// pictures by OCR, else by OCR), answered once its last page is done. Every model file the engine runs, it runs:
// on the CPU FP32, FP16 and INT8 on ONNX Runtime or the native backend (engine.precision, engine.backend), on the
// GPU FP16, FP32 and symmetric INT8 on TensorRT (gpu_set_model). A page's answer is what carbon-ocr ocr writes for
// it, by the same writers (carbon/output.h).
//
//   POST /ocr[?format=json|text|hocr|alto|page|markdown|html|pdf][&name=NAME][&text_only=1][&html=1][&dpi=N]
//        the body a page's image (PNG, JPEG, BMP, PNM, GIF or WebP) or a PDF, each by its first bytes (file_kind).
//        For an image, json (the default) is the
//        page's object (OcrPage::json; html=1 adds "html", its body as format=html writes it, to_html_body, as
//        carbon_ocr's html=True does), text its lines (OcrPage::text); hocr, alto and page (PAGE XML) name its
//        image NAME ("page" by default); pdf is a searchable PDF of it at image_dpi (text_only=1: its words
//        alone). A PDF's pages are rendered at dpi (200 by default) and answered together: json {"pages": [...]},
//        each page's object with its "page" number first; the other formats as carbon-ocr ocr writes the
//        document's pages (page, PAGE XML, for a PDF of one page alone).
//   POST /parse  a route for document-parsing benchmark clients: the body a JSON object {"page": the page's
//        file in base64 (an image or a PDF), "suffix": its extension, "settings": {"dpi": N}} (settings
//        optional, dpi the only one); the answer {"pages": [each page's object with "html" and "markdown"],
//        "served": what answers, "vendor": {"carbon-ocr": its version}}.
//   GET  /health   200, "ok"
//   GET  /served   {"weights": the set's name, "revision": a hash of its model files}, as /parse's "served"
//   GET  /info     {"version", "set", "device", "precision", "workers", "in_flight", "queue", "pdf", "served"}
//
// Statuses: 200; 400 a request it cannot read, an unknown format or setting, or format=page for a PDF of more
// than a page; 404 an unknown path; 405 a method the path does not take; 411 a body without a Content-Length; 413
// a body past max_body; 415 a body that is no image or PDF it reads (a PDF where the build reads none); 422 an
// image that does not decode, or a PDF that does not open or has no pages; 431 a head too large; 500 the engine's
// error; 503 the queue full: the pages in flight and `queue` requests more held (Retry-After: 1). An error's body is
// its reason, as text.
//
// Its threads: one for the network (poll over the listening socket and every connection, nothing else on it), a
// few that give the pages to the stream (on a GPU decoding the images; a PDF's pages rendered one at a time, a
// PDF's before the next request's), one that takes them as they finish, and a few that write the answers. The
// engine's workers do the rest, as for carbon-ocr ocr.
#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "carbon/document.h"
#include "carbon/engine.h"
#include "carbon/pipeline.h"

namespace carbon {

struct ServerOptions {
  std::string host = "127.0.0.1";  // the address to listen on: 0.0.0.0 for every interface
  int port = 8080;                 // 0: a free one (Server::port)
  size_t max_body = size_t{64} << 20;  // a request's body, past which 413
  // Requests waiting for the engine, past its pages in flight; past them, 503 at once. A request is held from when
  // it is read until its answer goes out, wherever it waits (the threads that give the pages each hold one while
  // the stream is full), so at most the pages in flight and this many are held.
  int queue = 64;
  EngineOptions engine;
  bool cuda = false;          // the batched pipeline on the GPU, with `pipeline`'s options
  PipelineOptions pipeline;   // its engine options are `engine`
  // The threads that give the pages to the stream: on a GPU decoding the images, a PDF's pages rendered on them
  // (0: half the physical cores, at least 4 on a GPU, 2 on the CPU)
  int decoders = 0;
  // The threads that write the pages' answers in their formats, off the thread that takes the pages as they
  // finish (0: 2 on the CPU, half the decoders on a GPU, at least 2).
  int writers = 0;
  double image_dpi = 300;     // format=pdf: the page's image at this dpi
  // A PDF's pages as carbon-ocr ocr reads them (DocumentOptions): their text layers, pictures pass and lines.
  bool text_layer = true;
  PictureRead pictures = PictureRead::kFound;
  bool layer_lines = true;
  std::filesystem::path pdf_helper;  // the PDF helpers' program (PdfPool); empty: carbon-pdf beside this one
  std::function<void(const std::string&)> log;  // a line a request, where given (called on several threads)
};

class Server {
 public:
  // Loads the set's models and listens; throws Error where either fails.
  Server(const std::filesystem::path& set, ServerOptions options);
  ~Server();
  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  int port() const;  // the port it listens on
  // Serves until stop(): then no more requests are read, the pages being read are finished, and every
  // connection is closed.
  void run();
  // From any thread, a signal handler among them (it sets a flag and writes a byte to a socket).
  void stop();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace carbon
