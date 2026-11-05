// carbon-ocr serve: the engine behind HTTP/1.1 (carbon/http.h). Its models load once, and every client's pages
// keep its workers busy together: on the CPU the engine's stream (Engine::submit_encoded, next_finished), each
// page decoded by the worker that reads it and answered as soon as it is done; on a GPU the batched pipeline
// (carbon/pipeline.h), pages from every client in its batches. Every model file the engine runs, it runs: on the
// CPU FP32, FP16 and INT8 on ONNX Runtime or the native backend (engine.precision, engine.backend), on the GPU
// FP16, FP32 and symmetric INT8 on TensorRT (gpu_set_model). A page's answer is what carbon-ocr ocr writes
// for that page, by the same writers (carbon/output.h).
//
//   POST /ocr[?format=json|text|hocr|alto|page|markdown|html|pdf][&name=NAME][&text_only=1][&html=1]
//        the body a page's image: PNG, JPEG, BMP or PNM. json (the default) is the page's object
//        (OcrPage::json; html=1 adds "html", its body as format=html writes it, to_html_body, as
//        carbon_ocr's html=True does), text its lines (OcrPage::text); hocr, alto and page (PAGE XML)
//        name its image NAME ("page" by default); pdf is a searchable PDF of it at image_dpi
//        (text_only=1: its words alone).
//   GET  /health   200, "ok"
//   GET  /info     {"version", "set", "device", "precision", "workers", "in_flight", "queue"}
//
// Statuses: 200; 400 a request it cannot read, or an unknown format; 404 an unknown path; 405 a method the path
// does not take; 411 a body without a Content-Length; 413 a body past max_body; 415 a body that is no image it
// reads (a PDF among them: carbon-ocr ocr reads PDFs); 422 an image that does not decode; 431 a head too large;
// 500 the engine's error; 503 the queue full (Retry-After: 1). An error's body is its reason, as text.
//
// Its threads: one for the network (poll over the listening socket and every connection, nothing else on it),
// one that gives the pages to the engine, one that takes them as they finish and writes their answers; on a GPU,
// a few that decode the images. The engine's workers do the rest, as for carbon-ocr ocr.
#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

#include "carbon/engine.h"
#include "carbon/pipeline.h"

namespace carbon {

struct ServerOptions {
  std::string host = "127.0.0.1";  // the address to listen on: 0.0.0.0 for every interface
  int port = 8080;                 // 0: a free one (Server::port)
  size_t max_body = size_t{64} << 20;  // a request's body, past which 413
  int queue = 64;    // requests waiting for the engine, past its pages in flight; past them, 503
  EngineOptions engine;
  bool cuda = false;          // the batched pipeline on the GPU, with `pipeline`'s options
  PipelineOptions pipeline;   // its engine options are `engine`
  int decoders = 0;  // on a GPU, the threads that decode the pages (0: half the physical cores, at least 4)
  // The threads that write the pages' answers in their formats, off the thread that takes the pages as they
  // finish (0: 2 on the CPU, half the decoders on a GPU, at least 2).
  int writers = 0;
  double image_dpi = 300;     // format=pdf: the page's image at this dpi
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
