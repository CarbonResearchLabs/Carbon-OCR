// Documents' pages read as one stream (carbon/document.h): each page's stages as tasks of the engine's or the
// pipeline's stream, a thread for each stream taking its tasks as they finish and giving the page its next one.

#include "carbon/document.h"

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "carbon/error.h"
#include "carbon/image.h"
#include "carbon/textlayer.h"

namespace carbon {

namespace {

// What a task of a stream is for: a page's OCR, a text layer's page's lines found (the pipeline's detect-only
// pages), its pictures pass, or its reading order.
enum class Stage : uint8_t { kOcr, kLines, kPictures, kOrder };

}  // namespace

struct DocumentStream::Impl {
  // A text layer's page between its steps.
  struct TextPage {
    OcrPage page;           // read from its layer, its pictures' lines in once its pass is done
    bool order = false;     // its reading order still to come (every page but a tagged one)
    bool pictures = false;  // its pictures pass run
    TextLayer layer;        // while its lines are being found: the layer, and its pictures' page and regions
    std::optional<GrayImage> pictures_page;
    std::vector<LineBox> to_read;
    bool set_aside = false;  // lines bad-code runs touch among them (read where the pictures' page shows them)
    int turn = 0;            // the quarter turns it was read at, for its rotation (read_turned) before its order
  };
  struct Job {
    uint64_t tag = 0;
    Stage stage = Stage::kOcr;
    std::string note;  // an OCR'd PDF page: why its text layer was not used
    std::shared_ptr<TextPage> text;
  };
  // A stream the pages' tasks go to (the CPU engine's or the pipeline's): its tasks by their numbers, and the
  // thread that takes them as they finish.
  struct Backend {
    std::unique_ptr<Engine> engine;
    std::unique_ptr<Pipeline> pipe;
    std::mutex submit_m;   // a task's number and its place in the stream, together
    size_t submitted = 0;  // under submit_m
    std::mutex m;
    std::condition_variable cv;
    std::unordered_map<size_t, Job> jobs;  // given and not yet taken, under m
    size_t outstanding = 0;                // under m
    bool stop = false;
    std::thread collector;
  };

  std::filesystem::path set;
  DocumentOptions o;
  std::unique_ptr<Backend> ocr;   // OCR pages; on the CPU every task
  std::unique_ptr<Backend> text;  // on a GPU: the text layer's pages' tasks, a CPU engine made at the first
  std::mutex text_m;
  std::mutex finders_m;
  std::vector<std::unique_ptr<LineFinder>> finders;  // free ones, for the threads that give PDF pages
  // the pages out
  std::mutex out_m;
  std::condition_variable out_cv;
  std::deque<Done> out;
  size_t given = 0, delivered = 0;
  bool closed = false;

  Impl(const std::filesystem::path& s, DocumentOptions opts) : set(s), o(std::move(opts)) {
    ocr = std::make_unique<Backend>();
    if (o.cuda) {
      PipelineOptions po = o.pipeline;
      po.engine = o.engine;
      po.device = Device::kCuda;
      ocr->pipe = std::make_unique<Pipeline>(set, po);
    } else {
      ocr->engine = std::make_unique<Engine>(set, o.engine);
    }
    start(*ocr);
  }

  ~Impl() {
    // The OCR stream's thread first: what it finishes may still give the text layer's stream a task (on a GPU)
    stop(*ocr);
    Backend* t = nullptr;
    {
      std::lock_guard lk(text_m);
      t = text.get();
    }
    if (t) stop(*t);
    if (ocr->pipe) ocr->pipe->close();
  }

  // A stream's thread ended once every task out is taken.
  static void stop(Backend& b) {
    {
      std::lock_guard lk(b.m);
      b.stop = true;
    }
    b.cv.notify_all();
    if (b.collector.joinable()) b.collector.join();
  }

  void start(Backend& b) {
    b.collector = std::thread([this, &b] { collect(b); });
  }

  // The text layer's pages' tasks: the stream's own (the CPU engine's; on a GPU the pipeline's), or with
  // text_on_gpu off on a GPU a CPU engine of their own, made at the first page that needs one.
  Backend& text_backend() {
    if (!o.cuda || o.text_on_gpu) return *ocr;
    std::lock_guard lk(text_m);
    if (!text) {
      auto b = std::make_unique<Backend>();
      b->engine = std::make_unique<Engine>(set, o.engine);
      text = std::move(b);
      start(*text);
    }
    return *text;
  }

  // ---- the pages out
  void deliver(Done d) {
    {
      std::lock_guard lk(out_m);
      out.push_back(std::move(d));
      ++delivered;
    }
    out_cv.notify_all();
  }
  void deliver_error(uint64_t tag, std::exception_ptr e, bool unreadable = false, bool from_text = false) {
    Done d;
    d.tag = tag;
    d.error = std::move(e);
    d.unreadable = unreadable;
    d.from_text_layer = from_text;
    deliver(std::move(d));
  }
  void count_given() {
    std::lock_guard lk(out_m);
    if (closed) throw Error("DocumentStream: closed");
    ++given;
  }

  // ---- the tasks
  // A task into a stream: its number taken with its place, the job kept under it before the stream can finish
  // it; a task the stream refuses is its page's error.
  template <class Submit>
  void give(Backend& b, Job job, Submit&& submit) {
    std::lock_guard sl(b.submit_m);
    const size_t n = b.submitted;
    {
      std::lock_guard lk(b.m);
      b.jobs.emplace(n, std::move(job));
    }
    try {
      const size_t got = submit();
      if (got != n) throw Error("DocumentStream: the stream numbered a task " + std::to_string(got) + ", not " +
                                std::to_string(n));
    } catch (...) {
      Job lost;
      {
        std::lock_guard lk(b.m);
        lost = std::move(b.jobs.at(n));
        b.jobs.erase(n);
      }
      deliver_error(lost.tag, std::current_exception(), false, lost.text != nullptr);
      return;
    }
    ++b.submitted;
    {
      std::lock_guard lk(b.m);
      ++b.outstanding;
    }
    b.cv.notify_all();
  }

  void ocr_page(GrayImage image, uint64_t tag, std::string note = {}) {
    Job j;
    j.tag = tag;
    j.note = std::move(note);
    give(*ocr, std::move(j), [&] { return ocr->engine ? ocr->engine->submit(std::move(image)) : ocr->pipe->submit(std::move(image)); });
  }

  // A text layer's page's task into a stream, the engine's or the pipeline's.
  void text_task(Backend& b, uint64_t tag, Stage stage, const std::shared_ptr<TextPage>& tp,
                 const std::function<size_t(Backend&)>& submit) {
    Job j;
    j.tag = tag;
    j.stage = stage;
    j.text = tp;
    give(b, std::move(j), [&] { return submit(b); });
  }

  // A text layer's page's lines found (here, or on a GPU by the pipeline): its words into them, then its
  // pictures.
  void layer_lines(uint64_t tag, const std::shared_ptr<TextPage>& tp, const OcrPage& found) {
    std::vector<LineBox> boxes;
    for (const OcrLine& l : found.lines) boxes.push_back(l.box);
    TextLayer detected = std::move(tp->layer);
    lines_from_detection(detected, boxes);
    std::vector<LineBox> pictures = std::move(tp->page.pictures);
    tp->page = page_from_text_layer(detected, found.width, found.height);
    tp->page.pictures = std::move(pictures);
    pictures_step(tag, tp, &found);
  }

  // A text layer's page's pictures pass, where it has pictures or uncovered ink to read: as given, from the lines
  // the detector found in them where it ran (none: no pass); then its next step.
  void pictures_step(uint64_t tag, const std::shared_ptr<TextPage>& tp, const OcrPage* found) {
    if (tp->pictures_page) {
      PageTask task;
      task.as_given = true;  // orientation and deskew read whole pages
      bool skip = false;
      // The lines found, unless lines were set aside: those are read on the pictures' page, as before
      if (found && o.pictures == PictureRead::kFound && !tp->set_aside) {
        OcrPage picked = picture_lines(*found, tp->page, tp->to_read);
        if (picked.lines.empty()) {
          skip = true;
        } else {
          task.found = std::make_shared<const OcrPage>(std::move(picked));
        }
      }
      GrayImage pictures = std::move(*tp->pictures_page);
      tp->pictures_page.reset();
      tp->to_read.clear();
      if (!skip) {
        text_task(text_backend(), tag, Stage::kPictures, tp, [&](Backend& b) {
          return b.engine ? b.engine->submit(std::move(pictures), std::move(task))
                          : b.pipe->submit(std::move(pictures), std::move(task));
        });
        return;
      }
    }
    text_step(tag, tp);
  }

  // A text layer's page's next step: its order, else it is done; a page read turned upright first given its
  // rotation, a page of the page as shown (its boxes stay the upright page's, as an OCR page's).
  void text_step(uint64_t tag, const std::shared_ptr<TextPage>& tp) {
    read_turned(tp->page, tp->turn);
    tp->turn = 0;
    if (tp->order) {
      text_task(text_backend(), tag, Stage::kOrder, tp, [&](Backend& b) {
        return b.engine ? b.engine->submit_order(std::move(tp->page)) : b.pipe->submit_order(std::move(tp->page));
      });
      return;
    }
    Done d;
    d.tag = tag;
    d.page = std::move(tp->page);
    d.from_text_layer = true;
    d.pictures_read = tp->pictures;
    deliver(std::move(d));
  }

  // A task done: the page's next step, or the page out.
  void finished(Job& job, OcrPage page, std::exception_ptr error, bool unreadable) {
    if (error) {
      deliver_error(job.tag, std::move(error), unreadable, job.text != nullptr);
      return;
    }
    try {
      switch (job.stage) {
        case Stage::kOcr: {
          Done d;
          d.tag = job.tag;
          d.page = std::move(page);
          d.page.text_layer = job.note;
          deliver(std::move(d));
          return;
        }
        case Stage::kLines:
          layer_lines(job.tag, job.text, page);
          return;
        case Stage::kPictures:
          add_picture_lines(job.text->page, page);
          job.text->pictures = true;
          text_step(job.tag, job.text);
          return;
        case Stage::kOrder: {
          Done d;
          d.tag = job.tag;
          d.page = std::move(page);
          d.from_text_layer = true;
          d.pictures_read = job.text->pictures;
          deliver(std::move(d));
          return;
        }
      }
    } catch (...) {
      deliver_error(job.tag, std::current_exception(), false, job.text != nullptr);
    }
  }

  // A stream's thread: its tasks as they finish, while any is out.
  void collect(Backend& b) {
    for (;;) {
      {
        std::unique_lock lk(b.m);
        b.cv.wait(lk, [&] { return b.stop || b.outstanding > 0; });
        if (b.outstanding == 0) return;  // stopping, nothing out
      }
      size_t n = 0;
      OcrPage page;
      std::exception_ptr error;
      bool unreadable = false;
      if (b.engine) {
        std::optional<Engine::Finished> f = b.engine->next_finished();
        if (!f) continue;
        n = f->number;
        page = std::move(f->page);
        error = f->error;
        unreadable = f->unreadable;
      } else {
        std::optional<Pipeline::Finished> f;
        try {
          f = b.pipe->next_finished();
        } catch (...) {  // the pipeline stopped (its device broken): every task out fails with its error
          const std::exception_ptr e = std::current_exception();
          std::vector<Job> lost;
          {
            std::lock_guard lk(b.m);
            for (auto& [k, j] : b.jobs) lost.push_back(std::move(j));
            b.jobs.clear();
            b.outstanding = 0;
          }
          for (Job& j : lost) deliver_error(j.tag, e, false, j.text != nullptr);
          continue;
        }
        if (!f) continue;
        n = f->number;
        page = std::move(f->page);
        error = f->error;
      }
      Job job;
      {
        std::lock_guard lk(b.m);
        auto it = b.jobs.find(n);
        if (it == b.jobs.end()) continue;
        job = std::move(it->second);
        b.jobs.erase(it);
        --b.outstanding;
      }
      finished(job, std::move(page), std::move(error), unreadable);
    }
  }

  // ---- the detector's lines for a text layer, on the giving thread: a free finder, or a new one
  std::unique_ptr<LineFinder> take_finder() {
    {
      std::lock_guard lk(finders_m);
      if (!finders.empty()) {
        std::unique_ptr<LineFinder> f = std::move(finders.back());
        finders.pop_back();
        return f;
      }
    }
    return std::make_unique<LineFinder>(set, o.engine);
  }
  void give_back(std::unique_ptr<LineFinder> f) {
    std::lock_guard lk(finders_m);
    finders.push_back(std::move(f));
  }

  void pdf_page(PdfPage pg, uint64_t tag, const std::function<GrayImage()>& as_shown) {
    if (!o.text_layer) {
      ocr_page(std::move(pg.image), tag);
      return;
    }
    const GrayView view = pg.image.view();
    TextLayerRule rule;
    rule.set_aside_bad_runs = o.pictures != PictureRead::kOff;  // the lines bad-code runs touch read with the
                                                               // pictures
    rule.rules = o.engine.grid && o.engine.grid->check;  // its ruled lines where the false-table check asks
    const TextLayer layer = read_text_layer(pg.chars, view, rule);
    if (!layer.check.usable) {  // to OCR as shown (a page read turned upright for its layer rendered as shown)
      GrayImage image = !pg.turn   ? std::move(pg.image)
                        : as_shown ? as_shown()
                                   : rotate_quarter(view, 4 - pg.turn);
      ocr_page(std::move(image), tag, layer.check.reason);
      return;
    }
    auto tp = std::make_shared<TextPage>();
    tp->turn = pg.turn;
    // a tagged page in its tree's order and roles, else the XY-cut's, the order model's after
    tp->page = page_from_tagged_layer(layer, pg.structure, pg.image.width, pg.image.height);
    tp->order = tp->page.order != "tagged";
    if (o.pictures != PictureRead::kOff) {  // its pictures' own text, which the layer does not hold
      tp->page.pictures = std::move(pg.pictures);
      tp->to_read = pictures_to_read(tp->page.pictures, pg.image.width, pg.image.height);
      // And its ink the layer and the pictures do not account for (text drawn as paths, in a form)
      for (const LineBox& b : uncovered_ink(view, layer, pg.chars, tp->page.pictures)) tp->to_read.push_back(b);
      // And the lines bad-code runs touch
      const std::vector<LineBox> set_aside = set_aside_to_read(layer, pg.image.width, pg.image.height);
      tp->set_aside = !set_aside.empty();
      tp->to_read.insert(tp->to_read.end(), set_aside.begin(), set_aside.end());
      if (!tp->to_read.empty()) tp->pictures_page = pictures_only(view, tp->to_read);
    }
    // the detector's lines, for a page whose text reads across upright (a turned one keeps the layer's): on a GPU
    // a detect-only page of the pipeline's, else here
    if (o.layer_lines && tp->page.order != "tagged" && reads_across(layer)) {
      if (ocr->pipe && o.text_on_gpu) {
        tp->layer = layer;
        text_task(*ocr, tag, Stage::kLines, tp, [&](Backend& b) { return b.pipe->submit_detect(std::move(pg.image)); });
        return;
      }
      std::unique_ptr<LineFinder> finder = take_finder();
      const OcrPage found = finder->detect(view);
      give_back(std::move(finder));
      tp->layer = layer;
      layer_lines(tag, tp, found);
      return;
    }
    pictures_step(tag, tp, nullptr);
  }
};

DocumentStream::DocumentStream(const std::filesystem::path& set, DocumentOptions options)
    : impl_(std::make_unique<Impl>(set, std::move(options))) {}

DocumentStream::~DocumentStream() = default;

unsigned DocumentStream::pdf_want() const {
  const DocumentOptions& o = impl_->o;
  return kPdfImage | (o.text_layer ? kPdfChars | kPdfStructure | kPdfUpright : 0u) |
         (o.text_layer && o.pictures != PictureRead::kOff ? kPdfPictures : 0u);
}

void DocumentStream::image(GrayImage page, uint64_t tag) {
  impl_->count_given();
  impl_->ocr_page(std::move(page), tag);
}

void DocumentStream::image_file(const std::filesystem::path& file, uint64_t tag) {
  impl_->count_given();
  Impl::Backend& b = *impl_->ocr;
  if (b.engine) {  // decoded by the worker that takes it
    Impl::Job j;
    j.tag = tag;
    impl_->give(b, std::move(j), [&] { return b.engine->submit(file); });
    return;
  }
  GrayImage page;
  try {
    page = load_image(file);
  } catch (...) {
    impl_->deliver_error(tag, std::current_exception(), true);
    return;
  }
  impl_->ocr_page(std::move(page), tag);
}

void DocumentStream::encoded(std::shared_ptr<const std::string> bytes, uint64_t tag) {
  impl_->count_given();
  Impl::Backend& b = *impl_->ocr;
  const std::span<const uint8_t> raw(reinterpret_cast<const uint8_t*>(bytes->data()), bytes->size());
  if (b.engine) {
    Impl::Job j;
    j.tag = tag;
    impl_->give(b, std::move(j), [&] { return b.engine->submit_encoded(raw, bytes); });
    return;
  }
  GrayImage page;
  try {
    page = decode_image(raw);
  } catch (...) {
    impl_->deliver_error(tag, std::current_exception(), true);
    return;
  }
  impl_->ocr_page(std::move(page), tag);
}

void DocumentStream::pdf_page(PdfPage page, uint64_t tag, const std::function<GrayImage()>& as_shown) {
  impl_->count_given();
  try {
    impl_->pdf_page(std::move(page), tag, as_shown);
  } catch (...) {  // its text layer's steps on this thread: the page's own error
    impl_->deliver_error(tag, std::current_exception());
  }
}

std::optional<DocumentStream::Done> DocumentStream::next() {
  std::unique_lock lk(impl_->out_m);
  impl_->out_cv.wait(lk, [&] {
    return !impl_->out.empty() || (impl_->closed && impl_->delivered == impl_->given);
  });
  if (impl_->out.empty()) return std::nullopt;
  Done d = std::move(impl_->out.front());
  impl_->out.pop_front();
  return d;
}

void DocumentStream::close() {
  {
    std::lock_guard lk(impl_->out_m);
    impl_->closed = true;
  }
  impl_->out_cv.notify_all();
}

bool DocumentStream::cuda() const { return impl_->o.cuda; }

int DocumentStream::workers() const {
  if (impl_->ocr->engine) return impl_->ocr->engine->workers();
  std::lock_guard lk(impl_->text_m);
  return impl_->text ? impl_->text->engine->workers() : 0;
}

int DocumentStream::in_flight() const {
  if (impl_->ocr->pipe) return impl_->o.pipeline.in_flight;
  return impl_->o.engine.in_flight > 0 ? impl_->o.engine.in_flight : 2 * impl_->ocr->engine->workers();
}

const Pipeline* DocumentStream::pipeline() const { return impl_->ocr->pipe.get(); }

}  // namespace carbon
