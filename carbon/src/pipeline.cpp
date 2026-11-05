// The batched OCR pipeline (carbon/pipeline.h): its stages, queues and the CPU
// device's models.  The GPU's are in cuda/pipeline_cuda.cpp.

#include "carbon/pipeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <thread>
#include <tuple>
#include <utility>

#include "carbon/error.h"
#include "carbon/parallel.h"
#include "carbon/session.h"
#include "engine_steps.h"

namespace carbon {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

int round_up(int v, int step) { return (v + step - 1) / step * step; }

// A queue between stages: unbounded (submit's in_flight bounds the pages in the
// pipeline), closed when the pipeline ends.
template <class T>
class Channel {
 public:
  void push(T v) {
    {
      std::lock_guard lk(m_);
      if (closed_) return;
      q_.push_back(std::move(v));
    }
    cv_.notify_one();
  }
  // Waits for an item; nullopt once closed.
  std::optional<T> pop() {
    std::unique_lock lk(m_);
    cv_.wait(lk, [&] { return closed_ || !q_.empty(); });
    if (closed_) return std::nullopt;
    T v = std::move(q_.front());
    q_.pop_front();
    return v;
  }
  std::optional<T> try_pop() {
    std::lock_guard lk(m_);
    if (closed_ || q_.empty()) return std::nullopt;
    T v = std::move(q_.front());
    q_.pop_front();
    return v;
  }
  void close() {
    {
      std::lock_guard lk(m_);
      closed_ = true;
      q_.clear();
    }
    cv_.notify_all();
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<T> q_;
  bool closed_ = false;
};

// ---------------------------------------------------------------- the CPU device

class CpuModels final : public PipelineModels {
 public:
  CpuModels(const std::filesystem::path& set, const EngineOptions& o, int threads) {
    SessionOptions so;
    so.threads = threads;
    orient_ = open_set_model(set, "orient", o.backend, so, o.precision);
    det_ = open_set_model(set, "det", o.backend, so, o.precision);
    rec_ = open_set_model(set, o.rec, o.backend, so, o.precision);
  }

  Device device() const override { return Device::kCpu; }

  std::vector<float> orient(const float* thumbs, int n, int side) override {
    Tensor x = Tensor::empty(DType::kFloat32, {n, 1, side, side});
    std::copy(thumbs, thumbs + static_cast<size_t>(n) * side * side, x.data<float>().begin());
    const Tensor y = std::move(orient_->run(std::span<const Tensor>(&x, 1))[0]);
    const auto z = y.data<float>();
    return {z.begin(), z.end()};
  }

  // The CPU engine's: the page's own size, padded to a multiple of 32.
  std::pair<int, int> detect_size(int width, int height) const override {
    return {round_up(height, 32), round_up(width, 32)};
  }

  std::vector<Detection> detect(std::span<const GrayView> pages, int ph, int pw, const DetRule& rule) override {
    const int n = static_cast<int>(pages.size());
    Tensor x = Tensor::empty(DType::kFloat32, {n, 1, ph, pw});
    float* px = x.data<float>().data();
    for (int i = 0; i < n; ++i) {
      ink_input(pages[i], pages[i].width, pages[i].height, pw, ph, px + static_cast<size_t>(i) * ph * pw);
    }
    const Tensor maps = std::move(det_->run(std::span<const Tensor>(&x, 1))[0]);
    const Shape& s = maps.shape();  // [n, 4, mh, mw]
    const int mh = static_cast<int>(s[2]), mw = static_cast<int>(s[3]);
    std::vector<Detection> out;
    const float* m = maps.data<float>().data();
    for (int i = 0; i < n; ++i) {
      out.push_back(decode_det(m + static_cast<size_t>(i) * 4 * mh * mw, mh, mw, 1.0, rule));
    }
    return out;
  }

  bool second_pass() const override { return true; }

  Tensor detect_one(const Tensor& input) override {
    return std::move(det_->run(std::span<const Tensor>(&input, 1))[0]);
  }

  void set_pages(std::span<const GrayView> pages, std::span<const RowRuns>) override {
    pages_.assign(pages.begin(), pages.end());
  }

  void recognize(std::span<const int32_t> page_of, std::span<const LineBox> boxes, int bw, const CropRule& crop,
                 std::vector<CtcLine>& lines, std::vector<int32_t>& lengths) override {
    const std::vector<GrayView>& pages = pages_;
    const int m = static_cast<int>(boxes.size()), h = crop.height;
    std::vector<uint8_t> crops(static_cast<size_t>(m) * h * bw);
    std::vector<int32_t> widths(m);
    for (int i = 0; i < m;) {  // each page's run of lines in one call
      int e = i + 1;
      while (e < m && page_of[e] == page_of[i]) ++e;
      crop_lines(pages[page_of[i]], boxes.subspan(i, e - i), bw, crops.data() + static_cast<size_t>(i) * h * bw,
                 std::span<int32_t>(widths.data() + i, e - i), crop);
      i = e;
    }
    Tensor x = Tensor::empty(DType::kFloat32, {m, 1, h, bw});
    float* px = x.data<float>().data();
    for (size_t i = 0; i < crops.size(); ++i) px[i] = 1.0f - crops[i] / 255.0f;
    const Tensor logits = std::move(rec_->run(std::span<const Tensor>(&x, 1))[0]);
    const Shape& s = logits.shape();
    lengths.resize(m);
    for (int i = 0; i < m; ++i) {
      lengths[i] = std::min<int32_t>(static_cast<int32_t>(s[2]), widths[i] / crop.frame_px);
    }
    lines = ctc_greedy(logits.data<float>().data(), m, static_cast<int>(s[1]), static_cast<int>(s[2]), lengths);
  }

  const Session& rec() const { return *rec_; }

 private:
  std::unique_ptr<Session> orient_, det_, rec_;
  std::vector<GrayView> pages_;
};

}  // namespace

Tensor PipelineModels::detect_one(const Tensor&) {
  throw Error("this device's pipeline does not run the second detection pass");
}

std::unique_ptr<PipelineModels> cpu_pipeline_models(const std::filesystem::path& set, const EngineOptions& o,
                                                    int threads) {
  return std::make_unique<CpuModels>(set, o, threads);
}

// ---------------------------------------------------------------- the pipeline

struct Pipeline::Impl {
  struct Page {
    size_t index = 0;
    GrayImage image;         // as given
    GrayImage small;         // at the detector's scale, until made upright
    std::vector<float> thumb;
    detail::Upright up;
    OcrPage out;
    int ph = 0, pw = 0;      // the detector's padded size
    Clock::time_point start, lines_start;
  };
  using PagePtr = std::shared_ptr<Page>;  // shared: a CPU task's copy of it

  PipelineOptions o;
  EngineOptions& e;
  std::vector<std::string> vocab;
  std::unique_ptr<PipelineModels> models;
  detail::OrderModel order_model;

  // CPU threads: a task queue, each thread with its own order session
  Channel<std::function<void(int)>> tasks;
  std::vector<std::thread> cpu, stage;
  std::vector<std::unique_ptr<Session>> order;

  Channel<PagePtr> to_orient, to_detect, to_recognize;

  std::mutex m;
  std::condition_variable cv;
  size_t submitted = 0, delivered = 0, taken = 0;
  bool closed = false, stopping = false;
  bool by_finished = false;  // next_finished took a page: next may not
  std::exception_ptr error;
  std::map<size_t, OcrPage> ready;
  std::map<size_t, std::exception_ptr> failed;  // the pages that met an error, each with its own

  std::atomic<int64_t> pages{0}, lines{0}, orient_runs{0}, detect_runs{0}, recognize_runs{0};

  Impl(const std::filesystem::path& set, PipelineOptions options, std::unique_ptr<PipelineModels> given = nullptr)
      : o(std::move(options)), e(o.engine) {
    if (o.page_batch < 1 || o.line_batch < 1 || o.in_flight < 1) {
      throw Error("Pipeline: page_batch, line_batch and in_flight are at least 1");
    }
    vocab = load_vocab(set / (e.rec + ".vocab.json"));
    e.crop = detail::model_crop(set, e.rec, e.crop);
    if (!e.turn_probability) e.turn_probability = detail::model_turn_probability(set);
    const int cores = cpu_topology().default_workers();
    if (given) {  // the caller's own (Pipeline's third constructor)
      models = std::move(given);
    } else {
      const Device device = o.device.value_or(cuda_pipeline_available() ? Device::kCuda : Device::kCpu);
      models = device == Device::kCuda ? cuda_pipeline_models(set, e, o)
                                       : cpu_pipeline_models(set, e, o.model_threads > 0 ? o.model_threads : cores);
      if (device == Device::kCpu) check_recognizer(static_cast<const CpuModels&>(*models).rec());
    }
    order_model = detail::choose_order(set, e);
    const int n = o.threads > 0 ? o.threads : cores;
    order.resize(n);
    if (!order_model.file.empty()) {
      SessionOptions single;
      for (auto& s : order) s = open_session(order_model.backend, order_model.file, single);
    }
    for (int k = 0; k < n; ++k) {
      cpu.emplace_back([this, k] {
        while (auto t = tasks.pop()) run(*t, k);
      });
    }
    stage.emplace_back([this] { guarded([&] { orient_stage(); }); });
    stage.emplace_back([this] { guarded([&] { detect_stage(); }); });
    stage.emplace_back([this] { guarded([&] { recognize_stage(); }); });
  }

  ~Impl() {
    {
      std::lock_guard lk(m);
      stopping = true;
    }
    cv.notify_all();
    to_orient.close();
    to_detect.close();
    to_recognize.close();
    tasks.close();
    for (auto& t : stage) t.join();
    for (auto& t : cpu) t.join();
  }

  void check_recognizer(const Session& rec) const {
    const Shape& lines_in = rec.inputs().at(0).shape;  // [N, 1, height, W]
    if (lines_in.size() == 4 && lines_in[2] > 0 && lines_in[2] != e.crop.height) {
      throw Error(e.rec + " reads lines " + std::to_string(lines_in[2]) + " px high, but its crop rule makes them " +
                  std::to_string(e.crop.height));
    }
    const Shape& logits = rec.outputs().at(0).shape;
    if (logits.size() != 3 || (logits[1] > 0 && logits[1] != static_cast<int64_t>(vocab.size()))) {
      throw Error(e.rec + " gives " + shape_string(logits) + " but its vocabulary has " + std::to_string(vocab.size()) +
                  " classes");
    }
  }

  // ---- failures. A page's own (a run of it, or a step of it, met an error): that page comes out with it, next
  // throws it, and the stream goes on, as the CPU engine's (Engine::next). Anything else, outside a page's work,
  // ends the pipeline: the first such error, then every call's.
  void fail_page(Page& p, std::exception_ptr ep) {
    p.image = GrayImage{};
    p.small = GrayImage{};
    p.up = detail::Upright{};
    {
      std::lock_guard lk(m);
      if (ready.count(p.index) || failed.count(p.index)) return;
      failed.emplace(p.index, std::move(ep));
      ++delivered;
    }
    cv.notify_all();
  }
  void fail_pages(std::span<PagePtr> run, const std::exception_ptr& ep) {
    for (PagePtr& p : run) fail_page(*p, ep);
  }
  template <class F>
  void for_page(Page& p, F&& f) {
    try {
      f();
    } catch (...) {
      fail_page(p, std::current_exception());
    }
  }
  void fail(std::exception_ptr ep) {
    {
      std::lock_guard lk(m);
      if (!error) error = ep;
      stopping = true;
    }
    cv.notify_all();
    to_orient.close();
    to_detect.close();
    to_recognize.close();
  }
  template <class F>
  void guarded(F&& f) {
    try {
      f();
    } catch (...) {
      fail(std::current_exception());
    }
  }
  void run(std::function<void(int)>& t, int k) {
    guarded([&] { t(k); });
  }
  void cpu_task(std::function<void(int)> t) { tasks.push(std::move(t)); }

  // ---- the stages
  void prepare(Page& p) {
    p.start = Clock::now();
    p.out.width = p.image.width;
    p.out.height = p.image.height;
    p.small = detail::det_scale(p.image.view(), e.det_long_side);
    const int side = e.orient_side;
    p.thumb.resize(static_cast<size_t>(side) * side);
    detail::thumbnail_input(p.small.view(), p.image.width, p.image.height, side, p.thumb.data());
  }

  void orient_stage() {
    const int side = e.orient_side;
    std::vector<float> thumbs;
    while (auto first = to_orient.pop()) {
      std::vector<PagePtr> batch;
      batch.push_back(std::move(*first));
      while (static_cast<int>(batch.size()) < o.page_batch) {
        auto more = to_orient.try_pop();
        if (!more) break;
        batch.push_back(std::move(*more));
      }
      const size_t per = static_cast<size_t>(side) * side;
      thumbs.resize(batch.size() * per);
      for (size_t i = 0; i < batch.size(); ++i) {
        std::copy(batch[i]->thumb.begin(), batch[i]->thumb.end(), thumbs.begin() + i * per);
      }
      std::vector<float> z;
      try {
        z = models->orient(thumbs.data(), static_cast<int>(batch.size()), side);
      } catch (...) {  // the run's pages fail with it; the next run goes on, unless the device is broken
        fail_pages(batch, std::current_exception());
        if (models->broken()) fail(std::current_exception());
        continue;
      }
      const size_t classes = z.size() / batch.size();
      ++orient_runs;
      for (size_t i = 0; i < batch.size(); ++i) {
        PagePtr p = std::move(batch[i]);
        for_page(*p, [&] {
          p->out.rotation = detail::rotation_of(std::span<const float>(z.data() + i * classes, classes),
                                                *e.turn_probability);
          p->thumb = {};
          cpu_task([this, page = p](int) {
            for_page(*page, [&] {
              detail::make_upright(page->image.view(), page->out.rotation, std::move(page->small), e, page->up);
              page->out.skew = page->up.skew;
              const GrayView sv = page->up.small.view();
              std::tie(page->ph, page->pw) = models->detect_size(sv.width, sv.height);
              page->out.ms.orient = ms_since(page->start);
              to_detect.push(page);
            });
          });
        });
      }
    }
  }

  void detect_stage() {
    // The second detection pass's runs, here, the one thread that runs the detector;
    // none where the device does not run it.
    detail::DetectRun second;
    if (models->second_pass()) second = [this](const Tensor& x) { return models->detect_one(x); };
    std::deque<PagePtr> held;  // pages of another size than the batch's, for the next one
    for (;;) {
      PagePtr first;
      if (!held.empty()) {
        first = std::move(held.front());
        held.pop_front();
      } else if (auto p = to_detect.pop()) {
        first = std::move(*p);
      } else {
        return;
      }
      std::vector<PagePtr> batch;
      const int ph = first->ph, pw = first->pw;
      batch.push_back(std::move(first));
      for (auto it = held.begin(); it != held.end() && static_cast<int>(batch.size()) < o.page_batch;) {
        if ((*it)->ph == ph && (*it)->pw == pw) {
          batch.push_back(std::move(*it));
          it = held.erase(it);
        } else {
          ++it;
        }
      }
      while (static_cast<int>(batch.size()) < o.page_batch) {
        auto more = to_detect.try_pop();
        if (!more) break;
        if ((*more)->ph == ph && (*more)->pw == pw) {
          batch.push_back(std::move(*more));
        } else {
          held.push_back(std::move(*more));
        }
      }
      const auto t = Clock::now();
      std::vector<GrayView> smalls;
      for (const PagePtr& p : batch) smalls.push_back(p->up.small.view());
      std::vector<Detection> found;
      try {
        found = models->detect(smalls, ph, pw, e.det);
      } catch (...) {  // the run's pages fail with it; the next run goes on, unless the device is broken
        fail_pages(batch, std::current_exception());
        if (models->broken()) fail(std::current_exception());
        continue;
      }
      ++detect_runs;
      const double ms = ms_since(t) / batch.size();
      for (size_t i = 0; i < batch.size(); ++i) {
        PagePtr p = std::move(batch[i]);
        for_page(*p, [&] {  // its second pass among it: a page's own
          p->out.ms.detect = ms;
          const auto t2 = Clock::now();
          detail::place_detection(found[i], p->up.view, smalls[i].width, smalls[i].height, e, second, p->out);
          p->up.small = GrayImage{};
          p->out.ms.decode = ms_since(t2) - p->out.ms.second_pass;
          p->lines_start = Clock::now();
          if (p->out.lines.empty()) {
            cpu_task([this, p](int k) { for_page(*p, [&] { finish(*p, k); }); });
          } else {
            to_recognize.push(p);
          }
        });
      }
    }
  }

  // The rows a page's crops read: each line's crop region (crop_region) widened by the
  // resampler's support, twice over, and two rows more; runs closer than 16 rows joined. Rows past the page's are
  // the crop rule's paper, read from nowhere.
  static RowRuns rows_read(const std::vector<OcrLine>& lines, const CropRule& crop, int height) {
    RowRuns r;
    r.reserve(lines.size());
    for (const OcrLine& l : lines) {
      const CropRegion c = crop_region(l.box, crop);
      const double scale = std::max(1.0, (c.y1 - c.y0) / crop.height);
      const int m = static_cast<int>(std::ceil(2 * scale)) + 2;
      const int y0 = std::max(0, static_cast<int>(std::floor(c.y0)) - m);
      const int y1 = std::min(height, static_cast<int>(std::ceil(c.y1)) + m);
      if (y1 > y0) r.push_back({y0, y1});
    }
    std::sort(r.begin(), r.end());
    RowRuns merged;
    for (const auto& run : r) {
      if (!merged.empty() && run.first <= merged.back().second + 16) {
        merged.back().second = std::max(merged.back().second, run.second);
      } else {
        merged.push_back(run);
      }
    }
    return merged;
  }

  void recognize_stage() {
    const size_t cap = static_cast<size_t>(o.line_batch) * 4;  // lines gathered before the runs start
    // The next group, taken from the pages already waiting while this one is read and given to the models then
    // (prefetch_pages: a GPU uploads them while it reads this group's lines); it is not topped up after.
    std::vector<PagePtr> ahead;
    for (;;) {
      std::vector<PagePtr> group;
      size_t n = 0;
      if (!ahead.empty()) {
        group = std::move(ahead);
        ahead.clear();
        for (const PagePtr& p : group) n += p->out.lines.size();
      } else {
        auto first = to_recognize.pop();
        if (!first) break;
        n = (*first)->out.lines.size();
        group.push_back(std::move(*first));
        while (n < cap) {
          auto more = to_recognize.try_pop();
          if (!more) break;
          n += (*more)->out.lines.size();
          group.push_back(std::move(*more));
        }
      }
      // every line of the group by its width bucket, then its page
      struct Item {
        int bucket;
        int32_t page;
        int32_t line;
      };
      std::vector<Item> items;
      items.reserve(n);
      std::vector<GrayView> views;
      for (size_t g = 0; g < group.size(); ++g) {
        views.push_back(group[g]->up.view);
        const auto& ls = group[g]->out.lines;
        for (size_t i = 0; i < ls.size(); ++i) {
          items.push_back({width_bucket(crop_region(ls[i].box, e.crop).width), static_cast<int32_t>(g),
                           static_cast<int32_t>(i)});
        }
      }
      std::stable_sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.bucket < b.bucket; });
      // A run's error is every page's with a line in it; those pages fail once the group's runs are done (the
      // runs read the group's pages), and the rest of the group goes on.
      std::vector<std::exception_ptr> errors(group.size());
      std::vector<RowRuns> rows;
      for (const PagePtr& p : group) rows.push_back(rows_read(p->out.lines, e.crop, p->up.view.height));
      try {
        models->set_pages(views, rows);
      } catch (...) {
        fail_pages(group, std::current_exception());
        continue;
      }
      if (o.prefetch) {  // the next group from the pages waiting now, its upload started beside this group's runs
        size_t ahead_lines = 0;
        while (ahead_lines < cap) {
          auto more = to_recognize.try_pop();
          if (!more) break;
          ahead_lines += (*more)->out.lines.size();
          ahead.push_back(std::move(*more));
        }
        if (!ahead.empty()) {
          std::vector<GrayView> next;
          std::vector<RowRuns> next_rows;
          for (const PagePtr& p : ahead) {
            next.push_back(p->up.view);
            next_rows.push_back(rows_read(p->out.lines, e.crop, p->up.view.height));
          }
          try {
            models->prefetch_pages(next, next_rows);
          } catch (...) {  // its set_pages uploads them again, and meets the error then, if it is still there
          }
        }
      }
      std::vector<int32_t> page_of, lengths;
      std::vector<LineBox> boxes;
      std::vector<CtcLine> decoded;
      for (size_t a = 0; a < items.size();) {
        size_t b = a;
        while (b < items.size() && items[b].bucket == items[a].bucket && b - a < static_cast<size_t>(o.line_batch)) {
          ++b;
        }
        page_of.clear();
        boxes.clear();
        for (size_t i = a; i < b; ++i) {
          page_of.push_back(items[i].page);
          boxes.push_back(group[items[i].page]->out.lines[items[i].line].box);
        }
        try {
          models->recognize(page_of, boxes, items[a].bucket, e.crop, decoded, lengths);
          ++recognize_runs;
          lines += static_cast<int64_t>(b - a);
        } catch (...) {
          const std::exception_ptr ep = std::current_exception();
          for (size_t i = a; i < b; ++i) {
            if (!errors[items[i].page]) errors[items[i].page] = ep;
          }
          if (models->broken()) fail(ep);
          a = b;
          continue;
        }
        for (size_t i = a; i < b; ++i) {
          std::exception_ptr& err = errors[items[i].page];
          if (err) continue;
          try {
            detail::read_line(group[items[i].page]->out.lines[items[i].line], decoded[i - a], lengths[i - a],
                              e.crop, vocab, nullptr, 0, nullptr, e.char_confidence);
          } catch (...) {
            err = std::current_exception();
          }
        }
        a = b;
      }
      for (size_t g = 0; g < group.size(); ++g) {
        PagePtr& p = group[g];
        if (errors[g]) {
          fail_page(*p, errors[g]);
        } else {
          cpu_task([this, p](int k) { for_page(*p, [&] { finish(*p, k); }); });
        }
      }
    }
  }

  void finish(Page& p, int k) {
    OcrPage& out = p.out;
    out.ms.recognize = ms_since(p.lines_start);
    detail::finish_page(out, order[k].get(), e);  // the gate, the split and the order, as the CPU engine's
    out.ms.total = ms_since(p.start);
    ++pages;
    p.image = GrayImage{};
    p.up = detail::Upright{};
    {
      std::lock_guard lk(m);
      ready.emplace(p.index, std::move(out));
      ++delivered;
    }
    cv.notify_all();
  }

  // ---- the caller's side
  size_t submit(GrayImage image) {
    if (image.pixels.empty() || image.width <= 0 || image.height <= 0) throw Error("Pipeline::submit: an empty page");
    std::unique_lock lk(m);
    cv.wait(lk, [&] { return stopping || submitted - delivered < static_cast<size_t>(o.in_flight); });
    if (error) std::rethrow_exception(error);
    if (closed || stopping) throw Error("Pipeline::submit: the pipeline is closed");
    auto page = std::make_shared<Page>();
    page->index = submitted++;
    page->image = std::move(image);
    lk.unlock();
    const size_t index = page->index;
    cpu_task([this, page](int) {
      for_page(*page, [&] {
        prepare(*page);
        to_orient.push(page);
      });
    });
    return index;
  }

  void close() {
    {
      std::lock_guard lk(m);
      closed = true;
    }
    cv.notify_all();
  }

  std::optional<OcrPage> next() {
    std::unique_lock lk(m);
    if (by_finished) throw Error("Pipeline::next: this stream's pages are taken by next_finished");
    cv.wait(lk, [&] {
      return error || ready.count(taken) || failed.count(taken) || (closed && taken == submitted) || stopping;
    });
    if (error) std::rethrow_exception(error);
    if (const auto f = failed.find(taken); f != failed.end()) {  // this page's error; the next page goes on
      const std::exception_ptr ep = f->second;
      failed.erase(f);
      ++taken;
      std::rethrow_exception(ep);
    }
    auto it = ready.find(taken);
    if (it == ready.end()) return std::nullopt;
    OcrPage out = std::move(it->second);
    ready.erase(it);
    ++taken;
    return out;
  }

  // The first given among the pages done, as each finishes.
  std::optional<Pipeline::Finished> next_finished() {
    std::unique_lock lk(m);
    by_finished = true;
    cv.wait(lk, [&] {
      return error || !ready.empty() || !failed.empty() || (closed && taken == submitted) || stopping;
    });
    if (error) std::rethrow_exception(error);
    const size_t r = ready.empty() ? SIZE_MAX : ready.begin()->first;
    const size_t f = failed.empty() ? SIZE_MAX : failed.begin()->first;
    if (r == SIZE_MAX && f == SIZE_MAX) return std::nullopt;
    Pipeline::Finished out;
    if (f < r) {
      out.number = f;
      out.error = failed.begin()->second;
      failed.erase(failed.begin());
    } else {
      out.number = r;
      out.page = std::move(ready.begin()->second);
      ready.erase(ready.begin());
    }
    ++taken;
    return out;
  }
};

Pipeline::Pipeline(const std::filesystem::path& set, PipelineOptions options)
    : impl_(std::make_unique<Impl>(set, std::move(options))) {}

Pipeline::Pipeline(const std::filesystem::path& set, PipelineOptions options, std::unique_ptr<PipelineModels> models)
    : impl_(std::make_unique<Impl>(set, std::move(options), std::move(models))) {
  if (!impl_->models) throw Error("Pipeline: no models");
}

Pipeline::~Pipeline() = default;

size_t Pipeline::submit(GrayImage page) { return impl_->submit(std::move(page)); }

void Pipeline::close() { impl_->close(); }

std::optional<OcrPage> Pipeline::next() { return impl_->next(); }

std::optional<Pipeline::Finished> Pipeline::next_finished() { return impl_->next_finished(); }

std::vector<OcrPage> Pipeline::ocr(std::span<const GrayView> pages) {
  std::vector<OcrPage> out;
  out.reserve(pages.size());
  const size_t base = [&] {
    std::lock_guard lk(impl_->m);
    return impl_->submitted;
  }();
  if (base != impl_->taken) throw Error("Pipeline::ocr: pages given by submit are still in it");
  std::exception_ptr first_error;
  auto take = [&] {
    try {
      out.push_back(*impl_->next());
    } catch (...) {
      if (!first_error) first_error = std::current_exception();
      out.emplace_back();
    }
  };
  for (const GrayView& p : pages) {
    if (!p.data || p.width <= 0 || p.height <= 0) throw Error("Pipeline::ocr: an empty page");
    GrayImage copy{p.width, p.height, std::vector<uint8_t>(static_cast<size_t>(p.width) * p.height)};
    for (int y = 0; y < p.height; ++y) {
      std::copy(p.data + y * p.stride, p.data + y * p.stride + p.width,
                copy.pixels.data() + static_cast<size_t>(y) * p.width);
    }
    impl_->submit(std::move(copy));
    // take what is out as it comes, so the pages in flight stay few
    std::unique_lock lk(impl_->m);
    while (impl_->ready.count(impl_->taken) || impl_->failed.count(impl_->taken)) {
      lk.unlock();
      take();
      lk.lock();
    }
  }
  while (out.size() < pages.size()) take();
  if (first_error) std::rethrow_exception(first_error);  // every page through first: the pipeline goes on
  return out;
}

Device Pipeline::device() const { return impl_->models->device(); }

bool Pipeline::stopped() const {
  std::lock_guard lk(impl_->m);
  return impl_->stopping && impl_->error != nullptr;
}

PipelineStats Pipeline::stats() const {
  PipelineStats s;
  s.pages = impl_->pages;
  s.lines = impl_->lines;
  s.orient_runs = impl_->orient_runs;
  s.detect_runs = impl_->detect_runs;
  s.recognize_runs = impl_->recognize_runs;
  return s;
}

const CropRule& Pipeline::crop() const { return impl_->e.crop; }

std::filesystem::path gpu_set_model(const std::filesystem::path& set, const std::string& name, Precision precision) {
  auto file = [&](const char* kind) { return set / (name + kind + ".onnx"); };
  std::filesystem::path f;
  switch (precision) {
    case Precision::kFp32: f = file(""); break;
    case Precision::kFp16: f = file(".fp16"); break;
    case Precision::kInt8: f = file(".int8-sym"); break;
    case Precision::kAuto:
      f = file(".fp16");
      if (!std::filesystem::is_regular_file(f)) f = file("");
      break;
  }
  if (!std::filesystem::is_regular_file(f)) {
    throw Error(f.string() + ": the set has no " +
                (precision == Precision::kAuto ? std::string("fp16 or fp32") : std::string(precision_name(precision))) +
                " file for " + name + " on the GPU");
  }
  return f;
}

#if !(defined(CARBON_WITH_CUDA) && defined(CARBON_WITH_TENSORRT))
bool cuda_pipeline_available() { return false; }

std::unique_ptr<PipelineModels> cuda_pipeline_models(const std::filesystem::path&, const EngineOptions&,
                                                     const PipelineOptions&) {
  throw Error("the GPU pipeline needs CUDA and TensorRT, not built in");
}
#endif

}  // namespace carbon
