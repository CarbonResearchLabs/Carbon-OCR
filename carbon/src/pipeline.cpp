// The batched OCR pipeline (carbon/pipeline.h): its stages, queues and the CPU
// device's models.  The GPU's are in cuda/pipeline_cuda.cpp.

#include "carbon/pipeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
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
    orient_ = open_set_model(set, "orient", o.backend, so);
    det_ = open_set_model(set, "det", o.backend, so);
    rec_ = open_set_model(set, o.rec, o.backend, so);
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

  void set_pages(std::span<const GrayView> pages) override { pages_.assign(pages.begin(), pages.end()); }

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
  std::exception_ptr error;
  std::map<size_t, OcrPage> ready;

  std::atomic<int64_t> pages{0}, lines{0}, orient_runs{0}, detect_runs{0}, recognize_runs{0};

  Impl(const std::filesystem::path& set, PipelineOptions options) : o(std::move(options)), e(o.engine) {
    if (o.page_batch < 1 || o.line_batch < 1 || o.in_flight < 1) {
      throw Error("Pipeline: page_batch, line_batch and in_flight are at least 1");
    }
    vocab = load_vocab(set / (e.rec + ".vocab.json"));
    e.crop = detail::model_crop(set, e.rec, e.crop);
    if (!e.turn_probability) e.turn_probability = detail::model_turn_probability(set);
    const int cores = cpu_topology().default_workers();
    const Device device = o.device.value_or(cuda_pipeline_available() ? Device::kCuda : Device::kCpu);
    models = device == Device::kCuda ? cuda_pipeline_models(set, e, o)
                                     : cpu_pipeline_models(set, e, o.model_threads > 0 ? o.model_threads : cores);
    if (device == Device::kCpu) check_recognizer(static_cast<const CpuModels&>(*models).rec());
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

  // ---- failures: the first stage's error ends the pipeline
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
      const std::vector<float> z = models->orient(thumbs.data(), static_cast<int>(batch.size()), side);
      const size_t classes = z.size() / batch.size();
      ++orient_runs;
      for (size_t i = 0; i < batch.size(); ++i) {
        PagePtr p = std::move(batch[i]);
        p->out.rotation = detail::rotation_of(std::span<const float>(z.data() + i * classes, classes),
                                              *e.turn_probability);
        p->thumb = {};
        cpu_task([this, page = std::move(p)](int) {
          detail::make_upright(page->image.view(), page->out.rotation, std::move(page->small), e, page->up);
          page->out.skew = page->up.skew;
          const GrayView sv = page->up.small.view();
          std::tie(page->ph, page->pw) = models->detect_size(sv.width, sv.height);
          page->out.ms.orient = ms_since(page->start);
          to_detect.push(page);
        });
      }
    }
  }

  void detect_stage() {
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
      const std::vector<Detection> found = models->detect(smalls, ph, pw, e.det);
      ++detect_runs;
      const double ms = ms_since(t) / batch.size();
      for (size_t i = 0; i < batch.size(); ++i) {
        PagePtr p = std::move(batch[i]);
        p->out.ms.detect = ms;
        const auto t2 = Clock::now();
        detail::place_detection(found[i], p->up.view.width, p->up.view.height, smalls[i].width, smalls[i].height,
                                p->out);
        p->up.small = GrayImage{};
        p->out.ms.decode = ms_since(t2);
        p->lines_start = Clock::now();
        if (p->out.lines.empty()) {
          cpu_task([this, p](int k) { finish(*p, k); });
        } else {
          to_recognize.push(std::move(p));
        }
      }
    }
  }

  void recognize_stage() {
    const size_t cap = static_cast<size_t>(o.line_batch) * 4;  // lines gathered before the runs start
    while (auto first = to_recognize.pop()) {
      std::vector<PagePtr> group;
      size_t n = (*first)->out.lines.size();
      group.push_back(std::move(*first));
      while (n < cap) {
        auto more = to_recognize.try_pop();
        if (!more) break;
        n += (*more)->out.lines.size();
        group.push_back(std::move(*more));
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
      models->set_pages(views);
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
        models->recognize(page_of, boxes, items[a].bucket, e.crop, decoded, lengths);
        ++recognize_runs;
        lines += static_cast<int64_t>(b - a);
        for (size_t i = a; i < b; ++i) {
          detail::read_line(group[items[i].page]->out.lines[items[i].line], decoded[i - a], lengths[i - a], e.crop,
                            vocab);
        }
        a = b;
      }
      for (PagePtr& p : group) cpu_task([this, p](int k) { finish(*p, k); });
    }
  }

  void finish(Page& p, int k) {
    OcrPage& out = p.out;
    out.ms.recognize = ms_since(p.lines_start);
    if (e.split) out.lines = split_lines(std::move(out.lines), *e.split);
    detail::order_page(out, order[k].get(), e);
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
      prepare(*page);
      to_orient.push(page);
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
    cv.wait(lk, [&] { return error || ready.count(taken) || (closed && taken == submitted) || stopping; });
    if (error) std::rethrow_exception(error);
    auto it = ready.find(taken);
    if (it == ready.end()) return std::nullopt;
    OcrPage out = std::move(it->second);
    ready.erase(it);
    ++taken;
    return out;
  }
};

Pipeline::Pipeline(const std::filesystem::path& set, PipelineOptions options)
    : impl_(std::make_unique<Impl>(set, std::move(options))) {}

Pipeline::~Pipeline() = default;

size_t Pipeline::submit(GrayImage page) { return impl_->submit(std::move(page)); }

void Pipeline::close() { impl_->close(); }

std::optional<OcrPage> Pipeline::next() { return impl_->next(); }

std::vector<OcrPage> Pipeline::ocr(std::span<const GrayView> pages) {
  std::vector<OcrPage> out;
  out.reserve(pages.size());
  const size_t base = [&] {
    std::lock_guard lk(impl_->m);
    return impl_->submitted;
  }();
  if (base != impl_->taken) throw Error("Pipeline::ocr: pages given by submit are still in it");
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
    while (impl_->ready.count(impl_->taken)) {
      lk.unlock();
      out.push_back(*impl_->next());
      lk.lock();
    }
  }
  while (out.size() < pages.size()) out.push_back(*impl_->next());
  return out;
}

Device Pipeline::device() const { return impl_->models->device(); }

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

#if !(defined(CARBON_WITH_CUDA) && defined(CARBON_WITH_TENSORRT))
bool cuda_pipeline_available() { return false; }

std::unique_ptr<PipelineModels> cuda_pipeline_models(const std::filesystem::path&, const EngineOptions&,
                                                     const PipelineOptions&) {
  throw Error("the GPU pipeline needs CUDA and TensorRT, not built in");
}
#endif

}  // namespace carbon
