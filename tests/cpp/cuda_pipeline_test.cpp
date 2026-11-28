// carbon::Pipeline on a GPU (Device::kCuda: TensorRT, the cuda:: kernels) against
// the CPU device, on tests/data/fake-set, at every precision the GPU runs:
//   auto   orient.fp16.onnx and det.fp16.onnx, and rec-latin.onnx in float32 (the
//          fake set has no trunk and head files)
//   fp32   the .onnx files
//   int8   the .int8-sym.onnx files, built by TensorRT as INT8 networks
//   fp16   with fused_head off: the FP16 files, the recognizer whole in float32
// and fp16 with the fused head on refused, the fake set having no trunk and head.
// A page of tiny text among them takes the second detection pass on both devices, and a
// page's line wider than every recognizer bucket is read as the CPU device reads it.
// Skipped without a GPU.
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "carbon/deskew.h"
#include "carbon/document.h"
#include "carbon/engine.h"
#include "carbon/error.h"
#include "carbon/image.h"
#include "carbon/lines.h"
#include "carbon/lm.h"
#include "carbon/pipeline.h"
#include "random_tlm.h"

namespace carbon {
namespace {

const std::filesystem::path kSet = std::filesystem::path(CARBON_TEST_DATA) / "fake-set";

PipelineOptions gpu_options(Precision precision, bool fused_head) {
  PipelineOptions gpu;
  gpu.device = Device::kCuda;
  gpu.page_batch = 4;
  gpu.engine.precision = precision;
  gpu.fused_head = fused_head;
  gpu.engine_cache = std::filesystem::temp_directory_path() / "carbon-test-engines";
  return gpu;
}

// A page of tiny text at the detector's scale: bars 5 px high, every other one starting on an odd row, which
// the fake detector finds only at twice the scale (batch_pipeline_test.cpp's tiny_page).
GrayImage tiny_page() {
  GrayImage p{800, 1024, std::vector<uint8_t>(800 * 1024, 255)};
  const int dashes[] = {3, 4, 2, 3, 4, 2};
  for (int i = 0; i < 6; ++i) {
    const int y = 200 + 41 * i;
    for (int d = 0; d < dashes[i]; ++d) {
      for (int r = y; r < y + 5; ++r) std::fill_n(p.pixels.begin() + r * 800 + 100 + 32 * d, 24, uint8_t{20});
    }
  }
  return p;
}

// A page with one line far wider than it is high (a bar 760 px long and 6 px high): its crop is wider than the
// GPU recognizer's widest bucket, 2048 px, so the GPU gives it to the CPU device's recognizer.
GrayImage wide_line_page() {
  GrayImage p{800, 1024, std::vector<uint8_t>(800 * 1024, 255)};
  for (int r = 500; r < 506; ++r) std::fill_n(p.pixels.begin() + r * 800 + 20, 760, uint8_t{20});
  return p;
}

TEST(CudaPipeline, ReadsPagesAsTheCpuDeviceDoesAtEveryPrecision) {
  if (!cuda_pipeline_available()) GTEST_SKIP() << "no GPU, or no TensorRT in this build";
  const GrayImage page = load_image(kSet / "page.png");
  const GrayImage skewed = rotate_page(page.view(), 2.5);
  const GrayImage blank{400, 300, std::vector<uint8_t>(400 * 300, 255)};
  const GrayImage tiny = tiny_page();
  const GrayImage wide = wide_line_page();
  const GrayView kinds[] = {page.view(), blank.view(), skewed.view(), tiny.view(), wide.view()};
  std::vector<GrayView> pages;
  for (int i = 0; i < 15; ++i) pages.push_back(kinds[i % 5]);
  PipelineOptions cpu;
  cpu.device = Device::kCpu;
  const std::vector<OcrPage> want = Pipeline(kSet, cpu).ocr(pages);
  ASSERT_TRUE(want[3].second_pass);
  ASSERT_EQ(want[3].lines.size(), 6u);
  ASSERT_EQ(want[4].lines.size(), 1u);
  {
    const Pipeline crop_of(kSet, cpu);
    ASSERT_GT(width_bucket(crop_region(want[4].lines[0].box, crop_of.crop()).width), 2048);
  }
  struct Case {
    Precision precision;
    bool fused_head;
  };
  for (const Case c : {Case{Precision::kAuto, true}, Case{Precision::kFp32, true}, Case{Precision::kInt8, true},
                       Case{Precision::kFp16, false}}) {
    SCOPED_TRACE(std::string(precision_name(c.precision)) + (c.fused_head ? "" : ", fused head off"));
    Pipeline pipe(kSet, gpu_options(c.precision, c.fused_head));
    EXPECT_EQ(pipe.device(), Device::kCuda);
    const std::vector<OcrPage> got = pipe.ocr(pages);
    ASSERT_EQ(got.size(), want.size());
    for (size_t p = 0; p < got.size(); ++p) {
      EXPECT_EQ(got[p].rotation, want[p].rotation) << "page " << p;
      EXPECT_EQ(got[p].second_pass, want[p].second_pass) << "page " << p;
      ASSERT_EQ(got[p].lines.size(), want[p].lines.size()) << "page " << p;
      for (size_t i = 0; i < got[p].lines.size(); ++i) {
        EXPECT_EQ(got[p].lines[i].text, want[p].lines[i].text) << "page " << p << " line " << i;
        // detection on a page padded to a fixed size, in float16 or int8: the boxes within a pixel or two
        EXPECT_NEAR(got[p].lines[i].box.y0, want[p].lines[i].box.y0, 2.0);
        EXPECT_NEAR(got[p].lines[i].box.y1, want[p].lines[i].box.y1, 2.0);
      }
    }
    EXPECT_GE(pipe.stats().pages_per_detect_run(), 1.0);
  }
}

// Counts its calls from any thread; every class alike (ln 0.2).
class Counting : public LanguageModel {
 public:
  uint64_t start() const override { return 1; }
  float score(uint64_t state, int32_t c, uint64_t& next) const override {
    calls.fetch_add(1);
    next = state * 1315423911u + static_cast<uint64_t>(c);
    return -1.609438f;
  }
  mutable std::atomic<int64_t> calls{0};
};

// The beam (EngineOptions::beam) on every line, its candidates from the GPU (cuda::ctc_topk; the wide line's from
// the CPU device it is read on): the lines the CPU device's beam reads, a language model asked on both.
TEST(CudaPipeline, ReadsUnsureLinesByTheBeamAsTheCpuDeviceDoes) {
  if (!cuda_pipeline_available()) GTEST_SKIP() << "no GPU, or no TensorRT in this build";
  const GrayImage page = load_image(kSet / "page.png");
  const GrayImage tiny = tiny_page();
  const GrayImage wide = wide_line_page();
  const GrayView kinds[] = {page.view(), tiny.view(), wide.view()};
  std::vector<GrayView> pages;
  for (int i = 0; i < 9; ++i) pages.push_back(kinds[i % 3]);
  auto beamed = [](PipelineOptions o, std::shared_ptr<const LanguageModel> lm) {
    o.engine.beam.width = 8;
    o.engine.beam.gate = 2.0f;  // every line
    o.engine.beam.alpha = 0.2f;
    o.engine.language_model = std::move(lm);
    return o;
  };
  PipelineOptions cpu;
  cpu.device = Device::kCpu;
  auto lm_cpu = std::make_shared<Counting>();
  const std::vector<OcrPage> want = Pipeline(kSet, beamed(cpu, lm_cpu)).ocr(pages);
  auto lm_gpu = std::make_shared<Counting>();
  Pipeline pipe(kSet, beamed(gpu_options(Precision::kFp32, true), lm_gpu));
  const std::vector<OcrPage> got = pipe.ocr(pages);
  ASSERT_EQ(got.size(), want.size());
  for (size_t p = 0; p < got.size(); ++p) {
    ASSERT_EQ(got[p].lines.size(), want[p].lines.size()) << "page " << p;
    for (size_t i = 0; i < got[p].lines.size(); ++i) {
      EXPECT_EQ(got[p].lines[i].text, want[p].lines[i].text) << "page " << p << " line " << i;
      EXPECT_NEAR(got[p].lines[i].confidence, want[p].lines[i].confidence, 1e-3);
    }
  }
  EXPECT_GT(lm_cpu->calls.load(), 0);
  EXPECT_GT(lm_gpu->calls.load(), 0);
}

// The transformer's rescoring (EngineOptions::tlm) on the pipeline's GPU, as it is by default where the pipeline
// runs on one (cuda_transformer_lm), and on the CPU (tlm_gpu -1): the same lines, every one of them gated.
TEST(CudaPipeline, RescoresWithTheTransformerOnItsGpuAsOnTheCpu) {
  if (!cuda_pipeline_available()) GTEST_SKIP() << "no GPU, or no TensorRT in this build";
  const std::vector<std::string> vocab = load_vocab(kSet / "rec-latin.vocab.json");
  const auto tlm = random_model("pipeline", 64, 2, 2, 128, 64, static_cast<int>(vocab.size()), 9u, vocab_hash(vocab));
  const GrayImage page = load_image(kSet / "page.png");
  const GrayImage tiny = tiny_page();
  const std::vector<GrayView> pages = {page.view(), tiny.view(), page.view(), tiny.view()};
  auto rescored = [&](PipelineOptions o, std::optional<int> gpu) {
    o.engine.beam.width = 8;
    o.engine.beam.gate = 2.0f;  // every line
    o.engine.tlm = tlm;
    o.engine.tlm_alpha = 1.0f;
    o.engine.tlm_gpu = gpu;
    return o;
  };
  const std::vector<OcrPage> on_cpu = Pipeline(kSet, rescored(gpu_options(Precision::kFp32, true), -1)).ocr(pages);
  const std::vector<OcrPage> on_gpu =
      Pipeline(kSet, rescored(gpu_options(Precision::kFp32, true), std::nullopt)).ocr(pages);
  ASSERT_EQ(on_gpu.size(), on_cpu.size());
  for (size_t p = 0; p < on_gpu.size(); ++p) {
    ASSERT_EQ(on_gpu[p].lines.size(), on_cpu[p].lines.size()) << "page " << p;
    for (size_t i = 0; i < on_gpu[p].lines.size(); ++i) {
      EXPECT_EQ(on_gpu[p].lines[i].text, on_cpu[p].lines[i].text) << "page " << p << " line " << i;
    }
  }
}

TEST(CudaPipeline, Fp16NeedsTheTrunkAndFusedHead) {
  if (!cuda_pipeline_available()) GTEST_SKIP() << "no GPU, or no TensorRT in this build";
  EXPECT_THROW(Pipeline(kSet, gpu_options(Precision::kFp16, true)), Error);
}

// On the GPU: a page's lines found alone, read from them, read as given and ordered alone, as the CPU device
// does them (in FP32, the same lines); and a text layer's page through the document stream with its steps on the
// GPU's pipeline, its picture's bar read there.
TEST(CudaPipeline, ReadsAPagesTasksAsTheCpuDeviceDoes) {
  if (!cuda_pipeline_available()) GTEST_SKIP() << "no GPU, or no TensorRT in this build";
  const GrayImage page = load_image(kSet / "page.png");
  const GrayImage skewed = rotate_page(page.view(), 2.5);
  PipelineOptions cpu = gpu_options(Precision::kFp32, true);
  cpu.device = Device::kCpu;
  std::vector<OcrPage> want, got;
  for (PipelineOptions po : {cpu, gpu_options(Precision::kFp32, true)}) {
    Pipeline pipe(kSet, po);
    pipe.submit_detect(GrayImage(page));
    PageTask as_given;
    as_given.as_given = true;
    pipe.submit(GrayImage(skewed), as_given);
    pipe.close();
    std::vector<OcrPage>& out = po.device == Device::kCpu ? want : got;
    while (auto p = pipe.next()) out.push_back(std::move(*p));
  }
  ASSERT_EQ(want.size(), 2u);
  ASSERT_EQ(got.size(), 2u);
  ASSERT_EQ(got[0].lines.size(), want[0].lines.size());
  for (size_t i = 0; i < got[0].lines.size(); ++i) {
    EXPECT_TRUE(got[0].lines[i].text.empty());
    EXPECT_NEAR(got[0].lines[i].box.y0, want[0].lines[i].box.y0, 1.0);
  }
  ASSERT_EQ(got[1].lines.size(), want[1].lines.size());
  EXPECT_EQ(got[1].skew, 0.0);
  for (size_t i = 0; i < got[1].lines.size(); ++i) EXPECT_EQ(got[1].lines[i].text, want[1].lines[i].text);
  {  // from the lines found, on the GPU
    Pipeline pipe(kSet, gpu_options(Precision::kFp32, true));
    PageTask from;
    from.found = std::make_shared<const OcrPage>(want[0]);
    pipe.submit(GrayImage(page), from);
    OcrPage text_page = want[0];
    text_page.lines[0].text = "first";
    pipe.submit_order(text_page);
    pipe.close();
    const OcrPage read = *pipe.next();
    EXPECT_EQ(read.text(), "aaa\naaaaa\naa\n");
    EXPECT_EQ(pipe.next()->lines.size(), text_page.lines.size());
  }
}

#if defined(CARBON_PDF_HELPER)
TEST(CudaPipeline, ReadsATextLayersPageOnTheGpu) {
  if (!cuda_pipeline_available()) GTEST_SKIP() << "no GPU, or no TensorRT in this build";
  const std::filesystem::path data = CARBON_TEST_DATA;
  for (const bool on_gpu : {true, false}) {
    DocumentOptions d;
    d.cuda = true;
    d.pipeline = gpu_options(Precision::kFp32, true);
    d.text_on_gpu = on_gpu;
    DocumentStream stream(kSet, d);
    PdfPool pool(1, CARBON_PDF_HELPER);
    stream.pdf_page(pool.page(data / "figure.pdf", 0, 200 / 72.0, stream.pdf_want()), 0);
    stream.pdf_page(pool.page(data / "text.pdf", 0, 200 / 72.0, stream.pdf_want()), 1);
    stream.close();
    std::vector<OcrPage> out(2);
    while (auto done = stream.next()) {
      ASSERT_FALSE(done->error) << on_gpu;
      EXPECT_TRUE(done->from_text_layer);
      out[done->tag] = std::move(done->page);
    }
    EXPECT_EQ(out[0].source, "hybrid") << on_gpu;
    EXPECT_NE(out[0].text().find("aaa"), std::string::npos) << out[0].text();
    EXPECT_EQ(stream.workers() > 0, !on_gpu) << "a CPU engine only for the text pages off the GPU";
    EXPECT_EQ(out[1].source, "text layer");
  }
  {  // On the GPU: a layer that runs up the page found and read upright, its rotation the turn
    DocumentOptions d;
    d.cuda = true;
    d.pipeline = gpu_options(Precision::kFp32, true);
    DocumentStream stream(kSet, d);
    PdfPool pool(1, CARBON_PDF_HELPER);
    stream.pdf_page(pool.page(data / "turned.pdf", 0, 200 / 72.0, stream.pdf_want()), 0);
    stream.close();
    const auto done = stream.next();
    ASSERT_TRUE(done.has_value());
    ASSERT_FALSE(done->error);
    EXPECT_EQ(done->page.rotation, 270);
    EXPECT_EQ(done->page.source, "text layer");
  }
}
#endif

}  // namespace
}  // namespace carbon
