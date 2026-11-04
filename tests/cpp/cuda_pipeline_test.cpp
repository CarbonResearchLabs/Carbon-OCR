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
#include <filesystem>
#include <vector>

#include "carbon/deskew.h"
#include "carbon/error.h"
#include "carbon/image.h"
#include "carbon/lines.h"
#include "carbon/pipeline.h"

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

TEST(CudaPipeline, Fp16NeedsTheTrunkAndFusedHead) {
  if (!cuda_pipeline_available()) GTEST_SKIP() << "no GPU, or no TensorRT in this build";
  EXPECT_THROW(Pipeline(kSet, gpu_options(Precision::kFp16, true)), Error);
}

}  // namespace
}  // namespace carbon
