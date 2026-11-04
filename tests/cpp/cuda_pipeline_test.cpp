// carbon::Pipeline on a GPU (Device::kCuda: TensorRT, the cuda:: kernels) against
// the CPU device, on tests/data/fake-set, at every precision the GPU runs:
//   auto   orient.fp16.onnx and det.fp16.onnx, and rec-latin.onnx in float32 (the
//          fake set has no trunk and head files)
//   fp32   the .onnx files
//   int8   the .int8-sym.onnx files, built by TensorRT as INT8 networks
//   fp16   with fused_head off: the FP16 files, the recognizer whole in float32
// and fp16 with the fused head on refused, the fake set having no trunk and head.
// Skipped without a GPU.
#include <gtest/gtest.h>

#include <filesystem>
#include <vector>

#include "carbon/deskew.h"
#include "carbon/error.h"
#include "carbon/image.h"
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

TEST(CudaPipeline, ReadsPagesAsTheCpuDeviceDoesAtEveryPrecision) {
  if (!cuda_pipeline_available()) GTEST_SKIP() << "no GPU, or no TensorRT in this build";
  const GrayImage page = load_image(kSet / "page.png");
  const GrayImage skewed = rotate_page(page.view(), 2.5);
  const GrayImage blank{400, 300, std::vector<uint8_t>(400 * 300, 255)};
  std::vector<GrayView> pages;
  for (int i = 0; i < 12; ++i) pages.push_back(i % 3 == 0 ? page.view() : i % 3 == 1 ? blank.view() : skewed.view());
  PipelineOptions cpu;
  cpu.device = Device::kCpu;
  const std::vector<OcrPage> want = Pipeline(kSet, cpu).ocr(pages);
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
