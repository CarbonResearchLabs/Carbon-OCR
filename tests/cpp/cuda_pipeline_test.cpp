// carbon::Pipeline on a GPU (Device::kCuda: TensorRT, the cuda:: kernels) against
// the CPU device, on tests/data/fake-set (its orient.fp16.onnx and det.fp16.onnx,
// and rec-latin.onnx in float32: the fake set has no trunk and head files).
// Skipped without a GPU.
#include <gtest/gtest.h>

#include <filesystem>
#include <vector>

#include "carbon/deskew.h"
#include "carbon/image.h"
#include "carbon/pipeline.h"

namespace carbon {
namespace {

const std::filesystem::path kSet = std::filesystem::path(CARBON_TEST_DATA) / "fake-set";

TEST(CudaPipeline, ReadsPagesAsTheCpuDeviceDoes) {
  if (!cuda_pipeline_available()) GTEST_SKIP() << "no GPU, or no TensorRT in this build";
  const GrayImage page = load_image(kSet / "page.png");
  const GrayImage skewed = rotate_page(page.view(), 2.5);
  const GrayImage blank{400, 300, std::vector<uint8_t>(400 * 300, 255)};
  std::vector<GrayView> pages;
  for (int i = 0; i < 12; ++i) pages.push_back(i % 3 == 0 ? page.view() : i % 3 == 1 ? blank.view() : skewed.view());
  PipelineOptions cpu;
  cpu.device = Device::kCpu;
  PipelineOptions gpu;
  gpu.device = Device::kCuda;
  gpu.page_batch = 4;
  gpu.engine_cache = std::filesystem::temp_directory_path() / "carbon-test-engines";
  const std::vector<OcrPage> want = Pipeline(kSet, cpu).ocr(pages);
  Pipeline pipe(kSet, gpu);
  EXPECT_EQ(pipe.device(), Device::kCuda);
  const std::vector<OcrPage> got = pipe.ocr(pages);
  ASSERT_EQ(got.size(), want.size());
  for (size_t p = 0; p < got.size(); ++p) {
    EXPECT_EQ(got[p].rotation, want[p].rotation) << "page " << p;
    ASSERT_EQ(got[p].lines.size(), want[p].lines.size()) << "page " << p;
    for (size_t i = 0; i < got[p].lines.size(); ++i) {
      EXPECT_EQ(got[p].lines[i].text, want[p].lines[i].text) << "page " << p << " line " << i;
      // detection on a page padded to a fixed size in float16: the boxes within a pixel or two
      EXPECT_NEAR(got[p].lines[i].box.y0, want[p].lines[i].box.y0, 2.0);
      EXPECT_NEAR(got[p].lines[i].box.y1, want[p].lines[i].box.y1, 2.0);
    }
  }
  EXPECT_GE(pipe.stats().pages_per_detect_run(), 1.0);
}

}  // namespace
}  // namespace carbon
