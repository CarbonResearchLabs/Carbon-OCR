// Page images (carbon/image.h) and the kernels under them (carbon/cpu.h): every
// kernel's quarter turns against where each byte must land, on pages of sizes
// around the kernels' tiles (16 rows by 16 or 32 columns) and blocks (64 columns
// by 256 rows), rows longer than the page; and the ink map's values, each
// written once.

#include "carbon/image.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "carbon/cpu.h"

namespace carbon {
namespace {

// Where (x, y) of a w x h page lands, turned clockwise q times: its index in the
// packed output.
size_t landing(int x, int y, int w, int h, int q) {
  switch (q) {
    case 1: return static_cast<size_t>(x) * h + (h - 1 - y);
    case 2: return static_cast<size_t>(h - 1 - y) * w + (w - 1 - x);
    case 3: return static_cast<size_t>(w - 1 - x) * h + y;
    default: return static_cast<size_t>(y) * w + x;
  }
}

TEST(Image, EveryKernelTurnsByQuartersAsTheReference) {
  std::mt19937 rng(5);
  const std::pair<int, int> sizes[] = {{1, 1},   {1, 7},   {7, 1},   {2, 2},   {15, 17},  {16, 16},   {17, 15},
                                       {31, 33}, {32, 32}, {33, 64}, {48, 16}, {64, 65},  {100, 37},  {50, 150},
                                       {129, 257}, {120, 300}, {300, 70}};
  for (const auto& [w, h] : sizes) {
    const int stride = w + static_cast<int>(rng() % 5);
    std::vector<uint8_t> page(static_cast<size_t>(stride) * h);
    for (auto& v : page) v = static_cast<uint8_t>(rng());
    const GrayView view{page.data(), w, h, stride};
    for (int q = 0; q < 4; ++q) {
      std::vector<uint8_t> want(static_cast<size_t>(w) * h);
      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) want[landing(x, y, w, h, q)] = page[static_cast<size_t>(y) * stride + x];
      }
      for (cpu::Isa isa : cpu::isas_available()) {
        std::vector<uint8_t> got(want.size(), 0);
        cpu::rotate_quarters(page.data(), w, h, stride, q, got.data(), isa);
        EXPECT_EQ(got, want) << cpu::isa_name(isa) << " " << w << "x" << h << " q " << q;
      }
      const GrayImage turned = rotate_quarter(view, q);
      EXPECT_EQ(turned.width, q % 2 ? h : w);
      EXPECT_EQ(turned.height, q % 2 ? w : h);
      EXPECT_EQ(turned.pixels, want) << w << "x" << h << " q " << q;
      EXPECT_EQ(rotate_quarter(view, q - 4).pixels, want) << w << "x" << h << " q " << q - 4;
    }
  }
}

TEST(Image, InkInputIsOneMinusGrayOver255) {
  std::mt19937 rng(9);
  GrayImage page;
  page.width = 37;
  page.height = 23;
  page.pixels.resize(static_cast<size_t>(page.width) * page.height);
  for (auto& v : page.pixels) v = static_cast<uint8_t>(rng());
  page.pixels[0] = 0;
  page.pixels[1] = 255;
  const int out_w = 48, out_h = 30;
  std::vector<float> in(static_cast<size_t>(out_w) * out_h, NAN);
  ink_input(page.view(), page.width, page.height, out_w, out_h, in.data());
  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      const float want = x < page.width && y < page.height
                             ? 1.0f - page.pixels[static_cast<size_t>(y) * page.width + x] / 255.0f
                             : 0.0f;  // paper past the page
      EXPECT_EQ(in[static_cast<size_t>(y) * out_w + x], want) << x << "," << y;
    }
  }
  EXPECT_EQ(in[0], 1.0f);
  EXPECT_EQ(in[1], 0.0f);
}

}  // namespace
}  // namespace carbon
