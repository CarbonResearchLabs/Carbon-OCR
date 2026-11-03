// A page's small skew (carbon/deskew.h): found on a page of text-like bars turned
// by known angles, and the page turned back; every kernel of the turn against
// the reference.

#include "carbon/deskew.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

#include "carbon/cpu.h"

namespace carbon {
namespace {

// 16 rows of "words": dark bars 12 px high, of varied widths, on white.
GrayImage text_page() {
  GrayImage p;
  p.width = 900;
  p.height = 700;
  p.pixels.assign(static_cast<size_t>(p.width) * p.height, 250);
  uint32_t seed = 7;
  auto next = [&] { return (seed = seed * 1103515245u + 12345u) >> 16; };
  for (int row = 0; row < 16; ++row) {
    const int y0 = 60 + row * 36;
    for (int x = 80; x < 820;) {
      const int w = 20 + static_cast<int>(next() % 70);
      for (int y = y0; y < y0 + 12; ++y) {
        for (int xx = x; xx < std::min(820, x + w); ++xx) p.pixels[static_cast<size_t>(y) * p.width + xx] = 20;
      }
      x += w + 12;
    }
  }
  return p;
}

TEST(Deskew, FindsTheAngleThatLevelsAPage) {
  const GrayImage level = text_page();
  const Skew none = find_skew(level.view());
  EXPECT_NEAR(none.degrees, 0.0, 0.05);
  for (double angle : {3.0, -1.5, 0.7, -4.2}) {
    const GrayImage turned = rotate_page(level.view(), angle);
    const Skew s = find_skew(turned.view());
    EXPECT_NEAR(s.degrees, -angle, 0.1) << angle;
    EXPECT_GT(s.gain, 1.2) << angle;  // well over the engine's 1.05
    // turned back by it, level again
    const GrayImage back = rotate_page(turned.view(), s.degrees);
    EXPECT_NEAR(find_skew(back.view()).degrees, 0.0, 0.1) << angle;
  }
  // a blank page: nothing to level
  GrayImage blank;
  blank.width = 300;
  blank.height = 200;
  blank.pixels.assign(300 * 200, 255);
  const Skew b = find_skew(blank.view());
  EXPECT_EQ(b.degrees, 0.0);
  EXPECT_EQ(b.gain, 1.0);
}

TEST(Deskew, TurnsAPageAboutItsCentre) {
  const GrayImage page = text_page();
  const GrayImage same = rotate_page(page.view(), 0.0);
  EXPECT_EQ(same.pixels, page.pixels);
  // there and back: the inside of the page as it was, but for the resampling's blur
  const GrayImage there = rotate_page(page.view(), 2.0), back = rotate_page(there.view(), -2.0);
  int64_t diff = 0, n = 0;
  for (int y = 100; y < 600; ++y) {
    for (int x = 100; x < 800; ++x) {
      diff += std::abs(back.pixels[static_cast<size_t>(y) * 900 + x] - page.pixels[static_cast<size_t>(y) * 900 + x]);
      ++n;
    }
  }
  EXPECT_LT(static_cast<double>(diff) / n, 6.0);
  EXPECT_EQ(there.pixels[0], 255);  // a corner of the turned page: paper
}

// Every kernel of the turn (carbon::cpu::rotate_bilinear) gives the scalar
// reference's bytes: random pages of odd sizes, some a pixel or two wide, rows
// longer than the page, angles within 5 degrees either way, a few large ones, and
// turns with a zoom whose samples fall halfway between pixels (rounding ties).
TEST(Deskew, EveryKernelTurnsAsTheReference) {
  std::mt19937 rng(11);
  const std::pair<int, int> sizes[] = {{1, 1},  {2, 3},   {3, 2},   {7, 5},    {8, 2},     {9, 1},   {1, 9},
                                       {17, 13}, {16, 40}, {64, 64}, {123, 77}, {301, 199}, {640, 37}};
  const std::pair<double, double> zooms[] = {{0.0, 0.5}, {0.5, 0.0}, {0.25, -0.75}, {0.0, 2.0}, {0.125, 0.5}};
  std::uniform_real_distribution<double> small_angle(-5.0, 5.0);
  for (const auto& [w, h] : sizes) {
    const int stride = w + static_cast<int>(rng() % 9);
    std::vector<uint8_t> page(static_cast<size_t>(stride) * h);
    for (auto& v : page) v = rng() % 3 == 0 ? static_cast<uint8_t>(rng() % 2 * 255) : static_cast<uint8_t>(rng());
    std::vector<std::pair<double, double>> turns;  // sine, cosine
    std::vector<double> degrees = {0.0, 0.05, -0.05, 4.99, -4.99, 30.0, 90.0, -135.0, 180.0};
    for (int i = 0; i < 8; ++i) degrees.push_back(small_angle(rng));
    for (double d : degrees) {
      const double a = d * 3.14159265358979323846 / 180.0;
      turns.emplace_back(std::sin(a), std::cos(a));
    }
    for (const auto& zoom : zooms) turns.push_back(zoom);
    for (const auto& [s, c] : turns) {
      std::vector<uint8_t> want(static_cast<size_t>(w) * h, 255);
      cpu::rotate_bilinear(page.data(), w, h, stride, s, c, want.data(), cpu::Isa::kScalar);
      for (cpu::Isa isa : cpu::isas_available()) {
        std::vector<uint8_t> got(static_cast<size_t>(w) * h, 255);
        cpu::rotate_bilinear(page.data(), w, h, stride, s, c, got.data(), isa);
        EXPECT_EQ(got, want) << cpu::isa_name(isa) << " " << w << "x" << h << " sin " << s << " cos " << c;
      }
    }
  }
  // rotate_page itself on a page of text, the reference's bytes too
  const GrayImage text = text_page();
  for (double d : {2.5, -1.25, 4.4}) {
    const double a = d * 3.14159265358979323846 / 180.0;
    std::vector<uint8_t> want(text.pixels.size(), 255);
    cpu::rotate_bilinear(text.pixels.data(), text.width, text.height, text.width, std::sin(a), std::cos(a),
                         want.data(), cpu::Isa::kScalar);
    EXPECT_EQ(rotate_page(text.view(), d).pixels, want) << d;
  }
}

}  // namespace
}  // namespace carbon
