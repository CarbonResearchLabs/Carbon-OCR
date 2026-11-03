// A page's small skew (carbon/deskew.h): found on a page of text-like bars turned
// by known angles, and the page turned back; every kernel of the turn against
// the reference.

#include "carbon/deskew.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
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

// find_skew as it was before it was rewritten for speed (one histogram, a division a sample), the reference its
// rewrite must give exactly: the same counts at every angle, so the same degrees and gain.
constexpr double kPi = 3.14159265358979323846;
// Otsu's threshold of the page's histogram: the gray level that splits it best
// into ink and paper.
int reference_otsu(const GrayView& page) {
  std::array<int64_t, 256> hist{};
  for (int y = 0; y < page.height; ++y) {
    const uint8_t* row = page.data + y * page.stride;
    for (int x = 0; x < page.width; ++x) ++hist[row[x]];
  }
  const int64_t total = static_cast<int64_t>(page.width) * page.height;
  double sum = 0;
  for (int i = 0; i < 256; ++i) sum += static_cast<double>(i) * hist[i];
  double sum_b = 0, best = -1;
  int64_t w_b = 0;
  int t = 128;
  for (int i = 0; i < 256; ++i) {
    w_b += hist[i];
    if (w_b == 0) continue;
    const int64_t w_f = total - w_b;
    if (w_f == 0) break;
    sum_b += static_cast<double>(i) * hist[i];
    const double m_b = sum_b / w_b, m_f = (sum - sum_b) / w_f;
    const double between = static_cast<double>(w_b) * w_f * (m_b - m_f) * (m_b - m_f);
    if (between > best) {
      best = between;
      t = i;
    }
  }
  return t;
}


Skew reference_find_skew(const GrayView& page, double max_degrees) {
  Skew out;
  if (page.width < 16 || page.height < 16) return out;
  const int t = reference_otsu(page);
  // the ink, a sample of at most 60000 pixels spread over the page
  int64_t ink = 0;
  for (int y = 0; y < page.height; ++y) {
    const uint8_t* row = page.data + y * page.stride;
    for (int x = 0; x < page.width; ++x) ink += row[x] <= t;
  }
  if (ink < 64 || ink > static_cast<int64_t>(page.width) * page.height / 2) return out;  // blank, or not text
  const int64_t step = std::max<int64_t>(1, ink / 60000);
  std::vector<float> xs, ys;
  int64_t k = 0;
  const double cx = page.width / 2.0, cy = page.height / 2.0;
  for (int y = 0; y < page.height; ++y) {
    const uint8_t* row = page.data + y * page.stride;
    for (int x = 0; x < page.width; ++x) {
      if (row[x] <= t && k++ % step == 0) {
        xs.push_back(static_cast<float>(x - cx));
        ys.push_back(static_cast<float>(y - cy));
      }
    }
  }
  // a score for each angle: the sum of squared counts of the rows the points fall
  // in, the page turned by it
  const int rows = static_cast<int>(std::ceil(std::hypot(page.width, page.height))) + 2;
  std::vector<int32_t> count(rows);
  auto score = [&](double deg) {
    const double a = deg * kPi / 180.0, s = std::sin(a), c = std::cos(a);
    std::fill(count.begin(), count.end(), 0);
    for (size_t i = 0; i < xs.size(); ++i) {
      // the row the point is on after the page turns `deg` counterclockwise
      const double r = ys[i] * c - xs[i] * s + rows / 2.0;
      ++count[std::clamp(static_cast<int>(r), 0, rows - 1)];
    }
    double sum = 0;
    for (int32_t n : count) sum += static_cast<double>(n) * n;
    return sum;
  };
  const double level = score(0);
  double best = 0, best_score = level;
  for (double d = -max_degrees; d <= max_degrees + 1e-9; d += 0.5) {
    const double v = score(d);
    if (v > best_score) {
      best_score = v;
      best = d;
    }
  }
  const double coarse = best;
  for (double d = coarse - 0.5; d <= coarse + 0.5 + 1e-9; d += 0.05) {
    if (std::abs(d) > max_degrees + 1e-9) continue;
    const double v = score(d);
    if (v > best_score) {
      best_score = v;
      best = d;
    }
  }
  out.degrees = std::round(best * 100) / 100;
  out.gain = level > 0 ? best_score / level : 1;
  return out;
}

TEST(Deskew, TheSearchGivesTheReferencesDegreesAndGainExactly) {
  std::mt19937 rng(17);
  const GrayImage level = text_page();
  std::vector<GrayImage> pages;
  for (double a = -5.2; a <= 5.2; a += 0.37) pages.push_back(rotate_page(level.view(), a));
  // denser ink (more than 60,000 pixels, so the sample takes every step-th) and odd sizes
  GrayImage dense;
  dense.width = 1333;
  dense.height = 1001;
  dense.pixels.assign(static_cast<size_t>(dense.width) * dense.height, 240);
  for (int y = 0; y < dense.height; ++y) {
    for (int x = 0; x < dense.width; ++x) {
      if ((y / 9) % 3 == 0 && (x / 23) % 4 != 3) dense.pixels[static_cast<size_t>(y) * dense.width + x] = 30;
    }
  }
  for (double a : {0.0, 1.3, -2.7, 4.05}) pages.push_back(rotate_page(dense.view(), a));
  GrayImage noise;
  noise.width = 257;
  noise.height = 129;
  noise.pixels.resize(static_cast<size_t>(noise.width) * noise.height);
  for (auto& v : noise.pixels) v = static_cast<uint8_t>(rng() % 3 == 0 ? rng() % 120 : 200 + rng() % 56);
  pages.push_back(noise);
  for (const GrayImage& p : pages) {
    for (double max : {5.0, 2.0}) {
      const Skew want = reference_find_skew(p.view(), max), got = find_skew(p.view(), max);
      EXPECT_EQ(got.degrees, want.degrees) << p.width << "x" << p.height << " max " << max;
      EXPECT_EQ(got.gain, want.gain) << p.width << "x" << p.height << " max " << max;
    }
  }
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
        // in row bands, as idle workers turn a page alone: the same bytes however the rows are cut
        std::vector<uint8_t> bands(static_cast<size_t>(w) * h, 255);
        const int cuts[] = {0, h / 3, h / 3, h / 2 + 1, h};
        for (int b = 0; b + 1 < 5; ++b) {
          cpu::rotate_bilinear_rows(page.data(), w, h, stride, s, c, bands.data(), cuts[b], cuts[b + 1], isa);
        }
        EXPECT_EQ(bands, want) << cpu::isa_name(isa) << " in bands, " << w << "x" << h;
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
