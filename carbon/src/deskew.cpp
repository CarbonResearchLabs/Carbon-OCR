// A page's small skew and the page levelled (carbon/deskew.h).

#include "carbon/deskew.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include "carbon/cpu.h"

namespace carbon {
namespace {

constexpr double kPi = 3.14159265358979323846;

// The page's histogram, counted in four interleaved histograms so that a run of
// one gray level (paper) is not one long chain of increments to one counter.
std::array<int64_t, 256> histogram(const GrayView& page) {
  std::array<std::array<int32_t, 256>, 4> part{};
  for (int y = 0; y < page.height; ++y) {
    const uint8_t* row = page.data + static_cast<int64_t>(y) * page.stride;
    int x = 0;
    for (; x + 4 <= page.width; x += 4) {
      ++part[0][row[x]];
      ++part[1][row[x + 1]];
      ++part[2][row[x + 2]];
      ++part[3][row[x + 3]];
    }
    for (; x < page.width; ++x) ++part[0][row[x]];
  }
  std::array<int64_t, 256> hist{};
  for (int i = 0; i < 256; ++i) hist[i] = int64_t{part[0][i]} + part[1][i] + part[2][i] + part[3][i];
  return hist;
}

// Otsu's threshold of the page's histogram: the gray level that splits it best
// into ink and paper.
int otsu(const std::array<int64_t, 256>& hist, int64_t total) {
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

// The search's points: each turned by an angle to the row it falls in, counted in four interleaved histograms
// (consecutive points share rows), their squares summed in row order: the same score as one histogram.
struct Points {
  std::vector<double> xs, ys;  // from the page's centre
  int rows = 0;
  double half = 0;             // rows / 2.0
  std::vector<int32_t> count;  // four histograms of rows, interleaved: count[4 * row + k]

  double score(double deg) {
    const double a = deg * kPi / 180.0, s = std::sin(a), c = std::cos(a);
    std::fill(count.begin(), count.end(), 0);
    const size_t n = xs.size();
    const double* x = xs.data();
    const double* y = ys.data();
    int32_t* h = count.data();
    const int last = rows - 1;
    const double mid = half;
    for (size_t i = 0; i < n; ++i) {
      // the row the point is on after the page turns `deg` counterclockwise
      const double r = y[i] * c - x[i] * s + mid;
      int v = static_cast<int>(r);
      v = v < 0 ? 0 : v > last ? last : v;
      ++h[4 * v + static_cast<int>(i & 3)];
    }
    double sum = 0;
    for (int j = 0; j < rows; ++j) {
      const int64_t m = int64_t{h[4 * j]} + h[4 * j + 1] + h[4 * j + 2] + h[4 * j + 3];
      sum += static_cast<double>(m) * m;
    }
    return sum;
  }
};

}  // namespace

Skew find_skew(const GrayView& page, double max_degrees) {
  Skew out;
  if (page.width < 16 || page.height < 16) return out;
  const std::array<int64_t, 256> hist = histogram(page);
  const int64_t total = static_cast<int64_t>(page.width) * page.height;
  const int t = otsu(hist, total);
  // the ink, a sample of every step-th ink pixel spread over the page (60,000 to
  // 119,999 of them where there are more than 60,000)
  int64_t ink = 0;
  for (int i = 0; i <= t; ++i) ink += hist[i];
  if (ink < 64 || ink > total / 2) return out;  // blank, or not text
  const int64_t step = std::max<int64_t>(1, ink / 60000);
  Points p;
  p.xs.reserve(static_cast<size_t>(ink / step + 1));
  p.ys.reserve(static_cast<size_t>(ink / step + 1));
  int64_t until = 0;  // ink pixels until the next one sampled: the 0th, the step-th, ...
  const double cx = page.width / 2.0, cy = page.height / 2.0;
  for (int y = 0; y < page.height; ++y) {
    const uint8_t* row = page.data + static_cast<int64_t>(y) * page.stride;
    for (int x = 0; x < page.width; ++x) {
      if (row[x] > t) continue;
      if (until == 0) {
        // widened once, as the search's arithmetic widens each float point
        p.xs.push_back(static_cast<double>(static_cast<float>(x - cx)));
        p.ys.push_back(static_cast<double>(static_cast<float>(y - cy)));
        until = step;
      }
      --until;
    }
  }
  // a score for each angle: the sum of squared counts of the rows the points fall
  // in, the page turned by it
  p.rows = static_cast<int>(std::ceil(std::hypot(page.width, page.height))) + 2;
  p.half = p.rows / 2.0;
  p.count.assign(static_cast<size_t>(4) * p.rows, 0);
  const double level = p.score(0);
  double best = 0, best_score = level;
  for (double d = -max_degrees; d <= max_degrees + 1e-9; d += 0.5) {
    const double v = p.score(d);
    if (v > best_score) {
      best_score = v;
      best = d;
    }
  }
  const double coarse = best;
  for (double d = coarse - 0.5; d <= coarse + 0.5 + 1e-9; d += 0.05) {
    if (std::abs(d) > max_degrees + 1e-9) continue;
    const double v = p.score(d);
    if (v > best_score) {
      best_score = v;
      best = d;
    }
  }
  out.degrees = std::round(best * 100) / 100;
  out.gain = level > 0 ? best_score / level : 1;
  return out;
}

// Each pixel of the turned page is where it was turned back clockwise, sampled
// bilinear (carbon::cpu::rotate_bilinear: vectors where the CPU has them, the same
// bytes as the scalar reference).
GrayImage rotate_page(const GrayView& page, double degrees) {
  return rotate_page(page, degrees, [](int rows, const std::function<void(int, int)>& band) { band(0, rows); });
}

GrayImage rotate_page(const GrayView& page, double degrees, const RowSplit& split) {
  GrayImage out;
  out.width = page.width;
  out.height = page.height;
  out.pixels.assign(static_cast<size_t>(page.width) * page.height, 255);
  const double a = degrees * kPi / 180.0, s = std::sin(a), c = std::cos(a);
  split(page.height, [&](int row0, int row1) {
    cpu::rotate_bilinear_rows(page.data, page.width, page.height, page.stride, s, c, out.pixels.data(), row0, row1);
  });
  return out;
}

}  // namespace carbon
