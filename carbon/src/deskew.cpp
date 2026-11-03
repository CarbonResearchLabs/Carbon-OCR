// A page's small skew and the page levelled (carbon/deskew.h).

#include "carbon/deskew.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace carbon {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Otsu's threshold of the page's histogram: the gray level that splits it best
// into ink and paper.
int otsu(const GrayView& page) {
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

}  // namespace

Skew find_skew(const GrayView& page, double max_degrees) {
  Skew out;
  if (page.width < 16 || page.height < 16) return out;
  const int t = otsu(page);
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

GrayImage rotate_page(const GrayView& page, double degrees) {
  GrayImage out;
  out.width = page.width;
  out.height = page.height;
  out.pixels.assign(static_cast<size_t>(page.width) * page.height, 255);
  const double a = degrees * kPi / 180.0, s = std::sin(a), c = std::cos(a);
  const double cx = (page.width - 1) / 2.0, cy = (page.height - 1) / 2.0;
  for (int y = 0; y < page.height; ++y) {
    uint8_t* dst = out.pixels.data() + static_cast<int64_t>(y) * page.width;
    const double dy = y - cy;
    for (int x = 0; x < page.width; ++x) {
      // where this pixel of the turned page was: turned back clockwise
      const double dx = x - cx;
      const double sx = dx * c - dy * s + cx, sy = dx * s + dy * c + cy;
      const int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
      if (x0 < -1 || y0 < -1 || x0 >= page.width || y0 >= page.height) continue;
      const double fx = sx - x0, fy = sy - y0;
      auto at = [&](int xx, int yy) -> double {
        if (xx < 0 || yy < 0 || xx >= page.width || yy >= page.height) return 255.0;
        return page.data[yy * page.stride + xx];
      };
      const double v = (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) +
                       (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
      dst[x] = static_cast<uint8_t>(std::lround(std::clamp(v, 0.0, 255.0)));
    }
  }
  return out;
}

}  // namespace carbon
