// Line detection's targets and decoder (carbon/detect.h).

#include "carbon/detect.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <utility>

#include "carbon/cpu.h"

namespace carbon {

namespace {

// A run of set map pixels in one row, columns [c0, c1]; v is its first pixel's
// index among the set pixels in row-major order (where its values are).
struct Run {
  int row, c0, c1, v;
};

int find(std::vector<int>& parent, int i) {
  while (parent[i] != i) {
    parent[i] = parent[parent[i]];
    i = parent[i];
  }
  return i;
}

// The first column at or after c whose bit is `set`, or w.
int next_bit(const uint32_t* row, int words, int w, int c, bool set) {
  int k = c >> 5;
  if (k >= words) return w;
  uint32_t m = (set ? row[k] : ~row[k]) & (~0u << (c & 31));
  while (!m && ++k < words) m = set ? row[k] : ~row[k];
  return k >= words ? w : std::min(w, k * 32 + std::countr_zero(m));
}

// The 8-connected components of a mask's set bits, as runs grouped by component:
// runs[first[k] .. first[k + 1]) are component k's, in row order.
void components(const uint32_t* bits, int h, int w, int words, std::vector<Run>& runs, std::vector<int>& first) {
  std::vector<Run> all;
  std::vector<int> row_start(static_cast<size_t>(h) + 1, 0);
  int v = 0;
  for (int r = 0; r < h; ++r) {
    row_start[r] = static_cast<int>(all.size());
    const uint32_t* row = bits + static_cast<size_t>(r) * words;
    for (int c = next_bit(row, words, w, 0, true); c < w; c = next_bit(row, words, w, c, true)) {
      const int c1 = next_bit(row, words, w, c, false);
      all.push_back({r, c, c1 - 1, v});
      v += c1 - c;
      c = c1;
    }
  }
  row_start[h] = static_cast<int>(all.size());
  std::vector<int> parent(all.size());
  std::iota(parent.begin(), parent.end(), 0);
  for (int r = 1; r < h; ++r) {
    int j = row_start[r - 1];
    const int prev_end = row_start[r];
    for (int i = row_start[r]; i < row_start[r + 1]; ++i) {
      const Run& run = all[i];
      while (j < prev_end && all[j].c1 < run.c0 - 1) ++j;  // 8-connected: diagonal neighbours touch
      for (int k = j; k < prev_end && all[k].c0 <= run.c1 + 1; ++k) {
        const int a = find(parent, i), b = find(parent, k);
        if (a != b) parent[std::max(a, b)] = std::min(a, b);
      }
    }
  }
  // Dense component ids in order of each component's first run, then the runs
  // grouped by component (a counting sort keeps their row order).
  std::vector<int> id(all.size(), -1);
  int n = 0;
  for (size_t i = 0; i < all.size(); ++i) {
    const int root = find(parent, static_cast<int>(i));
    if (id[root] < 0) id[root] = n++;
    id[i] = id[root];
  }
  first.assign(static_cast<size_t>(n) + 1, 0);
  for (size_t i = 0; i < all.size(); ++i) ++first[id[i] + 1];
  std::partial_sum(first.begin(), first.end(), first.begin());
  runs.resize(all.size());
  std::vector<int> at(first.begin(), first.end() - 1);
  for (size_t i = 0; i < all.size(); ++i) runs[at[id[i]]++] = all[i];
}


}  // namespace

float det_logit(float p) {
  p = std::clamp(p, 1e-6f, 1.0f - 1e-6f);
  return std::log(p / (1.0f - p));
}

namespace {

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// The value at quantile q of v (reordered): the element at floor(q (n - 1)).
float quantile(std::vector<float>& v, double q) {
  const size_t k = static_cast<size_t>(std::floor(q * static_cast<double>(v.size() - 1)));
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
  return v[k];
}

bool reading_order(const DetBox& a, const DetBox& b) {
  return a.box.y0 != b.box.y0 ? a.box.y0 < b.box.y0 : a.box.x0 < b.box.x0;
}

// The map indices whose pixel centres s i + s / 2 lie in [lo, hi]; the one
// nearest the middle when none does.
std::pair<int, int> centres_in(double lo, double hi, double s) {
  int i0 = static_cast<int>(std::ceil((lo - s / 2) / s));
  int i1 = static_cast<int>(std::floor((hi - s / 2) / s));
  if (i0 > i1) i0 = i1 = static_cast<int>(std::lround(((lo + hi) / 2 - s / 2) / s));
  return {i0, i1};
}

double iou(const LineBox& a, const LineBox& b) {
  const double w = std::min(a.x1, b.x1) - std::max(a.x0, b.x0), h = std::min(a.y1, b.y1) - std::max(a.y0, b.y0);
  if (w <= 0 || h <= 0) return 0;
  const double both = w * h;
  return both / ((a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - both);
}

// One kind of box of the two passes merged (merge_passes) into out: a pair is
// kept from the second pass where its mean height is under `tiny` page pixels.
void merge_boxes(const std::vector<DetBox>& first, const std::vector<DetBox>& second, double tiny, double min_iou,
                 std::vector<DetBox>& out) {
  // The second pass's boxes by their tops: a box topped more than the tallest's
  // height above another ends above it, so each first box scans only those that
  // can reach it.
  std::vector<uint32_t> by_top(second.size());
  std::iota(by_top.begin(), by_top.end(), 0u);
  std::sort(by_top.begin(), by_top.end(), [&](uint32_t p, uint32_t q) { return second[p].box.y0 < second[q].box.y0; });
  double tallest = 0;
  for (const DetBox& b : second) tallest = std::max(tallest, b.box.y1 - b.box.y0);
  struct Pair {
    double iou;
    uint32_t i, j;
  };
  std::vector<Pair> pairs;
  for (uint32_t i = 0; i < first.size(); ++i) {
    const LineBox& a = first[i].box;
    auto it = std::lower_bound(by_top.begin(), by_top.end(), a.y0 - tallest,
                               [&](uint32_t j, double y) { return second[j].box.y0 < y; });
    for (; it != by_top.end() && second[*it].box.y0 < a.y1; ++it) {
      const double v = iou(a, second[*it].box);
      if (v > min_iou) pairs.push_back({v, i, *it});
    }
  }
  std::sort(pairs.begin(), pairs.end(), [](const Pair& p, const Pair& q) {
    return p.iou != q.iou ? p.iou > q.iou : p.i != q.i ? p.i < q.i : p.j < q.j;
  });
  std::vector<uint8_t> paired_first(first.size(), 0), paired_second(second.size(), 0);
  for (const Pair& p : pairs) {
    if (paired_first[p.i] || paired_second[p.j]) continue;
    paired_first[p.i] = paired_second[p.j] = 1;
    const LineBox &a = first[p.i].box, &b = second[p.j].box;
    out.push_back((a.y1 - a.y0 + b.y1 - b.y0) / 2 < tiny ? second[p.j] : first[p.i]);
  }
  // A first box no second box pairs with is dropped where the second pass read the
  // same text as tiny lines: two or more of them lie mostly inside it (the first pass
  // ran tiny lines together), or one does and fills most of it (the same line, boxed
  // loosely). Kept, its text would be read twice. A second box that is not tiny (a
  // heading the larger scale split) never drops a first one.
  auto inside = [](const LineBox& b, const LineBox& a) {  // the share of b within a
    const double w = std::min(a.x1, b.x1) - std::max(a.x0, b.x0), h = std::min(a.y1, b.y1) - std::max(a.y0, b.y0);
    const double area = (b.x1 - b.x0) * (b.y1 - b.y0);
    return w > 0 && h > 0 && area > 0 ? w * h / area : 0.0;
  };
  for (size_t i = 0; i < first.size(); ++i) {
    if (paired_first[i]) continue;
    const LineBox& a = first[i].box;
    const double a_area = (a.x1 - a.x0) * (a.y1 - a.y0);
    int held = 0;
    double held_area = 0;
    auto it = std::lower_bound(by_top.begin(), by_top.end(), a.y0 - tallest,
                               [&](uint32_t j, double y) { return second[j].box.y0 < y; });
    for (; it != by_top.end() && second[*it].box.y0 < a.y1; ++it) {
      const LineBox& b = second[*it].box;
      if (b.y1 - b.y0 < tiny && inside(b, a) >= 0.7) {
        ++held;
        held_area += (b.x1 - b.x0) * (b.y1 - b.y0);
      }
    }
    if (held >= 2 || (held == 1 && a_area > 0 && held_area >= 0.4 * a_area)) continue;
    out.push_back(first[i]);
  }
  for (size_t j = 0; j < second.size(); ++j) {
    if (!paired_second[j]) out.push_back(second[j]);
  }
}

}  // namespace

bool takes_second_pass(const Detection& first, const DetRule& rule) {
  return rule.second_pass_below > 0 && !first.lines.empty() && first.median_line_height() < rule.second_pass_below;
}

Detection merge_passes(const Detection& first, const Detection& second, double first_scale, const DetRule& rule) {
  const double tiny = rule.second_pass_below * first_scale;
  Detection d;
  d.lines.reserve(first.lines.size() + second.lines.size());
  merge_boxes(first.lines, second.lines, tiny, rule.merge_iou, d.lines);
  merge_boxes(first.pictures, second.pictures, tiny, rule.merge_iou, d.pictures);
  std::sort(d.lines.begin(), d.lines.end(), reading_order);
  std::sort(d.pictures.begin(), d.pictures.end(), reading_order);
  return d;
}

double Detection::median_line_height() const {
  if (lines.empty()) return 0;
  std::vector<double> h;
  for (const DetBox& l : lines) h.push_back(l.box.y1 - l.box.y0);
  std::nth_element(h.begin(), h.begin() + static_cast<std::ptrdiff_t>(h.size() / 2), h.end());
  return h[h.size() / 2];
}

void det_targets(std::span<const LineBox> lines, std::span<const LineBox> pictures, int map_h, int map_w,
                 float* maps, const DetRule& rule) {
  const size_t plane = static_cast<size_t>(map_h) * map_w;
  float* core = maps;
  float* top = maps + plane;
  float* bottom = maps + 2 * plane;
  float* picture = maps + 3 * plane;
  std::fill(maps, maps + 4 * plane, 0.0f);
  const double s = rule.stride;
  std::vector<int32_t> owner(plane, -1);  // the line whose band holds the pixel; -2 for two lines'
  for (size_t i = 0; i < lines.size(); ++i) {
    const LineBox& b = lines[i];
    const double lh = b.y1 - b.y0;
    if (lh <= 0 || b.x1 <= b.x0) continue;
    const double margin = (1 - rule.core_frac) / 2 * lh, shrink = rule.core_x_shrink * lh;
    auto [r0, r1] = centres_in(b.y0 + margin, b.y1 - margin, s);
    auto [c0, c1] = centres_in(b.x0 + shrink, std::max(b.x0 + shrink, b.x1 - shrink), s);
    r0 = std::max(r0, 0);
    r1 = std::min(r1, map_h - 1);
    c0 = std::max(c0, 0);
    c1 = std::min(c1, map_w - 1);
    for (int r = r0; r <= r1; ++r) {
      const double y = s * r + s / 2;
      for (int c = c0; c <= c1; ++c) {
        const size_t k = static_cast<size_t>(r) * map_w + c;
        if (owner[k] == -1) {
          owner[k] = static_cast<int32_t>(i);
          core[k] = 1;
          top[k] = static_cast<float>(y - b.y0);
          bottom[k] = static_cast<float>(b.y1 - y);
        } else if (owner[k] >= 0 && owner[k] != static_cast<int32_t>(i)) {
          owner[k] = -2;
          core[k] = top[k] = bottom[k] = 0;
        }
      }
    }
  }
  for (const LineBox& b : pictures) {
    const int r0 = std::max(0, static_cast<int>(std::ceil((b.y0 - s / 2) / s)));
    const int r1 = std::min(map_h - 1, static_cast<int>(std::floor((b.y1 - s / 2) / s)));
    const int c0 = std::max(0, static_cast<int>(std::ceil((b.x0 - s / 2) / s)));
    const int c1 = std::min(map_w - 1, static_cast<int>(std::floor((b.x1 - s / 2) / s)));
    for (int r = r0; r <= r1; ++r) {
      std::fill(picture + static_cast<size_t>(r) * map_w + c0, picture + static_cast<size_t>(r) * map_w + c1 + 1,
                1.0f);
    }
  }
}

DetMasks threshold_det(const float* maps, int map_h, int map_w, const DetRule& rule) {
  const size_t plane = static_cast<size_t>(map_h) * map_w;
  const float* core = maps;
  const float* top = maps + plane;
  const float* bottom = maps + 2 * plane;
  const float* picture = maps + 3 * plane;
  const float tc = det_logit(rule.core_threshold), tp = det_logit(rule.picture_threshold);
  DetMasks m;
  m.height = map_h;
  m.width = map_w;
  m.words = (map_w + 31) / 32;
  m.core.resize(static_cast<size_t>(map_h) * m.words);
  m.picture.resize(m.core.size());
  // The masks a row at a time on the CPU's vectors, then each set pixel's values, in the pixels' order, into lists
  // reserved for them.
  size_t cores = 0, pictures = 0;
  for (int r = 0; r < map_h; ++r) {
    const size_t at = static_cast<size_t>(r) * map_w, word = static_cast<size_t>(r) * m.words;
    cpu::above_bits(core + at, map_w, tc, &m.core[word]);
    cpu::above_bits(picture + at, map_w, tp, &m.picture[word]);
    for (int k = 0; k < m.words; ++k) {
      cores += static_cast<size_t>(std::popcount(m.core[word + k]));
      pictures += static_cast<size_t>(std::popcount(m.picture[word + k]));
    }
  }
  m.core_values.reserve(3 * cores);
  m.picture_values.reserve(pictures);
  for (size_t word = 0; word < m.core.size(); ++word) {
    const size_t at = word / m.words * map_w + word % m.words * 32;
    for (uint32_t b = m.core[word]; b; b &= b - 1) {
      const size_t i = at + static_cast<size_t>(std::countr_zero(b));
      m.core_values.insert(m.core_values.end(), {core[i], top[i], bottom[i]});
    }
    for (uint32_t b = m.picture[word]; b; b &= b - 1) m.picture_values.push_back(picture[at + std::countr_zero(b)]);
  }
  return m;
}

Detection decode_det(const float* maps, int map_h, int map_w, double scale, const DetRule& rule) {
  return decode_det(threshold_det(maps, map_h, map_w, rule), scale, rule);
}

Detection decode_det(const DetMasks& m, double scale, const DetRule& rule) {
  const int map_h = m.height, map_w = m.width;
  const float* cv = m.core_values.data();
  const double s = rule.stride;
  Detection d;
  std::vector<Run> runs;
  std::vector<int> first;
  std::vector<float> tops, bottoms;

  components(m.core.data(), map_h, map_w, m.words, runs, first);
  for (size_t k = 0; k + 1 < first.size(); ++k) {
    tops.clear();
    bottoms.clear();
    int cmin = map_w, cmax = -1;
    double prob = 0;
    for (int i = first[k]; i < first[k + 1]; ++i) {
      const Run& run = runs[i];
      const float y = static_cast<float>(s * run.row + s / 2);
      for (int c = run.c0; c <= run.c1; ++c) {
        const float* px = cv + 3 * static_cast<size_t>(run.v + c - run.c0);
        tops.push_back(y - px[1]);
        bottoms.push_back(y + px[2]);
        prob += sigmoid(px[0]);
      }
      cmin = std::min(cmin, run.c0);
      cmax = std::max(cmax, run.c1);
    }
    const int n = static_cast<int>(tops.size());
    if (n < rule.min_core_pixels) continue;
    const double y0 = quantile(tops, rule.edge_quantile), y1 = quantile(bottoms, 1 - rule.edge_quantile);
    const double lh = y1 - y0;
    if (lh <= 0) continue;
    const double shrink = rule.core_x_shrink * lh;
    const LineBox box{(s * cmin - shrink) * scale, y0 * scale, (s * (cmax + 1) + shrink) * scale, y1 * scale};
    d.lines.push_back({box, static_cast<float>(prob / n), n});
  }

  components(m.picture.data(), map_h, map_w, m.words, runs, first);
  for (size_t k = 0; k + 1 < first.size(); ++k) {
    int cmin = map_w, cmax = -1, rmin = map_h, rmax = -1, n = 0;
    double prob = 0;
    for (int i = first[k]; i < first[k + 1]; ++i) {
      const Run& run = runs[i];
      cmin = std::min(cmin, run.c0);
      cmax = std::max(cmax, run.c1);
      rmin = std::min(rmin, run.row);
      rmax = std::max(rmax, run.row);
      n += run.c1 - run.c0 + 1;
      for (int c = run.c0; c <= run.c1; ++c) prob += sigmoid(m.picture_values[static_cast<size_t>(run.v + c - run.c0)]);
    }
    if (n < rule.min_picture_pixels) continue;
    const LineBox box{s * cmin * scale, s * rmin * scale, s * (cmax + 1) * scale, s * (rmax + 1) * scale};
    d.pictures.push_back({box, static_cast<float>(prob / n), n});
  }
  std::sort(d.lines.begin(), d.lines.end(), reading_order);
  std::sort(d.pictures.begin(), d.pictures.end(), reading_order);
  return d;
}

}  // namespace carbon
