// Line detection's two ends: the targets the detector learns and
// the decoder that turns its maps into line and picture boxes.  Each is the
// other's inverse, decode_det(det_targets(boxes)) giving the boxes back within
// a pixel (a test checks it), and training renders its targets through this
// code (the Python bindings, python/), so the detector learns exactly what the
// engine decodes.
//
// The maps, float32 [4][h][w] at the detector's output stride (2):
//   0 core     a logit: the line's core band, the middle 40% of its height,
//              shortened at both ends by 0.15 of its height, so the bands of
//              neighbouring lines do not touch
//   1 top      at a core pixel, the distance in input pixels from the pixel's
//              centre up to the line's top
//   2 bottom   the same, down to the line's bottom
//   3 picture  a logit: inside a picture
// Map pixel (r, c) covers input rows [2r, 2r + 2) and columns [2c, 2c + 2), and
// is in a band when its centre (2c + 1, 2r + 1) is.  Boxes are in the
// detector's input pixels; decode_det scales them to the page's.
#pragma once

#include <span>
#include <vector>

#include "carbon/lines.h"

namespace carbon {

struct DetRule {
  int stride = 2;                  // input pixels per map pixel
  double core_frac = 0.4;          // the band's share of a line's height, centred
  double core_x_shrink = 0.15;     // the band's shortening at each end, in line heights
  float core_threshold = 0.5f;     // a probability
  float picture_threshold = 0.5f;  // a probability
  int min_core_pixels = 2;         // a smaller band is noise
  int min_picture_pixels = 16;
  // A line's top is this quantile of its band pixels' top edges, its bottom
  // the (1 - q) quantile of their bottom edges, so the box of a slightly
  // slanted line holds both its ends.
  double edge_quantile = 0.05;
};

struct DetBox {
  LineBox box;
  float score = 0;  // the mean probability over its pixels
  int pixels = 0;   // its map pixels
};

struct Detection {
  std::vector<DetBox> lines;     // top to bottom, then left to right
  std::vector<DetBox> pictures;  // the same order
  // The median line height, in the boxes' pixels: under 9 px at the detector's
  // scale, the page takes a second pass at a larger one.  0
  // without lines.
  double median_line_height() const;
};

// The targets for one page, in maps[4][map_h][map_w]: core and picture 0 or 1,
// top and bottom in input pixels at core pixels and 0 elsewhere.  A pixel in
// the bands of two lines goes to neither.  A band too thin or too short to
// hold a pixel centre gets the one pixel nearest its middle, so every line has
// a band.
void det_targets(std::span<const LineBox> lines, std::span<const LineBox> pictures, int map_h, int map_w,
                 float* maps, const DetRule& rule = {});

// The maps thresholded: all decode_det reads of them, and all a GPU sends back
// (carbon::cuda::threshold_det).  Each map row is `words` 32-bit words
// of bits, bit j of word k for pixel 32 k + j: the core above its threshold, the
// picture above its.  The set pixels' values follow in row-major order: a core
// pixel's logit, top and bottom (three floats), a picture pixel's logit.
struct DetMasks {
  int height = 0, width = 0, words = 0;
  std::vector<uint32_t> core, picture;  // [height][words]
  std::vector<float> core_values, picture_values;
};

DetMasks threshold_det(const float* maps, int map_h, int map_w, const DetRule& rule = {});

// A threshold as the maps hold it: the logit of p, clamped to [1e-6, 1 - 1e-6].
float det_logit(float p);

// Decodes thresholded maps: each 8-connected component of the core band is a
// line, its box from the distance maps; each component of the picture map a
// picture.  Boxes come back in input pixels times `scale` (the page's pixels per
// input pixel).
Detection decode_det(const DetMasks& masks, double scale = 1.0, const DetRule& rule = {});

// decode_det(threshold_det(maps)), for maps[4][map_h][map_w] (core and picture as
// logits).
Detection decode_det(const float* maps, int map_h, int map_w, double scale = 1.0, const DetRule& rule = {});

}  // namespace carbon
