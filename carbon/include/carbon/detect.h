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
//
// Tiny text takes a second pass.  At stride 2 a line under 9 px
// has a band one or two map rows high, so the detector misses some and places
// the rest coarsely; the page detected again at twice the scale finds them as
// lines of 18 px.  On a page of print the second pass would find again what the
// first found, and where the two disagree it costs precision, so it runs only
// where the first pass's median line is under 9 px (takes_second_pass), and the
// passes are merged line by line, each line kept from the pass whose scale suits
// its height, and a first box over text the second read as tiny lines dropped, so
// nothing is read twice (merge_passes).
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
  // ... except a band of one pixel this sure (its probability; 0: never): training gives every line a band of at
  // least one map pixel, so a line under about 5 px at the detector's scale, or a lone character under about 7,
  // has only that one (det_targets' centres_in).
  float single_pixel = 0.8f;
  int min_picture_pixels = 16;
  // A line's top is this quantile of its band pixels' top edges, its bottom
  // the (1 - q) quantile of their bottom edges, so the box of a slightly
  // slanted line holds both its ends.
  double edge_quantile = 0.05;
  // The second pass: a page whose first pass's median line is under
  // second_pass_below px at the detector's scale (0: never) is detected again
  // at second_pass_scale times that scale; two boxes, one of each pass, whose
  // IoU is over merge_iou are one line.
  double second_pass_below = 9.0;
  double second_pass_scale = 2.0;
  double merge_iou = 0.5;
  // The second pass also where the median line is not tiny but at least this share of the lines, and at least
  // second_pass_lines of them, are (0, the default: by the median alone): a page whose small print is a fifth of
  // its lines. At 0.15 the whole page's second pass helped OmniParseBench's tiny_plans (85.2 to 86.9) and
  // large_format and cost 6-8% of the CPU's pages a second and 3% of the GPU's: off until the
  // pass runs on the tiny lines' bands alone.
  double second_pass_share = 0;
  int second_pass_lines = 8;
  // A pass by share (its median line not tiny) runs on the bands of rows that hold the tiny lines, not the
  // page: each tiny line's rows at the second scale, a line height either side, bands closer than
  // second_pass_join rows joined, each run with the strips' halo; a band's line is kept where its centre lies in
  // the band's own rows. False: the whole page, as a pass by the median.
  bool second_pass_bands = true;
  int second_pass_join = 256;
  // The second pass at the scale that makes the tiny lines' median height this many pixels, capped by the
  // page's own resolution over the first pass's (and at least second_pass_scale), its strips keeping each run
  // about a first pass's pixels. 0: second_pass_scale, within second_pass_long_side (the default).
  double second_pass_target = 0;
};

struct DetBox {
  LineBox box;
  float score = 0;  // the mean probability over its pixels
  int pixels = 0;   // its map pixels
};

// The picture map thresholded (DetMasks::picture), kept with a detection: bit j of word k of row r is map pixel
// (r, 32 k + j), which covers the boxes' pixels [px_x c, px_x (c + 1)) x [px_y r, px_y (r + 1)). The engine's gate
// reads it (a line read out of a photograph, carbon/engine.h's GateRule); empty where no pass made one.
struct PictureMask {
  int height = 0, width = 0, words = 0;
  double px_x = 0, px_y = 0;  // the boxes' pixels a map pixel covers
  std::vector<uint32_t> bits;
  // The share of the map pixels under a box that are set (0 for an empty mask or box).
  double share(const LineBox& b) const;
};

struct Detection {
  std::vector<DetBox> lines;     // top to bottom, then left to right
  std::vector<DetBox> pictures;  // the same order
  PictureMask picture_mask;      // the first pass's (decode_det from masks; merge_passes keeps the first's)
  // The median line height, in the boxes' pixels: under 9 px at the detector's
  // scale, the page takes a second pass at a larger one (takes_second_pass).  0
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

// Whether a page takes the second pass: its first pass's median line, at the
// detector's scale (decode_det's scale 1), under rule.second_pass_below.  A
// page without lines takes none.
bool takes_second_pass(const Detection& first, const DetRule& rule = {});

// The two passes' detections of a page merged, their boxes in the page's pixels,
// first_scale of them to one input pixel of the first pass.  A line of each pass
// whose boxes' IoU is over rule.merge_iou are one line (the pairs of largest IoU
// first, each line in one pair at most), kept from the pass whose scale suits
// its height: the second's where the pair's mean height is under
// rule.second_pass_below at the first pass's scale, the first's otherwise.
// Every other line of either pass is kept, one pass's miss being the other's find,
// except a first-pass line over text the second read as tiny lines (two or more
// of them mostly inside it, or one that fills most of it): kept, its text would be
// read twice.  Pictures likewise.  In reading order, as decode_det gives them.
Detection merge_passes(const Detection& first, const Detection& second, double first_scale,
                       const DetRule& rule = {});

}  // namespace carbon
