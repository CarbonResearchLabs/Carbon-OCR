// A page's small skew: the angle that makes its text level,
// found by a projection search, and the page turned by it.  Orientation turns a
// page by quarter turns; this levels what is left, a few degrees, before the
// detector reads it.
//
// The search: the page's ink (Otsu's threshold), sampled to a fixed number of
// pixels, projected onto the rows of the page turned by each candidate angle;
// the sum of the squared row counts is largest when the lines are level (their
// ink falls in few rows, the gaps between them empty).  0.5-degree steps within
// +-max_degrees, then 0.05-degree steps around the best.
#pragma once

#include "carbon/image.h"
#include "carbon/lines.h"

namespace carbon {

struct Skew {
  double degrees = 0;  // turning the page this much counterclockwise levels it
  double gain = 1;     // the best angle's score over the level page's (1: no better)
};

Skew find_skew(const GrayView& page, double max_degrees = 5.0);

// The page turned `degrees` counterclockwise about its centre, the same size,
// bilinear, paper (255) where nothing of the page falls.
GrayImage rotate_page(const GrayView& page, double degrees);

}  // namespace carbon
