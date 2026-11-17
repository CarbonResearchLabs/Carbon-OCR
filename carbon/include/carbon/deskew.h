// A page's small skew: the angle that makes its text level,
// found by a projection search, and the page turned by it.  Orientation turns a
// page by quarter turns; this levels what is left, a few degrees, before the
// detector reads it.
//
// The search: the page's ink (Otsu's threshold), sampled to a fixed number of
// pixels, projected onto the rows of the page turned by each candidate angle;
// the sum of the squared row counts is largest when the lines are level (their
// ink falls in few rows, the gaps between them empty).  0.5-degree steps within
// +-max_degrees (5 at most), then 0.05-degree steps around the best.  Past 5 degrees
// (max_degrees up to 45: a page photographed or scanned askew, what orientation's
// quarter turns leave), 1-degree steps on a quarter of the points find the best wide
// angle, which half-degree steps on all of them weigh against the narrow search's; a
// best at the wide search's edge (its whole degrees) is no peak, and is not weighed.
#pragma once

#include <functional>

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

// Runs band(row0, row1) over rows 0 to rows - 1, in as many bands at once as it can (a caller's idle workers);
// band(0, rows) alone does the same.
using RowSplit = std::function<void(int rows, const std::function<void(int row0, int row1)>& band)>;
// The same page as rotate_page, its rows turned by `split`'s bands: the same bytes however they are cut.
GrayImage rotate_page(const GrayView& page, double degrees, const RowSplit& split);

}  // namespace carbon
