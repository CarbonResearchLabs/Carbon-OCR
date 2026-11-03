// Detection in strips: a page's rows cut into strips that run
// side by side on idle cores and are stitched back into the page's maps.
//
// Each strip runs on its own rows plus `halo` rows of the page above and below,
// at least the detector's receptive field, so the rows it keeps come out with
// the same bytes as from the whole page in one run.  Every cut falls on a
// multiple of `align`, the detector's largest stride, so a strip's feature maps
// line up with the page's at every level.  The halos cost compute (8 strips of
// a 1024-row page run 2.5 times the rows), so strips are for a page that finds
// many cores idle; with the cores busy a page is detected in one run.
#pragma once

#include <vector>

#include "carbon/tensor.h"

namespace carbon {

struct StripRule {
  int halo = 128;      // rows of context; fullerene det's receptive field reaches 127 px
  int align = 32;      // its largest stride
  int out_stride = 2;  // input rows per output row: its maps are at stride 2
};

struct Strip {
  int y0, y1;        // the page's input rows the strip runs on, [y0, y1)
  int keep0, keep1;  // the rows of its own output that are kept, [keep0, keep1)
  int out0;          // the page output row the first kept row goes to
};

// `strips` strips (fewer if the page has fewer align-row blocks) covering a page
// `height` rows tall, a multiple of rule.align (the detector's input is padded
// to one).  Throws Error on a height or rule that does not fit.
std::vector<Strip> plan_strips(int height, int strips, const StripRule& rule = {});

// The input rows of one strip: page [N, C, H, W] to [N, C, y1 - y0, W].
Tensor strip_input(const Tensor& page, const Strip& strip);

// Copies a strip's kept output rows ([N, C, rows, W'] from the detector) into
// the page's output [N, C, H / out_stride, W'].
void stitch_strip(const Tensor& strip_out, const Strip& strip, Tensor& page_out);

}  // namespace carbon
