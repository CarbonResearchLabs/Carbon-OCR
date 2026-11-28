// The crop rule's resampling (carbon/lines.h's resample_gray) of a whole image whose rows come one at a time
// (carbon/src/lines.cpp): the PNG decoder's pages resampled as they are unfiltered (carbon/src/png.cpp), so a page
// read at another scale is never whole in memory.
#pragma once

#include <cstdint>

#include "carbon/cpu.h"

namespace carbon::detail {

// Image row y of a source (ctx), valid until the next call.
using RowOf = const uint8_t* (*)(void* ctx, int y);

// resample_gray of the whole width x height image (the rectangle [0, width) x [0, height)) to ow x oh into out
// (rows os bytes apart), the same bytes, image row y from row(ctx, y): the rows are asked for in increasing order,
// each once, as the horizontal pass reads them. False, no row asked for, where isa has no vector kernels (the scalar
// reference's passes read the image whole).
bool resample_gray_from(int width, int height, RowOf row, void* ctx, int ow, int oh, uint8_t* out, int64_t os,
                        cpu::Isa isa);

}  // namespace carbon::detail
