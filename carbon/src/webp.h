// WebP pages (carbon/src/webp.cpp): libwebp's decoder to RGB, or RGBA where the file has alpha, then each pixel to
// gray as stb_image weighs RGB ((77 R + 150 G + 29 B) >> 8) and laid over white by its alpha (over_white), so a WebP
// and a PNG of the same pixels give the same page (carbon/image.h).
#pragma once

#include <cstdint>
#include <span>

#include "carbon/image.h"

namespace carbon::detail {

// A still image (lossy VP8 or lossless VP8L, either with alpha), or an animation's first frame (its first ANMF
// chunk) set at its offset on the animation's canvas, the rest of the canvas paper. Throws Error ("the image could
// not be read: WebP: ...") where libwebp refuses the file, an animation has no frame, or a side is over 32768 px.
GrayImage webp_to_gray(std::span<const uint8_t> webp);

}  // namespace carbon::detail
