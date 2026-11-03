// PNG pages straight to gray (carbon/src/png.cpp), the bytes stb_image gives when it is
// asked for one channel, in a fraction of its time: libdeflate inflates the image data,
// and each row is unfiltered and turned to gray as it comes, by the CPU kernels of
// carbon/src/cpu/png_kernels.h.
#pragma once

#include <cstdint>
#include <span>

#include "carbon/cpu.h"
#include "carbon/image.h"

namespace carbon::detail {

// Decodes `png` into `out` and returns true, `out` then byte for byte what
// stbi_load_from_memory(..., 1) returns; or returns false, `out` unspecified, for a
// file this decoder leaves to stb_image: not a PNG, interlaced (Adam7), Apple's CgBI,
// a bit depth below 8 for a colour type that does not allow it, anything stb_image
// would reject (a damaged chunk, image data that does not inflate to exactly the
// rows), a palette index past the palette, and sizes near stb_image's own limits.  The
// kernels are those of `isa` (carbon/cpu.h), which must be available.
bool png_to_gray(std::span<const uint8_t> png, cpu::Isa isa, GrayImage& out);

}  // namespace carbon::detail
