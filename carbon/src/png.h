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

// A zlib stream inflated into out (out_n bytes, its whole size), by libdeflate's decoder, or (avx2, where the build
// has it: zlib_inflate_avx2_built) by its copy compiled with /arch:AVX2 (MSVC), the same bytes. False where
// the stream is bad or does not fill out_n exactly.
bool zlib_inflate(const uint8_t* in, size_t n, uint8_t* out, size_t out_n, bool avx2);
bool zlib_inflate_avx2_built();

}  // namespace carbon::detail
