// Page images: reading them, turning them upright, and the models' inputs.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "carbon/cpu.h"
#include "carbon/lines.h"

namespace carbon {

// A grayscale image that owns its pixels: paper 255, ink 0, rows packed.
struct GrayImage {
  int width = 0, height = 0;
  std::vector<uint8_t> pixels;

  GrayView view() const { return {pixels.data(), width, height, width}; }
};

// Reads a PNG, JPEG, BMP or PNM file and converts it to gray: the bytes stb_image
// gives when asked for one channel.  PNG through Carbon's own decoder (libdeflate,
// and the CPU kernels of `isa`, best_isa() by default: carbon/cpu.h), which leaves
// interlaced and damaged files and a few rare kinds to stb_image; the other
// formats through stb_image.  Throws Error if it cannot be read or is larger than
// 32768 px a side, or if `isa` is not available.
GrayImage load_image(const std::filesystem::path& file);
GrayImage decode_image(std::span<const uint8_t> bytes);
GrayImage decode_image(std::span<const uint8_t> bytes, cpu::Isa isa);

// The image turned clockwise by quarter_turns times 90 degrees.
GrayImage rotate_quarter(const GrayView& image, int quarter_turns);

// The image resampled to width x height with the crop rule's antialiased
// bilinear (carbon/lines.h).
GrayImage resize_gray(const GrayView& image, int width, int height);

// A model's input from a page: the page resampled to fit_w x fit_h (the crop
// rule's antialiased bilinear; copied when it is that size already) into the
// top-left of an out_w x out_h float32 ink map (1 - gray / 255: ink 1, paper
// 0), the rest paper.
void ink_input(const GrayView& page, int fit_w, int fit_h, int out_w, int out_h, float* out);

}  // namespace carbon
