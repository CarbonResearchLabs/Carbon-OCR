// Page images: reading them, turning them upright, and the models' inputs.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

#include "carbon/cpu.h"
#include "carbon/lines.h"

namespace carbon {

// A gray sample g of alpha a over white paper: round((g a + 255 (255 - a)) / 255), exactly; a = 255 keeps g.
inline uint8_t over_white(uint32_t g, uint32_t a) {
  const uint32_t t = g * a + 255 * (255 - a) + 128;
  return static_cast<uint8_t>((t + (t >> 8)) >> 8);
}

// A grayscale image that owns its pixels: paper 255, ink 0, rows packed.
struct GrayImage {
  int width = 0, height = 0;
  std::vector<uint8_t> pixels;

  GrayView view() const { return {pixels.data(), width, height, width}; }
};

// What a file is by its first bytes (its signature), whatever its name says: one of the images decode_image reads, a
// PDF, or none of them. PNG 89 'PNG' 0D 0A 1A 0A; JPEG FF D8 FF; BMP "BM"; PNM 'P' and a digit 1 to 6; GIF "GIF87a"
// or "GIF89a"; WebP "RIFF", 4 bytes, "WEBP"; PDF "%PDF-".
enum class FileKind : uint8_t { kUnknown, kPng, kJpeg, kBmp, kPnm, kGif, kWebp, kPdf };
FileKind file_kind(std::span<const uint8_t> head);      // from its first 12 bytes at most
FileKind file_kind(const std::filesystem::path& file);  // its first bytes; kUnknown where it cannot be read
std::string_view file_kind_name(FileKind kind);         // "png", "jpeg", "bmp", "pnm", "gif", "webp", "pdf", "unknown"

// Reads a PNG, JPEG, BMP, PNM (binary PGM and PPM), GIF (its first frame) or WebP (lossy or lossless; an animation's
// first frame) file, the format told by its first bytes (file_kind), not its name, and converts it to gray: the
// bytes stb_image gives when asked for one channel, RGB weighed (77 R + 150 G + 29 B) / 256 and a pixel's alpha laid
// over white paper (over_white), whatever the format.  PNG through Carbon's own decoder (libdeflate, and the CPU
// kernels of `isa`, best_isa() by default: carbon/cpu.h), which leaves interlaced and damaged files and a few rare
// kinds to stb_image; WebP through libwebp's decoder (carbon/src/webp.cpp), to RGB or RGBA and then to gray as
// stb_image weighs RGB, so a WebP and a PNG of the same pixels give the same page; the other formats through
// stb_image.  Throws Error if it cannot be read or is larger than 32768 px a side, or if `isa` is not available.
GrayImage load_image(const std::filesystem::path& file);
GrayImage decode_image(std::span<const uint8_t> bytes);
GrayImage decode_image(std::span<const uint8_t> bytes, cpu::Isa isa);

// An image's size at another scale, from its own (width, height): {width, height} there.
using ImageFit = std::function<std::array<int, 2>(int width, int height)>;

// decode_image, then resize_gray to the size fit gives for the image's width and height: the same bytes, but a PNG's
// rows are resampled as they are unfiltered, a block at a time, so the image is never whole in memory (its inflated
// rows alone): a page read at a smaller scale (the detector's, say) at half the memory's traffic.
// fit is called once. Throws as decode_image, and Error for an empty size.
GrayImage decode_image_resized(std::span<const uint8_t> bytes, const ImageFit& fit);
GrayImage decode_image_resized(std::span<const uint8_t> bytes, const ImageFit& fit, cpu::Isa isa);

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
