// Page images (carbon/image.h).

#include "carbon/image.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <new>
#include <string>

#include "carbon/cpu.h"
#include "carbon/error.h"
#include "png.h"

// stb_image for PNG, JPEG, BMP and PNM only, from memory (paths with any
// characters work through std::filesystem), at most 32768 px a side.  PNG goes
// to png.cpp first, which gives the same bytes faster and leaves to stb_image
// what it does not read itself.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_PNM
#define STBI_NO_STDIO
#define STBI_MAX_DIMENSIONS 32768
#include <stb_image.h>

namespace carbon {

GrayImage decode_image(std::span<const uint8_t> bytes) { return decode_image(bytes, cpu::best_isa()); }

GrayImage decode_image(std::span<const uint8_t> bytes, cpu::Isa isa) {
  if (!cpu::isa_available(isa)) {
    throw Error(std::string("decode_image: ") + cpu::isa_name(isa) + " is not available on this CPU");
  }
  try {
    GrayImage img;
    if (detail::png_to_gray(bytes, isa, img)) return img;
  } catch (const std::bad_alloc&) {
    // stb_image reports running out of memory itself
  }
  int w = 0, h = 0, channels = 0;
  stbi_uc* data = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &w, &h, &channels, 1);
  if (!data) throw Error(std::string("the image could not be read: ") + stbi_failure_reason());
  GrayImage img;
  img.width = w;
  img.height = h;
  img.pixels.assign(data, data + static_cast<size_t>(w) * h);
  stbi_image_free(data);
  return img;
}

GrayImage load_image(const std::filesystem::path& file) {
  // One read of the whole file (a byte at a time through istreambuf_iterator was 4-7 ms for a page's PNG).
  std::ifstream f(file, std::ios::binary | std::ios::ate);
  if (!f) throw Error("cannot open " + file.string());
  const std::streamoff size = f.tellg();
  if (size < 0) throw Error("cannot read " + file.string());
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  f.seekg(0);
  if (size > 0 && !f.read(reinterpret_cast<char*>(bytes.data()), size)) throw Error("cannot read " + file.string());
  try {
    return decode_image(bytes);
  } catch (const Error& e) {
    throw Error(file.string() + ": " + e.what());
  }
}

// The bytes moved by carbon::cpu::rotate_quarters: rows copied or reversed, or
// tiles transposed in registers where the CPU has vectors.
GrayImage rotate_quarter(const GrayView& image, int quarter_turns) {
  const int q = ((quarter_turns % 4) + 4) % 4;
  const int w = image.width, h = image.height;
  GrayImage out;
  out.width = q % 2 ? h : w;
  out.height = q % 2 ? w : h;
  out.pixels.resize(static_cast<size_t>(w) * h);
  cpu::rotate_quarters(image.data, w, h, image.stride, q, out.pixels.data());
  return out;
}

GrayImage resize_gray(const GrayView& image, int width, int height) {
  if (width <= 0 || height <= 0) throw Error("resize_gray: an empty size");
  GrayImage out;
  out.width = width;
  out.height = height;
  out.pixels.resize(static_cast<size_t>(width) * height);
  resample_gray(image, 0, 0, image.width, image.height, width, height, out.pixels.data(), width);
  return out;
}

void ink_input(const GrayView& page, int fit_w, int fit_h, int out_w, int out_h, float* out) {
  if (fit_w <= 0 || fit_h <= 0 || fit_w > out_w || fit_h > out_h) {
    throw Error("ink_input: the page does not fit the input");
  }
  GrayImage small;
  GrayView src = page;
  if (fit_w != page.width || fit_h != page.height) {
    small = resize_gray(page, fit_w, fit_h);
    src = small.view();
  }
  // Each value written once: the page's, then paper to the right of it and below.
  // The conversion is a loop compilers vectorize (a division, not a multiply by
  // 1 / 255, which would round differently).
  for (int y = 0; y < fit_h; ++y) {
    const uint8_t* row = src.data + y * src.stride;
    float* o = out + static_cast<size_t>(y) * out_w;
    for (int x = 0; x < fit_w; ++x) o[x] = 1.0f - row[x] / 255.0f;
  }
  if (fit_w < out_w) {
    for (int y = 0; y < fit_h; ++y) {
      std::fill(out + static_cast<size_t>(y) * out_w + fit_w, out + static_cast<size_t>(y + 1) * out_w, 0.0f);
    }
  }
  std::fill(out + static_cast<size_t>(fit_h) * out_w, out + static_cast<size_t>(out_w) * out_h, 0.0f);
}

}  // namespace carbon
