// Page images (carbon/image.h).

#include "carbon/image.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>

#include "carbon/error.h"

// stb_image for PNG, JPEG, BMP and PNM only, from memory (paths with any
// characters work through std::filesystem), at most 32768 px a side.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_PNM
#define STBI_NO_STDIO
#define STBI_MAX_DIMENSIONS 32768
#include <stb_image.h>

namespace carbon {

GrayImage decode_image(std::span<const uint8_t> bytes) {
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
  std::ifstream f(file, std::ios::binary);
  if (!f) throw Error("cannot open " + file.string());
  const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  try {
    return decode_image(bytes);
  } catch (const Error& e) {
    throw Error(file.string() + ": " + e.what());
  }
}

GrayImage rotate_quarter(const GrayView& image, int quarter_turns) {
  const int q = ((quarter_turns % 4) + 4) % 4;
  const int w = image.width, h = image.height;
  GrayImage out;
  out.width = q % 2 ? h : w;
  out.height = q % 2 ? w : h;
  out.pixels.resize(static_cast<size_t>(w) * h);
  for (int y = 0; y < h; ++y) {
    const uint8_t* row = image.data + y * image.stride;
    for (int x = 0; x < w; ++x) {
      int nx = x, ny = y;  // where (x, y) lands, turned clockwise q times
      if (q == 1) {
        nx = h - 1 - y;
        ny = x;
      } else if (q == 2) {
        nx = w - 1 - x;
        ny = h - 1 - y;
      } else if (q == 3) {
        nx = y;
        ny = w - 1 - x;
      }
      out.pixels[static_cast<size_t>(ny) * out.width + nx] = row[x];
    }
  }
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
  std::fill(out, out + static_cast<size_t>(out_w) * out_h, 0.0f);
  for (int y = 0; y < fit_h; ++y) {
    const uint8_t* row = src.data + y * src.stride;
    float* o = out + static_cast<size_t>(y) * out_w;
    for (int x = 0; x < fit_w; ++x) o[x] = 1.0f - row[x] / 255.0f;
  }
}

}  // namespace carbon
