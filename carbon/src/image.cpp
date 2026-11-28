// Page images (carbon/image.h).

#include "carbon/image.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <new>
#include <optional>
#include <string>
#include <string_view>

#include "carbon/cpu.h"
#include "carbon/error.h"
#include "png.h"
#include "webp.h"

// stb_image for PNG, JPEG, BMP, PNM and GIF only (a GIF's first frame), from memory (paths with any characters work
// through std::filesystem), at most 32768 px a side.  PNG goes to png.cpp first, which gives the same bytes faster and
// leaves to stb_image what it does not read itself; WebP to webp.cpp (libwebp).
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_PNM
#define STBI_ONLY_GIF
#define STBI_NO_STDIO
#define STBI_MAX_DIMENSIONS 32768
#include <stb_image.h>

namespace carbon {

FileKind file_kind(std::span<const uint8_t> b) {
  auto starts = [&](std::string_view m, size_t at = 0) {
    return b.size() >= at + m.size() && std::memcmp(b.data() + at, m.data(), m.size()) == 0;
  };
  if (starts("\x89PNG\r\n\x1a\n")) return FileKind::kPng;
  if (starts("\xff\xd8\xff")) return FileKind::kJpeg;
  if (starts("GIF87a") || starts("GIF89a")) return FileKind::kGif;
  if (starts("RIFF") && starts("WEBP", 8)) return FileKind::kWebp;
  if (starts("%PDF-")) return FileKind::kPdf;
  if (starts("BM")) return FileKind::kBmp;
  if (b.size() >= 2 && b[0] == 'P' && b[1] >= '1' && b[1] <= '6') return FileKind::kPnm;
  return FileKind::kUnknown;
}

FileKind file_kind(const std::filesystem::path& file) {
  std::ifstream f(file, std::ios::binary);
  if (!f) return FileKind::kUnknown;
  uint8_t head[12] = {};
  f.read(reinterpret_cast<char*>(head), sizeof head);
  return file_kind(std::span<const uint8_t>(head, static_cast<size_t>(f.gcount())));
}

std::string_view file_kind_name(FileKind kind) {
  switch (kind) {
    case FileKind::kPng: return "png";
    case FileKind::kJpeg: return "jpeg";
    case FileKind::kBmp: return "bmp";
    case FileKind::kPnm: return "pnm";
    case FileKind::kGif: return "gif";
    case FileKind::kWebp: return "webp";
    case FileKind::kPdf: return "pdf";
    default: return "unknown";
  }
}

GrayImage decode_image(std::span<const uint8_t> bytes) { return decode_image(bytes, cpu::best_isa()); }

GrayImage decode_image(std::span<const uint8_t> bytes, cpu::Isa isa) {
  if (!cpu::isa_available(isa)) {
    throw Error(std::string("decode_image: ") + cpu::isa_name(isa) + " is not available on this CPU");
  }
  // the format by the file's signature; stb_image reads the rest (and refuses what it does not know)
  switch (file_kind(bytes)) {
    case FileKind::kWebp:
      return detail::webp_to_gray(bytes);
    case FileKind::kPdf:
      throw Error("the image could not be read: a PDF, whose pages are rendered (carbon/pdf.h), not an image");
    case FileKind::kPng:
      try {
        GrayImage img;
        if (detail::png_to_gray(bytes, isa, img)) return img;
      } catch (const std::bad_alloc&) {
        // stb_image reports running out of memory itself
      }
      break;
    default:
      break;
  }
  int w = 0, h = 0, channels = 0;
  // gray and alpha, the gray over white by the alpha, as the PNG decoder above
  stbi_uc* data = stbi_load_from_memory(bytes.data(), static_cast<int>(bytes.size()), &w, &h, &channels, 2);
  if (!data) throw Error(std::string("the image could not be read: ") + stbi_failure_reason());
  GrayImage img;
  img.width = w;
  img.height = h;
  img.pixels.resize(static_cast<size_t>(w) * h);
  for (size_t i = 0; i < img.pixels.size(); ++i) img.pixels[i] = over_white(data[2 * i], data[2 * i + 1]);
  stbi_image_free(data);
  return img;
}

GrayImage decode_image_resized(std::span<const uint8_t> bytes, const ImageFit& fit) {
  return decode_image_resized(bytes, fit, cpu::best_isa());
}

GrayImage decode_image_resized(std::span<const uint8_t> bytes, const ImageFit& fit, cpu::Isa isa) {
  if (!cpu::isa_available(isa)) {
    throw Error(std::string("decode_image_resized: ") + cpu::isa_name(isa) + " is not available on this CPU");
  }
  // fit once, whichever reading it is: a caller drawing a scale in it draws one
  std::optional<std::array<int, 2>> got;
  const ImageFit once = [&](int w, int h) {
    if (!got) got = fit(w, h);
    return *got;
  };
  try {
    GrayImage img;
    if (detail::png_to_gray_resized(bytes, isa, once, img)) return img;
  } catch (const std::bad_alloc&) {
    // decode_image's reading, and its own report
  }
  const GrayImage whole = decode_image(bytes, isa);
  const std::array<int, 2> size = once(whole.width, whole.height);
  if (size[0] <= 0 || size[1] <= 0) throw Error("decode_image_resized: an empty size");
  GrayImage out;
  out.width = size[0];
  out.height = size[1];
  out.pixels.resize(static_cast<size_t>(size[0]) * size[1]);
  resample_gray(whole.view(), 0, 0, whole.width, whole.height, size[0], size[1], out.pixels.data(), size[0], isa);
  return out;
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
