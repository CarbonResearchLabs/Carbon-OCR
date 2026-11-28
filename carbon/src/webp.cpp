// WebP pages (carbon/src/webp.h): libwebp's decoder, then gray as stb_image gives it.
//
// A still file goes to libwebp whole. An animation (VP8X with its animation flag) is a canvas and frames, which
// WebPDecode does not read: its first ANMF chunk's frame (an ALPH and a VP8 chunk, or a VP8L chunk) is put in a
// still file of its own (RIFF, VP8X with the frame's size and its alpha flag, the frame's chunks as they are),
// decoded so, and laid on a white canvas at the frame's offset, as the animation's first frame shows over paper.

#include "webp.h"

#include <webp/decode.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "carbon/error.h"

namespace carbon::detail {
namespace {

constexpr int kMaxSide = 32768;  // decode_image's for every format (stb_image's STBI_MAX_DIMENSIONS)

[[noreturn]] void refuse(const std::string& why) { throw Error("the image could not be read: WebP: " + why); }

const char* status_text(VP8StatusCode s) {
  switch (s) {
    case VP8_STATUS_OUT_OF_MEMORY: return "out of memory";
    case VP8_STATUS_INVALID_PARAM: return "invalid parameters";
    case VP8_STATUS_BITSTREAM_ERROR: return "a damaged bitstream";
    case VP8_STATUS_UNSUPPORTED_FEATURE: return "a feature libwebp does not read";
    case VP8_STATUS_SUSPENDED:
    case VP8_STATUS_NOT_ENOUGH_DATA: return "the file is cut short";
    default: return "not read";
  }
}

uint32_t le24(const uint8_t* p) { return p[0] | uint32_t{p[1]} << 8 | uint32_t{p[2]} << 16; }
uint32_t le32(const uint8_t* p) { return le24(p) | uint32_t{p[3]} << 24; }
void put24(std::vector<uint8_t>& v, uint32_t x) {
  for (int s = 0; s < 24; s += 8) v.push_back(static_cast<uint8_t>(x >> s));
}
void put32(std::vector<uint8_t>& v, uint32_t x) {
  for (int s = 0; s < 32; s += 8) v.push_back(static_cast<uint8_t>(x >> s));
}

WebPBitstreamFeatures features(const uint8_t* data, size_t size) {
  WebPBitstreamFeatures f;
  const VP8StatusCode s = WebPGetFeatures(data, size, &f);
  if (s != VP8_STATUS_OK) refuse(status_text(s));
  if (f.width <= 0 || f.height <= 0 || f.width > kMaxSide || f.height > kMaxSide) {
    refuse(std::to_string(f.width) + " x " + std::to_string(f.height) + " px, over " + std::to_string(kMaxSide) +
           " a side");
  }
  return f;
}

// A still file decoded and laid on `out` at (x0, y0), what falls off the canvas cut: each pixel's gray as stb_image
// weighs RGB (stbi__compute_y), its alpha over white (over_white).
void lay_still(const uint8_t* data, size_t size, GrayImage& out, int x0, int y0) {
  const WebPBitstreamFeatures f = features(data, size);
  if (f.has_animation) refuse("a frame that is an animation");
  const int c = f.has_alpha ? 4 : 3;
  const size_t stride = static_cast<size_t>(f.width) * c;
  std::vector<uint8_t> px(stride * f.height);
  const uint8_t* got =
      f.has_alpha ? WebPDecodeRGBAInto(data, size, px.data(), px.size(), static_cast<int>(stride))
                  : WebPDecodeRGBInto(data, size, px.data(), px.size(), static_cast<int>(stride));
  if (!got) refuse("a damaged bitstream");
  const int xa = std::max(0, x0), xb = std::min(out.width, x0 + f.width);
  const int ya = std::max(0, y0), yb = std::min(out.height, y0 + f.height);
  for (int y = ya; y < yb; ++y) {
    const uint8_t* s = px.data() + static_cast<size_t>(y - y0) * stride + static_cast<size_t>(xa - x0) * c;
    uint8_t* o = out.pixels.data() + static_cast<size_t>(y) * out.width;
    if (c == 3) {
      for (int x = xa; x < xb; ++x, s += 3) o[x] = static_cast<uint8_t>((s[0] * 77 + s[1] * 150 + s[2] * 29) >> 8);
    } else {
      for (int x = xa; x < xb; ++x, s += 4) {
        o[x] = over_white(static_cast<uint32_t>(s[0] * 77 + s[1] * 150 + s[2] * 29) >> 8, s[3]);
      }
    }
  }
}

GrayImage canvas(int w, int h) {
  GrayImage g;
  g.width = w;
  g.height = h;
  g.pixels.assign(static_cast<size_t>(w) * h, 255);
  return g;
}

}  // namespace

GrayImage webp_to_gray(std::span<const uint8_t> b) {
  const WebPBitstreamFeatures f = features(b.data(), b.size());
  GrayImage out = canvas(f.width, f.height);
  if (!f.has_animation) {
    lay_still(b.data(), b.size(), out, 0, 0);
    return out;
  }
  // The chunks after "RIFF", its size and "WEBP" (VP8X, ANIM, then the frames), each padded to an even size.
  for (size_t at = 12; at + 8 <= b.size();) {
    const uint8_t* chunk = b.data() + at;
    const size_t n = le32(chunk + 4);
    if (n > b.size() - at - 8) refuse("the file is cut short");
    if (std::memcmp(chunk, "ANMF", 4) == 0) {
      if (n < 16) refuse("a frame's header cut short");
      const uint8_t* p = chunk + 8;
      const int fx = static_cast<int>(2 * le24(p)), fy = static_cast<int>(2 * le24(p + 3));
      const uint32_t fw = 1 + le24(p + 6), fh = 1 + le24(p + 9);
      bool alpha = false;  // the frame's own chunks: ALPH and VP8, or VP8L, maybe others
      for (size_t k = 16; k + 8 <= n; k += 8 + le32(p + k + 4) + (le32(p + k + 4) & 1)) {
        if (std::memcmp(p + k, "ALPH", 4) == 0) alpha = true;
      }
      std::vector<uint8_t> still = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'E', 'B', 'P', 'V', 'P', '8', 'X'};
      put32(still, 10);
      still.push_back(alpha ? 0x10 : 0);  // VP8X's alpha flag
      still.insert(still.end(), 3, 0);
      put24(still, fw - 1);
      put24(still, fh - 1);
      still.insert(still.end(), p + 16, p + n);
      const uint32_t riff = static_cast<uint32_t>(still.size() - 8);
      for (int s = 0; s < 4; ++s) still[4 + s] = static_cast<uint8_t>(riff >> (8 * s));
      lay_still(still.data(), still.size(), out, fx, fy);
      return out;
    }
    at += 8 + n + (n & 1);
  }
  refuse("an animation without a frame");
}

}  // namespace carbon::detail
