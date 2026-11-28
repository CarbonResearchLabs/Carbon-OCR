// Every image format decode_image reads (carbon/image.h), each told by its first bytes, not its name: files written
// here (WebP by libwebp's encoder, still and lossless, lossy, with alpha, an animation built chunk by chunk; GIF, BMP
// and PPM by hand; two small JPEGs Pillow wrote), each page held to the gray it must be: RGB weighed as stb_image
// weighs it, (77 R + 150 G + 29 B) >> 8, and alpha over white (over_white). PNG has png_test.cpp.

#include <gtest/gtest.h>
#include <webp/decode.h>
#include <webp/encode.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "carbon/cpu.h"
#include "carbon/error.h"
#include "carbon/image.h"

namespace carbon {
namespace {

namespace fs = std::filesystem;
using namespace std::string_literals;

using Sizes = std::vector<std::array<int, 2>>;

uint8_t weigh(uint32_t r, uint32_t g, uint32_t b) { return static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8); }

std::vector<uint8_t> bytes_of(const std::string& s) { return {s.begin(), s.end()}; }

std::vector<uint8_t> from_hex(const char* hex) {
  std::vector<uint8_t> out;
  for (const char* p = hex; p[0] && p[1]; p += 2) {
    out.push_back(static_cast<uint8_t>(std::stoi(std::string(p, 2), nullptr, 16)));
  }
  return out;
}

// Pixels, c channels a pixel (3 RGB, 4 RGBA), smooth with a little noise as a page's are, some alpha where c is 4.
std::vector<uint8_t> pixels(int w, int h, int c, std::mt19937& rng) {
  std::vector<uint8_t> px(static_cast<size_t>(w) * h * c);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      uint8_t* p = px.data() + (static_cast<size_t>(y) * w + x) * c;
      for (int k = 0; k < 3; ++k) p[k] = static_cast<uint8_t>(x * 7 + y * 3 + k * 70 + rng() % 9);
      if (c == 4) p[3] = static_cast<uint8_t>(x < w / 4 ? 0 : x < w / 2 ? (x * 37 + y * 11) % 256 : 255);
    }
  }
  return px;
}

// The page those pixels make: each one's gray, over white by its alpha.
std::vector<uint8_t> page_of(const uint8_t* px, int w, int h, int c) {
  std::vector<uint8_t> g(static_cast<size_t>(w) * h);
  for (size_t i = 0; i < g.size(); ++i, px += c) {
    const uint8_t v = weigh(px[0], px[1], px[2]);
    g[i] = c == 4 ? over_white(v, px[3]) : v;
  }
  return g;
}

void expect_page(const GrayImage& got, int w, int h, const std::vector<uint8_t>& want, const std::string& what) {
  ASSERT_EQ(got.width, w) << what;
  ASSERT_EQ(got.height, h) << what;
  const auto miss = std::mismatch(got.pixels.begin(), got.pixels.end(), want.begin());
  if (miss.first != got.pixels.end()) {
    const auto i = miss.first - got.pixels.begin();
    ADD_FAILURE() << what << ": pixel (" << i % w << ", " << i / w << ") is " << int{*miss.first} << ", not "
                  << int{*miss.second};
  }
}

// ------------------------------------------------------------------ WebP

std::vector<uint8_t> webp(const std::vector<uint8_t>& px, int w, int h, int c, bool lossless, float quality = 90) {
  uint8_t* out = nullptr;
  size_t n = 0;
  if (lossless) {
    n = c == 4 ? WebPEncodeLosslessRGBA(px.data(), w, h, w * 4, &out)
               : WebPEncodeLosslessRGB(px.data(), w, h, w * 3, &out);
  } else {
    n = c == 4 ? WebPEncodeRGBA(px.data(), w, h, w * 4, quality, &out)
               : WebPEncodeRGB(px.data(), w, h, w * 3, quality, &out);
  }
  EXPECT_GT(n, 0u);
  std::vector<uint8_t> file(out, out + n);
  WebPFree(out);
  return file;
}

// libwebp's own decoding of a still file, to RGBA: the pixels the page must be made of.
std::vector<uint8_t> webp_rgba(const std::vector<uint8_t>& file, int& w, int& h) {
  uint8_t* d = WebPDecodeRGBA(file.data(), file.size(), &w, &h);
  EXPECT_NE(d, nullptr);
  std::vector<uint8_t> px(d, d + static_cast<size_t>(w) * h * 4);
  WebPFree(d);
  return px;
}

void le(std::vector<uint8_t>& v, uint32_t x, int n) {
  for (int i = 0; i < n; ++i) v.push_back(static_cast<uint8_t>(x >> (8 * i)));
}

void riff_chunk(std::vector<uint8_t>& out, const char* tag, const std::vector<uint8_t>& data) {
  out.insert(out.end(), tag, tag + 4);
  le(out, static_cast<uint32_t>(data.size()), 4);
  out.insert(out.end(), data.begin(), data.end());
  if (data.size() & 1) out.push_back(0);
}

// A still file's image chunks (ALPH and VP8, or VP8L): what follows RIFF's header, its VP8X left out.
std::vector<uint8_t> image_chunks(const std::vector<uint8_t>& file) {
  std::vector<uint8_t> out;
  for (size_t at = 12; at + 8 <= file.size();) {
    const uint32_t n = file[at + 4] | file[at + 5] << 8 | file[at + 6] << 16 | uint32_t{file[at + 7]} << 24;
    const size_t end = at + 8 + n + (n & 1);
    if (std::string(file.begin() + at, file.begin() + at + 4) != "VP8X") {
      out.insert(out.end(), file.begin() + at, file.begin() + std::min(end, file.size()));
    }
    at = end;
  }
  return out;
}

struct Frame {
  int x = 0, y = 0, w = 0, h = 0;  // x and y even, as ANMF stores them halved
  std::vector<uint8_t> still;      // the frame as a still file of its own
};

// An animation on a w x h canvas: VP8X (animation and alpha), ANIM, an ANMF chunk a frame.
std::vector<uint8_t> animation(int w, int h, const std::vector<Frame>& frames) {
  std::vector<uint8_t> body = {'W', 'E', 'B', 'P'};
  std::vector<uint8_t> vp8x = {0x12, 0, 0, 0};
  le(vp8x, static_cast<uint32_t>(w - 1), 3);
  le(vp8x, static_cast<uint32_t>(h - 1), 3);
  riff_chunk(body, "VP8X", vp8x);
  riff_chunk(body, "ANIM", {255, 255, 255, 255, 0, 0});
  for (const Frame& f : frames) {
    std::vector<uint8_t> anmf;
    le(anmf, static_cast<uint32_t>(f.x / 2), 3);
    le(anmf, static_cast<uint32_t>(f.y / 2), 3);
    le(anmf, static_cast<uint32_t>(f.w - 1), 3);
    le(anmf, static_cast<uint32_t>(f.h - 1), 3);
    le(anmf, 100, 3);  // its duration, ms
    anmf.push_back(0);
    const std::vector<uint8_t> chunks = image_chunks(f.still);
    anmf.insert(anmf.end(), chunks.begin(), chunks.end());
    riff_chunk(body, "ANMF", anmf);
  }
  std::vector<uint8_t> file = {'R', 'I', 'F', 'F'};
  le(file, static_cast<uint32_t>(body.size()), 4);
  file.insert(file.end(), body.begin(), body.end());
  return file;
}

TEST(ImageFormats, WebPStillFilesAsTheirPixels) {
  std::mt19937 rng(41);
  for (const auto& [w, h] : Sizes{{61, 37}, {1, 1}, {300, 17}, {16, 400}}) {
    for (int c : {3, 4}) {
      const std::vector<uint8_t> px = pixels(w, h, c, rng);
      const std::string what = std::to_string(w) + " x " + std::to_string(h) + (c == 4 ? " RGBA" : " RGB");
      // lossless: the pixels themselves
      const std::vector<uint8_t> lossless = webp(px, w, h, c, true);
      EXPECT_EQ(file_kind(lossless), FileKind::kWebp);
      for (cpu::Isa isa : cpu::isas_available()) {
        expect_page(decode_image(lossless, isa), w, h, page_of(px.data(), w, h, c), what + ", lossless");
      }
      // lossy (VP8, and with alpha VP8X, ALPH and VP8): libwebp's own decoding of the file
      const std::vector<uint8_t> lossy = webp(px, w, h, c, false, 80);
      int dw = 0, dh = 0;
      const std::vector<uint8_t> rgba = webp_rgba(lossy, dw, dh);
      expect_page(decode_image(lossy), w, h, page_of(rgba.data(), dw, dh, 4), what + ", lossy");
    }
  }
}

TEST(ImageFormats, WebPAnimationReadsItsFirstFrameOverPaper) {
  std::mt19937 rng(43);
  const int cw = 50, ch = 40;
  // the first frame smaller than the canvas, at an offset, part transparent; the second over all of it
  const int fw = 21, fh = 13, fx = 6, fy = 10;
  const std::vector<uint8_t> first = pixels(fw, fh, 4, rng), second = pixels(cw, ch, 3, rng);
  for (bool lossless : {true, false}) {
    const Frame a{fx, fy, fw, fh, webp(first, fw, fh, 4, lossless, 85)};
    const Frame b{0, 0, cw, ch, webp(second, cw, ch, 3, true)};
    const std::vector<uint8_t> file = animation(cw, ch, {a, b});
    ASSERT_EQ(file_kind(file), FileKind::kWebp);
    int dw = 0, dh = 0;
    const std::vector<uint8_t> rgba = webp_rgba(a.still, dw, dh);
    const std::vector<uint8_t> frame = page_of(rgba.data(), dw, dh, 4);
    std::vector<uint8_t> want(static_cast<size_t>(cw) * ch, 255);
    for (int y = 0; y < fh; ++y) {
      for (int x = 0; x < fw; ++x) {
        want[static_cast<size_t>(y + fy) * cw + x + fx] = frame[static_cast<size_t>(y) * fw + x];
      }
    }
    expect_page(decode_image(file), cw, ch, want, lossless ? "lossless frames" : "a lossy frame with alpha");
  }
  // a frame past the canvas's edge is cut there
  const Frame wide{40, 30, fw, fh, webp(first, fw, fh, 4, true)};
  const GrayImage cut = decode_image(animation(cw, ch, {wide}));
  EXPECT_EQ(cut.width, cw);
  EXPECT_EQ(cut.pixels[0], 255);
}

TEST(ImageFormats, DamagedWebPIsRefusedWithItsReason) {
  std::mt19937 rng(47);
  const std::vector<uint8_t> px = pixels(40, 30, 3, rng);
  for (bool lossless : {true, false}) {
    std::vector<uint8_t> file = webp(px, 40, 30, 3, lossless);
    file.resize(file.size() / 2);
    try {
      decode_image(file);
      ADD_FAILURE() << "a WebP cut in half read";
    } catch (const Error& e) {
      EXPECT_EQ(std::string(e.what()).rfind("the image could not be read: WebP: ", 0), 0u) << e.what();
    }
  }
  // an animation with no frame
  std::vector<uint8_t> empty = animation(20, 20, {});
  EXPECT_THROW(decode_image(empty), Error);
}

TEST(ImageFormats, WebPDecodedToAScaleIsTheScaledPage) {
  std::mt19937 rng(53);
  const std::vector<uint8_t> file = webp(pixels(90, 70, 4, rng), 90, 70, 4, true);
  const GrayImage whole = decode_image(file);
  int calls = 0;
  const GrayImage got = decode_image_resized(file, [&](int w, int h) {
    ++calls;
    return std::array<int, 2>{w / 3, h / 3};
  });
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(got.pixels, resize_gray(whole.view(), 30, 23).pixels);
}

// ------------------------------------------------------------------ GIF

struct GifFrame {
  int x = 0, y = 0, w = 0, h = 0;
  std::vector<uint8_t> index;    // w x h palette indices, row by row
  std::vector<uint8_t> palette;  // a local colour table (RGB); empty: the global one
  int transparent = -1;
  bool interlace = false;
};

int table_bits(size_t entries) {
  int b = 1;
  while ((size_t{1} << b) < entries) ++b;
  return b;
}

// LZW as a GIF's raster holds it, every index a code of its own: a Clear before the table would grow past the codes'
// width, so they all stay min_code + 1 bits.
std::vector<uint8_t> lzw(const std::vector<uint8_t>& idx, int min_code) {
  const uint32_t clear = 1u << min_code;
  const int width = min_code + 1;
  const size_t run = (size_t{1} << min_code) - 2;
  std::vector<uint8_t> out;
  uint32_t acc = 0;
  int bits = 0;
  auto put = [&](uint32_t code) {
    acc |= code << bits;
    bits += width;
    for (; bits >= 8; bits -= 8, acc >>= 8) out.push_back(static_cast<uint8_t>(acc));
  };
  for (size_t i = 0; i < idx.size(); ++i) {
    if (i % run == 0) put(clear);
    put(idx[i]);
  }
  put(clear + 1);
  if (bits) out.push_back(static_cast<uint8_t>(acc));
  return out;
}

std::vector<uint8_t> gif(int w, int h, const std::vector<uint8_t>& global, const std::vector<GifFrame>& frames) {
  std::vector<uint8_t> o = {'G', 'I', 'F', '8', '9', 'a'};
  auto put16 = [&](int v) { le(o, static_cast<uint32_t>(v), 2); };
  put16(w);
  put16(h);
  const int gb = table_bits(global.size() / 3);
  o.insert(o.end(), {static_cast<uint8_t>(0xf0 | (gb - 1)), 0, 0});  // a global table; background 0
  std::vector<uint8_t> table = global;
  table.resize(size_t{3} << gb, 0);
  o.insert(o.end(), table.begin(), table.end());
  for (const GifFrame& f : frames) {
    if (f.transparent >= 0) o.insert(o.end(), {0x21, 0xf9, 4, 1, 0, 0, static_cast<uint8_t>(f.transparent), 0});
    o.push_back(0x2c);
    put16(f.x);
    put16(f.y);
    put16(f.w);
    put16(f.h);
    const int lb = f.palette.empty() ? 0 : table_bits(f.palette.size() / 3);
    o.push_back(static_cast<uint8_t>((lb ? 0x80 | (lb - 1) : 0) | (f.interlace ? 0x40 : 0)));
    if (lb) {
      std::vector<uint8_t> local = f.palette;
      local.resize(size_t{3} << lb, 0);
      o.insert(o.end(), local.begin(), local.end());
    }
    std::vector<int> rows;  // the rows in the order the raster holds them
    if (f.interlace) {
      for (const auto& [start, step] : Sizes{{0, 8}, {4, 8}, {2, 4}, {1, 2}}) {
        for (int y = start; y < f.h; y += step) rows.push_back(y);
      }
    } else {
      for (int y = 0; y < f.h; ++y) rows.push_back(y);
    }
    std::vector<uint8_t> idx;
    for (int y : rows) idx.insert(idx.end(), f.index.begin() + y * f.w, f.index.begin() + (y + 1) * f.w);
    const int mcs = std::max(2, lb ? lb : gb);
    o.push_back(static_cast<uint8_t>(mcs));
    const std::vector<uint8_t> data = lzw(idx, mcs);
    for (size_t at = 0; at < data.size(); at += 255) {
      const size_t n = std::min<size_t>(255, data.size() - at);
      o.push_back(static_cast<uint8_t>(n));
      o.insert(o.end(), data.begin() + at, data.begin() + at + n);
    }
    o.push_back(0);
  }
  o.push_back(0x3b);
  return o;
}

std::vector<uint8_t> palette(int n, std::mt19937& rng) {
  std::vector<uint8_t> p(static_cast<size_t>(n) * 3);
  for (auto& v : p) v = static_cast<uint8_t>(rng());
  return p;
}

GifFrame frame(int x, int y, int w, int h, int colours, std::mt19937& rng) {
  GifFrame f{x, y, w, h, {}, {}, -1, false};
  f.index.resize(static_cast<size_t>(w) * h);
  for (auto& v : f.index) v = static_cast<uint8_t>(rng() % colours);
  return f;
}

// The page a frame makes on a w x h screen that starts as paper: its pixels' palette colours weighed, its
// transparent ones and the screen it does not cover white.
std::vector<uint8_t> gif_page(int w, int h, const std::vector<uint8_t>& global, const GifFrame& f) {
  std::vector<uint8_t> g(static_cast<size_t>(w) * h, 255);
  const std::vector<uint8_t>& pal = f.palette.empty() ? global : f.palette;
  for (int y = 0; y < f.h; ++y) {
    for (int x = 0; x < f.w; ++x) {
      const int i = f.index[static_cast<size_t>(y) * f.w + x];
      if (i == f.transparent || x + f.x >= w || y + f.y >= h) continue;
      g[static_cast<size_t>(y + f.y) * w + x + f.x] = weigh(pal[3 * i], pal[3 * i + 1], pal[3 * i + 2]);
    }
  }
  return g;
}

TEST(ImageFormats, GifFirstFrame) {
  std::mt19937 rng(59);
  const std::vector<uint8_t> four = palette(4, rng), many = palette(256, rng);
  // a frame over the whole screen: 4 colours, 256, interlaced
  for (const auto* pal : {&four, &many}) {
    for (bool interlace : {false, true}) {
      GifFrame f = frame(0, 0, 37, 23, static_cast<int>(pal->size() / 3), rng);
      f.interlace = interlace;
      const std::vector<uint8_t> file = gif(37, 23, *pal, {f});
      EXPECT_EQ(file_kind(file), FileKind::kGif);
      const std::string what = std::to_string(pal->size() / 3) + " colours" + (interlace ? ", interlaced" : "");
      expect_page(decode_image(file), 37, 23, gif_page(37, 23, *pal, f), what);
    }
  }
  // a transparent colour; a frame smaller than the screen, with a local table; an animation's first frame alone
  GifFrame clear = frame(0, 0, 30, 20, 4, rng);
  clear.transparent = 2;
  expect_page(decode_image(gif(30, 20, four, {clear})), 30, 20, gif_page(30, 20, four, clear), "transparent");
  GifFrame small = frame(5, 3, 12, 9, 16, rng);
  small.palette = palette(16, rng);
  expect_page(decode_image(gif(30, 20, four, {small})), 30, 20, gif_page(30, 20, four, small), "a local table");
  const GifFrame next = frame(0, 0, 30, 20, 4, rng);
  expect_page(decode_image(gif(30, 20, four, {small, next})), 30, 20, gif_page(30, 20, four, small),
              "an animation");
}

// ------------------------------------------------------------------ BMP, PNM, JPEG

std::vector<uint8_t> bmp24(const std::vector<uint8_t>& rgb, int w, int h) {
  const int row = (w * 3 + 3) / 4 * 4;
  std::vector<uint8_t> o = {'B', 'M'};
  le(o, static_cast<uint32_t>(54 + row * h), 4);
  le(o, 0, 4);
  le(o, 54, 4);
  le(o, 40, 4);
  le(o, static_cast<uint32_t>(w), 4);
  le(o, static_cast<uint32_t>(h), 4);  // rows from the bottom up
  le(o, 1, 2);
  le(o, 24, 2);
  for (int i = 0; i < 6; ++i) le(o, 0, 4);
  for (int y = h - 1; y >= 0; --y) {
    for (int x = 0; x < w; ++x) {
      const uint8_t* p = rgb.data() + (static_cast<size_t>(y) * w + x) * 3;
      o.insert(o.end(), {p[2], p[1], p[0]});
    }
    o.insert(o.end(), static_cast<size_t>(row - w * 3), 0);
  }
  return o;
}

TEST(ImageFormats, BmpAndPpmAsTheirPixels) {
  std::mt19937 rng(61);
  for (const auto& [w, h] : Sizes{{13, 7}, {1, 1}, {64, 3}}) {
    const std::vector<uint8_t> rgb = pixels(w, h, 3, rng);
    const std::vector<uint8_t> want = page_of(rgb.data(), w, h, 3);
    const std::vector<uint8_t> bmp = bmp24(rgb, w, h);
    EXPECT_EQ(file_kind(bmp), FileKind::kBmp);
    expect_page(decode_image(bmp), w, h, want, "BMP");
    std::vector<uint8_t> ppm = bytes_of("P6\n" + std::to_string(w) + " " + std::to_string(h) + "\n255\n");
    ppm.insert(ppm.end(), rgb.begin(), rgb.end());
    EXPECT_EQ(file_kind(ppm), FileKind::kPnm);
    expect_page(decode_image(ppm), w, h, want, "PPM");
  }
}

// Pillow's JPEGs of 16 x 16 px (quality 95): gray 40 on the left half and 200 on the right; RGB (200, 30, 60) on
// the top half and (20, 180, 240) on the bottom, 4:4:4. Within 3 levels of each colour's gray.
TEST(ImageFormats, JpegNearItsColours) {
  const std::vector<uint8_t> gray = from_hex(
      "ffd8ffe000104a46494600010100000100010000ffdb0043000201010101010201010102020202020403020202020504040304060506"
      "060605060606070908060709070606080b08090a0a0a0a0a06080b0c0b0a0c090a0a0affc0000b080010001001011100ffc4001f0000"
      "010501010101010100000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d010203000411"
      "05122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445"
      "464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9"
      "aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffda00080101"
      "00003f00fc9fafea02bf97fafea02bffd9");
  const std::vector<uint8_t> rgb = from_hex(
      "ffd8ffe000104a46494600010100000100010000ffdb0043000201010101010201010102020202020403020202020504040304060506"
      "060605060606070908060709070606080b08090a0a0a0a0a06080b0c0b0a0c090a0a0affdb004301020202020202050303050a070607"
      "0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a0a"
      "0a0a0a0a0affc00011080010001003011100021101031101ffc4001f0000010501010101010100000000000000000102030405060708"
      "090a0bffc400b5100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552"
      "d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475"
      "767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6"
      "d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffc4001f0100030101010101010101010000000000000102030405060708"
      "090a0bffc400b51100020102040403040705040400010277000102031104052131061241510761711322328108144291a1b1c1092333"
      "52f0156272d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758595a636465666768696a7374"
      "75767778797a82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4"
      "d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffda000c03010002110311003f00f93ebe1cff005302803edcaff470ff00"
      "95f0a00fffd9");
  EXPECT_EQ(file_kind(gray), FileKind::kJpeg);
  EXPECT_EQ(file_kind(rgb), FileKind::kJpeg);
  const GrayImage g = decode_image(gray), c = decode_image(rgb);
  ASSERT_EQ(g.width, 16);
  ASSERT_EQ(c.height, 16);
  for (int y = 0; y < 16; ++y) {
    for (int x = 0; x < 16; ++x) {
      const size_t i = static_cast<size_t>(y) * 16 + x;
      EXPECT_NEAR(g.pixels[i], x < 8 ? 40 : 200, 3) << x << ", " << y;
      EXPECT_NEAR(c.pixels[i], y < 8 ? weigh(200, 30, 60) : weigh(20, 180, 240), 3) << x << ", " << y;
    }
  }
}

// ------------------------------------------------------------------ by content, not by name

TEST(ImageFormats, KindsByTheirFirstBytes) {
  const std::pair<std::string, FileKind> heads[] = {
      {"\x89PNG\r\n\x1a\n...."s, FileKind::kPng}, {"\xff\xd8\xff\xe0"s, FileKind::kJpeg},
      {"BM\x36\x00"s, FileKind::kBmp},           {"P5\n2 2"s, FileKind::kPnm},
      {"P1\n"s, FileKind::kPnm},                  {"GIF87a.."s, FileKind::kGif},
      {"GIF89a.."s, FileKind::kGif},              {"RIFF\x10\x00\x00\x00WEBPVP8 "s, FileKind::kWebp},
      {"%PDF-1.7\n"s, FileKind::kPdf},            {"RIFF\x10\x00\x00\x00WAVEfmt "s, FileKind::kUnknown},
      {"GIF90a"s, FileKind::kUnknown},            {"\x89PN"s, FileKind::kUnknown},
      {"P9"s, FileKind::kUnknown},                {"not an image"s, FileKind::kUnknown},
      {""s, FileKind::kUnknown}};
  for (const auto& [s, kind] : heads) {
    EXPECT_EQ(file_kind(std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(s.data()), s.size())), kind) << s;
  }
  EXPECT_EQ(file_kind_name(FileKind::kWebp), "webp");
  EXPECT_EQ(file_kind_name(FileKind::kUnknown), "unknown");
  EXPECT_EQ(file_kind(fs::path("no-such-file.png")), FileKind::kUnknown);
  try {
    decode_image(bytes_of("%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"));
    ADD_FAILURE() << "a PDF read as an image";
  } catch (const Error& e) {
    EXPECT_NE(std::string(e.what()).find("a PDF"), std::string::npos) << e.what();
  }
}

// Files whose names say another format (a WebP and a GIF named .png, a BMP named .jpg): read by what they are.
TEST(ImageFormats, FilesReadByWhatTheyAreNotTheirNames) {
  std::mt19937 rng(67);
  const fs::path dir = fs::temp_directory_path() / "carbon-image-formats-test";
  fs::create_directories(dir);
  const std::vector<uint8_t> rgb = pixels(33, 21, 3, rng);
  const std::vector<uint8_t> four = palette(4, rng);
  const GifFrame f = frame(0, 0, 33, 21, 4, rng);
  const std::pair<std::string, std::vector<uint8_t>> files[] = {
      {"webp.png", webp(rgb, 33, 21, 3, true)}, {"gif.png", gif(33, 21, four, {f})}, {"bmp.jpg", bmp24(rgb, 33, 21)}};
  for (const auto& [name, data] : files) {
    {
      std::ofstream o(dir / name, std::ios::binary);
      o.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }
    EXPECT_EQ(file_kind(dir / name), file_kind(data)) << name;
    EXPECT_EQ(load_image(dir / name).pixels, decode_image(data).pixels) << name;
  }
  EXPECT_EQ(load_image(dir / "webp.png").pixels, page_of(rgb.data(), 33, 21, 3));
  EXPECT_EQ(load_image(dir / "gif.png").pixels, gif_page(33, 21, four, f));
  fs::remove_all(dir);
}

}  // namespace
}  // namespace carbon
