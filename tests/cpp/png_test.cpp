// PNG pages (carbon/src/png.cpp) against stb_image, byte for byte, with every
// instruction set's kernels: PNGs made here of every colour type and bit depth, every
// filter, odd widths, runs of Paeth rows, several IDAT chunks, tRNS; and the files the
// decoder leaves to stb_image (interlaced, damaged, odd), which must come out as
// stb_image's, errors included.

#include <gtest/gtest.h>
#include <libdeflate.h>
#include <stb_image.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "carbon/cpu.h"
#include "carbon/error.h"
#include "carbon/image.h"

namespace carbon {
namespace {

// ------------------------------------------------------------------ writing PNGs

// An image's samples, channels a pixel (a palette index is one), row-major.
struct Samples {
  int w = 0, h = 0, depth = 8, color = 0;
  std::vector<uint32_t> s;

  int channels() const { return color == 2 ? 3 : color == 4 ? 2 : color == 6 ? 4 : 1; }
  uint32_t at(int x, int y, int c) const { return s[(static_cast<size_t>(y) * w + x) * channels() + c]; }
};

// Random samples below `top`, or a gradient with a little noise (smooth == true), which
// gives the filters small residuals as a page does.
Samples samples(int w, int h, int depth, int color, uint32_t top, bool smooth, std::mt19937& rng) {
  Samples im{w, h, depth, color, {}};
  im.s.resize(static_cast<size_t>(w) * h * im.channels());
  size_t i = 0;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int c = 0; c < im.channels(); ++c, ++i) {
        const uint32_t v = smooth ? static_cast<uint32_t>(x * 3 + y * 5 + c * 40) + rng() % 3 : rng();
        im.s[i] = v % top;
      }
    }
  }
  return im;
}

// The bytes of the pixels xs of row y, packed as PNG packs them.
std::vector<uint8_t> pack(const Samples& im, int y, const std::vector<int>& xs) {
  std::vector<uint8_t> out;
  uint32_t acc = 0;
  int bits = 0;
  for (int x : xs) {
    for (int c = 0; c < im.channels(); ++c) {
      const uint32_t v = im.at(x, y, c);
      if (im.depth == 16) {
        out.push_back(static_cast<uint8_t>(v >> 8));
        out.push_back(static_cast<uint8_t>(v));
      } else if (im.depth == 8) {
        out.push_back(static_cast<uint8_t>(v));
      } else {
        acc = acc << im.depth | v;
        bits += im.depth;
        if (bits == 8) {
          out.push_back(static_cast<uint8_t>(acc));
          acc = 0;
          bits = 0;
        }
      }
    }
  }
  if (bits) out.push_back(static_cast<uint8_t>(acc << (8 - bits)));
  return out;
}

int paeth(int a, int b, int c) {  // the PNG specification's
  const int p = a + b - c, pa = std::abs(p - a), pb = std::abs(p - b), pc = std::abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

// A row filtered with `filter` against the row above (unfiltered; empty: zeros), bpp
// bytes a pixel as stb_image counts them.
std::vector<uint8_t> filter_row(int filter, const std::vector<uint8_t>& row, const std::vector<uint8_t>& above,
                                size_t bpp) {
  std::vector<uint8_t> out(row.size() + 1);
  out[0] = static_cast<uint8_t>(filter);
  for (size_t i = 0; i < row.size(); ++i) {
    const int a = i >= bpp ? row[i - bpp] : 0, b = above.empty() ? 0 : above[i];
    const int c = i >= bpp && !above.empty() ? above[i - bpp] : 0;
    const int pred = filter == 1 ? a : filter == 2 ? b : filter == 3 ? (a + b) / 2 : filter == 4 ? paeth(a, b, c) : 0;
    out[i + 1] = static_cast<uint8_t>(row[i] - pred);
  }
  return out;
}

void put32(std::vector<uint8_t>& v, uint32_t x) {
  for (int s = 24; s >= 0; s -= 8) v.push_back(static_cast<uint8_t>(x >> s));
}

void chunk(std::vector<uint8_t>& out, const char* type, const std::vector<uint8_t>& data) {
  put32(out, static_cast<uint32_t>(data.size()));
  const size_t at = out.size();
  out.insert(out.end(), type, type + 4);
  out.insert(out.end(), data.begin(), data.end());
  put32(out, libdeflate_crc32(0, out.data() + at, data.size() + 4));
}

std::vector<uint8_t> zlib(const std::vector<uint8_t>& raw) {
  libdeflate_compressor* c = libdeflate_alloc_compressor(6);
  std::vector<uint8_t> z(libdeflate_zlib_compress_bound(c, raw.size()));
  z.resize(libdeflate_zlib_compress(c, raw.data(), raw.size(), z.data(), z.size()));
  libdeflate_free_compressor(c);
  return z;
}

// How a PNG is written.
struct Write {
  std::function<int(int)> filter = [](int y) { return y % 5; };  // a row's filter
  bool interlace = false;
  size_t idat_piece = 0;              // the zlib stream in IDAT chunks of this many bytes (0: one)
  std::vector<uint8_t> palette;       // RGB entries (colour type 3; or a suggested palette)
  std::vector<uint8_t> trns;          // the tRNS chunk's data, if any
  std::function<void(std::vector<uint8_t>&)> edit_raw;   // changes to the filtered rows before deflating
  std::function<void(std::vector<uint8_t>&)> edit_zlib;  // changes to the zlib stream
  bool text = false;                  // tEXt chunks before and among the IDAT chunks
};

// The filtered rows: the image's, or Adam7's seven passes' one after another.
std::vector<uint8_t> filtered(const Samples& im, const Write& how) {
  const size_t bpp = im.depth < 8 ? 1 : static_cast<size_t>(im.channels() * im.depth / 8);
  std::vector<uint8_t> raw;
  auto image = [&](int x0, int y0, int dx, int dy) {
    std::vector<int> xs;
    for (int x = x0; x < im.w; x += dx) xs.push_back(x);
    if (xs.empty()) return;
    std::vector<uint8_t> above;
    for (int y = y0, k = 0; y < im.h; y += dy, ++k) {
      const std::vector<uint8_t> row = pack(im, y, xs);
      const std::vector<uint8_t> f = filter_row(how.filter(k), row, above, bpp);
      raw.insert(raw.end(), f.begin(), f.end());
      above = row;
    }
  };
  if (!how.interlace) {
    image(0, 0, 1, 1);
  } else {
    const int x0[7] = {0, 4, 0, 2, 0, 1, 0}, y0[7] = {0, 0, 4, 0, 2, 0, 1};
    const int dx[7] = {8, 8, 4, 4, 2, 2, 1}, dy[7] = {8, 8, 8, 4, 4, 2, 2};
    for (int p = 0; p < 7; ++p) image(x0[p], y0[p], dx[p], dy[p]);
  }
  return raw;
}

std::vector<uint8_t> png(const Samples& im, const Write& how = {}) {
  std::vector<uint8_t> raw = filtered(im, how);
  if (how.edit_raw) how.edit_raw(raw);
  std::vector<uint8_t> z = zlib(raw);
  if (how.edit_zlib) how.edit_zlib(z);
  std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
  std::vector<uint8_t> ihdr;
  put32(ihdr, static_cast<uint32_t>(im.w));
  put32(ihdr, static_cast<uint32_t>(im.h));
  ihdr.insert(ihdr.end(), {static_cast<uint8_t>(im.depth), static_cast<uint8_t>(im.color), 0, 0,
                           static_cast<uint8_t>(how.interlace)});
  chunk(out, "IHDR", ihdr);
  const std::vector<uint8_t> text = {'C', 'o', 'm', 'm', 'e', 'n', 't', 0, 'h', 'i'};
  if (how.text) chunk(out, "tEXt", text);
  if (!how.palette.empty()) chunk(out, "PLTE", how.palette);
  if (!how.trns.empty()) chunk(out, "tRNS", how.trns);
  const size_t piece = how.idat_piece ? how.idat_piece : z.size();
  for (size_t at = 0; at < z.size(); at += piece) {
    chunk(out, "IDAT", std::vector<uint8_t>(z.begin() + at, z.begin() + std::min(z.size(), at + piece)));
    if (how.text && at == 0) chunk(out, "tEXt", text);
  }
  chunk(out, "IEND", {});
  return out;
}

std::vector<uint8_t> palette(int n, std::mt19937& rng) {
  std::vector<uint8_t> p(static_cast<size_t>(n) * 3);
  for (auto& v : p) v = static_cast<uint8_t>(rng());
  return p;
}

// ------------------------------------------------------------------ against stb_image

// decode_image with every instruction set against stbi_load_from_memory(..., 1): the same
// bytes, or the same error.  Whether stb_image read the file.
bool as_stb(const std::vector<uint8_t>& file, const std::string& what) {
  int w = 0, h = 0, n = 0;
  stbi_uc* ref = stbi_load_from_memory(file.data(), static_cast<int>(file.size()), &w, &h, &n, 1);
  const std::string reason = ref ? "" : stbi_failure_reason();
  for (cpu::Isa isa : cpu::isas_available()) {
    const std::string at = what + " (" + cpu::isa_name(isa) + ")";
    if (!ref) {
      try {
        decode_image(file, isa);
        ADD_FAILURE() << at << ": read, where stb_image says " << reason;
      } catch (const Error& e) {
        EXPECT_EQ(std::string(e.what()), "the image could not be read: " + reason) << at;
      }
      continue;
    }
    const GrayImage g = decode_image(file, isa);
    EXPECT_EQ(g.width, w) << at;
    EXPECT_EQ(g.height, h) << at;
    if (g.width != w || g.height != h) continue;
    const auto miss = std::mismatch(g.pixels.begin(), g.pixels.end(), ref);
    if (miss.first != g.pixels.end()) {
      const auto i = miss.first - g.pixels.begin();
      ADD_FAILURE() << at << ": pixel (" << i % w << ", " << i / w << ") is " << int{*miss.first} << ", stb_image's "
                    << int{*miss.second};
    }
  }
  stbi_image_free(ref);
  return ref != nullptr;
}

std::string name(const Samples& im, const std::string& more = "") {
  return "colour " + std::to_string(im.color) + ", " + std::to_string(im.depth) + " bits, " + std::to_string(im.w) +
         " x " + std::to_string(im.h) + (more.empty() ? "" : ", " + more);
}

struct Kind {
  int color, depth;
};
const Kind kKinds[] = {{0, 1}, {0, 2}, {0, 4}, {0, 8}, {0, 16}, {2, 8}, {2, 16}, {3, 1},
                       {3, 2}, {3, 4}, {3, 8}, {4, 8}, {4, 16}, {6, 8}, {6, 16}};

// Every colour type and bit depth PNG allows, odd widths, the filters in turn, all Paeth,
// and at random; random and smooth samples.
TEST(Png, EveryKindAsStbImage) {
  std::mt19937 rng(7);
  const std::function<int(int)> filters[] = {[](int y) { return y % 5; }, [](int) { return 4; },
                                             [&rng](int) { return static_cast<int>(rng() % 5); }};
  const char* filter_names[] = {"filters in turn", "Paeth", "filters at random"};
  for (const Kind& k : kKinds) {
    for (int w : {1, 3, 17, 33, 61, 250}) {
      for (int f = 0; f < 3; ++f) {
        for (bool smooth : {false, true}) {
          const uint32_t top = k.color == 3 ? (k.depth == 8 ? 256u : 1u << k.depth) : 1u << k.depth;
          const Samples im = samples(w, 37, k.depth, k.color, top, smooth, rng);
          Write how;
          how.filter = filters[f];
          if (k.color == 3) how.palette = palette(static_cast<int>(top), rng);
          as_stb(png(im, how), name(im, filter_names[f]) + (smooth ? ", smooth" : ""));
        }
      }
    }
  }
}

// Runs of Paeth rows of every length to 40 among Up rows, wide enough for the
// instruction sets' 16-row wavefront, at every number of bytes a pixel.
TEST(Png, RunsOfPaethRowsAsStbImage) {
  std::mt19937 rng(11);
  const Kind kinds[] = {{0, 8}, {0, 1}, {0, 16}, {2, 8}, {3, 8}, {4, 8}, {6, 8}, {2, 16}, {6, 16}};
  for (const Kind& k : kinds) {
    for (int w : {32, 33, 47, 100, 517}) {
      for (bool smooth : {false, true}) {
        const uint32_t top = k.color == 3 ? 256u : 1u << k.depth;
        const Samples im = samples(w, 120, k.depth, k.color, top, smooth, rng);
        Write how;
        // rows in runs: 1 Paeth, 1 Up, 2 Paeth, 1 Up, ..., up to 40 Paeth
        std::vector<int> f;
        for (int run = 1; f.size() < 120; ++run) {
          f.insert(f.end(), static_cast<size_t>(run), 4);
          f.push_back(run % 3 == 0 ? 1 : 2);
        }
        how.filter = [f](int y) { return f[static_cast<size_t>(y)]; };
        if (k.color == 3) how.palette = palette(256, rng);
        as_stb(png(im, how), name(im, smooth ? "Paeth runs, smooth" : "Paeth runs"));
        how.filter = [](int) { return 4; };
        as_stb(png(im, how), name(im, smooth ? "all Paeth, smooth" : "all Paeth"));
      }
    }
  }
}

// Blocks of 16 rows that are mostly Paeth with every other filter among them (the wavefront
// takes such a block whole, each lane by its row's filter), and blocks just under and over
// its threshold, at every number of bytes a pixel it takes.
TEST(Png, MixedFiltersInTheWavefrontAsStbImage) {
  std::mt19937 rng(23);
  const Kind kinds[] = {{0, 8}, {2, 8}, {6, 8}, {4, 8}};
  for (const Kind& k : kinds) {
    for (int w : {32, 33, 100, 517}) {
      for (int share : {30, 40, 60, 90}) {  // per cent of rows Paeth or Average
        const Samples im = samples(w, 97, k.depth, k.color, 1u << k.depth, share > 50, rng);
        Write how;
        std::vector<int> f(97);
        for (int& v : f) v = static_cast<int>(rng() % 100) < share ? 3 + static_cast<int>(rng() % 2) : static_cast<int>(rng() % 3);
        how.filter = [f](int y) { return f[static_cast<size_t>(y)]; };
        as_stb(png(im, how), name(im, "mixed filters, " + std::to_string(share) + "% serial"));
      }
    }
  }
}

// tRNS (an alpha the gray drops), short palettes, a palette in an RGB file, IDAT in
// pieces down to a byte, chunks among them.
TEST(Png, ChunksAsStbImage) {
  std::mt19937 rng(13);
  for (const Kind& k : kKinds) {
    const bool pal = k.color == 3;
    if (k.color == 4 || k.color == 6) continue;  // tRNS with alpha: stb_image refuses it (below)
    const uint32_t top = pal ? std::min(5u, 1u << k.depth) : 1u << k.depth;
    const Samples im = samples(45, 21, k.depth, k.color, top, false, rng);
    Write how;
    if (pal) {
      how.palette = palette(5, rng);
      how.trns = {0, 128, 255};
    } else {
      how.trns.assign(static_cast<size_t>(im.channels()) * 2, 0);
      how.palette = k.color == 2 ? palette(7, rng) : std::vector<uint8_t>{};
    }
    EXPECT_TRUE(as_stb(png(im, how), name(im, "tRNS")));
    how.idat_piece = 1;
    how.text = true;
    EXPECT_TRUE(as_stb(png(im, how), name(im, "tRNS, IDAT a byte a chunk, tEXt")));
    how.idat_piece = 97;
    EXPECT_TRUE(as_stb(png(im, how), name(im, "tRNS, IDAT 97 bytes a chunk, tEXt")));
  }
}

// Adam7: left to stb_image.
TEST(Png, InterlacedAsStbImage) {
  std::mt19937 rng(17);
  for (const Kind& k : kKinds) {
    for (int w : {1, 7, 33}) {
      const uint32_t top = k.color == 3 ? 1u << std::min(k.depth, 8) : 1u << k.depth;
      const Samples im = samples(w, 19, k.depth, k.color, top, false, rng);
      Write how;
      how.interlace = true;
      if (k.color == 3) how.palette = palette(static_cast<int>(top), rng);
      EXPECT_TRUE(as_stb(png(im, how), name(im, "interlaced")));
    }
  }
}

// Files the decoder leaves to stb_image, which reads some and refuses others.
TEST(Png, DamagedAndOddFilesAsStbImage) {
  std::mt19937 rng(19);
  const Samples gray = samples(40, 30, 8, 0, 256, false, rng);
  const Samples rgb = samples(40, 30, 8, 2, 256, false, rng);
  const std::vector<uint8_t> good = png(gray);
  EXPECT_TRUE(as_stb(good, "good"));

  // read by both: CRCs (stb_image checks none), the Adler-32 (libdeflate's check fails),
  // more data than the rows, a palette index past a short palette is stb_image's own
  std::vector<uint8_t> f = good;
  f[29] ^= 0xff;  // IHDR's CRC
  EXPECT_TRUE(as_stb(f, "a bad CRC"));
  Write how;
  how.edit_zlib = [](std::vector<uint8_t>& z) { z.back() ^= 1; };
  EXPECT_TRUE(as_stb(png(gray, how), "a bad Adler-32"));
  how = {};
  how.edit_raw = [](std::vector<uint8_t>& raw) { raw.insert(raw.end(), 100, 0); };
  EXPECT_TRUE(as_stb(png(gray, how), "data past the rows"));
  Samples odd = samples(40, 30, 4, 2, 16, false, rng);  // RGB of 4 bits, which PNG does not allow
  EXPECT_TRUE(as_stb(png(odd), "RGB of 4 bits"));
  odd = samples(40, 30, 2, 4, 4, false, rng);
  EXPECT_TRUE(as_stb(png(odd), "gray + alpha of 2 bits"));

  // refused by both, with stb_image's message
  how = {};
  how.edit_raw = [](std::vector<uint8_t>& raw) { raw[41 * 3] = 5; };
  EXPECT_FALSE(as_stb(png(gray, how), "an invalid filter"));
  how.edit_raw = [](std::vector<uint8_t>& raw) { raw.resize(raw.size() - 41); };
  EXPECT_FALSE(as_stb(png(gray, how), "a row short"));
  how.edit_zlib = [](std::vector<uint8_t>& z) { z.resize(z.size() / 2); };
  how.edit_raw = nullptr;
  EXPECT_FALSE(as_stb(png(gray, how), "the zlib stream cut short"));
  f = good;
  f.resize(f.size() - 20);
  EXPECT_FALSE(as_stb(f, "the file cut short"));
  f = std::vector<uint8_t>(good.begin(), good.begin() + 8);
  chunk(f, "zzZz", {1, 2, 3});  // a chunk before IHDR
  f.insert(f.end(), good.begin() + 8, good.end());
  EXPECT_FALSE(as_stb(f, "a chunk before IHDR"));
  f = std::vector<uint8_t>(good.begin(), good.begin() + 33);
  chunk(f, "ZZZZ", {1, 2, 3});  // an unknown critical chunk
  f.insert(f.end(), good.begin() + 33, good.end());
  EXPECT_FALSE(as_stb(f, "an unknown critical chunk"));
  const Samples pal = samples(40, 30, 8, 3, 256, false, rng);
  EXPECT_FALSE(as_stb(png(pal), "a palette's indices without PLTE"));
  how = {};
  how.trns = {0, 0, 0, 0};
  EXPECT_FALSE(as_stb(png(rgb, how), "tRNS of the wrong length"));
  const Samples rgba = samples(40, 30, 8, 6, 256, false, rng);
  how.trns = {0, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_FALSE(as_stb(png(rgba, how), "tRNS with an alpha channel"));
  f = good;
  f[24] = 3;  // a bit depth of 3
  EXPECT_FALSE(as_stb(f, "a bit depth of 3"));
}

// The pages the engine reads are 8-bit gray from rendering at 200 dpi: a large one, then
// small ones, so each thread's kept buffers are reused with other sizes.
TEST(Png, BuffersKeptBetweenPagesAsStbImage) {
  std::mt19937 rng(23);
  for (int w : {1700, 9, 300, 1700}) {
    const Samples im = samples(w, w > 100 ? 220 : 5, 8, 0, 256, true, rng);
    Write how;
    how.filter = [](int y) { return (y / 17) % 2 ? 4 : y % 5; };
    how.idat_piece = 8192;
    EXPECT_TRUE(as_stb(png(im, how), name(im)));
  }
}

TEST(Png, OnlyTheInstructionSetsThisCpuHas) {
  const std::vector<uint8_t> file = png(Samples{1, 1, 8, 0, {0}});
  for (cpu::Isa isa : {cpu::Isa::kNeon, cpu::Isa::kAvx2, cpu::Isa::kAvxVnni, cpu::Isa::kAvx512, cpu::Isa::kNeonDot}) {
    if (!cpu::isa_available(isa)) EXPECT_THROW(decode_image(file, isa), Error) << cpu::isa_name(isa);
  }
}

}  // namespace
}  // namespace carbon
