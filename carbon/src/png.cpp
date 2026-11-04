// PNG pages straight to gray (carbon/src/png.h).
//
// The chunks are walked as stb_image walks them (stbi__parse_png_file in stb_image.h),
// and wherever stb_image would fail, or would do something this file does not repeat,
// the page goes back to stb_image, so its result is stb_image's, error message
// included.  What is left is the common case, done fast:
//
//   - the IDAT chunks' zlib stream inflated by libdeflate in one call into a buffer of
//     exactly the rows' bytes.  A stream that inflates to more goes back to stb_image,
//     which reads it too, and so does one whose Adler-32 is wrong, which libdeflate
//     rejects and stb_image does not check.  The other way round, libdeflate reads the
//     length codes 286 and 287, the distance codes 30 and 31 and a lone length-1 code
//     word sent as 1, all of which DEFLATE forbids and stb_image rejects: only a file
//     made so has them and the right Adler-32 too.
//   - the rows unfiltered by the CPU kernels (carbon/src/cpu/png_kernels.h), a row or a
//     block of rows at a time (the row above the first as zeros), then turned to
//     gray from the bytes still in the cache, as stbi__convert_format and
//     stbi__convert_16_to_8 do it:
//
//       gray            the sample; 16 bits: its high byte; 1, 2 and 4 bits: times 255,
//                       85 or 17
//       gray + alpha    the gray sample, as above (alpha dropped)
//       RGB, RGBA       (77 r + 150 g + 29 b) >> 8; 16 bits: (77 r + 150 g + 29 b) >> 16
//       palette         the entry's RGB as for RGB (1 to 8 bits an index)
//
//     tRNS gives the others an alpha channel, which the gray drops as well.  8-bit gray,
//     most pages, is unfiltered straight into the image's pixels, the rest in place in
//     the inflated buffer and converted from there.
//
// The inflated rows, and the IDAT chunks end to end when there are several, are in
// buffers each thread keeps from page to page: a fresh page of memory costs the
// operating system's zeroing on first touch, about half a millisecond a megabyte on
// Windows, as much as inflating it.  A buffer past 64 MiB is let go after its page.

#include "png.h"

#include <libdeflate.h>

#include <climits>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include "cpu/png_kernels.h"

namespace carbon::detail {
namespace {

using cpu::detail::PngRows;

constexpr uint32_t chunk_type(const char (&s)[5]) {
  return static_cast<uint32_t>(static_cast<uint8_t>(s[0])) << 24 |
         static_cast<uint32_t>(static_cast<uint8_t>(s[1])) << 16 |
         static_cast<uint32_t>(static_cast<uint8_t>(s[2])) << 8 | static_cast<uint8_t>(s[3]);
}
constexpr uint32_t kIHDR = chunk_type("IHDR"), kPLTE = chunk_type("PLTE"), kTRNS = chunk_type("tRNS"),
                   kIDAT = chunk_type("IDAT"), kIEND = chunk_type("IEND"), kCgBI = chunk_type("CgBI");

uint32_t be32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) << 24 | static_cast<uint32_t>(p[1]) << 16 | static_cast<uint32_t>(p[2]) << 8 |
         p[3];
}

// libdeflate's decompressor holds its tables (some tens of KB): one per thread, kept.
libdeflate_decompressor* decompressor() {
  struct Owned {
    libdeflate_decompressor* d = libdeflate_alloc_decompressor();
    ~Owned() { libdeflate_free_decompressor(d); }
  };
  thread_local Owned owned;
  return owned.d;
}

// A thread's buffer of at least n bytes (above), and its letting go of a large one.
constexpr size_t kKeep = size_t{64} << 20;

uint8_t* scratch(std::vector<uint8_t>& v, size_t n) {
  if (v.size() < n) {
    v.clear();
    v.resize(n);
  }
  return v.data();
}

struct Scratch {
  std::vector<uint8_t> rows, zlib;
};

struct Release {
  Scratch& s;
  ~Release() {
    for (std::vector<uint8_t>* v : {&s.rows, &s.zlib}) {
      if (v->size() > kKeep) std::vector<uint8_t>().swap(*v);
    }
  }
};

// ------------------------------------------------------------------ kernels

// The reference: png_kernels.h's scalar code, a row at a time.
size_t unfilter_scalar(const PngRows& r) {
  cpu::detail::png_unfilter_row(r.raw[-1], r.raw, r.prior, r.cur, r.n, r.bpp);
  return 1;
}

void gray_rgb8_scalar(const uint8_t* px, uint8_t* out, size_t w, int step) {
  for (size_t x = 0; x < w; ++x, px += step) {
    out[x] = static_cast<uint8_t>((px[0] * 77 + px[1] * 150 + px[2] * 29) >> 8);
  }
}

struct Kernels {
  size_t (*unfilter)(const PngRows&);
  void (*gray_rgb8)(const uint8_t*, uint8_t*, size_t, int);
};

// AVX2 for every x86-64 set, NEON for Arm, as the depthwise kernels.
Kernels kernels(cpu::Isa isa) {
  switch (isa) {
#if defined(CARBON_KERNEL_AVX2)
    case cpu::Isa::kAvx2:
    case cpu::Isa::kAvxVnni:
    case cpu::Isa::kAvx512: return {cpu::detail::png_unfilter_avx2, cpu::detail::png_gray_rgb8_avx2};
#endif
#if defined(CARBON_KERNEL_NEON)
    case cpu::Isa::kNeon:
    case cpu::Isa::kNeonDot: return {cpu::detail::png_unfilter_neon, cpu::detail::png_gray_rgb8_neon};
#endif
    default: return {unfilter_scalar, gray_rgb8_scalar};
  }
}

// ------------------------------------------------------------------ the file

struct Png {
  uint32_t width = 0, height = 0;
  int depth = 0, color = 0;
  int channels = 0;     // samples a pixel: 1 gray or a palette index, 2 gray + alpha, 3 RGB, 4 RGBA
  bool trns = false;    // a tRNS chunk: stb_image adds an alpha channel to all but a palette's
  int palette_len = 0;  // entries
  uint8_t palette[256][3] = {};
  const uint8_t* zlib = nullptr;  // the IDAT chunks' data end to end
  size_t zlib_len = 0;
};

// The chunks up to IEND, as stbi__parse_png_file reads them; false wherever it would
// fail, and for what this file leaves to it.  No CRC is checked, as stb_image checks none.
bool parse(const uint8_t* data, size_t size, Png& p, Scratch& s) {
  static const uint8_t kSignature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  if (size < 8 || std::memcmp(data, kSignature, 8) != 0 || size > INT_MAX) return false;
  std::vector<std::pair<const uint8_t*, uint32_t>> idat;
  bool first = true, any_idat = false;
  for (size_t at = 8;;) {
    if (size - at < 8) return false;
    const uint32_t len = be32(data + at), type = be32(data + at + 4);
    const uint8_t* d = data + at + 8;
    if (type == kIEND) break;  // its length and CRC are not read
    if (size - at - 8 < 4 || len > size - at - 12) return false;  // the data and the CRC in the file
    switch (type) {
      case kIHDR: {
        if (!first || len != 13) return false;
        first = false;
        p.width = be32(d);
        p.height = be32(d + 4);
        p.depth = d[8];
        p.color = d[9];
        // stb_image's STBI_MAX_DIMENSIONS (carbon/src/image.cpp); compression and filter
        // method 0; Adam7 interlacing left to it
        if (!p.width || !p.height || p.width > 32768 || p.height > 32768 || d[10] || d[11] || d[12]) return false;
        const int depth = p.depth;
        const bool sub8 = depth == 1 || depth == 2 || depth == 4;
        switch (p.color) {
          case 0: p.channels = 1; break;
          case 2: p.channels = 3; break;
          case 3: p.channels = 1; break;
          case 4: p.channels = 2; break;
          case 6: p.channels = 4; break;
          default: return false;
        }
        // The depths PNG allows each type: stb_image reads 1, 2 and 4 bits of RGB and
        // alpha types too, its own way.
        if (depth != 8 && !(depth == 16 && p.color != 3) && !(sub8 && (p.color == 0 || p.color == 3))) return false;
        if ((1u << 30) / p.width / static_cast<uint32_t>(p.color == 3 ? 4 : p.channels) < p.height) return false;
        break;
      }
      case kPLTE:
        if (first || len > 256 * 3 || len % 3) return false;
        p.palette_len = static_cast<int>(len / 3);
        std::memcpy(p.palette, d, len);
        break;
      case kTRNS:
        if (first || any_idat) return false;
        if (p.color == 3) {
          if (p.palette_len == 0 || len > static_cast<uint32_t>(p.palette_len)) return false;
        } else {
          if (!(p.channels & 1) || len != 2u * static_cast<uint32_t>(p.channels)) return false;
          p.trns = true;
        }
        break;  // the alpha it gives, the gray drops
      case kIDAT:
        if (first || (p.color == 3 && p.palette_len == 0) || len > (1u << 30)) return false;
        if (p.zlib_len + len > INT_MAX) return false;
        idat.emplace_back(d, len);
        p.zlib_len += len;
        any_idat = any_idat || len;
        break;
      default:
        // stb_image rejects a critical chunk it does not know and anything before IHDR,
        // skips the other ancillary chunks, and reads CgBI's files its own way.
        if (first || type == kCgBI || !(type & (1u << 29))) return false;
        break;
    }
    at += 12 + static_cast<size_t>(len);
  }
  if (first || !any_idat) return false;
  // stb_image's buffers, each of which must fit an int: the pixels as it unfilters them
  // (a channel more with tRNS), the inflated rows, and a few of 4 bytes a pixel.
  const uint64_t px = static_cast<uint64_t>(p.width) * p.height;
  const uint64_t row = (static_cast<uint64_t>(p.channels) * p.width * p.depth + 7) / 8;
  const uint64_t out_n = static_cast<uint64_t>(p.channels) + (p.trns ? 1 : 0);
  if (px * out_n * (p.depth == 16 ? 2 : 1) > INT_MAX || row * p.height + row > INT_MAX || px * 4 > INT_MAX) {
    return false;
  }
  if (idat.size() == 1) {
    p.zlib = idat[0].first;
  } else {
    uint8_t* z = scratch(s.zlib, p.zlib_len);
    for (const auto& [d, len] : idat) {
      std::memcpy(z, d, len);
      z += len;
    }
    p.zlib = s.zlib.data();
  }
  return true;
}

// The gray of a palette entry, or 0x100 for an index past the palette (stb_image reads
// whatever its buffer holds there): the entries a page used, OR-ed together, say
// whether it used one.
void palette_gray(const Png& p, uint16_t lut[256]) {
  for (int i = 0; i < 256; ++i) {
    const uint8_t* e = p.palette[i];
    lut[i] = i < p.palette_len ? static_cast<uint16_t>((e[0] * 77 + e[1] * 150 + e[2] * 29) >> 8) : uint16_t{0x100};
  }
}

// A row of indices or gray samples of D bits (1, 2 or 4) through lut: each byte's
// 8 / D pixels from a table made once for the image.
template <int D>
struct Unpack {
  static constexpr int kPer = 8 / D;
  uint8_t table[256][kPer];
  uint16_t used[256];  // the lut values of the byte's samples, OR-ed

  explicit Unpack(const uint16_t* lut) {
    for (int b = 0; b < 256; ++b) {
      used[b] = 0;
      for (int k = 0; k < kPer; ++k) {
        const uint16_t v = lut[(b >> (8 - D * (k + 1))) & ((1 << D) - 1)];
        table[b][k] = static_cast<uint8_t>(v);
        used[b] = static_cast<uint16_t>(used[b] | v);
      }
    }
  }
  // A row of w pixels; its last byte's bits past them are not read, as in stb_image.
  void row(const uint8_t* in, uint8_t* out, size_t w, const uint16_t* lut, uint16_t& seen) const {
    const size_t whole = w / kPer;
    uint16_t s = 0;
    for (size_t i = 0; i < whole; ++i) {
      std::memcpy(out + i * kPer, table[in[i]], kPer);
      s = static_cast<uint16_t>(s | used[in[i]]);
    }
    for (size_t x = whole * kPer, k = 0; x < w; ++x, ++k) {
      const uint16_t v = lut[(in[whole] >> (8 - D * (k + 1))) & ((1 << D) - 1)];
      out[x] = static_cast<uint8_t>(v);
      s = static_cast<uint16_t>(s | v);
    }
    seen = static_cast<uint16_t>(seen | s);
  }
};

// One unfiltered row to gray, for all but 8-bit gray.
class ToGray {
 public:
  ToGray(const Png& p, const Kernels& k) : p_(p), k_(k) {
    if (p.color == 3) {
      palette_gray(p, lut_);
    } else if (p.depth < 8) {
      static const uint8_t kScale[5] = {0, 0xff, 0x55, 0, 0x11};  // stbi__depth_scale_table
      for (int i = 0; i < (1 << p.depth); ++i) lut_[i] = static_cast<uint16_t>(i * kScale[p.depth]);
    }
    if (p.depth == 1) unpack1_.emplace(lut_);
    if (p.depth == 2) unpack2_.emplace(lut_);
    if (p.depth == 4) unpack4_.emplace(lut_);
  }

  void row(const uint8_t* in, uint8_t* out, size_t w) {
    switch (p_.depth) {
      case 1: unpack1_->row(in, out, w, lut_, seen_); return;
      case 2: unpack2_->row(in, out, w, lut_, seen_); return;
      case 4: unpack4_->row(in, out, w, lut_, seen_); return;
      default: break;
    }
    const bool wide = p_.depth == 16;
    const size_t step = static_cast<size_t>(p_.channels) * (wide ? 2 : 1);
    switch (p_.color) {
      case 0:  // gray 16: the high byte
      case 4:  // gray + alpha: the gray sample's (high) byte
        for (size_t x = 0; x < w; ++x) out[x] = in[x * step];
        return;
      case 2:
      case 6:
        if (!wide) {
          k_.gray_rgb8(in, out, w, p_.channels);
          return;
        }
        for (size_t x = 0; x < w; ++x, in += step) {
          const uint32_t r = static_cast<uint32_t>(in[0]) << 8 | in[1], g = static_cast<uint32_t>(in[2]) << 8 | in[3],
                         b = static_cast<uint32_t>(in[4]) << 8 | in[5];
          out[x] = static_cast<uint8_t>((r * 77 + g * 150 + b * 29) >> 16);
        }
        return;
      default: {  // 3, 8-bit indices
        uint16_t s = 0;
        for (size_t x = 0; x < w; ++x) {
          const uint16_t v = lut_[in[x]];
          out[x] = static_cast<uint8_t>(v);
          s = static_cast<uint16_t>(s | v);
        }
        seen_ = static_cast<uint16_t>(seen_ | s);
        return;
      }
    }
  }

  bool past_palette() const { return seen_ & 0x100; }  // an index past the palette: stb_image's

 private:
  const Png& p_;
  const Kernels& k_;
  uint16_t lut_[256] = {};
  uint16_t seen_ = 0;  // the lut values used, OR-ed (palette_gray)
  std::optional<Unpack<1>> unpack1_;  // made for its depth only
  std::optional<Unpack<2>> unpack2_;
  std::optional<Unpack<4>> unpack4_;
};

}  // namespace

bool png_to_gray(std::span<const uint8_t> png, cpu::Isa isa, GrayImage& out) {
  thread_local Scratch s;
  const Release release{s};
  Png p;
  if (!parse(png.data(), png.size(), p, s)) return false;
  const Kernels k = kernels(isa);
  const size_t w = p.width, h = p.height;
  const size_t n = (static_cast<size_t>(p.channels) * w * p.depth + 7) / 8;  // a row's bytes past its filter byte
  const size_t stride = n + 1;
  uint8_t* rows = scratch(s.rows, stride * h);
  libdeflate_decompressor* d = decompressor();
  if (!d || libdeflate_zlib_decompress(d, p.zlib, p.zlib_len, rows, stride * h, nullptr) != LIBDEFLATE_SUCCESS) {
    return false;
  }
  for (size_t y = 0; y < h; ++y) {
    if (rows[y * stride] > 4) return false;  // stb_image: "invalid filter"
  }
  out.width = static_cast<int>(w);
  out.height = static_cast<int>(h);
  out.pixels.resize(w * h);
  uint8_t* pixels = out.pixels.data();
  const std::vector<uint8_t> zeros(n, 0);
  PngRows r;
  r.raw_stride = stride;
  r.n = n;
  r.bpp = p.depth < 8 ? 1 : p.channels * p.depth / 8;  // stb_image's filter_bytes
  if (p.color == 0 && p.depth == 8) {  // straight into the pixels
    r.cur_stride = w;
    for (size_t y = 0; y < h;) {
      r.raw = rows + y * stride + 1;
      r.cur = pixels + y * w;
      r.prior = y ? r.cur - w : zeros.data();
      r.rows = h - y;
      y += k.unfilter(r);
    }
    return true;
  }
  ToGray gray(p, k);
  r.cur_stride = stride;
  for (size_t y = 0; y < h;) {
    r.raw = rows + y * stride + 1;
    r.cur = rows + y * stride + 1;
    r.prior = y ? r.cur - stride : zeros.data();
    r.rows = h - y;
    const size_t done = k.unfilter(r);
    for (size_t t = 0; t < done; ++t) gray.row(r.cur + t * stride, pixels + (y + t) * w, w);
    y += done;
  }
  return !gray.past_palette();
}

}  // namespace carbon::detail
