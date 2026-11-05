// The crop rule and CTC greedy decoding in CUDA (carbon/cuda.h): the same bytes
// as carbon/src/lines.cpp.  Compiled with -fmad=false, so the double-precision
// coefficient math rounds exactly as it does on the CPU (resample_math.h).
//
// Cropping is four launches for every line of every page: the horizontal
// coefficients, the horizontal pass into an 8-bit intermediate, the vertical
// pass, and the finish (finish_crop's): the ground, the light-on-dark flip, and
// the padding to the batch width.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "carbon/cuda.h"
#include "carbon/error.h"
#include "buffer.h"
#include "../resample_math.h"

namespace carbon::cuda {

namespace {

using mem::Buffer;
using mem::check;

struct LineMeta {
  double x0, x1;   // the horizontal extent of the crop region
  int width;       // the crop's own width
  int page;
  int top, rows;   // the source rows of the horizontal pass
  int64_t tmp;     // offset of this line's intermediate
};

constexpr int kThreads = 128;

// The fused crop: what a line's block needs, its region in both axes.
struct LineRegion {
  double x0, x1, y0, y1;
  int width, page;
};
constexpr int kMaxTaps = 32;  // the fused kernel's taps a column or a row, in registers and shared memory

// A block for each (128-column tile, line): the line's vertical coefficients made once into shared memory; each
// thread's horizontal taps for its column, in registers; the horizontal pass over the line's source rows into a
// shared tile [rows][128] (a column each thread's own, so it needs no barrier); the vertical pass from it. The
// bytes of horizontal_pass and vertical_pass: the same coefficients, sums and rounding.
__global__ void __launch_bounds__(kThreads) crop_fused(const LineRegion* lines, const DevicePage* pages, int height,
                                                       int out_width, int kv, uint8_t* out) {
  extern __shared__ int32_t smem[];
  const int i = blockIdx.y, tid = threadIdx.x;
  const LineRegion m = lines[i];
  const int xx = blockIdx.x * kThreads + tid;
  if (blockIdx.x * kThreads >= m.width) return;  // the whole block: past this line's columns
  int32_t* vk = smem;                       // [height][kv]
  int32_t* vfirst = vk + height * kv;       // [height]
  int32_t* vcount = vfirst + height;        // [height]
  uint8_t* tile = reinterpret_cast<uint8_t*>(vcount + height);  // [rows][kThreads]
  const detail::Filter fy = detail::filter_for(m.y0, m.y1, height);
  for (int yy = tid; yy < height; yy += kThreads) {
    int c = 0;
    vfirst[yy] = detail::coefficients(fy, m.y0, yy, vk + yy * kv, &c);
    vcount[yy] = c;
  }
  const bool live = xx < m.width;
  int32_t hk[kMaxTaps];
  int hcount = 0, hfirst = 0;
  if (live) hfirst = detail::coefficients(detail::filter_for(m.x0, m.x1, m.width), m.x0, xx, hk, &hcount);
  __syncthreads();
  if (!live) return;
  const int top = vfirst[0], rows = vfirst[height - 1] + vcount[height - 1] - top;
  const DevicePage p = pages[m.page];
  for (int r = 0; r < rows; ++r) {
    const int y = top + r;
    const uint8_t* row = (y >= 0 && y < p.height) ? p.data + static_cast<int64_t>(y) * p.stride : nullptr;
    int32_t ss = detail::kHalf;
    for (int t = 0; t < hcount; ++t) {
      const int x = hfirst + t;
      ss += ((row && x >= 0 && x < p.width) ? row[x] : 255) * hk[t];
    }
    tile[r * kThreads + tid] = detail::clip8(ss);
  }
  uint8_t* o = out + static_cast<size_t>(i) * height * out_width + xx;
  for (int yy = 0; yy < height; ++yy) {
    const int32_t* k = vk + yy * kv;
    const uint8_t* src = tile + (vfirst[yy] - top) * kThreads + tid;
    int32_t ss = detail::kHalf;
    for (int t = 0; t < vcount[yy]; ++t) ss += src[t * kThreads] * k[t];
    o[static_cast<size_t>(yy) * out_width] = detail::clip8(ss);
  }
}

__global__ void horizontal_coefficients(const LineMeta* lines, int max_w, int kh, int32_t* hk,
                                        int32_t* hfirst, int32_t* hcount) {
  const int i = blockIdx.y, xx = blockIdx.x * kThreads + threadIdx.x;
  const LineMeta m = lines[i];
  if (xx >= m.width) return;
  const detail::Filter f = detail::filter_for(m.x0, m.x1, m.width);
  const size_t at = static_cast<size_t>(i) * max_w + xx;
  int count = 0;
  hfirst[at] = detail::coefficients(f, m.x0, xx, hk + at * kh, &count);
  hcount[at] = count;
}

__global__ void horizontal_pass(const LineMeta* lines, const DevicePage* pages, int max_w, int kh,
                                const int32_t* hk, const int32_t* hfirst, const int32_t* hcount,
                                uint8_t* tmp) {
  const int i = blockIdx.z, r = blockIdx.y, xx = blockIdx.x * kThreads + threadIdx.x;
  const LineMeta m = lines[i];
  if (r >= m.rows || xx >= m.width) return;
  const DevicePage p = pages[m.page];
  const int y = m.top + r;
  const uint8_t* row = (y >= 0 && y < p.height) ? p.data + y * p.stride : nullptr;
  const size_t at = static_cast<size_t>(i) * max_w + xx;
  const int32_t* k = hk + at * kh;
  const int xmin = hfirst[at], n = hcount[at];
  int32_t ss = detail::kHalf;
  for (int t = 0; t < n; ++t) {
    const int x = xmin + t;
    const int v = (row && x >= 0 && x < p.width) ? row[x] : 255;
    ss += v * k[t];
  }
  tmp[m.tmp + static_cast<int64_t>(r) * m.width + xx] = detail::clip8(ss);
}

__global__ void vertical_pass(const LineMeta* lines, int height, int out_width, int kv,
                              const int32_t* vk, const int32_t* vfirst, const int32_t* vcount,
                              const uint8_t* tmp, uint8_t* out) {
  const int i = blockIdx.z, yy = blockIdx.y, xx = blockIdx.x * kThreads + threadIdx.x;
  const LineMeta m = lines[i];
  if (xx >= m.width) return;
  const size_t at = static_cast<size_t>(i) * height + yy;
  const int32_t* k = vk + at * kv;
  const uint8_t* src = tmp + m.tmp + static_cast<int64_t>(vfirst[at] - m.top) * m.width + xx;
  const int n = vcount[at];
  int32_t ss = detail::kHalf;
  for (int t = 0; t < n; ++t) ss += src[static_cast<int64_t>(t) * m.width] * k[t];
  out[(static_cast<size_t>(i) * height + yy) * out_width + xx] = detail::clip8(ss);
}

constexpr int kFinishThreads = 256;

// The n-th smallest value (from 0) of a 256-bin histogram, by one warp: each lane
// holds 8 bins, a shuffle scan finds the lane the n-th falls in, then its bins.
__device__ int nth_of(const int* hist, int n) {
  const int lane = threadIdx.x & 31;
  int bins[8], own = 0;
#pragma unroll
  for (int j = 0; j < 8; ++j) own += bins[j] = hist[8 * lane + j];
  int incl = own;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int t = __shfl_up_sync(0xffffffffu, incl, d);
    if (lane >= d) incl += t;
  }
  const unsigned past = __ballot_sync(0xffffffffu, incl > n);  // the lanes whose bins reach past n
  const int at = past ? __ffs(past) - 1 : 31;
  int seen = __shfl_sync(0xffffffffu, incl - own, at), v = 255;
#pragma unroll
  for (int j = 0; j < 8; ++j) {  // the first of that lane's bins whose running count passes n
    seen += bins[j];
    if (seen > n && v == 255) v = 8 * lane + j;
  }
  return __shfl_sync(0xffffffffu, v, at);
}

// Adds one to hist[v] for each active lane, a warp's equal values added at once
// (most border pixels are paper: the lanes collide on a few bins).
__device__ void count_in(int* hist, unsigned active, int v) {
  const unsigned same = __match_any_sync(active, v);
  if ((threadIdx.x & 31) == __ffs(same) - 1) atomicAdd(&hist[v], __popc(same));
}

// finish_crop on a GPU, a block for each crop of out[i][height][out_width]: the
// ground (the median of the first row, last row and last column), whether it is
// light on dark (bytes compared four to a word, the counts summed by REDUX), the
// flip (four pixels to an XOR), and the padding with the flipped crop's ground.
// Widths and out_width are multiples of 4, `out` 4-byte aligned.
__global__ void __launch_bounds__(kFinishThreads) finish(const int32_t* widths, int height, int out_width, int margin,
                                                         bool need_dark, uint8_t* out) {
  __shared__ int hist[256];
  __shared__ int sums[3][kFinishThreads / 32];
  __shared__ int rank[2];
  const int i = blockIdx.x, w = widths[i], tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  uint8_t* c = out + static_cast<size_t>(i) * height * out_width;
  hist[tid] = 0;  // kFinishThreads is 256
  __syncthreads();
  const int border = 2 * w + height;
  for (int base = 0; base < border; base += kFinishThreads) {
    const int j = base + tid;
    const unsigned active = __ballot_sync(0xffffffffu, j < border);
    if (j < border) {
      const int v = j < w ? c[j] : j < 2 * w ? c[(height - 1) * out_width + j - w] : c[(j - 2 * w) * out_width + w - 1];
      count_in(hist, active, v);
    }
  }
  __syncthreads();
  if (warp == 0) {
    const int a = nth_of(hist, border % 2 ? border / 2 : border / 2 - 1), b = nth_of(hist, border / 2);
    if (lane == 0) {
      rank[0] = a;
      rank[1] = b;
    }
  }
  __syncthreads();
  const int a = rank[0], b = rank[1];
  const int ground = border % 2 ? b : (a + b) / 2;
  const int words = w >> 2, row_words = out_width >> 2;
  auto* px = reinterpret_cast<uint32_t*>(c);
  __syncthreads();  // rank is reused below
  bool flip = false;
  if (margin > 0) {
    const int hi = ground + margin, lo = ground - margin;  // lighter: v >= hi; darker: v <= lo
    const uint32_t hi4 = 0x01010101u * static_cast<uint32_t>(hi <= 255 ? hi : 0);
    const uint32_t lo4 = 0x01010101u * static_cast<uint32_t>(lo >= 0 ? lo : 0);
    int lighter = 0, darker = 0, dark = 0;
    const uint32_t mid4 = 0x7F7F7F7Fu;  // bytes at most 127
    for (int y = 0; y < height; ++y) {
      const uint32_t* row = px + y * row_words;
      for (int k = tid; k < words; k += kFinishThreads) {
        const uint32_t q = row[k];
        lighter += __popc(__vcmpgeu4(q, hi4));
        darker += __popc(__vcmpleu4(q, lo4));
        if (need_dark) dark += __popc(__vcmpleu4(q, mid4));
      }
    }
    lighter = hi <= 255 ? __reduce_add_sync(0xffffffffu, static_cast<unsigned>(lighter)) >> 3 : 0;
    darker = lo >= 0 ? __reduce_add_sync(0xffffffffu, static_cast<unsigned>(darker)) >> 3 : 0;
    dark = __reduce_add_sync(0xffffffffu, static_cast<unsigned>(dark)) >> 3;
    if (lane == 0) {
      sums[0][warp] = lighter;
      sums[1][warp] = darker;
      sums[2][warp] = dark;
    }
    __syncthreads();
    if (warp == 0) {
      const bool held = lane < kFinishThreads / 32;
      const unsigned l = held ? sums[0][lane] : 0, d = held ? sums[1][lane] : 0, k = held ? sums[2][lane] : 0;
      const unsigned tl = __reduce_add_sync(0xffffffffu, l), td = __reduce_add_sync(0xffffffffu, d);
      const unsigned tk = __reduce_add_sync(0xffffffffu, k);
      if (lane == 0) rank[0] = tl > td && (!need_dark || 2 * static_cast<int64_t>(tk) > static_cast<int64_t>(w) * height);
    }
    __syncthreads();
    flip = rank[0];
  }
  if (flip) {
    for (int y = 0; y < height; ++y)
      for (int k = tid; k < words; k += kFinishThreads) px[y * row_words + k] ^= 0xffffffffu;
  }
  // The flipped crop's n-th smallest value is 255 less the original's (m - 1 - n)-th.
  const int g = flip ? (border % 2 ? 255 - b : (510 - a - b) / 2) : ground;
  const int pad = row_words - words;
  if (pad > 0) {
    const uint32_t g4 = 0x01010101u * static_cast<uint32_t>(g);
    for (int y = 0; y < height; ++y)
      for (int k = tid; k < pad; k += kFinishThreads) px[y * row_words + words + k] = g4;
  }
}

// One block per line, a thread per frame: the reads run along the frames for
// each class.  The softmax normalizer is kept online against the running max.
__global__ void ctc_argmax(const float* logits, int classes, int frames, const int32_t* lengths,
                           int32_t* arg, float* prob) {
  const int i = blockIdx.x;
  const int t_n = min(max(lengths[i], 0), frames);
  const float* line = logits + static_cast<size_t>(i) * classes * frames;
  for (int t = threadIdx.x; t < t_n; t += blockDim.x) {
    float best = -INFINITY, sum = 0.0f;
    int a = 0;
    for (int c = 0; c < classes; ++c) {
      const float v = line[static_cast<size_t>(c) * frames + t];
      if (v > best) {
        sum = sum * expf(best - v) + 1.0f;
        best = v;
        a = c;
      } else {
        sum += expf(v - best);
      }
    }
    arg[static_cast<size_t>(i) * frames + t] = a;
    prob[static_cast<size_t>(i) * frames + t] = 1.0f / sum;
  }
}

// Collapse repeats, drop blanks: one thread per line.
__global__ void ctc_collapse(int n, int frames, const int32_t* lengths, const int32_t* arg, const float* prob,
                             int32_t* ids, int32_t* starts, float* probs, int32_t* count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int t_n = min(max(lengths[i], 0), frames);
  const size_t base = static_cast<size_t>(i) * frames;
  int prev = 0, k = 0;
  for (int t = 0; t < t_n; ++t) {
    const int id = arg[base + t];
    const float p = prob[base + t];
    if (id != 0 && id != prev) {
      ids[base + k] = id;
      starts[base + k] = t;
      probs[base + k] = p;
      ++k;
    } else if (id != 0) {
      probs[base + k - 1] = fmaxf(probs[base + k - 1], p);
    }
    prev = id;
  }
  count[i] = k;
}

}  // namespace

bool available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

void crop_lines(std::span<const DevicePage> pages, std::span<const int32_t> page_of,
                std::span<const LineBox> boxes, int out_width, uint8_t* out, int32_t* widths,
                const CropRule& rule, void* stream) {
  const auto s = static_cast<cudaStream_t>(stream);
  const int n = static_cast<int>(boxes.size()), h = rule.height;
  if (n == 0) return;
  if (page_of.size() < boxes.size()) throw Error("cuda::crop_lines: page_of is shorter than boxes");

  // The fused kernel where every line's taps fit its registers and its source rows a shared tile: the
  // regions alone go up, the coefficients are made on the device. Else the kernels below, a pass each.
  {
    std::vector<LineRegion> regions(n);
    std::vector<int32_t> width_of(n);
    int kh = 1, kv = 1, max_w = 1, max_rows = 1;
    for (int i = 0; i < n; ++i) {
      const CropRegion r = crop_region(boxes[i], rule);
      if (r.width > out_width) {
        throw Error("cuda::crop_lines: line " + std::to_string(i) + " is " + std::to_string(r.width) +
                    " px wide, the batch " + std::to_string(out_width));
      }
      if (page_of[i] < 0 || page_of[i] >= static_cast<int>(pages.size())) {
        throw Error("cuda::crop_lines: line " + std::to_string(i) + " has no page " + std::to_string(page_of[i]));
      }
      regions[i] = {r.x0, r.x1, r.y0, r.y1, r.width, page_of[i]};
      width_of[i] = r.width;
      const detail::Filter fy = detail::filter_for(r.y0, r.y1, h);
      kh = std::max(kh, detail::max_taps(detail::filter_for(r.x0, r.x1, r.width)));
      kv = std::max(kv, detail::max_taps(fy));
      max_w = std::max(max_w, r.width);
      int32_t k[kMaxTaps * 4];
      int c0 = 0, c1 = 0;
      if (detail::max_taps(fy) <= kMaxTaps * 4) {
        const int top = detail::coefficients(fy, r.y0, 0, k, &c0);
        const int last = detail::coefficients(fy, r.y0, h - 1, k, &c1);
        max_rows = std::max(max_rows, last + c1 - top);
      } else {
        max_rows = INT32_MAX / 2;
      }
    }
    const size_t shared = (static_cast<size_t>(h) * kv + 2 * h) * sizeof(int32_t) +
                          static_cast<size_t>(max_rows) * kThreads;
    int device = 0, limit = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    check(cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device), "the shared memory");
    if (kh <= kMaxTaps && kv <= kMaxTaps && shared <= static_cast<size_t>(limit) && out_width % 4 == 0 &&
        reinterpret_cast<uintptr_t>(out) % 4 == 0) {
      if (shared > 48 * 1024) {
        check(cudaFuncSetAttribute(crop_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared)),
              "the fused crop's shared memory");
      }
      const std::vector<DevicePage> page_list(pages.begin(), pages.end());
      Buffer<LineRegion> d_regions(regions, s);
      Buffer<DevicePage> d_pages(page_list, s);
      const unsigned bx = static_cast<unsigned>((max_w + kThreads - 1) / kThreads);
      crop_fused<<<dim3(bx, n), kThreads, shared, s>>>(d_regions.get(), d_pages.get(), h, out_width, kv, out);
      check(cudaMemcpyAsync(widths, width_of.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice, s),
            "copy the widths");
      finish<<<n, kFinishThreads, 0, s>>>(widths, h, out_width, rule.flip_margin, rule.flip_needs_dark, out);
      check(cudaGetLastError(), "crop kernels");
      // No sync: the buffers are freed in the stream's order, and the pageable copies were taken before they
      // returned; the caller's next work on the stream follows the crops.
      return;
    }
  }

  // The regions and the vertical coefficients on the host (32 rows a line); the
  // horizontal ones, one per output column, on the device.
  std::vector<LineMeta> meta(n);
  std::vector<int32_t> width_of(n);
  std::vector<detail::Filter> fy(n);
  int kh = 1, kv = 1, max_w = 1, max_rows = 1;
  for (int i = 0; i < n; ++i) {
    const CropRegion r = crop_region(boxes[i], rule);
    if (r.width > out_width) {
      throw Error("cuda::crop_lines: line " + std::to_string(i) + " is " + std::to_string(r.width) +
                  " px wide, the batch " + std::to_string(out_width));
    }
    if (page_of[i] < 0 || page_of[i] >= static_cast<int>(pages.size())) {
      throw Error("cuda::crop_lines: line " + std::to_string(i) + " has no page " + std::to_string(page_of[i]));
    }
    meta[i] = {r.x0, r.x1, r.width, page_of[i], 0, 0, 0};
    width_of[i] = r.width;
    fy[i] = detail::filter_for(r.y0, r.y1, h);
    kh = std::max(kh, detail::max_taps(detail::filter_for(r.x0, r.x1, r.width)));
    kv = std::max(kv, detail::max_taps(fy[i]));
    max_w = std::max(max_w, r.width);
  }
  std::vector<int32_t> vk(static_cast<size_t>(n) * h * kv, 0), vfirst(static_cast<size_t>(n) * h),
      vcount(static_cast<size_t>(n) * h);
  int64_t tmp_bytes = 0;
  for (int i = 0; i < n; ++i) {
    const CropRegion r = crop_region(boxes[i], rule);
    for (int yy = 0; yy < h; ++yy) {
      const size_t at = static_cast<size_t>(i) * h + yy;
      vfirst[at] = detail::coefficients(fy[i], r.y0, yy, &vk[at * kv], &vcount[at]);
    }
    const size_t last = static_cast<size_t>(i) * h + h - 1;
    meta[i].top = vfirst[static_cast<size_t>(i) * h];
    meta[i].rows = vfirst[last] + vcount[last] - meta[i].top;
    meta[i].tmp = tmp_bytes;
    tmp_bytes += static_cast<int64_t>(meta[i].rows) * meta[i].width;
    max_rows = std::max(max_rows, meta[i].rows);
  }

  const std::vector<DevicePage> page_list(pages.begin(), pages.end());
  Buffer<LineMeta> d_meta(meta, s);
  Buffer<DevicePage> d_pages(page_list, s);
  Buffer<int32_t> d_vk(vk, s), d_vfirst(vfirst, s), d_vcount(vcount, s);
  const size_t hn = static_cast<size_t>(n) * max_w;
  Buffer<int32_t> d_hk(hn * kh, s), d_hfirst(hn, s), d_hcount(hn, s);
  Buffer<uint8_t> d_tmp(static_cast<size_t>(tmp_bytes), s);

  const unsigned bx = static_cast<unsigned>((max_w + kThreads - 1) / kThreads);
  horizontal_coefficients<<<dim3(bx, n), kThreads, 0, s>>>(d_meta.get(), max_w, kh, d_hk.get(),
                                                           d_hfirst.get(), d_hcount.get());
  horizontal_pass<<<dim3(bx, max_rows, n), kThreads, 0, s>>>(d_meta.get(), d_pages.get(), max_w, kh,
                                                             d_hk.get(), d_hfirst.get(), d_hcount.get(),
                                                             d_tmp.get());
  vertical_pass<<<dim3(bx, h, n), kThreads, 0, s>>>(d_meta.get(), h, out_width, kv, d_vk.get(),
                                                    d_vfirst.get(), d_vcount.get(), d_tmp.get(), out);
  check(cudaMemcpyAsync(widths, width_of.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice, s),
        "copy the widths");
  if (out_width % 4 == 0 && reinterpret_cast<uintptr_t>(out) % 4 == 0) {
    finish<<<n, kFinishThreads, 0, s>>>(widths, h, out_width, rule.flip_margin, rule.flip_needs_dark, out);
  } else {
    throw Error("cuda::crop_lines: the rows must be 4-byte aligned (out_width a multiple of 4)");
  }
  check(cudaGetLastError(), "crop kernels");
  check(cudaStreamSynchronize(s), "crop_lines");
}

void finish_crops(uint8_t* out, int n, int out_width, std::span<const int32_t> widths, const CropRule& rule,
                  void* stream) {
  const auto s = static_cast<cudaStream_t>(stream);
  if (n <= 0) return;
  if (static_cast<int>(widths.size()) < n) throw Error("cuda::finish_crops: widths is shorter than the batch");
  if (out_width % 4 || reinterpret_cast<uintptr_t>(out) % 4) {
    throw Error("cuda::finish_crops: the rows must be 4-byte aligned (out_width a multiple of 4)");
  }
  for (int i = 0; i < n; ++i) {
    if (widths[i] < 4 || widths[i] % 4 || widths[i] > out_width) {
      throw Error("cuda::finish_crops: crop " + std::to_string(i) + " is " + std::to_string(widths[i]) +
                  " px wide, not a multiple of 4 up to " + std::to_string(out_width));
    }
  }
  const std::vector<int32_t> w(widths.begin(), widths.begin() + n);
  Buffer<int32_t> d_w(w, s);
  finish<<<n, kFinishThreads, 0, s>>>(d_w.get(), rule.height, out_width, rule.flip_margin, rule.flip_needs_dark,
                                       out);
  check(cudaGetLastError(), "finish_crops");
}

std::vector<CtcLine> ctc_greedy(const float* logits, int n, int classes, int frames,
                                std::span<const int32_t> lengths, void* stream) {
  const auto s = static_cast<cudaStream_t>(stream);
  if (static_cast<int>(lengths.size()) < n) throw Error("cuda::ctc_greedy: lengths is shorter than the batch");
  std::vector<CtcLine> out(n);
  if (n == 0 || frames == 0) return out;
  const std::vector<int32_t> len(lengths.begin(), lengths.begin() + n);
  const size_t cells = static_cast<size_t>(n) * frames;
  Buffer<int32_t> d_len(len, s), d_arg(cells, s), d_ids(cells, s), d_starts(cells, s), d_count(n, s);
  Buffer<float> d_prob(cells, s), d_probs(cells, s);
  ctc_argmax<<<n, 256, 0, s>>>(logits, classes, frames, d_len.get(), d_arg.get(), d_prob.get());
  ctc_collapse<<<(n + 127) / 128, 128, 0, s>>>(n, frames, d_len.get(), d_arg.get(), d_prob.get(), d_ids.get(),
                                               d_starts.get(), d_probs.get(), d_count.get());
  check(cudaGetLastError(), "ctc kernels");
  std::vector<int32_t> ids(cells), starts(cells), count(n);
  std::vector<float> probs(cells);
  check(cudaMemcpyAsync(count.data(), d_count.get(), n * sizeof(int32_t), cudaMemcpyDeviceToHost, s), "copy");
  check(cudaMemcpyAsync(ids.data(), d_ids.get(), cells * sizeof(int32_t), cudaMemcpyDeviceToHost, s), "copy");
  check(cudaMemcpyAsync(starts.data(), d_starts.get(), cells * sizeof(int32_t), cudaMemcpyDeviceToHost, s), "copy");
  check(cudaMemcpyAsync(probs.data(), d_probs.get(), cells * sizeof(float), cudaMemcpyDeviceToHost, s), "copy");
  check(cudaStreamSynchronize(s), "ctc_greedy");
  for (int i = 0; i < n; ++i) {
    const size_t b = static_cast<size_t>(i) * frames;
    out[i].ids.assign(ids.begin() + b, ids.begin() + b + count[i]);
    out[i].frames.assign(starts.begin() + b, starts.begin() + b + count[i]);
    out[i].probs.assign(probs.begin() + b, probs.begin() + b + count[i]);
  }
  return out;
}

}  // namespace carbon::cuda
