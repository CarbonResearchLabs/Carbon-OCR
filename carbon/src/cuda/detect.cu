// The detector's maps thresholded on the GPU (carbon/cuda.h): the same DetMasks as
// carbon::threshold_det, so the host decodes what it would have from the whole maps,
// but only the bits and the set pixels' values cross the bus.
//
// Three launches for every page in flight:
//   ballot   a warp a 32-pixel word: each lane compares its pixel, VOTE.ALL packs
//            the 32 decisions into the word, POPC counts them
//   scan     one block for every page: the exclusive sums of the counts, words in
//            page-major, row-major order (each thread a run of words, a shuffle scan
//            across the warps), so every page's values land in one array
//   scatter  a warp a word again: a set lane writes its values at the word's offset
//            plus the set lanes below it (POPC of the word and the lanes-below mask), into
//            device memory: every core value first, then every picture value (their base
//            the scan's core total), so the values of every page are one array
// Then two copies back into pinned memory: the scan's header (each page's first offsets
// and the totals) and the bits, one array, and once its totals are in, the values, as many
// as are set, in one more; the host splits them by page. The buffers are DetScratch's,
// sized once by a caller that thresholds batch after batch (the GPU pipeline), else the
// thread's own, grown to the largest call it has made.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "carbon/cpu.h"
#include "carbon/cuda.h"
#include "carbon/error.h"
#include "buffer.h"

namespace carbon::cuda {

namespace {

using mem::check;

__device__ __forceinline__ float to_float(float v) { return v; }
__device__ __forceinline__ float to_float(__half v) { return __half2float(v); }

constexpr int kWarps = 8;  // words a block of the ballot and scatter launches

// Bits and counts of one page's words: maps [n][4][h][w], word k of row r covers
// pixels 32 k .. 32 k + 31.  Lanes past the row read nothing and vote 0.
template <class T>
__global__ void __launch_bounds__(32 * kWarps) ballot(const T* maps, int h, int w, int words, float tc, float tp,
                                                      uint32_t* core_bits, uint32_t* picture_bits, int32_t* counts) {
  const int page = blockIdx.y, lane = threadIdx.x & 31;
  const int word = blockIdx.x * kWarps + (threadIdx.x >> 5), total = h * words;
  if (word >= total) return;
  const int r = word / words, c = (word - r * words) * 32 + lane;
  const size_t plane = static_cast<size_t>(h) * w;
  const T* core = maps + static_cast<size_t>(page) * 4 * plane;
  bool cs = false, ps = false;
  if (c < w) {
    const size_t at = static_cast<size_t>(r) * w + c;
    cs = to_float(core[at]) > tc;
    ps = to_float(core[3 * plane + at]) > tp;
  }
  const uint32_t cb = __ballot_sync(0xffffffffu, cs), pb = __ballot_sync(0xffffffffu, ps);
  if (lane == 0) {
    const size_t at = static_cast<size_t>(page) * total + word;
    core_bits[at] = cb;
    picture_bits[at] = pb;
    counts[2 * at] = __popc(cb);
    counts[2 * at + 1] = __popc(pb);
  }
}

constexpr int kScanThreads = 1024;

// The exclusive sums of every page's counts in place (core and picture as int2), and
// a header: each page's first offsets (2 n ints), then the totals (2 ints).
__global__ void __launch_bounds__(kScanThreads) scan(int n, int total, int32_t* counts, int32_t* header) {
  __shared__ int2 warp_sums[kScanThreads / 32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, all = n * total;
  int2* c = reinterpret_cast<int2*>(counts);
  const int per = (all + kScanThreads - 1) / kScanThreads, k0 = min(all, tid * per), k1 = min(all, k0 + per);
  int2 own = {0, 0};
  for (int k = k0; k < k1; ++k) {
    const int2 v = c[k];
    own.x += v.x;
    own.y += v.y;
  }
  int2 incl = own;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int x = __shfl_up_sync(0xffffffffu, incl.x, d), y = __shfl_up_sync(0xffffffffu, incl.y, d);
    if (lane >= d) {
      incl.x += x;
      incl.y += y;
    }
  }
  if (lane == 31) warp_sums[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    int2 t = warp_sums[lane];
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int x = __shfl_up_sync(0xffffffffu, t.x, d), y = __shfl_up_sync(0xffffffffu, t.y, d);
      if (lane >= d) {
        t.x += x;
        t.y += y;
      }
    }
    warp_sums[lane] = t;
    if (lane == 31) {
      header[2 * n] = t.x;
      header[2 * n + 1] = t.y;
    }
  }
  __syncthreads();
  int2 run = {incl.x - own.x, incl.y - own.y};
  if (warp > 0) {
    run.x += warp_sums[warp - 1].x;
    run.y += warp_sums[warp - 1].y;
  }
  for (int k = k0; k < k1; ++k) {
    if (k % total == 0) {
      header[2 * (k / total)] = run.x;
      header[2 * (k / total) + 1] = run.y;
    }
    const int2 v = c[k];
    c[k] = run;
    run.x += v.x;
    run.y += v.y;
  }
}

// Each set pixel's values to its place: a core pixel's logit, top and bottom, a
// picture pixel's logit, at its offset among every page's set pixels, in device memory
// (the 2-byte writes across the bus into mapped host memory took most of the GPU's
// time). The picture values follow the core values: their base is 3 x the scan's core
// total (totals[0]), so all of them are one array, copied back at once. In the maps'
// own type: FP16 maps send half the bytes, widened on the host exactly.
template <class T>
__global__ void __launch_bounds__(32 * kWarps) scatter(const T* maps, int h, int w, int words,
                                                       const uint32_t* core_bits,
                                                       const uint32_t* picture_bits, const int32_t* offsets,
                                                       const int32_t* totals, T* values) {
  const int page = blockIdx.y, lane = threadIdx.x & 31;
  const int word = blockIdx.x * kWarps + (threadIdx.x >> 5), total = h * words;
  if (word >= total) return;
  const size_t at = static_cast<size_t>(page) * total + word;
  const uint32_t cb = core_bits[at], pb = picture_bits[at], below = (1u << lane) - 1u;
  if (!(cb | pb)) return;
  const int r = word / words, c = (word - r * words) * 32 + lane;
  const size_t plane = static_cast<size_t>(h) * w, px = static_cast<size_t>(r) * w + c;
  const T* m = maps + static_cast<size_t>(page) * 4 * plane;
  if (cb >> lane & 1u) {
    T* v = values + (static_cast<size_t>(offsets[2 * at]) + __popc(cb & below)) * 3;
    v[0] = m[px];
    v[1] = m[plane + px];
    v[2] = m[2 * plane + px];
  }
  if (pb >> lane & 1u) {
    T* picture_values = values + 3 * static_cast<size_t>(totals[0]);
    picture_values[static_cast<size_t>(offsets[2 * at + 1]) + __popc(pb & below)] = m[3 * plane + px];
  }
}

}  // namespace

// The device's counts, its header and bits (one array: the header's 2 n + 2 ints, then the core and the picture
// bits) and its values (every page's core values, then the picture values), and the pinned copies of the header,
// the bits and the values on the host.
struct DetScratch::Impl {
  mem::DeviceArray counts, meta, values;
  mem::PinnedArray meta_host, values_host;
  // Room for n pages of h x w maps of `value` bytes a value: every pixel a core pixel and a picture pixel at most.
  void reserve(size_t n, size_t h, size_t w, size_t value) {
    const size_t nw = n * h * ((w + 31) / 32), plane = h * w;
    counts.reserve(2 * nw * sizeof(int32_t));
    meta.reserve((2 * n + 2 + 2 * nw) * sizeof(int32_t));
    meta_host.reserve(meta.size());
    values.reserve(n * plane * 4 * value);
    values_host.reserve(values.size());
  }
};

DetScratch::DetScratch() : impl_(std::make_unique<Impl>()) {}
DetScratch::~DetScratch() = default;

void DetScratch::reserve(int n, int map_h, int map_w, bool fp16) {
  if (n <= 0 || map_h <= 0 || map_w <= 0) return;
  impl_->reserve(static_cast<size_t>(n), static_cast<size_t>(map_h), static_cast<size_t>(map_w), fp16 ? 2 : 4);
}

namespace {

template <class T>
std::vector<DetMasks> threshold(const T* maps, int n, int h, int w, const DetRule& rule, cudaStream_t s,
                                DetScratch::Impl& k) {
  std::vector<DetMasks> out(n);
  if (n <= 0 || h <= 0 || w <= 0) return out;
  const int words = (w + 31) / 32, total = h * words;
  const size_t plane = static_cast<size_t>(h) * w, nw = static_cast<size_t>(n) * total;
  k.reserve(static_cast<size_t>(n), static_cast<size_t>(h), static_cast<size_t>(w), sizeof(T));
  const size_t head_ints = 2 * static_cast<size_t>(n) + 2;
  int32_t* header = k.meta.as<int32_t>();
  uint32_t* core_bits = reinterpret_cast<uint32_t*>(header + head_ints);
  uint32_t* picture_bits = core_bits + nw;
  int32_t* counts = k.counts.as<int32_t>();
  const float tc = det_logit(rule.core_threshold), tp = det_logit(rule.picture_threshold);
  const dim3 grid((total + kWarps - 1) / kWarps, n);
  ballot<<<grid, 32 * kWarps, 0, s>>>(maps, h, w, words, tc, tp, core_bits, picture_bits, counts);
  scan<<<1, kScanThreads, 0, s>>>(n, total, counts, header);
  scatter<<<grid, 32 * kWarps, 0, s>>>(maps, h, w, words, core_bits, picture_bits, counts, header + 2 * n,
                                       k.values.as<T>());
  check(cudaGetLastError(), "threshold kernels");
  // the header and the bits in one copy; then, their totals known, the values set, in one more
  const size_t head = head_ints * sizeof(int32_t), bits = nw * sizeof(uint32_t);
  uint8_t* staged = k.meta_host.as<uint8_t>();
  check(cudaMemcpyAsync(staged, header, head + 2 * bits, cudaMemcpyDeviceToHost, s), "copy");
  check(cudaStreamSynchronize(s), "threshold_det");
  std::vector<int32_t> starts(head_ints);
  std::memcpy(starts.data(), staged, head);
  const size_t nc = static_cast<size_t>(starts[2 * n]), np = static_cast<size_t>(starts[2 * n + 1]);
  if (3 * nc + np > static_cast<size_t>(n) * plane * 4) throw Error("threshold_det: more values set than pixels");
  if (3 * nc + np > 0) {
    check(cudaMemcpyAsync(k.values_host.as<T>(), k.values.as<T>(), (3 * nc + np) * sizeof(T), cudaMemcpyDeviceToHost,
                          s),
          "copy");
    check(cudaStreamSynchronize(s), "threshold_det");
  }
  const auto* cb = reinterpret_cast<const uint32_t*>(staged + head);
  const auto* pb = reinterpret_cast<const uint32_t*>(staged + head + bits);
  const auto* cf = k.values_host.as<const T>();
  const auto* pf = cf + 3 * nc;
  // the values as floats: FP16 widened on the host's vectors, the same bits as on the device
  auto widen = [](const T* from, size_t count, std::vector<float>& to) {
    if constexpr (std::is_same_v<T, float>) {
      to.assign(from, from + count);
    } else {
      to.resize(count);
      if (count) cpu::to_f32(reinterpret_cast<const uint16_t*>(from), to.data(), count);
    }
  };
  for (int p = 0; p < n; ++p) {
    DetMasks& m = out[p];
    m.height = h;
    m.width = w;
    m.words = words;
    m.core.assign(cb + static_cast<size_t>(p) * total, cb + static_cast<size_t>(p + 1) * total);
    m.picture.assign(pb + static_cast<size_t>(p) * total, pb + static_cast<size_t>(p + 1) * total);
    const size_t c0 = starts[2 * p], c1 = p + 1 < n ? starts[2 * p + 2] : nc;
    const size_t p0 = starts[2 * p + 1], p1 = p + 1 < n ? starts[2 * p + 3] : np;
    widen(cf + 3 * c0, 3 * (c1 - c0), m.core_values);
    widen(pf + p0, p1 - p0, m.picture_values);
  }
  return out;
}

}  // namespace

std::vector<DetMasks> threshold_det(const void* maps, bool fp16, int n, int map_h, int map_w, const DetRule& rule,
                                    void* stream, DetScratch* scratch) {
  const auto s = static_cast<cudaStream_t>(stream);
  thread_local DetScratch own;  // a caller without its own: the thread's, grown to its largest call
  DetScratch::Impl& k = scratch ? scratch->impl() : own.impl();
  return fp16 ? threshold(static_cast<const __half*>(maps), n, map_h, map_w, rule, s, k)
              : threshold(static_cast<const float*>(maps), n, map_h, map_w, rule, s, k);
}

std::vector<DetMasks> threshold_det(const void* maps, bool fp16, int n, int map_h, int map_w, const DetRule& rule,
                                    void* stream) {
  return threshold_det(maps, fp16, n, map_h, map_w, rule, stream, nullptr);
}

}  // namespace carbon::cuda
