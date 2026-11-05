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
//            plus the set lanes below it (POPC of the word and the lanes-below mask)
// Then the copies back, into a pinned buffer each thread keeps: the scan's header (each
// page's first offsets and the totals), and after it the bits and the values, four
// copies; the host splits them by page.

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

using mem::Buffer;
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
// picture pixel's logit, at its offset among every page's set pixels. In the maps'
// own type: FP16 maps send half the bytes, widened on the host exactly.
template <class T>
__global__ void __launch_bounds__(32 * kWarps) scatter(const T* maps, int h, int w, int words,
                                                       const uint32_t* core_bits,
                                                       const uint32_t* picture_bits, const int32_t* offsets,
                                                       T* core_values, T* picture_values) {
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
    T* v = core_values + (static_cast<size_t>(offsets[2 * at]) + __popc(cb & below)) * 3;
    v[0] = m[px];
    v[1] = m[plane + px];
    v[2] = m[2 * plane + px];
  }
  if (pb >> lane & 1u) {
    picture_values[static_cast<size_t>(offsets[2 * at + 1]) + __popc(pb & below)] = m[3 * plane + px];
  }
}

// A pinned host buffer a thread keeps for the copies back, grown as it needs: pinned
// memory is copied at the bus's speed, without the driver's staging.
uint8_t* pinned(size_t bytes) {
  struct Pinned {
    uint8_t* p = nullptr;
    size_t size = 0;
    ~Pinned() {
      if (p) cudaFreeHost(p);  // at exit the runtime may be gone: the error is ignored
    }
  };
  thread_local Pinned h;
  if (bytes > h.size) {
    if (h.p) check(cudaFreeHost(h.p), "cudaFreeHost");
    h.p = nullptr;
    h.size = 0;
    check(cudaMallocHost(reinterpret_cast<void**>(&h.p), bytes), "cudaMallocHost");
    h.size = bytes;
  }
  return h.p;
}

// Mapped pinned host memory a thread keeps for the values the scatter writes: the device writes them across
// the bus as it makes them, so their count need not come back first. Grown as it needs; its device address too.
struct Mapped {
  uint8_t* host = nullptr;
  uint8_t* device = nullptr;
  size_t size = 0;
  ~Mapped() {
    if (host) cudaFreeHost(host);  // at exit the runtime may be gone: the error is ignored
  }
};

Mapped& mapped(size_t bytes) {
  thread_local Mapped m;
  if (bytes > m.size) {
    if (m.host) check(cudaFreeHost(m.host), "cudaFreeHost");
    m = Mapped{};
    check(cudaHostAlloc(reinterpret_cast<void**>(&m.host), bytes, cudaHostAllocMapped), "cudaHostAlloc");
    check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&m.device), m.host, 0), "cudaHostGetDevicePointer");
    m.size = bytes;
  }
  return m;
}

template <class T>
std::vector<DetMasks> threshold(const T* maps, int n, int h, int w, const DetRule& rule, cudaStream_t s) {
  std::vector<DetMasks> out(n);
  if (n <= 0 || h <= 0 || w <= 0) return out;
  const int words = (w + 31) / 32, total = h * words;
  const size_t plane = static_cast<size_t>(h) * w, nw = static_cast<size_t>(n) * total;
  Buffer<uint32_t> core_bits(nw, s), picture_bits(nw, s);
  Buffer<int32_t> counts(2 * nw, s), header(2 * static_cast<size_t>(n) + 2, s);
  // the values' room: every pixel a core pixel and a picture pixel at most, in the maps' type
  const size_t core_room = static_cast<size_t>(n) * plane * 3 * sizeof(T);
  Mapped& values = mapped(core_room + static_cast<size_t>(n) * plane * sizeof(T));
  T* core_values = reinterpret_cast<T*>(values.device);
  T* picture_values = reinterpret_cast<T*>(values.device + core_room);
  const float tc = det_logit(rule.core_threshold), tp = det_logit(rule.picture_threshold);
  const dim3 grid((total + kWarps - 1) / kWarps, n);
  ballot<<<grid, 32 * kWarps, 0, s>>>(maps, h, w, words, tc, tp, core_bits.get(), picture_bits.get(), counts.get());
  scan<<<1, kScanThreads, 0, s>>>(n, total, counts.get(), header.get());
  scatter<<<grid, 32 * kWarps, 0, s>>>(maps, h, w, words, core_bits.get(), picture_bits.get(), counts.get(),
                                       core_values, picture_values);
  check(cudaGetLastError(), "threshold kernels");
  // the header and the bits copied while the values cross the bus: one sync for all
  const size_t head = (2 * static_cast<size_t>(n) + 2) * sizeof(int32_t), bits = nw * sizeof(uint32_t);
  uint8_t* staged = pinned(head + 2 * bits);
  check(cudaMemcpyAsync(staged, header.get(), head, cudaMemcpyDeviceToHost, s), "copy");
  check(cudaMemcpyAsync(staged + head, core_bits.get(), bits, cudaMemcpyDeviceToHost, s), "copy");
  check(cudaMemcpyAsync(staged + head + bits, picture_bits.get(), bits, cudaMemcpyDeviceToHost, s), "copy");
  check(cudaStreamSynchronize(s), "threshold_det");
  std::vector<int32_t> starts(2 * static_cast<size_t>(n) + 2);
  std::memcpy(starts.data(), staged, head);
  const size_t nc = static_cast<size_t>(starts[2 * n]), np = static_cast<size_t>(starts[2 * n + 1]);
  const auto* cb = reinterpret_cast<const uint32_t*>(staged + head);
  const auto* pb = reinterpret_cast<const uint32_t*>(staged + head + bits);
  const auto* cf = reinterpret_cast<const T*>(values.host);
  const auto* pf = reinterpret_cast<const T*>(values.host + core_room);
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
                                    void* stream) {
  const auto s = static_cast<cudaStream_t>(stream);
  return fp16 ? threshold(static_cast<const __half*>(maps), n, map_h, map_w, rule, s)
              : threshold(static_cast<const float*>(maps), n, map_h, map_w, rule, s);
}

}  // namespace carbon::cuda
