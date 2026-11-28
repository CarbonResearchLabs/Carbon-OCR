// Device scratch for one call of the CUDA code (lines.cu, detect.cu): from
// the device's default memory pool, freed on the call's stream.
//
// The pool keeps what it holds: its release threshold is set, once a device, to the
// most it may hold. With the default of 0 it hands its memory back at every stream
// synchronize, and the next call's allocation goes to the driver, a millisecond each,
// longer than the kernels themselves. What it keeps is the largest scratch a call
// has needed (tens of MB for eight pages of maps).
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "carbon/error.h"

namespace carbon::cuda::mem {

inline void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw Error(std::string("CUDA: ") + what + ": " + cudaGetErrorString(e));
}

// Device memory a caller keeps from call to call, grown to the most it was asked for (reserve) and never shrunk,
// so a caller that reserves its largest at once allocates once.
class DeviceArray {
 public:
  DeviceArray() = default;
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  ~DeviceArray() {
    if (p_) cudaFree(p_);  // at exit the runtime may be gone: the error is ignored
  }
  void reserve(size_t bytes) {
    if (bytes <= size_) return;
    if (p_) check(cudaFree(p_), "cudaFree");
    p_ = nullptr;
    size_ = 0;
    check(cudaMalloc(&p_, bytes), "cudaMalloc");
    size_ = bytes;
  }
  template <class T>
  T* as() const {
    return static_cast<T*>(p_);
  }
  size_t size() const { return size_; }

 private:
  void* p_ = nullptr;
  size_t size_ = 0;
};

// Pinned host memory kept so too: copied at the bus's speed, without the driver's staging.
class PinnedArray {
 public:
  PinnedArray() = default;
  PinnedArray(const PinnedArray&) = delete;
  PinnedArray& operator=(const PinnedArray&) = delete;
  ~PinnedArray() {
    if (p_) cudaFreeHost(p_);  // at exit the runtime may be gone: the error is ignored
  }
  void reserve(size_t bytes) {
    if (bytes <= size_) return;
    if (p_) check(cudaFreeHost(p_), "cudaFreeHost");
    p_ = nullptr;
    size_ = 0;
    check(cudaMallocHost(&p_, bytes), "cudaMallocHost");
    size_ = bytes;
  }
  template <class T>
  T* as() const {
    return static_cast<T*>(p_);
  }
  size_t size() const { return size_; }

 private:
  void* p_ = nullptr;
  size_t size_ = 0;
};

// A ring of pinned host memory that uploads pass through. A copy from pageable memory is staged by the
// driver through buffers of its own, on the calling thread; from pinned memory the copy engine takes it while the
// thread goes on. Each upload takes the next room of the ring (take), and send queues its copy and marks the room
// with an event, so a later take waits only for the copy that last used the room it gets. The events are kept
// for the next uses, none made a copy. Sized once: an upload past the whole ring grows it, once, to twice that.
// One thread's at a time (mine(): each thread its own, for the small uploads of a call's tables).
class Staging {
 public:
  explicit Staging(size_t capacity = size_t{8} << 20) : want_(capacity) {}
  Staging(const Staging&) = delete;
  Staging& operator=(const Staging&) = delete;
  ~Staging() {  // at exit the runtime may be gone: errors are ignored
    for (const Use& u : busy_) cudaEventDestroy(u.done);
    for (cudaEvent_t e : spare_) cudaEventDestroy(e);
    if (p_) cudaFreeHost(p_);
  }
  static Staging& mine() {
    thread_local Staging s;
    return s;
  }

  // Room for `bytes` in the ring, the copies that last read it waited for. Valid until the next take.
  uint8_t* take(size_t bytes) {
    retire();
    if (!p_ || bytes > cap_) grow(std::max({bytes, 2 * cap_, want_}));
    size_t at = (head_ + kAlign - 1) / kAlign * kAlign;
    if (at + bytes > cap_) at = 0;  // the ring's end: from its start again
    while (!busy_.empty() && overlaps(at, at + bytes)) {
      check(cudaEventSynchronize(busy_.front().done), "cudaEventSynchronize");
      spare_.push_back(busy_.front().done);
      busy_.pop_front();
    }
    taken_ = at;
    taken_end_ = at + bytes;
    return p_ + at;
  }
  // The first `bytes` of the room last taken copied to `to` on `s`, and the room marked busy until they are.
  void send(void* to, size_t bytes, cudaStream_t s) {
    if (bytes > taken_end_ - taken_) throw Error("Staging::send: more bytes than the room taken");
    if (bytes == 0) return;
    check(cudaMemcpyAsync(to, p_ + taken_, bytes, cudaMemcpyHostToDevice, s), "copy to device");
    cudaEvent_t e = nullptr;
    if (spare_.empty()) {
      check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreate");
    } else {
      e = spare_.back();
      spare_.pop_back();
    }
    if (const cudaError_t r = cudaEventRecord(e, s); r != cudaSuccess) {
      spare_.push_back(e);
      check(r, "cudaEventRecord");
    }
    busy_.push_back({taken_, taken_ + bytes, e});
    head_ = taken_ + bytes;
    taken_end_ = taken_;  // sent: the room is the ring's again
  }
  // `bytes` of host memory at `from` to the device at `to` on `s`, through the ring.
  void upload(void* to, const void* from, size_t bytes, cudaStream_t s) {
    if (bytes == 0) return;
    std::memcpy(take(bytes), from, bytes);
    send(to, bytes, s);
  }
  size_t capacity() const { return cap_; }

 private:
  static constexpr size_t kAlign = 256;
  struct Use {
    size_t begin, end;
    cudaEvent_t done;
  };
  bool overlaps(size_t a, size_t b) const {
    for (const Use& u : busy_) {
      if (u.begin < b && a < u.end) return true;
    }
    return false;
  }
  // The copies done, their rooms and events free again (none waited for).
  void retire() {
    while (!busy_.empty() && cudaEventQuery(busy_.front().done) == cudaSuccess) {
      spare_.push_back(busy_.front().done);
      busy_.pop_front();
    }
  }
  void grow(size_t bytes) {
    for (const Use& u : busy_) {
      check(cudaEventSynchronize(u.done), "cudaEventSynchronize");
      spare_.push_back(u.done);
    }
    busy_.clear();
    if (p_) check(cudaFreeHost(p_), "cudaFreeHost");
    p_ = nullptr;
    cap_ = head_ = 0;
    check(cudaMallocHost(reinterpret_cast<void**>(&p_), bytes), "cudaMallocHost");
    cap_ = bytes;
  }
  uint8_t* p_ = nullptr;
  size_t want_, cap_ = 0, head_ = 0, taken_ = 0, taken_end_ = 0;
  std::deque<Use> busy_;
  std::vector<cudaEvent_t> spare_;
};

inline void keep_pool() {
  constexpr int kDevices = 64;
  static std::once_flag once[kDevices];
  int device = 0;
  check(cudaGetDevice(&device), "cudaGetDevice");
  if (device < 0 || device >= kDevices) return;
  std::call_once(once[device], [device] {
    cudaMemPool_t pool;
    check(cudaDeviceGetDefaultMemPool(&pool, device), "cudaDeviceGetDefaultMemPool");
    uint64_t keep = UINT64_MAX;
    check(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep), "cudaMemPoolSetAttribute");
  });
}

// A pinned host buffer a thread keeps for its uploads (pinned memory is copied at the
// bus's speed, without the driver's staging copy), grown as needed.  get() waits for
// the last copy out of it to have run before handing it out again; sent() marks that copy.
class Upload {
 public:
  static Upload& mine() {
    thread_local Upload u;
    return u;
  }
  uint8_t* get(size_t bytes) {
    if (pending_) check(cudaEventSynchronize(done_), "cudaEventSynchronize");
    pending_ = false;
    if (bytes > size_) {
      if (p_) check(cudaFreeHost(p_), "cudaFreeHost");
      p_ = nullptr;
      size_ = 0;
      check(cudaMallocHost(reinterpret_cast<void**>(&p_), bytes), "cudaMallocHost");
      size_ = bytes;
    }
    return p_;
  }
  void sent(cudaStream_t s) {
    if (!done_) check(cudaEventCreateWithFlags(&done_, cudaEventDisableTiming), "cudaEventCreate");
    check(cudaEventRecord(done_, s), "cudaEventRecord");
    pending_ = true;
  }
  ~Upload() {  // at exit the runtime may be gone: errors are ignored
    if (done_) cudaEventDestroy(done_);
    if (p_) cudaFreeHost(p_);
  }

 private:
  Upload() = default;
  uint8_t* p_ = nullptr;
  size_t size_ = 0;
  cudaEvent_t done_ = nullptr;
  bool pending_ = false;
};

// A device array that lives as long as the call, freed on its stream.
template <class T>
class Buffer {
 public:
  Buffer(size_t n, cudaStream_t s) : s_(s) {
    if (n) {
      keep_pool();
      check(cudaMallocAsync(reinterpret_cast<void**>(&p_), n * sizeof(T), s), "cudaMallocAsync");
    }
  }
  // The host's values uploaded through the thread's staging ring: the thread does not wait for the copy.
  Buffer(const std::vector<T>& host, cudaStream_t s) : Buffer(host.size(), s) {
    if (!host.empty()) Staging::mine().upload(p_, host.data(), host.size() * sizeof(T), s);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() {
    if (p_) cudaFreeAsync(p_, s_);
  }
  T* get() const { return p_; }

 private:
  T* p_ = nullptr;
  cudaStream_t s_;
};

}  // namespace carbon::cuda::mem
