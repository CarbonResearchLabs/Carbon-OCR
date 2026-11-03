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

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "carbon/error.h"

namespace carbon::cuda::mem {

inline void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw Error(std::string("CUDA: ") + what + ": " + cudaGetErrorString(e));
}

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
  Buffer(const std::vector<T>& host, cudaStream_t s) : Buffer(host.size(), s) {
    if (!host.empty()) {
      check(cudaMemcpyAsync(p_, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice, s), "copy to device");
    }
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
