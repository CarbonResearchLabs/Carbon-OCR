// The pinned staging ring the GPU's uploads pass through (carbon/src/cuda/buffer.h): every upload lands whole
// and in order, whatever the ring had to wait for, wrap around or grow. Skipped without a GPU.

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "../../carbon/src/cuda/buffer.h"
#include "carbon/cuda.h"

namespace {

using carbon::cuda::mem::Staging;

// Uploads of many sizes, through a ring far smaller than their sum (so it wraps and waits on its own copies),
// one larger than the whole ring (so it grows), on two streams, each from a host buffer overwritten at once after
// its upload returns: the device holds every one as it was when uploaded.
TEST(CudaStaging, UploadsLandWholeThroughASmallRing) {
  if (!carbon::cuda::available()) GTEST_SKIP() << "no CUDA device";
  Staging ring(size_t{64} << 10);  // 64 KiB
  cudaStream_t s[2];
  for (cudaStream_t& x : s) ASSERT_EQ(cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking), cudaSuccess);
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> size(1, 20000);
  std::vector<size_t> sizes;
  for (int i = 0; i < 300; ++i) sizes.push_back(static_cast<size_t>(size(rng)));
  sizes[150] = size_t{200} << 10;  // past the ring: it grows
  const size_t total = std::accumulate(sizes.begin(), sizes.end(), size_t{0});
  uint8_t* dev = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dev), total), cudaSuccess);
  std::vector<uint8_t> want(total), host;
  size_t at = 0;
  for (size_t i = 0; i < sizes.size(); ++i) {
    host.resize(sizes[i]);
    for (uint8_t& v : host) v = static_cast<uint8_t>(rng());
    std::copy(host.begin(), host.end(), want.begin() + static_cast<std::ptrdiff_t>(at));
    ring.upload(dev + at, host.data(), host.size(), s[i % 2]);
    std::fill(host.begin(), host.end(), uint8_t{0});  // the host's memory is free as soon as upload returns
    at += sizes[i];
  }
  EXPECT_GE(ring.capacity(), size_t{200} << 10);
  for (cudaStream_t x : s) ASSERT_EQ(cudaStreamSynchronize(x), cudaSuccess);
  std::vector<uint8_t> got(total);
  ASSERT_EQ(cudaMemcpy(got.data(), dev, total, cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_TRUE(got == want);
  cudaFree(dev);
  for (cudaStream_t x : s) cudaStreamDestroy(x);
}

// take and send: a room filled in pieces, only its first bytes sent, and the next room after it.
TEST(CudaStaging, PartOfARoomIsSent) {
  if (!carbon::cuda::available()) GTEST_SKIP() << "no CUDA device";
  Staging ring(size_t{4} << 10);
  uint8_t* dev = nullptr;
  ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&dev), 8192), cudaSuccess);
  ASSERT_EQ(cudaMemset(dev, 0, 8192), cudaSuccess);
  std::vector<uint8_t> want(8192, 0);
  for (int k = 0; k < 6; ++k) {  // 6 rooms of 3000 bytes through a 4 KiB ring, 1000 of each sent
    uint8_t* room = ring.take(3000);
    for (int i = 0; i < 3000; ++i) room[i] = static_cast<uint8_t>(k * 31 + i);
    ring.send(dev + 1000 * k, 1000, nullptr);
    for (int i = 0; i < 1000; ++i) want[static_cast<size_t>(1000 * k + i)] = static_cast<uint8_t>(k * 31 + i);
  }
  EXPECT_THROW(ring.send(dev, 10, nullptr), carbon::Error);  // nothing taken since
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint8_t> got(8192);
  ASSERT_EQ(cudaMemcpy(got.data(), dev, got.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_TRUE(got == want);
  cudaFree(dev);
}

}  // namespace
