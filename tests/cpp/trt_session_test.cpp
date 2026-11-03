#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <vector>

#include "carbon/session.h"
#include "test_model.h"

namespace carbon {
namespace {

bool have_gpu() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

class TrtSessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!have_gpu()) GTEST_SKIP() << "no CUDA device";
    cache_ = std::filesystem::temp_directory_path() /
             ("carbon-trt-test-" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()));
    std::filesystem::remove_all(cache_);
  }
  void TearDown() override { std::filesystem::remove_all(cache_); }

  std::unique_ptr<Session> open_model(std::vector<Profile> profiles = {}, bool graphs = true) {
    SessionOptions o;
    const Profile range = {{"x", {1, 1, 1, 1}, {2, 1, 32, 64}, {8, 1, 64, 512}}};
    o.profiles = profiles.empty() ? std::vector<Profile>{range} : std::move(profiles);
    o.engine_cache = cache_;
    o.cuda_graphs = graphs;
    return open_session(Backend::kTensorRT, test::model_path(), o);
  }

  std::filesystem::path cache_;
};

std::vector<Tensor> one(Tensor t) {
  std::vector<Tensor> v;
  v.push_back(std::move(t));
  return v;
}

void expect_hand_computed(std::span<const Tensor> out) {
  ASSERT_EQ(out.size(), 2u);
  auto colsum = out[0].data<float>();
  ASSERT_EQ(colsum.size(), test::expected_colsum().size());
  for (size_t i = 0; i < colsum.size(); ++i) {
    EXPECT_NEAR(colsum[i], test::expected_colsum()[i], 1e-5) << "column " << i;
  }
  auto ids = out[1].data<int64_t>();
  EXPECT_EQ(std::vector<int64_t>(ids.begin(), ids.end()), test::expected_ids());
}

// The input with line 0 and line 1 swapped: the outputs swap too.
Tensor swapped_input() {
  Tensor a = test::input(), b = test::input();
  const size_t line = a.bytes() / 2;
  std::memcpy(static_cast<char*>(b.raw()), static_cast<const char*>(a.raw()) + line, line);
  std::memcpy(static_cast<char*>(b.raw()) + line, a.raw(), line);
  return b;
}

TEST_F(TrtSessionTest, RunMatchesHandComputedOutputs) {
  auto s = open_model();
  expect_hand_computed(s->run(one(test::input())));
}

TEST_F(TrtSessionTest, ShapeOutsideTheRangeThrows) {
  auto s = open_model();
  EXPECT_THROW(s->run(one(Tensor(DType::kFloat32, {9, 1, 32, 64}))), Error);
}

TEST_F(TrtSessionTest, SecondOpenLoadsTheCachedEngine) {
  open_model();
  std::vector<std::filesystem::path> engines;
  bool timing = false;
  for (const auto& e : std::filesystem::directory_iterator(cache_)) {
    if (e.path().extension() == ".engine") engines.push_back(e.path());
    timing = timing || e.path().filename().string().starts_with("timing-");
  }
  ASSERT_EQ(engines.size(), 1u);
  EXPECT_TRUE(timing) << "no timing cache next to the engine";
  auto written = std::filesystem::last_write_time(engines[0]);
  auto s = open_model();
  EXPECT_EQ(std::filesystem::last_write_time(engines[0]), written);
  EXPECT_EQ(s->run(one(test::input()))[1].data<int64_t>()[0], 2);
}

// An exact bucket and a range: runs alternate between the two contexts, and
// the bucket's runs replay a graph, which must read the inputs of each run.
TEST_F(TrtSessionTest, BucketsReplayGraphsOnNewInputs) {
  for (bool graphs : {true, false}) {
    auto s = open_model({{ShapeRange::exact("x", {2, 1, 3, 4})},
                         {{"x", {1, 1, 1, 1}, {1, 1, 3, 4}, {4, 1, 8, 8}}}},
                        graphs);
    for (int round = 0; round < 3; ++round) {
      expect_hand_computed(s->run(one(test::input())));
      auto out = s->run(one(swapped_input()));
      EXPECT_EQ(out[1].data<int64_t>()[0], 0) << "graphs " << graphs << " round " << round;
      EXPECT_EQ(out[1].data<int64_t>()[1], 2) << "graphs " << graphs << " round " << round;
      // the range's context: line 0 alone
      Tensor first = Tensor::empty(DType::kFloat32, {1, 1, 3, 4});
      std::memcpy(first.raw(), test::input().raw(), first.bytes());
      out = s->run(one(std::move(first)));
      ASSERT_EQ(out[1].shape(), (Shape{1, 1}));
      EXPECT_EQ(out[1].data<int64_t>()[0], 2);
    }
  }
}

TEST_F(TrtSessionTest, OutputShapesFollowTheInputs) {
  auto s = open_model();
  const std::vector<Shape> in = {{3, 1, 5, 7}};
  const auto out = s->output_shapes(in);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_EQ(out[0], (Shape{3, 1, 7}));
  EXPECT_EQ(out[1], (Shape{3, 1}));
  EXPECT_THROW(s->output_shapes(std::vector<Shape>{{9, 1, 5, 7}}), Error);
}

template <class T>
struct Device {
  void* p = nullptr;
  size_t bytes;
  explicit Device(size_t n) : bytes(n * sizeof(T)) { EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess); }
  ~Device() { cudaFree(p); }
  std::vector<T> get() const {
    std::vector<T> h(bytes / sizeof(T));
    EXPECT_EQ(cudaMemcpy(h.data(), p, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
  }
};

// The device run: the caller's buffers and stream; a graph after the first run.
TEST_F(TrtSessionTest, EnqueueRunsOnTheCallersBuffers) {
  auto s = open_model({{ShapeRange::exact("x", {2, 1, 3, 4})}});
  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
  Device<float> x(24), colsum(8);
  Device<int64_t> ids(2);
  const std::vector<DeviceTensor> in = {{x.p, DType::kFloat32, {2, 1, 3, 4}}};
  const std::vector<DeviceTensor> out = {{colsum.p, DType::kFloat32, {2, 1, 4}},
                                         {ids.p, DType::kInt64, {2, 1}}};
  for (int round = 0; round < 3; ++round) {
    const Tensor input = round % 2 ? swapped_input() : test::input();
    ASSERT_EQ(cudaMemcpyAsync(x.p, input.raw(), input.bytes(), cudaMemcpyHostToDevice, stream),
              cudaSuccess);
    s->enqueue(in, out, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    const auto got = ids.get();
    const std::vector<int64_t> want = round % 2 ? std::vector<int64_t>{0, 2} : test::expected_ids();
    EXPECT_EQ(got, want) << "round " << round;
    const auto sums = colsum.get();
    EXPECT_NEAR(sums[round % 2 ? 6 : 2], 2.4f, 1e-5) << "round " << round;
  }
  // a wrong output buffer, the default stream
  const std::vector<DeviceTensor> narrow = {{colsum.p, DType::kFloat32, {2, 1, 3}},
                                            {ids.p, DType::kInt64, {2, 1}}};
  EXPECT_THROW(s->enqueue(in, narrow, stream), Error);
  EXPECT_THROW(s->enqueue(in, out, nullptr), Error);
  cudaStreamDestroy(stream);
}

TEST(TrtSession, FreeDimensionsNeedARange) {
  if (!have_gpu()) GTEST_SKIP() << "no CUDA device";
  EXPECT_THROW(open_session(Backend::kTensorRT, test::model_path()), Error);
}

TEST(TrtSession, AProfileForAnInputTheModelLacksThrows) {
  if (!have_gpu()) GTEST_SKIP() << "no CUDA device";
  SessionOptions o;
  o.profiles = {{ShapeRange::exact("x", {2, 1, 3, 4}), ShapeRange::exact("y", {1})}};
  EXPECT_THROW(open_session(Backend::kTensorRT, test::model_path(), o), Error);
}

}  // namespace
}  // namespace carbon
