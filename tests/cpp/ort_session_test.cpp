#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

#include "carbon/parallel.h"
#include "carbon/session.h"
#include "carbon/strips.h"
#include "test_model.h"

namespace carbon {
namespace {

std::unique_ptr<Session> open_model(int threads = 1) {
  SessionOptions o;
  o.threads = threads;
  return open_session(Backend::kOnnxRuntime, test::model_path(), o);
}

std::vector<Tensor> one(Tensor t) {
  std::vector<Tensor> v;
  v.push_back(std::move(t));
  return v;
}

TEST(OrtSession, ReportsInputsAndOutputs) {
  auto s = open_model();
  EXPECT_EQ(s->backend(), Backend::kOnnxRuntime);
  ASSERT_EQ(s->inputs().size(), 1u);
  EXPECT_EQ(s->inputs()[0].name, "x");
  EXPECT_EQ(s->inputs()[0].dtype, DType::kFloat32);
  EXPECT_EQ(s->inputs()[0].shape, (Shape{-1, 1, -1, -1}));
  ASSERT_EQ(s->outputs().size(), 2u);
  EXPECT_EQ(s->outputs()[0].name, "colsum");
  EXPECT_EQ(s->outputs()[0].shape, (Shape{-1, 1, -1}));
  EXPECT_EQ(s->outputs()[1].name, "ids");
  EXPECT_EQ(s->outputs()[1].dtype, DType::kInt64);
}

TEST(OrtSession, RunMatchesHandComputedOutputs) {
  for (int threads : {1, 2}) {
    auto s = open_model(threads);
    auto out = s->run(one(test::input()));
    ASSERT_EQ(out.size(), 2u);
    EXPECT_EQ(out[0].shape(), (Shape{2, 1, 4}));
    auto colsum = out[0].data<float>();
    for (size_t i = 0; i < colsum.size(); ++i) {
      EXPECT_NEAR(colsum[i], test::expected_colsum()[i], 1e-6) << "column " << i;
    }
    EXPECT_EQ(out[1].shape(), (Shape{2, 1}));
    auto ids = out[1].data<int64_t>();
    EXPECT_EQ(std::vector<int64_t>(ids.begin(), ids.end()), test::expected_ids());
  }
}

TEST(OrtSession, FreeDimensionsTakeAnySize) {
  auto s = open_model();
  auto out = s->run(one(Tensor(DType::kFloat32, {5, 1, 32, 640})));
  EXPECT_EQ(out[0].shape(), (Shape{5, 1, 640}));
}

TEST(OrtSession, RejectsInputsThatDoNotMatch) {
  auto s = open_model();
  EXPECT_THROW(s->run(one(Tensor(DType::kInt64, {2, 1, 3, 4}))), Error);    // dtype
  EXPECT_THROW(s->run(one(Tensor(DType::kFloat32, {2, 3, 4}))), Error);     // rank
  EXPECT_THROW(s->run(one(Tensor(DType::kFloat32, {2, 3, 3, 4}))), Error);  // fixed dim
  EXPECT_THROW(s->run(one(Tensor(DType::kFloat32, {2, 1, 3, 0}))), Error);  // empty
  EXPECT_THROW(s->run(std::vector<Tensor>{}), Error);                       // count
}

TEST(OrtSession, DeviceRunsThrow) {
  auto s = open_model();
  EXPECT_THROW(s->output_shapes(std::vector<Shape>{{2, 1, 3, 4}}), Error);
  EXPECT_THROW(s->enqueue({}, {}, nullptr), Error);
}

TEST(OrtSession, MissingModelThrows) {
  EXPECT_THROW(open_session(Backend::kOnnxRuntime, "no/such/model.onnx"), Error);
}

// The CPU layout: a session per worker, opened on the worker's own thread, all
// running at once.
TEST(OrtSession, ASessionPerWorkerRunsConcurrently) {
  WorkerPool pool({4});
  std::vector<std::unique_ptr<Session>> sessions(pool.size());
  pool.on_each_worker([&](int w) { sessions[w] = open_model(); });
  std::atomic<int> wrong{0};
  for (int i = 0; i < 64; ++i) {
    pool.submit([&](int w) {
      auto out = sessions[w]->run(one(test::input()));
      auto ids = out[1].data<int64_t>();
      if (std::vector<int64_t>(ids.begin(), ids.end()) != test::expected_ids()) wrong.fetch_add(1);
    });
  }
  pool.wait();
  EXPECT_EQ(wrong.load(), 0);
  pool.on_each_worker([&](int w) { sessions[w].reset(); });
}

// Detection in strips: with a halo that covers the receptive field the stitched
// maps are the whole page's, byte for byte; with less they are not.
TEST(OrtSession, StripsGiveTheWholePagesBytes) {
  auto s = open_session(Backend::kOnnxRuntime, test::strips_model_path());
  std::vector<float> v(256 * 96);
  uint32_t r = 12345;
  for (float& x : v) {
    r = r * 1664525u + 1013904223u;
    x = static_cast<float>(r >> 8) / 16777216.0f;
  }
  const Tensor page = Tensor::from<float>({1, 1, 256, 96}, v);
  const Tensor full = std::move(s->run(one(page.clone()))[0]);
  ASSERT_EQ(full.shape(), (Shape{1, 2, 128, 48}));
  struct Case {
    int halo, strips;
    bool exact;
  };
  for (const Case c : {Case{32, 4, true}, Case{32, 5, true}, Case{64, 3, true}, Case{8, 4, false}}) {
    Tensor out(DType::kFloat32, full.shape());
    for (const Strip& st : plan_strips(256, c.strips, {c.halo, 8, 2})) {
      stitch_strip(s->run(one(strip_input(page, st)))[0], st, out);
    }
    EXPECT_EQ(std::memcmp(out.raw(), full.raw(), full.bytes()) == 0, c.exact)
        << "halo " << c.halo << ", " << c.strips << " strips";
  }
  // the same through run_in_strips, as the second detection pass runs its page
  const Tensor run = run_in_strips(
      page, 5, [&](const Tensor& x) { return std::move(s->run(std::span<const Tensor>(&x, 1))[0]); }, {32, 8, 2});
  ASSERT_EQ(run.shape(), full.shape());
  EXPECT_EQ(std::memcmp(run.raw(), full.raw(), full.bytes()), 0);
}

TEST(Backends, NamesRoundTrip) {
  EXPECT_EQ(parse_backend("onnxruntime"), Backend::kOnnxRuntime);
  EXPECT_EQ(parse_backend(backend_name(Backend::kTensorRT)), Backend::kTensorRT);
  EXPECT_EQ(parse_backend("cuda"), std::nullopt);
}

}  // namespace
}  // namespace carbon
