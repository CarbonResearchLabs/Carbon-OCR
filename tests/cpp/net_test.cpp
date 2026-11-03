// The native backend (carbon/src/net/) on Fullerene's networks in miniature
// (tools/make_test_models.py): ONNX Runtime's outputs, saved beside each model,
// so the check runs on every machine; every instruction set's bits alike.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "carbon/cpu.h"
#include "carbon/engine.h"
#include "carbon/error.h"
#include "carbon/session.h"
#include "test_model.h"

namespace carbon {
namespace {

std::filesystem::path data(const std::string& name) { return std::filesystem::path(CARBON_TEST_DATA) / name; }

// <name>.io: b"CIO1", a count, then each tensor's name, rank, dims and floats.
std::vector<std::pair<std::string, Tensor>> read_io(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  const std::string b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  size_t at = 0;
  auto take = [&](void* out, size_t n) {
    if (at + n > b.size()) throw Error(path.string() + ": too short");
    std::memcpy(out, b.data() + at, n);
    at += n;
  };
  char magic[4];
  take(magic, 4);
  if (std::string(magic, 4) != "CIO1") throw Error(path.string() + ": not a CIO1 file");
  uint32_t count;
  take(&count, 4);
  std::vector<std::pair<std::string, Tensor>> out;
  for (uint32_t i = 0; i < count; ++i) {
    uint32_t len, rank;
    take(&len, 4);
    std::string name(len, '\0');
    take(name.data(), len);
    take(&rank, 4);
    Shape s(rank);
    take(s.data(), 8 * rank);
    Tensor t = Tensor::empty(DType::kFloat32, s);
    take(t.raw(), t.bytes());
    out.emplace_back(name, std::move(t));
  }
  return out;
}

class NativeNet : public ::testing::TestWithParam<const char*> {};

TEST_P(NativeNet, GivesOnnxRuntimesOutputs) {
  const std::string name = GetParam();
  auto io = read_io(data(name + ".io"));
  auto s = open_session(Backend::kNative, data(name + ".onnx"));
  EXPECT_EQ(s->backend(), Backend::kNative);
  ASSERT_EQ(s->inputs().size(), 1u);
  ASSERT_EQ(s->outputs().size(), io.size() - 1);
  std::vector<Tensor> in;
  in.push_back(std::move(io[0].second));
  const std::vector<Tensor> out = s->run(in);
  for (size_t k = 0; k < out.size(); ++k) {
    const Tensor& want = io[k + 1].second;
    SCOPED_TRACE(name + " output " + io[k + 1].first);
    ASSERT_EQ(out[k].shape(), want.shape());
    const auto got = out[k].data<float>();
    const auto w = want.data<float>();
    float scale = 0;
    for (float v : w) scale = std::max(scale, std::abs(v));
    for (size_t i = 0; i < w.size(); ++i) ASSERT_NEAR(got[i], w[i], 2e-5f * scale + 1e-6f) << "at " << i;
  }
  // every instruction set: the same bits
  for (cpu::Isa isa : cpu::isas_available()) {
    SessionOptions o;
    o.isa = cpu::isa_name(isa);
    auto si = open_session(Backend::kNative, data(name + ".onnx"), o);
    const std::vector<Tensor> again = si->run(in);
    for (size_t k = 0; k < out.size(); ++k) {
      ASSERT_EQ(std::memcmp(again[k].raw(), out[k].raw(), out[k].bytes()), 0) << name << " on " << o.isa;
    }
  }
  // a second run of other shapes, then the first again: the buffers resize and
  // their borders stay zero
  Shape other = in[0].shape();
  other[0] = 1;
  other[3] += 8;
  std::vector<Tensor> in2;
  in2.push_back(Tensor(DType::kFloat32, other));
  s->run(in2);
  const std::vector<Tensor> third = s->run(in);
  for (size_t k = 0; k < out.size(); ++k) EXPECT_EQ(std::memcmp(third[k].raw(), out[k].raw(), out[k].bytes()), 0);
}

INSTANTIATE_TEST_SUITE_P(Fullerene, NativeNet, ::testing::Values("native-rec", "native-det", "native-cls"),
                         [](const auto& info) {
                           std::string n = info.param;
                           n.erase(std::remove(n.begin(), n.end(), '-'), n.end());
                           return n;
                         });

// The Q/DQ files ONNX Runtime's quantizer makes of them run in int8: as close to
// the FP32 network as ONNX Runtime's own int8 run of the file (the outputs
// leave the last layer as float, finer than its quantized outputs), and with the
// same bits on every instruction set.
class NativeInt8 : public ::testing::TestWithParam<const char*> {};

TEST_P(NativeInt8, IsAsCloseToFp32AsOnnxRuntime) {
  const std::string name = GetParam();
  auto fp32 = read_io(data(name + ".io"));
  auto ort8 = read_io(data(name + ".int8.io"));
  ASSERT_EQ(std::memcmp(fp32[0].second.raw(), ort8[0].second.raw(), fp32[0].second.bytes()), 0) << "one input";
  std::vector<Tensor> in;
  in.push_back(std::move(fp32[0].second));
  auto s = open_session(Backend::kNative, data(name + ".int8.onnx"));
  const std::vector<Tensor> out = s->run(in);
  ASSERT_EQ(out.size(), fp32.size() - 1);
  for (size_t k = 0; k < out.size(); ++k) {
    const auto want = fp32[k + 1].second.data<float>(), theirs = ort8[k + 1].second.data<float>();
    const auto got = out[k].data<float>();
    ASSERT_EQ(got.size(), want.size());
    // the error's root mean square (the largest error of a few hundred values
    // is one rounding's luck), and the largest
    double ours = 0, ort = 0, ours_max = 0, scale = 0;
    for (size_t i = 0; i < want.size(); ++i) {
      ours += (got[i] - want[i]) * static_cast<double>(got[i] - want[i]);
      ort += (theirs[i] - want[i]) * static_cast<double>(theirs[i] - want[i]);
      ours_max = std::max(ours_max, static_cast<double>(std::abs(got[i] - want[i])));
      scale = std::max(scale, static_cast<double>(std::abs(want[i])));
    }
    ours = std::sqrt(ours / want.size());
    ort = std::sqrt(ort / want.size());
    std::printf("%s %s: int8 against FP32, rms error %.5f native, %.5f ONNX Runtime; largest %.4f (of %.3f)\n",
                name.c_str(), fp32[k + 1].first.c_str(), ours, ort, ours_max, scale);
    EXPECT_LE(ours, 1.1 * ort) << name;
    EXPECT_LE(ours_max, 0.05 * scale) << name;
  }
  for (cpu::Isa isa : cpu::isas_available()) {
    SessionOptions o;
    o.isa = cpu::isa_name(isa);
    const std::vector<Tensor> again = open_session(Backend::kNative, data(name + ".int8.onnx"), o)->run(in);
    for (size_t k = 0; k < out.size(); ++k) {
      ASSERT_EQ(std::memcmp(again[k].raw(), out[k].raw(), out[k].bytes()), 0) << name << " on " << o.isa;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Fullerene, NativeInt8, ::testing::Values("native-rec", "native-det", "native-cls"),
                         [](const auto& info) {
                           std::string n = info.param;
                           n.erase(std::remove(n.begin(), n.end(), '-'), n.end());
                           return n;
                         });

// A set's model on the CPU: its int8 file on the native backend when the set has
// one, else ONNX Runtime where it is built in; a backend asked for is the one.
TEST(NativeBackend, ASetChoosesItsFastestModel) {
  const auto dir = std::filesystem::temp_directory_path() / "carbon-set-choice";
  std::filesystem::create_directories(dir);
  for (const char* f : {"a.onnx", "a.int8.onnx", "b.onnx"}) std::ofstream(dir / f) << "x";
  const auto built = compiled_backends();
  const bool ort = std::find(built.begin(), built.end(), Backend::kOnnxRuntime) != built.end();
  EXPECT_EQ(choose_set_model(dir, "a", std::nullopt).backend, Backend::kNative);
  EXPECT_EQ(choose_set_model(dir, "a", std::nullopt).file, dir / "a.int8.onnx");
  EXPECT_EQ(choose_set_model(dir, "b", std::nullopt).backend, ort ? Backend::kOnnxRuntime : Backend::kNative);
  EXPECT_EQ(choose_set_model(dir, "b", std::nullopt).file, dir / "b.onnx");
  EXPECT_EQ(choose_set_model(dir, "a", Backend::kOnnxRuntime).file, dir / "a.onnx");
  EXPECT_EQ(choose_set_model(dir, "b", Backend::kNative).backend, Backend::kNative);
  EXPECT_EQ(choose_set_model(dir, "b", Backend::kNative).file, dir / "b.onnx");
  std::filesystem::remove_all(dir);
}

TEST(NativeBackend, RefusesWhatItDoesNotRun) {
  EXPECT_EQ(parse_backend("native"), Backend::kNative);
  EXPECT_EQ(backend_name(Backend::kNative), "native");
  // a Relu on the input, ReduceSum and ArgMax: none of them Fullerene's
  try {
    open_session(Backend::kNative, test::model_path());
    FAIL() << "opened";
  } catch (const Error& e) {
    EXPECT_NE(std::string(e.what()).find("native backend: Relu"), std::string::npos) << e.what();
  }
  SessionOptions o;
  o.isa = "no-such-set";
  EXPECT_THROW(open_session(Backend::kNative, data("native-cls.onnx"), o), Error);
}

}  // namespace
}  // namespace carbon
