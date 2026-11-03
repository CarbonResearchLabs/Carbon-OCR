// The native backend (carbon/src/net/) on Fullerene's networks in miniature
// (tools/make_test_models.py): ONNX Runtime's outputs, saved beside each model,
// so the check runs on every machine; every instruction set's bits alike.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// An environment variable set for a scope (CARBON_FP16_ACTIVATIONS), then removed.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) { set(value); }
  ~ScopedEnv() { set(""); }
  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  void set(const char* value) {
#if defined(_WIN32)
    _putenv_s(name_, value);  // "" removes it
#else
    if (*value) {
      setenv(name_, value, 1);
    } else {
      unsetenv(name_);
    }
#endif
  }
  const char* name_;
};

// The decisions the engine takes from an output: the best class along the
// channels (a recognizer's frames, a classifier's line), or for a detector's
// maps whether each pixel's core logit (channel 0) is above 0.  -1 where the
// reference's margin is under `margin` (a near tie either way is fair).
std::vector<int> decisions(const std::string& name, const Tensor& t, const Tensor& ref, float margin) {
  const Shape& s = t.shape();
  const auto v = t.data<float>(), r = ref.data<float>();
  std::vector<int> out;
  if (name == "native-det") {  // [N, 4, H, W]
    const int64_t plane = s[2] * s[3];
    for (int64_t i = 0; i < s[0]; ++i) {
      for (int64_t p = 0; p < plane; ++p) {
        const size_t at = static_cast<size_t>(i * s[1] * plane + p);
        out.push_back(std::abs(r[at]) < margin ? -1 : v[at] > 0);
      }
    }
    return out;
  }
  const int64_t classes = s[1], inner = s.size() == 3 ? s[2] : 1;  // [N, C, T] or [N, C]
  for (int64_t i = 0; i < s[0]; ++i) {
    for (int64_t f = 0; f < inner; ++f) {
      auto at = [&](int64_t c) { return static_cast<size_t>((i * classes + c) * inner + f); };
      int64_t best = 0, ref_best = 0;
      for (int64_t c = 1; c < classes; ++c) {
        if (v[at(c)] > v[at(best)]) best = c;
        if (r[at(c)] > r[at(ref_best)]) ref_best = c;
      }
      float second = -INFINITY;
      for (int64_t c = 0; c < classes; ++c) {
        if (c != ref_best) second = std::max(second, r[at(c)]);
      }
      out.push_back(r[at(ref_best)] - second < margin ? -1 : static_cast<int>(best));
    }
  }
  return out;
}

// The FP16 files (tools/make_test_models.py: the FP32 networks' weights in
// float16, input and output float16, as the export writes them) run on the
// FP16 kernels with float32 at the session: as close to ONNX Runtime's FP32
// outputs (native-*.io) as FP16's rounding allows, and the same decisions where
// FP32's are clear of a tie.  With the tensors between layers in FP32 (the FP16
// weights the one rounding) and in FP16 (as on a CPU with FP16 arithmetic,
// CARBON_FP16_ACTIVATIONS=1); every instruction set with the reference's bits but
// the FP16 arithmetic ones (within their rounding); and a second run of other
// shapes, then the first again, the same bits.
class NativeFp16 : public ::testing::TestWithParam<const char*> {};

TEST_P(NativeFp16, IsFp32WithinFp16Rounding) {
  const std::string name = GetParam();
  auto fp32 = read_io(data(name + ".io"));
  std::vector<Tensor> in;
  in.push_back(std::move(fp32[0].second));
  // The outputs against FP32's: the largest error and the root mean square under
  // these shares of the largest output, the decisions clear of a tie FP32's.
  auto near_fp32 = [&](const std::vector<Tensor>& out, double largest, double rms_share, const std::string& what) {
    ASSERT_EQ(out.size(), fp32.size() - 1);
    for (size_t k = 0; k < out.size(); ++k) {
      const Tensor& want = fp32[k + 1].second;
      ASSERT_EQ(out[k].shape(), want.shape());
      const auto w = want.data<float>(), got = out[k].data<float>();
      double rms = 0, worst = 0, scale = 0;
      for (size_t i = 0; i < w.size(); ++i) {
        const double d = std::abs(static_cast<double>(got[i]) - w[i]);
        rms += d * d;
        worst = std::max(worst, d);
        scale = std::max(scale, static_cast<double>(std::abs(w[i])));
      }
      rms = std::sqrt(rms / w.size());
      std::printf("%s %s, %s: against FP32, rms error %.2e, largest %.2e (of %.3f)\n", name.c_str(),
                  fp32[k + 1].first.c_str(), what.c_str(), rms, worst, scale);
      EXPECT_LE(worst, largest * scale) << name << " " << what;
      EXPECT_LE(rms, rms_share * scale) << name << " " << what;
      const float margin = static_cast<float>(4 * worst);
      const std::vector<int> mine = decisions(name, out[k], want, margin), theirs = decisions(name, want, want, margin);
      int clear = 0;
      for (size_t i = 0; i < mine.size(); ++i) {
        if (theirs[i] < 0) continue;
        ++clear;
        ASSERT_EQ(mine[i], theirs[i]) << name << " " << what << ": decision " << i;
      }
      EXPECT_GT(clear, static_cast<int>(mine.size()) / 2) << name << " " << what << ": most decisions clear of a tie";
    }
  };
  for (const char* act : {"0", "1"}) {
    const ScopedEnv env("CARBON_FP16_ACTIVATIONS", act);
    const bool act16 = std::string(act) == "1";
    const std::string tensors = act16 ? "FP16 tensors" : "FP32 tensors";
    SessionOptions scalar;
    scalar.isa = "scalar";
    auto s = open_session(Backend::kNative, data(name + ".fp16.onnx"), scalar);
    ASSERT_EQ(s->inputs().at(0).dtype, DType::kFloat32);
    for (const TensorInfo& o : s->outputs()) EXPECT_EQ(o.dtype, DType::kFloat32);
    const std::vector<Tensor> out = s->run(in);
    near_fp32(out, act16 ? 1e-2 : 2.5e-3, act16 ? 2.5e-3 : 6e-4, tensors);
    for (cpu::Isa isa : cpu::isas_available()) {
      SessionOptions o;
      o.isa = cpu::isa_name(isa);
      const std::vector<Tensor> again = open_session(Backend::kNative, data(name + ".fp16.onnx"), o)->run(in);
      if (act16 && cpu::f16_arithmetic(isa)) {  // sums rounded to FP16 too
        near_fp32(again, 3e-2, 8e-3, tensors + " on " + o.isa);
        continue;
      }
      for (size_t k = 0; k < out.size(); ++k) {
        ASSERT_EQ(std::memcmp(again[k].raw(), out[k].raw(), out[k].bytes()), 0) << name << " on " << o.isa;
      }
    }
    Shape other = in[0].shape();
    other[0] = 1;
    other[3] += 8;
    std::vector<Tensor> in2;
    in2.push_back(Tensor(DType::kFloat32, other));
    s->run(in2);
    const std::vector<Tensor> third = s->run(in);
    for (size_t k = 0; k < out.size(); ++k) EXPECT_EQ(std::memcmp(third[k].raw(), out[k].raw(), out[k].bytes()), 0);
  }
}

INSTANTIATE_TEST_SUITE_P(Fullerene, NativeFp16, ::testing::Values("native-rec", "native-det", "native-cls"),
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

// A precision asked for: its file, on the native backend for FP16 and int8 unless
// int8 is asked of ONNX Runtime, FP32 as without one but never the int8 file; a
// set without the file, or FP16 asked of another backend, is an error.
TEST(NativeBackend, APrecisionChoosesItsFile) {
  EXPECT_EQ(parse_precision("fp16"), Precision::kFp16);
  EXPECT_EQ(parse_precision("auto"), Precision::kAuto);
  EXPECT_EQ(parse_precision("half"), std::nullopt);
  EXPECT_EQ(precision_name(Precision::kInt8), "int8");
  const auto dir = std::filesystem::temp_directory_path() / "carbon-set-precision";
  std::filesystem::create_directories(dir);
  for (const char* f : {"a.onnx", "a.int8.onnx", "a.fp16.onnx", "b.onnx"}) std::ofstream(dir / f) << "x";
  const auto built = compiled_backends();
  const bool ort = std::find(built.begin(), built.end(), Backend::kOnnxRuntime) != built.end();
  const Backend fastest_fp32 = ort ? Backend::kOnnxRuntime : Backend::kNative;
  // auto: the fastest, as without a precision
  EXPECT_EQ(choose_set_model(dir, "a", std::nullopt, Precision::kAuto).file, dir / "a.int8.onnx");
  EXPECT_EQ(choose_set_model(dir, "b", std::nullopt, Precision::kAuto).backend, fastest_fp32);
  // fp32
  EXPECT_EQ(choose_set_model(dir, "a", std::nullopt, Precision::kFp32).file, dir / "a.onnx");
  EXPECT_EQ(choose_set_model(dir, "a", std::nullopt, Precision::kFp32).backend, fastest_fp32);
  EXPECT_EQ(choose_set_model(dir, "a", Backend::kNative, Precision::kFp32).backend, Backend::kNative);
  EXPECT_EQ(choose_set_model(dir, "a", Backend::kNative, Precision::kFp32).file, dir / "a.onnx");
  // fp16: native alone
  EXPECT_EQ(choose_set_model(dir, "a", std::nullopt, Precision::kFp16).file, dir / "a.fp16.onnx");
  EXPECT_EQ(choose_set_model(dir, "a", std::nullopt, Precision::kFp16).backend, Backend::kNative);
  EXPECT_EQ(choose_set_model(dir, "a", Backend::kNative, Precision::kFp16).file, dir / "a.fp16.onnx");
  EXPECT_THROW(choose_set_model(dir, "a", Backend::kOnnxRuntime, Precision::kFp16), Error);
  EXPECT_THROW(choose_set_model(dir, "b", std::nullopt, Precision::kFp16), Error);
  // int8
  EXPECT_EQ(choose_set_model(dir, "a", std::nullopt, Precision::kInt8).file, dir / "a.int8.onnx");
  EXPECT_EQ(choose_set_model(dir, "a", std::nullopt, Precision::kInt8).backend, Backend::kNative);
  EXPECT_EQ(choose_set_model(dir, "a", Backend::kOnnxRuntime, Precision::kInt8).backend, Backend::kOnnxRuntime);
  EXPECT_THROW(choose_set_model(dir, "b", std::nullopt, Precision::kInt8), Error);
  try {
    choose_set_model(dir, "b", std::nullopt, Precision::kInt8);
  } catch (const Error& e) {
    EXPECT_NE(std::string(e.what()).find("b.int8.onnx"), std::string::npos) << e.what();
  }
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
