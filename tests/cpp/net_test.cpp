// The native backend (carbon/src/net/) on Fullerene's networks in miniature
// (tools/make_test_models.py): ONNX Runtime's outputs, saved beside each model,
// so the check runs on every machine; every instruction set's bits alike.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "carbon/cpu.h"
#include "carbon/engine.h"
#include "carbon/error.h"
#include "carbon/session.h"
#include "test_model.h"
#include "../../carbon/src/det_head.h"

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

// The detector's upsamplings read in place by the convolutions whose residual they are, not written: the same
// bits as with them written (CARBON_NATIVE_UPSAMPLE_IN_PLACE=0), FP32, int8 and FP16 (its tensors FP32 and FP16), on
// every instruction set.
TEST(Net, TheUpsamplingReadInPlaceIsTheUpsamplingWritten) {
  auto io = read_io(data("native-det.io"));
  std::vector<Tensor> in;
  in.push_back(std::move(io[0].second));
  for (const char* file : {"native-det.onnx", "native-det.int8.onnx", "native-det.fp16.onnx"}) {
    for (const char* act : {"0", "1"}) {
      const ScopedEnv tensors("CARBON_FP16_ACTIVATIONS", act);
      for (cpu::Isa isa : cpu::isas_available()) {
        SessionOptions o;
        o.isa = cpu::isa_name(isa);
        std::vector<Tensor> written;
        {
          const ScopedEnv off("CARBON_NATIVE_UPSAMPLE_IN_PLACE", "0");
          written = open_session(Backend::kNative, data(file), o)->run(in);
        }
        const std::vector<Tensor> read = open_session(Backend::kNative, data(file), o)->run(in);
        ASSERT_EQ(read.size(), written.size());
        for (size_t k = 0; k < read.size(); ++k) {
          ASSERT_EQ(read[k].shape(), written[k].shape());
          ASSERT_EQ(std::memcmp(read[k].raw(), written[k].raw(), read[k].bytes()), 0)
              << file << ", FP16 tensors " << act << ", on " << o.isa;
        }
      }
    }
  }
}

// FP16 tensors on an instruction set without FP16 arithmetic, each input row widened once into an FP32 ring:
// the same bits as a value widened a tap (CARBON_NATIVE_WIDEN_ONCE=0), every network, on every such instruction set,
// and a second run of other shapes, then the first again, the same.
TEST(Net, Fp16TensorsWidenedOnceAreWidenedATap) {
  const ScopedEnv tensors("CARBON_FP16_ACTIVATIONS", "1");
  for (const char* name : {"native-det", "native-rec", "native-cls"}) {
    auto io = read_io(data(std::string(name) + ".io"));
    std::vector<Tensor> in;
    in.push_back(std::move(io[0].second));
    const std::filesystem::path file = data(std::string(name) + ".fp16.onnx");
    for (cpu::Isa isa : cpu::isas_available()) {
      if (cpu::f16_arithmetic(isa)) continue;
      SessionOptions o;
      o.isa = cpu::isa_name(isa);
      std::vector<Tensor> tap;
      {
        const ScopedEnv off("CARBON_NATIVE_WIDEN_ONCE", "0");
        tap = open_session(Backend::kNative, file, o)->run(in);
      }
      auto s = open_session(Backend::kNative, file, o);
      const std::vector<Tensor> once = s->run(in);
      Shape other = in[0].shape();
      other[0] = 1;
      other[3] += 8;
      std::vector<Tensor> in2;
      in2.push_back(Tensor(DType::kFloat32, other));
      s->run(in2);
      const std::vector<Tensor> again = s->run(in);
      ASSERT_EQ(once.size(), tap.size());
      for (size_t k = 0; k < once.size(); ++k) {
        ASSERT_EQ(std::memcmp(once[k].raw(), tap[k].raw(), once[k].bytes()), 0) << name << " on " << o.isa;
        ASSERT_EQ(std::memcmp(again[k].raw(), tap[k].raw(), once[k].bytes()), 0) << name << " again on " << o.isa;
      }
    }
  }
}

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


// The detector cut before its head (carbon/src/det_head.h), its head then run by the engine (cpu::det_head), gives
// the whole detector's maps: FP32 on the native backend and on ONNX Runtime and FP16 on the native backend within
// float rounding; int8 within the quantization the whole file's head adds and the engine's FP32 head does not.
TEST(Net, TheDetectorCutBeforeItsHeadGivesItsMaps) {
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  Tensor x = Tensor::empty(DType::kFloat32, {1, 1, 64, 48});
  for (float& v : x.data<float>()) v = u(rng) < 0.3f ? u(rng) : 0.0f;  // ink, sparse
  struct Case {
    const char* file;
    Backend backend;
    float tolerance;  // of the largest map value
  };
  std::vector<Case> cases = {{"native-det.onnx", Backend::kNative, 1e-4f},
                             {"native-det.fp16.onnx", Backend::kNative, 1e-2f},
                             {"native-det.int8.onnx", Backend::kNative, 0.05f}};
  const auto built = compiled_backends();
  if (std::find(built.begin(), built.end(), Backend::kOnnxRuntime) != built.end()) {
    cases.push_back({"native-det.onnx", Backend::kOnnxRuntime, 1e-4f});
  }
  for (const Case& c : cases) {
    const auto head = detail::fused_det_head(data(c.file));
    ASSERT_TRUE(head.has_value()) << c.file;
    EXPECT_EQ(head->mid, 8);
    EXPECT_EQ(head->out, 4);
    const auto whole = open_session(c.backend, data(c.file));
    SessionOptions cut;
    cut.model_bytes = head->cut;
    const auto part = open_session(c.backend, data(c.file), cut);
    const std::span<const Tensor> in(&x, 1);
    const Tensor want = std::move(whole->run(in)[0]);
    const Tensor y = std::move(part->run(in)[0]);
    ASSERT_EQ(y.shape(), (Shape{1, head->in, 16, 12})) << c.file;
    ASSERT_EQ(want.shape(), (Shape{1, 4, 32, 24})) << c.file;
    Tensor got = Tensor::empty(DType::kFloat32, {1, 4, 32, 24});
    cpu::det_head(y.data<float>().data(), 16, 12, head->weights(), got.data<float>().data(), 0, 32);
    float most = 0, worst = 0;
    const std::span<const float> a = want.data<float>();
    const std::span<const float> b = std::as_const(got).data<float>();
    for (size_t i = 0; i < a.size(); ++i) {
      most = std::max(most, std::abs(a[i]));
      worst = std::max(worst, std::abs(a[i] - b[i]));
    }
    EXPECT_GT(most, 0.0f) << c.file;
    EXPECT_LE(worst, c.tolerance * most) << c.file << " on " << backend_name(c.backend);
  }
  EXPECT_FALSE(detail::fused_det_head(data("native-rec.onnx")).has_value());  // no detector head there
}

// The time on a real detector (CARBON_DET_TIME: its det.onnx) at 1024 x 800, one thread: ONNX Runtime from the
// ink map against the engine's stem and ONNX Runtime from the stride-4 tensor, the head the engine's in both.
TEST(Net, DetStemTime) {
  const char* file = std::getenv("CARBON_DET_TIME");
  if (!file) GTEST_SKIP() << "set CARBON_DET_TIME to a det.onnx to time it";
  auto head = detail::fused_det_head(file);
  ASSERT_TRUE(head.has_value());
  auto stemmed = head;
  ASSERT_TRUE(detail::with_det_stem(*stemmed));
  SessionOptions a, b;
  a.model_bytes = head->cut;
  b.model_bytes = stemmed->cut;
  const auto front = open_session(Backend::kOnnxRuntime, file, a), back = open_session(Backend::kOnnxRuntime, file, b);
  std::mt19937 rng(4);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  const int H = 1024, Wd = 800;
  Tensor x = Tensor::empty(DType::kFloat32, {1, 1, H, Wd});
  for (float& v : x.data<float>()) v = u(rng) < 0.1f ? u(rng) : 0.0f;
  Tensor s4 = Tensor::empty(DType::kFloat32, {1, stemmed->stem->c2, H / 4, Wd / 4});
  auto time = [&](const char* what, const auto& fn) {
    double best = 1e30;
    for (int round = 0; round < 7; ++round) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < 5; ++r) fn();
      best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 5);
    }
    std::printf("%s: %.2f ms\n", what, best);
  };
  time("ONNX Runtime from the ink map", [&] { front->run(std::span<const Tensor>(&x, 1)); });
  time("the stem alone", [&] {
    cpu::det_stem(x.data<float>().data(), H, Wd, stemmed->stem->weights(), s4.data<float>().data(), 0, H / 4);
  });
  time("the stem and ONNX Runtime from the stride-4 tensor", [&] {
    cpu::det_stem(x.data<float>().data(), H, Wd, stemmed->stem->weights(), s4.data<float>().data(), 0, H / 4);
    back->run(std::span<const Tensor>(&s4, 1));
  });
}

// The detector cut at both ends, fed the stride-4 tensor cpu::det_stem makes, its head run by the engine, gives
// the whole detector's maps within float rounding on ONNX Runtime (FP32 files alone take the stem).
TEST(Net, TheDetectorFedItsStrideFourTensorGivesItsMaps) {
  const auto built = compiled_backends();
  if (std::find(built.begin(), built.end(), Backend::kOnnxRuntime) == built.end()) GTEST_SKIP() << "no ONNX Runtime";
  std::mt19937 rng(4);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  Tensor x = Tensor::empty(DType::kFloat32, {1, 1, 64, 48});
  for (float& v : x.data<float>()) v = u(rng) < 0.3f ? u(rng) : 0.0f;
  auto head = detail::fused_det_head(data("native-det.onnx"));
  ASSERT_TRUE(head.has_value());
  ASSERT_TRUE(detail::with_det_stem(*head));
  ASSERT_TRUE(head->stem.has_value());
  EXPECT_EQ(head->stem->c1, 8);
  EXPECT_EQ(head->stem->c2, 12);
  EXPECT_FALSE(detail::with_det_stem(*head));  // once
  const auto whole = open_session(Backend::kOnnxRuntime, data("native-det.onnx"));
  SessionOptions cut;
  cut.model_bytes = head->cut;
  const auto part = open_session(Backend::kOnnxRuntime, data("native-det.onnx"), cut);
  ASSERT_EQ(part->inputs().size(), 1u);
  EXPECT_EQ(part->inputs()[0].name, head->stem->output);
  Tensor s4 = Tensor::empty(DType::kFloat32, {1, 12, 16, 12});
  cpu::det_stem(x.data<float>().data(), 64, 48, head->stem->weights(), s4.data<float>().data(), 0, 16);
  const Tensor want = std::move(whole->run(std::span<const Tensor>(&x, 1))[0]);
  const Tensor y = std::move(part->run(std::span<const Tensor>(&s4, 1))[0]);
  ASSERT_EQ(y.shape(), (Shape{1, head->in, 16, 12}));
  Tensor got = Tensor::empty(DType::kFloat32, {1, 4, 32, 24});
  cpu::det_head(y.data<float>().data(), 16, 12, head->weights(), got.data<float>().data(), 0, 32);
  float most = 0, worst = 0;
  const std::span<const float> a = want.data<float>();
  const std::span<const float> b = std::as_const(got).data<float>();
  for (size_t i = 0; i < a.size(); ++i) {
    most = std::max(most, std::abs(a[i]));
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  EXPECT_GT(most, 0.0f);
  EXPECT_LE(worst, 1e-4f * most);
  // an FP16 file keeps its front
  auto half = detail::fused_det_head(data("native-det.fp16.onnx"));
  ASSERT_TRUE(half.has_value());
  EXPECT_FALSE(detail::with_det_stem(*half));
}

}  // namespace
}  // namespace carbon
