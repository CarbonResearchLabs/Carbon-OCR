// The engine's own executor for Fullerene's convolutional networks (det, orient,
// script, the recognizers): an ONNX file compiled into layers that run on the
// CPU kernels (carbon/cpu.h), behind carbon::Session (Backend::kNative).
//
// Compiling: the nodes are read in order and fused.  A Conv takes the Relu after
// it and a residual Add (its output used by nothing else), so its output stage
// does the bias, the residual and the ReLU while the sums are still in L1; an
// Identity or a Squeeze is a name for its input.  Conv (dense or depthwise),
// Resize (nearest, whole scales), AveragePool, ReduceMean over the image, Gemm
// and Add are the layers; any other op is refused when the file is opened.
//
// int8: a file with QuantizeLinear and DequantizeLinear nodes (the Q/DQ format
// ONNX Runtime's quantizer and quantization-aware training write) runs in int8.
// A first pass reads the Q/DQ nodes away: a weight's DequantizeLinear gives its
// int8 values and per-channel scales, a bias's its float values, and a Q then DQ
// on a tensor says the tensor is kept as uint8 with that scale and zero point.
// What is left is the float graph, compiled as above; a convolution whose input
// is uint8 and whose weights are int8 runs on the int8 kernels, and every layer
// writes its output as uint8 or float, as its tensor is kept.  A residual Add
// joins its convolution in float (the sum is quantized once, not twice).
//
// FP16: a file whose input is float16 (the export's <model>.fp16.onnx: the
// same graph, float16 weights and input and output) runs its convolutions on the
// FP16 kernels (carbon/cpu.h), the weights packed in FP16.  Its tensors between
// the layers are kept in FP16 where the instruction set has FP16 arithmetic
// (avx512-fp16, neon-fp16: the kernels read and write FP16, twice the lanes of
// FP32), else in FP32.  Without FP16 arithmetic (F16C, plain NEON) every value
// is widened to FP32 before it is summed, so FP16 tensors only add conversions:
// measured on an i7-13620H (AVX2, F16C), one thread on a P-core, the median of
// three rounds of 30 to 100 runs, det at 1024 x 800 took 59.7 ms with FP32
// tensors and 64.9 ms with FP16 ones, orient at 512 x 512 6.3 and 7.8 ms,
// rec-latin on 8 lines of 512 px 34.6 and 35.0 ms (the FP32 file: 59.7, 6.5 and
// 34.4 ms), and FP32 tensors keep a rounding fewer.  The graph's outputs are
// FP32, as the layer's float sums give them (finer than FP16).
// CARBON_FP16_ACTIVATIONS=1 or 0 in the environment chooses (tests, timing).  At
// the session the input and outputs are float32, as every other file's.
//
// Running: every tensor is NHWC, channels innermost, with a border of zeros (the
// zero point, for uint8) as wide as its consumers' padding, so each tap of a
// convolution reads inside the buffer and an output row of a convolution is one
// GEMM whose rows of A are a grid on its input (base, the pixels' step, a tap's
// offset): nothing is copied into columns and no pointers are built per pixel.
// The graph's input and outputs are float NCHW, as ONNX has them.

#include "net.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "carbon/cpu.h"
#include "carbon/error.h"
#include "onnx.h"

namespace carbon::net {
namespace {

[[noreturn]] void refuse(const std::string& why) { throw Error("native backend: " + why); }

struct Quant {
  float scale = 1;
  int32_t zero = 0;
  bool operator==(const Quant& o) const { return scale == o.scale && zero == o.zero; }
};

// round(v / scale) + zero, clamped to 0 .. 255, rounding half to even as
// QuantizeLinear does: adding and taking away 1.5 * 2^23 leaves a float of
// magnitude under 2^22 rounded to a whole number (no call, so it vectorizes).
inline uint8_t quantize(float v, const Quant& q) {
  float r = v / q.scale;
  r = std::min(4194304.0f, std::max(-4194304.0f, r));
  r = (r + 12582912.0f) - 12582912.0f;
  r += static_cast<float>(q.zero);
  return static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, r)));
}

// A tensor: NHWC with a border ph rows and pw columns wide that holds zeros
// (uint8: the zero point), kept as float, as uint8 (real = (v - zero) * scale) or
// as FP16 (an FP16 file's, where the instruction set sums in FP16).
struct Value {
  std::string name, last;  // last: the name of the op it ends as (fused ops give it theirs)
  int c = 0;
  int ph = 0, pw = 0;
  bool u8 = false, f16 = false;
  Quant q;
  int n = 0, h = 0, w = 0;  // this run's
  std::vector<float> f;
  std::vector<uint8_t> b;
  std::vector<uint16_t> hv;  // FP16

  int64_t hp() const { return h + 2 * ph; }
  int64_t wp() const { return w + 2 * pw; }
  int64_t pixel(int i, int y, int x) const { return ((static_cast<int64_t>(i) * hp() + y + ph) * wp() + x + pw) * c; }
  float* fat(int i, int y, int x) { return f.data() + pixel(i, y, x); }
  const float* fat(int i, int y, int x) const { return f.data() + pixel(i, y, x); }
  uint8_t* bat(int i, int y, int x) { return b.data() + pixel(i, y, x); }
  const uint8_t* bat(int i, int y, int x) const { return b.data() + pixel(i, y, x); }
  uint16_t* hat(int i, int y, int x) { return hv.data() + pixel(i, y, x); }
  const uint16_t* hat(int i, int y, int x) const { return hv.data() + pixel(i, y, x); }
  float real(int i, int y, int x, int ch) const {
    if (f16) return cpu::f16_to_f32(hat(i, y, x)[ch]);
    return u8 ? static_cast<float>(static_cast<int32_t>(bat(i, y, x)[ch]) - q.zero) * q.scale : fat(i, y, x)[ch];
  }
  void set(int i, int y, int x, int ch, float v) {
    if (u8) {
      bat(i, y, x)[ch] = quantize(v, q);
    } else if (f16) {
      hat(i, y, x)[ch] = cpu::f32_to_f16(v);
    } else {
      fat(i, y, x)[ch] = v;
    }
  }
  // Bytes a value: what a copy of a pixel's channels moves.
  size_t element() const { return u8 ? 1 : f16 ? 2 : 4; }
  const void* raw(int i, int y, int x) const {
    return u8 ? static_cast<const void*>(bat(i, y, x)) : f16 ? static_cast<const void*>(hat(i, y, x)) : fat(i, y, x);
  }
  void* raw(int i, int y, int x) { return const_cast<void*>(static_cast<const Value*>(this)->raw(i, y, x)); }
  bool same_storage(const Value& o) const { return u8 == o.u8 && f16 == o.f16 && (!u8 || q == o.q); }

  // Sized for n x h x w with the border filled (the inside is written by the
  // layer that makes it).  The slack past the end is for the kernels' wide reads.
  void shape(int n_, int h_, int w_) {
    n = n_;
    h = h_;
    w = w_;
    const size_t size = static_cast<size_t>(n) * hp() * wp() * c + 64;
    if (u8) {
      fill(b, size, static_cast<uint8_t>(q.zero));
    } else if (f16) {
      fill(hv, size, uint16_t{0});
    } else {
      fill(f, size, 0.0f);
    }
  }

 private:
  template <typename T>
  void fill(std::vector<T>& v, size_t size, T zero) {
    // A new size: resized, not refilled whole: the inside is the making layer's to write, the
    // border is zeroed below and the slack here, so a run whose batch or width changed (most of recognition's) no
    // longer zeroes every tensor end to end.
    if (v.size() != size) {
      v.resize(size, zero);
      std::fill(v.end() - 64, v.end(), zero);  // the slack, for the kernels' wide reads
    }
    if (!ph && !pw) return;
    for (int i = 0; i < n; ++i) {
      for (int y = -ph; y < h + ph; ++y) {
        T* row = v.data() + pixel(i, y, -pw);
        if (y < 0 || y >= h) {
          std::fill(row, row + wp() * c, zero);
        } else {
          std::fill(row, row + static_cast<int64_t>(pw) * c, zero);
          T* right = v.data() + pixel(i, y, w);
          std::fill(right, right + static_cast<int64_t>(pw) * c, zero);
        }
      }
    }
  }
};

// A stored value as the pools sum it: uint8 as its integer (dequantized after the
// sum), FP16 widened.
inline double stored(uint8_t v) { return v; }
inline double stored(uint16_t v) { return cpu::f16_to_f32(v); }
inline double stored(float v) { return v; }

// A weight read from a DequantizeLinear: its int8 values and its scales (one, or
// one per output channel).
struct QWeight {
  std::vector<int8_t> values;
  std::vector<float> scales;
};

enum class Kind { kConv, kDwConv, kResize, kAvgPool, kMean, kAdd };

struct Layer {
  Kind kind = Kind::kConv;
  std::string node;  // for errors
  int in = -1, out = -1, residual = -1, other = -1;  // values (other: Add's second input)
  int kh = 1, kw = 1, sh = 1, sw = 1, pt = 0, pl = 0, pb = 0, pr = 0;  // a convolution or a pool
  int cin = 0, cout = 0;
  bool relu = false;
  std::vector<float> bias;
  // the weights as read, [cout][cin / groups][kh][kw], until they are packed
  std::vector<float> wf;
  std::optional<QWeight> w8;
  // packed: FP32, int8 with the sums' scales (the input's times the weights'), or FP16
  bool int8 = false, im2col = false, half = false;
  cpu::PackedF32 weights;
  cpu::PackedB weights8;
  cpu::PackedF16 weights16;
  std::vector<float> dw;      // depthwise FP32: [taps][c]
  std::vector<int8_t> dw8;    // depthwise int8: [taps][c]
  std::vector<uint16_t> dw16;  // depthwise FP16: [taps][c]
  std::vector<float> scale;
  int fh = 1, fw = 1;  // Resize's scales
};

struct Output {
  int value = -1;
  int rank = 4;
};

// An environment variable's value, or empty.
std::string env(const char* name) {
  std::string v;
#if defined(_MSC_VER)
  char* e = nullptr;
  size_t len = 0;
  if (_dupenv_s(&e, &len, name) == 0 && e) v = e;
  std::free(e);
#else
  if (const char* e = std::getenv(name)) v = e;
#endif
  return v;
}

class Net {
 public:
  // act16: an FP16 file's tensors between layers in FP16 (else FP32).
  Net(const onnx::Model& model, cpu::Isa isa, bool act16) : isa_(isa), act16_(act16) { compile(model.graph); }

  std::vector<TensorInfo> inputs, outputs;

  std::vector<Tensor> run(const Tensor& x) {
    const Shape& s = x.shape();
    Value& in = values_[0];
    in.shape(static_cast<int>(s[0]), static_cast<int>(s[2]), static_cast<int>(s[3]));
    const float* src = x.data<float>().data();
    for (int i = 0; i < in.n; ++i) {  // NCHW to NHWC, quantized or rounded to FP16 if the network takes it so
      for (int y = 0; y < in.h; ++y) {
        if (in.c == 1) {  // the same order: a row at a time
          const float* from = src + (static_cast<int64_t>(i) * in.h + y) * in.w;
          if (in.u8) {
            uint8_t* to = in.bat(i, y, 0);
            for (int x0 = 0; x0 < in.w; ++x0) to[x0] = quantize(from[x0], in.q);
          } else if (in.f16) {
            cpu::to_f16(from, in.hat(i, y, 0), static_cast<size_t>(in.w), isa_);
          } else {
            std::memcpy(in.fat(i, y, 0), from, sizeof(float) * in.w);
          }
          continue;
        }
        for (int x0 = 0; x0 < in.w; ++x0) {
          for (int c = 0; c < in.c; ++c) {
            in.set(i, y, x0, c, src[((static_cast<int64_t>(i) * in.c + c) * in.h + y) * in.w + x0]);
          }
        }
      }
    }
    if (profile_.empty()) {
      for (const Layer& l : layers_) step(l);
    } else {
      for (size_t i = 0; i < layers_.size(); ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        step(layers_[i]);
        profile_[i] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      }
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<Tensor> out;
    for (const Output& o : outputs_) out.push_back(to_nchw(values_[o.value], o.rank));
    if (!profile_.empty()) {
      profile_.back() += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      ++runs_;
    }
    return out;
  }

  // CARBON_NATIVE_PROFILE=1: each layer's time, summed over the runs, printed
  // to stderr when the net goes.
  void profile() {
    const std::string v = env("CARBON_NATIVE_PROFILE");
    if (!v.empty() && v != "0") profile_.assign(layers_.size() + 1, 0.0);
  }

  ~Net() {
    if (profile_.empty() || !runs_) return;
    double total = 0;
    for (double t : profile_) total += t;
    std::fprintf(stderr, "native: %d runs, %.3f ms a run\n", runs_, total / runs_);
    for (size_t i = 0; i <= layers_.size(); ++i) {
      std::string what = "outputs to NCHW";
      if (i < layers_.size()) {
        const Layer& l = layers_[i];
        const char* kinds[] = {"conv", "dwconv", "resize", "avgpool", "mean", "add"};
        const Value& y = values_[l.out];
        what = std::string(kinds[static_cast<int>(l.kind)]) + (l.int8 ? " int8 " : l.half ? " fp16 " : " ") +
               std::to_string(l.kh) + "x" + std::to_string(l.kw) + "/" + std::to_string(l.sh) + "," +
               std::to_string(l.sw) + " " + std::to_string(l.cin) + "->" + std::to_string(l.cout) + " out " +
               std::to_string(y.h) + "x" + std::to_string(y.w) + (y.u8 ? " u8" : y.f16 ? " f16" : "") +
               (l.relu ? " relu" : "") + (l.residual >= 0 ? " +res" : "");
      }
      std::fprintf(stderr, "  %8.3f ms %5.1f%%  %s\n", profile_[i] / runs_, 100 * profile_[i] / total, what.c_str());
    }
  }

 private:
  // ---------------------------------------------------------------- Q/DQ

  float scalar(const std::string& name) const {
    const std::vector<float> v = constant(name).floats();
    if (v.size() != 1) refuse("'" + name + "': a scale for the whole tensor, not per channel");
    return v[0];
  }

  // The nodes without their QuantizeLinear and DequantizeLinear: the weights
  // and biases they dequantize become constants, a Q then DQ on a tensor a note
  // of its quantization, and every input renamed to the float tensor it stands for.
  std::vector<onnx::Node> read_qdq(const onnx::Graph& g) {
    std::map<std::string, std::string> rename;                     // DQ's output -> the tensor
    std::map<std::string, std::pair<std::string, Quant>> pending;  // Q's output -> its input, quantization
    std::vector<onnx::Node> out;
    auto renamed = [&](const std::string& n) {
      auto it = rename.find(n);
      return it == rename.end() ? n : it->second;
    };
    for (onnx::Node n : g.nodes) {
      for (std::string& i : n.inputs) i = renamed(i);
      if (n.op == "Constant") {
        const onnx::Attribute* a = n.attribute("value");
        if (!a || !a->t) refuse("Constant '" + n.name + "' without a tensor value");
        owned_.push_back(*a->t);
        constants_[n.outputs[0]] = &owned_.back();
      } else if (fp16_ && (n.op == "QuantizeLinear" || n.op == "DequantizeLinear")) {
        refuse(n.op + " '" + n.name + "' in a float16 network");
      } else if (n.op == "QuantizeLinear") {
        const onnx::TensorData* z = n.inputs.size() > 2 && !n.inputs[2].empty() ? &constant(n.inputs[2]) : nullptr;
        if (z && z->dtype != onnx::kUint8) refuse("QuantizeLinear '" + n.name + "': uint8 activations only");
        pending[n.outputs[0]] = {n.inputs[0], {scalar(n.inputs[1]), z ? static_cast<int32_t>(z->uint8s()[0]) : 0}};
      } else if (n.op == "DequantizeLinear" && pending.count(n.inputs[0])) {
        const auto& [x, q] = pending[n.inputs[0]];
        rename[n.outputs[0]] = x;
        // A graph output's last Q/DQ is not kept: the output is float anyway,
        // and the layer's float sums are finer than 256 levels.  But a zero point
        // of 0 is also a ReLU the quantizer folded into it (uint8 holds nothing
        // below 0), so that one stays, as a ReLU of its own: the output of a model
        // cut inside (the detector before its head, carbon/src/det_head.h) is a
        // ReLU's.
        const auto is_output = [&](const onnx::ValueInfo& o) { return o.name == n.outputs[0]; };
        if (std::any_of(g.outputs.begin(), g.outputs.end(), is_output)) {
          if (q.zero == 0) {
            onnx::Node relu;
            relu.op = "Relu";
            relu.name = n.outputs[0] + "/relu";
            relu.inputs = {x};
            relu.outputs = {x + "/relu"};
            out.push_back(std::move(relu));
            output_names_[n.outputs[0]] = x + "/relu";
          } else {
            output_names_[n.outputs[0]] = x;
          }
          continue;
        }
        auto it = quant_.find(x);
        if (it != quant_.end() && !(it->second == q)) refuse("'" + x + "' is quantized twice, differently");
        quant_[x] = q;
      } else if (n.op == "DequantizeLinear") {
        dequantize_constant(n);
      } else {
        out.push_back(std::move(n));
      }
    }
    for (const auto& [name, p] : pending) {
      bool read = false;
      for (const onnx::Node& n : out) {
        read = read || std::find(n.inputs.begin(), n.inputs.end(), name) != n.inputs.end();
      }
      if (read) refuse("QuantizeLinear '" + name + "' is read without a DequantizeLinear");
    }
    return out;
  }

  // A DequantizeLinear of a constant: int8 weights (kept as int8 too), int32
  // biases, or anything else, as the float constant it gives.
  void dequantize_constant(const onnx::Node& n) {
    const onnx::TensorData& t = constant(n.inputs[0]);
    const std::vector<float> s = constant(n.inputs[1]).floats();
    std::vector<int64_t> z(s.size(), 0);
    if (n.inputs.size() > 2 && !n.inputs[2].empty()) z = constant(n.inputs[2]).ints();
    const int64_t axis = n.i("axis", 1);
    const std::vector<int64_t> v = t.ints();
    int64_t inner = 1;  // the elements under one index of the axis
    if (s.size() > 1) {
      if (axis < 0 || axis >= static_cast<int64_t>(t.dims.size()) || t.dims[axis] != static_cast<int64_t>(s.size())) {
        refuse("DequantizeLinear '" + n.name + "': the scales do not match the axis");
      }
      for (size_t d = axis + 1; d < t.dims.size(); ++d) inner *= t.dims[d];
    }
    onnx::TensorData f;
    f.name = n.outputs[0];
    f.dtype = onnx::kFloat;
    f.dims = t.dims;
    f.bytes.resize(v.size() * 4);
    for (size_t i = 0; i < v.size(); ++i) {
      const size_t k = s.size() > 1 ? (i / inner) % s.size() : 0;
      const float r = static_cast<float>(v[i] - z[k]) * s[k];
      std::memcpy(f.bytes.data() + 4 * i, &r, 4);
    }
    owned_.push_back(std::move(f));
    constants_[n.outputs[0]] = &owned_.back();
    if (t.dtype == onnx::kInt8 && axis == 0) {
      if (std::any_of(z.begin(), z.end(), [](int64_t v0) { return v0 != 0; })) {
        refuse("DequantizeLinear '" + n.name + "': int8 weights must be symmetric (zero point 0)");
      }
      qweights_[n.outputs[0]] = {t.int8s(), s};
    }
  }

  // ---------------------------------------------------------------- compiling

  int value(const std::string& name) const {
    auto it = names_.find(name);
    if (it == names_.end()) refuse("'" + name + "' is not a tensor the network computes");
    return it->second;
  }

  const onnx::TensorData& constant(const std::string& name) const {
    auto it = constants_.find(name);
    if (it == constants_.end()) refuse("'" + name + "' is not a constant");
    return *it->second;
  }

  int add_value(const std::string& name, int c) {
    Value v;
    v.name = v.last = name;
    v.c = c;
    values_.push_back(std::move(v));
    names_[name] = static_cast<int>(values_.size()) - 1;
    return static_cast<int>(values_.size()) - 1;
  }

  // The layer that makes value v, or -1.
  int producer(int v) const {
    for (size_t i = 0; i < layers_.size(); ++i) {
      if (layers_[i].out == v) return static_cast<int>(i);
    }
    return -1;
  }

  void compile(const onnx::Graph& g) {
    if (g.inputs.size() != 1) refuse("networks with one input only");
    const onnx::ValueInfo& x = g.inputs[0];
    if ((x.dtype != onnx::kFloat && x.dtype != onnx::kFloat16) || x.dims.size() != 4 || x.dims[1] <= 0) {
      refuse("the input must be float or float16 [N, C, H, W] with C fixed");
    }
    fp16_ = x.dtype == onnx::kFloat16;
    // float32 at the session whatever the file's: an FP16 file's are converted here
    inputs.push_back({x.name, DType::kFloat32, Shape(x.dims.begin(), x.dims.end())});
    add_value(x.name, static_cast<int>(x.dims[1]));
    for (const onnx::TensorData& t : g.initializers) constants_[t.name] = &t;
    const std::vector<onnx::Node> nodes = read_qdq(g);
    // How many nodes (and graph outputs) read each name: a fused output must
    // have no other reader.
    for (const onnx::Node& n : nodes) {
      for (const std::string& i : n.inputs) ++uses_[i];
    }
    for (const onnx::ValueInfo& o : g.outputs) ++uses_[o.name];
    for (const onnx::Node& n : nodes) compile(n);
    for (const onnx::ValueInfo& o : g.outputs) {
      if (o.dtype != (fp16_ ? onnx::kFloat16 : onnx::kFloat)) {
        refuse("output '" + o.name + "' is not " + (fp16_ ? "float16, as the input is" : "float"));
      }
      auto it = output_names_.find(o.name);
      outputs_.push_back({value(it == output_names_.end() ? o.name : it->second), static_cast<int>(o.dims.size())});
      outputs.push_back({o.name, DType::kFloat32, Shape(o.dims.begin(), o.dims.end())});
    }
    // Each tensor kept as its last name's quantization says, or in FP16 (an FP16
    // file's, where its tensors are: the outputs stay FP32); each border as wide
    // as the widest padding of the convolutions that read it.
    for (Value& v : values_) {
      auto it = quant_.find(v.last);
      if (it != quant_.end()) {
        v.u8 = true;
        v.q = it->second;
      }
      v.f16 = fp16_ && act16_;
    }
    for (const Output& o : outputs_) values_[o.value].f16 = false;
    for (const Layer& l : layers_) {
      if (l.kind != Kind::kConv && l.kind != Kind::kDwConv) continue;
      Value& v = values_[l.in];
      v.ph = std::max({v.ph, l.pt, l.pb});
      v.pw = std::max({v.pw, l.pl, l.pr});
    }
    for (Layer& l : layers_) pack(l);
    constants_.clear();  // the layers keep their own copies; the model goes after compiling
    owned_.clear();
    qweights_.clear();
  }

  // A fused op's output: the value it joins, which now ends as this name.
  void join(const std::string& name, int v) {
    names_[name] = v;
    values_[v].last = name;
  }

  void compile(const onnx::Node& n) {
    const std::string& op = n.op;
    if (!n.domain.empty() && n.domain != "ai.onnx") refuse("op " + n.domain + "." + op);
    if (op == "Constant") {
      const onnx::Attribute* a = n.attribute("value");
      if (!a || !a->t) refuse("Constant '" + n.name + "' without a tensor value");
      owned_.push_back(*a->t);
      constants_[n.outputs[0]] = &owned_.back();
    } else if (op == "Identity" || op == "Squeeze") {
      if (constants_.count(n.inputs[0])) {
        constants_[n.outputs[0]] = constants_[n.inputs[0]];
        if (qweights_.count(n.inputs[0])) qweights_[n.outputs[0]] = qweights_[n.inputs[0]];
      } else {
        names_[n.outputs[0]] = value(n.inputs[0]);
      }
    } else if (op == "Conv" || op == "Gemm") {
      conv(n);
    } else if (op == "Relu") {
      const int v = value(n.inputs[0]);
      const int p = producer(v);
      const bool joins = p >= 0 && (layers_[p].kind == Kind::kConv || layers_[p].kind == Kind::kDwConv ||
                                    layers_[p].kind == Kind::kAdd);
      if (!joins || layers_[p].relu || uses_[n.inputs[0]] != 1) {
        refuse("Relu '" + n.name + "' does not follow a layer it can join");
      }
      layers_[p].relu = true;
      join(n.outputs[0], v);
    } else if (op == "Add") {
      add(n);
    } else if (op == "Resize") {
      resize(n);
    } else if (op == "AveragePool") {
      Layer l;
      l.kind = Kind::kAvgPool;
      l.node = n.name;
      l.in = value(n.inputs[0]);
      const auto k = n.ints("kernel_shape", {});
      const auto s = n.ints("strides", k);
      const auto pads = n.ints("pads", {0, 0, 0, 0});
      if (k.size() != 2 || s.size() != 2 || std::any_of(pads.begin(), pads.end(), [](int64_t p) { return p != 0; }) ||
          n.i("ceil_mode", 0) != 0) {
        refuse("AveragePool '" + n.name + "': 2-d, no padding, floor");
      }
      l.kh = static_cast<int>(k[0]);
      l.kw = static_cast<int>(k[1]);
      l.sh = static_cast<int>(s[0]);
      l.sw = static_cast<int>(s[1]);
      l.cin = l.cout = values_[l.in].c;
      l.out = add_value(n.outputs[0], l.cout);
      layers_.push_back(std::move(l));
    } else if (op == "ReduceMean") {
      std::vector<int64_t> axes = n.ints("axes", {});
      if (n.inputs.size() > 1 && !n.inputs[1].empty()) axes = constant(n.inputs[1]).ints();
      if (axes != std::vector<int64_t>{2, 3} || n.i("keepdims", 1) != 0) {
        refuse("ReduceMean '" + n.name + "': over H and W, keepdims 0");
      }
      Layer l;
      l.kind = Kind::kMean;
      l.node = n.name;
      l.in = value(n.inputs[0]);
      l.cin = l.cout = values_[l.in].c;
      l.out = add_value(n.outputs[0], l.cout);
      layers_.push_back(std::move(l));
    } else {
      refuse("op " + op + " ('" + n.name + "')");
    }
  }

  void conv(const onnx::Node& n) {
    Layer l;
    l.node = n.name;
    l.in = value(n.inputs[0]);
    const onnx::TensorData& w = constant(n.inputs[1]);
    l.wf = w.floats();
    if (auto it = qweights_.find(n.inputs[1]); it != qweights_.end()) l.w8 = it->second;
    int64_t groups = 1;
    if (n.op == "Gemm") {  // [N, in] x W^T: a 1x1 convolution on a 1x1 image
      if (n.f("alpha", 1) != 1 || n.f("beta", 1) != 1 || n.i("transA", 0) != 0 || w.dims.size() != 2) {
        refuse("Gemm '" + n.name + "': alpha 1, beta 1, A not transposed");
      }
      if (n.i("transB", 0) == 0) {  // W is [in, out]: make it [out, in]
        const int64_t rows = w.dims[0], cols = w.dims[1];
        auto transpose = [&](auto& v) {
          auto t = v;
          for (int64_t i = 0; i < rows; ++i) {
            for (int64_t j = 0; j < cols; ++j) t[j * rows + i] = v[i * cols + j];
          }
          v.swap(t);
        };
        transpose(l.wf);
        if (l.w8) {
          if (l.w8->scales.size() > 1) refuse("Gemm '" + n.name + "': per-channel scales on W not transposed");
          transpose(l.w8->values);
        }
        l.cout = static_cast<int>(cols);
        l.cin = static_cast<int>(rows);
      } else {
        l.cout = static_cast<int>(w.dims[0]);
        l.cin = static_cast<int>(w.dims[1]);
      }
    } else {
      if (w.dims.size() != 4) refuse("Conv '" + n.name + "': 2-d only");
      if (n.s("auto_pad", "NOTSET") != "NOTSET") refuse("Conv '" + n.name + "': auto_pad");
      const auto d = n.ints("dilations", {1, 1});
      if (d != std::vector<int64_t>{1, 1}) refuse("Conv '" + n.name + "': dilation");
      groups = n.i("group", 1);
      l.cout = static_cast<int>(w.dims[0]);
      l.cin = static_cast<int>(w.dims[1] * groups);
      l.kh = static_cast<int>(w.dims[2]);
      l.kw = static_cast<int>(w.dims[3]);
      const auto s = n.ints("strides", {1, 1});
      const auto p = n.ints("pads", {0, 0, 0, 0});
      l.sh = static_cast<int>(s[0]);
      l.sw = static_cast<int>(s[1]);
      l.pt = static_cast<int>(p[0]);
      l.pl = static_cast<int>(p[1]);
      l.pb = static_cast<int>(p[2]);
      l.pr = static_cast<int>(p[3]);
    }
    if (l.cin != values_[l.in].c) refuse("'" + n.name + "' takes " + std::to_string(l.cin) + " channels");
    l.bias.assign(l.cout, 0.0f);
    if (n.inputs.size() > 2 && !n.inputs[2].empty()) {
      l.bias = constant(n.inputs[2]).floats();
      if (static_cast<int>(l.bias.size()) != l.cout) refuse("'" + n.name + "': the bias has the wrong size");
    }
    if (groups == 1) {
      l.kind = Kind::kConv;
    } else if (groups == l.cin && l.cout == l.cin) {
      l.kind = Kind::kDwConv;
    } else {
      refuse("Conv '" + n.name + "': groups other than 1 and depthwise");
    }
    l.out = add_value(n.outputs[0], l.cout);
    layers_.push_back(std::move(l));
  }

  // A convolution's weights packed for its kernel, once every tensor's storage
  // is known: int8 where its input is uint8 and its weights came as int8; FP16 in
  // an FP16 network (its weights FP16 values, widened exactly when read, so
  // rounding them back is exact too).
  void pack(Layer& l) {
    if (l.kind != Kind::kConv && l.kind != Kind::kDwConv) return;
    const Value& x = values_[l.in];
    const int taps = l.kh * l.kw;
    if (x.u8 && !l.w8) refuse("'" + l.node + "': a uint8 input needs int8 weights");
    l.int8 = x.u8;
    l.half = fp16_;
    auto halves = [](const std::vector<float>& v) {
      std::vector<uint16_t> h(v.size());
      for (size_t i = 0; i < v.size(); ++i) h[i] = cpu::f32_to_f16(v[i]);
      return h;
    };
    if (l.int8) {
      l.scale.resize(l.cout);
      for (int co = 0; co < l.cout; ++co) {
        l.scale[co] = x.q.scale * l.w8->scales[l.w8->scales.size() > 1 ? co : 0];
      }
    }
    if (l.kind == Kind::kConv) {
      // [cout][cin][kh][kw] to B[(tap, cin)][cout]
      auto to_b = [&](const auto& w, auto& b) {
        b.resize(static_cast<size_t>(taps) * l.cin * l.cout);
        for (int co = 0; co < l.cout; ++co) {
          for (int ci = 0; ci < l.cin; ++ci) {
            for (int t = 0; t < taps; ++t) {
              const size_t from = (static_cast<size_t>(co) * l.cin + ci) * taps + t;
              b[(static_cast<size_t>(t) * l.cin + ci) * l.cout + co] = w[from];
            }
          }
        }
      };
      if (l.int8) {
        std::vector<int8_t> b;
        to_b(l.w8->values, b);
        // A piece of A must be a whole number of the kernel's groups of k: when
        // a tap's channels are not (the stem's one channel), the taps are copied
        // into one row per pixel first.
        l.im2col = taps > 1 && cpu::k_padded(l.cin, isa_) != l.cin;
        l.weights8 = cpu::pack_b(b.data(), taps * l.cin, l.cout, isa_, l.im2col ? 1 : taps);
      } else {
        std::vector<float> b;
        to_b(l.wf, b);
        if (l.half) {
          l.weights16 = cpu::pack_f16(halves(b).data(), taps * l.cin, l.cout, isa_);
        } else {
          l.weights = cpu::pack_f32(b.data(), taps * l.cin, l.cout, isa_);
        }
      }
    } else {
      auto to_dw = [&](const auto& w, auto& d) {
        d.resize(static_cast<size_t>(taps) * l.cin);
        for (int c = 0; c < l.cin; ++c) {
          for (int t = 0; t < taps; ++t) d[static_cast<size_t>(t) * l.cin + c] = w[static_cast<size_t>(c) * taps + t];
        }
      };
      if (l.int8) {
        to_dw(l.w8->values, l.dw8);
      } else {
        to_dw(l.wf, l.dw);
        if (l.half) {
          l.dw16 = halves(l.dw);
          l.dw.clear();
        }
      }
    }
    l.wf.clear();
    l.wf.shrink_to_fit();
    l.w8.reset();
  }

  // A residual Add joins the convolution that makes one of its inputs, when
  // nothing else reads that output and the convolution has no ReLU yet (the
  // output stage adds the residual before the ReLU).  The convolution then runs
  // where the Add is, after the residual is made.  Otherwise the Add is a layer.
  void add(const onnx::Node& n) {
    const int a = value(n.inputs[0]), b = value(n.inputs[1]);
    for (const auto& [mine, theirs] : {std::pair{0, 1}, std::pair{1, 0}}) {
      const int v = mine == 0 ? a : b, r = theirs == 0 ? a : b;
      const int p = producer(v);
      if (p < 0 || uses_[n.inputs[mine]] != 1 || v == r) continue;
      Layer& l = layers_[p];
      if ((l.kind != Kind::kConv && l.kind != Kind::kDwConv) || l.relu || l.residual >= 0) continue;
      if (values_[r].c != l.cout) refuse("Add '" + n.name + "': channels differ");
      Layer moved = std::move(l);
      layers_.erase(layers_.begin() + p);
      moved.residual = r;
      join(n.outputs[0], moved.out);
      layers_.push_back(std::move(moved));
      return;
    }
    if (values_[a].c != values_[b].c) refuse("Add '" + n.name + "': channels differ (no broadcasting)");
    Layer l;
    l.kind = Kind::kAdd;
    l.node = n.name;
    l.in = a;
    l.other = b;
    l.cin = l.cout = values_[a].c;
    l.out = add_value(n.outputs[0], l.cout);
    layers_.push_back(std::move(l));
  }

  void resize(const onnx::Node& n) {
    if (n.s("mode", "nearest") != "nearest" || n.s("coordinate_transformation_mode", "half_pixel") != "asymmetric" ||
        n.s("nearest_mode", "round_prefer_floor") != "floor") {
      refuse("Resize '" + n.name + "': nearest, asymmetric, floor");
    }
    if (n.inputs.size() < 3 || n.inputs[2].empty()) refuse("Resize '" + n.name + "': scales, not sizes");
    const std::vector<float> s = constant(n.inputs[2]).floats();
    auto whole = [](float v) { return v >= 1 && v == static_cast<float>(static_cast<int>(v)); };
    if (s.size() != 4 || s[0] != 1 || s[1] != 1 || !whole(s[2]) || !whole(s[3])) {
      refuse("Resize '" + n.name + "': whole scales over H and W");
    }
    Layer l;
    l.kind = Kind::kResize;
    l.node = n.name;
    l.in = value(n.inputs[0]);
    l.fh = static_cast<int>(s[2]);
    l.fw = static_cast<int>(s[3]);
    l.cin = l.cout = values_[l.in].c;
    l.out = add_value(n.outputs[0], l.cout);
    layers_.push_back(std::move(l));
  }

  // ---------------------------------------------------------------- running

  void step(const Layer& l) {
    const Value& x = values_[l.in];
    Value& y = values_[l.out];
    switch (l.kind) {
      case Kind::kConv:
      case Kind::kDwConv: conv_step(l, x, y); return;
      case Kind::kResize: {
        y.shape(x.n, x.h * l.fh, x.w * l.fw);
        const bool same = x.same_storage(y);
        const int c = x.c;
        // Each input row widened once into its first output row; the rows it
        // repeats are copies of that one.
        auto widen = [&](const auto* src, auto* dst) {
          for (int ox = 0; ox < y.w; ++ox) {
            const auto* from = src + static_cast<int64_t>(ox / l.fw) * c;
            auto* to = dst + static_cast<int64_t>(ox) * c;
            for (int ch = 0; ch < c; ++ch) to[ch] = from[ch];
          }
        };
        for (int i = 0; i < y.n; ++i) {
          for (int oy = 0; oy < y.h; ++oy) {
            const int iy = oy / l.fh;
            if (oy % l.fh && same) {
              std::memcpy(y.raw(i, oy, 0), y.raw(i, oy - 1, 0), static_cast<size_t>(y.w) * c * y.element());
            } else if (same && x.u8) {
              widen(x.bat(i, iy, 0), y.bat(i, oy, 0));
            } else if (same && x.f16) {
              widen(x.hat(i, iy, 0), y.hat(i, oy, 0));
            } else if (same) {
              widen(x.fat(i, iy, 0), y.fat(i, oy, 0));
            } else {
              for (int ox = 0; ox < y.w; ++ox) {
                for (int ch = 0; ch < c; ++ch) y.set(i, oy, ox, ch, x.real(i, oy / l.fh, ox / l.fw, ch));
              }
            }
          }
        }
        return;
      }
      case Kind::kAvgPool: {
        const int oh = (x.h - l.kh) / l.sh + 1, ow = (x.w - l.kw) / l.sw + 1;
        if (oh <= 0 || ow <= 0) refuse("'" + l.node + "': the input is smaller than the pool");
        y.shape(x.n, oh, ow);
        const int count = l.kh * l.kw, c = x.c;
        sums_.resize(static_cast<size_t>(ow) * c);
        for (int i = 0; i < y.n; ++i) {
          for (int oy = 0; oy < oh; ++oy) {
            // A row of windows at a time: the window's rows summed into one row
            // of sums (uint8 as integers, exactly), then each sum to its mean.
            std::fill(sums_.begin(), sums_.end(), 0.0);
            // one channel: the row's values in order
            auto add_row = [&](const auto* v) {
              for (int ox = 0; ox < ow; ++ox) {
                for (int kx = 0; kx < l.kw; ++kx) sums_[ox] += stored(v[ox * l.sw + kx]);
              }
            };
            auto add_pixel = [&](const auto* v, double* sum) {
              for (int ch = 0; ch < c; ++ch) sum[ch] += stored(v[ch]);
            };
            for (int ky = 0; ky < l.kh; ++ky) {
              const int iy = oy * l.sh + ky;
              if (c == 1) {
                if (x.u8) {
                  add_row(x.bat(i, iy, 0));
                } else if (x.f16) {
                  add_row(x.hat(i, iy, 0));
                } else {
                  add_row(x.fat(i, iy, 0));
                }
                continue;
              }
              for (int ox = 0; ox < ow; ++ox) {
                double* sum = sums_.data() + static_cast<int64_t>(ox) * c;
                for (int kx = 0; kx < l.kw; ++kx) {
                  const int ix = ox * l.sw + kx;
                  if (x.u8) {
                    add_pixel(x.bat(i, iy, ix), sum);
                  } else if (x.f16) {
                    add_pixel(x.hat(i, iy, ix), sum);
                  } else {
                    add_pixel(x.fat(i, iy, ix), sum);
                  }
                }
              }
            }
            for (int ox = 0; ox < ow; ++ox) {
              for (int ch = 0; ch < c; ++ch) {
                const double sum = sums_[static_cast<size_t>(ox) * c + ch];
                const float mean = x.u8 ? static_cast<float>((sum - count * x.q.zero) * x.q.scale / count)
                                        : static_cast<float>(sum / count);
                y.set(i, oy, ox, ch, mean);
              }
            }
          }
        }
        return;
      }
      case Kind::kMean: {
        y.shape(x.n, 1, 1);
        const double count = static_cast<double>(x.h) * x.w;
        const int c = x.c;
        sums_.resize(c);
        for (int i = 0; i < y.n; ++i) {
          std::fill(sums_.begin(), sums_.end(), 0.0);
          auto add_pixel = [&](const auto* v) {
            for (int ch = 0; ch < c; ++ch) sums_[ch] += stored(v[ch]);
          };
          for (int py = 0; py < x.h; ++py) {
            for (int px = 0; px < x.w; ++px) {
              if (x.u8) {
                add_pixel(x.bat(i, py, px));
              } else if (x.f16) {
                add_pixel(x.hat(i, py, px));
              } else {
                add_pixel(x.fat(i, py, px));
              }
            }
          }
          for (int ch = 0; ch < c; ++ch) {
            const double sum = sums_[ch];
            y.set(i, 0, 0, ch, static_cast<float>(x.u8 ? (sum - count * x.q.zero) * x.q.scale / count : sum / count));
          }
        }
        return;
      }
      case Kind::kAdd: {
        const Value& z = values_[l.other];
        if (z.n != x.n || z.h != x.h || z.w != x.w) refuse("Add '" + l.node + "': the shapes differ");
        y.shape(x.n, x.h, x.w);
        for (int i = 0; i < y.n; ++i) {
          for (int py = 0; py < y.h; ++py) {
            for (int px = 0; px < y.w; ++px) {
              for (int c = 0; c < y.c; ++c) {
                const float v = x.real(i, py, px, c) + z.real(i, py, px, c);
                y.set(i, py, px, c, l.relu ? (v > 0 ? v : 0.0f) : v);
              }
            }
          }
        }
        return;
      }
    }
  }

  void conv_step(const Layer& l, const Value& x, Value& y) {
    const int oh = (x.h + l.pt + l.pb - l.kh) / l.sh + 1, ow = (x.w + l.pl + l.pr - l.kw) / l.sw + 1;
    if (oh <= 0 || ow <= 0) refuse("'" + l.node + "': the input is smaller than the kernel");
    y.shape(x.n, oh, ow);
    const int taps = l.kh * l.kw;
    offsets_.resize(taps);
    for (int ky = 0; ky < l.kh; ++ky) {
      for (int kx = 0; kx < l.kw; ++kx) offsets_[ky * l.kw + kx] = (ky * x.wp() + kx) * x.c;
    }
    const Value* r = l.residual >= 0 ? &values_[l.residual] : nullptr;
    if (r && (r->n != y.n || r->h != y.h || r->w != y.w)) refuse("'" + l.node + "': the residual's shape differs");
    cpu::Epilogue e;
    e.scale = l.int8 ? l.scale.data() : nullptr;
    e.bias = l.bias.data();
    e.relu = l.relu;
    e.out_ld = y.c;
    if (y.u8) {
      e.out_inv_scale = 1.0f / y.q.scale;
      e.out_zero = y.q.zero;
    }
    if (r) {
      e.residual_ld = r->c;
      e.residual_scale = r->q.scale;
      e.residual_zero = r->q.zero;
    }
    const int64_t kp = l.im2col ? cpu::k_padded(taps * l.cin, isa_) : 0;
    // (the bytes of a row past its taps meet B's zero padding: any value)
    if (l.im2col && cols_.size() < static_cast<size_t>(ow) * kp + 64) cols_.resize(static_cast<size_t>(ow) * kp + 64);
    for (int i = 0; i < y.n; ++i) {
      for (int oy = 0; oy < oh; ++oy) {
        if (y.u8) {
          e.out_u8 = y.bat(i, oy, 0);
        } else if (y.f16) {
          e.out_f16 = y.hat(i, oy, 0);
        } else {
          e.out_f32 = y.fat(i, oy, 0);
        }
        if (r && r->u8) e.residual_u8 = r->bat(i, oy, 0);
        if (r && r->f16) e.residual_f16 = r->hat(i, oy, 0);
        if (r && !r->u8 && !r->f16) e.residual_f32 = r->fat(i, oy, 0);
        const int y0 = oy * l.sh - l.pt, x0 = -l.pl;
        const int64_t step = int64_t{l.sw} * x.c;
        if (l.half && x.f16) {  // FP16 activations: the FP16 kernels' own arithmetic
          const cpu::Rows<uint16_t> rows = cpu::grid(x.hat(i, y0, x0), step, taps, offsets_.data());
          if (l.kind == Kind::kConv) {
            cpu::gemm_f16(rows, l.weights16, ow, e);
          } else {
            cpu::dwconv_f16(rows, l.dw16.data(), l.cin, ow, e, isa_);
          }
        } else if (l.half) {  // FP32 activations, FP16 weights
          const cpu::Rows<float> rows = cpu::grid(x.fat(i, y0, x0), step, taps, offsets_.data());
          if (l.kind == Kind::kConv) {
            cpu::gemm_f16(rows, l.weights16, ow, e);
          } else {
            cpu::dwconv_f16(rows, l.dw16.data(), l.cin, ow, e, isa_);
          }
        } else if (!l.int8) {
          const cpu::Rows<float> rows = cpu::grid(x.fat(i, y0, x0), int64_t{l.sw} * x.c, taps, offsets_.data());
          if (l.kind == Kind::kConv) {
            cpu::gemm_f32(rows, l.weights, ow, e);
          } else {
            cpu::dwconv_f32(rows, l.dw.data(), l.cin, ow, e, isa_);
          }
        } else if (l.kind == Kind::kDwConv) {
          const cpu::Rows<uint8_t> rows = cpu::grid(x.bat(i, y0, x0), int64_t{l.sw} * x.c, taps, offsets_.data());
          cpu::dwconv_u8s8(rows, x.q.zero, l.dw8.data(), l.cin, ow, e, isa_);
        } else if (!l.im2col) {
          const cpu::Rows<uint8_t> rows = cpu::grid(x.bat(i, y0, x0), int64_t{l.sw} * x.c, taps, offsets_.data());
          cpu::gemm_u8s8(rows, x.q.zero, l.weights8, ow, e);
        } else {
          for (int ox = 0; ox < ow; ++ox) {
            uint8_t* col = cols_.data() + ox * kp;
            const uint8_t* base = x.bat(i, y0, x0 + ox * l.sw);
            if (l.cin == 1) {
              for (int t = 0; t < taps; ++t) col[t] = base[offsets_[t]];
            } else {
              for (int t = 0; t < taps; ++t) std::memcpy(col + t * l.cin, base + offsets_[t], l.cin);
            }
          }
          cpu::gemm_u8s8(cpu::rows_of(cols_.data(), kp), x.q.zero, l.weights8, ow, e);
        }
      }
    }
  }

  // An output as ONNX has it: float NCHW, then without the dims of size 1 a
  // lower rank drops (a squeezed height, a pooled image).
  static Tensor to_nchw(const Value& v, int rank) {
    Shape s;
    if (rank == 4) {
      s = {v.n, v.c, v.h, v.w};
    } else if (rank == 3 && (v.h == 1 || v.w == 1)) {
      s = {v.n, v.c, v.h == 1 ? v.w : v.h};
    } else if (rank == 2 && v.h == 1 && v.w == 1) {
      s = {v.n, v.c};
    } else {
      refuse("output '" + v.name + "' of rank " + std::to_string(rank) + " from " + std::to_string(v.h) + " x " +
             std::to_string(v.w));
    }
    Tensor t = Tensor::empty(DType::kFloat32, s);
    float* o = t.data<float>().data();
    if (v.f16) {  // not a graph output's storage (those are FP32), but any value's
      for (int i = 0; i < v.n; ++i) {
        for (int y = 0; y < v.h; ++y) {
          for (int x = 0; x < v.w; ++x) {
            for (int ch = 0; ch < v.c; ++ch) {
              o[((static_cast<int64_t>(i) * v.c + ch) * v.h + y) * v.w + x] = v.real(i, y, x, ch);
            }
          }
        }
      }
      return t;
    }
    float table[256];
    for (int q = 0; q < 256; ++q) table[q] = static_cast<float>(q - v.q.zero) * v.q.scale;
    // Pixels x channels to channels x pixels, in blocks of 16 x 16 (both sides
    // in cache: one by one, each channel would read every pixel's line).
    constexpr int kB = 16;
    const int64_t plane = static_cast<int64_t>(v.h) * v.w, c = v.c;
    for (int i = 0; i < v.n; ++i) {
      for (int y = 0; y < v.h; ++y) {
        float* plane0 = o + static_cast<int64_t>(i) * c * plane + static_cast<int64_t>(y) * v.w;
        const float* rf = v.u8 ? nullptr : v.fat(i, y, 0);
        const uint8_t* rb = v.u8 ? v.bat(i, y, 0) : nullptr;
        for (int x0 = 0; x0 < v.w; x0 += kB) {
          const int xe = std::min(v.w, x0 + kB);
          for (int64_t c0 = 0; c0 < c; c0 += kB) {
            const int64_t ce = std::min(c, c0 + kB);
            for (int64_t ch = c0; ch < ce; ++ch) {
              float* dst = plane0 + ch * plane;
              if (rf) {
                for (int x = x0; x < xe; ++x) dst[x] = rf[x * c + ch];
              } else {
                for (int x = x0; x < xe; ++x) dst[x] = table[rb[x * c + ch]];
              }
            }
          }
        }
      }
    }
    return t;
  }

  cpu::Isa isa_;
  bool act16_ = false;  // an FP16 file's tensors in FP16
  bool fp16_ = false;   // the file is FP16 (its input float16)
  std::vector<Value> values_;
  std::vector<Layer> layers_;
  std::vector<Output> outputs_;
  std::map<std::string, int> names_;
  std::map<std::string, const onnx::TensorData*> constants_;
  std::deque<onnx::TensorData> owned_;  // Constant nodes' and dequantized tensors (constants_ points into it)
  std::map<std::string, QWeight> qweights_;
  std::map<std::string, Quant> quant_;  // a tensor's name -> its uint8 quantization
  std::map<std::string, std::string> output_names_;  // a graph output -> the tensor before its Q/DQ
  std::map<std::string, int> uses_;
  std::vector<int64_t> offsets_;
  std::vector<uint8_t> cols_;  // im2col rows
  std::vector<double> sums_;   // a pool's sums
  std::vector<double> profile_;
  int runs_ = 0;
};

cpu::Isa isa(const std::string& name) {
  if (name.empty()) return cpu::best_isa();
  for (cpu::Isa i : cpu::isas_available()) {
    if (name == cpu::isa_name(i)) return i;
  }
  refuse("instruction set '" + name + "' is not available on this CPU");
}

// An FP16 file's tensors between layers: FP16 where the instruction set sums in
// FP16, else FP32 (see the top of this file); CARBON_FP16_ACTIVATIONS=1 or 0 chooses.
bool fp16_activations(cpu::Isa isa) {
  const std::string v = env("CARBON_FP16_ACTIVATIONS");
  if (v == "1") return true;
  if (v == "0") return false;
  return cpu::f16_arithmetic(isa);
}

// The file compiled for isa (or `bytes`, read in its place); what it refuses names the file.
Net open_net(const std::filesystem::path& path, cpu::Isa isa, const std::string* bytes = nullptr) {
  onnx::Model model;
  try {
    model = bytes ? onnx::parse(*bytes) : onnx::load(path);
  } catch (const Error& e) {
    if (!bytes) throw;
    throw Error(path.string() + ": " + e.what());
  }
  try {
    return Net(model, isa, fp16_activations(isa));
  } catch (const Error& e) {
    throw Error(path.string() + ": " + e.what());
  }
}

class NativeSession final : public Session {
 public:
  NativeSession(const std::filesystem::path& path, const SessionOptions& options)
      : net_(open_net(path, isa(options.isa), options.model_bytes.get())) {
    inputs_ = net_.inputs;
    outputs_ = net_.outputs;
    net_.profile();
  }

  Backend backend() const override { return Backend::kNative; }
  using Session::run;  // and run() with its outputs in place: the base's, over this one

  std::vector<Tensor> run(std::span<const Tensor> inputs) override {
    check_inputs(inputs);
    return net_.run(inputs[0]);
  }

 private:
  Net net_;
};

}  // namespace

std::unique_ptr<Session> open_native_session(const std::filesystem::path& model, const SessionOptions& options) {
  if (is_order_net(model)) return open_order_net(model, isa(options.isa));
  return std::make_unique<NativeSession>(model, options);
}

}  // namespace carbon::net
