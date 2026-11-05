#include "det_head.h"

#include <algorithm>
#include <fstream>
#include <iterator>

#include "carbon/error.h"
#include "net/onnx.h"

namespace carbon::detail {
namespace {

using onnx::Graph;
using onnx::Node;
using onnx::TensorData;

struct Weights {
  std::vector<float> v;
  std::vector<int64_t> dims;
};

// The graph read for the head: who makes a tensor, its constants, its weights as floats.
struct View {
  const Graph& g;

  const Node* producer(const std::string& t) const {
    for (const Node& n : g.nodes) {
      for (const std::string& o : n.outputs) {
        if (o == t) return &n;
      }
    }
    return nullptr;
  }
  const TensorData* init(const std::string& t) const {
    for (const TensorData& i : g.initializers) {
      if (i.name == t) return &i;
    }
    return nullptr;
  }
  // An initializer, or a Constant node's value.
  std::optional<TensorData> constant(const std::string& t) const {
    if (const TensorData* i = init(t)) return *i;
    const Node* n = producer(t);
    if (n && n->op == "Constant") {
      const onnx::Attribute* a = n->attribute("value");
      if (a && a->t) return *a->t;
    }
    return std::nullopt;
  }
  bool zero_point_zero(const Node& q) const {
    if (q.inputs.size() < 3 || q.inputs[2].empty()) return true;  // ONNX's default zero point is 0
    const auto z = constant(q.inputs[2]);
    if (!z) return false;
    for (int64_t v : z->ints()) {
      if (v != 0) return false;
    }
    return true;
  }
  // The tensor `t` is, past QuantizeLinear and DequantizeLinear pairs (int8 Q/DQ files): what was quantized, and
  // whether the nearest quantization's zero point is 0 (a ReLU the quantizer folded into it, for uint8).
  struct Through {
    std::string tensor;
    bool quantized = false, zero = false;
  };
  Through through(std::string t) const {
    Through r{t};
    for (;;) {
      const Node* dq = producer(t);
      if (!dq || dq->op != "DequantizeLinear") break;
      const Node* q = producer(dq->inputs[0]);
      if (!q || q->op != "QuantizeLinear") break;
      if (!r.quantized) {
        r.quantized = true;
        r.zero = zero_point_zero(*q);
      }
      t = q->inputs[0];
      r.tensor = t;
    }
    return r;
  }
  // A weight or a bias: a float or float16 initializer (widened exactly), or an int8 or int32 one dequantized as
  // DequantizeLinear does, float(q - zero) * scale, per tensor or along its axis.
  std::optional<Weights> weights(const std::string& t) const {
    if (const TensorData* i = init(t)) {
      if (i->dtype != onnx::kFloat && i->dtype != onnx::kFloat16) return std::nullopt;
      return Weights{i->floats(), i->dims};
    }
    const Node* dq = producer(t);
    if (!dq || dq->op != "DequantizeLinear" || dq->inputs.size() < 2) return std::nullopt;
    const TensorData* q = init(dq->inputs[0]);
    const auto scale = constant(dq->inputs[1]);
    if (!q || !scale) return std::nullopt;
    std::vector<int64_t> zero;
    if (dq->inputs.size() > 2 && !dq->inputs[2].empty()) {
      const auto z = constant(dq->inputs[2]);
      if (!z) return std::nullopt;
      zero = z->ints();
    }
    const std::vector<int64_t> v = q->ints();
    const std::vector<float> s = scale->floats();
    Weights w{std::vector<float>(v.size()), q->dims};
    int64_t axis = dq->i("axis", 1), inner = 1;
    if (axis < 0) axis += static_cast<int64_t>(q->dims.size());
    if (s.size() > 1) {
      if (axis < 0 || axis >= static_cast<int64_t>(q->dims.size()) || q->dims[axis] != static_cast<int64_t>(s.size())) {
        return std::nullopt;
      }
      for (size_t d = static_cast<size_t>(axis) + 1; d < q->dims.size(); ++d) inner *= q->dims[d];
    }
    for (size_t e = 0; e < v.size(); ++e) {
      const size_t c = s.size() > 1 ? static_cast<size_t>((static_cast<int64_t>(e) / inner) % static_cast<int64_t>(s.size())) : 0;
      const int64_t z = zero.empty() ? 0 : zero.size() > 1 ? zero[c] : zero[0];
      w.v[e] = static_cast<float>(v[e] - z) * s[c];
    }
    return w;
  }
  // The tensor a ReLU's output `t` is the ReLU of: a Relu node's input, or, in a Q/DQ file, the quantized tensor
  // whose zero point is 0. Nullopt where `t` is no ReLU's.
  std::optional<std::string> relu_of(const std::string& t) const {
    const Through r = through(t);
    const Node* n = producer(r.tensor);
    if (n && n->op == "Relu") return through(n->inputs[0]).tensor;
    if (r.quantized && r.zero) return r.tensor;
    return std::nullopt;
  }
};

bool all_are(const std::vector<int64_t>& v, int64_t x) {
  for (int64_t e : v) {
    if (e != x) return false;
  }
  return true;
}

// A Conv of `group` groups, a k x k kernel (1 or 3), stride and dilation 1, padding k / 2, with a bias; its
// weights and bias.
bool conv(const View& v, const Node* n, int64_t group, int k, Weights& w, Weights& b) {
  if (!n || n->op != "Conv" || n->inputs.size() < 3 || n->i("group", 1) != group) return false;
  if (!all_are(n->ints("strides", {1, 1}), 1) || !all_are(n->ints("dilations", {1, 1}), 1)) return false;
  if (!all_are(n->ints("pads", {0, 0, 0, 0}), k / 2) || n->s("auto_pad", "NOTSET") != "NOTSET") return false;
  const auto wt = v.weights(n->inputs[1]), bt = v.weights(n->inputs[2]);
  if (!wt || !bt || wt->dims.size() != 4 || wt->dims[2] != k || wt->dims[3] != k) return false;
  if (bt->dims.size() != 1 || bt->dims[0] != wt->dims[0]) return false;
  w = *wt;
  b = *bt;
  return true;
}

// A Conv of a 3x3 kernel, stride 2, padding 1, dilation 1, one group, with a bias (the stem).
bool conv_s2(const View& v, const Node* n, Weights& w, Weights& b) {
  if (!n || n->op != "Conv" || n->inputs.size() < 3 || n->i("group", 1) != 1) return false;
  if (!all_are(n->ints("strides", {1, 1}), 2) || !all_are(n->ints("dilations", {1, 1}), 1)) return false;
  if (!all_are(n->ints("pads", {0, 0, 0, 0}), 1) || n->s("auto_pad", "NOTSET") != "NOTSET") return false;
  const auto wt = v.weights(n->inputs[1]), bt = v.weights(n->inputs[2]);
  if (!wt || !bt || wt->dims.size() != 4 || wt->dims[2] != 3 || wt->dims[3] != 3) return false;
  if (bt->dims.size() != 1 || bt->dims[0] != wt->dims[0]) return false;
  w = *wt;
  b = *bt;
  return true;
}

}  // namespace

bool with_det_stem(FusedHead& head) {
  if (!head.cut || head.stem) return false;
  onnx::Model m;
  try {
    m = onnx::parse(*head.cut);
  } catch (const Error&) {
    return false;
  }
  const Graph& g = m.graph;
  std::vector<const onnx::ValueInfo*> inputs;
  for (const onnx::ValueInfo& i : g.inputs) {
    const bool init = std::any_of(g.initializers.begin(), g.initializers.end(),
                                  [&](const TensorData& t) { return t.name == i.name; });
    if (!init) inputs.push_back(&i);
  }
  if (inputs.size() != 1 || inputs[0]->dtype != onnx::kFloat || inputs[0]->dims.size() != 4 ||
      inputs[0]->dims[1] != 1) {
    return false;  // FP32, one channel
  }
  const View v{g};
  // The input's one reader, a stride-2 convolution, then ReLU, a second, ReLU.
  auto reader = [&](const std::string& t) -> const Node* {
    const Node* only = nullptr;
    for (const Node& n : g.nodes) {
      for (const std::string& i : n.inputs) {
        if (i != t) continue;
        if (only && only != &n) return nullptr;
        only = &n;
      }
    }
    return only;
  };
  Weights w1, b1, w2, b2;
  const Node* stem = reader(inputs[0]->name);
  if (!conv_s2(v, stem, w1, b1) || w1.dims[1] != 1) return false;
  const Node* r1 = reader(stem->outputs[0]);
  if (!r1 || r1->op != "Relu") return false;
  const Node* down = reader(r1->outputs[0]);
  if (!conv_s2(v, down, w2, b2) || w2.dims[1] != w1.dims[0]) return false;
  const Node* r2 = reader(down->outputs[0]);
  if (!r2 || r2->op != "Relu") return false;
  FusedStem s;
  s.output = r2->outputs[0];
  s.c1 = static_cast<int>(w1.dims[0]);
  s.c2 = static_cast<int>(w2.dims[0]);
  try {
    head.cut = std::make_shared<const std::string>(onnx::cut_input(*head.cut, s.output, onnx::kFloat, {-1, s.c2, -1, -1}));
  } catch (const Error&) {
    return false;
  }
  s.w1 = std::move(w1.v);
  s.b1 = std::move(b1.v);
  s.w2 = std::move(w2.v);
  s.b2 = std::move(b2.v);
  head.stem = std::move(s);
  return true;
}

std::optional<RecHead> fused_rec_head(const std::filesystem::path& file) {
  std::ifstream f(file, std::ios::binary);
  if (!f) return std::nullopt;
  const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  onnx::Model m;
  try {
    m = onnx::parse(bytes);
  } catch (const Error&) {
    return std::nullopt;
  }
  const Graph& g = m.graph;
  if (g.outputs.size() != 1 || g.outputs[0].dtype != onnx::kFloat) return std::nullopt;  // FP32 files alone
  const View v{g};
  const Node* sq = v.producer(g.outputs[0].name);
  if (!sq || sq->op != "Squeeze") return std::nullopt;
  std::vector<int64_t> axes = sq->ints("axes", {});
  if (axes.empty() && sq->inputs.size() > 1) {
    const auto a = v.constant(sq->inputs[1]);
    if (a) axes = a->ints();
  }
  if (axes != std::vector<int64_t>{2}) return std::nullopt;
  Weights w, b;
  const Node* head = v.producer(sq->inputs[0]);
  if (!conv(v, head, 1, 1, w, b)) return std::nullopt;
  RecHead r;
  try {
    r.cut = std::make_shared<const std::string>(onnx::cut(bytes, head->inputs[0], onnx::kFloat, {-1, w.dims[1], 1, -1}));
  } catch (const Error&) {
    return std::nullopt;
  }
  r.head = cpu::pack_rec_head(w.v.data(), b.v.data(), static_cast<int>(w.dims[1]), static_cast<int>(w.dims[0]));
  return r;
}

std::optional<FusedHead> fused_det_head(const std::filesystem::path& file) {
  std::ifstream f(file, std::ios::binary);
  if (!f) return std::nullopt;
  const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  onnx::Model m;
  try {
    m = onnx::parse(bytes);
  } catch (const Error&) {
    return std::nullopt;
  }
  const Graph& g = m.graph;
  if (g.outputs.size() != 1) return std::nullopt;
  const View v{g};
  // From the maps back: 1x1 to the maps, ReLU, depthwise 3x3, nearest 2x, ReLU, 1x1 to the head.
  Weights w1, b1, w2, b2, w3, b3;
  const Node* maps = v.producer(v.through(g.outputs[0].name).tensor);
  if (!conv(v, maps, 1, 1, w3, b3)) return std::nullopt;
  const int64_t mid = w3.dims[1], out = w3.dims[0];
  const auto dw_out = v.relu_of(maps->inputs[0]);
  if (!dw_out) return std::nullopt;
  const Node* dw = v.producer(*dw_out);
  if (!conv(v, dw, mid, 3, w2, b2) || w2.dims[0] != mid || w2.dims[1] != 1) return std::nullopt;
  const Node* up = v.producer(v.through(dw->inputs[0]).tensor);
  if (!up || up->op != "Resize" || up->s("mode", "nearest") != "nearest" ||
      up->s("coordinate_transformation_mode", "half_pixel") != "asymmetric" ||
      up->s("nearest_mode", "round_prefer_floor") != "floor" || up->inputs.size() < 3 ||
      (up->inputs.size() > 3 && !up->inputs[3].empty())) {
    return std::nullopt;
  }
  const auto scales = v.constant(up->inputs[2]);
  if (!scales || scales->floats() != std::vector<float>{1.0f, 1.0f, 2.0f, 2.0f}) return std::nullopt;
  const auto head_out = v.relu_of(up->inputs[0]);
  if (!head_out) return std::nullopt;
  const Node* head = v.producer(*head_out);
  if (!conv(v, head, 1, 1, w1, b1) || w1.dims[0] != mid) return std::nullopt;
  if (out < 1 || out > 8) return std::nullopt;
  FusedHead h;
  h.input = head->inputs[0];
  try {
    h.cut = std::make_shared<const std::string>(onnx::cut(bytes, h.input, g.outputs[0].dtype, {-1, w1.dims[1], -1, -1}));
  } catch (const Error&) {
    return std::nullopt;
  }
  h.in = static_cast<int>(w1.dims[1]);
  h.mid = static_cast<int>(mid);
  h.out = static_cast<int>(out);
  h.w1 = std::move(w1.v);
  h.b1 = std::move(b1.v);
  h.w2 = std::move(w2.v);
  h.b2 = std::move(b2.v);
  h.w3 = std::move(w3.v);
  h.b3 = std::move(b3.v);
  return h;
}

}  // namespace carbon::detail
