// The order model on the engine's own
// CPU kernels (carbon/cpu.h), behind carbon::Session (Backend::kNative): its weights
// from <set>/order.native.bin, which the export writes beside order.onnx, its
// forward written out, so a build without ONNX Runtime reads pages in the model's
// order too, and one with it skips ONNX Runtime's per-node work on a model of 22 MMAC.
//
// order.native.bin, little-endian: "ORDERNT1", int32 d, layers, heads, ffn,
// features (16) and labels (13), then float32 weights as PyTorch holds them
// ([out][in]): embed's weight and bias; for each layer qkv's, out's, ff1's and
// ff2's; successor's, same_block's, table_row's, table_col's and table_cell's
// weights (no biases); labels' weight and bias.
//
// The forward is order.py's: the start token (the embedding of a vector of -1s)
// and the lines; per layer a pre-norm attention (layer norms without weights, eps
// 1e-5; the heads' products Q K^T / sqrt(d / heads), a softmax, P V) and a GELU
// feed-forward, each with its residual; a last layer norm; the successor's form over
// every token, the others over the lines made symmetric, and the labels.  Every
// product is carbon::cpu::gemm_f32 on the best instruction set (fused multiply-adds,
// so the bits do not depend on it), the layers' weights packed once at open, a
// head's K^T and V packed per run; a residual joins its GEMM's output stage.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "carbon/cpu.h"
#include "carbon/error.h"
#include "carbon/order.h"
#include "net.h"

namespace carbon::net {

namespace {

[[noreturn]] void refuse(const std::string& why) { throw Error("order.native.bin: " + why); }

constexpr char kMagic[8] = {'O', 'R', 'D', 'E', 'R', 'N', 'T', '1'};

// A linear layer: W^T packed for x W^T, and its bias.
struct Linear {
  cpu::PackedF32 w;
  std::vector<float> b;
  int in = 0, out = 0;
};

class OrderNet {
 public:
  OrderNet(const std::filesystem::path& file, cpu::Isa isa) : isa_(isa) {
    std::ifstream f(file, std::ios::binary);
    if (!f) refuse("cannot open " + file.string());
    char magic[8];
    int32_t dims[6];
    if (!f.read(magic, 8) || std::memcmp(magic, kMagic, 8) != 0) refuse(file.string() + " is not an order model");
    if (!f.read(reinterpret_cast<char*>(dims), sizeof dims)) refuse("its header is cut short");
    d_ = dims[0];
    layers_ = dims[1];
    heads_ = dims[2];
    ffn_ = dims[3];
    if (dims[4] != kOrderFeatures || dims[5] != kBlockLabels || d_ <= 0 || heads_ <= 0 || d_ % heads_ ||
        layers_ <= 0 || layers_ > 64 || ffn_ <= 0 || d_ > 4096 || ffn_ > 16384) {
      refuse("its sizes are not an order model's");
    }
    auto read = [&](int out, int in, bool bias) {
      Linear l;
      l.in = in;
      l.out = out;
      std::vector<float> w(static_cast<size_t>(out) * in), t(w.size());
      if (!f.read(reinterpret_cast<char*>(w.data()), static_cast<std::streamsize>(w.size() * 4))) {
        refuse("its weights are cut short");
      }
      for (int o = 0; o < out; ++o) {  // [out][in] to [in][out]: B of x W^T
        for (int i = 0; i < in; ++i) t[static_cast<size_t>(i) * out + o] = w[static_cast<size_t>(o) * in + i];
      }
      l.w = cpu::pack_f32(t.data(), in, out, isa_);
      if (bias) {
        l.b.resize(out);
        if (!f.read(reinterpret_cast<char*>(l.b.data()), static_cast<std::streamsize>(out * 4))) {
          refuse("its biases are cut short");
        }
      }
      return l;
    };
    embed_ = read(d_, kOrderFeatures, true);
    for (int k = 0; k < layers_; ++k) {
      Layer L;
      L.qkv = read(3 * d_, d_, true);
      L.out = read(d_, d_, true);
      L.ff1 = read(ffn_, d_, true);
      L.ff2 = read(d_, ffn_, true);
      layer_.push_back(std::move(L));
    }
    for (Linear& form : forms_) form = read(d_, d_, false);
    labels_ = read(kBlockLabels, d_, true);
    char extra;
    if (f.read(&extra, 1)) refuse("it is longer than its sizes say");
  }

  // features [n][16] -> successor [(n+1)^2], the four pairwise [n^2], labels [n][13]; pairs_only: the four
  // pairwise left to chain() and among(), out[1..4] empty
  void run(const float* x, int n, std::vector<float>* out, bool pairs_only = false) {
    const int t = n + 1, d = d_, dh = d_ / heads_;
    h_.assign(static_cast<size_t>(t) * d, 0.0f);
    in_.assign(static_cast<size_t>(t) * kOrderFeatures, -1.0f);  // the start token's features
    std::copy(x, x + static_cast<size_t>(n) * kOrderFeatures, in_.begin() + kOrderFeatures);
    gemm(in_.data(), t, kOrderFeatures, embed_, h_.data(), d);
    y_.resize(h_.size());
    h2_.resize(h_.size());
    qkv_.resize(static_cast<size_t>(t) * 3 * d);
    att_.resize(static_cast<size_t>(t) * d);
    ff_.resize(static_cast<size_t>(t) * ffn_);
    p_.resize(static_cast<size_t>(t) * t);
    kt_.resize(static_cast<size_t>(dh) * t);
    v_.resize(static_cast<size_t>(t) * dh);
    const float qk_scale = 1.0f / std::sqrt(static_cast<float>(dh));
    for (Layer& L : layer_) {
      layer_norm(h_.data(), y_.data(), t, d);
      gemm(y_.data(), t, d, L.qkv, qkv_.data(), 3 * d);
      for (int hd = 0; hd < heads_; ++hd) {  // [q | k | v], each head's dh columns
        const float* q = qkv_.data() + hd * dh;
        const float* k = qkv_.data() + d + hd * dh;
        const float* v = qkv_.data() + 2 * d + hd * dh;
        for (int j = 0; j < t; ++j) {
          for (int c = 0; c < dh; ++c) {
            kt_[static_cast<size_t>(c) * t + j] = k[static_cast<size_t>(j) * 3 * d + c];
            v_[static_cast<size_t>(j) * dh + c] = v[static_cast<size_t>(j) * 3 * d + c];
          }
        }
        const cpu::PackedF32 kb = cpu::pack_f32(kt_.data(), dh, t, isa_);
        product(q, t, 3 * d, kb, p_.data(), t);
        cpu::softmax_rows(p_.data(), t, t, qk_scale, isa_);  // the CPU's vectors, the same bits everywhere
        const cpu::PackedF32 vb = cpu::pack_f32(v_.data(), t, dh, isa_);
        product(p_.data(), t, t, vb, att_.data() + hd * dh, d);
      }
      gemm(att_.data(), t, d, L.out, h2_.data(), d, h_.data());  // h + attention
      std::swap(h_, h2_);
      layer_norm(h_.data(), y_.data(), t, d);
      gemm(y_.data(), t, d, L.ff1, ff_.data(), ffn_);
      cpu::gelu(ff_.data(), ff_.size(), isa_);
      gemm(ff_.data(), t, ffn_, L.ff2, h2_.data(), d, h_.data());  // h + feed-forward
      std::swap(h_, h2_);
    }
    layer_norm(h_.data(), y_.data(), t, d);  // y: the tokens, row 0 the start
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));
    // the successor's form over every token
    std::vector<float>& succ = out[0];
    succ.resize(static_cast<size_t>(t) * t);
    gemm(y_.data(), t, d, forms_[0], h2_.data(), d);
    pair(h2_.data(), y_.data(), t, d, succ.data(), scale, false);
    // the others over the lines, made symmetric
    const float* lines = y_.data() + d;
    n_ = n;
    out[5].resize(static_cast<size_t>(n) * kBlockLabels);
    gemm(lines, n, d, labels_, out[5].data(), kBlockLabels);
    if (pairs_only) {  // same_block's w x kept for its chain; the table heads as their blocks ask
      for (int f = 1; f < 5; ++f) out[f].clear();
      wx_.resize(static_cast<size_t>(n) * d);
      gemm(lines, n, d, forms_[1], wx_.data(), d);
      return;
    }
    for (int f = 1; f < 5; ++f) {
      out[f].resize(static_cast<size_t>(n) * n);
      gemm(lines, n, d, forms_[f], h2_.data(), d);
      pair(h2_.data(), lines, n, d, out[f].data(), scale, true);
    }
  }

  // After a pairs_only run: same_block's logits between consecutive lines of order, each (s_ab scale + s_ba
  // scale) / 2 from the chains gemm_f32 sums, as pair() makes them
  std::vector<float> chain(std::span<const size_t> order) const {
    const size_t m = order.size() < 2 ? 0 : order.size() - 1;
    if (static_cast<int64_t>(n_) * d_ >= (int64_t{1} << 31)) refuse("too many lines for the pairs' indices");
    std::vector<int32_t> ia(2 * m), ib(2 * m);
    for (size_t k = 0; k < m; ++k) {
      ia[2 * k] = ib[2 * k + 1] = static_cast<int32_t>(order[k]);
      ib[2 * k] = ia[2 * k + 1] = static_cast<int32_t>(order[k + 1]);
    }
    std::vector<float> s(2 * m);
    const float* lines = y_.data() + d_;
    cpu::dot_chains(wx_.data(), lines, ia.data(), ib.data(), static_cast<int>(2 * m), d_, s.data(), isa_);
    const float scale = 1.0f / std::sqrt(static_cast<float>(d_));
    std::vector<float> out(m);
    for (size_t k = 0; k < m; ++k) out[k] = (s[2 * k] * scale + s[2 * k + 1] * scale) * 0.5f;
    return out;
  }

  // head (1 table_row, 2 table_col, 3 table_cell)'s logits among these lines, [m][m]: pair() on their rows,
  // whose entries are the full matrix's (a GEMM's rows and columns do not change one another's bits)
  std::vector<float> among(int head, std::span<const size_t> which) {
    const int m = static_cast<int>(which.size()), d = d_;
    sub_.resize(static_cast<size_t>(m) * d);
    const float* lines = y_.data() + d;
    for (int r = 0; r < m; ++r) std::copy_n(lines + which[r] * d, d, sub_.data() + static_cast<size_t>(r) * d);
    std::vector<float> s(static_cast<size_t>(m) * m);
    if (m == 0) return s;
    sub_wx_.resize(static_cast<size_t>(m) * d);
    gemm(sub_.data(), m, d, forms_[head + 1], sub_wx_.data(), d);
    pair(sub_wx_.data(), sub_.data(), m, d, s.data(), 1.0f / std::sqrt(static_cast<float>(d)), true);
    return s;
  }

  int d() const { return d_; }

 private:
  struct Layer {
    Linear qkv, out, ff1, ff2;
  };

  // out[m][l.out] = a[m][l.in] W^T + b (+ residual[m][l.out])
  void gemm(const float* a, int m, int64_t lda, const Linear& l, float* out, int64_t ldo,
            const float* residual = nullptr) const {
    cpu::Epilogue e;
    e.bias = l.b.empty() ? nullptr : l.b.data();
    e.residual_f32 = residual;
    e.residual_ld = residual ? ldo : 0;
    e.out_f32 = out;
    e.out_ld = ldo;
    cpu::gemm_f32(cpu::rows_of(a, lda), l.w, m, e);
  }

  // out[m][b.n] = a[m][b.k] B, a product of two activations
  static void product(const float* a, int m, int64_t lda, const cpu::PackedF32& b, float* out, int64_t ldo) {
    cpu::Epilogue e;
    e.out_f32 = out;
    e.out_ld = ldo;
    cpu::gemm_f32(cpu::rows_of(a, lda), b, m, e);
  }

  // s = (w x) y^T * scale, and (s + s^T) / 2 when symmetric
  void pair(const float* wx, const float* y, int m, int d, float* s, float scale, bool symmetric) {
    yt_.resize(static_cast<size_t>(d) * m);
    for (int j = 0; j < m; ++j) {
      for (int c = 0; c < d; ++c) yt_[static_cast<size_t>(c) * m + j] = y[static_cast<size_t>(j) * d + c];
    }
    const cpu::PackedF32 b = cpu::pack_f32(yt_.data(), d, m, isa_);
    product(wx, m, d, b, s, m);
    for (size_t i = 0; i < static_cast<size_t>(m) * m; ++i) s[i] *= scale;
    if (!symmetric) return;
    for (int i = 0; i < m; ++i) {
      for (int j = i + 1; j < m; ++j) {
        const float v = (s[static_cast<size_t>(i) * m + j] + s[static_cast<size_t>(j) * m + i]) * 0.5f;
        s[static_cast<size_t>(i) * m + j] = s[static_cast<size_t>(j) * m + i] = v;
      }
    }
  }

  static void layer_norm(const float* x, float* y, int rows, int d) {
    for (int r = 0; r < rows; ++r) {
      const float* a = x + static_cast<size_t>(r) * d;
      float* b = y + static_cast<size_t>(r) * d;
      float mean = 0;
      for (int c = 0; c < d; ++c) mean += a[c];
      mean /= static_cast<float>(d);
      float var = 0;
      for (int c = 0; c < d; ++c) var += (a[c] - mean) * (a[c] - mean);
      var /= static_cast<float>(d);
      const float inv = 1.0f / std::sqrt(var + 1e-5f);
      for (int c = 0; c < d; ++c) b[c] = (a[c] - mean) * inv;
    }
  }

  cpu::Isa isa_;
  int d_ = 0, layers_ = 0, heads_ = 0, ffn_ = 0;
  Linear embed_, labels_;
  Linear forms_[5];  // successor, same_block, table_row, table_col, table_cell
  std::vector<Layer> layer_;
  std::vector<float> in_, h_, h2_, y_, qkv_, att_, ff_, p_, kt_, v_, yt_;
  std::vector<float> wx_, sub_, sub_wx_;  // same_block's w x; a block's rows and theirs
  int n_ = 0;
};

class OrderSession final : public Session, public OrderPairs {
 public:
  OrderSession(const std::filesystem::path& file, cpu::Isa isa) : net_(file, isa) {
    inputs_ = {{"x", DType::kFloat32, {1, -1, kOrderFeatures}}};
    outputs_ = {{"successor", DType::kFloat32, {1, -1, -1}}, {"same_block", DType::kFloat32, {1, -1, -1}},
                {"table_row", DType::kFloat32, {1, -1, -1}},  {"table_col", DType::kFloat32, {1, -1, -1}},
                {"table_cell", DType::kFloat32, {1, -1, -1}}, {"labels", DType::kFloat32, {1, -1, kBlockLabels}}};
  }

  Backend backend() const override { return Backend::kNative; }

  // Successor and labels, the pairwise heads left to chain() and among(); the spans until the next run
  void run_pairs(const float* x, int n, OrderOutputs& o) {
    net_.run(x, n, pairs_out_, true);
    o = OrderOutputs{};
    o.n = n;
    o.successor = pairs_out_[0];
    o.labels = pairs_out_[5];
    o.pairs = this;
  }
  std::vector<float> chain(int head, std::span<const size_t> order) const override {
    if (head != 0) throw Error("the order model's chain is same_block's");
    return net_.chain(order);
  }
  std::vector<float> among(int head, std::span<const size_t> lines) const override {
    if (head < 1 || head > 3) throw Error("the order model's table heads are 1 to 3");
    return const_cast<OrderNet&>(net_).among(head, lines);  // its buffers, a worker's session
  }

  using Session::run;  // and run() with its outputs in place: the base's, over this one

  std::vector<Tensor> run(std::span<const Tensor> inputs) override {
    check_inputs(inputs);
    const Shape& s = inputs[0].shape();
    const int n = static_cast<int>(s[1]);
    std::vector<float> out[6];
    net_.run(inputs[0].data<float>().data(), n, out);
    std::vector<Tensor> y;
    const int64_t t = n + 1;
    const Shape shapes[6] = {{1, t, t}, {1, n, n}, {1, n, n}, {1, n, n}, {1, n, n}, {1, n, kBlockLabels}};
    for (int i = 0; i < 6; ++i) {
      Tensor o = Tensor::empty(DType::kFloat32, shapes[i]);
      std::copy(out[i].begin(), out[i].end(), o.data<float>().begin());
      y.push_back(std::move(o));
    }
    return y;
  }

 private:
  OrderNet net_;
  std::vector<float> pairs_out_[6];
};

}  // namespace

bool is_order_net(const std::filesystem::path& file) {
  std::ifstream f(file, std::ios::binary);
  char magic[8];
  return f.read(magic, 8) && std::memcmp(magic, kMagic, 8) == 0;
}

std::unique_ptr<Session> open_order_net(const std::filesystem::path& file, cpu::Isa isa) {
  return std::make_unique<OrderSession>(file, isa);
}

bool run_order_pairs(Session& s, std::span<const float> x, int n, OrderOutputs& out) {
  auto* o = dynamic_cast<OrderSession*>(&s);
  if (!o) return false;
  if (x.size() != static_cast<size_t>(n) * kOrderFeatures) throw Error("the order model's features are not n x 16");
  o->run_pairs(x.data(), n, out);
  return true;
}

}  // namespace carbon::net
