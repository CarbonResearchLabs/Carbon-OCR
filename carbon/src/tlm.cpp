// The small transformer language model (carbon/tlm.h) on the CPU's kernels: its file read in any of its
// precisions (tlm_file.h), and run in it.
#include "carbon/tlm.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "carbon/cpu.h"
#include "carbon/error.h"
#include "carbon/lm.h"
#include "tlm_file.h"

namespace carbon {

const char* precision_name(TlmPrecision p) {
  switch (p) {
    case TlmPrecision::kF32: return "fp32";
    case TlmPrecision::kF16: return "fp16";
    case TlmPrecision::kInt8: return "int8";
  }
  return "unknown";
}

namespace tlm {

File read(const std::filesystem::path& file) {
  std::ifstream f(file, std::ios::binary | std::ios::ate);
  if (!f) throw Error("cannot open " + file.string());
  const auto size = static_cast<size_t>(f.tellg());
  f.seekg(0);
  std::vector<char> raw(size);
  f.read(raw.data(), static_cast<std::streamsize>(size));
  const std::string what = file.string();
  if (size < sizeof(TlmHeader) || std::memcmp(raw.data(), kTlmMagic, sizeof(kTlmMagic)) != 0) {
    throw Error(what + ": not a transformer language model (CRBNTLM1)");
  }
  File out;
  std::memcpy(&out.h, raw.data(), sizeof(TlmHeader));
  const TlmHeader& h = out.h;
  const TlmPrecision P = h.precision;
  const bool known = P == TlmPrecision::kF32 || P == TlmPrecision::kF16 || P == TlmPrecision::kInt8;
  if (!(h.version == 1 && P == TlmPrecision::kF32) && !(h.version == 2 && known)) {
    throw Error(what + ": version " + std::to_string(h.version) + ", precision " +
                std::to_string(static_cast<uint32_t>(P)) + ": not a file this engine reads (1: FP32; 2: FP32, FP16, INT8)");
  }
  if (h.d == 0 || h.heads == 0 || h.d % h.heads || (h.d / h.heads) % 8 || h.d / h.heads > 128 || h.d % 8 ||
      h.ffn == 0 || h.ctx < 2 || h.layers == 0 || h.tokens != h.classes + 1) {
    throw Error(what + ": a shape the engine does not run (d " + std::to_string(h.d) + ", " + std::to_string(h.heads) +
                " heads, ffn " + std::to_string(h.ffn) + ", ctx " + std::to_string(h.ctx) + ")");
  }
  const size_t d = h.d, ffn = h.ffn, T = h.tokens, C = h.ctx;
  const bool i8 = P == TlmPrecision::kInt8, f16 = P == TlmPrecision::kF16;
  // its size: FP32 and FP16 every value in 4 or 2 bytes; INT8 a weight's values a byte each, then its columns'
  // scales and its input's scale and zero point, every other value in 4
  const size_t e = f16 ? 2 : 4;
  auto matrix_bytes = [&](size_t in, size_t n) { return i8 ? in * n + 4 * n + 8 : e * in * n; };
  size_t want = i8 ? T * d + 4 * T : e * T * d;
  want += e * C * d;
  want += h.layers * (e * 2 * d + matrix_bytes(d, 3 * d) + e * 3 * d + matrix_bytes(d, d) + e * d + e * 2 * d +
                      matrix_bytes(d, ffn) + e * ffn + matrix_bytes(ffn, d) + e * d);
  want += e * 2 * d + (i8 ? 8 : 0);
  if (size != sizeof(TlmHeader) + want) {
    throw Error(what + ": " + std::to_string(size) + " bytes, its shapes take " +
                std::to_string(sizeof(TlmHeader) + want) + " in " + precision_name(P));
  }
  const char* p = raw.data() + sizeof(TlmHeader);
  auto floats = [&](size_t n) {  // n float32 values
    std::vector<float> v(n);
    std::memcpy(v.data(), p, 4 * n);
    p += 4 * n;
    return v;
  };
  auto halves = [&](size_t n) {
    std::vector<uint16_t> v(n);
    std::memcpy(v.data(), p, 2 * n);
    p += 2 * n;
    return v;
  };
  auto bytes = [&](size_t n) {
    std::vector<int8_t> v(n);
    std::memcpy(v.data(), p, n);
    p += n;
    return v;
  };
  auto widened = [](const std::vector<uint16_t>& v) {
    std::vector<float> w(v.size());
    cpu::to_f32(v.data(), w.data(), v.size());
    return w;
  };
  auto values = [&](size_t n) { return f16 ? widened(halves(n)) : floats(n); };  // not a weight
  auto input_quant = [&](Matrix& m, const char* which) {
    const std::vector<float> q = floats(2);
    m.in_scale = q[0], m.in_zero = q[1];
    if (!(m.in_scale > 0) || !(m.in_zero >= 0 && m.in_zero <= 255) || m.in_zero != std::nearbyint(m.in_zero)) {
      throw Error(what + ": " + which + "'s input quantization (scale " + std::to_string(m.in_scale) + ", zero point " +
                  std::to_string(m.in_zero) + ") is not uint8's");
    }
  };
  auto matrix = [&](size_t in, size_t n, const char* which) {
    Matrix m;
    m.in = static_cast<int>(in), m.out = static_cast<int>(n);
    if (f16) {
      m.f16 = halves(in * n);
      m.f32 = widened(m.f16);
    } else if (i8) {
      m.i8 = bytes(in * n);
      m.scale = floats(n);
      input_quant(m, which);
      m.f32.resize(in * n);
      for (size_t k = 0; k < in; ++k) {
        for (size_t j = 0; j < n; ++j) m.f32[k * n + j] = static_cast<float>(m.i8[k * n + j]) * m.scale[j];
      }
    } else {
      m.f32 = floats(in * n);
    }
    return m;
  };
  // the embedding: INT8's per token (the head's output channels), else as any tensor
  std::vector<int8_t> emb8;
  std::vector<float> emb_scale;
  std::vector<uint16_t> emb16;
  if (i8) {
    emb8 = bytes(T * d);
    emb_scale = floats(T);
    out.emb.resize(T * d);
    for (size_t t = 0; t < T; ++t) {
      for (size_t j = 0; j < d; ++j) out.emb[t * d + j] = static_cast<float>(emb8[t * d + j]) * emb_scale[t];
    }
  } else if (f16) {
    emb16 = halves(T * d);
    out.emb = widened(emb16);
  } else {
    out.emb = floats(T * d);
  }
  out.pos = values(C * d);
  for (uint32_t l = 0; l < h.layers; ++l) {
    Layer L;
    L.ln1w = values(d);
    L.ln1b = values(d);
    L.qkv = matrix(d, 3 * d, "qkv");
    L.bqkv = values(3 * d);
    L.o = matrix(d, d, "out");
    L.bo = values(d);
    L.ln2w = values(d);
    L.ln2b = values(d);
    L.w1 = matrix(d, ffn, "ffn1");
    L.b1 = values(ffn);
    L.w2 = matrix(ffn, d, "ffn2");
    L.b2 = values(d);
    out.layers.push_back(std::move(L));
  }
  out.lnfw = values(d);
  out.lnfb = values(d);
  // the tied head: the embedding, transposed ([d][tokens])
  Matrix& hd = out.head;
  hd.in = static_cast<int>(d), hd.out = static_cast<int>(T);
  hd.f32.resize(d * T);
  for (size_t t = 0; t < T; ++t) {
    for (size_t j = 0; j < d; ++j) hd.f32[j * T + t] = out.emb[t * d + j];
  }
  if (f16) {
    hd.f16.resize(d * T);
    for (size_t t = 0; t < T; ++t) {
      for (size_t j = 0; j < d; ++j) hd.f16[j * T + t] = emb16[t * d + j];
    }
  } else if (i8) {
    hd.i8.resize(d * T);
    for (size_t t = 0; t < T; ++t) {
      for (size_t j = 0; j < d; ++j) hd.i8[j * T + t] = emb8[t * d + j];
    }
    hd.scale = std::move(emb_scale);
    input_quant(hd, "the head");
  }
  return out;
}

}  // namespace tlm

namespace {

// A linear layer packed for the CPU's kernels, in its file's precision.
struct Linear {
  TlmPrecision p = TlmPrecision::kF32;
  int k = 0, n = 0;
  cpu::PackedF32 f32;
  cpu::PackedF16 f16;
  cpu::PackedB i8;
  std::vector<float> scale;  // kInt8: the input's scale times each column's, the sums' scale
  float in_scale = 1;
  int32_t in_zero = 0;
};

Linear pack(const tlm::Matrix& m, TlmPrecision p, cpu::Isa isa) {
  Linear L;
  L.p = p, L.k = m.in, L.n = m.out;
  if (p == TlmPrecision::kF16) {
    L.f16 = cpu::pack_f16(m.f16.data(), m.in, m.out, isa);
  } else if (p == TlmPrecision::kInt8) {
    L.i8 = cpu::pack_b(m.i8.data(), m.in, m.out, isa);
    L.scale.resize(static_cast<size_t>(m.out));
    for (int j = 0; j < m.out; ++j) L.scale[static_cast<size_t>(j)] = m.in_scale * m.scale[static_cast<size_t>(j)];
    L.in_scale = m.in_scale;
    L.in_zero = static_cast<int32_t>(m.in_zero);
  } else {
    L.f32 = cpu::pack_f32(m.f32.data(), m.in, m.out, isa);
  }
  return L;
}

struct Layer {
  std::vector<float> ln1w, ln1b, bqkv, bo, ln2w, ln2b, b1, b2;
  Linear wqkv, wo, w1, w2;
};

// A thread's buffers, kept between calls.
struct Scratch {
  std::vector<float> x, x2, h, qkv, att, f, logits, sc, lp;
  std::vector<std::vector<float>> cache;  // each layer's qkv of the shared rows
  std::vector<int32_t> tok, pos, tgt, row_of;
  cpu::Vector64<uint8_t> q;  // an INT8 layer's input, quantized
};

thread_local Scratch tls;

// round(v / scale) + zero, clamped to 0 .. 255, rounding half to even as QuantizeLinear does (the native backend's
// arithmetic, net/net.cpp: adding and taking away 1.5 * 2^23 rounds a float under 2^22 to a whole number)
inline uint8_t quantize(float v, float scale, int32_t zero) {
  float r = v / scale;
  r = std::min(4194304.0f, std::max(-4194304.0f, r));
  r = (r + 12582912.0f) - 12582912.0f;
  r += static_cast<float>(zero);
  return static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, r)));
}

// out = a w + bias (+ residual): a's rows w.k values apart, out's w.n
void linear(const float* a, int rows, const Linear& w, const float* bias, const float* residual, float* out,
            Scratch& s) {
  cpu::Epilogue e;
  e.bias = bias;
  e.out_f32 = out;
  e.out_ld = w.n;
  if (residual) {
    e.residual_f32 = residual;
    e.residual_ld = w.n;
  }
  if (w.p == TlmPrecision::kF16) {
    cpu::gemm_f16(cpu::rows_of(a, w.k), w.f16, rows, e);
  } else if (w.p == TlmPrecision::kInt8) {
    // each row quantized into k_padded(k) bytes (those past k meet the packing's zero padding: any value)
    const int kp = cpu::k_padded(w.k, w.i8.isa);
    s.q.resize(static_cast<size_t>(rows) * kp + 64);
    for (int r = 0; r < rows; ++r) {
      const float* src = a + static_cast<size_t>(r) * w.k;
      uint8_t* dst = s.q.data() + static_cast<size_t>(r) * kp;
      for (int j = 0; j < w.k; ++j) dst[j] = quantize(src[j], w.in_scale, w.in_zero);
    }
    e.scale = w.scale.data();
    cpu::gemm_u8s8(cpu::rows_of<uint8_t>(s.q.data(), kp), w.in_zero, w.i8, rows, e);
  } else {
    cpu::gemm_f32(cpu::rows_of(a, w.k), w.f32, rows, e);
  }
}

}  // namespace

struct TransformerLM::Weights {
  std::vector<float> emb, pos, lnfw, lnfb;
  std::vector<Layer> layers;
  Linear head;  // the embedding's transpose, [d][tokens]
};

TransformerLM::TransformerLM() = default;
TransformerLM::~TransformerLM() = default;

std::shared_ptr<TransformerLM> TransformerLM::open(const std::filesystem::path& file) {
  tlm::File f = tlm::read(file);
  std::shared_ptr<TransformerLM> m(new TransformerLM());
  m->h_ = f.h;
  const TlmPrecision P = f.h.precision;
  const cpu::Isa isa = cpu::best_isa();
  m->w_ = std::make_unique<Weights>();
  Weights& w = *m->w_;
  w.emb = std::move(f.emb);
  w.pos = std::move(f.pos);
  for (tlm::Layer& fl : f.layers) {
    Layer L;
    L.ln1w = std::move(fl.ln1w);
    L.ln1b = std::move(fl.ln1b);
    L.wqkv = pack(fl.qkv, P, isa);
    L.bqkv = std::move(fl.bqkv);
    L.wo = pack(fl.o, P, isa);
    L.bo = std::move(fl.bo);
    L.ln2w = std::move(fl.ln2w);
    L.ln2b = std::move(fl.ln2b);
    L.w1 = pack(fl.w1, P, isa);
    L.b1 = std::move(fl.b1);
    L.w2 = pack(fl.w2, P, isa);
    L.b2 = std::move(fl.b2);
    w.layers.push_back(std::move(L));
  }
  w.lnfw = std::move(f.lnfw);
  w.lnfb = std::move(f.lnfb);
  w.head = pack(f.head, P, isa);
  return m;
}

void TransformerLM::check_vocab(const std::vector<std::string>& vocab, const std::string& what) const {
  if (vocab.size() != h_.classes || vocab_hash(vocab) != h_.vocab_hash) {
    throw Error(what + ": its transformer language model was trained on another vocabulary");
  }
}

namespace {

// The blocks over `rows` rows of x (rows d apart), each row's queries attending as `keys(layer, row)` says: a span
// of KeyRuns, the head's offset added here. qkv_out, where given, receives each layer's qkv rows (the shared pass's
// cache).
template <class Keys>
void blocks(const std::vector<Layer>& layers, int d, int heads, int ffn, int rows, Scratch& s, Keys&& keys,
            std::vector<std::vector<float>>* qkv_out) {
  const int dh = d / heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
  s.h.resize(static_cast<size_t>(rows) * d);
  s.x2.resize(static_cast<size_t>(rows) * d);
  s.qkv.resize(static_cast<size_t>(rows) * 3 * d);
  s.att.resize(static_cast<size_t>(rows) * d);
  s.f.resize(static_cast<size_t>(rows) * ffn);
  for (size_t l = 0; l < layers.size(); ++l) {
    const Layer& L = layers[l];
    cpu::layer_norm_rows(s.x.data(), d, rows, d, L.ln1w.data(), L.ln1b.data(), 1e-5f, s.h.data(), d);
    linear(s.h.data(), rows, L.wqkv, L.bqkv.data(), nullptr, s.qkv.data(), s);
    if (qkv_out) (*qkv_out)[l].assign(s.qkv.begin(), s.qkv.end());
    for (int r = 0; r < rows; ++r) {
      cpu::KeyRun runs[2];
      const int n_runs = keys(static_cast<int>(l), r, runs);
      int total = 0;
      for (int i = 0; i < n_runs; ++i) total += runs[i].n;
      s.sc.resize(static_cast<size_t>(std::max(total, 8)));
      for (int hh = 0; hh < heads; ++hh) {
        cpu::KeyRun hr[2];
        for (int i = 0; i < n_runs; ++i) {
          hr[i] = runs[i];
          hr[i].k += hh * dh;
          hr[i].v += hh * dh;
        }
        cpu::attend(s.qkv.data() + static_cast<size_t>(r) * 3 * d + hh * dh, dh, hr, n_runs, scale,
                    s.att.data() + static_cast<size_t>(r) * d + hh * dh, s.sc.data());
      }
    }
    linear(s.att.data(), rows, L.wo, L.bo.data(), s.x.data(), s.x2.data(), s);
    std::swap(s.x, s.x2);
    cpu::layer_norm_rows(s.x.data(), d, rows, d, L.ln2w.data(), L.ln2b.data(), 1e-5f, s.h.data(), d);
    linear(s.h.data(), rows, L.w1, L.b1.data(), nullptr, s.f.data(), s);
    cpu::gelu(s.f.data(), s.f.size());
    linear(s.f.data(), rows, L.w2, L.b2.data(), s.x.data(), s.x2.data(), s);
    std::swap(s.x, s.x2);
  }
}

void embed(const std::vector<float>& emb, const std::vector<float>& pos, int d, std::span<const int32_t> tok,
           std::span<const int32_t> at, float* x) {
  for (size_t r = 0; r < tok.size(); ++r) {
    const float* e = emb.data() + static_cast<size_t>(tok[r]) * d;
    const float* p = pos.data() + static_cast<size_t>(at[r]) * d;
    float* o = x + r * d;
    for (int j = 0; j < d; ++j) o[j] = e[j] + p[j];
  }
}

}  // namespace

double TransformerLM::logprob(std::span<const int32_t> context, std::span<const int32_t> tokens, bool end) const {
  std::vector<int32_t> r(tokens.begin(), tokens.end());
  double out = 0;
  if (end) {
    score(context, std::span<const std::vector<int32_t>>(&r, 1), std::span<double>(&out, 1));
    return out;
  }
  // without the end: the full score less the end's own term
  std::vector<int32_t> ctx(context.begin(), context.end());
  ctx.insert(ctx.end(), tokens.begin(), tokens.end());
  double with_end = 0, end_only = 0;
  score(context, std::span<const std::vector<int32_t>>(&r, 1), std::span<double>(&with_end, 1));
  const std::vector<int32_t> none;
  score(ctx, std::span<const std::vector<int32_t>>(&none, 1), std::span<double>(&end_only, 1));
  return with_end - end_only;
}

void TransformerLM::score(std::span<const int32_t> context, std::span<const std::vector<int32_t>> readings,
                          std::span<double> out) const {
  const int R = static_cast<int>(readings.size());
  if (R == 0) return;
  if (context.empty()) throw Error("TransformerLM::score: a context of at least one token (the boundary)");
  const int d = static_cast<int>(h_.d), heads = static_cast<int>(h_.heads), ffn = static_cast<int>(h_.ffn);
  const int C = static_cast<int>(h_.ctx), T = static_cast<int>(h_.tokens), B = static_cast<int>(h_.classes);
  for (const auto& rd : readings) {
    for (const int32_t t : rd) {
      if (t < 1 || t > B) throw Error("TransformerLM::score: token " + std::to_string(t));
    }
  }
  for (const int32_t t : context) {
    if (t < 1 || t > B) throw Error("TransformerLM::score: context token " + std::to_string(t));
  }
  size_t longest = 0;
  for (const auto& rd : readings) longest = std::max(longest, rd.size());
  // a reading longer than the window: its own windows, ctx / 2 tokens of the one before as context
  if (static_cast<int>(longest) + 1 > C) {
    for (int r = 0; r < R; ++r) {
      const auto& rd = readings[static_cast<size_t>(r)];
      if (static_cast<int>(rd.size()) + 1 <= C) {
        score(context, std::span<const std::vector<int32_t>>(&rd, 1), out.subspan(static_cast<size_t>(r), 1));
        continue;
      }
      double sum = 0;
      const size_t step = static_cast<size_t>(C / 2);
      std::vector<int32_t> ctx(context.end() - 1, context.end());
      for (size_t a = 0; a < rd.size(); a += step) {
        const size_t b = std::min(rd.size(), a + step);
        const std::vector<int32_t> piece(rd.begin() + static_cast<std::ptrdiff_t>(a),
                                         rd.begin() + static_cast<std::ptrdiff_t>(b));
        double v = 0, v_end = 0;
        score(ctx, std::span<const std::vector<int32_t>>(&piece, 1), std::span<double>(&v, 1));
        if (b < rd.size()) {  // not the line's end yet: take the end's term back out
          std::vector<int32_t> c2 = ctx;
          c2.insert(c2.end(), piece.begin(), piece.end());
          const std::vector<int32_t> none;
          score(c2, std::span<const std::vector<int32_t>>(&none, 1), std::span<double>(&v_end, 1));
        }
        sum += v - v_end;
        ctx.assign(piece.begin(), piece.end());
      }
      out[static_cast<size_t>(r)] = sum;
    }
    return;
  }
  // the context, its oldest tokens dropped to fit the window with the longest reading
  const size_t keep = std::min(context.size(), static_cast<size_t>(C) - longest);
  const std::span<const int32_t> ctx = context.subspan(context.size() - keep);
  // the readings' common prefix
  size_t P = readings[0].size();
  for (int r = 1; r < R; ++r) {
    const auto& rd = readings[static_cast<size_t>(r)];
    size_t k = 0;
    while (k < P && k < rd.size() && rd[k] == readings[0][k]) ++k;
    P = k;
  }
  Scratch& s = tls;
  const Weights& w = *w_;
  // ---- the shared rows: the context and the common prefix, causal among themselves
  const int Ls = static_cast<int>(ctx.size() + P);
  s.tok.assign(ctx.begin(), ctx.end());
  s.tok.insert(s.tok.end(), readings[0].begin(), readings[0].begin() + static_cast<std::ptrdiff_t>(P));
  s.pos.resize(static_cast<size_t>(Ls));
  for (int i = 0; i < Ls; ++i) s.pos[static_cast<size_t>(i)] = i;
  s.x.resize(static_cast<size_t>(Ls) * d);
  embed(w.emb, w.pos, d, s.tok, s.pos, s.x.data());
  s.cache.resize(w.layers.size());
  blocks(w.layers, d, heads, ffn, Ls, s,
         [&](int, int r, cpu::KeyRun* runs) {
           runs[0] = {s.qkv.data() + d, s.qkv.data() + 2 * d, 3 * d, r + 1};
           return 1;
         },
         &s.cache);
  // the shared rows' predictions: the prefix's tokens, and each reading's first token after it (or its end)
  const int first = static_cast<int>(ctx.size()) - 1, nrows = Ls - first;
  s.h.resize(static_cast<size_t>(nrows) * d);
  cpu::layer_norm_rows(s.x.data() + static_cast<size_t>(first) * d, d, nrows, d, w.lnfw.data(), w.lnfb.data(), 1e-5f,
                       s.h.data(), d);
  s.logits.resize(static_cast<size_t>(nrows) * T);
  linear(s.h.data(), nrows, w.head, nullptr, nullptr, s.logits.data(), s);
  double shared = 0;
  if (P > 0) {
    s.tgt.assign(readings[0].begin(), readings[0].begin() + static_cast<std::ptrdiff_t>(P));
    s.lp.resize(P);
    cpu::log_softmax_pick(s.logits.data(), T, static_cast<int>(P), T, s.tgt.data(), s.lp.data());
    for (size_t i = 0; i < P; ++i) shared += s.lp[i];
  }
  s.tgt.resize(static_cast<size_t>(R));
  for (int r = 0; r < R; ++r) {
    const auto& rd = readings[static_cast<size_t>(r)];
    s.tgt[static_cast<size_t>(r)] = rd.size() > P ? rd[P] : B;
  }
  s.lp.resize(static_cast<size_t>(R));
  cpu::log_softmax_pick(s.logits.data() + static_cast<size_t>(nrows - 1) * T, 0, R, T, s.tgt.data(), s.lp.data());
  for (int r = 0; r < R; ++r) out[static_cast<size_t>(r)] = shared + s.lp[static_cast<size_t>(r)];
  // ---- every reading's rest, in one batch: each row attends to the shared rows and its own reading's rows
  std::vector<int> start(static_cast<size_t>(R) + 1, 0);
  for (int r = 0; r < R; ++r) {
    start[static_cast<size_t>(r) + 1] =
        start[static_cast<size_t>(r)] + static_cast<int>(readings[static_cast<size_t>(r)].size() - P);
  }
  const int M = start[static_cast<size_t>(R)];
  if (M == 0) return;
  s.tok.resize(static_cast<size_t>(M));
  s.pos.resize(static_cast<size_t>(M));
  s.row_of.resize(static_cast<size_t>(M));
  s.tgt.resize(static_cast<size_t>(M));
  for (int r = 0; r < R; ++r) {
    const auto& rd = readings[static_cast<size_t>(r)];
    for (int j = start[static_cast<size_t>(r)]; j < start[static_cast<size_t>(r) + 1]; ++j) {
      const size_t at = P + static_cast<size_t>(j - start[static_cast<size_t>(r)]);
      s.tok[static_cast<size_t>(j)] = rd[at];
      s.pos[static_cast<size_t>(j)] = Ls + static_cast<int>(at - P);
      s.row_of[static_cast<size_t>(j)] = r;
      s.tgt[static_cast<size_t>(j)] = at + 1 < rd.size() ? rd[at + 1] : B;
    }
  }
  s.x.resize(static_cast<size_t>(M) * d);
  embed(w.emb, w.pos, d, s.tok, s.pos, s.x.data());
  blocks(w.layers, d, heads, ffn, M, s,
         [&](int l, int j, cpu::KeyRun* runs) {
           const float* c = s.cache[static_cast<size_t>(l)].data();
           runs[0] = {c + d, c + 2 * d, 3 * d, Ls};
           const int r0 = start[static_cast<size_t>(s.row_of[static_cast<size_t>(j)])];
           const float* q = s.qkv.data() + static_cast<size_t>(r0) * 3 * d;
           runs[1] = {q + d, q + 2 * d, 3 * d, j - r0 + 1};
           return 2;
         },
         nullptr);
  s.h.resize(static_cast<size_t>(M) * d);
  cpu::layer_norm_rows(s.x.data(), d, M, d, w.lnfw.data(), w.lnfb.data(), 1e-5f, s.h.data(), d);
  s.logits.resize(static_cast<size_t>(M) * T);
  linear(s.h.data(), M, w.head, nullptr, nullptr, s.logits.data(), s);
  s.lp.resize(static_cast<size_t>(M));
  cpu::log_softmax_pick(s.logits.data(), T, M, T, s.tgt.data(), s.lp.data());
  for (int j = 0; j < M; ++j) out[static_cast<size_t>(s.row_of[static_cast<size_t>(j)])] += s.lp[static_cast<size_t>(j)];
}

#ifndef CARBON_WITH_CUDA
std::shared_ptr<LineRescorer> cuda_transformer_lm(const std::filesystem::path& file, int) {
  throw Error(file.string() + ": the transformer on a GPU needs CUDA, and this build has none");
}
#endif

}  // namespace carbon
