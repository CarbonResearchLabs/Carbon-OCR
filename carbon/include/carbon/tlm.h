// The small transformer language model: a causal character transformer over a
// recognizer's classes and the boundary, run here on the CPU's own kernels to
// rescore the readings the beam keeps for a line (carbon/beam.h, ctc_beam_nbest).
//
// The file (tlm-<group>.bin), little-endian: a 256-byte TlmHeader, then the
// tensors in this order, each linear layer's weight as [in][out]:
//   embedding [tokens][d], positions [ctx][d]
//   for each layer: ln1 w, b [d]; qkv w [d][3d], b [3d]; out w [d][d], b [d];
//                   ln2 w, b [d]; ffn1 w [d][ffn], b [ffn]; ffn2 w [ffn][d], b [d]
//   final ln w, b [d]
// The output is the embedding's transpose (tied).  Tokens are the n-gram's
// (carbon/lm.h): the classes 1 .. classes - 1, and the boundary, `classes`.
//
// Three precisions, as Fullerene's models ship, each gated against FP32 when it
// is exported:
//   FP32  tlm-<group>.bin       version 1: every tensor float32
//   FP16  tlm-<group>.fp16.bin  version 2, precision kF16: every tensor binary16
//   INT8  tlm-<group>.int8.bin  version 2, precision kInt8, the scheme of
//         Fullerene's INT8 model files: each weight (the embedding
//         too) int8 per output channel, symmetric, then its float32 scales [out]
//         (the embedding's, a token's: the head's output channels); each linear
//         layer's input uint8 per tensor with a zero point, its float32 scale and
//         zero point after its weight's scales (calibrated on held-out text, the
//         range widened to hold 0); the head's input's two last in the file.
//         Positions, layer norms and biases float32.
// The CPU runs INT8 in int8 (cpu::gemm_u8s8: each input quantized as
// QuantizeLinear rounds, the sums scaled by the input's and the column's scales)
// and FP16 on gemm_f16; the GPU runs every precision on its FP16 tensor cores, an
// INT8 file's as whole numbers FP16 holds exactly (each input's quantized steps,
// less its zero point, where it is made, and the int8 weights), so its FP32 sums
// are the CPU's int8 sums, scaled as the CPU scales them.
//
// A rescore runs a line's context (its own start, or the line before and the
// boundary) and the readings' common prefix once, keeping every layer's keys and
// values, then every reading's rest in one batch against that cache: a beam's
// readings differ in a few characters, so most of their tokens run once.  Every
// matrix product is cpu::gemm_f32 on packed weights (AVX2 or AVX-512 FMA, the bias
// and the residual fused), and the layer norms, the attention and the final
// log-softmax are cpu::layer_norm_rows, cpu::attend and cpu::log_softmax_pick.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace carbon {

constexpr char kTlmMagic[8] = {'C', 'R', 'B', 'N', 'T', 'L', 'M', '1'};

enum class TlmPrecision : uint32_t { kF32 = 0, kF16 = 1, kInt8 = 2 };

struct TlmHeader {
  char magic[8];
  uint32_t version, d, layers, heads, ffn, ctx, tokens, classes;
  uint64_t vocab_hash;
  float alpha, beta;  // the rescore's weight and a character's bonus, tuned; 0, 0: not tuned
  uint64_t params;
  float heldout;      // its held-out nats a token when it was exported (FP32's, in every precision's file)
  char note[128];
  TlmPrecision precision;  // version 2; version 1 is FP32 (this word 0)
  float gate;              // the beam's gate tuned for the transformer alone (EngineOptions::lm_mode); 0: not tuned
  uint8_t reserved[52];
};
static_assert(sizeof(TlmHeader) == 256);

const char* precision_name(TlmPrecision p);  // "fp32", "fp16", "int8"

// One line's rescoring: its context (tokens, oldest first: at least the boundary that starts a line) and its
// readings.
struct RescoreQuery {
  std::span<const int32_t> context;
  std::span<const std::vector<int32_t>> readings;
};

// What rescores the beam's readings of a line: ln P of each reading (its classes, and the line's end) after a
// context. Thread-safe.
class LineRescorer {
 public:
  virtual ~LineRescorer() = default;
  virtual void score(std::span<const int32_t> context, std::span<const std::vector<int32_t>> readings,
                     std::span<double> out) const = 0;
  // A page's lines at once (out[i] sized to query i's readings): a GPU's in one run; by default one at a time.
  virtual void score_lines(std::span<const RescoreQuery> queries, std::span<std::vector<double>> out) const {
    for (size_t i = 0; i < queries.size(); ++i) {
      out[i].assign(queries[i].readings.size(), 0.0);
      score(queries[i].context, queries[i].readings, out[i]);
    }
  }
  virtual int boundary() const = 0;
};

// The same model on a GPU (src/cuda/tlm.cu): a page's lines rescored in one run, every product on the tensor
// cores (FP16 in, FP32 sums, the bias, GELU and residual fused), the layer norms, attention and log-softmax a warp a
// row; within a few hundredths of a nat of the CPU's. A line longer than its window is scored in windows, as the
// CPU's model scores it, in the same run. Callers at once run at once, each on a stream and buffers of its own (up to
// 8; more wait for one). A page's lines run in batches of a bounded size, one after another, so each caller's buffers
// are a batch's (about 120 MB for Latin's model) however large its pages. Throws Error where CUDA is not built in, or
// where the model's heads are not 32, 64 or 128 wide.
std::shared_ptr<LineRescorer> cuda_transformer_lm(const std::filesystem::path& file, int gpu = 0);

class TransformerLM final : public LineRescorer {
 public:
  // Reads and packs the file; throws Error if it is not a model or does not hold its weights.
  static std::shared_ptr<TransformerLM> open(const std::filesystem::path& file);
  ~TransformerLM() override;

  void score(std::span<const int32_t> context, std::span<const std::vector<int32_t>> readings,
             std::span<double> out) const override;
  int boundary() const override { return static_cast<int>(h_.classes); }
  // ln P(tokens, then the line's end where `end`) after context.
  double logprob(std::span<const int32_t> context, std::span<const int32_t> tokens, bool end = true) const;

  const TlmHeader& header() const { return h_; }
  TlmPrecision precision() const { return h_.precision; }
  int ctx() const { return static_cast<int>(h_.ctx); }
  // Throws Error unless the model was trained on this vocabulary.
  void check_vocab(const std::vector<std::string>& vocab, const std::string& what) const;

 private:
  struct Weights;
  TransformerLM();
  TlmHeader h_{};
  std::unique_ptr<Weights> w_;
};

}  // namespace carbon
