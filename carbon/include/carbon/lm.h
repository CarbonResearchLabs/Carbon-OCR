// The n-gram language model the beam reads unsure lines with (carbon/beam.h): a
// character n-gram over a recognizer's own classes, in a file mapped into memory.
//
// Tokens are the recognizer's classes 1 .. classes - 1 (0, the CTC blank, never
// occurs in text) and the boundary, `classes`: the context at a line's start and
// the token that ends it.  A model of order N reads at most N - 1 tokens back.
//
// The file (lm-<group>.bin), little-endian, every section on 64 bytes:
//   NgramHeader                    512 bytes
//   unigrams  [classes + 1]        {logp, bow}: token t's probability and its
//                                  back-off weight as a context, natural logs
//   order n's table, n = 2 .. N    an open-addressing hash table of NgramEntry,
//                                  `capacity` (a power of 2) slots, key 0 empty:
//                                  slot ngram_slot(key, capacity), then the next
// An entry's key is a 64-bit fingerprint of its tokens (ngram_key), not the tokens:
// any order and any alphabet in one 16-byte entry, four to a cache line.  Two
// n-grams of a table share a fingerprint with a probability of about one in 2^64
// over the pair.
//
// A query's state is the context packed in 64 bits: its length L in the top 3
// bits, its last L tokens below, `bits` bits each, the newest lowest.  So a model
// holds (N - 1) * bits <= 61: order 7 at 800 classes (10 bits), order 5 at 12,000.
// The state after a token is as short as the model allows (KenLM's state
// minimization): the longest suffix that is a context in the model, so the next
// query starts where an entry can be.
#pragma once

#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "carbon/beam.h"

namespace carbon {

constexpr char kNgramMagic[8] = {'C', 'R', 'B', 'N', 'G', 'R', 'M', '1'};
constexpr int kNgramMaxOrder = 8;

struct NgramTable {
  uint64_t offset = 0;    // from the file's start
  uint64_t capacity = 0;  // slots, a power of 2; 0 for the unigrams, which are dense
  uint64_t count = 0;     // entries
};

struct NgramHeader {
  char magic[8];
  uint32_t version;    // 1
  uint32_t order;      // N
  uint32_t classes;    // the recognizer's, the blank among them; the boundary token is `classes`
  uint32_t bits;       // bits a token takes in a packed context: bit_width(classes)
  uint64_t vocab_hash; // vocab_hash() of the recognizer's vocabulary
  NgramTable tables[kNgramMaxOrder + 1];  // [n] order n's; [1] the unigrams
  float alpha, beta, gate;  // the beam's weights tuned for this model (BeamRule); 0, 0, 0: not tuned
  float discounts[kNgramMaxOrder + 1][3];  // modified Kneser-Ney's D1, D2, D3+ of each order (a record)
  uint64_t tokens, lines;  // the corpus it was counted on
  char note[120];          // what it was built from
  uint8_t reserved[8];
};
static_assert(sizeof(NgramHeader) == 512);

struct NgramEntry {
  uint64_t key;  // ngram_key; 0: an empty slot
  float logp;    // ln P(the last token | the others)
  float bow;     // ln of the back-off weight of the n-gram as a context; 0 where it is none
};
static_assert(sizeof(NgramEntry) == 16);

struct NgramUnigram {
  float logp, bow;
};

// MurmurHash3's finalizer: every input bit moves every output bit.
inline uint64_t ngram_mix(uint64_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}

// The fingerprint of an n-gram: its first n - 1 tokens packed as a state packs a
// context (ctx, len = n - 1) and its last token.  Never 0, the empty slot.
inline uint64_t ngram_key(uint64_t ctx, int len, uint32_t last) {
  const uint64_t k = ngram_mix(ngram_mix(ctx + (static_cast<uint64_t>(len) << 61)) ^
                               (last * 0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL));
  return k ? k : 1;
}

// A key's first slot in a table of `capacity` slots: its top bits.
inline uint64_t ngram_slot(uint64_t key, uint64_t capacity) {
  return capacity > 1 ? key >> (64 - std::countr_zero(capacity)) : 0;
}

// FNV-1a over the vocabulary's strings, each ended by a 0 byte: a model reads the
// recognizer whose classes it was counted on.
uint64_t vocab_hash(const std::vector<std::string>& vocab);

class NgramModel final : public LanguageModel {
 public:
  // Maps the file; throws Error if it is not a model or its tables do not fit it.
  static std::shared_ptr<NgramModel> open(const std::filesystem::path& file);
  ~NgramModel() override;

  uint64_t start() const override;
  float score(uint64_t state, int32_t c, uint64_t& next) const override;
  float end(uint64_t state) const override;
  // Many queries at once, their tables' slots prefetched a round ahead of the probes that read them: a probe that
  // misses the cache waits beside the others, not before them.
  void score_batch(std::span<const uint64_t> states, std::span<const int32_t> classes, std::span<float> out,
                   std::span<uint64_t> next) const override;

  const NgramHeader& header() const { return *h_; }
  int order() const { return static_cast<int>(h_->order); }
  int classes() const { return static_cast<int>(h_->classes); }
  int boundary() const { return static_cast<int>(h_->classes); }
  size_t bytes() const;
  // ln P(tokens, and the line's end where `end`) from a line's start: the sum of score over them.
  double logprob(std::span<const int32_t> tokens, bool end = true) const;
  // Throws Error unless the model was counted on this vocabulary.
  void check_vocab(const std::vector<std::string>& vocab, const std::string& what) const;

  // The state's context: its length and its tokens, oldest first (tests, tools).
  std::vector<int32_t> context(uint64_t state) const;

 private:
  struct Mapping;
  NgramModel() = default;
  const NgramEntry* find(int n, uint64_t key) const;
  const NgramEntry* slot(int n, uint64_t key) const;
  uint64_t pack(uint64_t ctx, int len) const { return (static_cast<uint64_t>(len) << 61) | ctx; }

  std::unique_ptr<Mapping> map_;
  const NgramHeader* h_ = nullptr;
  const NgramUnigram* uni_ = nullptr;
  std::array<const NgramEntry*, kNgramMaxOrder + 1> tab_{};
  std::array<uint64_t, kNgramMaxOrder + 1> mask_{};
  std::array<int, kNgramMaxOrder + 1> shift_{};
  int bits_ = 0, max_ctx_ = 0;
  uint64_t token_mask_ = 0;
};

}  // namespace carbon
