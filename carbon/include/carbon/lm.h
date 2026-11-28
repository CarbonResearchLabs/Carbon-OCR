// The n-gram language model the beam reads unsure lines with (carbon/beam.h): a
// character n-gram over a recognizer's own classes, in a file mapped into memory.
//
// Tokens are the recognizer's classes 1 .. classes - 1 (0, the CTC blank, never
// occurs in text) and the boundary, `classes`: the context at a line's start and
// the token that ends it.  A model of order N reads at most N - 1 tokens back.
//
// The file (lm-<group>.bin), little-endian, every section on 64 bytes.  Version 2, the
// compact file:
//   NgramHeader                    512 bytes
//   codebooks [9][2][256] float    at `codebooks`: order n's 256 log-probabilities
//                                  ([n][0]) and 256 back-off weights ([n][1]), natural
//                                  logs; a back-off code of 0 is exactly 0
//   unigrams  [classes + 1]        {logp, bow}: token t's probability and its
//                                  back-off weight as a context, natural logs, FP32
//   order n's table, n = 2 .. N    an open-addressing hash table of `capacity` slots
//                                  (any number, at most `load` full), each 6 bytes
//                                  (order N's 5): the key's tag (ngram_tag, 0 an
//                                  empty slot), its log-probability's code and its
//                                  back-off weight's code (order N has none); slot
//                                  ngram_home(key, capacity), then the next, round;
//                                  8 bytes of zeros after it (a slot is read as 8)
// A key is a 64-bit fingerprint of its tokens (ngram_key); a slot keeps its low 32 bits
// and the slot's place keeps its high ones, so a query that misses meets another
// n-gram's tag about once in 2^32 slots it reads (an n-gram whose tag another meets on
// its probe is left out of the file).  Values in
// 8-bit codebooks (KenLM's -q 8 -b 8: equal-count bins, then 1-D k-means) cost
// lm-latin's 7-gram 0.001 nats a token on held-out text; the file is a seventh of
// version 1's.
//
// Version 1 (still read): the unigrams after the header, then each order's table of
// NgramEntry (a 64-bit key, FP32 logp and bow), `capacity` a power of 2 slots, at most
// half full, key 0 empty: slot ngram_slot(key, capacity), then the next.
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
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
#include <intrin.h>
#endif

#include "carbon/beam.h"

namespace carbon {

constexpr char kNgramMagic[8] = {'C', 'R', 'B', 'N', 'G', 'R', 'M', '1'};
constexpr int kNgramMaxOrder = 8;
constexpr int kNgramCodes = 256;  // a version 2 codebook's entries

struct NgramTable {
  uint64_t offset = 0;    // from the file's start
  uint64_t capacity = 0;  // slots (version 1: a power of 2); 0 for the unigrams, which are dense
  uint64_t count = 0;     // entries
};

struct NgramHeader {
  char magic[8];
  uint32_t version;    // 2, the compact file; 1
  uint32_t order;      // N
  uint32_t classes;    // the recognizer's, the blank among them; the boundary token is `classes`
  uint32_t bits;       // bits a token takes in a packed context: bit_width(classes)
  uint64_t vocab_hash; // vocab_hash() of the recognizer's vocabulary
  NgramTable tables[kNgramMaxOrder + 1];  // [n] order n's; [1] the unigrams
  float alpha, beta, gate;  // the beam's weights tuned for this model (BeamRule); 0, 0, 0: not tuned
  float discounts[kNgramMaxOrder + 1][3];  // modified Kneser-Ney's D1, D2, D3+ of each order (a record)
  uint64_t tokens, lines;  // the corpus it was counted on
  char note[120];          // what it was built from
  uint64_t codebooks;      // version 2: the codebooks' offset; version 1: 0
};
static_assert(sizeof(NgramHeader) == 512);

// A version 2 slot's bytes: order N's (no back-off weight), the others'.
constexpr uint32_t kNgramSlotTop = 5, kNgramSlot = 6;

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

// A key's first slot in a version 1 table of `capacity` (a power of 2) slots: its top bits.
inline uint64_t ngram_slot(uint64_t key, uint64_t capacity) {
  return capacity > 1 ? key >> (64 - std::countr_zero(capacity)) : 0;
}

// The high 64 bits of x * y, in 32-bit halves (where the compiler has no 128-bit product).
inline uint64_t mulhi_portable(uint64_t x, uint64_t y) {
  const uint64_t a = x >> 32, b = x & 0xffffffffu, c = y >> 32, d = y & 0xffffffffu;
  const uint64_t bd = b * d, ad = a * d, bc = b * c, ac = a * c;
  const uint64_t mid = (bd >> 32) + (ad & 0xffffffffu) + (bc & 0xffffffffu);
  return ac + (ad >> 32) + (bc >> 32) + (mid >> 32);
}

// A key's first slot in a version 2 table of any number of slots: the high half of key x capacity (its top bits,
// scaled).
inline uint64_t ngram_home(uint64_t key, uint64_t capacity) {
#if defined(__SIZEOF_INT128__)
  return static_cast<uint64_t>((static_cast<unsigned __int128>(key) * capacity) >> 64);
#elif defined(_MSC_VER) && (defined(_M_X64) || defined(_M_ARM64))
  return __umulh(key, capacity);
#else
  return mulhi_portable(key, capacity);
#endif
}

// A key's tag in a version 2 slot: its low 32 bits, never 0 (the empty slot).
inline uint32_t ngram_tag(uint64_t key) {
  const uint32_t t = static_cast<uint32_t>(key);
  return t ? t : 1u;
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
  // A version 2 file's bound from its values alone: the largest log-probability any order holds, plus the largest
  // back-off weight of each order a query can back off through where it is positive (a query adds each order's at
  // most once), plus a margin for float sums. +inf for version 1 (its values are in its tables, not codebooks).
  float max_score() const override { return max_score_; }
  // Many queries at once, their tables' slots prefetched a round ahead of the probes that read them: a probe that
  // misses the cache waits beside the others, not before them.
  void score_batch(std::span<const uint64_t> states, std::span<const int32_t> classes, std::span<float> out,
                   std::span<uint64_t> next) const override;

  const NgramHeader& header() const { return *h_; }
  int version() const { return static_cast<int>(h_->version); }
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
  // order n's entry for key: its logp and bow; false where the table has none
  bool lookup(int n, uint64_t key, float& logp, float& bow) const;
  const void* slot(int n, uint64_t key) const;  // where key's probe starts (to prefetch)
  uint64_t pack(uint64_t ctx, int len) const { return (static_cast<uint64_t>(len) << 61) | ctx; }

  std::unique_ptr<Mapping> map_;
  const NgramHeader* h_ = nullptr;
  const NgramUnigram* uni_ = nullptr;
  int version_ = 0;
  std::array<const uint8_t*, kNgramMaxOrder + 1> tab_{};  // each order's table
  std::array<uint64_t, kNgramMaxOrder + 1> cap_{};        // its slots
  std::array<uint64_t, kNgramMaxOrder + 1> mask_{};       // version 1: slots - 1
  std::array<int, kNgramMaxOrder + 1> shift_{};           // version 1: 64 - log2(slots)
  std::array<uint32_t, kNgramMaxOrder + 1> stride_{};     // version 2: a slot's bytes
  const float* codes_ = nullptr;                          // version 2: [n][0 logp, 1 bow][256]
  int bits_ = 0, max_ctx_ = 0;
  uint64_t token_mask_ = 0;
  float max_score_ = std::numeric_limits<float>::infinity();
};

}  // namespace carbon
