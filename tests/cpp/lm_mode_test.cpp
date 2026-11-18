// The language models as one switch (carbon::EngineOptions::lm_mode): each mode's beam, models and tuned weights.
// The n-gram is tests/data/lm-mode-ngram.bin, built ahead: order 3 over the vocabulary below, its header's alpha
// 0.3, beta 2.0 and gate 1.5. The transformer is a random one (random_tlm.h). And the n-gram file's two versions:
// tests/data/lm-compact-v1.bin and lm-compact-v2.bin, one toy 4-gram over the same vocabulary written as each.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <random>
#include <string>
#include <vector>

#include "carbon/beam.h"
#include "carbon/engine.h"
#include "carbon/error.h"
#include "carbon/lm.h"
#include "carbon/tlm.h"
#include "../../carbon/src/engine_steps.h"
#include "random_tlm.h"

namespace {

namespace fs = std::filesystem;

fs::path temp(const std::string& name) { return fs::temp_directory_path() / ("carbon-lm-mode-test-" + name); }

// a model file's tuned weights, written into its header
void put_floats(const fs::path& file, size_t offset, std::initializer_list<float> v) {
  std::fstream f(file, std::ios::binary | std::ios::in | std::ios::out);
  f.seekp(static_cast<std::streamoff>(offset));
  for (const float x : v) f.write(reinterpret_cast<const char*>(&x), sizeof(x));
}

// Each mode turns the beam on (or off) and opens the set's models it names, at the weights and gate tuned into
// their files; what is set by hand still wins; "auto" is the engine as it was; a model a mode needs is required.
TEST(LmMode, EachModeOpensItsModelsAtTheirTunedWeights) {
  using carbon::LmMode;
  const std::vector<std::string> vocab{"", "a", "b", " ", "c", "d", "e", "f"};
  const int classes = static_cast<int>(vocab.size());
  const uint64_t vh = carbon::vocab_hash(vocab);
  const fs::path ngram = CARBON_TEST_DATA "/lm-mode-ngram.bin";
  {  // the file is the one this test is written for
    const auto m = carbon::NgramModel::open(ngram);
    ASSERT_EQ(m->order(), 3);
    ASSERT_EQ(m->classes(), classes);
    ASSERT_EQ(m->header().vocab_hash, vh);
    ASSERT_FLOAT_EQ(m->header().alpha, 0.3f);
    ASSERT_FLOAT_EQ(m->header().beta, 2.0f);
    ASSERT_FLOAT_EQ(m->header().gate, 1.5f);
  }
  const fs::path set = temp("set"), half = temp("half");
  fs::remove_all(set), fs::remove_all(half);
  fs::create_directories(set), fs::create_directories(half);
  fs::copy_file(ngram, set / "lm-latin.bin");
  fs::copy_file(ngram, half / "lm-latin.bin");
  fs::copy_file(carbon::random_model("lm-mode", 32, 1, 1, 64, 16, classes, 1u, vh), set / "tlm-latin.bin");
  put_floats(set / "tlm-latin.bin", offsetof(carbon::TlmHeader, alpha), {0.1f});
  put_floats(set / "tlm-latin.bin", offsetof(carbon::TlmHeader, gate), {0.97f});
  auto attach = [&](LmMode m, const fs::path& dir, auto&& tweak) {
    carbon::EngineOptions o;
    o.lm_mode = m;
    tweak(o);
    carbon::detail::attach_language_model(o, dir, vocab, -1);
    return o;
  };
  const auto as_is = [](carbon::EngineOptions&) {};
  const float gate0 = carbon::BeamRule{}.gate;

  auto o = attach(LmMode::kAuto, set, as_is);  // as before the switch: the beam off, so no model
  EXPECT_FALSE(o.beam.on());
  EXPECT_FALSE(o.language_model);
  EXPECT_FALSE(o.rescorer);
  o = attach(LmMode::kAuto, set, [](carbon::EngineOptions& e) { e.beam.width = 8; });  // by hand: the set's n-gram
  EXPECT_TRUE(o.language_model);
  EXPECT_FALSE(o.rescorer);  // the transformer only where a mode or a file names it
  EXPECT_FLOAT_EQ(o.beam.gate, gate0);  // the gate as set, not the files'
  EXPECT_FLOAT_EQ(o.beam.alpha, 0.3f);
  o = attach(LmMode::kAuto, set, [&](carbon::EngineOptions& e) {
    e.beam.width = 8;
    e.tlm = set / "tlm-latin.bin";
  });
  EXPECT_TRUE(o.language_model && o.rescorer);

  o = attach(LmMode::kNone, set, [](carbon::EngineOptions& e) { e.beam.width = 8; });
  EXPECT_FALSE(o.beam.on());
  EXPECT_FALSE(o.language_model);
  EXPECT_FALSE(o.rescorer);

  o = attach(LmMode::kNgram, set, as_is);
  EXPECT_EQ(o.beam.width, carbon::kLmBeamWidth);
  EXPECT_TRUE(o.language_model);
  EXPECT_FALSE(o.rescorer);
  EXPECT_FLOAT_EQ(o.beam.alpha, 0.3f);
  EXPECT_FLOAT_EQ(o.beam.beta, 2.0f);
  EXPECT_FLOAT_EQ(o.beam.gate, 1.5f);

  o = attach(LmMode::kTransformer, set, as_is);
  EXPECT_EQ(o.beam.width, carbon::kLmBeamWidth);
  EXPECT_FALSE(o.language_model);
  EXPECT_TRUE(o.rescorer);
  EXPECT_FLOAT_EQ(o.beam.rescore, 0.1f);
  EXPECT_FLOAT_EQ(o.beam.gate, 0.97f);

  o = attach(LmMode::kBoth, set, as_is);
  EXPECT_TRUE(o.language_model && o.rescorer);
  EXPECT_FLOAT_EQ(o.beam.gate, 1.5f);  // the n-gram's, which drives the beam
  EXPECT_FLOAT_EQ(o.beam.alpha, 0.3f);
  EXPECT_FLOAT_EQ(o.beam.rescore, 0.1f);

  o = attach(LmMode::kBoth, set, [](carbon::EngineOptions& e) {  // what is set by hand wins
    e.beam.width = 4;
    e.lm_gate = 0.5f;
    e.lm_alpha = 0.7f;
    e.tlm_alpha = 0.4f;
  });
  EXPECT_EQ(o.beam.width, 4);
  EXPECT_FLOAT_EQ(o.beam.gate, 0.5f);
  EXPECT_FLOAT_EQ(o.beam.alpha, 0.7f);
  EXPECT_FLOAT_EQ(o.beam.rescore, 0.4f);

  // a set with the n-gram alone: the modes that need the transformer refused, unless it is given
  EXPECT_NO_THROW(attach(LmMode::kNgram, half, as_is));
  EXPECT_THROW(attach(LmMode::kTransformer, half, as_is), carbon::Error);
  EXPECT_THROW(attach(LmMode::kBoth, half, as_is), carbon::Error);
  o = attach(LmMode::kBoth, half, [&](carbon::EngineOptions& e) { e.tlm = set / "tlm-latin.bin"; });
  EXPECT_TRUE(o.language_model && o.rescorer);
  EXPECT_THROW(attach(LmMode::kNgram, half, [](carbon::EngineOptions& e) { e.lm = "off"; }), carbon::Error);

  for (const char* name : {"auto", "none", "ngram", "transformer", "both"}) {
    ASSERT_TRUE(carbon::parse_lm_mode(name));
    EXPECT_EQ(carbon::lm_mode_name(*carbon::parse_lm_mode(name)), name);
  }
  EXPECT_EQ(carbon::parse_lm_mode("off"), LmMode::kNone);
  EXPECT_EQ(carbon::parse_lm_mode("tlm"), LmMode::kTransformer);
  EXPECT_FALSE(carbon::parse_lm_mode("trigram"));
}

// The compact file (version 2: 32-bit tags, 8-bit codebooks) reads as the full one (version 1: 64-bit keys, FP32)
// it was written beside, from the same counts: the same states after every token, scores within the codebooks'
// rounding.
TEST(NgramFile, TheCompactFileReadsAsTheFullOne) {
  const auto m1 = carbon::NgramModel::open(CARBON_TEST_DATA "/lm-compact-v1.bin");
  const auto m2 = carbon::NgramModel::open(CARBON_TEST_DATA "/lm-compact-v2.bin");
  ASSERT_EQ(m1->version(), 1);
  ASSERT_EQ(m2->version(), 2);
  ASSERT_EQ(m2->order(), 4);
  ASSERT_EQ(m1->order(), m2->order());
  ASSERT_EQ(m1->classes(), m2->classes());
  ASSERT_EQ(m1->header().vocab_hash, m2->header().vocab_hash);
  EXPECT_FLOAT_EQ(m2->header().alpha, 0.3f);
  EXPECT_FLOAT_EQ(m2->header().gate, 1.5f);
  for (int n = 2; n <= m2->order(); ++n) {
    EXPECT_EQ(m1->header().tables[n].count, m2->header().tables[n].count) << n;
    EXPECT_LE(m2->header().tables[n].count * 4, m2->header().tables[n].capacity * 3 + 4) << n;  // at most 0.75 full
  }
  std::mt19937 rng(3);
  std::uniform_int_distribution<int> pick(0, m1->classes() - 1);
  uint64_t a = m1->start(), b = m2->start(), na = 0, nb = 0;
  double diff = 0, worst = 0;
  const int tokens = 20000;
  for (int i = 0; i < tokens; ++i) {
    const int t = pick(rng);  // 0, the line's end
    const int32_t c = t == 0 ? m1->boundary() : t;
    const double d = std::abs(m1->score(a, c, na) - m2->score(b, c, nb));
    ASSERT_EQ(na, nb) << i;  // the same contexts kept: the same states
    diff += d;
    worst = std::max(worst, d);
    a = t == 0 ? m1->start() : na;
    b = t == 0 ? m2->start() : nb;
  }
  EXPECT_LT(diff / tokens, 0.01);
  EXPECT_LT(worst, 0.3);
  // a key's home slot: the high half of key x capacity, whichever way the compiler makes it
  std::mt19937_64 r64(5);
  for (int i = 0; i < 10000; ++i) {
    const uint64_t x = r64(), y = i % 2 ? r64() : r64() >> 20;
    ASSERT_EQ(carbon::ngram_home(x, y), carbon::mulhi_portable(x, y));
    if (y) ASSERT_LT(carbon::ngram_home(x, y), y);
  }
}

}  // namespace
