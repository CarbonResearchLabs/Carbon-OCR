// The language models as one switch (carbon::EngineOptions::lm_mode): each mode's beam, models and tuned weights.
// The n-gram is tests/data/lm-mode-ngram.bin, built ahead: order 3 over the vocabulary below, its header's alpha
// 0.3, beta 2.0 and gate 1.5. The transformer is a random one (random_tlm.h).
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
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
  o = attach(LmMode::kAuto, set, [](carbon::EngineOptions& e) { e.beam.width = 8; });  // by hand: the set's models
  EXPECT_TRUE(o.language_model && o.rescorer);
  EXPECT_FLOAT_EQ(o.beam.gate, gate0);  // the gate as set, not the files'
  EXPECT_FLOAT_EQ(o.beam.alpha, 0.3f);

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

}  // namespace
