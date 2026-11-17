// The CTC prefix beam (carbon/beam.h) and its candidates (cpu::ctc_topk, cpu::rec_head_logits).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <vector>

#include "carbon/beam.h"
#include "carbon/cpu.h"
#include "carbon/error.h"
#include "carbon/lines.h"

namespace {

using Ids = std::vector<int32_t>;

// Logits [classes][frames], the recognizers' layout, from rows [frames][classes].
std::vector<float> logits_of(const std::vector<std::vector<float>>& rows) {
  const int frames = static_cast<int>(rows.size()), classes = static_cast<int>(rows[0].size());
  std::vector<float> out(static_cast<size_t>(classes) * frames);
  for (int t = 0; t < frames; ++t) {
    for (int c = 0; c < classes; ++c) out[static_cast<size_t>(c) * frames + t] = rows[t][c];
  }
  return out;
}

std::vector<float> random_logits(std::mt19937& rng, int classes, int frames, float spread) {
  std::normal_distribution<float> d(0.0f, spread);
  std::vector<float> out(static_cast<size_t>(classes) * frames);
  for (float& v : out) v = d(rng);
  return out;
}

// Every path's probability summed by the prefix it collapses to (exponential: tiny cases only).
std::map<Ids, double> brute_force(const std::vector<float>& logits, int classes, int frames) {
  std::vector<std::vector<double>> p(frames, std::vector<double>(classes));
  for (int t = 0; t < frames; ++t) {
    double m = -1e300, s = 0;
    for (int c = 0; c < classes; ++c) m = std::max(m, double(logits[static_cast<size_t>(c) * frames + t]));
    for (int c = 0; c < classes; ++c) s += std::exp(double(logits[static_cast<size_t>(c) * frames + t]) - m);
    for (int c = 0; c < classes; ++c) p[t][c] = std::exp(double(logits[static_cast<size_t>(c) * frames + t]) - m) / s;
  }
  std::map<Ids, double> out;
  std::vector<int> path(frames, 0);
  while (true) {
    double q = 1;
    Ids ids;
    int last = 0;
    for (int t = 0; t < frames; ++t) {
      q *= p[t][path[t]];
      if (path[t] != 0 && path[t] != last) ids.push_back(path[t]);
      last = path[t];
    }
    out[ids] += q;
    int t = 0;
    while (t < frames && ++path[t] == classes) path[t++] = 0;
    if (t == frames) break;
  }
  return out;
}

carbon::FrameCandidates all_of(const std::vector<float>& logits, int classes, int frames) {
  return carbon::frame_candidates(logits.data(), classes, frames, frames, classes - 1);
}

carbon::BeamRule rule_of(int width, float blank_skip = 1.0f) {
  carbon::BeamRule r;
  r.width = width;
  r.blank_skip = blank_skip;
  return r;
}

TEST(Beam, CandidatesAreTheLogSoftmaxsBest) {
  std::mt19937 rng(7);
  const int classes = 9, frames = 13, k = 4;
  const std::vector<float> logits = random_logits(rng, classes, frames, 2.0f);
  const carbon::FrameCandidates f = carbon::frame_candidates(logits.data(), classes, frames, 11, k);
  ASSERT_EQ(f.frames, 11);
  ASSERT_EQ(f.k, k);
  for (int t = 0; t < f.frames; ++t) {
    double m = -1e300, s = 0;
    for (int c = 0; c < classes; ++c) m = std::max(m, double(logits[static_cast<size_t>(c) * frames + t]));
    for (int c = 0; c < classes; ++c) s += std::exp(double(logits[static_cast<size_t>(c) * frames + t]) - m);
    std::vector<std::pair<float, int>> rank;
    for (int c = 1; c < classes; ++c) rank.push_back({-logits[static_cast<size_t>(c) * frames + t], c});
    std::sort(rank.begin(), rank.end());
    for (int j = 0; j < k; ++j) {
      EXPECT_EQ(f.ids[t * k + j], rank[j].second) << "frame " << t;
      const double want = -rank[j].first - m - std::log(s);
      EXPECT_NEAR(f.logp[t * k + j], want, 1e-5);
    }
    EXPECT_NEAR(f.blank[t], double(logits[t]) - m - std::log(s), 1e-5);
  }
}

TEST(Beam, CandidatesTieToTheLowerClassAndMatchGreedysProbability) {
  // frame 0: classes 2 and 3 tie at the top; frame 1: the blank's
  const std::vector<float> logits = logits_of({{0.0f, 1.0f, 5.0f, 5.0f}, {4.0f, 1.0f, 2.0f, 2.0f}});
  const carbon::FrameCandidates f = carbon::frame_candidates(logits.data(), 4, 2, 2, 3);
  EXPECT_EQ(f.ids[0], 2);
  EXPECT_EQ(f.ids[1], 3);
  EXPECT_EQ(f.ids[2], 1);
  EXPECT_EQ(f.ids[3], 2);  // the tie of frame 1's 2 and 3
  const int32_t len[] = {2};
  const auto g = carbon::ctc_greedy(logits.data(), 1, 4, 2, len);
  ASSERT_EQ(g[0].ids, Ids({2}));
  EXPECT_NEAR(std::exp(f.logp[0]), g[0].probs[0], 1e-6);
}

TEST(Beam, CandidatesAreTheSameBitsOnEveryIsa) {
  std::mt19937 rng(37);
  for (int t_n : {29, 16, 7, 1}) {
    for (int k : {0, 1, 5, 12}) {
      const int classes = 13, frames = 33;
      std::vector<float> logits = random_logits(rng, classes, frames, 3.0f);
      logits[static_cast<size_t>(4) * frames + 2] = logits[static_cast<size_t>(9) * frames + 2];  // a tie
      std::vector<int32_t> ids_a(static_cast<size_t>(t_n) * k), ids_b(ids_a.size());
      std::vector<float> lp_a(ids_a.size()), lp_b(ids_a.size()), bl_a(t_n), bl_b(t_n);
      carbon::cpu::ctc_topk(logits.data(), classes, frames, t_n, k, ids_a.data(), lp_a.data(), bl_a.data(),
                            carbon::cpu::Isa::kScalar);
      carbon::cpu::ctc_topk(logits.data(), classes, frames, t_n, k, ids_b.data(), lp_b.data(), bl_b.data(),
                            carbon::cpu::best_isa());
      EXPECT_EQ(ids_a, ids_b) << t_n << " " << k;
      EXPECT_EQ(lp_a, lp_b) << t_n << " " << k;
      EXPECT_EQ(bl_a, bl_b) << t_n << " " << k;
    }
  }
}

TEST(Beam, CandidatesRefuseKPastTheClasses) {
  const std::vector<float> logits(4 * 3, 0.0f);
  EXPECT_THROW(carbon::frame_candidates(logits.data(), 4, 3, 3, 4), carbon::Error);
  EXPECT_NO_THROW(carbon::frame_candidates(logits.data(), 4, 3, 3, 0));
}

TEST(Beam, LogprobIsEveryPathsSum) {
  std::mt19937 rng(11);
  for (int trial = 0; trial < 20; ++trial) {
    const int classes = 3 + trial % 2, frames = 4 + trial % 3;
    const std::vector<float> logits = random_logits(rng, classes, frames, 1.5f);
    const auto truth = brute_force(logits, classes, frames);
    const carbon::FrameCandidates f = all_of(logits, classes, frames);
    for (const auto& [ids, p] : truth) {
      EXPECT_NEAR(carbon::ctc_logprob(f, ids), std::log(p), 1e-5) << "trial " << trial;
    }
    EXPECT_EQ(carbon::ctc_logprob(f, Ids(frames + 1, 1)), -std::numeric_limits<double>::infinity());
  }
}

TEST(Beam, AWideEnoughBeamFindsTheMostProbableReading) {
  std::mt19937 rng(13);
  for (int trial = 0; trial < 30; ++trial) {
    const int classes = 3 + trial % 2, frames = 4 + trial % 3;
    const std::vector<float> logits = random_logits(rng, classes, frames, 1.0f + 0.1f * trial);
    const auto truth = brute_force(logits, classes, frames);
    double best = 0;
    for (const auto& kv : truth) best = std::max(best, kv.second);
    const carbon::BeamLine b = carbon::ctc_beam(all_of(logits, classes, frames), rule_of(4096));
    EXPECT_NEAR(b.ctc, std::log(best), 1e-6) << "trial " << trial;
    EXPECT_NEAR(b.ctc, std::log(truth.at(b.line.ids)), 1e-6);
    EXPECT_EQ(b.score, b.ctc);
    EXPECT_EQ(b.lm, 0.0);
  }
}

TEST(Beam, NeverLessProbableThanGreedy) {
  // flat logits, where the argmax path's prefix and the most probable prefix part
  std::mt19937 rng(17);
  int better = 0;
  for (int trial = 0; trial < 300; ++trial) {
    const int classes = 6, frames = 14;
    const std::vector<float> logits = random_logits(rng, classes, frames, 0.6f);
    const carbon::FrameCandidates f = carbon::frame_candidates(logits.data(), classes, frames, frames, 3);
    const carbon::BeamLine g = carbon::ctc_beam(f, rule_of(1));
    for (int width : {2, 3, 5, 8}) {
      const carbon::BeamLine b = carbon::ctc_beam(f, rule_of(width));
      ASSERT_GE(b.ctc, g.ctc - 1e-9) << "trial " << trial << " width " << width;
      better += b.ctc > g.ctc + 1e-9;
    }
  }
  EXPECT_GT(better, 0);  // the beam does find more probable readings
}

TEST(Beam, WidthOneIsGreedy) {
  std::mt19937 rng(19);
  for (int trial = 0; trial < 50; ++trial) {
    const int classes = 7, frames = 20;
    const std::vector<float> logits = random_logits(rng, classes, frames, trial % 2 ? 4.0f : 0.7f);
    const int32_t len[] = {frames};
    const carbon::CtcLine g = carbon::ctc_greedy(logits.data(), 1, classes, frames, len)[0];
    const carbon::BeamLine b = carbon::ctc_beam(all_of(logits, classes, frames), rule_of(1));
    EXPECT_EQ(b.line.ids, g.ids);
    EXPECT_EQ(b.line.frames, g.frames);
    ASSERT_EQ(b.line.probs.size(), g.probs.size());
    for (size_t i = 0; i < g.probs.size(); ++i) EXPECT_NEAR(b.line.probs[i], g.probs[i], 1e-5);
  }
}

TEST(Beam, ARepeatExtendsOnlyAcrossABlank) {
  const float hi = 8.0f;
  // 1 _ 1: two ones; 1 1: one
  const std::vector<float> across = logits_of({{0, hi, 0}, {hi, 0, 0}, {0, hi, 0}});
  const std::vector<float> run = logits_of({{0, hi, 0}, {0, hi, 0}, {hi, 0, 0}});
  for (int width : {1, 4}) {
    EXPECT_EQ(carbon::ctc_beam(all_of(across, 3, 3), rule_of(width)).line.ids, Ids({1, 1})) << width;
    EXPECT_EQ(carbon::ctc_beam(all_of(run, 3, 3), rule_of(width)).line.ids, Ids({1})) << width;
  }
  const carbon::BeamLine b = carbon::ctc_beam(all_of(across, 3, 3), rule_of(4));
  EXPECT_EQ(b.line.frames, Ids({0, 2}));
  const carbon::BeamLine r = carbon::ctc_beam(all_of(run, 3, 3), rule_of(4));
  EXPECT_EQ(r.line.frames, Ids({0}));
}

TEST(Beam, OnClearLinesItReadsAsGreedyDoes) {
  // peaked logits: the beam, its frames skipped where the blank owns them or not, reads greedy's line
  std::mt19937 rng(23);
  std::uniform_int_distribution<int> cls(0, 11);
  for (int trial = 0; trial < 40; ++trial) {
    const int classes = 12, frames = 60;
    std::vector<std::vector<float>> rows(frames, std::vector<float>(classes, 0.0f));
    for (int t = 0; t < frames; ++t) rows[t][t % 3 == 0 ? cls(rng) : 0] = 9.0f;
    const std::vector<float> logits = logits_of(rows);
    const int32_t len[] = {frames};
    const carbon::CtcLine g = carbon::ctc_greedy(logits.data(), 1, classes, frames, len)[0];
    const carbon::FrameCandidates f = carbon::frame_candidates(logits.data(), classes, frames, frames, 5);
    for (float skip : {0.999f, 1.0f}) {
      const carbon::BeamLine b = carbon::ctc_beam(f, rule_of(8, skip));
      EXPECT_EQ(b.line.ids, g.ids) << "trial " << trial << " skip " << skip;
      EXPECT_EQ(b.line.frames, g.frames);
    }
  }
}

TEST(Beam, EmptyAndBlankLines) {
  carbon::FrameCandidates none;
  const carbon::BeamLine e = carbon::ctc_beam(none, rule_of(8));
  EXPECT_TRUE(e.line.ids.empty());
  EXPECT_EQ(e.ctc, 0.0);
  const std::vector<float> blank = logits_of({{9, 0, 0}, {9, 0, 0}, {9, 0, 0}});
  EXPECT_TRUE(carbon::ctc_beam(all_of(blank, 3, 3), rule_of(8, 0.999f)).line.ids.empty());
}

TEST(Beam, RefusesMalformedInputs) {
  carbon::FrameCandidates f;
  f.frames = 2;
  f.k = 1;
  f.ids = {1};
  f.logp = {0.0f};
  f.blank = {0.0f, 0.0f};
  EXPECT_THROW(carbon::ctc_beam(f, rule_of(4)), carbon::Error);
  EXPECT_THROW(carbon::ctc_beam(carbon::FrameCandidates{}, rule_of(0)), carbon::Error);
  EXPECT_THROW(carbon::ctc_beam(carbon::FrameCandidates{}, rule_of(4, 0.0f)), carbon::Error);
}

// A toy model: class `liked` costs nothing, every other class ln 0.01, the line's end ln 0.5; it counts its calls.
class Toy : public carbon::LanguageModel {
 public:
  explicit Toy(int32_t liked) : liked_(liked) {}
  uint64_t start() const override { return 0; }
  float score(uint64_t state, int32_t c, uint64_t& next) const override {
    ++calls;
    next = state * 31 + static_cast<uint64_t>(c);
    return c == liked_ ? 0.0f : std::log(0.01f);
  }
  float end(uint64_t) const override { return std::log(0.5f); }
  mutable int calls = 0;

 private:
  int32_t liked_;
};

TEST(Beam, TheLanguageModelBreaksCloseCalls) {
  // frame 1: classes 1 and 2 nearly tied, 1 a little ahead
  const std::vector<float> logits = logits_of({{6, 0, 0}, {0, 3.1f, 3.0f}, {6, 0, 0}});
  const carbon::FrameCandidates f = all_of(logits, 3, 3);
  carbon::BeamRule r = rule_of(4);
  EXPECT_EQ(carbon::ctc_beam(f, r).line.ids, Ids({1}));
  Toy toy(2);
  EXPECT_EQ(carbon::ctc_beam(f, r, &toy).line.ids, Ids({1}));  // alpha 0: the recognizer's alone
  r.alpha = 0.5f;
  const carbon::BeamLine b = carbon::ctc_beam(f, r, &toy);
  EXPECT_EQ(b.line.ids, Ids({2}));
  EXPECT_NEAR(b.lm, std::log(0.5), 1e-6);
  EXPECT_NEAR(b.score, b.ctc + 0.5 * b.lm, 1e-9);
  EXPECT_NEAR(b.ctc, carbon::ctc_logprob(f, Ids({2})), 1e-12);
}

TEST(Beam, EachPrefixAsksTheModelOnce) {
  std::mt19937 rng(29);
  const int classes = 8, frames = 40;
  const std::vector<float> logits = random_logits(rng, classes, frames, 1.0f);
  const carbon::FrameCandidates f = carbon::frame_candidates(logits.data(), classes, frames, frames, 4);
  Toy toy(3);
  carbon::BeamRule r = rule_of(6);
  r.alpha = 0.3f;
  (void)carbon::ctc_beam(f, r, &toy);
  // at most a new prefix per kept prefix and candidate a frame, plus greedy's
  EXPECT_LE(toy.calls, frames * (r.width + 1) * f.k + frames);
  EXPECT_GT(toy.calls, 0);
}

TEST(Beam, ABonusPerCharacterKeepsCharacters) {
  // frame 1 is a near tie between a character and the blank: beta tips it either way
  const std::vector<float> logits = logits_of({{6, 0, 0}, {3.0f, 2.9f, 0}, {6, 0, 0}});
  const carbon::FrameCandidates f = all_of(logits, 3, 3);
  carbon::BeamRule r = rule_of(4);
  EXPECT_TRUE(carbon::ctc_beam(f, r).line.ids.empty());
  r.beta = 0.5f;
  EXPECT_EQ(carbon::ctc_beam(f, r).line.ids, Ids({1}));
}

TEST(Beam, AlignmentIsTheBestPath) {
  // "12" with 2 first likelier at frame 1 than at frame 2: 2 starts at frame 1
  const std::vector<float> logits = logits_of({{0, 5, 0}, {0, 2, 4}, {1, 0, 3}, {5, 0, 0}});
  const carbon::FrameCandidates f = all_of(logits, 3, 4);
  const carbon::CtcLine a = carbon::ctc_align(f, Ids({1, 2}));
  EXPECT_EQ(a.frames, Ids({0, 1}));
  ASSERT_EQ(a.probs.size(), 2u);
  EXPECT_NEAR(a.probs[1], std::exp(f.logp[1 * 2 + 0]), 1e-6);  // frame 1's best, class 2, over frame 2's
  EXPECT_TRUE(carbon::ctc_align(f, Ids(9, 1)).frames.empty());  // more characters than frames: no path
}

TEST(Beam, HeadLogitsAreTheFusedHeadsSums) {
  std::mt19937 rng(31);
  const int k = 16, classes = 37, frames = 21;
  std::normal_distribution<float> d(0.0f, 1.0f);
  std::vector<float> w(static_cast<size_t>(classes) * k), b(classes), feat(static_cast<size_t>(k) * frames);
  for (float& v : w) v = d(rng);
  for (float& v : b) v = d(rng);
  for (float& v : feat) v = d(rng);
  const carbon::cpu::PackedHead head = carbon::cpu::pack_rec_head(w.data(), b.data(), k, classes);
  for (carbon::cpu::Isa isa : {carbon::cpu::Isa::kScalar, carbon::cpu::best_isa()}) {
    for (int t_n : {frames, 13, 5}) {
      std::vector<int32_t> ids(frames, -1);
      std::vector<float> probs(frames, 0.0f);
      carbon::cpu::ctc_head(feat.data(), k, frames, t_n, head, ids.data(), probs.data(), isa);
      std::vector<float> logits(static_cast<size_t>(classes) * frames, 0.0f);
      carbon::cpu::rec_head_logits(feat.data(), k, frames, t_n, head, logits.data(), isa);
      std::vector<int32_t> arg(frames, -1);
      std::vector<float> p(frames, 0.0f);
      carbon::cpu::ctc_argmax(logits.data(), classes, frames, t_n, arg.data(), p.data(), isa);
      for (int t = 0; t < t_n; ++t) {
        EXPECT_EQ(arg[t], ids[t]) << "frame " << t;
        if (ids[t] != 0) EXPECT_EQ(p[t], probs[t]);
        double want = b[static_cast<size_t>(ids[t])];
        for (int j = 0; j < k; ++j) want += double(w[static_cast<size_t>(ids[t]) * k + j]) * feat[j * frames + t];
        EXPECT_NEAR(logits[static_cast<size_t>(ids[t]) * frames + t], want, 1e-4);
      }
    }
  }
}

}  // namespace
