// The CTC prefix beam, and a language model's seam into it.
//
// The engine decodes every line greedily (carbon/lines.h).  With a BeamRule on,
// a line whose confidence (read_line's: the mean probability of its characters)
// is under the rule's gate is read again: its frames' likeliest classes
// (FrameCandidates) go to a prefix beam that scores each reading by the
// recognizer's probability of it, plus alpha times a language model's, plus beta
// a character.  A line over the gate keeps its greedy decode and costs nothing
// more.  Without a language model the beam is the recognizer's own: a reading
// whose paths sum to more than the greedy path's prefix.
//
// The search has two guards:
// - the greedy prefix so far is carried at every frame as a hypothesis of its
//   own, besides the width's best, so it is never pruned;
// - the reading returned is the survivor with the best exact score: the CTC
//   forward sum over the frames' candidates (ctc_logprob), not the beam's
//   running score, which pruning only bounds from below.  Without a language
//   model the answer is therefore never less probable than greedy's.
// A repeated class extends a prefix only across a blank (1 _ 1 is "11", 1 1 is
// "1").  Width 1 is greedy: the argmax path, collapsed.
//
// The search keeps its working memory on the thread from line to line, and asks a
// language model that bounds its scores (LanguageModel::max_score) only for the
// prefixes that could be kept; neither changes a bit of what it reads.
//
// What the beam cannot do: write a class that is not among a frame's candidates
// (the top `topk` classes and the blank).  It chooses among what the pixels
// allow.  The exact scores are over the candidates too, so the classes past
// them count as impossible.
#pragma once

#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "carbon/lines.h"

namespace carbon {

// A line's frames as the beam reads them: at each of `frames` frames, its `k`
// likeliest classes but the blank, best first (ties to the lower id), with their
// log-probabilities (natural log, a log-softmax over every class), and the
// blank's.  A class outside a frame's k reads as impossible there.
struct FrameCandidates {
  int frames = 0, k = 0;
  std::vector<int32_t> ids;  // [frames][k]
  std::vector<float> logp;   // [frames][k]
  std::vector<float> blank;  // [frames]
};

// One line's candidates from its logits [classes][frames] (the recognizers'
// layout, class 0 the blank), its first t_n frames, k at most classes - 1.  The
// softmax is ctc_argmax's (cpu::ctc_topk), so exp of the best class's
// log-probability is the probability greedy decoding gives it.
FrameCandidates frame_candidates(const float* logits, int classes, int frames, int t_n, int k);

struct BeamRule {
  int width = 0;              // prefixes kept besides greedy's own; under 2: off, every line greedy
  int topk = 5;               // the classes a frame offers besides the blank
  float gate = 0.9f;          // a line is read again where its confidence (read_line's) is under this
  float blank_skip = 0.999f;  // a frame whose blank has at least this probability starts no character; >= 1: none
  float alpha = 0.0f;         // the language model's weight
  float beta = 0.0f;          // a bonus a character (it offsets the language model's cost of each)
  float rescore = 0.0f;       // the rescorer's weight (EngineOptions::rescorer) on its ln P of each of the n best
  // Span-level decoding: a frame whose likeliest class (a character, or the blank) has a probability of at least
  // `span`, with no frame under it within span_margin frames, is read as greedy reads it (sure_spans: that class
  // alone). The beam then searches the unsure spans and their margins alone, and the sure characters keep their
  // greedy reading. Off (every frame's candidates) at 0, the default, or over 1.
  float span = 0.0f;
  int span_margin = 2;
  bool on() const { return width >= 2; }
  bool spans() const { return span > 0.0f && span <= 1.0f; }
};

// The candidates with every sure frame cut to its likeliest class (BeamRule::span): a frame whose best (its first
// candidate or the blank, the likelier; ties to the blank, as greedy reads it) has a probability of at least `span`,
// and no frame within `margin` of it less, keeps that one: a character's frame its character (its blank and other
// candidates -inf), a blank's frame its blank (its candidates -inf). Every other frame as it was.
FrameCandidates sure_spans(const FrameCandidates& f, float span, int margin);

// The beam's view of a language model: states are the model's own (an n-gram's
// last classes packed, say).  Each prefix asks it once, when the prefix is
// first made.  It must be safe to call from several threads at once.
class LanguageModel {
 public:
  virtual ~LanguageModel() = default;
  // The state at a line's start.
  virtual uint64_t start() const = 0;
  // ln P(c | state), and the state after c in `next`.  c is a recognizer class
  // (the space among them), never the blank.
  virtual float score(uint64_t state, int32_t c, uint64_t& next) const = 0;
  // ln P(the line ends | state).
  virtual float end(uint64_t state) const {
    (void)state;
    return 0.0f;
  }
  // An upper bound on score() at every state and class, or +inf where the model knows none. With a bound (and a
  // positive alpha) the beam asks only for the prefixes it could keep: a frame's new prefix whose score at the bound
  // is under the width's best exact scores is pruned unasked, so the beam reads every line as it would asking for
  // them all, with fewer queries.
  virtual float max_score() const { return std::numeric_limits<float>::infinity(); }
  // Many queries at once: out[i] = score(states[i], classes[i], next[i]). The beam asks a frame's new prefixes
  // together, so a model may overlap their memory reads (NgramModel prefetches).
  virtual void score_batch(std::span<const uint64_t> states, std::span<const int32_t> classes, std::span<float> out,
                           std::span<uint64_t> next) const {
    for (size_t i = 0; i < states.size(); ++i) out[i] = score(states[i], classes[i], next[i]);
  }
};

// A line the beam read.
struct BeamLine {
  CtcLine line;     // as ctc_greedy gives it: the ids, the frame each starts at in the best path, the best
                    // probability over its run there
  double ctc = 0;   // ln P(ids | frames), over the candidates (ctc_logprob)
  double lm = 0;    // the language model's ln P(ids, end); 0 without one
  double score = 0; // ctc + alpha lm + beta characters: what the beam chose it by
};

// Throws Error for a malformed FrameCandidates.
BeamLine ctc_beam(const FrameCandidates& f, const BeamRule& rule, const LanguageModel* lm = nullptr);
// The beam's n best survivors by their exact scores, best first (at most the width and greedy's): what a second
// model rescores (carbon/tlm.h). Width 1 gives one.
std::vector<BeamLine> ctc_beam_nbest(const FrameCandidates& f, const BeamRule& rule, const LanguageModel* lm, int n);

// ln P(ids | frames) over the candidates: the CTC forward algorithm over the
// extended sequence (blank, c1, blank, c2, ..., blank); -inf where no path through
// the candidates reads ids.
double ctc_logprob(const FrameCandidates& f, std::span<const int32_t> ids);

// ids aligned to the frames by the CTC path through the candidates that is most
// likely (Viterbi): the frame each starts at and its best probability over its
// run, as ctc_collapse gives them.  Empty frames and probabilities where no path
// reads ids.
CtcLine ctc_align(const FrameCandidates& f, std::span<const int32_t> ids);

}  // namespace carbon
