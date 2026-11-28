// The CTC prefix beam (carbon/beam.h), with a language model's score on each prefix.
#include "carbon/beam.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <string>
#include <utility>

#include "carbon/cpu.h"
#include "carbon/error.h"

namespace carbon {

namespace {

constexpr double kNeg = -std::numeric_limits<double>::infinity();

// ln(exp(a) + exp(b)), exact at the infinities.
double lae(double a, double b) {
  if (a == kNeg) return b;
  if (b == kNeg) return a;
  if (a < b) std::swap(a, b);
  return a + std::log1p(std::exp(b - a));
}

void check(const FrameCandidates& f, const char* what) {
  if (f.frames < 0 || f.k < 0) throw Error(std::string(what) + ": negative frames or k");
  const size_t n = static_cast<size_t>(f.frames) * static_cast<size_t>(f.k);
  if (f.ids.size() != n || f.logp.size() != n || f.blank.size() != static_cast<size_t>(f.frames)) {
    throw Error(std::string(what) + ": candidates of " + std::to_string(f.frames) + " frames and k " +
                std::to_string(f.k) + " hold " + std::to_string(f.ids.size()) + " ids, " +
                std::to_string(f.logp.size()) + " log-probabilities and " + std::to_string(f.blank.size()) +
                " blanks");
  }
}

// Each class's log-probability and probability at every frame: a row a class, -inf (0) where a frame does not
// offer it, made the first time a reading asks for the class (most classes the candidates offer are in no
// reading scored), so the exact scores and the alignment read a class at a frame in O(1).
class Emissions {
 public:
  explicit Emissions(const FrameCandidates& f) : f_(f) {
    int32_t top = 0;
    for (int32_t c : f.ids) top = std::max(top, c);
    row_.assign(static_cast<size_t>(top) + 1, -1);
    blank_p_.resize(static_cast<size_t>(f.frames));
    for (int t = 0; t < f.frames; ++t) blank_p_[static_cast<size_t>(t)] = std::exp(static_cast<double>(f.blank[t]));
  }
  // A class's row of log-probabilities (probabilities) over the frames; null for one no frame offers (the blank:
  // blank()). The rows stay where they are while others are made.
  const double* of(int32_t c) const {
    const int32_t r = row(c);
    return r < 0 ? nullptr : lp_[static_cast<size_t>(r)].data();
  }
  const double* p_of(int32_t c) const {
    const int32_t r = row(c);
    return r < 0 ? nullptr : p_[static_cast<size_t>(r)].data();
  }
  double blank(int t) const { return f_.blank[static_cast<size_t>(t)]; }
  double blank_p(int t) const { return blank_p_[static_cast<size_t>(t)]; }
  const double* blank_p_row() const { return blank_p_.data(); }
  int frames() const { return f_.frames; }

 private:
  int32_t row(int32_t c) const {
    if (c <= 0 || static_cast<size_t>(c) >= row_.size()) return -1;
    int32_t& r = row_[static_cast<size_t>(c)];
    if (r == -1) {
      std::vector<double> lp(static_cast<size_t>(f_.frames), kNeg), p(static_cast<size_t>(f_.frames), 0.0);
      bool any = false;
      for (int t = 0; t < f_.frames; ++t) {
        const size_t i0 = static_cast<size_t>(t) * f_.k;
        for (int j = 0; j < f_.k; ++j) {
          if (f_.ids[i0 + j] != c) continue;
          lp[static_cast<size_t>(t)] = f_.logp[i0 + j];
          p[static_cast<size_t>(t)] = std::exp(static_cast<double>(f_.logp[i0 + j]));
          any = true;
          break;
        }
      }
      if (!any) {
        r = -2;  // offered by no frame
      } else {
        r = static_cast<int32_t>(lp_.size());
        lp_.push_back(std::move(lp));
        p_.push_back(std::move(p));
      }
    }
    return r;
  }
  const FrameCandidates& f_;
  mutable std::vector<int32_t> row_;  // -1: not asked yet; -2: offered by no frame
  mutable std::vector<std::vector<double>> lp_, p_;
  std::vector<double> blank_p_;
};

// The extended sequence's emissions at frame t: state s even is the blank, odd the class ids[s / 2].
struct Extended {
  const Emissions& em;
  std::span<const int32_t> ids;
  std::vector<const double*> rows, p_rows;  // a character's rows, null where no frame offers it
  Extended(const Emissions& e, std::span<const int32_t> i) : em(e), ids(i), rows(i.size()), p_rows(i.size()) {
    for (size_t j = 0; j < i.size(); ++j) {
      rows[j] = e.of(i[j]);
      p_rows[j] = e.p_of(i[j]);
    }
  }
  size_t states() const { return 2 * ids.size() + 1; }
  double at(int t, size_t s) const {
    if (s % 2 == 0) return em.blank(t);
    const double* r = rows[s / 2];
    return r ? r[t] : kNeg;
  }
  double p_at(int t, size_t s) const {
    if (s % 2 == 0) return em.blank_p(t);
    const double* r = p_rows[s / 2];
    return r ? r[t] : 0.0;
  }
  // Whether the move from s - 2 to s is allowed: s a class, not the class two states back (a blank between
  // two equal classes is what lets both appear).
  bool skip_allowed(size_t s) const { return s >= 2 && s % 2 == 1 && ids[s / 2] != ids[s / 2 - 1]; }
};

// The CTC forward sum in probabilities, scaled to 1 at every frame (the scales' logarithms summed), over the
// band of states a path can be in: at frame t, at most state 2t + 1 from the start, and at least the state from
// which the frames left can still reach the end (states below it read no path that ends in ids). Each state's
// row of probabilities and its move from two states back are looked up once a state, not once a state and frame
// (a class no frame offers reads a row of zeros): the same products and sums, in the same order.
double logprob(const Emissions& em, std::span<const int32_t> ids) {
  const int frames = em.frames();
  if (frames == 0) return ids.empty() ? 0.0 : kNeg;
  const size_t states = 2 * ids.size() + 1;
  if (states > 2 * static_cast<size_t>(frames) + 1) return kNeg;  // more characters than the frames can hold
  thread_local std::vector<double> a, zeros;
  thread_local std::vector<const double*> p;
  thread_local std::vector<uint8_t> skip;
  if (zeros.size() < static_cast<size_t>(frames)) zeros.assign(static_cast<size_t>(frames), 0.0);
  p.resize(states);
  skip.resize(states);
  for (size_t s = 0; s < states; ++s) {
    if (s % 2 == 0) {
      p[s] = em.blank_p_row();
      skip[s] = 0;
    } else {
      const double* r = em.p_of(ids[s / 2]);
      p[s] = r ? r : zeros.data();
      skip[s] = s >= 2 && ids[s / 2] != ids[s / 2 - 1];
    }
  }
  a.assign(states, 0.0);
  a[0] = p[0][0];
  if (states > 1) a[1] = p[1][0];
  double scale = a[0] + (states > 1 ? a[1] : 0.0);
  if (!(scale > 0.0)) return kNeg;
  double logs = std::log(scale);
  a[0] /= scale;
  if (states > 1) a[1] /= scale;
  for (int t = 1; t < frames; ++t) {
    const size_t hi = std::min(states - 1, static_cast<size_t>(2 * t + 1));
    const std::ptrdiff_t low = static_cast<std::ptrdiff_t>(states) - 2 * static_cast<std::ptrdiff_t>(frames - t);
    const size_t lo = low > 0 ? static_cast<size_t>(low) : 0;
    double sum = 0.0;
    for (size_t s = hi + 1; s-- > lo;) {  // downwards, in place: s reads s, s - 1 and s - 2 of the frame before
      double v = a[s];
      if (s >= 1) v += a[s - 1];
      if (skip[s]) v += a[s - 2];
      v *= p[s][t];
      a[s] = v;
      sum += v;
    }
    if (!(sum > 0.0)) return kNeg;
    logs += std::log(sum);
    const double inv = 1.0 / sum;
    for (size_t s = lo; s <= hi; ++s) a[s] *= inv;
  }
  const double end = states == 1 ? a[0] : a[states - 1] + a[states - 2];
  return end > 0.0 ? logs + std::log(end) : kNeg;
}

CtcLine align(const Emissions& em, std::span<const int32_t> ids) {
  CtcLine out;
  out.ids.assign(ids.begin(), ids.end());
  const int frames = em.frames();
  if (ids.empty() || frames == 0) return out;
  const Extended x(em, ids);
  const size_t states = x.states();
  if (states > 2 * static_cast<size_t>(frames) + 1) return out;
  std::vector<double> a(states, kNeg), b(states, kNeg);
  std::vector<uint8_t> back(static_cast<size_t>(frames) * states, 0);  // 0: stayed, 1: from s - 1, 2: from s - 2
  a[0] = x.at(0, 0);
  a[1] = x.at(0, 1);
  for (int t = 1; t < frames; ++t) {
    uint8_t* bt = back.data() + static_cast<size_t>(t) * states;
    for (size_t s = 0; s < states; ++s) {
      double v = a[s];
      uint8_t from = 0;
      if (s >= 1 && a[s - 1] > v) {
        v = a[s - 1];
        from = 1;
      }
      if (x.skip_allowed(s) && a[s - 2] > v) {
        v = a[s - 2];
        from = 2;
      }
      b[s] = v == kNeg ? kNeg : v + x.at(t, s);
      bt[s] = from;
    }
    std::swap(a, b);
  }
  size_t s = states - 1;
  if (states > 1 && a[states - 2] > a[states - 1]) s = states - 2;
  if (a[s] == kNeg) return out;  // no path through the candidates reads ids
  out.frames.assign(ids.size(), -1);
  out.probs.assign(ids.size(), 0.0f);
  for (int t = frames - 1; t >= 0; --t) {
    if (s % 2 == 1) {
      const size_t c = s / 2;
      out.frames[c] = t;  // the walk runs backwards: the run's first frame is the last written
      out.probs[c] = std::max(out.probs[c], static_cast<float>(std::exp(x.at(t, s))));
    }
    if (t > 0) s -= back[static_cast<size_t>(t) * states + s];
  }
  return out;
}

// The prefixes a line's beam has made, a tree: each its parent and last class, made once.
struct Node {
  int32_t parent, cls, depth;
  uint64_t state;  // the language model's after the prefix (once scored)
  double lm;       // its ln P_lm(prefix) (once scored; its parent's until then)
};

struct Entry {
  int32_t node;
  double pb, pnb;  // ln P(the frames so far read the prefix, its path ending in a blank / in its last class)
};

// A line's working memory, a thread's own and kept from line to line (most lines are read on the threads that read
// the ones before them), so a line allocates little; a search that finds its thread's in use has its own.
struct Scratch {
  std::vector<Node> nodes;
  std::vector<uint8_t> flags;              // a node's: kScored, kQueued
  std::vector<uint64_t> keys;              // the children's table: (parent << 32 | class), kNoKey where empty
  std::vector<int32_t> values;             // the child of the key in the same slot
  std::vector<uint32_t> used;              // the table's filled slots, to empty them for the next line
  std::vector<int32_t> pending, cls, batch, which;
  std::vector<uint64_t> states, next_states;
  std::vector<float> out;
  std::vector<Entry> beam, next, kept;
  std::vector<int32_t> slot, stamp;
  std::vector<size_t> order, bounded;
  std::vector<double> score, acoustic, thr;
  std::vector<int32_t> ids;
  bool busy = false;
};

constexpr uint64_t kNoKey = ~uint64_t{0};
constexpr uint8_t kScored = 1, kQueued = 2;

// The scratch of the thread, or `own` where a search on it is under way (re-entered from a language model).
class ScratchLease {
 public:
  ScratchLease() {
    thread_local Scratch mine;
    s_ = mine.busy ? &own_ : &mine;
    s_->busy = true;
  }
  ~ScratchLease() { s_->busy = false; }
  ScratchLease(const ScratchLease&) = delete;
  ScratchLease& operator=(const ScratchLease&) = delete;
  Scratch& operator*() const { return *s_; }

 private:
  Scratch own_;
  Scratch* s_;
};

class Prefixes {
 public:
  Prefixes(const LanguageModel* lm, Scratch& s) : lm_(lm), s_(s) {
    s_.nodes.clear();
    s_.flags.clear();
    s_.pending.clear();
    for (uint32_t i : s_.used) s_.keys[i] = kNoKey;
    s_.used.clear();
    if (s_.keys.empty()) {
      s_.keys.assign(size_t{1} << 12, kNoKey);
      s_.values.assign(s_.keys.size(), 0);
    }
    s_.nodes.push_back({-1, -1, 0, lm ? lm->start() : 0, 0.0});
    s_.flags.push_back(kScored);
  }
  // The child of p by class c, made where it is new. A child the language model has not scored is queued for the
  // next resolve (once a frame).
  int32_t child(int32_t p, int32_t c) {
    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(p)) << 32) | static_cast<uint32_t>(c);
    const uint64_t mask = s_.keys.size() - 1;
    for (uint64_t i = home(key);; i = (i + 1) & mask) {
      const uint64_t k = s_.keys[i];
      if (k == key) {
        const int32_t n = s_.values[i];
        queue(n);
        return n;
      }
      if (k != kNoKey) continue;
      const int32_t n = static_cast<int32_t>(s_.nodes.size());
      const Node parent = s_.nodes[static_cast<size_t>(p)];
      s_.nodes.push_back(Node{p, c, parent.depth + 1, parent.state, parent.lm});
      s_.flags.push_back(0);
      s_.keys[i] = key;
      s_.values[i] = n;
      s_.used.push_back(static_cast<uint32_t>(i));
      if (2 * s_.used.size() > s_.keys.size()) grow();
      queue(n);
      return n;
    }
  }
  // The language model's scores of every queued prefix, asked in one batch (LanguageModel::score_batch), each from
  // its parent's state: a frame's new prefixes together. Their parents must be scored already.
  void resolve() {
    resolve(s_.pending);
    s_.pending.clear();
  }
  // The same for these prefixes alone (queued or not; those scored already are passed over). The others stay
  // queued until end_frame.
  void resolve(std::span<const int32_t> which) {
    if (!lm_) return;
    size_t m = 0;
    s_.batch.clear();
    for (int32_t i : which) {
      if (s_.flags[static_cast<size_t>(i)] & kScored) continue;
      s_.batch.push_back(i);
      ++m;
    }
    if (m == 0) return;
    s_.states.resize(m);
    s_.cls.resize(m);
    s_.out.resize(m);
    s_.next_states.resize(m);
    for (size_t i = 0; i < m; ++i) {
      const Node& n = s_.nodes[static_cast<size_t>(s_.batch[i])];
      s_.states[i] = s_.nodes[static_cast<size_t>(n.parent)].state;
      s_.cls[i] = n.cls;
    }
    lm_->score_batch(s_.states, s_.cls, s_.out, s_.next_states);
    for (size_t i = 0; i < m; ++i) {
      const size_t at = static_cast<size_t>(s_.batch[i]);
      Node& n = s_.nodes[at];
      n.lm = s_.nodes[static_cast<size_t>(n.parent)].lm + s_.out[i];
      n.state = s_.next_states[i];
      s_.flags[at] = kScored;
    }
  }
  // The frame's queue let go: a prefix left unscored is queued again when a frame reaches it again.
  void end_frame() {
    for (int32_t i : s_.pending) s_.flags[static_cast<size_t>(i)] &= static_cast<uint8_t>(~kQueued);
    s_.pending.clear();
  }
  bool scored(int32_t i) const { return (s_.flags[static_cast<size_t>(i)] & kScored) != 0; }
  const Node& operator[](int32_t i) const { return s_.nodes[static_cast<size_t>(i)]; }
  size_t size() const { return s_.nodes.size(); }
  void ids(int32_t i, std::vector<int32_t>& out) const {
    out.resize(static_cast<size_t>(s_.nodes[static_cast<size_t>(i)].depth));
    for (int32_t n = i; n > 0; n = s_.nodes[static_cast<size_t>(n)].parent) {
      out[static_cast<size_t>(s_.nodes[static_cast<size_t>(n)].depth - 1)] = s_.nodes[static_cast<size_t>(n)].cls;
    }
  }

 private:
  uint64_t home(uint64_t key) const {
    return (key * 0x9E3779B97F4A7C15ULL) >> (64 - std::countr_zero(s_.keys.size()));
  }
  void queue(int32_t n) {
    uint8_t& f = s_.flags[static_cast<size_t>(n)];
    if (!lm_ || (f & (kScored | kQueued))) return;
    f |= kQueued;
    s_.pending.push_back(n);
  }
  void grow() {
    std::vector<uint64_t> keys(s_.keys.size() * 2, kNoKey);
    std::vector<int32_t> values(keys.size(), 0);
    std::vector<uint32_t> used;
    used.reserve(s_.used.size());
    const uint64_t mask = keys.size() - 1;
    s_.keys.swap(keys);
    for (uint32_t j : s_.used) {
      uint64_t i = home(keys[j]);
      while (s_.keys[i] != kNoKey) i = (i + 1) & mask;
      s_.keys[i] = keys[j];
      values[i] = s_.values[j];
      used.push_back(static_cast<uint32_t>(i));
    }
    s_.values.swap(values);
    s_.used.swap(used);
  }

  const LanguageModel* lm_;
  Scratch& s_;
};

// The frame's greedy class: the blank unless its best candidate beats it (ties to the blank, the lower id).
int32_t argmax_at(const FrameCandidates& f, int t) {
  const size_t i = static_cast<size_t>(t) * f.k;
  return f.k > 0 && f.logp[i] > f.blank[static_cast<size_t>(t)] ? f.ids[i] : 0;
}

// A survivor's reading: its exact score (ctc_logprob over its ids, already found where ctc is given) and its
// alignment.
BeamLine finish(const Emissions& em, const Prefixes& trie, int32_t node, const BeamRule& rule,
                const LanguageModel* lm, std::vector<int32_t>& ids, const double* ctc = nullptr,
                const double* lmv = nullptr) {
  BeamLine out;
  trie.ids(node, ids);
  out.ctc = ctc ? *ctc : logprob(em, ids);
  const Node& n = trie[node];
  out.lm = lmv ? *lmv : lm ? n.lm + lm->end(n.state) : 0.0;
  out.score = out.ctc + rule.alpha * out.lm + rule.beta * n.depth;
  out.line = align(em, ids);
  return out;
}

}  // namespace

FrameCandidates frame_candidates(const float* logits, int classes, int frames, int t_n, int k) {
  FrameCandidates f;
  f.frames = std::clamp(t_n, 0, frames);
  f.k = k;
  f.ids.resize(static_cast<size_t>(f.frames) * k);
  f.logp.resize(static_cast<size_t>(f.frames) * k);
  f.blank.resize(static_cast<size_t>(f.frames));
  cpu::ctc_topk(logits, classes, frames, f.frames, k, f.ids.data(), f.logp.data(), f.blank.data());
  return f;
}

FrameCandidates sure_spans(const FrameCandidates& f, float span, int margin) {
  check(f, "sure_spans");
  FrameCandidates out = f;
  if (!(span > 0.0f && span <= 1.0f) || f.frames == 0) return out;
  const double at_least = std::log(static_cast<double>(span));
  const int n = f.frames, m = std::max(0, margin);
  // a frame is unsure where its best is under span; one within the margin of an unsure frame is searched with it
  std::vector<int> unsure(static_cast<size_t>(n) + 1, 0);  // prefix counts of the unsure frames
  for (int t = 0; t < n; ++t) {
    const size_t i = static_cast<size_t>(t) * f.k;
    const double best = std::max(f.k > 0 ? static_cast<double>(f.logp[i]) : kNeg, static_cast<double>(f.blank[t]));
    unsure[static_cast<size_t>(t) + 1] = unsure[static_cast<size_t>(t)] + (best >= at_least ? 0 : 1);
  }
  const float neg = -std::numeric_limits<float>::infinity();
  for (int t = 0; t < n; ++t) {
    const int lo = std::max(0, t - m), hi = std::min(n - 1, t + m);
    if (unsure[static_cast<size_t>(hi) + 1] - unsure[static_cast<size_t>(lo)] > 0) continue;
    const size_t i = static_cast<size_t>(t) * f.k;
    const bool character = f.k > 0 && f.logp[i] > f.blank[static_cast<size_t>(t)];  // argmax_at's choice
    for (int j = character ? 1 : 0; j < f.k; ++j) out.logp[i + static_cast<size_t>(j)] = neg;
    if (character) out.blank[static_cast<size_t>(t)] = neg;
  }
  return out;
}

double ctc_logprob(const FrameCandidates& f, std::span<const int32_t> ids) {
  check(f, "ctc_logprob");
  return logprob(Emissions(f), ids);
}

CtcLine ctc_align(const FrameCandidates& f, std::span<const int32_t> ids) {
  check(f, "ctc_align");
  return align(Emissions(f), ids);
}

namespace {

// The search; its `nbest` best survivors, best first. With spans (BeamRule::span), over the given candidates' sure
// frames cut to their one class.
std::vector<BeamLine> search(const FrameCandidates& given, const BeamRule& rule, const LanguageModel* lm, int nbest) {
  check(given, "ctc_beam");
  if (rule.width < 1) throw Error("ctc_beam: a width of at least 1, not " + std::to_string(rule.width));
  if (!(rule.blank_skip > 0.0f)) throw Error("ctc_beam: blank_skip above 0 (1 or more: no frame skipped)");
  FrameCandidates cut;
  if (rule.spans()) cut = sure_spans(given, rule.span, rule.span_margin);
  const FrameCandidates& f = rule.spans() ? cut : given;
  const ScratchLease lease;
  Scratch& sc = *lease;
  const Emissions em(f);
  Prefixes trie(lm, sc);
  if (rule.width == 1) {  // greedy, by definition: the argmax path, collapsed
    int32_t node = 0, prev = 0;
    for (int t = 0; t < f.frames; ++t) {
      const int32_t a = argmax_at(f, t);
      if (a != 0 && a != prev) {
        node = trie.child(node, a);
        trie.resolve();  // the next prefix's parent is this one
      }
      prev = a;
    }
    return {finish(em, trie, node, rule, lm, sc.ids)};
  }
  const double quiet_at = rule.blank_skip >= 1.0f ? std::numeric_limits<double>::infinity()
                                                   : std::log(static_cast<double>(rule.blank_skip));
  const double alpha = rule.alpha, beta = rule.beta;
  const size_t width = static_cast<size_t>(rule.width);
  // With a model that bounds its scores (LanguageModel::max_score) and a positive weight, a frame's new prefixes are
  // scored only where they could be kept: a prefix's score is at most its parent's ln P_lm plus the bound, so one
  // whose bound is under the width's best exact scores is pruned unscored. The prefixes kept, and their order, are
  // those every prefix scored would give.
  const double bound = lm ? static_cast<double>(lm->max_score()) : 0.0;
  const bool lazy = lm && std::isfinite(bound) && alpha > 0.0 && std::isfinite(alpha);
  std::vector<Entry>&beam = sc.beam, &next = sc.next, &kept = sc.kept;
  std::vector<int32_t>&slot = sc.slot, &stamp = sc.stamp;
  std::vector<size_t>& order = sc.order;
  std::vector<double>& score = sc.score;
  beam.assign(1, Entry{0, 0.0, kNeg});
  next.clear();
  kept.clear();
  slot.clear();
  stamp.clear();  // next's entry of a node, where stamp says it is this frame's
  int t = 0;
  auto at = [&](int32_t node) -> Entry& {
    const size_t n = static_cast<size_t>(node);
    if (n >= stamp.size()) {
      stamp.resize(std::max(trie.size(), n + 1), -1);
      slot.resize(stamp.size(), 0);
    }
    if (stamp[n] != t) {
      stamp[n] = t;
      slot[n] = static_cast<int32_t>(next.size());
      next.push_back({node, kNeg, kNeg});
    }
    return next[static_cast<size_t>(slot[n])];
  };
  // The ranking's order: by score (ties to the earlier prefix).
  auto better = [&](size_t x, size_t y) {
    return score[x] > score[y] || (score[x] == score[y] && next[x].node < next[y].node);
  };
  int32_t greedy = 0, prev = 0;  // the greedy prefix so far, and the frame before's greedy class
  for (; t < f.frames; ++t) {
    next.clear();
    const size_t i0 = static_cast<size_t>(t) * f.k;
    const double lb = f.blank[static_cast<size_t>(t)];
    // A frame the blank all but owns starts no character: its prefixes go on through the blank, or through
    // their last class again (that path stays the prefix).
    const bool quiet = lb >= quiet_at;
    for (const Entry& e : beam) {
      const double total = lae(e.pb, e.pnb);
      const int32_t last = trie[e.node].cls;  // -1 at the root
      {
        Entry& x = at(e.node);
        x.pb = lae(x.pb, total + lb);
      }
      for (int j = 0; j < f.k; ++j) {
        const int32_t c = f.ids[i0 + j];
        const double v = f.logp[i0 + j];
        if (v == kNeg) continue;
        if (c == last) {  // a repeat: across a blank it extends the prefix (1 _ 1), else it stays it (1 1)
          if (e.pnb != kNeg) {
            Entry& x = at(e.node);
            x.pnb = lae(x.pnb, e.pnb + v);
          }
          if (!quiet && e.pb != kNeg) {
            Entry& x = at(trie.child(e.node, c));
            x.pnb = lae(x.pnb, e.pb + v);
          }
        } else if (!quiet) {
          Entry& x = at(trie.child(e.node, c));
          x.pnb = lae(x.pnb, total + v);
        }
      }
    }
    const int32_t a = argmax_at(f, t);
    if (a != 0 && a != prev) greedy = trie.child(greedy, a);
    prev = a;
    const size_t g = static_cast<size_t>(greedy);
    if (g >= stamp.size() || stamp[g] != t) {
      // Defensive: the greedy class and the blank are always among a frame's candidates, so the moves above
      // always reach greedy's prefix. Were it ever missed, its state from the frame before's beam: itself
      // through the blank or its last class again, or its parent through its last class.
      double pb_prev = kNeg, pnb_prev = kNeg, parent_total = kNeg;
      const int32_t parent = trie[greedy].parent;
      for (const Entry& e : beam) {
        if (e.node == greedy) {
          pb_prev = e.pb;
          pnb_prev = e.pnb;
        }
        if (e.node == parent) parent_total = lae(e.pb, e.pnb);
      }
      Entry& x = at(greedy);
      x.pb = lae(pb_prev, pnb_prev) + lb;
      const int32_t c = trie[greedy].cls;
      if (c > 0) {
        const double* r = em.of(c);
        const double v = r ? r[t] : kNeg;
        x.pnb = lae(pnb_prev + v, parent_total + v);
      }
    }
    if (next.size() <= width || !lazy) {
      trie.resolve();  // the frame's new prefixes scored together, before they are ranked
      if (next.size() <= width) {
        beam.swap(next);
        continue;
      }
      // The width's best by score (ties to the earlier prefix), and greedy's prefix besides, never pruned.
      order.resize(next.size());
      score.resize(next.size());
      for (size_t i = 0; i < next.size(); ++i) {
        order[i] = i;
        const Node& n = trie[next[i].node];
        score[i] = lae(next[i].pb, next[i].pnb) + alpha * n.lm + beta * n.depth;
      }
    } else {
      // The same ranking, the unscored prefixes at their bounds until they are scored: those whose bound reaches
      // the width's best exact score so far (all the bounds' best while fewer are exact) scored in a batch, again
      // until none is left that could be kept. Greedy's prefix is scored with the first batch (it is kept).
      std::vector<double>& acoustic = sc.acoustic;
      std::vector<size_t>& bounded = sc.bounded;
      const size_t m = next.size();
      score.resize(m);
      acoustic.resize(m);
      order.clear();
      bounded.clear();
      for (size_t i = 0; i < m; ++i) {
        const Node& n = trie[next[i].node];
        acoustic[i] = lae(next[i].pb, next[i].pnb);
        if (trie.scored(next[i].node)) {
          score[i] = acoustic[i] + alpha * n.lm + beta * n.depth;
          order.push_back(i);
        } else {
          score[i] = acoustic[i] + alpha * (trie[n.parent].lm + bound) + beta * n.depth;
          bounded.push_back(i);
        }
      }
      std::vector<int32_t>& which = sc.which;
      bool first = true;
      while (!bounded.empty()) {
        which.clear();
        if (order.size() < width) {
          // the bounds' best, as many as the width lacks
          const size_t want = std::min(width - order.size(), bounded.size());
          std::partial_sort(bounded.begin(), bounded.begin() + static_cast<std::ptrdiff_t>(want), bounded.end(),
                            better);
          for (size_t q = 0; q < want; ++q) which.push_back(next[bounded[q]].node);
        } else {
          std::vector<double>& thr = sc.thr;
          thr.clear();
          for (size_t i : order) thr.push_back(score[i]);
          std::nth_element(thr.begin(), thr.begin() + static_cast<std::ptrdiff_t>(width - 1), thr.end(),
                           std::greater<double>());
          const double at_least = thr[width - 1];
          for (size_t i : bounded) {
            if (!(score[i] < at_least)) which.push_back(next[i].node);
          }
        }
        if (first && !trie.scored(greedy) && std::find(which.begin(), which.end(), greedy) == which.end()) {
          which.push_back(greedy);
        }
        first = false;
        if (which.empty()) break;
        trie.resolve(which);
        size_t left = 0;
        for (size_t i : bounded) {
          if (trie.scored(next[i].node)) {
            const Node& n = trie[next[i].node];
            score[i] = acoustic[i] + alpha * n.lm + beta * n.depth;
            order.push_back(i);
          } else {
            bounded[left++] = i;
          }
        }
        bounded.resize(left);
      }
      if (!trie.scored(greedy)) {
        const int32_t gw[] = {greedy};
        trie.resolve(gw);
      }
      trie.end_frame();
    }
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(width), order.end(), better);
    kept.clear();
    bool has_greedy = false;
    for (size_t i = 0; i < width; ++i) {
      kept.push_back(next[order[i]]);
      has_greedy = has_greedy || next[order[i]].node == greedy;
    }
    if (!has_greedy) kept.push_back(next[static_cast<size_t>(slot[g])]);
    beam.swap(kept);
  }
  // The survivors by their exact scores: the beam's running score only bounds a prefix's probability from below
  // (pruning drops paths), and greedy's prefix is among the survivors.
  struct Ranked {
    double s, b, ctc, lm;
    int32_t node;
  };
  std::vector<Ranked> ranked;
  for (const Entry& e : beam) {
    const Node& n = trie[e.node];
    trie.ids(e.node, sc.ids);
    const double ctc = logprob(em, sc.ids);
    const double lmv = lm ? n.lm + lm->end(n.state) : 0.0;
    ranked.push_back({ctc + alpha * lmv + beta * n.depth, lae(e.pb, e.pnb) + alpha * n.lm + beta * n.depth, ctc, lmv,
                      e.node});
  }
  std::sort(ranked.begin(), ranked.end(), [](const Ranked& x, const Ranked& y) {
    return x.s > y.s || (x.s == y.s && (x.b > y.b || (x.b == y.b && x.node < y.node)));
  });
  std::vector<BeamLine> out;
  for (size_t i = 0; i < ranked.size() && static_cast<int>(i) < nbest; ++i) {
    out.push_back(finish(em, trie, ranked[i].node, rule, lm, sc.ids, &ranked[i].ctc, &ranked[i].lm));
  }
  return out;
}

}  // namespace

BeamLine ctc_beam(const FrameCandidates& f, const BeamRule& rule, const LanguageModel* lm) {
  return search(f, rule, lm, 1).front();
}

std::vector<BeamLine> ctc_beam_nbest(const FrameCandidates& f, const BeamRule& rule, const LanguageModel* lm,
                                     int n) {
  if (n < 1) throw Error("ctc_beam_nbest: n of at least 1");
  return search(f, rule, lm, n);
}

}  // namespace carbon
