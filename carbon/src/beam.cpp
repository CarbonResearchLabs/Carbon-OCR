// The CTC prefix beam (carbon/beam.h), with a language model's score on each prefix.
#include "carbon/beam.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
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
// which the frames left can still reach the end (states below it read no path that ends in ids).
double logprob(const Emissions& em, std::span<const int32_t> ids) {
  const int frames = em.frames();
  if (frames == 0) return ids.empty() ? 0.0 : kNeg;
  const Extended x(em, ids);
  const size_t states = x.states();
  if (states > 2 * static_cast<size_t>(frames) + 1) return kNeg;  // more characters than the frames can hold
  thread_local std::vector<double> a;
  a.assign(states, 0.0);
  a[0] = x.p_at(0, 0);
  if (states > 1) a[1] = x.p_at(0, 1);
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
      if (x.skip_allowed(s)) v += a[s - 2];
      v *= x.p_at(t, s);
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

// The prefixes a line's beam has made, a tree: each its parent and last class, made once, when the language
// model scores it.
struct Node {
  int32_t parent, cls, depth;
  uint64_t state;  // the language model's after the prefix
  double lm;       // its ln P_lm(prefix)
};

class Prefixes {
 public:
  explicit Prefixes(const LanguageModel* lm) : lm_(lm) { nodes_.push_back({-1, -1, 0, lm ? lm->start() : 0, 0.0}); }
  int32_t child(int32_t p, int32_t c) {
    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(p)) << 32) | static_cast<uint32_t>(c);
    const auto [it, made] = kids_.try_emplace(key, static_cast<int32_t>(nodes_.size()));
    if (made) {
      const Node parent = nodes_[static_cast<size_t>(p)];
      Node n{p, c, parent.depth + 1, 0, parent.lm};
      if (lm_) {
        uint64_t next = 0;
        n.lm += lm_->score(parent.state, c, next);
        n.state = next;
      }
      nodes_.push_back(n);
    }
    return it->second;
  }
  const Node& operator[](int32_t i) const { return nodes_[static_cast<size_t>(i)]; }
  size_t size() const { return nodes_.size(); }
  std::vector<int32_t> ids(int32_t i) const {
    std::vector<int32_t> out(static_cast<size_t>(nodes_[static_cast<size_t>(i)].depth));
    for (int32_t n = i; n > 0; n = nodes_[static_cast<size_t>(n)].parent) {
      out[static_cast<size_t>(nodes_[static_cast<size_t>(n)].depth - 1)] = nodes_[static_cast<size_t>(n)].cls;
    }
    return out;
  }

 private:
  const LanguageModel* lm_;
  std::vector<Node> nodes_;
  std::unordered_map<uint64_t, int32_t> kids_;
};

struct Entry {
  int32_t node;
  double pb, pnb;  // ln P(the frames so far read the prefix, its path ending in a blank / in its last class)
};

// The frame's greedy class: the blank unless its best candidate beats it (ties to the blank, the lower id).
int32_t argmax_at(const FrameCandidates& f, int t) {
  const size_t i = static_cast<size_t>(t) * f.k;
  return f.k > 0 && f.logp[i] > f.blank[static_cast<size_t>(t)] ? f.ids[i] : 0;
}

BeamLine finish(const Emissions& em, const Prefixes& trie, int32_t node, const BeamRule& rule,
                const LanguageModel* lm) {
  BeamLine out;
  const std::vector<int32_t> ids = trie.ids(node);
  out.ctc = logprob(em, ids);
  const Node& n = trie[node];
  out.lm = lm ? n.lm + lm->end(n.state) : 0.0;
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

double ctc_logprob(const FrameCandidates& f, std::span<const int32_t> ids) {
  check(f, "ctc_logprob");
  return logprob(Emissions(f), ids);
}

CtcLine ctc_align(const FrameCandidates& f, std::span<const int32_t> ids) {
  check(f, "ctc_align");
  return align(Emissions(f), ids);
}

BeamLine ctc_beam(const FrameCandidates& f, const BeamRule& rule, const LanguageModel* lm) {
  check(f, "ctc_beam");
  if (rule.width < 1) throw Error("ctc_beam: a width of at least 1, not " + std::to_string(rule.width));
  if (!(rule.blank_skip > 0.0f)) throw Error("ctc_beam: blank_skip above 0 (1 or more: no frame skipped)");
  const Emissions em(f);
  Prefixes trie(lm);
  if (rule.width == 1) {  // greedy, by definition: the argmax path, collapsed
    int32_t node = 0, prev = 0;
    for (int t = 0; t < f.frames; ++t) {
      const int32_t a = argmax_at(f, t);
      if (a != 0 && a != prev) node = trie.child(node, a);
      prev = a;
    }
    return finish(em, trie, node, rule, lm);
  }
  const double quiet_at = rule.blank_skip >= 1.0f ? std::numeric_limits<double>::infinity()
                                                   : std::log(static_cast<double>(rule.blank_skip));
  const double alpha = rule.alpha, beta = rule.beta;
  const size_t width = static_cast<size_t>(rule.width);
  std::vector<Entry> beam{{0, 0.0, kNeg}}, next, kept;
  std::vector<int32_t> slot, stamp;  // next's entry of a node, where stamp says it is this frame's
  std::vector<size_t> order;
  std::vector<double> score;
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
    if (next.size() <= width) {
      beam.swap(next);
      continue;
    }
    // The width's best by score (ties to the earlier prefix), and greedy's prefix besides, never pruned.
    order.resize(next.size());
    std::iota(order.begin(), order.end(), size_t{0});
    score.resize(next.size());
    for (size_t i = 0; i < next.size(); ++i) {
      const Node& n = trie[next[i].node];
      score[i] = lae(next[i].pb, next[i].pnb) + alpha * n.lm + beta * n.depth;
    }
    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(width), order.end(),
                      [&](size_t x, size_t y) {
                        return score[x] > score[y] || (score[x] == score[y] && next[x].node < next[y].node);
                      });
    kept.clear();
    bool has_greedy = false;
    for (size_t i = 0; i < width; ++i) {
      kept.push_back(next[order[i]]);
      has_greedy = has_greedy || next[order[i]].node == greedy;
    }
    if (!has_greedy) kept.push_back(next[static_cast<size_t>(slot[g])]);
    beam.swap(kept);
  }
  // The survivor with the best exact score: the beam's running score only bounds a prefix's probability from
  // below (pruning drops paths), and greedy's prefix is among the survivors.
  int32_t best = -1;
  double best_score = kNeg, best_beam = kNeg;
  for (const Entry& e : beam) {
    const Node& n = trie[e.node];
    const double ctc = logprob(em, trie.ids(e.node));
    const double lmv = lm ? n.lm + lm->end(n.state) : 0.0;
    const double s = ctc + alpha * lmv + beta * n.depth;
    const double b = lae(e.pb, e.pnb) + alpha * n.lm + beta * n.depth;
    if (best < 0 || s > best_score || (s == best_score && (b > best_beam || (b == best_beam && e.node < best)))) {
      best = e.node;
      best_score = s;
      best_beam = b;
    }
  }
  return finish(em, trie, best, rule, lm);
}

}  // namespace carbon
