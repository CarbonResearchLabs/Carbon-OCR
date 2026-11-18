// The n-gram language model (carbon/lm.h): its file mapped, its queries probed.
#include "carbon/lm.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#include "carbon/error.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#define CARBON_PREFETCH(p) _mm_prefetch(reinterpret_cast<const char*>(p), _MM_HINT_T0)
#elif defined(__GNUC__) || defined(__clang__)
#define CARBON_PREFETCH(p) __builtin_prefetch(p)
#else
#define CARBON_PREFETCH(p) ((void)(p))
#endif

namespace carbon {

// The file, read-only and mapped: every engine worker and process shares the one copy in the page cache.
struct NgramModel::Mapping {
  const uint8_t* data = nullptr;
  size_t size = 0;
#if defined(_WIN32)
  HANDLE file = INVALID_HANDLE_VALUE, map = nullptr;
#else
  int fd = -1;
#endif
  explicit Mapping(const std::filesystem::path& path) {
#if defined(_WIN32)
    file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                       FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (file == INVALID_HANDLE_VALUE) throw Error("cannot open " + path.string());
    LARGE_INTEGER s;
    GetFileSizeEx(file, &s);
    size = static_cast<size_t>(s.QuadPart);
    if (size) {
      map = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
      if (!map) throw Error("cannot map " + path.string());
      data = static_cast<const uint8_t*>(MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0));
      if (!data) throw Error("cannot map " + path.string());
    }
#else
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw Error("cannot open " + path.string());
    struct stat st;
    ::fstat(fd, &st);
    size = static_cast<size_t>(st.st_size);
    if (size) {
      void* p = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
      if (p == MAP_FAILED) throw Error("cannot map " + path.string());
      ::madvise(p, size, MADV_RANDOM);
      data = static_cast<const uint8_t*>(p);
    }
#endif
  }
  ~Mapping() {
#if defined(_WIN32)
    if (data) UnmapViewOfFile(data);
    if (map) CloseHandle(map);
    if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
#else
    if (data) ::munmap(const_cast<uint8_t*>(data), size);
    if (fd >= 0) ::close(fd);
#endif
  }
  Mapping(const Mapping&) = delete;
  Mapping& operator=(const Mapping&) = delete;
};

uint64_t vocab_hash(const std::vector<std::string>& vocab) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (const std::string& s : vocab) {
    for (const unsigned char c : s) {
      h ^= c;
      h *= 0x100000001b3ULL;
    }
    h *= 0x100000001b3ULL;  // the 0 byte that ends it: h ^= 0, then the multiply
  }
  return h;
}

NgramModel::~NgramModel() = default;

std::shared_ptr<NgramModel> NgramModel::open(const std::filesystem::path& file) {
  std::shared_ptr<NgramModel> m(new NgramModel());
  m->map_ = std::make_unique<Mapping>(file);
  const uint8_t* d = m->map_->data;
  const size_t size = m->map_->size;
  const std::string what = file.string();
  if (size < sizeof(NgramHeader) || std::memcmp(d, kNgramMagic, sizeof(kNgramMagic)) != 0) {
    throw Error(what + ": not an n-gram model (CRBNGRM1)");
  }
  const NgramHeader* h = reinterpret_cast<const NgramHeader*>(d);
  if (h->version != 1) throw Error(what + ": version " + std::to_string(h->version) + ", not 1");
  if (h->order < 1 || h->order > kNgramMaxOrder) throw Error(what + ": order " + std::to_string(h->order));
  if (h->classes < 2 || h->bits != static_cast<uint32_t>(std::bit_width(h->classes))) {
    throw Error(what + ": " + std::to_string(h->classes) + " classes at " + std::to_string(h->bits) + " bits");
  }
  if ((h->order - 1) * h->bits > 61) {
    throw Error(what + ": order " + std::to_string(h->order) + " at " + std::to_string(h->bits) +
                " bits a token does not fit a state's 61 bits");
  }
  const NgramTable& u = h->tables[1];
  if (u.offset % 8 || u.offset + (static_cast<uint64_t>(h->classes) + 1) * sizeof(NgramUnigram) > size) {
    throw Error(what + ": its unigrams lie past its end");
  }
  m->h_ = h;
  m->uni_ = reinterpret_cast<const NgramUnigram*>(d + u.offset);
  for (uint32_t n = 2; n <= h->order; ++n) {
    const NgramTable& t = h->tables[n];
    if (t.capacity == 0 || !std::has_single_bit(t.capacity) || t.count >= t.capacity || t.offset % 16 ||
        t.offset + t.capacity * sizeof(NgramEntry) > size) {
      throw Error(what + ": order " + std::to_string(n) + "'s table does not fit the file");
    }
    m->tab_[n] = reinterpret_cast<const NgramEntry*>(d + t.offset);
    m->mask_[n] = t.capacity - 1;
    m->shift_[n] = 64 - std::countr_zero(t.capacity);
  }
  m->bits_ = static_cast<int>(h->bits);
  m->max_ctx_ = static_cast<int>(h->order) - 1;
  m->token_mask_ = (uint64_t{1} << h->bits) - 1;
  return m;
}

size_t NgramModel::bytes() const { return map_->size; }

inline const NgramEntry* NgramModel::slot(int n, uint64_t key) const {
  return tab_[n] + (mask_[n] ? key >> shift_[n] : 0);
}

// Linear probing from the key's slot to the key or an empty slot (the tables are at most half full, so a probe
// reads one or two cache lines).
inline const NgramEntry* NgramModel::find(int n, uint64_t key) const {
  const NgramEntry* base = tab_[n];
  uint64_t i = mask_[n] ? key >> shift_[n] : 0;
  for (;;) {
    const NgramEntry* e = base + i;
    if (e->key == key) return e;
    if (e->key == 0) return nullptr;
    i = (i + 1) & mask_[n];
  }
}

namespace {

inline uint64_t low_bits(int n) { return n >= 64 ? ~uint64_t{0} : (uint64_t{1} << n) - 1; }

}  // namespace

uint64_t NgramModel::start() const {
  return max_ctx_ >= 1 ? pack(static_cast<uint64_t>(h_->classes), 1) : 0;
}

float NgramModel::score(uint64_t state, int32_t c, uint64_t& next) const {
  if (c < 1 || static_cast<uint32_t>(c) > h_->classes) {
    throw Error("NgramModel::score: token " + std::to_string(c) + " of " + std::to_string(h_->classes) + " classes");
  }
  const int len = static_cast<int>(state >> 61);
  const uint64_t ctx = state & low_bits(61);
  float bo = 0.0f, logp = 0.0f;
  int matched = 1;
  int k = std::min(len, max_ctx_);
  for (; k >= 1; --k) {
    const uint64_t ck = ctx & low_bits(k * bits_);
    if (const NgramEntry* e = find(k + 1, ngram_key(ck, k, static_cast<uint32_t>(c)))) {
      logp = bo + e->logp;
      matched = k + 1;
      break;
    }
    // back off: the context's own weight, the context an n-gram of order k
    if (k == 1) {
      bo += uni_[ck].bow;
    } else if (const NgramEntry* b = find(k, ngram_key(ck >> bits_, k - 1, static_cast<uint32_t>(ck & token_mask_)))) {
      bo += b->bow;
    }
  }
  if (k == 0) logp = bo + uni_[c].logp;
  // the state after c: the longest suffix that can be a context, the n-gram just matched at most
  const int nl = std::min(matched, max_ctx_);
  next = pack(((ctx << bits_) | static_cast<uint64_t>(c)) & low_bits(nl * bits_), nl);
  return logp;
}

float NgramModel::end(uint64_t state) const {
  uint64_t next = 0;
  return score(state, boundary(), next);
}

void NgramModel::score_batch(std::span<const uint64_t> states, std::span<const int32_t> classes,
                             std::span<float> out, std::span<uint64_t> next) const {
  // Each query a little machine: at its context length k, the probe of (ctx_k, c) at order k + 1 and, should it
  // miss, of ctx_k's back-off at order k. A round computes the keys of every pending query and prefetches their
  // slots; the next round probes them, by when the lines are on their way.
  struct Q {
    uint64_t ctx, key, bkey;
    float bo;
    int32_t k, c;
    uint32_t i;
  };
  const size_t n = states.size();
  thread_local std::vector<Q> q;
  q.clear();
  auto aim = [&](Q& x) {
    const uint64_t ck = x.ctx & low_bits(x.k * bits_);
    x.key = ngram_key(ck, x.k, static_cast<uint32_t>(x.c));
    CARBON_PREFETCH(slot(x.k + 1, x.key));
    if (x.k == 1) {
      x.bkey = ck;
      CARBON_PREFETCH(uni_ + ck);
    } else {
      x.bkey = ngram_key(ck >> bits_, x.k - 1, static_cast<uint32_t>(ck & token_mask_));
      CARBON_PREFETCH(slot(x.k, x.bkey));
    }
  };
  auto finish = [&](const Q& x, float logp, int matched) {
    out[x.i] = logp;
    const int nl = std::min(matched, max_ctx_);
    next[x.i] = pack(((x.ctx << bits_) | static_cast<uint64_t>(x.c)) & low_bits(nl * bits_), nl);
  };
  for (size_t i = 0; i < n; ++i) {
    const int32_t c = classes[i];
    if (c < 1 || static_cast<uint32_t>(c) > h_->classes) {
      throw Error("NgramModel::score_batch: token " + std::to_string(c) + " of " + std::to_string(h_->classes) +
                  " classes");
    }
    Q x{states[i] & low_bits(61), 0, 0, 0.0f, std::min(static_cast<int>(states[i] >> 61), max_ctx_), c,
        static_cast<uint32_t>(i)};
    if (x.k == 0) {
      finish(x, uni_[c].logp, 1);
      continue;
    }
    aim(x);
    q.push_back(x);
  }
  while (!q.empty()) {
    size_t kept = 0;
    for (size_t j = 0; j < q.size(); ++j) {
      Q x = q[j];
      if (const NgramEntry* e = find(x.k + 1, x.key)) {
        finish(x, x.bo + e->logp, x.k + 1);
        continue;
      }
      if (x.k == 1) {
        x.bo += uni_[x.bkey].bow;
      } else if (const NgramEntry* b = find(x.k, x.bkey)) {
        x.bo += b->bow;
      }
      if (--x.k == 0) {
        finish(x, x.bo + uni_[x.c].logp, 1);
        continue;
      }
      aim(x);
      q[kept++] = x;
    }
    q.resize(kept);
  }
}

double NgramModel::logprob(std::span<const int32_t> tokens, bool with_end) const {
  uint64_t s = start(), next = 0;
  double sum = 0;
  for (const int32_t t : tokens) {
    sum += score(s, t, next);
    s = next;
  }
  if (with_end) sum += end(s);
  return sum;
}

void NgramModel::check_vocab(const std::vector<std::string>& vocab, const std::string& what) const {
  if (vocab.size() != h_->classes || vocab_hash(vocab) != h_->vocab_hash) {
    throw Error(what + ": its language model was counted on another vocabulary (" + std::to_string(h_->classes) +
                " classes; the recognizer has " + std::to_string(vocab.size()) + ")");
  }
}

std::vector<int32_t> NgramModel::context(uint64_t state) const {
  const int len = static_cast<int>(state >> 61);
  std::vector<int32_t> out(static_cast<size_t>(len));
  for (int i = 0; i < len; ++i) {
    out[static_cast<size_t>(len - 1 - i)] = static_cast<int32_t>((state >> (i * bits_)) & token_mask_);
  }
  return out;
}

}  // namespace carbon
