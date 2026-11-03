// Text scored against its truth (carbon/score.h).

#include "carbon/score.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace carbon {

std::u32string code_points(std::string_view s) {
  std::u32string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    const int n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
    if (n == 0 || i + n > s.size()) {
      out += c;
      ++i;
      continue;
    }
    char32_t v = n == 1 ? c : c & (0x7F >> n);
    for (int k = 1; k < n; ++k) v = (v << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    out += v;
    i += n;
  }
  return out;
}

std::string utf8_of(char32_t c) {
  std::string s;
  if (c == 0) return s;
  if (c < 0x80) {
    s += static_cast<char>(c);
  } else if (c < 0x800) {
    s += static_cast<char>(0xC0 | (c >> 6));
    s += static_cast<char>(0x80 | (c & 0x3F));
  } else if (c < 0x10000) {
    s += static_cast<char>(0xE0 | (c >> 12));
    s += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
    s += static_cast<char>(0x80 | (c & 0x3F));
  } else {
    s += static_cast<char>(0xF0 | (c >> 18));
    s += static_cast<char>(0x80 | ((c >> 12) & 0x3F));
    s += static_cast<char>(0x80 | ((c >> 6) & 0x3F));
    s += static_cast<char>(0x80 | (c & 0x3F));
  }
  return s;
}

bool is_space(char32_t c) {
  if (c <= 0x20) return c == 0x20 || (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x1F);
  if (c < 0x85) return false;
  return c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 ||
         c == 0x202F || c == 0x205F || c == 0x3000;
}

namespace {

// One row of the table at a time, over what is left once the common ends are taken off: a
// recognized line is mostly its truth, so the table is small.
template <class T>
int64_t distance(std::span<const T> a, std::span<const T> b) {
  while (!a.empty() && !b.empty() && a.front() == b.front()) a = a.subspan(1), b = b.subspan(1);
  while (!a.empty() && !b.empty() && a.back() == b.back()) a = a.first(a.size() - 1), b = b.first(b.size() - 1);
  if (a.size() < b.size()) std::swap(a, b);
  if (b.empty()) return static_cast<int64_t>(a.size());
  thread_local std::vector<int64_t> row;
  row.resize(b.size() + 1);
  for (size_t j = 0; j <= b.size(); ++j) row[j] = static_cast<int64_t>(j);
  for (size_t i = 1; i <= a.size(); ++i) {
    int64_t diag = row[0];  // the row above, one to the left
    row[0] = static_cast<int64_t>(i);
    for (size_t j = 1; j <= b.size(); ++j) {
      const int64_t up = row[j];
      row[j] = std::min({up + 1, row[j - 1] + 1, diag + (a[i - 1] != b[j - 1])});
      diag = up;
    }
  }
  return row[b.size()];
}

// The words of a text, each as a number: the same word, the same number, across both texts.
std::vector<uint32_t> words(std::u32string_view t, std::unordered_map<std::u32string_view, uint32_t>& ids) {
  std::vector<uint32_t> out;
  for (size_t i = 0; i < t.size();) {
    if (is_space(t[i])) {
      ++i;
      continue;
    }
    size_t j = i;
    while (j < t.size() && !is_space(t[j])) ++j;
    out.push_back(ids.try_emplace(t.substr(i, j - i), static_cast<uint32_t>(ids.size())).first->second);
    i = j;
  }
  return out;
}

}  // namespace

int64_t edit_distance(std::span<const char32_t> a, std::span<const char32_t> b) { return distance(a, b); }
int64_t edit_distance(std::span<const int32_t> a, std::span<const int32_t> b) { return distance(a, b); }

std::vector<CharEdit> char_edits(std::u32string_view a, std::u32string_view b) {
  // The common ends need no edits; the whole table over what is left, then back from its end.
  size_t head = 0;
  while (head < a.size() && head < b.size() && a[head] == b[head]) ++head;
  a.remove_prefix(head), b.remove_prefix(head);
  while (!a.empty() && !b.empty() && a.back() == b.back()) a.remove_suffix(1), b.remove_suffix(1);
  const size_t n = a.size(), m = b.size(), w = m + 1;
  thread_local std::vector<int32_t> d;
  d.resize((n + 1) * w);
  for (size_t j = 0; j <= m; ++j) d[j] = static_cast<int32_t>(j);
  for (size_t i = 1; i <= n; ++i) {
    int32_t* row = &d[i * w];
    const int32_t* up = row - w;
    row[0] = static_cast<int32_t>(i);
    for (size_t j = 1; j <= m; ++j) {
      row[j] = std::min({up[j] + 1, row[j - 1] + 1, up[j - 1] + (a[i - 1] != b[j - 1])});
    }
  }
  std::vector<CharEdit> out;
  out.reserve(static_cast<size_t>(d[n * w + m]));
  for (size_t i = n, j = m; i > 0 || j > 0;) {
    const int32_t here = d[i * w + j];
    if (i > 0 && j > 0 && here == d[(i - 1) * w + j - 1] + (a[i - 1] != b[j - 1])) {
      if (a[i - 1] != b[j - 1]) out.push_back({a[i - 1], b[j - 1]});
      --i, --j;
    } else if (i > 0 && here == d[(i - 1) * w + j] + 1) {
      out.push_back({a[i - 1], 0});
      --i;
    } else {
      out.push_back({0, b[j - 1]});
      --j;
    }
  }
  std::reverse(out.begin(), out.end());
  return out;
}

Confusions char_confusions(std::span<const std::string> truths, std::span<const std::string> predicted) {
  Confusions c;
  std::unordered_map<uint64_t, int64_t> count;
  for (size_t k = 0; k < truths.size() && k < predicted.size(); ++k) {
    const std::u32string t = code_points(truths[k]), p = code_points(predicted[k]);
    c.chars += static_cast<int64_t>(t.size());
    for (const CharEdit& e : char_edits(t, p)) {
      ++count[(static_cast<uint64_t>(e.truth) << 32) | e.predicted];
      (e.truth == 0 ? c.insertions : e.predicted == 0 ? c.deletions : c.substitutions) += 1;
    }
  }
  c.edits.reserve(count.size());
  for (const auto& [key, n] : count) {
    c.edits.push_back({static_cast<char32_t>(key >> 32), static_cast<char32_t>(key & 0xFFFFFFFFu), n});
  }
  std::sort(c.edits.begin(), c.edits.end(), [](const Confusion& x, const Confusion& y) {
    if (x.count != y.count) return x.count > y.count;
    return x.truth != y.truth ? x.truth < y.truth : x.predicted < y.predicted;
  });
  return c;
}

TextErrors text_errors(std::string_view truth, std::string_view predicted) {
  const std::u32string t = code_points(truth), p = code_points(predicted);
  std::unordered_map<std::u32string_view, uint32_t> ids;
  const std::vector<uint32_t> tw = words(t, ids), pw = words(p, ids);
  TextErrors e;
  e.char_errors = distance<char32_t>(t, p);
  e.chars = static_cast<int64_t>(t.size());
  e.word_errors = distance<uint32_t>(tw, pw);
  e.words = static_cast<int64_t>(tw.size());
  return e;
}

}  // namespace carbon
