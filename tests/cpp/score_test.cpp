#include "carbon/score.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

namespace carbon {
namespace {

// The whole table, nothing taken off the ends: what the distance must equal.
int64_t full_table(const std::u32string& a, const std::u32string& b) {
  std::vector<std::vector<int64_t>> d(a.size() + 1, std::vector<int64_t>(b.size() + 1));
  for (size_t i = 0; i <= a.size(); ++i) d[i][0] = static_cast<int64_t>(i);
  for (size_t j = 0; j <= b.size(); ++j) d[0][j] = static_cast<int64_t>(j);
  for (size_t i = 1; i <= a.size(); ++i)
    for (size_t j = 1; j <= b.size(); ++j)
      d[i][j] = std::min({d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1])});
  return d[a.size()][b.size()];
}

TEST(Score, EditDistance) {
  const std::u32string kitten = U"kitten", sitting = U"sitting", empty;
  EXPECT_EQ(edit_distance(kitten, sitting), 3);
  EXPECT_EQ(edit_distance(sitting, kitten), 3);
  EXPECT_EQ(edit_distance(kitten, kitten), 0);
  EXPECT_EQ(edit_distance(kitten, empty), 6);
  EXPECT_EQ(edit_distance(empty, empty), 0);
  std::mt19937 rng(7);
  for (int k = 0; k < 2000; ++k) {  // lines a few edits from their truth, and lines nothing like it
    std::u32string a, b;
    for (int n = static_cast<int>(rng() % 40); n--;) a += U"ab c"[rng() % 4];
    b = a;
    for (int e = static_cast<int>(rng() % 6); e--;) {
      const size_t at = b.empty() ? 0 : rng() % b.size();
      switch (rng() % 3) {
        case 0: b.insert(b.begin() + static_cast<std::ptrdiff_t>(at), U"xa "[rng() % 3]); break;
        case 1: if (!b.empty()) b.erase(at, 1); break;
        default: if (!b.empty()) b[at] = U"xa "[rng() % 3];
      }
    }
    if (k % 4 == 0) std::shuffle(b.begin(), b.end(), rng);
    ASSERT_EQ(edit_distance(a, b), full_table(a, b)) << k;
  }
}

TEST(Score, TextErrorsCountCodePointsAndWords) {
  const TextErrors e = text_errors("caf\xC3\xA9 au lait", "cafe au  lait");  // é, then a doubled space
  EXPECT_EQ(e.chars, 12);
  EXPECT_EQ(e.char_errors, 2);  // é for e, one space more
  EXPECT_EQ(e.words, 3);
  EXPECT_EQ(e.word_errors, 1);  // only "café" differs: words are the runs between spaces
  const TextErrors none = text_errors("", "");
  EXPECT_EQ(none.chars + none.char_errors + none.words + none.word_errors, 0);
  const TextErrors gone = text_errors("two words", "");
  EXPECT_EQ(gone.char_errors, 9);
  EXPECT_EQ(gone.word_errors, 2);
}

TEST(Score, WhiteSpaceIsPythons) {
  // str.split()'s separators: no-break space, the en quad to the hair space, the ideographic space...
  const TextErrors e = text_errors("a\xC2\xA0" "b\xE2\x80\x82" "c\xE3\x80\x80" "d\x1F" "e\xC2\x85" "f", "a b c d e f");
  EXPECT_EQ(e.words, 6);
  EXPECT_EQ(e.word_errors, 0);
  EXPECT_FALSE(is_space(0x200B));  // the zero-width space is not white space to Python
  EXPECT_FALSE(is_space(0x180E));
  EXPECT_TRUE(is_space(0x2029));
  EXPECT_EQ(text_errors("a\xE2\x80\x8B" "b", "a b").words, 1);
}

TEST(Score, CharEditsAlignWithTheFewestEdits) {
  const std::vector<CharEdit> e = char_edits(U"kitten", U"sitting");
  ASSERT_EQ(e.size(), 3u);
  EXPECT_EQ(e[0].truth, U'k');
  EXPECT_EQ(e[0].predicted, U's');
  EXPECT_EQ(e[1].truth, U'e');
  EXPECT_EQ(e[1].predicted, U'i');
  EXPECT_EQ(e[2].truth, 0u);  // the g, inserted
  EXPECT_EQ(e[2].predicted, U'g');
  EXPECT_TRUE(char_edits(U"same", U"same").empty());
  std::mt19937 rng(11);
  for (int k = 0; k < 2000; ++k) {  // as many edits as the distance, and the characters they leave line up
    std::u32string a, b;
    for (int n = static_cast<int>(rng() % 30); n--;) a += U"ab c"[rng() % 4];
    for (int n = static_cast<int>(rng() % 30); n--;) b += U"abx "[rng() % 4];
    const std::vector<CharEdit> ed = char_edits(a, b);
    ASSERT_EQ(static_cast<int64_t>(ed.size()), edit_distance(a, b));
    size_t from_a = 0, from_b = 0;  // the characters the edits take from each side
    for (const CharEdit& x : ed) from_a += x.truth != 0, from_b += x.predicted != 0;
    EXPECT_EQ(a.size() - from_a, b.size() - from_b);  // the rest match one to one
  }
}

TEST(Score, CharConfusionsCountEachEditOverTheLines) {
  const std::vector<std::string> truths = {"ab", "ab", "a", "caf\xC3\xA9"}, predicted = {"xb", "xb", "a!", "cafe"};
  const Confusions c = char_confusions(truths, predicted);
  ASSERT_EQ(c.edits.size(), 3u);
  EXPECT_EQ(c.edits[0].truth, U'a');  // the most frequent first
  EXPECT_EQ(c.edits[0].predicted, U'x');
  EXPECT_EQ(c.edits[0].count, 2);
  EXPECT_EQ(c.edits[1].truth, 0u);  // then by code point: the insertion's none first
  EXPECT_EQ(c.edits[1].predicted, U'!');
  EXPECT_EQ(c.edits[2].truth, U'é');
  EXPECT_EQ(c.edits[2].predicted, U'e');
  EXPECT_EQ(c.substitutions, 3);
  EXPECT_EQ(c.insertions, 1);
  EXPECT_EQ(c.deletions, 0);
  EXPECT_EQ(c.chars, 9);
  for (const char32_t cp : {U'a', U'é', U'€', U'\U0001F600'}) {  // 1 to 4 bytes, and back
    EXPECT_EQ(code_points(utf8_of(cp)), std::u32string(1, cp));
  }
  EXPECT_EQ(utf8_of(0), "");
}

}  // namespace
}  // namespace carbon
