// Text scored against its truth, as the models' evaluation counts it: the Levenshtein distance in code points for the
// character error rate, and in words for the word error rate.  Words are the runs between white
// space, white space as Python's str.split() has it, so the counts are those of the Python they
// replaced.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace carbon {

// UTF-8 to code points; a malformed byte stands for itself.
std::u32string code_points(std::string_view utf8);

// A code point as UTF-8 (0: the empty string).
std::string utf8_of(char32_t c);

// White space as Python's str.isspace() has it (str.split()'s separators).
bool is_space(char32_t c);

// The Levenshtein distance: insertions, deletions and substitutions, each 1; between texts' code
// points, or a recognizer's classes.
int64_t edit_distance(std::span<const char32_t> a, std::span<const char32_t> b);
int64_t edit_distance(std::span<const int32_t> a, std::span<const int32_t> b);

struct TextErrors {
  int64_t char_errors = 0, chars = 0;  // the distance in code points; the truth's code points
  int64_t word_errors = 0, words = 0;  // the distance in words; the truth's words
};

TextErrors text_errors(std::string_view truth, std::string_view predicted);

// One edit of an alignment: the truth's code point and the prediction's, 0 for none (an
// insertion has no truth's, a deletion no prediction's).
struct CharEdit {
  char32_t truth = 0, predicted = 0;
};

// The edits of one alignment with the fewest (edit_distance of them), from the start of the
// line.  Walking back from the end, a tie goes to a substitution, then a deletion, then an
// insertion, so the same texts always give the same edits.
std::vector<CharEdit> char_edits(std::u32string_view truth, std::u32string_view predicted);

// The confusions of many lines (a recognizer's evaluation): each distinct edit and how often
// it was made, the most frequent first (ties by the code points), and the totals by kind.
struct Confusion {
  char32_t truth = 0, predicted = 0;
  int64_t count = 0;
};
struct Confusions {
  std::vector<Confusion> edits;
  int64_t substitutions = 0, insertions = 0, deletions = 0;
  int64_t chars = 0;  // the truths' code points
};
Confusions char_confusions(std::span<const std::string> truths, std::span<const std::string> predicted);

}  // namespace carbon
