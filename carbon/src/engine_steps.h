// The page's steps the CPU engine (engine.cpp) and the batched pipeline (pipeline.cpp)
// share, so a page reads the same through either: the detector's scale and the
// orientation thumbnail, the turn orientation asks for, the page made upright and
// level, the second detection pass and the detection placed on the upright page, a
// line's text and words from its CTC decoding, and the reading order with blocks and
// tables.
#pragma once

#include <algorithm>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "carbon/deskew.h"
#include "carbon/detect.h"
#include "carbon/engine.h"
#include "carbon/image.h"
#include "carbon/lines.h"
#include "carbon/session.h"
#include "carbon/tensor.h"

namespace carbon::detail {

// A page's side scaled so its long side is `to` (at least 1 px).
int fit_side(int side, int longest, int to);

// A page's lines as recognition tasks: `lines` the lines' indices task by task, task t from starts[t] to
// starts[t + 1], each task's lines by width bucket so a bucket's lines are one batch. With parts > 0 (the free
// fast cores) that many tasks of about equal cost, a line costing its bucket's width; else tasks of line_batch
// width-sorted lines. Every line keeps its bucket either way, so the text is the same.
struct LineTasks {
  std::vector<size_t> lines;
  std::vector<size_t> starts;
};
LineTasks line_tasks(std::span<const OcrLine> lines, const CropRule& crop, int parts, int line_batch,
                     double ratio = 1.3);

// The page at the detector's scale: its long side long_side, the crop rule's
// antialiased bilinear.
GrayImage det_scale(const GrayView& page, int long_side, int64_t area = 0);

// The orientation model's input, [side][side] float32: the page at the detector's
// scale fitted to side x side (its long side side), as an ink map, the rest paper.
void thumbnail_input(const GrayView& small, int page_w, int page_h, int side, float* out);

// The quarter turn the orientation logits (13 classes) ask for, in degrees
// clockwise (0, 90, 180 or 270): only a quarter turn, only when the softmax
// probability reaches turn_probability: the likeliest bin's, where it is a quarter
// turn's; with `quarters` (EngineOptions::turn_by_quarters), the likeliest quarter's,
// the sum of its three bins (for 90: 60, 90 and 120 degrees).
int rotation_of(std::span<const float> logits, float turn_probability, bool quarters = false);

// A page turned upright and levelled: the full page (view) and its copy at the
// detector's scale (small), each turned back by `rotation` and levelled when
// deskew finds a skew that clearly wins (EngineOptions' rule).
struct Upright {
  GrayImage turned, levelled, small;
  GrayView view{};
  double skew = 0;
};
// skew: the page's own, where the caller found it already on `small` as given (a page orientation does not
// turn: the search ran beside orientation); split: the full page's levelling turned in row bands.
void make_upright(const GrayView& page, int rotation, GrayImage small, const EngineOptions& o, Upright& u,
                  std::optional<Skew> skew = std::nullopt, const RowSplit& split = {});

// A page's line pitch in its pixels: the lag, 4 to 64 rows, where the autocorrelation of its rows' ink
// counts (pixels of 160 or darker, the mean taken away) peaks, where that peak is at least 0.3 of the zero lag's:
// the smallest lag that is a peak within 90% of the largest (a periodic profile peaks again at twice its pitch);
// 0 where none is (a page of no clear lines).
double line_pitch(const GrayView& page);

// A detector run: its input [1, 1, h, w] (an ink map, h and w multiples of 32), its maps.
using DetectRun = std::function<Tensor(const Tensor&)>;

// A detection made at a scale of dw x dh, its lines and pictures in the upright
// page's pixels (vw x vh), clamped.
Detection on_page(const Detection& d, int vw, int vh, int dw, int dh);

// The second detection pass (carbon/detect.h) of the upright page, its first pass
// made at dw x dh: the page resampled to det.second_pass_scale times that scale
// (its long side at most o.second_pass_long_side), as an ink map padded to a
// multiple of 32, run by `run` in strips of about a first pass's pixels (the
// detector's long side squared) where it has more (run_in_strips: a whole run's
// maps) and decoded; its boxes in the upright page's pixels.  Nothing where that
// scale is no larger than dw x dh.
std::optional<Detection> second_pass(const GrayView& upright, int dw, int dh, const EngineOptions& o,
                                     const DetectRun& run, const Detection& first);

// The page's detection on the upright page: the first pass's, made at the
// detector's scale (dw x dh), merged with the second pass's (merge_passes) where it
// takes one and `run` is given (out.second_pass, its time in out.ms.second_pass);
// its lines (those of a pixel or more) and pictures into out.
void place_detection(const Detection& first, const GrayView& upright, int dw, int dh, const EngineOptions& o,
                     const DetectRun& run, OcrPage& out);

// A line's text, words (each from the frame its first character starts on) and
// confidence from its CTC decoding over `length` frames.
// With `ink` (a flag a crop column, ink_width of them: the crop's columns that hold ink, ink_columns) and `spaces`,
// an unsure space the ink does not bear out is dropped (SpaceRule).
void read_line(OcrLine& line, const CtcLine& d, int32_t length, const CropRule& crop,
               const std::vector<std::string>& vocab, const uint8_t* ink = nullptr, int ink_width = 0,
               const SpaceRule* spaces = nullptr, bool char_confidence = false, bool words_at_ink = false);

// The confidence read_line gives a line from its decoding, without reading it: the mean probability of its
// characters, its spaces left out (char_confidence), or of every id the decoder kept; 0 with none. The beam's gate.
float decoded_confidence(const CtcLine& d, const std::vector<std::string>& vocab, bool char_confidence);

// A gated line that waits for the page's rescoring (rescore_page): its index among the page's lines, its frame
// count, the beam's n best, and its crop's ink columns where read_line takes them (EngineOptions::spaces,
// word_boxes_at_ink).
struct UnsureLine {
  size_t line = 0;
  int32_t length = 0;
  std::vector<BeamLine> nbest;
  std::vector<uint8_t> ink;  // empty: none
  int ink_width = 0;
};

// Whether gated lines wait for a rescorer (EngineOptions::rescorer, with a weight) before they are read.
inline bool rescoring(const EngineOptions& o) { return o.beam.on() && o.rescorer && o.beam.rescore != 0.0f; }

// A gated line read by the beam alone (EngineOptions::beam, language_model).
inline CtcLine beam_read(const FrameCandidates& f, const EngineOptions& o) {
  return ctc_beam(f, o.beam, o.language_model.get()).line;
}

// The lines of a recognition run under the beam's gate (EngineOptions::beam) read again: for each line i of the
// run whose decoded[i] is under it, logits_of(i, scratch) its logits [classes][frames] (null: not read again), and
// decoded[i] the beam's; where the page's lines wait for a rescorer (rescoring) and `waiting` is given, the beam's
// n best go there with i instead, decoded[i] left as it was. Nothing where the beam is off.
template <class LogitsOf>
void beam_lines(std::vector<CtcLine>& decoded, std::span<const int32_t> lengths, int classes, int frames,
                const EngineOptions& o, const std::vector<std::string>& vocab, LogitsOf&& logits_of,
                std::vector<std::pair<size_t, std::vector<BeamLine>>>* waiting = nullptr) {
  if (!o.beam.on() || classes < 2) return;
  std::vector<float> scratch;
  for (size_t i = 0; i < decoded.size(); ++i) {
    if (decoded_confidence(decoded[i], vocab, o.char_confidence) >= o.beam.gate) continue;
    const float* logits = logits_of(i, scratch);
    if (!logits) continue;
    const FrameCandidates f = frame_candidates(logits, classes, frames, lengths[i], std::min(o.beam.topk, classes - 1));
    if (waiting && rescoring(o)) {
      waiting->emplace_back(i, ctc_beam_nbest(f, o.beam, o.language_model.get(), o.beam.width));
      continue;
    }
    decoded[i] = beam_read(f, o);
  }
}

// The page's waiting lines read (before the split and the order): each reading's score plus beam.rescore times the
// rescorer's ln P of it after its context, the line above it on the page (the nearest whose box ends above its
// middle, within 1.5 of its heights, overlapping it across by half the narrower's width), that line's text as
// read (a waiting line's its best reading) between boundaries, its last 96 characters where it is longer (then
// with no boundary before them, as a training window starts), or the boundary alone; every waiting line of the
// page in one call (LineRescorer::score_lines); the best into its line by read_line.
void rescore_page(OcrPage& page, std::vector<UnsureLine>& lines, const EngineOptions& o,
                  const std::vector<std::string>& vocab);

// The beam's language model, where it is on and none is given: EngineOptions::lm's file, or the set's
// lm-<group>.bin beside the recognizer, checked against its vocabulary, its tuned weights into o.beam unless
// lm_alpha and lm_beta are given. EngineOptions::lm_mode first (the beam on, the models it does not name off, a model
// it names that is missing refused), and under a mode the beam's gate the models' tuned one (lm_gate over it).
void attach_language_model(EngineOptions& o, const std::filesystem::path& set, const std::vector<std::string>& vocab,
                           int rescorer_gpu);
// The same for the rescorer (EngineOptions::tlm, the set's tlm-<group>.bin), which attach_language_model calls: on
// that GPU (cuda_transformer_lm), or the CPU where it is -1. Returns its file's tuned gate (0: none, or no file).
float attach_rescorer(EngineOptions& o, const std::filesystem::path& set, const std::vector<std::string>& vocab,
                      int gpu);

// A crop's columns that hold ink (SpaceRule): out[x] 1 or 0 for x < width, of an h-row crop `stride` bytes a row;
// false (out untouched) where the crop's contrast is under 32 levels.
bool ink_columns(const uint8_t* crop, int width, int h, int64_t stride, const CropRule& rule, uint8_t* out);

// The page's lines in reading order: the order model's with its blocks (order not
// null), or the XY-cut's; tables from geometry where no model gave blocks.
void order_page(OcrPage& page, Session* order, const EngineOptions& o);

// The gate (GateRule), in its two places: on whole lines before they are split at their gaps, and on the pieces
// after, before they are ordered. Each dropped line goes to page.dropped with its reason, in the order it was.
void gate_whole_lines(OcrPage& page, const GateRule& rule);
void gate_pieces(OcrPage& page, const GateRule& rule);

// The order model's blocks put in an order of their own by geometry (EngineOptions::blocks_by_geometry): an XY-cut
// over the blocks' boxes, a gutter measured in the lines' median character width and a gap down the page in their
// median height, so columns come before the space between paragraphs; each block's lines in the model's order.
// po.order follows. boxes and texts are the page's lines, as po indexes them.
void blocks_by_geometry(PageOrder& po, std::span<const LineBox> boxes, std::span<const std::string> texts);

// The order model's tables whose median cell holds more than max_cell_chars characters (find_tables' own test,
// TableRule::max_cell_chars): body text the model took for a table, relabelled Text, its cells let go. The page's
// lines must be in their reading order (the blocks' indices are places in it).
void text_tables_as_text(OcrPage& page, int max_cell_chars);

// The order model's tables laid out again from their cells' boxes, with spans (EngineOptions::table_spans): the
// page's lines in their reading order, as text_tables_as_text's.
void table_spans(OcrPage& page);

// The order model's tables by geometry (GridRule, EngineOptions::grid; carbon/tables.cpp): Table blocks that
// are one table joined into one, at the first's place, each grown by the lines inside its box and the rows beside
// it, its cells by its rows and columns, its lines cell by cell; a table with no two columns keeps the model's cells
// and blocks. The page's lines follow the blocks' order again (each block's lines one run of them).
void tables_by_geometry(OcrPage& page, const GridRule& rule);

// A page's last steps, the CPU's and the GPU's alike: the gate, the split at wide gaps, the reading order (and
// tables), the picture mask let go.
void finish_page(OcrPage& page, Session* order, const EngineOptions& o);

// The order model a page is read in, where use_order: order_model (an ONNX file on
// ONNX Runtime, or an order.native.bin on the native backend); else the set's
// order.native.bin on the native backend (not when the options' backend is ONNX
// Runtime), else its order.onnx on ONNX Runtime where that is built in.  None: an
// empty file (the XY-cut).
struct OrderModel {
  Backend backend = Backend::kNative;
  std::filesystem::path file;
};
OrderModel choose_order(const std::filesystem::path& set, const EngineOptions& o);

// The crop rule a recognizer's ONNX file carries (carbon.crop.*), over `rule`.
CropRule model_crop(const std::filesystem::path& set, const std::string& rec, CropRule rule);
// How sure orientation must be to turn a page: the set's own (orient's carbon.orient.turn_probability), else 0.95.
float model_turn_probability(const std::filesystem::path& set);
// An orient file's own; and the detector's decoding thresholds from its file's carbon.det.* metadata over
// `rule`.
float model_turn_probability_of(const std::filesystem::path& file);
DetRule model_det_rule(const std::filesystem::path& file, DetRule rule);

}  // namespace carbon::detail
