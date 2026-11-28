// The OCR pipeline on the CPU: page images in, their lines' boxes
// and text out; a PDF's pages come from carbon/pdf.h's helper processes, and a
// born-digital page's text from its text layer (carbon/textlayer.h).
//
// A page is resampled once, to the detector's scale (its long side 1024, the
// crop rule's antialiased bilinear), and `orient` reads a thumbnail of that
// (its long side 512), which is what their trainers must give them too.  The
// page is turned upright by `orient` (only by quarter turns, and only when it
// is sure) and levelled by deskew (carbon/deskew.h), its lines found by `det`
// and the line decoder (carbon/detect.h), a page of tiny text detected again at
// twice the scale and the two passes merged, its lines cropped by the crop rule
// (carbon/lines.h), read by one recognizer and CTC decoding, split at wide gaps,
// and put in reading order, with blocks and tables, by the set's order
// model (order.native.bin on the native backend, or order.onnx on ONNX Runtime;
// carbon/order.h), or by an XY-cut where the set has none or the model is not
// sure.  Pages run on a WorkerPool (carbon/parallel.h):
// each worker has its own sessions, and each page is planned for the cores
// free when it gets to them (detection on threads when fast cores are idle,
// its lines split over them), as `carbon-ocr pages` measures.  Not yet: the
// script classifier and other recognizers, and the language model.
#pragma once

#include <array>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "carbon/beam.h"
#include "carbon/tlm.h"
#include "carbon/detect.h"
#include "carbon/image.h"
#include "carbon/lines.h"
#include "carbon/order.h"
#include "carbon/parallel.h"
#include "carbon/session.h"

namespace carbon {

// Line splitting: a line split between two of its words
// where the gap is wide, at a column gutter the detector or a text layer ran
// across, or between a table row's cells.  The gaps are in the line's heights; a
// line's word spaces are the median of its gaps under min_gap, or `space` where it
// has none.
struct SplitRule {
  double min_gap = 0.6;  // a narrower gap never splits
  double ratio = 2.5;    // a gap this many times the line's word spaces splits it
  double space = 0.25;
  double always = 1.5;   // a gap this wide splits it whatever its spaces
  // A justified line: where none of a line's gaps is under min_gap, its word space is its smallest gap, not
  // `space`, and a line of one gap splits only at `always`, so a narrow justified column's wide spaces do not
  // split its lines. False: `space` (the default).
  bool justified = false;
  // A line's first word that is a list's bullet alone (GateRule::bullet_join puts a lone one there) stays with the
  // word after it within this many line heights, however wide its spaces say the gap is (0: it splits as any word).
  // A dash is not held: it can be an empty cell's.
  double marker = 2.5;
};

// Tables from geometry, for a page the order model gave no blocks: lines (split at
// their gutters) in rows by their heights' overlap; a run of rows of two cells or
// more, each row's cells in columns of their own (the columns the cells' extents
// make, overlapping), is a table.  Rows of long lines are columns of text, not a
// table's: the median cell is at most max_cell_chars.
struct TableRule {
  int min_rows = 2, min_cols = 2;
  double row_overlap = 0.5;  // two lines share a row where their heights overlap this much of the smaller's
  double max_row_gap = 2.0;  // line heights between a table's rows at most
  int max_cell_chars = 30;
};

// Tables by geometry inside the order model's Table blocks (EngineOptions::grid). Distances are in the table's
// median line height.
//   1. Blocks: two Table blocks are one table where they overlap or lie within merge_gap of each other down the
//      page, their spans across overlapping by half the narrower's; or side by side (no overlap across, their spans
//      down overlapping by half the shorter's) where side_rows of the smaller's lines share a row with the other's
//      (centres within side_tol); and where no other line lies in the gap between them (a caption, a paragraph:
//      two tables).
//   2. Extent: the lines of other blocks wholly inside the table's box are its (a header the model labelled a heading,
//      a cell's long line), and those whose centres are inside it of at most inside_chars characters (a paragraph's
//      line that runs past the table's edge is not); then
//      the rows next to it, above and below, within grow_gap, whose lines are two or more each in a column of its
//      own, their median at most grow_chars characters (a cell's, not two columns of text), while there are such;
//      then, with heading_up, the row just above it within head_gap, nothing between, where it heads the columns:
//      each of its lines inside the table across, with a letter, no number alone or caption, a phrase (not begun as
//      a continuation, a lowercase letter first, nor ended with a sentence's full stop), the whole of its block (a
//      heading of one line, not a paragraph's last), each in a column of its own, and one of them over two columns
//      or more (by header_cover, 4), not over the first, its middle over them (a heading over the values, not the
//      stub).
//      With caption_out a caption line ("Table 3", "TABLE II.", "Tab. 2:") is never taken, and the table's own
//      caption lines above or below all its others leave it, a Caption block before or after it.
//   3. Rows: its lines by their centres down the page, a line joining the row whose mean centre is within row_tol.
//   4. Columns: from its fullest rows (the most lines), each column the median of their k-th lines' edges, columns
//      that overlap joined; with label_band, before them a column of the lines wholly left of the first, where
//      there are two or more (group labels set apart from the rows they head). A line covering more than `cover`
//      of a column's width is in it, from the first such column to the last (a column span), else in the column
//      its centre is in or nearest; a header line (in the rows above the first that holds a number) by
//      header_cover instead (0: by `cover`), so a heading spans each column it lies over in part.
//   5. A cell's next lines: a row whose lines each overlap a different line of the row above across, no more of
//      them, and none of them a number alone (digits, no letter: a value, in a row of its own), joins that row,
//      cell by cell: in a header of at most header_rows rows (the rows down to the first full row), or nearer the
//      row above than wrap_pitch of the table's median row pitch. With body_wrap, a body row that does not join so
//      goes on the cells of the row above it where it reads as their next lines: fewer lines than that row, under
//      half the columns, none a number alone, each under a different line of it within wrap_gap line heights of
//      white, and each starting on a lowercase letter, a closing bracket, a percent sign or a bracket opening on a
//      lowercase letter (a unit), or under a line that ends open (a bracket left open, a hyphen, a comma, a slash,
//      a joining word such as "and" or "of").
//   6. Row labels: in the rows from the first that holds a number (a value: the rows above are the header's), a
//      first-column label over the rows next to it whose first cells are empty and which hold a number spans them,
//      set at the top of its group where the first label has none above it, else in the middle (each row the
//      nearer label's); a row whose first cell spans columns (a section's) bounds the groups.
//   7. Header cells between two header rows: a header row each of whose cells takes only columns that neither the
//      row above nor the row below has, its lines' middle between theirs (a "Route" and a "Year" beside a heading
//      that spans the quarters, level with neither it nor the quarters), joins the row above, each cell two rows tall.
//   8. Section rows (section_rows): in a table of three columns or more, a body row of a single cell, not in the
//      first column and under no cell that spans rows, whose text holds a letter, is centred in the table (its
//      middle within section_centre of the table's width of the table's) and does not begin as a cell's next line
//      (5), heads the rows below it and spans every column.
//   9. A title row (title_out): where the grid's first row is one cell, from the first column over two or more,
//      its text with a letter and no line a number alone, over a row of two cells or more and another row, it is
//      the table's title, not a row: the lines the table took from other blocks go back to them, its own lines are
//      a Caption block before it, and the grid is laid out again without them.
//  10. The false-table check (check; an option, off: at the cut its weights carry it takes false tables out but
//      writes too many real ones as text on pages unlike those it was fitted on): a logistic over the table's grid
//      (its columns, rows, how full), its text
//      (characters a cell, numbers, digits, sentences, words a line, mathematical signs, lowercase starts), its
//      lines' geometry (how many, its width on the page, their heights' spread, the gutters between cells, the rows'
//      pitch), the order model's label probabilities over its lines (Table, Text, Formula, Picture, List-Group),
//      the page's ruled lines over its rows and columns (OcrPage::rules) and the detector's pictures over it,
//      with weights fitted on labelled pages in no benchmark (one set for pages read by OCR, one for pages read
//      from their text layer: carbon/src/table_check_weights.h), the cut at check_keep
//      moved into the weights so that each set's held-out folds keep all but 2% of its true tables. A table whose
//      probability is under check_keep is written as text: its blocks Text, in the model's line order. The engine
//      finds an OCR page's ruled lines while the check is on; a text layer's are found where TextLayerRule::rules
//      asks (read_text_layer), as the weights for those pages were fitted with them.
// Each cell's lines are a line each in the writers' tables.
struct GridRule {
  double merge_gap = 1.5;
  double side_rows = 0.6, side_tol = 0.3;
  int inside_chars = 40;
  double grow_gap = 1.5;
  int grow_chars = 30;
  bool heading_up = true;
  double head_gap = 2.5;
  bool caption_out = true;
  bool title_out = true;
  double row_tol = 0.6;
  double cover = 0.5;
  double header_cover = 0.15;
  bool label_band = true;
  double wrap_pitch = 0.75;
  int header_rows = 4;
  bool body_wrap = true;
  double wrap_gap = 0.6;
  bool section_rows = true;
  double section_centre = 0.15;
  bool check = false;
  float check_keep = 0.5f;
};

// Lines that are not text, dropped before they are ordered and written: what the recognizer
// read out of a photograph, a rule, a speck or a page's edge. Before the lines are split at their gaps, a line that
// read nothing, or only rules and marks (_ - — – . | = ~ · • / \), or a sliver (taller than `sliver` times its
// width) that read at most `sliver_chars` characters: narrow ones a rule reads as (| l I 1 ! i j ( ) [ ] { } / \ :
// ; . , ' "), or any where it is taller than sliver_thin times its width (0: any, whatever its height), since a
// letter's box alone (a table's "P", "n" or "F") is up to about three times taller than wide. After the split, a
// piece at least `picture_share` on the
// detector's picture mask that read unsurely (its confidence under `picture_sure`, or a character's probability
// under `picture_least`); and a piece of at most `lone_chars` letters and digits, its confidence under `lone_sure`,
// with no other piece on its row within a line height. The JSON keeps each, with its reason (OcrPage::dropped).
//
// Two marks are kept from that, on whole lines: a list's marker the detector found apart from its item (a bullet,
// or a dash with nothing before it on its row) joins the line on its right, within bullet_join of its line heights,
// as that line's first word; and a lone dash (- ‐ – — −) with other text on its row, at least dash_cell of its line
// heights from the text on either side of it (a hyphen at a line's end is not), is an empty table cell's mark, kept
// as a line of its own. 0 turns either off.
struct GateRule {
  double sliver = 1.6;
  int sliver_chars = 2;
  double sliver_thin = 4.0;
  double picture_share = 0.8;
  float picture_sure = 0.9f, picture_least = 0.5f;
  int lone_chars = 3;
  float lone_sure = 0.6f;
  double bullet_join = 2.5;
  double dash_cell = 1.0;
  // After the split: a piece of at most overlap_chars characters whose box lies over another, longer line on its row
  // by more than overlap_share of its own width is dropped (the detector found a rule's piece beside a cell, which
  // the cell's line read already; a table's "d" beside "Homes served|"), and a | at a line's edge, beside its text,
  // is taken off (edge_bars: a column rule the line took in, read as |; 0 keeps it).
  int overlap_chars = 2;
  double overlap_share = 0.2;
  bool edge_bars = true;
};

// Spaces the recognizer read where the ink does not part the characters: a space whose
// probability is under `sure` is dropped where the crop's columns between its two neighbouring characters hold no
// blank run `wider` times the line's median blank run between inks, and at least `min_gap` of the text's height
// (letter-spaced and wide type read a space between letters; a word's space is wider than the line's letters' gaps,
// however tight the type). A column is ink where the crop's darkest pixel in it, in the text's rows, is darker than
// halfway between the crop's darkest and its 90th percentile; a crop of under 32 levels' contrast keeps every space.
struct SpaceRule {
  float sure = 0.7f;
  double wider = 2.0;
  double min_gap = 0.08;
};

// Turns read before they are made: the recognizer reads a page's `lines` longest lines (its widest boxes, found
// on the page as orientation would turn it) as they stand and turned half round, and a reading's sureness is the mean
// probability of its characters, spaces aside (0 for none). The readings judge a turn only where one of them reaches
// `floor`: rec-latin reads a script it was not trained on unsurely either way up, so there they are no evidence
// either way, and orientation's turn stands (no veto without evidence).
//   - A half turn orientation is sure of: where a reading reaches the floor, made where the turned reading is at least
//     as sure as the page's as given, else the page is read as given; where none does, orientation's turn as it
//     makes it without the check (by a bin's probability unless turn_by_quarters, below).
//   - A quarter turn it is sure of (with `quarters`) goes the other way round where that reading is the surer and
//     reaches the floor.
//   - A page it is not sure enough to turn whose sideways bins (60 to 120 and 240 to 300 degrees) hold `sideways` of
//     its probability is read as given and turned to its likelier side (both ways), and turned where a sideways
//     reading is the surest of the three and reaches the floor.
// With `sum_bins` a turn's probability is its quarter's three bins' sum (EngineOptions::turn_by_quarters) wherever
// turns are read: the readings bear out what the sums risk (upright pages in scripts orientation never saw taken for
// upside down), and a half turn no reading can judge is the bins' (the sums' half turns are uncalibrated). Nothing is
// read where nothing can turn (a turn_probability over 1). The CPU engine's: the batched pipeline turns by
// orientation alone, by a bin's probability unless turn_by_quarters.
struct TurnCheck {
  int lines = 8;
  float floor = 0.8f;
  double sideways = 0.8;
  bool quarters = true;
  bool sum_bins = true;
};

// The precision a set's models run in on the CPU: which of each model's files
// (choose_set_model).  auto: the fastest the set has.
enum class Precision : uint8_t { kAuto, kFp32, kFp16, kInt8 };
std::string_view precision_name(Precision precision);  // "auto", "fp32", "fp16", "int8"
std::optional<Precision> parse_precision(std::string_view name);

// Which language models read a page's lines (EngineOptions::lm_mode):
//   kAuto         as set by hand (EngineOptions::beam, lm, tlm): the beam, and the set's models with it, only where
//                 beam.width is set
//   kNone         greedy: no beam, no language model
//   kNgram        the beam with the n-gram (lm-<group>.bin)
//   kTransformer  the beam, its best readings rescored by the transformer (tlm-<group>.bin) after the line above
//   kBoth         the beam with the n-gram, its best readings rescored by the transformer
enum class LmMode : uint8_t { kAuto, kNone, kNgram, kTransformer, kBoth };
constexpr int kLmBeamWidth = 8;  // the beam's width under a mode where none is set: the width the models were tuned at
std::string_view lm_mode_name(LmMode mode);  // "auto", "none", "ngram", "transformer", "both"
std::optional<LmMode> parse_lm_mode(std::string_view name);  // those, or "off"/"greedy" for none, "tlm" for transformer

// A line's kind by its characters, spaces aside: digits and codes (a digit among them and at most half of them
// letters: amounts, dates, phone numbers, IDs, part numbers), prose (at least 60% letters and at most 10% digits),
// else mixed; a line of none is mixed. The beam may weigh and gate each kind apart (EngineOptions::lm_kinds), from its
// greedy reading.
enum class LineKind : uint8_t { kProse, kMixed, kCode };
constexpr int kLineKinds = 3;
LineKind line_kind(std::string_view utf8);
std::string_view line_kind_name(LineKind kind);  // "prose", "mixed", "code"
std::optional<LineKind> parse_line_kind(std::string_view name);

struct EngineOptions {
  PoolOptions pool;               // the workers: one per physical core by default
  std::optional<Backend> backend; // the models' backend (open_set_model); none: the fastest the set has
  Precision precision = Precision::kAuto;  // the models' files (open_set_model)
  std::string rec = "rec-latin";  // the recognizer: <set>/<rec>.onnx and <rec>.vocab.json
  int det_long_side = 1024;       // the page's long side at detection, padded to a multiple of 32
  // A page whose long side fit (det_long_side) leaves it under half this many pixels (a page over about 2.5:1,
  // a receipt, a strip) is detected at this many pixels instead, at most its own; letter and A4 pages as before.
  // 0: by the long side alone (the default). The CPU engine's; the batched pipeline's sizes are fixed.
  int64_t det_area = 0;
  // The second detection pass's page (det.second_pass_below and second_pass_scale, carbon/detect.h): its long
  // side at most this, run in strips of about a first pass's pixels (det_long_side squared; carbon/strips.h)
  // where it has more, so a worker's detector never holds much more than a first pass's memory.
  int second_pass_long_side = 2048;
  int orient_side = 512;          // the thumbnail orientation reads
  // How sure orientation must be to turn a page. None: the set's own, where its orient.onnx
  // carries one (carbon.orient.turn_probability, calibrated on held-out pages when the
  // model is exported), else 0.95.
  std::optional<float> turn_probability;
  // A quarter turn's probability the sum of its three bins' (for 90 degrees: 60, 90 and 120), the turn the
  // likeliest quarter, where it reaches turn_probability; what is left of the angle, up to 45 degrees, is deskew's
  // (max_skew). False: a quarter turn's own bin, the likeliest of the 12 (the default), unless the turn is read
  // (turn_check's sum_bins). The sums run higher than a bin's, and no threshold is calibrated for them: at a bin's
  // 0.65, pages in scripts the model was not trained on, upright, were turned 180 degrees, which the turn check reads
  // and undoes.
  bool turn_by_quarters = false;
  // Turns read before they are made (TurnCheck). None: orientation's turn alone.
  std::optional<TurnCheck> turn_check = TurnCheck{};
  // Deskew (carbon/deskew.h): a page skewed by at least min_skew degrees (and at
  // most max_skew), whose level angle clearly beats level (its projection score
  // min_skew_gain times the level page's), is levelled before detection; past 5 degrees (a page photographed or
  // scanned askew, up to 45) its score must be wide_skew_gain times the level page's. 45 by default: on real pages
  // turned 8 to 40 degrees, 0.7 to 0.8 of the upright page's text read back against 0.2 to 0.45 at 5, and upright
  // and slightly skewed pages read the same; 5: the narrow search alone.
  bool deskew = true;
  double max_skew = 45.0, min_skew = 0.2, min_skew_gain = 1.05, wide_skew_gain = 1.25;
  int line_batch = 8;             // lines a recognition task holds when every core is busy
  // The CPU engine's width buckets (width_bucket): the trainer's ladder, 1.3 times wider a step. Finer (1.15) pads
  // the lines less (7.5% of recognition's pixels, against 13%), but the recognizer then reads
  // paddings it was not trained on, so it is a graded choice. The GPU's profiles keep 1.3.
  double bucket_ratio = 1.3;
  // The stream (Engine::submit): pages being read at once, past which submit waits. 0: two a worker, so a
  // worker that finishes a page always finds the next one queued.
  int in_flight = 0;
  // Reading order: <set>/order.native.bin on the native backend (unless backend is ONNX Runtime),
  // else <set>/order.onnx where ONNX Runtime is built in, or order_model; with none (or use_order
  // false) the XY-cut.
  bool use_order = true;
  std::filesystem::path order_model;
  OrderRule order;
  // The order model's blocks in an order of their own, by geometry: an XY-cut over the blocks' boxes, its gaps in
  // the page's units (a gutter in character widths, a gap down the page in line heights), so columns are read one
  // after the other; the model's order inside each block, its labels and cells, kept. False: the model's order.
  bool blocks_by_geometry = true;
  // The order model's tables with their spans: a cell over more than half of two cells that
  // lie apart across the page (a header over columns) spans the model's columns it covers more than half of, and a
  // column the model made of such cells alone goes; every other cell keeps the model's row and column, so a table
  // without a wide cell is the model's as it was. False: the model's grid, no spans.
  bool table_spans = true;
  // The order model's tables laid out by geometry (GridRule, tables_by_geometry): its Table blocks that are one
  // table joined, each grown by the lines inside its box and the rows beside it that fall in its columns, its grid
  // from its lines' rows and its fullest rows' columns. A table with no two columns keeps the model's cells. None:
  // the model's cells for every table (with table_spans).
  std::optional<GridRule> grid = GridRule{};
  // One ONNX Runtime session a model for every worker, not one a worker: its packed weights
  // once in the caches the cores share, its run on the caller's thread (one intra-op thread) as before. The native
  // backend's sessions keep each run's buffers and stay a worker's own. Measured on the speed set: the same bytes,
  // and 4-6% more pages a second on 6 and 10 workers (one alone: the same). False: a session a worker.
  bool share_sessions = true;
  std::optional<SplitRule> split = SplitRule{};  // lines split at wide gaps before the order; none: not
  std::optional<TableRule> tables = TableRule{};  // tables from geometry without the order model; none: not
  std::optional<GateRule> gate = GateRule{};      // lines that are not text dropped; none: every line kept
  std::optional<SpaceRule> spaces;                // unsure spaces the ink does not bear out dropped; none (default)
  // A line's confidence the mean of its characters' probabilities, its spaces left out, the scale its split
  // pieces have: on by default (graded). False: over every id the decoder kept, spaces too.
  bool char_confidence = true;
  // Each word's box at its ink (the crop's columns of ink): its x0 out of its first glyph to the blank before it,
  // its x1 back from the gap to its last ink. False: from its first character's frame to the next space's (the
  // default). The CPU engine's: the GPU's crops give no ink columns back.
  bool word_boxes_at_ink = false;
  // The recognizer's head (its last 1x1 convolution) and CTC's argmax fused on the CPU's vectors, the logits
  // never written; each line's own frames, not its bucket's padding. An FP32 file on a CPU with AVX2; else the
  // file whole. The logits are the engine's sums, not ONNX Runtime's. False: the file whole (the default). Its one
  // kernel, AVX2's tile, runs at the engine's FP32 GEMM's rate, under ONNX Runtime's, so it is slower on every CPU
  // it runs on that was timed (AMD Zen 2 and Intel Raptor Lake, both one worker): an option for experiments.
  bool fused_rec_head = false;
  // The detector's stem and first downsampling run by the engine in row bands (cpu::det_stem, the stride-2
  // tensor never whole) and ONNX Runtime fed the stride-4 tensor: an FP32 detector whose head the engine fuses.
  // The stem's sums are the engine's, not ONNX Runtime's. False: the detector as it is (the default).
  bool det_stem = false;
  // Pixels that follow the text's size. A page whose line pitch at the detector's scale (line_pitch: the
  // autocorrelation of its rows' ink) is over det_pitch px is detected at the scale that makes it det_pitch, at
  // least half the detector's side (slides, posters: 1.8 to 4 times fewer pixels). 0: every page at det_long_side
  // (the default).
  double det_pitch = 0;
  DetRule det;
  // The CTC prefix beam for the lines the recognizer is unsure of (carbon/beam.h): a line whose
  // confidence (as above: read_line's) is under beam.gate read again over its frames' beam.topk likeliest classes,
  // scored with language_model where one is given (beam.alpha its weight, beam.beta a character's bonus). Every
  // line is decoded greedily first; the rest keep it. beam.width under 2: off (the default), every line greedy.
  BeamRule beam;
  std::shared_ptr<const LanguageModel> language_model;
  // The n-gram model the beam reads with where language_model is not given (carbon/lm.h): this file, else the
  // set's own beside the recognizer (lm-latin.bin for rec-latin) where it has one; "off": none. Opened only while
  // the beam is on, and refused if it was counted on another vocabulary. lm_alpha and lm_beta set beam.alpha and
  // beam.beta then; none: the weights tuned into the model's file.
  std::filesystem::path lm;
  std::optional<float> lm_alpha, lm_beta;
  // A second model that rescores the beam's beam.width best readings of a line (carbon/tlm.h): rescorer where given,
  // else this file, else, under lm_mode transformer or both, the set's tlm-<group>.bin (the beam set by hand, auto,
  // reads with the set's n-gram alone: the transformer gains little over it at about four times the CPU's time);
  // "off": none. Each reading's score gains
  // beam.rescore times its ln P after the line above it on the page, as read (or after the line's start where it
  // has none); a page's gated lines are rescored together once all its lines are read. tlm_alpha sets
  // beam.rescore (none: the file's tuned weight).
  std::shared_ptr<const LineRescorer> rescorer;
  std::filesystem::path tlm;
  std::optional<float> tlm_alpha;
  // The GPU the file's rescorer runs on (cuda_transformer_lm: a page's lines in one run); none: the batched
  // pipeline's own where it runs on one, else the CPU; -1: the CPU.
  std::optional<int> tlm_gpu;
  // The language models as one switch (LmMode), each model at the weights tuned into its file for the set's
  // recognizer (an n-gram's alpha, beta and gate; a transformer's weight and gate).
  // kAuto, the default, leaves the beam and the models above as set by hand. Under kNgram, kTransformer and kBoth the
  // beam is on (beam.width kLmBeamWidth where it is under 2), the model the mode does not name is off, and the beam's
  // gate is the n-gram's tuned one where the n-gram reads, else the transformer's (beam.gate where the file has
  // none), lm_gate over it; lm_alpha, lm_beta and tlm_alpha still set the weights. A model the mode needs that is
  // neither given nor in the set is refused (Error). kNone: greedy, the beam off.
  LmMode lm_mode = LmMode::kAuto;
  std::optional<float> lm_gate;
  // The beam's alpha, beta and gate by the kind of line its greedy reading is (line_kind; [LineKind]): each one
  // set replaces the beam's own (beam.alpha, beta and gate, as the mode and the files set them) for lines of that
  // kind. None set (the default): every line by the beam's own.
  struct KindWeights {
    std::optional<float> alpha, beta, gate;
  };
  std::array<KindWeights, kLineKinds> lm_kinds{};
  // The crop rule the recognizer reads lines by. A recognizer that carries its own (its ONNX
  // file's carbon.crop.height, pad_v and pad_h, written at export) sets those three.
  CropRule crop;
};

// A run of a word's or a line's text set as a superscript or a subscript: a text layer's characters set smaller
// and raised or lowered (carbon/textlayer.h). Its bytes of the text, [begin, end).
struct TextScript {
  uint32_t begin = 0, end = 0;
  bool sub = false;  // a subscript; else a superscript
  bool operator==(const TextScript&) const = default;
};

// A word of a line: its characters between spaces.  By OCR, its box spans the CTC
// frames its characters start on (to the next character's), at the line's
// height; from a text layer, its characters' boxes.
struct OcrWord {
  LineBox box;
  std::string text;      // UTF-8
  float confidence = 0;  // the mean probability of its characters
  std::vector<TextScript> scripts = {};  // a text layer's: its runs set as scripts, in order
};

struct OcrLine {
  LineBox box;           // page pixels, of the page as turned upright
  std::string text;      // UTF-8
  float confidence = 0;  // the mean probability of its characters (0 for no characters)
  float det_score = 0;   // the detector's mean core probability
  std::vector<OcrWord> words;
  float least = 0;       // the least probability of its characters, spaces aside (0 for none)
  std::vector<TextScript> scripts = {};  // its words' (scripts_of_words), where its text is theirs
};

// A line's scripts made of its words' (OcrWord::scripts), the line's text its words' joined by single spaces (a
// text layer's lines, and the pieces split_lines cuts).
std::vector<TextScript> scripts_of_words(std::span<const OcrWord> words);

struct OcrCell {
  int row = 0, col = 0;
  std::vector<int> lines;  // indices into OcrPage::lines
  int row_span = 1, col_span = 1;  // the rows and columns it covers (EngineOptions::table_spans)
};

// A block of the order model's: a run of the page's lines in reading order.
struct OcrBlock {
  std::string label;        // carbon/order.h: Text, Section-Header, Table, ...
  float confidence = 0;
  std::vector<int> lines;   // indices into OcrPage::lines
  std::vector<OcrCell> cells;  // a table's, by row then column
};

struct OcrPage {
  int width = 0, height = 0;  // the page as given
  int rotation = 0;           // degrees its text was turned clockwise (0, 90, 180, 270); undone before detection
  double skew = 0;            // then degrees it was turned counterclockwise to level its lines; its boxes are the
                              // levelled page's
  std::vector<OcrLine> lines; // in reading order
  std::vector<LineBox> pictures;
  std::vector<LineBox> rules;  // its ruled lines (long thin runs of ink, across or down), for the false-table check
                               // (GridRule::check): found where its image is read; not written
  struct Dropped {            // a line the gate took for something other than text (GateRule)
    OcrLine line;
    std::string why;          // "empty", "rules", "sliver", "picture", "lone", "overlap"
  };
  std::vector<Dropped> dropped;
  PictureMask picture_mask;   // the detector's, for the gate while the page is read; cleared once it is
  std::string order = "xy-cut";  // the reading order's: "model" (the order model's), the XY-cut's, or "tagged" (a
                                 // tagged PDF page's structure tree's: page_from_tagged_layer)
  bool second_pass = false;      // detected again at a larger scale, its median line tiny (carbon/detect.h)
  std::vector<OcrBlock> blocks;  // the order model's, or a tagged page's by its roles; none without them
  std::string source = "ocr";  // or "text layer": a born-digital page read from its own text (carbon/textlayer.h);
                               // "hybrid": from its text, and its pictures' text by OCR
  std::string text_layer;      // a PDF page read by OCR: why its text layer was not used
  struct TurnRead {            // a reading the turn check made (TurnCheck): the page read turned so, and how surely
    int rotation = 0;
    double sure = 0;
  };
  std::vector<TurnRead> turn_reads;  // in the order read; none where no turn was checked
  struct Timings {
    // ms; second_pass the second detection pass's, all of it (its page, detection, decoding and the merge);
    // load the page's file read and decoded, where the engine was given a file (Engine::submit), outside total
    double orient = 0, detect = 0, decode = 0, second_pass = 0, recognize = 0, total = 0, load = 0;
  } ms;

  std::string text() const;  // the lines, a row each
  // {"width", "height", "rotation", "skew", "source", "order", "second_pass", "lines": [{box, text, confidence,
  // det_score, words: [{box, text, confidence}]}], "pictures", "blocks": [{label, confidence, lines, cells: [{row,
  // col, lines}]}] (with the order model), "ms"}
  std::string json() const;
};

// A box of a page as read (turned upright, then levelled: the boxes an OcrPage holds) on the page as given: its
// four corners (x, y), in the order of the box's top-left, top-right, bottom-right and bottom-left as read, a
// levelled page's kept inside the page; and the box around them. On a page read as it was given (no turn, no
// skew) both are the box itself. The writers place every box so (carbon/output.h).
std::array<std::array<double, 2>, 4> corners_as_given(const LineBox& box, const OcrPage& page);
LineBox box_as_given(const LineBox& box, const OcrPage& page);

// A model of a Fullerene set, <set>/<name>, on the CPU, by precision:
//
//   auto  by backend:
//           kNative       the engine's own kernels, on <name>.int8.onnx (Q/DQ int8)
//                         when the set has it, else on <name>.onnx
//           kOnnxRuntime  ONNX Runtime on <name>.onnx
//           none          the fastest the set has: native on <name>.int8.onnx, else
//                         ONNX Runtime on <name>.onnx where it is built in, else native
//   fp32  <name>.onnx, on the backend given, else ONNX Runtime where it is built in,
//         else native
//   fp16  <name>.fp16.onnx (float16 weights, input and output) on the native
//         backend's FP16 kernels (carbon/cpu.h); another backend given throws, as
//         ONNX Runtime has no FP16 convolutions on x86-64 CPUs
//   int8  <name>.int8.onnx, on the backend given, else native
//
// A precision given (not auto) throws Error when the set has no file for it.  On
// a CPU with an int8 dot product (AVX-VNNI, AVX-512 VNNI, Arm's sdot) the native
// int8 recognizer runs about 2.5 times as fast as ONNX Runtime in FP32 on one core.
// Where the int8 products widen to 16 bits (AVX2 without VNNI: cpu::int8_widens)
// INT8 is no faster than FP32 an instruction and wins by its memory traffic alone:
// one worker reads a page sooner in fp32, many workers more pages in int8, so auto
// stays int8 everywhere and fp32 is the one-worker latency choice there (carbon-ocr
// cpus says which a CPU is).  The native backend in FP32 is slower than ONNX
// Runtime's, and in FP16 about as fast as in FP32 on a CPU without
// FP16 arithmetic (net.cpp).
struct SetModel {
  Backend backend;
  std::filesystem::path file;
};
SetModel choose_set_model(const std::filesystem::path& set, const std::string& name, std::optional<Backend> backend,
                          Precision precision = Precision::kAuto);
std::unique_ptr<Session> open_set_model(const std::filesystem::path& set, const std::string& name,
                                        std::optional<Backend> backend, const SessionOptions& options = {},
                                        Precision precision = Precision::kAuto);

// How the engine reads one page of its stream where not as its options say (Engine::submit's task; the batched
// pipeline's too, carbon/pipeline.h).
struct PageTask {
  // The page read as it stands: no quarter turn and no deskew, as EngineOptions' turn_probability over 1 and deskew
  // false read it (a page's pictures read on their own, carbon/textlayer.h). Orientation is not run, as it is not
  // for any page where turn_probability is over 1: it could not turn it.
  bool as_given = false;
  // The page's lines found already, on the page as given (LineFinder::detect: their boxes and detector scores, the
  // pictures and the picture mask): detection skipped, these lines read from the page, then gated, split and ordered
  // as found lines are. The page is read as given.
  std::shared_ptr<const OcrPage> found;
};

// The Fullerene set in `set` (a directory: orient.onnx, det.onnx, <rec>.onnx,
// <rec>.vocab.json, and .int8.onnx and .fp16.onnx files where it has them), each
// model as open_set_model chooses for the options' backend and precision;
// detection on several threads (fast cores idle) is ONNX Runtime's, on det.onnx
// (not with precision fp16 or int8).  Throws Error if a file is missing or the
// recognizer's classes do not match its vocabulary.  Use from one thread.
class Engine {
 public:
  explicit Engine(const std::filesystem::path& set, EngineOptions options = {});
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Reads every page, in parallel; the results in the pages' order.
  std::vector<OcrPage> ocr(std::span<const GrayView> pages);
  OcrPage ocr(const GrayView& page);

  // A stream of pages, for a caller that keeps every core busy across calls (a server, the benchmark's
  // workers): each page starts as soon as a worker is free, with no wait for a whole batch to end. submit
  // gives a page and returns its number, from 0 in the order given; it waits while in_flight pages are being
  // read. next returns the next page's result in that order, waiting for it, and rethrows the error that page
  // met (the stream goes on); nullopt once every page given is out. The view's pixels must stay valid until
  // next has returned that page: `keep` holds them (released when it has), a GrayImage is the engine's. The
  // same bytes as ocr(). One thread may submit while another takes; not from inside the engine's workers.
  size_t submit(const GrayView& page, std::shared_ptr<const void> keep = {});
  size_t submit(GrayImage page);
  // A page read by its task (PageTask) rather than as the options say.
  size_t submit(const GrayView& page, std::shared_ptr<const void> keep, PageTask task);
  size_t submit(GrayImage page, PageTask task);
  // A page read otherwise (from its text layer) put in reading order, as order() puts it, by a worker of the
  // stream: its result that page in order.
  size_t submit_order(OcrPage page);
  // A page's image file (as load_image reads it), read and decoded by the worker that takes the page, so the
  // decode runs beside the other pages' work; a file that cannot be read is that page's error, from next.
  size_t submit(const std::filesystem::path& file);
  // A page's encoded image (PNG, JPEG, BMP, PNM, GIF or WebP, as decode_image reads it), decoded by the worker that
  // takes the page, as a file is; `keep` holds the bytes until they are decoded. Bytes that do not decode are
  // that page's error (Finished::unreadable).
  size_t submit_encoded(std::span<const uint8_t> bytes, std::shared_ptr<const void> keep);
  std::optional<OcrPage> next();

  // A page of the stream as it finished: its number (submit's), and its result or the error it met;
  // unreadable where that error was its file's or bytes' (not found, or not an image decode_image reads).
  struct Finished {
    size_t number = 0;
    OcrPage page;
    std::exception_ptr error;
    bool unreadable = false;
  };
  // The next page to finish, the first given among those done, waiting for one; nullopt once every page given
  // is out. For a caller with many pages from many sources at once (a server): a slow page holds back no
  // other. A stream takes its results by next or by next_finished, not both (next throws once
  // next_finished has taken one).
  std::optional<Finished> next_finished();

  // Puts pages read otherwise (from a text layer) in reading order as ocr does: the
  // order model's, with blocks, or the XY-cut's.
  void order(std::span<OcrPage> pages);
  bool has_order_model() const;
  // The crop rule in use: the options', with the recognizer's own height and padding.
  const CropRule& crop() const;
  // How sure orientation must be to turn a page: the options', else the set's own, else 0.95.
  float turn_probability() const;

  int workers() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The set's detector alone, for a page whose text comes from elsewhere (a born-digital page's text layer, its
// words put into the lines the detector finds, carbon/textlayer.h): the lines' boxes on the page as given (not
// turned or levelled), found as Engine finds them (at its scale, with its decoding thresholds, a page of tiny text
// detected again). Its own one-thread session of the set's det file (the options' backend and precision); use from
// one thread.
class LineFinder {
 public:
  explicit LineFinder(const std::filesystem::path& set, const EngineOptions& options = {});
  ~LineFinder();
  LineFinder(const LineFinder&) = delete;
  LineFinder& operator=(const LineFinder&) = delete;
  std::vector<LineBox> find(const GrayView& page);
  // The whole detection: the lines (boxes and detector scores, no text), the pictures and the picture mask, on the
  // page as given, as an OcrPage (its size the page's, source "ocr"); find's boxes are its lines'. What a page's
  // own pictures pass reads its lines from (PageTask::found).
  OcrPage detect(const GrayView& page);

 private:
  std::unique_ptr<Session> det_;
  EngineOptions o_;
};

// A recognizer's classes, from <rec>.vocab.json ({"chars": [...]}, class 0 the
// blank), written beside the model when it is exported.
std::vector<std::string> load_vocab(const std::filesystem::path& file);

// The lines split at their wide gaps (SplitRule): each piece a line of its own,
// its words', its box theirs at the line's height, its confidence their mean by
// characters, its text theirs joined by spaces.  A line without words stays whole.
std::vector<OcrLine> split_lines(std::vector<OcrLine> lines, const SplitRule& rule = {});

// Tables from geometry (TableRule) on a page without blocks: where it finds one,
// the page gets blocks, each table a Table block with its cells, its lines read
// row by row and placed where its first line was, the other lines Text blocks, a
// paragraph where the gap between lines is wider than a line (every block's
// confidence 0: no model's).  A page without a table, or with blocks, is left as
// it is.  Returns the tables found.
int find_tables(OcrPage& page, const TableRule& rule = {});

// Reading order by XY-cut: the boxes split, again and again, at the widest gap
// across them, a column gutter before the space between lines, and top to
// bottom, left to right where no gap is left.  Indices into `boxes`.
std::vector<size_t> reading_order(std::span<const LineBox> boxes);

// The width a line's crop is batched at: the smallest of the recognizers'
// training buckets (32 px, then 1.3 times wider, in steps of 8, to
// 2048) that holds it, or its own width rounded up to 8 past the last.
int width_bucket(int crop_width);
// The same ladder at another ratio (32 px, then `ratio` times wider, in steps of 8, to 2048): the CPU engine's
// buckets where EngineOptions::bucket_ratio is not the trainer's 1.3.
int width_bucket(int crop_width, double ratio);

}  // namespace carbon
