// The OCR pipeline on the CPU: page images in, their lines' boxes
// and text out; a PDF's pages come from carbon/pdf.h's helper processes, and a
// born-digital page's text from its text layer (carbon/textlayer.h).
//
// A page is resampled once, to the detector's scale (its long side 1024, the
// crop rule's antialiased bilinear), and `orient` reads a thumbnail of that
// (its long side 512), which is what their trainers must give them too.  The
// page is turned upright by `orient` (only by quarter turns, and only when it
// is sure) and levelled by deskew (carbon/deskew.h), its lines found by `det`
// and the line decoder (carbon/detect.h), cropped by the crop rule
// (carbon/lines.h), read by one recognizer and CTC decoding, split at wide gaps,
// and put in reading order, with blocks and tables, by the set's order
// model (order.onnx, on ONNX Runtime; carbon/order.h), or by an XY-cut where the
// set has none or the model is not sure.  Pages run on a WorkerPool (carbon/parallel.h):
// each worker has its own sessions, and each page is planned for the cores
// free when it gets to them (detection on threads when fast cores are idle,
// its lines split over them), as `carbon-ocr pages` measures.  Not yet: the
// script classifier and other recognizers, the second detection pass for tiny
// text, and the language model.
#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "carbon/detect.h"
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

struct EngineOptions {
  PoolOptions pool;               // the workers: one per physical core by default
  std::optional<Backend> backend; // the models' backend (open_set_model); none: the fastest the set has
  std::string rec = "rec-latin";  // the recognizer: <set>/<rec>.onnx and <rec>.vocab.json
  int det_long_side = 1024;       // the page's long side at detection, padded to a multiple of 32
  int orient_side = 512;          // the thumbnail orientation reads
  // How sure orientation must be to turn a page. None: the set's own, where its orient.onnx
  // carries one (carbon.orient.turn_probability, calibrated on held-out pages when the
  // model is exported), else 0.95.
  std::optional<float> turn_probability;
  // Deskew (carbon/deskew.h): a page skewed by at least min_skew degrees (and at
  // most max_skew), whose level angle clearly beats level (its projection score
  // min_skew_gain times the level page's), is levelled before detection.
  bool deskew = true;
  double max_skew = 5.0, min_skew = 0.2, min_skew_gain = 1.05;
  int line_batch = 8;             // lines a recognition task holds when every core is busy
  // Reading order: <set>/order.onnx where the set has one and ONNX Runtime is built in,
  // or order_model; with none (or use_order false) the XY-cut.
  bool use_order = true;
  std::filesystem::path order_model;
  OrderRule order;
  std::optional<SplitRule> split = SplitRule{};  // lines split at wide gaps before the order; none: not
  std::optional<TableRule> tables = TableRule{};  // tables from geometry without the order model; none: not
  DetRule det;
  // The crop rule the recognizer reads lines by. A recognizer that carries its own (its ONNX
  // file's carbon.crop.height, pad_v and pad_h, written at export) sets those three.
  CropRule crop;
};

// A word of a line: its characters between spaces.  By OCR, its box spans the CTC
// frames its characters start on (to the next character's), at the line's
// height; from a text layer, its characters' boxes.
struct OcrWord {
  LineBox box;
  std::string text;      // UTF-8
  float confidence = 0;  // the mean probability of its characters
};

struct OcrLine {
  LineBox box;           // page pixels, of the page as turned upright
  std::string text;      // UTF-8
  float confidence = 0;  // the mean probability of its characters (0 for no characters)
  float det_score = 0;   // the detector's mean core probability
  std::vector<OcrWord> words;
};

struct OcrCell {
  int row = 0, col = 0;
  std::vector<int> lines;  // indices into OcrPage::lines
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
  std::string order = "xy-cut";  // the reading order's: "model" (the order model's), or the XY-cut's
  std::vector<OcrBlock> blocks;  // the order model's; none without it
  std::string source = "ocr";  // or "text layer": a born-digital page read from its own text (carbon/textlayer.h);
                               // "hybrid": from its text, and its pictures' text by OCR
  std::string text_layer;      // a PDF page read by OCR: why its text layer was not used
  struct Timings {
    double orient = 0, detect = 0, decode = 0, recognize = 0, total = 0;  // ms
  } ms;

  std::string text() const;  // the lines, a row each
  // {"width", "height", "rotation", "skew", "source", "order", "lines": [{box, text, confidence, det_score,
  // words: [{box, text, confidence}]}], "pictures", "blocks": [{label, confidence, lines, cells: [{row,
  // col, lines}]}] (with the order model), "ms"}
  std::string json() const;
};

// A model of a Fullerene set, <set>/<name>, on the CPU:
//
//   kNative       the engine's own kernels, on <name>.int8.onnx (Q/DQ int8) when
//                 the set has it, else on <name>.onnx
//   kOnnxRuntime  ONNX Runtime on <name>.onnx
//   none          the fastest the set has: native on <name>.int8.onnx, else ONNX
//                 Runtime on <name>.onnx where it is built in, else native
//
// On one core the native int8 recognizer runs 2.5 times as fast as ONNX Runtime
// in FP32; the native backend in FP32 is slower than ONNX Runtime's.
struct SetModel {
  Backend backend;
  std::filesystem::path file;
};
SetModel choose_set_model(const std::filesystem::path& set, const std::string& name, std::optional<Backend> backend);
std::unique_ptr<Session> open_set_model(const std::filesystem::path& set, const std::string& name,
                                        std::optional<Backend> backend, const SessionOptions& options = {});

// The Fullerene set in `set` (a directory: orient.onnx, det.onnx, <rec>.onnx,
// <rec>.vocab.json, and .int8.onnx files where it has them), each model as
// open_set_model chooses; detection on several threads (fast cores idle) is ONNX
// Runtime's.  Throws Error if a file is missing or the recognizer's classes do
// not match its vocabulary.  Use from one thread.
class Engine {
 public:
  explicit Engine(const std::filesystem::path& set, EngineOptions options = {});
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // Reads every page, in parallel; the results in the pages' order.
  std::vector<OcrPage> ocr(std::span<const GrayView> pages);
  OcrPage ocr(const GrayView& page);

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

}  // namespace carbon
