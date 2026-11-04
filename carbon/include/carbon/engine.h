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

#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

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

// The precision a set's models run in on the CPU: which of each model's files
// (choose_set_model).  auto: the fastest the set has.
enum class Precision : uint8_t { kAuto, kFp32, kFp16, kInt8 };
std::string_view precision_name(Precision precision);  // "auto", "fp32", "fp16", "int8"
std::optional<Precision> parse_precision(std::string_view name);

struct EngineOptions {
  PoolOptions pool;               // the workers: one per physical core by default
  std::optional<Backend> backend; // the models' backend (open_set_model); none: the fastest the set has
  Precision precision = Precision::kAuto;  // the models' files (open_set_model)
  std::string rec = "rec-latin";  // the recognizer: <set>/<rec>.onnx and <rec>.vocab.json
  int det_long_side = 1024;       // the page's long side at detection, padded to a multiple of 32
  // The second detection pass's page (det.second_pass_below and second_pass_scale, carbon/detect.h): its long
  // side at most this, run in strips of about a first pass's pixels (det_long_side squared; carbon/strips.h)
  // where it has more, so a worker's detector never holds much more than a first pass's memory.
  int second_pass_long_side = 2048;
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
  // The stream (Engine::submit): pages being read at once, past which submit waits. 0: two a worker, so a
  // worker that finishes a page always finds the next one queued.
  int in_flight = 0;
  // Reading order: <set>/order.native.bin on the native backend (unless backend is ONNX Runtime),
  // else <set>/order.onnx where ONNX Runtime is built in, or order_model; with none (or use_order
  // false) the XY-cut.
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
  bool second_pass = false;      // detected again at a larger scale, its median line tiny (carbon/detect.h)
  std::vector<OcrBlock> blocks;  // the order model's; none without it
  std::string source = "ocr";  // or "text layer": a born-digital page read from its own text (carbon/textlayer.h);
                               // "hybrid": from its text, and its pictures' text by OCR
  std::string text_layer;      // a PDF page read by OCR: why its text layer was not used
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
// one core the native int8 recognizer runs 2.5 times as fast as ONNX Runtime in
// FP32; the native backend in FP32 is slower than ONNX Runtime's,
// and in FP16 about as fast as in FP32 on a CPU without FP16 arithmetic (net.cpp).
struct SetModel {
  Backend backend;
  std::filesystem::path file;
};
SetModel choose_set_model(const std::filesystem::path& set, const std::string& name, std::optional<Backend> backend,
                          Precision precision = Precision::kAuto);
std::unique_ptr<Session> open_set_model(const std::filesystem::path& set, const std::string& name,
                                        std::optional<Backend> backend, const SessionOptions& options = {},
                                        Precision precision = Precision::kAuto);

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
  // A page's image file (as load_image reads it), read and decoded by the worker that takes the page, so the
  // decode runs beside the other pages' work; a file that cannot be read is that page's error, from next.
  size_t submit(const std::filesystem::path& file);
  // A page's encoded image (PNG, JPEG, BMP or PNM, as decode_image reads it), decoded by the worker that
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
