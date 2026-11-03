// The page's steps the CPU engine (engine.cpp) and the batched pipeline (pipeline.cpp)
// share, so a page reads the same through either: the detector's scale and the
// orientation thumbnail, the turn orientation asks for, the page made upright and
// level, the detection placed on the upright page, a line's text and words from
// its CTC decoding, and the reading order with blocks and tables.
#pragma once

#include <span>
#include <string>
#include <vector>

#include "carbon/detect.h"
#include "carbon/engine.h"
#include "carbon/image.h"
#include "carbon/lines.h"
#include "carbon/session.h"

namespace carbon::detail {

// A page's side scaled so its long side is `to` (at least 1 px).
int fit_side(int side, int longest, int to);

// The page at the detector's scale: its long side long_side, the crop rule's
// antialiased bilinear.
GrayImage det_scale(const GrayView& page, int long_side);

// The orientation model's input, [side][side] float32: the page at the detector's
// scale fitted to side x side (its long side side), as an ink map, the rest paper.
void thumbnail_input(const GrayView& small, int page_w, int page_h, int side, float* out);

// The quarter turn the orientation logits (13 classes) ask for, in degrees
// clockwise (0, 90, 180 or 270): only a quarter turn, only when the softmax
// probability reaches turn_probability.
int rotation_of(std::span<const float> logits, float turn_probability);

// A page turned upright and levelled: the full page (view) and its copy at the
// detector's scale (small), each turned back by `rotation` and levelled when
// deskew finds a skew that clearly wins (EngineOptions' rule).
struct Upright {
  GrayImage turned, levelled, small;
  GrayView view{};
  double skew = 0;
};
void make_upright(const GrayView& page, int rotation, GrayImage small, const EngineOptions& o, Upright& u);

// A detection made at the detector's scale (dw x dh), its lines and pictures in the
// upright page's pixels (vw x vh), clamped; lines under a pixel left out.
void place_detection(const Detection& d, int vw, int vh, int dw, int dh, OcrPage& out);

// A line's text, words (each from the frame its first character starts on) and
// confidence from its CTC decoding over `length` frames.
void read_line(OcrLine& line, const CtcLine& d, int32_t length, const CropRule& crop,
               const std::vector<std::string>& vocab);

// The page's lines in reading order: the order model's with its blocks (order not
// null), or the XY-cut's; tables from geometry where no model gave blocks.
void order_page(OcrPage& page, Session* order, const EngineOptions& o);

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

}  // namespace carbon::detail
