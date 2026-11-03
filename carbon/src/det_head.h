// The line detector's head run by the engine (carbon::cpu::det_head), the rest of the detector by its backend: the
// model cut before its head (onnx::cut), so the head's two stride-2 tensors (12.5 MB each for a Letter page in
// FP32) are never written. Found in the file as fullerene det exports it: a 1x1 convolution and ReLU, a nearest 2x
// upsample, a depthwise 3x3 and ReLU, a 1x1 convolution to the maps; in FP32, in FP16 (its weights widened, exactly)
// and in int8 Q/DQ (the quantize and dequantize pairs passed over, the weights and biases dequantized, a ReLU the
// quantizer folded into a zero point of 0 taken as one). The head's arithmetic is then FP32 whatever the file's:
// its maps are the head's in full precision, not the backend's.
#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "carbon/cpu.h"

namespace carbon::detail {

struct FusedHead {
  std::shared_ptr<const std::string> cut;  // the model to the head's input (SessionOptions::model_bytes)
  std::string input;                       // that tensor: [1, in, h/4, w/4]
  int in = 0, mid = 0, out = 0;
  std::vector<float> w1, b1, w2, b2, w3, b3;

  cpu::DetHeadWeights weights() const {
    return {in, mid, out, w1.data(), b1.data(), w2.data(), b2.data(), w3.data(), b3.data()};
  }
};

// The head of the detector in `file`, and the file cut before it; nullopt where the file's head is not that
// pattern (another detector, a test's fake one), which then runs whole.
std::optional<FusedHead> fused_det_head(const std::filesystem::path& file);

}  // namespace carbon::detail
