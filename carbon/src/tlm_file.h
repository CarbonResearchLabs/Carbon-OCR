// A transformer language model's file (carbon/tlm.h) read into memory, in any of its precisions: what the CPU's
// model (tlm.cpp) and the GPU's (cuda/tlm.cu) each pack for their kernels.
#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "carbon/tlm.h"

namespace carbon::tlm {

// A linear layer's weight, [in][out] (the head's: the embedding's transpose, [d][tokens]).
struct Matrix {
  int in = 0, out = 0;
  std::vector<float> f32;     // the weights as they compute: FP32's, FP16's widened, INT8's dequantized (q scale)
  std::vector<uint16_t> f16;  // kF16: as stored
  std::vector<int8_t> i8;     // kInt8: as stored, with each column's scale
  std::vector<float> scale;   // kInt8: [out]
  float in_scale = 0, in_zero = 0;  // kInt8: the input's quantization (uint8, a zero point)
};

struct Layer {
  std::vector<float> ln1w, ln1b, bqkv, bo, ln2w, ln2b, b1, b2;
  Matrix qkv, o, w1, w2;
};

struct File {
  TlmHeader h{};
  std::vector<float> emb, pos, lnfw, lnfb;  // emb as it computes (INT8's dequantized), [tokens][d]
  std::vector<Layer> layers;
  Matrix head;
};

// Reads and checks the file (its magic, version, precision, shapes and size); throws Error.
File read(const std::filesystem::path& file);

}  // namespace carbon::tlm
