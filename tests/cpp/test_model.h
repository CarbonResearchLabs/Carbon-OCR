// tests/data/colsum_argmax.onnx (tools/make_test_models.py) and its expected
// outputs, shared by the backend tests.
#pragma once

#include <filesystem>
#include <vector>

#include "carbon/tensor.h"

namespace carbon::test {

inline std::filesystem::path model_path() {
  return std::filesystem::path(CARBON_TEST_DATA) / "colsum_argmax.onnx";
}

// tests/data/strips_fpn.onnx: a detector in miniature (stride 8, maps at
// stride 2), whose receptive field a 32-row halo covers.
inline std::filesystem::path strips_model_path() {
  return std::filesystem::path(CARBON_TEST_DATA) / "strips_fpn.onnx";
}

// x [2, 1, 3, 4].  Line 0's column 2 has the most ink (the negative pixel
// counts as none), line 1's column 0.
inline Tensor input() {
  const std::vector<float> v = {
      // line 0: rows of 4 columns
      0.1f, 0.0f, 0.9f, 0.2f,  //
      0.1f, -5.f, 0.8f, 0.2f,  //
      0.0f, 0.3f, 0.7f, 0.2f,  //
      // line 1
      1.0f, 0.5f, 0.0f, 0.1f,  //
      1.0f, 0.5f, 0.0f, 0.1f,  //
      1.0f, 0.5f, 0.0f, 0.1f,  //
  };
  return Tensor::from<float>({2, 1, 3, 4}, v);
}

inline const std::vector<float>& expected_colsum() {
  static const std::vector<float> v = {0.2f, 0.3f, 2.4f, 0.6f,  //
                                       3.0f, 1.5f, 0.0f, 0.3f};
  return v;
}

inline const std::vector<int64_t>& expected_ids() {
  static const std::vector<int64_t> v = {2, 0};
  return v;
}

}  // namespace carbon::test
