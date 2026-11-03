// Errors Carbon reports.
#pragma once

#include <stdexcept>

namespace carbon {

// Every failure the library reports: a model that does not load, inputs that
// do not match the model, a backend that is not built in or cannot start.
// Backend exceptions (ONNX Runtime, CUDA, TensorRT) are rethrown as Error, so
// callers catch one type.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace carbon
