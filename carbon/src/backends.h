// The backends behind open_session(), each in its own file and built only when
// its option is on (CARBON_WITH_ONNXRUNTIME, CARBON_WITH_TENSORRT).
#pragma once

#include <filesystem>
#include <memory>

#include "carbon/session.h"

namespace carbon::detail {

std::unique_ptr<Session> open_ort_session(const std::filesystem::path& model,
                                          const SessionOptions& options);
std::unique_ptr<Session> open_trt_session(const std::filesystem::path& model,
                                          const SessionOptions& options);

}  // namespace carbon::detail
