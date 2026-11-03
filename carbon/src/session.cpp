#include "carbon/session.h"

#include <string>
#include <type_traits>

#include "backends.h"
#include "net/net.h"

namespace carbon {

std::string_view backend_name(Backend backend) {
  switch (backend) {
    case Backend::kOnnxRuntime:
      return "ort";
    case Backend::kTensorRT:
      return "trt";
    case Backend::kNative:
      return "native";
  }
  return "?";
}

std::optional<Backend> parse_backend(std::string_view name) {
  if (name == "ort" || name == "onnxruntime") return Backend::kOnnxRuntime;
  if (name == "trt" || name == "tensorrt") return Backend::kTensorRT;
  if (name == "native") return Backend::kNative;
  return std::nullopt;
}

std::vector<Backend> compiled_backends() {
  std::vector<Backend> out;
#ifdef CARBON_WITH_ONNXRUNTIME
  out.push_back(Backend::kOnnxRuntime);
#endif
#ifdef CARBON_WITH_TENSORRT
  out.push_back(Backend::kTensorRT);
#endif
  out.push_back(Backend::kNative);
  return out;
}

namespace {

void check_input(const TensorInfo& want, DType dtype, const Shape& shape) {
  const std::string at = "input '" + want.name + "': ";
  if (dtype != want.dtype) {
    throw Error(at + "model takes " + std::string(dtype_name(want.dtype)) + ", got " +
                std::string(dtype_name(dtype)));
  }
  bool same = shape.size() == want.shape.size();
  for (size_t d = 0; same && d < want.shape.size(); ++d) {
    same = want.shape[d] < 0 || want.shape[d] == shape[d];
  }
  if (!same) {
    throw Error(at + "model takes shape " + shape_string(want.shape) + ", got " +
                shape_string(shape));
  }
  // A crop narrower than a model's window can crash deep inside a runtime
  //.  An empty input fails here, with its name.
  if (element_count(shape) == 0) {
    throw Error(at + "no elements in shape " + shape_string(shape) + ": skip the run instead");
  }
}

template <class T>
void check_all(const std::vector<TensorInfo>& want, std::span<const T> got) {
  if (got.size() != want.size()) {
    throw Error("model takes " + std::to_string(want.size()) + " inputs, got " +
                std::to_string(got.size()));
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if constexpr (std::is_same_v<T, Tensor>) {
      check_input(want[i], got[i].dtype(), got[i].shape());
    } else {
      check_input(want[i], got[i].dtype, got[i].shape);
    }
  }
}

}  // namespace

void Session::check_inputs(std::span<const Tensor> inputs) const { check_all(inputs_, inputs); }
void Session::check_inputs(std::span<const DeviceTensor> inputs) const {
  check_all(inputs_, inputs);
}

void Session::run(std::span<const Tensor> inputs, const std::function<void(std::span<const OutputView>)>& read) {
  const std::vector<Tensor> out = run(inputs);
  std::vector<OutputView> views;
  views.reserve(out.size());
  for (const Tensor& t : out) views.push_back({t.dtype(), t.shape(), t.raw()});
  read(views);
}

std::vector<Shape> Session::output_shapes(std::span<const Shape>) {
  throw Error("backend '" + std::string(backend_name(backend())) +
              "' runs on host memory only: use run()");
}

void Session::enqueue(std::span<const DeviceTensor>, std::span<const DeviceTensor>, void*) {
  throw Error("backend '" + std::string(backend_name(backend())) +
              "' runs on host memory only: use run()");
}

std::unique_ptr<Session> open_session(Backend backend,
                                      const std::filesystem::path& model,
                                      [[maybe_unused]] const SessionOptions& options) {
  if (!std::filesystem::is_regular_file(model)) {
    throw Error("no model file " + model.string());
  }
  switch (backend) {
    case Backend::kOnnxRuntime:
#ifdef CARBON_WITH_ONNXRUNTIME
      return detail::open_ort_session(model, options);
#else
      break;
#endif
    case Backend::kTensorRT:
#ifdef CARBON_WITH_TENSORRT
      return detail::open_trt_session(model, options);
#else
      break;
#endif
    case Backend::kNative:
      return net::open_native_session(model, options);
  }
  throw Error("backend '" + std::string(backend_name(backend)) +
              "' is not built in (CMake option CARBON_WITH_" +
              (backend == Backend::kTensorRT ? "TENSORRT" : "ONNXRUNTIME") + ")");
}

}  // namespace carbon
