// The ONNX Runtime backend: CPU, one session per worker (carbon/parallel.h).

#include <onnxruntime_cxx_api.h>

#include <cstring>
#include <string>
#include <vector>

#include "../backends.h"

namespace carbon::detail {

namespace {

Ort::Env& env() {
  static Ort::Env e(ORT_LOGGING_LEVEL_WARNING, "carbon");
  return e;
}

DType from_ort(ONNXTensorElementDataType t, const std::string& name) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return DType::kFloat32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return DType::kFloat16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return DType::kInt32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return DType::kInt64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return DType::kUInt8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return DType::kBool;
    default:
      throw Error("'" + name + "': ONNX element type " + std::to_string(t) +
                  " is not supported");
  }
}

ONNXTensorElementDataType to_ort(DType t) {
  switch (t) {
    case DType::kFloat32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DType::kFloat16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case DType::kInt32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DType::kInt64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DType::kUInt8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case DType::kBool:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  }
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

TensorInfo tensor_info(std::string name, const Ort::TypeInfo& type) {
  if (type.GetONNXType() != ONNX_TYPE_TENSOR) {
    throw Error("'" + name + "' is not a tensor (sequences and maps are not supported)");
  }
  auto info = type.GetTensorTypeAndShapeInfo();
  DType dtype = from_ort(info.GetElementType(), name);
  return {std::move(name), dtype, info.GetShape()};
}

class OrtSession final : public Session {
 public:
  OrtSession(const std::filesystem::path& model, const SessionOptions& options) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(options.threads);
    so.SetInterOpNumThreads(1);
    so.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    // With threads > 1 the session's own threads sleep when idle: spinning
    // would take cores the other workers need.  With 1 it has none.
    so.AddConfigEntry("session.intra_op.allow_spinning", "0");
    so.AddConfigEntry("session.inter_op.allow_spinning", "0");
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (options.threads > 1 && !options.thread_cpus.empty()) {
      // each of its threads (not the caller's) on any of these: ONNX Runtime numbers processors from 1
      std::string cpus;
      for (int c : options.thread_cpus) cpus += (cpus.empty() ? "" : ",") + std::to_string(c + 1);
      std::string all;
      for (int t = 1; t < options.threads; ++t) all += (all.empty() ? "" : ";") + cpus;
      so.AddConfigEntry("session.intra_op_thread_affinities", all.c_str());
    }
    // Denormals flushed to zero: no slow path on tiny activations.
    so.AddConfigEntry("session.set_denormal_as_zero", "1");
    // The arena (SessionOptions::cpu_arena).  Without it every run asks the OS
    // for its activations, and large blocks come back as fresh, zero-filled
    // pages under a lock the whole process shares, which costs more the more
    // workers run.
    if (!options.cpu_arena) so.DisableCpuMemArena();
    if (options.model_bytes) {
      session_ = Ort::Session(env(), options.model_bytes->data(), options.model_bytes->size(), so);
    } else {
      session_ = Ort::Session(env(), model.c_str(), so);
    }

    Ort::AllocatorWithDefaultOptions alloc;
    for (size_t i = 0; i < session_.GetInputCount(); ++i) {
      inputs_.push_back(tensor_info(session_.GetInputNameAllocated(i, alloc).get(),
                                    session_.GetInputTypeInfo(i)));
    }
    for (size_t i = 0; i < session_.GetOutputCount(); ++i) {
      outputs_.push_back(tensor_info(session_.GetOutputNameAllocated(i, alloc).get(),
                                     session_.GetOutputTypeInfo(i)));
    }
    for (const auto& t : inputs_) in_names_.push_back(t.name.c_str());
    for (const auto& t : outputs_) out_names_.push_back(t.name.c_str());
  }

  Backend backend() const override { return Backend::kOnnxRuntime; }

  // The run itself: its outputs ONNX Runtime's own values.
  std::vector<Ort::Value> run_values(std::span<const Tensor> inputs) {
    check_inputs(inputs);
    auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> in;
    in.reserve(inputs.size());
    for (const Tensor& t : inputs) {
      // ONNX Runtime reads inputs and never writes them.
      in.push_back(Ort::Value::CreateTensor(mem, const_cast<void*>(t.raw()), t.bytes(), t.shape().data(),
                                            t.shape().size(), to_ort(t.dtype())));
    }
    return session_.Run(run_options_, in_names_.data(), in.data(), in.size(), out_names_.data(), out_names_.size());
  }

  void run(std::span<const Tensor> inputs, const std::function<void(std::span<const OutputView>)>& read) override {
    std::vector<Ort::Value> out;
    std::vector<OutputView> views;
    try {
      out = run_values(inputs);
      for (size_t i = 0; i < out.size(); ++i) {
        auto info = out[i].GetTensorTypeAndShapeInfo();
        views.push_back({from_ort(info.GetElementType(), outputs_[i].name), info.GetShape(), out[i].GetTensorRawData()});
      }
    } catch (const Ort::Exception& e) {
      throw Error(std::string("ONNX Runtime: ") + e.what());
    }
    read(views);  // the values held until it returns
  }

  std::vector<Tensor> run(std::span<const Tensor> inputs) override {
    try {
      std::vector<Ort::Value> out = run_values(inputs);

      std::vector<Tensor> result;
      result.reserve(out.size());
      for (size_t i = 0; i < out.size(); ++i) {
        auto info = out[i].GetTensorTypeAndShapeInfo();
        Tensor t = Tensor::empty(from_ort(info.GetElementType(), outputs_[i].name),
                                 info.GetShape());
        if (t.bytes()) std::memcpy(t.raw(), out[i].GetTensorRawData(), t.bytes());
        result.push_back(std::move(t));
      }
      return result;
    } catch (const Ort::Exception& e) {
      throw Error(std::string("ONNX Runtime: ") + e.what());
    }
  }

 private:
  Ort::Session session_{nullptr};
  Ort::RunOptions run_options_;
  std::vector<const char*> in_names_, out_names_;  // point into inputs_, outputs_
};

}  // namespace

std::unique_ptr<Session> open_ort_session(const std::filesystem::path& model,
                                          const SessionOptions& options) {
  try {
    return std::make_unique<OrtSession>(model, options);
  } catch (const Ort::Exception& e) {
    throw Error("ONNX Runtime could not load " + model.string() + ": " + e.what());
  }
}

}  // namespace carbon::detail
