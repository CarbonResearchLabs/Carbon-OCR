// Model sessions: one ONNX model loaded on one backend.
//
// Every model in the pipeline (orientation, detector, script classifier,
// recognizers, order model) runs through this interface, so a stage does not
// know whether it runs on ONNX Runtime (CPU) or TensorRT (GPU).
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "carbon/tensor.h"

namespace carbon {

// kNative: the engine's own executor for Fullerene's convolutional networks, on
// its CPU kernels (carbon/cpu.h), in every build.
enum class Backend : uint8_t { kOnnxRuntime, kTensorRT, kNative };

std::string_view backend_name(Backend backend);  // "ort", "trt", "native"
// "ort" / "onnxruntime", "trt" / "tensorrt", "native"; nullopt otherwise.
std::optional<Backend> parse_backend(std::string_view name);
// The backends this build has.  TensorRT also needs a CUDA device at run time.
std::vector<Backend> compiled_backends();

// One input or output of a model.
struct TensorInfo {
  std::string name;
  DType dtype = DType::kFloat32;
  Shape shape;  // -1 for free dimensions
};

// The shapes a TensorRT engine is built for, for one input with free
// dimensions.  The engine runs only inside the range, and a wide range costs
// speed: give the buckets the pipeline actually batches into.
struct ShapeRange {
  std::string input;
  Shape min, opt, max;

  static ShapeRange exact(std::string input, Shape shape) {
    return {std::move(input), shape, shape, shape};
  }
};

// One TensorRT optimization profile: a range for every input with free
// dimensions.  Each profile gets an execution context of its own, so runs
// never pay for switching profiles.  A profile whose ranges are all exact is a
// bucket: TensorRT tunes for its one shape, and its runs are CUDA graphs,
// captured on the first run and replayed after.
using Profile = std::vector<ShapeRange>;

struct SessionOptions {
  // CPU threads inside one run.  1 suits a WorkerPool (carbon/parallel.h): a
  // worker per core, each with its own sessions, the layout that scales on CPU
  //.
  int threads = 1;
  // ONNX Runtime's memory arena: a session keeps its runs' activation memory
  // and reuses it.  Measured on rec-latin (29 October 2026, the laptop's 10
  // cores, two interleaved rounds): 1% to 16% more runs a second with the
  // arena, from 1 to 16 workers.  It grows to the largest batch the session
  // has run (with shapes that keep changing it grows further: +1.4 GB over 10
  // runs was seen in another engine), which the width buckets bound.
  bool cpu_arena = true;
  // ONNX Runtime with threads > 1: the logical processors (numbered from 0, in processor group 0) its own threads
  // may run on, each of them; empty, wherever the OS puts them. The engine keeps a page's multi-threaded
  // detection on the fast cores of a hybrid CPU with it.
  std::vector<int> thread_cpus;
  // The model's bytes, run instead of the file's (the path still names it in errors): a model the engine has
  // changed, as the detector cut before its head (carbon/src/det_head.h). ONNX Runtime and the native backend.
  std::shared_ptr<const std::string> model_bytes;

  // The native backend: the instruction set its kernels use, by its name in
  // carbon/cpu.h ("avx2", "avx512", "neon-dot", ...).  Empty: the best the CPU
  // has.  Every set gives the same bits, but an FP16 file's on a set with FP16
  // arithmetic (avx512-fp16, neon-fp16), within FP16's rounding; this is for
  // tests and timing.
  std::string isa;

  // TensorRT.
  int device = 0;
  // Required for inputs with free dimensions.  A run takes the first profile
  // that is exact for its shapes, else the first whose ranges hold them.
  std::vector<Profile> profiles;
  // Built engines are cached here, keyed by the model bytes, the profiles, the
  // build settings, the GPU and the TensorRT version, next to a timing cache
  // that makes rebuilds quicker.  Empty: build on every open (minutes).
  std::filesystem::path engine_cache;
  size_t workspace_bytes = size_t{1} << 30;
  // TensorRT's builder optimization level, 0 to 5 (its default is 3).  Higher
  // levels try more kernels: builds take longer, engines may run faster.
  int optimization_level = 3;
  // Replay runs of exact profiles as CUDA graphs: one launch instead of one
  // per layer.
  bool cuda_graphs = true;
};

// A tensor in GPU memory.  Not owned: the caller allocates and frees it.
struct DeviceTensor {
  void* data = nullptr;
  DType dtype = DType::kFloat32;
  Shape shape;
};

// A loaded model.  run() takes the inputs in the order of inputs() and returns
// the outputs in the order of outputs().  A Session is not safe to run from two
// threads at once: give each worker its own.
class Session {
 public:
  virtual ~Session() = default;
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  virtual Backend backend() const = 0;
  const std::vector<TensorInfo>& inputs() const { return inputs_; }
  const std::vector<TensorInfo>& outputs() const { return outputs_; }

  // Host tensors in, host tensors out, through the session's own device
  // buffers.  For tests, tools and CPU-fed stages.
  virtual std::vector<Tensor> run(std::span<const Tensor> inputs) = 0;

  // An output where the backend holds it: valid while the function given to run() runs.
  struct OutputView {
    DType dtype;
    Shape shape;
    const void* data;
    template <class T>
    const T* as() const {
      return static_cast<const T*>(data);
    }
  };
  // The same run, its outputs read where the backend wrote them, not copied out: ONNX Runtime's own buffers
  // (its arena's, reused run to run), for an output read once (the recognizer's logits, about 20 MB a page, and
  // the detector's maps). Other backends pass run()'s tensors.
  virtual void run(std::span<const Tensor> inputs, const std::function<void(std::span<const OutputView>)>& read);

  // GPU sessions (TensorRT).  The output shapes for inputs of these shapes;
  // throws Error when no profile holds them.
  virtual std::vector<Shape> output_shapes(std::span<const Shape> inputs);
  // GPU sessions (TensorRT): device tensors in, device tensors out.  Queues a
  // run on `stream` (a cudaStream_t, not the default stream) and returns
  // without waiting: leave the inputs alone and the outputs unread until the
  // stream is past the run.  The outputs must have the shapes output_shapes()
  // gives.  The session's scratch memory serves all its runs, so queue them on
  // one stream.  Keep the buffers fixed from run to run: a new set of
  // addresses costs a graph capture.  Throws Error on CPU backends.
  virtual void enqueue(std::span<const DeviceTensor> inputs,
                       std::span<const DeviceTensor> outputs, void* stream);

 protected:
  Session() = default;
  // Throws Error unless the inputs match inputs(): count, element type, rank,
  // fixed dimensions, and at least one element (a run on an empty batch is a
  // caller bug: skip it).
  void check_inputs(std::span<const Tensor> inputs) const;
  void check_inputs(std::span<const DeviceTensor> inputs) const;

  std::vector<TensorInfo> inputs_, outputs_;
};

// Loads a model.  Throws Error when the backend is not built in, the file does
// not load, or (TensorRT) an input with free dimensions has no range in a
// profile.
std::unique_ptr<Session> open_session(Backend backend,
                                      const std::filesystem::path& model,
                                      const SessionOptions& options = {});

}  // namespace carbon
