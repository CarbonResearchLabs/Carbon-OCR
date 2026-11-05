// The TensorRT backend: an engine built from the ONNX file (or loaded from the
// engine cache) with an optimization profile per bucket, an execution context
// per profile, and one scratch buffer the contexts share.  Runs of an exact
// profile are captured as CUDA graphs and replayed.
//
// TensorRT 11 builds strongly typed networks only, so the ONNX file decides
// the precision: export FP16 (or Q/DQ for INT8) to run in it.

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "../backends.h"

namespace carbon::detail {

namespace {

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
    if (severity <= Severity::kWARNING) std::fprintf(stderr, "[TensorRT] %s\n", msg);
  }
};

Logger& logger() {
  static Logger l;
  return l;
}

void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw Error(std::string(what) + ": " + cudaGetErrorString(e));
  }
}

DType from_trt(nvinfer1::DataType t, const std::string& name) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
      return DType::kFloat32;
    case nvinfer1::DataType::kHALF:
      return DType::kFloat16;
    case nvinfer1::DataType::kINT32:
      return DType::kInt32;
    case nvinfer1::DataType::kINT64:
      return DType::kInt64;
    case nvinfer1::DataType::kUINT8:
      return DType::kUInt8;
    case nvinfer1::DataType::kBOOL:
      return DType::kBool;
    default:
      throw Error("'" + name + "': TensorRT data type " +
                  std::to_string(static_cast<int>(t)) + " is not supported");
  }
}

nvinfer1::Dims to_dims(const Shape& s) {
  if (s.size() > static_cast<size_t>(nvinfer1::Dims::MAX_DIMS)) {
    throw Error("shape " + shape_string(s) + " has more dimensions than TensorRT allows");
  }
  nvinfer1::Dims d{};
  d.nbDims = static_cast<int32_t>(s.size());
  for (size_t i = 0; i < s.size(); ++i) d.d[i] = s[i];
  return d;
}

Shape from_dims(const nvinfer1::Dims& d) {
  return Shape(d.d, d.d + std::max(d.nbDims, 0));
}

// FNV-1a: the cache key has to be stable across runs, not secret.
// A model file of FP16 weights and tensors (the export's <model>.fp16.onnx, <rec>.features.fp16.onnx): TF32
// has nothing to change in its engine. Every other file (FP32, int8 Q/DQ) is built without TF32.
bool fp16_file(const std::filesystem::path& model) { return model.filename().string().find(".fp16.") != std::string::npos; }

uint64_t fnv1a(std::string_view bytes, uint64_t h = 14695981039346656037ull) {
  for (unsigned char c : bytes) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

std::string read_file(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw Error("cannot read " + p.string());
  std::ostringstream s;
  s << f.rdbuf();
  return std::move(s).str();
}

// Written under a temporary name, then renamed: a crash never leaves a
// truncated engine for the next open to load.
void write_file(const std::filesystem::path& p, std::string_view data) {
  std::filesystem::create_directories(p.parent_path());
  auto tmp = p;
  tmp += ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary);
    f.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!f) throw Error("cannot write " + tmp.string());
  }
  std::filesystem::rename(tmp, p);
}

const ShapeRange* find_range(const Profile& p, std::string_view name) {
  for (const auto& r : p) {
    if (r.input == name) return &r;
  }
  return nullptr;
}

bool inside(const Shape& s, const Shape& lo, const Shape& hi) {
  if (s.size() != lo.size() || s.size() != hi.size()) return false;
  for (size_t d = 0; d < s.size(); ++d) {
    if (s[d] < lo[d] || s[d] > hi[d]) return false;
  }
  return true;
}

std::string shapes_string(std::span<const Shape> shapes) {
  std::string out;
  for (const auto& s : shapes) out += (out.empty() ? "" : " ") + shape_string(s);
  return out;
}

// A file name part from a GPU name: "NVIDIA GeForce RTX 4060" -> "NVIDIA-GeForce-RTX-4060".
std::string file_part(std::string_view name) {
  std::string out;
  for (char c : name) {
    const bool keep = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    if (keep) {
      out += c;
    } else if (!out.empty() && out.back() != '-') {
      out += '-';
    }
  }
  while (!out.empty() && out.back() == '-') out.pop_back();
  return out;
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes) : bytes_(bytes) {
    if (bytes) cuda_check(cudaMalloc(&ptr_, bytes), "cudaMalloc");
  }
  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_) {
    o.ptr_ = nullptr;
    o.bytes_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    std::swap(ptr_, o.ptr_);
    std::swap(bytes_, o.bytes_);
    return *this;
  }
  ~DeviceBuffer() {
    if (ptr_) cudaFree(ptr_);
  }
  void* get() const { return ptr_; }
  size_t bytes() const { return bytes_; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

// Created by create(), after cudaSetDevice: a stream belongs to the device
// current when it is made.
class Stream {
 public:
  Stream() = default;
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  ~Stream() {
    if (s_) cudaStreamDestroy(s_);
  }
  void create() { cuda_check(cudaStreamCreate(&s_), "cudaStreamCreate"); }
  cudaStream_t get() const { return s_; }

 private:
  cudaStream_t s_ = nullptr;
};

// One run captured as a CUDA graph, for one set of tensor addresses.
class Graph {
 public:
  Graph(std::vector<void*> addresses, cudaGraphExec_t exec)
      : addresses_(std::move(addresses)), exec_(exec) {}
  Graph(Graph&& o) noexcept : addresses_(std::move(o.addresses_)), exec_(o.exec_) {
    o.exec_ = nullptr;
  }
  Graph& operator=(Graph&& o) noexcept {
    std::swap(addresses_, o.addresses_);
    std::swap(exec_, o.exec_);
    return *this;
  }
  ~Graph() {
    if (exec_) cudaGraphExecDestroy(exec_);
  }
  const std::vector<void*>& addresses() const { return addresses_; }
  cudaGraphExec_t get() const { return exec_; }

 private:
  std::vector<void*> addresses_;
  cudaGraphExec_t exec_ = nullptr;
};

// An optimization profile and the context that runs it.
struct Context {
  int profile = 0;
  std::vector<Shape> min, max;  // per input; a fixed input's own shape
  bool exact = false;           // one shape for every input: runs are graphs
  std::unique_ptr<nvinfer1::IExecutionContext> ctx;
  std::vector<Shape> shapes;     // the input shapes set on ctx
  std::vector<void*> addresses;  // the addresses set on ctx: inputs, then outputs
  bool capture = true;           // false once a capture fails: plain enqueues
  std::vector<Graph> graphs;     // destroyed before ctx
};

// Graphs kept per context.  A pipeline reuses a few buffers per bucket; more
// means addresses that change every run, where capturing does not pay.
constexpr size_t kMaxGraphs = 8;

class TrtSession final : public Session {
 public:
  TrtSession(const std::filesystem::path& model, const SessionOptions& options)
      : graphs_(options.cuda_graphs) {
    cuda_check(cudaSetDevice(options.device), "cudaSetDevice");
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, options.device), "cudaGetDeviceProperties");
    stream_.create();

    const std::string onnx = read_file(model);
    std::filesystem::path cached, timing;
    if (!options.engine_cache.empty()) {
      std::string key = std::to_string(getInferLibVersion()) + "|" + prop.name + "|" +
                        std::to_string(prop.major) + "." + std::to_string(prop.minor) +
                        "|" + std::to_string(options.workspace_bytes) + "|O" +
                        std::to_string(options.optimization_level);
      if (!fp16_file(model)) key += "|no-tf32";  // FP32's engines built without TF32 (build); FP16's keys as before
      for (const auto& p : options.profiles) {
        key += "|profile";
        for (const auto& r : p) {
          key += "|" + r.input + shape_string(r.min) + shape_string(r.opt) + shape_string(r.max);
        }
      }
      char hex[17];
      std::snprintf(hex, sizeof hex, "%016llx",
                    static_cast<unsigned long long>(fnv1a(key, fnv1a(onnx))));
      cached = options.engine_cache / (model.stem().string() + "-" + hex + ".engine");
      timing = options.engine_cache / ("timing-trt" + std::to_string(getInferLibVersion()) + "-" +
                                       file_part(prop.name) + ".cache");
    }

    runtime_.reset(nvinfer1::createInferRuntime(logger()));
    if (!runtime_) throw Error("TensorRT: createInferRuntime failed");
    if (!cached.empty() && std::filesystem::exists(cached)) {
      const std::string plan = read_file(cached);
      engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
    }
    if (!engine_) {  // no cache, or a stale one
      const std::string plan = build(onnx, model, options, timing);
      engine_.reset(runtime_->deserializeCudaEngine(plan.data(), plan.size()));
      if (!engine_) throw Error("TensorRT could not load the engine it built for " + model.string());
      if (!cached.empty()) write_file(cached, plan);
    }

    for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
      const char* name = engine_->getIOTensorName(i);
      TensorInfo info{name, from_trt(engine_->getTensorDataType(name), name),
                      from_dims(engine_->getTensorShape(name))};
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        inputs_.push_back(std::move(info));
      } else {
        outputs_.push_back(std::move(info));
      }
    }

    // A context per profile, all in one scratch buffer: the runs of a session
    // are serial on one stream, so no two use it at once.
    const int64_t scratch = engine_->getDeviceMemorySizeV2();
    scratch_ = DeviceBuffer(static_cast<size_t>(scratch));
    contexts_.reserve(static_cast<size_t>(engine_->getNbOptimizationProfiles()));
    for (int32_t p = 0; p < engine_->getNbOptimizationProfiles(); ++p) {
      Context& c = contexts_.emplace_back();
      c.profile = p;
      c.ctx.reset(engine_->createExecutionContext(
          nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
      if (!c.ctx) throw Error("TensorRT: createExecutionContext failed");
      c.ctx->setDeviceMemoryV2(scratch_.get(), scratch);
      if (!c.ctx->setOptimizationProfileAsync(p, stream_.get())) {
        throw Error("TensorRT: cannot select optimization profile " + std::to_string(p));
      }
      c.exact = true;
      for (const auto& t : inputs_) {
        using nvinfer1::OptProfileSelector;
        Shape lo = t.shape, hi = t.shape;  // a fixed shape has no range
        if (std::any_of(t.shape.begin(), t.shape.end(), [](int64_t d) { return d < 0; })) {
          lo = from_dims(engine_->getProfileShape(t.name.c_str(), p, OptProfileSelector::kMIN));
          hi = from_dims(engine_->getProfileShape(t.name.c_str(), p, OptProfileSelector::kMAX));
        }
        c.exact = c.exact && lo == hi;
        c.min.push_back(std::move(lo));
        c.max.push_back(std::move(hi));
      }
      set_shapes(c, c.max);
      for (const auto& t : outputs_) {
        for (int64_t d : from_dims(c.ctx->getTensorShape(t.name.c_str()))) {
          if (d < 0) throw Error("output '" + t.name + "' has a data-dependent shape: not supported");
        }
      }
    }
    cuda_check(cudaStreamSynchronize(stream_.get()), "select the profiles");
  }

  Backend backend() const override { return Backend::kTensorRT; }
  using Session::run;  // and run() with its outputs in place: the base's, over this one

  std::vector<Tensor> run(std::span<const Tensor> inputs) override {
    check_inputs(inputs);
    std::vector<Shape> shapes;
    for (const auto& t : inputs) shapes.push_back(t.shape());
    Context& c = pick(shapes);
    if (in_buf_.empty()) allocate_run_buffers();
    set_shapes(c, shapes);
    std::vector<void*> addresses;
    for (size_t i = 0; i < inputs.size(); ++i) {
      cuda_check(cudaMemcpyAsync(in_buf_[i].get(), inputs[i].raw(), inputs[i].bytes(),
                                 cudaMemcpyHostToDevice, stream_.get()),
                 "copy to device");
      addresses.push_back(in_buf_[i].get());
    }
    for (const auto& b : out_buf_) addresses.push_back(b.get());
    launch(c, addresses, stream_.get());

    std::vector<Tensor> result;
    result.reserve(outputs_.size());
    for (size_t i = 0; i < outputs_.size(); ++i) {
      Tensor t = Tensor::empty(outputs_[i].dtype,
                               from_dims(c.ctx->getTensorShape(outputs_[i].name.c_str())));
      if (t.bytes()) {
        cuda_check(cudaMemcpyAsync(t.raw(), out_buf_[i].get(), t.bytes(),
                                   cudaMemcpyDeviceToHost, stream_.get()),
                   "copy to host");
      }
      result.push_back(std::move(t));
    }
    cuda_check(cudaStreamSynchronize(stream_.get()), "cudaStreamSynchronize");
    return result;
  }

  std::vector<Shape> output_shapes(std::span<const Shape> inputs) override {
    if (inputs.size() != inputs_.size()) {
      throw Error("model takes " + std::to_string(inputs_.size()) + " inputs, got " +
                  std::to_string(inputs.size()));
    }
    Context& c = pick(inputs);
    set_shapes(c, inputs);
    std::vector<Shape> out;
    for (const auto& t : outputs_) out.push_back(from_dims(c.ctx->getTensorShape(t.name.c_str())));
    return out;
  }

  void enqueue(std::span<const DeviceTensor> inputs, std::span<const DeviceTensor> outputs,
               void* stream) override {
    check_inputs(inputs);
    if (!stream) {
      throw Error("enqueue needs a stream of the caller's: on the default stream TensorRT "
                  "synchronizes the device, and a run cannot be captured");
    }
    if (outputs.size() != outputs_.size()) {
      throw Error("model has " + std::to_string(outputs_.size()) + " outputs, got " +
                  std::to_string(outputs.size()));
    }
    std::vector<Shape> shapes;
    std::vector<void*> addresses;
    for (const auto& t : inputs) {
      shapes.push_back(t.shape);
      addresses.push_back(t.data);
    }
    Context& c = pick(shapes);
    set_shapes(c, shapes);
    for (size_t i = 0; i < outputs.size(); ++i) {
      const Shape want = from_dims(c.ctx->getTensorShape(outputs_[i].name.c_str()));
      if (outputs[i].dtype != outputs_[i].dtype || outputs[i].shape != want) {
        throw Error("output '" + outputs_[i].name + "': this run gives " +
                    std::string(dtype_name(outputs_[i].dtype)) + " " + shape_string(want) +
                    ", got a buffer for " + std::string(dtype_name(outputs[i].dtype)) + " " +
                    shape_string(outputs[i].shape));
      }
      addresses.push_back(outputs[i].data);
    }
    launch(c, addresses, static_cast<cudaStream_t>(stream));
  }

 private:
  // The first profile exact for these shapes, else the first that holds them.
  Context& pick(std::span<const Shape> shapes) {
    Context* holds = nullptr;
    for (Context& c : contexts_) {
      bool in = true, same = true;
      for (size_t i = 0; i < shapes.size(); ++i) {
        in = in && inside(shapes[i], c.min[i], c.max[i]);
        same = same && shapes[i] == c.min[i];
      }
      if (in && same && c.exact) return c;
      if (in && !holds) holds = &c;
    }
    if (!holds) {
      throw Error("input shapes " + shapes_string(shapes) +
                  " are outside every profile of the engine (SessionOptions::profiles)");
    }
    return *holds;
  }

  void set_shapes(Context& c, std::span<const Shape> shapes) {
    if (std::equal(shapes.begin(), shapes.end(), c.shapes.begin(), c.shapes.end())) return;
    for (size_t i = 0; i < shapes.size(); ++i) {
      if (!c.ctx->setInputShape(inputs_[i].name.c_str(), to_dims(shapes[i]))) {
        c.shapes.clear();
        throw Error("input '" + inputs_[i].name + "': TensorRT rejected shape " +
                    shape_string(shapes[i]));
      }
    }
    c.shapes.assign(shapes.begin(), shapes.end());
  }

  // Runs c on the tensors at `addresses` (inputs, then outputs) on `stream`:
  // a replay when c has a graph for them, else an enqueue, captured for the
  // next run when c is exact.
  void launch(Context& c, const std::vector<void*>& addresses, cudaStream_t stream) {
    const bool graphs = graphs_ && c.exact && c.capture;
    if (graphs) {
      for (const Graph& g : c.graphs) {
        if (g.addresses() == addresses) {
          cuda_check(cudaGraphLaunch(g.get(), stream), "cudaGraphLaunch");
          return;
        }
      }
    }
    if (c.addresses != addresses) {
      for (size_t i = 0; i < addresses.size(); ++i) {
        const char* name = i < inputs_.size() ? inputs_[i].name.c_str()
                                              : outputs_[i - inputs_.size()].name.c_str();
        if (!c.ctx->setTensorAddress(name, addresses[i])) {
          c.addresses.clear();
          throw Error(std::string("TensorRT rejected the address of '") + name + "'");
        }
      }
      c.addresses = addresses;
    }
    // This enqueue is the run.  It is also the one TensorRT wants after new
    // shapes or addresses, before a capture.
    if (!c.ctx->enqueueV3(stream)) throw Error("TensorRT: enqueueV3 failed");
    if (graphs) capture(c, addresses, stream);
  }

  void capture(Context& c, const std::vector<void*>& addresses, cudaStream_t stream) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    bool ok = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
    if (ok) {
      const bool queued = c.ctx->enqueueV3(stream);
      ok = cudaStreamEndCapture(stream, &graph) == cudaSuccess && queued && graph;
    }
    ok = ok && cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
    if (graph) cudaGraphDestroy(graph);
    if (!ok) {
      (void)cudaGetLastError();  // the failed capture's error, not the next call's
      c.capture = false;
      std::fprintf(stderr, "[carbon] profile %d: CUDA graph capture failed; enqueueing instead\n",
                   c.profile);
      return;
    }
    if (c.graphs.size() == kMaxGraphs) c.graphs.erase(c.graphs.begin());
    c.graphs.emplace_back(addresses, exec);
  }

  // run()'s buffers, sized for the largest shapes of every profile and made
  // on the first run, so an enqueue-only pipeline never pays for them.  Memory
  // is fixed after (GPU memory that grows with every page).
  void allocate_run_buffers() {
    std::vector<size_t> in(inputs_.size()), out(outputs_.size());
    for (Context& c : contexts_) {
      const std::vector<Shape> was = c.shapes;
      set_shapes(c, c.max);
      for (size_t i = 0; i < inputs_.size(); ++i) {
        in[i] = std::max(in[i], static_cast<size_t>(element_count(c.max[i])) *
                                    dtype_size(inputs_[i].dtype));
      }
      for (size_t i = 0; i < outputs_.size(); ++i) {
        const Shape s = from_dims(c.ctx->getTensorShape(outputs_[i].name.c_str()));
        out[i] = std::max(out[i], static_cast<size_t>(element_count(s)) *
                                      dtype_size(outputs_[i].dtype));
      }
      set_shapes(c, was);
    }
    for (size_t b : in) in_buf_.emplace_back(b);
    for (size_t b : out) out_buf_.emplace_back(b);
  }

  static std::string build(const std::string& onnx, const std::filesystem::path& model,
                           const SessionOptions& options, const std::filesystem::path& timing) {
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger()));
    if (!builder) throw Error("TensorRT: createInferBuilder failed");
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger()));
    const std::string path = model.string();  // for external weights next to the model
    if (!parser->parse(onnx.data(), onnx.size(), path.c_str())) {
      std::string why;
      for (int i = 0; i < parser->getNbErrors(); ++i) {
        why += std::string("\n  ") + parser->getError(i)->desc();
      }
      throw Error("TensorRT could not parse " + path + ":" + why);
    }

    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, options.workspace_bytes);
    config->setBuilderOptimizationLevel(options.optimization_level);
    // FP32 that is FP32: TensorRT lets kFLOAT tensors' convolutions and matrix products run in TF32 (10-bit
    // mantissas) unless kTF32 is cleared, so an FP32 file's engine would not read as the CPU's FP32 does.
    if (!fp16_file(model)) config->clearFlag(nvinfer1::BuilderFlag::kTF32);

    std::vector<nvinfer1::ITensor*> free_inputs;
    for (int32_t i = 0; i < network->getNbInputs(); ++i) {
      nvinfer1::ITensor* in = network->getInput(i);
      const Shape dims = from_dims(in->getDimensions());
      if (std::any_of(dims.begin(), dims.end(), [](int64_t d) { return d < 0; })) {
        free_inputs.push_back(in);
      }
    }
    for (const Profile& p : options.profiles) {
      for (const auto& r : p) {
        bool known = false;
        for (int32_t i = 0; i < network->getNbInputs(); ++i) {
          known = known || r.input == network->getInput(i)->getName();
        }
        if (!known) {
          throw Error("a profile gives a range for '" + r.input + "', not an input of " + path);
        }
      }
    }
    if (!free_inputs.empty() && options.profiles.empty()) {
      throw Error("input '" + std::string(free_inputs[0]->getName()) + "' has free dimensions " +
                  shape_string(from_dims(free_inputs[0]->getDimensions())) +
                  ": give SessionOptions::profiles a range for it");
    }
    for (size_t k = 0; !free_inputs.empty() && k < options.profiles.size(); ++k) {
      nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
      for (nvinfer1::ITensor* in : free_inputs) {
        const ShapeRange* r = find_range(options.profiles[k], in->getName());
        if (!r) {
          throw Error("profile " + std::to_string(k) + " has no range for input '" +
                      in->getName() + "', which has free dimensions " +
                      shape_string(from_dims(in->getDimensions())));
        }
        using nvinfer1::OptProfileSelector;
        if (!profile->setDimensions(in->getName(), OptProfileSelector::kMIN, to_dims(r->min)) ||
            !profile->setDimensions(in->getName(), OptProfileSelector::kOPT, to_dims(r->opt)) ||
            !profile->setDimensions(in->getName(), OptProfileSelector::kMAX, to_dims(r->max))) {
          throw Error("input '" + r->input + "': range " + shape_string(r->min) + " .. " +
                      shape_string(r->max) + " does not fit the model's shape " +
                      shape_string(from_dims(in->getDimensions())));
        }
      }
      config->addOptimizationProfile(profile);
    }

    // The timing cache: kernel timings from earlier builds, so a rebuild (a
    // new profile, a retrained model) skips most of the measuring.
    std::unique_ptr<nvinfer1::ITimingCache> cache;
    if (!timing.empty()) {
      const std::string blob = std::filesystem::exists(timing) ? read_file(timing) : std::string();
      cache.reset(config->createTimingCache(blob.data(), blob.size()));
      if (!cache || !config->setTimingCache(*cache, false)) {  // another GPU's or version's
        cache.reset(config->createTimingCache(nullptr, 0));
        if (cache) config->setTimingCache(*cache, false);
      }
    }

    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) throw Error("TensorRT could not build an engine for " + path);
    if (cache) {
      std::unique_ptr<nvinfer1::IHostMemory> t(config->getTimingCache()->serialize());
      if (t) write_file(timing, std::string_view(static_cast<const char*>(t->data()), t->size()));
    }
    return std::string(static_cast<const char*>(plan->data()), plan->size());
  }

  // Members are destroyed in reverse order: graphs and contexts before the
  // scratch they run in and the engine they came from, the engine before the
  // runtime.
  bool graphs_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  DeviceBuffer scratch_;
  std::vector<Context> contexts_;
  Stream stream_;
  std::vector<DeviceBuffer> in_buf_, out_buf_;
};

}  // namespace

std::unique_ptr<Session> open_trt_session(const std::filesystem::path& model,
                                          const SessionOptions& options) {
  return std::make_unique<TrtSession>(model, options);
}

}  // namespace carbon::detail
