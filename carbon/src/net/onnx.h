// An ONNX file read for the engine's own executor (net.h): the model's graph, its
// nodes and attributes, its initializers and the shapes of its inputs and
// outputs.  A protobuf decoder for these messages only, so the executor needs no
// protobuf library and no ONNX Runtime.  External data is not read: Fullerene's
// files hold their weights.
#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace carbon::onnx {

// TensorProto.DataType
enum DataType : int32_t { kFloat = 1, kUint8 = 2, kInt8 = 3, kInt32 = 6, kInt64 = 7, kFloat16 = 10 };

struct TensorData {
  std::string name;
  int32_t dtype = 0;
  std::vector<int64_t> dims;
  std::vector<uint8_t> bytes;  // the elements, little-endian

  int64_t count() const;
  std::vector<float> floats() const;    // a float or float16 tensor (float16 widened, exactly)
  std::vector<int64_t> ints() const;    // any integer tensor
  std::vector<int8_t> int8s() const;    // an int8 tensor
  std::vector<uint8_t> uint8s() const;  // a uint8 tensor
};

struct Attribute {
  std::string name;
  float f = 0;
  int64_t i = 0;
  std::string s;
  std::vector<float> floats;
  std::vector<int64_t> ints;
  std::optional<TensorData> t;
};

struct Node {
  std::string op, name, domain;
  std::vector<std::string> inputs, outputs;  // an empty name is an absent optional input
  std::vector<Attribute> attributes;

  const Attribute* attribute(std::string_view name) const;
  int64_t i(std::string_view name, int64_t otherwise) const;
  float f(std::string_view name, float otherwise) const;
  std::string s(std::string_view name, std::string otherwise) const;
  std::vector<int64_t> ints(std::string_view name, std::vector<int64_t> otherwise) const;
};

struct ValueInfo {
  std::string name;
  int32_t dtype = 0;
  std::vector<int64_t> dims;  // -1 for a free dimension
};

struct Graph {
  std::vector<Node> nodes;
  std::vector<TensorData> initializers;
  std::vector<ValueInfo> inputs, outputs;  // the inputs without the initializers
};

struct Model {
  int64_t opset = 0;  // the default domain's
  Graph graph;
};

// Throws carbon::Error on a file that does not parse.
Model load(const std::filesystem::path& path);
Model parse(std::string_view bytes);

// A model's metadata_props (ModelProto field 14: key and value strings), its graph
// skipped: where the export writes a recognizer's crop rule (carbon.crop.*).
std::map<std::string, std::string> metadata(const std::filesystem::path& path);

// The model (a file's bytes) cut at `output`: the same ModelProto with its graph's one output that tensor
// (of `dtype`, of shape `dims`), and only the nodes it depends on and the initializers they read; every
// other field as it was. For a runtime to run part of a model, the rest done by the engine (the detector's head:
// carbon/src/det_head.cpp). Throws Error where no node makes `output`.
// dims: its shape where the runtime needs it given (the native backend does), -1 for a free dimension.
std::string cut(std::string_view model, const std::string& output, int32_t dtype = kFloat,
                const std::vector<int64_t>& dims = {});

// The model cut at its input side: `input` (of `dtype`, of shape `dims`) its one graph input, the nodes the
// outputs need with the walk back stopped there, and only the initializers they read; every other field as it was.
// For the engine to run a model's first layers itself. Throws Error where a node the outputs need still reads the
// old input (the tensor does not part the graph) or no node makes `input`.
std::string cut_input(std::string_view model, const std::string& input, int32_t dtype = kFloat,
                      const std::vector<int64_t>& dims = {});

}  // namespace carbon::onnx
