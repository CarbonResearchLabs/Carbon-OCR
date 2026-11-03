// The ONNX reader (onnx.h): protobuf's wire format, for the fields of
// onnx.proto3 the executor uses; every other field is skipped.

#include "onnx.h"

#include <cstring>
#include <fstream>
#include <iterator>

#include "carbon/error.h"

namespace carbon::onnx {
namespace {

[[noreturn]] void bad(const std::string& why) { throw Error("ONNX: " + why); }

// A protobuf message's bytes, read field by field.
class Reader {
 public:
  explicit Reader(std::string_view b)
      : p_(reinterpret_cast<const uint8_t*>(b.data())), end_(p_ + b.size()) {}

  bool at_end() const { return p_ == end_; }

  // The next field's number and wire type; false at the end.
  bool next(uint32_t& field, uint32_t& wire) {
    if (p_ == end_) return false;
    const uint64_t key = varint();
    field = static_cast<uint32_t>(key >> 3);
    wire = static_cast<uint32_t>(key & 7);
    return true;
  }

  uint64_t varint() {
    uint64_t v = 0;
    for (int shift = 0; shift < 64; shift += 7) {
      if (p_ == end_) bad("a varint runs past its message");
      const uint8_t b = *p_++;
      v |= static_cast<uint64_t>(b & 0x7f) << shift;
      if (!(b & 0x80)) return v;
    }
    bad("a varint longer than 10 bytes");
  }

  std::string_view bytes() {
    const uint64_t n = varint();
    if (n > static_cast<uint64_t>(end_ - p_)) bad("a field runs past its message");
    std::string_view out(reinterpret_cast<const char*>(p_), static_cast<size_t>(n));
    p_ += n;
    return out;
  }

  uint32_t fixed32() {
    if (end_ - p_ < 4) bad("a fixed32 runs past its message");
    uint32_t v;
    std::memcpy(&v, p_, 4);
    p_ += 4;
    return v;
  }

  void skip(uint32_t wire) {
    switch (wire) {
      case 0: varint(); break;
      case 1:
        if (end_ - p_ < 8) bad("a fixed64 runs past its message");
        p_ += 8;
        break;
      case 2: bytes(); break;
      case 5: fixed32(); break;
      default: bad("wire type " + std::to_string(wire) + " (groups are not read)");
    }
  }

 private:
  const uint8_t* p_;
  const uint8_t* end_;
};

float as_float(uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// A repeated varint field, packed (wire 2: varints end to end) or one value a
// field (wire 0).
void read_ints(Reader& r, uint32_t wire, std::vector<int64_t>& out) {
  if (wire == 2) {
    Reader packed(r.bytes());
    while (!packed.at_end()) out.push_back(static_cast<int64_t>(packed.varint()));
    return;
  }
  if (wire != 0) bad("an integer field of wire type " + std::to_string(wire));
  out.push_back(static_cast<int64_t>(r.varint()));
}

// A repeated float field, packed (wire 2) or one value a field (wire 5).
void read_floats(Reader& r, uint32_t wire, std::vector<float>& out) {
  if (wire == 2) {
    const std::string_view b = r.bytes();
    if (b.size() % 4) bad("a packed float field of " + std::to_string(b.size()) + " bytes");
    const size_t at = out.size();
    out.resize(at + b.size() / 4);
    std::memcpy(out.data() + at, b.data(), b.size());
    return;
  }
  if (wire != 5) bad("a float field of wire type " + std::to_string(wire));
  out.push_back(as_float(r.fixed32()));
}

std::string text(std::string_view b) { return std::string(b); }

size_t element_size(int32_t dtype) {
  switch (dtype) {
    case kFloat:
    case kInt32: return 4;
    case kUint8:
    case kInt8: return 1;
    case kInt64: return 8;
    default: return 0;
  }
}

TensorData read_tensor(std::string_view msg) {
  TensorData t;
  Reader r(msg);
  std::vector<float> float_data;
  std::vector<int64_t> int_data;  // int32_data and int64_data
  bool external = false;
  uint32_t field, wire;
  while (r.next(field, wire)) {
    switch (field) {
      case 1: read_ints(r, wire, t.dims); break;
      case 2: t.dtype = static_cast<int32_t>(r.varint()); break;
      case 4: read_floats(r, wire, float_data); break;
      case 5:
      case 7: read_ints(r, wire, int_data); break;
      case 8: t.name = text(r.bytes()); break;
      case 9: {
        const std::string_view b = r.bytes();
        t.bytes.assign(b.begin(), b.end());
        break;
      }
      case 14: external = r.varint() == 1; break;
      default: r.skip(wire);
    }
  }
  if (external) bad("tensor '" + t.name + "' keeps its data in another file");
  const size_t size = element_size(t.dtype);
  if (!size) bad("tensor '" + t.name + "' has data type " + std::to_string(t.dtype));
  if (t.bytes.empty()) {
    if (!float_data.empty()) {
      if (t.dtype != kFloat) bad("tensor '" + t.name + "' has float_data but is not float");
      t.bytes.resize(float_data.size() * 4);
      std::memcpy(t.bytes.data(), float_data.data(), t.bytes.size());
    } else if (!int_data.empty()) {
      t.bytes.resize(int_data.size() * size);
      for (size_t i = 0; i < int_data.size(); ++i) {
        const int64_t v = int_data[i];
        if (size == 8) {
          std::memcpy(t.bytes.data() + 8 * i, &v, 8);
        } else if (size == 4) {
          const int32_t v32 = static_cast<int32_t>(v);
          std::memcpy(t.bytes.data() + 4 * i, &v32, 4);
        } else {
          t.bytes[i] = static_cast<uint8_t>(v);
        }
      }
    }
  }
  if (t.bytes.size() != static_cast<size_t>(t.count()) * size) {
    bad("tensor '" + t.name + "' holds " + std::to_string(t.bytes.size()) + " bytes for " +
        std::to_string(t.count()) + " elements");
  }
  return t;
}

Attribute read_attribute(std::string_view msg) {
  Attribute a;
  Reader r(msg);
  uint32_t field, wire;
  while (r.next(field, wire)) {
    switch (field) {
      case 1: a.name = text(r.bytes()); break;
      case 2: a.f = as_float(r.fixed32()); break;
      case 3: a.i = static_cast<int64_t>(r.varint()); break;
      case 4: a.s = text(r.bytes()); break;
      case 5: a.t = read_tensor(r.bytes()); break;
      case 7: read_floats(r, wire, a.floats); break;
      case 8: read_ints(r, wire, a.ints); break;
      default: r.skip(wire);
    }
  }
  return a;
}

Node read_node(std::string_view msg) {
  Node n;
  Reader r(msg);
  uint32_t field, wire;
  while (r.next(field, wire)) {
    switch (field) {
      case 1: n.inputs.push_back(text(r.bytes())); break;
      case 2: n.outputs.push_back(text(r.bytes())); break;
      case 3: n.name = text(r.bytes()); break;
      case 4: n.op = text(r.bytes()); break;
      case 5: n.attributes.push_back(read_attribute(r.bytes())); break;
      case 7: n.domain = text(r.bytes()); break;
      default: r.skip(wire);
    }
  }
  return n;
}

// TensorShapeProto.Dimension: a value, or a name (free).
int64_t read_dim(std::string_view msg) {
  Reader r(msg);
  int64_t v = -1;
  uint32_t field, wire;
  while (r.next(field, wire)) {
    if (field == 1 && wire == 0) {
      v = static_cast<int64_t>(r.varint());
    } else {
      r.skip(wire);
    }
  }
  return v;
}

ValueInfo read_value_info(std::string_view msg) {
  ValueInfo v;
  Reader r(msg);
  uint32_t field, wire;
  while (r.next(field, wire)) {
    if (field == 1) {
      v.name = text(r.bytes());
    } else if (field == 2) {  // TypeProto
      Reader type(r.bytes());
      uint32_t f2, w2;
      while (type.next(f2, w2)) {
        if (f2 != 1) {  // tensor_type
          type.skip(w2);
          continue;
        }
        Reader tensor(type.bytes());
        uint32_t f3, w3;
        while (tensor.next(f3, w3)) {
          if (f3 == 1) {
            v.dtype = static_cast<int32_t>(tensor.varint());
          } else if (f3 == 2) {  // TensorShapeProto
            Reader shape(tensor.bytes());
            uint32_t f4, w4;
            while (shape.next(f4, w4)) {
              if (f4 == 1) {
                v.dims.push_back(read_dim(shape.bytes()));
              } else {
                shape.skip(w4);
              }
            }
          } else {
            tensor.skip(w3);
          }
        }
      }
    } else {
      r.skip(wire);
    }
  }
  return v;
}

Graph read_graph(std::string_view msg) {
  Graph g;
  Reader r(msg);
  std::vector<ValueInfo> inputs;
  uint32_t field, wire;
  while (r.next(field, wire)) {
    switch (field) {
      case 1: g.nodes.push_back(read_node(r.bytes())); break;
      case 5: g.initializers.push_back(read_tensor(r.bytes())); break;
      case 11: inputs.push_back(read_value_info(r.bytes())); break;
      case 12: g.outputs.push_back(read_value_info(r.bytes())); break;
      default: r.skip(wire);
    }
  }
  for (ValueInfo& v : inputs) {  // older exports list the initializers as inputs too
    bool init = false;
    for (const TensorData& t : g.initializers) init = init || t.name == v.name;
    if (!init) g.inputs.push_back(std::move(v));
  }
  return g;
}

}  // namespace

int64_t TensorData::count() const {
  int64_t n = 1;
  for (int64_t d : dims) n *= d;
  return n;
}

std::vector<float> TensorData::floats() const {
  if (dtype != kFloat) throw Error("ONNX: tensor '" + name + "' is not float");
  std::vector<float> out(static_cast<size_t>(count()));
  std::memcpy(out.data(), bytes.data(), bytes.size());
  return out;
}

std::vector<int64_t> TensorData::ints() const {
  std::vector<int64_t> out(static_cast<size_t>(count()));
  for (size_t i = 0; i < out.size(); ++i) {
    switch (dtype) {
      case kInt64: std::memcpy(&out[i], bytes.data() + 8 * i, 8); break;
      case kInt32: {
        int32_t v;
        std::memcpy(&v, bytes.data() + 4 * i, 4);
        out[i] = v;
        break;
      }
      case kInt8: out[i] = static_cast<int8_t>(bytes[i]); break;
      case kUint8: out[i] = bytes[i]; break;
      default: throw Error("ONNX: tensor '" + name + "' is not an integer tensor");
    }
  }
  return out;
}

std::vector<int8_t> TensorData::int8s() const {
  if (dtype != kInt8) throw Error("ONNX: tensor '" + name + "' is not int8");
  return std::vector<int8_t>(reinterpret_cast<const int8_t*>(bytes.data()),
                             reinterpret_cast<const int8_t*>(bytes.data()) + bytes.size());
}

std::vector<uint8_t> TensorData::uint8s() const {
  if (dtype != kUint8) throw Error("ONNX: tensor '" + name + "' is not uint8");
  return bytes;
}

const Attribute* Node::attribute(std::string_view n) const {
  for (const Attribute& a : attributes) {
    if (a.name == n) return &a;
  }
  return nullptr;
}

int64_t Node::i(std::string_view n, int64_t otherwise) const {
  const Attribute* a = attribute(n);
  return a ? a->i : otherwise;
}

float Node::f(std::string_view n, float otherwise) const {
  const Attribute* a = attribute(n);
  return a ? a->f : otherwise;
}

std::string Node::s(std::string_view n, std::string otherwise) const {
  const Attribute* a = attribute(n);
  return a ? a->s : otherwise;
}

std::vector<int64_t> Node::ints(std::string_view n, std::vector<int64_t> otherwise) const {
  const Attribute* a = attribute(n);
  return a ? a->ints : otherwise;
}

Model parse(std::string_view bytes) {
  Model m;
  Reader r(bytes);
  bool graph = false;
  uint32_t field, wire;
  while (r.next(field, wire)) {
    if (field == 7) {
      m.graph = read_graph(r.bytes());
      graph = true;
    } else if (field == 8) {  // OperatorSetIdProto
      Reader set(r.bytes());
      std::string domain;
      int64_t version = 0;
      uint32_t f2, w2;
      while (set.next(f2, w2)) {
        if (f2 == 1) {
          domain = text(set.bytes());
        } else if (f2 == 2) {
          version = static_cast<int64_t>(set.varint());
        } else {
          set.skip(w2);
        }
      }
      if (domain.empty() || domain == "ai.onnx") m.opset = version;
    } else {
      r.skip(wire);
    }
  }
  if (!graph) bad("no graph");
  return m;
}

Model load(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw Error("cannot read " + path.string());
  const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  try {
    return parse(bytes);
  } catch (const Error& e) {
    throw Error(path.string() + ": " + e.what());
  }
}

std::map<std::string, std::string> metadata(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw Error("cannot read " + path.string());
  const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::map<std::string, std::string> out;
  try {
    Reader r(bytes);
    uint32_t field, wire;
    while (r.next(field, wire)) {
      if (field != 14 || wire != 2) {
        r.skip(wire);  // the graph's bytes passed over, not copied
        continue;
      }
      Reader entry(r.bytes());
      std::string key, value;
      uint32_t f2, w2;
      while (entry.next(f2, w2)) {
        if (f2 == 1 && w2 == 2) {
          key = text(entry.bytes());
        } else if (f2 == 2 && w2 == 2) {
          value = text(entry.bytes());
        } else {
          entry.skip(w2);
        }
      }
      out[key] = value;
    }
  } catch (const Error& e) {
    throw Error(path.string() + ": " + e.what());
  }
  return out;
}

}  // namespace carbon::onnx
