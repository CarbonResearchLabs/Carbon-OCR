// The ONNX reader (onnx.h): protobuf's wire format, for the fields of
// onnx.proto3 the executor uses; every other field is skipped.

#include "onnx.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>

#include "carbon/cpu.h"
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
    case kFloat16: return 2;
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
    } else if (!int_data.empty()) {  // float16 too: its bits in int32_data
      t.bytes.resize(int_data.size() * size);
      for (size_t i = 0; i < int_data.size(); ++i) {
        const int64_t v = int_data[i];
        if (size == 8) {
          std::memcpy(t.bytes.data() + 8 * i, &v, 8);
        } else if (size == 4) {
          const int32_t v32 = static_cast<int32_t>(v);
          std::memcpy(t.bytes.data() + 4 * i, &v32, 4);
        } else if (size == 2) {
          const uint16_t v16 = static_cast<uint16_t>(v);
          std::memcpy(t.bytes.data() + 2 * i, &v16, 2);
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
  std::vector<float> out(static_cast<size_t>(count()));
  if (dtype == kFloat16) {
    for (size_t i = 0; i < out.size(); ++i) {
      uint16_t h;
      std::memcpy(&h, bytes.data() + 2 * i, 2);
      out[i] = cpu::f16_to_f32(h);
    }
    return out;
  }
  if (dtype != kFloat) throw Error("ONNX: tensor '" + name + "' is not float");
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

namespace {

// Protobuf's wire format, written: a key, then a varint or a length and its bytes.
void put_varint(std::string& out, uint64_t v) {
  while (v >= 0x80) {
    out.push_back(static_cast<char>((v & 0x7f) | 0x80));
    v >>= 7;
  }
  out.push_back(static_cast<char>(v));
}
void put_bytes(std::string& out, uint32_t field, std::string_view bytes) {
  put_varint(out, (static_cast<uint64_t>(field) << 3) | 2);
  put_varint(out, bytes.size());
  out.append(bytes);
}
void put_int(std::string& out, uint32_t field, uint64_t v) {
  put_varint(out, static_cast<uint64_t>(field) << 3);
  put_varint(out, v);
}

// A name in a message: a ValueInfoProto's is field 1, a TensorProto's field 8.
std::string name_in(std::string_view msg, uint32_t name_field) {
  Reader r(msg);
  uint32_t field, wire;
  while (r.next(field, wire)) {
    if (field == name_field && wire == 2) return text(r.bytes());
    r.skip(wire);
  }
  return {};
}

}  // namespace

std::string cut(std::string_view model, const std::string& output, int32_t dtype, const std::vector<int64_t>& dims) {
  // The nodes `output` depends on, from the graph's end back.
  const Model m = parse(model);
  std::set<std::string> needed{output};
  std::vector<bool> keep(m.graph.nodes.size(), false);
  bool made = false;
  for (size_t k = m.graph.nodes.size(); k-- > 0;) {
    const Node& n = m.graph.nodes[k];
    bool want = false;
    for (const std::string& o : n.outputs) want = want || needed.count(o);
    if (!want) continue;
    keep[k] = true;
    for (const std::string& o : n.outputs) made = made || o == output;
    for (const std::string& i : n.inputs) {
      if (!i.empty()) needed.insert(i);
    }
  }
  if (!made) throw Error("ONNX: no node makes " + output);
  // The graph again, field by field: the nodes kept, the initializers they read, the inputs, and the new output.
  std::string graph;
  std::string_view graph_bytes;
  Reader top(model);
  uint32_t field, wire;
  while (top.next(field, wire)) {
    if (field == 7 && wire == 2) graph_bytes = top.bytes();
    else top.skip(wire);
  }
  Reader g(graph_bytes);
  size_t node = 0;
  while (g.next(field, wire)) {
    if (wire != 2) {  // no graph field the cut changes is a varint
      g.skip(wire);
      continue;
    }
    const std::string_view b = g.bytes();
    switch (field) {
      case 1:  // a node
        if (keep[node++]) put_bytes(graph, 1, b);
        break;
      case 5:  // an initializer: kept where a kept node reads it
        if (needed.count(name_in(b, 8))) put_bytes(graph, 5, b);
        break;
      case 11: {  // an input; older exports list the initializers as inputs too, kept as theirs are
        const std::string name = name_in(b, 1);
        const bool init = std::any_of(m.graph.initializers.begin(), m.graph.initializers.end(),
                                      [&](const TensorData& t) { return t.name == name; });
        if (!init || needed.count(name)) put_bytes(graph, 11, b);
        break;
      }
      case 12:  // the outputs: the cut's alone, below
      case 13:  // value_info: the shapes of tensors that may be gone
        break;
      default: put_bytes(graph, field, b);
    }
  }
  std::string tensor_type, type, value;
  put_int(tensor_type, 1, static_cast<uint64_t>(dtype));  // TypeProto.Tensor: elem_type
  if (!dims.empty()) {
    std::string shape;  // TensorShapeProto: a Dimension each, a value or a name
    for (size_t d = 0; d < dims.size(); ++d) {
      std::string dim;
      if (dims[d] >= 0) put_int(dim, 1, static_cast<uint64_t>(dims[d]));
      else put_bytes(dim, 2, "d" + std::to_string(d));
      put_bytes(shape, 1, dim);
    }
    put_bytes(tensor_type, 2, shape);
  }
  put_bytes(type, 1, tensor_type);  // TypeProto: tensor_type
  put_bytes(value, 1, output);      // ValueInfoProto: name
  put_bytes(value, 2, type);        //                 type
  put_bytes(graph, 12, value);
  // The model with that graph, every other field as it was.
  std::string out;
  Reader r(model);
  while (r.next(field, wire)) {
    if (field == 7 && wire == 2) {
      r.bytes();
      put_bytes(out, 7, graph);
      continue;
    }
    switch (wire) {
      case 0: put_int(out, field, r.varint()); break;
      case 2: put_bytes(out, field, r.bytes()); break;
      default: throw Error("ONNX: a ModelProto field of wire type " + std::to_string(wire));
    }
  }
  return out;
}

std::string cut_input(std::string_view model, const std::string& input, int32_t dtype, const std::vector<int64_t>& dims) {
  // The nodes the outputs depend on, from the graph's end back, stopped at `input`.
  const Model m = parse(model);
  std::set<std::string> needed;
  for (const ValueInfo& o : m.graph.outputs) needed.insert(o.name);
  std::vector<bool> keep(m.graph.nodes.size(), false);
  bool made = false;
  for (size_t k = m.graph.nodes.size(); k-- > 0;) {
    const Node& n = m.graph.nodes[k];
    bool want = false, makes = false;
    for (const std::string& o : n.outputs) {
      want = want || (needed.count(o) && o != input);
      makes = makes || o == input;
    }
    made = made || makes;
    if (!want) continue;
    keep[k] = true;
    for (const std::string& i : n.inputs) {
      if (!i.empty()) needed.insert(i);
    }
  }
  if (!made) throw Error("ONNX: no node makes " + input);
  for (const ValueInfo& in : m.graph.inputs) {
    const bool init = std::any_of(m.graph.initializers.begin(), m.graph.initializers.end(),
                                  [&](const TensorData& t) { return t.name == in.name; });
    if (!init && in.name != input && needed.count(in.name)) {
      throw Error("ONNX: the graph still reads " + in.name + " past " + input);
    }
  }
  // The new input's ValueInfoProto: its name and type.
  std::string tensor_type, type, value;
  put_int(tensor_type, 1, static_cast<uint64_t>(dtype));
  if (!dims.empty()) {
    std::string shape;
    for (size_t d = 0; d < dims.size(); ++d) {
      std::string dim;
      if (dims[d] >= 0) put_int(dim, 1, static_cast<uint64_t>(dims[d]));
      else put_bytes(dim, 2, "d" + std::to_string(d));
      put_bytes(shape, 1, dim);
    }
    put_bytes(tensor_type, 2, shape);
  }
  put_bytes(type, 1, tensor_type);
  put_bytes(value, 1, input);
  put_bytes(value, 2, type);
  // The graph again: the nodes kept, the initializers they read, the new input first, the outputs as they were.
  std::string graph;
  std::string_view graph_bytes;
  Reader top(model);
  uint32_t field, wire;
  while (top.next(field, wire)) {
    if (field == 7 && wire == 2) graph_bytes = top.bytes();
    else top.skip(wire);
  }
  Reader g(graph_bytes);
  size_t node = 0;
  bool input_put = false;
  while (g.next(field, wire)) {
    if (wire != 2) {
      g.skip(wire);
      continue;
    }
    const std::string_view b = g.bytes();
    switch (field) {
      case 1:  // a node
        if (keep[node++]) put_bytes(graph, 1, b);
        break;
      case 5:  // an initializer a kept node reads
        if (needed.count(name_in(b, 8))) put_bytes(graph, 5, b);
        break;
      case 11: {  // the inputs: the new one in the old one's place; initializers listed as inputs kept as theirs are
        const std::string name = name_in(b, 1);
        const bool init = std::any_of(m.graph.initializers.begin(), m.graph.initializers.end(),
                                      [&](const TensorData& t) { return t.name == name; });
        if (init) {
          if (needed.count(name)) put_bytes(graph, 11, b);
        } else if (!input_put) {
          put_bytes(graph, 11, value);
          input_put = true;
        }
        break;
      }
      case 13:  // value_info: the shapes of tensors that may be gone
        break;
      default: put_bytes(graph, field, b);
    }
  }
  if (!input_put) put_bytes(graph, 11, value);
  std::string out;
  Reader r(model);
  while (r.next(field, wire)) {
    if (field == 7 && wire == 2) {
      r.bytes();
      put_bytes(out, 7, graph);
      continue;
    }
    switch (wire) {
      case 0: put_int(out, field, r.varint()); break;
      case 2: put_bytes(out, field, r.bytes()); break;
      default: throw Error("ONNX: a ModelProto field of wire type " + std::to_string(wire));
    }
  }
  return out;
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
