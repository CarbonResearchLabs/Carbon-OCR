// Dense tensors in host memory: what model sessions take and return.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "carbon/error.h"

namespace carbon {

enum class DType : uint8_t { kFloat32, kFloat16, kInt32, kInt64, kUInt8, kBool };

size_t dtype_size(DType dtype);
std::string_view dtype_name(DType dtype);

// Dimensions, outermost first.  In a TensorInfo, -1 marks a dimension the
// model leaves free (batch, line width).
using Shape = std::vector<int64_t>;

// Number of elements (1 for a scalar, 0 if any dimension is 0).  Throws Error
// on a negative dimension.
int64_t element_count(const Shape& shape);
std::string shape_string(const Shape& shape);  // "[1, 1, 32, -1]"

// C++ element types for Tensor::data<T>().  kFloat16 and kBool have none:
// read them through raw().
template <class T>
struct dtype_of;
template <>
struct dtype_of<float> {
  static constexpr DType value = DType::kFloat32;
};
template <>
struct dtype_of<int32_t> {
  static constexpr DType value = DType::kInt32;
};
template <>
struct dtype_of<int64_t> {
  static constexpr DType value = DType::kInt64;
};
template <>
struct dtype_of<uint8_t> {
  static constexpr DType value = DType::kUInt8;
};

// A row-major tensor in host memory, 64-byte aligned (a cache line, and the
// widest SIMD load).  Movable, not copyable: page and line batches are large,
// so copies are explicit (clone()).  A large tensor's memory (256 KB and up) goes
// back to its thread's cache when the tensor goes, for the next tensor that thread
// makes: on Windows a fresh block of that size is the system's, first touched at
// about half a millisecond a megabyte, and a page made 46-55 MB of them.
class Tensor {
 public:
  Tensor() = default;
  Tensor(DType dtype, Shape shape);  // zero-filled

  // Uninitialized memory, for buffers that are written whole right away.
  static Tensor empty(DType dtype, Shape shape);

  template <class T>
  static Tensor from(Shape shape, std::span<const T> values) {
    Tensor t = empty(dtype_of<T>::value, std::move(shape));
    if (static_cast<int64_t>(values.size()) != t.size_) {
      throw Error("Tensor::from: " + std::to_string(values.size()) +
                  " values for shape " + shape_string(t.shape_));
    }
    if (!values.empty()) std::memcpy(t.raw(), values.data(), t.bytes());
    return t;
  }

  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(Tensor&&) noexcept = default;
  Tensor clone() const;

  DType dtype() const { return dtype_; }
  const Shape& shape() const { return shape_; }
  int64_t size() const { return size_; }
  size_t bytes() const { return static_cast<size_t>(size_) * dtype_size(dtype_); }

  void* raw() { return data_.get(); }
  const void* raw() const { return data_.get(); }

  // Throws Error when T is not the tensor's element type.
  template <class T>
  std::span<T> data() {
    check_dtype(dtype_of<T>::value);
    return {reinterpret_cast<T*>(data_.get()), static_cast<size_t>(size_)};
  }
  template <class T>
  std::span<const T> data() const {
    check_dtype(dtype_of<T>::value);
    return {reinterpret_cast<const T*>(data_.get()), static_cast<size_t>(size_)};
  }

 private:
  struct Free {
    // Constructors of its own: inside Tensor a member's initializer (or a default argument) cannot yet make it
    // default-constructible, which gcc and clang need for data_'s default.
    Free() noexcept : capacity(0) {}
    explicit Free(size_t bytes) noexcept : capacity(bytes) {}
    size_t capacity;  // the block's bytes, for its thread's cache
    void operator()(std::byte* p) const noexcept;
  };
  struct Uninitialized {};
  Tensor(DType dtype, Shape shape, Uninitialized);
  void check_dtype(DType want) const;

  DType dtype_ = DType::kFloat32;
  Shape shape_;
  int64_t size_ = 0;
  std::unique_ptr<std::byte[], Free> data_;
};

}  // namespace carbon
