#include "carbon/tensor.h"

#include <new>

namespace carbon {

namespace {

constexpr std::align_val_t ALIGN{64};

}  // namespace

size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
    case DType::kInt32:
      return 4;
    case DType::kFloat16:
      return 2;
    case DType::kInt64:
      return 8;
    case DType::kUInt8:
    case DType::kBool:
      return 1;
  }
  return 0;
}

std::string_view dtype_name(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
      return "float32";
    case DType::kFloat16:
      return "float16";
    case DType::kInt32:
      return "int32";
    case DType::kInt64:
      return "int64";
    case DType::kUInt8:
      return "uint8";
    case DType::kBool:
      return "bool";
  }
  return "?";
}

int64_t element_count(const Shape& shape) {
  int64_t n = 1;
  for (int64_t d : shape) {
    if (d < 0) throw Error("negative dimension in shape " + shape_string(shape));
    n *= d;
  }
  return n;
}

std::string shape_string(const Shape& shape) {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) s += ", ";
    s += std::to_string(shape[i]);
  }
  return s + "]";
}

void Tensor::Free::operator()(std::byte* p) const noexcept {
  ::operator delete[](p, ALIGN);
}

Tensor::Tensor(DType dtype, Shape shape, Uninitialized)
    : dtype_(dtype), shape_(std::move(shape)), size_(element_count(shape_)) {
  if (size_ > 0) {
    data_.reset(static_cast<std::byte*>(::operator new[](bytes(), ALIGN)));
  }
}

Tensor::Tensor(DType dtype, Shape shape)
    : Tensor(dtype, std::move(shape), Uninitialized{}) {
  if (size_ > 0) std::memset(data_.get(), 0, bytes());
}

Tensor Tensor::empty(DType dtype, Shape shape) {
  return Tensor(dtype, std::move(shape), Uninitialized{});
}

Tensor Tensor::clone() const {
  Tensor t = empty(dtype_, shape_);
  if (size_ > 0) std::memcpy(t.raw(), raw(), bytes());
  return t;
}

void Tensor::check_dtype(DType want) const {
  if (want != dtype_) {
    throw Error("tensor is " + std::string(dtype_name(dtype_)) + ", not " +
                std::string(dtype_name(want)));
  }
}

}  // namespace carbon
