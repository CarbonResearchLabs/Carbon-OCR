#include "carbon/tensor.h"

#include <new>
#include <vector>

namespace carbon {

namespace {

constexpr std::align_val_t ALIGN{64};

// Each thread's freed large blocks, for its next tensors: a block is taken for a tensor that needs at least half
// of it, the smallest that fits first; at most kKept bytes kept, the oldest given back past that.
constexpr size_t kLarge = 256 << 10, kGrain = 64 << 10, kKept = size_t{48} << 20;

struct Blocks {
  struct Block {
    std::byte* p;
    size_t bytes;
  };
  std::vector<Block> held;
  size_t total = 0;

  ~Blocks();
  std::byte* take(size_t bytes, size_t& capacity) {
    size_t best = held.size();
    for (size_t i = 0; i < held.size(); ++i) {
      if (held[i].bytes >= bytes && held[i].bytes <= 2 * bytes &&
          (best == held.size() || held[i].bytes < held[best].bytes)) {
        best = i;
      }
    }
    if (best == held.size()) return nullptr;
    std::byte* p = held[best].p;
    capacity = held[best].bytes;
    total -= capacity;
    held.erase(held.begin() + static_cast<std::ptrdiff_t>(best));
    return p;
  }
  void keep(std::byte* p, size_t bytes) {
    held.push_back({p, bytes});
    total += bytes;
    while (total > kKept && !held.empty()) {
      total -= held.front().bytes;
      ::operator delete[](held.front().p, ALIGN);
      held.erase(held.begin());
    }
  }
};

thread_local bool gone = false;  // the thread's blocks already freed (it is ending): a tensor freed now goes back

Blocks::~Blocks() {
  gone = true;
  for (const Block& b : held) ::operator delete[](b.p, ALIGN);
}

Blocks& blocks() {
  thread_local Blocks b;
  return b;
}

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
  if (capacity >= kLarge && !gone) {
    try {
      blocks().keep(p, capacity);
      return;
    } catch (...) {  // no room to note it: given back
    }
  }
  ::operator delete[](p, ALIGN);
}

Tensor::Tensor(DType dtype, Shape shape, Uninitialized)
    : dtype_(dtype), shape_(std::move(shape)), size_(element_count(shape_)) {
  if (size_ <= 0) return;
  const size_t need = bytes();
  size_t capacity = need;
  std::byte* p = nullptr;
  if (need >= kLarge && !gone) {
    p = blocks().take(need, capacity);
    if (!p) capacity = (need + kGrain - 1) / kGrain * kGrain;  // a whole grain, so near sizes share blocks
  }
  if (!p) p = static_cast<std::byte*>(::operator new[](capacity, ALIGN));
  data_ = std::unique_ptr<std::byte[], Free>(p, Free{capacity});
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
