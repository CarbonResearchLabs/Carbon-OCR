#include "carbon/tensor.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace carbon {
namespace {

TEST(DType, SizesAndNames) {
  EXPECT_EQ(dtype_size(DType::kFloat32), 4u);
  EXPECT_EQ(dtype_size(DType::kFloat16), 2u);
  EXPECT_EQ(dtype_size(DType::kInt64), 8u);
  EXPECT_EQ(dtype_size(DType::kBool), 1u);
  EXPECT_EQ(dtype_name(DType::kInt32), "int32");
}

TEST(Shape, ElementCount) {
  EXPECT_EQ(element_count({}), 1);  // a scalar
  EXPECT_EQ(element_count({2, 3, 4}), 24);
  EXPECT_EQ(element_count({2, 0, 4}), 0);
  EXPECT_THROW(element_count({2, -1}), Error);
}

TEST(Shape, String) {
  EXPECT_EQ(shape_string({}), "[]");
  EXPECT_EQ(shape_string({1, -1, 32}), "[1, -1, 32]");
}

TEST(Tensor, ZeroFilledAndAligned) {
  Tensor t(DType::kFloat32, {3, 5});
  EXPECT_EQ(t.size(), 15);
  EXPECT_EQ(t.bytes(), 60u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(t.raw()) % 64, 0u);
  for (float v : t.data<float>()) EXPECT_EQ(v, 0.0f);
}

TEST(Tensor, EmptyShapeHasNoMemory) {
  Tensor t(DType::kInt64, {4, 0});
  EXPECT_EQ(t.size(), 0);
  EXPECT_EQ(t.raw(), nullptr);
  EXPECT_TRUE(t.data<int64_t>().empty());
}

TEST(Tensor, FromValues) {
  const std::vector<int32_t> v = {1, 2, 3, 4, 5, 6};
  Tensor t = Tensor::from<int32_t>({2, 3}, v);
  EXPECT_EQ(t.dtype(), DType::kInt32);
  EXPECT_EQ(t.data<int32_t>()[5], 6);
  EXPECT_THROW(Tensor::from<int32_t>({4, 2}, v), Error);
}

TEST(Tensor, WrongElementTypeThrows) {
  Tensor t(DType::kFloat32, {2});
  EXPECT_THROW(t.data<int64_t>(), Error);
}

TEST(Tensor, CloneIsDeepAndMoveEmpties) {
  const std::vector<float> v = {1, 2, 3};
  Tensor a = Tensor::from<float>({3}, v);
  Tensor b = a.clone();
  b.data<float>()[0] = 9;
  EXPECT_EQ(a.data<float>()[0], 1);

  Tensor c = std::move(a);
  EXPECT_EQ(c.data<float>()[2], 3);
  EXPECT_EQ(a.raw(), nullptr);  // NOLINT(bugprone-use-after-move): moved-from is empty
}

}  // namespace
}  // namespace carbon
