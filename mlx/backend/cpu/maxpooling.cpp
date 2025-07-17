// Copyright © 2023-2024 Apple Inc.

#include <cassert>
#include <cmath>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cpu/simd/simd.h"
#include "mlx/primitives.h"
#include "mlx/types/limits.h"

namespace mlx::core {

namespace {

using namespace mlx::core::simd;

template <typename T, typename AccT>
void maxpooling(const array& in, array& out, Stream stream) {
  printf("NYI\n");
}

} // namespace

void MaxPooling::eval_cpu(const std::vector<array>& inputs, array& out) {
  printf("NYI\n");
}

} // namespace mlx::core
