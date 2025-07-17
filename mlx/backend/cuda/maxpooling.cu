// Copyright © 2025 Apple Inc.

#include "mlx/allocator.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/cast_op.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <nvtx3/nvtx3.hpp>
#include <cub/block/block_load.cuh>

#include <cassert>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename T>
__global__ void maxpooling_kernel(
    const T* input,
    T* output,
    size_t outer_size,
    size_t inner_size) {
  printf("NYI\n");
}

} // namespace cu

void MaxPooling::eval_gpu(const std::vector<array>& inputs, array& out) {
  printf("NYI\n");
}

} // namespace mlx::core
