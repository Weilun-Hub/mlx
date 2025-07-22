// Copyright © 2023-2024 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>

using namespace metal;

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/topk_to_uint64.h"

#define instantiate_topk_to_uint64(name)                               \
  instantiate_kernel("topk_to_uint64_" #name, topk_to_uint64)         \

instantiate_topk_to_uint64(uint64_t)
