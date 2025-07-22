// Copyright © 2023-2024 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>

using namespace metal;

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/topk_to_uint64.h"

#define instantiate_topk_to_uint64(name, itype)                               \
  instantiate_kernel("topk_to_uint64_" #name, topk_to_uint64, itype)         \

instantiate_topk_to_uint64(float32, float)
instantiate_topk_to_uint64(float16, half)
instantiate_topk_to_uint64(bfloat16, bfloat16_t) // clang-format on
