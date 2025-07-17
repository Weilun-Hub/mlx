// Copyright © 2023-2024 Apple Inc.

#include <metal_common>
#include <metal_simdgroup>

using namespace metal;

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/maxpooling.h"

#define instantiate_maxpooling(name, itype)                               \
  instantiate_kernel("maxpooling_" #name, maxpooling, itype)         \

instantiate_maxpooling(float32, float)
instantiate_maxpooling(float16, half)
instantiate_maxpooling(bfloat16, bfloat16_t) // clang-format on
