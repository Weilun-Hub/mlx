// Copyright © 2025 Apple Inc.

#include "mlx/backend/metal/maxpooling.h"

using mlx::core::MaxPoolingParams;

template <typename T, int THREADGROUP_SIZE = 128>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_SIZE)]] void maxpooling(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant MaxPoolingParams* params [[buffer(2)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  (void)lid;
  
  ulong3 tidl{gid.x, gid.y, gid.z}; // #q, #head, #batch
  int tid = lid.x;

  const device T* in_ptr = in + tidl.z * params->in_strides[0] + tidl.y * params->in_strides[1] + tidl.x * params->in_strides[2];
  device T* out_ptr = out + tidl.z * params->out_strides[0] + tidl.y * params->out_strides[1] + tidl.x * params->out_strides[2];

  int q_block = (tidl.x + params->cache_len) / params->block_size;

  constexpr auto neg_inf = Limits<T>::finite_min;
  constexpr auto pos_inf = Limits<T>::finite_max;

  for (int k = tid; k < params->out_strides[2]; k += THREADGROUP_SIZE) {
    int start = k * params->stride - params->padding;
    int end = start + params->kernel_size;
    start = start > 0 ? start : 0;
    end = end < params->in_strides[2] ? end : params->in_strides[2];

    T max_val = -1;
    if (k < params->init_blocks) {
      max_val = pos_inf;
    } else if (q_block - params->local_blocks < k) {
      max_val = pos_inf; // neg_inf in cuda impl
    } else {
      max_val = in_ptr[start];
      for (int i = start + 1; i < end; i++) {
        if (in_ptr[i] > max_val) {
          max_val = in_ptr[i];
        }
      }
    }
    out_ptr[k] = max_val;
  }
}
