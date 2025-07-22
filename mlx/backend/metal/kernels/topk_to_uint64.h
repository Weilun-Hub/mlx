// Copyright © 2025 Apple Inc.

#include "mlx/backend/metal/topk_to_uint64.h"

using mlx::core::TopkToUint64Params;

template <int THREADGROUP_SIZE = 128>
[[kernel, max_total_threads_per_threadgroup(THREADGROUP_SIZE)]] void topk_to_uint64(
    const device int32_t* in [[buffer(0)]],
    device uint64_t* out [[buffer(1)]],
    const constant TopkToUint64Params* params [[buffer(2)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) {
  (void)lid;
  
  ulong3 tidl{gid.x, gid.y, gid.z}; // #blocks_per_row, #n_uint64_per_row, 1
  int tid = lid.x;

  int row = gid.x * THREADGROUP_SIZE + simd_group_id * 32 + simd_lane_id;
  int col = gid.y;

  if ((row >= params->batch_size) || (col >= params->n_uint64_per_row)) {
    return;
  }

  int out_idx = row * params->n_uint64_per_row + col;

  int bit_start = col * 64;
  uint64_t packed_value = 0;

  for (int i = 0; i < params->k; ++i) {
    int idx_offset = row * params->k + i;
    int idx = in[idx_offset];
    if (idx == -1) {
      continue;
    }

    if (idx >= bit_start && idx < bit_start + 64) {
      int local_bit = idx - bit_start;
      packed_value |= (1ULL << local_bit);
    }
  }

  out[out_idx] = packed_value;
}
