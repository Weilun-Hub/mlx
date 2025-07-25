// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"

///////////////////////////////////////////////////////////////////////////////
// Block mask
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

struct BlockMaskIterator {
  const device uint64_t* blockmask_ptr;
  int max_blockmask_idx, max_k_block_idx;
  int num_k_per_blockmask, num_k_per_block;

  METAL_FUNC BlockMaskIterator(
    const int qL,
    const int kL,
    const int num_k_per_blockmask, // 64
    const int B,
    const int num_k_heads,
    const int num_q_per_block, // 16
    const int uint64_per_row,
    const device uint64_t* blockmask,
    const int max_k_block_idx,
    const int num_k_per_block, // 16
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]], 
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]] // q_block_idx, group_idx, batch_idx
  ) {
    this->max_blockmask_idx = (kL + num_k_per_blockmask - 1) / num_k_per_blockmask;
    this->max_k_block_idx = max_k_block_idx;
    this->num_k_per_blockmask = num_k_per_blockmask;
    this->num_k_per_block = num_k_per_block;

    this->blockmask_ptr = blockmask
      + lid.z * num_k_heads * qL * uint64_per_row 
      + lid.y * qL * uint64_per_row
      + lid.x * num_q_per_block * uint64_per_row; // offset to blockmask_ptr for current block
  }

  METAL_FUNC ~BlockMaskIterator() {
    this->blockmask_ptr = nullptr;
  }

  METAL_FUNC int clzll(uint64_t x) const {
    if (x == 0) { return 64; }

    int n = 0;
    if (x >> 32 == 0) {n += 32; x <<= 32;}
    if (x >> 48 == 0) {n += 16; x <<= 16;}
    if (x >> 56 == 0) {n += 8; x <<= 8;}
    if (x >> 60 == 0) {n += 4; x <<= 4;}
    if (x >> 62 == 0) {n += 2; x <<= 2;}
    if (x >> 63 == 0) {n += 1;}
    return n;
  }

  METAL_FUNC int max_no_larger(int k_block_idx) const {

    k_block_idx = min(k_block_idx, this->max_k_block_idx - 1);

    int blockmask_idx = k_block_idx / (this->num_k_per_blockmask / this->num_k_per_block);

    int blockmask_uint64_idx = blockmask_idx / 64;

    int blockmask_uint64_bit_idx = blockmask_idx % 64;

    uint64_t bit_pos_in_1_uint64 = blockmask_uint64_bit_idx != 63 ? (1ULL << (blockmask_uint64_bit_idx + 1)) - 1 : 0xFFFFFFFFFFFFFFFFULL;

    uint64_t mask = blockmask_ptr[blockmask_uint64_idx] & bit_pos_in_1_uint64;

    int target_blockmask_idx = -1;

    if (mask != 0) {
      int highest_bit = 63 - clzll(mask);
      target_blockmask_idx = highest_bit + blockmask_uint64_idx * 64;
    } else {
      for (int i = blockmask_uint64_idx - 1; i >= 0; --i) {
        mask = blockmask_ptr[i];
        if (mask != 0) {
          int highest_bit = 63 - clzll(mask);
          target_blockmask_idx = highest_bit + i * 64;
          break;
        }
      }
    }

    if (target_blockmask_idx == -1) {
      return -1;
    }

    int target_k_block_idx = target_blockmask_idx * (this->num_k_per_blockmask / this->num_k_per_block);

    return target_k_block_idx;
  }
};

} // namespace steel
} // namespace mlx
