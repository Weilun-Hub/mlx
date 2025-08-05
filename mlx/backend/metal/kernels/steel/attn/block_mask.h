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
  int num_k_per_blockmask, num_k_per_block, num_block_per_blockmask;
  int blockmask_idx_left, blockmask_idx_right;

  METAL_FUNC BlockMaskIterator(
    const int qL,
    const int kL,
    const int num_k_per_blockmask, // 64
    const int num_k_per_block, // 16
    const int B, // 1
    const int num_k_heads, // 2
    const int uint64_per_row, // 1
    const int block_window_size, // 2048 / 64 = 32
    const device uint64_t* blockmask,
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]] // q_block_idx, group_idx, batch_idx
  ) {
    this->max_blockmask_idx = (kL + num_k_per_blockmask - 1) / num_k_per_blockmask; // 2048 / 64 = 32
    this->max_k_block_idx = (kL + num_k_per_block - 1) / num_k_per_block; // 2048 / 16 = 128
    this->num_k_per_blockmask = num_k_per_blockmask;
    this->num_k_per_block = num_k_per_block;
    this->num_block_per_blockmask = num_k_per_blockmask / num_k_per_block;

    // tid: q_idx, group_idx (kv_head_idx), batch_idx
    // blockmask: bs * num_k_heads * qL * uint64_per_row, eg: 1 * 2 * 1024 * 1
    this->blockmask_ptr = blockmask
      + tid.z * num_k_heads * qL * uint64_per_row 
      + tid.y * qL * uint64_per_row
      + tid.x * uint64_per_row; // offset to blockmask_ptr for current block

    this->blockmask_idx_right = (kL - qL + tid.x) / num_k_per_blockmask;
    this->blockmask_idx_left = this->blockmask_idx_right - block_window_size + 1;
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

    if (blockmask_ptr == nullptr) { return k_block_idx; }

    k_block_idx = min(k_block_idx, this->max_k_block_idx - 1);

    int blockmask_idx = k_block_idx / this->num_block_per_blockmask; // blockmask_idx = k_block_idx / 4
    
    if (this->blockmask_idx_left <= blockmask_idx && blockmask_idx <= this->blockmask_idx_right) {
      return k_block_idx;
    }

    // int blockmask_uint64_idx = blockmask_idx / 64;
    int blockmask_uint64_idx = blockmask_idx >> 6;

    // int blockmask_uint64_bit_idx = blockmask_idx % 64;
    int blockmask_uint64_bit_idx = blockmask_idx & 63;

    uint64_t bit_pos_in_1_uint64 = blockmask_uint64_bit_idx != 63 ? (1ULL << (blockmask_uint64_bit_idx + 1)) - 1 : 0xFFFFFFFFFFFFFFFFULL;

    // if (blockmask_ptr[blockmask_uint64_idx] == 0) {
    //   if (blockmask_uint64_idx == 0) {
    //     return 0;
    //   } else {
    //     return (blockmask_uint64_idx - 1) * 64 + this->num_block_per_blockmask - 1;
    //   }
    // }

    uint64_t mask = blockmask_ptr[blockmask_uint64_idx] & bit_pos_in_1_uint64;

    int target_blockmask_idx = -1;

    if (mask != 0) {
      int highest_bit = 63 - clzll(mask);
      // target_blockmask_idx = highest_bit + blockmask_uint64_idx * 64;
      target_blockmask_idx = highest_bit + (blockmask_uint64_idx << 6);
    } else {
      for (int i = blockmask_uint64_idx - 1; i >= 0; --i) {
        mask = blockmask_ptr[i];
        if (mask != 0) {
          int highest_bit = 63 - clzll(mask);
          // target_blockmask_idx = highest_bit + i * 64;
          target_blockmask_idx = highest_bit + (i << 6);
          break;
        }
      }
    }

    if (target_blockmask_idx == -1) {
      return -1;
    }

    int target_k_block_idx = target_blockmask_idx * this->num_block_per_blockmask; // target_k_block_idx is the start block index of the blockmask

    return target_k_block_idx + this->num_block_per_blockmask - 1;
  }
};

} // namespace steel
} // namespace mlx
