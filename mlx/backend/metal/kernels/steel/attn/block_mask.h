// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"

///////////////////////////////////////////////////////////////////////////////
// Block mask
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

struct BlockMask {
  device uint64_t* blockmask_ptr;
  int row_offset;
  int uint64_per_row;
  int cache_seqlen_k;
  int max_block_idx;
  int m_block_dim, n_block_dim;
  int n_block_min, n_block_max;
  int batch_idx, head_idx;
  int k_window_left, k_window_right;

  METAL_FUNC BlockMask(const InfLLMV2AttentionStage2Params& params, const BlockInfo& binfo, const device uint64_t* blockmask, const int kBlockM, const int kBlockN, const int batch_idx, const int loop_step_idx, int n_block_min, int n_block_max) {
    this->cache_seqlen_k = binfo.actual_seqlen_k - binfo.actual_seqlen_q / params.m_block_dim;
    this->max_block_idx = (binfo.actual_seqlen_k + params.n_block_dim - 1) / params.n_block_dim;
    this->m_block_dim = params.m_block_dim;
    this->n_block_dim = params.n_block_dim;
    this->n_block_min = n_block_min;
    this->n_block_max = n_block_max;
    this->batch_idx = batch_idx;
    this->head_idx = head_idx;

    int num_blocks_m = params.num_block_m;
    int num_blocks_n = params.num_blocks_n;
    int uint64_per_row = (num_blocks_n + 64 - 1) / 64;
    int row_offset = params.cu_seqlens_q != nullptr ? binfo.blockmask_q_offset(m_block_dim, batch_idx) : batch_idx * params.num_k_heads * params.num_blocks_m;

    this->blockmask_ptr = blockmask + head_idx * params.num_blocks_n * uint64_per_row + row_offset * uint64_per_row + loop_step_idx * uint64_per_row;

    int q_block_idx = loop_step_idx * cache_seqlen_k;
    this->k_window_right = q_block_idx / n_block_dim;
    this->k_window_left = this->k_window_right - params.block_window_size + 1;
  }

  METAL_FUNC ~BlockMask() {
    this->blockmask_ptr = nullptr;
  }

  METAL_FUNC int clzll(const uint64_t x) {
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

  METAL_FUNC int max_no_larger(int target) const {
    if (this->max_block_idx == 0) {
      return -1;
    }

    if ((k_window_left <= target) && (target <= k_window_right)) {
      return target;
    }

    target = min(target, max_block_idx - 1);

    int target_bit_pos = target;

    int uint64_offset = target_bit_pos / 64;

    int bit_pos = target_bit_pos % 64;

    uint64_t mask = bit_pos != 63 ? (1ULL << (bit_pos + 1)) - 1 : 0xFFFFFFFFFFFFFFFFULL;

    uint64_t value = blockmask_ptr[uint64_offset] & mask;

    int result = -1;
    if (value != 0) {
      int highest_bit = 63 - clzll(value);
      result = highest_bit + (uint64_offset * 64);
    } else {
      for (int i = uint64_offset - 1; i >= 0; --i) {
        value = blockmask_ptr[i];
        if (value != 0) {
          int highest_bit = 63 - clzll(value);
          result = highest_bit + (i * 64);
          break;
        }
      } 
    }

    if (target > k_window_right && result <= k_window_right && k_window_left <= k_window_right) { return k_window_right; }

    return result;
  }
};

} // namespace steel
} // namespace mlx
