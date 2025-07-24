// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/steel/defines.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"

///////////////////////////////////////////////////////////////////////////////
// Block info
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

struct BlockInfo {

  int sum_s_q;
  int sum_s_k;
  int actual_seqlen_q;
  int leftpad_k;
  int seqlen_k_cache;
  int actual_seqlen_k;

  // [DEBUG ZWL] correctness of the constructor NOT guaranteed
  METAL_FUNC BlockInfo(const InfLLMV2AttentionStage2Params& params, const int bidb)
    : sum_s_q(params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb]),
      sum_s_k(params.cu_seqlens_k == nullptr ? -1 : params.cu_seqlens_k[bidb]),
      actual_seqlen_q(params.cu_seqlens_q == nullptr ? params.max_seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q),
      leftpad_k(0),
      seqlen_k_cache(params.cu_seqlens_k == nullptr ? params.max_seqlen_k : params.cu_seqlens_k[bidb + 1] - sum_s_k - leftpad_k),
      actual_seqlen_k(seqlen_k_cache) {}

  METAL_FUNC int blockmask_q_offset(const int m_block_dim, const int bidb) const {
    return sum_s_q == -1 ? bidb * (actual_seqlen_q / m_block_dim) : uint32_t(sum_s_q) / m_block_dim;
  }

};

} // namespace steel
} // namespace mlx
