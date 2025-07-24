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
  METAL_FUNC BlockInfo(const int cu_seqlens_q_0, const int cu_seqlens_q_1, const int cu_seqlens_k_0, const int cu_seqlens_k_1, const int max_seqlen_q, const int max_seqlen_k, const int bidb)
    : sum_s_q(cu_seqlens_q_0 == -1 ? -1 : cu_seqlens_q_0 + bidb * cu_seqlens_q_1),
      sum_s_k(cu_seqlens_k_0 == -1 ? -1 : cu_seqlens_k_0 + bidb * cu_seqlens_k_1),
      actual_seqlen_q(cu_seqlens_q_0 == -1 ? max_seqlen_q : cu_seqlens_q_0 + bidb * cu_seqlens_q_1 - sum_s_q),
      leftpad_k(0),
      seqlen_k_cache(cu_seqlens_k_0 == -1 ? max_seqlen_k : cu_seqlens_k_0 + bidb * cu_seqlens_k_1 - sum_s_k - leftpad_k),
      actual_seqlen_k(seqlen_k_cache) {}

  METAL_FUNC int blockmask_q_offset(const int m_block_dim, const int bidb) const {
    return sum_s_q == -1 ? bidb * (actual_seqlen_q / m_block_dim) : uint32_t(sum_s_q) / m_block_dim;
  }

};

} // namespace steel
} // namespace mlx
