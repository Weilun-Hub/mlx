// Copyright © 2024-25 Apple Inc.

using namespace mlx::steel;

///////////////////////////////////////////////////////////////////////////////
// GEMM kernels
///////////////////////////////////////////////////////////////////////////////

constant bool align_Q [[function_constant(200)]];
constant bool align_K [[function_constant(201)]];

constant bool has_mask [[function_constant(300)]];
constant bool do_causal [[function_constant(301)]];

template <typename T>
struct TransformScale {
  T scale;
  METAL_FUNC TransformScale(T scale_) : scale(scale_) {}

  METAL_FUNC T apply(T x) const {
    return scale * x;
  }
};

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct SubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x - y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

struct DivOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x / y;
  }
};

// clang-format off
template <
    typename T,
    int BQ, // 32
    int BK, // 16
    int BD, // 128
    int WM, // 4
    int WN, // 1
    typename MaskType = float,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void infllmv2_attention_stage1(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    const constant AttnParams* params [[buffer(4)]],
    const constant AttnMaskParams* mask_params [[buffer(5), function_constant(has_mask)]],
    const device MaskType* mask [[buffer(6), function_constant(has_mask)]],
    uint simd_lane_id [[thread_index_in_simdgroup]], // 0, 1, 2, ..., 31
    uint simd_group_id [[simdgroup_index_in_threadgroup]], // 0, 1, 2, 3
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // Pacifying compiler
  (void)lid;

  // Move to correct block
  ulong3 tidl{tid.x, tid.y, tid.z}; // #tile_q_len, #head, #batch

  Q += tidl.z * params->Q_strides[0] + // Batch
      tidl.y * params->Q_strides[1] + // Head
      tidl.x * BQ * params->Q_strides[2]; // Sequence. move to q this block prepared to process, 

  ulong kv_head_idx = int(tid.y) / params->gqa_factor; // 32 / 2 = 16
  K += tidl.z * params->K_strides[0] + // Batch
      kv_head_idx * params->K_strides[1]; // Head. 
  // move to k this block prepared to process, 
  // since in a block, q attends to all k, so only need to move to start position of k w.r.t seq len

  O += tidl.z * params->O_strides[0] + // Batch
      tidl.y * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Sequence, BQ = 32, each block process q_len of 32

  O_save += tidl.z * params->O_strides[0] + // Batch
      tidl.y / 16 * params->O_strides[1] + // Head
      tidl.x * BQ * params->O_strides[2]; // Sequence, BQ = 32, each block process q_len of 32

  if (has_mask) {
    mask += tidl.z * mask_params->M_strides[0] + // Batch
        tidl.y * mask_params->M_strides[1]; // Head
  }

  // Prepare threadgroup memory
  constexpr short padQ = 16 / sizeof(T); // 16 / 2 = 8
  constexpr short padK = 16 / sizeof(T); // 16 / 2 = 8
  constexpr short padV = 16 / sizeof(T); // 16 / 2 = 8

  constexpr short LDQ_tgp = BD + padQ; // 128 + 8 = 136
  constexpr short LDK_tgp = BK + padK; // 16 + 8 = 24
  constexpr short LDV_tgp = BD + padV; // 128 + 8 = 136

  constexpr short tgp_mem_0 = (BK + padK) * (BD); // (16 + 8) * 128 = 3072
  constexpr short tgp_mem_1 = BK * (BD + padV); // 16 * (128 + 8) = 2176
  constexpr short tgp_mem_s = tgp_mem_0 > tgp_mem_1 ? tgp_mem_0 : tgp_mem_1; // 3072

  threadgroup T Q_smem[BQ * (BD + padQ)]; // smem: each thread process q_len of 32 * head_dim of 128
  threadgroup T KV_smem[tgp_mem_s]; // smem: each thread process k_len of 16 * head_dim of 128

  threadgroup T* Qs = Q_smem;
  threadgroup T* Ks = KV_smem; // kv share same smem, k first loaded, then v
  threadgroup T* Ss = Q_smem;

  // Prepare block loaders
  using QBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ, // 32
      /* short BCOLS = */ BD, // 128
      /* short kDstStrRow = */ LDQ_tgp, // 128 + 8 = 136
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

  // K is loaded in transposed
  using KBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BK,
      /* short BCOLS = */ BD,
      /* short kDstStrRow = */ 1,
      /* short kDstStrCol = */ LDK_tgp,
      /* short reduction_dim = */ 0,
      /* short tgp_size = */ WM * WN * 32>;

  using SBlockLoader = BlockLoaderT<
      /* typename T = */ T,
      /* short BROWS = */ BQ,
      /* short BCOLS = */ BK,
      /* short kDstStrRow = */ LDK_tgp,
      /* short kDstStrCol = */ 1,
      /* short reduction_dim = */ 1,
      /* short tgp_size = */ WM * WN * 32>;

  QBlockLoader loader_q(
      Q, params->Q_strides[2], Qs, simd_group_id, simd_lane_id);
  KBlockLoader loader_k(
      K, params->K_strides[2], Ks, simd_group_id, simd_lane_id);
  SBlockLoader loader_s(
      O, params->O_strides[2], Ss, simd_group_id, simd_lane_id);

  TransformScale<T> ts(static_cast<T>(params->scale * 1.44269504089));

  // Prepare MMA tiles
  constexpr short kFragSize = 8; // MMAFrag size
  using MMAFrag_acc_t = BaseMMAFrag<AccumType, kFragSize, kFragSize>; // 8 * 8

  constexpr int kNWarps = WM * WN; // 4 * 1 = 4
  static_assert(
      BQ >= (kNWarps * kFragSize) && BQ % (kNWarps * kFragSize) == 0,
      "Each simdgroup must host atleast 1 simdgroup matrix along Q sequence.");

  // Q seq frags per warp
  constexpr int TQ = BQ / (kNWarps * kFragSize); // 32 / (4 * 8) = 1
  // KV sequence frags (all warps load the same frags)
  constexpr int TK = BK / kFragSize; // 16 / 8 = 2
  // HeadDim frags (all warps load the same frags)
  constexpr int TD = BD / kFragSize; // 128 / 8 = 16

  static_assert(TQ == 1, "Check TQ");

  MMATile<AccumType, TQ, 1, MMAFrag_acc_t> Qtile; // 1 * 1
  MMATile<AccumType, 1, TK, MMAFrag_acc_t> Ktile; // 1 * 2
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Stile; // 1 * 2
  MMATile<AccumType, TQ, TK, MMAFrag_acc_t> Otile; // same as Stile

  Otile.clear();

  // Prepare mma tile offsets
  const short2 simd_coord = MMAFrag_acc_t::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id; // 8 * 1 * (0, 1, 2, 3) = 0, 8, 16, 24

  // in ONE simdgroup (8 x 8)
  const short Qs_offset = (tm + sm) * LDQ_tgp + sn; // LDQ_tgp = BD + padQ = 128 + 8 = 136
  const short Ks_offset = sm * LDK_tgp + sn; // LDK_tgp = BK + padK = 16 + 8 = 24
  const short Vs_offset = sm * LDV_tgp + sn; // LDV_tgp = BD + padV = 128 + 8 = 136
  const short Ss_offset = (tm + sm) * LDK_tgp + sn; // LDK_tgp = BK + padK = 16 + 8 = 24

  constexpr short Qs_tile_stride = kFragSize; // 8
  constexpr short Ks_tile_stride = kFragSize * LDK_tgp; // 8 * (16 + 8) = 192

  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Load Q blocks apply scale
  if (!align_Q && int(tid.x) == (params->NQ_aligned)) {
    loader_q.load_safe(short2(BD, params->qL_rem));
  } else {
    loader_q.load_unsafe();
  }
  loader_q.apply_inplace_op(ts);

  // Init row reduction variables
  constexpr short kRowsPT = decltype(Stile)::kRowsPerThread; // 1 row per thread, 1 warp process 32 rows

  static_assert(kRowsPT == 1, "Check kRowsPT");

  AccumType max_score[kRowsPT];
  AccumType sum_score[kRowsPT] = {0};

  // Init to -Inf
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    max_score[i] = Limits<AccumType>::finite_min;
  }

  int kb_lim = params->NK; // 8 iters over k

  if (do_causal) {
    int q_max = (tid.x + 1) * BQ + params->qL_off;
    kb_lim = (q_max + BK - 1) / BK;
    kb_lim = min(params->NK, kb_lim);
  }

  O += (tm + sm) * params->O_strides[2] + sn;

  // 1st Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) { // 0, 1, 2, 3, 4, 5, 6, 7
    // Load K block and apply scale
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!align_K && kb == (params->NK_aligned)) {
      loader_k.load_safe(short2(BD, params->kL_rem));
    } else {
      loader_k.load_unsafe();
    }

    // Do S = Q @ K.T
    Stile.clear();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_UNROLL
    for (short dd = 0; dd < TD; dd++) { // loop over head dim dimension 0, 1, 2, ..., 15. TD = 16
      simdgroup_barrier(mem_flags::mem_none);

      Qtile.template load<T, 1, 1, LDQ_tgp, 1>(
          &Qs[Qs_offset + dd * Qs_tile_stride]);
      Ktile.template load<T, 1, 1, LDK_tgp, 1>(
          &Ks[Ks_offset + dd * Ks_tile_stride]);

      simdgroup_barrier(mem_flags::mem_none);

      tile_matmad(Stile, Qtile, Ktile, Stile);
    } // finish warp level operation

    // Mask out length sequence
    if (!align_K && kb == (params->NK_aligned)) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          short col_pos = sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if ((col_pos + jj) >= params->kL_rem) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Mask out if causal
    if (do_causal && kb >= (kb_lim - ((BQ + BK - 1) / BK) - int(!align_K))) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos =
            tid.x * BQ + params->qL_off + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);
          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemCols; jj++) {
            if (row_pos < (col_pos + jj)) {
              Stile.frag_at(i, j)[jj] = neg_inf;
            }
          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = Limits<selem_t>::finite_min;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, selem_t>;

      using MMAFrag_mask_t = BaseMMAFrag<melem_t, kFragSize, kFragSize>;
      using frag_t = typename MMAFrag_mask_t::frag_type;

      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; i++) {
        const int row_pos = tid.x * BQ + tm + sm + (i * stile_t::kFragRows);
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; j++) {
          const int col_pos = kb * BK + sn + (j * stile_t::kFragCols);

          frag_t mfrag;

          MMAFrag_mask_t::load_safe(
              mfrag,
              mask,
              int(mask_params->M_strides[2]),
              Int<1>{},
              params->qL,
              params->kL,
              row_pos,
              col_pos);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < stile_t::MMAFrag_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              Stile.frag_at(i, j)[jj] =
                  mfrag[jj] ? Stile.frag_at(i, j)[jj] : neg_inf;
            } else {
              Stile.frag_at(i, j)[jj] += 1.44269504089 * selem_t(mfrag[jj]);
            }
          }
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    Stile.template store<T, 1, 1>(O + kb * BK, params->O_strides[2]);
    
    // Do softmax

    // Temp variables
    AccumType new_max[kRowsPT]; // each thread handles one row
    AccumType factor[kRowsPT];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      new_max[i] = max_score[i];
    }

    // Row max
    Stile.template row_reduce<MaxOp>(new_max);

    // exp(Si - rowmax(Si))
    Stile.template row_bin_op<ExpSubOp>(new_max);

    // Factor exp(rowmax(Si) - rowmax(Si-1))
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      factor[i] = fast::exp2(max_score[i] - new_max[i]);
    }

    // Row Sum
    AccumType sum_score_tmp[kRowsPT] = {0};
    Stile.template row_reduce<SumOp>(sum_score_tmp);

    // Update norm
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kRowsPT; ++i) {
      max_score[i] = new_max[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Prepare for next iteration
    loader_k.next();
  }
  
  // 2nd Loop over KV seq length
  for (int kb = 0; kb < kb_lim; kb++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_s.load_unsafe();
    Stile.template load<T, 1, 1, LDK_tgp, 1>(&Ss[Ss_offset]);

    Stile.template row_bin_op<ExpSubOp>(max_score);
    Stile.template row_bin_op<DivOp>(sum_score);

    Stile.template store<T, 1, 1, LDK_tgp, 1>(&Ss[Ss_offset]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id % 2 == 0) {
      Ss[Ss_offset] += Ss[Ss_offset + kFragSize * TQ * LDK_tgp];
      Ss[Ss_offset + 1] += Ss[Ss_offset + kFragSize * TQ * LDK_tgp + 1];
      // threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    Stile.template load<T, 1, 1, LDK_tgp, 1>(&Ss[Ss_offset]);
    
    Stile.template col_reduce<SumOp>();

    simdgroup_barrier(mem_flags::mem_threadgroup);
    Stile.template store<T, 1, 1>(O + kb * BK, params->O_strides[2]);

    simdgroup_barrier(mem_flags::mem_threadgroup);
    loader_s.next();
  }
}
