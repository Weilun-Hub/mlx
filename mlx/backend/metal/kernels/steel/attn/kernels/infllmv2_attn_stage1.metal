// Copyright © 2024-25 Apple Inc.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"

#include "mlx/backend/metal/kernels/steel/attn/attn.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/infllmv2_attn_stage1.h"

#define instantiate_infllmv2_attn_stage1(tname, dtype, bq, bk, bd, wm, wn, mname, mtype) \
  instantiate_kernel(                                                    \
      "infllmv2_attention_stage1_" #tname "_bq" #bq "_bk" #bk "_bd" #bd            \
      "_wm" #wm "_wn" #wn "_mask" #mname,                                \
  infllmv2_attention_stage1, dtype, bq, bk, bd, wm, wn, mtype, float)

#define instantiate_infllmv2_attn_stage1_shapes_helper(iname, itype, mname, mtype)  \
    instantiate_infllmv2_attn_stage1(iname, itype, 32, 16, 128, 4, 1, mname, mtype) \
    instantiate_infllmv2_attn_stage1(iname, itype, 32, 32,  80, 4, 1, mname, mtype) \
    instantiate_infllmv2_attn_stage1(iname, itype, 32, 32,  64, 4, 1, mname, mtype)

#define instantiate_infllmv2_attn_stage1_mask_helper(iname, itype) \
    instantiate_infllmv2_attn_stage1_shapes_helper(iname, itype, iname, itype) \
    instantiate_infllmv2_attn_stage1_shapes_helper(iname, itype, bool_, bool)

instantiate_infllmv2_attn_stage1_mask_helper(float16, half);
instantiate_infllmv2_attn_stage1_mask_helper(bfloat16, bfloat16_t);

instantiate_infllmv2_attn_stage1_mask_helper(float32, float);
// clang-format on
