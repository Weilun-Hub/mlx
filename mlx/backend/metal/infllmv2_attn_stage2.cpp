// Copyright © 2024 Apple Inc.
#include <sstream>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/transforms_impl.h"
#include "mlx/utils.h"
#include <iostream>

namespace mlx::core::fast {

namespace {
void infllmv2_attention_stage2_metal(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const array& cu_seqlens_q,
    const array& cu_seqlens_k,
    const int max_seqlen_q,
    const int max_seqlen_k,
    const int window_size_left,
    const int window_size_right,
    const int block_window_size,
    const array& blockmask_uint64,
    const float scale,
    array& o,
    bool do_causal_ = false,
    const std::optional<array>& mask = std::nullopt) {
  using namespace mlx::steel;

  /*
  wm: Number of SIMD groups (warps) in the M dimension
  wn: Number of SIMD groups (warps) in the N dimension
  These control the threadgroup size for the Metal compute kernel
  Used in: MTL::Size group_dims = MTL::Size(32, wm, wn) → MTL::Size(32, 2, 1)
  Total threads per threadgroup = 32 * wm * wn = 32 * 2 * 1 = 64
  */
  int wm = 2;
  int wn = 1; 
  
  /*
  bd: Block dimension, equals the head dimension of the query tensor
  This is the feature dimension (e.g., 64, 128, 256)
  Used in kernel specialization for different head dimensions

  bq: Block size for the query sequence dimension
  Each threadgroup processes 16 query tokens at a time
  Used to tile the query sequence length for parallel processing

  bk: Block size for the key sequence dimension
  Adaptive sizing:
    If head dimension < 128: bk = 32
    If head dimension ≥ 128: bk = 16
  Trades off memory usage vs. parallelism based on head dimension
  Used to tile the key sequence length for parallel processing
  */
  int bd = q.shape(-1);
  int bq = 16;
  int bk = bd < 128 ? 32 : 16;
  printf("[DEBUG ZWL] bd: %d, bq: %d, bk: %d\n", bd, bq, bk);

  int B = q.shape(0);
  int H = q.shape(1);
  int D = q.shape(3);
  int gqa_factor = q.shape(1) / k.shape(1);
  printf("[DEBUG ZWL] B: %d, H: %d, D: %d, gqa_factor: %d\n", B, H, D, gqa_factor);

  int qL = q.shape(2);
  int kL = k.shape(2);
  printf("[DEBUG ZWL] qL: %d, kL: %d\n", qL, kL);

  const bool align_Q = (qL % bq) == 0;
  const bool align_K = (kL % bk) == 0;
  const bool has_mask = !!mask;
  const bool do_causal = do_causal_;
  printf("[DEBUG ZWL] align_Q: %d, align_K: %d, has_mask: %d, do_causal: %d\n", align_Q, align_K, has_mask, do_causal);

  metal::MTLFCList func_consts = {
      {&align_Q, MTL::DataType::DataTypeBool, 200},
      {&align_K, MTL::DataType::DataTypeBool, 201},
      {&has_mask, MTL::DataType::DataTypeBool, 300},
      {&do_causal, MTL::DataType::DataTypeBool, 301}};

  std::ostringstream kname;
  // clang-format off
  kname << "infllmv2_attention_stage2_"
        << type_to_name(q)
        << "_bq" << bq
        << "_bk" << bk
        << "_bd" << bd
        << "_wm" << wm
        << "_wn" << wn
        << "_mask" << (type_to_name(has_mask ? *mask : q)); // clang-format on

  std::string base_name = kname.str();
  printf("[DEBUG ZWL] base_name: %s\n", base_name.c_str());

  // clang-format off
  kname << "_align_Q_" << (align_Q ? 't' : 'n')
        << "_align_K_" << (align_K ? 't' : 'n')
        << "_has_mask_" << (has_mask ? 't' : 'n')
        << "_do_causal_" << (do_causal ? 't' : 'n'); // clang-format on

  std::string hash_name = kname.str();
  printf("[DEBUG ZWL] hash_name: %s\n", hash_name.c_str());

  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(base_name, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  const int NQ = (qL + bq - 1) / bq;
  const int NK = (kL + bk - 1) / bk;
  printf("[DEBUG ZWL] NQ: %d, NK: %d\n", NQ, NK);

  const int NQ_aligned = qL / bq;
  const int NK_aligned = kL / bk;
  printf("[DEBUG ZWL] NQ_aligned: %d, NK_aligned: %d\n", NQ_aligned, NK_aligned);

  InfLLMV2AttnStage2Params params{
      /* int B = */ B,
      /* int H = */ H,
      /* int D = */ D,

      /* int qL = */ qL,
      /* int kL = */ kL,

      /* int cu_seqlens_q = */ {*cu_seqlens_q.data<int>(), *(cu_seqlens_q.data<int>() + 1)},
      /* int cu_seqlens_k = */ {*cu_seqlens_k.data<int>(), *(cu_seqlens_k.data<int>() + 1)},
      /* int max_seqlen_q = */ max_seqlen_q,
      /* int max_seqlen_k = */ max_seqlen_k,
      /* int window_size_left = */ window_size_left,
      /* int window_size_right = */ window_size_right,

      /* int num_q_per_block = */ 1,
      /* int num_kv_per_blockmask = */ 64,
      /* int num_k_heads = */ 2,
      /* int block_window_size = */ block_window_size,
      /* int uint64_per_row = */ blockmask_uint64.shape(3), // bs * num_k_heads * qL * uint64_per_row, eg: 1 * 2 * 1024 * 1

      /* int gqa_factor = */ gqa_factor,
      /* float scale = */ scale,

      /* int NQ = */ NQ,
      /* int NK = */ NK,

      /* int NQ_aligned = */ NQ_aligned,
      /* int NK_aligned = */ NK_aligned,

      /* int qL_rem = */ (qL - NQ_aligned * bq),
      /* int kL_rem = */ (kL - NK_aligned * bk),
      /* int qL_off = */ (kL - qL),

      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)}};

  printf("[DEBUG ZWL] cu_seqlens_q: %d, %d\n", params.cu_seqlens_q[0], params.cu_seqlens_q[1]);
  printf("[DEBUG ZWL] cu_seqlens_k: %d, %d\n", params.cu_seqlens_k[0], params.cu_seqlens_k[1]);

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_input_array(blockmask_uint64, 3);
  compute_encoder.set_output_array(o, 4);
  compute_encoder.set_bytes(params, 5);

  if (mask) {
    printf("[DEBUG ZWL] mask exists\n");
    auto m = *mask;

    AttnMaskParams mask_params{/* int64_t M_strides[3] = */ {
        m.strides(0), m.strides(1), m.strides(2)}};

    compute_encoder.set_bytes(mask_params, 5);
    compute_encoder.set_input_array(m, 6);
  }

  int num_q_per_block = 1;
  int num_block_q = (qL + num_q_per_block - 1) / num_q_per_block;
  printf("[DEBUG ZWL] num_q_per_block: %d, num_block_q: %d\n", num_q_per_block, num_block_q);

  int head_group_num = k.shape(1);
  printf("[DEBUG ZWL] head group num: %d\n", head_group_num);

  MTL::Size grid_dims = MTL::Size(num_block_q, head_group_num, B);
  MTL::Size group_dims = MTL::Size(32, wm, wn);
  printf("[DEBUG ZWL] grid_dims: %d, %d, %d\n", num_block_q, head_group_num, B);
  printf("[DEBUG ZWL] group_dims: %d, %d, %d\n", 32, wm, wn);
  
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

} // namespace

bool InfLLMV2AttentionStage2::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    const array& cu_seqlens_q,
    const array& cu_seqlens_k,
    const int max_seqlen_q,
    const int max_seqlen_k,
    const int window_size_left,
    const int window_size_right,
    const array& blockmask_uint64,
    const int block_window_size,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    Stream s) {
  printf("[DEBUG ZWL] InfLLMV2AttentionStage2::use_fallback NYI\n");
  return false;
}

void InfLLMV2AttentionStage2::eval_gpu(
    const std::vector<array>& inputs,
    array& out) {
  auto& s = stream();
  auto& d = metal::device(s.device);

  auto& q_pre = inputs[0];
  auto& k_pre = inputs[1];
  auto& v_pre = inputs[2];
  auto& o = out;

  auto& q_pre_shape = q_pre.shape();
  auto& k_pre_shape = k_pre.shape();
  auto& v_pre_shape = v_pre.shape();
  auto& o_shape = o.shape();
  printf("[DEBUG ZWL] q_pre.shape().size: %d, [%d, %d, %d, %d]\n", q_pre_shape.size(), q_pre_shape[0], q_pre_shape[1], q_pre_shape[2], q_pre_shape[3]);
  printf("[DEBUG ZWL] k_pre.shape().size: %d, [%d, %d, %d, %d]\n", k_pre_shape.size(), k_pre_shape[0], k_pre_shape[1], k_pre_shape[2], k_pre_shape[3]);
  printf("[DEBUG ZWL] v_pre.shape().size: %d, [%d, %d, %d, %d]\n", v_pre_shape.size(), v_pre_shape[0], v_pre_shape[1], v_pre_shape[2], v_pre_shape[3]);
  printf("[DEBUG ZWL] o.shape().size: %d, [%d, %d, %d, %d]\n", o_shape.size(), o_shape[0], o_shape[1], o_shape[2], o_shape[3]);

  std::vector<array> copies;

  // Define some copy functions to ensure the layout of the inputs is as
  // expected.
  copies.reserve(3);
  auto copy_unless = [&copies, &s](
                         auto predicate, const array& arr) -> const array& {
    if (!predicate(arr)) {
      array arr_copy(arr.shape(), arr.dtype(), nullptr, {});
      copy_gpu(arr, arr_copy, CopyType::General, s);
      copies.push_back(std::move(arr_copy));
      return copies.back();
    } else {
      return arr;
    }
  };

  // Checks that the headdim dimension has stride 1.
  auto is_matrix_contiguous = [](const array& arr) {
    return arr.strides(-1) == 1;
  };

  // We are in vector mode ie single query
  // assert(q_pre.shape(2) > 8);

  // assert(q_pre.shape(0) == 1); // batch size must be 1 currently
  
  // printf("[DEBUG ZWL] q_pre.shape(2) > 8, full attention mode\n");
  const auto& q = copy_unless(is_matrix_contiguous, q_pre);
  const auto& k = copy_unless(is_matrix_contiguous, k_pre);
  const auto& v = copy_unless(is_matrix_contiguous, v_pre);

  int64_t str_oD = 1;
  int64_t str_oH = o.shape(3); // 128
  int64_t str_oL = o.shape(1) * str_oH; // 32 * 128 = 4096
  int64_t str_oB = o.shape(2) * str_oL; // 2048 * 4096 = 8388608
  size_t data_size = o.shape(0) * str_oB; // 1 * 8388608 = 8388608
  printf("[DEBUG ZWL] str_oD: %d, str_oH: %d, str_oL: %d, str_oB: %d, data_size: %zu\n", str_oD, str_oH, str_oL, str_oB, data_size);

  array::Flags flags{
      /* bool contiguous = */ 1,
      /* bool row_contiguous = */ 0,
      /* bool col_contiguous = */ 0,
  };

  o.set_data(
      allocator::malloc(o.nbytes()),
      data_size,
      {str_oB, str_oH, str_oL, str_oD},
      flags);

  auto mask = inputs.size() > 3
      ? std::optional<array>{copy_unless(is_matrix_contiguous, inputs[3])}
      : std::nullopt;

  infllmv2_attention_stage2_metal(s, d, q, k, v, cu_seqlens_q_, cu_seqlens_k_, max_seqlen_q_, max_seqlen_k_, window_size_left_, window_size_right_, block_window_size_, blockmask_uint64_, scale_, o, do_causal_, mask);

  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core::fast
