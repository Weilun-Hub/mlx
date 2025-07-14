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
void sdpa_full_self_attention_metal(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal_ = false,
    const std::optional<array>& mask = std::nullopt) {
  using namespace mlx::steel;

  /*
  wm: Number of SIMD groups (warps) in the M dimension
  wn: Number of SIMD groups (warps) in the N dimension
  These control the threadgroup size for the Metal compute kernel
  Used in: MTL::Size group_dims = MTL::Size(32, wm, wn) → MTL::Size(32, 4, 1)
  Total threads per threadgroup = 32 * wm * wn = 32 * 4 * 1 = 128
  */
  int wm = 4;
  int wn = 1; 
  
  /*
  bd: Block dimension, equals the head dimension of the query tensor
  This is the feature dimension (e.g., 64, 128, 256)
  Used in kernel specialization for different head dimensions

  bq: Block size for the query sequence dimension
  Each threadgroup processes 32 query tokens at a time
  Used to tile the query sequence length for parallel processing

  bk: Block size for the key sequence dimension
  Adaptive sizing:
    If head dimension < 128: bk = 32
    If head dimension ≥ 128: bk = 16
  Trades off memory usage vs. parallelism based on head dimension
  Used to tile the key sequence length for parallel processing
  */
  int bd = q.shape(-1);
  int bq = 32;
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
  kname << "steel_attention_"
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

  AttnParams params{
      /* int B = */ B,
      /* int H = */ H,
      /* int D = */ D,

      /* int qL = */ qL,
      /* int kL = */ kL,

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

  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(o, 3);
  compute_encoder.set_bytes(params, 4);

  if (mask) {
    printf("[DEBUG ZWL] mask exists\n");
    auto m = *mask;

    AttnMaskParams mask_params{/* int64_t M_strides[3] = */ {
        m.strides(0), m.strides(1), m.strides(2)}};

    compute_encoder.set_bytes(mask_params, 5);
    compute_encoder.set_input_array(m, 6);
  }

  MTL::Size grid_dims = MTL::Size(NQ, H, B);
  MTL::Size group_dims = MTL::Size(32, wm, wn);
  printf("[DEBUG ZWL] grid_dims: %d, %d, %d\n", NQ, H, B);
  printf("[DEBUG ZWL] group_dims: %d, %d, %d\n", 32, wm, wn);
  
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void sdpa_vector(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    array& out,
    float scale,
    bool do_causal,
    const std::optional<array>& mask) {
  // Set the kernel name
  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Compute the necessary sizes
  int gqa_factor = q.shape(1) / k.shape(1);
  int N = k.shape(2);
  int B = q.shape(0) * q.shape(1);
  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides()[2];
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides()[2];

  MTL::Size group_dims(1024, 1, 1);
  MTL::Size grid_dims(B, q.shape(2), 1);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;
  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
  };
  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";

  // Get the kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set its arguments
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(out, 3);
  compute_encoder.set_bytes(gqa_factor, 4);
  compute_encoder.set_bytes(N, 5);
  compute_encoder.set_bytes(k_head_stride, 6);
  compute_encoder.set_bytes(k_seq_stride, 7);
  compute_encoder.set_bytes(v_head_stride, 8);
  compute_encoder.set_bytes(v_seq_stride, 9);

  compute_encoder.set_bytes(scale, 10);
  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(m, 11 + float_mask);
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 13);
    compute_encoder.set_bytes(q_seq_stride, 14);
    compute_encoder.set_bytes(head_stride, 15);
  }

  // Launch
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void sdpa_vector_2pass(
    const Stream& s,
    metal::Device& d,
    const array& q,
    const array& k,
    const array& v,
    array& out,
    float scale,
    bool do_causal,
    const std::optional<array>& mask) {
  // Set the kernel name
  std::string kname;
  kname.reserve(64);
  kname += "sdpa_vector_2pass_1_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(q.shape(-1));
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Compute the necessary sizes
  int gqa_factor = q.shape(1) / k.shape(1);
  int N = k.shape(2);
  int blocks = 32;
  int B = q.shape(0) * q.shape(1);

  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides()[2];
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides()[2];
  MTL::Size group_dims(8 * 32, 1, 1);
  MTL::Size grid_dims(B, q.shape(2), blocks);

  // Allocate the intermediates
  Shape intermediate_shape;
  intermediate_shape.reserve(out.ndim() + 1);
  intermediate_shape.insert(
      intermediate_shape.end(), out.shape().begin(), out.shape().end() - 1);
  intermediate_shape.push_back(blocks);
  intermediate_shape.push_back(out.shape().back());
  array intermediate(intermediate_shape, float32, nullptr, {});
  intermediate_shape.pop_back();
  array sums(intermediate_shape, float32, nullptr, {});
  array maxs(std::move(intermediate_shape), float32, nullptr, {});
  intermediate.set_data(allocator::malloc(intermediate.nbytes()));
  sums.set_data(allocator::malloc(sums.nbytes()));
  maxs.set_data(allocator::malloc(maxs.nbytes()));
  d.add_temporary(intermediate, s.index);
  d.add_temporary(sums, s.index);
  d.add_temporary(maxs, s.index);

  bool has_mask = mask.has_value();
  bool bool_mask = has_mask && (*mask).dtype() == bool_;
  bool float_mask = has_mask && !bool_mask;
  bool query_transposed = !q.flags().row_contiguous;
  metal::MTLFCList func_consts = {
      {&has_mask, MTL::DataType::DataTypeBool, 20},
      {&query_transposed, MTL::DataType::DataTypeBool, 21},
      {&do_causal, MTL::DataType::DataTypeBool, 22},
      {&bool_mask, MTL::DataType::DataTypeBool, 23},
      {&float_mask, MTL::DataType::DataTypeBool, 24},
  };
  std::string hash_name = kname;
  hash_name += has_mask ? (bool_mask ? "_boolmask" : "_floatmask") : "_nomask";
  hash_name += query_transposed ? "_qt" : "_qnt";
  hash_name += do_causal ? "_c" : "_nc";

  // Get the kernel
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto kernel = d.get_kernel(kname, hash_name, func_consts);

  compute_encoder.set_compute_pipeline_state(kernel);

  // Set its arguments
  compute_encoder.set_input_array(q, 0);
  compute_encoder.set_input_array(k, 1);
  compute_encoder.set_input_array(v, 2);
  compute_encoder.set_output_array(intermediate, 3);
  compute_encoder.set_output_array(sums, 4);
  compute_encoder.set_output_array(maxs, 5);
  compute_encoder.set_bytes(gqa_factor, 6);
  compute_encoder.set_bytes(N, 7);
  compute_encoder.set_bytes(k_head_stride, 8);
  compute_encoder.set_bytes(k_seq_stride, 9);
  compute_encoder.set_bytes(v_head_stride, 10);
  compute_encoder.set_bytes(v_seq_stride, 11);
  compute_encoder.set_bytes(scale, 12);
  if (has_mask) {
    auto& m = *mask;
    compute_encoder.set_input_array(m, 13 + float_mask);
    int32_t kv_seq_stride = m.shape(3) > 1 ? m.strides(3) : 0;
    int32_t q_seq_stride = m.shape(2) > 1 ? m.strides(2) : 0;
    int32_t head_stride =
        m.shape(1) > 1 ? m.strides(1) : (m.shape(0) > 1 ? m.strides(0) : 0);
    compute_encoder.set_bytes(kv_seq_stride, 15);
    compute_encoder.set_bytes(q_seq_stride, 16);
    compute_encoder.set_bytes(head_stride, 17);
  }

  // Launch
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);

  // Final pass
  kname.clear();
  kname += "sdpa_vector_2pass_2_";
  kname += get_type_string(q.dtype());
  kname += "_";
  kname += std::to_string(v.shape(-1));

  // Get the kernel
  kernel = d.get_kernel(kname);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set its arguments
  compute_encoder.set_input_array(intermediate, 0);
  compute_encoder.set_input_array(sums, 1);
  compute_encoder.set_input_array(maxs, 2);
  compute_encoder.set_output_array(out, 3);

  // Launch
  group_dims = MTL::Size(1024, 1, 1);
  grid_dims = MTL::Size(B, q.shape(2), 1);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

} // namespace

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    Stream s) {
  if (detail::in_grad_tracing()) {
    std::cout << "[DEBUG ZWL] " << __FILE__ << " : " << __LINE__ << " ScaledDotProductAttention::use_fallback in grad tracing" << std::endl;
    return true;
  }
  if (s.device == Device::cpu) {
    std::cout << "[DEBUG ZWL] " << __FILE__ << " : " << __LINE__ << " ScaledDotProductAttention::use_fallback on cpu" << std::endl;
    return true;
  }

  const int value_head_dim = v.shape(-1);
  const int query_head_dim = q.shape(-1);
  const int query_sequence_length = q.shape(2);
  const int key_sequence_length = k.shape(2);

  const bool sdpa_vector_supported_head_dim =
      query_head_dim == value_head_dim &&
      (query_head_dim == 64 || query_head_dim == 96 || query_head_dim == 128 ||
       query_head_dim == 256);
  const bool sdpa_full_supported_head_dim = query_head_dim == value_head_dim &&
      (query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 128);

  const bool sdpa_full_supported_mask = !has_mask || has_arr_mask ||
      (query_sequence_length <= key_sequence_length && do_causal);

  const bool supports_sdpa_full = query_sequence_length > 8 &&
      sdpa_full_supported_mask && sdpa_full_supported_head_dim;

  const bool supports_sdpa_vector = (query_sequence_length <= 8) &&
      (query_sequence_length <= key_sequence_length) &&
      sdpa_vector_supported_head_dim;

  std::cout << "[DEBUG ZWL] " << __FILE__ << " : " << __LINE__ << " !(supports_sdpa_full || supports_sdpa_vector): " << !(supports_sdpa_full || supports_sdpa_vector) << std::endl;

  return !(supports_sdpa_full || supports_sdpa_vector);
}

void ScaledDotProductAttention::eval_gpu(
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
  if (q_pre.shape(2) <= 8) {
    printf("[DEBUG ZWL] q_pre.shape(2) <= 8, vector mode\n");
    auto q_copy_unless = [](const array& arr) {
      if (arr.flags().row_contiguous) {
        return true;
      }
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      if (shape[0] == 1 || shape[1] == 1) {
        // If either the batch or head dimension is a singleton, the other can
        // be transposed with the sequence dimension
        auto bidx = shape[0] == 1 ? 1 : 0;
        return (strides[3] == 1) && (strides[2] == shape[3] * shape[bidx]) &&
            (strides[bidx] == shape[3]);
      }
      return false;
    };

    auto kv_copy_unless = [](const array& arr) {
      // keys and values should be copied if:
      // - the last dimension is not contiguous
      // - the batch and head dim are not contiguous
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      if (strides.back() != 1) {
        return false;
      }
      if (shape[0] == 1 || shape[1] == 1) {
        return true;
      }
      return (strides[0] == strides[1] * shape[1]);
    };

    const auto& q = copy_unless(q_copy_unless, q_pre);
    const auto& k = copy_unless(kv_copy_unless, k_pre);
    const auto& v = copy_unless(kv_copy_unless, v_pre);

    // Donate the query if possible
    if (q.is_donatable() && q.flags().row_contiguous && q.size() == o.size()) {
      o.copy_shared_buffer(q);
    } else {
      o.set_data(allocator::malloc(o.nbytes()));
    }

    auto mask_copy_unless = [&q](const array& arr) {
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      return arr.flags().row_contiguous || q.shape(0) == 1 || q.shape(1) == 1 ||
          (strides[0] == strides[1] * shape[1]);
    };

    auto mask = inputs.size() > 3
        ? std::optional<array>{copy_unless(mask_copy_unless, inputs[3])}
        : std::nullopt;

    // We route to the 2 pass fused attention if
    // - The device is large and the sequence length long
    // - The sequence length is even longer and we have gqa
    bool do_causal = do_causal_ && q.shape(2) > 1;
    char devc = d.get_architecture().back();
    if ((devc == 'd' && k.shape(2) >= 1024) ||
        (k.shape(1) < q.shape(1) && k.shape(2) >= 4096)) {
      sdpa_vector_2pass(s, d, q, k, v, o, scale_, do_causal, mask);
    } else {
      sdpa_vector(s, d, q, k, v, o, scale_, do_causal, mask);
    }
  }

  // Full attention mode
  else {
    printf("[DEBUG ZWL] q_pre.shape(2) > 8, full attention mode\n");
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

    sdpa_full_self_attention_metal(s, d, q, k, v, scale_, o, do_causal_, mask);
  }

  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core::fast
