// Copyright © 2023-2024 Apple Inc.
#include <algorithm>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/backend/metal/maxpooling.h"
#include <iostream>
#include <cassert>

namespace mlx::core {

void MaxPooling::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error(
        "[maxpooling] Does not support non-floating point types.");
  }
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Make sure that the last dimension is contiguous
  auto ensure_contiguous = [&s, &d](const array& x) {
    if (x.flags().contiguous && x.strides()[x.ndim() - 1] == 1) {
      return x;
    } else {
      auto x_copy = array(x.shape(), x.dtype(), nullptr, {});
      copy_gpu(x, x_copy, CopyType::General, s);
      d.add_temporary(x_copy, s.index);
      return x_copy;
    }
  };

  auto in = ensure_contiguous(inputs[0]);
  // const auto& in = inputs[0];

  assert(in.flags().row_contiguous);
  out.set_data(allocator::malloc(out.nbytes()));

  // printf("[DEBUG ZWL] in.shape: %d, %d, %d, %d\n", in.shape(0), in.shape(1), in.shape(2), in.shape(3));

  int axis_size = in.shape().back();
  // printf("[DEBUG ZWL] axis_size: %d\n", axis_size);
  int n_rows = in.data_size() / axis_size;
  // printf("[DEBUG ZWL] n_rows: %d\n", n_rows);

  int batch_size = in.shape(0);
  int num_head = in.shape(1);
  int q_len = in.shape(2);
  int k_len = in.shape(3);
  // printf("[DEBUG ZWL] batch_size: %d, num_head: %d, q_len: %d, k_len: %d\n", batch_size, num_head, q_len, k_len);

  const int simd_size = 32;
  const int n_reads = 4;

  std::string kernel_name = "maxpooling_";
  kernel_name += type_to_name(out);

  int out_len = out.shape(3);

  // printf("[DEBUG ZWL] in.strides: %d, %d, %d\n", num_head * q_len * k_len, q_len * k_len, k_len);
  // printf("[DEBUG ZWL] in.strides: %d, %d, %d, %d\n", in.strides(0), in.strides(1), in.strides(2), in.strides(3));
  // printf("[DEBUG ZWL] out.strides: %d, %d, %d\n", out.strides(0), out.strides(1), out.strides(2));
  // printf("[DEBUG ZWL] cache_len: %d, init_blocks: %d, local_blocks: %d, kernel_size: %d, stride: %d, padding: %d, block_size: %d\n", cache_len_, init_blocks_, local_blocks_, kernel_size_, stride_, padding_, block_size_);
  MaxPoolingParams params{
    /* cache_len = */ cache_len_,
    /* init_blocks = */ init_blocks_,
    /* local_blocks = */ local_blocks_,
    /* kernel_size = */ kernel_size_,
    /* stride = */ stride_,
    /* padding = */ padding_,
    /* block_size = */ block_size_,
    /* k_len = */ k_len,
    /* out_len = */ out_len,
    /* in_strides = */ {in.strides(0), in.strides(1), in.strides(2)},
    // /* in_strides = */ {num_head * q_len * k_len, q_len * k_len, k_len},
    /* out_strides = */ {out.strides(0), out.strides(1), out.strides(2)}
  };

  auto kernel = get_maxpooling_kernel(d, kernel_name, out);
  auto& compute_encoder = d.get_command_encoder(s.index);
  {
    size_t threadgroup_size = 128;

    MTL::Size grid_dims = MTL::Size(q_len, num_head, batch_size);
    MTL::Size group_dims = MTL::Size(threadgroup_size, 1, 1);

    // printf("[DEBUG ZWL] grid_dims: %d, %d, %d\n", q_len, num_head, 1);
    // printf("[DEBUG ZWL] group_dims: %d, %d, %d\n", threadgroup_size, 1, 1);

    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_bytes(params, 2);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }
}

} // namespace mlx::core
