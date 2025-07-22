// Copyright © 2023-2024 Apple Inc.
#include <algorithm>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"
#include <iostream>
#include <cassert>
#include "mlx/backend/metal/topk_to_uint64.h"

namespace mlx::core {

void TopkToUint64::eval_gpu(const std::vector<array>& inputs, array& out) {

  std::cout << "[DEBUG ZWL] " << __FILE__ << " : " << __LINE__ << std::endl;
  
  assert(inputs[0].dtype() == int32);
  assert(inputs.size() == 1);
  if (!issubdtype(out.dtype(), uint64)) {
    throw std::runtime_error(
        "[topk_to_uint64] Does not support non-uint64 types.");
  }
  auto& s = stream();
  auto& d = metal::device(s.device);

  // Make sure that the last dimension is contiguous
  auto ensure_contiguous = [&s, &d](const array& x) {
    if (x.flags().contiguous && x.strides()[x.ndim() - 1] == 1) {
      printf("[DEBUG ZWL] x.flags().contiguous && x.strides()[x.ndim() - 1] == 1\n");
      return x;
    } else {
      auto x_copy = array(x.shape(), x.dtype(), nullptr, {});
      copy_gpu(x, x_copy, CopyType::General, s);
      d.add_temporary(x_copy, s.index);
      return x_copy;
    }
  };

  auto in = ensure_contiguous(inputs[0]);

  assert(in.flags().row_contiguous);
  out.set_data(allocator::malloc(out.nbytes()));

  printf("[DEBUG ZWL] in.shape: %d, %d, %d, %d\n", in.shape(0), in.shape(1), in.shape(2), in.shape(3));
  printf("[DEBUG ZWL] in.shape: %d, %d, %d\n", in.strides(0), in.strides(1), in.strides(2));

  int batch_size = in.shape(0);
  int num_head = in.shape(1);
  int q_len = in.shape(2);
  int k_len = in.shape(3);
  printf("[DEBUG ZWL] batch_size: %d, num_head: %d, q_len: %d, k_len: %d\n", batch_size, num_head, q_len, k_len);

  std::string kernel_name = "topk_to_uint64_";
  kernel_name += type_to_name(out) + "_t";

  printf("[DEBUG ZWL] in.strides: %d, %d, %d\n", in.strides(0), in.strides(1), in.strides(2));
  printf("[DEBUG ZWL] out.strides: %d, %d, %d\n", out.strides(0), out.strides(1), out.strides(2));
  printf("[DEBUG ZWL] max_seqlen_k_: %d, block_size_: %d\n", max_seqlen_k_, block_size_);
  int k_blocks = (max_seqlen_k_ + block_size_ - 1) / block_size_;
  printf("[DEBUG ZWL] k_blocks: %d\n", k_blocks);
  printf("[DEBUG ZWL] n_uint64_per_row: %d\n", out.shape(3));

  int flat_dims = batch_size * num_head * q_len;
  printf("[DEBUG ZWL] flat_dims: %d\n", flat_dims);

  size_t threadgroup_size = 128;

  int blocks_per_row = (flat_dims + threadgroup_size - 1) / threadgroup_size;

  int n_uint64_per_row = out.shape(3);
  printf("[DEBUG ZWL] n_uint64_per_row: %d\n", n_uint64_per_row);

  int k = in.shape(3);
  printf("[DEBUG ZWL] k: %d\n", k);

  TopkToUint64Params params{
    /* batch_size = */ flat_dims,
    /* k = */ k,
    /* k_blocks = */ k_blocks,
    /* n_uint64_per_row = */ n_uint64_per_row,
    /* in_strides = */ {in.strides(0), in.strides(1), in.strides(2)},
    /* out_strides = */ {out.strides(0), out.strides(1), out.strides(2)}
  };

  printf("[DEBUG ZWL] params.in_strides: %d, %d, %d\n", params.in_strides[0], params.in_strides[1], params.in_strides[2]);
  printf("[DEBUG ZWL] params.out_strides: %d, %d, %d\n", params.out_strides[0], params.out_strides[1], params.out_strides[2]);

  auto kernel = get_topk_to_uint64_kernel(d, kernel_name, out);
  auto& compute_encoder = d.get_command_encoder(s.index);
  {

    MTL::Size grid_dims = MTL::Size(blocks_per_row, n_uint64_per_row, 1);
    MTL::Size group_dims = MTL::Size(threadgroup_size, 1, 1);

    printf("[DEBUG ZWL] grid_dims: %d, %d, %d\n", blocks_per_row, n_uint64_per_row, 1);
    printf("[DEBUG ZWL] group_dims: %d, %d, %d\n", threadgroup_size, 1, 1);

    compute_encoder.set_compute_pipeline_state(kernel);
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(out, 1);
    compute_encoder.set_bytes(params, 2);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }
}

} // namespace mlx::core
