#pragma once

namespace mlx::core {

struct TopkToUint64Params {
    int batch_size;
    int k;
    int k_blocks;
    int n_uint64_per_row;
    int64_t in_strides[3];
    int64_t out_strides[3];
};

} // namespace mlx::core