#pragma once

namespace mlx::core {

struct MaxPoolingParams {
    int cache_len;
    int init_blocks;
    int local_blocks;
    int kernel_size;
    int stride;
    int padding;
    int block_size;
    int k_len;
    int out_len;
    int64_t in_strides[3];
    int64_t out_strides[3];
};

} // namespace mlx::core