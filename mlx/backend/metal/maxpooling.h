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
};

} // namespace mlx::core