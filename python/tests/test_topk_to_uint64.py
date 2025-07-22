import numpy as np
import mlx.core as mx
import torch

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=4, suppress=True)

"""
namespace {
/**
 * CUDA kernel to convert topk indices directly to uint64 representation
 * Each thread processes one element in the output array
 */
__global__ void kernel_topk_to_uint64(
    const int* topk_idx,       // Input topk indices [num_heads, total_seqlen, k]
    uint64_t* result,          // Output uint64 array
    int batch_size,            // Total number of rows (flattened batch dimensions)
    int k,                     // Number of topk values per row
    int k_blocks,              // Number of key blocks
    int n_uint64_per_row       // Number of uint64 needed per row
) {
    // Calculate global position
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    
    if (row >= batch_size || col >= n_uint64_per_row) return;
    
    // Calculate output offset
    int out_idx = row * n_uint64_per_row + col;
    
    // Calculate starting bit position for this uint64
    int bit_start = col * 64;
    
    // Initialize result
    uint64_t packed_value = 0;
    
    // For each topk index in this row
    for (int i = 0; i < k; i++) {
        // Get the index value
        int idx_offset = row * k + i;
        int idx = topk_idx[idx_offset];
        
        // Skip if the index is -1 (invalid)
        if (idx == -1) continue;
        
        // Check if this idx falls within the current uint64 chunk
        if (idx >= bit_start && idx < bit_start + 64) {
            // Set the corresponding bit in the packed value
            int local_bit = idx - bit_start;
            packed_value |= (1ULL << local_bit);
        }
    }
    
    // Store the result
    result[out_idx] = packed_value;
}
} // namespace

/**
 * Function to convert topk indices directly to uint64 representation
 */
void topk_to_uint64_func(
    cudaStream_t stream,
    const int* topk_idx,          // Input topk indices
    uint64_t* result,             // Output uint64 array
    int batch_size,               // Total number of rows (flattened batch dimensions)
    int k,                        // Number of topk values per row
    int k_blocks,                 // Number of key blocks
    int n_uint64_per_row          // Number of uint64 needed per row
) {
    const int threads_per_block = 256;
    const int blocks_per_row = (batch_size + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(blocks_per_row, n_uint64_per_row);
    dim3 block(threads_per_block, 1);
    
    kernel_topk_to_uint64<<<grid, block, 0, stream>>>(
        topk_idx, result, batch_size, k, k_blocks, n_uint64_per_row
    );
} 

"""

def topk_to_uint64_numpy(
    topk_idx: np.ndarray,
    max_seqlen_k: int,
    block_size: int
) -> tuple:
    """
    NumPy implementation of topk_to_uint64
    
    Args:
        topk_idx: NumPy array of shape [num_heads, total_seqlen, k] or [batch, num_heads, seq_len, k]
        max_seqlen_k: Maximum key sequence length
        block_size: Block size
        
    Returns:
        uint64_array: NumPy array of uint64 values
        last_dim: Size of the last dimension
    """
    # Check input dimensions
    is_4d = len(topk_idx.shape) == 4
    
    if is_4d:
        batch, num_heads, seq_len, k = topk_idx.shape
        # Reshape to 3D for processing
        topk_idx_3d = topk_idx.reshape(batch * num_heads, seq_len, k) # (4, 8, 128, 32) -> (32, 128, 32)
    else:
        num_heads, seq_len, k = topk_idx.shape
        topk_idx_3d = topk_idx
    
    # Calculate number of blocks
    k_blocks = (max_seqlen_k + block_size - 1) // block_size # (1024 + 16 - 1) // 16 = 64
    
    # Calculate how many uint64 values we need per row
    uint64_bits = 64
    last_dim = (k_blocks + uint64_bits - 1) // uint64_bits # 1
    
    # Create output array
    if is_4d:
        uint64_output = np.zeros((batch, num_heads, seq_len, last_dim), dtype=np.uint64) # (4, 8, 128, 1)
        uint64_output_3d = uint64_output.reshape(batch * num_heads, seq_len, last_dim) # (32, 128, 1)
    else:
        uint64_output = np.zeros((num_heads, seq_len, last_dim), dtype=np.uint64)
        uint64_output_3d = uint64_output
    
    # Process each (head, sequence) position
    for head_idx in range(uint64_output_3d.shape[0]):
        for seq_idx in range(seq_len):
            # Get valid indices (not -1)
            indices = topk_idx_3d[head_idx, seq_idx]
            valid_indices = indices[indices >= 0]
            
            # Set bits in uint64 array
            for idx in valid_indices:
                uint64_idx = idx // uint64_bits
                bit_pos = idx % uint64_bits
                # Use np.uint64 explicitly for bitwise operations
                uint64_output_3d[head_idx, seq_idx, uint64_idx] |= np.uint64(1) << np.uint64(bit_pos)
    
    return uint64_output, last_dim

if __name__ == "__main__":

    torch.manual_seed(0)

    shape = (4, 8, 128, 32) # (batch, num_heads, seq_len, k)
    max_seqlen_k = 1024
    block_size = 16

    k_blocks = (max_seqlen_k + block_size - 1) // block_size
    print(f"k_blocks: {k_blocks}")

    topk_idx = torch.randint(-1, k_blocks, shape, dtype=torch.int32)
    print(f"topk_idx.shape: {topk_idx.shape}")
    print(f"topk_idx.min(): {topk_idx.min()}, topk_idx.max(): {topk_idx.max()}")

    topk_idx_cpu = topk_idx.cpu().numpy()
    print("topk_idx_cpu.dtype: ", topk_idx_cpu.dtype)

    topk_idx_cpu_cp = topk_idx_cpu.copy()

    uint64_np, last_dim_np = topk_to_uint64_numpy(topk_idx_cpu, max_seqlen_k, block_size)
    # uint64_np_signed = uint64_np.view(np.int64)
    print(f"last_dim_np: {last_dim_np}")

    print(f"uint64_np.shape: {uint64_np.shape}")
    print(f"uint64_np.min(): {uint64_np.min()}, uint64_np.max(): {uint64_np.max()}")
    # print(f"uint64_np_signed.min(): {uint64_np_signed.min()}, uint64_np_signed.max(): {uint64_np_signed.max()}")

    topk_idx = mx.array(topk_idx_cpu_cp, dtype=mx.int32)
    a = mx.topk_to_uint64(topk_idx, max_seqlen_k, block_size)
    a_mlx = np.array(a)
    print(f"a_mlx.dtype: {a_mlx.dtype}")
    print(f"a_mlx.shape: {a_mlx.shape}")
    print(f"a_mlx.min(): {a_mlx.min()}, a_mlx.max(): {a_mlx.max()}")

    print("+------------- gt -------------+")
    print(uint64_np[0, 0, :4, 0])
    # print(uint64_np[0, 0, :, 0])
    # print(bin(uint64_np[0, 0, 0, 0]))

    print("+------------- mlx -------------+")
    print(np.array(a[0, 0, :4, 0]))
    # print(a_mlx[0, 0, 0, 0])
    # print(bin(a_mlx[0, 0, 0, 0]))

    diff = np.abs(a_mlx - uint64_np)
    print("max diff: ", diff.max())

    # exit()
