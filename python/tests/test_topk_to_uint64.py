import numpy as np
import mlx.core as mx
import torch

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

    uint64_np, last_dim_np = topk_to_uint64_numpy(topk_idx_cpu, max_seqlen_k, block_size)

    print(f"uint64_np.shape: {uint64_np.shape}")
    print(f"uint64_np.min(): {uint64_np.min()}, uint64_np.max(): {uint64_np.max()}")

    print(f"last_dim_np: {last_dim_np}")
