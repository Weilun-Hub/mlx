import numpy as np
import mlx.core as mx
import torch
import gc
import time

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=4, suppress=True)

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

BATCH_SIZE = 1
NUM_HEAD = 2
LEN_Q = 1
LEN_K = 4096 * 2
TOPK = 32
BLOCK_SIZE = 64
NUM_BLOCKS = (LEN_K + BLOCK_SIZE - 1) // BLOCK_SIZE
print(f"[INFO] NUM_BLOCKS: {NUM_BLOCKS}")
NUM_UINT64_PER_ROW = (NUM_BLOCKS + 64 - 1) // 64
print(f"[INFO]NUM_UINT64_PER_ROW: {NUM_UINT64_PER_ROW}")

TEST_ITERS_MLX = 20

if __name__ == "__main__":

    torch.manual_seed(0)

    topk_idx = torch.randint(-1, NUM_BLOCKS - 1, (BATCH_SIZE, NUM_HEAD, LEN_Q, NUM_UINT64_PER_ROW), dtype=torch.int32)
    
    topk_idx_cpu = topk_idx.cpu().numpy()

    topk_idx_cpu_cp = topk_idx_cpu.copy()

    uint64_np, last_dim_np = topk_to_uint64_numpy(topk_idx_cpu, LEN_K, BLOCK_SIZE)
    
    topk_idx = mx.array(topk_idx_cpu_cp, dtype=mx.int32)
    a = mx.topk_to_uint64(topk_idx, LEN_K, BLOCK_SIZE)
    a_mlx = np.array(a)
    diff = np.abs(a_mlx - uint64_np)
    print(f"[INFO] max diff: {diff.max()}")

    start_event = torch.mps.Event(enable_timing=True)
    end_event = torch.mps.Event(enable_timing=True)

    time_list = []

    for i in range(TEST_ITERS_MLX):
        try:
            torch.mps.empty_cache()
            start_event.record()
            a = mx.topk_to_uint64(topk_idx, LEN_K, BLOCK_SIZE)
            end_event.record()
            torch.mps.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
        except Exception as e:
            print(f"[ERROR]: {e}, iteration: {i}")
            continue

        time_list.append(elapsed_time)
        torch.mps.empty_cache()
        gc.collect()
        print(f"[INFO] mlx impl time @ iteration {i:02d}: {elapsed_time:.4f} ms")
        time.sleep(0.5) # sleep 0.5 s
    
    del start_event
    del end_event
    gc.collect()

    print(f"[INFO] mlx mean time: {np.mean(time_list):.2f} ms")
    print(f"[INFO] mlx std time: {np.std(time_list):.2f} ms")
    print(f"[INFO] mlx min time: {np.min(time_list):.2f} ms")
    print(f"[INFO] mlx max time: {np.max(time_list):.2f} ms")
