import math
import time
import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
import gc
from einops import rearrange, repeat

np.random.seed(233)

BATCH_SIZE = 1
NUM_ATTN_HEADS = 32
NUM_KEY_VALUE_HEADS = 2
HEAD_DIM = 128
# Q_LEN = 2048
Q_LEN = 1
K_LEN = 16 * 1024 # here K_LEN is the sum of current k len and historical k len
COMPRESSED_K_LEN = K_LEN // 16
print(f"[DEBUG] COMPRESSED_K_LEN: {COMPRESSED_K_LEN}")
CACHE_LEN = K_LEN - Q_LEN
TOPK = 64
BLOCK_SIZE = 64
NUM_BLOCKS = K_LEN // BLOCK_SIZE
print(f"[DEBUG] NUM_BLOCKS: {NUM_BLOCKS}")
DTYPE = np.float16

INIT_BLOCK = 1
LOCAL_BLOCK = 8
KERNEL_SIZE = 5
STRIDE = 4
PADDING = 1
BLOCK_WINDOW_SIZE = 0

TORCH_MASK = False if Q_LEN == 1 else True
MLX_MASK = "causal" if TORCH_MASK else None

TEST_ITERS_INFLLMV2 = 10
TEST_ITERS_FA = 10

CHECK_ACCURACY = True
DEVICE = "mps"

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=4, suppress=True)

def naive_infllmv2_attn_stage1_torch(q, k, v, causal=False):
    k = k.repeat_interleave(q.shape[0] // k.shape[0], dim=0) # (2, 128, 128) -> (32, 128, 128)
    v = v.repeat_interleave(q.shape[0] // v.shape[0], dim=0) # (2, 128, 128) -> (32, 128, 128)
    
    attn = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5) # (32, 2048, 128) @ (32, 128, 128) -> (32, 2048, 128)
    if causal:
        causal_mask = torch.zeros(q.shape[1], k.shape[1], device=q.device).bool() # (2048, 128)
        for i in range(q.shape[1]): # 2048
            for j in range(k.shape[1]): # 128
                if i >= (j * 16 + 32 - 1):
                    causal_mask[i, j] = True
        attn = attn.masked_fill(~causal_mask, -float('inf'))
    score = F.softmax(attn, dim=-1)
    score = score.reshape(2, 16, q.shape[1], k.shape[1]).sum(dim=1)
    if causal:
        score = torch.nan_to_num(score, nan=0.0)
    return score

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

def generate_topk_indices(nheads_k, total_seqlen, seqlen_k, sparsity, block_size, device):
    """
    Generate random topk indices for infllmv2_sparse_attention.
    
    Args:
        nheads_k: Number of key heads
        total_seqlen: Total sequence length (batch * seqlen_q)
        seqlen_k: Key sequence length
        sparsity: Sparsity level (0.0 to 1.0)
        block_size: Size of each block
        device: Device to create the indices on
        
    Returns:
        torch.Tensor: Topk indices with shape [nheads_k, total_seqlen, k]
    """
    # Calculate number of blocks in key dimension
    k_blocks = (seqlen_k + block_size - 1) // block_size
    
    # Calculate how many blocks to keep (top-k)
    k = max(0, int(k_blocks * (1 - sparsity)))
    
    # Option 1: Generate random valid indices for each query position
    # For each head and query position, sample k random indices from 0 to k_blocks-1
    indices = torch.stack([
        torch.stack([
            torch.randperm(k_blocks, device=device)[:k]
            for _ in range(total_seqlen)
        ])
        for _ in range(nheads_k)
    ])
    
    # Make sure indices are sorted for better performance
    indices = indices.sort(dim=-1)[0].to(torch.int32)
    
    return indices

def convert_topk_to_base_blockmask(
    topk_idx: torch.Tensor,
    max_seqlen_k: int,
    block_size: int,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Convert topk indices to infllmv2_sparse_attention mask
    
    Args:
        topk_idx: Tensor of shape [num_heads, total_seqlen, k] containing block indices
        max_seqlen_k: Maximum sequence length for keys
        block_size: Size of each block
        device: Output device
    
    Returns:
        mask: Boolean mask of shape [num_heads, total_seqlen, k_blocks]
    """
    # Calculate number of key blocks
    k_blocks = (max_seqlen_k + block_size - 1) // block_size
    num_heads, total_seqlen, k = topk_idx.shape

    # Initialize all-False mask
    mask = torch.zeros(num_heads, total_seqlen, k_blocks, 
                       dtype=torch.bool, device=device)

    # Filter out any -1 values (if present)
    valid_mask = topk_idx != -1
    
    # Generate index mask - get head, seq positions and corresponding indices
    batch_idx, seq_idx, k_idx = torch.where(valid_mask)
    block_idx = topk_idx[valid_mask]
    
    # Set corresponding positions to True
    mask[batch_idx, seq_idx, block_idx] = True

    return mask

def prepare_mixed_mask(base_blockmask, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k, batch_size, nheads = 32, nheads_k= 2, m_block_dim=128, n_block_dim=128, block_window_size=0):
    """
    Expand a block-level sparsity mask to a token-level mask for attention_blocksparse_ref,
    handling variable sequence lengths per batch item.
    
    Parameters:
        base_blockmask: Bool tensor of shape [nheads_k, total_unpadded_tokens, ncol] 
            where True values indicate blocks that should be attended to
        cu_seqlens_q: Cumulative sequence lengths for queries [batch_size + 1]
        cu_seqlens_k: Cumulative sequence lengths for keys [batch_size + 1]
        seqlen_q: Maximum query sequence length (padded)
        seqlen_k: Maximum key sequence length (padded)
        batch_size: Number of batches
        m_block_dim: Block size for query dimension (default: 128)
        n_block_dim: Block size for key dimension (default: 128)
        block_window_size: Number of blocks to the left of each query block to attend to (default: 0)
    
    Returns:
        mixed_mask: Bool tensor of shape [batch_size, num_heads, seqlen_q, seqlen_k]
            where True values indicate positions that should be masked out
    """
    # print("+" + '-' * 50 + "+")
    # print("|start debug prepare_mixed_mask")
    # print("+" + '-' * 50 + "+")
    # print("base_blockmask.shape", base_blockmask.shape)
    # print("cu_seqlens_q", cu_seqlens_q.cpu().numpy())
    # print("cu_seqlens_k", cu_seqlens_k.cpu().numpy())
    # print("seqlen_q", seqlen_q)
    # print("seqlen_k", seqlen_k)
    # print("batch_size", batch_size)
    # print("nheads", nheads)
    # print("nheads_k", nheads_k)
    # print("m_block_dim", m_block_dim)
    # print("n_block_dim", n_block_dim)
    # print("block_window_size", block_window_size)
    # Make a copy of base_blockmask to avoid modifying the original
    modified_blockmask = base_blockmask.clone()
    
    # Helper function to round up to multiple
    def round_to_multiple(x, base):
        return ((x + base - 1) // base) * base
    
    # Apply block window logic if block_window_size > 0
    if block_window_size > 0:
        num_blocks_k = modified_blockmask.shape[2]  # number of blocks in key dimension, [2, 2046, 32] -> 32
        
        # Generate block indices for each query position
        total_unpadded_tokens = modified_blockmask.shape[1] # [2, 2046, 32] -> 2046
        
        # Calculate batch-relative positions for each token in the unpadded sequence
        q_positions = torch.arange(total_unpadded_tokens, device=modified_blockmask.device) # [0, 1, 2, ..., 2045], 2046 elements
        
        # Properly determine which batch each position belongs to
        batch_indices = torch.zeros_like(q_positions) # [0, 0, 0, ..., 0], 2046 elements
        for b in range(1, len(cu_seqlens_q)): # [0, 2046] -> 2046
            batch_indices = torch.where(
                q_positions >= cu_seqlens_q[b-1],
                torch.where(
                    q_positions < cu_seqlens_q[b],
                    torch.tensor(b-1, device=q_positions.device),
                    batch_indices
                ),
                batch_indices
            ) # [0, 0, 0, ..., 0], 2046 elements
            # print("batch_indices.shape", batch_indices.shape)
            # print(f"batch_indices.min(): {batch_indices.min().cpu().numpy()}, batch_indices.max(): {batch_indices.max().cpu().numpy()}")
        
        # Calculate relative positions within each batch
        relative_positions = q_positions - cu_seqlens_q[batch_indices] # 2046 elements, [0, 1, 2, ..., 2045]
        # print("relative_positions.shape", relative_positions.shape)
        # print("relative_positions", relative_positions)
        
        # For each query position, check if key blocks are within the window range
        # according to kernel logic: k_idx >= q_block_idx - (block_window_size * n_block_dim) && k_idx <= q_block_idx
        for i in range(total_unpadded_tokens): # 0, 1, 2, ..., 2045
            # Get the batch this token belongs to
            batch_idx = batch_indices[i] # 0
            
            # Calculate cache_seqlen_k for this batch (matching CUDA logic)
            # cache_seqlen_k = actual_seqlen_k - actual_seqlen_q / m_block_dim
            actual_seqlen_q = cu_seqlens_q[batch_idx + 1] - cu_seqlens_q[batch_idx] # 2046
            actual_seqlen_k = cu_seqlens_k[batch_idx + 1] - cu_seqlens_k[batch_idx] # 2047
            cache_seqlen_k = actual_seqlen_k - actual_seqlen_q # 1
            
            # Calculate q_block_idx as token position (matching CUDA: loop_step_idx + cache_seqlen_k)
            loop_step_idx = relative_positions[i] # 0, 1, 2, ..., 2045
            q_block_idx = loop_step_idx + cache_seqlen_k # 1, 2, 3, ..., 2046
            
            # Calculate window boundaries (matching CUDA logic)
            k_window_right = q_block_idx // n_block_dim # 0 // 64 = 0
            k_window_left = k_window_right - block_window_size + 1 # 0 - 1 + 1 = 0
            
            for k_block in range(num_blocks_k): # 0, 1, 2, ..., 31
                # Matching the kernel logic exactly
                if k_window_left <= k_block and k_block <= k_window_right:
                    # Set this position to True for all heads
                    modified_blockmask[:, i, k_block] = True
    
    # print("modified_blockmask[0, 0, :]", modified_blockmask[0, 0, :])
    # print(f"modified_blockmask[:, :, 0].min(): {modified_blockmask[:, :, 0].min().cpu().numpy()}, modified_blockmask[:, :, 0].max(): {modified_blockmask[:, :, 0].max().cpu().numpy()}")

    # Expand blocks to token level for all heads
    expanded_mask = repeat(modified_blockmask, "h r c -> (h g) r (c n)", 
                         g = int(nheads / nheads_k), n=n_block_dim) # (2, 2046, 32) -> (2 * 16, 2046, 32 * 64) = (32, 2046, 2048)
    # print(f"g: {int(nheads / nheads_k)}, n: {n_block_dim}")
    # print(f"modefied_blockmask changed from {modified_blockmask.shape} to {expanded_mask.shape}")
    # print("expanded_mask[:, -1, 0]", expanded_mask[:, -1, 0])
    # print("modified_blockmask[:, -1, 0]", modified_blockmask[:, -1, 0])
    # breakpoint()
    # Create batch of masks with padding
    batch_masks = []
    for b in range(batch_size):
        # Get actual sequence lengths and start/end indices for this batch item
        q_start, q_end = cu_seqlens_q[b].item(), cu_seqlens_q[b+1].item() # 0, 2046
        k_start, k_end = cu_seqlens_k[b].item(), cu_seqlens_k[b+1].item() # 0, 2047
        q_len = q_end - q_start # 2046
        k_len = k_end - k_start # 2047
        
        # Create padded mask for this batch item
        # Initialize with all masked out (True)
        batch_mask = torch.ones(nheads, seqlen_q, seqlen_k, dtype=torch.bool, 
                               device=base_blockmask.device) # (32, 2046, 2048)
        
        # Copy the relevant portion from expanded mask corresponding to THIS batch item
        # We need to use the correct slice from the unpadded tokens
        # And invert since True in expanded_mask means "attend to"
        # but True in final mask means "mask out" (set to -inf)
        batch_mask[:, :q_len, :k_len] = ~expanded_mask[:, q_start:q_end, :k_len]

        batch_masks.append(batch_mask)
    
    # Stack all batch masks
    mixed_mask = torch.stack(batch_masks, dim=0) # (1, 32, 2048, 2048)
    # print("mixed_mask.shape", mixed_mask.shape)
    # print("+" + '-' * 50 + "+")
    # print("|end debug prepare_mixed_mask")
    # print("+" + '-' * 50 + "+")
    return mixed_mask

def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )

def attention_blocksparse_ref(
    q, k, v, 
    mixed_mask,
    query_padding_mask=None,
    key_padding_mask=None, 
    p_dropout=0.0, 
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),
    upcast=True,
    reorder_ops=False,
    ):
    # q, k, v = qkv.float().unbind(dim=2)
    # print("+" + '-' * 50 + "+")
    # print("|start debug attention_blocksparse_ref")
    # print("+" + '-' * 50 + "+")
    if causal:
        # print("causal is True, window_size", window_size)
        window_size = (window_size[0], 0) # (-1 , -1) -> (-1, 0)
        # print("causal is True, window_size", window_size)
    dtype_og = q.dtype
    if upcast:
        # print("upcast is True")
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1] # (1, 2048, 32, 128) -> 2048, (1, 2048, 2, 128) -> 2048
    # print("seqlen_q", seqlen_q)
    # print("seqlen_k", seqlen_k)
    # print("k.shape", k.shape, "v.shape", v.shape)
    # exit()
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2]) # (1, 2048, 2, 128) -> (1, 2048, 32, 128)
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2]) # (1, 2048, 2, 128) -> (1, 2048, 32, 128)
    # print("k.shape", k.shape, "v.shape", v.shape)
    d = q.shape[-1] # 128
    # exit()
    # print("reorder_ops", reorder_ops)
    if not reorder_ops: # reorder_ops is False
        # here q.shape = (1, 2048, 32, 128), k.shape = (1, 2048, 32, 128)
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k) # (1, 2048, 32, 128) * (1, 2048, 32, 128) -> (1, 32, 2048, 2048)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    # print("scores.shape", scores.shape)
    if key_padding_mask is not None: # 1, 2048
        # print("key_padding_mask.shape", key_padding_mask.shape)
        # print("key_padding_mask", key_padding_mask)
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    # local mask
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        ) # (1, 1, 2048, 2048)
        # print("local mask.shape", local_mask.shape)
        scores.masked_fill_(local_mask, float("-inf"))
        
    
    scores.masked_fill_(rearrange(mixed_mask, "b h t s -> b h t s"), float("-inf"))
    
    # print("processed blockmask: ", rearrange(~base_blockmask, "h t s -> 1 h t s"))
    
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
     
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(torch.bitwise_or(local_mask, rearrange(mixed_mask, "b h t s -> b h t s")), dim=-1, keepdim=True), 0.0)
    
    attention = attention.masked_fill(rearrange(mixed_mask, "b h t s -> b h t s"), 0.0)  
    
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - p_dropout)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    # print("+" + '-' * 50 + "+")
    # print("|end debug attention_blocksparse_ref")
    # print("+" + '-' * 50 + "+")
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)

def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask

if __name__ == "__main__":

    q_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_ATTN_HEADS, Q_LEN, HEAD_DIM)).astype(DTYPE)
    k_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_KEY_VALUE_HEADS, K_LEN, HEAD_DIM)).astype(DTYPE)
    compressed_k_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_KEY_VALUE_HEADS, COMPRESSED_K_LEN, HEAD_DIM)).astype(DTYPE)
    v_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_KEY_VALUE_HEADS, K_LEN, HEAD_DIM)).astype(DTYPE)
    scale = float(1.0 / math.sqrt(HEAD_DIM))

    q_mlx = mx.array(q_npy)
    k_mlx = mx.array(k_npy)
    compressed_k_mlx = mx.array(compressed_k_npy)
    v_mlx = mx.array(v_npy)

    q_torch_ = torch.from_numpy(q_npy).permute(0, 2, 1, 3).to(DEVICE).contiguous()
    k_torch_ = torch.from_numpy(k_npy).permute(0, 2, 1, 3).to(DEVICE).contiguous()
    v_torch_ = torch.from_numpy(v_npy).permute(0, 2, 1, 3).to(DEVICE).contiguous()
    query_padding_mask = generate_random_padding_mask(Q_LEN, BATCH_SIZE, DEVICE, mode="random")
    key_padding_mask = generate_random_padding_mask(K_LEN, BATCH_SIZE, DEVICE, mode="random")
    query_padding_mask[query_padding_mask == False] = True
    key_padding_mask[key_padding_mask == False] = True

    print(f"[DEBUG] q_torch_.shape: {q_torch_.shape}")
    print(f"[DEBUG] k_torch_.shape: {k_torch_.shape}")
    print(f"[DEBUG] v_torch_.shape: {v_torch_.shape}")
    print(f"[DEBUG] query_padding_mask.shape: {query_padding_mask.shape}")
    print(f"[DEBUG] key_padding_mask.shape: {key_padding_mask.shape}")
    # exit()

    # exit()

    cu_seqlens_q_mlx = mx.array([0, Q_LEN])
    cu_seqlens_k_mlx = mx.array([0, K_LEN])
    cu_seqlens_q = torch.tensor([0, Q_LEN], device=DEVICE)
    cu_seqlens_k = torch.tensor([0, K_LEN], device=DEVICE)
    max_seqlen_q = Q_LEN
    max_seqlen_k = K_LEN
    window_size = (-1, -1)
    window_size_left = window_size[0]
    window_size_right = window_size[1]
    
    # block score
    score = mx.fast.infllmv2_attention_stage1(q_mlx, compressed_k_mlx, v_mlx, scale=scale, mask=MLX_MASK) # (1, 2, 1024, 256)
    print(f"[DEBUG] score.shape: {score.shape}")
    # print(f"[DEBUG] score[0, 0, 0, 0]: {score[0, 0, 0, 0]}")
    if CHECK_ACCURACY:
        q_torch = torch.from_numpy(q_npy.squeeze(0))
        compressed_k_torch = torch.from_numpy(compressed_k_npy.squeeze(0))
        v_torch = torch.from_numpy(v_npy.squeeze(0))
        
        score_torch = naive_infllmv2_attn_stage1_torch(q_torch, compressed_k_torch, v_torch, causal=TORCH_MASK)
        
        score_torch_npy = score_torch.numpy()
        score_mlx_npy = np.array(score).squeeze(0)
        diff = np.abs(score_torch_npy - score_mlx_npy)
        print(f"[DEBUG] stage1 diff.shape: {diff.shape}, min diff: {np.min(diff):.4f}, max diff: {np.max(diff):.4f}")
    
    # max pooling
    pooled_score = mx.maxpooling(score, CACHE_LEN, INIT_BLOCK, LOCAL_BLOCK, KERNEL_SIZE, STRIDE, PADDING, BLOCK_SIZE) # (1, 2, 1024, 16)
    print(f"[DEBUG] pooled_score.shape: {pooled_score.shape}")
    if CHECK_ACCURACY:
        gt_pooled_score = np.fromfile(f"./data.maxpooling/block_score.{NUM_KEY_VALUE_HEADS}x{Q_LEN}x{NUM_BLOCKS}.fp16.bin", dtype=DTYPE).reshape(BATCH_SIZE, NUM_KEY_VALUE_HEADS, Q_LEN, NUM_BLOCKS)
        print(f"[DEBUG] gt_pooled_score.shape: {gt_pooled_score.shape}")

        # print(gt_pooled_score[0, 0, 0])
        pooled_score_npy = np.array(pooled_score)
        pooled_score_npy[pooled_score_npy == -65504] = -float('inf')
        pooled_score_npy[pooled_score_npy == 65504] = float('inf')

        pooled_score_npy[pooled_score_npy == -float('inf')] = -1000
        pooled_score_npy[pooled_score_npy == float('inf')] = 1000
        gt_pooled_score[gt_pooled_score == -float('inf')] = -1000
        gt_pooled_score[gt_pooled_score == float('inf')] = 1000

        diff = np.abs(gt_pooled_score - pooled_score_npy)
        print(f"[DEBUG] diff.shape: {diff.shape}, min diff: {np.min(diff):.4f}, max diff: {np.max(diff):.4f}")

    # topk
    topk_idx = mx.argtopk(pooled_score, TOPK, axis=-1) # (1, 2, 1024, 32)
    # print(f"[DEBUG] topk_idx.shape: {topk_idx.shape}")
    if CHECK_ACCURACY:
        pooled_score_torch = torch.from_numpy(np.array(pooled_score))
        topk_idx_gt_torch = pooled_score_torch.topk(TOPK, dim=-1).indices.sort(-1).values
        topk_idx_gt_npy = topk_idx_gt_torch.numpy()
        topk_idx_npy = np.array(topk_idx)
        diff = np.abs(topk_idx_gt_npy - topk_idx_npy)
        print(f"[DEBUG] topk diff.shape: {diff.shape}, min diff: {np.min(diff):.4f}, max diff: {np.max(diff):.4f}")
        # print(f"[DEBUG] topk_idx[0, 0, -1]: {topk_idx_npy[0, 0, -1]}")
        # for i in range(Q_LEN):
        # i = 2010
        # print("-" * 100)
        # print(f"[DEBUG] q_idx: {i}")
        # for j in range(pooled_score_npy.shape[3]):
        #     print(f"[DEBUG] pooled_score[0, 0, {i}, {j}]: {pooled_score_npy[0, 0, i, j]}")
        # print(f"[DEBUG] topk_idx_gt_npy[0, 0, {i}, :]: {topk_idx_gt_npy[0, 0, i, :]}")
        # print(f"[DEBUG] topk_idx_npy[0, 0, {i}, :]: {topk_idx_npy[0, 0, i, :]}")
        # cur_diff = np.abs(topk_idx_gt_npy[0, 0, i, :] - topk_idx_npy[0, 0, i, :])
        # print(f"[DEBUG] diff.min(): {cur_diff.min():.4f}, diff.max(): {cur_diff.max():.4f}")
    
    # topk to uint64
    blockmask_uint64 = mx.topk_to_uint64(topk_idx, K_LEN, BLOCK_SIZE) # (1, 2, 1024, 2)
    print(f"[DEBUG] blockmask_uint64.shape: {blockmask_uint64.shape}")
    # print(f"[DEBUG] blockmask_uint64[0, 0, 0, 0]: {blockmask_uint64[0, 0, 0, 0]:0b}")
    # exit()
    if CHECK_ACCURACY:
        topk_idx_npy = np.array(topk_idx).copy()
        # print(f"[DEBUG] topk_idx_npy.shape: {topk_idx_npy.shape}")
        blockmask_uint64_ref, _ = topk_to_uint64_numpy(topk_idx_npy, K_LEN, BLOCK_SIZE)
        # print(f"[DEBUG] blockmask_uint64_ref.shape: {blockmask_uint64_ref.shape}")
        blockmask_uint64_npy = np.array(blockmask_uint64)

        diff = np.abs(blockmask_uint64_ref - blockmask_uint64_npy)
        print(f"[DEBUG] diff.shape: {diff.shape}, min diff: {np.min(diff):.4f}, max diff: {np.max(diff):.4f}")
        # for i in range(16):
        #     print(f"[DEBUG] blockmask_uint64[0, 0, -1, {i}]: {blockmask_uint64[0, 0, -16, i]:0b}")
        #     print(f"[DEBUG] blockmask_uint64[0, 0, -1, {i}]: {blockmask_uint64[0, 0, -16, i]:0b}")
    # block sparse attention
    out_mlx = mx.fast.infllmv2_attention_stage2(q_mlx, k_mlx, v_mlx, cu_seqlens_q_mlx, cu_seqlens_k_mlx, max_seqlen_q, max_seqlen_k, window_size_left, window_size_right, blockmask_uint64, BLOCK_WINDOW_SIZE, scale=scale, mask="causal")
    
    if CHECK_ACCURACY:
        topk_idx_torch = torch.from_numpy(np.array(topk_idx)).to(torch.int32)
        base_blockmask = convert_topk_to_base_blockmask(topk_idx_torch[0], max_seqlen_k, BLOCK_SIZE, DEVICE)
        mixed_mask = prepare_mixed_mask(base_blockmask, cu_seqlens_q, cu_seqlens_k, Q_LEN, K_LEN, BATCH_SIZE, m_block_dim=1, n_block_dim=BLOCK_SIZE, block_window_size=BLOCK_WINDOW_SIZE)
        # print(f"[DEBUG] mixed_mask.shape: {mixed_mask.shape}")
        # print(f"[DEBUG] v_torch.shape: {v_torch.shape}")
        # exit()
        out_ref, attn_ref = attention_blocksparse_ref(
            q_torch_, # (1, 2048, 32, 128)
            k_torch_, # (1, 2048, 2, 128)
            v_torch_, # (1, 2048, 2, 128)
            mixed_mask, # (1, 32, 2048, 2048)
            query_padding_mask, # (1, 2048)
            key_padding_mask, # (1, 2048)
            0.0,
            None,  # dropout_mask
            causal=True, # True
        )

        out_ref = out_ref.permute(0, 2, 1, 3).contiguous() # (1, 2048, 32, 128) -> (1, 32, 2048, 128)
        # print("out_ref.shape", out_ref.shape)
        # print(f"out_ref min: {out_ref.min()}, max: {out_ref.max()}, mean: {out_ref.mean()}, std: {out_ref.std()}")
        out_ref_mlx = mx.array(out_ref.detach().cpu().numpy())

        diff = mx.abs(out_mlx - out_ref_mlx)
        print(f"[DEBUG] diff.shape: {diff.shape}, min diff: {diff.max():.4f}, max diff: {diff.min():.4f}")
        # print(f"[DEBUG] base_blockmask.shape: {base_blockmask.shape}")
        # print(f"[DEBUG] base_blockmask[0, 0, 0, 0]: {base_blockmask[0, 0, 0, 0]}")
        # exit()
    
    # print(f"[DEBUG] out_mlx.shape: {out_mlx.shape}")
    # print(f"[DEBUG] out_mlx[0, 0, 0, 0]: {out_mlx[0, 0, 0, 0]}")
    
    
