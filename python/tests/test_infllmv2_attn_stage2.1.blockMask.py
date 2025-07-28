import math
import time
import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

np.random.seed(42)
torch.random.manual_seed(42)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=4, suppress=True)

BATCH_SIZE = 1
SEQLEN_Q = 2048
SEQLEN_K = 2048
D = 128
NHEADS = 32
NHEADS_K = 2
BLOCK_SIZE = 64
CAUSAL = False
DTYPE = torch.float16
SPARSITY = 0.8
# SPARSITY = 0
BLOCK_WINDOW_SIZE = 0
DEVICE = "mps"
REQUIRE_GRAD = False

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


def pad_input(tensor_unpad, indices, batch_size, seqlen):
    """
    简化的pad_input替代函数
    将unpadded tensor重新padding为指定的shape
    """
    # 这是一个简化版本，适用于当前的使用场景
    total_tokens, *other_dims = tensor_unpad.shape
    output_shape = [batch_size, seqlen] + other_dims
    output = torch.zeros(output_shape, dtype=tensor_unpad.dtype, device=tensor_unpad.device)
    
    # 使用indices重建tensor
    # 注意：这是一个简化实现，可能需要根据具体使用场景调整
    if indices is not None and len(indices) == total_tokens:
        # 将tokens放回到正确的位置
        for i, (batch_idx, seq_idx) in enumerate(indices):
            if batch_idx < batch_size and seq_idx < seqlen:
                output[batch_idx, seq_idx] = tensor_unpad[i]
    else:
        # 如果没有indices，按顺序填充（假设是连续的）
        tokens_per_batch = total_tokens // batch_size
        for b in range(batch_size):
            start_idx = b * tokens_per_batch
            end_idx = min((b + 1) * tokens_per_batch, total_tokens)
            actual_len = min(end_idx - start_idx, seqlen)
            if actual_len > 0:
                output[b, :actual_len] = tensor_unpad[start_idx:start_idx + actual_len]
    
    return output

def unpad_input(input_tensor, attention_mask):
    """
    简化的unpad_input替代函数
    根据attention mask移除padding
    """
    batch_size, seqlen = attention_mask.shape
    # 获取有效token的位置
    valid_mask = attention_mask.bool()
    
    # 计算cumulative lengths
    seqlens = valid_mask.sum(dim=1)
    cu_seqlens = torch.cat([torch.tensor([0], device=input_tensor.device), seqlens.cumsum(dim=0)])
    
    # 获取有效tokens的indices
    indices = []
    for b in range(batch_size):
        for s in range(seqlen):
            if valid_mask[b, s]:
                indices.append((b, s))
    
    # 提取有效tokens
    output_list = []
    for b, s in indices:
        output_list.append(input_tensor[b, s])
    
    output_unpad = torch.stack(output_list) if output_list else torch.empty((0,) + input_tensor.shape[2:], device=input_tensor.device, dtype=input_tensor.dtype)
    max_seqlen = seqlens.max().item()
    
    return output_unpad, indices, cu_seqlens.to(torch.int32), max_seqlen

def generate_qkv(
    q, k, v, query_padding_mask=None, key_padding_mask=None, kvpacked=False, qkvpacked=False
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )

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
    print("+" + '-' * 50 + "+")
    print("|start debug prepare_mixed_mask")
    print("+" + '-' * 50 + "+")
    print("base_blockmask.shape", base_blockmask.shape)
    print("cu_seqlens_q", cu_seqlens_q.cpu().numpy())
    print("cu_seqlens_k", cu_seqlens_k.cpu().numpy())
    print("seqlen_q", seqlen_q)
    print("seqlen_k", seqlen_k)
    print("batch_size", batch_size)
    print("nheads", nheads)
    print("nheads_k", nheads_k)
    print("m_block_dim", m_block_dim)
    print("n_block_dim", n_block_dim)
    print("block_window_size", block_window_size)
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
            print("batch_indices.shape", batch_indices.shape)
            print(f"batch_indices.min(): {batch_indices.min().cpu().numpy()}, batch_indices.max(): {batch_indices.max().cpu().numpy()}")
        
        # Calculate relative positions within each batch
        relative_positions = q_positions - cu_seqlens_q[batch_indices] # 2046 elements, [0, 1, 2, ..., 2045]
        print("relative_positions.shape", relative_positions.shape)
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
    print(f"g: {int(nheads / nheads_k)}, n: {n_block_dim}")
    print(f"modefied_blockmask changed from {modified_blockmask.shape} to {expanded_mask.shape}")
    print("expanded_mask[:, -1, 0]", expanded_mask[:, -1, 0])
    print("modified_blockmask[:, -1, 0]", modified_blockmask[:, -1, 0])
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
    print("mixed_mask.shape", mixed_mask.shape)
    print("+" + '-' * 50 + "+")
    print("|end debug prepare_mixed_mask")
    print("+" + '-' * 50 + "+")
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
    print("+" + '-' * 50 + "+")
    print("|start debug attention_blocksparse_ref")
    print("+" + '-' * 50 + "+")
    if causal:
        print("causal is True, window_size", window_size)
        window_size = (window_size[0], 0) # (-1 , -1) -> (-1, 0)
        print("causal is True, window_size", window_size)
    dtype_og = q.dtype
    if upcast:
        print("upcast is True")
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1] # (1, 2048, 32, 128) -> 2048, (1, 2048, 2, 128) -> 2048
    print("seqlen_q", seqlen_q)
    print("seqlen_k", seqlen_k)
    print("k.shape", k.shape, "v.shape", v.shape)
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2]) # (1, 2048, 2, 128) -> (1, 2048, 32, 128)
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2]) # (1, 2048, 2, 128) -> (1, 2048, 32, 128)
    print("k.shape", k.shape, "v.shape", v.shape)
    d = q.shape[-1] # 128
    print("reorder_ops", reorder_ops)
    if not reorder_ops: # reorder_ops is False
        # here q.shape = (1, 2048, 32, 128), k.shape = (1, 2048, 32, 128)
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k) # (1, 2048, 32, 128) * (1, 2048, 32, 128) -> (1, 32, 2048, 2048)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    print("scores.shape", scores.shape)
    if key_padding_mask is not None: # 1, 2048
        print("key_padding_mask.shape", key_padding_mask.shape)
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
        print("local mask.shape", local_mask.shape)
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
    print("+" + '-' * 50 + "+")
    print("|end debug attention_blocksparse_ref")
    print("+" + '-' * 50 + "+")
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)

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

# SDPA for MHA (n_heads == n_kv_heads)
def mlx_primitives_sdpa(q, k, v, scale, mask=None):
    p = (q * scale) @ k.transpose(0, 1, 3, 2)
    if mask is not None:
        if mask == "causal":
            q_offset = max(0, k.shape[2] - q.shape[2])
            q_indices = mx.arange(q_offset, q_offset + q.shape[2])
            k_indices = mx.arange(k.shape[2])
            mask = q_indices[:, None] >= k_indices[None]
            p = mx.where(mask, p, mx.finfo(mx.float32).min)
        elif mask.dtype == mx.bool_:
            p = mx.where(mask, p, mx.finfo(mx.float32).min)
        else:
            p += mask
    scores = mx.softmax(p.astype(mx.float32), axis=-1).astype(p.dtype)
    return scores @ v

# SDPA for GQA (n_heads > n_kv_heads, n_kv_heads > 1, n_heads % n_kv_heads == 0)
def mlx_primitives_sdpa_with_gqa(q, k, v, scale, mask=None):
    n_repeats = q.shape[1] // k.shape[1]

    # borrowing kv cache tiling from mlx-examples/llms/mistral/mistral.py
    n_heads = q.shape[1]
    B = q.shape[0]
    L = k.shape[2]

    def repeat(a):
        a = mx.concatenate([mx.expand_dims(a, 2)] * n_repeats, axis=2)
        return a.reshape([B, n_heads, L, -1])

    k, v = map(repeat, (k, v))

    return mlx_primitives_sdpa(q, k, v, scale, mask=mask)

if __name__ == "__main__":

    q = torch.randn(BATCH_SIZE, SEQLEN_Q, NHEADS, D, device=DEVICE, dtype=DTYPE, requires_grad=REQUIRE_GRAD)
    k = torch.randn(BATCH_SIZE, SEQLEN_K, NHEADS_K, D, device=DEVICE, dtype=DTYPE, requires_grad=REQUIRE_GRAD)
    v = torch.randn(BATCH_SIZE, SEQLEN_K, NHEADS_K, D, device=DEVICE, dtype=DTYPE, requires_grad=REQUIRE_GRAD)
    print("q.shape", q.shape)
    print("k.shape", k.shape)
    print("v.shape", v.shape)

    query_padding_mask = generate_random_padding_mask(SEQLEN_Q, BATCH_SIZE, DEVICE, mode="random")
    key_padding_mask = generate_random_padding_mask(SEQLEN_K, BATCH_SIZE, DEVICE, mode="random")
    print("query_padding_mask.shape", query_padding_mask.shape)
    print("key_padding_mask.shape", key_padding_mask.shape)

    query_padding_mask[query_padding_mask == False] = True
    key_padding_mask[key_padding_mask == False] = True
    assert query_padding_mask.sum() == query_padding_mask.numel()
    assert key_padding_mask.sum() == key_padding_mask.numel()

    # print(query_padding_mask)
    # print(key_padding_mask[0,-10:])
    # exit()

    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    print("q_unpad.shape", q_unpad.shape)
    print("k_unpad.shape", k_unpad.shape)
    print("v_unpad.shape", v_unpad.shape)
    print("cu_seqlens_q", cu_seqlens_q)
    print("cu_seqlens_k", cu_seqlens_k)
    print("max_seqlen_q", max_seqlen_q)
    print("max_seqlen_k", max_seqlen_k)
    print("q.shape", q.shape)
    print("k.shape", k.shape)
    print("v.shape", v.shape)

    assert cu_seqlens_q[0] == 0 and cu_seqlens_q[1] == max_seqlen_q
    assert cu_seqlens_k[0] == 0 and cu_seqlens_k[1] == max_seqlen_k
    assert (q_unpad == q[0, cu_seqlens_q[0]:cu_seqlens_q[1], : , :]).all()
    assert (k_unpad == k[0, cu_seqlens_k[0]:cu_seqlens_k[1], : , :]).all()
    assert (v_unpad == v[0, cu_seqlens_k[0]:cu_seqlens_k[1], : , :]).all()

    # exit()

    total_seqlen_q = q_unpad.shape[0]
    topk_idx = generate_topk_indices(NHEADS_K, total_seqlen_q, max_seqlen_k, SPARSITY, BLOCK_SIZE, DEVICE)
    print("topk_idx.shape", topk_idx.shape)

    base_blockmask = convert_topk_to_base_blockmask(topk_idx, max_seqlen_k, BLOCK_SIZE, DEVICE)
    print("base_blockmask.shape", base_blockmask.shape)

    mixed_mask = prepare_mixed_mask(base_blockmask, cu_seqlens_q, cu_seqlens_k, SEQLEN_Q, SEQLEN_K, BATCH_SIZE, m_block_dim=1, n_block_dim=BLOCK_SIZE, block_window_size=BLOCK_WINDOW_SIZE)
    # mixed_mask.shape: (1, 32, 2048, 2048)
    print("q.shape", q.shape)
    print("k.shape", k.shape)
    print("v.shape", v.shape)
    print("mixed_mask.shape", mixed_mask.shape)
    print("query_padding_mask.shape", query_padding_mask.shape)
    print("key_padding_mask.shape", key_padding_mask.shape)
    print("causal", CAUSAL)

    out_ref, attn_ref = attention_blocksparse_ref(
        q, # (1, 2048, 32, 128)
        k, # (1, 2048, 2, 128)
        v, # (1, 2048, 2, 128)
        mixed_mask, # (1, 32, 2048, 2048)
        query_padding_mask, # (1, 2048)
        key_padding_mask, # (1, 2048)
        0.0,
        None,  # dropout_mask
        causal=CAUSAL, # True
    )
    # print("out_ref.shape", out_ref.shape)
    # print(f"out_ref min: {out_ref.min()}, max: {out_ref.max()}, mean: {out_ref.mean()}, std: {out_ref.std()}")

    out_ref = out_ref.permute(0, 2, 1, 3).contiguous() # (1, 2048, 32, 128) -> (1, 32, 2048, 128)
    print("out_ref.shape", out_ref.shape)
    print(f"out_ref min: {out_ref.min()}, max: {out_ref.max()}, mean: {out_ref.mean()}, std: {out_ref.std()}")
    out_ref_mlx = mx.array(out_ref.detach().cpu().numpy())

    q_mlx = mx.array(q.permute(0, 2, 1, 3).contiguous().detach().cpu().numpy())
    k_mlx = mx.array(k.permute(0, 2, 1, 3).contiguous().detach().cpu().numpy())
    v_mlx = mx.array(v.permute(0, 2, 1, 3).contiguous().detach().cpu().numpy())
    print("q_mlx.shape", q_mlx.shape)
    print("k_mlx.shape", k_mlx.shape)
    print("v_mlx.shape", v_mlx.shape)

    scale = float(1.0 / math.sqrt(D))

    # reference = mlx_primitives_sdpa_with_gqa(q_mlx, k_mlx, v_mlx, scale) # , mask="causal"
    # print(f"reference: {reference.shape}")

    # diff = mx.abs(out_ref_mlx - reference)
    # print(f"diff.max(): {diff.max():.4f}")
    # print(f"diff.min(): {diff.min():.4f}")

    # exit()
    # o_mlx = mx.fast.scaled_dot_product_attention(q_mlx, k_mlx, v_mlx, scale=scale)
    # print(f"o_mlx: {o_mlx.shape}")

    # diff = mx.abs(o_mlx - reference)
    # print(f"diff.max(): {diff.max():.4f}")
    # print(f"diff.min(): {diff.min():.4f}")

    window_size = (-1, -1)
    window_size_left = window_size[0]
    window_size_right = window_size[1]
    
    blockmask_uint64, last_dim = topk_to_uint64_numpy(topk_idx.cpu().numpy(), max_seqlen_k, BLOCK_SIZE)
    print("blockmask_uint64.shape", blockmask_uint64.shape)
    blockmask_uint64 = blockmask_uint64.reshape(1, NHEADS_K, SEQLEN_Q, last_dim)
    print("blockmask_uint64.shape", blockmask_uint64.shape)
    blockmask_uint64 = mx.array(blockmask_uint64)
    
    # exit()
    
    cu_seqlens_q_mlx = mx.array(cu_seqlens_q.detach().cpu().numpy())
    cu_seqlens_k_mlx = mx.array(cu_seqlens_k.detach().cpu().numpy())
    out_mlx = mx.fast.infllmv2_attention_stage2(q_mlx, k_mlx, v_mlx, cu_seqlens_q_mlx, cu_seqlens_k_mlx, max_seqlen_q, max_seqlen_k, window_size_left, window_size_right, blockmask_uint64, BLOCK_WINDOW_SIZE, scale=scale)
    print("out_mlx.shape", out_mlx.shape)
    
    diff = mx.abs(out_mlx - out_ref_mlx)
    print("diff.shape:",diff.shape)
    print(f"diff.max(): {diff.max():.4f}")
    print(f"diff.min(): {diff.min():.4f}")
    print(f"pred.shape: {out_mlx.shape}")
    print(f"gt.shape  : {out_ref_mlx.shape}")
    print(f"pred[0, 0, 0, 0 : 16]:")
    print(np.array(out_mlx[0, 0, 0, : 16]))
    # print(f"pred[0, 0, 0, 0]: {out_mlx[0, 0, 0, 0]:0b}")
    print(f"gt[0, 0, 0, 0 : 16]:")
    print(np.array(out_ref_mlx[0, 0, 0, : 16]))
    print(f"blockmask_uint64.shape: {blockmask_uint64.shape}")
    print(f"blockmask_uint64[0, 0, 0, 0]: {blockmask_uint64[0, 0, 0, 0]:0b}")
    # exit()
