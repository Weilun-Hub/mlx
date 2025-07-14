import math
import time
import mlx.core as mx
import numpy as np

BATCH_SIZE = 1
NUM_ATTN_HEADS = 32
NUM_KEY_VALUE_HEADS = 2
HEAD_DIM = 128
Q_LEN = 1024 * 2  # Reduced from 16384 to 2048 to fit in GPU memory
K_LEN = 1024 * 2 // 16  # 128
DTYPE = np.float16

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
    q_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_ATTN_HEADS, Q_LEN, HEAD_DIM)).astype(DTYPE)
    k_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_KEY_VALUE_HEADS, K_LEN, HEAD_DIM)).astype(DTYPE)
    v_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_KEY_VALUE_HEADS, K_LEN, HEAD_DIM)).astype(DTYPE)

    q_mlx = mx.array(q_npy)
    k_mlx = mx.array(k_npy)
    v_mlx = mx.array(v_npy)

    scale = float(1.0 / math.sqrt(HEAD_DIM))

    reference = mlx_primitives_sdpa_with_gqa(q_mlx, k_mlx, v_mlx, scale)
    print(f"reference: {reference.shape}")
    o_mlx = mx.fast.scaled_dot_product_attention(q_mlx, k_mlx, v_mlx, scale=scale)
    print(f"o_mlx: {o_mlx.shape}")

    diff = mx.abs(o_mlx - reference)
    print(f"diff.max(): {diff.max():.4f}")
    print(f"diff.min(): {diff.min():.4f}")

    exit()

    # Warmup runs
    for _ in range(2):
        result = mx.fast.scaled_dot_product_attention(q_mlx, k_mlx, v_mlx, scale=scale)
        mx.eval(result)
    
    # Actual timing runs
    nIter = 20
    start = time.perf_counter()
    for _ in range(nIter):
        result = mx.fast.scaled_dot_product_attention(q_mlx, k_mlx, v_mlx, scale=scale)
        mx.eval(result)  # Ensure computation is complete
    end = time.perf_counter()
    elapsedTime = (end - start) / nIter
    print(f"fast.scaled_dot_product_attention time: {elapsedTime * 1000:.2f} ms")

    start = time.perf_counter()
    for _ in range(nIter):
        result = mlx_primitives_sdpa_with_gqa(q_mlx, k_mlx, v_mlx, scale=scale)
        mx.eval(result)  # Ensure computation is complete
    end = time.perf_counter()
    elapsedTime = (end - start) / nIter
    print(f"mlx_primitives_sdpa_with_gqa time: {elapsedTime * 1000:.2f} ms")
    