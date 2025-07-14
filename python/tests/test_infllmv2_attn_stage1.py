import math
import time
import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

BATCH_SIZE = 1
NUM_ATTN_HEADS = 32
NUM_KEY_VALUE_HEADS = 2
HEAD_DIM = 128
Q_LEN = 1024 * 2  # Reduced from 16384 to 2048 to fit in GPU memory
K_LEN = 1024 * 2 // 16  # 128
DTYPE = np.float16

def naive_infllmv2_attn_stage1_mlx(q, k, v, causal=False):

    batch_size, n_q_head, q_len, head_dim = q.shape
    _, n_kv_head, k_len, _ = k.shape

    n_repeat = n_q_head // n_kv_head  # 32 // 2 = 16
    k = mx.repeat(k, n_repeat, axis=1)
    v = mx.repeat(v, n_repeat, axis=1)

    scale = float(1.0 / math.sqrt(head_dim))

    score = q @ k.transpose(0, 1, 3, 2) * scale
    if causal:
        print("NYI")
        exit()
    score = mx.softmax(score, axis=-1)
    score = score.reshape(batch_size, n_kv_head, n_repeat, q_len, k_len)
    score = score.sum(axis=2)
    
    return score

def naive_infllmv2_attn_stage1_torch(q, k, v, causal=False):
    k = k.repeat_interleave(q.shape[0] // k.shape[0], dim=0)
    v = v.repeat_interleave(q.shape[0] // v.shape[0], dim=0)
    
    attn = q @ k.transpose(-2, -1) / (q.size(-1) ** 0.5)
    if causal:
        causal_mask = torch.zeros(q.shape[1], k.shape[1], device=q.device).bool()
        for i in range(q.shape[1]):
            for j in range(k.shape[1]):
                if i >= (j * 16 + 32 - 1):
                    causal_mask[i, j] = True
        attn = attn.masked_fill(~causal_mask, -float('inf'))
    score = F.softmax(attn, dim=-1)
    score = score.reshape(2, 16, q.shape[1], k.shape[1]).sum(dim=1)
    return score


if __name__ == "__main__":
    q_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_ATTN_HEADS, Q_LEN, HEAD_DIM)).astype(DTYPE)
    k_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_KEY_VALUE_HEADS, K_LEN, HEAD_DIM)).astype(DTYPE)
    v_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_KEY_VALUE_HEADS, K_LEN, HEAD_DIM)).astype(DTYPE)

    q_mlx = mx.array(q_npy)
    k_mlx = mx.array(k_npy)
    v_mlx = mx.array(v_npy)

    q_torch = torch.from_numpy(q_npy.squeeze(0))
    k_torch = torch.from_numpy(k_npy.squeeze(0))
    v_torch = torch.from_numpy(v_npy.squeeze(0))

    score_mlx = naive_infllmv2_attn_stage1_mlx(q_mlx, k_mlx, v_mlx, causal=False)
    score_mlx_npy = np.array(score_mlx).squeeze(0)
    score_torch = naive_infllmv2_attn_stage1_torch(q_torch, k_torch, v_torch, causal=False)
    score_torch_npy = score_torch.numpy()

    diff = np.abs(score_mlx_npy - score_torch_npy)
    print("max |diff| between mlx and torch: ", diff.max())

    scale = float(1.0 / math.sqrt(HEAD_DIM))
    
    o_mlx = mx.fast.infllmv2_attention_stage1(q_mlx, k_mlx, v_mlx, scale=scale)
    print(f"o_mlx.shape: {o_mlx.shape}")
    print(o_mlx.sum())