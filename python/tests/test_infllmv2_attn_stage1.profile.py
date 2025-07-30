import math
import time
import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
import gc

BATCH_SIZE = 1
NUM_ATTN_HEADS = 32
NUM_KEY_VALUE_HEADS = 2
HEAD_DIM = 128
Q_LEN = 1
K_LEN = (8 * 1024) // 16
DTYPE = np.float16

TORCH_MASK = False if Q_LEN == 1 else True
MLX_MASK = "causal" if TORCH_MASK else None

TEST_ITERS_TORCH = 1
TEST_ITERS_MLX = 10

np.set_printoptions(linewidth=np.inf)
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
    return score

if __name__ == "__main__":

    np.random.seed(0)
    q_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_ATTN_HEADS, Q_LEN, HEAD_DIM)).astype(DTYPE)
    k_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_KEY_VALUE_HEADS, K_LEN, HEAD_DIM)).astype(DTYPE)
    v_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_KEY_VALUE_HEADS, K_LEN, HEAD_DIM)).astype(DTYPE)

    q_mlx = mx.array(q_npy)
    k_mlx = mx.array(k_npy)
    v_mlx = mx.array(v_npy)

    q_torch = torch.from_numpy(q_npy.squeeze(0))
    k_torch = torch.from_numpy(k_npy.squeeze(0))
    v_torch = torch.from_numpy(v_npy.squeeze(0))

    # exit()

    # warm up
    torch.mps.empty_cache()
    # exit()
    score_torch = naive_infllmv2_attn_stage1_torch(q_torch, k_torch, v_torch, causal=TORCH_MASK)
    # exit()
    
    start_event = torch.mps.Event(enable_timing=True)
    end_event = torch.mps.Event(enable_timing=True)

    time_list = []
    for _ in range(TEST_ITERS_TORCH):
        torch.mps.empty_cache()

        start_event.record()

        score_torch = naive_infllmv2_attn_stage1_torch(q_torch, k_torch, v_torch, causal=TORCH_MASK)
        
        end_event.record()
        torch.mps.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        
        time_list.append(elapsed_time)
        torch.mps.empty_cache()
        gc.collect()
        print(f"[INFO] torch impl time: {elapsed_time:.4f} ms")
        time.sleep(0.5) # sleep 0.5 s
        # exit()
    
    del start_event
    del end_event
    gc.collect()

    print(f"[INFO] torch mean time: {np.mean(time_list):.2f} ms")
    print(f"[INFO] torch std time: {np.std(time_list):.2f} ms")
    print(f"[INFO] torch min time: {np.min(time_list):.2f} ms")
    print(f"[INFO] torch max time: {np.max(time_list):.2f} ms")
    
    score_torch_npy = score_torch.numpy()

    scale = float(1.0 / math.sqrt(HEAD_DIM))

    # warm up
    o_mlx = mx.fast.infllmv2_attention_stage1(q_mlx, k_mlx, v_mlx, scale=scale, mask=MLX_MASK)

    start_event = torch.mps.Event(enable_timing=True)
    end_event = torch.mps.Event(enable_timing=True)

    time_list = []
    for i in range(TEST_ITERS_MLX):
        try:
            torch.mps.empty_cache()
            
            start_event.record()
            o_mlx = mx.fast.infllmv2_attention_stage1(q_mlx, k_mlx, v_mlx, scale=scale, mask=MLX_MASK)
            end_event.record()
            torch.mps.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
        except Exception as e:
            print(f"[ERROR]: {e}, iteration: {i}")
            continue
        time_list.append(elapsed_time)
        torch.mps.empty_cache()
        gc.collect()
        print(f"[INFO] mlx impl time: {elapsed_time:.4f} ms")
        time.sleep(0.5) # sleep 0.5 s

    del start_event
    del end_event
    gc.collect()

    print(f"[INFO] mlx mean time: {np.mean(time_list):.2f} ms")
    print(f"[INFO] mlx std time: {np.std(time_list):.2f} ms")
    print(f"[INFO] mlx min time: {np.min(time_list):.2f} ms")
    print(f"[INFO] mlx max time: {np.max(time_list):.2f} ms")

    o_mlx_npy = np.array(o_mlx).squeeze(0)
    diff = np.abs(score_torch_npy - o_mlx_npy)
    print(f"min |diff|: {np.nanmin(diff)}, max |diff|: {np.nanmax(diff)}")

