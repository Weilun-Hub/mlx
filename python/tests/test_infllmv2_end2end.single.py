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
K_LEN = 16 * 1024
print(f"[DEBUG] K_LEN: {K_LEN}")
COMPRESSED_K_LEN = K_LEN // 16
print(f"[DEBUG] COMPRESSED_K_LEN: {COMPRESSED_K_LEN}")
TOPK = 64
BLOCK_SIZE = 64
DTYPE = np.float16

INIT_BLOCK = 1
LOCAL_BLOCK = 8
KERNEL_SIZE = 5
STRIDE = 4
PADDING = 1
BLOCK_WINDOW_SIZE = 0

TORCH_MASK = False if Q_LEN == 1 else True
MLX_MASK = "causal" if TORCH_MASK else None

TEST_ITERS_INFLLMV2 = 20
TEST_ITERS_FA = 20

TEST_FA = False

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=4, suppress=True)

if __name__ == "__main__":

    np.random.seed(2)
    q_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_ATTN_HEADS, Q_LEN, HEAD_DIM)).astype(DTYPE)
    k_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_KEY_VALUE_HEADS, K_LEN, HEAD_DIM)).astype(DTYPE)
    compressed_k_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_KEY_VALUE_HEADS, COMPRESSED_K_LEN, HEAD_DIM)).astype(DTYPE)
    v_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_KEY_VALUE_HEADS, K_LEN, HEAD_DIM)).astype(DTYPE)
    scale = float(1.0 / math.sqrt(HEAD_DIM))

    q_mlx = mx.array(q_npy)
    k_mlx = mx.array(k_npy)
    compressed_k_mlx = mx.array(compressed_k_npy)
    v_mlx = mx.array(v_npy)

    print(f"[DEBUG] q_mlx.shape: {q_mlx.shape}")
    print(f"[DEBUG] k_mlx.shape: {k_mlx.shape}")
    print(f"[DEBUG] compressed_k_mlx.shape: {compressed_k_mlx.shape}")
    print(f"[DEBUG] v_mlx.shape: {v_mlx.shape}")

    # warmup
    o_mlx = mx.fast.scaled_dot_product_attention(q_mlx, k_mlx, v_mlx, scale=scale)

    start_event = torch.mps.Event(enable_timing=True)
    end_event = torch.mps.Event(enable_timing=True)

    start_event.record()

    o_mlx = mx.fast.scaled_dot_product_attention(q_mlx, k_mlx, v_mlx, scale=scale)
    print(f"[DEBUG] o_mlx[0, 0, 0, 0]: {o_mlx[0, 0, 0, 0]}")

    end_event.record()
    torch.mps.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    print(f"[INFO] fa impl time: {elapsed_time:.4f} ms")
    time.sleep(0.5) # sleep 0.5 s

    torch.mps.empty_cache()
    gc.collect()
    del start_event
    del end_event

    cu_seqlens_q_mlx = mx.array([0, Q_LEN])
    cu_seqlens_k_mlx = mx.array([0, K_LEN])
    max_seqlen_q = Q_LEN
    max_seqlen_k = K_LEN
    window_size = (-1, -1)
    window_size_left = window_size[0]
    window_size_right = window_size[1]

    torch.mps.empty_cache()
    start_event = torch.mps.Event(enable_timing=True)
    end_event = torch.mps.Event(enable_timing=True)

    start_event.record()
    # block score
    score = mx.fast.infllmv2_attention_stage1(q_mlx, compressed_k_mlx, v_mlx, scale=scale, mask=MLX_MASK) # (1, 2, 1024, 256)
    print(f"[DEBUG] score[0, 0, 0, 0]: {score[0, 0, 0, 0]}")
    # max pooling
    pooled_score = mx.maxpooling(score, K_LEN, INIT_BLOCK, LOCAL_BLOCK, KERNEL_SIZE, STRIDE, PADDING, BLOCK_SIZE) # (1, 2, 1024, 16)
    print(f"[DEBUG] pooled_score[0, 0, 0, 0]: {pooled_score[0, 0, 0, 0]}")
    # topk
    topk_idx = mx.argtopk(pooled_score, TOPK, axis=-1) # (1, 2, 1024, 32)
    print(f"[DEBUG] topk_idx[0, 0, 0, 0]: {topk_idx[0, 0, 0, 0]}")
    # topk to uint64
    blockmask_uint64 = mx.topk_to_uint64(topk_idx, K_LEN, BLOCK_SIZE) # (1, 2, 1024, 2)
    print(f"[DEBUG] blockmask_uint64[0, 0, 0, 0]: {blockmask_uint64[0, 0, 0, 0]}")
    # block sparse attention
    out_mlx = mx.fast.infllmv2_attention_stage2(q_mlx, k_mlx, v_mlx, cu_seqlens_q_mlx, cu_seqlens_k_mlx, max_seqlen_q, max_seqlen_k, window_size_left, window_size_right, blockmask_uint64, BLOCK_WINDOW_SIZE, scale=scale, mask="causal")
    print(f"[DEBUG] out_mlx[0, 0, 0, 0]: {out_mlx[0, 0, 0, 0]}")
    # get topkK_LEN
    end_event.record()
    torch.mps.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    torch.mps.empty_cache()
    gc.collect()
    print(f"[INFO] infllmv2 impl time: {elapsed_time:.4f} ms")
    time.sleep(0.5) # sleep 0.5 s

    del start_event
    del end_event
    gc.collect()

    exit()



    exit()

    # tmp = 0

    if TEST_FA:
        print(f"[DEBUG] q_mlx.shape: {q_mlx.shape}")
        print(f"[DEBUG] k_mlx.shape: {k_mlx.shape}")
        print(f"[DEBUG] compressed_k_mlx.shape: {compressed_k_mlx.shape}")
        print(f"[DEBUG] v_mlx.shape: {v_mlx.shape}")

        # warmup
        o_mlx = mx.fast.scaled_dot_product_attention(q_mlx, k_mlx, v_mlx, scale=scale)
        # tmp = o_mlx[0, 0, 0, 0]
        print(f"[DEBUG] o_mlx[0, 0, 0, 0]: {o_mlx[0, 0, 0, 0]}")

        start_event = torch.mps.Event(enable_timing=True)
        end_event = torch.mps.Event(enable_timing=True)

        time_list = []

        for i in range(TEST_ITERS_FA):
            try:
                torch.mps.empty_cache()
                start_event.record()

                o_mlx = mx.fast.scaled_dot_product_attention(q_mlx, k_mlx, v_mlx, scale=scale)
                # tmp += o_mlx[0, 0, 0, 0]
                print(f"[DEBUG] o_mlx[0, 0, 0, 0]: {o_mlx[0, 0, 0, 0]}")

                end_event.record()
                torch.mps.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)
            except Exception as e:
                print(f"[ERROR]: {e}, iteration: {i}")
                continue
            time_list.append(elapsed_time)
            torch.mps.empty_cache()
            gc.collect()
            print(f"[INFO] fa impl time @ iteration {i:02d}: {elapsed_time:.4f} ms")
            time.sleep(0.5) # sleep 0.5 s
        
        del start_event
        del end_event
        gc.collect()

        # print(f"[INFO] time_list: {time_list}")
        time_list_np = np.array(time_list)
        # filtered_time_list = time_list_np[(time_list_np != np.max(time_list_np)) & (time_list_np != np.min(time_list_np))]
        print(f"[INFO] fa mean time: {np.mean(time_list_np):.4f} ms")
        print(f"[INFO] fa std time: {np.std(time_list_np):.4f} ms")
        print(f"[INFO] fa min time: {np.min(time_list_np):.4f} ms")
        print(f"[INFO] fa max time: {np.max(time_list_np):.4f} ms")

        exit()

    cu_seqlens_q_mlx = mx.array([0, Q_LEN])
    cu_seqlens_k_mlx = mx.array([0, K_LEN])
    max_seqlen_q = Q_LEN
    max_seqlen_k = K_LEN
    window_size = (-1, -1)
    window_size_left = window_size[0]
    window_size_right = window_size[1]
    
    torch.mps.empty_cache()
    start_event = torch.mps.Event(enable_timing=True)
    end_event = torch.mps.Event(enable_timing=True)
    # tmp = 0

    start_event.record()
    # block score
    score = mx.fast.infllmv2_attention_stage1(q_mlx, compressed_k_mlx, v_mlx, scale=scale, mask=MLX_MASK) # (1, 2, 1024, 256)
    print(f"[DEBUG] score.shape: {score.shape}")
    print(f"[DEBUG] score[0, 0, 0, 0]: {score[0, 0, 0, 0]}")
    # max pooling
    pooled_score = mx.maxpooling(score, K_LEN, INIT_BLOCK, LOCAL_BLOCK, KERNEL_SIZE, STRIDE, PADDING, BLOCK_SIZE) # (1, 2, 1024, 16)
    print(f"[DEBUG] pooled_score.shape: {pooled_score.shape}")
    print(f"[DEBUG] pooled_score[0, 0, 0, 0]: {pooled_score[0, 0, 0, 0]}")
    # topk
    topk_idx = mx.argtopk(pooled_score, TOPK, axis=-1) # (1, 2, 1024, 32)
    print(f"[DEBUG] topk_idx.shape: {topk_idx.shape}")
    print(f"[DEBUG] topk_idx[0, 0, 0, 0]: {topk_idx[0, 0, 0, 0]}")
    # topk to uint64
    blockmask_uint64 = mx.topk_to_uint64(topk_idx, K_LEN, BLOCK_SIZE) # (1, 2, 1024, 2)
    print(f"[DEBUG] blockmask_uint64.shape: {blockmask_uint64.shape}")
    print(f"[DEBUG] blockmask_uint64[0, 0, 0, 0]: {blockmask_uint64[0, 0, 0, 0]}")
    # block sparse attention
    out_mlx = mx.fast.infllmv2_attention_stage2(q_mlx, k_mlx, v_mlx, cu_seqlens_q_mlx, cu_seqlens_k_mlx, max_seqlen_q, max_seqlen_k, window_size_left, window_size_right, blockmask_uint64, BLOCK_WINDOW_SIZE, scale=scale, mask="causal")
    print(f"[DEBUG] out_mlx.shape: {out_mlx.shape}")
    print(f"[DEBUG] out_mlx[0, 0, 0, 0]: {out_mlx[0, 0, 0, 0]}")
    # get topkK_LEN
    end_event.record()
    torch.mps.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    torch.mps.empty_cache()
    gc.collect()
    print(f"[INFO] infllmv2 impl time: {elapsed_time:.4f} ms")
    time.sleep(0.5) # sleep 0.5 s

    del start_event
    del end_event
    gc.collect()

    # exit()

    start_event = torch.mps.Event(enable_timing=True)
    end_event = torch.mps.Event(enable_timing=True)

    time_list = []

    for i in range(TEST_ITERS_INFLLMV2):
        try:
            torch.mps.empty_cache()
            start_event.record()

            score = mx.fast.infllmv2_attention_stage1(q_mlx, compressed_k_mlx, v_mlx, scale=scale, mask=MLX_MASK)
            print(f"[DEBUG] score[0, 0, 0, 0]: {score[0, 0, 0, 0]}")
            pooled_score = mx.maxpooling(score, K_LEN, INIT_BLOCK, LOCAL_BLOCK, KERNEL_SIZE, STRIDE, PADDING, BLOCK_SIZE)
            print(f"[DEBUG] pooled_score[0, 0, 0, 0]: {pooled_score[0, 0, 0, 0]}")
            topk_idx = mx.argtopk(pooled_score, TOPK, axis=-1)
            print(f"[DEBUG] topk_idx[0, 0, 0, 0]: {topk_idx[0, 0, 0, 0]}")
            blockmask_uint64 = mx.topk_to_uint64(topk_idx, K_LEN, BLOCK_SIZE)
            out_mlx = mx.fast.infllmv2_attention_stage2(q_mlx, k_mlx, v_mlx, cu_seqlens_q_mlx, cu_seqlens_k_mlx, max_seqlen_q, max_seqlen_k, window_size_left, window_size_right, blockmask_uint64, BLOCK_WINDOW_SIZE, scale=scale, mask="causal")
            # tmp += out_mlx[0, 0, 0, 0]
            print(f"[DEBUG] out_mlx[0, 0, 0, 0]: {out_mlx[0, 0, 0, 0]}")
            end_event.record()
            torch.mps.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
            torch.mps.empty_cache()
            gc.collect()
        except Exception as e:
            print(f"[ERROR]: {e}, iteration: {i}")
            continue
        time_list.append(elapsed_time)
        torch.mps.empty_cache()
        gc.collect()
        print(f"[INFO] infllmv2 impl time @ iteration {i:02d}: {elapsed_time:.4f} ms")
        time.sleep(0.5) # sleep 0.5 s
    
    del start_event
    del end_event
    gc.collect()

    # print(f"[INFO] time_list: {time_list}")
    time_list_np = np.array(time_list)
    filtered_time_list = time_list_np[(time_list_np != np.max(time_list_np)) & (time_list_np != np.min(time_list_np))]
    print(f"[INFO] infllmv2 mean time: {np.mean(filtered_time_list):.4f} ms")
    print(f"[INFO] infllmv2 std time: {np.std(filtered_time_list):.4f} ms")
    print(f"[INFO] infllmv2 min time: {np.min(filtered_time_list):.4f} ms")
    print(f"[INFO] infllmv2 max time: {np.max(filtered_time_list):.4f} ms")

