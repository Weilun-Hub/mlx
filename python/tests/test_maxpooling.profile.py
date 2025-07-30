import mlx.core as mx
import numpy as np
import torch
import gc
import time

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=4, suppress=True)
np.random.seed(0)

round_q = lambda len_q : (len_q * 16 + 128 - 1) // 128 * 128 // 16

BATCH_SIZE = 1
NUM_HEAD = 2
LEN_Q = 1
LEN_Q_ROUND = round_q(LEN_Q)
print(f"[INFO] LEN_Q: {LEN_Q}, LEN_Q_ROUND: {LEN_Q_ROUND}")
LEN_K = 4096 // 16
LEN_CACHE = 0
INIT_BLOCK = 1
LOCAL_BLOCK = 8
KERNEL_SIZE = 5
STRIDE = 4
PADDING = 1
BLOCK_SIZE = 64
DTYPE = np.float16

TEST_ITERS_MLX = 10

if __name__ == "__main__":
    
    score_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_HEAD, LEN_Q, LEN_K)).astype(DTYPE)

    score_mlx = mx.array(score_npy)

    # warm up
    max_val_mlx_pred = mx.maxpooling(score_mlx, LEN_CACHE, INIT_BLOCK, LOCAL_BLOCK, KERNEL_SIZE, STRIDE, PADDING, BLOCK_SIZE)
    max_val_npy_pred = np.array(max_val_mlx_pred)
    
    start_event = torch.mps.Event(enable_timing=True)
    end_event = torch.mps.Event(enable_timing=True)

    time_list = []

    for i in range(TEST_ITERS_MLX):
        try:
            torch.mps.empty_cache()
            start_event.record()
            max_val_mlx_pred = mx.maxpooling(score_mlx, LEN_CACHE, INIT_BLOCK, LOCAL_BLOCK, KERNEL_SIZE, STRIDE, PADDING, BLOCK_SIZE)
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