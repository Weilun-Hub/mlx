import mlx.core as mx
import numpy as np
import torch
import gc
import time

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=4, suppress=True)
np.random.seed(0)

TOPK = 32
BLOCK_SIZE = 64
BATCH_SIZE = 1
NUM_HEAD = 2
LEN_Q = 1
LEN_K = 8192
NUM_BLOCK = LEN_K // BLOCK_SIZE

DEVICE = "mps"
REQUIRE_GRAD = False
DTYPE = torch.float16

TEST_ITERS_MLX = 10

if __name__ == "__main__":

    score = torch.randn(BATCH_SIZE, NUM_HEAD, LEN_Q, NUM_BLOCK, device=DEVICE, dtype=DTYPE, requires_grad=REQUIRE_GRAD)
    
    score_mlx = mx.array(score.cpu().numpy())
    
    # warm up
    mx_argtopk = mx.argtopk(score_mlx, TOPK, axis=-1)
    # print(mx_argtopk.shape)

    start_event = torch.mps.Event(enable_timing=True)
    end_event = torch.mps.Event(enable_timing=True)

    time_list = []

    for i in range(TEST_ITERS_MLX):
        try:
            torch.mps.empty_cache()
            start_event.record()
            mx_argtopk = mx.argtopk(score_mlx, TOPK, axis=-1)
            end_event.record()
            torch.mps.synchronize()
            elapsed_time = start_event.elapsed_time(end_event)
        except Exception as e:
            print(f"[ERROR] {e}, iteration: {i}")
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
