import mlx.core as mx
import numpy as np

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=4, suppress=True)
np.random.seed(0)

TOPK = 64

BATCH_SIZE = 1
NUM_HEAD = 2
LEN_Q = 14001
NUM_BLOCK = 219


if __name__ == "__main__":
    
    block_score = np.fromfile("./data/block_score.2x14001x219.fp16.bin", dtype=np.float16).reshape(BATCH_SIZE, NUM_HEAD, LEN_Q, NUM_BLOCK)
    topk_index = np.fromfile("./data/topk_idx.2x14001x64.int32.bin", dtype=np.int32).reshape(BATCH_SIZE, NUM_HEAD, LEN_Q, TOPK)
    print(block_score.shape)

    block_score_mx = mx.array(block_score)
    
    mx_topk = mx.topk(block_score_mx, TOPK, axis=-1)
    print(mx_topk.shape)
    print(mx_topk[0, 0, -1, :])
    # exit()

    mx_argtopk = mx.argtopk(block_score_mx, TOPK, axis=-1)
    npy_argtopk = np.array(mx_argtopk)
    # print(mx_argtopk.shape)
    # print(np.sort(npy_argtopk[0, 0, -1, :]))
    print(npy_argtopk[0, 0, -1, :])
    print(topk_index[0, 0, -1, :])
    diff = np.abs(npy_argtopk - topk_index)
    print("max diff", diff.max())

