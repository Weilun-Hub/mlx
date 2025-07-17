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

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(
    precision=4,   # 小数点后位数
    suppress=True  # 禁止科学计数法（如 0.0001 显示为 0.0001 而非 1e-4）
)

def naive_infllmv2_attn_stage1_mlx(q, k, v, causal=False):

    batch_size, n_q_head, q_len, head_dim = q.shape
    _, n_kv_head, k_len, _ = k.shape

    n_repeat = n_q_head // n_kv_head  # 32 // 2 = 16
    k = mx.repeat(k, n_repeat, axis=1)
    v = mx.repeat(v, n_repeat, axis=1)

    scale = float(1.0 / math.sqrt(head_dim))

    score = q @ k.transpose(0, 1, 3, 2) * scale
    # print(f"score.shape: {score.shape}")
    if causal:
        print("NYI")
        exit()
    score = mx.softmax(score, axis=-1)

    # score = score.reshape(batch_size, n_kv_head, n_repeat, q_len, k_len)
    # score = score.sum(axis=2)
    
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

    score_mlx = naive_infllmv2_attn_stage1_mlx(q_mlx, k_mlx, v_mlx, causal=False)
    score_mlx_npy = np.array(score_mlx).squeeze(0)
    # print(score_mlx_npy.shape)
    # exit()
    score_mlx_npy = score_mlx_npy.reshape(NUM_ATTN_HEADS, Q_LEN // 16, 16, HEAD_DIM).sum(axis=-2)
    # print(score_mlx_npy.shape)
    # exit()
    # score_torch = naive_infllmv2_attn_stage1_torch(q_torch, k_torch, v_torch, causal=False)
    # score_torch_npy = score_torch.numpy()

    # diff = np.abs(score_mlx_npy - score_torch_npy)
    # print("max |diff| between mlx and torch: ", diff.max())
    # exit()

    scale = float(1.0 / math.sqrt(HEAD_DIM))

    o_mlx = mx.fast.infllmv2_attention_stage1(q_mlx, k_mlx, v_mlx, scale=scale)
    print(f"o_mlx.shape: {o_mlx.shape}")
    # print(f"o_mlx.min(): {o_mlx.min()}, o_mlx.max(): {o_mlx.max()}")
    # exit()

    o_mlx_npy = np.array(o_mlx).squeeze(0)
    print("o_mlx_npy.shape: ", o_mlx_npy.shape)
    # exit()
    # print("pred zero elements: ", (o_mlx_npy == 0).sum(), "out of ", o_mlx_npy.size, "=", (o_mlx_npy == 0).sum() / o_mlx_npy.size)
    # print("gt zero elements: ", (score_mlx_npy == 0).sum(), "out of ", score_mlx_npy.size, "=", (score_mlx_npy == 0).sum() / score_mlx_npy.size)
    
    diff = np.abs(o_mlx_npy - score_mlx_npy)
    print(f"max |diff| between mlx and torch: {diff.max()}")
    exit()

    # diff = np.abs(o_mlx_npy[:, :, :8] - score_mlx_npy[:, :, :8])
    # print(f"max |diff|[:, :, :8] between mlx and torch: {diff.max()}")
    # diff = np.abs(o_mlx_npy[:, :, 8:] - score_mlx_npy[:, :, 8:])
    # print(f"max |diff|[:, :, 8:] between mlx and torch: {diff.max()}")

    # head = 3
    # q_start, q_end = 0, 16
    # k_start, k_end = 0, 16
    # print("+------- pred ---------+")
    # print("line 0 - 15")
    # for i in range(q_start, q_end):
    #     print(o_mlx_npy[head, i, k_start : k_end])
    
    # print("line 16 - 31")
    # for i in range(q_start, q_end):
    #     print(o_mlx_npy[head, i + 16, k_start : k_end])

    # print("line 32 - 47")
    # for i in range(q_start, q_end):
    #     print(o_mlx_npy[head, i + 32, k_start : k_end])
    # print("sum")
    # print(o_mlx_npy[head, q_start + 32, k_start : k_end])
    # print(o_mlx_npy[head, q_start + 32 + 8, k_start : k_end])
    # print(o_mlx_npy[head, q_start + 32, k_start : k_end] + o_mlx_npy[head, q_start + 32 + 8, k_start : k_end])
    # print("line 48 - 63")
    # for i in range(q_start, q_end):
    #     print(o_mlx_npy[head, i + 48, k_start : k_end])

    # print("+------- gt ---------+")
    # for i in range(q_start, q_end):
    #     print(score_mlx_npy[head, i + 32, k_start : k_end])
    # print("line 0 - 7")
    # print(score_mlx_npy[head, 0:8, k_start : k_end].sum(axis=-2))
    # print("line 8 - 15")
    # print(score_mlx_npy[head, 8:16, k_start : k_end].sum(axis=-2))
    # print("line 0 - 15")
    # print(score_mlx_npy[head, 0:16, k_start : k_end].sum(axis=-2))
    # print("line 16 - 31")
    # print(score_mlx_npy[head, 16:32, k_start : k_end].sum(axis=-2))
    # print("line 32 - 47")
    # print(score_mlx_npy[head, 32:48, k_start : k_end].sum(axis=-2))
    # print("line 48 - 63")
    # print(score_mlx_npy[head, 48:64, k_start : k_end].sum(axis=-2))、
    
    # exit()

    # print("=" * 10)
    # gt = score_mlx_npy[head, 32:48, k_start : k_end].sum(axis=-2)
    # pred = o_mlx_npy[head, q_start + 32, k_start : k_end]
    # diff = np.abs(gt - pred)
    # print(f"max |diff| between mlx and torch: {diff.max()}")
    # for idx, (cur_gt, cur_pred) in enumerate(zip(gt, pred)):
    #     print(f"idx: {idx:03d}, cur_gt: {cur_gt:.4f}, cur_pred: {cur_pred:.4f}, diff: {np.abs(cur_gt - cur_pred):.4f}")
    #     # print("=" * 10)
    
    # exit()
    # print("=" * 10)
    # gt = score_mlx_npy.reshape(NUM_ATTN_HEADS, Q_LEN // 16, 16, HEAD_DIM).sum(axis=-2)
    # print("gt")
    # # print(gt[head, 0, k_start : k_end])
    # # gt = gt[head, 0]
    # print(gt[30, q_start:q_end, k_start:k_end])
    # pred = o_mlx_npy[:, : Q_LEN // 16, :] # [head, 0]
    # # print(pred.shape)
    # print("pred")
    # print(pred[30, q_start:q_end, k_start:k_end])
    # diff = np.abs(gt - pred)#[:, :, k_start : 16]
    # print(f"max |diff| between mlx and torch: {diff.max()}")
    # # exit()

    # for idx, (cur_gt, cur_pred) in enumerate(zip(gt, pred)):
    #     print(f"idx: {idx:.4f}, cur_gt: {cur_gt:.4f}, cur_pred: {cur_pred:.4f}, diff: {np.abs(cur_gt - cur_pred):.4f}")
    #     # print("=" * 10)
    # exit()
    