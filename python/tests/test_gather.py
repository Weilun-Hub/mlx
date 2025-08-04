import mlx.core as mx
import torch

K_LEN = 14001
NUM_KEY_VALUE_HEAD = 2
HEAD_DIM = 128
CU_SEQLEN = torch.tensor([0, K_LEN], dtype=torch.int32)
KERNEL_SIZE = 32
KERNEL_STRIDE = 16
DEVICE = "mps"
REQUIRE_GRAD = False
DTYPE = torch.float16

def calc_chunks_with_stride(cu_seqlen, chunk_size, kernel_stride):
    """
    Compute the chunks that require Sparse attention, with stride support.

    Args:
        cu_seqlen (torch.Tensor): Cumulative sequence lengths for each sample.
        chunk_size (int): Chunk size used for Sparse attention.
        kernel_stride (int): Stride size when sliding over the sequence.

    Returns:
        filtered_indices (torch.Tensor): Indices used to directly index into the key/value tensors.
        cu_seqlens_compressed (torch.Tensor): Cumulative sequence lengths after compression.
    """
    # 1. Compute the length of each sequence
    batch_sizes = cu_seqlen[1:] - cu_seqlen[:-1] # [14001]

    # 2. Compute the start positions of chunks for each sequence (with stride)
    max_seq_len = torch.max(batch_sizes) # 14001
    max_num_chunks_per_seq = (max_seq_len - chunk_size) // kernel_stride + 1 # (14001 - 32) // 16 + 1 = 874
    chunk_start_offsets = torch.arange(0, max_num_chunks_per_seq * kernel_stride, kernel_stride, device=cu_seqlen.device) # [0, 16, 32, ..., 13952, 13968], shape = 874
    seq_starts = cu_seqlen[:-1] # [0]
    chunk_start_in_seq = seq_starts[:, None] + chunk_start_offsets[None, :]  # [batch_size, max_num_chunks_per_seq], shape = (1, 874)

    # 3. Filter out chunks that exceed sequence length or are smaller than the full chunk size
    chunk_end_in_seq = chunk_start_in_seq + chunk_size # [0 + 32, 16 + 32, 32 + 32, ..., 13952 + 32, 13968 + 32] = [32, 48, 64, ..., 14000, 14016], shape = (1, 874)
    valid_chunk_mask = (chunk_end_in_seq <= (seq_starts[:, None] + batch_sizes[:, None])) # shape = (1, 874), [True, True, True, ..., True, True]

    # 4. Filter valid chunk start positions using the valid_chunk_mask
    valid_chunk_starts = chunk_start_in_seq[valid_chunk_mask]  # [num_valid_chunks], shape = 874, [0, 16, 32, ..., 13952, 13968]
    del chunk_start_in_seq
    # 5. Generate filtered_indices
    chunk_indices = torch.arange(
        0, chunk_size, device=cu_seqlen.device
    )[None, :]  # [1, chunk_size], [0, 1, 2, ..., 31]
    filtered_indices = valid_chunk_starts[:, None] + chunk_indices  # [num_valid_chunks, chunk_size], shape = (874, 32)
    filtered_indices = filtered_indices.view(-1)  # Flatten to 1D indices [0, 1, 2, ..., 30, 31, | 16, 17, 46, 47, | 32, 33, ..., 62, 63, ... ]

    # 6. Compute compressed cumulative sequence lengths
    num_filtered_chunks_per_batch = valid_chunk_mask.sum(dim=1)  # Number of valid chunks per batch, [874]
    cu_seqlens_compressed = torch.zeros(
        len(cu_seqlen), dtype=torch.int32, device=cu_seqlen.device
    ) # [0, 0]
    cu_seqlens_compressed[1:] = num_filtered_chunks_per_batch.cumsum(dim=0) # [0, 874]
    del num_filtered_chunks_per_batch, chunk_start_offsets, seq_starts, chunk_end_in_seq, valid_chunk_mask, chunk_indices
    return filtered_indices, cu_seqlens_compressed # (874 * 32), [0, 874]

if __name__ == "__main__":

    q = torch.randn(K_LEN, NUM_KEY_VALUE_HEAD, HEAD_DIM, device=DEVICE, dtype=DTYPE, requires_grad=REQUIRE_GRAD)
    q = mx.array(q.detach().cpu().numpy())
    print(q.shape) # (14001, 2, 128)
    filtered_indices, cu_seqlens_compressed = calc_chunks_with_stride(CU_SEQLEN, KERNEL_SIZE, KERNEL_STRIDE)
    filtered_indices = mx.array(filtered_indices.detach().cpu().numpy())
    cu_seqlens_compressed = mx.array(cu_seqlens_compressed.detach().cpu().numpy())
    print(filtered_indices.shape) # (874 * 32)
    # print(filtered_indices[0:32])
    # print(filtered_indices[32:64])
    # print(filtered_indices[-32:])
    print(cu_seqlens_compressed) # [0, 874]

    gathered_q = mx.gather(q, filtered_indices, axis=0, slice_sizes=(1, 2, 128))
    gathered_q = mx.squeeze(gathered_q, axis=1)  # Remove the dimension of size 1
    print(gathered_q.shape) # (874 * 32, 2, 128)
    print(gathered_q[-32, 0, :])
    print(q[13968, 0, :])
    
