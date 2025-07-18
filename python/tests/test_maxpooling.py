import mlx.core as mx
import numpy as np

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=4, suppress=True)
np.random.seed(0)

round_q = lambda len_q : (len_q * 16 + 128 - 1) // 128 * 128 // 16

BATCH_SIZE = 1
NUM_HEAD = 2
LEN_Q = 14001
LEN_Q_ROUND = round_q(LEN_Q)
LEN_K = 896
LEN_CACHE = 0
INIT_BLOCK = 1
LOCAL_BLOCK = 32
KERNEL_SIZE = 5
STRIDE = 4
PADDING = 2
BLOCK_SIZE = 64
DTYPE = np.float16

"""
template <typename T>
__global__ void maxpooling_kernel(
    const T* input,
    T* output,
    int num_heads, // 2
    int q_len, // 14001
    int q_round, // 14008
    int k_len, // 896
    int out_len, // 233
    int cache_len, // 0
    int init_blocks, // 1
    int local_blocks, // 32
    int kernel_size, // 5
    int stride, // 4
    int padding, // 1
    int block_size // 64
) {
    int bidh = blockIdx.y;
    int bidq = blockIdx.x;
    const T* in = input + bidh * q_round * k_len + bidq * k_len;
    T* out = output + bidh * q_len * out_len + bidq * out_len;
    int q_block = (bidq + cache_len) / block_size;

    for (int k = threadIdx.x; k < out_len; k += blockDim.x) { // blockDim.x = 256
        int start = k * stride - padding;
        int end = start + kernel_size;
        start = max(start, 0);
        end = min(end, k_len);
        
        T max_val;
        if (k < init_blocks) {
            max_val = TypeTraits<T>::inf();
        } else if (q_block - local_blocks < k) {
            max_val = -TypeTraits<T>::inf();
        } else {
            max_val = in[start];
            for (int i = start + 1; i < end; i++) {
                if (in[i] > max_val) {
                    max_val = in[i];
                }
            }
        }
        out[k] = max_val;
    }
}

template <typename T>
void maxpooling_func(
    cudaStream_t stream,
    const T* input, // num_heads x q_len x k_len (2, 14001, 896)
    T* output, // num_heads x q_len x out_len (2, 14001, 233)
    int num_heads, // 2
    int q_len, // 14001
    int q_round, // 14008,  q_len * 16 = 224016, (q_len + 128 - 1) / 128 = 224128, 224128 / 16 = 14008
    int k_len, // 896
    int cache_len, // 0
    int init_blocks, // 1
    int local_blocks, // 32
    int &out_len, // 233
    int kernel_size=5,
    int stride=4,
    int padding=1,
    int block_size=64
) {
    out_len = (cache_len + block_size - 1) / block_size;
    maxpooling_kernel<<<dim3(q_len, num_heads), 256, 0, stream>>>(
        input, output, num_heads, q_len, q_round, k_len, out_len, cache_len, init_blocks, local_blocks, kernel_size, stride, padding, block_size
    );
} 
"""

def naive_maxpooling_npy(score, cache_len, init_blocks, local_blocks, kernel_size, stride, padding, block_size):
    
    batch_size, num_head, q_len, k_len = score.shape
    assert batch_size == 1

    total_len = q_len + cache_len
    q_block = total_len // block_size
    out_len = (total_len + block_size - 1) // block_size
    
    max_val = np.zeros((batch_size, num_head, q_len, out_len), dtype=score.dtype)

    for i in range(out_len):
        start = i * stride - padding
        end = start + kernel_size
        start = max(start, 0)
        end = min(end, q_len)
        if (i < init_blocks):
            max_val[:, :, :, i] = np.inf
        elif (q_block - local_blocks < i):
            max_val[:, :, :, i] = -np.inf
        else:
            max_val[:, :, :, i] = score[:, :, :, start:end].max(axis=-1)
    
    return max_val


if __name__ == "__main__":
    
    score_npy = np.random.normal(0.0, 1.0, (BATCH_SIZE, NUM_HEAD, LEN_Q, LEN_K)).astype(DTYPE)
    print(score_npy.shape)

    max_val_npy = naive_maxpooling_npy(score_npy, LEN_CACHE, INIT_BLOCK, LOCAL_BLOCK, KERNEL_SIZE, STRIDE, PADDING, BLOCK_SIZE)
    # print(max_val_npy)
    # exit()

    score_mlx = mx.array(score_npy)

    output = mx.maxpooling(score_mlx, LEN_CACHE, INIT_BLOCK, LOCAL_BLOCK, KERNEL_SIZE, STRIDE, PADDING, BLOCK_SIZE)

    print(output.shape)
    
    