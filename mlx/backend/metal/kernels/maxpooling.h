// Copyright © 2025 Apple Inc.

#include "mlx/backend/metal/maxpooling.h"

using mlx::core::MaxPoolingParams;

template <typename T, typename AccT = float, int N_READS = 4>
[[kernel]] void maxpooling(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant MaxPoolingParams* params [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint _lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  int lid = _lid;

  constexpr int SIMD_SIZE = 32;

  threadgroup AccT local_max[SIMD_SIZE];
  threadgroup AccT local_normalizer[SIMD_SIZE];

  AccT ld[N_READS];

  // in += gid * size_t(axis_size) + lid * N_READS;
  // if (lid * N_READS + N_READS <= axis_size) {
  //   for (int i = 0; i < N_READS; i++) {
  //     ld[i] = AccT(in[i]);
  //   }
  // } else {
  //   for (int i = 0; i < N_READS; i++) {
  //     ld[i] =
  //         ((lid * N_READS + i) < axis_size) ? AccT(in[i]) : Limits<AccT>::min;
  //   }
  // }
  // if (simd_group_id == 0) {
  //   local_max[simd_lane_id] = Limits<AccT>::min;
  //   local_normalizer[simd_lane_id] = 0;
  // }
  // threadgroup_barrier(mem_flags::mem_threadgroup);

  // // Get the max
  // AccT maxval = Limits<AccT>::finite_min;
  // for (int i = 0; i < N_READS; i++) {
  //   maxval = (maxval < ld[i]) ? ld[i] : maxval;
  // }
  // maxval = simd_max(maxval);
  // if (simd_lane_id == 0) {
  //   local_max[simd_group_id] = maxval;
  // }
  // threadgroup_barrier(mem_flags::mem_threadgroup);
  // if (simd_group_id == 0) {
  //   maxval = simd_max(local_max[simd_lane_id]);
  //   if (simd_lane_id == 0) {
  //     local_max[0] = maxval;
  //   }
  // }
  // threadgroup_barrier(mem_flags::mem_threadgroup);
  // maxval = local_max[0];

  // // Compute exp(x_i - maxval) and store the partial sums in local_normalizer
  // AccT normalizer = 0;
  // for (int i = 0; i < N_READS; i++) {
  //   normalizer += fast::exp(ld[i] - maxval);
  // }
  // normalizer = simd_sum(normalizer);
  // if (simd_lane_id == 0) {
  //   local_normalizer[simd_group_id] = normalizer;
  // }
  // threadgroup_barrier(mem_flags::mem_threadgroup);
  // if (simd_group_id == 0) {
  //   normalizer = simd_sum(local_normalizer[simd_lane_id]);
  //   if (simd_lane_id == 0) {
  //     out[gid] = isinf(maxval) ? T(maxval) : T(log(normalizer) + maxval);
  //   }
  // }
}
