// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

struct DecodePlan {
  int G;
  int npacks;
  int np4;  // 1 -> pack of up to 4 heads, 2 -> up to 8
  int splits;
  int pps;     // logical 32-token chunks per split
  int nwarps;  // warps per CTA (1..4)
  int one_wave;
};

inline DecodePlan make_decode_plan(int B, int HQ, int HKV, int NP,
                                   int page_size, int sm_count) {
  DecodePlan pl;
  pl.G = HQ / HKV;
  pl.npacks = (pl.G + 7) / 8;
  const int pmax = pl.G < 8 ? pl.G : 8;
  pl.np4 = (pmax + 3) / 4;
  const int base = B * HKV * pl.npacks;
  const int nchunks = (NP * page_size + 31) / 32;
  // Aim just under four resident CTAs per SM so the final wave remains
  // populated without forcing an extra split solely for a small tail.
  // 2026-09-30 local: resident-CTA target is tunable for A/B
  // (TQ_SIMT_CTAS_PER_SM, default 4 = upstream). On sm_120 the SIMT kernel
  // fits only 2 CTAs/SM; brasrox measured 2 as up to 1.29x at 30k context.
  static const int ctas_per_sm = [] {
    const char* e = std::getenv("TQ_SIMT_CTAS_PER_SM");
    const int v = e ? std::atoi(e) : 4;
    return v >= 1 && v <= 8 ? v : 4;
  }();
  const int target = ctas_per_sm * sm_count - sm_count / 8;
  int desired = (target + base - 1) / base;
  if (desired < 1) desired = 1;
  int splits = desired;
  if (splits > nchunks) splits = nchunks;
  if (splits > 256) splits = 256;
  // Use at least two 32-token warps per CTA when the allocated cache permits
  // it; the policy is invariant to the physical page template.
  const int two_chunk_cap = nchunks / 2 > 0 ? nchunks / 2 : 1;
  if (splits > two_chunk_cap) splits = two_chunk_cap;
  const int one_wave = (sm_count + base - 1) / base;
  const int four_chunk_cap = nchunks / 4 > 0 ? nchunks / 4 : 1;
  const int warp_full_cap =
      one_wave > four_chunk_cap ? one_wave : four_chunk_cap;
  if (splits > warp_full_cap) splits = warp_full_cap;
  pl.pps = (nchunks + splits - 1) / splits;
  pl.splits = (nchunks + pl.pps - 1) / pl.pps;
  pl.nwarps = pl.pps < 4 ? pl.pps : 4;
  pl.one_wave = one_wave;
  return pl;
}

void turboquant_decode_launch(const void* q, const void* kv_cache,
                              const void* page_table, const void* seq_lens,
                              const void* rotation, const void* centroids,
                              void* out, float* workspace, int B, int HQ,
                              int HKV, int page_table_stride, int page_size,
                              int head_dim, const DecodePlan& pl,
                              cudaStream_t stream);
#pragma once
// SPDX-License-Identifier: Apache-2.0
#pragma once
