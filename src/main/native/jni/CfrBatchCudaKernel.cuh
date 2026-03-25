#pragma once

#include <cstdint>
#include <cstdio>

namespace cfrbatch {

constexpr int kMaxInlineActions = 8;

// ---- device helpers ----

__device__ __forceinline__ void regret_matching_f(
    const float* cumulative_regret,
    int start, uint8_t count,
    float* out_strategy)
{
  float positive_sum = 0.0f;
  for (uint8_t i = 0; i < count; ++i) {
    float pos = fmaxf(cumulative_regret[start + i], 0.0f);
    out_strategy[i] = pos;
    positive_sum += pos;
  }
  float has_pos = (positive_sum > 0.0f) ? 1.0f : 0.0f;
  float inv = has_pos / (positive_sum + (1.0f - has_pos));
  float uniform = (1.0f - has_pos) / static_cast<float>(count);
  for (uint8_t i = 0; i < count; ++i) {
    out_strategy[i] = out_strategy[i] * inv + uniform;
  }
}

__device__ inline int averaging_weight(
    int iteration, int averaging_delay, bool linear_averaging)
{
  return (iteration > averaging_delay) * (linear_averaging ? (iteration - averaging_delay) : 1);
}

// ---- batch kernel ----

// One thread per tree. Trees share topology, differ in terminal utilities and chance weights.
__global__ void cfr_batch_kernel(
    // Shared topology (read-only, same for all trees)
    const int* __restrict__ node_types,
    const int* __restrict__ node_starts,
    const int* __restrict__ node_counts,
    const int* __restrict__ node_infosets,
    const int* __restrict__ edge_child_ids,
    const int* __restrict__ infoset_action_counts,
    const int* __restrict__ infoset_offsets,
    // Per-tree varying data (read-only, indexed by [tree_id * size + element])
    const float* __restrict__ all_terminal_utilities,
    const float* __restrict__ all_chance_weights,
    // Dimensions
    int root_node_id, int node_count, int edge_count,
    int strategy_size, int iterations, int averaging_delay,
    bool cfr_plus, bool linear_averaging,
    // Per-tree working memory (pre-zeroed, indexed by [tree_id * strategy_size + offset])
    float* all_cumulative_regret,
    float* all_cumulative_strategy,
    // Output (indexed by [tree_id * strategy_size + offset] and [tree_id])
    float* all_out_strategies,
    float* all_out_ev,
    int infoset_count,
    int batch_size)
{
  const int tree_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (tree_id >= batch_size) return;

  // Per-tree pointers
  const float* terminal_utilities = all_terminal_utilities + tree_id * node_count;
  const float* chance_weights = all_chance_weights + tree_id * edge_count;
  float* cum_regret = all_cumulative_regret + tree_id * strategy_size;
  float* cum_strategy = all_cumulative_strategy + tree_id * strategy_size;

  // Iterative DFS stack (max depth 4 for HoldemDecisionGame trees, +1 headroom)
  constexpr int kMaxDepth = 5;
  struct Frame {
    int node_id;
    float reach_p0;
    float reach_p1;
    int action_idx;  // next child to process
    float node_value; // accumulated so far
    float strategy[kMaxInlineActions];
    float action_values[kMaxInlineActions];
  };
  Frame stack[kMaxDepth];
  int depth = 0;

  // CFR iteration loop
  for (int iter = 1; iter <= iterations; ++iter) {
    const int avg_weight = averaging_weight(iter, averaging_delay, linear_averaging);

    // Push root
    depth = 0;
    stack[0].node_id = root_node_id;
    stack[0].reach_p0 = 1.0f;
    stack[0].reach_p1 = 1.0f;
    stack[0].action_idx = 0;
    stack[0].node_value = 0.0f;

    int node_type = __ldg(&node_types[root_node_id]);
    int count = __ldg(&node_counts[root_node_id]);

    // Pre-compute strategy for root if player node
    if (node_type == 2 || node_type == 3) {
      int infoset = __ldg(&node_infosets[root_node_id]);
      int is_start = __ldg(&infoset_offsets[infoset]);
      regret_matching_f(cum_regret, is_start, static_cast<uint8_t>(count), stack[0].strategy);
    }

    while (depth >= 0) {
      Frame& f = stack[depth];
      int ntype = __ldg(&node_types[f.node_id]);
      int nstart = __ldg(&node_starts[f.node_id]);
      int ncount = __ldg(&node_counts[f.node_id]);

      if (ntype == 0) {
        // Terminal node
        float value = __ldg(&terminal_utilities[f.node_id]);
        depth--;
        if (depth >= 0) {
          Frame& parent = stack[depth];
          int pidx = parent.action_idx - 1;
          parent.action_values[pidx] = value;
          int ptype = __ldg(&node_types[parent.node_id]);
          if (ptype == 1) { // chance
            parent.node_value += chance_weights[__ldg(&node_starts[parent.node_id]) + pidx] * value;
          } else {
            parent.node_value += parent.strategy[pidx] * value;
          }
        } else {
          // Root was terminal (shouldn't happen, but handle)
          all_out_ev[tree_id] = value;
        }
        continue;
      }

      if (f.action_idx < ncount) {
        // Push next child
        int edge = nstart + f.action_idx;
        int child_id = __ldg(&edge_child_ids[edge]);
        f.action_idx++;

        float child_reach_p0 = f.reach_p0;
        float child_reach_p1 = f.reach_p1;
        if (ntype == 1) { // chance
          float w = __ldg(&chance_weights[edge]);
          child_reach_p0 *= w;
          child_reach_p1 *= w;
        } else if (ntype == 2) { // player 0
          child_reach_p0 *= f.strategy[f.action_idx - 1];
        } else { // player 1
          child_reach_p1 *= f.strategy[f.action_idx - 1];
        }

        depth++;
        Frame& child = stack[depth];
        child.node_id = child_id;
        child.reach_p0 = child_reach_p0;
        child.reach_p1 = child_reach_p1;
        child.action_idx = 0;
        child.node_value = 0.0f;

        int ctype = __ldg(&node_types[child_id]);
        if (ctype == 2 || ctype == 3) {
          int cinfoset = __ldg(&node_infosets[child_id]);
          int cis_start = __ldg(&infoset_offsets[cinfoset]);
          int ccount = __ldg(&node_counts[child_id]);
          regret_matching_f(cum_regret, cis_start, static_cast<uint8_t>(ccount), child.strategy);
        }
        continue;
      }

      // All children processed. Update regrets/strategy, then pop.
      float node_value = f.node_value;

      if (ntype == 2 || ntype == 3) {
        int infoset = __ldg(&node_infosets[f.node_id]);
        int is_start = __ldg(&infoset_offsets[infoset]);

        if (ntype == 2) { // player 0
          float strategy_scale = static_cast<float>(avg_weight) * f.reach_p0;
          for (uint8_t i = 0; i < static_cast<uint8_t>(ncount); ++i) {
            float regret_delta = f.reach_p1 * (f.action_values[i] - node_value);
            float updated = cum_regret[is_start + i] + regret_delta;
            cum_regret[is_start + i] = cfr_plus ? fmaxf(updated, 0.0f) : updated;
            cum_strategy[is_start + i] += strategy_scale * f.strategy[i];
          }
        } else { // player 1
          float strategy_scale = static_cast<float>(avg_weight) * f.reach_p1;
          for (uint8_t i = 0; i < static_cast<uint8_t>(ncount); ++i) {
            float regret_delta = f.reach_p0 * (node_value - f.action_values[i]);
            float updated = cum_regret[is_start + i] + regret_delta;
            cum_regret[is_start + i] = cfr_plus ? fmaxf(updated, 0.0f) : updated;
            cum_strategy[is_start + i] += strategy_scale * f.strategy[i];
          }
        }
      }

      depth--;
      if (depth >= 0) {
        Frame& parent = stack[depth];
        int pidx = parent.action_idx - 1;
        parent.action_values[pidx] = node_value;
        int ptype = __ldg(&node_types[parent.node_id]);
        if (ptype == 1) {
          parent.node_value += chance_weights[__ldg(&node_starts[parent.node_id]) + pidx] * node_value;
        } else {
          parent.node_value += parent.strategy[pidx] * node_value;
        }
      }
    }
    // End of one CFR iteration
  }

  // Extract average strategies from cumulative_strategy
  float* out_strat = all_out_strategies + tree_id * strategy_size;

  for (int is = 0; is < infoset_count; ++is) {
    int is_start = __ldg(&infoset_offsets[is]);
    int is_count = __ldg(&infoset_action_counts[is]);
    float sum = 0.0f;
    for (uint8_t i = 0; i < static_cast<uint8_t>(is_count); ++i) sum += cum_strategy[is_start + i];
    if (sum > 0.0f) {
      float inv = 1.0f / sum;
      for (uint8_t i = 0; i < static_cast<uint8_t>(is_count); ++i)
        out_strat[is_start + i] = cum_strategy[is_start + i] * inv;
    } else {
      regret_matching_f(cum_regret, is_start, static_cast<uint8_t>(is_count), out_strat + is_start);
    }
  }
}

// ---- host launcher ----

struct BatchSpec {
  int root_node_id;
  int node_count;
  int edge_count;
  int infoset_count;
  int strategy_size;
  int iterations;
  int averaging_delay;
  bool cfr_plus;
  bool linear_averaging;
  int batch_size;
};

inline int launch_batch_solve(
    const BatchSpec& spec,
    // Host arrays: shared topology
    const int* h_node_types,
    const int* h_node_starts,
    const int* h_node_counts,
    const int* h_node_infosets,
    const int* h_edge_child_ids,
    const int* h_infoset_action_counts,
    const int* h_infoset_offsets,
    // Host arrays: per-tree data [batch_size * dim]
    const float* h_terminal_utilities,
    const float* h_chance_weights,
    // Host output
    float* h_out_strategies,
    float* h_out_ev)
{
  const int B = spec.batch_size;
  const int N = spec.node_count;
  const int E = spec.edge_count;
  const int S = spec.strategy_size;
  const int I = spec.infoset_count;

  // Helper macro for CUDA error checking
  #define CHECK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) return -1; \
  } while(0)

  // Device topology arrays
  int *d_node_types = nullptr, *d_node_starts = nullptr;
  int *d_node_counts = nullptr, *d_node_infosets = nullptr;
  int *d_edge_child_ids = nullptr, *d_infoset_action_counts = nullptr;
  int *d_infoset_offsets = nullptr;
  CHECK_CUDA(cudaMalloc(&d_node_types, N * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_node_starts, N * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_node_counts, N * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_node_infosets, N * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_edge_child_ids, E * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_infoset_action_counts, I * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_infoset_offsets, (I + 1) * sizeof(int)));

  CHECK_CUDA(cudaMemcpy(d_node_types, h_node_types, N * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_node_starts, h_node_starts, N * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_node_counts, h_node_counts, N * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_node_infosets, h_node_infosets, N * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_edge_child_ids, h_edge_child_ids, E * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_infoset_action_counts, h_infoset_action_counts, I * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_infoset_offsets, h_infoset_offsets, (I + 1) * sizeof(int), cudaMemcpyHostToDevice));

  // Device per-tree arrays
  float *d_terminal_utilities = nullptr, *d_chance_weights = nullptr;
  float *d_cum_regret = nullptr, *d_cum_strategy = nullptr;
  float *d_out_strategies = nullptr, *d_out_ev = nullptr;
  CHECK_CUDA(cudaMalloc(&d_terminal_utilities, B * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_chance_weights, B * E * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_cum_regret, B * S * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_cum_strategy, B * S * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out_strategies, B * S * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_out_ev, B * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(d_terminal_utilities, h_terminal_utilities, B * N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_chance_weights, h_chance_weights, B * E * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(d_cum_regret, 0, B * S * sizeof(float)));
  CHECK_CUDA(cudaMemset(d_cum_strategy, 0, B * S * sizeof(float)));

  // Launch
  const int threads_per_block = 64; // tunable; low because each thread uses registers heavily
  const int blocks = (B + threads_per_block - 1) / threads_per_block;

  cfr_batch_kernel<<<blocks, threads_per_block>>>(
      d_node_types, d_node_starts, d_node_counts, d_node_infosets,
      d_edge_child_ids, d_infoset_action_counts, d_infoset_offsets,
      d_terminal_utilities, d_chance_weights,
      spec.root_node_id, N, E, S,
      spec.iterations, spec.averaging_delay,
      spec.cfr_plus, spec.linear_averaging,
      d_cum_regret, d_cum_strategy,
      d_out_strategies, d_out_ev,
      I, B);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    // Cleanup
    cudaFree(d_node_types); cudaFree(d_node_starts); cudaFree(d_node_counts);
    cudaFree(d_node_infosets); cudaFree(d_edge_child_ids);
    cudaFree(d_infoset_action_counts); cudaFree(d_infoset_offsets);
    cudaFree(d_terminal_utilities); cudaFree(d_chance_weights);
    cudaFree(d_cum_regret); cudaFree(d_cum_strategy);
    cudaFree(d_out_strategies); cudaFree(d_out_ev);
    return -1;
  }

  // Read back results
  CHECK_CUDA(cudaMemcpy(h_out_strategies, d_out_strategies, B * S * sizeof(float), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_out_ev, d_out_ev, B * sizeof(float), cudaMemcpyDeviceToHost));

  // Cleanup
  cudaFree(d_node_types); cudaFree(d_node_starts); cudaFree(d_node_counts);
  cudaFree(d_node_infosets); cudaFree(d_edge_child_ids);
  cudaFree(d_infoset_action_counts); cudaFree(d_infoset_offsets);
  cudaFree(d_terminal_utilities); cudaFree(d_chance_weights);
  cudaFree(d_cum_regret); cudaFree(d_cum_strategy);
  cudaFree(d_out_strategies); cudaFree(d_out_ev);

  #undef CHECK_CUDA
  return 0;
}

} // namespace cfrbatch
