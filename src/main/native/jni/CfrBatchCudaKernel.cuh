/*
 * CfrBatchCudaKernel.cuh -- CUDA kernel for batched CFR (Counterfactual Regret
 * Minimization) game-tree solving on the GPU.
 *
 * This header implements a batched CFR solver where multiple game trees with
 * shared topology but different terminal utilities and chance weights are solved
 * in parallel — one CUDA thread per tree. Each thread runs the full iterative
 * CFR algorithm (regret matching, tree traversal, regret/strategy accumulation)
 * independently using an iterative DFS with an explicit stack (no recursion on GPU).
 *
 * Design decisions:
 *   - One thread per tree (not one thread per node) because the game trees are
 *     small (HoldemDecisionGame trees have max depth 4) but the batch size can
 *     be large (hundreds of different card configurations).
 *   - Iterative DFS with explicit stack frames replaces the recursive approach
 *     used in CfrNativeSolverCore.hpp, since CUDA does not support deep recursion.
 *   - Float32 arithmetic (not double) for GPU throughput — Maxwell sm_50 has
 *     1/32 double throughput vs float.
 *   - __ldg() intrinsic used for read-only topology data to leverage the texture
 *     cache / read-only data cache on Maxwell+.
 *   - 64 threads per block (not 128/256) because each thread uses significant
 *     register space for the DFS stack frames and strategy/action_values arrays.
 *
 * Node type encoding:
 *   0 = terminal (leaf), 1 = chance, 2 = player 0, 3 = player 1.
 *
 * Memory layout for per-tree arrays:
 *   all_terminal_utilities[tree_id * node_count + node_id]
 *   all_chance_weights[tree_id * edge_count + edge_idx]
 *   all_cumulative_regret[tree_id * strategy_size + infoset_offset + action_idx]
 *
 * The host-side launch_batch_solve() function handles all cudaMalloc/cudaMemcpy
 * and provides a clean C++ interface for the JNI binding.
 */

#pragma once

#include <cstdint>
#include <cstdio>

namespace cfrbatch {

/* Maximum number of actions per infoset that can be stored inline in stack frames.
 * Trees with more actions per decision point would require heap allocation (not
 * supported on GPU), so this must be >= max actions in any HoldemDecisionGame tree. */
constexpr int kMaxInlineActions = 8;

/* ---- device helpers ---- */

/*
 * Regret matching in float32 on the GPU. Converts cumulative regret into a
 * probability distribution: positive regrets are normalized to probabilities,
 * zero/negative regrets get uniform probability. Uses branchless arithmetic
 * to avoid warp divergence.
 */
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

/*
 * Computes the strategy averaging weight for a given CFR iteration.
 * Returns 0 during the delay period (no averaging), 1 for uniform averaging,
 * or (iteration - delay) for linear averaging. Branchless implementation.
 */
__device__ inline int averaging_weight(
    int iteration, int averaging_delay, bool linear_averaging)
{
  return (iteration > averaging_delay) * (linear_averaging ? (iteration - averaging_delay) : 1);
}

/* ---- batch kernel ----
 *
 * Main GPU kernel: one thread per game tree in the batch. All trees share the
 * same topology (node_types, node_starts, node_counts, edge_child_ids, etc.)
 * but each has its own terminal utilities, chance weights, cumulative regret,
 * and cumulative strategy arrays. Each thread independently runs 'iterations'
 * passes of the CFR algorithm using iterative DFS, then extracts the average
 * strategy from the cumulative strategy sums.
 *
 * Parameters — Shared topology (constant across the batch, accessed via __ldg):
 *   node_types           — per-node type: 0=terminal, 1=chance, 2=player0, 3=player1
 *   node_starts          — per-node index into edge_child_ids where children begin
 *   node_counts          — per-node number of children (actions)
 *   node_infosets        — per-node infoset ID (only meaningful for player nodes)
 *   edge_child_ids       — flat array of child node IDs for all edges
 *   infoset_action_counts — per-infoset number of actions
 *   infoset_offsets      — per-infoset offset into the flat strategy arrays
 *
 * Parameters — Per-tree data (indexed by tree_id * dimension + element):
 *   all_terminal_utilities — terminal payoffs for player 0 (player 1 = negated)
 *   all_chance_weights     — edge weights for chance nodes
 *
 * Parameters — Working memory (pre-zeroed by host):
 *   all_cumulative_regret   — running sum of counterfactual regrets per action
 *   all_cumulative_strategy — running sum of reach-weighted strategies per action
 *
 * Parameters — Output:
 *   all_out_strategies — final average strategy per infoset per tree
 *   all_out_ev         — root expected value per tree (from the last iteration)
 */
__global__ void cfr_batch_kernel(
    const int* __restrict__ node_types,
    const int* __restrict__ node_starts,
    const int* __restrict__ node_counts,
    const int* __restrict__ node_infosets,
    const int* __restrict__ edge_child_ids,
    const int* __restrict__ infoset_action_counts,
    const int* __restrict__ infoset_offsets,
    const float* __restrict__ all_terminal_utilities,
    const float* __restrict__ all_chance_weights,
    int root_node_id, int node_count, int edge_count,
    int strategy_size, int iterations, int averaging_delay,
    bool cfr_plus, bool linear_averaging,
    float* all_cumulative_regret,
    float* all_cumulative_strategy,
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

  /*
   * Iterative DFS stack. HoldemDecisionGame trees have max depth 4, so
   * kMaxDepth = 5 provides headroom. Each Frame holds the DFS state for one
   * node being explored:
   *   node_id      — which node this frame represents
   *   reach_p0/p1  — factored reach probabilities for player 0 and player 1
   *                   (used to weight regret and strategy contributions)
   *   action_idx   — index of the next child to push (0..ncount-1); when
   *                   action_idx == ncount, all children are done and we pop
   *   node_value   — running weighted sum of child values (EV at this node)
   *   strategy[]   — current regret-matched strategy for this infoset
   *   action_values[] — child EV per action, filled as children return
   */
  constexpr int kMaxDepth = 5;
  struct Frame {
    int node_id;
    int ntype;    /* cached node type — avoids re-reading from global memory */
    int nstart;   /* cached edge start index */
    int ncount;   /* cached child count */
    float reach_p0;
    float reach_p1;
    int action_idx;
    float node_value;
    float strategy[kMaxInlineActions];
    float action_values[kMaxInlineActions];
  };
  Frame stack[kMaxDepth];
  int depth = 0;

  // CFR iteration loop
  for (int iter = 1; iter <= iterations; ++iter) {
    const int avg_weight = averaging_weight(iter, averaging_delay, linear_averaging);

    // Push root — cache node metadata to avoid redundant __ldg in the DFS loop.
    depth = 0;
    stack[0].node_id = root_node_id;
    stack[0].ntype = __ldg(&node_types[root_node_id]);
    stack[0].nstart = __ldg(&node_starts[root_node_id]);
    stack[0].ncount = __ldg(&node_counts[root_node_id]);
    stack[0].reach_p0 = 1.0f;
    stack[0].reach_p1 = 1.0f;
    stack[0].action_idx = 0;
    stack[0].node_value = 0.0f;

    // Pre-compute strategy for root if player node
    if (stack[0].ntype == 2 || stack[0].ntype == 3) {
      int infoset = __ldg(&node_infosets[root_node_id]);
      int is_start = __ldg(&infoset_offsets[infoset]);
      regret_matching_f(cum_regret, is_start, static_cast<uint8_t>(stack[0].ncount), stack[0].strategy);
    }

    while (depth >= 0) {
      Frame& f = stack[depth];
      const int ntype = f.ntype;
      const int nstart = f.nstart;
      const int ncount = f.ncount;

      if (ntype == 0) {
        /* Terminal node: read the payoff (from player 0's perspective) and
         * propagate it up to the parent frame's action_values. For chance
         * parents, weight by the chance edge probability; for player parents,
         * weight by the strategy probability for that action. */
        float value = __ldg(&terminal_utilities[f.node_id]);
        depth--;
        if (depth >= 0) {
          Frame& parent = stack[depth];
          int pidx = parent.action_idx - 1;
          parent.action_values[pidx] = value;
          if (parent.ntype == 1) { // chance
            parent.node_value += chance_weights[parent.nstart + pidx] * value;
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
        /* Push the next unvisited child onto the DFS stack. Update the
         * child's reach probabilities: chance nodes multiply both players'
         * reach by the edge weight, player nodes multiply only the acting
         * player's reach by the strategy probability for the chosen action. */
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
        child.ntype = __ldg(&node_types[child_id]);
        child.nstart = __ldg(&node_starts[child_id]);
        child.ncount = __ldg(&node_counts[child_id]);
        child.reach_p0 = child_reach_p0;
        child.reach_p1 = child_reach_p1;
        child.action_idx = 0;
        child.node_value = 0.0f;

        if (child.ntype == 2 || child.ntype == 3) {
          int cinfoset = __ldg(&node_infosets[child_id]);
          int cis_start = __ldg(&infoset_offsets[cinfoset]);
          regret_matching_f(cum_regret, cis_start, static_cast<uint8_t>(child.ncount), child.strategy);
        }
        continue;
      }

      /* All children have been processed for this node. For player nodes,
       * update cumulative regrets and cumulative strategy:
       *   - Regret for action a = opponent_reach * (action_value[a] - node_value).
       *     For player 0, opponent_reach = reach_p1. For player 1, the sign is
       *     flipped because utilities are stored from player 0's perspective.
       *   - CFR+ variant: clamp regrets to non-negative after each update.
       *   - Cumulative strategy += avg_weight * acting_player_reach * strategy[a].
       * Then pop this frame and propagate node_value to the parent. */
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
        if (parent.ntype == 1) {
          parent.node_value += chance_weights[parent.nstart + pidx] * node_value;
        } else {
          parent.node_value += parent.strategy[pidx] * node_value;
        }
      }
    }
    // End of one CFR iteration
  }

  /* After all CFR iterations, extract the average strategy from the cumulative
   * strategy sums. For each infoset, normalize cum_strategy to sum to 1.
   * If the total is zero (infoset was never reached), fall back to regret
   * matching on the final cumulative regrets as a best-effort approximation. */
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

/* ---- host launcher ----
 *
 * BatchSpec bundles all scalar parameters needed to configure the kernel launch:
 * tree topology dimensions, iteration control, and batch size.
 */

struct BatchSpec {
  int root_node_id;      /* root of the shared game tree */
  int node_count;        /* total nodes in the tree */
  int edge_count;        /* total edges (sum of all node action counts) */
  int infoset_count;     /* number of distinct information sets */
  int strategy_size;     /* total strategy entries = sum of actions across infosets */
  int iterations;        /* CFR iterations to run per tree */
  int averaging_delay;   /* skip strategy accumulation for this many early iterations */
  bool cfr_plus;         /* if true, clamp regrets to non-negative (CFR+) */
  bool linear_averaging; /* if true, weight later iterations more heavily */
  int batch_size;        /* number of trees in this batch */
};

/*
 * Host-side launcher for the batched CFR kernel. Allocates device memory,
 * copies topology and per-tree data to the GPU, launches cfr_batch_kernel,
 * synchronizes, copies results back, and frees all device memory.
 *
 * Returns 0 on success, -1 on any CUDA error. All cudaMalloc/cudaMemcpy
 * failures are caught by the CHECK_CUDA macro and trigger early return.
 *
 * Memory allocation strategy:
 *   - Topology arrays: allocated once at [node_count] or [edge_count] size
 *   - Per-tree arrays: allocated at [batch_size * dim] size
 *   - Working arrays (cum_regret, cum_strategy): zeroed via cudaMemset
 */
inline int launch_batch_solve(
    const BatchSpec& spec,
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
