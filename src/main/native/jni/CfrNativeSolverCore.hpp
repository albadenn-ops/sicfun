/*
 * CfrNativeSolverCore.hpp -- Pure C++ CFR (Counterfactual Regret Minimization)
 * solver engine, shared by both CPU and GPU JNI bindings.
 *
 * Provides three solver variants:
 *   1. solve()       — Full CFR with double-precision arithmetic. Returns average
 *                      strategies for all infosets and the root expected value.
 *   2. solve_root()  — Same as solve() but only returns the average strategy for
 *                      a single specified root infoset (saves copying all strategies
 *                      back through JNI when only the root decision matters).
 *   3. solve_fixed() — Fixed-point CFR using Q30 probabilities and Q13 values.
 *                      Avoids floating-point entirely for deterministic, bit-exact
 *                      results across platforms. Includes overflow protection via
 *                      emergency rescaling (halving) of regret/strategy accumulators.
 *
 * The tree is represented as flat arrays (node_types, node_starts, node_counts,
 * edge_child_ids, etc.) matching the layout serialized by the Scala JNI caller.
 * All three variants use recursive DFS (via C++ generic lambdas with self-call)
 * — unlike the GPU kernel in CfrBatchCudaKernel.cuh which uses iterative DFS.
 *
 * Node types: 0=terminal, 1=chance, 2=player0, 3=player1.
 *
 * Key design decisions:
 *   - Template on float type F allows sharing train_impl between double and
 *     potential float instantiations without code duplication.
 *   - Inline action buffers (kInlineActionBufferSize=8) avoid heap allocation
 *     for typical poker game trees. Heap fallback for larger trees.
 *   - Branchless regret matching avoids branch misprediction on the
 *     positive-sum vs uniform fallback path.
 *   - Fixed-point variant uses round-to-nearest (not truncation) for Q30 shifts
 *     to match the Scala reference implementation exactly.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

/* CFR_FORCE_INLINE: cross-platform forced inlining for hot-path functions.
 * MSVC uses __forceinline; GCC/Clang use __attribute__((always_inline)).
 * Also handles nvcc compilation (__CUDACC__) which uses MSVC-style on Windows. */
#if defined(_MSC_VER) || defined(__CUDACC__)
#define CFR_FORCE_INLINE __forceinline
#else
#define CFR_FORCE_INLINE inline __attribute__((always_inline))
#endif

namespace cfrnative {

/* Status codes shared with JNI callers. 0 = success; 100+ = errors.
 * These match the constants in CfrSolver.scala for cross-layer consistency. */
constexpr int kStatusOk = 0;
constexpr int kStatusNullArray = 100;             /* JNI array argument was null */
constexpr int kStatusLengthMismatch = 101;        /* array sizes don't match expected dimensions */
constexpr int kStatusReadFailure = 102;           /* JNI array read failed */
constexpr int kStatusInvalidMode = 111;           /* unrecognized solver mode */
constexpr int kStatusWriteFailure = 124;          /* JNI array write failed */
constexpr int kStatusInvalidRootNode = 146;       /* root_node_id out of range */
constexpr int kStatusInvalidIterations = 147;     /* iterations <= 0 or averaging_delay < 0 */
constexpr int kStatusInvalidNodeType = 148;       /* node type not in {0,1,2,3} */
constexpr int kStatusInvalidNodeLayout = 149;     /* node_starts/node_counts out of range */
constexpr int kStatusInvalidChildIndex = 150;     /* edge_child_ids references invalid node */
constexpr int kStatusInvalidInfoSetIndex = 151;   /* infoset index out of range or wrong player */
constexpr int kStatusInfoSetActionMismatch = 152; /* node action count != infoset action count */
constexpr int kStatusInvalidChanceProbabilities = 153; /* negative or zero-sum chance probs */

/* Node type constants matching the Scala GameTree encoding. */
constexpr int kNodeTerminal = 0;  /* leaf node — has a utility value, no children */
constexpr int kNodeChance = 1;    /* nature/chance node — children weighted by edge probs */
constexpr int kNodePlayer0 = 2;   /* player 0 decision node */
constexpr int kNodePlayer1 = 3;   /* player 1 decision node */

/* Stack-allocated action buffer size. 8 covers all HoldemDecisionGame trees
 * (max ~6 actions: fold/check/call/bet sizes). Falls back to heap if exceeded. */
constexpr int kInlineActionBufferSize = 8;

/* Fixed-point arithmetic scales for solve_fixed():
 *   Q30 for probabilities: 1.0 = 2^30 = 1,073,741,824
 *   Q13 for values/utilities: 1.0 = 2^13 = 8,192
 * The different scales balance precision (probabilities need ~9 decimal digits)
 * against range (values can exceed 1.0 by large factors in pot-sized bets). */
constexpr int kProbScaleBits = 30;
constexpr int kProbScale = 1 << kProbScaleBits;     /* 2^30 ≈ 1.07 billion */
constexpr int kFixedValScaleBits = 13;
constexpr int kFixedValScale = 1 << kFixedValScaleBits; /* 2^13 = 8192 */

/* Rescale thresholds: when any accumulator exceeds 1/4 of the integer range,
 * halve the entire infoset's accumulators to prevent overflow on the next update.
 * This matches the Scala CfrSolver's emergency rescaling logic. */
constexpr int kRegretRescaleThreshold = INT32_MAX / 4;
constexpr int64_t kStrategyRescaleThreshold = INT64_MAX / 4;

/*
 * TreeSpec: game tree specification for double-precision CFR.
 * All arrays use flat indexing. Nodes are numbered 0..N-1. Edges (parent->child
 * links) are numbered 0..E-1. For each node, node_starts[i] is the first edge
 * index and node_counts[i] is how many children it has. terminal_utilities are
 * from player 0's perspective (player 1 = negated internally by CFR).
 */
struct TreeSpec {
  int iterations = 0;
  int averaging_delay = 0;
  bool cfr_plus = true;
  bool linear_averaging = true;

  int root_node_id = 0;
  std::vector<int> node_types;
  std::vector<int> node_starts;
  std::vector<int> node_counts;
  std::vector<int> node_infosets;
  std::vector<int> edge_child_ids;
  std::vector<double> edge_probabilities;
  std::vector<double> terminal_utilities;
  std::vector<int> infoset_players;
  std::vector<int> infoset_action_counts;
};

/* SolveOutput: result from solve() — average strategies for all infosets
 * (flat array indexed by infoset_offset + action_idx) and the game value. */
struct SolveOutput {
  std::vector<double> average_strategies;
  double expected_value_player0 = 0.0;
};

/* RootSolveOutput: result from solve_root() — only the root infoset's strategy. */
struct RootSolveOutput {
  std::vector<double> root_strategy;
};

/*
 * TreeSpecFixed: game tree specification for fixed-point CFR (solve_fixed).
 * Same topology as TreeSpec, but edge_probabilities_raw and terminal_utilities_raw
 * are Q30 and Q13 fixed-point integers respectively. This avoids all floating-point
 * arithmetic for deterministic, bit-exact results matching the Scala implementation.
 */
struct TreeSpecFixed {
  int iterations = 0;
  int averaging_delay = 0;
  bool cfr_plus = true;
  bool linear_averaging = true;

  int root_node_id = 0;
  std::vector<int> node_types;
  std::vector<int> node_starts;
  std::vector<int> node_counts;
  std::vector<int> node_infosets;
  std::vector<int> edge_child_ids;
  std::vector<int> edge_probabilities_raw;
  std::vector<int> terminal_utilities_raw;
  std::vector<int> infoset_players;
  std::vector<int> infoset_action_counts;
};

/* SolveOutputFixed: result from solve_fixed() — Q30 average strategies and Q13 EV. */
struct SolveOutputFixed {
  std::vector<int> average_strategies_raw;  /* Q30 probabilities per action */
  int expected_value_player0_raw = 0;       /* Q13 expected value for player 0 */
};

/*
 * Validates a TreeSpec for structural correctness before solving. Checks:
 *   - Positive iterations and non-negative averaging delay
 *   - Consistent array sizes (node arrays all same length, etc.)
 *   - Valid root node ID
 *   - Terminal nodes have 0 children; non-terminal nodes have > 0 children
 *   - Chance node edge probabilities are finite, non-negative, and sum > 0
 *   - Chance nodes have infoset == -1 (no information set)
 *   - Player nodes reference valid infosets with matching player and action count
 *   - All edge child IDs reference valid node indices
 * Returns kStatusOk or a specific error code.
 */
inline int validate_tree(const TreeSpec& spec) {
  if (spec.iterations <= 0 || spec.averaging_delay < 0) {
    return kStatusInvalidIterations;
  }

  const int node_count = static_cast<int>(spec.node_types.size());
  if (node_count <= 0) {
    return kStatusInvalidNodeLayout;
  }
  if (static_cast<int>(spec.node_starts.size()) != node_count ||
      static_cast<int>(spec.node_counts.size()) != node_count ||
      static_cast<int>(spec.node_infosets.size()) != node_count ||
      static_cast<int>(spec.terminal_utilities.size()) != node_count) {
    return kStatusLengthMismatch;
  }
  if (spec.root_node_id < 0 || spec.root_node_id >= node_count) {
    return kStatusInvalidRootNode;
  }

  const int edge_count = static_cast<int>(spec.edge_child_ids.size());
  if (static_cast<int>(spec.edge_probabilities.size()) != edge_count) {
    return kStatusLengthMismatch;
  }

  const int infoset_count = static_cast<int>(spec.infoset_players.size());
  if (infoset_count <= 0 || static_cast<int>(spec.infoset_action_counts.size()) != infoset_count) {
    return kStatusLengthMismatch;
  }

  for (int infoset = 0; infoset < infoset_count; ++infoset) {
    if ((spec.infoset_players[infoset] != 0 && spec.infoset_players[infoset] != 1) ||
        spec.infoset_action_counts[infoset] <= 0) {
      return kStatusInvalidInfoSetIndex;
    }
  }

  for (int node = 0; node < node_count; ++node) {
    const int node_type = spec.node_types[node];
    if (node_type < kNodeTerminal || node_type > kNodePlayer1) {
      return kStatusInvalidNodeType;
    }

    const int start = spec.node_starts[node];
    const int count = spec.node_counts[node];
    if (start < 0 || count < 0 || start > edge_count || (start + count) > edge_count) {
      return kStatusInvalidNodeLayout;
    }

    if (node_type == kNodeTerminal) {
      if (count != 0) {
        return kStatusInvalidNodeLayout;
      }
      continue;
    }

    if (count <= 0) {
      return kStatusInvalidNodeLayout;
    }

    if (node_type == kNodeChance) {
      double prob_sum = 0.0;
      for (int edge = start; edge < start + count; ++edge) {
        const int child = spec.edge_child_ids[edge];
        if (child < 0 || child >= node_count) {
          return kStatusInvalidChildIndex;
        }
        const double probability = spec.edge_probabilities[edge];
        if (!std::isfinite(probability) || probability < 0.0) {
          return kStatusInvalidChanceProbabilities;
        }
        prob_sum += probability;
      }
      if (!(prob_sum > 0.0)) {
        return kStatusInvalidChanceProbabilities;
      }
      if (spec.node_infosets[node] != -1) {
        return kStatusInvalidInfoSetIndex;
      }
      continue;
    }

    const int infoset_index = spec.node_infosets[node];
    if (infoset_index < 0 || infoset_index >= infoset_count) {
      return kStatusInvalidInfoSetIndex;
    }
    const int expected_player = (node_type == kNodePlayer0) ? 0 : 1;
    if (spec.infoset_players[infoset_index] != expected_player) {
      return kStatusInvalidInfoSetIndex;
    }
    if (spec.infoset_action_counts[infoset_index] != count) {
      return kStatusInfoSetActionMismatch;
    }
    for (int edge = start; edge < start + count; ++edge) {
      const int child = spec.edge_child_ids[edge];
      if (child < 0 || child >= node_count) {
        return kStatusInvalidChildIndex;
      }
    }
  }

  return kStatusOk;
}

/*
 * Computes strategy averaging weight for a given CFR iteration.
 * During the delay period (iteration <= averaging_delay), returns 0 (no averaging).
 * After delay: returns 1 for uniform averaging, or (iteration - delay) for linear
 * averaging (which gives more weight to later, more-converged iterations).
 * Branchless: multiplies by the active flag (0 or 1).
 */
CFR_FORCE_INLINE int averaging_weight(const TreeSpec& spec, const int iteration) {
  const int delayed_iteration = iteration - spec.averaging_delay;
  // Branchless: active=0 when delayed<=0, so whole expression collapses to 0.
  // When linear_averaging=true: returns delayed_iteration.
  // When linear_averaging=false: returns 1 (if active).
  const int active = static_cast<int>(delayed_iteration > 0);
  return active * (spec.linear_averaging ? delayed_iteration : 1);
}

/* Fixed-point overload — same logic as the double-precision version. */
CFR_FORCE_INLINE int averaging_weight(const TreeSpecFixed& spec, const int iteration) {
  const int delayed_iteration = iteration - spec.averaging_delay;
  const int active = static_cast<int>(delayed_iteration > 0);
  return active * (spec.linear_averaging ? delayed_iteration : 1);
}

/*
 * Regret matching: converts cumulative regret values into a current strategy
 * (probability distribution). Actions with positive regret get probability
 * proportional to their regret; if no action has positive regret, falls back
 * to uniform. Template on F (double or float) for reuse across precision levels.
 *
 * Uses branchless arithmetic: computes both the normalized and uniform paths,
 * then blends them based on whether positive_sum > 0, avoiding a branch.
 */
template <typename F>
CFR_FORCE_INLINE void regret_matching_fp(
    const std::vector<F>& regrets,
    const int start,
    const int count,
    F* __restrict__ out_strategy) {
  // Guard against out-of-bounds access if upstream validation is bypassed.
  if (start < 0 || count <= 0 ||
      static_cast<size_t>(start + count) > regrets.size()) {
    // Fallback: uniform strategy.
    const F uniform = F(1) / static_cast<F>(count > 0 ? count : 1);
    for (int idx = 0; idx < count; ++idx) {
      out_strategy[idx] = uniform;
    }
    return;
  }
  F positive_sum = F(0);
  const F* __restrict__ regret_ptr = regrets.data() + start;
  for (int idx = 0; idx < count; ++idx) {
    const F positive = regret_ptr[idx] > F(0) ? regret_ptr[idx] : F(0);
    out_strategy[idx] = positive;
    positive_sum += positive;
  }

  // Branchless uniform/normalize: compute both paths, select by whether
  // positive_sum > 0. Avoids branch misprediction on the uniform fallback.
  const F has_positive = static_cast<F>(positive_sum > F(0));
  // When positive_sum==0: denominator becomes 1.0 (avoids div-by-zero),
  // and the positive path contributes 0 since all out_strategy[i]==0.
  const F inv = has_positive / (positive_sum + (F(1) - has_positive));
  const F uniform = (F(1) - has_positive) / static_cast<F>(count);
  for (int idx = 0; idx < count; ++idx) {
    out_strategy[idx] = out_strategy[idx] * inv + uniform;
  }
}

/* Double-precision convenience wrapper for regret_matching_fp. */
inline void regret_matching(
    const std::vector<double>& regrets,
    const int start,
    const int count,
    double* __restrict__ out_strategy) {
  regret_matching_fp(regrets, start, count, out_strategy);
}

/*
 * SolveWorkspace: mutable working state for a CFR training run.
 *   infoset_offsets  — prefix-sum array: infoset_offsets[i] is the flat index
 *                      where infoset i's action entries begin in the strategy arrays.
 *   cumulative_regret — running sum of counterfactual regrets per action (F=double or float).
 *   cumulative_strategy — running sum of reach-weighted strategies per action.
 *   chance_edge_weights — pre-normalized chance probabilities (sum to 1.0 per chance node).
 */
template <typename F>
struct SolveWorkspace {
  std::vector<int> infoset_offsets;
  std::vector<F> cumulative_regret;
  std::vector<F> cumulative_strategy;
  std::vector<F> chance_edge_weights;
};

/*
 * Extracts the average strategy for a single infoset from cumulative_strategy.
 * If the cumulative sum > 0, normalizes to a probability distribution.
 * Otherwise falls back to regret matching on cumulative_regret as a best-effort
 * approximation (this happens when the infoset was never reached during training).
 */
template <typename F>
inline void materialize_average_strategy_for_infoset(
    const TreeSpec& spec,
    const SolveWorkspace<F>& workspace,
    const int infoset,
    double* const out_strategy) {
  const int start = workspace.infoset_offsets[infoset];
  const int count = spec.infoset_action_counts[infoset];
  F sum = F(0);
  for (int idx = 0; idx < count; ++idx) {
    sum += workspace.cumulative_strategy[start + idx];
  }
  if (sum > F(0)) {
    const F inv = F(1) / sum;
    for (int idx = 0; idx < count; ++idx) {
      out_strategy[idx] = static_cast<double>(workspace.cumulative_strategy[start + idx] * inv);
    }
    return;
  }

  std::array<F, kInlineActionBufferSize> fallback_inline{};
  std::vector<F> fallback_heap;
  F* fallback = nullptr;
  if (count <= kInlineActionBufferSize) {
    fallback = fallback_inline.data();
  } else {
    fallback_heap.assign(static_cast<size_t>(count), F(0));
    fallback = fallback_heap.data();
  }
  regret_matching_fp(workspace.cumulative_regret, start, count, fallback);
  for (int idx = 0; idx < count; ++idx) {
    out_strategy[idx] = static_cast<double>(fallback[idx]);
  }
}

/*
 * Core CFR training loop. Validates the tree, builds the workspace (infoset
 * offsets, zero-initialized accumulators, normalized chance weights), then runs
 * spec.iterations passes of the CFR algorithm.
 *
 * Each iteration does a full DFS traversal of the game tree via a recursive
 * lambda 'cfr'. At each player node:
 *   1. Compute current strategy via regret matching on cumulative_regret.
 *   2. Recurse into each child, passing updated reach probabilities.
 *   3. Compute counterfactual regret for each action:
 *        regret(a) = opponent_reach * (action_value(a) - node_value)
 *      For player 1, the sign flips because utilities are from player 0's POV.
 *   4. Update cumulative_regret (with CFR+ clamping if enabled).
 *   5. Update cumulative_strategy weighted by avg_weight * acting_player_reach.
 *
 * Returns kStatusOk on success, or a validation error code.
 */
template <typename F>
inline int train_impl(const TreeSpec& spec, SolveWorkspace<F>& workspace) {
  const int validation = validate_tree(spec);
  if (validation != kStatusOk) {
    return validation;
  }

  const int infoset_count = static_cast<int>(spec.infoset_action_counts.size());
  workspace.infoset_offsets.assign(infoset_count + 1, 0);
  for (int infoset = 0; infoset < infoset_count; ++infoset) {
    workspace.infoset_offsets[infoset + 1] =
        workspace.infoset_offsets[infoset] + spec.infoset_action_counts[infoset];
  }
  const int strategy_size = workspace.infoset_offsets[infoset_count];

  workspace.cumulative_regret.assign(strategy_size, F(0));
  workspace.cumulative_strategy.assign(strategy_size, F(0));
  workspace.chance_edge_weights.assign(spec.edge_probabilities.size(), F(0));

  const int node_count = static_cast<int>(spec.node_types.size());
  for (int node = 0; node < node_count; ++node) {
    if (spec.node_types[node] != kNodeChance) {
      continue;
    }
    const int start = spec.node_starts[node];
    const int count = spec.node_counts[node];
    F prob_sum = F(0);
    for (int edge = start; edge < start + count; ++edge) {
      prob_sum += static_cast<F>(spec.edge_probabilities[edge]);
    }
    const F inv_prob_sum = F(1) / prob_sum;
    for (int edge = start; edge < start + count; ++edge) {
      workspace.chance_edge_weights[edge] =
          static_cast<F>(spec.edge_probabilities[edge]) * inv_prob_sum;
    }
  }

  auto cfr = [&](auto&& self, const int node_id, const F reach_p0, const F reach_p1,
                 const int avg_weight) -> F {
    const int node_type = spec.node_types[node_id];
    if (node_type == kNodeTerminal) {
      return static_cast<F>(spec.terminal_utilities[node_id]);
    }

    const int start = spec.node_starts[node_id];
    const int count = spec.node_counts[node_id];

    if (node_type == kNodeChance) {
      F value = F(0);
      for (int edge = start; edge < start + count; ++edge) {
        const F probability = workspace.chance_edge_weights[edge];
        const int child = spec.edge_child_ids[edge];
        value += probability * self(self, child, reach_p0 * probability, reach_p1 * probability, avg_weight);
      }
      return value;
    }

    const int infoset = spec.node_infosets[node_id];
    const int infoset_start = workspace.infoset_offsets[infoset];

    std::array<F, kInlineActionBufferSize> strategy_inline{};
    std::array<F, kInlineActionBufferSize> action_utility_inline{};
    std::vector<F> strategy_heap;
    std::vector<F> action_utility_heap;
    F* strategy = nullptr;
    F* action_utility = nullptr;
    if (count <= kInlineActionBufferSize) {
      strategy = strategy_inline.data();
      action_utility = action_utility_inline.data();
    } else {
      strategy_heap.assign(static_cast<size_t>(count), F(0));
      action_utility_heap.assign(static_cast<size_t>(count), F(0));
      strategy = strategy_heap.data();
      action_utility = action_utility_heap.data();
    }
    regret_matching_fp(workspace.cumulative_regret, infoset_start, count, strategy);

    F node_value = F(0);
    for (int idx = 0; idx < count; ++idx) {
      const int child = spec.edge_child_ids[start + idx];
      const F action_value =
          (node_type == kNodePlayer0)
              ? self(self, child, reach_p0 * strategy[idx], reach_p1, avg_weight)
              : self(self, child, reach_p0, reach_p1 * strategy[idx], avg_weight);
      action_utility[idx] = action_value;
      node_value += strategy[idx] * action_value;
    }

    const bool accumulate_strategy = avg_weight != 0;
    if (node_type == kNodePlayer0) {
      const F strategy_scale = static_cast<F>(avg_weight) * reach_p0;
      for (int idx = 0; idx < count; ++idx) {
        const F regret_delta = reach_p1 * (action_utility[idx] - node_value);
        F updated = workspace.cumulative_regret[infoset_start + idx] + regret_delta;
        updated = spec.cfr_plus ? std::max(updated, F(0)) : updated;
        workspace.cumulative_regret[infoset_start + idx] = updated;
        if (accumulate_strategy) {
          workspace.cumulative_strategy[infoset_start + idx] += strategy_scale * strategy[idx];
        }
      }
    } else {
      const F strategy_scale = static_cast<F>(avg_weight) * reach_p1;
      for (int idx = 0; idx < count; ++idx) {
        const F regret_delta = reach_p0 * (node_value - action_utility[idx]);
        F updated = workspace.cumulative_regret[infoset_start + idx] + regret_delta;
        updated = spec.cfr_plus ? std::max(updated, F(0)) : updated;
        workspace.cumulative_regret[infoset_start + idx] = updated;
        if (accumulate_strategy) {
          workspace.cumulative_strategy[infoset_start + idx] += strategy_scale * strategy[idx];
        }
      }
    }
    return node_value;
  };

  for (int iteration = 1; iteration <= spec.iterations; ++iteration) {
    const int avg_weight = averaging_weight(spec, iteration);
    (void)cfr(cfr, spec.root_node_id, F(1), F(1), avg_weight);
  }

  return kStatusOk;
}

/*
 * Full CFR solve: trains the workspace, then extracts average strategies for
 * all infosets and computes the root expected value by traversing the tree
 * once more using the average strategies.
 */
template <typename F>
inline int solve_impl(const TreeSpec& spec, SolveOutput& output) {
  SolveWorkspace<F> workspace;
  const int status = train_impl(spec, workspace);
  if (status != kStatusOk) {
    return status;
  }

  const int infoset_count = static_cast<int>(spec.infoset_action_counts.size());
  const int strategy_size = workspace.infoset_offsets[infoset_count];
  output.average_strategies.assign(strategy_size, 0.0);

  for (int infoset = 0; infoset < infoset_count; ++infoset) {
    materialize_average_strategy_for_infoset(
        spec,
        workspace,
        infoset,
        output.average_strategies.data() + workspace.infoset_offsets[infoset]);
  }

  auto expected_value = [&](auto&& self, const int node_id) -> double {
    const int node_type = spec.node_types[node_id];
    if (node_type == kNodeTerminal) {
      return spec.terminal_utilities[node_id];
    }

    const int start = spec.node_starts[node_id];
    const int count = spec.node_counts[node_id];
    if (node_type == kNodeChance) {
      double value = 0.0;
      for (int edge = start; edge < start + count; ++edge) {
        const double probability = static_cast<double>(workspace.chance_edge_weights[edge]);
        value += probability * self(self, spec.edge_child_ids[edge]);
      }
      return value;
    }

    const int infoset = spec.node_infosets[node_id];
    const int infoset_start = workspace.infoset_offsets[infoset];
    double value = 0.0;
    for (int idx = 0; idx < count; ++idx) {
      value += output.average_strategies[infoset_start + idx] *
               self(self, spec.edge_child_ids[start + idx]);
    }
    return value;
  };

  output.expected_value_player0 = expected_value(expected_value, spec.root_node_id);
  return kStatusOk;
}

/* Public entry point for double-precision CFR. */
inline int solve(const TreeSpec& spec, SolveOutput& output) {
  return solve_impl<double>(spec, output);
}

/*
 * Root-only CFR solve: trains the workspace, then extracts the average strategy
 * for only the specified root infoset. Used by the JNI solveTreeRoot entry point
 * when the caller only needs the decision at a single infoset (avoids copying
 * the entire strategy array back through JNI).
 */
template <typename F>
inline int solve_root_impl(const TreeSpec& spec, const int root_infoset_index, RootSolveOutput& output) {
  SolveWorkspace<F> workspace;
  const int status = train_impl(spec, workspace);
  if (status != kStatusOk) {
    return status;
  }

  const int infoset_count = static_cast<int>(spec.infoset_action_counts.size());
  if (root_infoset_index < 0 || root_infoset_index >= infoset_count) {
    return kStatusInvalidInfoSetIndex;
  }

  const int count = spec.infoset_action_counts[root_infoset_index];
  output.root_strategy.assign(static_cast<size_t>(count), 0.0);
  materialize_average_strategy_for_infoset(
      spec,
      workspace,
      root_infoset_index,
      output.root_strategy.data());
  return kStatusOk;
}

/* Public entry point for root-only double-precision CFR. */
inline int solve_root(const TreeSpec& spec, const int root_infoset_index, RootSolveOutput& output) {
  return solve_root_impl<double>(spec, root_infoset_index, output);
}

/* Validates a TreeSpecFixed — same structural checks as validate_tree but
 * for fixed-point arrays (edge_probabilities_raw, terminal_utilities_raw).
 * Chance probabilities must be non-negative integers summing to > 0. */
inline int validate_tree_fixed(const TreeSpecFixed& spec) {
  if (spec.iterations <= 0 || spec.averaging_delay < 0) {
    return kStatusInvalidIterations;
  }

  const int node_count = static_cast<int>(spec.node_types.size());
  if (node_count <= 0) {
    return kStatusInvalidNodeLayout;
  }
  if (static_cast<int>(spec.node_starts.size()) != node_count ||
      static_cast<int>(spec.node_counts.size()) != node_count ||
      static_cast<int>(spec.node_infosets.size()) != node_count ||
      static_cast<int>(spec.terminal_utilities_raw.size()) != node_count) {
    return kStatusLengthMismatch;
  }
  if (spec.root_node_id < 0 || spec.root_node_id >= node_count) {
    return kStatusInvalidRootNode;
  }

  const int edge_count = static_cast<int>(spec.edge_child_ids.size());
  if (static_cast<int>(spec.edge_probabilities_raw.size()) != edge_count) {
    return kStatusLengthMismatch;
  }

  const int infoset_count = static_cast<int>(spec.infoset_players.size());
  if (infoset_count <= 0 || static_cast<int>(spec.infoset_action_counts.size()) != infoset_count) {
    return kStatusLengthMismatch;
  }

  for (int infoset = 0; infoset < infoset_count; ++infoset) {
    if ((spec.infoset_players[infoset] != 0 && spec.infoset_players[infoset] != 1) ||
        spec.infoset_action_counts[infoset] <= 0) {
      return kStatusInvalidInfoSetIndex;
    }
  }

  for (int node = 0; node < node_count; ++node) {
    const int node_type = spec.node_types[node];
    if (node_type < kNodeTerminal || node_type > kNodePlayer1) {
      return kStatusInvalidNodeType;
    }

    const int start = spec.node_starts[node];
    const int count = spec.node_counts[node];
    if (start < 0 || count < 0 || start > edge_count || (start + count) > edge_count) {
      return kStatusInvalidNodeLayout;
    }

    if (node_type == kNodeTerminal) {
      if (count != 0) {
        return kStatusInvalidNodeLayout;
      }
      continue;
    }

    if (count <= 0) {
      return kStatusInvalidNodeLayout;
    }

    if (node_type == kNodeChance) {
      int64_t prob_sum = 0;
      for (int edge = start; edge < start + count; ++edge) {
        const int child = spec.edge_child_ids[edge];
        if (child < 0 || child >= node_count) {
          return kStatusInvalidChildIndex;
        }
        const int probability_raw = spec.edge_probabilities_raw[edge];
        if (probability_raw < 0) {
          return kStatusInvalidChanceProbabilities;
        }
        prob_sum += probability_raw;
      }
      if (prob_sum <= 0) {
        return kStatusInvalidChanceProbabilities;
      }
      if (spec.node_infosets[node] != -1) {
        return kStatusInvalidInfoSetIndex;
      }
      continue;
    }

    const int infoset_index = spec.node_infosets[node];
    if (infoset_index < 0 || infoset_index >= infoset_count) {
      return kStatusInvalidInfoSetIndex;
    }
    const int expected_player = (node_type == kNodePlayer0) ? 0 : 1;
    if (spec.infoset_players[infoset_index] != expected_player) {
      return kStatusInvalidInfoSetIndex;
    }
    if (spec.infoset_action_counts[infoset_index] != count) {
      return kStatusInfoSetActionMismatch;
    }
    for (int edge = start; edge < start + count; ++edge) {
      const int child = spec.edge_child_ids[edge];
      if (child < 0 || child >= node_count) {
        return kStatusInvalidChildIndex;
      }
    }
  }

  return kStatusOk;
}

/* Q30 * Q30 -> Q30 probability multiplication: (left * right) >> 30.
 * Uses int64_t intermediate to avoid overflow (two Q30 values can be up to ~10^9). */
CFR_FORCE_INLINE int multiply_prob_raw(const int left, const int right) {
  return static_cast<int>((static_cast<int64_t>(left) * static_cast<int64_t>(right)) >> kProbScaleBits);
}

// Signed round-to-nearest shift by 30 bits. Matches Scala roundShift30Signed:
// arithmetic right-shift alone biases small negative products toward -infinity,
// so we round on absolute magnitude before restoring the sign.
//
// Branchless implementation using arithmetic right-shift sign propagation.
// sign_mask: all-ones (-1) if product < 0, else 0.
// Branchless abs: (x ^ mask) - mask.
// Branchless negate: (x ^ mask) - mask applied to rounded result.
CFR_FORCE_INLINE int round_shift_30_signed(const int64_t product) {
  const int64_t sign_mask = product >> 63;                      // 0 or -1
  const int64_t abs_product = (product ^ sign_mask) - sign_mask; // branchless abs
  const int rounded = static_cast<int>((abs_product + (1LL << 29)) >> kProbScaleBits);
  // Restore sign branchlessly: negate rounded iff sign_mask == -1
  return (rounded ^ static_cast<int>(sign_mask)) - static_cast<int>(sign_mask);
}

/* Q13_value * Q30_prob -> Q13 result via round_shift_30_signed.
 * Used to weight utility values by probabilities in fixed-point CFR. */
CFR_FORCE_INLINE int multiply_fixed_by_prob_raw(const int value_raw, const int probability_raw) {
  return round_shift_30_signed(
      static_cast<int64_t>(value_raw) * static_cast<int64_t>(probability_raw));
}

/* Writes a uniform Q30 probability distribution: each action gets kProbScale/count,
 * with the remainder distributed to the first (remainder) actions to sum exactly
 * to kProbScale. This avoids floating-point rounding issues. */
inline void write_uniform_probabilities_raw(const int count, int* __restrict__ out_strategy_raw) {
  const int base = kProbScale / count;
  const int remainder = kProbScale - (base * count);
  for (int idx = 0; idx < count; ++idx) {
    out_strategy_raw[idx] = base + static_cast<int>(idx < remainder);
  }
}

/*
 * Normalizes non-negative int64 weights to Q30 probabilities summing to kProbScale.
 * Uses the "last positive gets the remainder" trick to ensure exact summation.
 * Pre-reduces weights by right-shifting if they could overflow when multiplied
 * by kProbScale. Falls back to uniform if all weights are zero.
 */
inline void normalize_non_negative_weights_to_prob_raw(
    const int64_t* __restrict__ weights,
    const int count,
    int* __restrict__ out_probabilities_raw) {
  int64_t sum = 0;
  int last_positive = -1;
  for (int idx = 0; idx < count; ++idx) {
    const int64_t weight = weights[idx];
    if (weight > 0) {
      sum += weight;
      last_positive = idx;
    }
  }
  if (sum <= 0 || last_positive < 0) {
    write_uniform_probabilities_raw(count, out_probabilities_raw);
    return;
  }

  // Pre-reduce weights to prevent overflow in weight * kProbScale.
  // We need (sum >> shift) * kProbScale <= INT64_MAX.
  int shift = 0;
  {
    int64_t s = sum;
    constexpr int64_t safe_limit = INT64_MAX / static_cast<int64_t>(kProbScale);
    while (s > safe_limit) {
      s >>= 1;
      shift++;
    }
  }

  const int64_t shifted_sum = sum >> shift;
  int assigned = 0;
  for (int idx = 0; idx < count; ++idx) {
    const int64_t weight = weights[idx];
    if (weight <= 0) {
      out_probabilities_raw[idx] = 0;
    } else if (idx == last_positive) {
      out_probabilities_raw[idx] = kProbScale - assigned;
    } else {
      const int raw = static_cast<int>(
          ((weight >> shift) * static_cast<int64_t>(kProbScale)) / shifted_sum);
      out_probabilities_raw[idx] = raw;
      assigned += raw;
    }
  }
}

/* Same as normalize_non_negative_weights_to_prob_raw but reads from a contiguous
 * subrange of a vector<int> (edge weights for a single chance node). Simpler
 * because int32 weights don't need the overflow pre-reduction step. */
inline void normalize_non_negative_edge_weights_to_prob_raw(
    const std::vector<int>& weights,
    const int start,
    const int count,
    int* __restrict__ out_probabilities_raw) {
  int64_t sum = 0;
  int last_positive = -1;
  for (int idx = 0; idx < count; ++idx) {
    const int weight = weights[start + idx];
    if (weight > 0) {
      sum += weight;
      last_positive = idx;
    }
  }
  if (sum <= 0 || last_positive < 0) {
    write_uniform_probabilities_raw(count, out_probabilities_raw);
    return;
  }

  int assigned = 0;
  for (int idx = 0; idx < count; ++idx) {
    const int weight = weights[start + idx];
    if (weight <= 0) {
      out_probabilities_raw[idx] = 0;
    } else if (idx == last_positive) {
      out_probabilities_raw[idx] = kProbScale - assigned;
    } else {
      const int raw = static_cast<int>((static_cast<int64_t>(weight) * static_cast<int64_t>(kProbScale)) / sum);
      out_probabilities_raw[idx] = raw;
      assigned += raw;
    }
  }
}

/* Saturating cast from int64 to int32. Clamps to [INT32_MIN, INT32_MAX] to
 * prevent undefined behavior from overflow in fixed-point accumulation. */
inline int checked_fixed_raw(const int64_t raw) {
  return static_cast<int>(std::clamp(
      raw,
      static_cast<int64_t>(INT32_MIN),
      static_cast<int64_t>(INT32_MAX)));
}

inline void halve_int_array(std::vector<int32_t>& values, const int start, const int count) {
  for (int idx = 0; idx < count; ++idx) {
    values[start + idx] >>= 1;
  }
}

inline void halve_long_array(std::vector<int64_t>& values, const int start, const int count) {
  for (int idx = 0; idx < count; ++idx) {
    values[start + idx] >>= 1;
  }
}

inline bool needs_regret_emergency_rescale(
    const std::vector<int32_t>& cumulative_regret,
    const int* regret_deltas,
    const int start,
    const int count) {
  for (int idx = 0; idx < count; ++idx) {
    const int64_t updated =
        static_cast<int64_t>(cumulative_regret[start + idx]) +
        static_cast<int64_t>(regret_deltas[idx]);
    if (updated < static_cast<int64_t>(INT32_MIN) ||
        updated > static_cast<int64_t>(INT32_MAX)) {
      return true;
    }
  }
  return false;
}

inline bool needs_strategy_emergency_rescale(
    const std::vector<int64_t>& cumulative_strategy,
    const int64_t* strategy_deltas,
    const int start,
    const int count) {
  for (int idx = 0; idx < count; ++idx) {
    const int64_t delta = strategy_deltas[idx];
    if (cumulative_strategy[start + idx] > INT64_MAX - delta) {
      return true;
    }
  }
  return false;
}

CFR_FORCE_INLINE void regret_matching_fixed(
    const std::vector<int32_t>& regrets,
    const int start,
    const int count,
    int* __restrict__ out_strategy_raw) {
  // Guard against out-of-bounds access if upstream validation is bypassed.
  if (start < 0 || count <= 0 ||
      static_cast<size_t>(start + count) > regrets.size()) {
    write_uniform_probabilities_raw(count > 0 ? count : 0, out_strategy_raw);
    return;
  }
  int64_t positive_sum = 0;
  int last_positive = -1;
  for (int idx = 0; idx < count; ++idx) {
    const int32_t regret = regrets[start + idx];
    if (regret > 0) {
      positive_sum += static_cast<int64_t>(regret);
      last_positive = idx;
    }
  }

  if (positive_sum > 0 && last_positive >= 0) {
    int assigned = 0;
    for (int idx = 0; idx < count; ++idx) {
      const int32_t regret = regrets[start + idx];
      if (regret <= 0) {
        out_strategy_raw[idx] = 0;
      } else if (idx == last_positive) {
        out_strategy_raw[idx] = kProbScale - assigned;
      } else {
        const int raw = static_cast<int>(
            (static_cast<int64_t>(regret) * static_cast<int64_t>(kProbScale)) / positive_sum);
        out_strategy_raw[idx] = raw;
        assigned += raw;
      }
    }
  } else {
    write_uniform_probabilities_raw(count, out_strategy_raw);
  }
}

inline int solve_fixed(const TreeSpecFixed& spec, SolveOutputFixed& output) {
  const int validation = validate_tree_fixed(spec);
  if (validation != kStatusOk) {
    return validation;
  }

  const int infoset_count = static_cast<int>(spec.infoset_action_counts.size());
  std::vector<int> infoset_offsets(infoset_count + 1, 0);
  for (int infoset = 0; infoset < infoset_count; ++infoset) {
    infoset_offsets[infoset + 1] = infoset_offsets[infoset] + spec.infoset_action_counts[infoset];
  }
  const int strategy_size = infoset_offsets[infoset_count];
  output.average_strategies_raw.assign(strategy_size, 0);

  std::vector<int32_t> cumulative_regret(strategy_size, 0);
  std::vector<int64_t> cumulative_strategy(strategy_size, 0);
  std::vector<int> chance_edge_weights_raw(spec.edge_probabilities_raw.size(), 0);

  const int node_count = static_cast<int>(spec.node_types.size());
  for (int node = 0; node < node_count; ++node) {
    if (spec.node_types[node] != kNodeChance) {
      continue;
    }
    const int start = spec.node_starts[node];
    const int count = spec.node_counts[node];
    normalize_non_negative_edge_weights_to_prob_raw(
        spec.edge_probabilities_raw,
        start,
        count,
        chance_edge_weights_raw.data() + start);
  }

  auto cfr = [&](auto&& self, const int node_id, const int reach_p0_raw, const int reach_p1_raw,
                 const int avg_weight) -> int {
    const int node_type = spec.node_types[node_id];
    if (node_type == kNodeTerminal) {
      return spec.terminal_utilities_raw[node_id];
    }

    const int start = spec.node_starts[node_id];
    const int count = spec.node_counts[node_id];

    if (node_type == kNodeChance) {
      int value_raw = 0;
      for (int edge = start; edge < start + count; ++edge) {
        const int probability_raw = chance_edge_weights_raw[edge];
        const int child = spec.edge_child_ids[edge];
        const int child_value_raw = self(
            self,
            child,
            multiply_prob_raw(reach_p0_raw, probability_raw),
            multiply_prob_raw(reach_p1_raw, probability_raw),
            avg_weight);
        value_raw = checked_fixed_raw(
            static_cast<int64_t>(value_raw) +
            static_cast<int64_t>(multiply_fixed_by_prob_raw(child_value_raw, probability_raw)));
      }
      return value_raw;
    }

    const int infoset = spec.node_infosets[node_id];
    const int infoset_start = infoset_offsets[infoset];

    std::array<int, kInlineActionBufferSize> strategy_inline{};
    std::array<int, kInlineActionBufferSize> action_utility_inline{};
    std::vector<int> strategy_heap;
    std::vector<int> action_utility_heap;
    int* strategy_raw = nullptr;
    int* action_utility_raw = nullptr;
    if (count <= kInlineActionBufferSize) {
      strategy_raw = strategy_inline.data();
      action_utility_raw = action_utility_inline.data();
    } else {
      strategy_heap.assign(static_cast<size_t>(count), 0);
      action_utility_heap.assign(static_cast<size_t>(count), 0);
      strategy_raw = strategy_heap.data();
      action_utility_raw = action_utility_heap.data();
    }
    regret_matching_fixed(cumulative_regret, infoset_start, count, strategy_raw);

    int node_value_raw = 0;
    for (int idx = 0; idx < count; ++idx) {
      const int child = spec.edge_child_ids[start + idx];
      const int action_value_raw =
          (node_type == kNodePlayer0)
              ? self(self, child, multiply_prob_raw(reach_p0_raw, strategy_raw[idx]), reach_p1_raw, avg_weight)
              : self(self, child, reach_p0_raw, multiply_prob_raw(reach_p1_raw, strategy_raw[idx]), avg_weight);
      action_utility_raw[idx] = action_value_raw;
      node_value_raw = checked_fixed_raw(
          static_cast<int64_t>(node_value_raw) +
          static_cast<int64_t>(multiply_fixed_by_prob_raw(action_value_raw, strategy_raw[idx])));
    }

    // Compute regret deltas, and strategy deltas only once averaging activates.
    const bool accumulate_strategy = avg_weight != 0;
    std::array<int, kInlineActionBufferSize> regret_delta_inline{};
    std::array<int64_t, kInlineActionBufferSize> strategy_delta_inline{};
    std::vector<int> regret_delta_heap;
    std::vector<int64_t> strategy_delta_heap;
    int* regret_deltas = nullptr;
    int64_t* strategy_deltas = nullptr;
    if (count <= kInlineActionBufferSize) {
      regret_deltas = regret_delta_inline.data();
      strategy_deltas = strategy_delta_inline.data();
    } else {
      regret_delta_heap.assign(static_cast<size_t>(count), 0);
      if (accumulate_strategy) {
        strategy_delta_heap.assign(static_cast<size_t>(count), 0);
      }
      regret_deltas = regret_delta_heap.data();
      strategy_deltas = accumulate_strategy ? strategy_delta_heap.data() : nullptr;
    }

    if (node_type == kNodePlayer0) {
      for (int idx = 0; idx < count; ++idx) {
        regret_deltas[idx] =
            multiply_fixed_by_prob_raw(action_utility_raw[idx] - node_value_raw, reach_p1_raw);
        if (accumulate_strategy) {
          strategy_deltas[idx] =
              static_cast<int64_t>(avg_weight) *
              static_cast<int64_t>(multiply_prob_raw(reach_p0_raw, strategy_raw[idx]));
        }
      }
    } else {
      for (int idx = 0; idx < count; ++idx) {
        regret_deltas[idx] =
            multiply_fixed_by_prob_raw(node_value_raw - action_utility_raw[idx], reach_p0_raw);
        if (accumulate_strategy) {
          strategy_deltas[idx] =
              static_cast<int64_t>(avg_weight) *
              static_cast<int64_t>(multiply_prob_raw(reach_p1_raw, strategy_raw[idx]));
        }
      }
    }

    // Emergency rescaling before update (matches Scala applyFixedActionUpdates).
    while (needs_regret_emergency_rescale(cumulative_regret, regret_deltas, infoset_start, count)) {
      halve_int_array(cumulative_regret, infoset_start, count);
    }
    while (accumulate_strategy &&
           needs_strategy_emergency_rescale(cumulative_strategy, strategy_deltas, infoset_start, count)) {
      halve_long_array(cumulative_strategy, infoset_start, count);
    }

    // Apply regret updates with CFR+ clamping and threshold rescaling.
    bool rescale_regret = false;
    for (int idx = 0; idx < count; ++idx) {
      int64_t updated =
          static_cast<int64_t>(cumulative_regret[infoset_start + idx]) +
          static_cast<int64_t>(regret_deltas[idx]);
      // Branchless CFR+ clamp: if cfr_plus, zero out negative values.
      // updated &= ~(cfr_plus_mask & sign_bit) where cfr_plus_mask = -spec.cfr_plus
      updated &= ~(static_cast<int64_t>(-static_cast<int>(spec.cfr_plus)) & (updated >> 63));
      cumulative_regret[infoset_start + idx] = checked_fixed_raw(updated);
      if (std::abs(static_cast<int64_t>(cumulative_regret[infoset_start + idx])) >=
          static_cast<int64_t>(kRegretRescaleThreshold)) {
        rescale_regret = true;
      }
    }
    if (rescale_regret) {
      halve_int_array(cumulative_regret, infoset_start, count);
    }

    // Apply strategy updates with threshold rescaling.
    bool rescale_strategy = false;
    if (accumulate_strategy) {
      for (int idx = 0; idx < count; ++idx) {
        cumulative_strategy[infoset_start + idx] += strategy_deltas[idx];
        if (cumulative_strategy[infoset_start + idx] >= kStrategyRescaleThreshold) {
          rescale_strategy = true;
        }
      }
    }
    if (rescale_strategy) {
      halve_long_array(cumulative_strategy, infoset_start, count);
    }

    return node_value_raw;
  };

  for (int iteration = 1; iteration <= spec.iterations; ++iteration) {
    const int avg_weight = averaging_weight(spec, iteration);
    (void)cfr(cfr, spec.root_node_id, kProbScale, kProbScale, avg_weight);
  }

  for (int infoset = 0; infoset < infoset_count; ++infoset) {
    const int start = infoset_offsets[infoset];
    const int count = spec.infoset_action_counts[infoset];
    int64_t sum = 0;
    for (int idx = 0; idx < count; ++idx) {
      sum += cumulative_strategy[start + idx];
    }
    if (sum > 0) {
      normalize_non_negative_weights_to_prob_raw(
          cumulative_strategy.data() + start,
          count,
          output.average_strategies_raw.data() + start);
    } else {
      regret_matching_fixed(cumulative_regret, start, count, output.average_strategies_raw.data() + start);
    }
  }

  auto expected_value = [&](auto&& self, const int node_id) -> int {
    const int node_type = spec.node_types[node_id];
    if (node_type == kNodeTerminal) {
      return spec.terminal_utilities_raw[node_id];
    }

    const int start = spec.node_starts[node_id];
    const int count = spec.node_counts[node_id];
    if (node_type == kNodeChance) {
      int value_raw = 0;
      for (int edge = start; edge < start + count; ++edge) {
        const int probability_raw = chance_edge_weights_raw[edge];
        value_raw = checked_fixed_raw(
            static_cast<int64_t>(value_raw) +
            static_cast<int64_t>(multiply_fixed_by_prob_raw(self(self, spec.edge_child_ids[edge]), probability_raw)));
      }
      return value_raw;
    }

    const int infoset = spec.node_infosets[node_id];
    const int infoset_start = infoset_offsets[infoset];
    int value_raw = 0;
    for (int idx = 0; idx < count; ++idx) {
      value_raw = checked_fixed_raw(
          static_cast<int64_t>(value_raw) +
          static_cast<int64_t>(multiply_fixed_by_prob_raw(
              self(self, spec.edge_child_ids[start + idx]),
              output.average_strategies_raw[infoset_start + idx])));
    }
    return value_raw;
  };

  output.expected_value_player0_raw = expected_value(expected_value, spec.root_node_id);
  return kStatusOk;
}

}  // namespace cfrnative

#undef CFR_FORCE_INLINE
