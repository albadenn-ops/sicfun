# Batch CFR CUDA Kernel Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CUDA kernel that solves many CFR game trees in parallel using float arithmetic, providing 100-600x speedup over sequential CPU solving for the CFR iteration phase of bulk hand analysis.

**Architecture:** One CUDA thread per game tree. All trees in a batch share identical topology (same action structure, branching) but differ in terminal utilities and chance weights (per hero hand). The kernel runs the full CFR iteration loop per thread using float arithmetic (30x faster than double on GPU, no fixed-point complexity). Working memory (cumulative regret + strategy) lives in device global memory, batched per tree.

**Tech Stack:** CUDA C++ (nvcc, sm_50+), JNI, Scala 3

---

## Context

### Why float, not double or fixed-point?

The CUDA throughput benchmark proved on real hardware:
- **T4 (server)**: float 30.1x faster than double, int32 19.9x faster
- **GTX 960M (dev laptop)**: float 14.5x faster than double, int32 9.1x faster

Float avoids:
- The double FP64 penalty on consumer/datacenter GPUs (1:32 to 1:64 ratio)
- The fixed-point parity bug documented in Phase 5 of the design spec
- The x86 float conversion overhead (irrelevant on GPU where float is native width)

### Why batch?

Current CFR solving is single-tree sequential: each `solveDecisionPolicy()` call sends one tree to the CPU native solver (~3-10ms each). For bulk analysis (all 1326 starting hands at a game state), that's 4-13 seconds.

With batch GPU solving: 1326 trees × 1500 iterations × ~1250 nodes each = ~25 GFLOP total. A T4 at 8.1 TFLOPS finishes this in ~3-10ms. **CFR iteration speedup: 100-600x.**

**Important caveat:** The end-to-end `solveBatchDecisionPolicies` time includes equity computation per hero hand (the dominant cost at ~2-5ms per hand). The GPU batch path only accelerates the CFR iteration phase. A future optimization could batch equity computation as well, but that is out of scope for this plan.

### Tree structure (critical for kernel design)

From `HoldemCfrSolver.toNativeTreeSpec` (line 2016), every tree for a given game state has:
- **Depth exactly 4**: root chance → hero root → villain response → hero re-response → terminal
- **Identical topology** across all hero hands (villainCount=96 always, since maxVillainHands=96 < available villain hands)
- **~1250 nodes**, ~1440 edges, ~195 infosets, ~584 strategy elements per tree
- **~7KB working memory** per tree (regret float[584] + strategy float[584])

Because topology is shared, **all CUDA threads execute identical control flow = zero warp divergence.**

### Files overview

| File | Action | Purpose |
|------|--------|---------|
| `src/main/native/jni/CfrBatchCudaKernel.cuh` | Create | CUDA kernel + host launcher |
| `src/main/native/jni/HoldemCfrNativeGpuBindings.cu` | Modify | Add JNI `solveTreeBatch` entry point |
| `src/main/java/sicfun/holdem/HoldemCfrNativeGpuBindings.java` | Modify | Add `solveTreeBatch` native method |
| `src/main/scala/sicfun/holdem/cfr/HoldemCfrNativeRuntime.scala` | Modify | Add `solveTreeBatch` wrapper |
| `src/main/scala/sicfun/holdem/cfr/HoldemCfrSolver.scala` | Modify | Add `solveBatch` / `solveBatchDecisionPolicies` |
| `src/test/scala/sicfun/holdem/cfr/HoldemCfrBatchSolverTest.scala` | Create | Batch vs single-tree parity tests |
| `src/main/scala/sicfun/holdem/bench/HoldemCfrBatchBenchmark.scala` | Create | Throughput benchmark |

---

## Chunk 1: CUDA Kernel + Host Launcher

### Task 1: Create the batch CFR CUDA kernel header

**Files:**
- Create: `src/main/native/jni/CfrBatchCudaKernel.cuh`

The kernel solves N independent CFR trees with shared topology in parallel.
Each CUDA thread handles one complete tree (all iterations).

- [ ] **Step 1: Create `CfrBatchCudaKernel.cuh` with device helper functions**

```cuda
#pragma once

#include <cstdint>
#include <cstdio>

namespace cfrbatch {

constexpr int kMaxInlineActions = 8;

// ---- device helpers ----

__device__ __forceinline__ void regret_matching_f(
    const float* cumulative_regret,
    int start, int count,
    float* out_strategy)
{
  float positive_sum = 0.0f;
  for (int i = 0; i < count; ++i) {
    float pos = cumulative_regret[start + i] > 0.0f ? cumulative_regret[start + i] : 0.0f;
    out_strategy[i] = pos;
    positive_sum += pos;
  }
  if (positive_sum > 0.0f) {
    float inv = 1.0f / positive_sum;
    for (int i = 0; i < count; ++i) out_strategy[i] *= inv;
  } else {
    float uniform = 1.0f / static_cast<float>(count);
    for (int i = 0; i < count; ++i) out_strategy[i] = uniform;
  }
}

__device__ inline int averaging_weight(
    int iteration, int averaging_delay, bool linear_averaging)
{
  if (iteration <= averaging_delay) return 0;
  return linear_averaging ? (iteration - averaging_delay) : 1;
}

} // namespace cfrbatch
```

- [ ] **Step 2: Add the main batch kernel**

Append to `CfrBatchCudaKernel.cuh`:

```cuda
namespace cfrbatch {

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

  // Iterative DFS stack (max depth 4 for HoldemDecisionGame trees)
  constexpr int kMaxDepth = 5; // HoldemDecisionGame depth is 4, +1 headroom
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

    int node_type = node_types[root_node_id];
    int start = node_starts[root_node_id];
    int count = node_counts[root_node_id];

    // Pre-compute strategy for root if player node
    if (node_type == 2 || node_type == 3) {
      int infoset = node_infosets[root_node_id];
      int is_start = infoset_offsets[infoset];
      regret_matching_f(cum_regret, is_start, count, stack[0].strategy);
    }

    while (depth >= 0) {
      Frame& f = stack[depth];
      int ntype = node_types[f.node_id];
      int nstart = node_starts[f.node_id];
      int ncount = node_counts[f.node_id];

      if (ntype == 0) {
        // Terminal node
        float value = terminal_utilities[f.node_id];
        depth--;
        if (depth >= 0) {
          Frame& parent = stack[depth];
          int pidx = parent.action_idx - 1;
          parent.action_values[pidx] = value;
          int ptype = node_types[parent.node_id];
          if (ptype == 1) { // chance
            parent.node_value += chance_weights[node_starts[parent.node_id] + pidx] * value;
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
        int child_id = edge_child_ids[edge];
        f.action_idx++;

        float child_reach_p0 = f.reach_p0;
        float child_reach_p1 = f.reach_p1;
        if (ntype == 1) { // chance
          float w = chance_weights[edge];
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

        int ctype = node_types[child_id];
        if (ctype == 2 || ctype == 3) {
          int cinfoset = node_infosets[child_id];
          int cis_start = infoset_offsets[cinfoset];
          int ccount = node_counts[child_id];
          regret_matching_f(cum_regret, cis_start, ccount, child.strategy);
        }
        continue;
      }

      // All children processed. Update regrets/strategy, then pop.
      float node_value = f.node_value;

      if (ntype == 2 || ntype == 3) {
        int infoset = node_infosets[f.node_id];
        int is_start = infoset_offsets[infoset];

        if (ntype == 2) { // player 0
          float strategy_scale = static_cast<float>(avg_weight) * f.reach_p0;
          for (int i = 0; i < ncount; ++i) {
            float regret_delta = f.reach_p1 * (f.action_values[i] - node_value);
            float updated = cum_regret[is_start + i] + regret_delta;
            cum_regret[is_start + i] = (cfr_plus && updated < 0.0f) ? 0.0f : updated;
            cum_strategy[is_start + i] += strategy_scale * f.strategy[i];
          }
        } else { // player 1
          float strategy_scale = static_cast<float>(avg_weight) * f.reach_p1;
          for (int i = 0; i < ncount; ++i) {
            float regret_delta = f.reach_p0 * (node_value - f.action_values[i]);
            float updated = cum_regret[is_start + i] + regret_delta;
            cum_regret[is_start + i] = (cfr_plus && updated < 0.0f) ? 0.0f : updated;
            cum_strategy[is_start + i] += strategy_scale * f.strategy[i];
          }
        }
      }

      depth--;
      if (depth >= 0) {
        Frame& parent = stack[depth];
        int pidx = parent.action_idx - 1;
        parent.action_values[pidx] = node_value;
        int ptype = node_types[parent.node_id];
        if (ptype == 1) {
          parent.node_value += chance_weights[node_starts[parent.node_id] + pidx] * node_value;
        } else {
          parent.node_value += parent.strategy[pidx] * node_value;
        }
      }
    }
    // End of one CFR iteration
  }

  // Extract average strategies
  float* out_strat = all_out_strategies + tree_id * strategy_size;
  int infoset_count = 0;
  // Count infosets from infoset_offsets: find how many entries exist
  // We need infoset_count passed or derivable. Use: last offset with non-zero actions.
  // Actually, strategy_size == infoset_offsets[infoset_count], so we scan.
  // Better: pass infoset_count as a parameter. Adding it.
  // For now, iterate over all strategy positions and normalize per-infoset.
  // We'll handle this by having the host launcher pass infoset_count.
}

} // namespace cfrbatch
```

Wait — the kernel needs `infoset_count` to extract strategies at the end. I'll add it as a parameter and also add a separate finalization helper.

- [ ] **Step 3: Refine kernel — add `infoset_count` param and strategy extraction**

Replace the strategy extraction section at the bottom of the kernel with:

```cuda
  // Extract average strategies from cumulative_strategy
  // (infoset_count is a kernel parameter, added in the full signature above)
  float* out_strat = all_out_strategies + tree_id * strategy_size;
  float ev_value = 0.0f; // will be computed in a separate pass

  for (int is = 0; is < infoset_count; ++is) {
    int is_start = infoset_offsets[is];
    int is_count = infoset_action_counts[is];
    float sum = 0.0f;
    for (int i = 0; i < is_count; ++i) sum += cum_strategy[is_start + i];
    if (sum > 0.0f) {
      float inv = 1.0f / sum;
      for (int i = 0; i < is_count; ++i)
        out_strat[is_start + i] = cum_strategy[is_start + i] * inv;
    } else {
      regret_matching_f(cum_regret, is_start, is_count, out_strat + is_start);
    }
  }
}
```

- [ ] **Step 4: Add host-side launcher function**

Append to `CfrBatchCudaKernel.cuh`:

```cuda
namespace cfrbatch {

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
  cudaMemcpy(h_out_strategies, d_out_strategies, B * S * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_out_ev, d_out_ev, B * sizeof(float), cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_node_types); cudaFree(d_node_starts); cudaFree(d_node_counts);
  cudaFree(d_node_infosets); cudaFree(d_edge_child_ids);
  cudaFree(d_infoset_action_counts); cudaFree(d_infoset_offsets);
  cudaFree(d_terminal_utilities); cudaFree(d_chance_weights);
  cudaFree(d_cum_regret); cudaFree(d_cum_strategy);
  cudaFree(d_out_strategies); cudaFree(d_out_ev);

  return 0;
}

} // namespace cfrbatch
```

- [ ] **Step 5: Compile the kernel as part of `sicfun_cfr_cuda.dll`**

The kernel header is `#include`d by `HoldemCfrNativeGpuBindings.cu`, which is already compiled by `build-windows-cuda11.ps1` line 70-97. No build script changes needed — just add the include.

Verify it compiles:
```powershell
.\src\main\native\build-windows-cuda11.ps1
```
Expected: `Built: ...\sicfun_cfr_cuda.dll` (may take 20-30s)

- [ ] **Step 6: Commit**

```bash
git add src/main/native/jni/CfrBatchCudaKernel.cuh
git commit -m "feat: add batch CFR CUDA kernel with float arithmetic

One thread per tree, shared topology across batch.
Iterative DFS with inline stack (max depth 8).
Host launcher handles cudaMalloc/memcpy/launch/cleanup."
```

---

### Task 2: Add JNI bridge for batch solve

**Files:**
- Modify: `src/main/java/sicfun/holdem/HoldemCfrNativeGpuBindings.java`
- Modify: `src/main/native/jni/HoldemCfrNativeGpuBindings.cu`

- [ ] **Step 1: Add native method declaration to Java bindings**

In `HoldemCfrNativeGpuBindings.java`, add after the `solveTreeFixed` method:

```java
/**
 * Batch CFR solve: N trees with shared topology, float arithmetic on GPU.
 *
 * <p>Per-tree data is concatenated: tree0 data, tree1 data, ..., treeN data.
 * Terminal utilities are indexed as [treeIdx * nodeCount + nodeId].
 * Chance weights are indexed as [treeIdx * edgeCount + edgeIdx].
 * Output strategies are indexed as [treeIdx * strategySize + offset].
 *
 * @return 0 on success, negative on CUDA error, positive on validation error
 */
public static native int solveTreeBatch(
    int iterations,
    int averagingDelay,
    boolean cfrPlus,
    boolean linearAveraging,
    int rootNodeId,
    int[] nodeTypes,
    int[] nodeStarts,
    int[] nodeCounts,
    int[] nodeInfosets,
    int[] edgeChildIds,
    int[] infosetPlayers,     // not consumed by kernel; passed for future validation
    int[] infosetActionCounts,
    int[] infosetOffsets,
    float[] terminalUtilities,
    float[] chanceWeights,
    float[] outAverageStrategies,
    float[] outExpectedValues,
    int batchSize
);
```

- [ ] **Step 2: Add JNI C++ implementation in `HoldemCfrNativeGpuBindings.cu`**

Add `#include "CfrBatchCudaKernel.cuh"` at top.

Add float array helpers in the anonymous namespace (alongside existing `read_int_array`, `read_double_array`, etc.):

```cpp
int read_float_array(JNIEnv* env, jfloatArray array, std::vector<float>& out) {
  if (array == nullptr) return cfrnative::kStatusNullArray;
  const jsize length = env->GetArrayLength(array);
  out.resize(static_cast<size_t>(length));
  if (length > 0) {
    env->GetFloatArrayRegion(array, 0, length, reinterpret_cast<jfloat*>(out.data()));
    if (clear_pending_jni_exception(env)) return cfrnative::kStatusReadFailure;
  }
  return cfrnative::kStatusOk;
}

int write_float_array(JNIEnv* env, jfloatArray array, const std::vector<float>& values) {
  if (array == nullptr) return cfrnative::kStatusNullArray;
  const jsize length = env->GetArrayLength(array);
  if (length != static_cast<jsize>(values.size())) return cfrnative::kStatusLengthMismatch;
  if (length > 0) {
    env->SetFloatArrayRegion(array, 0, length, reinterpret_cast<const jfloat*>(values.data()));
    if (clear_pending_jni_exception(env)) return cfrnative::kStatusWriteFailure;
  }
  return cfrnative::kStatusOk;
}
```

Add the JNI function:

```cuda
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemCfrNativeGpuBindings_solveTreeBatch(
    JNIEnv* env,
    jclass /*clazz*/,
    jint iterations,
    jint averagingDelay,
    jboolean cfrPlus,
    jboolean linearAveraging,
    jint rootNodeId,
    jintArray nodeTypesArray,
    jintArray nodeStartsArray,
    jintArray nodeCountsArray,
    jintArray nodeInfosetsArray,
    jintArray edgeChildIdsArray,
    jintArray infosetPlayersArray,
    jintArray infosetActionCountsArray,
    jintArray infosetOffsetsArray,
    jfloatArray terminalUtilitiesArray,
    jfloatArray chanceWeightsArray,
    jfloatArray outAverageStrategiesArray,
    jfloatArray outExpectedValuesArray,
    jint batchSize) {

  // Read shared topology
  std::vector<int> node_types, node_starts, node_counts, node_infosets;
  std::vector<int> edge_child_ids, infoset_players, infoset_action_counts, infoset_offsets;

  int status;
  status = read_int_array(env, nodeTypesArray, node_types);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, nodeStartsArray, node_starts);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, nodeCountsArray, node_counts);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, nodeInfosetsArray, node_infosets);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, edgeChildIdsArray, edge_child_ids);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, infosetPlayersArray, infoset_players);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, infosetActionCountsArray, infoset_action_counts);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, infosetOffsetsArray, infoset_offsets);
  if (status != cfrnative::kStatusOk) return status;

  const int N = static_cast<int>(node_types.size());
  const int E = static_cast<int>(edge_child_ids.size());
  const int I = static_cast<int>(infoset_action_counts.size());
  const int S = infoset_offsets.back(); // strategy_size = last offset
  const int B = static_cast<int>(batchSize);

  // Read per-tree float arrays
  std::vector<float> terminal_utilities, chance_weights;
  status = read_float_array(env, terminalUtilitiesArray, terminal_utilities);
  if (status != cfrnative::kStatusOk) return status;
  status = read_float_array(env, chanceWeightsArray, chance_weights);
  if (status != cfrnative::kStatusOk) return status;

  if (static_cast<int>(terminal_utilities.size()) != B * N) return cfrnative::kStatusLengthMismatch;
  if (static_cast<int>(chance_weights.size()) != B * E) return cfrnative::kStatusLengthMismatch;

  // Output buffers
  std::vector<float> out_strategies(B * S);
  std::vector<float> out_ev(B);

  cfrbatch::BatchSpec spec;
  spec.root_node_id = static_cast<int>(rootNodeId);
  spec.node_count = N;
  spec.edge_count = E;
  spec.infoset_count = I;
  spec.strategy_size = S;
  spec.iterations = static_cast<int>(iterations);
  spec.averaging_delay = static_cast<int>(averagingDelay);
  spec.cfr_plus = (cfrPlus == JNI_TRUE);
  spec.linear_averaging = (linearAveraging == JNI_TRUE);
  spec.batch_size = B;

  int result = cfrbatch::launch_batch_solve(
      spec,
      node_types.data(), node_starts.data(), node_counts.data(),
      node_infosets.data(), edge_child_ids.data(),
      infoset_action_counts.data(), infoset_offsets.data(),
      terminal_utilities.data(), chance_weights.data(),
      out_strategies.data(), out_ev.data());

  if (result != 0) return -1; // CUDA error

  // Write back
  status = write_float_array(env, outAverageStrategiesArray, out_strategies);
  if (status != cfrnative::kStatusOk) return status;
  status = write_float_array(env, outExpectedValuesArray, out_ev);
  if (status != cfrnative::kStatusOk) return status;

  g_last_engine_code.store(kEngineGpu, std::memory_order_relaxed);
  return cfrnative::kStatusOk;
}
```

- [ ] **Step 3: Rebuild the CUDA DLL**

```powershell
.\src\main\native\build-windows-cuda11.ps1
```
Expected: `Built: ...\sicfun_cfr_cuda.dll` with no compile errors.

- [ ] **Step 4: Commit**

```bash
git add src/main/java/sicfun/holdem/HoldemCfrNativeGpuBindings.java
git add src/main/native/jni/HoldemCfrNativeGpuBindings.cu
git commit -m "feat: add JNI bridge for batch CFR CUDA kernel

solveTreeBatch accepts shared topology + batched per-tree
terminal utilities and chance weights as float arrays."
```

---

## Chunk 2: Scala Integration + Parity Tests

### Task 3: Add Scala runtime wrapper for batch solve

**Files:**
- Modify: `src/main/scala/sicfun/holdem/cfr/HoldemCfrNativeRuntime.scala`

- [ ] **Step 1: Add `BatchTreeSpec` and `BatchSolveResult` types**

Add after the existing `NativeSolveResult` case class:

```scala
final case class BatchTreeSpec(
    rootNodeId: Int,
    nodeTypes: Array[Int],
    nodeStarts: Array[Int],
    nodeCounts: Array[Int],
    nodeInfosets: Array[Int],
    edgeChildIds: Array[Int],
    infosetPlayers: Array[Int],
    infosetActionCounts: Array[Int],
    infosetOffsets: Array[Int],
    // Per-tree data concatenated: [tree0, tree1, ..., treeN]
    terminalUtilities: Array[Float],
    chanceWeights: Array[Float],
    batchSize: Int
):
  require(batchSize > 0, "batchSize must be positive")
  val nodeCount: Int = nodeTypes.length
  val edgeCount: Int = edgeChildIds.length
  val infosetCount: Int = infosetActionCounts.length
  val strategySize: Int = infosetOffsets.last
  require(terminalUtilities.length == batchSize * nodeCount, "terminal utilities size mismatch")
  require(chanceWeights.length == batchSize * edgeCount, "chance weights size mismatch")

/** NOTE: expectedValues are NOT computed by the current kernel (always 0.0f).
  * EV computation requires an additional tree walk after the CFR loop, which
  * is deferred to a follow-up. For decision policies, only strategies are needed.
  */
final case class BatchSolveResult(
    averageStrategiesFlattened: Array[Float],  // [batchSize * strategySize]
    expectedValues: Array[Float],               // [batchSize] -- currently always 0
    batchSize: Int,
    strategySize: Int
):
  def strategiesForTree(treeIdx: Int): Array[Float] =
    val offset = treeIdx * strategySize
    java.util.Arrays.copyOfRange(averageStrategiesFlattened, offset, offset + strategySize)
```

- [ ] **Step 2: Add `solveTreeBatch` method**

Add to `HoldemCfrNativeRuntime`:

```scala
def solveTreeBatch(
    spec: BatchTreeSpec,
    config: CfrSolver.Config
): Either[String, BatchSolveResult] =
  gpuLoadResult() match
    case Left(reason) => Left(reason)
    case Right(_) =>
      try
        val outStrategies = new Array[Float](spec.batchSize * spec.strategySize)
        val outEv = new Array[Float](spec.batchSize)
        val status = HoldemCfrNativeGpuBindings.solveTreeBatch(
          config.iterations,
          config.averagingDelay,
          config.cfrPlus,
          config.linearAveraging,
          spec.rootNodeId,
          spec.nodeTypes,
          spec.nodeStarts,
          spec.nodeCounts,
          spec.nodeInfosets,
          spec.edgeChildIds,
          spec.infosetPlayers,
          spec.infosetActionCounts,
          spec.infosetOffsets,
          spec.terminalUtilities,
          spec.chanceWeights,
          outStrategies,
          outEv,
          spec.batchSize
        )
        if status != 0 then Left(describeStatus(status))
        else Right(BatchSolveResult(
          averageStrategiesFlattened = outStrategies,
          expectedValues = outEv,
          batchSize = spec.batchSize,
          strategySize = spec.strategySize
        ))
      catch
        case ex: UnsatisfiedLinkError =>
          Left(s"GPU batch CFR symbols not found: ${ex.getMessage}")
        case ex: Throwable =>
          Left(Option(ex.getMessage).map(_.trim).filter(_.nonEmpty)
            .getOrElse(ex.getClass.getSimpleName))
```

- [ ] **Step 3: Verify compilation**

```bash
sbt compile
```
Expected: compiles successfully.

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/cfr/HoldemCfrNativeRuntime.scala
git commit -m "feat: add Scala runtime wrapper for batch CFR GPU solve"
```

---

### Task 4: Add batch tree builder to HoldemCfrSolver

**Files:**
- Modify: `src/main/scala/sicfun/holdem/cfr/HoldemCfrSolver.scala`

The batch builder takes one `NativeTreeSpec` as the topology template and builds per-tree
arrays for a set of hero hands (which differ only in equities and villain weights).

- [ ] **Step 1: Add `buildBatchTreeSpec` to `HoldemDecisionGame` inner class**

This method takes a list of single-tree specs (all sharing the same topology) and packs them
into one `BatchTreeSpec`. Add inside `HoldemCfrSolver`:

```scala
private def buildBatchTreeSpec(
    specs: IndexedSeq[HoldemCfrNativeRuntime.NativeTreeSpec],
    config: CfrSolver.Config
): HoldemCfrNativeRuntime.BatchTreeSpec =
  require(specs.nonEmpty, "batch must have at least one tree")
  val template = specs.head
  val nodeCount = template.nodeTypes.length
  val edgeCount = template.edgeChildIds.length
  val infosetCount = template.infosetActionCounts.length

  // Build infoset offsets
  val infosetOffsets = new Array[Int](infosetCount + 1)
  var offset = 0
  var is = 0
  while is < infosetCount do
    infosetOffsets(is) = offset
    offset += template.infosetActionCounts(is)
    is += 1
  infosetOffsets(infosetCount) = offset
  val strategySize = offset

  // Validate all specs share identical topology (not just sizes)
  val batchSize = specs.length
  specs.indices.foreach { i =>
    val s = specs(i)
    require(s.nodeTypes.length == nodeCount, s"tree $i nodeCount mismatch")
    require(s.edgeChildIds.length == edgeCount, s"tree $i edgeCount mismatch")
    require(java.util.Arrays.equals(s.nodeTypes, template.nodeTypes), s"tree $i nodeTypes differ")
    require(java.util.Arrays.equals(s.nodeStarts, template.nodeStarts), s"tree $i nodeStarts differ")
    require(java.util.Arrays.equals(s.nodeCounts, template.nodeCounts), s"tree $i nodeCounts differ")
    require(java.util.Arrays.equals(s.nodeInfosets, template.nodeInfosets), s"tree $i nodeInfosets differ")
    require(java.util.Arrays.equals(s.edgeChildIds, template.edgeChildIds), s"tree $i edgeChildIds differ")
    require(java.util.Arrays.equals(s.infosetActionCounts, template.infosetActionCounts),
      s"tree $i infosetActionCounts differ")
  }

  // Pack per-tree data
  val terminalUtilities = new Array[Float](batchSize * nodeCount)
  val chanceWeights = new Array[Float](batchSize * edgeCount)

  var treeIdx = 0
  while treeIdx < batchSize do
    val s = specs(treeIdx)
    val tuBase = treeIdx * nodeCount
    val cwBase = treeIdx * edgeCount

    // Convert double terminal utilities to float
    var n = 0
    while n < nodeCount do
      terminalUtilities(tuBase + n) = s.terminalUtilities(n).toFloat
      n += 1

    // Normalize chance weights to float (same logic as CPU solver)
    var node = 0
    while node < nodeCount do
      if s.nodeTypes(node) == 1 then // chance
        val start = s.nodeStarts(node)
        val count = s.nodeCounts(node)
        var probSum = 0.0
        var e = start
        while e < start + count do
          probSum += s.edgeProbabilities(e)
          e += 1
        val inv = if probSum > 0.0 then 1.0 / probSum else 0.0
        e = start
        while e < start + count do
          chanceWeights(cwBase + e) = (s.edgeProbabilities(e) * inv).toFloat
          e += 1
      node += 1
    treeIdx += 1

  HoldemCfrNativeRuntime.BatchTreeSpec(
    rootNodeId = template.rootNodeId,
    nodeTypes = template.nodeTypes,
    nodeStarts = template.nodeStarts,
    nodeCounts = template.nodeCounts,
    nodeInfosets = template.nodeInfosets,
    edgeChildIds = template.edgeChildIds,
    infosetPlayers = template.infosetPlayers,
    infosetActionCounts = template.infosetActionCounts,
    infosetOffsets = infosetOffsets,
    terminalUtilities = terminalUtilities,
    chanceWeights = chanceWeights,
    batchSize = batchSize
  )
```

- [ ] **Step 2: Add public `solveBatchDecisionPolicies` method**

```scala
def solveBatchDecisionPolicies(
    heroHands: IndexedSeq[HoleCards],
    state: GameState,
    villainPosterior: DiscreteDistribution[HoleCards],
    candidateActions: Vector[PokerAction],
    config: HoldemCfrConfig = HoldemCfrConfig()
): IndexedSeq[(HoleCards, HoldemCfrDecisionPolicy)] =
  require(heroHands.nonEmpty, "heroHands must be non-empty")

  // Build individual tree specs (reuses existing prepareGame + toNativeTreeSpec)
  val preparedSpecs = heroHands.map { hero =>
    val prepared = prepareGame(
      hero = hero,
      state = state,
      villainPosterior = villainPosterior,
      candidateActions = candidateActions,
      config = config
    )
    (hero, prepared)
  }

  // Try GPU batch path
  val batchResult = tryGpuBatch(preparedSpecs, config)
  batchResult match
    case Right(results) => results
    case Left(_) =>
      // Fallback: sequential single-tree solve
      preparedSpecs.map { case (hero, prepared) =>
        val policy = solveDecisionPolicy(
          hero = hero,
          state = state,
          villainPosterior = villainPosterior,
          candidateActions = candidateActions,
          config = config
        )
        (hero, policy)
      }

private def tryGpuBatch(
    preparedSpecs: IndexedSeq[(HoleCards, PreparedGame)],
    config: HoldemCfrConfig
): Either[String, IndexedSeq[(HoleCards, HoldemCfrDecisionPolicy)]] =
  val gpuAvail = HoldemCfrNativeRuntime.availability(HoldemCfrNativeRuntime.Backend.Gpu)
  if !gpuAvail.available then return Left("GPU not available")

  val nativeSpecs = preparedSpecs.map { case (hero, prepared) =>
    (hero, prepared, prepared.game.toNativeTreeSpec)
  }

  val cfrConfig = CfrSolver.Config(
    iterations = config.iterations,
    cfrPlus = config.cfrPlus,
    averagingDelay = config.averagingDelay,
    linearAveraging = config.linearAveraging
  )

  val batchSpec = buildBatchTreeSpec(nativeSpecs.map(_._3), cfrConfig)
  HoldemCfrNativeRuntime.solveTreeBatch(batchSpec, cfrConfig) match
    case Left(error) => Left(error)
    case Right(result) =>
      val policies = nativeSpecs.indices.map { i =>
        val (hero, prepared, spec) = nativeSpecs(i)
        val strategies = result.strategiesForTree(i)
        val rootInfosetStart = batchSpec.infosetOffsets(spec.rootInfoSetIndex)
        val heroActions = prepared.heroActions
        val actionProbs = heroActions.indices.map { a =>
          heroActions(a) -> strategies(rootInfosetStart + a).toDouble
        }.toMap
        val normalizedProbs = normalizedPolicyForActions(heroActions, actionProbs)
        val bestAction = normalizedProbs.maxBy(_._2)._1
        val policy = HoldemCfrDecisionPolicy(
          actionProbabilities = normalizedProbs,
          bestAction = bestAction,
          iterations = config.iterations,
          infoSetKey = spec.infosetKeys.head,
          villainSupport = prepared.villainSupport.length,
          provider = "batch-gpu"
        )
        (hero, policy)
      }
      Right(policies)
```

- [ ] **Step 3: Verify compilation**

```bash
sbt compile
```

- [ ] **Step 4: Commit**

```bash
git add src/main/scala/sicfun/holdem/cfr/HoldemCfrSolver.scala
git commit -m "feat: add solveBatchDecisionPolicies with GPU batch path + CPU fallback"
```

---

### Task 5: Parity tests — batch GPU vs sequential CPU

**Files:**
- Create: `src/test/scala/sicfun/holdem/cfr/HoldemCfrBatchSolverTest.scala`

These tests verify that the batch GPU solver produces the same decision policies
as the existing single-tree CPU solver (within float precision tolerance).

- [ ] **Step 1: Write the test suite skeleton**

```scala
package sicfun.holdem.cfr

import munit.FunSuite
import sicfun.core.{Card, Deck, DiscreteDistribution}
import sicfun.holdem.types.*
import sicfun.holdem.equity.{HoldemEquity, HoldemCombinator}

class HoldemCfrBatchSolverTest extends FunSuite:

  private val GpuAvailable: Boolean =
    HoldemCfrNativeRuntime.availability(HoldemCfrNativeRuntime.Backend.Gpu).available

  private val StrategyTolerance = 0.02 // float vs double rounding

  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private val PreflopState: GameState = GameState(
    street = Street.Preflop,
    board = Board.empty,
    pot = 6.0,
    toCall = 2.0,
    position = Position.Button,
    stackSize = 100.0,
    betHistory = Vector.empty
  )

  private def uniformPosteriorExcluding(hero: HoleCards): DiscreteDistribution[HoleCards] =
    val remaining = Deck.full.filterNot(c => c == hero.first || c == hero.second)
    DiscreteDistribution.uniform(HoldemCombinator.holeCardsFrom(remaining))

  override def munitTestTransforms: List[TestTransform] = super.munitTestTransforms ++
    List(new TestTransform("gpu-only", { test =>
      if GpuAvailable then test
      else test.tag(munit.Ignore)
    }))
```

- [ ] **Step 2: Add single-tree parity test**

```scala
  test("batch-of-1 matches single-tree CPU solve".tag(munit.Slow)) {
    val hero = hole("As", "Ks")
    val state = PreflopState
    val posterior = uniformPosteriorExcluding(hero)
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(6.0))
    val config = HoldemCfrConfig(iterations = 500, maxVillainHands = 48)

    val singleResult = HoldemCfrSolver.solveDecisionPolicy(
      hero = hero, state = state,
      villainPosterior = posterior,
      candidateActions = actions, config = config
    )

    val batchResults = HoldemCfrSolver.solveBatchDecisionPolicies(
      heroHands = IndexedSeq(hero), state = state,
      villainPosterior = posterior,
      candidateActions = actions, config = config
    )

    assertEquals(batchResults.length, 1)
    val (batchHero, batchPolicy) = batchResults.head
    assertEquals(batchHero, hero)
    assertEquals(batchPolicy.bestAction, singleResult.bestAction)

    actions.foreach { action =>
      val singleProb = singleResult.actionProbabilities.getOrElse(action, 0.0)
      val batchProb = batchPolicy.actionProbabilities.getOrElse(action, 0.0)
      assertEqualsDouble(batchProb, singleProb, StrategyTolerance,
        s"action $action: batch=$batchProb vs single=$singleProb")
    }
  }
```

- [ ] **Step 3: Add multi-tree parity test**

```scala
  test("batch-of-10 matches sequential single-tree solves".tag(munit.Slow)) {
    val heroes = IndexedSeq(
      hole("As", "Ks"), hole("Ah", "Kh"),
      hole("Qs", "Qh"), hole("Jc", "Tc"),
      hole("9s", "8s"), hole("7h", "6h"),
      hole("5d", "4d"), hole("3c", "2c"),
      hole("Ac", "Qd"), hole("Kd", "Js")
    )
    val state = PreflopState
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(6.0))
    val config = HoldemCfrConfig(iterations = 500, maxVillainHands = 48)

    val batchResults = HoldemCfrSolver.solveBatchDecisionPolicies(
      heroHands = heroes, state = state,
      villainPosterior = DiscreteDistribution.uniform(/* all hole cards */),
      candidateActions = actions, config = config
    )

    assertEquals(batchResults.length, heroes.length)

    heroes.zip(batchResults).foreach { case (hero, (batchHero, batchPolicy)) =>
      assertEquals(batchHero, hero)
      val singleResult = HoldemCfrSolver.solveDecisionPolicy(
        hero = hero, state = state,
        villainPosterior = uniformPosteriorExcluding(hero),
        candidateActions = actions, config = config
      )
      assertEquals(batchPolicy.bestAction, singleResult.bestAction,
        s"hero $hero: best action mismatch")
    }
  }
```

- [ ] **Step 4: Run tests**

```bash
sbt "testOnly sicfun.holdem.cfr.HoldemCfrBatchSolverTest"
```
Expected: All tests pass (or skip if no GPU). If parity fails, debug the kernel
by comparing per-tree outputs between batch GPU and single-tree CPU.

- [ ] **Step 5: Commit**

```bash
git add src/test/scala/sicfun/holdem/cfr/HoldemCfrBatchSolverTest.scala
git commit -m "test: add batch CFR GPU vs CPU parity tests"
```

---

## Chunk 3: Benchmark + Cleanup

### Task 6: Throughput benchmark

**Files:**
- Create: `src/main/scala/sicfun/holdem/bench/HoldemCfrBatchBenchmark.scala`

- [ ] **Step 1: Write the benchmark**

```scala
package sicfun.holdem.bench

import sicfun.core.{Card, Deck, DiscreteDistribution}
import sicfun.holdem.types.*
import sicfun.holdem.cfr.*
import sicfun.holdem.equity.{HoldemEquity, HoldemCombinator}

/** Measures batch CFR GPU throughput vs sequential CPU baseline. */
object HoldemCfrBatchBenchmark:
  def main(args: Array[String]): Unit =
    val batchSize = if args.length > 0 then args(0).toInt else 100
    val iterations = if args.length > 1 then args(1).toInt else 1500
    val warmup = 3
    val runs = 10

    val state = GameState(
      street = Street.Preflop, board = Board.empty,
      pot = 6.0, toCall = 2.0, position = Position.Button,
      stackSize = 100.0, betHistory = Vector.empty
    )
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(6.0))
    val allHeroes = HoldemCombinator.holeCardsFrom(Deck.full).take(batchSize)
    val config = HoldemCfrConfig(iterations = iterations, maxVillainHands = 48)

    println(s"Batch CFR benchmark: $batchSize trees, $iterations iterations")
    println(s"GPU available: ${HoldemCfrNativeRuntime.availability(HoldemCfrNativeRuntime.Backend.Gpu)}")

    // Warmup
    for _ <- 0 until warmup do
      HoldemCfrSolver.solveBatchDecisionPolicies(
        heroHands = allHeroes, state = state,
        villainPosterior = DiscreteDistribution.uniform(HoldemCombinator.holeCardsFrom(Deck.full)),
        candidateActions = actions, config = config
      )

    // Timed runs
    val times = (0 until runs).map { _ =>
      val start = System.nanoTime()
      HoldemCfrSolver.solveBatchDecisionPolicies(
        heroHands = allHeroes, state = state,
        villainPosterior = DiscreteDistribution.uniform(HoldemCombinator.holeCardsFrom(Deck.full)),
        candidateActions = actions, config = config
      )
      (System.nanoTime() - start) / 1e6
    }.sorted

    println(s"\n=== BATCH RESULTS (median of $runs) ===")
    println(f"Batch ${batchSize} trees: ${times(runs / 2)}%.1f ms")
    println(f"Per tree: ${times(runs / 2) / batchSize}%.3f ms")

    // Sequential baseline
    val seqTimes = (0 until math.min(runs, 3)).map { _ =>
      val start = System.nanoTime()
      allHeroes.foreach { hero =>
        HoldemCfrSolver.solveDecisionPolicy(
          hero = hero, state = state,
          villainPosterior = DiscreteDistribution.uniform(HoldemCombinator.holeCardsFrom(Deck.full)),
          candidateActions = actions, config = config
        )
      }
      (System.nanoTime() - start) / 1e6
    }.sorted

    println(f"\nSequential ${batchSize} trees: ${seqTimes(seqTimes.length / 2)}%.1f ms")
    println(f"Speedup: ${seqTimes(seqTimes.length / 2) / times(runs / 2)}%.1fx")
```

- [ ] **Step 2: Run the benchmark**

```bash
sbt "runMain sicfun.holdem.bench.HoldemCfrBatchBenchmark 100 1500"
```

Expected output (approximate):
```
Batch 100 trees: ~20 ms
Sequential 100 trees: ~500 ms
Speedup: ~25x
```

Note: The 960M will show lower speedup than a T4 due to fewer SMs and lower clock.
The real payoff is on server hardware.

- [ ] **Step 3: Commit**

```bash
git add src/main/scala/sicfun/holdem/bench/HoldemCfrBatchBenchmark.scala
git commit -m "bench: add batch CFR GPU throughput benchmark"
```

---

### Task 7: Clean up dead code from benchmarking experiments

**Files:**
- Modify: `src/main/native/jni/CfrNativeSolverCore.hpp`

- [ ] **Step 1: Remove dead `solve_float` function**

Remove lines 425-427:

```cpp
inline int solve_float(const TreeSpec& spec, SolveOutput& output) {
  return solve_impl<float>(spec, output);
}
```

This was added during the CPU float-vs-double benchmark experiment and is never called.

- [ ] **Step 2: Verify build still works**

```powershell
.\src\main\native\build-windows-cuda11.ps1
```

- [ ] **Step 3: Commit**

```bash
git add src/main/native/jni/CfrNativeSolverCore.hpp
git commit -m "chore: remove dead solve_float from CfrNativeSolverCore"
```

---

## Key Risks and Mitigations

1. **Float precision vs double**: The batch kernel uses float32 internally while the CPU solver uses float64. Strategy outputs may differ by ~1-2%. The parity tests use a 0.02 tolerance. If parity is worse, consider: (a) using double for the final strategy extraction only, or (b) accepting the tolerance for batch workloads.

2. **Windows TDR timeout**: With 1326 trees × 1500 iterations on the 960M (5 SMs), the kernel may exceed the 2-second TDR limit. Mitigations: (a) reduce batch size for dev laptop, (b) split into multiple kernel launches, (c) increase TDR timeout via registry. The benchmark should detect this (CUDA error on synchronize).

3. **Register pressure**: The `Frame` struct with `strategy[8]` and `action_values[8]` arrays uses ~100 bytes per stack entry × 8 depth = 800 bytes in registers per thread. nvcc may spill to local memory. If performance is poor, reduce `kMaxDepth` to 5 (still safe for depth-4 trees) and `kMaxInlineActions` to 5 (covers all current Hold'em action counts).

4. **GPU memory**: 1326 trees × 7KB working memory = 9MB. Plus topology arrays (~50KB shared). Total ~10MB. Well within any CUDA GPU's memory.

5. **EV computation**: The kernel currently doesn't compute per-tree EV (left as 0). This can be added in a follow-up by doing one more tree walk after the iteration loop using the output strategies. For decision policies, EV is not strictly required.
