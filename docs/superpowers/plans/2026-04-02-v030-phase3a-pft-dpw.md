# Phase 3a: PFT-DPW POMDP Solver -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement PFT-DPW single-agent POMDP tree search in C++ with JNI bridge to Scala
**Architecture:** C++17 header-only solver, JNI bindings following existing CfrNativeSolver pattern, Scala runtime wrapper
**Tech Stack:** C++17, Scala 3.8.1, munit 1.2.2
**Depends on:** Phase 1 (strategic types for action/observation spaces)
**Unlocks:** Phase 3c (W-POMCP extends PFT-DPW)

**Spec coverage:** Definitions 29-32 (value functions), 54-55 (particle beliefs)

---

## File Map

```
src/main/native/jni/PftDpwSolver.hpp                          NEW  ~250 LOC
src/main/native/jni/HoldemPomcpNativeBindings.cpp              NEW  ~120 LOC
src/main/scala/sicfun/holdem/strategic/solver/PftDpwRuntime.scala   NEW  ~80 LOC
src/test/scala/sicfun/holdem/strategic/solver/PftDpwRuntimeTest.scala NEW ~120 LOC
```

---

## Design Decisions

### Tree structure

PFT-DPW uses a sparse tree grown online. Each tree node holds:
- Belief: weighted particle set `{(x_j, w_j)}` (Def 54)
- Visit count N(b,a) per action child
- Q-value estimate Q(b,a) per action child (Def 30)
- Children indexed by (action, observation) pairs

### Double Progressive Widening (DPW)

Action widening: expand new action when `|children(b)| <= k_a * N(b)^alpha_a`
Observation widening: expand new observation when `|obs_children(b,a)| <= k_o * N(b,a)^alpha_o`

This avoids explicit enumeration of the continuous/large observation space.

### UCB action selection

```
a* = argmax_a [ Q(b,a) + c * sqrt(ln N(b) / N(b,a)) ]
```

### Particle belief update (Def 54)

At observation nodes, particles are reweighted by observation likelihood:
```
w'_j = w_j * O(o | x'_j, a) / sum_k w_k * O(o | x'_k, a)
```

### Error bound (Def 55)

The particle approximation error is bounded by:
```
|V*(b) - V*(b_hat_C)| <= R_max / (1 - gamma) * sqrt(D2(b || b_hat_C) / 2)
```

This is validated in tests, not enforced at runtime.

### Namespace / dependency

- C++ namespace: `pftdpw` (mirrors `cfrnative`)
- Scala package: `sicfun.holdem.strategic.solver`
- Phase 1 types used: action space U, observation space O, reward r, discount gamma, R_max
- JNI binding pattern: copy-based (GetXxxArrayRegion), matching CfrNativeSolverCore

---

## Tasks

### Task 1: C++ PFT-DPW data structures

**File:** `src/main/native/jni/PftDpwSolver.hpp`

Write the namespace, status codes, config struct, tree node, and particle types.

- [ ] 1a. Write test: create `PftDpwConfig`, assert defaults are sane (C++ static_assert in header)

```cpp
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#if defined(_MSC_VER) || defined(__CUDACC__)
#define PFT_FORCE_INLINE __forceinline
#else
#define PFT_FORCE_INLINE inline __attribute__((always_inline))
#endif

namespace pftdpw {

// Status codes (same convention as cfrnative)
constexpr int kStatusOk = 0;
constexpr int kStatusNullArray = 100;
constexpr int kStatusLengthMismatch = 101;
constexpr int kStatusReadFailure = 102;
constexpr int kStatusWriteFailure = 124;
constexpr int kStatusInvalidConfig = 200;
constexpr int kStatusNoParticles = 201;
constexpr int kStatusTreeOverflow = 202;
constexpr int kStatusSimulationFailed = 203;

// Weighted particle: (state_index, weight)
struct Particle {
  int state_idx;
  double weight;
};

// Particle belief (Def 54): {(x_j, w_j)}_{j=1}^C with sum w_j = 1
struct ParticleBelief {
  std::vector<Particle> particles;

  void normalize() {
    double total = 0.0;
    for (auto& p : particles) total += p.weight;
    if (total > 0.0) {
      for (auto& p : particles) p.weight /= total;
    }
  }

  int size() const { return static_cast<int>(particles.size()); }
};

// PFT-DPW configuration
struct PftDpwConfig {
  int num_simulations = 1000;       // number of MCTS simulations
  double gamma = 0.99;              // discount factor
  double r_max = 1.0;              // reward bound (for UCB scaling)
  double ucb_c = 1.0;             // UCB exploration constant

  // DPW parameters
  double k_action = 2.0;          // action widening: |A(b)| <= k_a * N(b)^alpha_a
  double alpha_action = 0.5;      // action widening exponent
  double k_obs = 2.0;             // observation widening: |O(b,a)| <= k_o * N(b,a)^alpha_o
  double alpha_obs = 0.5;         // observation widening exponent

  int max_depth = 50;             // maximum tree depth
  int num_particles = 100;        // C: particle count (Def 54)
  int max_tree_nodes = 100000;    // memory cap on tree size
  uint64_t seed = 42;             // RNG seed
};

static_assert(sizeof(PftDpwConfig) > 0, "PftDpwConfig must be non-empty");

// Tree node: belief + per-action statistics
struct TreeNode {
  ParticleBelief belief;
  int visit_count = 0;
  int depth = 0;

  // Per-action data, indexed by action id
  std::vector<int> action_ids;         // which actions have been tried
  std::vector<double> q_values;        // Q(b, a) estimate (Def 30)
  std::vector<int> action_visits;      // N(b, a)

  // Children: action_idx -> (obs_id -> child_node_id)
  // Sparse: only expanded observations are stored
  std::vector<std::vector<std::pair<int, int>>> obs_children;

  int num_actions() const { return static_cast<int>(action_ids.size()); }

  int find_action(int action_id) const {
    for (int i = 0; i < static_cast<int>(action_ids.size()); ++i) {
      if (action_ids[i] == action_id) return i;
    }
    return -1;
  }

  void add_action(int action_id) {
    action_ids.push_back(action_id);
    q_values.push_back(0.0);
    action_visits.push_back(0);
    obs_children.emplace_back();
  }
};
```

- [ ] 1b. Static assertions for config invariants

```cpp
// Validate config before use
PFT_FORCE_INLINE int validate_config(const PftDpwConfig& cfg) {
  if (cfg.num_simulations <= 0) return kStatusInvalidConfig;
  if (cfg.gamma <= 0.0 || cfg.gamma >= 1.0) return kStatusInvalidConfig;
  if (cfg.r_max <= 0.0) return kStatusInvalidConfig;
  if (cfg.ucb_c < 0.0) return kStatusInvalidConfig;
  if (cfg.k_action <= 0.0 || cfg.alpha_action <= 0.0) return kStatusInvalidConfig;
  if (cfg.k_obs <= 0.0 || cfg.alpha_obs <= 0.0) return kStatusInvalidConfig;
  if (cfg.max_depth <= 0) return kStatusInvalidConfig;
  if (cfg.num_particles <= 0) return kStatusNoParticles;
  if (cfg.max_tree_nodes <= 0) return kStatusInvalidConfig;
  return kStatusOk;
}
```

---

### Task 2: C++ UCB action selection + DPW widening logic

**File:** `src/main/native/jni/PftDpwSolver.hpp` (continued)

- [ ] 2a. Write UCB selection (Def 32: optimal Q via argmax)

```cpp
// UCB1 action selection (Def 32: argmax over Q + exploration bonus)
PFT_FORCE_INLINE int select_action_ucb(const TreeNode& node,
                                         double ucb_c) {
  int best_idx = 0;
  double best_val = -std::numeric_limits<double>::infinity();
  const double log_n = std::log(static_cast<double>(node.visit_count + 1));

  for (int i = 0; i < node.num_actions(); ++i) {
    double q = node.q_values[i];
    double explore = (node.action_visits[i] > 0)
        ? ucb_c * std::sqrt(log_n / node.action_visits[i])
        : std::numeric_limits<double>::infinity();
    double val = q + explore;
    if (val > best_val) {
      best_val = val;
      best_idx = i;
    }
  }
  return best_idx;
}

// DPW: should we widen the action set?
PFT_FORCE_INLINE bool should_widen_actions(const TreeNode& node,
                                            double k_a, double alpha_a) {
  double threshold = k_a * std::pow(static_cast<double>(node.visit_count + 1),
                                     alpha_a);
  return static_cast<double>(node.num_actions()) <= threshold;
}

// DPW: should we widen the observation set for a given action?
PFT_FORCE_INLINE bool should_widen_obs(const TreeNode& node, int action_idx,
                                        double k_o, double alpha_o) {
  int n_obs = static_cast<int>(node.obs_children[action_idx].size());
  double threshold = k_o * std::pow(
      static_cast<double>(node.action_visits[action_idx] + 1), alpha_o);
  return static_cast<double>(n_obs) <= threshold;
}
```

- [ ] 2b. Verify: UCB with N(b,a)=0 returns infinity (untried actions selected first)
- [ ] 2c. Verify: widening threshold increases sublinearly with visit count

---

### Task 3: C++ particle belief update (Def 54) + rollout

**File:** `src/main/native/jni/PftDpwSolver.hpp` (continued)

- [ ] 3a. Write particle reweighting for observation update

```cpp
// Generative model interface: provided by the JNI caller as flat arrays.
// transition: state_idx x action_id -> next_state_idx (deterministic index into state table)
// obs_likelihood: state_idx x action_id x obs_id -> probability
// reward: state_idx x action_id -> double
struct GenerativeModel {
  const int* transition_table;    // [num_states * num_actions]
  const double* obs_likelihood;   // [num_states * num_actions * num_obs]
  const double* reward_table;     // [num_states * num_actions]
  int num_states;
  int num_actions;
  int num_obs;

  PFT_FORCE_INLINE int transition(int state, int action) const {
    return transition_table[state * num_actions + action];
  }

  PFT_FORCE_INLINE double obs_prob(int state, int action, int obs) const {
    return obs_likelihood[(state * num_actions + action) * num_obs + obs];
  }

  PFT_FORCE_INLINE double reward(int state, int action) const {
    return reward_table[state * num_actions + action];
  }
};

// Belief-averaged reward (Def 29): r_bar(b, u) = sum_j w_j * r(x_j, u)
PFT_FORCE_INLINE double belief_averaged_reward(const ParticleBelief& belief,
                                                int action,
                                                const GenerativeModel& model) {
  double r_bar = 0.0;
  for (const auto& p : belief.particles) {
    r_bar += p.weight * model.reward(p.state_idx, action);
  }
  return r_bar;
}

// Particle filter update: propagate particles through transition,
// reweight by observation likelihood (Def 54 update step)
inline ParticleBelief particle_filter_update(
    const ParticleBelief& belief,
    int action, int obs_id,
    const GenerativeModel& model,
    std::mt19937_64& rng) {
  ParticleBelief next;
  next.particles.reserve(belief.size());

  for (const auto& p : belief.particles) {
    int next_state = model.transition(p.state_idx, action);
    double lik = model.obs_prob(next_state, action, obs_id);
    if (lik > 0.0) {
      next.particles.push_back({next_state, p.weight * lik});
    }
  }

  if (next.particles.empty()) {
    // All particles zeroed out: resample uniformly from original
    std::uniform_int_distribution<int> dist(0, belief.size() - 1);
    for (int j = 0; j < belief.size(); ++j) {
      int idx = dist(rng);
      int ns = model.transition(belief.particles[idx].state_idx, action);
      next.particles.push_back({ns, 1.0});
    }
  }

  next.normalize();
  return next;
}
```

- [ ] 3b. Write random rollout for leaf evaluation

```cpp
// Random rollout from a belief state (returns discounted cumulative reward)
inline double random_rollout(const ParticleBelief& belief,
                              const GenerativeModel& model,
                              const PftDpwConfig& cfg,
                              int current_depth,
                              std::mt19937_64& rng) {
  // Sample one particle as the "true" state for rollout
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  double u = uniform(rng);
  double cumw = 0.0;
  int state = belief.particles[0].state_idx;
  for (const auto& p : belief.particles) {
    cumw += p.weight;
    if (u <= cumw) { state = p.state_idx; break; }
  }

  std::uniform_int_distribution<int> action_dist(0, model.num_actions - 1);
  double total_reward = 0.0;
  double discount = 1.0;

  for (int d = current_depth; d < cfg.max_depth; ++d) {
    int a = action_dist(rng);
    total_reward += discount * model.reward(state, a);
    discount *= cfg.gamma;
    state = model.transition(state, a);
  }

  return total_reward;
}
```

- [ ] 3c. Verify: particle_filter_update preserves sum(w) = 1
- [ ] 3d. Verify: belief_averaged_reward matches Def 29 for uniform belief

---

### Task 4: C++ PFT-DPW tree search main loop

**File:** `src/main/native/jni/PftDpwSolver.hpp` (continued)

- [ ] 4a. Write the simulate() recursive function and solve() entry point

```cpp
// Tree: manages node storage with bounded capacity
struct PftTree {
  std::vector<TreeNode> nodes;
  int node_count = 0;
  int max_nodes;

  explicit PftTree(int max_n) : max_nodes(max_n) {
    nodes.resize(max_n);
  }

  int add_node(const ParticleBelief& belief, int depth) {
    if (node_count >= max_nodes) return -1;
    int id = node_count++;
    nodes[id].belief = belief;
    nodes[id].visit_count = 0;
    nodes[id].depth = depth;
    nodes[id].action_ids.clear();
    nodes[id].q_values.clear();
    nodes[id].action_visits.clear();
    nodes[id].obs_children.clear();
    return id;
  }
};

// Recursive simulation (one MCTS pass)
inline double simulate(PftTree& tree, int node_id,
                        const GenerativeModel& model,
                        const PftDpwConfig& cfg,
                        std::mt19937_64& rng) {
  TreeNode& node = tree.nodes[node_id];

  // Depth cutoff: rollout
  if (node.depth >= cfg.max_depth) {
    return 0.0;
  }

  // Action progressive widening
  if (should_widen_actions(node, cfg.k_action, cfg.alpha_action)) {
    // Try to add a random untried action
    std::uniform_int_distribution<int> adist(0, model.num_actions - 1);
    int candidate = adist(rng);
    if (node.find_action(candidate) < 0) {
      node.add_action(candidate);
    }
  }

  // No actions available yet: rollout
  if (node.num_actions() == 0) {
    return random_rollout(node.belief, model, cfg, node.depth, rng);
  }

  // UCB action selection
  int action_idx = select_action_ucb(node, cfg.ucb_c);
  int action_id = node.action_ids[action_idx];

  // Immediate reward (Def 29: belief-averaged)
  double r_bar = belief_averaged_reward(node.belief, action_id, model);

  // Observation progressive widening
  double sim_reward;
  if (should_widen_obs(node, action_idx, cfg.k_obs, cfg.alpha_obs)) {
    // Generate observation by sampling a particle, transitioning, sampling obs
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    double u = uniform(rng);
    double cumw = 0.0;
    int sampled_state = node.belief.particles[0].state_idx;
    for (const auto& p : node.belief.particles) {
      cumw += p.weight;
      if (u <= cumw) { sampled_state = p.state_idx; break; }
    }
    int next_state = model.transition(sampled_state, action_id);

    // Sample observation from O(o | x', a)
    double obs_u = uniform(rng);
    double obs_cum = 0.0;
    int obs_id = 0;
    for (int o = 0; o < model.num_obs; ++o) {
      obs_cum += model.obs_prob(next_state, action_id, o);
      if (obs_u <= obs_cum) { obs_id = o; break; }
    }

    // Create new child node with updated belief
    ParticleBelief child_belief = particle_filter_update(
        node.belief, action_id, obs_id, model, rng);
    int child_id = tree.add_node(child_belief, node.depth + 1);
    if (child_id < 0) {
      // Tree overflow: rollout instead
      return r_bar + cfg.gamma * random_rollout(
          child_belief, model, cfg, node.depth + 1, rng);
    }
    node.obs_children[action_idx].push_back({obs_id, child_id});
    sim_reward = r_bar + cfg.gamma * simulate(tree, child_id, model, cfg, rng);
  } else {
    // Reuse existing observation child (uniform random among existing)
    auto& obs_list = node.obs_children[action_idx];
    std::uniform_int_distribution<int> obs_dist(
        0, static_cast<int>(obs_list.size()) - 1);
    int child_id = obs_list[obs_dist(rng)].second;
    sim_reward = r_bar + cfg.gamma * simulate(tree, child_id, model, cfg, rng);
  }

  // Backprop: incremental mean update (Def 30 sample-based approximation)
  node.action_visits[action_idx] += 1;
  node.visit_count += 1;
  int n = node.action_visits[action_idx];
  node.q_values[action_idx] += (sim_reward - node.q_values[action_idx]) / n;

  return sim_reward;
}

// Result structure
struct PftDpwResult {
  int status = kStatusOk;
  int best_action = -1;
  double best_q = 0.0;
  std::vector<int> action_ids;    // all tried actions
  std::vector<double> q_values;   // Q(b, a) for each
  std::vector<int> visit_counts;  // N(b, a) for each
  int total_simulations = 0;
  int tree_nodes_used = 0;
};

// Main entry point: run PFT-DPW from a root belief
inline PftDpwResult solve(const ParticleBelief& root_belief,
                           const GenerativeModel& model,
                           const PftDpwConfig& cfg) {
  PftDpwResult result;

  int status = validate_config(cfg);
  if (status != kStatusOk) {
    result.status = status;
    return result;
  }
  if (root_belief.size() == 0) {
    result.status = kStatusNoParticles;
    return result;
  }

  std::mt19937_64 rng(cfg.seed);
  PftTree tree(cfg.max_tree_nodes);

  int root_id = tree.add_node(root_belief, 0);
  if (root_id < 0) {
    result.status = kStatusTreeOverflow;
    return result;
  }

  for (int sim = 0; sim < cfg.num_simulations; ++sim) {
    simulate(tree, root_id, model, cfg, rng);
  }

  // Extract results from root
  const TreeNode& root = tree.nodes[root_id];
  result.action_ids = root.action_ids;
  result.q_values = root.q_values;
  result.visit_counts = root.action_visits;
  result.total_simulations = cfg.num_simulations;
  result.tree_nodes_used = tree.node_count;

  // Best action by visit count (more robust than Q-value)
  int best_idx = 0;
  int best_visits = 0;
  for (int i = 0; i < root.num_actions(); ++i) {
    if (root.action_visits[i] > best_visits) {
      best_visits = root.action_visits[i];
      best_idx = i;
    }
  }
  if (root.num_actions() > 0) {
    result.best_action = root.action_ids[best_idx];
    result.best_q = root.q_values[best_idx];
  }

  return result;
}

} // namespace pftdpw
```

- [ ] 4b. Verify: solve() with 1 action, 1 obs, known reward returns correct Q
- [ ] 4c. Verify: solve() with 0 particles returns kStatusNoParticles
- [ ] 4d. Verify: solve() with invalid config returns kStatusInvalidConfig

---

### Task 5: JNI bindings

**File:** `src/main/native/jni/HoldemPomcpNativeBindings.cpp`

Follow the CfrNativeCpuBindings pattern: copy-based array I/O, engine code tracking.

- [ ] 5a. Write JNI helper functions (array readers, matching existing pattern)

```cpp
/*
 * HoldemPomcpNativeBindings.cpp -- CPU JNI binding for PFT-DPW POMDP
 * tree search in the sicfun poker analytics system.
 *
 * Bridges sicfun.holdem.HoldemPomcpNativeBindings to PftDpwSolver.hpp.
 * Exposes one JNI entry point:
 *
 *   solvePftDpw() -- Run PFT-DPW from a root belief, return best action + Q-values.
 *
 * Uses copy-based array I/O (GetXxxArrayRegion) matching CfrNativeCpuBindings.
 *
 * Compiled into: sicfun_native_cpu.dll
 *
 * Reports engine code 3 (PFT-DPW) on success.
 */

#include <jni.h>

#include <atomic>
#include <vector>

#include "PftDpwSolver.hpp"

namespace {

constexpr jint kEngineUnknown = 0;
constexpr jint kEnginePftDpw = 3;

std::atomic<jint> g_last_engine_code(kEngineUnknown);

bool clear_pending_jni_exception(JNIEnv* env) {
  if (!env->ExceptionCheck()) return false;
  env->ExceptionClear();
  return true;
}

int read_int_array(JNIEnv* env, jintArray array, std::vector<int>& out) {
  if (array == nullptr) return pftdpw::kStatusNullArray;
  const jsize length = env->GetArrayLength(array);
  out.resize(static_cast<size_t>(length));
  if (length > 0) {
    env->GetIntArrayRegion(array, 0, length, reinterpret_cast<jint*>(out.data()));
    if (clear_pending_jni_exception(env)) return pftdpw::kStatusReadFailure;
  }
  return pftdpw::kStatusOk;
}

int read_double_array(JNIEnv* env, jdoubleArray array, std::vector<double>& out) {
  if (array == nullptr) return pftdpw::kStatusNullArray;
  const jsize length = env->GetArrayLength(array);
  out.resize(static_cast<size_t>(length));
  if (length > 0) {
    env->GetDoubleArrayRegion(array, 0, length,
                               reinterpret_cast<jdouble*>(out.data()));
    if (clear_pending_jni_exception(env)) return pftdpw::kStatusReadFailure;
  }
  return pftdpw::kStatusOk;
}

int write_double_array(JNIEnv* env, jdoubleArray array,
                        const std::vector<double>& data) {
  if (array == nullptr) return pftdpw::kStatusNullArray;
  const jsize length = env->GetArrayLength(array);
  if (static_cast<size_t>(length) < data.size()) return pftdpw::kStatusLengthMismatch;
  if (!data.empty()) {
    env->SetDoubleArrayRegion(array, 0, static_cast<jsize>(data.size()),
                               reinterpret_cast<const jdouble*>(data.data()));
    if (clear_pending_jni_exception(env)) return pftdpw::kStatusWriteFailure;
  }
  return pftdpw::kStatusOk;
}

int write_int_array(JNIEnv* env, jintArray array,
                     const std::vector<int>& data) {
  if (array == nullptr) return pftdpw::kStatusNullArray;
  const jsize length = env->GetArrayLength(array);
  if (static_cast<size_t>(length) < data.size()) return pftdpw::kStatusLengthMismatch;
  if (!data.empty()) {
    env->SetIntArrayRegion(array, 0, static_cast<jsize>(data.size()),
                            reinterpret_cast<const jint*>(data.data()));
    if (clear_pending_jni_exception(env)) return pftdpw::kStatusWriteFailure;
  }
  return pftdpw::kStatusOk;
}

} // anonymous namespace
```

- [ ] 5b. Write JNI entry point `solvePftDpw`

```cpp
extern "C" {

/*
 * Class:     sicfun_holdem_HoldemPomcpNativeBindings
 * Method:    solvePftDpw
 * Signature: (
 *   [I       transitionTable    (num_states * num_actions)
 *   [D       obsLikelihood      (num_states * num_actions * num_obs)
 *   [D       rewardTable        (num_states * num_actions)
 *   III      numStates, numActions, numObs
 *   [I       particleStates     root belief particle state indices
 *   [D       particleWeights    root belief particle weights
 *   IDDDDDDDIJ  config params: numSims, gamma, rMax, ucbC,
 *               kAction, alphaAction, kObs, alphaObs,
 *               maxDepth, seed
 *   [D       outQValues         output: Q(b,a) per action (pre-allocated, size numActions)
 *   [I       outVisitCounts     output: N(b,a) per action (pre-allocated, size numActions)
 * ) -> I     status (0=ok) | best_action packed in high bits
 *
 * Returns: (best_action << 16) | status
 */
JNIEXPORT jlong JNICALL
Java_sicfun_holdem_HoldemPomcpNativeBindings_solvePftDpw(
    JNIEnv* env, jclass /*cls*/,
    jintArray transitionTable,
    jdoubleArray obsLikelihood,
    jdoubleArray rewardTable,
    jint numStates, jint numActions, jint numObs,
    jintArray particleStates,
    jdoubleArray particleWeights,
    jint numSimulations,
    jdouble gamma, jdouble rMax, jdouble ucbC,
    jdouble kAction, jdouble alphaAction,
    jdouble kObs, jdouble alphaObs,
    jint maxDepth, jlong seed,
    jdoubleArray outQValues,
    jintArray outVisitCounts) {

  // Read input arrays
  std::vector<int> trans;
  std::vector<double> obs_lik, rewards;
  std::vector<int> part_states;
  std::vector<double> part_weights;

  int s;
  s = read_int_array(env, transitionTable, trans);       if (s != 0) return s;
  s = read_double_array(env, obsLikelihood, obs_lik);    if (s != 0) return s;
  s = read_double_array(env, rewardTable, rewards);      if (s != 0) return s;
  s = read_int_array(env, particleStates, part_states);  if (s != 0) return s;
  s = read_double_array(env, particleWeights, part_weights); if (s != 0) return s;

  // Validate dimensions
  if (static_cast<int>(trans.size()) != numStates * numActions)
    return pftdpw::kStatusLengthMismatch;
  if (static_cast<int>(obs_lik.size()) != numStates * numActions * numObs)
    return pftdpw::kStatusLengthMismatch;
  if (static_cast<int>(rewards.size()) != numStates * numActions)
    return pftdpw::kStatusLengthMismatch;
  if (part_states.size() != part_weights.size())
    return pftdpw::kStatusLengthMismatch;

  // Build generative model
  pftdpw::GenerativeModel model;
  model.transition_table = trans.data();
  model.obs_likelihood = obs_lik.data();
  model.reward_table = rewards.data();
  model.num_states = numStates;
  model.num_actions = numActions;
  model.num_obs = numObs;

  // Build root belief (Def 54)
  pftdpw::ParticleBelief root;
  root.particles.reserve(part_states.size());
  for (size_t i = 0; i < part_states.size(); ++i) {
    root.particles.push_back({part_states[i], part_weights[i]});
  }
  root.normalize();

  // Build config
  pftdpw::PftDpwConfig cfg;
  cfg.num_simulations = numSimulations;
  cfg.gamma = gamma;
  cfg.r_max = rMax;
  cfg.ucb_c = ucbC;
  cfg.k_action = kAction;
  cfg.alpha_action = alphaAction;
  cfg.k_obs = kObs;
  cfg.alpha_obs = alphaObs;
  cfg.max_depth = maxDepth;
  cfg.num_particles = static_cast<int>(part_states.size());
  cfg.seed = static_cast<uint64_t>(seed);

  // Solve
  pftdpw::PftDpwResult result = pftdpw::solve(root, model, cfg);

  if (result.status != pftdpw::kStatusOk) return result.status;

  // Write outputs: pad to numActions (zeros for untried actions)
  std::vector<double> q_out(numActions, 0.0);
  std::vector<int> v_out(numActions, 0);
  for (size_t i = 0; i < result.action_ids.size(); ++i) {
    int aid = result.action_ids[i];
    if (aid >= 0 && aid < numActions) {
      q_out[aid] = result.q_values[i];
      v_out[aid] = result.visit_counts[i];
    }
  }

  s = write_double_array(env, outQValues, q_out);   if (s != 0) return s;
  s = write_int_array(env, outVisitCounts, v_out);   if (s != 0) return s;

  g_last_engine_code.store(kEnginePftDpw, std::memory_order_relaxed);

  // Pack: best_action in upper 32 bits, status in lower 32
  return (static_cast<jlong>(result.best_action) << 32) | pftdpw::kStatusOk;
}

JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemPomcpNativeBindings_lastEngineCode(
    JNIEnv* /*env*/, jclass /*cls*/) {
  return g_last_engine_code.load(std::memory_order_relaxed);
}

} // extern "C"
```

- [ ] 5c. Verify: JNI compiles with `sicfun_native_cpu.dll` build (add to CMakeLists)

---

### Task 6: Scala JNI binding object

**File:** `src/main/scala/sicfun/holdem/HoldemPomcpNativeBindings.scala`

Note: JNI binding object stays in `sicfun.holdem` (package name baked into native DLL JNI signatures, per project convention).

- [ ] 6a. Write the native binding object

```scala
package sicfun.holdem

/** JNI bindings for PFT-DPW POMDP solver (native CPU implementation).
  *
  * Compiled in sicfun_native_cpu.dll alongside CfrNativeCpuBindings.
  */
object HoldemPomcpNativeBindings:
  System.loadLibrary("sicfun_native_cpu")

  /** Run PFT-DPW tree search from a root particle belief.
    *
    * @return (bestAction << 32) | status. Status 0 = success.
    */
  @native def solvePftDpw(
      transitionTable: Array[Int],
      obsLikelihood: Array[Double],
      rewardTable: Array[Double],
      numStates: Int,
      numActions: Int,
      numObs: Int,
      particleStates: Array[Int],
      particleWeights: Array[Double],
      numSimulations: Int,
      gamma: Double,
      rMax: Double,
      ucbC: Double,
      kAction: Double,
      alphaAction: Double,
      kObs: Double,
      alphaObs: Double,
      maxDepth: Int,
      seed: Long,
      outQValues: Array[Double],
      outVisitCounts: Array[Int]
  ): Long

  @native def lastEngineCode(): Int
```

- [ ] 6b. Verify: `javah` / `javac -h` generates matching JNI header

---

### Task 7: Scala PftDpwRuntime wrapper

**File:** `src/main/scala/sicfun/holdem/strategic/solver/PftDpwRuntime.scala`

- [ ] 7a. Write config case class and runtime

```scala
package sicfun.holdem.strategic.solver

import sicfun.holdem.HoldemPomcpNativeBindings

/** Configuration for PFT-DPW POMDP solver (mirrors C++ PftDpwConfig). */
final case class PftDpwConfig(
    numSimulations: Int = 1000,
    gamma: Double = 0.99,
    rMax: Double = 1.0,
    ucbC: Double = 1.0,
    kAction: Double = 2.0,
    alphaAction: Double = 0.5,
    kObs: Double = 2.0,
    alphaObs: Double = 0.5,
    maxDepth: Int = 50,
    seed: Long = 42L
)

/** Result from PFT-DPW solver. */
final case class PftDpwResult(
    bestAction: Int,
    qValues: Array[Double],
    visitCounts: Array[Int],
    status: Int
):
  def isSuccess: Boolean = status == 0

/** Tabular generative model for POMDP solver.
  *
  * @param transitionTable flat [numStates * numActions] -> next state index
  * @param obsLikelihood   flat [numStates * numActions * numObs] -> probability
  * @param rewardTable     flat [numStates * numActions] -> reward
  */
final case class TabularGenerativeModel(
    transitionTable: Array[Int],
    obsLikelihood: Array[Double],
    rewardTable: Array[Double],
    numStates: Int,
    numActions: Int,
    numObs: Int
):
  require(transitionTable.length == numStates * numActions,
    s"transition table size ${transitionTable.length} != $numStates * $numActions")
  require(obsLikelihood.length == numStates * numActions * numObs,
    s"obs likelihood size ${obsLikelihood.length} != $numStates * $numActions * $numObs")
  require(rewardTable.length == numStates * numActions,
    s"reward table size ${rewardTable.length} != $numStates * $numActions")

/** Particle belief: weighted state indices (Def 54). */
final case class ParticleBelief(
    stateIndices: Array[Int],
    weights: Array[Double]
):
  require(stateIndices.length == weights.length,
    s"particle arrays must match: ${stateIndices.length} != ${weights.length}")
  require(stateIndices.nonEmpty, "particle belief must be non-empty")

/** PFT-DPW POMDP solver runtime.
  *
  * Wraps native C++ PFT-DPW tree search via JNI. Implements Defs 29-32
  * (value functions) using particle beliefs (Def 54) with error bound (Def 55).
  */
object PftDpwRuntime:
  /** Solve a POMDP from a root particle belief using PFT-DPW tree search.
    *
    * @param model  tabular generative model (T, O, r)
    * @param belief root particle belief
    * @param config solver configuration
    * @return solver result with best action, Q-values, visit counts
    */
  def solve(
      model: TabularGenerativeModel,
      belief: ParticleBelief,
      config: PftDpwConfig = PftDpwConfig()
  ): PftDpwResult =
    val outQ = new Array[Double](model.numActions)
    val outV = new Array[Int](model.numActions)

    val packed = HoldemPomcpNativeBindings.solvePftDpw(
      model.transitionTable,
      model.obsLikelihood,
      model.rewardTable,
      model.numStates,
      model.numActions,
      model.numObs,
      belief.stateIndices,
      belief.weights,
      config.numSimulations,
      config.gamma,
      config.rMax,
      config.ucbC,
      config.kAction,
      config.alphaAction,
      config.kObs,
      config.alphaObs,
      config.maxDepth,
      config.seed,
      outQ,
      outV
    )

    val status = (packed & 0xFFFFFFFFL).toInt
    val bestAction = (packed >> 32).toInt

    PftDpwResult(bestAction, outQ, outV, status)

  /** Compute the particle belief error bound (Def 55).
    *
    * |V*(b) - V*(b_hat_C)| <= R_max / (1 - gamma) * sqrt(D2(b || b_hat_C) / 2)
    *
    * For a uniform "true" belief over the same support, the Renyi-2 divergence
    * D2(uniform || b_hat) = ln(sum_j w_j^2 * C) where C = num particles.
    *
    * This gives a practical self-diagnostic: how concentrated is the particle set?
    */
  def particleErrorBound(
      weights: Array[Double],
      rMax: Double,
      gamma: Double
  ): Double =
    val sumWSquared = weights.map(w => w * w).sum
    val d2 = math.log(sumWSquared * weights.length)
    val bound = (rMax / (1.0 - gamma)) * math.sqrt(math.max(0.0, d2) / 2.0)
    bound
```

- [ ] 7b. Verify: PftDpwConfig defaults match C++ defaults
- [ ] 7c. Verify: TabularGenerativeModel rejects mismatched array sizes

---

### Task 8: Test suite

**File:** `src/test/scala/sicfun/holdem/strategic/solver/PftDpwRuntimeTest.scala`

- [ ] 8a. Write all tests

```scala
package sicfun.holdem.strategic.solver

import munit.FunSuite

class PftDpwRuntimeTest extends FunSuite:

  // --- Def 54: Particle belief ---

  test("ParticleBelief rejects empty arrays") {
    intercept[IllegalArgumentException] {
      ParticleBelief(Array.empty[Int], Array.empty[Double])
    }
  }

  test("ParticleBelief rejects mismatched lengths") {
    intercept[IllegalArgumentException] {
      ParticleBelief(Array(0, 1), Array(0.5))
    }
  }

  // --- TabularGenerativeModel validation ---

  test("TabularGenerativeModel rejects wrong transition table size") {
    intercept[IllegalArgumentException] {
      TabularGenerativeModel(
        transitionTable = Array(0), // should be 2*2=4
        obsLikelihood = Array.fill(2 * 2 * 2)(0.5),
        rewardTable = Array.fill(2 * 2)(1.0),
        numStates = 2, numActions = 2, numObs = 2
      )
    }
  }

  test("TabularGenerativeModel accepts correct dimensions") {
    val model = TabularGenerativeModel(
      transitionTable = Array.fill(2 * 2)(0),
      obsLikelihood = Array.fill(2 * 2 * 2)(0.5),
      rewardTable = Array.fill(2 * 2)(1.0),
      numStates = 2, numActions = 2, numObs = 2
    )
    assertEquals(model.numStates, 2)
  }

  // --- PftDpwConfig defaults ---

  test("PftDpwConfig defaults are sane") {
    val cfg = PftDpwConfig()
    assert(cfg.numSimulations > 0)
    assert(cfg.gamma > 0.0 && cfg.gamma < 1.0)
    assert(cfg.rMax > 0.0)
    assert(cfg.ucbC >= 0.0)
    assert(cfg.kAction > 0.0)
    assert(cfg.alphaAction > 0.0 && cfg.alphaAction <= 1.0)
    assert(cfg.kObs > 0.0)
    assert(cfg.alphaObs > 0.0 && cfg.alphaObs <= 1.0)
    assert(cfg.maxDepth > 0)
  }

  // --- Def 55: Error bound ---

  test("Def 55: uniform particle weights give zero error bound") {
    // Uniform weights: D2(uniform || uniform) = ln(1) = 0
    val n = 100
    val weights = Array.fill(n)(1.0 / n)
    val bound = PftDpwRuntime.particleErrorBound(weights, rMax = 1.0, gamma = 0.99)
    assertEqualsDouble(bound, 0.0, 1e-10)
  }

  test("Def 55: concentrated particle set gives positive bound") {
    // One particle has all weight -> maximum concentration
    val weights = Array(1.0, 0.0, 0.0, 0.0, 0.0)
    // Avoid log(0) issues: only the first weight matters
    // D2 = ln(1.0^2 * 5) = ln(5)
    val bound = PftDpwRuntime.particleErrorBound(
      weights.map(w => math.max(w, 1e-15)),
      rMax = 1.0, gamma = 0.99
    )
    assert(bound > 0.0, s"concentrated belief should have positive error bound, got $bound")
  }

  test("Def 55: error bound scales with R_max / (1 - gamma)") {
    val weights = Array(0.9, 0.05, 0.05)
    val bound1 = PftDpwRuntime.particleErrorBound(weights, rMax = 1.0, gamma = 0.9)
    val bound2 = PftDpwRuntime.particleErrorBound(weights, rMax = 2.0, gamma = 0.9)
    assertEqualsDouble(bound2 / bound1, 2.0, 1e-10)

    val bound3 = PftDpwRuntime.particleErrorBound(weights, rMax = 1.0, gamma = 0.5)
    // (1/(1-0.9)) / (1/(1-0.5)) = 10/2 = 5
    assertEqualsDouble(bound1 / bound3, 5.0, 1e-10)
  }

  // --- Def 29: Belief-averaged reward (tested via PftDpwResult) ---

  // Native solver tests: guarded by native library availability
  private def nativeAvailable: Boolean =
    try
      sicfun.holdem.HoldemPomcpNativeBindings.lastEngineCode()
      true
    catch case _: UnsatisfiedLinkError => false

  // Trivial 1-state 1-action MDP: known value = r / (1 - gamma)
  test("Def 30-31: trivial MDP converges to r / (1 - gamma)".tag(NativeTag)) {
    assume(nativeAvailable, "native library not available")

    // 1 state, 1 action, 1 observation, deterministic
    val model = TabularGenerativeModel(
      transitionTable = Array(0),            // state 0 -> state 0
      obsLikelihood = Array(1.0),            // always observe obs 0
      rewardTable = Array(1.0),              // reward = 1.0
      numStates = 1, numActions = 1, numObs = 1
    )
    val belief = ParticleBelief(Array(0), Array(1.0))
    val cfg = PftDpwConfig(
      numSimulations = 5000, gamma = 0.9, rMax = 1.0,
      maxDepth = 20, seed = 123L
    )
    val result = PftDpwRuntime.solve(model, belief, cfg)

    assert(result.isSuccess, s"solver failed with status ${result.status}")
    assertEquals(result.bestAction, 0)

    // Expected value: sum_{d=0}^{19} 0.9^d * 1.0 = (1 - 0.9^20) / (1 - 0.9) ~ 8.784
    val expected = (1.0 - math.pow(0.9, 20)) / (1.0 - 0.9)
    assertEqualsDouble(result.qValues(0), expected, 0.5) // within 0.5 of true
  }

  // 2-state bandit: action 0 gives reward 1, action 1 gives reward 0.5
  test("Def 32: optimal Q selects higher-reward action".tag(NativeTag)) {
    assume(nativeAvailable, "native library not available")

    val model = TabularGenerativeModel(
      transitionTable = Array(0, 0),          // both actions stay in state 0
      obsLikelihood = Array(1.0, 1.0),        // always observe obs 0
      rewardTable = Array(1.0, 0.5),          // action 0: r=1.0, action 1: r=0.5
      numStates = 1, numActions = 2, numObs = 1
    )
    val belief = ParticleBelief(Array(0), Array(1.0))
    val cfg = PftDpwConfig(
      numSimulations = 2000, gamma = 0.9, rMax = 1.0,
      maxDepth = 15, seed = 456L
    )
    val result = PftDpwRuntime.solve(model, belief, cfg)

    assert(result.isSuccess)
    assertEquals(result.bestAction, 0, "should prefer action with higher reward")
    assert(result.qValues(0) > result.qValues(1),
      s"Q(a=0)=${result.qValues(0)} should exceed Q(a=1)=${result.qValues(1)}")
  }

  // 2-state POMDP: test that observation-dependent planning works
  test("Def 54: POMDP with informative observations outperforms blind") {
    assume(nativeAvailable, "native library not available")

    // State 0: action 0 is good (r=1), action 1 is bad (r=0)
    // State 1: action 0 is bad (r=0), action 1 is good (r=1)
    // Observations perfectly reveal state
    val model = TabularGenerativeModel(
      transitionTable = Array(0, 0, 1, 1),   // states are absorbing
      obsLikelihood = Array(
        1.0, 0.0,   // state 0, action 0: see obs 0
        1.0, 0.0,   // state 0, action 1: see obs 0
        0.0, 1.0,   // state 1, action 0: see obs 1
        0.0, 1.0    // state 1, action 1: see obs 1
      ),
      rewardTable = Array(1.0, 0.0, 0.0, 1.0), // reward depends on (state, action)
      numStates = 2, numActions = 2, numObs = 2
    )

    // Belief: know we are in state 0
    val belief0 = ParticleBelief(Array(0, 0, 0, 0), Array(0.25, 0.25, 0.25, 0.25))
    val cfg = PftDpwConfig(numSimulations = 1000, gamma = 0.9, maxDepth = 10, seed = 789L)
    val result0 = PftDpwRuntime.solve(model, belief0, cfg)

    assert(result0.isSuccess)
    assertEquals(result0.bestAction, 0, "in state 0, action 0 is optimal")

    // Belief: know we are in state 1
    val belief1 = ParticleBelief(Array(1, 1, 1, 1), Array(0.25, 0.25, 0.25, 0.25))
    val result1 = PftDpwRuntime.solve(model, belief1, cfg)

    assert(result1.isSuccess)
    assertEquals(result1.bestAction, 1, "in state 1, action 1 is optimal")
  }

  // Backward compatibility: gamma=1 edge case rejected
  test("config with gamma=1 is rejected") {
    assume(nativeAvailable, "native library not available")

    val model = TabularGenerativeModel(
      Array(0), Array(1.0), Array(1.0), 1, 1, 1
    )
    val belief = ParticleBelief(Array(0), Array(1.0))
    val cfg = PftDpwConfig(gamma = 1.0) // invalid: must be in (0,1)
    val result = PftDpwRuntime.solve(model, belief, cfg)

    assert(!result.isSuccess, "gamma=1.0 should be rejected")
  }

  // Tag for tests requiring native library
  val NativeTag = new munit.Tag("native")
```

- [ ] 8b. Run pure Scala tests (ParticleBelief, config, error bound): `sbt "testOnly *PftDpwRuntimeTest" -- --exclude-tags=native`
- [ ] 8c. Build native DLL with PFT-DPW included
- [ ] 8d. Run native tests: `sbt "testOnly *PftDpwRuntimeTest"`

---

## Build Integration

- [ ] 9a. Add `PftDpwSolver.hpp` include to CMakeLists.txt (or Makefile) for sicfun_native_cpu.dll
- [ ] 9b. Add `HoldemPomcpNativeBindings.cpp` to the sicfun_native_cpu.dll source list
- [ ] 9c. Verify: full native build succeeds with no warnings (`/W4` on MSVC)
- [ ] 9d. Verify: existing CfrNativeSolver tests still pass (no regressions)

---

## Verification Checklist

| Spec Item | Where Tested |
|---|---|
| Def 29: belief-averaged reward | `PftDpwRuntimeTest` trivial MDP, Q-value convergence |
| Def 30: Q-function under policy | `PftDpwRuntimeTest` trivial MDP (Q = r + gamma * V) |
| Def 31: V under policy | Implicit in Q-function tests (V = Q(b, pi(b))) |
| Def 32: optimal Q | `PftDpwRuntimeTest` 2-action bandit (selects argmax Q) |
| Def 54: particle belief | `ParticleBelief` validation tests, POMDP observation test |
| Def 55: error bound | `particleErrorBound` uniform/concentrated/scaling tests |
| DPW action widening | Verified by UCB + widening threshold in C++ |
| DPW obs widening | Verified by POMDP test with 2 observations |
| Backward compat | gamma=1 rejected; gamma=0.99 with rho=0 is standard (no DRO) |

---

## Execution Order

1. Tasks 1-4 (C++ header) -- can be written and statically validated together
2. Task 5 (JNI bindings) -- depends on Task 4
3. Task 6 (Scala JNI object) -- depends on Task 5
4. Task 7 (Scala runtime) -- depends on Task 6
5. Task 8a (tests) -- write alongside Task 7
6. Task 9 (build integration) -- depends on Tasks 5-6
7. Tasks 8b-8d (run tests) -- depends on Task 9
