# Phase 3c: W-POMCP Multi-Agent Solver -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement W-POMCP multi-agent factored particle filter extending PFT-DPW
**Architecture:** C++17 header extending PftDpwSolver with per-rival belief particles and joint action sampling. JNI + Scala wrapper.
**Tech Stack:** C++17, Scala 3.8.1, munit 1.2.2
**Depends on:** Phase 3a (PFT-DPW base)
**Unlocks:** Phase 4a (kernel constructor uses POMCP for planning)

---

## Spec Reference (from SICFUN-v0.30.2)

### Def 54: Particle belief representation

```
b_hat_t^C := { (x_tilde_t^(j), w_t^(j)) }_{j=1}^{C}
```

Particles `x_tilde^(j)` are augmented hidden states `(theta^{R,i}, x^priv)` with
normalized weights `w^(j) > 0`, `sum_j w^(j) = 1`.

### Def 55: Particle belief MDP error bound

```
|V*(b_tilde) - V*(b_hat^C)| <= (R_max / (1 - gamma)) * sqrt(D_2(b_tilde || b_hat^C) / 2)
```

Error depends on `C` (particle count), not on `|X_tilde|`.

### Def 56: Factored particle filtering across rivals

```
b_hat_t^C = b_hat_t^{C,pub} (x) bigotimes_{i in R} b_hat_t^{C_i, i}
```

Each rival `i` gets `C_i` independent particles. Cross-rival dependency enters
only through `x_t^pub` (the public state).

### Def 12-14: Augmented hidden state (context)

```
X_tilde = X^pub x X^priv x prod_{i in R}(Theta^{R,i} x M^{R,i}) x Xi^S
```

The factored particle filter decomposes this product structure: the public component
is shared, while each rival's `(theta^{R,i}, m^{R,i})` is tracked independently.

---

## Architecture Overview

W-POMCP extends PFT-DPW (Phase 3a) to the multi-agent setting. The key differences:

1. **Factored belief** -- Instead of a single particle set over `X_tilde`, maintain
   one public belief `b^pub` plus per-rival particle sets `b^{C_i, i}`.
2. **Joint action sampling** -- At each simulation step, sample one action per rival
   from their respective belief-conditioned policies, producing a joint rival action.
3. **Weighted particles** -- Importance sampling weights attached to each particle,
   resampled when ESS drops below threshold.
4. **Variable rival count** -- Multiway mandate: `|R| >= 1`. Heads-up is `|R| = 1`.

### Inheritance from PFT-DPW

PftDpwSolver (Phase 3a) provides:
- UCB1-based action selection with progressive widening
- Tree node structure (visit counts, value estimates, children)
- Simulation/rollout framework
- Discount factor, reward bounds, exploration constant

WPomcpSolver specializes:
- `BeliefParticle` becomes `FactoredBelief` (public + per-rival particles)
- `simulate()` samples joint rival actions instead of single opponent action
- `updateBelief()` updates each rival's particle set independently
- Resampling is per-rival with ESS threshold

---

## File Map

| File | Responsibility | New/Modify |
|------|---------------|------------|
| `src/main/native/jni/WPomcpSolver.hpp` | C++ W-POMCP solver engine | NEW |
| `src/main/native/jni/HoldemPomcpNativeBindings.cpp` | JNI entry points for POMDP solvers | MODIFY (add W-POMCP entries) |
| `src/main/scala/sicfun/holdem/strategic/solver/WPomcpRuntime.scala` | Scala wrapper | NEW |
| `src/test/scala/sicfun/holdem/strategic/solver/WPomcpRuntimeTest.scala` | Tests | NEW |

---

## Task Execution Order

| Task | File(s) | Depends On | Description |
|------|---------|-----------|-------------|
| 1 | WPomcpSolver.hpp (types) | Phase 3a | Particle, factored belief, config structs |
| 2 | WPomcpSolver.hpp (resampling) | Task 1 | Systematic resampling + ESS |
| 3 | WPomcpSolver.hpp (joint action) | Task 1 | Joint rival action sampling |
| 4 | WPomcpSolver.hpp (belief update) | Tasks 1-3 | Per-rival factored belief update |
| 5 | WPomcpSolver.hpp (search) | Tasks 1-4 | W-POMCP tree search (simulate/select/expand) |
| 6 | WPomcpSolver.hpp (top-level API) | Task 5 | solve() entry point + validation |
| 7 | HoldemPomcpNativeBindings.cpp | Task 6 | JNI entry points for W-POMCP |
| 8 | WPomcpRuntime.scala | Task 7 | Scala wrapper over JNI |
| 9 | WPomcpRuntimeTest.scala | Task 8 | Full test suite |
| 10 | Backward compat + integration | Tasks 1-9 | |R|=1 recovery, error bound validation |

---

### Task 1: Particle and Factored Belief Types

**Files:**
- Create: `src/main/native/jni/WPomcpSolver.hpp` (initial types only)

- [ ] **Step 1: Write the test harness (C++ inline tests)**

The WPomcpSolver.hpp header includes a `wpomcp::self_test()` function gated behind
`WPOMCP_SELF_TEST`. This follows the pattern from BayesNativeUpdateCore.hpp where
correctness is validated via status codes. The JNI test in Task 9 calls this.

- [ ] **Step 2: Write particle and factored belief types**

```cpp
/*
 * WPomcpSolver.hpp -- W-POMCP multi-agent factored particle filter tree search.
 *
 * Implements Definition 56 (factored particle filtering across rivals) from
 * SICFUN-v0.30.2 as an extension of PFT-DPW (Phase 3a). Each rival i gets
 * an independent particle set of C_i particles, with cross-rival dependency
 * entering only through the shared public state x_t^pub.
 *
 * Key design:
 *   - Header-only, shared by CPU JNI binding (HoldemPomcpNativeBindings.cpp).
 *   - Extends PftDpwSolver's tree structure with factored belief nodes.
 *   - Per-rival particle sets with importance weights and systematic resampling.
 *   - Joint rival action sampling for multi-agent simulation.
 *   - Multiway-native: |R| >= 1 always. Heads-up is |R| = 1.
 *
 * JNI class: sicfun.holdem.HoldemPomcpNativeBindings
 */

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "PftDpwSolver.hpp"

namespace wpomcp {

/* ---------- Status codes (extend pftdpw codes) ---------- */

constexpr int kStatusOk = 0;
constexpr int kStatusNullArray = 100;
constexpr int kStatusLengthMismatch = 101;
constexpr int kStatusReadFailure = 102;
constexpr int kStatusWriteFailure = 124;
constexpr int kStatusInvalidConfig = 160;
constexpr int kStatusInvalidParticleCount = 170;
constexpr int kStatusInvalidRivalCount = 171;
constexpr int kStatusInvalidActionCount = 172;
constexpr int kStatusDegenerateWeights = 173;
constexpr int kStatusMaxRivalsExceeded = 174;
constexpr int kStatusSimulationOverflow = 175;

/* Maximum number of rivals. 8 covers 9-max tables (hero + 8 rivals). */
constexpr int kMaxRivals = 8;

/* Maximum actions per decision point. */
constexpr int kMaxActions = 16;

/* Default ESS threshold ratio for resampling trigger. */
constexpr double kDefaultEssThreshold = 0.5;

/* ---------- Particle types ---------- */

/** A single weighted particle for one rival's hidden state.
  *
  * Fields:
  *   rival_type  -- discrete type index theta^{R,i} (behavioral archetype)
  *   priv_state  -- discrete private state index (e.g., hand bucket)
  *   memory      -- rival's memory/history encoding m^{R,i}
  *   weight      -- importance weight, positive, not necessarily normalized
  */
struct RivalParticle {
  int rival_type = 0;
  int priv_state = 0;
  int memory = 0;
  double weight = 1.0;
};

/** Public state component shared across all rivals.
  *
  * This is the x^pub part of the augmented state X_tilde (Def 12).
  * In poker: board cards, pot, street, betting history.
  */
struct PublicState {
  int street = 0;           /* 0=preflop, 1=flop, 2=turn, 3=river */
  int board_hash = 0;       /* compact board representation */
  double pot = 0.0;         /* current pot size */
  int action_seq_len = 0;   /* length of public action sequence */
  /* Action sequence stored externally in flat array for JNI efficiency. */
};

/** Per-rival particle set: C_i weighted particles for rival i.
  *
  * Implements the b_hat_t^{C_i, i} factor from Def 56.
  * Particles are independently maintained per rival.
  * Resampling is triggered when ESS < ess_threshold * C_i.
  */
struct RivalBeliefSet {
  std::vector<RivalParticle> particles;
  double ess_threshold = kDefaultEssThreshold;

  /** Effective sample size: 1 / sum(w_j^2) after normalization.
    * Returns C_i when all weights are equal (best case).
    * Returns 1.0 when one particle dominates (worst case). */
  double ess() const {
    if (particles.empty()) return 0.0;
    double sum_w = 0.0;
    double sum_w2 = 0.0;
    for (const auto& p : particles) {
      sum_w += p.weight;
      sum_w2 += p.weight * p.weight;
    }
    if (sum_w <= 0.0 || sum_w2 <= 0.0) return 0.0;
    return (sum_w * sum_w) / sum_w2;
  }

  /** Normalize all weights to sum to 1. Returns false if sum is zero/negative. */
  bool normalize() {
    double sum = 0.0;
    for (const auto& p : particles) sum += p.weight;
    if (sum <= 0.0) return false;
    const double inv_sum = 1.0 / sum;
    for (auto& p : particles) p.weight *= inv_sum;
    return true;
  }

  /** Whether resampling is needed based on ESS threshold. */
  bool needs_resample() const {
    const double n = static_cast<double>(particles.size());
    return n > 0.0 && ess() < ess_threshold * n;
  }

  int particle_count() const {
    return static_cast<int>(particles.size());
  }
};

/** Factored belief state: b_hat = b^pub (x) bigotimes_{i in R} b^{C_i, i}.
  *
  * Implements Def 56. The public state is deterministic (fully observed).
  * Each rival has an independent particle set.
  * Cross-rival dependency enters only through public_state.
  */
struct FactoredBelief {
  PublicState public_state;
  std::vector<RivalBeliefSet> rival_beliefs;  /* indexed by rival 0...|R|-1 */

  int rival_count() const {
    return static_cast<int>(rival_beliefs.size());
  }

  /** Total particle count across all rivals. */
  int total_particles() const {
    int total = 0;
    for (const auto& rb : rival_beliefs) total += rb.particle_count();
    return total;
  }
};

/** Configuration for W-POMCP search. */
struct WPomcpConfig {
  int num_simulations = 1000;        /* Monte Carlo simulations per search call */
  double discount = 0.99;           /* gamma: discount factor */
  double exploration_constant = 1.0; /* UCB1 exploration constant c */
  double r_max = 1.0;               /* maximum absolute reward */
  int max_depth = 50;               /* maximum tree depth per simulation */
  double ess_threshold = 0.5;       /* ESS ratio for resampling trigger */

  /* Progressive widening parameters (inherited from PFT-DPW). */
  double pw_alpha = 0.5;            /* N(s)^alpha controls action widening */
  double pw_c = 1.0;                /* constant multiplier for widening */

  /* Per-rival particle counts. Length must equal rival_count. */
  std::vector<int> particles_per_rival;

  int rival_count() const {
    return static_cast<int>(particles_per_rival.size());
  }
};
```

- [ ] **Step 3: Write config validation**

```cpp
/** Validate a WPomcpConfig. Returns kStatusOk or an error code. */
inline int validate_config(const WPomcpConfig& cfg) {
  if (cfg.num_simulations <= 0) return kStatusInvalidConfig;
  if (cfg.discount <= 0.0 || cfg.discount >= 1.0) return kStatusInvalidConfig;
  if (cfg.exploration_constant < 0.0) return kStatusInvalidConfig;
  if (cfg.r_max <= 0.0) return kStatusInvalidConfig;
  if (cfg.max_depth <= 0) return kStatusInvalidConfig;
  if (cfg.ess_threshold <= 0.0 || cfg.ess_threshold > 1.0) return kStatusInvalidConfig;
  if (cfg.pw_alpha <= 0.0 || cfg.pw_alpha >= 1.0) return kStatusInvalidConfig;
  if (cfg.pw_c <= 0.0) return kStatusInvalidConfig;
  const int nr = cfg.rival_count();
  if (nr < 1 || nr > kMaxRivals) return kStatusInvalidRivalCount;
  for (int i = 0; i < nr; ++i) {
    if (cfg.particles_per_rival[i] < 1) return kStatusInvalidParticleCount;
  }
  return kStatusOk;
}
```

- [ ] **Step 4: Verify compilation**

Compile the header in isolation:
```
clang++ -std=c++17 -fsyntax-only -I. WPomcpSolver.hpp
```
This requires PftDpwSolver.hpp from Phase 3a to be present.

---

### Task 2: Systematic Resampling

**Files:**
- Modify: `src/main/native/jni/WPomcpSolver.hpp`

- [ ] **Step 1: Design the resampling interface**

Systematic resampling maintains particle diversity when weights become skewed.
Input: weighted particles. Output: equally-weighted resampled particles.

- [ ] **Step 2: Implement systematic resampling**

```cpp
/** Systematic resampling for a single rival's particle set.
  *
  * Given N weighted particles (normalized), produces N equally-weighted
  * particles by systematic sampling along the cumulative weight CDF.
  * This is O(N) and produces lower variance than multinomial resampling.
  *
  * After resampling, all weights are reset to 1/N.
  *
  * The rng parameter must be a uniform random engine (e.g., mt19937).
  */
template <typename Rng>
inline void systematic_resample(RivalBeliefSet& belief, Rng& rng) {
  const int n = belief.particle_count();
  if (n <= 1) return;

  /* Normalize weights before resampling. */
  if (!belief.normalize()) return;

  /* Build cumulative weight array. */
  std::vector<double> cumulative(n);
  cumulative[0] = belief.particles[0].weight;
  for (int i = 1; i < n; ++i) {
    cumulative[i] = cumulative[i - 1] + belief.particles[i].weight;
  }
  /* Fix floating-point: force last entry to exactly 1.0. */
  cumulative[n - 1] = 1.0;

  /* Draw single uniform offset u ~ U(0, 1/N). */
  std::uniform_real_distribution<double> dist(0.0, 1.0 / n);
  const double u0 = dist(rng);

  /* Walk through cumulative array, selecting particles. */
  std::vector<RivalParticle> resampled;
  resampled.reserve(n);
  int cursor = 0;
  const double equal_weight = 1.0 / static_cast<double>(n);

  for (int j = 0; j < n; ++j) {
    const double threshold = u0 + static_cast<double>(j) / static_cast<double>(n);
    while (cursor < n - 1 && cumulative[cursor] < threshold) {
      ++cursor;
    }
    RivalParticle p = belief.particles[cursor];
    p.weight = equal_weight;
    resampled.push_back(p);
  }

  belief.particles = std::move(resampled);
}

/** Conditionally resample each rival's belief set if ESS is below threshold. */
template <typename Rng>
inline void resample_if_needed(FactoredBelief& belief, Rng& rng) {
  for (auto& rb : belief.rival_beliefs) {
    if (rb.needs_resample()) {
      systematic_resample(rb, rng);
    }
  }
}
```

- [ ] **Step 3: Verify resampling invariants**

The following must hold after `systematic_resample`:
1. Output particle count equals input particle count.
2. All output weights are exactly `1/N`.
3. Output weight sum is 1.0 (within floating-point tolerance).
4. No particle has negative weight.
These are validated in the self_test function (Task 6) and via JNI tests (Task 9).

---

### Task 3: Joint Rival Action Sampling

**Files:**
- Modify: `src/main/native/jni/WPomcpSolver.hpp`

- [ ] **Step 1: Define the action model interface**

The solver needs a way to query each rival's action distribution given their
particle state and the public state. This is provided by the caller as a
flat callback-compatible structure (function pointer or flat array).

- [ ] **Step 2: Implement joint action sampling**

```cpp
/** Action distribution for a single rival conditioned on particle state.
  *
  * In JNI mode, this is a flat array of probabilities indexed by action,
  * passed from the Scala side. The solver samples from this distribution.
  *
  * action_probs[a] = P(rival i plays action a | particle state, public state)
  * Length: num_actions for this decision point.
  */
struct RivalActionDist {
  const double* action_probs = nullptr;  /* borrowed pointer, not owned */
  int num_actions = 0;
};

/** Joint rival action: one sampled action per rival.
  *
  * For |R| rivals at a decision point, this holds the sampled action index
  * for each rival. Used to advance the simulation one step.
  */
struct JointRivalAction {
  std::array<int, kMaxRivals> actions{};  /* action index per rival */
  int rival_count = 0;

  int operator[](int rival_idx) const { return actions[rival_idx]; }
  int& operator[](int rival_idx) { return actions[rival_idx]; }
};

/** Sample a joint rival action from per-rival action distributions.
  *
  * For each rival i, samples one action from action_dists[i] weighted by
  * the corresponding probabilities. Actions are sampled independently
  * because the factored belief (Def 56) treats rival hidden states as
  * independent given the public state.
  *
  * Returns kStatusOk on success, or an error code if any distribution is
  * invalid (empty, negative probs, zero sum).
  */
template <typename Rng>
inline int sample_joint_action(
    const RivalActionDist* action_dists,
    int rival_count,
    Rng& rng,
    JointRivalAction& out) {

  if (rival_count < 1 || rival_count > kMaxRivals) {
    return kStatusInvalidRivalCount;
  }
  out.rival_count = rival_count;

  for (int i = 0; i < rival_count; ++i) {
    const auto& dist = action_dists[i];
    if (dist.num_actions < 1 || dist.num_actions > kMaxActions) {
      return kStatusInvalidActionCount;
    }
    if (dist.action_probs == nullptr) {
      return kStatusNullArray;
    }

    /* Compute cumulative distribution. */
    double cumsum = 0.0;
    for (int a = 0; a < dist.num_actions; ++a) {
      if (dist.action_probs[a] < 0.0) return kStatusInvalidConfig;
      cumsum += dist.action_probs[a];
    }
    if (cumsum <= 0.0) return kStatusDegenerateWeights;

    /* Sample from CDF. */
    std::uniform_real_distribution<double> u(0.0, cumsum);
    const double draw = u(rng);
    double running = 0.0;
    int selected = dist.num_actions - 1;  /* fallback to last action */
    for (int a = 0; a < dist.num_actions; ++a) {
      running += dist.action_probs[a];
      if (draw <= running) {
        selected = a;
        break;
      }
    }
    out.actions[i] = selected;
  }
  return kStatusOk;
}
```

- [ ] **Step 3: Verify sampling correctness**

Self-test validates:
1. Deterministic distribution (prob=1 on one action) always selects that action.
2. Uniform distribution samples all actions over many trials.
3. Invalid inputs return appropriate error codes.

---

### Task 4: Factored Belief Update

**Files:**
- Modify: `src/main/native/jni/WPomcpSolver.hpp`

- [ ] **Step 1: Define the observation model interface**

After an action is taken and observation received, each rival's particles
must be updated. The update uses importance weighting: multiply each particle's
weight by the likelihood of the observation given that particle's state.

- [ ] **Step 2: Implement per-rival belief update**

```cpp
/** Observation likelihood for a single rival's particle.
  *
  * Provided by the caller: given rival i's particle state and the public
  * observation, what is the likelihood? This is used for importance weighting.
  *
  * In JNI mode, the Scala caller pre-computes a flat likelihood array
  * indexed by [rival][particle], avoiding per-particle JNI callbacks.
  */
struct ObservationLikelihoods {
  const double* likelihoods = nullptr;  /* borrowed, length = particle_count */
  int particle_count = 0;
};

/** Update a single rival's particle weights given observation likelihoods.
  *
  * For each particle j of rival i:
  *   w_j *= P(observation | x_tilde^(j))
  *
  * Then normalize. If all weights collapse to zero, returns false
  * (degenerate filter -- caller should reinitialize).
  *
  * This implements the weight update step of the SIR (Sequential Importance
  * Resampling) particle filter underlying Def 56.
  */
inline bool update_rival_weights(
    RivalBeliefSet& belief,
    const ObservationLikelihoods& obs_liks) {

  const int n = belief.particle_count();
  if (n != obs_liks.particle_count || obs_liks.likelihoods == nullptr) {
    return false;
  }

  for (int j = 0; j < n; ++j) {
    const double lik = obs_liks.likelihoods[j];
    if (lik < 0.0 || !std::isfinite(lik)) return false;
    belief.particles[j].weight *= lik;
  }

  return belief.normalize();
}

/** Update the full factored belief after an observation.
  *
  * Updates each rival's particle weights independently (Def 56: cross-rival
  * dependency only through public state, which is updated separately).
  * Then conditionally resamples each rival whose ESS drops below threshold.
  *
  * Parameters:
  *   belief      -- factored belief to update in-place
  *   pub_update  -- new public state after the observation
  *   rival_liks  -- per-rival observation likelihoods, length = rival_count
  *   rng         -- random engine for resampling
  *
  * Returns kStatusOk on success or an error code.
  */
template <typename Rng>
inline int update_factored_belief(
    FactoredBelief& belief,
    const PublicState& pub_update,
    const ObservationLikelihoods* rival_liks,
    int rival_count,
    Rng& rng) {

  if (rival_count != belief.rival_count()) {
    return kStatusInvalidRivalCount;
  }

  /* Update public state (deterministic, fully observed). */
  belief.public_state = pub_update;

  /* Update each rival's particle weights independently. */
  for (int i = 0; i < rival_count; ++i) {
    if (!update_rival_weights(belief.rival_beliefs[i], rival_liks[i])) {
      return kStatusDegenerateWeights;
    }
  }

  /* Resample rivals whose ESS dropped below threshold. */
  resample_if_needed(belief, rng);

  return kStatusOk;
}
```

- [ ] **Step 3: Verify belief update invariants**

After `update_factored_belief`:
1. Each rival's weights sum to 1.0.
2. Particle counts unchanged (resampling preserves count).
3. Public state updated to `pub_update`.
4. Degenerate likelihoods (all zero) return `kStatusDegenerateWeights`.

---

### Task 5: W-POMCP Tree Search

**Files:**
- Modify: `src/main/native/jni/WPomcpSolver.hpp`

- [ ] **Step 1: Define the search tree node**

The W-POMCP tree extends PFT-DPW's tree with factored belief at each node.

- [ ] **Step 2: Implement the search tree and simulation**

```cpp
/** Tree node for W-POMCP search.
  *
  * Each node holds:
  *   - A factored belief (public state + per-rival particles)
  *   - Per-action statistics (visit count, value sum)
  *   - Children indexed by (hero_action, joint_rival_action, observation)
  *
  * Children are lazily expanded during simulation (progressive widening
  * inherited from PFT-DPW).
  */
struct WPomcpNode {
  /* Per-action statistics for hero's actions. */
  struct ActionStats {
    int visit_count = 0;
    double value_sum = 0.0;
    double mean_value() const {
      return visit_count > 0 ? value_sum / visit_count : 0.0;
    }
  };

  int visit_count = 0;
  std::vector<ActionStats> action_stats;  /* indexed by hero action */
  std::vector<int> expanded_actions;      /* hero actions expanded so far */

  /* Children are stored as a flat vector keyed by an expanded-action index.
   * Each expanded action can have multiple observation-children (stochastic). */
  struct ChildKey {
    int hero_action = 0;
    int obs_hash = 0;  /* hash of the observation received after action */
  };
  std::vector<ChildKey> child_keys;
  std::vector<WPomcpNode> children;

  /** UCB1 action score for hero action a at this node.
    * score = Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
    * Unvisited actions return +infinity. */
  double ucb1_score(int action_idx, double c) const {
    const auto& stats = action_stats[action_idx];
    if (stats.visit_count == 0) {
      return std::numeric_limits<double>::infinity();
    }
    const double exploit = stats.mean_value();
    const double explore = c * std::sqrt(
        std::log(static_cast<double>(visit_count)) /
        static_cast<double>(stats.visit_count));
    return exploit + explore;
  }

  /** Select hero action by UCB1. Returns action index. */
  int select_action_ucb1(double c, int num_actions) const {
    int best = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    for (int a = 0; a < num_actions; ++a) {
      const double score = ucb1_score(a, c);
      if (score > best_score) {
        best_score = score;
        best = a;
      }
    }
    return best;
  }

  /** Check if progressive widening allows expanding a new action.
    * Condition: |expanded_actions| < c * N(s)^alpha */
  bool should_widen(double pw_c, double pw_alpha) const {
    const double limit = pw_c * std::pow(
        static_cast<double>(visit_count), pw_alpha);
    return static_cast<double>(expanded_actions.size()) < limit;
  }
};

/** W-POMCP solver: multi-agent factored POMCP extending PFT-DPW.
  *
  * The solver runs num_simulations Monte Carlo simulations from the root
  * belief, building a tree of WPomcpNodes. Each simulation:
  *   1. SELECT: traverse tree using UCB1 for hero actions
  *   2. EXPAND: add new child via progressive widening
  *   3. SIMULATE: roll out with joint rival action sampling
  *   4. BACKPROPAGATE: update visit counts and value estimates
  *
  * The factored belief (Def 56) is maintained at each node: per-rival
  * particle sets are updated independently, with cross-rival dependency
  * only through the public state.
  */
class WPomcpSolver {
public:
  /** Transition model interface.
    *
    * The caller provides this to define the game dynamics:
    *   - hero_actions: how many actions available at a state
    *   - rival_action_dist: rival i's action distribution given state
    *   - transition: given (public_state, hero_action, joint_rival_action),
    *     produce (next_public_state, observation_hash, reward, is_terminal)
    *   - rollout_value: default policy value estimate from a leaf
    *
    * All data is passed as flat arrays for JNI efficiency. The Scala side
    * pre-computes these and passes them through JNI.
    */
  struct TransitionModel {
    /* Flat arrays passed from JNI. Indexed by state/action pairs.
     * Layout documented in HoldemPomcpNativeBindings.cpp. */
    const double* rival_action_probs = nullptr;  /* [rival][action] per state */
    const double* transition_probs = nullptr;     /* [state][action] -> next */
    const double* rewards = nullptr;              /* [state][action] -> reward */
    const int* terminal_flags = nullptr;          /* [state] -> is_terminal */
    const double* rollout_values = nullptr;       /* [state] -> default value */
    int num_hero_actions = 0;
    int num_obs = 0;  /* observation space cardinality */
  };

  /** Result of a W-POMCP search.
    *
    * action_values[a] = estimated Q(root, a) from the search tree.
    * best_action = argmax_a action_values[a].
    * root_value = max_a action_values[a].
    */
  struct SearchResult {
    std::vector<double> action_values;
    int best_action = -1;
    double root_value = 0.0;
    int simulations_completed = 0;
    int tree_node_count = 0;
    int status = kStatusOk;
  };

private:
  WPomcpConfig config_;
  std::mt19937 rng_;
  WPomcpNode root_;

  /** Run a single simulation from the given node at the given depth.
    *
    * Returns the discounted cumulative reward from this point forward.
    * Modifies the node's visit counts and value estimates.
    */
  double simulate(
      WPomcpNode& node,
      FactoredBelief belief,  /* by value: each simulation works on a copy */
      const TransitionModel& model,
      int depth) {

    /* Base cases. */
    if (depth >= config_.max_depth) return 0.0;

    const int num_actions = model.num_hero_actions;
    if (num_actions <= 0) return 0.0;

    /* Initialize action stats on first visit. */
    if (node.action_stats.empty()) {
      node.action_stats.resize(num_actions);
    }

    /* SELECT: choose hero action by UCB1. */
    int hero_action;
    if (node.should_widen(config_.pw_c, config_.pw_alpha)) {
      /* Progressive widening: try a new action. */
      hero_action = pick_unexpanded_action(node, num_actions);
      node.expanded_actions.push_back(hero_action);
    } else {
      /* Exploit/explore among expanded actions. */
      hero_action = node.select_action_ucb1(
          config_.exploration_constant, num_actions);
    }

    /* Sample joint rival actions from factored belief. */
    JointRivalAction joint_action;
    {
      std::vector<RivalActionDist> rival_dists(config_.rival_count());
      for (int i = 0; i < config_.rival_count(); ++i) {
        /* Rival i's action distribution conditioned on their
         * particle-averaged state and the public state. */
        rival_dists[i].action_probs =
            model.rival_action_probs +
            i * model.num_hero_actions;  /* simplified: same action space */
        rival_dists[i].num_actions = model.num_hero_actions;
      }
      int status = sample_joint_action(
          rival_dists.data(), config_.rival_count(), rng_, joint_action);
      if (status != kStatusOk) return 0.0;
    }

    /* TRANSITION: get reward and next state. */
    double reward = 0.0;
    bool is_terminal = false;
    PublicState next_pub;
    int obs_hash = 0;

    /* Compute reward from model arrays. Flat index: hero_action. */
    if (model.rewards != nullptr) {
      reward = model.rewards[hero_action];
    }
    if (model.terminal_flags != nullptr && model.terminal_flags[0] != 0) {
      is_terminal = true;
    }

    if (is_terminal) {
      /* Terminal: return immediate reward. */
      node.visit_count++;
      node.action_stats[hero_action].visit_count++;
      node.action_stats[hero_action].value_sum += reward;
      return reward;
    }

    /* UPDATE factored belief with observation (per-rival, independent). */
    next_pub = belief.public_state;
    next_pub.street = std::min(next_pub.street + 1, 3);
    next_pub.pot += reward;

    /* Build per-rival observation likelihoods from model. */
    std::vector<ObservationLikelihoods> rival_obs(config_.rival_count());
    std::vector<std::vector<double>> lik_storage(config_.rival_count());
    for (int i = 0; i < config_.rival_count(); ++i) {
      const int np = belief.rival_beliefs[i].particle_count();
      lik_storage[i].assign(np, 1.0);  /* uniform likelihood as default */
      rival_obs[i].likelihoods = lik_storage[i].data();
      rival_obs[i].particle_count = np;
    }
    update_factored_belief(belief, next_pub,
                           rival_obs.data(), config_.rival_count(), rng_);

    /* EXPAND / RECURSE: find or create child node. */
    obs_hash = next_pub.street * 1000 + hero_action;
    WPomcpNode* child = find_or_create_child(node, hero_action, obs_hash);

    /* Recursive simulation from child. */
    double future = simulate(*child, std::move(belief), model, depth + 1);
    double total = reward + config_.discount * future;

    /* BACKPROPAGATE. */
    node.visit_count++;
    node.action_stats[hero_action].visit_count++;
    node.action_stats[hero_action].value_sum += total;

    return total;
  }

  /** Find an existing child node or create a new one. */
  WPomcpNode* find_or_create_child(
      WPomcpNode& parent, int hero_action, int obs_hash) {

    for (size_t i = 0; i < parent.child_keys.size(); ++i) {
      if (parent.child_keys[i].hero_action == hero_action &&
          parent.child_keys[i].obs_hash == obs_hash) {
        return &parent.children[i];
      }
    }
    /* Create new child. */
    parent.child_keys.push_back({hero_action, obs_hash});
    parent.children.emplace_back();
    return &parent.children.back();
  }

  /** Pick an action not yet in expanded_actions. */
  int pick_unexpanded_action(const WPomcpNode& node, int num_actions) {
    for (int a = 0; a < num_actions; ++a) {
      bool found = false;
      for (int ea : node.expanded_actions) {
        if (ea == a) { found = true; break; }
      }
      if (!found) return a;
    }
    /* All expanded: pick by UCB1. */
    return node.select_action_ucb1(config_.exploration_constant, num_actions);
  }

public:
  /** Construct a W-POMCP solver with the given config and random seed. */
  explicit WPomcpSolver(const WPomcpConfig& config, uint64_t seed = 42)
      : config_(config), rng_(seed) {}

  /** Run W-POMCP search from the given root belief.
    *
    * Performs config.num_simulations Monte Carlo simulations, building
    * a search tree. Returns the estimated action values at the root.
    *
    * The model parameter provides the game dynamics (transition, reward,
    * rival action distributions) via flat arrays for JNI efficiency.
    */
  SearchResult search(
      const FactoredBelief& root_belief,
      const TransitionModel& model) {

    SearchResult result;

    /* Validate config. */
    int status = validate_config(config_);
    if (status != kStatusOk) {
      result.status = status;
      return result;
    }

    if (model.num_hero_actions < 1 || model.num_hero_actions > kMaxActions) {
      result.status = kStatusInvalidActionCount;
      return result;
    }

    /* Clear tree for fresh search. */
    root_ = WPomcpNode{};

    /* Run simulations. */
    int completed = 0;
    for (int sim = 0; sim < config_.num_simulations; ++sim) {
      simulate(root_, root_belief, model, 0);
      ++completed;
    }

    /* Extract action values from root. */
    result.action_values.resize(model.num_hero_actions);
    double best_val = -std::numeric_limits<double>::infinity();
    int best_act = 0;
    for (int a = 0; a < model.num_hero_actions; ++a) {
      result.action_values[a] = root_.action_stats[a].mean_value();
      if (result.action_values[a] > best_val) {
        best_val = result.action_values[a];
        best_act = a;
      }
    }

    result.best_action = best_act;
    result.root_value = best_val;
    result.simulations_completed = completed;
    result.tree_node_count = count_nodes(root_);
    result.status = kStatusOk;
    return result;
  }

  /** Count total nodes in the tree rooted at the given node. */
  static int count_nodes(const WPomcpNode& node) {
    int count = 1;
    for (const auto& child : node.children) {
      count += count_nodes(child);
    }
    return count;
  }

  /** Access the config (for testing). */
  const WPomcpConfig& config() const { return config_; }
};
```

- [ ] **Step 3: Verify search produces valid output**

For a trivial 2-action game with known optimal action:
1. `best_action` matches the dominant action.
2. `action_values` for the dominant action exceeds the other.
3. `simulations_completed == num_simulations`.
4. `tree_node_count >= 1`.

---

### Task 6: Top-Level API and Self-Test

**Files:**
- Modify: `src/main/native/jni/WPomcpSolver.hpp`

- [ ] **Step 1: Add solve_raw entry point for JNI**

```cpp
/** JNI-friendly entry point: solve from flat arrays.
  *
  * All inputs are flat arrays (no pointers to structs) for zero-copy
  * JNI critical array access. This matches the protocol established by
  * BayesNativeUpdateCore.hpp and CfrNativeSolverCore.hpp.
  *
  * Parameters (all borrowed, not owned):
  *   rival_count           -- number of rivals (1..kMaxRivals)
  *   particles_per_rival   -- C_i for each rival, length = rival_count
  *   particle_types        -- rival type indices, flat [rival][particle]
  *   particle_priv_states  -- private state indices, flat [rival][particle]
  *   particle_weights      -- importance weights, flat [rival][particle]
  *   pub_street            -- public state street
  *   pub_pot               -- public state pot size
  *   num_hero_actions      -- hero's available action count
  *   rival_action_probs    -- per-rival action probs, flat [rival][action]
  *   rewards               -- reward per hero action, length = num_hero_actions
  *   num_simulations       -- MCTS simulations to run
  *   discount              -- gamma
  *   exploration           -- UCB1 exploration constant
  *   r_max                 -- reward bound
  *   max_depth             -- tree depth limit
  *   ess_threshold         -- resampling ESS ratio
  *   seed                  -- RNG seed
  *   out_action_values     -- output: Q(root, a), length = num_hero_actions
  *   out_best_action       -- output: single int[1], best action index
  *   out_root_value        -- output: single double[1], root value
  *
  * Returns kStatusOk on success, or an error code.
  */
inline int solve_raw(
    int rival_count,
    const int* particles_per_rival,
    const int* particle_types,
    const int* particle_priv_states,
    const double* particle_weights,
    int pub_street,
    double pub_pot,
    int num_hero_actions,
    const double* rival_action_probs,
    const double* rewards,
    int num_simulations,
    double discount,
    double exploration,
    double r_max,
    int max_depth,
    double ess_threshold,
    int64_t seed,
    double* out_action_values,
    int* out_best_action,
    double* out_root_value) {

  /* Validate rival count. */
  if (rival_count < 1 || rival_count > kMaxRivals) {
    return kStatusInvalidRivalCount;
  }
  if (particles_per_rival == nullptr || particle_types == nullptr ||
      particle_priv_states == nullptr || particle_weights == nullptr) {
    return kStatusNullArray;
  }
  if (rival_action_probs == nullptr || rewards == nullptr) {
    return kStatusNullArray;
  }
  if (out_action_values == nullptr || out_best_action == nullptr ||
      out_root_value == nullptr) {
    return kStatusNullArray;
  }

  /* Build config. */
  WPomcpConfig config;
  config.num_simulations = num_simulations;
  config.discount = discount;
  config.exploration_constant = exploration;
  config.r_max = r_max;
  config.max_depth = max_depth;
  config.ess_threshold = ess_threshold;
  config.particles_per_rival.resize(rival_count);
  for (int i = 0; i < rival_count; ++i) {
    config.particles_per_rival[i] = particles_per_rival[i];
  }

  int status = validate_config(config);
  if (status != kStatusOk) return status;

  /* Build factored belief from flat arrays. */
  FactoredBelief root_belief;
  root_belief.public_state.street = pub_street;
  root_belief.public_state.pot = pub_pot;
  root_belief.rival_beliefs.resize(rival_count);

  int flat_offset = 0;
  for (int i = 0; i < rival_count; ++i) {
    const int ci = particles_per_rival[i];
    auto& rb = root_belief.rival_beliefs[i];
    rb.ess_threshold = ess_threshold;
    rb.particles.resize(ci);
    for (int j = 0; j < ci; ++j) {
      rb.particles[j].rival_type = particle_types[flat_offset + j];
      rb.particles[j].priv_state = particle_priv_states[flat_offset + j];
      rb.particles[j].weight = particle_weights[flat_offset + j];
    }
    if (!rb.normalize()) return kStatusDegenerateWeights;
    flat_offset += ci;
  }

  /* Build transition model from flat arrays. */
  WPomcpSolver::TransitionModel model;
  model.rival_action_probs = rival_action_probs;
  model.rewards = rewards;
  model.num_hero_actions = num_hero_actions;

  /* Run solver. */
  WPomcpSolver solver(config, static_cast<uint64_t>(seed));
  auto result = solver.search(root_belief, model);

  if (result.status != kStatusOk) return result.status;

  /* Write outputs. */
  for (int a = 0; a < num_hero_actions; ++a) {
    out_action_values[a] = result.action_values[a];
  }
  *out_best_action = result.best_action;
  *out_root_value = result.root_value;

  return kStatusOk;
}

/* ---------- Self-test ---------- */

#ifdef WPOMCP_SELF_TEST

/** Run internal validation. Returns kStatusOk if all pass. */
inline int self_test() {
  /* Test 1: RivalBeliefSet ESS and normalization. */
  {
    RivalBeliefSet rb;
    rb.particles = {{0, 0, 0, 0.5}, {0, 1, 0, 0.5}};
    if (!rb.normalize()) return 1;
    double ess = rb.ess();
    /* Equal weights: ESS should be 2.0. */
    if (std::abs(ess - 2.0) > 1e-10) return 2;
  }
  /* Test 2: Skewed weights ESS. */
  {
    RivalBeliefSet rb;
    rb.particles = {{0, 0, 0, 0.99}, {0, 1, 0, 0.01}};
    if (!rb.normalize()) return 3;
    double ess = rb.ess();
    /* Highly skewed: ESS should be close to 1. */
    if (ess > 1.1) return 4;
  }
  /* Test 3: Systematic resampling preserves count. */
  {
    RivalBeliefSet rb;
    rb.particles = {{0, 0, 0, 0.9}, {0, 1, 0, 0.05}, {0, 2, 0, 0.05}};
    rb.normalize();
    std::mt19937 rng(12345);
    systematic_resample(rb, rng);
    if (rb.particle_count() != 3) return 5;
    /* After resampling, all weights should be 1/3. */
    for (const auto& p : rb.particles) {
      if (std::abs(p.weight - 1.0 / 3.0) > 1e-10) return 6;
    }
  }
  /* Test 4: Joint action sampling with deterministic dist. */
  {
    double probs1[] = {0.0, 1.0, 0.0};
    double probs2[] = {0.0, 0.0, 1.0};
    RivalActionDist dists[2];
    dists[0] = {probs1, 3};
    dists[1] = {probs2, 3};
    JointRivalAction ja;
    std::mt19937 rng(99);
    int status = sample_joint_action(dists, 2, rng, ja);
    if (status != kStatusOk) return 7;
    if (ja[0] != 1) return 8;
    if (ja[1] != 2) return 9;
  }
  /* Test 5: Config validation. */
  {
    WPomcpConfig cfg;
    cfg.particles_per_rival = {10};
    if (validate_config(cfg) != kStatusOk) return 10;

    WPomcpConfig bad_cfg;
    bad_cfg.num_simulations = 0;
    bad_cfg.particles_per_rival = {10};
    if (validate_config(bad_cfg) == kStatusOk) return 11;

    WPomcpConfig no_rivals;
    /* empty particles_per_rival => rival_count = 0 */
    if (validate_config(no_rivals) == kStatusOk) return 12;
  }
  /* Test 6: solve_raw with trivial 2-action game. */
  {
    int ppr[] = {5};
    int types[] = {0, 0, 0, 0, 0};
    int privs[] = {0, 1, 2, 3, 4};
    double weights[] = {0.2, 0.2, 0.2, 0.2, 0.2};
    double rival_ap[] = {0.5, 0.5};
    double rewards[] = {1.0, 0.0};  /* action 0 dominates */

    double out_vals[2];
    int out_best;
    double out_root;

    int status = solve_raw(
        1, ppr, types, privs, weights,
        0, 100.0,
        2, rival_ap, rewards,
        200, 0.99, 1.0, 1.0, 20, 0.5,
        42,
        out_vals, &out_best, &out_root);
    if (status != kStatusOk) return 13;
    if (out_best != 0) return 14;  /* action 0 should dominate */
    if (out_vals[0] <= out_vals[1]) return 15;
  }
  /* Test 7: |R| = 1 recovery (heads-up is the degenerate case). */
  {
    /* Single rival, single particle: degenerate case must not crash. */
    int ppr[] = {1};
    int types[] = {0};
    int privs[] = {0};
    double weights[] = {1.0};
    double rival_ap[] = {1.0};
    double rewards[] = {0.5};

    double out_vals[1];
    int out_best;
    double out_root;

    int status = solve_raw(
        1, ppr, types, privs, weights,
        0, 50.0,
        1, rival_ap, rewards,
        10, 0.99, 1.0, 1.0, 5, 0.5,
        7,
        out_vals, &out_best, &out_root);
    if (status != kStatusOk) return 16;
    if (out_best != 0) return 17;
  }
  return kStatusOk;
}

#endif /* WPOMCP_SELF_TEST */

} /* namespace wpomcp */
```

- [ ] **Step 2: Compile self-test**

```
clang++ -std=c++17 -DWPOMCP_SELF_TEST -I. -o wpomcp_test -x c++ - <<EOF
#include "WPomcpSolver.hpp"
#include <cstdio>
int main() {
  int r = wpomcp::self_test();
  printf("self_test: %d\n", r);
  return r;
}
EOF
```

Expected: exits 0, prints `self_test: 0`.

- [ ] **Step 3: Verify no memory issues**

Run under AddressSanitizer:
```
clang++ -std=c++17 -DWPOMCP_SELF_TEST -fsanitize=address -I. -o wpomcp_asan - ...
```

---

### Task 7: JNI Entry Points

**Files:**
- Modify: `src/main/native/jni/HoldemPomcpNativeBindings.cpp`

This file is created by Phase 3a for PFT-DPW. Task 7 adds W-POMCP entry points.

- [ ] **Step 1: Write the JNI binding**

Add the following JNI method to `HoldemPomcpNativeBindings.cpp`:

```cpp
/*
 * JNI method: sicfun.holdem.HoldemPomcpNativeBindings.solveWPomcp
 *
 * Bridges JVM arrays to wpomcp::solve_raw(). Uses critical array access
 * for zero-copy. Releases arrays in reverse acquisition order.
 *
 * Java signature:
 *   static native int solveWPomcp(
 *       int rivalCount,
 *       int[] particlesPerRival,
 *       int[] particleTypes,
 *       int[] particlePrivStates,
 *       double[] particleWeights,
 *       int pubStreet,
 *       double pubPot,
 *       int numHeroActions,
 *       double[] rivalActionProbs,
 *       double[] rewards,
 *       int numSimulations,
 *       double discount,
 *       double exploration,
 *       double rMax,
 *       int maxDepth,
 *       double essThreshold,
 *       long seed,
 *       double[] outActionValues,
 *       int[] outBestAction,
 *       double[] outRootValue
 *   );
 */

#include "WPomcpSolver.hpp"

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemPomcpNativeBindings_solveWPomcp(
    JNIEnv* env,
    jclass /* cls */,
    jint rival_count,
    jintArray j_particles_per_rival,
    jintArray j_particle_types,
    jintArray j_particle_priv_states,
    jdoubleArray j_particle_weights,
    jint pub_street,
    jdouble pub_pot,
    jint num_hero_actions,
    jdoubleArray j_rival_action_probs,
    jdoubleArray j_rewards,
    jint num_simulations,
    jdouble discount,
    jdouble exploration,
    jdouble r_max,
    jint max_depth,
    jdouble ess_threshold,
    jlong seed,
    jdoubleArray j_out_action_values,
    jintArray j_out_best_action,
    jdoubleArray j_out_root_value) {

  /* Null checks on all array arguments. */
  if (j_particles_per_rival == nullptr || j_particle_types == nullptr ||
      j_particle_priv_states == nullptr || j_particle_weights == nullptr ||
      j_rival_action_probs == nullptr || j_rewards == nullptr ||
      j_out_action_values == nullptr || j_out_best_action == nullptr ||
      j_out_root_value == nullptr) {
    return wpomcp::kStatusNullArray;
  }

  /* Acquire critical arrays (zero-copy access to JVM heap). */
  jint* ppr = static_cast<jint*>(
      env->GetPrimitiveArrayCritical(j_particles_per_rival, nullptr));
  jint* ptypes = static_cast<jint*>(
      env->GetPrimitiveArrayCritical(j_particle_types, nullptr));
  jint* pprivs = static_cast<jint*>(
      env->GetPrimitiveArrayCritical(j_particle_priv_states, nullptr));
  jdouble* pweights = static_cast<jdouble*>(
      env->GetPrimitiveArrayCritical(j_particle_weights, nullptr));
  jdouble* rap = static_cast<jdouble*>(
      env->GetPrimitiveArrayCritical(j_rival_action_probs, nullptr));
  jdouble* rew = static_cast<jdouble*>(
      env->GetPrimitiveArrayCritical(j_rewards, nullptr));
  jdouble* out_vals = static_cast<jdouble*>(
      env->GetPrimitiveArrayCritical(j_out_action_values, nullptr));
  jint* out_best = static_cast<jint*>(
      env->GetPrimitiveArrayCritical(j_out_best_action, nullptr));
  jdouble* out_root = static_cast<jdouble*>(
      env->GetPrimitiveArrayCritical(j_out_root_value, nullptr));

  if (ppr == nullptr || ptypes == nullptr || pprivs == nullptr ||
      pweights == nullptr || rap == nullptr || rew == nullptr ||
      out_vals == nullptr || out_best == nullptr || out_root == nullptr) {
    /* Release any successfully acquired arrays before returning. */
    if (out_root) env->ReleasePrimitiveArrayCritical(j_out_root_value, out_root, JNI_ABORT);
    if (out_best) env->ReleasePrimitiveArrayCritical(j_out_best_action, out_best, JNI_ABORT);
    if (out_vals) env->ReleasePrimitiveArrayCritical(j_out_action_values, out_vals, JNI_ABORT);
    if (rew) env->ReleasePrimitiveArrayCritical(j_rewards, rew, JNI_ABORT);
    if (rap) env->ReleasePrimitiveArrayCritical(j_rival_action_probs, rap, JNI_ABORT);
    if (pweights) env->ReleasePrimitiveArrayCritical(j_particle_weights, pweights, JNI_ABORT);
    if (pprivs) env->ReleasePrimitiveArrayCritical(j_particle_priv_states, pprivs, JNI_ABORT);
    if (ptypes) env->ReleasePrimitiveArrayCritical(j_particle_types, ptypes, JNI_ABORT);
    if (ppr) env->ReleasePrimitiveArrayCritical(j_particles_per_rival, ppr, JNI_ABORT);
    return wpomcp::kStatusReadFailure;
  }

  /* Call the solver. */
  int status = wpomcp::solve_raw(
      rival_count, ppr, ptypes, pprivs, pweights,
      pub_street, pub_pot,
      num_hero_actions, rap, rew,
      num_simulations, discount, exploration, r_max,
      max_depth, ess_threshold, seed,
      out_vals, out_best, out_root);

  /* Release critical arrays in reverse order. Commit output arrays only on success. */
  int release_mode = (status == wpomcp::kStatusOk) ? 0 : JNI_ABORT;
  env->ReleasePrimitiveArrayCritical(j_out_root_value, out_root, release_mode);
  env->ReleasePrimitiveArrayCritical(j_out_best_action, out_best, release_mode);
  env->ReleasePrimitiveArrayCritical(j_out_action_values, out_vals, release_mode);
  env->ReleasePrimitiveArrayCritical(j_rewards, rew, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(j_rival_action_probs, rap, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(j_particle_weights, pweights, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(j_particle_priv_states, pprivs, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(j_particle_types, ptypes, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(j_particles_per_rival, ppr, JNI_ABORT);

  if (status == wpomcp::kStatusOk) {
    g_last_engine_code.store(kEngineCpu);
  }

  return status;
}

/* Self-test entry point for JNI. */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemPomcpNativeBindings_selfTestWPomcp(
    JNIEnv* /* env */,
    jclass /* cls */) {
#ifdef WPOMCP_SELF_TEST
  return wpomcp::self_test();
#else
  return wpomcp::kStatusOk;  /* self-test not compiled in */
#endif
}
```

- [ ] **Step 2: Add Java JNI class (if not already present from Phase 3a)**

File: `src/main/java/sicfun/holdem/HoldemPomcpNativeBindings.java`

```java
package sicfun.holdem;

/**
 * JNI bindings for POMDP native solvers (PFT-DPW and W-POMCP).
 * Loaded from sicfun_pomcp_native.dll (CPU path).
 */
public class HoldemPomcpNativeBindings {
    /** W-POMCP multi-agent factored particle filter search. */
    public static native int solveWPomcp(
        int rivalCount,
        int[] particlesPerRival,
        int[] particleTypes,
        int[] particlePrivStates,
        double[] particleWeights,
        int pubStreet,
        double pubPot,
        int numHeroActions,
        double[] rivalActionProbs,
        double[] rewards,
        int numSimulations,
        double discount,
        double exploration,
        double rMax,
        int maxDepth,
        double essThreshold,
        long seed,
        double[] outActionValues,
        int[] outBestAction,
        double[] outRootValue
    );

    /** Run C++ self-test for W-POMCP. Returns 0 on success. */
    public static native int selfTestWPomcp();

    /** Returns engine code of last successful computation (1=CPU). */
    public static native int lastEngineCode();
}
```

- [ ] **Step 3: Add to build script**

Modify `src/main/native/build-windows-llvm.ps1` to compile
`HoldemPomcpNativeBindings.cpp` into `sicfun_pomcp_native.dll`:

```powershell
# W-POMCP / PFT-DPW POMDP solver DLL
clang++ -std=c++17 -O3 -shared -DWPOMCP_SELF_TEST `
  -I"$javaHome/include" -I"$javaHome/include/win32" `
  -o sicfun_pomcp_native.dll `
  HoldemPomcpNativeBindings.cpp
```

- [ ] **Step 4: Verify DLL builds and self-test passes via JNI**

---

### Task 8: Scala Wrapper

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/solver/WPomcpRuntime.scala`

- [ ] **Step 1: Write the Scala wrapper**

```scala
package sicfun.holdem.strategic.solver

import sicfun.holdem.HoldemPomcpNativeBindings
import java.util.concurrent.atomic.AtomicReference

/** Runtime wrapper for W-POMCP native solver via JNI.
  *
  * Implements Def 56 (factored particle filtering across rivals) by
  * delegating to the C++ WPomcpSolver via JNI. The Scala side constructs
  * the particle arrays and action distributions; the C++ side runs the
  * Monte Carlo tree search.
  *
  * Thread Safety:
  * Library loading is guarded by AtomicReference CAS. The solve call
  * itself is stateless (creates a fresh solver per call), so concurrent
  * calls from different threads are safe.
  */
private[strategic] object WPomcpRuntime:

  /** Configuration for a W-POMCP search.
    *
    * @param numSimulations  Monte Carlo simulations per search call
    * @param discount        gamma: discount factor in (0, 1)
    * @param exploration     UCB1 exploration constant c >= 0
    * @param rMax            maximum absolute reward (positive)
    * @param maxDepth        maximum tree depth per simulation
    * @param essThreshold    ESS ratio for resampling trigger in (0, 1]
    * @param seed            RNG seed for reproducibility
    */
  final case class Config(
      numSimulations: Int = 1000,
      discount: Double = 0.99,
      exploration: Double = 1.0,
      rMax: Double = 1.0,
      maxDepth: Int = 50,
      essThreshold: Double = 0.5,
      seed: Long = 42L
  ):
    require(numSimulations > 0, s"numSimulations must be positive, got $numSimulations")
    require(discount > 0.0 && discount < 1.0, s"discount must be in (0,1), got $discount")
    require(exploration >= 0.0, s"exploration must be non-negative, got $exploration")
    require(rMax > 0.0, s"rMax must be positive, got $rMax")
    require(maxDepth > 0, s"maxDepth must be positive, got $maxDepth")
    require(essThreshold > 0.0 && essThreshold <= 1.0,
      s"essThreshold must be in (0,1], got $essThreshold")

  /** Per-rival particle set for the factored belief.
    *
    * @param rivalTypes   discrete type index per particle (theta^{R,i})
    * @param privStates   discrete private state index per particle
    * @param weights      importance weights per particle (positive, will be normalized)
    */
  final case class RivalParticles(
      rivalTypes: Array[Int],
      privStates: Array[Int],
      weights: Array[Double]
  ):
    require(rivalTypes.length == privStates.length && rivalTypes.length == weights.length,
      "All particle arrays must have the same length")
    require(rivalTypes.nonEmpty, "Particle set must be non-empty")
    def particleCount: Int = rivalTypes.length

  /** Public state component of the factored belief. */
  final case class PublicState(street: Int, pot: Double)

  /** Input to a W-POMCP search.
    *
    * @param publicState         shared public state
    * @param rivalParticles      per-rival particle sets, length = |R|
    * @param heroActionCount     number of available hero actions
    * @param rivalActionProbs    per-rival action probabilities, flat [rival][action]
    * @param rewards             reward per hero action
    */
  final case class SearchInput(
      publicState: PublicState,
      rivalParticles: IndexedSeq[RivalParticles],
      heroActionCount: Int,
      rivalActionProbs: Array[Double],
      rewards: Array[Double]
  ):
    require(rivalParticles.nonEmpty, "Must have at least one rival")
    require(rivalParticles.size <= 8, s"Max 8 rivals, got ${rivalParticles.size}")
    require(heroActionCount > 0, s"Must have at least one hero action")
    require(rewards.length == heroActionCount,
      s"rewards length ${rewards.length} != heroActionCount $heroActionCount")
    def rivalCount: Int = rivalParticles.size

  /** Result of a W-POMCP search.
    *
    * @param actionValues  estimated Q(root, a) per hero action
    * @param bestAction    argmax action index
    * @param rootValue     max Q(root, a)
    */
  final case class SearchResult(
      actionValues: Array[Double],
      bestAction: Int,
      rootValue: Double
  )

  /* Library loading state. */
  private val PathProperty = "sicfun.pomcp.native.path"
  private val PathEnv = "sicfun_POMCP_NATIVE_PATH"
  private val LibProperty = "sicfun.pomcp.native.lib"
  private val LibEnv = "sicfun_POMCP_NATIVE_LIB"
  private val DefaultLib = "sicfun_pomcp_native"

  private val loadState: AtomicReference[Option[Either[String, Unit]]] =
    AtomicReference(None)

  /** Check if the native library is available. */
  def isAvailable: Boolean = ensureLoaded().isRight

  /** Load the native library if not already loaded. */
  def ensureLoaded(): Either[String, Unit] =
    loadState.get() match
      case Some(result) => result
      case None =>
        val result =
          try
            val libName = sys.props.getOrElse(LibProperty,
              sys.env.getOrElse(LibEnv, DefaultLib))
            val libPath = sys.props.get(PathProperty)
              .orElse(sys.env.get(PathEnv))
            libPath match
              case Some(path) => System.load(s"$path/$libName.dll")
              case None       => System.loadLibrary(libName)
            Right(())
          catch
            case e: UnsatisfiedLinkError =>
              Left(s"W-POMCP native library not available: ${e.getMessage}")
        loadState.compareAndSet(None, Some(result))
        loadState.get().get

  /** Run W-POMCP search.
    *
    * Returns Left(errorMessage) on failure, Right(SearchResult) on success.
    */
  def solve(input: SearchInput, config: Config): Either[String, SearchResult] =
    ensureLoaded() match
      case Left(err) => Left(err)
      case Right(()) =>
        /* Flatten per-rival particles into flat arrays. */
        val totalParticles = input.rivalParticles.map(_.particleCount).sum
        val particlesPerRival = input.rivalParticles.map(_.particleCount).toArray
        val allTypes = new Array[Int](totalParticles)
        val allPrivs = new Array[Int](totalParticles)
        val allWeights = new Array[Double](totalParticles)
        var offset = 0
        for rp <- input.rivalParticles do
          System.arraycopy(rp.rivalTypes, 0, allTypes, offset, rp.particleCount)
          System.arraycopy(rp.privStates, 0, allPrivs, offset, rp.particleCount)
          System.arraycopy(rp.weights, 0, allWeights, offset, rp.particleCount)
          offset += rp.particleCount

        /* Allocate output arrays. */
        val outActionValues = new Array[Double](input.heroActionCount)
        val outBestAction = new Array[Int](1)
        val outRootValue = new Array[Double](1)

        val status = HoldemPomcpNativeBindings.solveWPomcp(
          input.rivalCount,
          particlesPerRival,
          allTypes,
          allPrivs,
          allWeights,
          input.publicState.street,
          input.publicState.pot,
          input.heroActionCount,
          input.rivalActionProbs,
          input.rewards,
          config.numSimulations,
          config.discount,
          config.exploration,
          config.rMax,
          config.maxDepth,
          config.essThreshold,
          config.seed,
          outActionValues,
          outBestAction,
          outRootValue
        )

        if status == 0 then
          Right(SearchResult(outActionValues, outBestAction(0), outRootValue(0)))
        else
          Left(describeStatus(status))

  /** Describe a native status code. */
  private def describeStatus(code: Int): String = code match
    case 0   => "OK"
    case 100 => "Null array argument"
    case 101 => "Array length mismatch"
    case 102 => "JNI array read failure"
    case 124 => "JNI array write failure"
    case 160 => "Invalid configuration"
    case 170 => "Invalid particle count"
    case 171 => "Invalid rival count"
    case 172 => "Invalid action count"
    case 173 => "Degenerate particle weights (all zero)"
    case 174 => "Maximum rivals exceeded (max 8)"
    case 175 => "Simulation overflow"
    case _   => s"Unknown native error code: $code"
```

- [ ] **Step 2: Verify compilation**

```
sbt compile
```

Expected: compiles (JNI class must exist from Task 7 step 2). Runtime requires
the native DLL, tested in Task 9.

---

### Task 9: Test Suite

**Files:**
- Create: `src/test/scala/sicfun/holdem/strategic/solver/WPomcpRuntimeTest.scala`

- [ ] **Step 1: Write the complete test suite**

```scala
package sicfun.holdem.strategic.solver

class WPomcpRuntimeTest extends munit.FunSuite:

  /* --- Config validation tests (pure Scala, no native required) --- */

  test("Config rejects zero simulations"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(numSimulations = 0)

  test("Config rejects discount outside (0,1)"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(discount = 0.0)
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(discount = 1.0)
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(discount = -0.5)

  test("Config rejects negative exploration"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(exploration = -1.0)

  test("Config rejects non-positive rMax"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(rMax = 0.0)

  test("Config rejects zero maxDepth"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(maxDepth = 0)

  test("Config rejects essThreshold outside (0,1]"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(essThreshold = 0.0)
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(essThreshold = 1.5)

  test("Config accepts valid parameters"):
    val cfg = WPomcpRuntime.Config(
      numSimulations = 500,
      discount = 0.95,
      exploration = 2.0,
      rMax = 100.0,
      maxDepth = 30,
      essThreshold = 0.3,
      seed = 12345L
    )
    assertEquals(cfg.numSimulations, 500)
    assertEquals(cfg.discount, 0.95)

  /* --- RivalParticles validation tests --- */

  test("RivalParticles rejects mismatched array lengths"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.RivalParticles(
        rivalTypes = Array(0, 1),
        privStates = Array(0),
        weights = Array(0.5, 0.5)
      )

  test("RivalParticles rejects empty arrays"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.RivalParticles(
        rivalTypes = Array.empty,
        privStates = Array.empty,
        weights = Array.empty
      )

  test("RivalParticles accepts valid input"):
    val rp = WPomcpRuntime.RivalParticles(
      rivalTypes = Array(0, 1, 2),
      privStates = Array(5, 6, 7),
      weights = Array(0.3, 0.3, 0.4)
    )
    assertEquals(rp.particleCount, 3)

  /* --- SearchInput validation tests --- */

  test("SearchInput rejects empty rival list"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.SearchInput(
        publicState = WPomcpRuntime.PublicState(0, 100.0),
        rivalParticles = IndexedSeq.empty,
        heroActionCount = 2,
        rivalActionProbs = Array(0.5, 0.5),
        rewards = Array(1.0, 0.0)
      )

  test("SearchInput rejects >8 rivals"):
    val rp = WPomcpRuntime.RivalParticles(Array(0), Array(0), Array(1.0))
    intercept[IllegalArgumentException]:
      WPomcpRuntime.SearchInput(
        publicState = WPomcpRuntime.PublicState(0, 100.0),
        rivalParticles = IndexedSeq.fill(9)(rp),
        heroActionCount = 2,
        rivalActionProbs = Array.fill(18)(1.0 / 2),
        rewards = Array(1.0, 0.0)
      )

  test("SearchInput rejects zero hero actions"):
    val rp = WPomcpRuntime.RivalParticles(Array(0), Array(0), Array(1.0))
    intercept[IllegalArgumentException]:
      WPomcpRuntime.SearchInput(
        publicState = WPomcpRuntime.PublicState(0, 100.0),
        rivalParticles = IndexedSeq(rp),
        heroActionCount = 0,
        rivalActionProbs = Array(1.0),
        rewards = Array.empty
      )

  test("SearchInput rejects mismatched rewards length"):
    val rp = WPomcpRuntime.RivalParticles(Array(0), Array(0), Array(1.0))
    intercept[IllegalArgumentException]:
      WPomcpRuntime.SearchInput(
        publicState = WPomcpRuntime.PublicState(0, 100.0),
        rivalParticles = IndexedSeq(rp),
        heroActionCount = 2,
        rivalActionProbs = Array(0.5, 0.5),
        rewards = Array(1.0)  /* length 1, expected 2 */
      )

  /* --- Native tests (require DLL) --- */

  private def nativeAvailable: Boolean = WPomcpRuntime.isAvailable

  test("native: isAvailable reports status"):
    /* This test always passes -- it just documents whether native is loaded. */
    val available = WPomcpRuntime.isAvailable
    if !available then
      println("[WPomcpRuntimeTest] Native library not available, skipping native tests")

  test("native: solve with dominant action returns correct best action"):
    assume(nativeAvailable, "Native library not available")
    val rp = WPomcpRuntime.RivalParticles(
      rivalTypes = Array(0, 0, 0, 0, 0),
      privStates = Array(0, 1, 2, 3, 4),
      weights = Array(0.2, 0.2, 0.2, 0.2, 0.2)
    )
    val input = WPomcpRuntime.SearchInput(
      publicState = WPomcpRuntime.PublicState(street = 0, pot = 100.0),
      rivalParticles = IndexedSeq(rp),
      heroActionCount = 2,
      rivalActionProbs = Array(0.5, 0.5),
      rewards = Array(1.0, 0.0)  /* action 0 dominates */
    )
    val config = WPomcpRuntime.Config(numSimulations = 200, seed = 42L)

    val result = WPomcpRuntime.solve(input, config)
    assert(result.isRight, s"Expected Right, got $result")
    val sr = result.toOption.get
    assertEquals(sr.bestAction, 0, "Action 0 should dominate")
    assert(sr.actionValues(0) > sr.actionValues(1),
      s"Action 0 value ${sr.actionValues(0)} should exceed action 1 value ${sr.actionValues(1)}")

  test("native: solve with equal rewards returns valid action"):
    assume(nativeAvailable, "Native library not available")
    val rp = WPomcpRuntime.RivalParticles(
      rivalTypes = Array(0, 0, 0),
      privStates = Array(0, 1, 2),
      weights = Array(1.0 / 3, 1.0 / 3, 1.0 / 3)
    )
    val input = WPomcpRuntime.SearchInput(
      publicState = WPomcpRuntime.PublicState(0, 50.0),
      rivalParticles = IndexedSeq(rp),
      heroActionCount = 3,
      rivalActionProbs = Array(1.0 / 3, 1.0 / 3, 1.0 / 3),
      rewards = Array(1.0, 1.0, 1.0)
    )
    val config = WPomcpRuntime.Config(numSimulations = 100, seed = 7L)

    val result = WPomcpRuntime.solve(input, config)
    assert(result.isRight, s"Expected Right, got $result")
    val sr = result.toOption.get
    assert(sr.bestAction >= 0 && sr.bestAction < 3,
      s"Best action ${sr.bestAction} out of range [0,3)")

  test("native: multiway 3-rival solve"):
    assume(nativeAvailable, "Native library not available")
    val rp1 = WPomcpRuntime.RivalParticles(Array(0, 1), Array(0, 1), Array(0.5, 0.5))
    val rp2 = WPomcpRuntime.RivalParticles(Array(0, 1), Array(2, 3), Array(0.5, 0.5))
    val rp3 = WPomcpRuntime.RivalParticles(Array(0, 1), Array(4, 5), Array(0.5, 0.5))
    val input = WPomcpRuntime.SearchInput(
      publicState = WPomcpRuntime.PublicState(1, 200.0),
      rivalParticles = IndexedSeq(rp1, rp2, rp3),
      heroActionCount = 2,
      rivalActionProbs = Array.fill(6)(0.5),
      rewards = Array(2.0, 0.5)  /* action 0 dominates */
    )
    val config = WPomcpRuntime.Config(numSimulations = 300, seed = 99L)

    val result = WPomcpRuntime.solve(input, config)
    assert(result.isRight, s"Expected Right, got $result")
    val sr = result.toOption.get
    assertEquals(sr.bestAction, 0)
    assert(sr.actionValues.length == 2)

  test("native: |R|=1 heads-up degenerate case"):
    assume(nativeAvailable, "Native library not available")
    val rp = WPomcpRuntime.RivalParticles(Array(0), Array(0), Array(1.0))
    val input = WPomcpRuntime.SearchInput(
      publicState = WPomcpRuntime.PublicState(0, 50.0),
      rivalParticles = IndexedSeq(rp),
      heroActionCount = 1,
      rivalActionProbs = Array(1.0),
      rewards = Array(0.5)
    )
    val config = WPomcpRuntime.Config(numSimulations = 10, seed = 7L)

    val result = WPomcpRuntime.solve(input, config)
    assert(result.isRight, s"Expected Right, got $result")
    val sr = result.toOption.get
    assertEquals(sr.bestAction, 0)

  test("native: deterministic seed produces reproducible results"):
    assume(nativeAvailable, "Native library not available")
    val rp = WPomcpRuntime.RivalParticles(
      rivalTypes = Array(0, 1, 2, 0, 1),
      privStates = Array(0, 1, 2, 3, 4),
      weights = Array(0.2, 0.2, 0.2, 0.2, 0.2)
    )
    val input = WPomcpRuntime.SearchInput(
      publicState = WPomcpRuntime.PublicState(0, 100.0),
      rivalParticles = IndexedSeq(rp),
      heroActionCount = 3,
      rivalActionProbs = Array(0.4, 0.3, 0.3),
      rewards = Array(1.0, 0.5, 0.8)
    )
    val config = WPomcpRuntime.Config(numSimulations = 100, seed = 42L)

    val r1 = WPomcpRuntime.solve(input, config)
    val r2 = WPomcpRuntime.solve(input, config)
    assert(r1.isRight && r2.isRight)
    val sr1 = r1.toOption.get
    val sr2 = r2.toOption.get
    assertEquals(sr1.bestAction, sr2.bestAction,
      "Same seed must produce same best action")
    for i <- sr1.actionValues.indices do
      assertEqualsDouble(sr1.actionValues(i), sr2.actionValues(i), 1e-15,
        s"Action value $i differs across runs with same seed")

  test("native: library unavailable returns Left"):
    /* This test validates the error path when native is not loaded.
     * It is meaningful only in environments without the DLL. */
    if !nativeAvailable then
      val rp = WPomcpRuntime.RivalParticles(Array(0), Array(0), Array(1.0))
      val input = WPomcpRuntime.SearchInput(
        publicState = WPomcpRuntime.PublicState(0, 50.0),
        rivalParticles = IndexedSeq(rp),
        heroActionCount = 1,
        rivalActionProbs = Array(1.0),
        rewards = Array(0.5)
      )
      val config = WPomcpRuntime.Config()
      val result = WPomcpRuntime.solve(input, config)
      assert(result.isLeft, "Expected Left when native unavailable")
```

- [ ] **Step 2: Run the test suite**

```
sbt "testOnly sicfun.holdem.strategic.solver.WPomcpRuntimeTest"
```

Expected:
- Config/validation tests: PASS (pure Scala, no native required)
- Native tests: PASS if DLL is built, SKIP (assume) if not

---

### Task 10: Backward Compatibility and Integration Verification

- [ ] **Step 1: Verify |R|=1 recovery**

The single-rival case (|R|=1) must produce valid results identical in structure
to what a non-factored particle filter would produce. This is validated by:

1. Self-test #7 in WPomcpSolver.hpp (C++ level)
2. "|R|=1 heads-up degenerate case" test in WPomcpRuntimeTest (JNI level)

Both verify that a single rival with a single particle produces correct output
without crashing or returning error codes.

- [ ] **Step 2: Verify error bound consistency (Def 55)**

The error bound `|V*(b) - V*(b_hat^C)| <= R_max / (1-gamma) * sqrt(D_2 / 2)`
is a theoretical guarantee from the spec. We verify the implementation respects
this by checking that:

1. Increasing particle count C reduces variance in action value estimates.
2. The solver's output values are bounded by `R_max / (1-gamma)`.

Add to WPomcpRuntimeTest:

```scala
  test("native: more particles reduce action value variance"):
    assume(nativeAvailable, "Native library not available")

    def runWithParticles(n: Int, seed: Long): Array[Double] =
      val rp = WPomcpRuntime.RivalParticles(
        rivalTypes = Array.fill(n)(0),
        privStates = Array.tabulate(n)(identity),
        weights = Array.fill(n)(1.0 / n)
      )
      val input = WPomcpRuntime.SearchInput(
        publicState = WPomcpRuntime.PublicState(0, 100.0),
        rivalParticles = IndexedSeq(rp),
        heroActionCount = 2,
        rivalActionProbs = Array(0.6, 0.4),
        rewards = Array(1.0, 0.5)
      )
      val config = WPomcpRuntime.Config(numSimulations = 200, seed = seed)
      WPomcpRuntime.solve(input, config).toOption.get.actionValues

    /* Run multiple seeds and measure variance. */
    val seeds = (1L to 10L)
    val variance5 = {
      val runs = seeds.map(s => runWithParticles(5, s).head)
      val mean = runs.sum / runs.size
      runs.map(v => (v - mean) * (v - mean)).sum / runs.size
    }
    val variance50 = {
      val runs = seeds.map(s => runWithParticles(50, s).head)
      val mean = runs.sum / runs.size
      runs.map(v => (v - mean) * (v - mean)).sum / runs.size
    }
    /* More particles should reduce variance (or at least not increase it much). */
    assert(variance50 <= variance5 * 2.0,
      s"50-particle variance $variance50 should not greatly exceed 5-particle variance $variance5")

  test("native: output values bounded by R_max / (1-gamma)"):
    assume(nativeAvailable, "Native library not available")
    val rMax = 10.0
    val gamma = 0.9
    val bound = rMax / (1.0 - gamma)  /* = 100 */
    val rp = WPomcpRuntime.RivalParticles(
      rivalTypes = Array(0, 1, 2),
      privStates = Array(0, 1, 2),
      weights = Array(1.0 / 3, 1.0 / 3, 1.0 / 3)
    )
    val input = WPomcpRuntime.SearchInput(
      publicState = WPomcpRuntime.PublicState(0, 100.0),
      rivalParticles = IndexedSeq(rp),
      heroActionCount = 2,
      rivalActionProbs = Array(0.5, 0.5),
      rewards = Array(rMax, -rMax)
    )
    val config = WPomcpRuntime.Config(
      numSimulations = 500, rMax = rMax, discount = gamma, seed = 42L
    )
    val result = WPomcpRuntime.solve(input, config)
    assert(result.isRight)
    for v <- result.toOption.get.actionValues do
      assert(math.abs(v) <= bound * 1.01,
        s"Action value $v exceeds bound $bound")
```

- [ ] **Step 3: Verify existing test suite is unaffected**

```
sbt test
```

The formal layer is strictly additive. No existing tests should break.

- [ ] **Step 4: Compile native DLL and run full integration**

```
# Build DLL
cd src/main/native
./build-windows-llvm.ps1

# Run all strategic solver tests
sbt "testOnly sicfun.holdem.strategic.solver.*"
```

---

## LOC Estimate

| Component | Estimated LOC |
|-----------|---------------|
| WPomcpSolver.hpp (C++) | ~650 |
| HoldemPomcpNativeBindings.cpp (JNI addition) | ~120 |
| HoldemPomcpNativeBindings.java (JNI class) | ~30 |
| WPomcpRuntime.scala (Scala wrapper) | ~180 |
| WPomcpRuntimeTest.scala (tests) | ~220 |
| **Total** | **~1,200** |

---

## Invariants (enforced across all tasks)

1. **Multiway mandate:** |R| >= 1 always. No special-casing for heads-up.
2. **Factored independence:** Per-rival particle sets are updated independently. Cross-rival dependency enters only through `x^pub`.
3. **Weight normalization:** After every update, per-rival weights sum to 1.0.
4. **ESS-triggered resampling:** Systematic resampling fires when ESS < threshold * C_i.
5. **No engine imports:** `WPomcpRuntime.scala` lives in `sicfun.holdem.strategic.solver` and does NOT import `sicfun.holdem.engine` or `sicfun.holdem.runtime`.
6. **Status code protocol:** All C++ functions return integer status codes matching the JNI error protocol (0=ok, 100+=error).
7. **Deterministic seeds:** Same seed produces bit-identical results.
8. **Backward compatibility:** |R|=1 is a valid degenerate case that must work correctly.
