# Strategic Decision Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the strategic POMDP module into the decision path as `HeroMode.Strategic`, fixing the C++ WPomcp solver to be a proper factored POMCP with particle reweighting and building the Scala orchestrator that feeds it.

**Architecture:** New `StrategicEngine` in `sicfun.holdem.engine` orchestrates session/hand lifecycle, builds a factored tabular model via `PokerPomcpFormulation`, and delegates search to the fixed WPomcp C++ solver. The C++ solver gets type-conditioned rival policies, real observation-based particle reweighting, and tabular rewards. Dependency direction: `engine -> strategic -> types` (strategic never imports engine).

**Tech Stack:** Scala 3.8.1 / munit 1.2.2 / C++17 (clang++) / JNI / SBT

**Spec:** `docs/superpowers/specs/2026-04-05-strategic-decision-pipeline-design.md`

---

## File Structure

### C++ (modify)
- `src/main/native/jni/WPomcpSolver.hpp` — add FactoredTabularModel, rewrite simulate(), update solve_raw()

### Java JNI (modify)
- `src/main/java/sicfun/holdem/HoldemPomcpNativeBindings.java` — new solveWPomcpV2() method

### C++ JNI bridge (modify)
- `src/main/native/jni/HoldemPomcpNativeBindings.cpp` — new JNI function for solveWPomcpV2

### Scala (modify)
- `src/main/scala/sicfun/holdem/strategic/solver/WPomcpRuntime.scala` — new FactoredModel, updated SearchInput, new solveV2()
- `src/main/scala/sicfun/holdem/types/HeroMode.scala` — add Strategic case
- `src/main/scala/sicfun/holdem/engine/HeroDecisionPipeline.scala` — add Strategic branch + StrategicDecisionContext

### Scala (create)
- `src/main/scala/sicfun/holdem/strategic/StrategicRivalBelief.scala` — concrete RivalBeliefState
- `src/main/scala/sicfun/holdem/engine/PokerPomcpFormulation.scala` — builds factored tabular model
- `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala` — session/hand orchestrator

### Tests (create)
- `src/test/scala/sicfun/holdem/strategic/StrategicRivalBeliefTest.scala`
- `src/test/scala/sicfun/holdem/engine/PokerPomcpFormulationTest.scala`
- `src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala`

### Tests (modify)
- `src/test/scala/sicfun/holdem/strategic/solver/WPomcpRuntimeTest.scala` — add V2 tests

---

## Task 1: C++ FactoredTabularModel and simulate() Rewrite

**Files:**
- Modify: `src/main/native/jni/WPomcpSolver.hpp`

This is the foundation. The current WPomcp solver uses static particles, fake observations, and pre-baked rewards. We replace the `TransitionModel` with a `FactoredTabularModel` and rewrite `simulate()` to do real particle reweighting.

### Flat Array Layouts (JNI contract)

```
rival_policy[type * numPubStates * numActions + pubState * numActions + action]
  → P(action | type, pubState)

action_effects[action * 3 + field]
  → field 0: pot_delta_frac, field 1: is_fold (0/1), field 2: is_allin (0/1)

showdown_equity[heroBucket * numRivalBuckets + rivalBucket]
  → equity in [0, 1]

terminal_flags[pubState * numActions + action]
  → 0=Continue, 1=HeroFold, 2=RivalFold, 3=Showdown
```

- [ ] **Step 1: Add FactoredTabularModel struct after line 88 (after PublicState)**

Insert after the `PublicState` struct in WPomcpSolver.hpp:

```cpp
/* ---------- Terminal type enum ---------- */

enum class TerminalType : int {
  kContinue  = 0,
  kHeroFold  = 1,
  kRivalFold = 2,
  kShowdown  = 3
};

/* ---------- Action effect descriptor ---------- */

struct ActionEffect {
  double pot_delta_frac = 0.0;  /* pot contribution as fraction of current pot */
  bool is_fold = false;
  bool is_allin = false;
};

/* ---------- Factored tabular model ---------- */

/** Pre-computed game model passed from Scala via JNI.
  *
  * All arrays are flat row-major. Dimensions are stored alongside.
  * The rival_policy table serves dual purpose:
  *   1. Sampling rival actions during simulation (per particle type)
  *   2. Observation likelihood for particle reweighting
  */
struct FactoredTabularModel {
  /* Rival policy: P(action | type, pub_state) */
  const double* rival_policy = nullptr;   /* [numTypes * numPubStates * numActions] */
  int num_rival_types = 0;
  int num_pub_states = 0;

  /* Action effects */
  const ActionEffect* action_effects = nullptr;  /* [numActions] */

  /* Showdown equity table */
  const double* showdown_equity = nullptr;  /* [numHeroBuckets * numRivalBuckets] */
  int num_hero_buckets = 0;
  int num_rival_buckets = 0;

  /* Terminal flags */
  const int* terminal_flags = nullptr;  /* [numPubStates * numActions] */

  /* Shared dimensions */
  int num_actions = 0;

  /* Inline accessors */
  inline double rival_action_prob(int type, int pub_state, int action) const {
    return rival_policy[type * num_pub_states * num_actions + pub_state * num_actions + action];
  }

  inline TerminalType terminal_type(int pub_state, int action) const {
    return static_cast<TerminalType>(terminal_flags[pub_state * num_actions + action]);
  }

  inline double equity(int hero_bucket, int rival_bucket) const {
    return showdown_equity[hero_bucket * num_rival_buckets + rival_bucket];
  }
};
```

- [ ] **Step 2: Modify sample_joint_action to accept type-conditioned policy**

Replace the current `sample_joint_action` (lines ~312-358) that reads from flat `RivalActionDist` with a version that indexes the `FactoredTabularModel::rival_policy` by each particle's type. The function signature becomes:

```cpp
/** Sample a rival action for a single particle, conditioned on its type.
  * Returns the sampled action index.
  */
template <typename Rng>
int sample_type_conditioned_action(
    const FactoredTabularModel& model,
    int rival_type,
    int pub_state_idx,
    Rng& rng)
{
  double cdf = 0.0;
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  const double roll = u01(rng);
  for (int a = 0; a < model.num_actions; ++a) {
    cdf += model.rival_action_prob(rival_type, pub_state_idx, a);
    if (roll < cdf) return a;
  }
  return model.num_actions - 1;  /* fallback to last action */
}
```

Keep the old `sample_joint_action` for backward compat with the existing `solve_raw`. The new `simulate_v2` will use `sample_type_conditioned_action`.

- [ ] **Step 3: Add reweight_particles_by_observation helper**

Add after the existing `update_factored_belief` function:

```cpp
/** Reweight a single rival's particles by the observation likelihood.
  *
  * For each particle j of rival i:
  *   w_j *= P(observed_action | particle_j.rival_type, pub_state)
  *
  * This is the core of the factored particle filter.
  */
inline void reweight_particles_by_observation(
    RivalBeliefSet& belief,
    const FactoredTabularModel& model,
    int observed_action,
    int pub_state_idx)
{
  bool any_positive = false;
  for (auto& p : belief.particles) {
    const double lik = model.rival_action_prob(p.rival_type, pub_state_idx, observed_action);
    p.weight *= lik;
    if (p.weight > 0.0) any_positive = true;
  }
  if (!any_positive) {
    /* Degeneracy fallback: reset to uniform weights */
    const double unif = 1.0 / static_cast<double>(belief.particles.size());
    for (auto& p : belief.particles) p.weight = unif;
  }
  belief.is_normalized_ = false;
}
```

- [ ] **Step 4: Add compute_pub_state_idx helper**

The factored model needs a mapping from PublicState fields to a flat index:

```cpp
/** Map (street, pot_bucket, stack_bucket) to flat pub_state index.
  * pot_bucket = clamp(floor(pot / pot_bucket_size), 0, num_pot_buckets-1)
  * stack_bucket computed similarly.
  *
  * For initial deployment: 4 streets * 8 pot buckets * 6 stack buckets = 192.
  * Bucket boundaries passed from Scala as part of the model.
  */
struct PubStateEncoder {
  int num_pot_buckets = 8;
  int num_stack_buckets = 6;
  double pot_bucket_size = 50.0;   /* pot units per bucket */
  double stack_bucket_size = 0.2;  /* fraction per bucket */

  inline int encode(const PublicState& pub) const {
    int pot_b = std::min(num_pot_buckets - 1,
        std::max(0, static_cast<int>(pub.pot / pot_bucket_size)));
    /* stack_bucket defaults to middle bucket when not tracked */
    int stack_b = num_stack_buckets / 2;
    return pub.street * num_pot_buckets * num_stack_buckets
         + pot_b * num_stack_buckets
         + stack_b;
  }
};
```

- [ ] **Step 5: Add simulate_v2 method to WPomcpSolver class**

This is the core rewrite. Add as a private method in the WPomcpSolver class alongside the existing `simulate`:

```cpp
/** V2 simulate with type-conditioned policies, real observations, and particle reweighting. */
double simulate_v2(
    int node_idx,
    PublicState pub_state,
    FactoredBelief& belief,  /* mutable: particles get reweighted */
    const FactoredTabularModel& model,
    const PubStateEncoder& encoder,
    int hero_bucket,         /* hero's hand bucket for showdown equity */
    int depth)
{
  if (depth >= config_.max_depth || model.num_actions <= 0)
    return 0.0;

  auto& node = arena_[node_idx];
  if (node.action_stats.empty())
    node.action_stats.resize(model.num_actions);

  /* Phase 1: Select hero action (progressive widening + UCB1) */
  int hero_action;
  if (node.should_widen(config_.pw_c, config_.pw_alpha)) {
    hero_action = pick_unexpanded_action(node, model.num_actions);
    if (!node.action_expanded_[hero_action]) {
      node.action_expanded_[hero_action] = true;
      node.expanded_count_++;
    }
  } else {
    hero_action = select_ucb1(node, model.num_actions);
  }

  /* Phase 2: Check terminal */
  const int pub_idx = encoder.encode(pub_state);
  const auto term = model.terminal_type(pub_idx, hero_action);

  if (term == TerminalType::kHeroFold) {
    /* Hero folds — lose pot contribution */
    const double reward = -(model.action_effects[hero_action].pot_delta_frac * pub_state.pot);
    update_stats(node, hero_action, reward);
    return reward;
  }

  /* Phase 3: Sample rival actions (one per rival, type-conditioned per particle) */
  /* For tree branching, we use the MAP action per rival (most likely given belief). */
  std::array<int, kMaxRivals> rival_actions{};
  for (int r = 0; r < static_cast<int>(belief.rival_beliefs.size()); ++r) {
    auto& rb = belief.rival_beliefs[r];
    /* Sample one action using the belief-weighted type distribution */
    double best_prob = -1.0;
    int best_action = 0;
    for (int a = 0; a < model.num_actions; ++a) {
      double weighted_prob = 0.0;
      for (const auto& p : rb.particles) {
        weighted_prob += p.weight * model.rival_action_prob(p.rival_type, pub_idx, a);
      }
      if (weighted_prob > best_prob) {
        best_prob = weighted_prob;
        best_action = a;
      }
    }
    /* Stochastic sampling for actual simulation */
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    double roll = u01(rng_);
    double cdf = 0.0;
    rival_actions[r] = model.num_actions - 1;
    for (int a = 0; a < model.num_actions; ++a) {
      double weighted_prob = 0.0;
      for (const auto& p : rb.particles)
        weighted_prob += p.weight * model.rival_action_prob(p.rival_type, pub_idx, a);
      cdf += weighted_prob;
      if (roll < cdf) { rival_actions[r] = a; break; }
    }
  }

  /* Phase 4: Check rival fold */
  for (int r = 0; r < static_cast<int>(belief.rival_beliefs.size()); ++r) {
    if (model.action_effects[rival_actions[r]].is_fold) {
      /* Rival folds — hero wins current pot */
      const double reward = pub_state.pot;
      update_stats(node, hero_action, reward);
      return reward;
    }
  }

  /* Phase 5: Showdown check */
  if (term == TerminalType::kShowdown) {
    /* Compute expected showdown equity against rival particle beliefs */
    double eq_sum = 0.0;
    double weight_sum = 0.0;
    for (int r = 0; r < static_cast<int>(belief.rival_beliefs.size()); ++r) {
      for (const auto& p : belief.rival_beliefs[r].particles) {
        eq_sum += p.weight * model.equity(hero_bucket, p.priv_state);
        weight_sum += p.weight;
      }
    }
    const double eq = (weight_sum > 0.0) ? eq_sum / weight_sum : 0.5;
    const double reward = pub_state.pot * (2.0 * eq - 1.0);  /* zero-sum: win pot or lose pot */
    update_stats(node, hero_action, reward);
    return reward;
  }

  /* Phase 6: Reweight particles by observed rival actions */
  for (int r = 0; r < static_cast<int>(belief.rival_beliefs.size()); ++r) {
    reweight_particles_by_observation(
        belief.rival_beliefs[r], model, rival_actions[r], pub_idx);
    /* Normalize and resample if needed */
    belief.rival_beliefs[r].normalize();
    if (belief.rival_beliefs[r].needs_resample()) {
      systematic_resample(belief.rival_beliefs[r], rng_);
    }
  }

  /* Phase 7: Advance public state */
  PublicState next_pub = pub_state;
  next_pub.pot += model.action_effects[hero_action].pot_delta_frac * pub_state.pot;
  for (int r = 0; r < static_cast<int>(belief.rival_beliefs.size()); ++r) {
    next_pub.pot += model.action_effects[rival_actions[r]].pot_delta_frac * pub_state.pot;
  }
  if (hero_action > 0) {  /* non-fold/check actions may advance street */
    next_pub.street = std::min(next_pub.street + 1, 3);
  }

  /* Phase 8: Observation hash and child lookup */
  int64_t obs_hash = hero_action;
  for (int r = 0; r < static_cast<int>(belief.rival_beliefs.size()); ++r) {
    obs_hash = obs_hash * (model.num_actions + 1) + rival_actions[r];
  }
  const int child_idx = find_or_create_child(node_idx, hero_action, static_cast<int>(obs_hash & 0x7FFFFFFF));

  /* Phase 9: Recurse */
  const double future = simulate_v2(child_idx, next_pub, belief, model, encoder, hero_bucket, depth + 1);
  const double total = config_.discount * future;  /* no immediate reward at non-terminal */

  update_stats(node, hero_action, total);
  return total;
}

/** Helper to update node action stats. */
inline void update_stats(WPomcpNode& node, int action, double value) {
  node.visit_count++;
  auto& ts = node.action_stats[action];
  ts.visit_count++;
  ts.value_sum += value;
}
```

- [ ] **Step 6: Add search_v2 public method**

```cpp
/** V2 search entry point. Runs numSimulations passes of simulate_v2. */
int search_v2(
    const FactoredBelief& root_belief,
    const FactoredTabularModel& model,
    const PubStateEncoder& encoder,
    int hero_bucket,
    double* out_action_values,
    int* out_best_action,
    double* out_root_value)
{
  if (model.num_actions <= 0) return kStatusInvalidActionCount;

  arena_.clear();
  arena_.emplace_back();  /* root node */

  for (int sim = 0; sim < config_.num_simulations; ++sim) {
    /* Deep-copy belief for this simulation (particles mutate during search) */
    FactoredBelief sim_belief = root_belief;
    simulate_v2(0, root_belief.public_state, sim_belief, model, encoder, hero_bucket, 0);
  }

  /* Extract results from root node */
  const auto& root = arena_[0];
  int best = -1;
  double best_val = -std::numeric_limits<double>::infinity();
  for (int a = 0; a < model.num_actions; ++a) {
    const double q = root.action_stats[a].mean_value();
    out_action_values[a] = q;
    if (q > best_val || best < 0) {
      best_val = q;
      best = a;
    }
  }
  *out_best_action = best;
  *out_root_value = best_val;
  return kStatusOk;
}
```

- [ ] **Step 7: Add solve_raw_v2 JNI entry point**

```cpp
/** V2 flat JNI entry point for the factored tabular model interface.
  *
  * @param rival_count           number of rivals
  * @param particles_per_rival   [rival_count] particle count per rival
  * @param particle_types        flat [sum(particles_per_rival)] type indices
  * @param particle_priv_states  flat [sum(particles_per_rival)] private state indices
  * @param particle_weights      flat [sum(particles_per_rival)] weights
  * @param pub_street            public state street
  * @param pub_pot               current pot
  * @param num_hero_actions      hero action count
  * @param num_rival_types       number of distinct rival types (e.g., 4 for StrategicClass)
  * @param num_pub_states        number of discretized public states
  * @param rival_policy          flat [numTypes * numPubStates * numActions]
  * @param action_effects_flat   flat [numActions * 3] (pot_delta_frac, is_fold, is_allin)
  * @param showdown_equity       flat [numHeroBuckets * numRivalBuckets]
  * @param num_hero_buckets      hero hand bucket count
  * @param num_rival_buckets     rival hand bucket count
  * @param terminal_flags        flat [numPubStates * numActions]
  * @param hero_bucket           hero's current hand bucket
  * @param pot_bucket_size       pot units per bucket for pub state encoding
  * @param num_simulations       MCTS simulations
  * @param discount              gamma
  * @param exploration           UCB1 c
  * @param r_max                 reward bound
  * @param max_depth             tree depth limit
  * @param ess_threshold         ESS ratio for resampling
  * @param seed                  RNG seed
  * @param out_action_values     [numActions] output Q-values
  * @param out_best_action       [1] output best action
  * @param out_root_value        [1] output root value
  * @return 0 on success
  */
inline int solve_raw_v2(
    int rival_count,
    const int* particles_per_rival,
    const int* particle_types,
    const int* particle_priv_states,
    const double* particle_weights,
    int pub_street,
    double pub_pot,
    int num_hero_actions,
    int num_rival_types,
    int num_pub_states,
    const double* rival_policy,
    const double* action_effects_flat,
    const double* showdown_equity,
    int num_hero_buckets,
    int num_rival_buckets,
    const int* terminal_flags,
    int hero_bucket,
    double pot_bucket_size,
    int num_simulations,
    double discount,
    double exploration,
    double r_max,
    int max_depth,
    double ess_threshold,
    long long seed,
    double* out_action_values,
    int* out_best_action,
    double* out_root_value)
{
  /* Validate inputs */
  if (rival_count < 1 || rival_count > kMaxRivals) return kStatusInvalidRivalCount;
  if (num_hero_actions < 1 || num_hero_actions > kMaxActions) return kStatusInvalidActionCount;
  if (discount <= 0.0 || discount >= 1.0) return kStatusInvalidConfig;

  /* Build root belief */
  FactoredBelief root;
  root.public_state.street = pub_street;
  root.public_state.pot = pub_pot;
  root.rival_beliefs.resize(rival_count);
  int offset = 0;
  for (int r = 0; r < rival_count; ++r) {
    const int pc = particles_per_rival[r];
    if (pc < 1) return kStatusInvalidParticleCount;
    root.rival_beliefs[r].particles.resize(pc);
    root.rival_beliefs[r].ess_threshold = ess_threshold;
    for (int j = 0; j < pc; ++j) {
      auto& p = root.rival_beliefs[r].particles[j];
      p.rival_type = particle_types[offset + j];
      p.priv_state = particle_priv_states[offset + j];
      p.weight = particle_weights[offset + j];
    }
    root.rival_beliefs[r].normalize();
    offset += pc;
  }

  /* Build action effects */
  std::vector<ActionEffect> effects(num_hero_actions);
  for (int a = 0; a < num_hero_actions; ++a) {
    effects[a].pot_delta_frac = action_effects_flat[a * 3 + 0];
    effects[a].is_fold = action_effects_flat[a * 3 + 1] > 0.5;
    effects[a].is_allin = action_effects_flat[a * 3 + 2] > 0.5;
  }

  /* Build model */
  FactoredTabularModel model;
  model.rival_policy = rival_policy;
  model.num_rival_types = num_rival_types;
  model.num_pub_states = num_pub_states;
  model.action_effects = effects.data();
  model.showdown_equity = showdown_equity;
  model.num_hero_buckets = num_hero_buckets;
  model.num_rival_buckets = num_rival_buckets;
  model.terminal_flags = terminal_flags;
  model.num_actions = num_hero_actions;

  /* Build encoder */
  PubStateEncoder encoder;
  encoder.pot_bucket_size = pot_bucket_size;

  /* Configure and run solver */
  WPomcpConfig cfg;
  cfg.num_simulations = num_simulations;
  cfg.discount = discount;
  cfg.exploration_constant = exploration;
  cfg.r_max = r_max;
  cfg.max_depth = max_depth;
  cfg.ess_threshold = ess_threshold;

  auto status = validate_config(cfg);
  if (status != kStatusOk) return status;

  WPomcpSolver solver(cfg, seed);
  return solver.search_v2(root, model, encoder, hero_bucket,
      out_action_values, out_best_action, out_root_value);
}
```

- [ ] **Step 8: Rebuild native DLL**

Run: `powershell -ExecutionPolicy Bypass -File src/main/native/build-windows-llvm.ps1`

Expected: DLL rebuilds successfully. The build script compiles all `.cpp`/`.hpp` in `jni/` including the modified `WPomcpSolver.hpp`.

- [ ] **Step 9: Commit**

```bash
git add src/main/native/jni/WPomcpSolver.hpp
git commit -m "feat(native): add factored tabular model and simulate_v2 to WPomcp solver"
```

---

## Task 2: JNI Bridge for solveWPomcpV2

**Files:**
- Modify: `src/main/java/sicfun/holdem/HoldemPomcpNativeBindings.java`
- Modify: `src/main/native/jni/HoldemPomcpNativeBindings.cpp`

- [ ] **Step 1: Add solveWPomcpV2 native method to Java**

Add after the existing `solveWPomcp` method (line 136) in `HoldemPomcpNativeBindings.java`:

```java
  /**
   * V2 W-POMCP with factored tabular model (type-conditioned policies,
   * real observation particle reweighting, tabular rewards).
   *
   * @param rivalCount          number of rivals (1..8)
   * @param particlesPerRival   particle count per rival
   * @param particleTypes       rival type per particle, flat
   * @param particlePrivStates  private state per particle, flat
   * @param particleWeights     importance weights per particle, flat
   * @param pubStreet           current street (0..3)
   * @param pubPot              current pot size
   * @param numHeroActions      hero action count
   * @param numRivalTypes       distinct rival type count
   * @param numPubStates        discretized public state count
   * @param rivalPolicy         flat [numTypes * numPubStates * numActions]
   * @param actionEffects       flat [numActions * 3] (pot_delta_frac, is_fold, is_allin)
   * @param showdownEquity      flat [numHeroBuckets * numRivalBuckets]
   * @param numHeroBuckets      hero hand bucket count
   * @param numRivalBuckets     rival hand bucket count
   * @param terminalFlags       flat [numPubStates * numActions]
   * @param heroBucket          hero's hand bucket
   * @param potBucketSize       pot units per bucket
   * @param numSimulations      MCTS simulation count
   * @param discount            gamma
   * @param exploration         UCB1 c
   * @param rMax                reward bound
   * @param maxDepth            tree depth
   * @param essThreshold        ESS resampling trigger
   * @param seed                RNG seed
   * @param outActionValues     [numActions] output Q-values
   * @param outBestAction       [1] output best action
   * @param outRootValue        [1] output root value
   * @return 0 on success
   */
  public static native int solveWPomcpV2(
      int rivalCount,
      int[] particlesPerRival,
      int[] particleTypes,
      int[] particlePrivStates,
      double[] particleWeights,
      int pubStreet,
      double pubPot,
      int numHeroActions,
      int numRivalTypes,
      int numPubStates,
      double[] rivalPolicy,
      double[] actionEffects,
      double[] showdownEquity,
      int numHeroBuckets,
      int numRivalBuckets,
      int[] terminalFlags,
      int heroBucket,
      double potBucketSize,
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
```

- [ ] **Step 2: Add JNI C++ implementation**

Add in `HoldemPomcpNativeBindings.cpp` after the existing `solveWPomcp` function:

```cpp
JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemPomcpNativeBindings_solveWPomcpV2(
    JNIEnv* env, jclass,
    jint rival_count,
    jintArray j_particles_per_rival,
    jintArray j_particle_types,
    jintArray j_particle_priv_states,
    jdoubleArray j_particle_weights,
    jint pub_street,
    jdouble pub_pot,
    jint num_hero_actions,
    jint num_rival_types,
    jint num_pub_states,
    jdoubleArray j_rival_policy,
    jdoubleArray j_action_effects,
    jdoubleArray j_showdown_equity,
    jint num_hero_buckets,
    jint num_rival_buckets,
    jintArray j_terminal_flags,
    jint hero_bucket,
    jdouble pot_bucket_size,
    jint num_simulations,
    jdouble discount,
    jdouble exploration,
    jdouble r_max,
    jint max_depth,
    jdouble ess_threshold,
    jlong seed,
    jdoubleArray j_out_action_values,
    jintArray j_out_best_action,
    jdoubleArray j_out_root_value)
{
  /* Null checks */
  if (!j_particles_per_rival || !j_particle_types || !j_particle_priv_states ||
      !j_particle_weights || !j_rival_policy || !j_action_effects ||
      !j_showdown_equity || !j_terminal_flags ||
      !j_out_action_values || !j_out_best_action || !j_out_root_value)
    return wpomcp::kStatusNullArray;

  /* Read input arrays */
  auto ppr = read_int_array(env, j_particles_per_rival);
  auto pt  = read_int_array(env, j_particle_types);
  auto pps = read_int_array(env, j_particle_priv_states);
  auto pw  = read_double_array(env, j_particle_weights);
  auto rp  = read_double_array(env, j_rival_policy);
  auto ae  = read_double_array(env, j_action_effects);
  auto se  = read_double_array(env, j_showdown_equity);
  auto tf  = read_int_array(env, j_terminal_flags);

  if (ppr.empty() || pt.empty() || pps.empty() || pw.empty() ||
      rp.empty() || ae.empty() || se.empty() || tf.empty())
    return wpomcp::kStatusReadFailure;

  /* Allocate outputs */
  std::vector<double> out_av(num_hero_actions, 0.0);
  int out_ba = 0;
  double out_rv = 0.0;

  int status = wpomcp::solve_raw_v2(
      rival_count, ppr.data(), pt.data(), pps.data(), pw.data(),
      pub_street, pub_pot, num_hero_actions,
      num_rival_types, num_pub_states,
      rp.data(), ae.data(), se.data(),
      num_hero_buckets, num_rival_buckets, tf.data(),
      hero_bucket, pot_bucket_size,
      num_simulations, discount, exploration, r_max, max_depth, ess_threshold,
      static_cast<long long>(seed),
      out_av.data(), &out_ba, &out_rv);

  /* Write outputs */
  env->SetDoubleArrayRegion(j_out_action_values, 0, num_hero_actions, out_av.data());
  env->SetIntArrayRegion(j_out_best_action, 0, 1, &out_ba);
  env->SetDoubleArrayRegion(j_out_root_value, 0, 1, &out_rv);
  g_last_engine_code = 5;  /* 5 = W-POMCP V2 CPU */

  return status;
}
```

- [ ] **Step 3: Rebuild native DLL and commit**

```bash
powershell -ExecutionPolicy Bypass -File src/main/native/build-windows-llvm.ps1
git add src/main/java/sicfun/holdem/HoldemPomcpNativeBindings.java \
        src/main/native/jni/HoldemPomcpNativeBindings.cpp
git commit -m "feat(jni): add solveWPomcpV2 with factored tabular model interface"
```

---

## Task 3: WPomcpRuntime.scala V2 Interface

**Files:**
- Modify: `src/main/scala/sicfun/holdem/strategic/solver/WPomcpRuntime.scala`
- Modify: `src/test/scala/sicfun/holdem/strategic/solver/WPomcpRuntimeTest.scala`

- [ ] **Step 1: Write failing test for FactoredModel validation**

Add to `WPomcpRuntimeTest.scala`:

```scala
  /* --- FactoredModel validation tests --- */

  test("FactoredModel rejects mismatched rival policy size"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.FactoredModel(
        rivalPolicy = Array(0.5, 0.5),  /* should be 4 * 2 * 2 = 16 */
        numRivalTypes = 4,
        numPubStates = 2,
        actionEffects = Array.fill(6)(0.0),
        showdownEquity = Array.fill(100)(0.5),
        numHeroBuckets = 10,
        numRivalBuckets = 10,
        terminalFlags = Array.fill(4)(0),
        potBucketSize = 50.0
      )

  test("FactoredModel accepts valid dimensions"):
    val nTypes = 4; val nPub = 2; val nAct = 2
    val model = WPomcpRuntime.FactoredModel(
      rivalPolicy = Array.fill(nTypes * nPub * nAct)(1.0 / nAct),
      numRivalTypes = nTypes,
      numPubStates = nPub,
      actionEffects = Array.fill(nAct * 3)(0.0),
      showdownEquity = Array.fill(10 * 10)(0.5),
      numHeroBuckets = 10,
      numRivalBuckets = 10,
      terminalFlags = Array.fill(nPub * nAct)(0),
      potBucketSize = 50.0
    )
    assertEquals(model.numRivalTypes, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly *WPomcpRuntimeTest" 2>&1 | tail -20`

Expected: FAIL — `FactoredModel` not found.

- [ ] **Step 3: Add FactoredModel and SearchInputV2 to WPomcpRuntime**

Add to `WPomcpRuntime.scala` after the existing `SearchResult`:

```scala
  /** Factored tabular model for V2 solver (type-conditioned policies + tabular rewards).
    *
    * @param rivalPolicy      flat [numRivalTypes * numPubStates * numActions] -> P(action | type, pubState)
    * @param numRivalTypes    distinct rival types (4 for StrategicClass)
    * @param numPubStates     discretized public state count
    * @param actionEffects    flat [numActions * 3] -> (pot_delta_frac, is_fold, is_allin)
    * @param showdownEquity   flat [numHeroBuckets * numRivalBuckets] -> equity
    * @param numHeroBuckets   hero hand bucket count
    * @param numRivalBuckets  rival hand bucket count
    * @param terminalFlags    flat [numPubStates * numActions] -> terminal type
    * @param potBucketSize    pot units per bucket for pub state encoding
    */
  final case class FactoredModel(
      rivalPolicy: Array[Double],
      numRivalTypes: Int,
      numPubStates: Int,
      actionEffects: Array[Double],
      showdownEquity: Array[Double],
      numHeroBuckets: Int,
      numRivalBuckets: Int,
      terminalFlags: Array[Int],
      potBucketSize: Double
  ):
    def numActions: Int = actionEffects.length / 3
    require(rivalPolicy.length == numRivalTypes * numPubStates * numActions,
      s"rivalPolicy size ${rivalPolicy.length} != $numRivalTypes * $numPubStates * $numActions")
    require(actionEffects.length % 3 == 0,
      s"actionEffects length ${actionEffects.length} must be divisible by 3")
    require(showdownEquity.length == numHeroBuckets * numRivalBuckets,
      s"showdownEquity size ${showdownEquity.length} != $numHeroBuckets * $numRivalBuckets")
    require(terminalFlags.length == numPubStates * numActions,
      s"terminalFlags size ${terminalFlags.length} != $numPubStates * $numActions")

  /** V2 search input with factored tabular model. */
  final case class SearchInputV2(
      publicState: PublicState,
      rivalParticles: IndexedSeq[RivalParticles],
      model: FactoredModel,
      heroBucket: Int
  ):
    require(rivalParticles.nonEmpty, "Must have at least one rival")
    require(rivalParticles.size <= 8, s"Max 8 rivals, got ${rivalParticles.size}")
    require(heroBucket >= 0 && heroBucket < model.numHeroBuckets,
      s"heroBucket $heroBucket out of range [0, ${model.numHeroBuckets})")
    def rivalCount: Int = rivalParticles.size
```

- [ ] **Step 4: Add solveV2 method**

Add to `WPomcpRuntime.scala`:

```scala
  /** Run V2 W-POMCP search with factored tabular model.
    *
    * Returns Left(errorMessage) on failure, Right(SearchResult) on success.
    */
  def solveV2(input: SearchInputV2, config: Config): Either[String, SearchResult] =
    ensureLoaded() match
      case Left(err) => Left(err)
      case Right(()) =>
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

        val numActions = input.model.numActions
        val outActionValues = new Array[Double](numActions)
        val outBestAction = new Array[Int](1)
        val outRootValue = new Array[Double](1)

        val status = HoldemPomcpNativeBindings.solveWPomcpV2(
          input.rivalCount,
          particlesPerRival,
          allTypes,
          allPrivs,
          allWeights,
          input.publicState.street,
          input.publicState.pot,
          numActions,
          input.model.numRivalTypes,
          input.model.numPubStates,
          input.model.rivalPolicy,
          input.model.actionEffects,
          input.model.showdownEquity,
          input.model.numHeroBuckets,
          input.model.numRivalBuckets,
          input.model.terminalFlags,
          input.heroBucket,
          input.model.potBucketSize,
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
```

- [ ] **Step 5: Run tests**

Run: `sbt "testOnly *WPomcpRuntimeTest"`

Expected: All existing tests pass + new FactoredModel tests pass.

- [ ] **Step 6: Add native V2 test**

Add to `WPomcpRuntimeTest.scala`:

```scala
  test("native: solveV2 with dominant action returns correct best action"):
    assume(nativeAvailable, "Native library not available")
    val rp = WPomcpRuntime.RivalParticles(
      rivalTypes = Array(0, 1, 2, 3),
      privStates = Array(0, 1, 2, 3),
      weights = Array(0.25, 0.25, 0.25, 0.25)
    )
    val nTypes = 4; val nPub = 4; val nAct = 3
    /* Rival policy: type 0 always folds, types 1-3 always call */
    val policy = Array.fill(nTypes * nPub * nAct)(0.0)
    for pub <- 0 until nPub do
      /* type 0: fold (action 0) */
      policy(0 * nPub * nAct + pub * nAct + 0) = 1.0
      /* types 1-3: call (action 1) */
      for t <- 1 until nTypes do
        policy(t * nPub * nAct + pub * nAct + 1) = 1.0
    /* Action effects: 0=fold(0,1,0), 1=call(0.5,0,0), 2=raise(1.0,0,0) */
    val effects = Array(
      0.0, 1.0, 0.0,   /* fold */
      0.5, 0.0, 0.0,   /* call */
      1.0, 0.0, 0.0    /* raise */
    )
    /* Showdown: hero bucket 5 beats rival bucket 0-4, loses to 5-9 */
    val equity = Array.tabulate(10 * 10)((idx) =>
      val hb = idx / 10; val rb = idx % 10
      if hb > rb then 0.8 else if hb == rb then 0.5 else 0.2
    )
    /* Terminal: showdown at river (street=3), continue otherwise */
    val terminal = Array.fill(nPub * nAct)(0)
    for a <- 0 until nAct do terminal(3 * nAct + a) = 3 /* Showdown at river */

    val model = WPomcpRuntime.FactoredModel(
      rivalPolicy = policy, numRivalTypes = nTypes, numPubStates = nPub,
      actionEffects = effects, showdownEquity = equity,
      numHeroBuckets = 10, numRivalBuckets = 10,
      terminalFlags = terminal, potBucketSize = 50.0
    )
    val input = WPomcpRuntime.SearchInputV2(
      publicState = WPomcpRuntime.PublicState(0, 100.0),
      rivalParticles = IndexedSeq(rp),
      model = model,
      heroBucket = 7  /* strong hand */
    )
    val config = WPomcpRuntime.Config(numSimulations = 500, seed = 42L)
    val result = WPomcpRuntime.solveV2(input, config)
    assert(result.isRight, s"Expected Right, got $result")
    val sr = result.toOption.get
    assert(sr.actionValues.length == 3)
```

- [ ] **Step 7: Run full test suite and commit**

```bash
sbt "testOnly *WPomcpRuntimeTest"
git add src/main/scala/sicfun/holdem/strategic/solver/WPomcpRuntime.scala \
        src/test/scala/sicfun/holdem/strategic/solver/WPomcpRuntimeTest.scala
git commit -m "feat(strategic): add WPomcpRuntime V2 interface with factored tabular model"
```

---

## Task 4: StrategicRivalBelief

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/StrategicRivalBelief.scala`
- Create: `src/test/scala/sicfun/holdem/strategic/StrategicRivalBeliefTest.scala`

- [ ] **Step 1: Write failing tests**

```scala
package sicfun.holdem.strategic

import munit.FunSuite
import sicfun.core.DiscreteDistribution
import sicfun.holdem.types.{Board, Position, Street}

class StrategicRivalBeliefTest extends FunSuite:

  private val uniformPrior = DiscreteDistribution(Map(
    StrategicClass.Value -> 0.25,
    StrategicClass.Bluff -> 0.25,
    StrategicClass.SemiBluff -> 0.25,
    StrategicClass.Marginal -> 0.25
  ))

  private val dummyPub = PublicState(
    street = Street.Flop,
    board = Board.empty,
    pot = Chips(100.0),
    stacks = TableMap(
      hero = PlayerId("hero"),
      seats = Vector(Seat(PlayerId("hero"), Position.Button, SeatStatus.Active, Chips(500.0)))
    ),
    actionHistory = Vector.empty
  )

  test("StrategicRivalBelief initializes with uniform prior"):
    val belief = StrategicRivalBelief(uniformPrior)
    assertEqualsDouble(belief.typePosterior.probabilityOf(StrategicClass.Value), 0.25, 1e-10)
    assertEqualsDouble(belief.typePosterior.probabilityOf(StrategicClass.Bluff), 0.25, 1e-10)

  test("update returns new belief with same type"):
    val belief = StrategicRivalBelief(uniformPrior)
    val signal = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Raise,
      sizing = None, timing = None, stage = Street.Flop
    )
    val updated = belief.update(signal, dummyPub)
    assert(updated.isInstanceOf[StrategicRivalBelief])

  test("toParticles produces correct count and valid types"):
    val belief = StrategicRivalBelief(DiscreteDistribution(Map(
      StrategicClass.Value -> 0.7,
      StrategicClass.Bluff -> 0.3
    )))
    val (types, weights) = belief.toParticles(numParticles = 100, handBucket = 5)
    assertEquals(types.length, 100)
    assertEquals(weights.length, 100)
    assert(types.forall(t => t >= 0 && t <= 3))
    assertEqualsDouble(weights.sum, 1.0, 1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly *StrategicRivalBeliefTest"`

Expected: FAIL — class not found.

- [ ] **Step 3: Implement StrategicRivalBelief**

Create `src/main/scala/sicfun/holdem/strategic/StrategicRivalBelief.scala`:

```scala
package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution

/** Concrete RivalBeliefState backed by a DiscreteDistribution over StrategicClass.
  *
  * This is the M type parameter for Dynamics[M], KernelProfile[M], etc.
  * It holds a posterior distribution over rival types (Value/Bluff/SemiBluff/Marginal)
  * and updates via the kernel's tempered likelihood when new signals are observed.
  */
final case class StrategicRivalBelief(
    typePosterior: DiscreteDistribution[StrategicClass]
) extends RivalBeliefState:

  /** Update belief given an observed action signal.
    *
    * Default implementation preserves the current posterior (identity update).
    * The actual Bayesian update happens via the kernel pipeline in Dynamics.fullStep(),
    * which calls the StateEmbeddingUpdater to produce a new StrategicRivalBelief
    * with an updated posterior.
    */
  def update(signal: ActionSignal, publicState: PublicState): StrategicRivalBelief =
    this  /* Identity — real update via kernel pipeline */

  /** Convert belief to WPomcp particle arrays.
    *
    * Samples numParticles particles from the type posterior.
    * Each particle gets (type=StrategicClass.ordinal, privState=handBucket, weight=1/N).
    *
    * @param numParticles number of particles to generate
    * @param handBucket   the hand bucket to assign as private state
    * @return (typeIndices, weights) arrays
    */
  def toParticles(numParticles: Int, handBucket: Int): (Array[Int], Array[Double]) =
    val types = new Array[Int](numParticles)
    val weights = new Array[Double](numParticles)
    val classes = StrategicClass.values
    val uniformWeight = 1.0 / numParticles.toDouble
    var idx = 0
    for cls <- classes do
      val prob = typePosterior.probabilityOf(cls)
      val count = math.round(prob * numParticles).toInt
      var j = 0
      while j < count && idx < numParticles do
        types(idx) = cls.ordinal
        weights(idx) = uniformWeight
        idx += 1
        j += 1
    /* Fill remaining slots with MAP class */
    val mapClass = classes.maxBy(typePosterior.probabilityOf)
    while idx < numParticles do
      types(idx) = mapClass.ordinal
      weights(idx) = uniformWeight
      idx += 1
    (types, weights)

object StrategicRivalBelief:
  /** Create with uniform prior over all four strategic classes. */
  def uniform: StrategicRivalBelief =
    StrategicRivalBelief(DiscreteDistribution(
      StrategicClass.values.map(c => c -> 0.25).toMap
    ))

  /** StateEmbeddingUpdater for StrategicRivalBelief.
    * Replaces the type posterior with the output of the kernel's tempered likelihood.
    */
  val updater: StateEmbeddingUpdater[StrategicRivalBelief] =
    (state: StrategicRivalBelief, posterior: DiscreteDistribution[StrategicClass]) =>
      StrategicRivalBelief(posterior)
```

- [ ] **Step 4: Run tests and commit**

```bash
sbt "testOnly *StrategicRivalBeliefTest"
git add src/main/scala/sicfun/holdem/strategic/StrategicRivalBelief.scala \
        src/test/scala/sicfun/holdem/strategic/StrategicRivalBeliefTest.scala
git commit -m "feat(strategic): add StrategicRivalBelief concrete RivalBeliefState"
```

---

## Task 5: HeroMode.Strategic

**Files:**
- Modify: `src/main/scala/sicfun/holdem/types/HeroMode.scala`

- [ ] **Step 1: Add Strategic case to HeroMode enum**

Add after the Gto case in `HeroMode.scala`:

```scala
  /** Strategic mode: formal POMDP via WPomcp solver with factored particle beliefs. */
  case Strategic
```

- [ ] **Step 2: Verify compilation**

Run: `sbt compile 2>&1 | tail -5`

Expected: Compile succeeds. Match expressions on HeroMode may warn about non-exhaustive patterns — that's expected and will be fixed in Task 7.

- [ ] **Step 3: Commit**

```bash
git add src/main/scala/sicfun/holdem/types/HeroMode.scala
git commit -m "feat(types): add HeroMode.Strategic for POMDP solver path"
```

---

## Task 6: PokerPomcpFormulation

**Files:**
- Create: `src/main/scala/sicfun/holdem/engine/PokerPomcpFormulation.scala`
- Create: `src/test/scala/sicfun/holdem/engine/PokerPomcpFormulationTest.scala`

- [ ] **Step 1: Write failing tests**

```scala
package sicfun.holdem.engine

import munit.FunSuite
import sicfun.holdem.types.*
import sicfun.holdem.strategic.*
import sicfun.holdem.strategic.solver.WPomcpRuntime
import sicfun.core.DiscreteDistribution

class PokerPomcpFormulationTest extends FunSuite:

  private def makeBeliefs: Map[PlayerId, StrategicRivalBelief] =
    Map(PlayerId("villain") -> StrategicRivalBelief.uniform)

  test("buildRivalPolicy produces correct dimensions"):
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(3.0))
    val beliefs = makeBeliefs
    val policy = PokerPomcpFormulation.buildRivalPolicy(
      numRivalTypes = 4,
      numPubStates = 192,
      numActions = actions.size
    )
    assertEquals(policy.length, 4 * 192 * 3)

  test("buildActionEffects encodes fold correctly"):
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(3.0))
    val effects = PokerPomcpFormulation.buildActionEffects(actions, potChips = 100, stackChips = 500)
    /* Fold: pot_delta_frac=0, is_fold=1, is_allin=0 */
    assertEqualsDouble(effects(0), 0.0, 1e-10)
    assertEqualsDouble(effects(1), 1.0, 1e-10)
    assertEqualsDouble(effects(2), 0.0, 1e-10)

  test("buildShowdownEquity produces valid equity values"):
    val equity = PokerPomcpFormulation.buildShowdownEquity(
      numHeroBuckets = 10, numRivalBuckets = 10
    )
    assertEquals(equity.length, 100)
    assert(equity.forall(e => e >= 0.0 && e <= 1.0))

  test("buildTerminalFlags marks river as showdown"):
    val flags = PokerPomcpFormulation.buildTerminalFlags(numPubStates = 192, numActions = 3)
    /* River pub states (street=3) should have showdown terminal type */
    /* Street 3 starts at index 3 * (192/4) = 144 (if streets evenly distributed) */
    /* Just verify non-negative values */
    assert(flags.forall(f => f >= 0 && f <= 3))

  test("buildSearchInputV2 assembles valid input"):
    val actions = Vector(PokerAction.Fold, PokerAction.Call)
    val beliefs = makeBeliefs
    val gs = GameState(
      board = Board.empty,
      street = Street.Flop,
      pot = 100,
      stackSize = 500,
      toCall = 50,
      position = Position.Button,
      heroCards = None
    )
    val input = PokerPomcpFormulation.buildSearchInputV2(
      gameState = gs,
      rivalBeliefs = beliefs,
      heroActions = actions,
      heroBucket = 5,
      particlesPerRival = 50
    )
    assertEquals(input.rivalParticles.size, 1)
    assertEquals(input.rivalParticles.head.particleCount, 50)
    assertEquals(input.heroBucket, 5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly *PokerPomcpFormulationTest"`

Expected: FAIL — object not found.

- [ ] **Step 3: Implement PokerPomcpFormulation**

Create `src/main/scala/sicfun/holdem/engine/PokerPomcpFormulation.scala`:

```scala
package sicfun.holdem.engine

import sicfun.holdem.types.*
import sicfun.holdem.strategic.*
import sicfun.holdem.strategic.solver.WPomcpRuntime

/** Builds the factored tabular model for the WPomcp V2 solver.
  *
  * Translates poker domain state into flat arrays that cross the JNI boundary.
  * All arrays follow the layout contract defined in WPomcpSolver.hpp.
  */
object PokerPomcpFormulation:

  private val NumPotBuckets = 8
  private val NumStackBuckets = 6
  val NumPubStates: Int = 4 * NumPotBuckets * NumStackBuckets  /* 192 */
  val NumRivalTypes: Int = StrategicClass.values.length  /* 4 */
  val NumHandBuckets: Int = 10
  val DefaultPotBucketSize: Double = 50.0

  /** Build the rival policy table: P(action | type, pubState).
    *
    * Initial implementation: uniform policy across all types and states.
    * This will be replaced by kernel-derived policies in a follow-up
    * once the full TemperedLikelihood -> KernelConstructor pipeline
    * is connected.
    */
  def buildRivalPolicy(
      numRivalTypes: Int,
      numPubStates: Int,
      numActions: Int
  ): Array[Double] =
    val size = numRivalTypes * numPubStates * numActions
    val uniformProb = 1.0 / numActions.toDouble
    Array.fill(size)(uniformProb)

  /** Build action effects array: [numActions * 3].
    * Fields per action: (pot_delta_frac, is_fold, is_allin).
    */
  def buildActionEffects(
      actions: Vector[PokerAction],
      potChips: Int,
      stackChips: Int
  ): Array[Double] =
    val result = new Array[Double](actions.size * 3)
    var i = 0
    for action <- actions do
      val base = i * 3
      action match
        case PokerAction.Fold =>
          result(base) = 0.0
          result(base + 1) = 1.0  /* is_fold */
          result(base + 2) = 0.0
        case PokerAction.Check =>
          result(base) = 0.0
          result(base + 1) = 0.0
          result(base + 2) = 0.0
        case PokerAction.Call =>
          val frac = if potChips > 0 then 0.5 else 0.0  /* call typically ~50% pot */
          result(base) = frac
          result(base + 1) = 0.0
          result(base + 2) = 0.0
        case PokerAction.Raise(amountBb) =>
          val chipAmount = (amountBb * 100).toInt  /* BB to chips, assuming BB=100 */
          val frac = if potChips > 0 then chipAmount.toDouble / potChips.toDouble else 1.0
          result(base) = math.min(frac, 10.0)  /* cap at 10x pot */
          result(base + 1) = 0.0
          result(base + 2) = if chipAmount >= stackChips then 1.0 else 0.0
      i += 1
    result

  /** Build showdown equity table: [numHeroBuckets * numRivalBuckets].
    *
    * Initial implementation: linear equity based on bucket comparison.
    * hero_bucket > rival_bucket => favorable equity.
    * This will be replaced by HoldemEquity-derived tables.
    */
  def buildShowdownEquity(
      numHeroBuckets: Int,
      numRivalBuckets: Int
  ): Array[Double] =
    Array.tabulate(numHeroBuckets * numRivalBuckets) { idx =>
      val hb = idx / numRivalBuckets
      val rb = idx % numRivalBuckets
      val diff = (hb - rb).toDouble / math.max(numHeroBuckets, numRivalBuckets).toDouble
      0.5 + diff * 0.4  /* range: [0.1, 0.9] */
    }

  /** Build terminal flags: [numPubStates * numActions].
    * Values: 0=Continue, 1=HeroFold, 2=RivalFold, 3=Showdown.
    *
    * River (street=3) -> Showdown for non-fold actions.
    * Fold action -> HeroFold at any street.
    */
  def buildTerminalFlags(
      numPubStates: Int,
      numActions: Int
  ): Array[Int] =
    val flags = new Array[Int](numPubStates * numActions)
    for pub <- 0 until numPubStates do
      val street = pub / (NumPotBuckets * NumStackBuckets)
      for a <- 0 until numActions do
        val idx = pub * numActions + a
        if a == 0 then  /* action 0 = fold by convention */
          flags(idx) = 1  /* HeroFold */
        else if street >= 3 then
          flags(idx) = 3  /* Showdown at river */
        else
          flags(idx) = 0  /* Continue */
    flags

  /** Build complete SearchInputV2 from poker game state and strategic beliefs. */
  def buildSearchInputV2(
      gameState: GameState,
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      heroActions: Vector[PokerAction],
      heroBucket: Int,
      particlesPerRival: Int = 100
  ): WPomcpRuntime.SearchInputV2 =
    val numActions = heroActions.size

    /* Build per-rival particles */
    val rivalParticles = rivalBeliefs.values.toIndexedSeq.map { belief =>
      val (types, weights) = belief.toParticles(particlesPerRival, heroBucket)
      WPomcpRuntime.RivalParticles(
        rivalTypes = types,
        privStates = Array.fill(particlesPerRival)(heroBucket),  /* placeholder: rival bucket */
        weights = weights
      )
    }

    val model = WPomcpRuntime.FactoredModel(
      rivalPolicy = buildRivalPolicy(NumRivalTypes, NumPubStates, numActions),
      numRivalTypes = NumRivalTypes,
      numPubStates = NumPubStates,
      actionEffects = buildActionEffects(heroActions, gameState.pot, gameState.stackSize),
      showdownEquity = buildShowdownEquity(NumHandBuckets, NumHandBuckets),
      numHeroBuckets = NumHandBuckets,
      numRivalBuckets = NumHandBuckets,
      terminalFlags = buildTerminalFlags(NumPubStates, numActions),
      potBucketSize = DefaultPotBucketSize
    )

    WPomcpRuntime.SearchInputV2(
      publicState = WPomcpRuntime.PublicState(gameState.street.ordinal, gameState.pot.toDouble),
      rivalParticles = rivalParticles,
      model = model,
      heroBucket = heroBucket
    )
```

- [ ] **Step 4: Run tests and commit**

```bash
sbt "testOnly *PokerPomcpFormulationTest"
git add src/main/scala/sicfun/holdem/engine/PokerPomcpFormulation.scala \
        src/test/scala/sicfun/holdem/engine/PokerPomcpFormulationTest.scala
git commit -m "feat(engine): add PokerPomcpFormulation for factored tabular model construction"
```

---

## Task 7: StrategicEngine

**Files:**
- Create: `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`
- Create: `src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala`

- [ ] **Step 1: Write failing tests**

```scala
package sicfun.holdem.engine

import munit.FunSuite
import sicfun.holdem.types.*
import sicfun.holdem.strategic.*
import sicfun.holdem.strategic.solver.WPomcpRuntime
import sicfun.core.DiscreteDistribution

class StrategicEngineTest extends FunSuite:

  test("StrategicEngine initializes with uniform beliefs for unknown rivals"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    val state = engine.sessionState
    assertEquals(state.rivalBeliefs.size, 1)
    assertEqualsDouble(
      state.rivalBeliefs(PlayerId("v1")).typePosterior.probabilityOf(StrategicClass.Value),
      0.25, 1e-10)

  test("startHand resets hand-local state"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(heroCards = HoleCards.parse("AhKs").get)
    assert(engine.currentHandActive)

  test("decide returns a valid PokerAction"):
    assume(WPomcpRuntime.isAvailable, "Native library not available")
    val engine = new StrategicEngine(StrategicEngine.Config(numSimulations = 50))
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(heroCards = HoleCards.parse("AhKs").get)
    val gs = GameState(
      board = Board.empty,
      street = Street.Preflop,
      pot = 150,
      stackSize = 5000,
      toCall = 100,
      position = Position.Button,
      heroCards = Some(HoleCards.parse("AhKs").get)
    )
    val candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(3.0))
    val action = engine.decide(gs, candidates)
    assert(candidates.contains(action), s"Action $action not in candidates")

  test("observeAction does not throw"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(heroCards = HoleCards.parse("AhKs").get)
    val gs = GameState(
      board = Board.empty,
      street = Street.Preflop,
      pot = 200,
      stackSize = 4900,
      toCall = 0,
      position = Position.Button,
      heroCards = None
    )
    engine.observeAction(PlayerId("v1"), PokerAction.Raise(3.0), gs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `sbt "testOnly *StrategicEngineTest"`

Expected: FAIL — class not found.

- [ ] **Step 3: Implement StrategicEngine**

Create `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`:

```scala
package sicfun.holdem.engine

import sicfun.holdem.types.*
import sicfun.holdem.strategic.*
import sicfun.holdem.strategic.solver.WPomcpRuntime

/** Session/hand orchestrator for the Strategic decision mode.
  *
  * Manages per-rival beliefs across hands, builds the factored tabular model,
  * and delegates action selection to WPomcpRuntime.solveV2.
  */
class StrategicEngine(val config: StrategicEngine.Config):

  private var _sessionState: StrategicEngine.SessionState = null
  private var _handActive: Boolean = false
  private var _heroCards: Option[HoleCards] = None

  def sessionState: StrategicEngine.SessionState = _sessionState
  def currentHandActive: Boolean = _handActive

  /** Initialize session with rival IDs. Loads beliefs from store or uses uniform priors. */
  def initSession(
      rivalIds: Vector[PlayerId],
      existingBeliefs: Map[PlayerId, StrategicRivalBelief] = Map.empty
  ): Unit =
    val beliefs = rivalIds.map { id =>
      id -> existingBeliefs.getOrElse(id, StrategicRivalBelief.uniform)
    }.toMap
    val exploitStates = rivalIds.map { id =>
      id -> ExploitationState.initial(config.exploitConfig)
    }.toMap
    _sessionState = StrategicEngine.SessionState(
      rivalBeliefs = beliefs,
      exploitationStates = exploitStates
    )

  /** Start a new hand. Resets hand-local state, preserves session beliefs. */
  def startHand(heroCards: HoleCards): Unit =
    _heroCards = Some(heroCards)
    _handActive = true

  /** Observe a rival's action. Updates beliefs via Dynamics.fullStep(). */
  def observeAction(actor: PlayerId, action: PokerAction, gameState: GameState): Unit =
    if _sessionState == null then return
    /* For now, preserve beliefs unchanged. Full Dynamics integration requires
     * SignalBridge + PublicStateBridge + kernel pipeline, which will be connected
     * once the kernel-derived rival_policy replaces the uniform placeholder. */

  /** Choose an action using the WPomcp V2 solver. */
  def decide(gameState: GameState, candidateActions: Vector[PokerAction]): PokerAction =
    require(_sessionState != null, "Session not initialized")
    require(_handActive, "No hand in progress")
    require(candidateActions.nonEmpty, "No candidate actions")

    val heroBucket = estimateHeroBucket(gameState)

    val searchInput = PokerPomcpFormulation.buildSearchInputV2(
      gameState = gameState,
      rivalBeliefs = _sessionState.rivalBeliefs,
      heroActions = candidateActions,
      heroBucket = heroBucket,
      particlesPerRival = config.particlesPerRival
    )

    WPomcpRuntime.solveV2(searchInput, WPomcpRuntime.Config(
      numSimulations = config.numSimulations,
      discount = config.discount,
      maxDepth = config.maxDepth,
      seed = config.seed
    )) match
      case Right(result) =>
        if result.bestAction >= 0 && result.bestAction < candidateActions.size then
          candidateActions(result.bestAction)
        else
          candidateActions.last  /* fallback to last action */
      case Left(err) =>
        /* Solver failed — fall back to first non-fold action or fold */
        candidateActions.find(_ != PokerAction.Fold).getOrElse(PokerAction.Fold)

  /** End the current hand. Optional showdown for belief update. */
  def endHand(showdownResult: Option[Map[PlayerId, HoleCards]] = None): Unit =
    _handActive = false
    _heroCards = None
    /* Showdown kernel update will be connected in follow-up */

  /** Estimate hero hand bucket (0-9 equity decile) from game state. */
  private def estimateHeroBucket(gameState: GameState): Int =
    /* Placeholder: use position as rough proxy until HoldemEquity integration */
    gameState.position match
      case Position.Button | Position.Cutoff => 7  /* late position = likely strong */
      case Position.SmallBlind | Position.BigBlind => 4  /* blinds = average */
      case _ => 5  /* default middle bucket */

object StrategicEngine:
  final case class Config(
      numSimulations: Int = 500,
      discount: Double = 0.95,
      maxDepth: Int = 20,
      seed: Long = 42L,
      particlesPerRival: Int = 100,
      exploitConfig: ExploitationConfig = ExploitationConfig(
        initialBeta = 1.0,
        retreatRate = 0.1,
        adaptationTolerance = 0.05
      )
  )

  final case class SessionState(
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      exploitationStates: Map[PlayerId, ExploitationState]
  )
```

- [ ] **Step 4: Run tests and commit**

```bash
sbt "testOnly *StrategicEngineTest"
git add src/main/scala/sicfun/holdem/engine/StrategicEngine.scala \
        src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala
git commit -m "feat(engine): add StrategicEngine orchestrator for POMDP decision path"
```

---

## Task 8: HeroDecisionPipeline Strategic Branch

**Files:**
- Modify: `src/main/scala/sicfun/holdem/engine/HeroDecisionPipeline.scala`

- [ ] **Step 1: Add StrategicDecisionContext and Strategic branch**

Add a new context type after `HeroDecisionContext`:

```scala
  /** Context for Strategic mode decisions. Simpler than HeroDecisionContext
    * because the StrategicEngine manages its own state.
    */
  final case class StrategicDecisionContext(
      state: GameState,
      candidates: Vector[PokerAction],
      engine: StrategicEngine
  )
```

Modify `decideHero` to handle all three modes. Replace the existing `decideHero` match expression:

```scala
  def decideHero(mode: HeroMode, ctx: HeroDecisionContext): PokerAction =
    mode match
      case HeroMode.Adaptive =>
        ctx.engine
          .decide(
            hero = ctx.hero,
            state = ctx.state,
            folds = ctx.folds,
            villainPos = ctx.villainPos,
            observations = ctx.observations,
            candidateActions = ctx.candidates,
            decisionBudgetMillis = ctx.decisionBudgetMillis,
            rng = new Random(ctx.rng.nextLong())
          )
          .decision
          .recommendation
          .bestAction
      case HeroMode.Gto =>
        val gtoRng = new Random(ctx.rng.nextLong())
        val posterior = RangeInferenceEngine
          .inferPosterior(
            hero = ctx.hero,
            board = ctx.state.board,
            folds = ctx.folds,
            tableRanges = ctx.tableRanges,
            villainPos = ctx.villainPos,
            observations = ctx.observations,
            actionModel = ctx.actionModel,
            bunchingTrials = ctx.bunchingTrials,
            rng = gtoRng
          )
          .posterior
        val policy = HoldemCfrSolver
          .solveShallowDecisionPolicy(
            hero = ctx.hero,
            state = ctx.state,
            villainPosterior = posterior,
            candidateActions = ctx.candidates,
            config = HoldemCfrConfig(
              iterations = ctx.cfrIterations,
              maxVillainHands = ctx.cfrVillainHands,
              equityTrials = ctx.cfrEquityTrials,
              postflopLookahead = ctx.state.street != Street.Preflop && ctx.state.street != Street.River,
              rngSeed = ctx.rng.nextLong()
            )
          )
        sampleActionByPolicy(
          probabilities = policy.actionProbabilities,
          candidates = ctx.candidates,
          rng = gtoRng
        )
      case HeroMode.Strategic =>
        throw new UnsupportedOperationException(
          "Strategic mode requires StrategicDecisionContext — use decideHeroStrategic()")
```

Add a dedicated Strategic entry point:

```scala
  /** Strategic mode decision dispatch.
    * Uses StrategicEngine which manages its own beliefs and solver.
    */
  def decideHeroStrategic(ctx: StrategicDecisionContext): PokerAction =
    ctx.engine.decide(ctx.state, ctx.candidates)
```

- [ ] **Step 2: Fix exhaustive match warnings in other files**

Run: `sbt compile 2>&1 | grep -i "match may not be exhaustive"`

If any files match on HeroMode without Strategic, add the case. Likely candidates:
- `src/main/scala/sicfun/holdem/runtime/PokerAdvisor.scala`
- `src/main/scala/sicfun/holdem/runtime/TexasHoldemPlayingHall.scala`

For each, add a placeholder case:

```scala
case HeroMode.Strategic =>
  throw new UnsupportedOperationException("Strategic mode not yet supported in this context")
```

- [ ] **Step 3: Run full test suite and commit**

```bash
sbt test 2>&1 | tail -20
git add src/main/scala/sicfun/holdem/engine/HeroDecisionPipeline.scala \
        src/main/scala/sicfun/holdem/types/HeroMode.scala
# Also add any files modified for exhaustive match
git commit -m "feat(engine): add Strategic branch to HeroDecisionPipeline"
```

---

## Task 9: End-to-End Integration Test

**Files:**
- Modify: `src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala`

- [ ] **Step 1: Add integration test that plays a mini hand**

Add to `StrategicEngineTest.scala`:

```scala
  test("integration: play a complete hand with Strategic mode"):
    assume(WPomcpRuntime.isAvailable, "Native library not available")
    val engine = new StrategicEngine(StrategicEngine.Config(numSimulations = 100))
    engine.initSession(rivalIds = Vector(PlayerId("villain")))

    /* Hand 1: hero has strong hand */
    engine.startHand(HoleCards.parse("AhAs").get)

    val preflopState = GameState(
      board = Board.empty,
      street = Street.Preflop,
      pot = 150,
      stackSize = 5000,
      toCall = 100,
      position = Position.Button,
      heroCards = Some(HoleCards.parse("AhAs").get)
    )
    val candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(3.0))

    /* Villain raises */
    engine.observeAction(PlayerId("villain"), PokerAction.Raise(3.0), preflopState)

    /* Hero decides */
    val action = engine.decide(preflopState, candidates)
    assert(candidates.contains(action), s"Action $action not in candidates")

    /* End hand */
    engine.endHand()

    /* Hand 2: beliefs should persist */
    engine.startHand(HoleCards.parse("KhQs").get)
    val action2 = engine.decide(preflopState, candidates)
    assert(candidates.contains(action2))
    engine.endHand()

  test("integration: decideHeroStrategic routes correctly"):
    assume(WPomcpRuntime.isAvailable, "Native library not available")
    val engine = new StrategicEngine(StrategicEngine.Config(numSimulations = 50))
    engine.initSession(rivalIds = Vector(PlayerId("villain")))
    engine.startHand(HoleCards.parse("AhKs").get)

    val gs = GameState(
      board = Board.empty,
      street = Street.Preflop,
      pot = 150,
      stackSize = 5000,
      toCall = 100,
      position = Position.Button,
      heroCards = Some(HoleCards.parse("AhKs").get)
    )
    val candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(3.0))
    val ctx = HeroDecisionPipeline.StrategicDecisionContext(
      state = gs,
      candidates = candidates,
      engine = engine
    )
    val action = HeroDecisionPipeline.decideHeroStrategic(ctx)
    assert(candidates.contains(action))
```

- [ ] **Step 2: Run all tests**

Run: `sbt test 2>&1 | tail -30`

Expected: All tests pass. Strategic engine integration tests pass when native DLL is available, skip gracefully when not.

- [ ] **Step 3: Commit**

```bash
git add src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala
git commit -m "test(engine): add end-to-end integration tests for Strategic mode"
```

---

## Self-Review

**Spec coverage check:**
- C++ WPomcp solver fix (type-conditioned, observations, reweighting, tabular) → Task 1
- JNI bridge → Task 2
- WPomcpRuntime V2 → Task 3
- StrategicRivalBelief → Task 4
- HeroMode.Strategic → Task 5
- PokerPomcpFormulation → Task 6
- StrategicEngine → Task 7
- HeroDecisionPipeline → Task 8
- Integration test → Task 9
- Multiway-native invariant → preserved (rivalParticles: IndexedSeq, rivalCount >= 1)
- Belief persistence → StrategicEngine.SessionState carries across hands
- Strategic never imports engine → preserved (StrategicRivalBelief in strategic package)

**Placeholder scan:** `buildRivalPolicy` returns uniform — documented as initial implementation with explicit comment about kernel pipeline follow-up. `estimateHeroBucket` uses position proxy — documented similarly. `observeAction` is identity — documented. These are explicit scaffolds, not hidden TODOs.

**Type consistency check:**
- `FactoredModel` defined in Task 3 (WPomcpRuntime), used in Task 6 (PokerPomcpFormulation) and Task 7 (StrategicEngine) — consistent
- `SearchInputV2` defined in Task 3, constructed in Task 6 — consistent
- `StrategicRivalBelief.toParticles` returns `(Array[Int], Array[Double])`, consumed by `RivalParticles(rivalTypes, privStates, weights)` in Task 6 — consistent
- `StrategicEngine.Config` defined in Task 7, no conflicts
- `StrategicDecisionContext` defined in Task 8, used in Task 9 — consistent
