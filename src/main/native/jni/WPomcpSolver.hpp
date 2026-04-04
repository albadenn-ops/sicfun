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

/* ---------- Systematic resampling ---------- */

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

/* ---------- Joint rival action sampling ---------- */

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

/* ---------- Factored belief update ---------- */

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

/* ---------- W-POMCP tree search ---------- */

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

  /** Per-depth scratch storage to avoid deep-copying FactoredBelief on every
    * recursive simulate() call. Indexed by recursion depth so that each level
    * reuses a pre-allocated belief buffer instead of allocating a new one.
    */
  struct SimulationContext {
    std::vector<FactoredBelief> belief_stack;  // indexed by depth
    void ensure_depth(int depth, const FactoredBelief& tmpl) {
      if (static_cast<int>(belief_stack.size()) <= depth) {
        belief_stack.resize(depth + 1);
      }
      auto& b = belief_stack[depth];
      if (b.rival_beliefs.size() != tmpl.rival_beliefs.size()) {
        b.rival_beliefs.resize(tmpl.rival_beliefs.size());
        for (size_t i = 0; i < tmpl.rival_beliefs.size(); ++i) {
          b.rival_beliefs[i].particles.resize(tmpl.rival_beliefs[i].particles.size());
        }
      }
    }
  };
  SimulationContext sim_ctx_;

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

  /** Run a single simulation from the given node at the given depth.
    *
    * Returns the discounted cumulative reward from this point forward.
    * Modifies the node's visit counts and value estimates.
    */
  double simulate(
      WPomcpNode& node,
      const FactoredBelief& belief,  /* by const-ref: scratch buffer used for mutations */
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
      const int rc = config_.rival_count();
      std::vector<RivalActionDist> rival_dists(rc);
      for (int i = 0; i < rc; ++i) {
        /* Rival i's action distribution conditioned on their
         * particle-averaged state and the public state. */
        rival_dists[i].action_probs =
            model.rival_action_probs +
            i * model.num_hero_actions;  /* simplified: same action space */
        rival_dists[i].num_actions = model.num_hero_actions;
      }
      int status = sample_joint_action(
          rival_dists.data(), rc, rng_, joint_action);
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
    const int rc = config_.rival_count();
    std::vector<ObservationLikelihoods> rival_obs(rc);
    std::vector<std::vector<double>> lik_storage(rc);
    for (int i = 0; i < rc; ++i) {
      const int np = belief.rival_beliefs[i].particle_count();
      lik_storage[i].assign(np, 1.0);  /* uniform likelihood as default */
      rival_obs[i].likelihoods = lik_storage[i].data();
      rival_obs[i].particle_count = np;
    }
    /* Copy belief into per-depth scratch buffer and mutate the copy,
     * avoiding a deep copy of all rival particle vectors on each recursion. */
    sim_ctx_.ensure_depth(depth, belief);
    auto& scratch = sim_ctx_.belief_stack[depth];
    scratch.public_state = belief.public_state;
    for (int i = 0; i < rc; ++i) {
      auto& src = belief.rival_beliefs[i];
      auto& dst = scratch.rival_beliefs[i];
      dst.particles.resize(src.particles.size());
      std::copy(src.particles.begin(), src.particles.end(), dst.particles.begin());
      dst.ess_threshold = src.ess_threshold;
    }
    update_factored_belief(scratch, next_pub,
                           rival_obs.data(), rc, rng_);

    /* EXPAND / RECURSE: find or create child node. */
    obs_hash = next_pub.street * 1000 + hero_action;
    WPomcpNode* child = find_or_create_child(node, hero_action, obs_hash);

    /* Recursive simulation from child using scratch belief. */
    double future = simulate(*child, scratch, model, depth + 1);
    double total = reward + config_.discount * future;

    /* BACKPROPAGATE. */
    node.visit_count++;
    node.action_stats[hero_action].visit_count++;
    node.action_stats[hero_action].value_sum += total;

    return total;
  }

public:
  /** Construct a W-POMCP solver with the given config and random seed. */
  explicit WPomcpSolver(const WPomcpConfig& config, uint64_t seed = 42)
      : config_(config), rng_(static_cast<uint32_t>(seed)) {}

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

    /* Reset scratch belief stack; reserve up to max_depth to avoid
     * reallocation during the simulation loop. */
    sim_ctx_.belief_stack.clear();
    sim_ctx_.belief_stack.reserve(config_.max_depth);

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

    /* Initialize action stats if no simulations ran (edge case). */
    if (root_.action_stats.empty()) {
      root_.action_stats.resize(model.num_hero_actions);
    }

    for (int a = 0; a < model.num_hero_actions; ++a) {
      result.action_values[a] = root_.action_stats[a].mean_value();
      if (result.action_values[a] > best_val) {
        best_val = result.action_values[a];
        best_act = a;
      }
    }

    result.best_action = best_act;
    result.root_value = (best_val == -std::numeric_limits<double>::infinity())
                        ? 0.0 : best_val;
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

/* ---------- JNI-friendly flat entry point ---------- */

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
    double ess_val = rb.ess();
    /* Equal weights: ESS should be 2.0. */
    if (std::abs(ess_val - 2.0) > 1e-10) return 2;
  }
  /* Test 2: Skewed weights ESS. */
  {
    RivalBeliefSet rb;
    rb.particles = {{0, 0, 0, 0.99}, {0, 1, 0, 0.01}};
    if (!rb.normalize()) return 3;
    double ess_val = rb.ess();
    /* Highly skewed: ESS should be close to 1. */
    if (ess_val > 1.1) return 4;
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
    double rwds[] = {1.0, 0.0};  /* action 0 dominates */

    double out_vals[2];
    int out_best;
    double out_root;

    int status = solve_raw(
        1, ppr, types, privs, weights,
        0, 100.0,
        2, rival_ap, rwds,
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
    double rwds[] = {0.5};

    double out_vals[1];
    int out_best;
    double out_root;

    int status = solve_raw(
        1, ppr, types, privs, weights,
        0, 50.0,
        1, rival_ap, rwds,
        10, 0.99, 1.0, 1.0, 5, 0.5,
        7,
        out_vals, &out_best, &out_root);
    if (status != kStatusOk) return 16;
    if (out_best != 0) return 17;
  }
  return kStatusOk;
}

#endif /* WPOMCP_SELF_TEST */

}  /* namespace wpomcp */
