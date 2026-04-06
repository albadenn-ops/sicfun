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
#include <unordered_map>
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

/* ---------- Terminal type enum ---------- */

enum class TerminalType : int {
  kContinue  = 0,
  kHeroFold  = 1,
  kRivalFold = 2,
  kShowdown  = 3
};

/* ---------- Action effect descriptor ---------- */

struct ActionEffect {
  double pot_delta_frac = 0.0;
  bool is_fold = false;
  bool is_allin = false;
};

/* ---------- Factored tabular model ---------- */

struct FactoredTabularModel {
  const double* rival_policy = nullptr;   /* [numTypes * numPubStates * numActions] */
  int num_rival_types = 0;
  int num_pub_states = 0;
  const ActionEffect* action_effects = nullptr;  /* [numActions] */
  const double* showdown_equity = nullptr;  /* [numHeroBuckets * numRivalBuckets] */
  int num_hero_buckets = 0;
  int num_rival_buckets = 0;
  const int* terminal_flags = nullptr;  /* [numPubStates * numActions] */
  int num_actions = 0;

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

/** Per-rival particle set: C_i weighted particles for rival i.
  *
  * Implements the b_hat_t^{C_i, i} factor from Def 56.
  * Particles are independently maintained per rival.
  * Resampling is triggered when ESS < ess_threshold * C_i.
  */
struct RivalBeliefSet {
  std::vector<RivalParticle> particles;
  double ess_threshold = kDefaultEssThreshold;
  bool is_normalized_ = false;

  /** Effective sample size: (sum w)^2 / sum(w^2).
    * Returns C_i when all weights are equal (best case).
    * Returns 1.0 when one particle dominates (worst case).
    * Fast path: after normalize(), sum_w == 1.0 so ESS = 1/sum(w_j^2). */
  double ess() const {
    if (particles.empty()) return 0.0;
    if (is_normalized_) {
      // After normalize(), sum_w == 1.0, so ESS = 1/sum(w_j^2)
      double sum_w2 = 0.0;
      for (const auto& p : particles) sum_w2 += p.weight * p.weight;
      return (sum_w2 > 0.0) ? 1.0 / sum_w2 : 0.0;
    }
    // Full computation when not known to be normalized
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
    is_normalized_ = true;
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

  /* Reuse thread-local scratch buffers to avoid heap allocation per call.
   * After the first call, resize() is a no-op when n stays constant. */
  thread_local std::vector<double> cumulative;
  thread_local std::vector<RivalParticle> resampled;
  cumulative.resize(n);
  resampled.resize(n);

  /* Build cumulative weight array from raw (possibly unnormalized) weights.
   * Skip the separate normalize() pass -- resampling resets all weights
   * to 1/N, so normalizing particles beforehand is wasted work. */
  cumulative[0] = belief.particles[0].weight;
  for (int i = 1; i < n; ++i) {
    cumulative[i] = cumulative[i - 1] + belief.particles[i].weight;
  }
  const double total = cumulative[n - 1];
  if (total <= 0.0) return;

  /* Use scaled thresholds against the unnormalized CDF, eliminating
   * the O(N) normalization pass.  Step = total/N, offset ~ U(0, step). */
  const double step = total / static_cast<double>(n);
  std::uniform_real_distribution<double> dist(0.0, step);
  const double u0 = dist(rng);

  int cursor = 0;
  const double equal_weight = 1.0 / static_cast<double>(n);

  for (int j = 0; j < n; ++j) {
    const double threshold = u0 + static_cast<double>(j) * step;
    while (cursor < n - 1 && cumulative[cursor] < threshold) {
      ++cursor;
    }
    resampled[j] = belief.particles[cursor];
    resampled[j].weight = equal_weight;
  }

  belief.particles.swap(resampled);
  belief.is_normalized_ = true;  /* All weights are now exactly 1/N. */
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

    /* Single-pass: build CDF on the stack while validating, then sample.
     * Eliminates the separate cumsum validation loop (halves iterations). */
    double cdf[kMaxActions];
    double running = 0.0;
    bool has_negative = false;
    for (int a = 0; a < dist.num_actions; ++a) {
      has_negative |= (dist.action_probs[a] < 0.0);
      running += dist.action_probs[a];
      cdf[a] = running;
    }
    if (has_negative) return kStatusInvalidConfig;
    if (running <= 0.0) return kStatusDegenerateWeights;

    /* Sample from the pre-built CDF. */
    std::uniform_real_distribution<double> u(0.0, running);
    const double draw = u(rng);
    int selected = dist.num_actions - 1;  /* fallback to last action */
    for (int a = 0; a < dist.num_actions - 1; ++a) {
      if (draw <= cdf[a]) {
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

  belief.is_normalized_ = false;  // weights are about to change

  /* Branchless validation: accumulate bad-value flag via bitwise OR
   * instead of per-element early-exit.  The error path is rare, so
   * removing the unpredictable branch from the hot loop wins more
   * than the lost early-exit saves. */
  bool any_bad = false;
  for (int j = 0; j < n; ++j) {
    const double lik = obs_liks.likelihoods[j];
    any_bad |= (lik < 0.0) | !std::isfinite(lik);
    belief.particles[j].weight *= lik;
  }
  if (any_bad) return false;

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

/* ---------- V2 model-aware sampling and belief update ---------- */

/** Reweight a rival's particles by the likelihood of an observed action.
  *
  * For each particle, multiplies its weight by
  *   P(observed_action | particle.rival_type, pub_state_idx)
  * from the factored tabular model.  If all weights collapse to zero,
  * falls back to uniform weights.
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
    const double unif = 1.0 / static_cast<double>(belief.particles.size());
    for (auto& p : belief.particles) p.weight = unif;
  }
  belief.is_normalized_ = false;
}

/** Encoder that maps a PublicState to a flat index into the tabular model.
  *
  * Discretizes pot and stack into buckets.  The flat index is:
  *   street * num_pot_buckets * num_stack_buckets + pot_b * num_stack_buckets + stack_b
  */
struct PubStateEncoder {
  int num_pot_buckets = 8;
  int num_stack_buckets = 6;
  double pot_bucket_size = 50.0;
  double stack_bucket_size = 0.2;

  inline int encode(const PublicState& pub) const {
    int pot_b = std::min(num_pot_buckets - 1,
        std::max(0, static_cast<int>(pub.pot / pot_bucket_size)));
    int stack_b = num_stack_buckets / 2;
    return pub.street * num_pot_buckets * num_stack_buckets
         + pot_b * num_stack_buckets
         + stack_b;
  }
};

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
  std::array<bool, kMaxActions> action_expanded_{};  /* replaces vector<int> expanded_actions */
  int expanded_count_ = 0;

  /* Children indexed by packed (hero_action, obs_hash) key for O(1) lookup.
   * Values are indices into the solver's flat node arena. */
  std::unordered_map<int64_t, int> child_index_;

  static int64_t pack_key(int hero_action, int obs_hash) {
    return (static_cast<int64_t>(hero_action) << 32) | static_cast<uint32_t>(obs_hash);
  }

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

  /** Select hero action by UCB1. Returns action index.
    * Precomputes log(N(s)) once rather than per-action. */
  int select_action_ucb1(double c, int num_actions) const {
    int best = 0;
    double best_score = -std::numeric_limits<double>::infinity();
    const double log_n = std::log(static_cast<double>(visit_count));
    for (int a = 0; a < num_actions; ++a) {
      const auto& stats = action_stats[a];
      double score;
      if (stats.visit_count == 0) {
        score = std::numeric_limits<double>::infinity();
      } else {
        score = stats.value_sum / stats.visit_count
              + c * std::sqrt(log_n / static_cast<double>(stats.visit_count));
      }
      if (score > best_score) {
        best_score = score;
        best = a;
      }
    }
    return best;
  }

  /** Check if progressive widening allows expanding a new action.
    * Condition: expanded_count_ < c * N(s)^alpha
    * Uses exp(alpha*log(x)) instead of pow(x, alpha) (~10x faster). */
  bool should_widen(double pw_c, double pw_alpha) const {
    const double n = static_cast<double>(visit_count);
    const double limit = (n > 0.0)
        ? pw_c * std::exp(pw_alpha * std::log(n))
        : 0.0;
    return static_cast<double>(expanded_count_) < limit;
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

  /* Flat arena for tree nodes. Index 0 is the root.
   * Eliminates per-node heap allocation (make_unique) during search. */
  std::vector<WPomcpNode> arena_;

  /* Pre-allocated scratch buffer for per-rival action distributions in simulate(). */
  std::vector<RivalActionDist> scratch_rival_dists_;

  /** Pick an action not yet expanded (O(kMaxActions) bitset scan). */
  int pick_unexpanded_action(const WPomcpNode& node, int num_actions) {
    for (int a = 0; a < num_actions; ++a) {
      if (!node.action_expanded_[a]) return a;
    }
    return node.select_action_ucb1(config_.exploration_constant, num_actions);
  }

  /** Find an existing child node or create a new one (O(1) hash lookup).
    * Returns arena index. May grow arena_, invalidating references. */
  int find_or_create_child(int parent_idx, int hero_action, int obs_hash) {
    const int64_t key = WPomcpNode::pack_key(hero_action, obs_hash);
    auto it = arena_[parent_idx].child_index_.find(key);
    if (it != arena_[parent_idx].child_index_.end()) {
      return it->second;
    }
    const int child_idx = static_cast<int>(arena_.size());
    arena_.emplace_back();
    arena_[parent_idx].child_index_[key] = child_idx;
    return child_idx;
  }

  /** Run a single simulation from the given node at the given depth.
    *
    * Returns the discounted cumulative reward from this point forward.
    * Modifies the node's visit counts and value estimates.
    *
    * Takes an arena index (not a reference) because find_or_create_child
    * may grow arena_, invalidating any outstanding references.
    *
    * pub_state is passed separately from the (immutable) root belief because
    * the public state evolves with each transition while the particle sets
    * remain unchanged (current model uses uniform observation likelihoods).
    * This eliminates the O(R * C_i) per-depth particle copy that the
    * previous SimulationContext scratch-buffer approach required.
    */
  double simulate(
      int node_idx,
      const PublicState& pub_state,
      const TransitionModel& model,
      int depth) {

    /* Base cases. */
    if (depth >= config_.max_depth) return 0.0;

    const int num_actions = model.num_hero_actions;
    if (num_actions <= 0) return 0.0;

    /* Phase 1: select action. Arena is not mutated here; local ref is safe
     * and avoids repeated vector indexing in the hot loop. */
    int hero_action;
    {
      auto& node = arena_[node_idx];
      if (node.action_stats.empty()) {
        node.action_stats.resize(num_actions);
      }
      if (node.should_widen(config_.pw_c, config_.pw_alpha)) {
        hero_action = pick_unexpanded_action(node, num_actions);
        if (!node.action_expanded_[hero_action]) {
          node.action_expanded_[hero_action] = true;
          node.expanded_count_++;
        }
      } else {
        hero_action = node.select_action_ucb1(
            config_.exploration_constant, num_actions);
      }
    }

    /* Sample joint rival actions from factored belief. */
    JointRivalAction joint_action;
    {
      const int rc = config_.rival_count();
      auto& rival_dists = scratch_rival_dists_;
      for (int i = 0; i < rc; ++i) {
        rival_dists[i].action_probs =
            model.rival_action_probs +
            i * model.num_hero_actions;
        rival_dists[i].num_actions = model.num_hero_actions;
      }
      int status = sample_joint_action(
          rival_dists.data(), rc, rng_, joint_action);
      if (status != kStatusOk) return 0.0;
    }

    /* TRANSITION: get reward and next state. */
    double reward = 0.0;
    bool is_terminal = false;

    if (model.rewards != nullptr) {
      reward = model.rewards[hero_action];
    }
    if (model.terminal_flags != nullptr && model.terminal_flags[0] != 0) {
      is_terminal = true;
    }

    if (is_terminal) {
      auto& node = arena_[node_idx];
      node.visit_count++;
      auto& ts = node.action_stats[hero_action];
      ts.visit_count++;
      ts.value_sum += reward;
      return reward;
    }

    /* Advance public state for the transition. */
    PublicState next_pub = pub_state;
    next_pub.street = std::min(next_pub.street + 1, 3);
    next_pub.pot += reward;

    /* EXPAND / RECURSE: find or create child node.
     * find_or_create_child may grow arena_, invalidating references. */
    const int obs_hash = next_pub.street * 1000 + hero_action;
    const int child_idx = find_or_create_child(node_idx, hero_action, obs_hash);

    /* Recursive simulation from child with updated public state. */
    const double future = simulate(child_idx, next_pub, model, depth + 1);
    const double total = reward + config_.discount * future;

    /* BACKPROPAGATE (arena growth is complete; local ref is safe). */
    {
      auto& node = arena_[node_idx];
      node.visit_count++;
      auto& bs = node.action_stats[hero_action];
      bs.visit_count++;
      bs.value_sum += total;
    }

    return total;
  }

  /** UCB1 action selection delegating to node method. */
  int select_ucb1(const WPomcpNode& node, int num_actions) const {
    return node.select_action_ucb1(config_.exploration_constant, num_actions);
  }

  /** Update visit count and value sum for an action at a node. */
  inline void update_stats(WPomcpNode& node, int action, double value) {
    node.visit_count++;
    auto& ts = node.action_stats[action];
    ts.visit_count++;
    ts.value_sum += value;
  }

  /** V2 simulation: model-aware 9-phase flow with factored tabular model.
    *
    * Unlike simulate() which uses static particles and pre-baked rewards,
    * simulate_v2 uses the FactoredTabularModel to:
    *   - Check terminal states from the model's terminal_flags
    *   - Sample rival actions conditioned on belief-weighted type distribution
    *   - Compute showdown equity from the model's equity table
    *   - Reweight particles by observed rival actions (real observations)
    *   - Resample when ESS drops below threshold
    *
    * Takes a mutable FactoredBelief copy per simulation (particles evolve).
    */
  double simulate_v2(
      int node_idx,
      PublicState pub_state,
      FactoredBelief& belief,
      const FactoredTabularModel& model,
      const PubStateEncoder& encoder,
      int hero_bucket,
      int depth)
  {
    if (depth >= config_.max_depth || model.num_actions <= 0)
      return 0.0;

    /* Phase 1: Select hero action (progressive widening + UCB1).
     * Scoped so `node` expires before find_or_create_child can
     * grow arena_ and invalidate the reference. */
    int hero_action;
    {
      auto& node = arena_[node_idx];
      if (node.action_stats.empty())
        node.action_stats.resize(model.num_actions);

      if (node.should_widen(config_.pw_c, config_.pw_alpha)) {
        hero_action = pick_unexpanded_action(node, model.num_actions);
        if (!node.action_expanded_[hero_action]) {
          node.action_expanded_[hero_action] = true;
          node.expanded_count_++;
        }
      } else {
        hero_action = select_ucb1(node, model.num_actions);
      }
    }

    /* Phase 2: Check terminal -- hero fold */
    const int pub_idx = encoder.encode(pub_state);
    const auto term = model.terminal_type(pub_idx, hero_action);

    if (term == TerminalType::kHeroFold) {
      const double reward = -(model.action_effects[hero_action].pot_delta_frac * pub_state.pot);
      {
        auto& node = arena_[node_idx];
        update_stats(node, hero_action, reward);
      }
      return reward;
    }

    /* Phase 3: Sample rival actions (belief-weighted type distribution) */
    std::array<int, kMaxRivals> rival_actions{};
    for (int r = 0; r < static_cast<int>(belief.rival_beliefs.size()); ++r) {
      auto& rb = belief.rival_beliefs[r];
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
        const double reward = pub_state.pot;
        {
          auto& node = arena_[node_idx];
          update_stats(node, hero_action, reward);
        }
        return reward;
      }
    }

    /* Phase 5: Showdown check */
    if (term == TerminalType::kShowdown) {
      double eq_sum = 0.0;
      double weight_sum = 0.0;
      for (int r = 0; r < static_cast<int>(belief.rival_beliefs.size()); ++r) {
        for (const auto& p : belief.rival_beliefs[r].particles) {
          eq_sum += p.weight * model.equity(hero_bucket, p.priv_state);
          weight_sum += p.weight;
        }
      }
      const double eq = (weight_sum > 0.0) ? eq_sum / weight_sum : 0.5;
      const double reward = pub_state.pot * (2.0 * eq - 1.0);
      {
        auto& node = arena_[node_idx];
        update_stats(node, hero_action, reward);
      }
      return reward;
    }

    /* Phase 6: Reweight particles by observed rival actions */
    for (int r = 0; r < static_cast<int>(belief.rival_beliefs.size()); ++r) {
      reweight_particles_by_observation(
          belief.rival_beliefs[r], model, rival_actions[r], pub_idx);
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
    if (hero_action > 0) {
      next_pub.street = std::min(next_pub.street + 1, 3);
    }

    /* Phase 8: Observation hash and child lookup.
     * find_or_create_child may grow arena_, invalidating references. */
    int64_t obs_hash = hero_action;
    for (int r = 0; r < static_cast<int>(belief.rival_beliefs.size()); ++r) {
      obs_hash = obs_hash * (model.num_actions + 1) + rival_actions[r];
    }
    const int child_idx = find_or_create_child(node_idx, hero_action, static_cast<int>(obs_hash & 0x7FFFFFFF));

    /* Phase 9: Recurse */
    const double future = simulate_v2(child_idx, next_pub, belief, model, encoder, hero_bucket, depth + 1);
    const double total = config_.discount * future;

    /* Backpropagate: fresh reference after all arena growth is complete. */
    {
      auto& node = arena_[node_idx];
      update_stats(node, hero_action, total);
    }
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

    /* Clear tree for fresh search. Arena index 0 = root node.
     * Pre-reserve to avoid repeated reallocation during simulate(). */
    arena_.clear();
    arena_.reserve(std::max(1024, config_.num_simulations));
    arena_.emplace_back();

    /* Pre-allocate per-simulation scratch buffers sized to rival count. */
    const int rc = config_.rival_count();
    scratch_rival_dists_.resize(rc);

    /* Run simulations. Public state is passed separately from the
     * immutable particle beliefs to avoid O(R*C) copies per depth. */
    int completed = 0;
    for (int sim = 0; sim < config_.num_simulations; ++sim) {
      simulate(0, root_belief.public_state, model, 0);
      ++completed;
    }

    /* Extract action values from root. */
    result.action_values.resize(model.num_hero_actions);
    double best_val = -std::numeric_limits<double>::infinity();
    int best_act = 0;

    /* Initialize action stats if no simulations ran (edge case). */
    auto& root = arena_[0];
    if (root.action_stats.empty()) {
      root.action_stats.resize(model.num_hero_actions);
    }

    for (int a = 0; a < model.num_hero_actions; ++a) {
      result.action_values[a] = root.action_stats[a].mean_value();
      if (result.action_values[a] > best_val) {
        best_val = result.action_values[a];
        best_act = a;
      }
    }

    result.best_action = best_act;
    result.root_value = (best_val == -std::numeric_limits<double>::infinity())
                        ? 0.0 : best_val;
    result.simulations_completed = completed;
    result.tree_node_count = static_cast<int>(arena_.size());
    result.status = kStatusOk;
    return result;
  }

  /** V2 search: run simulations using the factored tabular model.
    *
    * Writes per-action Q-values into out_action_values, the best action
    * into out_best_action, and the root value into out_root_value.
    * Returns kStatusOk on success.
    */
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

    if (config_.num_simulations <= 0) {
      /* No simulations: return uniform Q-values */
      for (int a = 0; a < model.num_actions; ++a)
        out_action_values[a] = 0.0;
      *out_best_action = 0;
      *out_root_value = 0.0;
      return kStatusOk;
    }

    arena_.clear();
    arena_.emplace_back();

    for (int sim = 0; sim < config_.num_simulations; ++sim) {
      FactoredBelief sim_belief = root_belief;
      simulate_v2(0, root_belief.public_state, sim_belief, model, encoder, hero_bucket, 0);
    }

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

/* ---------- JNI-friendly flat entry point (V2) ---------- */

/** JNI-friendly entry point for V2 solver with factored tabular model.
  *
  * Validates inputs, builds a FactoredBelief from flat arrays, constructs
  * the FactoredTabularModel and PubStateEncoder, then runs search_v2.
  *
  * Additional parameters vs solve_raw:
  *   num_rival_types       -- type count for rival policy indexing
  *   num_pub_states        -- public state count for terminal/policy indexing
  *   rival_policy          -- [numTypes * numPubStates * numActions]
  *   action_effects_flat   -- [numActions * 3] (pot_delta_frac, is_fold, is_allin)
  *   showdown_equity       -- [numHeroBuckets * numRivalBuckets]
  *   num_hero_buckets      -- hero bucket count for equity indexing
  *   num_rival_buckets     -- rival bucket count for equity indexing
  *   terminal_flags        -- [numPubStates * numActions]
  *   hero_bucket           -- hero's current bucket index
  *   pot_bucket_size       -- pot discretization granularity
  *
  * Returns kStatusOk on success, or an error code.
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
  /* Null checks */
  if (!particles_per_rival || !particle_types || !particle_priv_states ||
      !particle_weights || !rival_policy || !action_effects_flat ||
      !showdown_equity || !terminal_flags ||
      !out_action_values || !out_best_action || !out_root_value)
    return kStatusNullArray;

  if (rival_count < 1 || rival_count > kMaxRivals) return kStatusInvalidRivalCount;
  if (num_hero_actions < 1 || num_hero_actions > kMaxActions) return kStatusInvalidActionCount;
  if (discount <= 0.0 || discount >= 1.0) return kStatusInvalidConfig;
  if (num_rival_types < 1 || num_pub_states < 1 || num_hero_buckets < 1 || num_rival_buckets < 1)
    return kStatusInvalidConfig;

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

  /* Bounds check particle indices */
  offset = 0;
  for (int r = 0; r < rival_count; ++r) {
    const int pc = particles_per_rival[r];
    for (int j = 0; j < pc; ++j) {
      if (particle_types[offset + j] < 0 || particle_types[offset + j] >= num_rival_types)
        return kStatusInvalidParticleCount;
      if (particle_priv_states[offset + j] < 0 || particle_priv_states[offset + j] >= num_rival_buckets)
        return kStatusInvalidParticleCount;
    }
    offset += pc;
  }

  std::vector<ActionEffect> effects(num_hero_actions);
  for (int a = 0; a < num_hero_actions; ++a) {
    effects[a].pot_delta_frac = action_effects_flat[a * 3 + 0];
    effects[a].is_fold = action_effects_flat[a * 3 + 1] > 0.5;
    effects[a].is_allin = action_effects_flat[a * 3 + 2] > 0.5;
  }

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

  PubStateEncoder encoder;
  encoder.pot_bucket_size = pot_bucket_size;

  WPomcpConfig cfg;
  cfg.num_simulations = num_simulations;
  cfg.discount = discount;
  cfg.exploration_constant = exploration;
  cfg.r_max = r_max;
  cfg.max_depth = max_depth;
  cfg.ess_threshold = ess_threshold;

  /* V2 validate_config requires particles_per_rival to be populated. */
  cfg.particles_per_rival.assign(particles_per_rival, particles_per_rival + rival_count);

  auto status = validate_config(cfg);
  if (status != kStatusOk) return status;

  WPomcpSolver solver(cfg, static_cast<uint64_t>(seed));
  return solver.search_v2(root, model, encoder, hero_bucket,
      out_action_values, out_best_action, out_root_value);
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
