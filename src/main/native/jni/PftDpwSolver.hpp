/*
 * PftDpwSolver.hpp -- Pure C++ PFT-DPW (Particle Filter Tree with Double
 * Progressive Widening) POMDP solver engine.
 *
 * Part of the sicfun poker analytics system's native acceleration layer.
 * Implements single-agent POMDP tree search as specified in SICFUN v0.30.2:
 *   - Definition 29: belief-averaged reward r_bar(b, u) = sum_j w_j * r(x_j, u)
 *   - Definition 30: Q-value estimate Q(b, a) via incremental mean backpropagation
 *   - Definition 31: value function V(b) = max_a Q(b, a)
 *   - Definition 32: optimal Q via argmax UCB1 selection
 *   - Definition 54: weighted particle belief {(x_j, w_j)} with sum w_j = 1
 *   - Definition 55: particle approximation error bound
 *
 * Design decisions:
 *   - Header-only so the JNI binding (HoldemPomcpNativeBindings.cpp) can
 *     include directly without a separate compilation step.
 *   - C++ namespace: pftdpw (mirrors cfrnative convention).
 *   - Uses copy-based JNI pattern (GetXxxArrayRegion) matching CfrNativeSolverCore,
 *     because the solver allocates working memory during execution (tree nodes).
 *   - DPW action widening: add action when |A(b)| <= k_a * N(b)^alpha_a
 *   - DPW observation widening: expand obs when |O(b,a)| <= k_o * N(b,a)^alpha_o
 *   - UCB1 action selection: argmax_a [ Q(b,a) + c * sqrt(ln N(b) / N(b,a)) ]
 *   - Particle belief update: propagate through transition, reweight by obs likelihood,
 *     re-normalize. Falls back to uniform resample if all particles zero out.
 *   - Rollout: sample one particle as "true" state, random actions to depth limit.
 *   - Backpropagation: incremental mean Q(b,a) += (r - Q(b,a)) / N(b,a).
 *
 * Status codes follow the shared JNI protocol (0=ok, 100=null, 101=mismatch,
 * 102=read failure, 124=write failure, 200+=POMDP-specific errors).
 *
 * JNI class: sicfun.holdem.HoldemPomcpNativeBindings
 */

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

/* Status codes (same convention as cfrnative / bayesnative). */
constexpr int kStatusOk = 0;
constexpr int kStatusNullArray = 100;
constexpr int kStatusLengthMismatch = 101;
constexpr int kStatusReadFailure = 102;
constexpr int kStatusWriteFailure = 124;
constexpr int kStatusInvalidConfig = 200;
constexpr int kStatusNoParticles = 201;
constexpr int kStatusTreeOverflow = 202;
constexpr int kStatusSimulationFailed = 203;

/* -------------------------------------------------------------------------
 * Particle belief (Definition 54): {(x_j, w_j)}_{j=1}^C, sum w_j = 1.
 * -------------------------------------------------------------------------*/

/* Weighted particle: a state index and its normalized weight. */
struct Particle {
  int state_idx;
  double weight;
};

/* Particle belief: a set of weighted particles representing P(x | history). */
struct ParticleBelief {
  std::vector<Particle> particles;

  /* Normalize weights so they sum to 1.0.
   * No-op if the particle set is empty or already normalized. */
  void normalize() {
    double total = 0.0;
    for (const auto& p : particles) {
      total += p.weight;
    }
    if (total > 0.0) {
      const double inv = 1.0 / total;
      for (auto& p : particles) {
        p.weight *= inv;
      }
    }
  }

  int size() const { return static_cast<int>(particles.size()); }
};

/* -------------------------------------------------------------------------
 * Configuration (mirrors Scala PftDpwConfig).
 * -------------------------------------------------------------------------*/

struct PftDpwConfig {
  int num_simulations = 1000;  /* Number of MCTS simulations to run. */
  double gamma = 0.99;         /* Discount factor in (0, 1). */
  double r_max = 1.0;          /* Maximum single-step reward (for UCB scaling). */
  double ucb_c = 1.0;          /* UCB1 exploration constant c >= 0. */

  /* Action progressive widening: expand when |A(b)| <= k_a * N(b)^alpha_a */
  double k_action = 2.0;
  double alpha_action = 0.5;

  /* Observation progressive widening: expand when |O(b,a)| <= k_o * N(b,a)^alpha_o */
  double k_obs = 2.0;
  double alpha_obs = 0.5;

  int max_depth = 50;          /* Maximum tree depth before rollout. */
  int num_particles = 100;     /* C: particle count per belief node (informational). */
  int max_tree_nodes = 100000; /* Hard memory cap on number of tree nodes. */
  uint64_t seed = 42;          /* RNG seed for reproducibility. */
};

/* Compile-time sanity: struct must be non-empty. */
static_assert(sizeof(PftDpwConfig) > 0, "PftDpwConfig must be non-empty");

/* -------------------------------------------------------------------------
 * Config validation.
 * -------------------------------------------------------------------------*/

/* Returns kStatusOk if cfg is valid, or a specific error code otherwise. */
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

/* -------------------------------------------------------------------------
 * Generative model: provided as flat arrays from the JNI caller.
 *   transition_table:  [num_states * num_actions] -> next state index (deterministic)
 *   obs_likelihood:    [num_states * num_actions * num_obs] -> P(obs | next_state, action)
 *   reward_table:      [num_states * num_actions] -> immediate reward
 * -------------------------------------------------------------------------*/

struct GenerativeModel {
  const int* transition_table;    /* Row-major: [state * num_actions + action] -> next_state */
  const double* obs_likelihood;   /* Row-major: [(state * num_actions + action) * num_obs + obs] */
  const double* reward_table;     /* Row-major: [state * num_actions + action] -> reward */
  int num_states;
  int num_actions;
  int num_obs;

  PFT_FORCE_INLINE int transition(int state, int action) const {
    return transition_table[state * num_actions + action];
  }

  PFT_FORCE_INLINE double obs_prob(int next_state, int action, int obs) const {
    return obs_likelihood[(next_state * num_actions + action) * num_obs + obs];
  }

  PFT_FORCE_INLINE double reward(int state, int action) const {
    return reward_table[state * num_actions + action];
  }
};

/* -------------------------------------------------------------------------
 * Tree node: holds a particle belief plus per-action statistics.
 * -------------------------------------------------------------------------*/

struct TreeNode {
  ParticleBelief belief;
  int visit_count = 0;
  int depth = 0;

  /* Per-action data, indexed by position in these vectors (action_idx). */
  std::vector<int> action_ids;        /* Which action IDs have been tried. */
  std::vector<double> q_values;       /* Q(b, a) estimate (Def 30), one per action. */
  std::vector<int> action_visits;     /* N(b, a), visit count per action. */

  /* Observation children: action_idx -> list of (obs_id, child_node_id) pairs.
   * Sparse: only expanded observations are stored. */
  std::vector<std::vector<std::pair<int, int>>> obs_children;

  int num_actions() const { return static_cast<int>(action_ids.size()); }

  /* Returns the position index of action_id, or -1 if not found. */
  int find_action(int action_id) const {
    for (int i = 0; i < static_cast<int>(action_ids.size()); ++i) {
      if (action_ids[i] == action_id) {
        return i;
      }
    }
    return -1;
  }

  /* Add a new action to this node's action set, initializing statistics to zero. */
  void add_action(int action_id) {
    action_ids.push_back(action_id);
    q_values.push_back(0.0);
    action_visits.push_back(0);
    obs_children.emplace_back();
  }
};

/* -------------------------------------------------------------------------
 * UCB1 action selection (Definition 32: argmax Q + exploration bonus).
 * -------------------------------------------------------------------------*/

/* Select the action index (into node.action_ids) that maximizes UCB1.
 * Untried actions (visit_count=0) return infinity, so they are always
 * selected before any tried action. */
PFT_FORCE_INLINE int select_action_ucb(const TreeNode& node, double ucb_c) {
  int best_idx = 0;
  double best_val = -std::numeric_limits<double>::infinity();
  /* Use ln(N(b) + 1) to avoid log(0) when node.visit_count == 0. */
  const double log_n = std::log(static_cast<double>(node.visit_count + 1));

  for (int i = 0; i < node.num_actions(); ++i) {
    const double q = node.q_values[i];
    const double explore = (node.action_visits[i] > 0)
        ? ucb_c * std::sqrt(log_n / static_cast<double>(node.action_visits[i]))
        : std::numeric_limits<double>::infinity();
    const double val = q + explore;
    if (val > best_val) {
      best_val = val;
      best_idx = i;
    }
  }
  return best_idx;
}

/* -------------------------------------------------------------------------
 * DPW widening predicates.
 * -------------------------------------------------------------------------*/

/* Should we expand a new action? True when the current action count is within
 * the progressive widening threshold: |A(b)| <= k_a * N(b)^alpha_a. */
PFT_FORCE_INLINE bool should_widen_actions(const TreeNode& node,
                                            double k_a, double alpha_a) {
  const double threshold = k_a * std::pow(
      static_cast<double>(node.visit_count + 1), alpha_a);
  return static_cast<double>(node.num_actions()) <= threshold;
}

/* Should we expand a new observation branch for action_idx?
 * True when: |O(b,a)| <= k_o * N(b,a)^alpha_o. */
PFT_FORCE_INLINE bool should_widen_obs(const TreeNode& node, int action_idx,
                                        double k_o, double alpha_o) {
  const int n_obs = static_cast<int>(node.obs_children[action_idx].size());
  const double threshold = k_o * std::pow(
      static_cast<double>(node.action_visits[action_idx] + 1), alpha_o);
  return static_cast<double>(n_obs) <= threshold;
}

/* -------------------------------------------------------------------------
 * Belief-averaged reward (Definition 29): r_bar(b, u) = sum_j w_j * r(x_j, u)
 * -------------------------------------------------------------------------*/

PFT_FORCE_INLINE double belief_averaged_reward(const ParticleBelief& belief,
                                                int action,
                                                const GenerativeModel& model) {
  double r_bar = 0.0;
  for (const auto& p : belief.particles) {
    r_bar += p.weight * model.reward(p.state_idx, action);
  }
  return r_bar;
}

/* -------------------------------------------------------------------------
 * Particle filter belief update (Definition 54 update step).
 *
 * For each particle (x_j, w_j):
 *   x'_j = T(x_j, action)
 *   w'_j = w_j * O(obs_id | x'_j, action)
 * Then normalize. If all particles collapse to zero weight, resample uniformly
 * from the original belief to prevent belief degeneracy.
 * -------------------------------------------------------------------------*/

inline ParticleBelief particle_filter_update(
    const ParticleBelief& belief,
    int action, int obs_id,
    const GenerativeModel& model,
    std::mt19937_64& rng) {
  ParticleBelief next;
  next.particles.reserve(static_cast<size_t>(belief.size()));

  for (const auto& p : belief.particles) {
    const int next_state = model.transition(p.state_idx, action);
    const double lik = model.obs_prob(next_state, action, obs_id);
    if (lik > 0.0) {
      next.particles.push_back({next_state, p.weight * lik});
    }
  }

  if (next.particles.empty()) {
    /* Particle degeneracy: resample uniformly from original belief. */
    std::uniform_int_distribution<int> dist(0, belief.size() - 1);
    for (int j = 0; j < belief.size(); ++j) {
      const int idx = dist(rng);
      const int ns = model.transition(belief.particles[idx].state_idx, action);
      next.particles.push_back({ns, 1.0});
    }
  }

  next.normalize();
  return next;
}

/* -------------------------------------------------------------------------
 * Random rollout from a particle belief (leaf value estimate).
 *
 * Samples one particle weighted by the belief distribution as the "true" state,
 * then executes random actions until max_depth, accumulating discounted reward.
 * -------------------------------------------------------------------------*/

inline double random_rollout(const ParticleBelief& belief,
                              const GenerativeModel& model,
                              const PftDpwConfig& cfg,
                              int current_depth,
                              std::mt19937_64& rng) {
  /* Sample one particle as the rollout starting state. */
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  double u = uniform(rng);
  double cumw = 0.0;
  int state = belief.particles[0].state_idx;
  for (const auto& p : belief.particles) {
    cumw += p.weight;
    if (u <= cumw) {
      state = p.state_idx;
      break;
    }
  }

  std::uniform_int_distribution<int> action_dist(0, model.num_actions - 1);
  double total_reward = 0.0;
  double discount = 1.0;

  for (int d = current_depth; d < cfg.max_depth; ++d) {
    const int a = action_dist(rng);
    total_reward += discount * model.reward(state, a);
    discount *= cfg.gamma;
    state = model.transition(state, a);
  }

  return total_reward;
}

/* -------------------------------------------------------------------------
 * PFT tree: manages node storage with bounded capacity.
 * -------------------------------------------------------------------------*/

struct PftTree {
  std::vector<TreeNode> nodes;
  int node_count = 0;
  int max_nodes;

  explicit PftTree(int max_n) : max_nodes(max_n) {
    nodes.reserve(static_cast<size_t>(max_n));
  }

  /* Add a new node with the given belief and depth.
   * Returns the node ID, or -1 if the tree is full (capacity exceeded). */
  int add_node(const ParticleBelief& belief, int depth) {
    if (node_count >= max_nodes) return -1;
    nodes.emplace_back();
    const int id = node_count++;
    nodes[id].belief = belief;
    nodes[id].depth = depth;
    return id;
  }
};

/* -------------------------------------------------------------------------
 * PFT-DPW recursive simulate: one MCTS pass from node_id.
 *
 * Returns the simulated reward (used for backpropagation).
 * -------------------------------------------------------------------------*/

inline double simulate(PftTree& tree, int node_id,
                        const GenerativeModel& model,
                        const PftDpwConfig& cfg,
                        std::mt19937_64& rng) {
  TreeNode& node = tree.nodes[node_id];

  /* Depth cutoff: return 0 (boundary condition for finite-horizon rollout). */
  if (node.depth >= cfg.max_depth) {
    return 0.0;
  }

  /* Action progressive widening: try to add a random untried action. */
  if (should_widen_actions(node, cfg.k_action, cfg.alpha_action)) {
    std::uniform_int_distribution<int> adist(0, model.num_actions - 1);
    const int candidate = adist(rng);
    if (node.find_action(candidate) < 0) {
      node.add_action(candidate);
    }
  }

  /* No actions yet: pure rollout. */
  if (node.num_actions() == 0) {
    return random_rollout(node.belief, model, cfg, node.depth, rng);
  }

  /* UCB1 action selection (Def 32). */
  const int action_idx = select_action_ucb(node, cfg.ucb_c);
  const int action_id = node.action_ids[action_idx];

  /* Immediate belief-averaged reward (Def 29). */
  const double r_bar = belief_averaged_reward(node.belief, action_id, model);

  double sim_reward;

  if (should_widen_obs(node, action_idx, cfg.k_obs, cfg.alpha_obs)) {
    /* Observation widening: generate a new observation by sampling a particle,
     * transitioning it, and sampling obs from O(o | x', a). */
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    /* Sample particle weighted by belief. */
    double u = uniform(rng);
    double cumw = 0.0;
    int sampled_state = node.belief.particles[0].state_idx;
    for (const auto& p : node.belief.particles) {
      cumw += p.weight;
      if (u <= cumw) {
        sampled_state = p.state_idx;
        break;
      }
    }
    const int next_state = model.transition(sampled_state, action_id);

    /* Sample observation from O(o | x', a). */
    const double obs_u = uniform(rng);
    double obs_cum = 0.0;
    int obs_id = 0;
    for (int o = 0; o < model.num_obs; ++o) {
      obs_cum += model.obs_prob(next_state, action_id, o);
      if (obs_u <= obs_cum) {
        obs_id = o;
        break;
      }
    }

    /* Create new child node with particle-filter updated belief. */
    ParticleBelief child_belief =
        particle_filter_update(node.belief, action_id, obs_id, model, rng);
    const int child_id = tree.add_node(child_belief, node.depth + 1);

    if (child_id < 0) {
      /* Tree overflow: fall back to rollout from the new belief. */
      return r_bar + cfg.gamma *
             random_rollout(child_belief, model, cfg, node.depth + 1, rng);
    }
    node.obs_children[action_idx].push_back({obs_id, child_id});
    sim_reward = r_bar + cfg.gamma * simulate(tree, child_id, model, cfg, rng);

  } else {
    /* Reuse an existing observation child (uniform random selection). */
    auto& obs_list = node.obs_children[action_idx];
    std::uniform_int_distribution<int> obs_dist(
        0, static_cast<int>(obs_list.size()) - 1);
    const int child_id = obs_list[obs_dist(rng)].second;
    sim_reward = r_bar + cfg.gamma * simulate(tree, child_id, model, cfg, rng);
  }

  /* Backpropagation: incremental mean update (Def 30 sample-based approximation).
   * Q(b,a) <- Q(b,a) + (R - Q(b,a)) / N(b,a) */
  node.action_visits[action_idx] += 1;
  node.visit_count += 1;
  const int n = node.action_visits[action_idx];
  node.q_values[action_idx] += (sim_reward - node.q_values[action_idx]) /
                                 static_cast<double>(n);

  return sim_reward;
}

/* -------------------------------------------------------------------------
 * Result structure: summary of the PFT-DPW solve.
 * -------------------------------------------------------------------------*/

struct PftDpwResult {
  int status = kStatusOk;
  int best_action = -1;     /* Action ID with highest visit count (robust policy). */
  double best_q = 0.0;      /* Q-value of the best action. */
  std::vector<int> action_ids;    /* All tried action IDs at root. */
  std::vector<double> q_values;   /* Q(b, a) for each tried action. */
  std::vector<int> visit_counts;  /* N(b, a) for each tried action. */
  int total_simulations = 0;
  int tree_nodes_used = 0;
};

/* -------------------------------------------------------------------------
 * Main entry point: run PFT-DPW from a root particle belief.
 *
 * Validates config, initializes the tree with the root belief, runs
 * num_simulations MCTS passes, then extracts the best action from root
 * statistics (by visit count, which is more robust than pure Q-value).
 * -------------------------------------------------------------------------*/

inline PftDpwResult solve(const ParticleBelief& root_belief,
                           const GenerativeModel& model,
                           const PftDpwConfig& cfg) {
  PftDpwResult result;

  const int cfg_status = validate_config(cfg);
  if (cfg_status != kStatusOk) {
    result.status = cfg_status;
    return result;
  }
  if (root_belief.size() == 0) {
    result.status = kStatusNoParticles;
    return result;
  }

  std::mt19937_64 rng(cfg.seed);
  PftTree tree(cfg.max_tree_nodes);

  const int root_id = tree.add_node(root_belief, 0);
  if (root_id < 0) {
    result.status = kStatusTreeOverflow;
    return result;
  }

  for (int sim = 0; sim < cfg.num_simulations; ++sim) {
    simulate(tree, root_id, model, cfg, rng);
  }

  /* Extract results from the root node. */
  const TreeNode& root = tree.nodes[root_id];
  result.action_ids = root.action_ids;
  result.q_values = root.q_values;
  result.visit_counts = root.action_visits;
  result.total_simulations = cfg.num_simulations;
  result.tree_nodes_used = tree.node_count;

  /* Best action by visit count (robust to Q-value noise at low visit counts). */
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

}  // namespace pftdpw
