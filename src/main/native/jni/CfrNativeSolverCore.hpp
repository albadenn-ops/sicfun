#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace cfrnative {

constexpr int kStatusOk = 0;
constexpr int kStatusNullArray = 100;
constexpr int kStatusLengthMismatch = 101;
constexpr int kStatusReadFailure = 102;
constexpr int kStatusInvalidMode = 111;
constexpr int kStatusWriteFailure = 124;
constexpr int kStatusInvalidRootNode = 146;
constexpr int kStatusInvalidIterations = 147;
constexpr int kStatusInvalidNodeType = 148;
constexpr int kStatusInvalidNodeLayout = 149;
constexpr int kStatusInvalidChildIndex = 150;
constexpr int kStatusInvalidInfoSetIndex = 151;
constexpr int kStatusInfoSetActionMismatch = 152;
constexpr int kStatusInvalidChanceProbabilities = 153;

constexpr int kNodeTerminal = 0;
constexpr int kNodeChance = 1;
constexpr int kNodePlayer0 = 2;
constexpr int kNodePlayer1 = 3;
constexpr int kInlineActionBufferSize = 8;
constexpr int kProbScaleBits = 30;
constexpr int kProbScale = 1 << kProbScaleBits;
constexpr int kFixedValScaleBits = 13;
constexpr int kFixedValScale = 1 << kFixedValScaleBits;
constexpr int kRegretRescaleThreshold = INT32_MAX / 4;
constexpr int64_t kStrategyRescaleThreshold = INT64_MAX / 4;

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

struct SolveOutput {
  std::vector<double> average_strategies;
  double expected_value_player0 = 0.0;
};

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

struct SolveOutputFixed {
  std::vector<int> average_strategies_raw;
  int expected_value_player0_raw = 0;
};

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

inline int averaging_weight(const TreeSpec& spec, const int iteration) {
  if (iteration <= spec.averaging_delay) {
    return 0;
  }
  if (!spec.linear_averaging) {
    return 1;
  }
  return iteration - spec.averaging_delay;
}

inline int averaging_weight(const TreeSpecFixed& spec, const int iteration) {
  if (iteration <= spec.averaging_delay) {
    return 0;
  }
  if (!spec.linear_averaging) {
    return 1;
  }
  return iteration - spec.averaging_delay;
}

template <typename F>
inline void regret_matching_fp(
    const std::vector<F>& regrets,
    const int start,
    const int count,
    F* out_strategy) {
  F positive_sum = F(0);
  const F* regret_ptr = regrets.data() + start;
  for (int idx = 0; idx < count; ++idx) {
    const F positive = regret_ptr[idx] > F(0) ? regret_ptr[idx] : F(0);
    out_strategy[idx] = positive;
    positive_sum += positive;
  }

  if (positive_sum > F(0)) {
    const F inv = F(1) / positive_sum;
    for (int idx = 0; idx < count; ++idx) {
      out_strategy[idx] *= inv;
    }
  } else {
    const F uniform = F(1) / static_cast<F>(count);
    for (int idx = 0; idx < count; ++idx) {
      out_strategy[idx] = uniform;
    }
  }
}

inline void regret_matching(
    const std::vector<double>& regrets,
    const int start,
    const int count,
    double* out_strategy) {
  regret_matching_fp(regrets, start, count, out_strategy);
}

template <typename F>
inline int solve_impl(const TreeSpec& spec, SolveOutput& output) {
  const int validation = validate_tree(spec);
  if (validation != kStatusOk) {
    return validation;
  }

  const int infoset_count = static_cast<int>(spec.infoset_action_counts.size());
  std::vector<int> infoset_offsets(infoset_count + 1, 0);
  for (int infoset = 0; infoset < infoset_count; ++infoset) {
    infoset_offsets[infoset + 1] = infoset_offsets[infoset] + spec.infoset_action_counts[infoset];
  }
  const int strategy_size = infoset_offsets[infoset_count];
  output.average_strategies.assign(strategy_size, 0.0);

  std::vector<F> cumulative_regret(strategy_size, F(0));
  std::vector<F> cumulative_strategy(strategy_size, F(0));
  std::vector<F> chance_edge_weights(spec.edge_probabilities.size(), F(0));

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
      chance_edge_weights[edge] = static_cast<F>(spec.edge_probabilities[edge]) * inv_prob_sum;
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
        const F probability = chance_edge_weights[edge];
        const int child = spec.edge_child_ids[edge];
        value += probability * self(self, child, reach_p0 * probability, reach_p1 * probability, avg_weight);
      }
      return value;
    }

    const int infoset = spec.node_infosets[node_id];
    const int infoset_start = infoset_offsets[infoset];

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
    regret_matching_fp(cumulative_regret, infoset_start, count, strategy);

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

    if (node_type == kNodePlayer0) {
      const F strategy_scale = static_cast<F>(avg_weight) * reach_p0;
      for (int idx = 0; idx < count; ++idx) {
        const F regret_delta = reach_p1 * (action_utility[idx] - node_value);
        F updated = cumulative_regret[infoset_start + idx] + regret_delta;
        if (spec.cfr_plus && updated < F(0)) {
          updated = F(0);
        }
        cumulative_regret[infoset_start + idx] = updated;
        cumulative_strategy[infoset_start + idx] += strategy_scale * strategy[idx];
      }
    } else {
      const F strategy_scale = static_cast<F>(avg_weight) * reach_p1;
      for (int idx = 0; idx < count; ++idx) {
        const F regret_delta = reach_p0 * (node_value - action_utility[idx]);
        F updated = cumulative_regret[infoset_start + idx] + regret_delta;
        if (spec.cfr_plus && updated < F(0)) {
          updated = F(0);
        }
        cumulative_regret[infoset_start + idx] = updated;
        cumulative_strategy[infoset_start + idx] += strategy_scale * strategy[idx];
      }
    }
    return node_value;
  };

  for (int iteration = 1; iteration <= spec.iterations; ++iteration) {
    const int avg_weight = averaging_weight(spec, iteration);
    (void)cfr(cfr, spec.root_node_id, F(1), F(1), avg_weight);
  }

  for (int infoset = 0; infoset < infoset_count; ++infoset) {
    const int start = infoset_offsets[infoset];
    const int count = spec.infoset_action_counts[infoset];
    F sum = F(0);
    for (int idx = 0; idx < count; ++idx) {
      sum += cumulative_strategy[start + idx];
    }
    if (sum > F(0)) {
      const F inv = F(1) / sum;
      for (int idx = 0; idx < count; ++idx) {
        output.average_strategies[start + idx] = static_cast<double>(cumulative_strategy[start + idx] * inv);
      }
    } else {
      std::array<F, kInlineActionBufferSize> fallback_inline{};
      std::vector<F> fallback_heap;
      F* fallback = nullptr;
      if (count <= kInlineActionBufferSize) {
        fallback = fallback_inline.data();
      } else {
        fallback_heap.assign(static_cast<size_t>(count), F(0));
        fallback = fallback_heap.data();
      }
      regret_matching_fp(cumulative_regret, start, count, fallback);
      for (int idx = 0; idx < count; ++idx) {
        output.average_strategies[start + idx] = static_cast<double>(fallback[idx]);
      }
    }
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
        const double probability = static_cast<double>(chance_edge_weights[edge]);
        value += probability * self(self, spec.edge_child_ids[edge]);
      }
      return value;
    }

    const int infoset = spec.node_infosets[node_id];
    const int infoset_start = infoset_offsets[infoset];
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

inline int solve(const TreeSpec& spec, SolveOutput& output) {
  return solve_impl<double>(spec, output);
}

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

inline int multiply_prob_raw(const int left, const int right) {
  return static_cast<int>((static_cast<int64_t>(left) * static_cast<int64_t>(right)) >> kProbScaleBits);
}

// Signed round-to-nearest shift by 30 bits. Matches Scala roundShift30Signed:
// arithmetic right-shift alone biases small negative products toward -infinity,
// so we round on absolute magnitude before restoring the sign.
inline int round_shift_30_signed(const int64_t product) {
  const int64_t abs_product = product >= 0 ? product : -product;
  const int rounded = static_cast<int>((abs_product + (1LL << 29)) >> kProbScaleBits);
  return product >= 0 ? rounded : -rounded;
}

inline int multiply_fixed_by_prob_raw(const int value_raw, const int probability_raw) {
  return round_shift_30_signed(
      static_cast<int64_t>(value_raw) * static_cast<int64_t>(probability_raw));
}

inline void write_uniform_probabilities_raw(const int count, int* out_strategy_raw) {
  const int base = kProbScale / count;
  const int remainder = kProbScale - (base * count);
  for (int idx = 0; idx < count; ++idx) {
    out_strategy_raw[idx] = base + (idx < remainder ? 1 : 0);
  }
}

inline void normalize_non_negative_weights_to_prob_raw(
    const int64_t* weights,
    const int count,
    int* out_probabilities_raw) {
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

inline void normalize_non_negative_edge_weights_to_prob_raw(
    const std::vector<int>& weights,
    const int start,
    const int count,
    int* out_probabilities_raw) {
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

inline int checked_fixed_raw(const int64_t raw) {
  if (raw < static_cast<int64_t>(INT32_MIN) || raw > static_cast<int64_t>(INT32_MAX)) {
    return raw > 0 ? INT32_MAX : INT32_MIN;
  }
  return static_cast<int>(raw);
}

inline void halve_int_array(std::vector<int32_t>& values, const int start, const int count) {
  for (int idx = 0; idx < count; ++idx) {
    values[start + idx] /= 2;
  }
}

inline void halve_long_array(std::vector<int64_t>& values, const int start, const int count) {
  for (int idx = 0; idx < count; ++idx) {
    values[start + idx] /= 2;
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

inline void regret_matching_fixed(
    const std::vector<int32_t>& regrets,
    const int start,
    const int count,
    int* out_strategy_raw) {
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

    // Compute regret deltas and strategy deltas into inline buffers.
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
      strategy_delta_heap.assign(static_cast<size_t>(count), 0);
      regret_deltas = regret_delta_heap.data();
      strategy_deltas = strategy_delta_heap.data();
    }

    if (node_type == kNodePlayer0) {
      for (int idx = 0; idx < count; ++idx) {
        regret_deltas[idx] =
            multiply_fixed_by_prob_raw(action_utility_raw[idx] - node_value_raw, reach_p1_raw);
        strategy_deltas[idx] =
            static_cast<int64_t>(avg_weight) * static_cast<int64_t>(multiply_prob_raw(reach_p0_raw, strategy_raw[idx]));
      }
    } else {
      for (int idx = 0; idx < count; ++idx) {
        regret_deltas[idx] =
            multiply_fixed_by_prob_raw(node_value_raw - action_utility_raw[idx], reach_p0_raw);
        strategy_deltas[idx] =
            static_cast<int64_t>(avg_weight) * static_cast<int64_t>(multiply_prob_raw(reach_p1_raw, strategy_raw[idx]));
      }
    }

    // Emergency rescaling before update (matches Scala applyFixedActionUpdates).
    while (needs_regret_emergency_rescale(cumulative_regret, regret_deltas, infoset_start, count)) {
      halve_int_array(cumulative_regret, infoset_start, count);
    }
    while (needs_strategy_emergency_rescale(cumulative_strategy, strategy_deltas, infoset_start, count)) {
      halve_long_array(cumulative_strategy, infoset_start, count);
    }

    // Apply regret updates with CFR+ clamping and threshold rescaling.
    bool rescale_regret = false;
    for (int idx = 0; idx < count; ++idx) {
      int64_t updated =
          static_cast<int64_t>(cumulative_regret[infoset_start + idx]) +
          static_cast<int64_t>(regret_deltas[idx]);
      if (spec.cfr_plus && updated < 0) {
        updated = 0;
      }
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
    for (int idx = 0; idx < count; ++idx) {
      cumulative_strategy[infoset_start + idx] += strategy_deltas[idx];
      if (cumulative_strategy[infoset_start + idx] >= kStrategyRescaleThreshold) {
        rescale_strategy = true;
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
