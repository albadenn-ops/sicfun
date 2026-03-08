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

inline void regret_matching(
    const std::vector<double>& regrets,
    const int start,
    const int count,
    double* out_strategy) {
  double positive_sum = 0.0;
  const double* regret_ptr = regrets.data() + start;
  for (int idx = 0; idx < count; ++idx) {
    const double positive = regret_ptr[idx] > 0.0 ? regret_ptr[idx] : 0.0;
    out_strategy[idx] = positive;
    positive_sum += positive;
  }

  if (positive_sum > 0.0) {
    const double inv = 1.0 / positive_sum;
    for (int idx = 0; idx < count; ++idx) {
      out_strategy[idx] *= inv;
    }
  } else {
    const double uniform = 1.0 / static_cast<double>(count);
    for (int idx = 0; idx < count; ++idx) {
      out_strategy[idx] = uniform;
    }
  }
}

inline int solve(const TreeSpec& spec, SolveOutput& output) {
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

  std::vector<double> cumulative_regret(strategy_size, 0.0);
  std::vector<double> cumulative_strategy(strategy_size, 0.0);
  std::vector<double> chance_edge_weights(spec.edge_probabilities.size(), 0.0);

  const int node_count = static_cast<int>(spec.node_types.size());
  for (int node = 0; node < node_count; ++node) {
    if (spec.node_types[node] != kNodeChance) {
      continue;
    }
    const int start = spec.node_starts[node];
    const int count = spec.node_counts[node];
    double prob_sum = 0.0;
    for (int edge = start; edge < start + count; ++edge) {
      prob_sum += spec.edge_probabilities[edge];
    }
    const double inv_prob_sum = 1.0 / prob_sum;
    for (int edge = start; edge < start + count; ++edge) {
      chance_edge_weights[edge] = spec.edge_probabilities[edge] * inv_prob_sum;
    }
  }

  auto cfr = [&](auto&& self, const int node_id, const double reach_p0, const double reach_p1,
                 const int avg_weight) -> double {
    const int node_type = spec.node_types[node_id];
    if (node_type == kNodeTerminal) {
      return spec.terminal_utilities[node_id];
    }

    const int start = spec.node_starts[node_id];
    const int count = spec.node_counts[node_id];

    if (node_type == kNodeChance) {
      double value = 0.0;
      for (int edge = start; edge < start + count; ++edge) {
        const double probability = chance_edge_weights[edge];
        const int child = spec.edge_child_ids[edge];
        value += probability * self(self, child, reach_p0 * probability, reach_p1 * probability, avg_weight);
      }
      return value;
    }

    const int infoset = spec.node_infosets[node_id];
    const int infoset_start = infoset_offsets[infoset];

    std::array<double, kInlineActionBufferSize> strategy_inline{};
    std::array<double, kInlineActionBufferSize> action_utility_inline{};
    std::vector<double> strategy_heap;
    std::vector<double> action_utility_heap;
    double* strategy = nullptr;
    double* action_utility = nullptr;
    if (count <= kInlineActionBufferSize) {
      strategy = strategy_inline.data();
      action_utility = action_utility_inline.data();
    } else {
      strategy_heap.assign(static_cast<size_t>(count), 0.0);
      action_utility_heap.assign(static_cast<size_t>(count), 0.0);
      strategy = strategy_heap.data();
      action_utility = action_utility_heap.data();
    }
    regret_matching(cumulative_regret, infoset_start, count, strategy);

    double node_value = 0.0;
    for (int idx = 0; idx < count; ++idx) {
      const int child = spec.edge_child_ids[start + idx];
      const double action_value =
          (node_type == kNodePlayer0)
              ? self(self, child, reach_p0 * strategy[idx], reach_p1, avg_weight)
              : self(self, child, reach_p0, reach_p1 * strategy[idx], avg_weight);
      action_utility[idx] = action_value;
      node_value += strategy[idx] * action_value;
    }

    if (node_type == kNodePlayer0) {
      const double strategy_scale = static_cast<double>(avg_weight) * reach_p0;
      for (int idx = 0; idx < count; ++idx) {
        const double regret_delta = reach_p1 * (action_utility[idx] - node_value);
        double updated = cumulative_regret[infoset_start + idx] + regret_delta;
        if (spec.cfr_plus && updated < 0.0) {
          updated = 0.0;
        }
        cumulative_regret[infoset_start + idx] = updated;
        cumulative_strategy[infoset_start + idx] += strategy_scale * strategy[idx];
      }
    } else {
      const double strategy_scale = static_cast<double>(avg_weight) * reach_p1;
      for (int idx = 0; idx < count; ++idx) {
        const double regret_delta = reach_p0 * (node_value - action_utility[idx]);
        double updated = cumulative_regret[infoset_start + idx] + regret_delta;
        if (spec.cfr_plus && updated < 0.0) {
          updated = 0.0;
        }
        cumulative_regret[infoset_start + idx] = updated;
        cumulative_strategy[infoset_start + idx] += strategy_scale * strategy[idx];
      }
    }
    return node_value;
  };

  for (int iteration = 1; iteration <= spec.iterations; ++iteration) {
    const int avg_weight = averaging_weight(spec, iteration);
    (void)cfr(cfr, spec.root_node_id, 1.0, 1.0, avg_weight);
  }

  for (int infoset = 0; infoset < infoset_count; ++infoset) {
    const int start = infoset_offsets[infoset];
    const int count = spec.infoset_action_counts[infoset];
    double sum = 0.0;
    for (int idx = 0; idx < count; ++idx) {
      sum += cumulative_strategy[start + idx];
    }
    if (sum > 0.0) {
      const double inv = 1.0 / sum;
      for (int idx = 0; idx < count; ++idx) {
        output.average_strategies[start + idx] = cumulative_strategy[start + idx] * inv;
      }
    } else {
      std::array<double, kInlineActionBufferSize> fallback_inline{};
      std::vector<double> fallback_heap;
      double* fallback = nullptr;
      if (count <= kInlineActionBufferSize) {
        fallback = fallback_inline.data();
      } else {
        fallback_heap.assign(static_cast<size_t>(count), 0.0);
        fallback = fallback_heap.data();
      }
      regret_matching(cumulative_regret, start, count, fallback);
      for (int idx = 0; idx < count; ++idx) {
        output.average_strategies[start + idx] = fallback[idx];
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
        const double probability = chance_edge_weights[edge];
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

}  // namespace cfrnative
