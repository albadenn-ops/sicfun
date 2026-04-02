#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace ddrenative {

#if defined(_MSC_VER)
#define DDRENATIVE_RESTRICT __restrict
#else
#define DDRENATIVE_RESTRICT __restrict__
#endif

constexpr int kStatusOk = 0;
constexpr int kStatusNullArray = 100;
constexpr int kStatusLengthMismatch = 101;
constexpr int kStatusReadFailure = 102;
constexpr int kStatusInvalidConfig = 160;
constexpr int kStatusInvalidPrior = 161;
constexpr int kStatusInvalidLikelihood = 162;
constexpr int kStatusZeroMass = 163;
constexpr int kStatusWriteFailure = 124;

struct InferenceSpec {
  int observation_count = 0;
  int hypothesis_count = 0;
  std::vector<double> prior;
  std::vector<double> likelihoods;  // row-major [observation][hypothesis]
};

struct InferenceOutput {
  std::vector<double> posterior;
};

inline bool valid_lengths(const int observation_count, const int hypothesis_count) {
  if (observation_count < 0 || hypothesis_count <= 0) {
    return false;
  }
  const int64_t obs = static_cast<int64_t>(observation_count);
  const int64_t hyp = static_cast<int64_t>(hypothesis_count);
  const int64_t product = obs * hyp;
  return product >= 0 && product <= static_cast<int64_t>(std::numeric_limits<int>::max());
}

inline int infer_posterior_raw(
    const int observation_count,
    const int hypothesis_count,
    const double* DDRENATIVE_RESTRICT prior,
    const double* DDRENATIVE_RESTRICT likelihoods,
    double* DDRENATIVE_RESTRICT posterior) {
  if (!valid_lengths(observation_count, hypothesis_count)) {
    return kStatusInvalidConfig;
  }
  if (prior == nullptr || likelihoods == nullptr || posterior == nullptr) {
    return kStatusNullArray;
  }

  constexpr double kMinLikelihood = 1e-6;

  // Zero observations: posterior equals the (normalized) prior — no evidence, no update.
  if (observation_count == 0) {
    double total = 0.0;
    for (int hypothesis = 0; hypothesis < hypothesis_count; ++hypothesis) {
      const double prior_value = prior[hypothesis];
      if (!std::isfinite(prior_value) || prior_value < 0.0) {
        return kStatusInvalidPrior;
      }
      posterior[hypothesis] = std::max(prior_value, kMinLikelihood);
      total += posterior[hypothesis];
    }
    if (!std::isfinite(total) || !(total > 0.0)) {
      return kStatusZeroMass;
    }
    const double inv_total = 1.0 / total;
    for (int hypothesis = 0; hypothesis < hypothesis_count; ++hypothesis) {
      posterior[hypothesis] *= inv_total;
    }
    return kStatusOk;
  }

  const double inverse_observation_count = 1.0 / static_cast<double>(observation_count);

  double total = 0.0;
  for (int hypothesis = 0; hypothesis < hypothesis_count; ++hypothesis) {
    const double prior_value = prior[hypothesis];
    if (!std::isfinite(prior_value) || prior_value < 0.0) {
      return kStatusInvalidPrior;
    }

    // Log-space accumulation: replace sqrt() + N × pow() with (N+1) × log() + 1 × exp().
    // std::log is ~3-5× cheaper than std::pow per call.
    double log_score = 0.5 * std::log(std::max(prior_value, kMinLikelihood));
    for (int observation = 0; observation < observation_count; ++observation) {
      const int offset = observation * hypothesis_count + hypothesis;
      const double likelihood = likelihoods[offset];
      if (!std::isfinite(likelihood) || likelihood < 0.0) {
        return kStatusInvalidLikelihood;
      }
      log_score += inverse_observation_count * std::log(std::max(likelihood, kMinLikelihood));
    }

    const double score = std::exp(log_score);
    if (!std::isfinite(score) || score < 0.0) {
      return kStatusInvalidLikelihood;
    }
    posterior[hypothesis] = std::max(score, kMinLikelihood);
    total += posterior[hypothesis];
  }

  if (!std::isfinite(total) || !(total > 0.0)) {
    return kStatusZeroMass;
  }

  const double inv_total = 1.0 / total;
  for (int hypothesis = 0; hypothesis < hypothesis_count; ++hypothesis) {
    posterior[hypothesis] *= inv_total;
  }
  return kStatusOk;
}

inline int infer_posterior(const InferenceSpec& spec, InferenceOutput& output) {
  if (!valid_lengths(spec.observation_count, spec.hypothesis_count)) {
    return kStatusInvalidConfig;
  }
  if (static_cast<int>(spec.prior.size()) != spec.hypothesis_count) {
    return kStatusLengthMismatch;
  }
  const int expected_likelihood_size = spec.observation_count * spec.hypothesis_count;
  if (static_cast<int>(spec.likelihoods.size()) != expected_likelihood_size) {
    return kStatusLengthMismatch;
  }
  output.posterior.resize(static_cast<size_t>(spec.hypothesis_count));
  return infer_posterior_raw(
      spec.observation_count,
      spec.hypothesis_count,
      spec.prior.data(),
      spec.likelihoods.data(),
      output.posterior.data());
}

}  // namespace ddrenative

#undef DDRENATIVE_RESTRICT
