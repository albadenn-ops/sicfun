#pragma once

#include <cmath>
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
  const long long obs = static_cast<long long>(observation_count);
  const long long hyp = static_cast<long long>(hypothesis_count);
  const long long product = obs * hyp;
  return product >= 0 && product <= static_cast<long long>(std::numeric_limits<int>::max());
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
  const int effective_observation_count = observation_count > 0 ? observation_count : 1;
  const double inverse_observation_count = 1.0 / static_cast<double>(effective_observation_count);

  double total = 0.0;
  for (int hypothesis = 0; hypothesis < hypothesis_count; ++hypothesis) {
    const double prior_value = prior[hypothesis];
    if (!std::isfinite(prior_value) || prior_value < 0.0) {
      return kStatusInvalidPrior;
    }

    double score = std::sqrt(std::max(prior_value, kMinLikelihood));
    for (int observation = 0; observation < observation_count; ++observation) {
      const int offset = observation * hypothesis_count + hypothesis;
      const double likelihood = likelihoods[offset];
      if (!std::isfinite(likelihood) || likelihood < 0.0) {
        return kStatusInvalidLikelihood;
      }
      score *= std::pow(std::max(likelihood, kMinLikelihood), inverse_observation_count);
    }

    if (!std::isfinite(score) || score < 0.0) {
      return kStatusInvalidLikelihood;
    }
    const double clipped = std::max(score, kMinLikelihood);
    posterior[hypothesis] = clipped;
    total += clipped;
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
