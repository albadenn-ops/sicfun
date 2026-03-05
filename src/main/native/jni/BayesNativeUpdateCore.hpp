#pragma once

#include <cmath>
#include <limits>
#include <vector>

namespace bayesnative {

#if defined(_MSC_VER)
#define BAYESNATIVE_RESTRICT __restrict
#else
#define BAYESNATIVE_RESTRICT __restrict__
#endif

constexpr int kStatusOk = 0;
constexpr int kStatusNullArray = 100;
constexpr int kStatusLengthMismatch = 101;
constexpr int kStatusReadFailure = 102;
constexpr int kStatusInvalidConfig = 160;
constexpr int kStatusInvalidPrior = 161;
constexpr int kStatusInvalidLikelihood = 162;
constexpr int kStatusZeroEvidence = 163;
constexpr int kStatusWriteFailure = 124;

struct UpdateSpec {
  int observation_count = 0;
  int hypothesis_count = 0;
  std::vector<double> prior;
  std::vector<double> likelihoods;  // row-major [observation][hypothesis]
};

struct UpdateOutput {
  std::vector<double> posterior;
  double log_evidence = 0.0;
};

inline bool valid_length_product(const int observation_count, const int hypothesis_count) {
  if (observation_count <= 0 || hypothesis_count <= 0) {
    return false;
  }
  const long long obs = static_cast<long long>(observation_count);
  const long long hyp = static_cast<long long>(hypothesis_count);
  const long long product = obs * hyp;
  return product > 0 && product <= static_cast<long long>(std::numeric_limits<int>::max());
}

inline int update_posterior_raw(
    const int observation_count,
    const int hypothesis_count,
    const double* BAYESNATIVE_RESTRICT prior,
    const double* BAYESNATIVE_RESTRICT likelihoods,
    double* BAYESNATIVE_RESTRICT posterior,
    double* out_log_evidence) {
  if (!valid_length_product(observation_count, hypothesis_count)) {
    return kStatusInvalidConfig;
  }

  if (prior == nullptr || likelihoods == nullptr || posterior == nullptr || out_log_evidence == nullptr) {
    return kStatusNullArray;
  }

  const double eps = 1e-12;
  double prior_sum = 0.0;
  int hypothesis = 0;
  for (; hypothesis + 3 < hypothesis_count; hypothesis += 4) {
    const double value0 = prior[hypothesis];
    const double value1 = prior[hypothesis + 1];
    const double value2 = prior[hypothesis + 2];
    const double value3 = prior[hypothesis + 3];
    if (!std::isfinite(value0) || value0 < 0.0 || !std::isfinite(value1) || value1 < 0.0 ||
        !std::isfinite(value2) || value2 < 0.0 || !std::isfinite(value3) || value3 < 0.0) {
      return kStatusInvalidPrior;
    }
    posterior[hypothesis] = value0;
    posterior[hypothesis + 1] = value1;
    posterior[hypothesis + 2] = value2;
    posterior[hypothesis + 3] = value3;
    prior_sum += (value0 + value1) + (value2 + value3);
  }
  for (; hypothesis < hypothesis_count; ++hypothesis) {
    const double value = prior[hypothesis];
    if (!std::isfinite(value) || value < 0.0) {
      return kStatusInvalidPrior;
    }
    posterior[hypothesis] = value;
    prior_sum += value;
  }

  if (!(prior_sum > 0.0)) {
    return kStatusInvalidPrior;
  }

  const double inv_prior_sum = 1.0 / prior_sum;
  hypothesis = 0;
  for (; hypothesis + 3 < hypothesis_count; hypothesis += 4) {
    posterior[hypothesis] *= inv_prior_sum;
    posterior[hypothesis + 1] *= inv_prior_sum;
    posterior[hypothesis + 2] *= inv_prior_sum;
    posterior[hypothesis + 3] *= inv_prior_sum;
  }
  for (; hypothesis < hypothesis_count; ++hypothesis) {
    posterior[hypothesis] *= inv_prior_sum;
  }

  double log_evidence = 0.0;
  for (int obs = 0; obs < observation_count; ++obs) {
    const double* row =
        likelihoods + static_cast<long long>(obs) * static_cast<long long>(hypothesis_count);
    double evidence = 0.0;
    hypothesis = 0;
    for (; hypothesis + 3 < hypothesis_count; hypothesis += 4) {
      const double updated0 = posterior[hypothesis] * row[hypothesis];
      const double updated1 = posterior[hypothesis + 1] * row[hypothesis + 1];
      const double updated2 = posterior[hypothesis + 2] * row[hypothesis + 2];
      const double updated3 = posterior[hypothesis + 3] * row[hypothesis + 3];
      posterior[hypothesis] = updated0;
      posterior[hypothesis + 1] = updated1;
      posterior[hypothesis + 2] = updated2;
      posterior[hypothesis + 3] = updated3;
      evidence += (updated0 + updated1) + (updated2 + updated3);
    }
    for (; hypothesis < hypothesis_count; ++hypothesis) {
      const double likelihood = row[hypothesis];
      const double updated = posterior[hypothesis] * likelihood;
      posterior[hypothesis] = updated;
      evidence += updated;
    }

    if (!std::isfinite(evidence)) {
      return kStatusInvalidLikelihood;
    }
    if (!(evidence > eps)) {
      return kStatusZeroEvidence;
    }

    const double inv_evidence = 1.0 / evidence;
    hypothesis = 0;
    for (; hypothesis + 3 < hypothesis_count; hypothesis += 4) {
      posterior[hypothesis] *= inv_evidence;
      posterior[hypothesis + 1] *= inv_evidence;
      posterior[hypothesis + 2] *= inv_evidence;
      posterior[hypothesis + 3] *= inv_evidence;
    }
    for (; hypothesis < hypothesis_count; ++hypothesis) {
      posterior[hypothesis] *= inv_evidence;
    }
    log_evidence += std::log(evidence);
  }

  *out_log_evidence = log_evidence;
  return kStatusOk;
}

inline int update_posterior(const UpdateSpec& spec, UpdateOutput& output) {
  if (!valid_length_product(spec.observation_count, spec.hypothesis_count)) {
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
  return update_posterior_raw(
      spec.observation_count,
      spec.hypothesis_count,
      spec.prior.data(),
      spec.likelihoods.data(),
      output.posterior.data(),
      &output.log_evidence);
}

}  // namespace bayesnative

#undef BAYESNATIVE_RESTRICT
