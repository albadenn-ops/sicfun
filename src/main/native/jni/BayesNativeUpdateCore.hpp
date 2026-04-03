/*
 * BayesNativeUpdateCore.hpp -- Pure C++ Bayesian posterior update engine.
 *
 * Part of the sicfun poker analytics system's native acceleration layer.
 * This header implements sequential Bayesian updating: given a prior probability
 * distribution over hypotheses (e.g., opponent hand ranges) and a matrix of
 * likelihoods for observed actions, it computes the posterior distribution and
 * the cumulative log-evidence.
 *
 * Design decisions:
 *   - Header-only so both CPU (HoldemBayesNativeCpuBindings.cpp) and GPU
 *     (HoldemBayesNativeGpuBindings.cu) JNI bindings can share the same core logic.
 *   - Manual 4-wide loop unrolling for the inner hypothesis loop to improve
 *     instruction-level parallelism on x86 without requiring SIMD intrinsics.
 *   - Uses __restrict pointers to allow the compiler to assume no aliasing
 *     between prior, likelihoods, and posterior buffers.
 *   - All status codes are integer constants matching the JNI error protocol
 *     used across all sicfun native bindings (100=null, 101=mismatch, etc.).
 *   - The update_posterior_raw function operates on raw pointers for zero-copy
 *     JNI critical array access; update_posterior is a convenience wrapper
 *     using std::vector for non-JNI callers.
 *
 * Algorithm:
 *   1. Normalize the prior to sum to 1.0.
 *   2. For each observation, multiply the current posterior by the likelihood
 *      row, then re-normalize. Accumulate log(evidence) for marginal likelihood.
 *   3. After all observations, posterior[h] = P(H=h | all observations).
 *
 * JNI class: sicfun.holdem.HoldemBayesNativeCpuBindings (CPU path)
 *            sicfun.holdem.HoldemBayesNativeGpuBindings (GPU path, same core)
 */

#pragma once

#include <cmath>
#include <limits>
#include <vector>

namespace bayesnative {

/* Compiler-portable restrict qualifier to enable aliasing optimizations. */
#if defined(_MSC_VER)
#define BAYESNATIVE_RESTRICT __restrict
#else
#define BAYESNATIVE_RESTRICT __restrict__
#endif

/* Status codes returned to JVM via JNI. Shared across all native bindings. */
constexpr int kStatusOk = 0;                 /* Success. */
constexpr int kStatusNullArray = 100;         /* A required JNI array argument was null. */
constexpr int kStatusLengthMismatch = 101;    /* Array dimensions do not match expected sizes. */
constexpr int kStatusReadFailure = 102;       /* JNI GetPrimitiveArrayCritical / GetXxxArrayRegion failed. */
constexpr int kStatusInvalidConfig = 160;     /* observation_count or hypothesis_count out of range. */
constexpr int kStatusInvalidPrior = 161;      /* Prior contains NaN, Inf, or negative values, or sums to zero. */
constexpr int kStatusInvalidLikelihood = 162; /* Likelihood contains NaN or Inf. */
constexpr int kStatusZeroEvidence = 163;      /* Evidence (sum of posterior * likelihood) collapsed to zero. */
constexpr int kStatusWriteFailure = 124;      /* JNI SetXxxArrayRegion / ReleasePrimitiveArrayCritical failed. */

/* Input specification for a Bayesian update. Used by the convenience wrapper. */
struct UpdateSpec {
  int observation_count = 0;    /* Number of observed actions (rows in likelihood matrix). */
  int hypothesis_count = 0;     /* Number of hypotheses, e.g., opponent hand buckets. */
  std::vector<double> prior;    /* Prior distribution, length = hypothesis_count. */
  std::vector<double> likelihoods;  /* Row-major likelihood matrix [observation][hypothesis].
                                       Length = observation_count * hypothesis_count. */
};

/* Output from a Bayesian update. */
struct UpdateOutput {
  std::vector<double> posterior;   /* Normalized posterior distribution, length = hypothesis_count. */
  double log_evidence = 0.0;      /* Cumulative log-evidence: sum of log(evidence) per observation.
                                     Useful for model comparison / marginal likelihood. */
};

/* Validates that observation_count * hypothesis_count does not overflow int.
 * Returns false if either dimension is non-positive or the product exceeds INT_MAX. */
inline bool valid_length_product(const int observation_count, const int hypothesis_count) {
  if (observation_count <= 0 || hypothesis_count <= 0) {
    return false;
  }
  const long long obs = static_cast<long long>(observation_count);
  const long long hyp = static_cast<long long>(hypothesis_count);
  const long long product = obs * hyp;
  return product > 0 && product <= static_cast<long long>(std::numeric_limits<int>::max());
}

/*
 * Core Bayesian posterior update on raw pointers.
 *
 * Parameters:
 *   observation_count  -- Number of sequential observations (rows in likelihood matrix).
 *   hypothesis_count   -- Number of hypotheses (columns).
 *   prior              -- Prior distribution array, length = hypothesis_count.
 *   likelihoods        -- Row-major likelihood matrix, length = observation_count * hypothesis_count.
 *                         likelihoods[obs * hypothesis_count + hyp] = P(obs | hyp).
 *   posterior          -- Output buffer, length = hypothesis_count. Will contain the
 *                         normalized posterior after all observations are applied.
 *   out_log_evidence   -- Output scalar. Receives the cumulative log-evidence
 *                         (sum of log P(observation_i | data_so_far)).
 *
 * Returns: kStatusOk on success, or an error status code.
 *
 * The algorithm processes observations sequentially:
 *   posterior[h] = prior[h] / sum(prior)
 *   for each observation:
 *     posterior[h] *= likelihood[obs][h]
 *     evidence = sum(posterior)
 *     posterior[h] /= evidence
 *     log_evidence += log(evidence)
 *
 * The inner loops use manual 4-wide unrolling to improve ILP on scalar CPUs.
 * This function is designed for zero-copy use with JNI GetPrimitiveArrayCritical.
 */
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

  const double eps = 1e-12;  /* Minimum evidence threshold to avoid division by zero. */

  /* Step 1: Validate prior values and copy into posterior buffer.
   * Uses 4-wide unrolling to reduce loop overhead and improve ILP. */
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

  /* Step 2: Normalize prior into a proper probability distribution. */
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

  /* Step 3: Sequential Bayesian update — for each observation, multiply posterior
   * by the likelihood row, re-normalize, and accumulate log-evidence. */
  double log_evidence = 0.0;
  for (int obs = 0; obs < observation_count; ++obs) {
    /* Pointer to the likelihood row for this observation. Uses long long
     * multiplication to avoid integer overflow for large matrices. */
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

/*
 * Tempered Bayesian posterior update (SICFUN v0.30.2 Def 15A/15B).
 *
 * Applies two-layer tempered likelihood before the standard Bayesian update:
 *   tempered_likelihood = pow(raw_likelihood, kappa_temp) + delta_floor * eta[h]
 *
 * When kappa_temp == 1.0 and delta_floor == 0.0, this reduces to the standard
 * update_posterior_raw (no pow, no additive floor).
 *
 * Legacy mode (use_legacy_form = true):
 *   tempered_likelihood = (1 - delta_floor) * raw_likelihood + delta_floor * eta[h]
 *   This recovers v0.29.1 epsilon-smoothing exactly.
 *
 * Parameters:
 *   observation_count  -- Number of sequential observations (rows in likelihood matrix).
 *   hypothesis_count   -- Number of hypotheses (columns).
 *   prior              -- Prior distribution array, length = hypothesis_count.
 *   likelihoods        -- Row-major likelihood matrix, length = observation_count * hypothesis_count.
 *   kappa_temp         -- Power-posterior exponent, in (0, 1].
 *   delta_floor        -- Additive safety floor, >= 0.
 *   eta                -- Full-support distribution over hypotheses, length = hypothesis_count.
 *                         If nullptr, uniform 1/hypothesis_count is used.
 *   use_legacy_form    -- If true, use legacy formula (1-eps)*L + eps*eta.
 *   posterior          -- Output buffer, length = hypothesis_count.
 *   out_log_evidence   -- Output scalar for cumulative log-evidence.
 *
 * Returns: kStatusOk on success, or an error status code.
 */
inline int update_posterior_tempered_raw(
    const int observation_count,
    const int hypothesis_count,
    const double* BAYESNATIVE_RESTRICT prior,
    const double* BAYESNATIVE_RESTRICT likelihoods,
    const double kappa_temp,
    const double delta_floor,
    const double* BAYESNATIVE_RESTRICT eta,
    const bool use_legacy_form,
    double* BAYESNATIVE_RESTRICT posterior,
    double* out_log_evidence) {
  if (!valid_length_product(observation_count, hypothesis_count)) {
    return kStatusInvalidConfig;
  }

  if (prior == nullptr || likelihoods == nullptr || posterior == nullptr || out_log_evidence == nullptr) {
    return kStatusNullArray;
  }

  /* Validate tempering parameters. */
  if (!(kappa_temp > 0.0 && kappa_temp <= 1.0)) {
    return kStatusInvalidConfig;
  }
  if (!(delta_floor >= 0.0)) {
    return kStatusInvalidConfig;
  }

  const double eps = 1e-12;

  /* Precompute default eta = 1/hypothesis_count if eta is null. */
  const double default_eta_val = 1.0 / static_cast<double>(hypothesis_count);

  /* Fast path: when kappa_temp == 1.0 and delta_floor == 0.0 and not legacy,
   * delegate to the original untampered function (no pow, no additive floor). */
  if (kappa_temp == 1.0 && delta_floor == 0.0 && !use_legacy_form) {
    return update_posterior_raw(
        observation_count, hypothesis_count, prior, likelihoods, posterior, out_log_evidence);
  }

  /* Step 1: Validate prior values and copy into posterior buffer. */
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

  /* Step 2: Normalize prior. */
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

  /* Step 3: Sequential Bayesian update with tempered likelihoods. */
  double log_evidence = 0.0;
  for (int obs = 0; obs < observation_count; ++obs) {
    const double* row =
        likelihoods + static_cast<long long>(obs) * static_cast<long long>(hypothesis_count);
    double evidence = 0.0;

    hypothesis = 0;
    for (; hypothesis + 3 < hypothesis_count; hypothesis += 4) {
      /* Apply tempering to each of 4 likelihoods. */
      double tempered0, tempered1, tempered2, tempered3;
      const double raw0 = row[hypothesis];
      const double raw1 = row[hypothesis + 1];
      const double raw2 = row[hypothesis + 2];
      const double raw3 = row[hypothesis + 3];
      const double eta0 = (eta != nullptr) ? eta[hypothesis]     : default_eta_val;
      const double eta1 = (eta != nullptr) ? eta[hypothesis + 1] : default_eta_val;
      const double eta2 = (eta != nullptr) ? eta[hypothesis + 2] : default_eta_val;
      const double eta3 = (eta != nullptr) ? eta[hypothesis + 3] : default_eta_val;

      if (use_legacy_form) {
        /* Legacy: (1 - delta_floor) * raw + delta_floor * eta */
        const double one_minus_delta = 1.0 - delta_floor;
        tempered0 = one_minus_delta * raw0 + delta_floor * eta0;
        tempered1 = one_minus_delta * raw1 + delta_floor * eta1;
        tempered2 = one_minus_delta * raw2 + delta_floor * eta2;
        tempered3 = one_minus_delta * raw3 + delta_floor * eta3;
      } else if (kappa_temp == 1.0) {
        /* Two-layer with kappa=1: raw + delta * eta (no pow). */
        tempered0 = raw0 + delta_floor * eta0;
        tempered1 = raw1 + delta_floor * eta1;
        tempered2 = raw2 + delta_floor * eta2;
        tempered3 = raw3 + delta_floor * eta3;
      } else {
        /* Full two-layer: pow(raw, kappa) + delta * eta. */
        tempered0 = std::pow(std::max(0.0, raw0), kappa_temp) + delta_floor * eta0;
        tempered1 = std::pow(std::max(0.0, raw1), kappa_temp) + delta_floor * eta1;
        tempered2 = std::pow(std::max(0.0, raw2), kappa_temp) + delta_floor * eta2;
        tempered3 = std::pow(std::max(0.0, raw3), kappa_temp) + delta_floor * eta3;
      }

      const double updated0 = posterior[hypothesis]     * tempered0;
      const double updated1 = posterior[hypothesis + 1] * tempered1;
      const double updated2 = posterior[hypothesis + 2] * tempered2;
      const double updated3 = posterior[hypothesis + 3] * tempered3;
      posterior[hypothesis]     = updated0;
      posterior[hypothesis + 1] = updated1;
      posterior[hypothesis + 2] = updated2;
      posterior[hypothesis + 3] = updated3;
      evidence += (updated0 + updated1) + (updated2 + updated3);
    }
    for (; hypothesis < hypothesis_count; ++hypothesis) {
      const double raw = row[hypothesis];
      const double eta_val = (eta != nullptr) ? eta[hypothesis] : default_eta_val;
      double tempered;
      if (use_legacy_form) {
        tempered = (1.0 - delta_floor) * raw + delta_floor * eta_val;
      } else if (kappa_temp == 1.0) {
        tempered = raw + delta_floor * eta_val;
      } else {
        tempered = std::pow(std::max(0.0, raw), kappa_temp) + delta_floor * eta_val;
      }
      const double updated = posterior[hypothesis] * tempered;
      posterior[hypothesis] = updated;
      evidence += updated;
    }

    if (!std::isfinite(evidence)) {
      return kStatusInvalidLikelihood;
    }

    /* Def 15B: when delta=0 and evidence=0, preserve prior. */
    if (!(evidence > eps)) {
      if (delta_floor == 0.0) {
        /* Preserve the current posterior (which at this point is the
         * normalized prior if obs==0, or the last valid posterior). */
        log_evidence += std::log(eps);
        continue;
      }
      return kStatusZeroEvidence;
    }

    const double inv_evidence = 1.0 / evidence;
    hypothesis = 0;
    for (; hypothesis + 3 < hypothesis_count; hypothesis += 4) {
      posterior[hypothesis]     *= inv_evidence;
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

/*
 * Convenience wrapper for update_posterior_tempered_raw using std::vector containers.
 */
inline int update_posterior_tempered(const UpdateSpec& spec, const double kappa_temp,
    const double delta_floor, const std::vector<double>& eta,
    const bool use_legacy_form, UpdateOutput& output) {
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
  if (!eta.empty() && static_cast<int>(eta.size()) != spec.hypothesis_count) {
    return kStatusLengthMismatch;
  }
  output.posterior.resize(static_cast<size_t>(spec.hypothesis_count));
  return update_posterior_tempered_raw(
      spec.observation_count,
      spec.hypothesis_count,
      spec.prior.data(),
      spec.likelihoods.data(),
      kappa_temp,
      delta_floor,
      eta.empty() ? nullptr : eta.data(),
      use_legacy_form,
      output.posterior.data(),
      &output.log_evidence);
}

/*
 * Convenience wrapper for update_posterior_raw using std::vector containers.
 * Validates dimensions, allocates the output posterior, and delegates to the
 * raw-pointer implementation. Used primarily by tests and non-JNI callers.
 */
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
