/*
 * DdreNativeInferenceCore.hpp -- DDRE (Data-Driven Range Estimation) posterior
 * inference engine for the sicfun poker analytics system.
 *
 * DDRE is a lightweight alternative to full Bayesian updating that computes
 * posterior distributions using a geometric-mean scoring rule in log-space.
 * Unlike the Bayesian engine (BayesNativeUpdateCore.hpp) which applies
 * observations sequentially, DDRE computes a single score per hypothesis:
 *
 *   score(h) = sqrt(prior(h)) * geometric_mean(likelihoods for h)
 *
 * This avoids numerical underflow for many observations and is computationally
 * cheaper (no sequential normalization passes), at the cost of not being a
 * true Bayesian posterior. The geometric mean treats all observations equally
 * regardless of order.
 *
 * Design decisions:
 *   - Header-only for sharing between CPU and GPU JNI bindings.
 *   - Log-space accumulation: replaces N multiplications + pow() with
 *     (N+1) log() + 1 exp(), which is ~3-5x cheaper per hypothesis.
 *   - Minimum likelihood floor (1e-6) prevents log(0) and ensures no
 *     hypothesis is completely zeroed out by a single observation.
 *   - Zero-observation case returns the normalized prior unchanged.
 *   - Status codes match the shared JNI error protocol.
 *
 * JNI classes: sicfun.holdem.HoldemDdreNativeCpuBindings (CPU path)
 *              sicfun.holdem.HoldemDdreNativeGpuBindings (GPU path)
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace ddrenative {

/* Compiler-portable restrict qualifier for aliasing optimizations. */
#if defined(_MSC_VER)
#define DDRENATIVE_RESTRICT __restrict
#else
#define DDRENATIVE_RESTRICT __restrict__
#endif

/* Status codes — shared convention across all sicfun native JNI bindings. */
constexpr int kStatusOk = 0;                 /* Success. */
constexpr int kStatusNullArray = 100;         /* A required JNI array argument was null. */
constexpr int kStatusLengthMismatch = 101;    /* Array dimensions do not match expected sizes. */
constexpr int kStatusReadFailure = 102;       /* JNI array read operation failed. */
constexpr int kStatusInvalidConfig = 160;     /* Observation/hypothesis counts out of range. */
constexpr int kStatusInvalidPrior = 161;      /* Prior contains NaN, Inf, or negative values. */
constexpr int kStatusInvalidLikelihood = 162; /* Likelihood contains NaN, Inf, or negative values. */
constexpr int kStatusZeroMass = 163;          /* Total posterior mass collapsed to zero or non-finite. */
constexpr int kStatusWriteFailure = 124;      /* JNI array write operation failed. */

/* Input specification for DDRE inference. Used by the convenience wrapper. */
struct InferenceSpec {
  int observation_count = 0;     /* Number of observed actions (may be 0 for prior-only output). */
  int hypothesis_count = 0;      /* Number of hypotheses (e.g., hand range buckets). */
  std::vector<double> prior;     /* Prior distribution, length = hypothesis_count. */
  std::vector<double> likelihoods;  /* Row-major likelihood matrix [observation][hypothesis].
                                       Length = observation_count * hypothesis_count. */
};

/* Output from DDRE inference. */
struct InferenceOutput {
  std::vector<double> posterior;  /* Normalized posterior distribution, length = hypothesis_count. */
};

/* Validates that observation_count >= 0, hypothesis_count > 0, and their
 * product does not overflow int. Allows zero observations (prior-only mode). */
inline bool valid_lengths(const int observation_count, const int hypothesis_count) {
  if (observation_count < 0 || hypothesis_count <= 0) {
    return false;
  }
  const int64_t obs = static_cast<int64_t>(observation_count);
  const int64_t hyp = static_cast<int64_t>(hypothesis_count);
  const int64_t product = obs * hyp;
  return product >= 0 && product <= static_cast<int64_t>(std::numeric_limits<int>::max());
}

/*
 * Core DDRE posterior inference on raw pointers.
 *
 * Parameters:
 *   observation_count  -- Number of observations (0 = prior-only output).
 *   hypothesis_count   -- Number of hypotheses.
 *   prior              -- Prior distribution, length = hypothesis_count.
 *   likelihoods        -- Row-major likelihood matrix [obs][hyp].
 *   posterior          -- Output buffer, length = hypothesis_count.
 *
 * Returns: kStatusOk on success, or an error status code.
 *
 * Scoring formula per hypothesis h:
 *   log_score(h) = 0.5 * log(prior(h))
 *                + (1/N) * sum_obs( log(likelihood(obs, h)) )
 *   score(h) = exp(log_score(h))
 *   posterior(h) = score(h) / sum(scores)
 *
 * The 0.5 exponent on the prior acts as a square-root dampening, preventing
 * the prior from dominating when observation counts are low. The (1/N)
 * averaging of log-likelihoods computes the geometric mean, which is robust
 * to the number of observations.
 */
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

  /* Floor value to prevent log(0). Any prior or likelihood below this threshold
   * is clamped up, ensuring every hypothesis retains some minimum probability. */
  constexpr double kMinLikelihood = 1e-6;

  /* Zero observations: posterior equals the (normalized) prior — no evidence, no update. */
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

  /* Phase 1: Validate prior distribution. */
  for (int h = 0; h < hypothesis_count; ++h) {
    if (!std::isfinite(prior[h]) || prior[h] < 0.0) {
      return kStatusInvalidPrior;
    }
  }

  /* Phase 2: Accumulate log-likelihoods with cache-friendly (row-major, stride-1)
   * access. The original per-hypothesis loop strided by hypothesis_count per
   * observation, causing ~hypothesis_count/8 cache-line misses per iteration.
   * This reorder accesses likelihoods sequentially within each observation row.
   * Uses the posterior buffer as scratch: posterior[h] = sum_obs log(lik[obs][h]). */
  {
    int h = 0;
    for (; h + 3 < hypothesis_count; h += 4) {
      posterior[h] = 0.0; posterior[h+1] = 0.0;
      posterior[h+2] = 0.0; posterior[h+3] = 0.0;
    }
    for (; h < hypothesis_count; ++h) posterior[h] = 0.0;
  }

  for (int obs = 0; obs < observation_count; ++obs) {
    const double* DDRENATIVE_RESTRICT row =
        likelihoods + static_cast<long long>(obs) * static_cast<long long>(hypothesis_count);
    int h = 0;
    for (; h + 3 < hypothesis_count; h += 4) {
      const double l0 = row[h], l1 = row[h+1], l2 = row[h+2], l3 = row[h+3];
      if (!std::isfinite(l0) || l0 < 0.0 || !std::isfinite(l1) || l1 < 0.0 ||
          !std::isfinite(l2) || l2 < 0.0 || !std::isfinite(l3) || l3 < 0.0) {
        return kStatusInvalidLikelihood;
      }
      posterior[h]   += std::log(std::max(l0, kMinLikelihood));
      posterior[h+1] += std::log(std::max(l1, kMinLikelihood));
      posterior[h+2] += std::log(std::max(l2, kMinLikelihood));
      posterior[h+3] += std::log(std::max(l3, kMinLikelihood));
    }
    for (; h < hypothesis_count; ++h) {
      const double lik = row[h];
      if (!std::isfinite(lik) || lik < 0.0) {
        return kStatusInvalidLikelihood;
      }
      posterior[h] += std::log(std::max(lik, kMinLikelihood));
    }
  }

  /* Phase 3: Combine 0.5*log(prior) + (1/N)*sum(log_lik), exponentiate, accumulate.
   * Applying the 1/N scale factor once at the end instead of per-observation
   * is algebraically equivalent and avoids N-1 extra multiplications. */
  double total = 0.0;
  {
    int h = 0;
    for (; h + 3 < hypothesis_count; h += 4) {
      const double s0 = std::exp(0.5 * std::log(std::max(prior[h],   kMinLikelihood)) + inverse_observation_count * posterior[h]);
      const double s1 = std::exp(0.5 * std::log(std::max(prior[h+1], kMinLikelihood)) + inverse_observation_count * posterior[h+1]);
      const double s2 = std::exp(0.5 * std::log(std::max(prior[h+2], kMinLikelihood)) + inverse_observation_count * posterior[h+2]);
      const double s3 = std::exp(0.5 * std::log(std::max(prior[h+3], kMinLikelihood)) + inverse_observation_count * posterior[h+3]);
      if (!std::isfinite(s0) || !std::isfinite(s1) || !std::isfinite(s2) || !std::isfinite(s3)) {
        return kStatusInvalidLikelihood;
      }
      posterior[h]   = std::max(s0, kMinLikelihood);
      posterior[h+1] = std::max(s1, kMinLikelihood);
      posterior[h+2] = std::max(s2, kMinLikelihood);
      posterior[h+3] = std::max(s3, kMinLikelihood);
      total += (posterior[h] + posterior[h+1]) + (posterior[h+2] + posterior[h+3]);
    }
    for (; h < hypothesis_count; ++h) {
      const double score = std::exp(
          0.5 * std::log(std::max(prior[h], kMinLikelihood))
          + inverse_observation_count * posterior[h]);
      if (!std::isfinite(score) || score < 0.0) {
        return kStatusInvalidLikelihood;
      }
      posterior[h] = std::max(score, kMinLikelihood);
      total += posterior[h];
    }
  }

  if (!std::isfinite(total) || !(total > 0.0)) {
    return kStatusZeroMass;
  }

  /* Phase 4: Normalize to probability distribution. */
  const double inv_total = 1.0 / total;
  {
    int h = 0;
    for (; h + 3 < hypothesis_count; h += 4) {
      posterior[h]   *= inv_total;
      posterior[h+1] *= inv_total;
      posterior[h+2] *= inv_total;
      posterior[h+3] *= inv_total;
    }
    for (; h < hypothesis_count; ++h) {
      posterior[h] *= inv_total;
    }
  }
  return kStatusOk;
}

/*
 * Convenience wrapper for infer_posterior_raw using std::vector containers.
 * Validates that the prior and likelihood vectors match the expected dimensions,
 * allocates the output posterior, and delegates to the raw-pointer implementation.
 * Used primarily by tests and non-JNI callers (the JNI path uses raw pointers
 * from GetPrimitiveArrayCritical for zero-copy access).
 */
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
