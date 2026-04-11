/*
 * HoldemBayesNativeCpuBindings.cpp -- CPU JNI binding for sequential Bayesian
 * posterior updating in the sicfun poker analytics system.
 *
 * This file bridges the JVM (sicfun.holdem.HoldemBayesNativeCpuBindings) to the
 * pure C++ engine in BayesNativeUpdateCore.hpp. It handles:
 *   - JNI null checks and array dimension validation.
 *   - Acquiring JNI critical arrays for zero-copy access to JVM heap memory.
 *   - Delegating the actual math to bayesnative::update_posterior_raw().
 *   - Releasing critical arrays with the correct mode (0 = commit writes,
 *     JNI_ABORT = discard writes on error).
 *   - Recording engine code 1 (CPU) on success for runtime dispatch telemetry.
 *
 * Compiled into: sicfun_native_cpu.dll
 *
 * Critical array protocol:
 *   GetPrimitiveArrayCritical gives a direct pointer into the JVM heap (or a
 *   pinned copy). While any critical array is held, the JNI code must not call
 *   back into the JVM, allocate Java objects, or block. Arrays must be released
 *   in reverse acquisition order to avoid GC deadlocks.
 *
 * Error status codes: shared with BayesNativeUpdateCore.hpp (0=ok, 100=null,
 * 101=mismatch, 102=read failure, 124=write failure, 160-163=computation errors).
 */

#include <jni.h>

#include <atomic>

#include "BayesNativeUpdateCore.hpp"

namespace {

constexpr jint kEngineUnknown = 0;  /* No computation has run yet. */
constexpr jint kEngineCpu = 1;      /* Last successful computation used CPU path. */

/* Tracks which engine last completed successfully. Read by lastEngineCode(). */
std::atomic<jint> g_last_engine_code(kEngineUnknown);

/*
 * Checks for a pending JNI exception, clears it if present, and returns true
 * if one was found. This is necessary after every JNI call that can throw
 * (GetArrayLength, GetPrimitiveArrayCritical, ReleasePrimitiveArrayCritical).
 */
bool clear_pending_jni_exception(JNIEnv* env) {
  if (!env->ExceptionCheck()) {
    return false;
  }
  env->ExceptionClear();
  return true;
}

/*
 * Releases a JNI critical array and updates the status if the release itself
 * triggers a JNI exception. The mode parameter controls whether writes are
 * committed back to the JVM heap:
 *   - 0:        commit changes (used for output arrays on success).
 *   - JNI_ABORT: discard changes (used for input arrays, or output on error).
 */
void release_critical_array(
    JNIEnv* env,
    jdoubleArray array,
    jdouble* values,
    const jint mode,
    const int failure_status,
    int& status) {
  if (values == nullptr) {
    return;
  }
  env->ReleasePrimitiveArrayCritical(array, values, mode);
  if (clear_pending_jni_exception(env) && status == bayesnative::kStatusOk) {
    status = failure_status;
  }
}

/*
 * Acquires a JNI critical array pointer for zero-copy access. Returns nullptr
 * (and sets status) if a prior error has already occurred or if the JNI call
 * fails. The caller must release the array via release_critical_array() even
 * if subsequent operations fail — critical arrays pin the GC.
 */
jdouble* acquire_critical_array(
    JNIEnv* env,
    jdoubleArray array,
    const int failure_status,
    int& status) {
  if (status != bayesnative::kStatusOk) {
    return nullptr;
  }
  void* raw = env->GetPrimitiveArrayCritical(array, nullptr);
  if (raw == nullptr || clear_pending_jni_exception(env)) {
    status = failure_status;
    return nullptr;
  }
  return static_cast<jdouble*>(raw);
}

}  // namespace

/*
 * JNI entry point: sicfun.holdem.HoldemBayesNativeCpuBindings.updatePosterior()
 *
 * Performs sequential Bayesian posterior updating on the CPU. The method:
 *   1. Validates null arrays and dimension config.
 *   2. Checks all array lengths match expected sizes (using long long
 *      multiplication to avoid int overflow for large observation x hypothesis products).
 *   3. Acquires critical pointers to prior, likelihood, posterior, and log-evidence arrays.
 *      Arrays are acquired one at a time; if any acquisition fails, all previously
 *      acquired arrays are released with JNI_ABORT before returning.
 *   4. Delegates to bayesnative::update_posterior_raw() for the actual computation.
 *   5. Releases input arrays with JNI_ABORT (read-only, no writes to commit).
 *   6. Releases output arrays with mode 0 (commit) on success, JNI_ABORT on error.
 *   7. Records engine code 1 (CPU) on success.
 *
 * Returns: 0 on success, or a status code from the shared JNI error protocol.
 */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemBayesNativeCpuBindings_updatePosterior(
    JNIEnv* env,
    jclass /*clazz*/,
    jint observationCount,
    jint hypothesisCount,
    jdoubleArray priorArray,
    jdoubleArray likelihoodArray,
    jdoubleArray outPosteriorArray,
    jdoubleArray outLogEvidenceArray) {
  if (priorArray == nullptr || likelihoodArray == nullptr || outPosteriorArray == nullptr ||
      outLogEvidenceArray == nullptr) {
    return bayesnative::kStatusNullArray;
  }

  if (!bayesnative::valid_length_product(static_cast<int>(observationCount),
                                         static_cast<int>(hypothesisCount))) {
    return bayesnative::kStatusInvalidConfig;
  }

  const jsize prior_length = env->GetArrayLength(priorArray);
  const jsize likelihood_length = env->GetArrayLength(likelihoodArray);
  const jsize out_posterior_length = env->GetArrayLength(outPosteriorArray);
  const jsize out_log_evidence_length = env->GetArrayLength(outLogEvidenceArray);

  if (clear_pending_jni_exception(env)) {
    return bayesnative::kStatusReadFailure;
  }

  const jsize expected_prior_length = static_cast<jsize>(hypothesisCount);
  const jsize expected_likelihood_length =
      static_cast<jsize>(static_cast<long long>(observationCount) *
                         static_cast<long long>(hypothesisCount));
  if (prior_length != expected_prior_length || likelihood_length != expected_likelihood_length ||
      out_posterior_length != expected_prior_length || out_log_evidence_length < 1) {
    return bayesnative::kStatusLengthMismatch;
  }

  int status = bayesnative::kStatusOk;
  jdouble* prior_values =
      acquire_critical_array(env, priorArray, bayesnative::kStatusReadFailure, status);
  if (prior_values == nullptr) {
    return status;
  }

  jdouble* likelihood_values =
      acquire_critical_array(env, likelihoodArray, bayesnative::kStatusReadFailure, status);
  if (likelihood_values == nullptr) {
    release_critical_array(env, priorArray, prior_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    return status;
  }

  jdouble* out_posterior_values =
      acquire_critical_array(env, outPosteriorArray, bayesnative::kStatusWriteFailure, status);
  if (out_posterior_values == nullptr) {
    release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    release_critical_array(env, priorArray, prior_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    return status;
  }

  jdouble* out_log_evidence_values =
      acquire_critical_array(env, outLogEvidenceArray, bayesnative::kStatusWriteFailure, status);
  if (out_log_evidence_values == nullptr) {
    release_critical_array(env, outPosteriorArray, out_posterior_values, JNI_ABORT,
                           bayesnative::kStatusWriteFailure, status);
    release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    release_critical_array(env, priorArray, prior_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    return status;
  }

  status = bayesnative::update_posterior_raw(
      static_cast<int>(observationCount), static_cast<int>(hypothesisCount), prior_values,
      likelihood_values, out_posterior_values, out_log_evidence_values);

  release_critical_array(env, priorArray, prior_values, JNI_ABORT, bayesnative::kStatusReadFailure,
                         status);
  release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                         bayesnative::kStatusReadFailure, status);

  const jint out_release_mode = (status == bayesnative::kStatusOk) ? 0 : JNI_ABORT;
  release_critical_array(env, outPosteriorArray, out_posterior_values, out_release_mode,
                         bayesnative::kStatusWriteFailure, status);
  release_critical_array(env, outLogEvidenceArray, out_log_evidence_values, out_release_mode,
                         bayesnative::kStatusWriteFailure, status);

  if (status != bayesnative::kStatusOk) {
    return status;
  }

  g_last_engine_code.store(kEngineCpu, std::memory_order_relaxed);
  return bayesnative::kStatusOk;
}

/*
 * JNI entry point: sicfun.holdem.HoldemBayesNativeCpuBindings.updatePosteriorTempered()
 *
 * Two-layer tempered Bayesian update (SICFUN v0.30.2 Def 15A/15B).
 * Same critical-array protocol as updatePosterior, plus three additional
 * parameters: kappaTemp (double), deltaFloor (double), eta (double[]),
 * and useLegacyForm (boolean).
 *
 * The eta array may be null, in which case uniform 1/hypothesisCount is used.
 */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemBayesNativeCpuBindings_updatePosteriorTempered(
    JNIEnv* env,
    jclass /*clazz*/,
    jint observationCount,
    jint hypothesisCount,
    jdoubleArray priorArray,
    jdoubleArray likelihoodArray,
    jdouble kappaTemp,
    jdouble deltaFloor,
    jdoubleArray etaArray,
    jboolean useLegacyForm,
    jdoubleArray outPosteriorArray,
    jdoubleArray outLogEvidenceArray) {
  if (priorArray == nullptr || likelihoodArray == nullptr || outPosteriorArray == nullptr ||
      outLogEvidenceArray == nullptr) {
    return bayesnative::kStatusNullArray;
  }

  if (!bayesnative::valid_length_product(static_cast<int>(observationCount),
                                         static_cast<int>(hypothesisCount))) {
    return bayesnative::kStatusInvalidConfig;
  }

  const jsize prior_length = env->GetArrayLength(priorArray);
  const jsize likelihood_length = env->GetArrayLength(likelihoodArray);
  const jsize out_posterior_length = env->GetArrayLength(outPosteriorArray);
  const jsize out_log_evidence_length = env->GetArrayLength(outLogEvidenceArray);

  if (clear_pending_jni_exception(env)) {
    return bayesnative::kStatusReadFailure;
  }

  const jsize expected_prior_length = static_cast<jsize>(hypothesisCount);
  const jsize expected_likelihood_length =
      static_cast<jsize>(static_cast<long long>(observationCount) *
                         static_cast<long long>(hypothesisCount));
  if (prior_length != expected_prior_length || likelihood_length != expected_likelihood_length ||
      out_posterior_length != expected_prior_length || out_log_evidence_length < 1) {
    return bayesnative::kStatusLengthMismatch;
  }

  /* Validate eta length if provided. */
  jsize eta_length = 0;
  if (etaArray != nullptr) {
    eta_length = env->GetArrayLength(etaArray);
    if (clear_pending_jni_exception(env)) {
      return bayesnative::kStatusReadFailure;
    }
    if (eta_length != expected_prior_length) {
      return bayesnative::kStatusLengthMismatch;
    }
  }

  int status = bayesnative::kStatusOk;
  jdouble* prior_values =
      acquire_critical_array(env, priorArray, bayesnative::kStatusReadFailure, status);
  if (prior_values == nullptr) {
    return status;
  }

  jdouble* likelihood_values =
      acquire_critical_array(env, likelihoodArray, bayesnative::kStatusReadFailure, status);
  if (likelihood_values == nullptr) {
    release_critical_array(env, priorArray, prior_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    return status;
  }

  /* Acquire eta critical array only if provided. */
  jdouble* eta_values = nullptr;
  if (etaArray != nullptr) {
    eta_values =
        acquire_critical_array(env, etaArray, bayesnative::kStatusReadFailure, status);
    if (eta_values == nullptr) {
      release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                             bayesnative::kStatusReadFailure, status);
      release_critical_array(env, priorArray, prior_values, JNI_ABORT,
                             bayesnative::kStatusReadFailure, status);
      return status;
    }
  }

  jdouble* out_posterior_values =
      acquire_critical_array(env, outPosteriorArray, bayesnative::kStatusWriteFailure, status);
  if (out_posterior_values == nullptr) {
    if (eta_values != nullptr) {
      release_critical_array(env, etaArray, eta_values, JNI_ABORT,
                             bayesnative::kStatusReadFailure, status);
    }
    release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    release_critical_array(env, priorArray, prior_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    return status;
  }

  jdouble* out_log_evidence_values =
      acquire_critical_array(env, outLogEvidenceArray, bayesnative::kStatusWriteFailure, status);
  if (out_log_evidence_values == nullptr) {
    release_critical_array(env, outPosteriorArray, out_posterior_values, JNI_ABORT,
                           bayesnative::kStatusWriteFailure, status);
    if (eta_values != nullptr) {
      release_critical_array(env, etaArray, eta_values, JNI_ABORT,
                             bayesnative::kStatusReadFailure, status);
    }
    release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    release_critical_array(env, priorArray, prior_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
    return status;
  }

  status = bayesnative::update_posterior_tempered_raw(
      static_cast<int>(observationCount), static_cast<int>(hypothesisCount), prior_values,
      likelihood_values, static_cast<double>(kappaTemp), static_cast<double>(deltaFloor),
      eta_values, static_cast<bool>(useLegacyForm),
      out_posterior_values, out_log_evidence_values);

  /* Release input arrays (JNI_ABORT = read-only, no writes to commit). */
  release_critical_array(env, priorArray, prior_values, JNI_ABORT, bayesnative::kStatusReadFailure,
                         status);
  release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                         bayesnative::kStatusReadFailure, status);
  if (eta_values != nullptr) {
    release_critical_array(env, etaArray, eta_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
  }

  /* Release output arrays (commit on success, discard on error). */
  const jint out_release_mode = (status == bayesnative::kStatusOk) ? 0 : JNI_ABORT;
  release_critical_array(env, outPosteriorArray, out_posterior_values, out_release_mode,
                         bayesnative::kStatusWriteFailure, status);
  release_critical_array(env, outLogEvidenceArray, out_log_evidence_values, out_release_mode,
                         bayesnative::kStatusWriteFailure, status);

  if (status != bayesnative::kStatusOk) {
    return status;
  }

  g_last_engine_code.store(kEngineCpu, std::memory_order_relaxed);
  return bayesnative::kStatusOk;
}

/*
 * JNI entry point: sicfun.holdem.HoldemBayesNativeCpuBindings.lastEngineCode()
 *
 * Returns the engine code from the last successful updatePosterior() call.
 * 0 = no computation yet, 1 = CPU. Used by the JVM-side hybrid dispatcher
 * to verify which native path actually executed.
 */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemBayesNativeCpuBindings_lastEngineCode(
    JNIEnv* /*env*/, jclass /*clazz*/) {
  return g_last_engine_code.load(std::memory_order_relaxed);
}
