/*
 * HoldemBayesNativeGpuBindings.cu -- GPU JNI binding for sequential Bayesian
 * posterior updating in the sicfun poker analytics system.
 *
 * Structurally identical to HoldemBayesNativeCpuBindings.cpp — same critical
 * array acquire/release pattern, same delegation to BayesNativeUpdateCore.hpp.
 * The only difference is the engine code reported on success: 2 (GPU) instead
 * of 1 (CPU). The .cu extension ensures this file is compiled by nvcc and
 * linked into sicfun_gpu_kernel.dll, allowing the JVM-side hybrid dispatcher
 * to select the GPU DLL at runtime.
 *
 * Note: The actual Bayesian update math runs on the CPU (via the shared
 * header-only engine). The "GPU" designation refers to the DLL context — this
 * binding lives in the GPU DLL so the dispatcher can probe GPU availability
 * by loading sicfun_gpu_kernel.dll and calling this entry point.
 *
 * Compiled into: sicfun_gpu_kernel.dll (requires CUDA toolkit for nvcc)
 */

#include <jni.h>

#include <atomic>

#include "BayesNativeUpdateCore.hpp"

namespace {

constexpr jint kEngineUnknown = 0;  /* No computation has run yet. */
constexpr jint kEngineGpu = 2;      /* Last successful computation used GPU-context path. */

std::atomic<jint> g_last_engine_code(kEngineUnknown);

/* See HoldemBayesNativeCpuBindings.cpp for detailed comments on exception handling. */
bool clear_pending_jni_exception(JNIEnv* env) {
  if (!env->ExceptionCheck()) {
    return false;
  }
  env->ExceptionClear();
  return true;
}

/* Releases a critical array; see CPU binding for detailed comments on mode semantics. */
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

/* Acquires a critical array pointer; see CPU binding for detailed comments. */
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
 * JNI entry point: sicfun.holdem.HoldemBayesNativeGpuBindings.updatePosterior()
 *
 * Identical logic to HoldemBayesNativeCpuBindings::updatePosterior — validates
 * arrays, acquires critical pointers, delegates to update_posterior_raw(), and
 * releases. Reports engine code 2 (GPU) on success.
 */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemBayesNativeGpuBindings_updatePosterior(
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

  g_last_engine_code.store(kEngineGpu, std::memory_order_relaxed);
  return bayesnative::kStatusOk;
}

/*
 * JNI entry point: sicfun.holdem.HoldemBayesNativeGpuBindings.updatePosteriorTempered()
 *
 * GPU-context variant of the two-layer tempered Bayesian update.
 * Identical logic to the CPU variant. Reports engine code 2 (GPU).
 */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemBayesNativeGpuBindings_updatePosteriorTempered(
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

  release_critical_array(env, priorArray, prior_values, JNI_ABORT, bayesnative::kStatusReadFailure,
                         status);
  release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                         bayesnative::kStatusReadFailure, status);
  if (eta_values != nullptr) {
    release_critical_array(env, etaArray, eta_values, JNI_ABORT,
                           bayesnative::kStatusReadFailure, status);
  }

  const jint out_release_mode = (status == bayesnative::kStatusOk) ? 0 : JNI_ABORT;
  release_critical_array(env, outPosteriorArray, out_posterior_values, out_release_mode,
                         bayesnative::kStatusWriteFailure, status);
  release_critical_array(env, outLogEvidenceArray, out_log_evidence_values, out_release_mode,
                         bayesnative::kStatusWriteFailure, status);

  if (status != bayesnative::kStatusOk) {
    return status;
  }

  g_last_engine_code.store(kEngineGpu, std::memory_order_relaxed);
  return bayesnative::kStatusOk;
}

/* Returns 0 (unknown) or 2 (GPU) depending on whether updatePosterior succeeded. */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemBayesNativeGpuBindings_lastEngineCode(
    JNIEnv* /*env*/, jclass /*clazz*/) {
  return g_last_engine_code.load(std::memory_order_relaxed);
}
