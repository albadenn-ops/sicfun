#include <jni.h>

#include <atomic>

#include "BayesNativeUpdateCore.hpp"

namespace {

constexpr jint kEngineUnknown = 0;
constexpr jint kEngineCpu = 1;

std::atomic<jint> g_last_engine_code(kEngineUnknown);

bool clear_pending_jni_exception(JNIEnv* env) {
  if (!env->ExceptionCheck()) {
    return false;
  }
  env->ExceptionClear();
  return true;
}

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

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemBayesNativeCpuBindings_lastEngineCode(
    JNIEnv* /*env*/, jclass /*clazz*/) {
  return g_last_engine_code.load(std::memory_order_relaxed);
}
