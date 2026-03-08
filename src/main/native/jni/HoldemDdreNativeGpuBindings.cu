#include <jni.h>

#include <atomic>

#include "DdreNativeInferenceCore.hpp"

namespace {

constexpr jint kEngineUnknown = 0;
constexpr jint kEngineGpu = 2;

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
  if (clear_pending_jni_exception(env) && status == ddrenative::kStatusOk) {
    status = failure_status;
  }
}

jdouble* acquire_critical_array(
    JNIEnv* env,
    jdoubleArray array,
    const int failure_status,
    int& status) {
  if (status != ddrenative::kStatusOk) {
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
Java_sicfun_holdem_HoldemDdreNativeGpuBindings_inferPosterior(
    JNIEnv* env,
    jclass /*clazz*/,
    jint observationCount,
    jint hypothesisCount,
    jdoubleArray priorArray,
    jdoubleArray likelihoodArray,
    jdoubleArray outPosteriorArray) {
  if (priorArray == nullptr || likelihoodArray == nullptr || outPosteriorArray == nullptr) {
    return ddrenative::kStatusNullArray;
  }

  if (!ddrenative::valid_lengths(static_cast<int>(observationCount),
                                 static_cast<int>(hypothesisCount))) {
    return ddrenative::kStatusInvalidConfig;
  }

  const jsize prior_length = env->GetArrayLength(priorArray);
  const jsize likelihood_length = env->GetArrayLength(likelihoodArray);
  const jsize out_posterior_length = env->GetArrayLength(outPosteriorArray);

  if (clear_pending_jni_exception(env)) {
    return ddrenative::kStatusReadFailure;
  }

  const jsize expected_prior_length = static_cast<jsize>(hypothesisCount);
  const jsize expected_likelihood_length =
      static_cast<jsize>(static_cast<long long>(observationCount) *
                         static_cast<long long>(hypothesisCount));
  if (prior_length != expected_prior_length ||
      likelihood_length != expected_likelihood_length ||
      out_posterior_length != expected_prior_length) {
    return ddrenative::kStatusLengthMismatch;
  }

  int status = ddrenative::kStatusOk;
  jdouble* prior_values =
      acquire_critical_array(env, priorArray, ddrenative::kStatusReadFailure, status);
  if (prior_values == nullptr) {
    return status;
  }

  jdouble* likelihood_values =
      acquire_critical_array(env, likelihoodArray, ddrenative::kStatusReadFailure, status);
  if (likelihood_values == nullptr) {
    release_critical_array(env, priorArray, prior_values, JNI_ABORT,
                           ddrenative::kStatusReadFailure, status);
    return status;
  }

  jdouble* out_posterior_values =
      acquire_critical_array(env, outPosteriorArray, ddrenative::kStatusWriteFailure, status);
  if (out_posterior_values == nullptr) {
    release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                           ddrenative::kStatusReadFailure, status);
    release_critical_array(env, priorArray, prior_values, JNI_ABORT,
                           ddrenative::kStatusReadFailure, status);
    return status;
  }

  status = ddrenative::infer_posterior_raw(
      static_cast<int>(observationCount),
      static_cast<int>(hypothesisCount),
      prior_values,
      likelihood_values,
      out_posterior_values);

  release_critical_array(env, priorArray, prior_values, JNI_ABORT, ddrenative::kStatusReadFailure,
                         status);
  release_critical_array(env, likelihoodArray, likelihood_values, JNI_ABORT,
                         ddrenative::kStatusReadFailure, status);

  const jint out_release_mode = (status == ddrenative::kStatusOk) ? 0 : JNI_ABORT;
  release_critical_array(env, outPosteriorArray, out_posterior_values, out_release_mode,
                         ddrenative::kStatusWriteFailure, status);

  if (status != ddrenative::kStatusOk) {
    return status;
  }

  g_last_engine_code.store(kEngineGpu, std::memory_order_relaxed);
  return ddrenative::kStatusOk;
}

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemDdreNativeGpuBindings_lastEngineCode(
    JNIEnv* /*env*/, jclass /*clazz*/) {
  return g_last_engine_code.load(std::memory_order_relaxed);
}
