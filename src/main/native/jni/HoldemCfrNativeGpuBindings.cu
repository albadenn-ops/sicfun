#include <jni.h>

#include <atomic>
#include <vector>

#include "CfrNativeSolverCore.hpp"

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

int read_int_array(JNIEnv* env, jintArray array, std::vector<int>& out) {
  if (array == nullptr) {
    return cfrnative::kStatusNullArray;
  }
  const jsize length = env->GetArrayLength(array);
  out.resize(static_cast<size_t>(length));
  if (length > 0) {
    env->GetIntArrayRegion(array, 0, length, reinterpret_cast<jint*>(out.data()));
    if (clear_pending_jni_exception(env)) {
      return cfrnative::kStatusReadFailure;
    }
  }
  return cfrnative::kStatusOk;
}

int read_double_array(JNIEnv* env, jdoubleArray array, std::vector<double>& out) {
  if (array == nullptr) {
    return cfrnative::kStatusNullArray;
  }
  const jsize length = env->GetArrayLength(array);
  out.resize(static_cast<size_t>(length));
  if (length > 0) {
    env->GetDoubleArrayRegion(array, 0, length, reinterpret_cast<jdouble*>(out.data()));
    if (clear_pending_jni_exception(env)) {
      return cfrnative::kStatusReadFailure;
    }
  }
  return cfrnative::kStatusOk;
}

int write_double_array(JNIEnv* env, jdoubleArray array, const std::vector<double>& values) {
  if (array == nullptr) {
    return cfrnative::kStatusNullArray;
  }
  const jsize length = env->GetArrayLength(array);
  if (length != static_cast<jsize>(values.size())) {
    return cfrnative::kStatusLengthMismatch;
  }
  if (length > 0) {
    env->SetDoubleArrayRegion(array, 0, length, reinterpret_cast<const jdouble*>(values.data()));
    if (clear_pending_jni_exception(env)) {
      return cfrnative::kStatusWriteFailure;
    }
  }
  return cfrnative::kStatusOk;
}

int write_single_double(JNIEnv* env, jdoubleArray array, const double value) {
  if (array == nullptr) {
    return cfrnative::kStatusNullArray;
  }
  const jsize length = env->GetArrayLength(array);
  if (length < 1) {
    return cfrnative::kStatusLengthMismatch;
  }
  const jdouble boxed = static_cast<jdouble>(value);
  env->SetDoubleArrayRegion(array, 0, 1, &boxed);
  if (clear_pending_jni_exception(env)) {
    return cfrnative::kStatusWriteFailure;
  }
  return cfrnative::kStatusOk;
}

}  // namespace

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemCfrNativeGpuBindings_solveTree(
    JNIEnv* env,
    jclass /*clazz*/,
    jint iterations,
    jint averagingDelay,
    jboolean cfrPlus,
    jboolean linearAveraging,
    jint rootNodeId,
    jintArray nodeTypesArray,
    jintArray nodeStartsArray,
    jintArray nodeCountsArray,
    jintArray nodeInfosetsArray,
    jintArray edgeChildIdsArray,
    jdoubleArray edgeProbabilitiesArray,
    jdoubleArray terminalUtilitiesArray,
    jintArray infosetPlayersArray,
    jintArray infosetActionCountsArray,
    jdoubleArray outAverageStrategiesArray,
    jdoubleArray outExpectedValueArray) {
  cfrnative::TreeSpec spec;
  spec.iterations = static_cast<int>(iterations);
  spec.averaging_delay = static_cast<int>(averagingDelay);
  spec.cfr_plus = (cfrPlus == JNI_TRUE);
  spec.linear_averaging = (linearAveraging == JNI_TRUE);
  spec.root_node_id = static_cast<int>(rootNodeId);

  int status = read_int_array(env, nodeTypesArray, spec.node_types);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, nodeStartsArray, spec.node_starts);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, nodeCountsArray, spec.node_counts);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, nodeInfosetsArray, spec.node_infosets);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, edgeChildIdsArray, spec.edge_child_ids);
  if (status != cfrnative::kStatusOk) return status;
  status = read_double_array(env, edgeProbabilitiesArray, spec.edge_probabilities);
  if (status != cfrnative::kStatusOk) return status;
  status = read_double_array(env, terminalUtilitiesArray, spec.terminal_utilities);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, infosetPlayersArray, spec.infoset_players);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, infosetActionCountsArray, spec.infoset_action_counts);
  if (status != cfrnative::kStatusOk) return status;

  cfrnative::SolveOutput output;
  status = cfrnative::solve(spec, output);
  if (status != cfrnative::kStatusOk) {
    return status;
  }

  status = write_double_array(env, outAverageStrategiesArray, output.average_strategies);
  if (status != cfrnative::kStatusOk) return status;
  status = write_single_double(env, outExpectedValueArray, output.expected_value_player0);
  if (status != cfrnative::kStatusOk) return status;

  g_last_engine_code.store(kEngineGpu, std::memory_order_relaxed);
  return cfrnative::kStatusOk;
}

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemCfrNativeGpuBindings_lastEngineCode(JNIEnv* /*env*/, jclass /*clazz*/) {
  return g_last_engine_code.load(std::memory_order_relaxed);
}
