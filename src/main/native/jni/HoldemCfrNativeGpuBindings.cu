#include <jni.h>

#include <atomic>
#include <vector>

#include "CfrNativeSolverCore.hpp"
#include "CfrBatchCudaKernel.cuh"

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

int write_int_array(JNIEnv* env, jintArray array, const std::vector<int>& values) {
  if (array == nullptr) {
    return cfrnative::kStatusNullArray;
  }
  const jsize length = env->GetArrayLength(array);
  if (length != static_cast<jsize>(values.size())) {
    return cfrnative::kStatusLengthMismatch;
  }
  if (length > 0) {
    env->SetIntArrayRegion(array, 0, length, reinterpret_cast<const jint*>(values.data()));
    if (clear_pending_jni_exception(env)) {
      return cfrnative::kStatusWriteFailure;
    }
  }
  return cfrnative::kStatusOk;
}

int write_single_int(JNIEnv* env, jintArray array, const int value) {
  if (array == nullptr) {
    return cfrnative::kStatusNullArray;
  }
  const jsize length = env->GetArrayLength(array);
  if (length < 1) {
    return cfrnative::kStatusLengthMismatch;
  }
  const jint boxed = static_cast<jint>(value);
  env->SetIntArrayRegion(array, 0, 1, &boxed);
  if (clear_pending_jni_exception(env)) {
    return cfrnative::kStatusWriteFailure;
  }
  return cfrnative::kStatusOk;
}

int read_float_array(JNIEnv* env, jfloatArray array, std::vector<float>& out) {
  if (array == nullptr) return cfrnative::kStatusNullArray;
  const jsize length = env->GetArrayLength(array);
  out.resize(static_cast<size_t>(length));
  if (length > 0) {
    env->GetFloatArrayRegion(array, 0, length, reinterpret_cast<jfloat*>(out.data()));
    if (clear_pending_jni_exception(env)) return cfrnative::kStatusReadFailure;
  }
  return cfrnative::kStatusOk;
}

int write_float_array(JNIEnv* env, jfloatArray array, const std::vector<float>& values) {
  if (array == nullptr) return cfrnative::kStatusNullArray;
  const jsize length = env->GetArrayLength(array);
  if (length != static_cast<jsize>(values.size())) return cfrnative::kStatusLengthMismatch;
  if (length > 0) {
    env->SetFloatArrayRegion(array, 0, length, reinterpret_cast<const jfloat*>(values.data()));
    if (clear_pending_jni_exception(env)) return cfrnative::kStatusWriteFailure;
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
Java_sicfun_holdem_HoldemCfrNativeGpuBindings_solveTreeFixed(
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
    jintArray edgeProbabilitiesRawArray,
    jintArray terminalUtilitiesRawArray,
    jintArray infosetPlayersArray,
    jintArray infosetActionCountsArray,
    jintArray outAverageStrategiesRawArray,
    jintArray outExpectedValueRawArray) {
  cfrnative::TreeSpecFixed spec;
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
  status = read_int_array(env, edgeProbabilitiesRawArray, spec.edge_probabilities_raw);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, terminalUtilitiesRawArray, spec.terminal_utilities_raw);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, infosetPlayersArray, spec.infoset_players);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, infosetActionCountsArray, spec.infoset_action_counts);
  if (status != cfrnative::kStatusOk) return status;

  cfrnative::SolveOutputFixed output;
  status = cfrnative::solve_fixed(spec, output);
  if (status != cfrnative::kStatusOk) {
    return status;
  }

  status = write_int_array(env, outAverageStrategiesRawArray, output.average_strategies_raw);
  if (status != cfrnative::kStatusOk) return status;
  status = write_single_int(env, outExpectedValueRawArray, output.expected_value_player0_raw);
  if (status != cfrnative::kStatusOk) return status;

  g_last_engine_code.store(kEngineGpu, std::memory_order_relaxed);
  return cfrnative::kStatusOk;
}

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemCfrNativeGpuBindings_solveTreeBatch(
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
    jintArray infosetPlayersArray,
    jintArray infosetActionCountsArray,
    jintArray infosetOffsetsArray,
    jfloatArray terminalUtilitiesArray,
    jfloatArray chanceWeightsArray,
    jfloatArray outAverageStrategiesArray,
    jfloatArray outExpectedValuesArray,
    jint batchSize) {

  // Read shared topology
  std::vector<int> node_types, node_starts, node_counts, node_infosets;
  std::vector<int> edge_child_ids, infoset_players, infoset_action_counts, infoset_offsets;

  int status;
  status = read_int_array(env, nodeTypesArray, node_types);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, nodeStartsArray, node_starts);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, nodeCountsArray, node_counts);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, nodeInfosetsArray, node_infosets);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, edgeChildIdsArray, edge_child_ids);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, infosetPlayersArray, infoset_players);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, infosetActionCountsArray, infoset_action_counts);
  if (status != cfrnative::kStatusOk) return status;
  status = read_int_array(env, infosetOffsetsArray, infoset_offsets);
  if (status != cfrnative::kStatusOk) return status;

  const int N = static_cast<int>(node_types.size());
  const int E = static_cast<int>(edge_child_ids.size());
  const int I = static_cast<int>(infoset_action_counts.size());
  const int S = infoset_offsets.back(); // strategy_size = last offset
  const int B = static_cast<int>(batchSize);

  // Read per-tree float arrays
  std::vector<float> terminal_utilities, chance_weights;
  status = read_float_array(env, terminalUtilitiesArray, terminal_utilities);
  if (status != cfrnative::kStatusOk) return status;
  status = read_float_array(env, chanceWeightsArray, chance_weights);
  if (status != cfrnative::kStatusOk) return status;

  if (static_cast<int>(terminal_utilities.size()) != B * N) return cfrnative::kStatusLengthMismatch;
  if (static_cast<int>(chance_weights.size()) != B * E) return cfrnative::kStatusLengthMismatch;

  // Output buffers
  std::vector<float> out_strategies(B * S);
  std::vector<float> out_ev(B);

  cfrbatch::BatchSpec spec;
  spec.root_node_id = static_cast<int>(rootNodeId);
  spec.node_count = N;
  spec.edge_count = E;
  spec.infoset_count = I;
  spec.strategy_size = S;
  spec.iterations = static_cast<int>(iterations);
  spec.averaging_delay = static_cast<int>(averagingDelay);
  spec.cfr_plus = (cfrPlus == JNI_TRUE);
  spec.linear_averaging = (linearAveraging == JNI_TRUE);
  spec.batch_size = B;

  int result = cfrbatch::launch_batch_solve(
      spec,
      node_types.data(), node_starts.data(), node_counts.data(),
      node_infosets.data(), edge_child_ids.data(),
      infoset_action_counts.data(), infoset_offsets.data(),
      terminal_utilities.data(), chance_weights.data(),
      out_strategies.data(), out_ev.data());

  if (result != 0) return -1; // CUDA error

  // Write back
  status = write_float_array(env, outAverageStrategiesArray, out_strategies);
  if (status != cfrnative::kStatusOk) return status;
  status = write_float_array(env, outExpectedValuesArray, out_ev);
  if (status != cfrnative::kStatusOk) return status;

  g_last_engine_code.store(kEngineGpu, std::memory_order_relaxed);
  return cfrnative::kStatusOk;
}

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemCfrNativeGpuBindings_lastEngineCode(JNIEnv* /*env*/, jclass /*clazz*/) {
  return g_last_engine_code.load(std::memory_order_relaxed);
}
