/*
 * HoldemCfrNativeCpuBindings.cpp -- CPU JNI binding for CFR (Counterfactual
 * Regret Minimization) game-tree solving in the sicfun poker analytics system.
 *
 * Bridges sicfun.holdem.HoldemCfrNativeCpuBindings to the pure C++ engine in
 * CfrNativeSolverCore.hpp. Exposes three JNI entry points:
 *
 *   solveTree()      -- Full solve: returns average strategies for all infosets
 *                       plus the expected value for player 0.
 *   solveTreeRoot()  -- Root-only solve: returns the average strategy for a
 *                       single specified information set (cheaper than full).
 *   solveTreeFixed() -- Fixed-point (Q30 probability / Q13 value) variant of
 *                       solveTree for reduced memory and better GPU throughput.
 *
 * Unlike the Bayes/DDRE bindings, CFR uses GetXxxArrayRegion (copy-based) rather
 * than GetPrimitiveArrayCritical (zero-copy), because the CFR solver allocates
 * working memory during execution and the critical-section restriction would
 * prevent that.
 *
 * Compiled into: sicfun_native_cpu.dll
 *
 * Reports engine code 1 (CPU) on success.
 */

#include <jni.h>

#include <atomic>
#include <vector>

#include "CfrNativeSolverCore.hpp"

namespace {

constexpr jint kEngineUnknown = 0;
constexpr jint kEngineCpu = 1;

std::atomic<jint> g_last_engine_code(kEngineUnknown);

/* Checks and clears any pending JNI exception. */
bool clear_pending_jni_exception(JNIEnv* env) {
  if (!env->ExceptionCheck()) {
    return false;
  }
  env->ExceptionClear();
  return true;
}

/* Reads a JNI int array into a std::vector via GetIntArrayRegion (copy-based). */
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

/* Reads a JNI double array into a std::vector via GetDoubleArrayRegion (copy-based). */
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

/* Writes a std::vector<double> back to a JNI array via SetDoubleArrayRegion. */
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

/* Writes a single double value into element 0 of a JNI double array. */
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

/* Writes a std::vector<int> back to a JNI int array. Used for fixed-point output. */
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

/* Writes a single int value into element 0 of a JNI int array. Used for fixed-point EV. */
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

}  // namespace

/*
 * JNI entry point: HoldemCfrNativeCpuBindings.solveTree()
 *
 * Full CFR solve using double-precision arithmetic. Reads the game tree topology
 * (node types, starts, counts, infosets, edges, probabilities, terminal utilities)
 * from JNI arrays into a TreeSpec, runs cfrnative::solve(), and writes back:
 *   - outAverageStrategiesArray: normalized average strategy for every infoset action.
 *   - outExpectedValueArray[0]: expected value for player 0 under the average strategy.
 */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemCfrNativeCpuBindings_solveTree(
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

  g_last_engine_code.store(kEngineCpu, std::memory_order_relaxed);
  return cfrnative::kStatusOk;
}

/*
 * JNI entry point: HoldemCfrNativeCpuBindings.solveTreeRoot()
 *
 * Root-only CFR solve. Same tree reading as solveTree(), but only extracts the
 * average strategy for the specified rootInfoSetIndex. Cheaper than a full solve
 * when only the root decision is needed (e.g., for real-time advice).
 */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemCfrNativeCpuBindings_solveTreeRoot(
    JNIEnv* env,
    jclass /*clazz*/,
    jint iterations,
    jint averagingDelay,
    jboolean cfrPlus,
    jboolean linearAveraging,
    jint rootNodeId,
    jint rootInfoSetIndex,
    jintArray nodeTypesArray,
    jintArray nodeStartsArray,
    jintArray nodeCountsArray,
    jintArray nodeInfosetsArray,
    jintArray edgeChildIdsArray,
    jdoubleArray edgeProbabilitiesArray,
    jdoubleArray terminalUtilitiesArray,
    jintArray infosetPlayersArray,
    jintArray infosetActionCountsArray,
    jdoubleArray outRootStrategyArray) {
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

  cfrnative::RootSolveOutput output;
  status = cfrnative::solve_root(spec, static_cast<int>(rootInfoSetIndex), output);
  if (status != cfrnative::kStatusOk) {
    return status;
  }

  status = write_double_array(env, outRootStrategyArray, output.root_strategy);
  if (status != cfrnative::kStatusOk) return status;

  g_last_engine_code.store(kEngineCpu, std::memory_order_relaxed);
  return cfrnative::kStatusOk;
}

/*
 * JNI entry point: HoldemCfrNativeCpuBindings.solveTreeFixed()
 *
 * Fixed-point CFR solve using Q30 probability scale and Q13 value scale.
 * All probabilities and utilities are passed as raw int32 values (no floating
 * point). Outputs are also int32: average_strategies_raw (Q30 probabilities)
 * and expected_value_player0_raw (Q13 utility). This variant uses ~50% less
 * memory than the double-precision version and is the preferred path for GPU
 * batch solving where float register pressure matters.
 */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemCfrNativeCpuBindings_solveTreeFixed(
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

  g_last_engine_code.store(kEngineCpu, std::memory_order_relaxed);
  return cfrnative::kStatusOk;
}

/* Returns 0 (unknown) or 1 (CPU) depending on whether any solve succeeded. */
extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemCfrNativeCpuBindings_lastEngineCode(JNIEnv* /*env*/, jclass /*clazz*/) {
  return g_last_engine_code.load(std::memory_order_relaxed);
}
