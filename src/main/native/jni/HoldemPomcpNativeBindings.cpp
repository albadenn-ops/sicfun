/*
 * HoldemPomcpNativeBindings.cpp -- CPU JNI binding for PFT-DPW POMDP tree
 * search in the sicfun poker analytics system.
 *
 * Bridges sicfun.holdem.HoldemPomcpNativeBindings (Scala) to the pure C++
 * engine in PftDpwSolver.hpp. Exposes two JNI entry points:
 *
 *   solvePftDpw()    -- Run PFT-DPW from a root particle belief (Def 54).
 *                       Returns best action and Q-values per action (Def 32).
 *   lastEngineCode() -- Returns 3 (PFT-DPW CPU) after a successful solve.
 *
 * Unlike the Bayes/DDRE bindings, this uses GetXxxArrayRegion (copy-based)
 * rather than GetPrimitiveArrayCritical (zero-copy), because the PFT-DPW
 * solver allocates significant working memory (tree nodes) during execution
 * and the critical-section restriction would prevent that.
 *
 * Compiled into: sicfun_pomcp_native.dll
 *
 * Error status codes: shared with PftDpwSolver.hpp (0=ok, 100=null,
 * 101=mismatch, 102=read failure, 124=write failure, 200+=POMDP errors).
 *
 * Return value of solvePftDpw: (best_action << 32) | status.
 * The Scala wrapper unpacks this: status = (packed & 0xFFFFFFFFL).toInt,
 *                                  best_action = (packed >> 32).toInt.
 */

#include <jni.h>

#include <atomic>
#include <vector>

#include "PftDpwSolver.hpp"
#include "WPomcpSolver.hpp"

namespace {

constexpr jint kEngineUnknown = 0;
constexpr jint kEnginePftDpw = 3;  /* Engine code 3: PFT-DPW CPU. */
constexpr jint kEngineCpu = 4;     /* Engine code 4: W-POMCP CPU. */

/* Tracks which engine last completed successfully. Read by lastEngineCode(). */
std::atomic<jint> g_last_engine_code(kEngineUnknown);

/* Checks for a pending JNI exception, clears it if present, returns true if found. */
bool clear_pending_jni_exception(JNIEnv* env) {
  if (!env->ExceptionCheck()) {
    return false;
  }
  env->ExceptionClear();
  return true;
}

/* Reads a JNI int array into a std::vector via GetIntArrayRegion (copy-based).
 * Returns kStatusOk on success, or an error status. */
int read_int_array(JNIEnv* env, jintArray array, std::vector<int>& out) {
  if (array == nullptr) {
    return pftdpw::kStatusNullArray;
  }
  const jsize length = env->GetArrayLength(array);
  out.resize(static_cast<size_t>(length));
  if (length > 0) {
    env->GetIntArrayRegion(array, 0, length,
                           reinterpret_cast<jint*>(out.data()));
    if (clear_pending_jni_exception(env)) {
      return pftdpw::kStatusReadFailure;
    }
  }
  return pftdpw::kStatusOk;
}

/* Reads a JNI double array into a std::vector via GetDoubleArrayRegion (copy-based). */
int read_double_array(JNIEnv* env, jdoubleArray array, std::vector<double>& out) {
  if (array == nullptr) {
    return pftdpw::kStatusNullArray;
  }
  const jsize length = env->GetArrayLength(array);
  out.resize(static_cast<size_t>(length));
  if (length > 0) {
    env->GetDoubleArrayRegion(array, 0, length,
                              reinterpret_cast<jdouble*>(out.data()));
    if (clear_pending_jni_exception(env)) {
      return pftdpw::kStatusReadFailure;
    }
  }
  return pftdpw::kStatusOk;
}

/* Writes a std::vector<double> back to a JNI double array via SetDoubleArrayRegion.
 * The JNI array must be at least as large as data.size(). */
int write_double_array(JNIEnv* env, jdoubleArray array,
                        const std::vector<double>& data) {
  if (array == nullptr) {
    return pftdpw::kStatusNullArray;
  }
  const jsize length = env->GetArrayLength(array);
  if (static_cast<size_t>(length) < data.size()) {
    return pftdpw::kStatusLengthMismatch;
  }
  if (!data.empty()) {
    env->SetDoubleArrayRegion(array, 0, static_cast<jsize>(data.size()),
                               reinterpret_cast<const jdouble*>(data.data()));
    if (clear_pending_jni_exception(env)) {
      return pftdpw::kStatusWriteFailure;
    }
  }
  return pftdpw::kStatusOk;
}

/* Writes a std::vector<int> back to a JNI int array via SetIntArrayRegion. */
int write_int_array(JNIEnv* env, jintArray array,
                     const std::vector<int>& data) {
  if (array == nullptr) {
    return pftdpw::kStatusNullArray;
  }
  const jsize length = env->GetArrayLength(array);
  if (static_cast<size_t>(length) < data.size()) {
    return pftdpw::kStatusLengthMismatch;
  }
  if (!data.empty()) {
    env->SetIntArrayRegion(array, 0, static_cast<jsize>(data.size()),
                            reinterpret_cast<const jint*>(data.data()));
    if (clear_pending_jni_exception(env)) {
      return pftdpw::kStatusWriteFailure;
    }
  }
  return pftdpw::kStatusOk;
}

}  // namespace

extern "C" {

/*
 * JNI entry point: sicfun.holdem.HoldemPomcpNativeBindings.solvePftDpw()
 *
 * Parameters:
 *   transitionTable  -- [numStates * numActions] int array: T(s, a) -> next state
 *   obsLikelihood    -- [numStates * numActions * numObs] double: O(o | s', a)
 *   rewardTable      -- [numStates * numActions] double: R(s, a)
 *   numStates        -- number of states in the model
 *   numActions       -- number of actions in the model
 *   numObs           -- number of observations in the model
 *   particleStates   -- [C] int: root belief particle state indices (Def 54)
 *   particleWeights  -- [C] double: root belief particle weights (Def 54)
 *   numSimulations   -- number of MCTS simulations
 *   gamma            -- discount factor (0, 1)
 *   rMax             -- maximum single-step reward
 *   ucbC             -- UCB1 exploration constant
 *   kAction          -- DPW action widening coefficient
 *   alphaAction      -- DPW action widening exponent
 *   kObs             -- DPW observation widening coefficient
 *   alphaObs         -- DPW observation widening exponent
 *   maxDepth         -- tree depth limit
 *   seed             -- RNG seed for reproducibility
 *   outQValues       -- [numActions] double output: Q(b,a) per action (zeros for untried)
 *   outVisitCounts   -- [numActions] int output: N(b,a) per action (zeros for untried)
 *
 * Returns: (best_action << 32) | status. Status 0 = success.
 *   Scala unpacks: status = (packed & 0xFFFFFFFFL).toInt, best_action = (packed >> 32).toInt
 */
JNIEXPORT jlong JNICALL
Java_sicfun_holdem_HoldemPomcpNativeBindings_solvePftDpw(
    JNIEnv* env, jclass /*cls*/,
    jintArray transitionTable,
    jdoubleArray obsLikelihood,
    jdoubleArray rewardTable,
    jint numStates, jint numActions, jint numObs,
    jintArray particleStates,
    jdoubleArray particleWeights,
    jint numSimulations,
    jdouble gamma, jdouble rMax, jdouble ucbC,
    jdouble kAction, jdouble alphaAction,
    jdouble kObs, jdouble alphaObs,
    jint maxDepth, jlong seed,
    jdoubleArray outQValues,
    jintArray outVisitCounts) {

  /* Read input arrays into local vectors. */
  std::vector<int> trans;
  std::vector<double> obs_lik;
  std::vector<double> rewards;
  std::vector<int> part_states;
  std::vector<double> part_weights;

  int s;
  s = read_int_array(env, transitionTable, trans);
  if (s != pftdpw::kStatusOk) return static_cast<jlong>(s);
  s = read_double_array(env, obsLikelihood, obs_lik);
  if (s != pftdpw::kStatusOk) return static_cast<jlong>(s);
  s = read_double_array(env, rewardTable, rewards);
  if (s != pftdpw::kStatusOk) return static_cast<jlong>(s);
  s = read_int_array(env, particleStates, part_states);
  if (s != pftdpw::kStatusOk) return static_cast<jlong>(s);
  s = read_double_array(env, particleWeights, part_weights);
  if (s != pftdpw::kStatusOk) return static_cast<jlong>(s);

  /* Validate array dimensions. Use long long for overflow-safe multiplication. */
  const long long ns = static_cast<long long>(numStates);
  const long long na = static_cast<long long>(numActions);
  const long long no = static_cast<long long>(numObs);

  if (static_cast<long long>(trans.size()) != ns * na) {
    return static_cast<jlong>(pftdpw::kStatusLengthMismatch);
  }
  if (static_cast<long long>(obs_lik.size()) != ns * na * no) {
    return static_cast<jlong>(pftdpw::kStatusLengthMismatch);
  }
  if (static_cast<long long>(rewards.size()) != ns * na) {
    return static_cast<jlong>(pftdpw::kStatusLengthMismatch);
  }
  if (part_states.size() != part_weights.size()) {
    return static_cast<jlong>(pftdpw::kStatusLengthMismatch);
  }
  if (part_states.empty()) {
    return static_cast<jlong>(pftdpw::kStatusNoParticles);
  }

  /* Build the generative model (no copies: points into local vectors). */
  pftdpw::GenerativeModel model;
  model.transition_table = trans.data();
  model.obs_likelihood = obs_lik.data();
  model.reward_table = rewards.data();
  model.num_states = static_cast<int>(numStates);
  model.num_actions = static_cast<int>(numActions);
  model.num_obs = static_cast<int>(numObs);

  /* Build root particle belief (Def 54): normalize weights on entry. */
  pftdpw::ParticleBelief root;
  root.particles.reserve(part_states.size());
  for (size_t i = 0; i < part_states.size(); ++i) {
    root.particles.push_back({part_states[i], part_weights[i]});
  }
  root.normalize();

  /* Build solver configuration. */
  pftdpw::PftDpwConfig cfg;
  cfg.num_simulations = static_cast<int>(numSimulations);
  cfg.gamma = static_cast<double>(gamma);
  cfg.r_max = static_cast<double>(rMax);
  cfg.ucb_c = static_cast<double>(ucbC);
  cfg.k_action = static_cast<double>(kAction);
  cfg.alpha_action = static_cast<double>(alphaAction);
  cfg.k_obs = static_cast<double>(kObs);
  cfg.alpha_obs = static_cast<double>(alphaObs);
  cfg.max_depth = static_cast<int>(maxDepth);
  cfg.num_particles = static_cast<int>(part_states.size());
  cfg.seed = static_cast<uint64_t>(seed);

  /* Run PFT-DPW tree search. */
  pftdpw::PftDpwResult result = pftdpw::solve(root, model, cfg);

  if (result.status != pftdpw::kStatusOk) {
    return static_cast<jlong>(result.status);
  }

  /* Write outputs: expand from tried-action-indexed vectors to full numActions arrays.
   * Untried actions get Q=0.0 and visit_count=0. */
  std::vector<double> q_out(static_cast<size_t>(numActions), 0.0);
  std::vector<int> v_out(static_cast<size_t>(numActions), 0);
  for (size_t i = 0; i < result.action_ids.size(); ++i) {
    const int aid = result.action_ids[i];
    if (aid >= 0 && aid < static_cast<int>(numActions)) {
      q_out[static_cast<size_t>(aid)] = result.q_values[i];
      v_out[static_cast<size_t>(aid)] = result.visit_counts[i];
    }
  }

  s = write_double_array(env, outQValues, q_out);
  if (s != pftdpw::kStatusOk) return static_cast<jlong>(s);
  s = write_int_array(env, outVisitCounts, v_out);
  if (s != pftdpw::kStatusOk) return static_cast<jlong>(s);

  g_last_engine_code.store(kEnginePftDpw, std::memory_order_relaxed);

  /* Pack result: best_action in upper 32 bits, status (0) in lower 32. */
  return (static_cast<jlong>(result.best_action) << 32) |
         static_cast<jlong>(pftdpw::kStatusOk);
}

/*
 * JNI entry point: sicfun.holdem.HoldemPomcpNativeBindings.lastEngineCode()
 *
 * Returns the engine code from the last successful solvePftDpw() call.
 * 0 = no computation yet, 3 = PFT-DPW CPU.
 */
JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemPomcpNativeBindings_lastEngineCode(
    JNIEnv* /*env*/, jclass /*cls*/) {
  return g_last_engine_code.load(std::memory_order_relaxed);
}

/*
 * JNI method: sicfun.holdem.HoldemPomcpNativeBindings.solveWPomcp
 *
 * Bridges JVM arrays to wpomcp::solve_raw(). Uses critical array access
 * for zero-copy. Releases arrays in reverse acquisition order.
 *
 * Java signature:
 *   static native int solveWPomcp(
 *       int rivalCount,
 *       int[] particlesPerRival,
 *       int[] particleTypes,
 *       int[] particlePrivStates,
 *       double[] particleWeights,
 *       int pubStreet,
 *       double pubPot,
 *       int numHeroActions,
 *       double[] rivalActionProbs,
 *       double[] rewards,
 *       int numSimulations,
 *       double discount,
 *       double exploration,
 *       double rMax,
 *       int maxDepth,
 *       double essThreshold,
 *       long seed,
 *       double[] outActionValues,
 *       int[] outBestAction,
 *       double[] outRootValue
 *   );
 */
JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemPomcpNativeBindings_solveWPomcp(
    JNIEnv* env,
    jclass /* cls */,
    jint rival_count,
    jintArray j_particles_per_rival,
    jintArray j_particle_types,
    jintArray j_particle_priv_states,
    jdoubleArray j_particle_weights,
    jint pub_street,
    jdouble pub_pot,
    jint num_hero_actions,
    jdoubleArray j_rival_action_probs,
    jdoubleArray j_rewards,
    jint num_simulations,
    jdouble discount,
    jdouble exploration,
    jdouble r_max,
    jint max_depth,
    jdouble ess_threshold,
    jlong seed,
    jdoubleArray j_out_action_values,
    jintArray j_out_best_action,
    jdoubleArray j_out_root_value) {

  /* Null checks on all array arguments. */
  if (j_particles_per_rival == nullptr || j_particle_types == nullptr ||
      j_particle_priv_states == nullptr || j_particle_weights == nullptr ||
      j_rival_action_probs == nullptr || j_rewards == nullptr ||
      j_out_action_values == nullptr || j_out_best_action == nullptr ||
      j_out_root_value == nullptr) {
    return wpomcp::kStatusNullArray;
  }

  /* Acquire critical arrays (zero-copy access to JVM heap). */
  jint* ppr = static_cast<jint*>(
      env->GetPrimitiveArrayCritical(j_particles_per_rival, nullptr));
  jint* ptypes = static_cast<jint*>(
      env->GetPrimitiveArrayCritical(j_particle_types, nullptr));
  jint* pprivs = static_cast<jint*>(
      env->GetPrimitiveArrayCritical(j_particle_priv_states, nullptr));
  jdouble* pweights = static_cast<jdouble*>(
      env->GetPrimitiveArrayCritical(j_particle_weights, nullptr));
  jdouble* rap = static_cast<jdouble*>(
      env->GetPrimitiveArrayCritical(j_rival_action_probs, nullptr));
  jdouble* rew = static_cast<jdouble*>(
      env->GetPrimitiveArrayCritical(j_rewards, nullptr));
  jdouble* out_vals = static_cast<jdouble*>(
      env->GetPrimitiveArrayCritical(j_out_action_values, nullptr));
  jint* out_best = static_cast<jint*>(
      env->GetPrimitiveArrayCritical(j_out_best_action, nullptr));
  jdouble* out_root = static_cast<jdouble*>(
      env->GetPrimitiveArrayCritical(j_out_root_value, nullptr));

  if (ppr == nullptr || ptypes == nullptr || pprivs == nullptr ||
      pweights == nullptr || rap == nullptr || rew == nullptr ||
      out_vals == nullptr || out_best == nullptr || out_root == nullptr) {
    /* Release any successfully acquired arrays before returning. */
    if (out_root) env->ReleasePrimitiveArrayCritical(j_out_root_value, out_root, JNI_ABORT);
    if (out_best) env->ReleasePrimitiveArrayCritical(j_out_best_action, out_best, JNI_ABORT);
    if (out_vals) env->ReleasePrimitiveArrayCritical(j_out_action_values, out_vals, JNI_ABORT);
    if (rew) env->ReleasePrimitiveArrayCritical(j_rewards, rew, JNI_ABORT);
    if (rap) env->ReleasePrimitiveArrayCritical(j_rival_action_probs, rap, JNI_ABORT);
    if (pweights) env->ReleasePrimitiveArrayCritical(j_particle_weights, pweights, JNI_ABORT);
    if (pprivs) env->ReleasePrimitiveArrayCritical(j_particle_priv_states, pprivs, JNI_ABORT);
    if (ptypes) env->ReleasePrimitiveArrayCritical(j_particle_types, ptypes, JNI_ABORT);
    if (ppr) env->ReleasePrimitiveArrayCritical(j_particles_per_rival, ppr, JNI_ABORT);
    return wpomcp::kStatusReadFailure;
  }

  /* Call the solver. */
  int status = wpomcp::solve_raw(
      rival_count, ppr, ptypes, pprivs, pweights,
      pub_street, pub_pot,
      num_hero_actions, rap, rew,
      num_simulations, discount, exploration, r_max,
      max_depth, ess_threshold, seed,
      out_vals, out_best, out_root);

  /* Release critical arrays in reverse order. Commit output arrays only on success. */
  int release_mode = (status == wpomcp::kStatusOk) ? 0 : JNI_ABORT;
  env->ReleasePrimitiveArrayCritical(j_out_root_value, out_root, release_mode);
  env->ReleasePrimitiveArrayCritical(j_out_best_action, out_best, release_mode);
  env->ReleasePrimitiveArrayCritical(j_out_action_values, out_vals, release_mode);
  env->ReleasePrimitiveArrayCritical(j_rewards, rew, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(j_rival_action_probs, rap, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(j_particle_weights, pweights, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(j_particle_priv_states, pprivs, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(j_particle_types, ptypes, JNI_ABORT);
  env->ReleasePrimitiveArrayCritical(j_particles_per_rival, ppr, JNI_ABORT);

  if (status == wpomcp::kStatusOk) {
    g_last_engine_code.store(kEngineCpu);
  }

  return status;
}

/* Self-test entry point for JNI. */
JNIEXPORT jint JNICALL
Java_sicfun_holdem_HoldemPomcpNativeBindings_selfTestWPomcp(
    JNIEnv* /* env */,
    jclass /* cls */) {
#ifdef WPOMCP_SELF_TEST
  return wpomcp::self_test();
#else
  return wpomcp::kStatusOk;  /* self-test not compiled in */
#endif
}

}  // extern "C"
