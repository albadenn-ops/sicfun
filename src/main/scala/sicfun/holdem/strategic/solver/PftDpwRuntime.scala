package sicfun.holdem.strategic.solver

import sicfun.holdem.HoldemPomcpNativeBindings
import sicfun.holdem.gpu.GpuRuntimeSupport

/** Configuration for PFT-DPW POMDP solver (mirrors C++ PftDpwConfig).
  *
  * Default values match the C++ defaults in PftDpwSolver.hpp.
  *
  * @param numSimulations  number of MCTS simulation passes
  * @param gamma           discount factor in (0, 1) -- must satisfy gamma > 0 && gamma < 1
  * @param rMax            maximum single-step reward bound (used in UCB scaling)
  * @param ucbC            UCB1 exploration constant (>= 0)
  * @param kAction         DPW action widening coefficient (> 0)
  * @param alphaAction     DPW action widening exponent (> 0)
  * @param kObs            DPW observation widening coefficient (> 0)
  * @param alphaObs        DPW observation widening exponent (> 0)
  * @param maxDepth        tree depth limit before random rollout
  * @param seed            RNG seed for reproducibility
  */
final case class PftDpwConfig(
    numSimulations: Int = 1000,
    gamma: Double = 0.99,
    rMax: Double = 1.0,
    ucbC: Double = 1.0,
    kAction: Double = 2.0,
    alphaAction: Double = 0.5,
    kObs: Double = 2.0,
    alphaObs: Double = 0.5,
    maxDepth: Int = 50,
    seed: Long = 42L
)

/** Result from PFT-DPW solver.
  *
  * @param bestAction  action ID with highest visit count (robust policy estimate, Def 32)
  * @param qValues     Q(b, a) per action index (Def 30); zero for untried actions
  * @param visitCounts N(b, a) per action index; zero for untried actions
  * @param status      0 = success; non-zero status code on failure
  */
final case class PftDpwResult(
    bestAction: Int,
    qValues: Array[Double],
    visitCounts: Array[Int],
    status: Int
):
  /** True when status == 0 (kStatusOk). */
  def isSuccess: Boolean = status == 0

/** Tabular generative model for the POMDP solver.
  *
  * Represents a finite POMDP with discrete state/action/observation spaces.
  * All arrays are row-major flat encodings:
  *   - transitionTable: T(s, a) = transitionTable[s * numActions + a]
  *   - obsLikelihood:   O(o | s', a) = obsLikelihood[(s' * numActions + a) * numObs + o]
  *   - rewardTable:     R(s, a) = rewardTable[s * numActions + a]
  *
  * @param transitionTable flat [numStates * numActions] -> next state index
  * @param obsLikelihood   flat [numStates * numActions * numObs] -> P(o | s', a)
  * @param rewardTable     flat [numStates * numActions] -> R(s, a)
  * @param numStates       number of states |S|
  * @param numActions      number of actions |A|
  * @param numObs          number of observations |O|
  */
final case class TabularGenerativeModel(
    transitionTable: Array[Int],
    obsLikelihood: Array[Double],
    rewardTable: Array[Double],
    numStates: Int,
    numActions: Int,
    numObs: Int
):
  require(
    transitionTable.length == numStates * numActions,
    s"transition table size ${transitionTable.length} != $numStates * $numActions"
  )
  require(
    obsLikelihood.length == numStates * numActions * numObs,
    s"obs likelihood size ${obsLikelihood.length} != $numStates * $numActions * $numObs"
  )
  require(
    rewardTable.length == numStates * numActions,
    s"reward table size ${rewardTable.length} != $numStates * $numActions"
  )

/** Particle belief: weighted state indices (Definition 54).
  *
  * Represents the agent's probabilistic state estimate as a finite particle set
  * {(x_j, w_j)}_{j=1}^C with sum(w_j) = 1 (enforced by the native solver).
  *
  * @param stateIndices particle state indices (length C)
  * @param weights      particle weights (length C); need not be normalized
  */
final case class ParticleBelief(
    stateIndices: Array[Int],
    weights: Array[Double]
):
  require(
    stateIndices.length == weights.length,
    s"particle arrays must match: ${stateIndices.length} != ${weights.length}"
  )
  require(stateIndices.nonEmpty, "particle belief must be non-empty")

/** PFT-DPW POMDP solver runtime.
  *
  * Wraps native C++ PFT-DPW tree search (PftDpwSolver.hpp) via JNI.
  * Implements Definitions 29-32 (value functions) using particle beliefs
  * (Definition 54) with error bound formula (Definition 55).
  *
  * ==Library loading==
  * The native library is loaded lazily on the first call. The library name
  * `sicfun_pomcp_native` is resolved via the standard JNI library path
  * (`java.library.path`). An absolute path can be injected via:
  *   - System property: `sicfun.pomcp.native.path`
  *   - Environment variable: `sicfun_POMCP_NATIVE_PATH`
  *
  * ==Thread safety==
  * Library loading is synchronized; concurrent solve() calls are safe (each
  * call uses its own RNG seeded by the config's seed parameter).
  */
object PftDpwRuntime:

  private val PathProperty = "sicfun.pomcp.native.path"
  private val PathEnv = "sicfun_POMCP_NATIVE_PATH"
  private val DefaultLib = "sicfun_pomcp_native"

  @volatile private var loadState: Either[Throwable, Unit] = null

  /** Load the native library (idempotent: loads at most once).
    *
    * Uses GpuRuntimeSupport.loadNativeLibrary which probes:
    *   1. Explicit path via system property / environment variable
    *   2. System.loadLibrary (java.library.path)
    *   3. Local build fallback: src/main/native/build/sicfun_pomcp_native.dll
    */
  private def ensureLoaded(): Unit =
    if loadState == null then
      synchronized {
        if loadState == null then
          loadState =
            GpuRuntimeSupport.loadNativeLibrary(
              pathProperty = PathProperty,
              pathEnv = PathEnv,
              libProperty = PathProperty, // no separate lib property for PftDpw
              libEnv = PathEnv,
              defaultLib = DefaultLib,
              label = "PFT-DPW POMCP"
            ) match
              case Right(_)      => Right(())
              case Left(reason)  => Left(new UnsatisfiedLinkError(reason))
      }
    loadState match
      case Right(_)  => ()
      case Left(err) => throw err

  /** Solve a POMDP from a root particle belief using PFT-DPW tree search.
    *
    * Runs `config.numSimulations` MCTS passes from the root belief, then
    * extracts the best action (by visit count) and Q-values.
    *
    * @param model  tabular generative model (T, O, R)
    * @param belief root particle belief (Definition 54)
    * @param config solver configuration
    * @return solver result with best action, Q-values, visit counts, and status
    */
  def solve(
      model: TabularGenerativeModel,
      belief: ParticleBelief,
      config: PftDpwConfig = PftDpwConfig()
  ): PftDpwResult =
    ensureLoaded()

    val outQ = new Array[Double](model.numActions)
    val outV = new Array[Int](model.numActions)

    val packed = HoldemPomcpNativeBindings.solvePftDpw(
      model.transitionTable,
      model.obsLikelihood,
      model.rewardTable,
      model.numStates,
      model.numActions,
      model.numObs,
      belief.stateIndices,
      belief.weights,
      config.numSimulations,
      config.gamma,
      config.rMax,
      config.ucbC,
      config.kAction,
      config.alphaAction,
      config.kObs,
      config.alphaObs,
      config.maxDepth,
      config.seed,
      outQ,
      outV
    )

    /* Unpack: status in lower 32 bits, best_action in upper 32 bits. */
    val status = (packed & 0xFFFFFFFFL).toInt
    val bestAction = (packed >> 32).toInt

    PftDpwResult(bestAction, outQ, outV, status)

  /** Compute the particle belief error bound (Definition 55).
    *
    * Bounds the value function approximation error for a finite particle set:
    * {{{
    *   |V*(b) - V*(b_hat_C)| <= R_max / (1 - gamma) * sqrt(D2(b || b_hat_C) / 2)
    * }}}
    *
    * For the self-diagnostic case (comparing the particle set against its own
    * implicit uniform distribution), the Renyi-2 divergence term is:
    * {{{
    *   D2 = ln(sum_j w_j^2 * C)  where C = weights.length
    * }}}
    *
    * Interpretation:
    *   - Uniform weights (all w_j = 1/C): D2 = ln(1) = 0, bound = 0.
    *   - Concentrated weights (one particle has all weight): D2 = ln(C), maximum bound.
    *
    * @param weights normalized particle weights (should sum to 1)
    * @param rMax    maximum single-step reward bound
    * @param gamma   discount factor in (0, 1)
    * @return upper bound on |V*(b) - V*(b_hat_C)|
    */
  def particleErrorBound(
      weights: Array[Double],
      rMax: Double,
      gamma: Double
  ): Double =
    val sumWSquared = weights.foldLeft(0.0)((acc, w) => acc + w * w)
    val d2 = math.log(sumWSquared * weights.length.toDouble)
    val bound = (rMax / (1.0 - gamma)) * math.sqrt(math.max(0.0, d2) / 2.0)
    bound
