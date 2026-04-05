package sicfun.holdem.strategic.solver

import sicfun.holdem.HoldemPomcpNativeBindings
import sicfun.holdem.gpu.GpuRuntimeSupport
import java.util.concurrent.atomic.AtomicReference

/** Runtime wrapper for W-POMCP native solver via JNI.
  *
  * Implements Def 56 (factored particle filtering across rivals) by
  * delegating to the C++ WPomcpSolver via JNI. The Scala side constructs
  * the particle arrays and action distributions; the C++ side runs the
  * Monte Carlo tree search.
  *
  * Thread Safety:
  * Library loading is guarded by AtomicReference CAS. The solve call
  * itself is stateless (creates a fresh solver per call), so concurrent
  * calls from different threads are safe.
  */
private[strategic] object WPomcpRuntime:

  /** Configuration for a W-POMCP search.
    *
    * @param numSimulations  Monte Carlo simulations per search call
    * @param discount        gamma: discount factor in (0, 1)
    * @param exploration     UCB1 exploration constant c >= 0
    * @param rMax            maximum absolute reward (positive)
    * @param maxDepth        maximum tree depth per simulation
    * @param essThreshold    ESS ratio for resampling trigger in (0, 1]
    * @param seed            RNG seed for reproducibility
    */
  final case class Config(
      numSimulations: Int = 1000,
      discount: Double = 0.99,
      exploration: Double = 1.0,
      rMax: Double = 1.0,
      maxDepth: Int = 50,
      essThreshold: Double = 0.5,
      seed: Long = 42L
  ):
    require(numSimulations > 0, s"numSimulations must be positive, got $numSimulations")
    require(discount > 0.0 && discount < 1.0, s"discount must be in (0,1), got $discount")
    require(exploration >= 0.0, s"exploration must be non-negative, got $exploration")
    require(rMax > 0.0, s"rMax must be positive, got $rMax")
    require(maxDepth > 0, s"maxDepth must be positive, got $maxDepth")
    require(essThreshold > 0.0 && essThreshold <= 1.0,
      s"essThreshold must be in (0,1], got $essThreshold")

  /** Per-rival particle set for the factored belief.
    *
    * @param rivalTypes   discrete type index per particle (theta^{R,i})
    * @param privStates   discrete private state index per particle
    * @param weights      importance weights per particle (positive, will be normalized)
    */
  final case class RivalParticles(
      rivalTypes: Array[Int],
      privStates: Array[Int],
      weights: Array[Double]
  ):
    require(rivalTypes.length == privStates.length && rivalTypes.length == weights.length,
      "All particle arrays must have the same length")
    require(rivalTypes.nonEmpty, "Particle set must be non-empty")
    def particleCount: Int = rivalTypes.length

  /** Public state component of the factored belief. */
  final case class PublicState(street: Int, pot: Double)

  /** Input to a W-POMCP search.
    *
    * @param publicState         shared public state
    * @param rivalParticles      per-rival particle sets, length = |R|
    * @param heroActionCount     number of available hero actions
    * @param rivalActionProbs    per-rival action probabilities, flat [rival][action]
    * @param rewards             reward per hero action
    */
  final case class SearchInput(
      publicState: PublicState,
      rivalParticles: IndexedSeq[RivalParticles],
      heroActionCount: Int,
      rivalActionProbs: Array[Double],
      rewards: Array[Double]
  ):
    require(rivalParticles.nonEmpty, "Must have at least one rival")
    require(rivalParticles.size <= 8, s"Max 8 rivals, got ${rivalParticles.size}")
    require(heroActionCount > 0, s"Must have at least one hero action")
    require(rewards.length == heroActionCount,
      s"rewards length ${rewards.length} != heroActionCount $heroActionCount")
    def rivalCount: Int = rivalParticles.size

  /** Result of a W-POMCP search.
    *
    * @param actionValues  estimated Q(root, a) per hero action
    * @param bestAction    argmax action index
    * @param rootValue     max Q(root, a)
    */
  final case class SearchResult(
      actionValues: Array[Double],
      bestAction: Int,
      rootValue: Double
  )

  /* Library loading state. */
  private val PathProperty = "sicfun.pomcp.native.path"
  private val PathEnv = "sicfun_POMCP_NATIVE_PATH"
  private val LibProperty = "sicfun.pomcp.native.lib"
  private val LibEnv = "sicfun_POMCP_NATIVE_LIB"
  private val DefaultLib = "sicfun_pomcp_native"

  private val loadState: AtomicReference[Option[Either[String, Unit]]] =
    new AtomicReference(None)

  /** Check if the native library is available. */
  def isAvailable: Boolean = ensureLoaded().isRight

  /** Load the native library if not already loaded.
    *
    * Uses GpuRuntimeSupport.loadNativeLibrary which probes:
    *   1. Explicit path via system property / environment variable
    *   2. System.loadLibrary (java.library.path)
    *   3. Local build fallback: src/main/native/build/sicfun_pomcp_native.dll
    */
  def ensureLoaded(): Either[String, Unit] =
    loadState.get() match
      case Some(result) => result
      case None =>
        val result = GpuRuntimeSupport.loadNativeLibrary(
          pathProperty = PathProperty,
          pathEnv = PathEnv,
          libProperty = LibProperty,
          libEnv = LibEnv,
          defaultLib = DefaultLib,
          label = "W-POMCP"
        ).map(_ => ())
        loadState.compareAndSet(None, Some(result))
        loadState.get().get

  /** Run W-POMCP search.
    *
    * Returns Left(errorMessage) on failure, Right(SearchResult) on success.
    */
  def solve(input: SearchInput, config: Config): Either[String, SearchResult] =
    ensureLoaded() match
      case Left(err) => Left(err)
      case Right(()) =>
        /* Flatten per-rival particles into flat arrays. */
        val totalParticles = input.rivalParticles.map(_.particleCount).sum
        val particlesPerRival = input.rivalParticles.map(_.particleCount).toArray
        val allTypes = new Array[Int](totalParticles)
        val allPrivs = new Array[Int](totalParticles)
        val allWeights = new Array[Double](totalParticles)
        var offset = 0
        for rp <- input.rivalParticles do
          System.arraycopy(rp.rivalTypes, 0, allTypes, offset, rp.particleCount)
          System.arraycopy(rp.privStates, 0, allPrivs, offset, rp.particleCount)
          System.arraycopy(rp.weights, 0, allWeights, offset, rp.particleCount)
          offset += rp.particleCount

        /* Allocate output arrays. */
        val outActionValues = new Array[Double](input.heroActionCount)
        val outBestAction = new Array[Int](1)
        val outRootValue = new Array[Double](1)

        val status = HoldemPomcpNativeBindings.solveWPomcp(
          input.rivalCount,
          particlesPerRival,
          allTypes,
          allPrivs,
          allWeights,
          input.publicState.street,
          input.publicState.pot,
          input.heroActionCount,
          input.rivalActionProbs,
          input.rewards,
          config.numSimulations,
          config.discount,
          config.exploration,
          config.rMax,
          config.maxDepth,
          config.essThreshold,
          config.seed,
          outActionValues,
          outBestAction,
          outRootValue
        )

        if status == 0 then
          Right(SearchResult(outActionValues, outBestAction(0), outRootValue(0)))
        else
          Left(describeStatus(status))

  /** Describe a native status code. */
  private def describeStatus(code: Int): String = code match
    case 0   => "OK"
    case 100 => "Null array argument"
    case 101 => "Array length mismatch"
    case 102 => "JNI array read failure"
    case 124 => "JNI array write failure"
    case 160 => "Invalid configuration"
    case 170 => "Invalid particle count"
    case 171 => "Invalid rival count"
    case 172 => "Invalid action count"
    case 173 => "Degenerate particle weights (all zero)"
    case 174 => "Maximum rivals exceeded (max 8)"
    case 175 => "Simulation overflow"
    case _   => s"Unknown native error code: $code"
