package sicfun.holdem.validation

import sicfun.core.DiscreteDistribution
import sicfun.holdem.cfr.{HoldemCfrConfig, HoldemCfrSolver}
import sicfun.holdem.engine.GtoSolveEngine
import sicfun.holdem.equity.{HoldemEquity, RangeParser}
import sicfun.holdem.types.*

import scala.collection.mutable
import scala.util.Random

/** Pluggable GTO baseline strategy for the simulated villain.
  *
  * The villain strategy provides the "competent baseline" action before
  * [[LeakInjectedVillain]] applies potential deviations. Two implementations exist:
  *   - [[EquityBasedStrategy]]: fast heuristic using equity thresholds (no solver)
  *   - [[CfrVillainStrategy]]: actual Nash equilibrium via CFR solver (slower, more accurate)
  *
  * @see [[HeadsUpSimulator]] which uses this to generate villain GTO actions
  */
trait VillainStrategy:
  /** Choose an action given the villain's hand, game state, available actions, and equity.
    *
    * @param hand           villain's hole cards
    * @param state          current game state snapshot
    * @param candidates     available actions (Fold, Check, Call, Raise variants)
    * @param equityVsRandom villain's equity against a uniform random range
    * @param rng            random number generator for mixed strategy sampling
    * @return the chosen poker action
    */
  def decide(
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      equityVsRandom: Double,
      rng: Random
  ): PokerAction

/** Equity-threshold heuristic — the original HeadsUpSimulator strategy.
  *
  * Uses simple equity-vs-random thresholds to decide actions. Fast (no solver call)
  * but not game-theoretically optimal. Produces "competent but imperfect" play that
  * is adequate as a baseline for leak-injected opponents — the leaks are the signal
  * being tested, not the baseline quality.
  *
  * Decision logic:
  *   - Facing a bet: fold/call/raise based on equity vs pot odds with street-specific
  *     adjustments (wider preflop defense, occasional floats postflop)
  *   - Checked to: value bet strong hands (equity >= 0.60), bluff occasionally with
  *     air (equity <= 0.30), check everything else
  */
final class EquityBasedStrategy extends VillainStrategy:
  def decide(
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      equityVsRandom: Double,
      rng: Random
  ): PokerAction =
    if candidates.size <= 1 then return candidates.headOption.getOrElse(PokerAction.Check)

    val potOdds = state.potOdds
    val street = state.street

    if state.toCall > 0 then
      val raiseActions = candidates.collect { case r: PokerAction.Raise => r }
      if street == Street.Preflop && state.position == Position.Button then
        if equityVsRandom >= 0.35 then
          if raiseActions.nonEmpty then raiseActions.head else PokerAction.Call
        else if equityVsRandom >= 0.20 && rng.nextDouble() < 0.4 then
          if raiseActions.nonEmpty then raiseActions.head else PokerAction.Call
        else PokerAction.Fold
      else if equityVsRandom >= 0.75 then
        if raiseActions.nonEmpty && rng.nextDouble() < 0.6 then raiseActions.head
        else PokerAction.Call
      else if equityVsRandom >= potOdds + 0.05 then
        if street == Street.Preflop && raiseActions.nonEmpty && rng.nextDouble() < 0.25 then
          raiseActions.head
        else if equityVsRandom >= 0.55 && raiseActions.nonEmpty && rng.nextDouble() < 0.20 then
          raiseActions.head
        else PokerAction.Call
      else if street == Street.Preflop && rng.nextDouble() < 0.70 then
        if raiseActions.nonEmpty && rng.nextDouble() < 0.20 then raiseActions.head
        else PokerAction.Call
      else if rng.nextDouble() < 0.25 then
        PokerAction.Call
      else PokerAction.Fold
    else
      if equityVsRandom >= 0.60 then
        val raiseActions = candidates.collect { case r: PokerAction.Raise => r }
        if raiseActions.nonEmpty then raiseActions(rng.nextInt(raiseActions.size))
        else PokerAction.Check
      else if equityVsRandom <= 0.30 && rng.nextDouble() < 0.25 then
        val raiseActions = candidates.collect { case r: PokerAction.Raise => r }
        if raiseActions.nonEmpty then raiseActions.head else PokerAction.Check
      else PokerAction.Check

/** CFR equilibrium strategy — solves each decision point via HoldemCfrSolver.
  *
  * Computes Nash equilibrium mixed strategy, then samples an action.
  * Slower than EquityBasedStrategy but produces actual equilibrium play.
  *
  * Uses street-appropriate opponent ranges: uniform preflop (opponent could
  * have anything), narrowed postflop (opponent continued with a preflop-viable
  * hand). Without narrowing, uniform postflop ranges make aggression dominant
  * because ~60% of random hands miss the flop and fold to any raise.
  */
final class CfrVillainStrategy(
    config: HoldemCfrConfig = HoldemCfrConfig(
      iterations = 500,
      equityTrials = 1_000,
      maxVillainHands = 64,
      includeVillainReraises = true
    ),
    allowHeuristicFallback: Boolean = true,
    collectSolveTiming: Boolean = false
) extends VillainStrategy:

  private[holdem] def allowsHeuristicFallback: Boolean = allowHeuristicFallback
  private val policyCache = mutable.HashMap.empty[CfrVillainStrategy.CacheKey, CfrVillainStrategy.CachedPolicy]
  private val cacheStats = CfrVillainStrategy.CacheStats()
  private val servedByProvider = mutable.HashMap.empty[String, Long].withDefaultValue(0L)
  private val solvedByProvider = mutable.HashMap.empty[String, Long].withDefaultValue(0L)
  private val solveWallByProviderNanos = mutable.HashMap.empty[String, Long].withDefaultValue(0L)
  private val solveTimingStats = CfrVillainStrategy.SolveTimingStats()

  // HU Button opening range — the hands opponent would have opened preflop.
  // Parsed once and reused across all decisions.
  private lazy val buttonOpenHands: Set[HoleCards] =
    RangeParser.parseWithHands(CfrVillainStrategy.HuButtonOpenRange) match
      case Right(result) => result.hands
      case Left(_) => Set.empty // fallback: will use fullRange

  private[holdem] def cacheStatsSnapshot: CfrVillainStrategy.CacheStatsSnapshot =
    CfrVillainStrategy.CacheStatsSnapshot(
      hits = cacheStats.hits,
      misses = cacheStats.misses,
      size = policyCache.size.toLong
    )

  private[holdem] def providerCountsSnapshot: Map[String, Long] =
    servedByProvider.toMap

  private[holdem] def solvedProviderCountsSnapshot: Map[String, Long] =
    solvedByProvider.toMap

  private[holdem] def solveTimingSnapshot: CfrVillainStrategy.SolveTimingSnapshot =
    CfrVillainStrategy.SolveTimingSnapshot(
      solveCount = solveTimingStats.solveCount,
      solveWallNanos = solveTimingStats.solveWallNanos,
      solveWallByProviderNanos = solveWallByProviderNanos.toMap,
      nativeProfileCount = solveTimingStats.nativeProfileCount,
      nativePrepareNanos = solveTimingStats.nativePrepareNanos,
      nativePrepareSupportNanos = solveTimingStats.nativePrepareSupportNanos,
      nativePrepareEquityNanos = solveTimingStats.nativePrepareEquityNanos,
      nativePrepareResponseNanos = solveTimingStats.nativePrepareResponseNanos,
      nativePrepareGameBuildNanos = solveTimingStats.nativePrepareGameBuildNanos,
      nativeSpecBuildNanos = solveTimingStats.nativeSpecBuildNanos,
      nativeSolveNanos = solveTimingStats.nativeSolveNanos,
      nativeUnpackNanos = solveTimingStats.nativeUnpackNanos
    )

  def decide(
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      equityVsRandom: Double,
      rng: Random
  ): PokerAction =
    if candidates.size <= 1 then return candidates.headOption.getOrElse(PokerAction.Check)
    try
      val rangeProfile = opponentRangeProfile(hand, state)
      val key = CfrVillainStrategy.CacheKey(
        hand = hand,
        board = state.board,
        streetOrdinal = state.street.ordinal,
        positionOrdinal = state.position.ordinal,
        potBits = java.lang.Double.doubleToLongBits(state.pot),
        toCallBits = java.lang.Double.doubleToLongBits(state.toCall),
        stackBits = java.lang.Double.doubleToLongBits(state.stackSize),
        candidateHash = GtoSolveEngine.hashActions(candidates),
        rangeKindOrdinal = rangeProfile.kind.ordinal
      )
      policyCache.get(key) match
        case Some(cached) =>
          cacheStats.hits += 1L
          recordProvider(cached.provider)
          GtoSolveEngine.sampleActionByPolicy(cached.orderedActionProbabilities, cached.bestAction, rng)
        case None =>
          val opponentRange = opponentRangeForProfile(hand, state, rangeProfile)
          val (policy, nativeProfileOpt, solveWallNanos) =
            if collectSolveTiming then
              val solveStarted = System.nanoTime()
              val profiled = HoldemCfrSolver.solveShallowDecisionPolicyProfiled(
                hero = hand,
                state = state,
                villainPosterior = opponentRange,
                candidateActions = candidates,
                config = config
              )
              (
                profiled.policy,
                profiled.nativeProfile,
                math.max(1L, System.nanoTime() - solveStarted)
              )
            else
              (
                HoldemCfrSolver.solveShallowDecisionPolicy(
                  hero = hand,
                  state = state,
                  villainPosterior = opponentRange,
                  candidateActions = candidates,
                  config = config
                ),
                None,
                0L
              )
          val orderedActionProbabilities =
            GtoSolveEngine.orderedPositiveProbabilities(candidates, policy.actionProbabilities)
          cacheStats.misses += 1L
          recordProvider(policy.provider)
          recordSolvedProvider(policy.provider)
          if collectSolveTiming then
            recordSolveTiming(policy.provider, solveWallNanos, nativeProfileOpt)
          if policyCache.size >= CfrVillainStrategy.MaxPolicyCacheEntries then policyCache.clear()
          policyCache.update(
            key,
            CfrVillainStrategy.CachedPolicy(
              orderedActionProbabilities = orderedActionProbabilities,
              bestAction = policy.bestAction,
              provider = policy.provider
            )
          )
          GtoSolveEngine.sampleActionByPolicy(orderedActionProbabilities, policy.bestAction, rng)
    catch
      case err: Exception =>
        if allowHeuristicFallback then
          EquityBasedStrategy().decide(hand, state, candidates, equityVsRandom, rng)
        else throw err

  /** Preflop: uniform range (opponent could hold anything).
    * Postflop: narrowed to hands the opponent would have opened/continued with
    * preflop, excluding dead cards. Falls back to uniform if narrowed range is
    * too small (< 20 hands).
    */
  private def opponentRangeProfile(
      hand: HoleCards,
      state: GameState
  ): CfrVillainStrategy.OpponentRangeProfile =
    if state.street == Street.Preflop || buttonOpenHands.isEmpty then
      CfrVillainStrategy.OpponentRangeProfile(CfrVillainStrategy.OpponentRangeKind.FullRange, Vector.empty)
    else
      val dead = hand.asSet ++ state.board.asSet
      val viable = buttonOpenHands.iterator.filter(h => !h.toVector.exists(dead.contains)).toVector
      if viable.size >= 20 then
        CfrVillainStrategy.OpponentRangeProfile(
          CfrVillainStrategy.OpponentRangeKind.NarrowedButtonOpenRange,
          viable
        )
      else
        CfrVillainStrategy.OpponentRangeProfile(CfrVillainStrategy.OpponentRangeKind.FullRange, Vector.empty)

  private def opponentRangeForProfile(
      hand: HoleCards,
      state: GameState,
      profile: CfrVillainStrategy.OpponentRangeProfile
  ): DiscreteDistribution[HoleCards] =
    profile.kind match
      case CfrVillainStrategy.OpponentRangeKind.FullRange =>
        HoldemEquity.fullRange(hand, state.board)
      case CfrVillainStrategy.OpponentRangeKind.NarrowedButtonOpenRange =>
        DiscreteDistribution.uniform(profile.viableHands)

  private def recordProvider(provider: String): Unit =
    servedByProvider.update(provider, servedByProvider(provider) + 1L)

  private def recordSolvedProvider(provider: String): Unit =
    solvedByProvider.update(provider, solvedByProvider(provider) + 1L)

  private def recordSolveTiming(
      provider: String,
      solveWallNanos: Long,
      nativeProfileOpt: Option[sicfun.holdem.cfr.HoldemCfrNativeDecisionProfile]
  ): Unit =
    solveTimingStats.solveCount += 1L
    solveTimingStats.solveWallNanos += solveWallNanos
    solveWallByProviderNanos.update(provider, solveWallByProviderNanos(provider) + solveWallNanos)
    nativeProfileOpt.foreach { nativeProfile =>
      solveTimingStats.nativeProfileCount += 1L
      solveTimingStats.nativePrepareNanos += nativeProfile.prepareNanos
      solveTimingStats.nativePrepareSupportNanos += nativeProfile.prepareSupportNanos
      solveTimingStats.nativePrepareEquityNanos += nativeProfile.prepareEquityNanos
      solveTimingStats.nativePrepareResponseNanos += nativeProfile.prepareResponseNanos
      solveTimingStats.nativePrepareGameBuildNanos += nativeProfile.prepareGameBuildNanos
      solveTimingStats.nativeSpecBuildNanos += nativeProfile.specBuildNanos
      solveTimingStats.nativeSolveNanos += nativeProfile.nativeSolveNanos
      solveTimingStats.nativeUnpackNanos += nativeProfile.unpackNanos
    }

object CfrVillainStrategy:
  private final case class CacheKey(
      hand: HoleCards,
      board: Board,
      streetOrdinal: Int,
      positionOrdinal: Int,
      potBits: Long,
      toCallBits: Long,
      stackBits: Long,
      candidateHash: Int,
      rangeKindOrdinal: Int
  )

  private final case class CachedPolicy(
      orderedActionProbabilities: Vector[(PokerAction, Double)],
      bestAction: PokerAction,
      provider: String
  )

  private final case class CacheStats(
      var hits: Long = 0L,
      var misses: Long = 0L
  )

  private final case class SolveTimingStats(
      var solveCount: Long = 0L,
      var solveWallNanos: Long = 0L,
      var nativeProfileCount: Long = 0L,
      var nativePrepareNanos: Long = 0L,
      var nativePrepareSupportNanos: Long = 0L,
      var nativePrepareEquityNanos: Long = 0L,
      var nativePrepareResponseNanos: Long = 0L,
      var nativePrepareGameBuildNanos: Long = 0L,
      var nativeSpecBuildNanos: Long = 0L,
      var nativeSolveNanos: Long = 0L,
      var nativeUnpackNanos: Long = 0L
  )

  private[holdem] final case class CacheStatsSnapshot(
      hits: Long,
      misses: Long,
      size: Long
  )

  private[holdem] final case class SolveTimingSnapshot(
      solveCount: Long,
      solveWallNanos: Long,
      solveWallByProviderNanos: Map[String, Long],
      nativeProfileCount: Long,
      nativePrepareNanos: Long,
      nativePrepareSupportNanos: Long,
      nativePrepareEquityNanos: Long,
      nativePrepareResponseNanos: Long,
      nativePrepareGameBuildNanos: Long,
      nativeSpecBuildNanos: Long,
      nativeSolveNanos: Long,
      nativeUnpackNanos: Long
  ):
    def profiledNativeTotalNanos: Long =
      nativePrepareNanos + nativeSpecBuildNanos + nativeSolveNanos + nativeUnpackNanos

  private enum OpponentRangeKind:
    case FullRange
    case NarrowedButtonOpenRange

  private final case class OpponentRangeProfile(
      kind: OpponentRangeKind,
      viableHands: Vector[HoleCards]
  )

  /** Maximum policy cache size before eviction (full clear). */
  private val MaxPolicyCacheEntries = 100000

  // Standard HU Button opening range (~85% of hands).
  // Source: TableFormat.defaultRangeStringsHeadsUp(Position.Button)
  val HuButtonOpenRange: String =
    "22+, A2s+, K2s+, Q4s+, J6s+, T6s+, 96s+, 86s+, 76s, 65s, 54s, A2o+, K7o+, Q8o+, J8o+, T8o+, 98o"
