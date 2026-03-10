package sicfun.holdem.cfr
import sicfun.holdem.types.*
import sicfun.holdem.engine.*
import sicfun.holdem.*
import sicfun.holdem.equity.*
import sicfun.holdem.gpu.*

import sicfun.core.{Card, CardId, DiscreteDistribution, HandEvaluator}

import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicReference
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/** Configuration for one-street Hold'em CFR solves. */
final case class HoldemCfrConfig(
    iterations: Int = 1_500,
    cfrPlus: Boolean = true,
    averagingDelay: Int = 200,
    linearAveraging: Boolean = true,
    maxVillainHands: Int = 96,
    equityTrials: Int = 4_000,
    includeVillainReraises: Boolean = true,
    villainReraiseMultipliers: Vector[Double] = Vector(2.0),
    preferNativeBatch: Boolean = true,
    rngSeed: Long = 1L
):
  require(iterations > 0, "iterations must be positive")
  require(averagingDelay >= 0, "averagingDelay must be non-negative")
  require(maxVillainHands > 0, "maxVillainHands must be positive")
  require(equityTrials > 0, "equityTrials must be positive")
  require(
    villainReraiseMultipliers.forall(m => m > 1.0 && m.isFinite),
    "villainReraiseMultipliers must be finite and > 1.0"
  )

/** Solved CFR baseline for a single decision point. */
final case class HoldemCfrSolution(
    actionProbabilities: Map[PokerAction, Double],
    actionEvaluations: Vector[ActionEvaluation],
    bestAction: PokerAction,
    expectedValuePlayer0: Double,
    heroRootBestResponseValue: Double,
    villainBestResponseValue: Double,
    rootDeviationGap: Double,
    villainDeviationGap: Double,
    localExploitability: Double,
    iterations: Int,
    infoSetKey: String,
    villainSupport: Int,
    provider: String
):
  require(actionProbabilities.nonEmpty, "actionProbabilities must be non-empty")
  require(actionEvaluations.nonEmpty, "actionEvaluations must be non-empty")
  require(iterations > 0, "iterations must be positive")
  require(villainSupport > 0, "villainSupport must be positive")
  require(provider.trim.nonEmpty, "provider must be non-empty")

/** Lightweight CFR output for decision-time sampling.
  *
  * Computes only the root mixed policy and fallback best action, skipping
  * exploitability diagnostics and per-action EV reports.
  */
final case class HoldemCfrDecisionPolicy(
    actionProbabilities: Map[PokerAction, Double],
    bestAction: PokerAction,
    iterations: Int,
    infoSetKey: String,
    villainSupport: Int,
    provider: String
):
  require(actionProbabilities.nonEmpty, "actionProbabilities must be non-empty")
  require(iterations > 0, "iterations must be positive")
  require(villainSupport > 0, "villainSupport must be positive")
  require(provider.trim.nonEmpty, "provider must be non-empty")

/** CFR baseline solver for a heads-up, one-street action abstraction.
  *
  * This module integrates with existing project equity engines:
  *  - preflop: optional native/hybrid batch path via [[HeadsUpGpuRuntime]]
  *  - fallback and postflop: [[HoldemEquity.equityMonteCarlo]]
  */
object HoldemCfrSolver:
  private val Epsilon = 1e-12
  private val CfrProviderProperty = "sicfun.cfr.provider"
  private val CfrProviderEnv = "sicfun_CFR_PROVIDER"
  private val CfrAutoBenchmarkIterationsProperty = "sicfun.cfr.auto.benchmarkIterations"
  private val CfrAutoBenchmarkIterationsEnv = "sicfun_CFR_AUTO_BENCHMARK_ITERATIONS"
  private val CfrAutoMinSpeedupProperty = "sicfun.cfr.auto.nativeMinSpeedup"
  private val CfrAutoMinSpeedupEnv = "sicfun_CFR_AUTO_NATIVE_MIN_SPEEDUP"
  private val DefaultAutoBenchmarkIterations = 240
  private val DefaultAutoMinSpeedup = 1.02
  private val MaxEquityLookupCacheEntries = 8_192
  private val MaxVillainSupportOrderingCacheEntries = 64
  private val autoChosenProviderRef = new AtomicReference[AutoSelection](AutoSelection.Unset)
  private val equityLookupCache = new ConcurrentHashMap[EquityLookupCacheKey, Array[Double]]()
  private val villainSupportOrderingCache =
    new ConcurrentHashMap[VillainSupportOrderingCacheKey, Array[OrderedVillainSupportEntry]]()

  private enum Provider:
    case Scala
    case NativeCpu
    case NativeGpu

  private enum AutoSelection:
    case Unset
    case Provider(provider: HoldemCfrSolver.Provider)

  private final case class PolicySolveResult(
      provider: Provider,
      iterations: Int,
      expectedValuePlayer0: Double,
      averagePolicy: Map[String, Map[PokerAction, Double]]
  )

  private final case class RootPolicySolveResult(
      provider: Provider,
      iterations: Int,
      actionProbabilities: Map[PokerAction, Double]
  )

  private final case class PreparedGame(
      heroActions: Vector[PokerAction],
      villainSupport: Vector[(HoleCards, Double)],
      game: HoldemDecisionGame
  )

  private final case class DirectShallowMixedReraise(
      foldValue: Double,
      callValues: Array[Double]
  )

  private final case class DirectShallowContext(
      publicState: GameState,
      villainDistribution: Vector[(HoleCards, Double)],
      villainResponseByRaise: Map[Double, Vector[PokerAction]],
      heroResponseByReraise: Map[(Double, Double), Vector[PokerAction]],
      equityByVillain: Array[Double],
      weightedEquity: Double,
      infoSetKey: String
  )

  private final case class EquityLookupCacheKey(
      heroId: Int,
      boardPacked: Long,
      villainIds: Vector[Int],
      trials: Int,
      preferNativeBatch: Boolean,
      rngSeed: Long
  )

  private final class VillainSupportOrderingCacheKey(private val ref: AnyRef):
    override def equals(other: Any): Boolean =
      other match
        case that: VillainSupportOrderingCacheKey => ref eq that.ref
        case _ => false

    override def hashCode(): Int =
      System.identityHashCode(ref)

  private final case class OrderedVillainSupportEntry(
      hand: HoleCards,
      weight: Double,
      handId: Int,
      mask: Long
  )

  def solve(
      hero: HoleCards,
      state: GameState,
      villainPosterior: DiscreteDistribution[HoleCards],
      candidateActions: Vector[PokerAction],
      config: HoldemCfrConfig = HoldemCfrConfig()
  ): HoldemCfrSolution =
    val prepared = prepareGame(
      hero = hero,
      state = state,
      villainPosterior = villainPosterior,
      candidateActions = candidateActions,
      config = config
    )
    val heroActions = prepared.heroActions
    val villainSupport = prepared.villainSupport
    val game = prepared.game
    val cfrConfig = CfrSolver.Config(
      iterations = config.iterations,
      cfrPlus = config.cfrPlus,
      averagingDelay = config.averagingDelay,
      linearAveraging = config.linearAveraging
    )
    val policySolve = solvePolicy(game, cfrConfig)
    val averagePolicy = policySolve.averagePolicy
    val actionProbabilities = normalizedPolicyForActions(
      heroActions,
      averagePolicy.getOrElse(game.heroRootInfoSetKey, Map.empty)
    )
    val strategyValue = game.evaluateAveragePolicy(averagePolicy)
    val actionEvaluations =
      heroActions.map { action =>
        ActionEvaluation(action, game.evaluateRootAction(action, averagePolicy))
      }
    val bestAction = actionEvaluations.maxBy(_.expectedValue).action
    val heroRootBestResponseValue = actionEvaluations.map(_.expectedValue).max
    val villainBestResponseValue = game.evaluateWithVillainBestResponse(averagePolicy)
    val rootDeviationGap = math.max(0.0, heroRootBestResponseValue - strategyValue)
    val villainDeviationGap = math.max(0.0, strategyValue - villainBestResponseValue)
    val localExploitability = rootDeviationGap + villainDeviationGap

    HoldemCfrSolution(
      actionProbabilities = actionProbabilities,
      actionEvaluations = actionEvaluations,
      bestAction = bestAction,
      expectedValuePlayer0 = strategyValue,
      heroRootBestResponseValue = heroRootBestResponseValue,
      villainBestResponseValue = villainBestResponseValue,
      rootDeviationGap = rootDeviationGap,
      villainDeviationGap = villainDeviationGap,
      localExploitability = localExploitability,
      iterations = policySolve.iterations,
      infoSetKey = game.heroRootInfoSetKey,
      villainSupport = villainSupport.length,
      provider = providerLabel(policySolve.provider)
    )

  def solveDecisionPolicy(
      hero: HoleCards,
      state: GameState,
      villainPosterior: DiscreteDistribution[HoleCards],
      candidateActions: Vector[PokerAction],
      config: HoldemCfrConfig = HoldemCfrConfig()
  ): HoldemCfrDecisionPolicy =
    val sanitizedHeroActions = sanitizeHeroActions(state, candidateActions)
    maybeSolveTerminalRootPolicy(
      hero = hero,
      state = state,
      villainPosterior = villainPosterior,
      heroActions = sanitizedHeroActions,
      config = config
    ) match
      case Some(policy) =>
        return policy
      case None => ()

    val prepared = prepareGame(
      hero = hero,
      state = state,
      villainPosterior = villainPosterior,
      candidateActions = sanitizedHeroActions,
      config = config
    )
    val cfrConfig = CfrSolver.Config(
      iterations = config.iterations,
      cfrPlus = config.cfrPlus,
      averagingDelay = config.averagingDelay,
      linearAveraging = config.linearAveraging
    )
    val rootPolicy = solveRootPolicy(prepared.game, cfrConfig)
    val bestAction = selectBestActionByProbability(prepared.heroActions, rootPolicy.actionProbabilities)
    HoldemCfrDecisionPolicy(
      actionProbabilities = rootPolicy.actionProbabilities,
      bestAction = bestAction,
      iterations = rootPolicy.iterations,
      infoSetKey = prepared.game.heroRootInfoSetKey,
      villainSupport = prepared.villainSupport.length,
      provider = providerLabel(rootPolicy.provider)
    )

  /** Direct solver for the shallow hall/action-abstraction tree.
    *
    * This bypasses iterative CFR when the tree shape is:
    * - hero root actions are terminal except optional raises
    * - villain responses may include multiple re-raise sizes
    * - hero can only fold/call versus each re-raise
    *
    * Falls back to [[solveDecisionPolicy]] when the abstraction is richer.
    */
  def solveShallowDecisionPolicy(
      hero: HoleCards,
      state: GameState,
      villainPosterior: DiscreteDistribution[HoleCards],
      candidateActions: Vector[PokerAction],
      config: HoldemCfrConfig = HoldemCfrConfig()
  ): HoldemCfrDecisionPolicy =
    val sanitizedHeroActions = sanitizeHeroActions(state, candidateActions)
    maybeSolveTerminalRootPolicy(
      hero = hero,
      state = state,
      villainPosterior = villainPosterior,
      heroActions = sanitizedHeroActions,
      config = config
    ) match
      case Some(policy) =>
        policy
      case None =>
        maybeSolveDirectShallowPolicyFast(
          hero = hero,
          state = state,
          villainPosterior = villainPosterior,
          heroActions = sanitizedHeroActions,
          config = config
        )
          .getOrElse {
            val prepared = prepareGame(
              hero = hero,
              state = state,
              villainPosterior = villainPosterior,
              candidateActions = sanitizedHeroActions,
              config = config
            )
            maybeSolveDirectShallowPolicy(prepared.game, prepared.heroActions, prepared.villainSupport.length)
              .getOrElse(
                solveDecisionPolicy(
                  hero = hero,
                  state = state,
                  villainPosterior = villainPosterior,
                  candidateActions = sanitizedHeroActions,
                  config = config
                )
              )
          }

  private def maybeSolveDirectShallowPolicyFast(
      hero: HoleCards,
      state: GameState,
      villainPosterior: DiscreteDistribution[HoleCards],
      heroActions: Vector[PokerAction],
      config: HoldemCfrConfig
  ): Option[HoldemCfrDecisionPolicy] =
    val deadMask = deadCardMask(hero, state.board)
    val villainSupport = trimVillainSupport(villainPosterior, deadMask, config.maxVillainHands)
    val equityByVillain = buildEquityLookup(
      hero = hero,
      board = state.board,
      villains = villainSupport.map(_._1),
      trials = config.equityTrials,
      preferNativeBatch = config.preferNativeBatch,
      rngSeed = config.rngSeed
    )
    val villainResponseByRaise = buildVillainResponses(state, heroActions, config)
    val heroResponsesByReraise = buildHeroReraiseResponses(state, villainResponseByRaise)
    val context = DirectShallowContext(
      publicState = state,
      villainDistribution = villainSupport,
      villainResponseByRaise = villainResponseByRaise,
      heroResponseByReraise = heroResponsesByReraise,
      equityByVillain = equityByVillain,
      weightedEquity = weightedEquity(villainSupport, equityByVillain),
      infoSetKey = rootDirectShallowInfoSetKey(hero, state)
    )
    maybeSolveDirectShallowPolicy(context, heroActions, villainSupport.length)

  private def rootDirectShallowInfoSetKey(hero: HoleCards, state: GameState): String =
    val boardToken =
      if state.board.cards.isEmpty then "preflop"
      else state.board.cards.map(_.toToken).mkString
    s"hero:${hero.toToken}|root-direct-shallow|${state.street}|$boardToken|${state.pot}|${state.toCall}|${state.stackSize}"

  private def maybeSolveDirectShallowPolicy(
      context: DirectShallowContext,
      heroActions: Vector[PokerAction],
      villainSupportSize: Int
  ): Option[HoldemCfrDecisionPolicy] =
    val actionValues = Vector.newBuilder[(PokerAction, Double)]
    var idx = 0
    while idx < heroActions.length do
      directShallowRootActionValue(context, heroActions(idx)) match
        case Some(value) =>
          actionValues += heroActions(idx) -> value
        case None =>
          return None
      idx += 1

    val resolved = actionValues.result()
    val bestValue = resolved.map(_._2).max
    val bestActions =
      resolved.collect { case (action, value) if math.abs(value - bestValue) <= Epsilon => action }
    val share = 1.0 / bestActions.length.toDouble
    val actionProbabilities =
      heroActions.map { action =>
        action -> (if bestActions.contains(action) then share else 0.0)
      }.toMap

    Some(
      HoldemCfrDecisionPolicy(
        actionProbabilities = actionProbabilities,
        bestAction = bestActions.head,
        iterations = 1,
        infoSetKey = context.infoSetKey,
        villainSupport = villainSupportSize,
        provider = "direct-shallow"
      )
    )

  private def directShallowRootActionValue(
      context: DirectShallowContext,
      action: PokerAction
  ): Option[Double] =
    action match
      case PokerAction.Fold =>
        Some(0.0)
      case PokerAction.Check =>
        Some(context.weightedEquity * context.publicState.pot)
      case PokerAction.Call =>
        Some((context.weightedEquity * (context.publicState.pot + context.publicState.toCall)) - context.publicState.toCall)
      case PokerAction.Raise(amount) =>
        directShallowRaiseValue(context, amount)

  private def directShallowRaiseValue(
      context: DirectShallowContext,
      heroRaise: Double
  ): Option[Double] =
    val responses = context.villainResponseByRaise.getOrElse(heroRaise, Vector(PokerAction.Fold, PokerAction.Call))
    val reraises = responses.collect { case PokerAction.Raise(amount) => amount }
    val villainCount = context.villainDistribution.length
    val probabilities = new Array[Double](villainCount)
    val equities = new Array[Double](villainCount)
    val baseValues = new Array[Double](villainCount)
    val foldValue =
      shallowTerminalUtility(
        state = context.publicState,
        equity = 0.0,
        heroInvestment = heroRaise,
        villainInvestment = 0.0,
        winnerByFold = Some(0)
      )

    var villainIdx = 0
    while villainIdx < villainCount do
      val (villain, probability) = context.villainDistribution(villainIdx)
      val equity = requireEquityLookup(context.equityByVillain, villain)
      probabilities(villainIdx) = probability
      equities(villainIdx) = equity
      baseValues(villainIdx) =
        math.min(
          foldValue,
          shallowTerminalUtility(
            state = context.publicState,
            equity = equity,
            heroInvestment = heroRaise,
            villainInvestment = heroRaise,
            winnerByFold = None
          )
        )
      villainIdx += 1

    val mixedReraises = ArrayBuffer.empty[DirectShallowMixedReraise]
    var reraiseIdx = 0
    while reraiseIdx < reraises.length do
      val villainRaise = reraises(reraiseIdx)
      val heroResponses = context.heroResponseByReraise.getOrElse((heroRaise, villainRaise), Vector(PokerAction.Fold))
      if heroResponses.exists {
          case PokerAction.Fold | PokerAction.Call => false
          case _                                   => true
        }
      then return None

      val canFold = heroResponses.contains(PokerAction.Fold)
      val canCall = heroResponses.contains(PokerAction.Call)
      if !canFold && !canCall then return None

      val foldToReraiseValue =
        shallowTerminalUtility(
          state = context.publicState,
          equity = 0.0,
          heroInvestment = heroRaise,
          villainInvestment = villainRaise,
          winnerByFold = Some(1)
        )

      if !canCall then
        villainIdx = 0
        while villainIdx < villainCount do
          baseValues(villainIdx) = math.min(baseValues(villainIdx), foldToReraiseValue)
          villainIdx += 1
      else
        val callReraiseValues = new Array[Double](villainCount)
        villainIdx = 0
        while villainIdx < villainCount do
          val callReraiseValue =
            shallowTerminalUtility(
              state = context.publicState,
              equity = equities(villainIdx),
              heroInvestment = villainRaise,
              villainInvestment = villainRaise,
              winnerByFold = None
            )
          if canFold then callReraiseValues(villainIdx) = callReraiseValue
          else baseValues(villainIdx) = math.min(baseValues(villainIdx), callReraiseValue)
          villainIdx += 1
        if canFold then
          mixedReraises += DirectShallowMixedReraise(
            foldValue = foldToReraiseValue,
            callValues = callReraiseValues
          )
      reraiseIdx += 1

    if mixedReraises.isEmpty then
      Some(weightedValue(probabilities, baseValues))
    else if mixedReraises.length == 1 then
      Some(directShallowSingleMixedReraiseValue(probabilities, baseValues, mixedReraises.head))
    else
      directShallowMultiMixedReraiseValue(probabilities, baseValues, mixedReraises.toVector)

  private def maybeSolveDirectShallowPolicy(
      game: HoldemDecisionGame,
      heroActions: Vector[PokerAction],
      villainSupportSize: Int
  ): Option[HoldemCfrDecisionPolicy] =
    maybeSolveDirectShallowPolicy(
      context = DirectShallowContext(
        publicState = game.publicState,
        villainDistribution = game.villainDistribution,
        villainResponseByRaise = game.villainResponseByRaise,
        heroResponseByReraise = game.heroResponseByReraise,
        equityByVillain = game.equityByVillain,
        weightedEquity = weightedEquity(game),
        infoSetKey = game.heroRootInfoSetKey
      ),
      heroActions = heroActions,
      villainSupportSize = villainSupportSize
    )

  private def maybeSolveTerminalRootPolicy(
      hero: HoleCards,
      state: GameState,
      villainPosterior: DiscreteDistribution[HoleCards],
      heroActions: Vector[PokerAction],
      config: HoldemCfrConfig
  ): Option[HoldemCfrDecisionPolicy] =
    if heroActions.isEmpty then None
    else if heroActions.exists {
        case PokerAction.Raise(_) => true
        case _                    => false
      }
    then None
    else
      val deadMask = deadCardMask(hero, state.board)
      val villainSupport = trimVillainSupport(villainPosterior, deadMask, config.maxVillainHands)
      val weightedEquityOpt =
        if heroActions.forall(_ == PokerAction.Fold) then None
        else
          val equityByVillain = buildEquityLookup(
            hero = hero,
            board = state.board,
            villains = villainSupport.map(_._1),
            trials = config.equityTrials,
            preferNativeBatch = config.preferNativeBatch,
            rngSeed = config.rngSeed
          )
          Some(
            villainSupport.foldLeft(0.0) { case (acc, (villain, probability)) =>
              acc + (probability * equityLookupOrZero(equityByVillain, villain))
            }
          )
      val actionValues = heroActions.map { action =>
        action -> terminalRootActionValue(state, action, weightedEquityOpt.getOrElse(0.0))
      }
      val bestValue = actionValues.map(_._2).max
      val tiedBest = actionValues.collect { case (action, value) if math.abs(value - bestValue) <= Epsilon => action }
      val bestAction = tiedBest.headOption.getOrElse(heroActions.head)
      val share = 1.0 / tiedBest.length.toDouble
      val actionProbabilities =
        heroActions.map { action =>
          action -> (if tiedBest.contains(action) then share else 0.0)
        }.toMap
      Some(
        HoldemCfrDecisionPolicy(
          actionProbabilities = actionProbabilities,
          bestAction = bestAction,
          iterations = 1,
          infoSetKey = s"hero:${hero.toToken}|root-direct|${state.street}|${state.board.cards.map(_.toToken).mkString}",
          villainSupport = villainSupport.length,
          provider = "direct"
        )
      )

  private def directShallowSingleMixedReraiseValue(
      probabilities: Array[Double],
      baseValues: Array[Double],
      mixedReraise: DirectShallowMixedReraise
  ): Double =
    val candidateCallProbabilities = scala.collection.mutable.ArrayBuffer[Double](0.0, 1.0)
    var villainIdx = 0
    while villainIdx < probabilities.length do
      val cappedValue = baseValues(villainIdx)
      val callReraiseValue = mixedReraise.callValues(villainIdx)
      val denominator = callReraiseValue - mixedReraise.foldValue
      if math.abs(denominator) > Epsilon then
        val callProbability = (cappedValue - mixedReraise.foldValue) / denominator
        if callProbability >= 0.0 && callProbability <= 1.0 then
          candidateCallProbabilities += callProbability
      villainIdx += 1

    var bestValue = Double.NegativeInfinity
    var candidateIdx = 0
    while candidateIdx < candidateCallProbabilities.length do
      val candidate = candidateCallProbabilities(candidateIdx)
      val expectedValue =
        directShallowExpectedMixedValue(
          probabilities = probabilities,
          baseValues = baseValues,
          mixedReraises = Vector(mixedReraise),
          callProbabilities = Array(candidate)
        )
      if expectedValue > bestValue then bestValue = expectedValue
      candidateIdx += 1
    bestValue

  private def directShallowMultiMixedReraiseValue(
      probabilities: Array[Double],
      baseValues: Array[Double],
      mixedReraises: Vector[DirectShallowMixedReraise]
  ): Option[Double] =
    val villainCount = probabilities.length
    val reraiseCount = mixedReraises.length
    val lowerBounds = new Array[Double](villainCount)
    var constantTerm = 0.0
    var villainIdx = 0
    while villainIdx < villainCount do
      var lowerBound = baseValues(villainIdx)
      var reraiseIdx = 0
      while reraiseIdx < reraiseCount do
        val mixedReraise = mixedReraises(reraiseIdx)
        lowerBound = math.min(lowerBound, mixedReraise.foldValue)
        lowerBound = math.min(lowerBound, mixedReraise.callValues(villainIdx))
        reraiseIdx += 1
      lowerBounds(villainIdx) = lowerBound
      constantTerm += probabilities(villainIdx) * lowerBound
      villainIdx += 1

    val variableCount = villainCount + reraiseCount
    val constraintCount = villainCount + (villainCount * reraiseCount) + reraiseCount
    val coefficients = Array.ofDim[Double](constraintCount, variableCount)
    val bounds = new Array[Double](constraintCount)
    var rowIdx = 0
    villainIdx = 0
    while villainIdx < villainCount do
      coefficients(rowIdx)(villainIdx) = 1.0
      bounds(rowIdx) = math.max(0.0, baseValues(villainIdx) - lowerBounds(villainIdx))
      rowIdx += 1
      villainIdx += 1

    var reraiseIdx = 0
    while reraiseIdx < reraiseCount do
      val mixedReraise = mixedReraises(reraiseIdx)
      villainIdx = 0
      while villainIdx < villainCount do
        coefficients(rowIdx)(villainIdx) = 1.0
        coefficients(rowIdx)(villainCount + reraiseIdx) =
          -(mixedReraise.callValues(villainIdx) - mixedReraise.foldValue)
        bounds(rowIdx) = math.max(0.0, mixedReraise.foldValue - lowerBounds(villainIdx))
        rowIdx += 1
        villainIdx += 1
      reraiseIdx += 1

    reraiseIdx = 0
    while reraiseIdx < reraiseCount do
      coefficients(rowIdx)(villainCount + reraiseIdx) = 1.0
      bounds(rowIdx) = 1.0
      rowIdx += 1
      reraiseIdx += 1

    val objective = new Array[Double](variableCount)
    villainIdx = 0
    while villainIdx < villainCount do
      objective(villainIdx) = probabilities(villainIdx)
      villainIdx += 1

    solveLinearProgramMax(
      coefficients = coefficients,
      bounds = bounds,
      objective = objective
    ).map(_ + constantTerm)

  private def directShallowExpectedMixedValue(
      probabilities: Array[Double],
      baseValues: Array[Double],
      mixedReraises: Vector[DirectShallowMixedReraise],
      callProbabilities: Array[Double]
  ): Double =
    var expectedValue = 0.0
    var villainIdx = 0
    while villainIdx < probabilities.length do
      var value = baseValues(villainIdx)
      var reraiseIdx = 0
      while reraiseIdx < mixedReraises.length do
        val mixedReraise = mixedReraises(reraiseIdx)
        val reraiseValue =
          mixedReraise.foldValue +
            (callProbabilities(reraiseIdx) * (mixedReraise.callValues(villainIdx) - mixedReraise.foldValue))
        value = math.min(value, reraiseValue)
        reraiseIdx += 1
      expectedValue += probabilities(villainIdx) * value
      villainIdx += 1
    expectedValue

  private def solveLinearProgramMax(
      coefficients: Array[Array[Double]],
      bounds: Array[Double],
      objective: Array[Double]
  ): Option[Double] =
    val rowCount = bounds.length
    val variableCount = objective.length
    if rowCount == 0 then Some(0.0)
    else
      val width = variableCount + rowCount + 1
      val tableau = Array.ofDim[Double](rowCount + 1, width)
      val basis = new Array[Int](rowCount)

      var rowIdx = 0
      while rowIdx < rowCount do
        if bounds(rowIdx) < -Epsilon then return None
        System.arraycopy(coefficients(rowIdx), 0, tableau(rowIdx), 0, variableCount)
        tableau(rowIdx)(variableCount + rowIdx) = 1.0
        tableau(rowIdx)(width - 1) = math.max(0.0, bounds(rowIdx))
        basis(rowIdx) = variableCount + rowIdx
        rowIdx += 1

      var colIdx = 0
      while colIdx < variableCount do
        tableau(rowCount)(colIdx) = -objective(colIdx)
        colIdx += 1

      val simplexEpsilon = 1e-10
      val maxIterations = math.max(256, rowCount * width * 8)
      var iteration = 0
      while iteration < maxIterations do
        var entering = -1
        colIdx = 0
        while colIdx < width - 1 && entering < 0 do
          if tableau(rowCount)(colIdx) < -simplexEpsilon then entering = colIdx
          colIdx += 1

        if entering < 0 then return Some(tableau(rowCount)(width - 1))

        var leaving = -1
        var bestRatio = Double.PositiveInfinity
        rowIdx = 0
        while rowIdx < rowCount do
          val coefficient = tableau(rowIdx)(entering)
          if coefficient > simplexEpsilon then
            val ratio = tableau(rowIdx)(width - 1) / coefficient
            if ratio < bestRatio - simplexEpsilon ||
              (math.abs(ratio - bestRatio) <= simplexEpsilon && (leaving < 0 || basis(rowIdx) < basis(leaving)))
            then
              bestRatio = ratio
              leaving = rowIdx
          rowIdx += 1

        if leaving < 0 then return None

        pivotTableau(tableau, leaving, entering)
        basis(leaving) = entering
        iteration += 1

      None

  private def pivotTableau(
      tableau: Array[Array[Double]],
      pivotRow: Int,
      pivotColumn: Int
  ): Unit =
    val width = tableau(pivotRow).length
    val pivotValue = tableau(pivotRow)(pivotColumn)
    var colIdx = 0
    while colIdx < width do
      tableau(pivotRow)(colIdx) /= pivotValue
      colIdx += 1

    var rowIdx = 0
    while rowIdx < tableau.length do
      if rowIdx != pivotRow then
        val factor = tableau(rowIdx)(pivotColumn)
        if math.abs(factor) > 1e-12 then
          colIdx = 0
          while colIdx < width do
            tableau(rowIdx)(colIdx) -= factor * tableau(pivotRow)(colIdx)
            colIdx += 1
      rowIdx += 1

  private def weightedValue(probabilities: Array[Double], values: Array[Double]): Double =
    var total = 0.0
    var idx = 0
    while idx < probabilities.length do
      total += probabilities(idx) * values(idx)
      idx += 1
    total

  private def weightedEquity(game: HoldemDecisionGame): Double =
    weightedEquity(game.villainDistribution, game.equityByVillain)

  private def weightedEquity(
      villainDistribution: Vector[(HoleCards, Double)],
      equityByVillain: Array[Double]
  ): Double =
    villainDistribution.foldLeft(0.0) { case (acc, (villain, probability)) =>
      acc + (probability * equityLookupOrZero(equityByVillain, villain))
    }

  private def shallowTerminalUtility(
      state: GameState,
      equity: Double,
      heroInvestment: Double,
      villainInvestment: Double,
      winnerByFold: Option[Int]
  ): Double =
    winnerByFold match
      case Some(0) =>
        state.pot + villainInvestment
      case Some(1) =>
        -heroInvestment
      case Some(other) =>
        throw new IllegalStateException(s"invalid fold winner marker: $other")
      case None =>
        val finalPot = state.pot + heroInvestment + villainInvestment
        (equity * finalPot) - heroInvestment

  private def prepareGame(
      hero: HoleCards,
      state: GameState,
      villainPosterior: DiscreteDistribution[HoleCards],
      candidateActions: Vector[PokerAction],
      config: HoldemCfrConfig
  ): PreparedGame =
    require(candidateActions.nonEmpty, "candidateActions must be non-empty")

    val heroActions = sanitizeHeroActions(state, candidateActions)
    require(heroActions.nonEmpty, "no legal hero actions after sanitization")

    val deadMask = deadCardMask(hero, state.board)
    val villainSupport = trimVillainSupport(villainPosterior, deadMask, config.maxVillainHands)

    val equityByVillain = buildEquityLookup(
      hero = hero,
      board = state.board,
      villains = villainSupport.map(_._1),
      trials = config.equityTrials,
      preferNativeBatch = config.preferNativeBatch,
      rngSeed = config.rngSeed
    )

    val villainResponseByRaise = buildVillainResponses(state, heroActions, config)
    val heroResponsesByReraise = buildHeroReraiseResponses(state, villainResponseByRaise)

    val game = HoldemDecisionGame(
      hero = hero,
      publicState = state,
      villainDistribution = villainSupport,
      heroActions = heroActions,
      villainResponseByRaise = villainResponseByRaise,
      heroResponseByReraise = heroResponsesByReraise,
      equityByVillain = equityByVillain
    )
    PreparedGame(
      heroActions = heroActions,
      villainSupport = villainSupport,
      game = game
    )

  private def toActionMap(
      snapshot: CfrSolver.InfoSetSnapshot[PokerAction]
  ): Map[PokerAction, Double] =
    snapshot.actions.zip(snapshot.strategy).toMap

  private def providerLabel(provider: Provider): String =
    provider match
      case Provider.Scala => "scala"
      case Provider.NativeCpu => "native-cpu"
      case Provider.NativeGpu => "native-gpu"

  private def normalizedPolicyForActions(
      actions: Vector[PokerAction],
      rawPolicy: Map[PokerAction, Double]
  ): Map[PokerAction, Double] =
    val cleaned = actions.map { action =>
      action -> math.max(0.0, rawPolicy.getOrElse(action, 0.0))
    }
    val total = cleaned.map(_._2).sum
    if total > Epsilon then
      val inv = 1.0 / total
      cleaned.map { case (action, probability) =>
        action -> (probability * inv)
      }.toMap
    else
      val uniform = 1.0 / actions.length.toDouble
      actions.map(action => action -> uniform).toMap

  private def normalizedPolicyFromVector(
      actions: Vector[PokerAction],
      strategy: Vector[Double]
  ): Map[PokerAction, Double] =
    require(actions.length == strategy.length, "actions/strategy length mismatch")
    val raw = actions.zip(strategy).toMap
    normalizedPolicyForActions(actions, raw)

  private def selectBestActionByProbability(
      actions: Vector[PokerAction],
      probabilities: Map[PokerAction, Double]
  ): PokerAction =
    actions.maxBy(action => probabilities.getOrElse(action, 0.0))

  private def solvePolicy(
      game: HoldemDecisionGame,
      config: CfrSolver.Config
  ): PolicySolveResult =
    val configured = resolveConfiguredProvider()
    configured match
      case Provider.Scala =>
        solveWithScala(game, config)
      case Provider.NativeCpu =>
        solveWithNative(game, config, HoldemCfrNativeRuntime.Backend.Cpu)
          .getOrElse(solveWithScala(game, config))
      case Provider.NativeGpu =>
        solveWithNative(game, config, HoldemCfrNativeRuntime.Backend.Gpu)
          .orElse(solveWithNative(game, config, HoldemCfrNativeRuntime.Backend.Cpu))
          .getOrElse(solveWithScala(game, config))

  private def solveRootPolicy(
      game: HoldemDecisionGame,
      config: CfrSolver.Config
  ): RootPolicySolveResult =
    val configured = resolveConfiguredProvider()
    configured match
      case Provider.Scala =>
        solveRootWithScala(game, config)
      case Provider.NativeCpu =>
        solveRootWithNative(game, config, HoldemCfrNativeRuntime.Backend.Cpu)
          .getOrElse(solveRootWithScala(game, config))
      case Provider.NativeGpu =>
        solveRootWithNative(game, config, HoldemCfrNativeRuntime.Backend.Gpu)
          .orElse(solveRootWithNative(game, config, HoldemCfrNativeRuntime.Backend.Cpu))
          .getOrElse(solveRootWithScala(game, config))

  private def solveWithScala(
      game: HoldemDecisionGame,
      config: CfrSolver.Config
  ): PolicySolveResult =
    val training = CfrSolver.solve(game = game, config = config)
    val averagePolicy = training.infosets.view.mapValues(toActionMap).toMap
    PolicySolveResult(
      provider = Provider.Scala,
      iterations = training.iterations,
      expectedValuePlayer0 = training.expectedValuePlayer0,
      averagePolicy = averagePolicy
    )

  private def solveWithNative(
      game: HoldemDecisionGame,
      config: CfrSolver.Config,
      backend: HoldemCfrNativeRuntime.Backend
  ): Option[PolicySolveResult] =
    val spec = game.toNativeTreeSpec
    HoldemCfrNativeRuntime.solveTree(
      backend = backend,
      spec = spec,
      config = config
    ) match
      case Left(reason) =>
        GpuRuntimeSupport.log(s"native CFR ${backend.toString.toLowerCase} solve unavailable: $reason")
        None
      case Right(nativeResult) =>
        val averagePolicy =
          policyFromNativeFlattened(spec, nativeResult.averageStrategiesFlattened)
        Some(
          PolicySolveResult(
            provider =
              backend match
                case HoldemCfrNativeRuntime.Backend.Cpu => Provider.NativeCpu
                case HoldemCfrNativeRuntime.Backend.Gpu => Provider.NativeGpu,
            iterations = config.iterations,
            expectedValuePlayer0 = nativeResult.expectedValuePlayer0,
            averagePolicy = averagePolicy
          )
        )

  private def solveRootWithScala(
      game: HoldemDecisionGame,
      config: CfrSolver.Config
  ): RootPolicySolveResult =
    val root = CfrSolver.solveRootPolicy(
      game = game,
      rootInfoSetKey = game.heroRootInfoSetKey,
      rootActions = game.heroActions,
      config = config
    )
    RootPolicySolveResult(
      provider = Provider.Scala,
      iterations = root.iterations,
      actionProbabilities = normalizedPolicyFromVector(root.actions, root.strategy)
    )

  private def solveRootWithNative(
      game: HoldemDecisionGame,
      config: CfrSolver.Config,
      backend: HoldemCfrNativeRuntime.Backend
  ): Option[RootPolicySolveResult] =
    val spec = game.toNativeTreeSpec
    HoldemCfrNativeRuntime.solveTree(
      backend = backend,
      spec = spec,
      config = config
    ) match
      case Left(reason) =>
        GpuRuntimeSupport.log(s"native CFR ${backend.toString.toLowerCase} root solve unavailable: $reason")
        None
      case Right(nativeResult) =>
        Some(
          RootPolicySolveResult(
            provider =
              backend match
                case HoldemCfrNativeRuntime.Backend.Cpu => Provider.NativeCpu
                case HoldemCfrNativeRuntime.Backend.Gpu => Provider.NativeGpu,
            iterations = config.iterations,
            actionProbabilities = rootPolicyFromNativeFlattened(
              spec = spec,
              flattened = nativeResult.averageStrategiesFlattened,
              rootActions = game.heroActions
            )
          )
        )

  private def policyFromNativeFlattened(
      spec: HoldemCfrNativeRuntime.NativeTreeSpec,
      flattened: Array[Double]
  ): Map[String, Map[PokerAction, Double]] =
    val expectedLength = spec.infosetActionCounts.sum
    require(flattened.length == expectedLength, s"native flattened strategy length mismatch: ${flattened.length} != $expectedLength")

    var cursor = 0
    val builder = Map.newBuilder[String, Map[PokerAction, Double]]
    var infosetIdx = 0
    while infosetIdx < spec.infosetKeys.length do
      val actions = spec.infosetActions(infosetIdx)
      val count = spec.infosetActionCounts(infosetIdx)
      val raw = Map.newBuilder[PokerAction, Double]
      var idx = 0
      var total = 0.0
      while idx < count do
        val p = math.max(0.0, flattened(cursor + idx))
        raw += actions(idx) -> p
        total += p
        idx += 1
      val normalized =
        if total > Epsilon then
          val inv = 1.0 / total
          raw.result().map { case (action, probability) => action -> (probability * inv) }
        else
          val uniform = 1.0 / actions.length.toDouble
          actions.map(action => action -> uniform).toMap
      builder += spec.infosetKeys(infosetIdx) -> normalized
      cursor += count
      infosetIdx += 1
    builder.result()

  private def rootPolicyFromNativeFlattened(
      spec: HoldemCfrNativeRuntime.NativeTreeSpec,
      flattened: Array[Double],
      rootActions: Vector[PokerAction]
  ): Map[PokerAction, Double] =
    val expectedLength = spec.infosetActionCounts.sum
    require(flattened.length == expectedLength, s"native flattened strategy length mismatch: ${flattened.length} != $expectedLength")
    var cursor = 0
    var infosetIdx = 0
    while infosetIdx < spec.rootInfoSetIndex do
      cursor += spec.infosetActionCounts(infosetIdx)
      infosetIdx += 1
    val count = spec.infosetActionCounts(spec.rootInfoSetIndex)
    val raw = Map.newBuilder[PokerAction, Double]
    var idx = 0
    while idx < count do
      raw += spec.infosetActions(spec.rootInfoSetIndex)(idx) -> math.max(0.0, flattened(cursor + idx))
      idx += 1
    normalizedPolicyForActions(rootActions, raw.result())

  private def resolveConfiguredProvider(): Provider =
    GpuRuntimeSupport.resolveNonEmptyLower(CfrProviderProperty, CfrProviderEnv) match
      case Some("scala" | "jvm") =>
        Provider.Scala
      case Some("native-cpu" | "cpu") =>
        val availability = HoldemCfrNativeRuntime.availability(HoldemCfrNativeRuntime.Backend.Cpu)
        if availability.available then Provider.NativeCpu
        else
          GpuRuntimeSupport.warn(s"CFR native CPU provider unavailable (${availability.detail}); falling back to Scala")
          Provider.Scala
      case Some("native-gpu" | "gpu" | "cuda") =>
        val gpuAvailability = HoldemCfrNativeRuntime.availability(HoldemCfrNativeRuntime.Backend.Gpu)
        if gpuAvailability.available then Provider.NativeGpu
        else
          val cpuAvailability = HoldemCfrNativeRuntime.availability(HoldemCfrNativeRuntime.Backend.Cpu)
          if cpuAvailability.available then
            GpuRuntimeSupport.warn(s"CFR native GPU provider unavailable (${gpuAvailability.detail}); using native CPU")
            Provider.NativeCpu
          else
            GpuRuntimeSupport.warn(s"CFR native GPU provider unavailable (${gpuAvailability.detail}); falling back to Scala")
            Provider.Scala
      case Some("auto") | None =>
        resolveAutoProvider()
      case Some(other) =>
        GpuRuntimeSupport.warn(s"unknown CFR provider '$other'; using auto selection")
        resolveAutoProvider()

  private def resolveAutoProvider(): Provider =
    autoChosenProviderRef.get() match
      case AutoSelection.Provider(provider) =>
        provider
      case AutoSelection.Unset =>
        val cpuAvailability = HoldemCfrNativeRuntime.availability(HoldemCfrNativeRuntime.Backend.Cpu)
        val gpuAvailability = HoldemCfrNativeRuntime.availability(HoldemCfrNativeRuntime.Backend.Gpu)
        val availableNative = Vector(
          if gpuAvailability.available then Some(Provider.NativeGpu) else None,
          if cpuAvailability.available then Some(Provider.NativeCpu) else None
        ).flatten

        val selected =
          if availableNative.isEmpty then Provider.Scala
          else
            val syntheticGame = benchmarkSyntheticGame()
            val benchmarkIterations = configuredAutoBenchmarkIterations
            val benchmarkConfig = CfrSolver.Config(
              iterations = benchmarkIterations,
              cfrPlus = true,
              averagingDelay = math.min(benchmarkIterations / 4, 64),
              linearAveraging = true
            )
            val scalaNanos = benchmarkNanos {
              solveWithScala(syntheticGame, benchmarkConfig)
            }
            val nativeTimings = availableNative.flatMap { provider =>
              provider match
                case Provider.NativeCpu =>
                  benchmarkNativeProvider(
                    provider = provider,
                    solveThunk = solveWithNative(syntheticGame, benchmarkConfig, HoldemCfrNativeRuntime.Backend.Cpu)
                  )
                case Provider.NativeGpu =>
                  benchmarkNativeProvider(
                    provider = provider,
                    solveThunk = solveWithNative(syntheticGame, benchmarkConfig, HoldemCfrNativeRuntime.Backend.Gpu)
                  )
                case Provider.Scala =>
                  None
            }

            if nativeTimings.isEmpty then Provider.Scala
            else
              val (bestNativeProvider, bestNativeNanos) = nativeTimings.minBy(_._2)
              val speedup = scalaNanos.toDouble / bestNativeNanos.toDouble
              if speedup >= configuredAutoMinSpeedup then
                GpuRuntimeSupport.log(
                  f"CFR auto-provider selected ${providerLabel(bestNativeProvider)} " +
                    f"(scala=${scalaNanos / 1e6}%.2fms native=${bestNativeNanos / 1e6}%.2fms speedup=${speedup}%.2fx)"
                )
                bestNativeProvider
              else
                GpuRuntimeSupport.log(
                  f"CFR auto-provider kept scala " +
                    f"(best=${providerLabel(bestNativeProvider)} scala=${scalaNanos / 1e6}%.2fms " +
                    f"bestNative=${bestNativeNanos / 1e6}%.2fms speedup=${speedup}%.2fx)"
                )
                Provider.Scala

        autoChosenProviderRef.compareAndSet(AutoSelection.Unset, AutoSelection.Provider(selected))
        autoChosenProviderRef.get() match
          case AutoSelection.Provider(provider) => provider
          case AutoSelection.Unset => selected

  private[holdem] def resetAutoProviderForTests(): Unit =
    autoChosenProviderRef.set(AutoSelection.Unset)
    equityLookupCache.clear()

  private def benchmarkSyntheticGame(): HoldemDecisionGame =
    val hero = HoleCardsIndex.byIdUnchecked(0)
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 6.0,
      toCall = 2.0,
      position = Position.Button,
      stackSize = 100.0,
      betHistory = Vector.empty
    )
    val villains = Vector(
      HoleCardsIndex.byIdUnchecked(120),
      HoleCardsIndex.byIdUnchecked(240),
      HoleCardsIndex.byIdUnchecked(360),
      HoleCardsIndex.byIdUnchecked(480)
    ).filter(_.isDisjointFrom(hero))
    val villainDistribution =
      if villains.nonEmpty then villains.map(_ -> (1.0 / villains.length.toDouble))
      else Vector(HoleCardsIndex.byIdUnchecked(600) -> 1.0)
    val heroActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(8.0))
    val villainResponseByRaise = Map(8.0 -> Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(16.0)))
    val heroResponseByReraise = Map((8.0, 16.0) -> Vector(PokerAction.Fold, PokerAction.Call))
    val equityByVillain = emptyEquityLookup()
    villainDistribution.foreach { case (villain, _) =>
      updateEquityLookup(equityByVillain, villain, 0.5)
    }

    HoldemDecisionGame(
      hero = hero,
      publicState = state,
      villainDistribution = villainDistribution,
      heroActions = heroActions,
      villainResponseByRaise = villainResponseByRaise,
      heroResponseByReraise = heroResponseByReraise,
      equityByVillain = equityByVillain
    )

  private def benchmarkNanos(thunk: => Any): Long =
    val started = System.nanoTime()
    thunk
    math.max(1L, System.nanoTime() - started)

  private def benchmarkNativeProvider(
      provider: Provider,
      solveThunk: => Option[PolicySolveResult]
  ): Option[(Provider, Long)] =
    val started = System.nanoTime()
    val result = solveThunk
    val elapsed = math.max(1L, System.nanoTime() - started)
    result.map(_ => provider -> elapsed)

  private def configuredAutoBenchmarkIterations: Int =
    GpuRuntimeSupport
      .resolveNonEmpty(CfrAutoBenchmarkIterationsProperty, CfrAutoBenchmarkIterationsEnv)
      .flatMap(_.toIntOption)
      .filter(_ > 0)
      .getOrElse(DefaultAutoBenchmarkIterations)

  private def configuredAutoMinSpeedup: Double =
    GpuRuntimeSupport
      .resolveNonEmpty(CfrAutoMinSpeedupProperty, CfrAutoMinSpeedupEnv)
      .flatMap(_.toDoubleOption)
      .filter(value => value > 1.0 && value.isFinite)
      .getOrElse(DefaultAutoMinSpeedup)

  private def sanitizeHeroActions(
      state: GameState,
      candidateActions: Vector[PokerAction]
  ): Vector[PokerAction] =
    val legal = candidateActions.filter {
      case PokerAction.Fold =>
        state.toCall > 0.0
      case PokerAction.Check =>
        state.toCall <= 0.0
      case PokerAction.Call =>
        state.toCall > 0.0
      case PokerAction.Raise(amount) =>
        amount.isFinite && amount > state.toCall && amount <= state.stackSize
    }.distinct

    if legal.nonEmpty then legal
    else if state.toCall > 0.0 then Vector(PokerAction.Fold, PokerAction.Call)
    else Vector(PokerAction.Check)

  private def trimVillainSupport(
      villainPosterior: DiscreteDistribution[HoleCards],
      deadMask: Long,
      maxVillainHands: Int
  ): Vector[(HoleCards, Double)] =
    val limit = math.max(1, maxVillainHands)
    val orderedSupport = orderedVillainSupportEntries(villainPosterior)
    val selectedHands = new Array[HoleCards](math.min(limit, orderedSupport.length))
    val selectedWeights = new Array[Double](selectedHands.length)
    val selectedIds = new Array[Int](selectedHands.length)
    var selectedSize = 0
    var supportIdx = 0
    while supportIdx < orderedSupport.length && selectedSize < limit do
      val entry = orderedSupport(supportIdx)
      if (entry.mask & deadMask) == 0L then
        selectedHands(selectedSize) = entry.hand
        selectedWeights(selectedSize) = entry.weight
        selectedIds(selectedSize) = entry.handId
        selectedSize += 1
      supportIdx += 1

    require(selectedSize > 0, "villain posterior is empty after dead-card filtering/capping")
    var insertionIdx = 1
    while insertionIdx < selectedSize do
      val hand = selectedHands(insertionIdx)
      val weight = selectedWeights(insertionIdx)
      val handId = selectedIds(insertionIdx)
      var dest = insertionIdx
      while dest > 0 && selectedIds(dest - 1) > handId do
        selectedHands(dest) = selectedHands(dest - 1)
        selectedWeights(dest) = selectedWeights(dest - 1)
        selectedIds(dest) = selectedIds(dest - 1)
        dest -= 1
      selectedHands(dest) = hand
      selectedWeights(dest) = weight
      selectedIds(dest) = handId
      insertionIdx += 1

    var total = 0.0
    var idx = 0
    while idx < selectedSize do
      total += selectedWeights(idx)
      idx += 1
    require(total > Epsilon, "villain posterior has zero retained weight after dead-card filtering/capping")

    val invTotal = 1.0 / total
    val builder = Vector.newBuilder[(HoleCards, Double)]
    idx = 0
    while idx < selectedSize do
      builder += selectedHands(idx) -> (selectedWeights(idx) * invTotal)
      idx += 1
    builder.result()

  private def orderedVillainSupportEntries(
      villainPosterior: DiscreteDistribution[HoleCards]
  ): Array[OrderedVillainSupportEntry] =
    val key = new VillainSupportOrderingCacheKey(villainPosterior)
    val cached = villainSupportOrderingCache.get(key)
    if cached != null then cached
    else
      val computed = villainPosterior.weights.iterator.collect {
        case (hand, weight) if weight > 0.0 =>
          OrderedVillainSupportEntry(
            hand = hand,
            weight = weight,
            handId = holeCardsId(hand),
            mask = cardMask(hand.first) | cardMask(hand.second)
          )
      }.toArray
      java.util.Arrays.sort(
        computed,
        (left: OrderedVillainSupportEntry, right: OrderedVillainSupportEntry) =>
          val byWeight = java.lang.Double.compare(right.weight, left.weight)
          if byWeight != 0 then byWeight
          else java.lang.Integer.compare(left.handId, right.handId)
      )
      if villainSupportOrderingCache.size() >= MaxVillainSupportOrderingCacheEntries then
        villainSupportOrderingCache.clear()
      val existing = villainSupportOrderingCache.putIfAbsent(key, computed)
      if existing != null then existing else computed

  private inline def cardMask(card: sicfun.core.Card): Long =
    1L << CardId.toId(card)

  private inline def holeCardsId(hand: HoleCards): Int =
    val firstId = CardId.toId(hand.first)
    val secondId = CardId.toId(hand.second)
    if firstId < secondId then
      (firstId * (103 - firstId)) / 2 + (secondId - firstId - 1)
    else
      (secondId * (103 - secondId)) / 2 + (firstId - secondId - 1)

  private def deadCardMask(hero: HoleCards, board: Board): Long =
    var mask = cardMask(hero.first) | cardMask(hero.second)
    var idx = 0
    while idx < board.cards.length do
      mask |= cardMask(board.cards(idx))
      idx += 1
    mask

  private def buildVillainResponses(
      state: GameState,
      heroActions: Vector[PokerAction],
      config: HoldemCfrConfig
  ): Map[Double, Vector[PokerAction]] =
    val raises = heroActions.collect { case PokerAction.Raise(amount) => amount }.distinct.sorted
    raises.map { raiseAmount =>
      val responses = ArrayBuffer(PokerAction.Fold, PokerAction.Call)
      if config.includeVillainReraises then
        config.villainReraiseMultipliers.foreach { multiplier =>
          val candidate = roundToHalf(raiseAmount * multiplier)
          if candidate > raiseAmount + Epsilon && candidate <= state.stackSize + Epsilon then
            responses += PokerAction.Raise(candidate)
        }
      raiseAmount -> responses.toVector.distinct
    }.toMap

  private def buildHeroReraiseResponses(
      state: GameState,
      villainResponseByRaise: Map[Double, Vector[PokerAction]]
  ): Map[(Double, Double), Vector[PokerAction]] =
    villainResponseByRaise.toVector.flatMap { case (heroRaise, responses) =>
      responses.collect { case PokerAction.Raise(villainRaise) =>
        val heroRemaining = state.stackSize - heroRaise
        val callAdditional = villainRaise - heroRaise
        val actions =
          if callAdditional <= heroRemaining + Epsilon then Vector(PokerAction.Fold, PokerAction.Call)
          else Vector(PokerAction.Fold)
        (heroRaise, villainRaise) -> actions
      }
    }.toMap

  private def terminalRootActionValue(
      state: GameState,
      action: PokerAction,
      equity: Double
  ): Double =
    action match
      case PokerAction.Fold => 0.0
      case PokerAction.Check => equity * state.pot
      case PokerAction.Call => (equity * (state.pot + state.toCall)) - state.toCall
      case PokerAction.Raise(_) =>
        throw new IllegalArgumentException("terminalRootActionValue only supports non-raise actions")

  private def buildEquityLookup(
      hero: HoleCards,
      board: Board,
      villains: Vector[HoleCards],
      trials: Int,
      preferNativeBatch: Boolean,
      rngSeed: Long
  ): Array[Double] =
    if villains.isEmpty then
      emptyEquityLookup()
    else
      val exactPostflop = exactPostflopEquity(hero, board, villains)
      val exactLookup = exactPostflop.nonEmpty
      val cacheKey = EquityLookupCacheKey(
        heroId = holeCardsId(hero),
        boardPacked = packBoard(board),
        villainIds = villains.map(HoleCardsIndex.idOf),
        trials = if exactLookup then 0 else trials,
        preferNativeBatch = if exactLookup then false else preferNativeBatch,
        rngSeed = if exactLookup then 0L else rngSeed
      )
      val cached = equityLookupCache.get(cacheKey)
      if cached != null then cached
      else
        val maybePreflopBatch =
          if preferNativeBatch && board.size == 0 then
            preflopBatchEquity(hero, villains, trials, rngSeed)
          else None

        val maybePostflopBatch =
          if board.size > 0 && board.size < 5 then postflopBatchEquity(hero, board, villains, trials, rngSeed)
          else None

        val computed =
          exactPostflop.orElse(maybePreflopBatch).orElse(maybePostflopBatch).getOrElse {
            val arr = emptyEquityLookup()
            var idx = 0
            while idx < villains.length do
              val villain = villains(idx)
              val estimate = HoldemEquity.equityMonteCarlo(
                hero = hero,
                board = board,
                villainRange = DiscreteDistribution(Map(villain -> 1.0)),
                trials = trials,
                rng = new Random(mixSeed(rngSeed, idx.toLong + 0x9E3779B97F4A7C15L))
              )
              updateEquityLookup(arr, villain, estimate.mean)
              idx += 1
            arr
          }
        if equityLookupCache.size() >= MaxEquityLookupCacheEntries then equityLookupCache.clear()
        equityLookupCache.putIfAbsent(cacheKey, computed)
        val published = equityLookupCache.get(cacheKey)
        if published != null then published else computed

  private[holdem] def exactPostflopEquity(
      hero: HoleCards,
      board: Board,
      villains: Vector[HoleCards]
  ): Option[Array[Double]] =
    board.size match
      case 5 => Some(riverExactEquity(hero, board, villains))
      case _ => None

  private def riverExactEquity(
      hero: HoleCards,
      board: Board,
      villains: Vector[HoleCards]
  ): Array[Double] =
    val arr = emptyEquityLookup()
    val boardCards = board.cards
    val heroRank = HandEvaluator.evaluate7PackedDirect(
      hero.first,
      hero.second,
      boardCards(0),
      boardCards(1),
      boardCards(2),
      boardCards(3),
      boardCards(4)
    )
    var idx = 0
    while idx < villains.length do
      val villain = villains(idx)
      val villainRank = HandEvaluator.evaluate7PackedDirect(
        villain.first,
        villain.second,
        boardCards(0),
        boardCards(1),
        boardCards(2),
        boardCards(3),
        boardCards(4)
      )
      val equity =
        if heroRank > villainRank then 1.0
        else if heroRank == villainRank then 0.5
        else 0.0
      updateEquityLookup(arr, villain, equity)
      idx += 1
    arr

  private def postflopBatchEquity(
      hero: HoleCards,
      board: Board,
      villains: Vector[HoleCards],
      trials: Int,
      rngSeed: Long
  ): Option[Array[Double]] =
    val villainArray = villains.toArray
    HoldemPostflopNativeRuntime.computePostflopBatch(
      hero = hero,
      board = board,
      villains = villainArray,
      trials = trials,
      seedBase = rngSeed
    ) match
      case Right(values) if values.length == villains.length =>
        val arr = emptyEquityLookup()
        var idx = 0
        while idx < values.length do
          updateEquityLookup(arr, villains(idx), values(idx).equity)
          idx += 1
        Some(arr)
      case _ =>
        None

  private def preflopBatchEquity(
      hero: HoleCards,
      villains: Vector[HoleCards],
      trials: Int,
      rngSeed: Long
  ): Option[Array[Double]] =
    val heroId = holeCardsId(hero)
    val packedKeys = new Array[Long](villains.length)
    val keyMaterial = new Array[Long](villains.length)
    val flipped = new Array[Boolean](villains.length)

    var idx = 0
    while idx < villains.length do
      val villainId = holeCardsId(villains(idx))
      val low = math.min(heroId, villainId)
      val high = math.max(heroId, villainId)
      packedKeys(idx) = HeadsUpEquityTable.pack(low, high)
      keyMaterial(idx) = packedKeys(idx) ^ ((idx.toLong + 1L) << 33)
      flipped(idx) = heroId > villainId
      idx += 1

    HeadsUpGpuRuntime.computeBatch(
      packedKeys = packedKeys,
      keyMaterial = keyMaterial,
      mode = HeadsUpEquityTable.Mode.MonteCarlo(trials),
      monteCarloSeedBase = rngSeed
    ) match
      case Right(values) if values.length == villains.length =>
        val arr = emptyEquityLookup()
        idx = 0
        while idx < values.length do
          val perspective = HeadsUpEquityTable.flipIfNeeded(values(idx), flipped(idx))
          updateEquityLookup(arr, villains(idx), perspective.equity)
          idx += 1
        Some(arr)
      case _ =>
        None

  private def emptyEquityLookup(): Array[Double] =
    val arr = new Array[Double](HoleCardsIndex.size)
    java.util.Arrays.fill(arr, Double.NaN)
    arr

  private def updateEquityLookup(lookup: Array[Double], hand: HoleCards, equity: Double): Unit =
    lookup(HoleCardsIndex.fastIdOf(hand)) = equity

  private def equityLookupOrZero(lookup: Array[Double], hand: HoleCards): Double =
    val eq = lookup(HoleCardsIndex.fastIdOf(hand))
    if eq.isNaN then 0.0 else eq

  private def requireEquityLookup(lookup: Array[Double], hand: HoleCards): Double =
    val eq = lookup(HoleCardsIndex.fastIdOf(hand))
    if eq.isNaN then throw new IllegalStateException(s"missing equity value for villain hand ${hand.toToken}")
    eq

  private def roundToHalf(value: Double): Double =
    math.round(value * 2.0) / 2.0

  private def mixSeed(seed: Long, salt: Long): Long =
    var z = seed ^ salt ^ 0x9E3779B97F4A7C15L
    z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L
    z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL
    z ^ (z >>> 31)

  private def packBoard(board: Board): Long =
    val ids = board.cards.map(CardId.toId).toArray
    java.util.Arrays.sort(ids)
    var packed = ids.length.toLong
    var idx = 0
    while idx < ids.length do
      packed = (packed << 6) | (ids(idx).toLong & 0x3fL)
      idx += 1
    packed

  private final case class HoldemDecisionGame(
      hero: HoleCards,
      publicState: GameState,
      villainDistribution: Vector[(HoleCards, Double)],
      heroActions: Vector[PokerAction],
      villainResponseByRaise: Map[Double, Vector[PokerAction]],
      heroResponseByReraise: Map[(Double, Double), Vector[PokerAction]],
      equityByVillain: Array[Double]
  ) extends CfrSolver.ExtensiveFormGame[HoldemDecisionGame.Node, PokerAction]:
    import HoldemDecisionGame.*

    private val boardToken =
      if publicState.board.cards.isEmpty then "preflop"
      else publicState.board.cards.map(_.toToken).mkString("")
    private val stateToken =
      s"${publicState.street}|$boardToken|${amountKey(publicState.pot)}|${amountKey(publicState.toCall)}|${amountKey(publicState.stackSize)}"

    val heroRootInfoSetKey: String =
      s"hero:${hero.toToken}|root|$stateToken"

    // Pre-computed chance outcomes cached once during construction.
    // Without this, chanceOutcomes creates a new Vector via .map on every
    // CFR iteration (~1500 Vectors of ~96 elements per solve).
    private val cachedRootChanceOutcomes: Vector[(Node, Double)] =
      villainDistribution.map { case (villain, probability) =>
        HeroRoot(villain) -> probability
      }

    // Pre-computed info set keys for villain and reraise nodes. Without this,
    // informationSetKey allocates a new String via interpolation on every node
    // visit (~288K string allocations per solve).
    private val villainInfoSetKeyCache: Map[(HoleCards, Double), String] =
      val builder = Map.newBuilder[(HoleCards, Double), String]
      villainDistribution.foreach { case (villain, _) =>
        villainResponseByRaise.keys.foreach { heroRaise =>
          builder += (villain, heroRaise) ->
            s"villain:${villain.toToken}|vsRaise:${amountKey(heroRaise)}|$stateToken"
        }
      }
      builder.result()

    private val heroReraiseInfoSetKeyCache: Map[(Double, Double), String] =
      val builder = Map.newBuilder[(Double, Double), String]
      heroResponseByReraise.keys.foreach { case (heroRaise, villainRaise) =>
        builder += (heroRaise, villainRaise) ->
          s"hero:${hero.toToken}|vs3bet:${amountKey(heroRaise)}:${amountKey(villainRaise)}|$stateToken"
      }
      builder.result()

    override def root: Node =
      RootChance

    override def actor(state: Node): CfrSolver.Actor =
      state match
        case RootChance              => CfrSolver.Actor.Chance
        case _: HeroRoot             => CfrSolver.Actor.Player0
        case _: VillainFacingRaise   => CfrSolver.Actor.Player1
        case _: HeroFacingReraise    => CfrSolver.Actor.Player0
        case _: Terminal             => CfrSolver.Actor.Terminal

    override def legalActions(state: Node): Vector[PokerAction] =
      state match
        case _: HeroRoot =>
          heroActions
        case VillainFacingRaise(_, heroRaise) =>
          villainResponseByRaise.getOrElse(heroRaise, Vector(PokerAction.Fold, PokerAction.Call))
        case HeroFacingReraise(_, heroRaise, villainRaise) =>
          heroResponseByReraise.getOrElse((heroRaise, villainRaise), Vector(PokerAction.Fold))
        case _ =>
          Vector.empty

    override def informationSetKey(state: Node, player: Int): String =
      state match
        case _: HeroRoot if player == 0 =>
          heroRootInfoSetKey
        case VillainFacingRaise(villain, heroRaise) if player == 1 =>
          villainInfoSetKeyCache.getOrElse(
            (villain, heroRaise),
            s"villain:${villain.toToken}|vsRaise:${amountKey(heroRaise)}|$stateToken"
          )
        case HeroFacingReraise(_, heroRaise, villainRaise) if player == 0 =>
          heroReraiseInfoSetKeyCache.getOrElse(
            (heroRaise, villainRaise),
            s"hero:${hero.toToken}|vs3bet:${amountKey(heroRaise)}:${amountKey(villainRaise)}|$stateToken"
          )
        case _ =>
          throw new IllegalArgumentException("invalid infoset query for node/player")

    override def transition(state: Node, action: PokerAction): Node =
      state match
        case HeroRoot(villain) =>
          action match
            case PokerAction.Fold =>
              Terminal(villain, heroInvestment = 0.0, villainInvestment = 0.0, winnerByFold = Some(1))
            case PokerAction.Check =>
              Terminal(villain, heroInvestment = 0.0, villainInvestment = 0.0, winnerByFold = None)
            case PokerAction.Call =>
              Terminal(villain, heroInvestment = publicState.toCall, villainInvestment = 0.0, winnerByFold = None)
            case PokerAction.Raise(raiseAmount) =>
              VillainFacingRaise(villain, raiseAmount)
        case VillainFacingRaise(villain, heroRaise) =>
          action match
            case PokerAction.Fold =>
              Terminal(villain, heroInvestment = heroRaise, villainInvestment = 0.0, winnerByFold = Some(0))
            case PokerAction.Call =>
              Terminal(villain, heroInvestment = heroRaise, villainInvestment = heroRaise, winnerByFold = None)
            case PokerAction.Raise(villainRaise) =>
              HeroFacingReraise(villain, heroRaise, villainRaise)
            case PokerAction.Check =>
              throw new IllegalArgumentException("villain cannot check facing a raise")
        case HeroFacingReraise(villain, heroRaise, villainRaise) =>
          action match
            case PokerAction.Fold =>
              Terminal(villain, heroInvestment = heroRaise, villainInvestment = villainRaise, winnerByFold = Some(1))
            case PokerAction.Call =>
              Terminal(villain, heroInvestment = villainRaise, villainInvestment = villainRaise, winnerByFold = None)
            case _ =>
              throw new IllegalArgumentException("hero can only fold/call versus 3-bet in this abstraction")
        case RootChance | _: Terminal =>
          throw new IllegalArgumentException("transition requested from non-action node")

    override def chanceOutcomes(state: Node): Vector[(Node, Double)] =
      state match
        case RootChance =>
          cachedRootChanceOutcomes
        case _ =>
          Vector.empty

    override def terminalUtilityPlayer0(state: Node): Double =
      state match
        case Terminal(villain, heroInvestment, villainInvestment, winnerByFold) =>
          val equity =
            if winnerByFold.isEmpty then
              requireEquityLookup(equityByVillain, villain)
            else 0.0
          terminalUtilityForEquity(equity, heroInvestment, villainInvestment, winnerByFold)
        case _ =>
          throw new IllegalArgumentException("terminalUtilityPlayer0 called on non-terminal node")

    private def terminalUtilityForEquity(
        equity: Double,
        heroInvestment: Double,
        villainInvestment: Double,
        winnerByFold: Option[Int]
    ): Double =
      winnerByFold match
        case Some(0) =>
          publicState.pot + villainInvestment
        case Some(1) =>
          -heroInvestment
        case Some(other) =>
          throw new IllegalStateException(s"invalid fold winner marker: $other")
        case None =>
          val finalPot = publicState.pot + heroInvestment + villainInvestment
          (equity * finalPot) - heroInvestment

    def toNativeTreeSpec: HoldemCfrNativeRuntime.NativeTreeSpec =
      val villainCount = villainDistribution.length
      require(villainCount > 0, "native tree spec missing villain support")

      val villainProbabilities = new Array[Double](villainCount)
      val villainTokens = new Array[String](villainCount)
      val villainEquities = new Array[Double](villainCount)
      var villainIdx = 0
      while villainIdx < villainCount do
        val (villain, probability) = villainDistribution(villainIdx)
        villainProbabilities(villainIdx) = probability
        villainTokens(villainIdx) = villain.toToken
        villainEquities(villainIdx) = requireEquityLookup(equityByVillain, villain)
        villainIdx += 1

      val heroActionCount = heroActions.length
      val heroRaiseSlotByAction = Array.fill(heroActionCount)(-1)
      val heroNonRaiseOrdinalByAction = Array.fill(heroActionCount)(-1)
      val heroRaiseAmounts = ArrayBuffer.empty[Double]
      var heroNonRaiseCount = 0
      var actionIdx = 0
      while actionIdx < heroActionCount do
        heroActions(actionIdx) match
          case PokerAction.Raise(amount) =>
            heroRaiseSlotByAction(actionIdx) = heroRaiseAmounts.length
            heroRaiseAmounts += amount
          case _ =>
            heroNonRaiseOrdinalByAction(actionIdx) = heroNonRaiseCount
            heroNonRaiseCount += 1
        actionIdx += 1

      val raiseCount = heroRaiseAmounts.length
      val villainActionsByRaiseSlot = new Array[Vector[PokerAction]](raiseCount)
      val pairIndexByRaiseAction = Array.ofDim[Array[Int]](raiseCount)
      val villainTerminalOrdinalByRaiseAction = Array.ofDim[Array[Int]](raiseCount)
      val villainTerminalOffsetByRaiseSlot = new Array[Int](raiseCount)
      val pairHeroRaises = ArrayBuffer.empty[Double]
      val pairVillainRaises = ArrayBuffer.empty[Double]
      val heroActionsByPair = ArrayBuffer.empty[Vector[PokerAction]]
      var villainResponseEdgesPerVillain = 0
      var villainTerminalsPerVillain = 0
      var raiseSlot = 0
      while raiseSlot < raiseCount do
        val heroRaise = heroRaiseAmounts(raiseSlot)
        val actions = villainResponseByRaise.getOrElse(heroRaise, Vector(PokerAction.Fold, PokerAction.Call))
        villainActionsByRaiseSlot(raiseSlot) = actions
        villainTerminalOffsetByRaiseSlot(raiseSlot) = villainTerminalsPerVillain
        villainResponseEdgesPerVillain += actions.length
        val pairIndices = Array.fill(actions.length)(-1)
        val terminalOrdinals = Array.fill(actions.length)(-1)
        var responseTerminalCount = 0
        var responseIdx = 0
        while responseIdx < actions.length do
          actions(responseIdx) match
            case PokerAction.Raise(villainRaise) =>
              pairIndices(responseIdx) = pairHeroRaises.length
              pairHeroRaises += heroRaise
              pairVillainRaises += villainRaise
              heroActionsByPair += heroResponseByReraise.getOrElse((heroRaise, villainRaise), Vector(PokerAction.Fold))
            case _ =>
              terminalOrdinals(responseIdx) = responseTerminalCount
              responseTerminalCount += 1
          responseIdx += 1
        pairIndexByRaiseAction(raiseSlot) = pairIndices
        villainTerminalOrdinalByRaiseAction(raiseSlot) = terminalOrdinals
        villainTerminalsPerVillain += responseTerminalCount
        raiseSlot += 1

      val pairCount = pairHeroRaises.length
      val heroActionsByPairArray = heroActionsByPair.toArray
      val heroTerminalOffsetByPair = new Array[Int](pairCount)
      var heroReraiseEdgesPerVillain = 0
      var heroReraiseTerminalsPerVillain = 0
      var pairIdx = 0
      while pairIdx < pairCount do
        val actions = heroActionsByPairArray(pairIdx)
        heroTerminalOffsetByPair(pairIdx) = heroReraiseTerminalsPerVillain
        heroReraiseEdgesPerVillain += actions.length
        heroReraiseTerminalsPerVillain += actions.length
        pairIdx += 1

      val rootNodeId = 0
      val heroRootBase = 1
      val villainFacingBase = heroRootBase + villainCount
      val heroFacingBase = villainFacingBase + (villainCount * raiseCount)
      val rootTerminalBase = heroFacingBase + (villainCount * pairCount)
      val villainTerminalBase = rootTerminalBase + (villainCount * heroNonRaiseCount)
      val heroReraiseTerminalBase = villainTerminalBase + (villainCount * villainTerminalsPerVillain)
      val nodeCount = heroReraiseTerminalBase + (villainCount * heroReraiseTerminalsPerVillain)
      val edgeCount =
        villainCount +
          (villainCount * heroActionCount) +
          (villainCount * villainResponseEdgesPerVillain) +
          (villainCount * heroReraiseEdgesPerVillain)

      val nodeTypes = new Array[Int](nodeCount)
      val nodeStarts = new Array[Int](nodeCount)
      val nodeCounts = new Array[Int](nodeCount)
      val nodeInfosets = Array.fill(nodeCount)(-1)
      val edgeChildIds = new Array[Int](edgeCount)
      val edgeProbabilities = new Array[Double](edgeCount)
      val terminalUtilities = new Array[Double](nodeCount)

      val heroToken = hero.toToken
      val infosetCount = 1 + (villainCount * raiseCount) + pairCount
      val infosetKeys = new Array[String](infosetCount)
      val infosetPlayers = new Array[Int](infosetCount)
      val infosetActions = new Array[Vector[PokerAction]](infosetCount)
      val infosetActionCounts = new Array[Int](infosetCount)

      infosetKeys(0) = heroRootInfoSetKey
      infosetPlayers(0) = 0
      infosetActions(0) = heroActions
      infosetActionCounts(0) = heroActionCount

      val villainInfoSetBase = 1
      villainIdx = 0
      while villainIdx < villainCount do
        raiseSlot = 0
        while raiseSlot < raiseCount do
          val infosetIdx = villainInfoSetBase + (villainIdx * raiseCount) + raiseSlot
          val actions = villainActionsByRaiseSlot(raiseSlot)
          infosetKeys(infosetIdx) =
            s"villain:${villainTokens(villainIdx)}|vsRaise:${amountKey(heroRaiseAmounts(raiseSlot))}|$stateToken"
          infosetPlayers(infosetIdx) = 1
          infosetActions(infosetIdx) = actions
          infosetActionCounts(infosetIdx) = actions.length
          raiseSlot += 1
        villainIdx += 1

      val heroReraiseInfoSetBase = villainInfoSetBase + (villainCount * raiseCount)
      pairIdx = 0
      while pairIdx < pairCount do
        val actions = heroActionsByPairArray(pairIdx)
        val infosetIdx = heroReraiseInfoSetBase + pairIdx
        infosetKeys(infosetIdx) =
          s"hero:${heroToken}|vs3bet:${amountKey(pairHeroRaises(pairIdx))}:${amountKey(pairVillainRaises(pairIdx))}|$stateToken"
        infosetPlayers(infosetIdx) = 0
        infosetActions(infosetIdx) = actions
        infosetActionCounts(infosetIdx) = actions.length
        pairIdx += 1

      var edgeCursor = 0
      nodeTypes(rootNodeId) = 1
      nodeStarts(rootNodeId) = edgeCursor
      nodeCounts(rootNodeId) = villainCount
      villainIdx = 0
      while villainIdx < villainCount do
        edgeChildIds(edgeCursor) = heroRootBase + villainIdx
        edgeProbabilities(edgeCursor) = villainProbabilities(villainIdx)
        edgeCursor += 1
        villainIdx += 1

      villainIdx = 0
      while villainIdx < villainCount do
        val heroRootNodeId = heroRootBase + villainIdx
        nodeTypes(heroRootNodeId) = 2
        nodeInfosets(heroRootNodeId) = 0
        nodeStarts(heroRootNodeId) = edgeCursor
        nodeCounts(heroRootNodeId) = heroActionCount
        actionIdx = 0
        while actionIdx < heroActionCount do
          val childId =
            heroActions(actionIdx) match
              case PokerAction.Raise(_) =>
                villainFacingBase + (villainIdx * raiseCount) + heroRaiseSlotByAction(actionIdx)
              case _ =>
                rootTerminalBase + (villainIdx * heroNonRaiseCount) + heroNonRaiseOrdinalByAction(actionIdx)
          edgeChildIds(edgeCursor) = childId
          edgeProbabilities(edgeCursor) = 0.0
          edgeCursor += 1
          actionIdx += 1

        raiseSlot = 0
        while raiseSlot < raiseCount do
          val actions = villainActionsByRaiseSlot(raiseSlot)
          val nodeId = villainFacingBase + (villainIdx * raiseCount) + raiseSlot
          nodeTypes(nodeId) = 3
          nodeInfosets(nodeId) = villainInfoSetBase + (villainIdx * raiseCount) + raiseSlot
          nodeStarts(nodeId) = edgeCursor
          nodeCounts(nodeId) = actions.length
          var responseIdx = 0
          while responseIdx < actions.length do
            val childId =
              actions(responseIdx) match
                case PokerAction.Raise(_) =>
                  heroFacingBase + (villainIdx * pairCount) + pairIndexByRaiseAction(raiseSlot)(responseIdx)
                case _ =>
                  villainTerminalBase +
                    (villainIdx * villainTerminalsPerVillain) +
                    villainTerminalOffsetByRaiseSlot(raiseSlot) +
                    villainTerminalOrdinalByRaiseAction(raiseSlot)(responseIdx)
            edgeChildIds(edgeCursor) = childId
            edgeProbabilities(edgeCursor) = 0.0
            edgeCursor += 1
            responseIdx += 1
          raiseSlot += 1

        pairIdx = 0
        while pairIdx < pairCount do
          val actions = heroActionsByPairArray(pairIdx)
          val nodeId = heroFacingBase + (villainIdx * pairCount) + pairIdx
          nodeTypes(nodeId) = 2
          nodeInfosets(nodeId) = heroReraiseInfoSetBase + pairIdx
          nodeStarts(nodeId) = edgeCursor
          nodeCounts(nodeId) = actions.length
          actionIdx = 0
          while actionIdx < actions.length do
            edgeChildIds(edgeCursor) =
              heroReraiseTerminalBase +
                (villainIdx * heroReraiseTerminalsPerVillain) +
                heroTerminalOffsetByPair(pairIdx) +
                actionIdx
            edgeProbabilities(edgeCursor) = 0.0
            edgeCursor += 1
            actionIdx += 1
          pairIdx += 1
        villainIdx += 1

      require(edgeCursor == edgeCount, s"native tree spec edge count mismatch: $edgeCursor != $edgeCount")
      val terminalStart = edgeCount

      villainIdx = 0
      while villainIdx < villainCount do
        val equity = villainEquities(villainIdx)

        actionIdx = 0
        while actionIdx < heroActionCount do
          heroActions(actionIdx) match
            case PokerAction.Raise(_) => ()
            case action =>
              val nodeId = rootTerminalBase + (villainIdx * heroNonRaiseCount) + heroNonRaiseOrdinalByAction(actionIdx)
              nodeTypes(nodeId) = 0
              nodeStarts(nodeId) = terminalStart
              nodeCounts(nodeId) = 0
              terminalUtilities(nodeId) = terminalRootActionValue(publicState, action, equity)
          actionIdx += 1

        raiseSlot = 0
        while raiseSlot < raiseCount do
          val actions = villainActionsByRaiseSlot(raiseSlot)
          val heroRaise = heroRaiseAmounts(raiseSlot)
          var responseIdx = 0
          while responseIdx < actions.length do
            actions(responseIdx) match
              case PokerAction.Raise(_) => ()
              case PokerAction.Fold =>
                val nodeId =
                  villainTerminalBase +
                    (villainIdx * villainTerminalsPerVillain) +
                    villainTerminalOffsetByRaiseSlot(raiseSlot) +
                    villainTerminalOrdinalByRaiseAction(raiseSlot)(responseIdx)
                nodeTypes(nodeId) = 0
                nodeStarts(nodeId) = terminalStart
                nodeCounts(nodeId) = 0
                terminalUtilities(nodeId) = terminalUtilityForEquity(
                  equity = equity,
                  heroInvestment = heroRaise,
                  villainInvestment = 0.0,
                  winnerByFold = Some(0)
                )
              case PokerAction.Call =>
                val nodeId =
                  villainTerminalBase +
                    (villainIdx * villainTerminalsPerVillain) +
                    villainTerminalOffsetByRaiseSlot(raiseSlot) +
                    villainTerminalOrdinalByRaiseAction(raiseSlot)(responseIdx)
                nodeTypes(nodeId) = 0
                nodeStarts(nodeId) = terminalStart
                nodeCounts(nodeId) = 0
                terminalUtilities(nodeId) = terminalUtilityForEquity(
                  equity = equity,
                  heroInvestment = heroRaise,
                  villainInvestment = heroRaise,
                  winnerByFold = None
                )
              case PokerAction.Check =>
                throw new IllegalArgumentException("villain cannot check facing a raise")
            responseIdx += 1
          raiseSlot += 1

        pairIdx = 0
        while pairIdx < pairCount do
          val actions = heroActionsByPairArray(pairIdx)
          val heroRaise = pairHeroRaises(pairIdx)
          val villainRaise = pairVillainRaises(pairIdx)
          actionIdx = 0
          while actionIdx < actions.length do
            val nodeId =
              heroReraiseTerminalBase +
                (villainIdx * heroReraiseTerminalsPerVillain) +
                heroTerminalOffsetByPair(pairIdx) +
                actionIdx
            nodeTypes(nodeId) = 0
            nodeStarts(nodeId) = terminalStart
            nodeCounts(nodeId) = 0
            actions(actionIdx) match
              case PokerAction.Fold =>
                terminalUtilities(nodeId) = terminalUtilityForEquity(
                  equity = equity,
                  heroInvestment = heroRaise,
                  villainInvestment = villainRaise,
                  winnerByFold = Some(1)
                )
              case PokerAction.Call =>
                terminalUtilities(nodeId) = terminalUtilityForEquity(
                  equity = equity,
                  heroInvestment = villainRaise,
                  villainInvestment = villainRaise,
                  winnerByFold = None
                )
              case other =>
                throw new IllegalArgumentException(s"invalid hero response against 3-bet: $other")
            actionIdx += 1
          pairIdx += 1
        villainIdx += 1

      HoldemCfrNativeRuntime.NativeTreeSpec(
        rootNodeId = rootNodeId,
        rootInfoSetIndex = 0,
        nodeTypes = nodeTypes,
        nodeStarts = nodeStarts,
        nodeCounts = nodeCounts,
        nodeInfosets = nodeInfosets,
        edgeChildIds = edgeChildIds,
        edgeProbabilities = edgeProbabilities,
        terminalUtilities = terminalUtilities,
        infosetKeys = infosetKeys.toVector,
        infosetPlayers = infosetPlayers,
        infosetActions = infosetActions.toVector,
        infosetActionCounts = infosetActionCounts
      )

    def evaluateRootAction(
        heroAction: PokerAction,
        averagePolicy: Map[String, Map[PokerAction, Double]]
    ): Double =
      val forcedAction = heroActions.find(_ == heroAction).getOrElse(
        throw new IllegalArgumentException(s"hero action $heroAction is not in root action set")
      )
      var value = 0.0
      var idx = 0
      while idx < villainDistribution.length do
        val (villain, probability) = villainDistribution(idx)
        val node = transition(HeroRoot(villain), forcedAction)
        value += probability * evaluateNode(node, averagePolicy)
        idx += 1
      value

    def evaluateAveragePolicy(
        averagePolicy: Map[String, Map[PokerAction, Double]]
    ): Double =
      evaluateNode(root, averagePolicy)

    def evaluateWithVillainBestResponse(
        averagePolicy: Map[String, Map[PokerAction, Double]]
    ): Double =
      evaluateNodeWithVillainBestResponse(root, averagePolicy)

    private def evaluateNode(
        node: Node,
        averagePolicy: Map[String, Map[PokerAction, Double]]
    ): Double =
      actor(node) match
        case CfrSolver.Actor.Terminal =>
          terminalUtilityPlayer0(node)
        case CfrSolver.Actor.Chance =>
          val outcomes = chanceOutcomes(node)
          val normalized = normalize(outcomes.map(_._2))
          var value = 0.0
          var idx = 0
          while idx < outcomes.length do
            val (nextNode, _) = outcomes(idx)
            value += normalized(idx) * evaluateNode(nextNode, averagePolicy)
            idx += 1
          value
        case CfrSolver.Actor.Player0 =>
          val actions = legalActions(node)
          val key = informationSetKey(node, player = 0)
          expectedFromPolicy(node, actions, key, averagePolicy)
        case CfrSolver.Actor.Player1 =>
          val actions = legalActions(node)
          val key = informationSetKey(node, player = 1)
          expectedFromPolicy(node, actions, key, averagePolicy)

    private def evaluateNodeWithVillainBestResponse(
        node: Node,
        averagePolicy: Map[String, Map[PokerAction, Double]]
    ): Double =
      actor(node) match
        case CfrSolver.Actor.Terminal =>
          terminalUtilityPlayer0(node)
        case CfrSolver.Actor.Chance =>
          val outcomes = chanceOutcomes(node)
          val normalized = normalize(outcomes.map(_._2))
          var value = 0.0
          var idx = 0
          while idx < outcomes.length do
            val (nextNode, _) = outcomes(idx)
            value += normalized(idx) * evaluateNodeWithVillainBestResponse(nextNode, averagePolicy)
            idx += 1
          value
        case CfrSolver.Actor.Player0 =>
          val actions = legalActions(node)
          val key = informationSetKey(node, player = 0)
          expectedFromPolicyVsVillainBestResponse(node, actions, key, averagePolicy)
        case CfrSolver.Actor.Player1 =>
          val actions = legalActions(node)
          require(actions.nonEmpty, "villain best-response node must have legal actions")
          var best = Double.PositiveInfinity
          var idx = 0
          while idx < actions.length do
            val nextNode = transition(node, actions(idx))
            val value = evaluateNodeWithVillainBestResponse(nextNode, averagePolicy)
            if value < best then best = value
            idx += 1
          best

    private def expectedFromPolicy(
        node: Node,
        actions: Vector[PokerAction],
        infosetKey: String,
        averagePolicy: Map[String, Map[PokerAction, Double]]
    ): Double =
      val policy = averagePolicy.get(infosetKey)
      val probs =
        policy match
          case Some(p) =>
            val raw = actions.map(a => p.getOrElse(a, 0.0))
            val total = raw.sum
            if total > 0.0 then raw.map(_ / total)
            else Vector.fill(actions.length)(1.0 / actions.length.toDouble)
          case None =>
            Vector.fill(actions.length)(1.0 / actions.length.toDouble)

      var value = 0.0
      var idx = 0
      while idx < actions.length do
        val nextNode = transition(node, actions(idx))
        value += probs(idx) * evaluateNode(nextNode, averagePolicy)
        idx += 1
      value

    private def expectedFromPolicyVsVillainBestResponse(
        node: Node,
        actions: Vector[PokerAction],
        infosetKey: String,
        averagePolicy: Map[String, Map[PokerAction, Double]]
    ): Double =
      val policy = averagePolicy.get(infosetKey)
      val probs =
        policy match
          case Some(p) =>
            val raw = actions.map(a => p.getOrElse(a, 0.0))
            val total = raw.sum
            if total > 0.0 then raw.map(_ / total)
            else Vector.fill(actions.length)(1.0 / actions.length.toDouble)
          case None =>
            Vector.fill(actions.length)(1.0 / actions.length.toDouble)

      var value = 0.0
      var idx = 0
      while idx < actions.length do
        val nextNode = transition(node, actions(idx))
        value += probs(idx) * evaluateNodeWithVillainBestResponse(nextNode, averagePolicy)
        idx += 1
      value

    private def amountKey(value: Double): String =
      f"$value%.3f"

    private def normalize(probabilities: Vector[Double]): Vector[Double] =
      val total = probabilities.sum
      if total <= 0.0 then Vector.fill(probabilities.length)(1.0 / probabilities.length.toDouble)
      else probabilities.map(_ / total)

  private object HoldemDecisionGame:
    sealed trait NativeInfoSetKey
    case object NativeHeroRootInfoSet extends NativeInfoSetKey
    final case class NativeVillainFacingRaiseInfoSet(villainId: Int, heroRaiseBits: Long) extends NativeInfoSetKey
    final case class NativeHeroFacingReraiseInfoSet(heroRaiseBits: Long, villainRaiseBits: Long) extends NativeInfoSetKey

    sealed trait Node
    case object RootChance extends Node
    final case class HeroRoot(villain: HoleCards) extends Node
    final case class VillainFacingRaise(villain: HoleCards, heroRaise: Double) extends Node
    final case class HeroFacingReraise(villain: HoleCards, heroRaise: Double, villainRaise: Double) extends Node
    final case class Terminal(
        villain: HoleCards,
        heroInvestment: Double,
        villainInvestment: Double,
        winnerByFold: Option[Int]
    ) extends Node

    def nativeInfoSetKey(node: Node, player: Int): NativeInfoSetKey =
      node match
        case _: HeroRoot if player == 0 =>
          NativeHeroRootInfoSet
        case VillainFacingRaise(villain, heroRaise) if player == 1 =>
          NativeVillainFacingRaiseInfoSet(
            villainId = holeCardsId(villain),
            heroRaiseBits = java.lang.Double.doubleToLongBits(heroRaise)
          )
        case HeroFacingReraise(_, heroRaise, villainRaise) if player == 0 =>
          NativeHeroFacingReraiseInfoSet(
            heroRaiseBits = java.lang.Double.doubleToLongBits(heroRaise),
            villainRaiseBits = java.lang.Double.doubleToLongBits(villainRaise)
          )
        case _ =>
          throw new IllegalArgumentException("invalid native infoset query for node/player")
