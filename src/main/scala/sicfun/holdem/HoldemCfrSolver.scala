package sicfun.holdem

import sicfun.core.DiscreteDistribution

import java.util.concurrent.atomic.AtomicReference
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable
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
  private val autoChosenProviderRef = new AtomicReference[AutoSelection](AutoSelection.Unset)

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

  def solve(
      hero: HoleCards,
      state: GameState,
      villainPosterior: DiscreteDistribution[HoleCards],
      candidateActions: Vector[PokerAction],
      config: HoldemCfrConfig = HoldemCfrConfig()
  ): HoldemCfrSolution =
    require(candidateActions.nonEmpty, "candidateActions must be non-empty")

    val heroActions = sanitizeHeroActions(state, candidateActions)
    require(heroActions.nonEmpty, "no legal hero actions after sanitization")

    val dead = hero.asSet ++ state.board.asSet
    val trimmedVillain = trimVillainDistribution(villainPosterior, dead, config.maxVillainHands)
    val villainSupport = trimmedVillain.weights.toVector.sortBy { case (hand, _) => hand.toToken }

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
    val equityByVillain = villainDistribution.map { case (villain, _) =>
      villain -> 0.5
    }.toMap

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

  private def trimVillainDistribution(
      villainPosterior: DiscreteDistribution[HoleCards],
      dead: Set[sicfun.core.Card],
      maxVillainHands: Int
  ): DiscreteDistribution[HoleCards] =
    val sorted = villainPosterior.weights.toVector
      .filter { case (hand, weight) =>
        weight > 0.0 && !hand.asSet.exists(dead.contains)
      }
      .sortBy { case (hand, weight) => (-weight, hand.toToken) }
      .take(maxVillainHands)

    require(sorted.nonEmpty, "villain posterior is empty after dead-card filtering/capping")
    DiscreteDistribution(sorted.toMap).normalized

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

  private def buildEquityLookup(
      hero: HoleCards,
      board: Board,
      villains: Vector[HoleCards],
      trials: Int,
      preferNativeBatch: Boolean,
      rngSeed: Long
  ): Map[HoleCards, Double] =
    if villains.isEmpty then Map.empty
    else
      val maybeBatch =
        if preferNativeBatch && board.size == 0 then
          preflopBatchEquity(hero, villains, trials, rngSeed)
        else None

      maybeBatch.getOrElse {
        villains.zipWithIndex.map { case (villain, idx) =>
          val estimate = HoldemEquity.equityMonteCarlo(
            hero = hero,
            board = board,
            villainRange = DiscreteDistribution(Map(villain -> 1.0)),
            trials = trials,
            rng = new Random(mixSeed(rngSeed, idx.toLong + 0x9E3779B97F4A7C15L))
          )
          villain -> estimate.mean
        }.toMap
      }

  private def preflopBatchEquity(
      hero: HoleCards,
      villains: Vector[HoleCards],
      trials: Int,
      rngSeed: Long
  ): Option[Map[HoleCards, Double]] =
    val heroId = HoleCardsIndex.idOf(hero)
    val packedKeys = new Array[Long](villains.length)
    val keyMaterial = new Array[Long](villains.length)
    val flipped = new Array[Boolean](villains.length)

    var idx = 0
    while idx < villains.length do
      val villainId = HoleCardsIndex.idOf(villains(idx))
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
        val builder = Map.newBuilder[HoleCards, Double]
        idx = 0
        while idx < values.length do
          val perspective = HeadsUpEquityTable.flipIfNeeded(values(idx), flipped(idx))
          builder += villains(idx) -> perspective.equity
          idx += 1
        Some(builder.result())
      case _ =>
        None

  private def roundToHalf(value: Double): Double =
    math.round(value * 2.0) / 2.0

  private def mixSeed(seed: Long, salt: Long): Long =
    var z = seed ^ salt ^ 0x9E3779B97F4A7C15L
    z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L
    z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL
    z ^ (z >>> 31)

  private final case class HoldemDecisionGame(
      hero: HoleCards,
      publicState: GameState,
      villainDistribution: Vector[(HoleCards, Double)],
      heroActions: Vector[PokerAction],
      villainResponseByRaise: Map[Double, Vector[PokerAction]],
      heroResponseByReraise: Map[(Double, Double), Vector[PokerAction]],
      equityByVillain: Map[HoleCards, Double]
  ) extends CfrSolver.ExtensiveFormGame[HoldemDecisionGame.Node, PokerAction]:
    import HoldemDecisionGame.*

    private val boardToken =
      if publicState.board.cards.isEmpty then "preflop"
      else publicState.board.cards.map(_.toToken).mkString("")
    private val stateToken =
      s"${publicState.street}|$boardToken|${amountKey(publicState.pot)}|${amountKey(publicState.toCall)}|${amountKey(publicState.stackSize)}"

    val heroRootInfoSetKey: String =
      s"hero:${hero.toToken}|root|$stateToken"

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
          s"villain:${villain.toToken}|vsRaise:${amountKey(heroRaise)}|$stateToken"
        case HeroFacingReraise(_, heroRaise, villainRaise) if player == 0 =>
          s"hero:${hero.toToken}|vs3bet:${amountKey(heroRaise)}:${amountKey(villainRaise)}|$stateToken"
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
          villainDistribution.map { case (villain, probability) =>
            HeroRoot(villain) -> probability
          }
        case _ =>
          Vector.empty

    override def terminalUtilityPlayer0(state: Node): Double =
      state match
        case Terminal(villain, heroInvestment, villainInvestment, winnerByFold) =>
          winnerByFold match
            case Some(0) =>
              publicState.pot + villainInvestment
            case Some(1) =>
              -heroInvestment
            case Some(other) =>
              throw new IllegalStateException(s"invalid fold winner marker: $other")
            case None =>
              val equity = equityByVillain.getOrElse(
                villain,
                throw new IllegalStateException(s"missing equity value for villain hand ${villain.toToken}")
              )
              val finalPot = publicState.pot + heroInvestment + villainInvestment
              (equity * finalPot) - heroInvestment
        case _ =>
          throw new IllegalArgumentException("terminalUtilityPlayer0 called on non-terminal node")

    def toNativeTreeSpec: HoldemCfrNativeRuntime.NativeTreeSpec =
      val nodeById = ArrayBuffer.empty[Node]
      val nodeIdByNode = mutable.HashMap.empty[Node, Int]

      def nodeId(node: Node): Int =
        nodeIdByNode.getOrElseUpdate(
          node, {
            val id = nodeById.length
            nodeById += node
            id
          }
        )

      nodeId(root)

      val infosetIndexByKey = mutable.HashMap.empty[String, Int]
      val infosetKeys = ArrayBuffer.empty[String]
      val infosetPlayers = ArrayBuffer.empty[Int]
      val infosetActions = ArrayBuffer.empty[Vector[PokerAction]]

      val nodeTypes = ArrayBuffer.empty[Int]
      val nodeInfosets = ArrayBuffer.empty[Int]
      val nodeChildren = ArrayBuffer.empty[Vector[Int]]
      val nodeEdgeProbabilities = ArrayBuffer.empty[Vector[Double]]
      val terminalUtilities = ArrayBuffer.empty[Double]

      var idx = 0
      while idx < nodeById.length do
        val node = nodeById(idx)
        actor(node) match
          case CfrSolver.Actor.Terminal =>
            nodeTypes += 0
            nodeInfosets += -1
            nodeChildren += Vector.empty
            nodeEdgeProbabilities += Vector.empty
            terminalUtilities += terminalUtilityPlayer0(node)
          case CfrSolver.Actor.Chance =>
            val outcomes = chanceOutcomes(node)
            val childIds = outcomes.map { case (child, _) =>
              nodeId(child)
            }
            nodeTypes += 1
            nodeInfosets += -1
            nodeChildren += childIds
            nodeEdgeProbabilities += outcomes.map(_._2)
            terminalUtilities += 0.0
          case CfrSolver.Actor.Player0 | CfrSolver.Actor.Player1 =>
            val player = if actor(node) == CfrSolver.Actor.Player0 then 0 else 1
            val actions = legalActions(node)
            val key = informationSetKey(node, player)
            val infosetIndex =
              infosetIndexByKey.getOrElseUpdate(
                key, {
                  val created = infosetKeys.length
                  infosetKeys += key
                  infosetPlayers += player
                  infosetActions += actions
                  created
                }
              )
            require(infosetPlayers(infosetIndex) == player, s"inconsistent infoset player for key '$key'")
            require(infosetActions(infosetIndex) == actions, s"inconsistent infoset actions for key '$key'")
            val children = actions.map(action => nodeId(transition(node, action)))
            nodeTypes += (if player == 0 then 2 else 3)
            nodeInfosets += infosetIndex
            nodeChildren += children
            nodeEdgeProbabilities += Vector.fill(actions.length)(0.0)
            terminalUtilities += 0.0
        idx += 1

      val nodeStarts = new Array[Int](nodeById.length)
      val nodeCounts = new Array[Int](nodeById.length)
      val edgeChildIds = ArrayBuffer.empty[Int]
      val edgeProbabilities = ArrayBuffer.empty[Double]

      idx = 0
      while idx < nodeById.length do
        nodeStarts(idx) = edgeChildIds.length
        nodeCounts(idx) = nodeChildren(idx).length
        edgeChildIds ++= nodeChildren(idx)
        edgeProbabilities ++= nodeEdgeProbabilities(idx)
        idx += 1

      HoldemCfrNativeRuntime.NativeTreeSpec(
        rootNodeId = nodeIdByNode(root),
        nodeTypes = nodeTypes.toArray,
        nodeStarts = nodeStarts,
        nodeCounts = nodeCounts,
        nodeInfosets = nodeInfosets.toArray,
        edgeChildIds = edgeChildIds.toArray,
        edgeProbabilities = edgeProbabilities.toArray,
        terminalUtilities = terminalUtilities.toArray,
        infosetKeys = infosetKeys.toVector,
        infosetPlayers = infosetPlayers.toArray,
        infosetActions = infosetActions.toVector,
        infosetActionCounts = infosetActions.map(_.length).toArray
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
