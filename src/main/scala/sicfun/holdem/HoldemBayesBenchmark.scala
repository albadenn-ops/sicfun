package sicfun.holdem

import sicfun.core.Card

import scala.util.Random

/** Benchmark harness for Bayesian inference hotspots and provider selection. */
object HoldemBayesBenchmark:
  private final case class Config(
      warmupRuns: Int = 2,
      measureRuns: Int = 8,
      bunchingTrials: Int = 1_200,
      equityTrials: Int = 8_000,
      provider: String = "auto",
      seed: Long = 17L
  )

  private final case class Sample(stage: String, elapsedNanos: Long):
    def elapsedMs: Double = elapsedNanos.toDouble / 1_000_000.0

  private final case class Stats(
      count: Int,
      meanMs: Double,
      medianMs: Double,
      minMs: Double,
      maxMs: Double
  )

  private final case class BenchmarkContext(
      hero: HoleCards,
      state: GameState,
      folds: Vector[PreflopFold],
      tableRanges: TableRanges,
      villainPos: Position,
      observations: Seq[VillainObservation],
      actionModel: PokerActionModel,
      candidateActions: Vector[PokerAction],
      baselinePrior: sicfun.core.DiscreteDistribution[HoleCards]
  )

  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    require(config.warmupRuns >= 0, "warmupRuns must be non-negative")
    require(config.measureRuns > 0, "measureRuns must be positive")
    require(config.bunchingTrials > 0, "bunchingTrials must be positive")
    require(config.equityTrials > 0, "equityTrials must be positive")

    System.setProperty("sicfun.bayes.provider", config.provider)
    HoldemBayesNativeRuntime.resetLoadCacheForTests()
    HoldemBayesProvider.resetAutoProviderForTests()
    RangeInferenceEngine.clearPosteriorCache()

    val context = buildContext(config.seed)

    println("=== Holdem Bayesian Benchmark ===")
    println(
      s"config: warmupRuns=${config.warmupRuns}, measureRuns=${config.measureRuns}, " +
        s"bunchingTrials=${config.bunchingTrials}, equityTrials=${config.equityTrials}, " +
        s"provider=${config.provider}, seed=${config.seed}"
    )
    println(s"resolvedBayesProvider: ${HoldemBayesProvider.currentProviderLabel}")

    val warmupRng = new Random(config.seed ^ 0x9E3779B97F4A7C15L)
    var warmup = 0
    while warmup < config.warmupRuns do
      runAllStages(context, config, warmupRng)
      warmup += 1

    val measureRng = new Random(config.seed ^ 0xC2B2AE3D27D4EB4FL)
    val samples = Vector.newBuilder[Sample]
    var run = 0
    while run < config.measureRuns do
      samples ++= runAllStages(context, config, measureRng)
      run += 1

    val byStage = samples.result().groupBy(_.stage).toVector.sortBy(_._1)
    println()
    byStage.foreach { case (stage, stageSamples) =>
      val stats = computeStats(stageSamples.map(_.elapsedMs))
      println(
        f"$stage%-24s count=${stats.count}%2d mean=${stats.meanMs}%.3fms " +
          f"median=${stats.medianMs}%.3fms min=${stats.minMs}%.3fms max=${stats.maxMs}%.3fms"
      )
    }
    println("=== Done ===")

  private def runAllStages(
      context: BenchmarkContext,
      config: Config,
      rng: Random
  ): Vector[Sample] =
    val stageSamples = Vector.newBuilder[Sample]

    val updateOnly = timed {
      HoldemBayesProvider.updatePosterior(
        prior = context.baselinePrior,
        observations = context.observations.map(o => o.action -> o.state),
        actionModel = context.actionModel
      )
    }
    stageSamples += Sample("bayesUpdateOnly", updateOnly)

    val inferPosterior = timed {
      RangeInferenceEngine.inferPosterior(
        hero = context.hero,
        board = context.state.board,
        folds = context.folds,
        tableRanges = context.tableRanges,
        villainPos = context.villainPos,
        observations = context.observations,
        actionModel = context.actionModel,
        bunchingTrials = config.bunchingTrials,
        rng = new Random(rng.nextLong()),
        useCache = false
      )
    }
    stageSamples += Sample("inferPosterior", inferPosterior)

    val recommend = timed {
      RangeInferenceEngine.recommendAction(
        hero = context.hero,
        state = context.state,
        posterior = context.baselinePrior,
        candidateActions = context.candidateActions,
        equityTrials = config.equityTrials,
        rng = new Random(rng.nextLong())
      )
    }
    stageSamples += Sample("recommendAction", recommend)

    val endToEnd = timed {
      RangeInferenceEngine.inferAndRecommend(
        hero = context.hero,
        state = context.state,
        folds = context.folds,
        tableRanges = context.tableRanges,
        villainPos = context.villainPos,
        observations = context.observations,
        actionModel = context.actionModel,
        candidateActions = context.candidateActions,
        bunchingTrials = config.bunchingTrials,
        equityTrials = config.equityTrials,
        rng = new Random(rng.nextLong())
      )
    }
    stageSamples += Sample("inferAndRecommend", endToEnd)

    stageSamples.result()

  private def timed(thunk: => Any): Long =
    val started = System.nanoTime()
    thunk
    math.max(1L, System.nanoTime() - started)

  private def computeStats(valuesMs: Vector[Double]): Stats =
    val sorted = valuesMs.sorted
    val count = sorted.length
    require(count > 0, "cannot compute benchmark stats for empty sample set")
    val mean = sorted.sum / count.toDouble
    val median = quantile(sorted, 0.5)
    Stats(
      count = count,
      meanMs = mean,
      medianMs = median,
      minMs = sorted.head,
      maxMs = sorted.last
    )

  private def quantile(sorted: Vector[Double], q: Double): Double =
    if sorted.length == 1 then sorted.head
    else
      val p = q * (sorted.length - 1).toDouble
      val lo = math.floor(p).toInt
      val hi = math.ceil(p).toInt
      if lo == hi then sorted(lo)
      else
        val w = p - lo.toDouble
        sorted(lo) * (1.0 - w) + sorted(hi) * w

  private def parseArgs(args: Vector[String]): Config =
    val options = args.flatMap { token =>
      token.split("=", 2) match
        case Array(key, value) if key.startsWith("--") && value.nonEmpty =>
          Some(key.drop(2) -> value)
        case _ =>
          None
    }.toMap

    Config(
      warmupRuns = options.get("warmupRuns").flatMap(_.toIntOption).getOrElse(2),
      measureRuns = options.get("measureRuns").flatMap(_.toIntOption).getOrElse(8),
      bunchingTrials = options.get("bunchingTrials").flatMap(_.toIntOption).getOrElse(1_200),
      equityTrials = options.get("equityTrials").flatMap(_.toIntOption).getOrElse(8_000),
      provider = options.getOrElse("provider", "auto"),
      seed = options.get("seed").flatMap(_.toLongOption).getOrElse(17L)
    )

  private def buildContext(seed: Long): BenchmarkContext =
    def card(token: String): Card =
      Card.parse(token).getOrElse(throw new IllegalArgumentException(s"invalid card token: $token"))

    def hole(a: String, b: String): HoleCards =
      HoleCards.from(Vector(card(a), card(b)))

    val hero = hole("Ac", "Kh")
    val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    val state = GameState(
      street = Street.Flop,
      board = board,
      pot = 22.0,
      toCall = 8.0,
      position = Position.Button,
      stackSize = 120.0,
      betHistory = Vector.empty
    )
    val tableRanges = TableRanges.defaults(TableFormat.NineMax)
    val folds = TableFormat.NineMax.foldsBeforeOpener(Position.Cutoff).map(PreflopFold(_))
    val villainPos = Position.BigBlind
    val observations = Seq(
      VillainObservation(PokerAction.Call, state.copy(toCall = 4.0)),
      VillainObservation(PokerAction.Raise(20.0), state.copy(toCall = 8.0))
    )
    val trainingData = Vector.fill(20)((state, hole("Ah", "Qh"), PokerAction.Raise(20.0))) ++
      Vector.fill(20)((state, hole("7c", "2d"), PokerAction.Fold)) ++
      Vector.fill(20)((state, hole("Jc", "Td"), PokerAction.Call))
    val actionModel = PokerActionModel.train(
      trainingData = trainingData,
      learningRate = 0.08,
      iterations = 500
    )

    val baselinePrior = BunchingEffect.adjustedRange(
      hero = hero,
      board = board,
      folds = folds,
      tableRanges = tableRanges,
      villainPos = villainPos,
      trials = 1_500,
      rng = new Random(seed ^ 0xD6E8FEB86659FD93L)
    )

    BenchmarkContext(
      hero = hero,
      state = state,
      folds = folds,
      tableRanges = tableRanges,
      villainPos = villainPos,
      observations = observations,
      actionModel = actionModel,
      candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(20.0)),
      baselinePrior = baselinePrior
    )

