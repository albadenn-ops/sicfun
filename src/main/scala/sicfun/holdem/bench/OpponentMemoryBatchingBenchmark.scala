package sicfun.holdem.bench
import sicfun.holdem.history.*
import sicfun.holdem.io.DecisionLoopEventFeedIO
import sicfun.holdem.model.{PokerActionModel, PokerActionModelArtifactIO}
import sicfun.holdem.runtime.AlwaysOnDecisionLoop
import sicfun.holdem.types.*

import sicfun.holdem.bench.BenchSupport.{card, hole}

import java.nio.file.{Files, Path, StandardCopyOption}
import scala.jdk.CollectionConverters.*

/** Coarse replay benchmark for opponent-memory overhead in the always-on loop.
  *
  * Measures end-to-end wall clock for realistic feed replays in one JVM, so the
  * comparison focuses on runtime behavior rather than sbt startup noise.
  */
object OpponentMemoryBatchingBenchmark:
  private final case class Config(
      hands: Int = 400,
      warmupRuns: Int = 1,
      measuredRuns: Int = 5,
      smallStoreProfiles: Int = 8,
      largeStoreProfiles: Int = 2000,
      bunchingTrials: Int = 50,
      equityTrials: Int = 200
  )

  private final case class Scenario(
      label: String,
      baselineStore: Option[Path]
  )

  private final case class ScenarioResult(
      label: String,
      storeBytes: Long,
      durationsMillis: Vector[Double]
  ):
    def meanMillis: Double =
      durationsMillis.sum / durationsMillis.length.toDouble

    def medianMillis: Double =
      val sorted = durationsMillis.sorted
      val mid = sorted.length / 2
      if sorted.length % 2 == 0 then (sorted(mid - 1) + sorted(mid)) / 2.0 else sorted(mid)

    def minMillis: Double = durationsMillis.min
    def maxMillis: Double = durationsMillis.max

  /** Entry point. Parses CLI args and runs three scenarios: memory-off (no opponent store),
    * memory-on with a small store (default 8 profiles), and memory-on with a large store
    * (default 2000 profiles). Measures wall clock for each, then reports the overhead delta
    * of opponent memory relative to the memory-off baseline.
    */
  def main(args: Array[String]): Unit =
    parseArgs(args) match
      case Left(err) =>
        System.err.println(err)
        sys.exit(1)
      case Right(config) =>
        run(config)

  private def run(config: Config): Unit =
    val root = Files.createTempDirectory("opponent-memory-benchmark-")
    try
      val shared = root.resolve("shared")
      Files.createDirectories(shared)
      val modelDir = shared.resolve("model")
      val feedPath = shared.resolve("feed.tsv")
      val smallStore = shared.resolve("store-small.json")
      val largeStore = shared.resolve("store-large.json")

      seedModelArtifact(modelDir)
      seedFeed(feedPath, config.hands)
      seedOpponentStore(smallStore, config.smallStoreProfiles)
      seedOpponentStore(largeStore, config.largeStoreProfiles)

      val scenarios = Vector(
        Scenario("memory-off", None),
        Scenario(s"memory-on-${config.smallStoreProfiles}p", Some(smallStore)),
        Scenario(s"memory-on-${config.largeStoreProfiles}p", Some(largeStore))
      )

      println("opponent-memory batching benchmark")
      println(s"hands=${config.hands} warmup=${config.warmupRuns} runs=${config.measuredRuns}")
      println()

      val results = scenarios.map { scenario =>
        repeat(config.warmupRuns)(_ => timedRun(root, scenario, modelDir, feedPath, config))
        val measured = repeat(config.measuredRuns)(_ => timedRun(root, scenario, modelDir, feedPath, config))
        ScenarioResult(
          label = scenario.label,
          storeBytes = scenario.baselineStore.map(Files.size).getOrElse(0L),
          durationsMillis = measured
        )
      }

      println(f"${"scenario"}%-20s ${"storeKiB"}%10s ${"meanMs"}%10s ${"medianMs"}%10s ${"minMs"}%10s ${"maxMs"}%10s")
      results.foreach { result =>
        println(
          f"${result.label}%-20s ${(result.storeBytes / 1024.0)}%10.1f ${result.meanMillis}%10.2f ${result.medianMillis}%10.2f ${result.minMillis}%10.2f ${result.maxMillis}%10.2f"
        )
      }

      val baseline = results.find(_.label == "memory-off")
      baseline.foreach { off =>
        println()
        results.filterNot(_.label == "memory-off").foreach { on =>
          val delta = on.meanMillis - off.meanMillis
          val pct =
            if off.meanMillis <= 0.0 then 0.0
            else (delta / off.meanMillis) * 100.0
          println(f"${on.label}: delta=${delta}%.2f ms (${pct}%+.1f%% vs memory-off)")
        }
      }
    finally
      deleteRecursively(root)

  /** Executes one full AlwaysOnDecisionLoop replay in an isolated temp directory.
    * Copies the opponent store (if any) so each run starts from the same baseline.
    * Returns elapsed wall time in milliseconds.
    */
  private def timedRun(
      benchmarkRoot: Path,
      scenario: Scenario,
      modelDir: Path,
      feedPath: Path,
      config: Config
  ): Double =
    val runRoot = Files.createTempDirectory(benchmarkRoot, sanitizeLabel(scenario.label) + "-")
    try
      val outDir = runRoot.resolve("out")
      val storeArg = scenario.baselineStore.map { baseline =>
        val copied = runRoot.resolve("profiles.json")
        Files.copy(baseline, copied, StandardCopyOption.REPLACE_EXISTING)
        copied
      }

      val args = Vector(
        s"--feedPath=$feedPath",
        s"--modelArtifactDir=$modelDir",
        s"--outputDir=$outDir",
        "--heroPlayerId=hero",
        "--heroCards=AcKh",
        "--villainPlayerId=villain",
        "--villainPosition=Button",
        "--tableFormat=ninemax",
        "--openerPosition=Cutoff",
        "--candidateActions=fold,call,raise:20",
        s"--bunchingTrials=${config.bunchingTrials}",
        s"--equityTrials=${config.equityTrials}",
        "--maxPolls=1",
        "--pollMillis=0"
      ) ++ storeArg.toVector.flatMap(path =>
        Vector(
          s"--opponentStore=$path",
          "--opponentSite=pokerstars",
          "--opponentName=villain"
        )
      )

      val started = System.nanoTime()
      val result = AlwaysOnDecisionLoop.run(args.toArray)
      val elapsedMillis = (System.nanoTime() - started).toDouble / 1_000_000.0
      result match
        case Left(err) =>
          throw new IllegalStateException(s"benchmark scenario '${scenario.label}' failed: $err")
        case Right(summary) =>
          val expectedEvents = config.hands * 2
          require(summary.processedEvents == expectedEvents, s"expected $expectedEvents events, got ${summary.processedEvents}")
          require(summary.decisionsEmitted == config.hands, s"expected ${config.hands} decisions, got ${summary.decisionsEmitted}")
      elapsedMillis
    finally
      deleteRecursively(runRoot)

  /** Generates a synthetic event feed with `hands` hand pairs (hero event + villain event).
    * Villain actions cycle through fold/call/raise to exercise all action paths.
    */
  private def seedFeed(path: Path, hands: Int): Unit =
    val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    (0 until hands).foreach { idx =>
      val handId = f"bench-hand-$idx%04d"
      val hero = PokerEvent(
        handId = handId,
        sequenceInHand = 0L,
        playerId = "hero",
        occurredAtEpochMillis = 1_800_000_000_000L + (idx * 100L),
        street = Street.Flop,
        position = Position.BigBlind,
        board = board,
        potBefore = 20.0,
        toCall = 0.0,
        stackBefore = 180.0,
        action = PokerAction.Raise(12.0),
        betHistory = Vector.empty
      )
      val villainAction =
        idx % 3 match
          case 0 => PokerAction.Fold
          case 1 => PokerAction.Call
          case _ => PokerAction.Raise(28.0)
      val villain = PokerEvent(
        handId = handId,
        sequenceInHand = 1L,
        playerId = "villain",
        occurredAtEpochMillis = 1_800_000_000_010L + (idx * 100L),
        street = Street.Flop,
        position = Position.Button,
        board = board,
        potBefore = 32.0,
        toCall = 12.0,
        stackBefore = 180.0,
        action = villainAction,
        betHistory = Vector(BetAction(0, PokerAction.Raise(12.0)))
      )
      DecisionLoopEventFeedIO.append(path, hero)
      DecisionLoopEventFeedIO.append(path, villain)
    }

  private def seedModelArtifact(modelDir: Path): Unit =
    val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    val state = GameState(
      street = Street.Flop,
      board = board,
      pot = 20.0,
      toCall = 10.0,
      position = Position.BigBlind,
      stackSize = 180.0,
      betHistory = Vector.empty
    )
    val checkState = state.copy(toCall = 0.0)
    val training = Vector.fill(16)((state, hole("Ah", "Ad"), PokerAction.Raise(25.0))) ++
      Vector.fill(16)((state, hole("Qc", "Jc"), PokerAction.Call)) ++
      Vector.fill(16)((state, hole("7c", "2d"), PokerAction.Fold)) ++
      Vector.fill(8)((checkState, hole("As", "Ks"), PokerAction.Check))

    val artifact = PokerActionModel.trainVersioned(
      trainingData = training,
      learningRate = 0.1,
      iterations = 200,
      l2Lambda = 0.001,
      validationFraction = 0.25,
      splitSeed = 7L,
      maxMeanBrierScore = 2.0,
      failOnGate = false,
      modelId = "opponent-memory-benchmark",
      source = "opponent-memory-benchmark",
      trainedAtEpochMillis = 777777L
    )
    PokerActionModelArtifactIO.save(modelDir, artifact)

  /** Generates a synthetic opponent profile store with the specified number of profiles.
    * Each profile has 3 recent events and 12 seen hand IDs. The first profile is always
    * named "villain" (matching the benchmark's villainPlayerId).
    */
  private def seedOpponentStore(path: Path, profiles: Int): Unit =
    val profileCount = math.max(1, profiles)
    val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    val built = Vector.tabulate(profileCount) { idx =>
      val name = if idx == 0 then "villain" else s"villain-$idx"
      val baseMillis = 1_700_000_000_000L + (idx.toLong * 10_000L)
      val recentEvents = Vector(
        PokerEvent(
          handId = s"$name-h1",
          sequenceInHand = 0L,
          playerId = name,
          occurredAtEpochMillis = baseMillis,
          street = Street.Flop,
          position = Position.Button,
          board = board,
          potBefore = 20.0,
          toCall = 0.0,
          stackBefore = 180.0,
          action = PokerAction.Check,
          betHistory = Vector.empty
        ),
        PokerEvent(
          handId = s"$name-h2",
          sequenceInHand = 1L,
          playerId = name,
          occurredAtEpochMillis = baseMillis + 1L,
          street = Street.Flop,
          position = Position.Button,
          board = board,
          potBefore = 28.0,
          toCall = 8.0,
          stackBefore = 176.0,
          action = PokerAction.Call,
          betHistory = Vector(BetAction(0, PokerAction.Raise(8.0)))
        ),
        PokerEvent(
          handId = s"$name-h3",
          sequenceInHand = 2L,
          playerId = name,
          occurredAtEpochMillis = baseMillis + 2L,
          street = Street.Flop,
          position = Position.Button,
          board = board,
          potBefore = 32.0,
          toCall = 12.0,
          stackBefore = 180.0,
          action = PokerAction.Fold,
          betHistory = Vector(BetAction(0, PokerAction.Raise(12.0)))
        )
      )
      OpponentProfile(
        site = "pokerstars",
        playerName = name,
        handsObserved = 12,
        firstSeenEpochMillis = baseMillis,
        lastSeenEpochMillis = baseMillis + 2L,
        actionSummary = OpponentActionSummary(folds = 3, raises = 5, calls = 3, checks = 2),
        raiseResponses = sicfun.holdem.engine.RaiseResponseCounts(folds = 2, calls = 3, raises = 1),
        recentEvents = recentEvents,
        seenHandIds = Vector.tabulate(12)(n => s"$name-seen-$n")
      )
    }
    OpponentProfileStore.save(path, OpponentProfileStore(built))

  private def repeat[A](times: Int)(thunk: Int => A): Vector[A] =
    Vector.tabulate(math.max(0, times))(thunk)

  private def sanitizeLabel(label: String): String =
    label.map {
      case c if c.isLetterOrDigit => c
      case _ => '-'
    }

  private def parseArgs(args: Array[String]): Either[String, Config] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      args.foldLeft[Either[String, Config]](Right(Config())) { (acc, raw) =>
        acc.flatMap { current =>
          if !raw.startsWith("--") || !raw.contains("=") then Left(s"invalid argument '$raw'")
          else
            val Array(key, value) = raw.drop(2).split("=", 2)
            key match
              case "hands" => parsePositiveInt(value, "--hands").map(v => current.copy(hands = v))
              case "warmup" => parseNonNegativeInt(value, "--warmup").map(v => current.copy(warmupRuns = v))
              case "runs" => parsePositiveInt(value, "--runs").map(v => current.copy(measuredRuns = v))
              case "smallStoreProfiles" =>
                parsePositiveInt(value, "--smallStoreProfiles").map(v => current.copy(smallStoreProfiles = v))
              case "largeStoreProfiles" =>
                parsePositiveInt(value, "--largeStoreProfiles").map(v => current.copy(largeStoreProfiles = v))
              case "bunchingTrials" =>
                parsePositiveInt(value, "--bunchingTrials").map(v => current.copy(bunchingTrials = v))
              case "equityTrials" =>
                parsePositiveInt(value, "--equityTrials").map(v => current.copy(equityTrials = v))
              case other => Left(s"unknown argument '--$other'")
        }
      }

  private def parsePositiveInt(raw: String, flag: String): Either[String, Int] =
    raw.toIntOption match
      case Some(value) if value > 0 => Right(value)
      case _ => Left(s"$flag must be > 0")

  private def parseNonNegativeInt(raw: String, flag: String): Either[String, Int] =
    raw.toIntOption match
      case Some(value) if value >= 0 => Right(value)
      case _ => Left(s"$flag must be >= 0")

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse.foreach(Files.deleteIfExists)
      finally
        stream.close()

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.bench.OpponentMemoryBatchingBenchmark [--key=value ...]
      |
      |Options:
      |  --hands=<int>                Default 400
      |  --warmup=<int>               Default 1
      |  --runs=<int>                 Default 5
      |  --smallStoreProfiles=<int>   Default 8
      |  --largeStoreProfiles=<int>   Default 2000
      |  --bunchingTrials=<int>       Default 50
      |  --equityTrials=<int>         Default 200
      |""".stripMargin
