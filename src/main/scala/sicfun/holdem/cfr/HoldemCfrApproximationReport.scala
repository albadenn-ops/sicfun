package sicfun.holdem.cfr
import sicfun.holdem.types.*
import sicfun.holdem.cli.*
import sicfun.holdem.equity.RangeParser

import sicfun.core.{Card, DiscreteDistribution}

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.Locale
import scala.jdk.CollectionConverters.*
import ujson.{Arr, Num, Obj, Str, Value}

/** Offline diagnostics for the repo's CFR approximation quality.
  *
  * Runs a fixed suite of representative heads-up poker spots covering all streets
  * (preflop through river) and multiple positions, then reports the solver's own
  * local quality metrics for each spot:
  *  - '''Root deviation gap''': how much hero could gain by switching to a best-response
  *    at the root, measuring how far the hero's strategy is from optimal.
  *  - '''Villain deviation gap''': how much villain could gain by switching to a
  *    best-response, measuring how exploitable our villain model is.
  *  - '''Local exploitability''': sum of root + villain deviation gaps, the key
  *    convergence metric. Lower is better; zero means Nash equilibrium.
  *
  * Outputs:
  *  - Console summary with aggregate statistics
  *  - Optional file outputs: summary.txt, spots.tsv, external-comparison.json
  *
  * The external-comparison JSON export includes full spot specifications (board, ranges,
  * bet history, spot signatures) enabling cross-validation against external solvers
  * like TexasSolver via [[HoldemCfrExternalComparison]].
  *
  * The default suite contains 6 carefully chosen spots that cover:
  *  - Preflop open from button
  *  - Flop defense from big blind
  *  - Turn facing a probe bet
  *  - Turn bet-or-check as aggressor
  *  - River bluff-catching
  *  - River bet-or-check from big blind
  */
object HoldemCfrApproximationReport:
  private val IsoFormatter = DateTimeFormatter.ISO_INSTANT.withZone(ZoneOffset.UTC)

  /** A diagnostic spot definition: a fully specified poker decision point that
    * can be solved by CFR and whose solution quality can be measured.
    */
  final case class DiagnosticSpot(
      id: String,
      hero: HoleCards,
      state: GameState,
      villainRange: DiscreteDistribution[HoleCards],
      candidateActions: Vector[PokerAction]
  ):
    require(id.trim.nonEmpty, "spot id must be non-empty")
    require(candidateActions.nonEmpty, "candidateActions must be non-empty")

  final case class SpotResult(
      spot: DiagnosticSpot,
      solution: HoldemCfrSolution
  )

  final case class ExportBetHistoryEntry(
      player: Int,
      action: String
  )

  final case class ExportRangeWeight(
      hand: String,
      probability: Double
  )

  /** Full spot export for external comparison, including the complete spot specification
    * (board, ranges, bet history, spot signature) and the solved policy/EV/exploitability.
    * This is serialized to JSON for cross-validation with external solvers.
    */
  final case class ExternalComparisonSpot(
      id: String,
      spotSignature: String,
      street: String,
      position: String,
      hero: String,
      board: Vector[String],
      pot: Double,
      toCall: Double,
      stackSize: Double,
      betHistory: Vector[ExportBetHistoryEntry],
      villainRange: Vector[ExportRangeWeight],
      candidateActions: Vector[String],
      policy: Map[String, Double],
      actionEvs: Map[String, Double],
      bestAction: String,
      expectedValuePlayer0: Double,
      rootDeviationGap: Double,
      villainDeviationGap: Double,
      localExploitability: Double,
      iterations: Int,
      villainSupport: Int,
      provider: String
  ):
    require(id.trim.nonEmpty, "export spot id must be non-empty")
    require(spotSignature.trim.nonEmpty, "export spotSignature must be non-empty")
    require(candidateActions.nonEmpty, "export candidateActions must be non-empty")

  /** Aggregate quality statistics across all spots in a suite run.
    * Used for both console output and JSON export.
    */
  final case class AggregateStats(
      spotCount: Int,
      meanLocalExploitability: Double,
      maxLocalExploitability: Double,
      meanRootDeviationGap: Double,
      maxRootDeviationGap: Double,
      meanVillainDeviationGap: Double,
      maxVillainDeviationGap: Double,
      providerCounts: Map[String, Int]
  ):
    require(spotCount >= 0, "spotCount must be non-negative")

  final case class ExternalComparisonExport(
      suiteName: String,
      generatedAtIso: String,
      cfrConfig: HoldemCfrConfig,
      aggregate: AggregateStats,
      spots: Vector[ExternalComparisonSpot]
  ):
    require(suiteName.trim.nonEmpty, "suiteName must be non-empty")

  final case class RunResult(
      suiteName: String,
      spotResults: Vector[SpotResult],
      aggregate: AggregateStats,
      externalComparison: ExternalComparisonExport,
      outDir: Option[Path]
  )

  private final case class CliConfig(
      suiteName: String,
      cfrConfig: HoldemCfrConfig,
      outDir: Option[Path]
  )

  private[cfr] val SpotsTsvHeader =
    "spotId\tstreet\tposition\thero\tboard\tpot\ttoCall\tstackSize\tcandidateActions\tbestAction\t" +
      "provider\titerations\tvillainSupport\texpectedValuePlayer0\trootDeviationGap\t" +
      "villainDeviationGap\tlocalExploitability\tpolicy"
  private[cfr] val ExternalComparisonFileName = "external-comparison.json"

  private[cfr] val DefaultSuite: Vector[DiagnosticSpot] =
    Vector(
      DiagnosticSpot(
        id = "hu_preflop_button_open",
        hero = hole("Ac", "Kh"),
        state = GameState(
          street = Street.Preflop,
          board = Board.empty,
          pot = 1.5,
          toCall = 0.5,
          position = Position.Button,
          stackSize = 99.5,
          betHistory = Vector.empty
        ),
        villainRange = range("22+,A2s+,K5s+,Q8s+,J8s+,T8s+,98s,87s,76s,A8o+,KTo+,QTo+,JTo"),
        candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(2.5))
      ),
      DiagnosticSpot(
        id = "hu_flop_bigblind_defense",
        hero = hole("9h", "8h"),
        state = GameState(
          street = Street.Flop,
          board = board("Tc", "7s", "2c"),
          pot = 8.0,
          toCall = 2.5,
          position = Position.BigBlind,
          stackSize = 92.0,
          betHistory = Vector(BetAction(0, PokerAction.Raise(2.5)))
        ),
        villainRange = range("22+,A2s+,K7s+,Q9s+,J9s+,T8s+,98s,87s,A8o+,KTo+,QTo+,JTo"),
        candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(8.0))
      ),
      DiagnosticSpot(
        id = "hu_turn_button_vs_probe",
        hero = hole("Ac", "Kd"),
        state = GameState(
          street = Street.Turn,
          board = board("2c", "7d", "Jh", "Qc"),
          pot = 18.0,
          toCall = 4.0,
          position = Position.Button,
          stackSize = 82.0,
          betHistory = Vector(BetAction(1, PokerAction.Raise(4.0)))
        ),
        villainRange = DiscreteDistribution(
          Map(
            hole("As", "Qd") -> 0.35,
            hole("Ts", "9s") -> 0.30,
            hole("Qh", "Js") -> 0.20,
            hole("7c", "7s") -> 0.15
          )
        ),
        candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(12.0))
      ),
      DiagnosticSpot(
        id = "hu_turn_button_bet_or_check",
        hero = hole("Ah", "Jh"),
        state = GameState(
          street = Street.Turn,
          board = board("9h", "8d", "2h", "Qc"),
          pot = 16.0,
          toCall = 0.0,
          position = Position.Button,
          stackSize = 84.0,
          betHistory = Vector.empty
        ),
        villainRange = range("22+,A2s+,K9s+,Q9s+,J9s+,T9s,98s,87s,A9o+,KTo+,QTo+,JTo"),
        candidateActions = Vector(PokerAction.Check, PokerAction.Raise(10.5), PokerAction.Raise(16.0))
      ),
      DiagnosticSpot(
        id = "hu_river_bluffcatch",
        hero = hole("Ad", "9d"),
        state = GameState(
          street = Street.River,
          board = board("Kd", "Qh", "7c", "2d", "2s"),
          pot = 22.0,
          toCall = 12.0,
          position = Position.Button,
          stackSize = 66.0,
          betHistory = Vector(BetAction(1, PokerAction.Raise(12.0)))
        ),
        villainRange = range("22+,A2s+,K9s+,Q9s+,JTs,A9o+,KTo+,QTo+,JTo"),
        candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(36.0))
      ),
      DiagnosticSpot(
        id = "hu_river_bigblind_bet_or_check",
        hero = hole("Qc", "Jc"),
        state = GameState(
          street = Street.River,
          board = board("Qh", "9d", "4c", "4h", "2s"),
          pot = 18.0,
          toCall = 0.0,
          position = Position.BigBlind,
          stackSize = 70.0,
          betHistory = Vector.empty
        ),
        villainRange = range("22+,A2s+,KTs+,Q8s+,J8s+,T8s+,98s,87s,A9o+,KTo+,QTo+,JTo"),
        candidateActions = Vector(PokerAction.Check, PokerAction.Raise(12.0), PokerAction.Raise(18.0))
      )
    )

  def main(args: Array[String]): Unit =
    val wantsHelp = args.contains("--help") || args.contains("-h")
    run(args) match
      case Right(result) =>
        println("=== Holdem CFR Approximation Report ===")
        println(s"suite: ${result.suiteName}")
        println(s"spots: ${result.aggregate.spotCount}")
        println(s"meanLocalExploitability: ${formatDouble(result.aggregate.meanLocalExploitability, 6)}")
        println(s"maxLocalExploitability: ${formatDouble(result.aggregate.maxLocalExploitability, 6)}")
        println(s"meanRootDeviationGap: ${formatDouble(result.aggregate.meanRootDeviationGap, 6)}")
        println(s"maxRootDeviationGap: ${formatDouble(result.aggregate.maxRootDeviationGap, 6)}")
        println(s"meanVillainDeviationGap: ${formatDouble(result.aggregate.meanVillainDeviationGap, 6)}")
        println(s"maxVillainDeviationGap: ${formatDouble(result.aggregate.maxVillainDeviationGap, 6)}")
        println(s"providerCounts: ${result.aggregate.providerCounts.toVector.sortBy(_._1).mkString(", ")}")
        result.outDir.foreach(path => println(s"outDir: ${path.toAbsolutePath.normalize()}"))
      case Left(error) =>
        if wantsHelp then println(error)
        else
          System.err.println(error)
          sys.exit(1)

  def run(args: Array[String]): Either[String, RunResult] =
    for
      config <- parseArgs(args)
      result <- runSuite(config.suiteName, DefaultSuite, config.cfrConfig, config.outDir)
    yield result

  /** Runs the full diagnostic suite: solves each spot, computes exploitability
    * metrics, builds aggregate statistics, and optionally writes output files.
    *
    * @param suiteName Name for the suite (used in output labeling)
    * @param spots     The diagnostic spots to solve
    * @param cfrConfig CFR solver configuration (iterations, averaging, etc.)
    * @param outDir    Optional output directory for summary.txt, spots.tsv, external-comparison.json
    * @return Right(RunResult) on success, Left(error) on failure
    */
  private[cfr] def runSuite(
      suiteName: String,
      spots: Vector[DiagnosticSpot],
      cfrConfig: HoldemCfrConfig,
      outDir: Option[Path]
  ): Either[String, RunResult] =
    try
      require(spots.nonEmpty, "diagnostic suite must contain at least one spot")

      val spotResults = spots.map { spot =>
        val solution = HoldemCfrSolver.solve(
          hero = spot.hero,
          state = spot.state,
          villainPosterior = spot.villainRange,
          candidateActions = spot.candidateActions,
          config = cfrConfig
        )
        SpotResult(spot, solution)
      }

      val aggregate = buildAggregate(spotResults)
      val externalComparison = buildExternalComparisonExport(suiteName, cfrConfig, aggregate, spotResults)

      val resolvedOutDir = outDir.map { dir =>
        Files.createDirectories(dir)
        Files.write(
          dir.resolve("summary.txt"),
          buildSummaryLines(suiteName, aggregate, spotResults).asJava,
          StandardCharsets.UTF_8
        )
        Files.write(
          dir.resolve("spots.tsv"),
          buildSpotRows(spotResults).asJava,
          StandardCharsets.UTF_8
        )
        Files.writeString(
          dir.resolve(ExternalComparisonFileName),
          ujson.write(writeExternalComparison(externalComparison), indent = 2),
          StandardCharsets.UTF_8
        )
        dir
      }

      Right(
        RunResult(
          suiteName = suiteName,
          spotResults = spotResults,
          aggregate = aggregate,
          externalComparison = externalComparison,
          outDir = resolvedOutDir
        )
      )
    catch
      case e: Exception =>
        Left(s"holdem CFR approximation report failed: ${e.getMessage}")

  /** Computes aggregate quality statistics (mean/max exploitability, deviation gaps,
    * provider distribution) across all solved spots.
    */
  private def buildAggregate(spotResults: Vector[SpotResult]): AggregateStats =
    def meanOf(f: SpotResult => Double): Double =
      if spotResults.isEmpty then 0.0 else spotResults.map(f).sum / spotResults.size.toDouble

    def maxOf(f: SpotResult => Double): Double =
      if spotResults.isEmpty then 0.0 else spotResults.map(f).max

    val providerCounts =
      spotResults
        .groupBy(_.solution.provider)
        .view
        .mapValues(_.size)
        .toMap

    AggregateStats(
      spotCount = spotResults.size,
      meanLocalExploitability = meanOf(_.solution.localExploitability),
      maxLocalExploitability = maxOf(_.solution.localExploitability),
      meanRootDeviationGap = meanOf(_.solution.rootDeviationGap),
      maxRootDeviationGap = maxOf(_.solution.rootDeviationGap),
      meanVillainDeviationGap = meanOf(_.solution.villainDeviationGap),
      maxVillainDeviationGap = maxOf(_.solution.villainDeviationGap),
      providerCounts = providerCounts
    )

  private def buildExternalComparisonExport(
      suiteName: String,
      cfrConfig: HoldemCfrConfig,
      aggregate: AggregateStats,
      spotResults: Vector[SpotResult]
  ): ExternalComparisonExport =
    ExternalComparisonExport(
      suiteName = suiteName,
      generatedAtIso = IsoFormatter.format(Instant.ofEpochMilli(System.currentTimeMillis())),
      cfrConfig = cfrConfig,
      aggregate = aggregate,
      spots = spotResults.map(exportSpot)
    )

  private def buildSummaryLines(
      suiteName: String,
      aggregate: AggregateStats,
      spotResults: Vector[SpotResult]
  ): Vector[String] =
    Vector(
      "Holdem CFR Approximation Report",
      s"generatedAt: ${IsoFormatter.format(Instant.ofEpochMilli(System.currentTimeMillis()))}",
      s"suite: $suiteName",
      s"spots: ${aggregate.spotCount}",
      s"meanLocalExploitability: ${formatDouble(aggregate.meanLocalExploitability, 8)}",
      s"maxLocalExploitability: ${formatDouble(aggregate.maxLocalExploitability, 8)}",
      s"meanRootDeviationGap: ${formatDouble(aggregate.meanRootDeviationGap, 8)}",
      s"maxRootDeviationGap: ${formatDouble(aggregate.maxRootDeviationGap, 8)}",
      s"meanVillainDeviationGap: ${formatDouble(aggregate.meanVillainDeviationGap, 8)}",
      s"maxVillainDeviationGap: ${formatDouble(aggregate.maxVillainDeviationGap, 8)}",
      s"providerCounts: ${aggregate.providerCounts.toVector.sortBy(_._1).mkString(", ")}",
      "",
      "spots:",
      spotResults.map { result =>
        String.format(
          Locale.ROOT,
          "%-24s exploitability=%s rootGap=%s villainGap=%s provider=%s",
          result.spot.id,
          formatDouble(result.solution.localExploitability, 8),
          formatDouble(result.solution.rootDeviationGap, 8),
          formatDouble(result.solution.villainDeviationGap, 8),
          result.solution.provider
        )
      }.mkString("\n")
    )

  private def buildSpotRows(spotResults: Vector[SpotResult]): Vector[String] =
    SpotsTsvHeader +:
      spotResults.map { result =>
        val spot = result.spot
        val solution = result.solution
        Vector(
          spot.id,
          spot.state.street.toString,
          spot.state.position.toString,
          spot.hero.toToken,
          renderBoard(spot.state.board),
          spot.state.pot.toString,
          spot.state.toCall.toString,
          spot.state.stackSize.toString,
          spot.candidateActions.map(renderAction).mkString(","),
          renderAction(solution.bestAction),
          solution.provider,
          solution.iterations.toString,
          solution.villainSupport.toString,
          solution.expectedValuePlayer0.toString,
          solution.rootDeviationGap.toString,
          solution.villainDeviationGap.toString,
          solution.localExploitability.toString,
          policySummary(solution)
        ).mkString("\t")
      }

  /** Converts a solved spot result into the full export format for external comparison.
    * Includes the spot specification (board, ranges, etc.), spot signature, solved policy,
    * per-action EVs, and exploitability metrics.
    */
  private def exportSpot(result: SpotResult): ExternalComparisonSpot =
    val spot = result.spot
    val solution = result.solution
    ExternalComparisonSpot(
      id = spot.id,
      spotSignature = buildSpotSignature(spot),
      street = spot.state.street.toString,
      position = spot.state.position.toString,
      hero = spot.hero.toToken,
      board = spot.state.board.cards.map(_.toToken),
      pot = spot.state.pot,
      toCall = spot.state.toCall,
      stackSize = spot.state.stackSize,
      betHistory = spot.state.betHistory.map(action =>
        ExportBetHistoryEntry(player = action.player, action = renderAction(action.action))
      ),
      villainRange = spot.villainRange.normalized.weights.toVector
        .collect { case (hand, probability) if probability > 0.0 =>
          ExportRangeWeight(hand = hand.toToken, probability = probability)
        }
        .sortBy(_.hand),
      candidateActions = spot.candidateActions.map(renderAction),
      policy = solution.actionProbabilities.toVector
        .map { case (action, probability) => renderAction(action) -> probability }
        .toMap,
      actionEvs = solution.actionEvaluations
        .map(eval => renderAction(eval.action) -> eval.expectedValue)
        .toMap,
      bestAction = renderAction(solution.bestAction),
      expectedValuePlayer0 = solution.expectedValuePlayer0,
      rootDeviationGap = solution.rootDeviationGap,
      villainDeviationGap = solution.villainDeviationGap,
      localExploitability = solution.localExploitability,
      iterations = solution.iterations,
      villainSupport = solution.villainSupport,
      provider = solution.provider
    )

  private def writeExternalComparison(report: ExternalComparisonExport): Value =
    Obj(
      "suiteName" -> Str(report.suiteName),
      "generatedAtIso" -> Str(report.generatedAtIso),
      "cfrConfig" -> writeCfrConfig(report.cfrConfig),
      "aggregate" -> writeAggregate(report.aggregate),
      "spots" -> Arr.from(report.spots.map(writeExternalComparisonSpot))
    )

  private def writeCfrConfig(config: HoldemCfrConfig): Value =
    Obj(
      "iterations" -> Num(config.iterations),
      "cfrPlus" -> ujson.Bool(config.cfrPlus),
      "averagingDelay" -> Num(config.averagingDelay),
      "linearAveraging" -> ujson.Bool(config.linearAveraging),
      "maxVillainHands" -> Num(config.maxVillainHands),
      "equityTrials" -> Num(config.equityTrials),
      "includeVillainReraises" -> ujson.Bool(config.includeVillainReraises),
      "villainReraiseMultipliers" -> Arr.from(config.villainReraiseMultipliers.map(Num(_))),
      "preferNativeBatch" -> ujson.Bool(config.preferNativeBatch),
      "rngSeed" -> Num(config.rngSeed.toDouble)
    )

  private def writeAggregate(aggregate: AggregateStats): Value =
    Obj(
      "spotCount" -> Num(aggregate.spotCount),
      "meanLocalExploitability" -> Num(aggregate.meanLocalExploitability),
      "maxLocalExploitability" -> Num(aggregate.maxLocalExploitability),
      "meanRootDeviationGap" -> Num(aggregate.meanRootDeviationGap),
      "maxRootDeviationGap" -> Num(aggregate.maxRootDeviationGap),
      "meanVillainDeviationGap" -> Num(aggregate.meanVillainDeviationGap),
      "maxVillainDeviationGap" -> Num(aggregate.maxVillainDeviationGap),
      "providerCounts" -> Obj.from(
        aggregate.providerCounts.toVector
          .sortBy(_._1)
          .map { case (provider, count) => provider -> Num(count) }
      )
    )

  private def writeExternalComparisonSpot(spot: ExternalComparisonSpot): Value =
    Obj(
      "id" -> Str(spot.id),
      "spotSignature" -> Str(spot.spotSignature),
      "state" -> Obj(
        "street" -> Str(spot.street),
        "position" -> Str(spot.position),
        "board" -> Arr.from(spot.board.map(Str(_))),
        "pot" -> Num(spot.pot),
        "toCall" -> Num(spot.toCall),
        "stackSize" -> Num(spot.stackSize),
        "betHistory" -> Arr.from(spot.betHistory.map(writeExportBetHistoryEntry))
      ),
      "hero" -> Str(spot.hero),
      "villainRange" -> Arr.from(spot.villainRange.map(writeExportRangeWeight)),
      "candidateActions" -> Arr.from(spot.candidateActions.map(Str(_))),
      "policy" -> writeDoubleMap(spot.policy),
      "actionEvs" -> writeDoubleMap(spot.actionEvs),
      "bestAction" -> Str(spot.bestAction),
      "expectedValuePlayer0" -> Num(spot.expectedValuePlayer0),
      "rootDeviationGap" -> Num(spot.rootDeviationGap),
      "villainDeviationGap" -> Num(spot.villainDeviationGap),
      "localExploitability" -> Num(spot.localExploitability),
      "iterations" -> Num(spot.iterations),
      "villainSupport" -> Num(spot.villainSupport),
      "provider" -> Str(spot.provider)
    )

  /** Builds a canonical spot signature that uniquely identifies the poker situation.
    * The signature encodes street, position, hero, board, pot geometry, bet history,
    * candidate actions, and normalized villain range in a deterministic format.
    * Used to verify that reference and external datasets solved the same problem.
    */
  private def buildSpotSignature(spot: DiagnosticSpot): String =
    val normalizedRange = spot.villainRange.normalized.weights.toVector
      .collect { case (hand, probability) if probability > 0.0 =>
        s"${hand.toToken}@${formatDouble(probability, 8)}"
      }
      .sortBy(identity)
      .mkString(";")
    val normalizedBoard = spot.state.board.cards.map(_.toToken).sorted.mkString("")
    val normalizedHistory = spot.state.betHistory
      .map(action => s"${action.player}:${renderAction(action.action)}")
      .mkString(";")
    val normalizedActions = spot.candidateActions
      .map(renderAction)
      .sorted
      .mkString(",")
    s"street=${spot.state.street.toString.toUpperCase(Locale.ROOT)}|" +
      s"position=${spot.state.position.toString.toUpperCase(Locale.ROOT)}|hero=${spot.hero.toToken}|" +
      s"board=$normalizedBoard|pot=${formatDouble(spot.state.pot, 6)}|" +
      s"toCall=${formatDouble(spot.state.toCall, 6)}|" +
      s"stackSize=${formatDouble(spot.state.stackSize, 6)}|" +
      s"betHistory=$normalizedHistory|candidateActions=$normalizedActions|villainRange=$normalizedRange"

  private def writeExportBetHistoryEntry(entry: ExportBetHistoryEntry): Value =
    Obj(
      "player" -> Num(entry.player),
      "action" -> Str(entry.action)
    )

  private def writeExportRangeWeight(entry: ExportRangeWeight): Value =
    Obj(
      "hand" -> Str(entry.hand),
      "probability" -> Num(entry.probability)
    )

  private def writeDoubleMap(values: Map[String, Double]): Value =
    Obj.from(
      values.toVector
        .sortBy(_._1)
        .map { case (key, value) => key -> Num(value) }
    )

  private def parseArgs(args: Array[String]): Either[String, CliConfig] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- CliHelpers.parseOptions(args)
        suiteName = options.getOrElse("suite", "default")
        _ <- if suiteName == "default" then Right(()) else Left("--suite only supports 'default'")
        iterations <- CliHelpers.parseIntOptionEither(options, "iterations", 1_200)
        _ <- if iterations > 0 then Right(()) else Left("--iterations must be > 0")
        averagingDelay <- CliHelpers.parseIntOptionEither(options, "averagingDelay", 200)
        _ <- if averagingDelay >= 0 then Right(()) else Left("--averagingDelay must be >= 0")
        maxVillainHands <- CliHelpers.parseIntOptionEither(options, "maxVillainHands", 48)
        _ <- if maxVillainHands > 0 then Right(()) else Left("--maxVillainHands must be > 0")
        equityTrials <- CliHelpers.parseIntOptionEither(options, "equityTrials", 1_200)
        _ <- if equityTrials > 0 then Right(()) else Left("--equityTrials must be > 0")
        includeVillainReraises <- CliHelpers.parseBooleanOptionEither(options, "includeVillainReraises", true)
        preferNativeBatch <- CliHelpers.parseBooleanOptionEither(options, "preferNativeBatch", true)
        rngSeed <- CliHelpers.parseLongOptionEither(options, "seed", 1L)
        outDir = options.get("outDir").map(Paths.get(_))
      yield
        CliConfig(
          suiteName = suiteName,
          cfrConfig = HoldemCfrConfig(
            iterations = iterations,
            averagingDelay = averagingDelay,
            maxVillainHands = maxVillainHands,
            equityTrials = equityTrials,
            includeVillainReraises = includeVillainReraises,
            postflopLookahead = true,
            preferNativeBatch = preferNativeBatch,
            rngSeed = rngSeed
          ),
          outDir = outDir
        )

  private def policySummary(solution: HoldemCfrSolution): String =
    solution.actionProbabilities.toVector
      .sortBy { case (action, _) => renderAction(action) }
      .map { case (action, probability) =>
        s"${renderAction(action)}=${probability}"
      }
      .mkString("|")

  private def renderAction(action: PokerAction): String =
    action match
      case PokerAction.Fold => "FOLD"
      case PokerAction.Check => "CHECK"
      case PokerAction.Call => "CALL"
      case PokerAction.Raise(amount) => String.format(Locale.ROOT, "RAISE:%.3f", java.lang.Double.valueOf(amount))

  private def renderBoard(board: Board): String =
    if board.cards.isEmpty then "[]"
    else board.cards.map(_.toToken).mkString("[", " ", "]")

  private def formatDouble(value: Double, digits: Int): String =
    String.format(Locale.ROOT, s"%.${digits}f", java.lang.Double.valueOf(value))

  private def range(expression: String): DiscreteDistribution[HoleCards] =
    RangeParser.parse(expression) match
      case Right(dist) => dist
      case Left(err) => throw new IllegalArgumentException(s"invalid built-in range '$expression': $err")

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def board(tokens: String*): Board =
    Board.from(tokens.map(card))

  private def card(token: String): Card =
    Card.parse(token).getOrElse(
      throw new IllegalArgumentException(s"invalid built-in card token '$token'")
    )

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.cfr.HoldemCfrApproximationReport [--key=value ...]
      |
      |Options:
      |  --suite=default                  Built-in representative spot suite
      |  --iterations=<int>               Default 1200
      |  --averagingDelay=<int>           Default 200
      |  --maxVillainHands=<int>          Default 48
      |  --equityTrials=<int>             Default 1200
      |  --includeVillainReraises=<bool>  Default true
      |  postflop lookahead               Enabled for postflop spots
      |  --preferNativeBatch=<bool>       Default true
      |  --seed=<long>                    Default 1
      |  --outDir=<path>                  Optional output directory for summary.txt/spots.tsv/external-comparison.json
      |""".stripMargin
