package sicfun.holdem.validation

import sicfun.holdem.engine.RealTimeAdaptiveEngine
import sicfun.holdem.equity.{TableFormat, TableRanges}
import sicfun.holdem.history.{HandHistoryImport, HandHistorySite, OpponentProfile}
import sicfun.holdem.model.{PokerActionModel, PokerActionModelArtifactIO}

import java.nio.file.{Files, Path, Paths}

/** Orchestrates the full profiling validation pipeline:
  *   1. Simulate hands for each leak-injected villain
  *   2. Export PokerStars-format hand histories (full + chunked)
  *   3. Feed through profiling pipeline (OpponentProfile.fromImportedHands)
  *   4. Track convergence — how many hands to detect each leak
  *   5. Produce scorecard
  */
object ValidationRunner:

  final case class Config(
      handsPerPlayer: Int = 1_000_000,
      chunkSize: Int = 1000,
      outputDir: Path = Paths.get("validation-output"),
      modelDir: Option[Path] = None,
      seed: Long = 42L,
      bunchingTrials: Int = 100,
      equityTrials: Int = 500,
      minEquityTrials: Int = 100,
      budgetMs: Long = 50L
  )

  /** All 18 players: 6 leaks x 3 severities. */
  def defaultPopulation: Vector[(InjectedLeak, String)] =
    val severities = Vector(("mild", 0.3), ("moderate", 0.6), ("severe", 0.9))
    val leakFactories: Vector[Double => InjectedLeak] = Vector(
      sev => OverfoldsToAggression(sev),
      sev => Overcalls(sev),
      sev => OverbluffsTurnBarrel(sev),
      sev => PassiveInBigPots(sev),
      sev => PreflopTooLoose(sev),
      sev => PreflopTooTight(sev)
    )
    for
      factory <- leakFactories
      (label, sev) <- severities
    yield
      val leak = factory(sev)
      (leak, s"${leak.id}_$label")

  def main(args: Array[String]): Unit =
    val config = parseConfig(args)
    run(config)

  def run(config: Config): Vector[PlayerValidationResult] =
    Files.createDirectories(config.outputDir)
    val population = defaultPopulation
    println("=== Profiling Validation Harness ===")
    println(s"Players: ${population.size}")
    println(s"Hands per player: ${config.handsPerPlayer}")
    println(s"Total hands: ${population.size.toLong * config.handsPerPlayer}")
    println()

    val results = population.zipWithIndex.map { case ((leak, villainName), idx) =>
      println(s"[${idx + 1}/${population.size}] Simulating $villainName ...")
      val result = runOnePlayer(config, leak, villainName, playerSeed = config.seed + idx)
      println(f"  -> ${result.leakFiredCount} leaks fired, hero EV ${result.heroNetBbPer100}%+.1f bb/100")
      result
    }

    val report = ValidationScorecard.format(results)
    println()
    println(report)

    val reportPath = config.outputDir.resolve("scorecard.txt")
    Files.writeString(reportPath, report)
    println(s"Scorecard saved to $reportPath")

    results

  private def runOnePlayer(
      config: Config,
      leak: InjectedLeak,
      villainName: String,
      playerSeed: Long
  ): PlayerValidationResult =
    val tableRanges = TableRanges.defaults(TableFormat.HeadsUp)
    val actionModel = config.modelDir
      .map(p => PokerActionModelArtifactIO.load(p).model)
      .getOrElse(PokerActionModel.uniform)

    val heroEngine = new RealTimeAdaptiveEngine(
      tableRanges = tableRanges,
      actionModel = actionModel,
      bunchingTrials = config.bunchingTrials,
      defaultEquityTrials = config.equityTrials,
      minEquityTrials = config.minEquityTrials
    )
    val villainEngine = new RealTimeAdaptiveEngine(
      tableRanges = tableRanges,
      actionModel = actionModel,
      bunchingTrials = config.bunchingTrials,
      defaultEquityTrials = config.equityTrials,
      minEquityTrials = config.minEquityTrials
    )
    val villainPlayer = LeakInjectedVillain(
      name = villainName,
      leaks = Vector(leak),
      baselineNoise = 0.03,
      seed = playerSeed
    )
    val simulator = new HeadsUpSimulator(
      heroEngine = heroEngine,
      villainEngine = villainEngine,
      villain = villainPlayer,
      seed = playerSeed,
      budgetMs = config.budgetMs
    )

    // Simulate all hands
    val records = (1 to config.handsPerPlayer).map(i => simulator.playHand(i)).toVector

    // Export files
    val playerDir = config.outputDir.resolve(villainName)
    Files.createDirectories(playerDir)
    val fullText = PokerStarsExporter.exportHands(records, "Hero", villainName)
    Files.writeString(playerDir.resolve("full_history.txt"), fullText)
    val chunks = PokerStarsExporter.exportChunked(records, "Hero", villainName, config.chunkSize)
    chunks.foreach { chunk =>
      Files.writeString(playerDir.resolve(f"chunk_${chunk.chunkIndex}%04d.txt"), chunk.text)
    }

    // Count leak firings
    val leakActions = records.flatMap(_.actions).filter(_.leakId.contains(leak.id))
    val firedCount = leakActions.size
    val applicableEstimate = if leak.severity > 0 then math.round(firedCount / leak.severity).toInt else firedCount

    // Hero EV
    val heroTotalNet = records.map(_.heroNet).sum
    val heroNetBbPer100 = (heroTotalNet / config.handsPerPlayer) * 100.0

    // Ground truth JSON
    val groundTruth = ujson.Obj(
      "leakId" -> ujson.Str(leak.id),
      "leakDescription" -> ujson.Str(leak.description),
      "severity" -> ujson.Num(leak.severity),
      "baselineNoise" -> ujson.Num(0.03),
      "totalHands" -> ujson.Num(config.handsPerPlayer.toDouble),
      "leakFiredCount" -> ujson.Num(firedCount.toDouble),
      "leakApplicableSpots" -> ujson.Num(applicableEstimate.toDouble),
      "heroNetBbPer100" -> ujson.Num(heroNetBbPer100)
    )
    Files.writeString(playerDir.resolve("ground_truth.json"), ujson.write(groundTruth, indent = 2))

    // Profiling pipeline validation — feed chunks through and track convergence
    val tracker = new ConvergenceTracker(leak.id)
    var lastArchetype = "Unknown"
    var archetypeStableChunk: Option[Int] = None
    var prevArchetype = ""

    val accumulatedText = new StringBuilder
    chunks.foreach { chunk =>
      accumulatedText.append(chunk.text)
      val parsed = HandHistoryImport.parseText(accumulatedText.toString(), Some(HandHistorySite.PokerStars), Some("Hero"))
      parsed.foreach { hands =>
        val profiles = OpponentProfile.fromImportedHands("simulated", hands, Set("Hero"))
        profiles.headOption.foreach { profile =>
          val hints = profile.exploitHints
          val archetype = profile.archetypePosterior.mapEstimate.toString
          lastArchetype = archetype
          if archetypeStableChunk.isEmpty && archetype == prevArchetype && prevArchetype.nonEmpty then
            archetypeStableChunk = Some(chunk.chunkIndex)
          prevArchetype = archetype

          val detected = hintMatchesLeak(hints, leak.id)
          val confidence = if detected then 0.8 else 0.1
          tracker.recordChunk(chunk.chunkIndex, detected, confidence, falsePositives = 0)
        }
      }
    }

    // TODO: cluster analysis (secondary) — use PlayerSignature.compute for cluster assignment
    PlayerValidationResult(
      villainName = villainName,
      leakId = leak.id,
      severity = leak.severity,
      totalHands = config.handsPerPlayer,
      leakApplicableSpots = applicableEstimate,
      leakFiredCount = firedCount,
      heroNetBbPer100 = heroNetBbPer100,
      convergence = tracker.summary(config.chunkSize),
      assignedArchetype = lastArchetype,
      archetypeConvergenceChunk = archetypeStableChunk,
      clusterId = None
    )

  /** Check if any exploit hint text matches the injected leak concept. */
  private def hintMatchesLeak(hints: Vector[String], leakId: String): Boolean =
    leakId match
      case "overfold-river-aggression" =>
        hints.exists(h => h.contains("over-fold") || h.contains("bluff pressure") || h.contains("fold"))
      case "overcall-big-bets" =>
        hints.exists(h => h.contains("calling station") || h.contains("Value bet") || h.contains("value bet"))
      case "overbluff-turn-barrel" =>
        hints.exists(h => h.contains("aggressive") || h.contains("bluff-catch"))
      case "passive-big-pots" =>
        hints.exists(h => h.contains("passive") || h.contains("sudden aggression") || h.contains("Respect"))
      case "preflop-too-loose" =>
        hints.exists(h => h.contains("calling station") || h.contains("value bet"))
      case "preflop-too-tight" =>
        hints.exists(h => h.contains("over-fold") || h.contains("wider") || h.contains("Open slightly"))
      case _ => false

  private def parseConfig(args: Array[String]): Config =
    var config = Config()
    args.foreach { arg =>
      if arg.startsWith("--hands=") then
        config = config.copy(handsPerPlayer = arg.stripPrefix("--hands=").toInt)
      else if arg.startsWith("--chunks=") then
        config = config.copy(chunkSize = arg.stripPrefix("--chunks=").toInt)
      else if arg.startsWith("--output=") then
        config = config.copy(outputDir = Paths.get(arg.stripPrefix("--output=")))
      else if arg.startsWith("--model=") then
        config = config.copy(modelDir = Some(Paths.get(arg.stripPrefix("--model="))))
      else if arg.startsWith("--seed=") then
        config = config.copy(seed = arg.stripPrefix("--seed=").toLong)
      else if arg.startsWith("--budget=") then
        config = config.copy(budgetMs = arg.stripPrefix("--budget=").toLong)
    }
    config
