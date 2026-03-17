package sicfun.holdem.validation

import sicfun.holdem.engine.RealTimeAdaptiveEngine
import sicfun.holdem.equity.{TableFormat, TableRanges}
import sicfun.holdem.history.{HandHistoryImport, HandHistorySite, OpponentProfile}
import sicfun.holdem.model.{PokerActionModel, PokerActionModelArtifactIO}
import sicfun.holdem.types.{PokerAction, Street}
import sicfun.holdem.web.HandHistoryReviewServer

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
      convergenceStep: Int = 100,
      outputDir: Path = Paths.get("validation-output"),
      modelDir: Option[Path] = None,
      seed: Long = 42L,
      bunchingTrials: Int = 100,
      equityTrials: Int = 500,
      minEquityTrials: Int = 100,
      budgetMs: Long = 50L,
      fastHero: Boolean = true,
      severityFilter: Option[String] = None
  )

  /** All 19 players: 6 leaks x 3 severities + 1 GTO control. */
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
    val leakyPlayers = for
      factory <- leakFactories
      (label, sev) <- severities
    yield
      val leak = factory(sev)
      (leak, s"${leak.id}_$label")
    // GTO control: no leak, no noise — false positive canary
    leakyPlayers :+ (NoLeak(), "gto-baseline")

  def main(args: Array[String]): Unit =
    val config = parseConfig(args)
    if args.contains("--web-spotcheck") then
      val results = run(config)
      runWebSpotCheck(config, results)
    else
      run(config)

  def run(config: Config): Vector[PlayerValidationResult] =
    Files.createDirectories(config.outputDir)
    val population = config.severityFilter match
      case Some(sev) => defaultPopulation.filter(_._2.endsWith(s"_$sev"))
      case None      => defaultPopulation
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
    val heroEngine = if config.fastHero then None
    else
      val tableRanges = TableRanges.defaults(TableFormat.HeadsUp)
      val actionModel = config.modelDir
        .map(p => PokerActionModelArtifactIO.load(p).model)
        .getOrElse(PokerActionModel.uniform)
      Some(new RealTimeAdaptiveEngine(
        tableRanges = tableRanges,
        actionModel = actionModel,
        bunchingTrials = config.bunchingTrials,
        defaultEquityTrials = config.equityTrials,
        minEquityTrials = config.minEquityTrials
      ))

    val isGtoControl = leak.id == "gto-baseline"
    val villainPlayer = LeakInjectedVillain(
      name = villainName,
      leaks = Vector(leak),
      baselineNoise = if isGtoControl then 0.0 else 0.03,
      seed = playerSeed
    )
    val simulator = new HeadsUpSimulator(
      heroEngine = heroEngine,
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

    // Count leak firings and actual applicable spots
    val leakActions = records.flatMap(_.actions).filter(_.leakId.contains(leak.id))
    val firedCount = leakActions.size
    val applicableCount = records.map(_.leakApplicableSpots).sum

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
      "leakApplicableSpots" -> ujson.Num(applicableCount.toDouble),
      "heroNetBbPer100" -> ujson.Num(heroNetBbPer100)
    )
    Files.writeString(playerDir.resolve("ground_truth.json"), ujson.write(groundTruth, indent = 2))

    // === CONVERGENCE ANALYSIS (decoupled from export) ===
    // Parse in chunks (resilient — one bad hand only loses its chunk, not all data),
    // then profile at fine-grained convergence steps.
    val tracker = new ConvergenceTracker(leak.id)
    var lastArchetype = "Unknown"
    var archetypeStableStep: Option[Int] = None
    var prevArchetype = ""

    var parseFailures = 0
    val allParsedHands = Vector.newBuilder[sicfun.holdem.history.ImportedHand]
    chunks.foreach { chunk =>
      HandHistoryImport.parseText(chunk.text, Some(HandHistorySite.PokerStars), Some("Hero")) match
        case Left(err) =>
          parseFailures += 1
          if parseFailures <= 3 then
            System.err.println(s"  [WARN] Parse failure at chunk ${chunk.chunkIndex} for $villainName: $err")
        case Right(hands) =>
          allParsedHands ++= hands
    }
    val parsedHands = allParsedHands.result()

    val step = config.convergenceStep
    val totalSteps = if parsedHands.nonEmpty then
      (parsedHands.size + step - 1) / step else 0

    for stepIdx <- 0 until totalSteps do
      val handsInWindow = math.min((stepIdx + 1) * step, parsedHands.size)
      val windowHands = parsedHands.take(handsInWindow)
      val profiles = OpponentProfile.fromImportedHands("simulated", windowHands, Set("Hero"))
      profiles.headOption.foreach { profile =>
        val hints = profile.exploitHints
        val archetype = profile.archetypePosterior.mapEstimate.toString
        lastArchetype = archetype
        if archetypeStableStep.isEmpty && archetype == prevArchetype && prevArchetype.nonEmpty then
          archetypeStableStep = Some(stepIdx)
        prevArchetype = archetype

        val detected = hintMatchesLeak(hints, leak.id)
        val confidence = if detected then 0.8 else 0.1
        tracker.recordChunk(stepIdx, detected, confidence, falsePositives = 0)
        // Debug: last step stats for undetected leaks
        if stepIdx == totalSteps - 1 && !detected then
          val evts = profile.recentEvents
          val turnEvts = evts.filter(_.street == Street.Turn)
          val turnRaises = turnEvts.count(_.action.category == PokerAction.Category.Raise)
          val riverFacing = evts.filter(e => e.street == Street.River && e.toCall > 0)
          val riverFolds = riverFacing.count(_.action == PokerAction.Fold)
          println(f"  [DBG] $villainName: evts=${evts.size} turn(n=${turnEvts.size} r=$turnRaises) rvr(facing=${riverFacing.size} folds=$riverFolds) hints=$hints")
      }

    // TODO: cluster analysis (secondary) — use PlayerSignature.compute for cluster assignment
    PlayerValidationResult(
      villainName = villainName,
      leakId = leak.id,
      severity = leak.severity,
      totalHands = config.handsPerPlayer,
      leakApplicableSpots = applicableCount,
      leakFiredCount = firedCount,
      heroNetBbPer100 = heroNetBbPer100,
      convergence = tracker.summary(step),
      assignedArchetype = lastArchetype,
      archetypeConvergenceChunk = archetypeStableStep,
      clusterId = None
    )

  /** Start the web review server with sample validation chunks for visual spot-checking.
    *
    * After a validation run, this picks one "severe" player per leak type and
    * serves their chunk files through HandHistoryReviewServer so the user can
    * visually inspect profiling reports in the browser.
    */
  def runWebSpotCheck(config: Config, results: Vector[PlayerValidationResult]): Unit =
    // Pick severe variants for spot-checking (most visible leaks)
    val severeResults = results.filter(_.severity >= 0.9)
    if severeResults.isEmpty then
      println("No severe-severity players to spot-check.")
      return

    // Copy sample chunks to a spot-check directory the web UI can reference
    val spotCheckDir = config.outputDir.resolve("spot-check")
    Files.createDirectories(spotCheckDir)
    severeResults.foreach { r =>
      val playerDir = config.outputDir.resolve(r.villainName)
      val chunkFile = playerDir.resolve("chunk_0000.txt")
      if Files.exists(chunkFile) then
        val dest = spotCheckDir.resolve(s"${r.villainName}_chunk_0000.txt")
        Files.copy(chunkFile, dest, java.nio.file.StandardCopyOption.REPLACE_EXISTING)
    }

    println()
    println("=== Web Spot-Check Mode ===")
    println(s"Starting HandHistoryReviewServer...")
    println(s"Upload files from: ${spotCheckDir.toAbsolutePath.normalize()}")
    println()

    val serverArgs = Array(
      s"--port=8090",
      s"--staticDir=docs/site-preview-hybrid",
      s"--bunchingTrials=${config.bunchingTrials}",
      s"--equityTrials=${config.equityTrials}",
      s"--budgetMs=${config.budgetMs}"
    )
    HandHistoryReviewServer.start(serverArgs) match
      case Right(server) =>
        println(s"Web server running at http://127.0.0.1:8090/")
        println(s"Upload the chunk files above to analyze in the browser.")
        println("Press Ctrl+C to stop.")
        try Thread.currentThread().join()
        catch case _: InterruptedException => ()
        finally server.close()
      case Left(err) =>
        println(s"Failed to start web server: $err")

  /** Check if any exploit hint text matches the injected leak concept.
    *
    * Uses specific profiler hint phrases — not generic keywords — to avoid
    * false positive matches on normal profiling language.
    */
  private def hintMatchesLeak(hints: Vector[String], leakId: String): Boolean =
    leakId match
      case "overfold-river-aggression" =>
        hints.exists(_.contains("Over-folds on the river"))
      case "overcall-big-bets" =>
        hints.exists(h => h.contains("calling station") || h.contains("Calls too often facing large bets"))
      case "overbluff-turn-barrel" =>
        hints.exists(_.contains("Very aggressive on the turn"))
      case "passive-big-pots" =>
        // Not detectable by action frequency — GTO checks ~100% in big pots.
        // Requires hand-strength-aware analysis to distinguish from equilibrium.
        false
      case "preflop-too-loose" =>
        hints.exists(_.contains("Calls too loose preflop"))
      case "preflop-too-tight" =>
        hints.exists(_.contains("Over-folds preflop"))
      case "gto-baseline" =>
        // False positive check: does the profiler incorrectly match ANY leak pattern?
        val leakIds = Vector("overfold-river-aggression", "overcall-big-bets", "overbluff-turn-barrel",
          "preflop-too-loose", "preflop-too-tight")
        leakIds.exists(id => hintMatchesLeak(hints, id))
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
      else if arg == "--slow-hero" then
        config = config.copy(fastHero = false)
      else if arg.startsWith("--severity=") then
        config = config.copy(severityFilter = Some(arg.stripPrefix("--severity=")))
      else if arg.startsWith("--step=") then
        config = config.copy(convergenceStep = arg.stripPrefix("--step=").toInt)
    }
    config
