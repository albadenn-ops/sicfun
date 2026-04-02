package sicfun.holdem.bench

import sicfun.holdem.cfr.{HoldemCfrConfig, HoldemCfrNativeRuntime, HoldemCfrSolver}
import sicfun.holdem.cli.CliHelpers
import sicfun.holdem.types.ScopedRuntimeProperties
import sicfun.holdem.validation.{CfrVillainStrategy, HeadsUpSimulator, LeakInjectedVillain, NoLeak}

/** End-to-end calibration-style workload benchmark for CFR villain decisions.
  *
  * Runs the same self-play loop used by the validation harness, but aggregates
  * cache behavior and native solver stage timings over many hands.
  */
object HoldemCfrCalibrationWorkloadBenchmark:
  private final case class Config(
      hands: Int = 10_000,
      provider: String = "auto",
      seed: Long = 42L
  )

  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    validateConfig(config)
    val propertyUpdates = Seq(
      "sicfun.cfr.provider" -> Some(config.provider),
      "sicfun.cfr.directShallowApproximation" -> Some("false")
    )

    ScopedRuntimeProperties.withOverrides(propertyUpdates) {
      HoldemCfrNativeRuntime.resetLoadCacheForTests()
      HoldemCfrSolver.resetAutoProviderForTests()
      try
        run(config)
      finally
        HoldemCfrNativeRuntime.resetLoadCacheForTests()
        HoldemCfrSolver.resetAutoProviderForTests()
    }

  private def run(config: Config): Unit =
    val cfrConfig = HoldemCfrConfig(
      iterations = 500,
      equityTrials = 1_000,
      maxVillainHands = 64,
      includeVillainReraises = true
    )
    val cfrStrategy = CfrVillainStrategy(
      config = cfrConfig,
      allowHeuristicFallback = false,
      collectSolveTiming = true
    )
    val villain = LeakInjectedVillain(
      name = "cfr_gto_control",
      leaks = Vector(NoLeak()),
      baselineNoise = 0.0,
      seed = config.seed
    )
    val simulator = new HeadsUpSimulator(
      heroEngine = None,
      villain = villain,
      seed = config.seed,
      villainStrategy = cfrStrategy
    )

    println("=== Holdem CFR Calibration Workload Benchmark ===")
    println(s"config: hands=${config.hands}, provider=${config.provider}, seed=${config.seed}")

    val started = System.nanoTime()
    var handNumber = 1
    while handNumber <= config.hands do
      simulator.playHand(handNumber)
      handNumber += 1
    val wallNanos = math.max(1L, System.nanoTime() - started)

    val cacheStats = cfrStrategy.cacheStatsSnapshot
    val servedProviders = cfrStrategy.providerCountsSnapshot
    val solvedProviders = cfrStrategy.solvedProviderCountsSnapshot
    val solveTiming = cfrStrategy.solveTimingSnapshot
    val totalServed = servedProviders.values.sum
    val cfrSolvedProviders = solvedProviders.removed("direct")
    val cfrMissSolveCount = cfrSolvedProviders.values.sum
    val missWallByProviderMs =
      solveTiming.solveWallByProviderNanos.toVector.sortBy(_._1).map { case (provider, nanos) =>
        provider -> (nanos.toDouble / 1_000_000.0)
      }.toMap
    val cfrMissSolveWallNanos =
      solveTiming.solveWallByProviderNanos.iterator.filterNot(_._1 == "direct").map(_._2).sum
    val cacheHitRate =
      if totalServed > 0L then cacheStats.hits.toDouble / totalServed.toDouble
      else 0.0
    val wallSeconds = wallNanos.toDouble / 1_000_000_000.0
    val handsPerSecond = config.hands.toDouble / wallSeconds
    val missSolveWallMs = solveTiming.solveWallNanos.toDouble / 1_000_000.0
    val cfrMissSolveWallMs = cfrMissSolveWallNanos.toDouble / 1_000_000.0
    val avgCfrMissSolveWallMs =
      if cfrMissSolveCount > 0L then cfrMissSolveWallMs / cfrMissSolveCount.toDouble else 0.0
    val profiledNativeTotalMs = solveTiming.profiledNativeTotalNanos.toDouble / 1_000_000.0
    val profiledCoverage =
      if cfrMissSolveWallNanos > 0L then
        solveTiming.profiledNativeTotalNanos.toDouble / cfrMissSolveWallNanos.toDouble
      else 0.0
    val nativeSolveShare =
      if solveTiming.profiledNativeTotalNanos > 0L then
        solveTiming.nativeSolveNanos.toDouble / solveTiming.profiledNativeTotalNanos.toDouble
      else 0.0
    val jvmMeasuredShare =
      if solveTiming.profiledNativeTotalNanos > 0L then
        (solveTiming.nativePrepareNanos + solveTiming.nativeSpecBuildNanos + solveTiming.nativeUnpackNanos).toDouble /
          solveTiming.profiledNativeTotalNanos.toDouble
      else 0.0

    println(f"wall=${wallSeconds}%.3fs handsPerSecond=${handsPerSecond}%.2f")
    println(s"servedProviders=$servedProviders")
    println(s"solvedProviders=$solvedProviders")
    println(
      f"policyCache: hits=${cacheStats.hits}%d misses=${cacheStats.misses}%d size=${cacheStats.size}%d " +
        f"hitRate=${cacheHitRate * 100.0}%.2f%%"
    )
    println(
      f"missSolveWall: solves=${solveTiming.solveCount}%d totalMs=${missSolveWallMs}%.3f " +
        f"byProviderMs=$missWallByProviderMs"
    )
    println(
      f"cfrMissSolveWall: solves=${cfrMissSolveCount}%d totalMs=${cfrMissSolveWallMs}%.3f " +
        f"avgMs=${avgCfrMissSolveWallMs}%.3f"
    )
    println(
      f"profiledNative: solves=${solveTiming.nativeProfileCount}%d totalMs=${profiledNativeTotalMs}%.3f " +
        f"coverageVsCfrMissWall=${profiledCoverage * 100.0}%.2f%%"
    )
    println(
      f"profiledStagesMs: prepare=${solveTiming.nativePrepareNanos / 1e6}%.3f " +
        f"spec=${solveTiming.nativeSpecBuildNanos / 1e6}%.3f " +
        f"nativeSolve=${solveTiming.nativeSolveNanos / 1e6}%.3f " +
        f"unpack=${solveTiming.nativeUnpackNanos / 1e6}%.3f"
    )
    println(
      f"profiledPrepareMs: support=${solveTiming.nativePrepareSupportNanos / 1e6}%.3f " +
        f"equity=${solveTiming.nativePrepareEquityNanos / 1e6}%.3f " +
        f"responses=${solveTiming.nativePrepareResponseNanos / 1e6}%.3f " +
        f"gameBuild=${solveTiming.nativePrepareGameBuildNanos / 1e6}%.3f"
    )
    println(
      f"profiledShares: nativeSolve=${nativeSolveShare * 100.0}%.2f%% " +
        f"jvmMeasured=${jvmMeasuredShare * 100.0}%.2f%%"
    )
    println("=== Done ===")

  private def parseArgs(args: Vector[String]): Config =
    val options = CliHelpers.requireOptions(args)
    CliHelpers.requireNoUnknownOptions(options, Set("hands", "provider", "seed"))
    Config(
      hands = CliHelpers.requireIntOption(options, "hands", 10_000),
      provider = options.getOrElse("provider", "auto").trim.toLowerCase(java.util.Locale.ROOT),
      seed = CliHelpers.requireLongOption(options, "seed", 42L)
    )

  private def validateConfig(config: Config): Unit =
    require(config.hands > 0, "hands must be positive")
    config.provider match
      case "auto" | "scala" | "native-cpu" | "native-gpu" => ()
      case other => throw new IllegalArgumentException(s"unsupported provider: $other")
