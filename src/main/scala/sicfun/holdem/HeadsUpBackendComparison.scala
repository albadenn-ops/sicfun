package sicfun.holdem

import sicfun.core.HandEvaluator

/** Three-way backend comparison harness: JVM CPU vs native CPU vs native CUDA.
  *
  * Benchmarks all three compute backends on the same batch of matchups, validates
  * result correctness against the JVM CPU baseline, and reports throughput and speedup
  * ratios. Auto-tuning of CUDA parameters is integrated and runs before the comparison
  * unless explicitly disabled or overridden.
  *
  * '''Usage:'''
  * {{{
  * HeadsUpBackendComparison [--key=value ...]
  * }}}
  *
  * '''Options:'''
  *   - `--table=full|canonical` (default: canonical)
  *   - `--mode=exact|mc` (default: mc)
  *   - `--trials=<int>` (default: 200)
  *   - `--maxMatchups=<long>` (default: 4000)
  *   - `--cpuParallelism=<int>` (default: available processors)
  *   - `--warmupRuns=<int>` / `--runs=<int>` / `--seed=<long>`
  *   - `--nativePath=<string>` — explicit path to the native shared library
  *   - `--nativeCpuEngine=<string>` / `--nativeGpuEngine=<string>`
  *   - `--nativeCudaBlockSize=<int>` / `--nativeCudaMaxChunkMatchups=<int>`
  *   - `--nativeAutoTune=<boolean>` (default: true)
  *
  * Exit codes: 0 = PASS (all validations pass), 2 = FAIL.
  */
object HeadsUpBackendComparison:
  private val NativeCudaBlockSizeProperty = "sicfun.gpu.native.cuda.blockSize"
  private val NativeCudaMaxChunkMatchupsProperty = "sicfun.gpu.native.cuda.maxChunkMatchups"

  /** Comparison configuration parsed from CLI arguments. */
  private final case class Config(
      table: String = "canonical",
      mode: String = "mc",
      trials: Int = 200,
      maxMatchups: Long = 4000L,
      cpuParallelism: Int = math.max(1, Runtime.getRuntime.availableProcessors()),
      warmupRuns: Int = 1,
      runs: Int = 2,
      seed: Long = 1L,
      nativePath: Option[String] = None,
      nativeCpuEngine: String = "cpu",
      nativeGpuEngine: String = "cuda",
      nativeCudaBlockSize: Option[Int] = None,
      nativeCudaMaxChunkMatchups: Option[Int] = None,
      nativeAutoTune: Boolean = true
  )

  private final case class BatchData(
      packedKeys: Array[Long],
      keyMaterial: Array[Long]
  ):
    def size: Int = packedKeys.length

  private final case class ValidationSummary(
      checked: Int,
      violations: Int
  ):
    def isPass: Boolean =
      checked > 0 && violations.toDouble / checked.toDouble <= 0.01

  private final case class BackendStats(
      name: String,
      avgSeconds: Double,
      throughput: Double
  )

  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    require(config.cpuParallelism > 0, "cpuParallelism must be positive")
    require(config.warmupRuns >= 0, "warmupRuns must be non-negative")
    require(config.runs > 0, "runs must be positive")
    config.nativeCudaBlockSize.foreach(value => require(value > 0, "nativeCudaBlockSize must be positive"))
    config.nativeCudaMaxChunkMatchups.foreach(value => require(value > 0, "nativeCudaMaxChunkMatchups must be positive"))

    val mode = parseMode(config.mode, config.trials)
    configureNativeRuntime(config)
    maybeAutoTune(config, mode)

    val batch = loadBatch(config.table, config.maxMatchups)
    val availability = HeadsUpGpuRuntime.availability

    println("heads-up backend comparison")
    println(
      s"table=${config.table}, mode=${config.mode}, trials=${config.trials}, maxMatchups=${config.maxMatchups}, " +
        s"entries=${batch.size}, cpuParallelism=${config.cpuParallelism}, warmupRuns=${config.warmupRuns}, runs=${config.runs}"
    )
    println(
      s"nativePath=${config.nativePath.getOrElse("(runtime default)")}, " +
        s"nativeCpuEngine=${config.nativeCpuEngine}, nativeGpuEngine=${config.nativeGpuEngine}, " +
        s"nativeCudaBlockSize=${config.nativeCudaBlockSize.map(_.toString).getOrElse("(runtime default)")}, " +
        s"nativeCudaMaxChunkMatchups=${config.nativeCudaMaxChunkMatchups.map(_.toString).getOrElse("(runtime default)")}, " +
        s"nativeAutoTune=${config.nativeAutoTune}"
    )
    println(
      s"effectiveNativeCudaBlockSize=${sys.props.get(NativeCudaBlockSizeProperty).getOrElse("(runtime default)")}, " +
        s"effectiveNativeCudaMaxChunkMatchups=${sys.props.get(NativeCudaMaxChunkMatchupsProperty).getOrElse("(runtime default)")}"
    )
    println(s"runtimeProvider=${availability.provider}, runtimeAvailable=${availability.available}")
    println(s"runtimeDetail=${availability.detail}")

    if !availability.available then
      println("comparison=FAIL reason=provider_unavailable")
      sys.exit(2)

    var warmup = 0
    while warmup < config.warmupRuns do
      val seedBase = config.seed + warmup.toLong
      runJvmCpu(batch, mode, config.cpuParallelism, seedBase)
      runNative(batch, mode, seedBase, config.nativeCpuEngine) match
        case Left(reason) =>
          println(s"comparison=FAIL reason=native_cpu_error detail=$reason")
          sys.exit(2)
        case Right(_) => ()
      runNative(batch, mode, seedBase, config.nativeGpuEngine) match
        case Left(reason) =>
          println(s"comparison=FAIL reason=native_gpu_error detail=$reason")
          sys.exit(2)
        case Right(_) => ()
      warmup += 1

    val jvmSeconds = new Array[Double](config.runs)
    val nativeCpuSeconds = new Array[Double](config.runs)
    val nativeGpuSeconds = new Array[Double](config.runs)
    var nativeCpuValidation = ValidationSummary(checked = 0, violations = 0)
    var nativeGpuValidation = ValidationSummary(checked = 0, violations = 0)

    var run = 0
    while run < config.runs do
      val seedBase = config.seed + 1000L + run.toLong
      val (jvmResults, jvmElapsed) = runJvmCpu(batch, mode, config.cpuParallelism, seedBase)

      val (nativeCpuResults, nativeCpuElapsed) =
        runNative(batch, mode, seedBase, config.nativeCpuEngine) match
          case Left(reason) =>
            println(s"comparison=FAIL reason=native_cpu_error detail=$reason")
            sys.exit(2)
            (Array.empty[EquityResultWithError], 0.0)
          case Right(value) => value

      val (nativeGpuResults, nativeGpuElapsed) =
        runNative(batch, mode, seedBase, config.nativeGpuEngine) match
          case Left(reason) =>
            println(s"comparison=FAIL reason=native_gpu_error detail=$reason")
            sys.exit(2)
            (Array.empty[EquityResultWithError], 0.0)
          case Right(value) => value

      jvmSeconds(run) = jvmElapsed
      nativeCpuSeconds(run) = nativeCpuElapsed
      nativeGpuSeconds(run) = nativeGpuElapsed
      nativeCpuValidation = mergeValidation(nativeCpuValidation, validate(mode, jvmResults, nativeCpuResults))
      nativeGpuValidation = mergeValidation(nativeGpuValidation, validate(mode, jvmResults, nativeGpuResults))
      run += 1

    val jvmStats = toStats("jvm-cpu", batch.size, jvmSeconds)
    val nativeCpuStats = toStats("native-cpu", batch.size, nativeCpuSeconds)
    val nativeGpuStats = toStats("native-cuda", batch.size, nativeGpuSeconds)

    val nativeCpuVsJvm = nativeCpuStats.throughput / jvmStats.throughput
    val nativeGpuVsJvm = nativeGpuStats.throughput / jvmStats.throughput
    val nativeGpuVsNativeCpu = nativeGpuStats.throughput / nativeCpuStats.throughput
    val allValid = nativeCpuValidation.isPass && nativeGpuValidation.isPass

    printStats(jvmStats)
    printStats(nativeCpuStats)
    printStats(nativeGpuStats)
    println(
      f"speedup(native-cpu/jvm)=${nativeCpuVsJvm}%.2fx " +
        f"speedup(native-cuda/jvm)=${nativeGpuVsJvm}%.2fx " +
        f"speedup(native-cuda/native-cpu)=${nativeGpuVsNativeCpu}%.2fx"
    )
    println(s"validation(native-cpu)=${nativeCpuValidation.violations}/${nativeCpuValidation.checked}")
    println(s"validation(native-cuda)=${nativeGpuValidation.violations}/${nativeGpuValidation.checked}")
    val lastTelemetry = HeadsUpGpuRuntime.lastBatchTelemetry
      .map(t => s"provider=${t.provider}, success=${t.success}, detail=${t.detail}")
      .getOrElse("none")
    println(s"lastTelemetry=$lastTelemetry")
    println(s"comparison=${if allValid then "PASS" else "FAIL"}")

    if allValid then sys.exit(0) else sys.exit(2)

  private def configureNativeRuntime(config: Config): Unit =
    sys.props.update("sicfun.gpu.provider", "native")
    config.nativePath.foreach(path => sys.props.update("sicfun.gpu.native.path", path))
    config.nativeCudaBlockSize match
      case Some(value) => sys.props.update(NativeCudaBlockSizeProperty, value.toString)
      case None => sys.props.remove(NativeCudaBlockSizeProperty)
    config.nativeCudaMaxChunkMatchups match
      case Some(value) => sys.props.update(NativeCudaMaxChunkMatchupsProperty, value.toString)
      case None => sys.props.remove(NativeCudaMaxChunkMatchupsProperty)

  private def maybeAutoTune(config: Config, mode: HeadsUpEquityTable.Mode): Unit =
    if !config.nativeAutoTune then
      println("gpu-autotune: disabled for backend comparison (--nativeAutoTune=false)")
    else if config.nativeGpuEngine != "cuda" then
      println(s"gpu-autotune: skipped for backend comparison (nativeGpuEngine=${config.nativeGpuEngine})")
    else if config.nativeCudaBlockSize.nonEmpty || config.nativeCudaMaxChunkMatchups.nonEmpty then
      println("gpu-autotune: skipped for backend comparison (explicit nativeCudaBlockSize/nativeCudaMaxChunkMatchups)")
    else
      HeadsUpBackendAutoTuner.configureForComparison(
        tableKind = config.table,
        mode = mode,
        maxMatchups = config.maxMatchups
      )

  private def loadBatch(table: String, maxMatchups: Long): BatchData =
    table.trim.toLowerCase match
      case "full" =>
        val batch = HeadsUpEquityTable.selectFullBatch(maxMatchups)
        BatchData(batch.packedKeys, batch.keyMaterial)
      case "canonical" =>
        val batch = HeadsUpEquityCanonicalTable.selectCanonicalBatch(maxMatchups)
        BatchData(batch.packedKeys, batch.keyMaterial)
      case other =>
        throw new IllegalArgumentException(s"unknown table '$other' (expected full or canonical)")

  private def runJvmCpu(
      batch: BatchData,
      mode: HeadsUpEquityTable.Mode,
      parallelism: Int,
      seedBase: Long
  ): (Array[EquityResultWithError], Double) =
    HandEvaluator.clearCaches()
    val started = System.nanoTime()
    val result =
      HeadsUpEquityTable.computeBatchCpu(
        mode = mode,
        packedKeys = batch.packedKeys,
        keyMaterial = batch.keyMaterial,
        parallelism = parallelism,
        monteCarloSeedBase = seedBase,
        progress = None
      )
    val elapsed = (System.nanoTime() - started).toDouble / 1_000_000_000.0
    (result, elapsed)

  private def runNative(
      batch: BatchData,
      mode: HeadsUpEquityTable.Mode,
      seedBase: Long,
      engine: String
  ): Either[String, (Array[EquityResultWithError], Double)] =
    sys.props.update("sicfun.gpu.native.engine", engine.trim.toLowerCase)
    HandEvaluator.clearCaches()
    val started = System.nanoTime()
    HeadsUpGpuRuntime.computeBatch(batch.packedKeys, batch.keyMaterial, mode, seedBase).map { values =>
      val elapsed = (System.nanoTime() - started).toDouble / 1_000_000_000.0
      (values, elapsed)
    }

  private def validate(
      mode: HeadsUpEquityTable.Mode,
      baseline: Array[EquityResultWithError],
      candidate: Array[EquityResultWithError]
  ): ValidationSummary =
    require(baseline.length == candidate.length, "baseline and candidate result lengths must match")
    var checked = 0
    var violations = 0
    var idx = 0
    while idx < baseline.length do
      val b = baseline(idx)
      val c = candidate(idx)
      val bad =
        mode match
          case HeadsUpEquityTable.Mode.Exact =>
            math.abs(b.win - c.win) > 1e-9 ||
            math.abs(b.tie - c.tie) > 1e-9 ||
            math.abs(b.loss - c.loss) > 1e-9
          case HeadsUpEquityTable.Mode.MonteCarlo(_) =>
            val diff = math.abs(b.equity - c.equity)
            val stderrBound = math.max(b.stderr, c.stderr) * 6.0
            val tolerance = math.max(stderrBound, 0.05)
            diff > tolerance
      checked += 1
      if bad then violations += 1
      idx += 1
    ValidationSummary(checked = checked, violations = violations)

  private def mergeValidation(a: ValidationSummary, b: ValidationSummary): ValidationSummary =
    ValidationSummary(checked = a.checked + b.checked, violations = a.violations + b.violations)

  private def toStats(name: String, entries: Int, seconds: Array[Double]): BackendStats =
    val avgSeconds = seconds.sum / seconds.length.toDouble
    val throughput = entries.toDouble / avgSeconds
    BackendStats(name = name, avgSeconds = avgSeconds, throughput = throughput)

  private def printStats(stats: BackendStats): Unit =
    println(
      f"backend=${stats.name} avg=${stats.avgSeconds}%.3fs throughput=${stats.throughput}%.1f matchups/s"
    )

  private def parseMode(mode: String, trials: Int): HeadsUpEquityTable.Mode =
    mode.trim.toLowerCase match
      case "exact" => HeadsUpEquityTable.Mode.Exact
      case "mc" | "montecarlo" => HeadsUpEquityTable.Mode.MonteCarlo(trials)
      case other => throw new IllegalArgumentException(s"unknown mode '$other' (expected exact or mc)")

  private def parseArgs(args: Vector[String]): Config =
    args.foldLeft(Config()) { (cfg, raw) =>
      val trimmed = raw.trim
      if !trimmed.startsWith("--") || !trimmed.contains("=") then
        throw new IllegalArgumentException(s"invalid argument '$raw' (expected --key=value)")
      val eq = trimmed.indexOf('=')
      val key = trimmed.substring(2, eq).trim
      val value = trimmed.substring(eq + 1).trim
      key match
        case "table" => cfg.copy(table = value.toLowerCase)
        case "mode" => cfg.copy(mode = value.toLowerCase)
        case "trials" => cfg.copy(trials = value.toInt)
        case "maxMatchups" => cfg.copy(maxMatchups = value.toLong)
        case "cpuParallelism" => cfg.copy(cpuParallelism = value.toInt)
        case "warmupRuns" => cfg.copy(warmupRuns = value.toInt)
        case "runs" => cfg.copy(runs = value.toInt)
        case "seed" => cfg.copy(seed = value.toLong)
        case "nativePath" => cfg.copy(nativePath = Some(value))
        case "nativeCpuEngine" => cfg.copy(nativeCpuEngine = value.toLowerCase)
        case "nativeGpuEngine" => cfg.copy(nativeGpuEngine = value.toLowerCase)
        case "nativeCudaBlockSize" => cfg.copy(nativeCudaBlockSize = Some(value.toInt))
        case "nativeCudaMaxChunkMatchups" => cfg.copy(nativeCudaMaxChunkMatchups = Some(value.toInt))
        case "nativeAutoTune" => cfg.copy(nativeAutoTune = parseBoolean(value, "nativeAutoTune"))
        case other => throw new IllegalArgumentException(s"unknown option '$other'")
    }

  private def parseBoolean(raw: String, key: String): Boolean =
    raw.trim.toLowerCase match
      case "1" | "true" | "yes" | "on" => true
      case "0" | "false" | "no" | "off" => false
      case other =>
        throw new IllegalArgumentException(s"invalid boolean for $key: '$other' (expected true/false)")
