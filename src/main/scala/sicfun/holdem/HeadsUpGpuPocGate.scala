package sicfun.holdem

import sicfun.core.HandEvaluator

/** GPU proof-of-concept gate: validates that the GPU backend produces correct results
  * and achieves a minimum speedup over multi-threaded CPU computation.
  *
  * The gate runs both CPU and GPU backends on the same batch of matchups, then:
  *   1. Validates correctness by comparing equity results (exact mode: 1e-9 tolerance;
  *      Monte Carlo: 6-sigma + 0.05 absolute tolerance).
  *   2. Measures throughput speedup of GPU over CPU.
  *   3. Passes only if validation has <= 1% violations AND speedup meets the threshold.
  *
  * Exit codes: 0 = PASS, 2 = FAIL.
  *
  * '''Usage:'''
  * {{{
  * HeadsUpGpuPocGate [--key=value ...]
  * }}}
  *
  * '''Options:'''
  *   - `--table=full|canonical` (default: full)
  *   - `--mode=exact|mc` (default: mc)
  *   - `--trials=<int>` (default: 200)
  *   - `--maxMatchups=<long>` (default: 2000)
  *   - `--cpuParallelism=<int>` (default: available processors)
  *   - `--speedupThreshold=<double>` (default: 5.0)
  *   - `--warmupRuns=<int>` / `--runs=<int>` / `--seed=<long>`
  */
object HeadsUpGpuPocGate:
  /** Gate configuration parsed from CLI arguments. */
  private final case class Config(
      table: String = "full",
      mode: String = "mc",
      trials: Int = 200,
      maxMatchups: Long = 2000L,
      cpuParallelism: Int = math.max(1, Runtime.getRuntime.availableProcessors()),
      speedupThreshold: Double = 5.0,
      warmupRuns: Int = 1,
      runs: Int = 2,
      seed: Long = 1L
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

  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    require(config.cpuParallelism > 0, "cpuParallelism must be positive")
    require(config.speedupThreshold > 0.0, "speedupThreshold must be positive")
    require(config.runs > 0, "runs must be positive")
    require(config.warmupRuns >= 0, "warmupRuns must be non-negative")

    val mode = parseMode(config.mode, config.trials)
    val batch = loadBatch(config.table, config.maxMatchups)
    val availability = HeadsUpGpuRuntime.availability

    println("gpu poc gate")
    println(
      s"table=${config.table}, mode=${config.mode}, trials=${config.trials}, maxMatchups=${config.maxMatchups}, " +
        s"entries=${batch.size}, cpuParallelism=${config.cpuParallelism}, speedupThreshold=${config.speedupThreshold}x, " +
        s"provider=${availability.provider}, providerAvailable=${availability.available}"
    )
    println(s"providerDetail=${availability.detail}")

    if !availability.available then
      println("gate=FAIL reason=provider_unavailable")
      sys.exit(2)

    var warmup = 0
    while warmup < config.warmupRuns do
      val seedBase = config.seed + warmup.toLong
      runCpu(batch, mode, config.cpuParallelism, seedBase)
      runGpu(batch, mode, seedBase) match
        case Left(reason) =>
          println(s"gate=FAIL reason=gpu_runtime_error detail=$reason")
          sys.exit(2)
        case Right(_) => ()
      warmup += 1

    val cpuSeconds = new Array[Double](config.runs)
    val gpuSeconds = new Array[Double](config.runs)
    var validation = ValidationSummary(checked = 0, violations = 0)
    var run = 0
    while run < config.runs do
      val seedBase = config.seed + 1000L + run.toLong

      val (cpuResults, cpuElapsed) = runCpu(batch, mode, config.cpuParallelism, seedBase)
      val (gpuResults, gpuElapsed) =
        runGpu(batch, mode, seedBase) match
          case Left(reason) =>
            println(s"gate=FAIL reason=gpu_runtime_error detail=$reason")
            sys.exit(2)
            (Array.empty[EquityResultWithError], 0.0)
          case Right(value) => value

      cpuSeconds(run) = cpuElapsed
      gpuSeconds(run) = gpuElapsed
      validation = mergeValidation(validation, validate(mode, cpuResults, gpuResults))
      run += 1

    val cpuAvg = cpuSeconds.sum / cpuSeconds.length.toDouble
    val gpuAvg = gpuSeconds.sum / gpuSeconds.length.toDouble
    val cpuThroughput = batch.size.toDouble / cpuAvg
    val gpuThroughput = batch.size.toDouble / gpuAvg
    val speedup = gpuThroughput / cpuThroughput
    val gatePass = validation.isPass && speedup >= config.speedupThreshold

    println(f"cpuAvg=${cpuAvg}%.3fs cpuThroughput=${cpuThroughput}%.1f matchups/s")
    println(f"gpuAvg=${gpuAvg}%.3fs gpuThroughput=${gpuThroughput}%.1f matchups/s")
    println(f"speedup=${speedup}%.2fx validationViolations=${validation.violations}/${validation.checked}")
    val lastTelemetry = HeadsUpGpuRuntime.lastBatchTelemetry
      .map(t => s"provider=${t.provider}, success=${t.success}, detail=${t.detail}")
      .getOrElse("none")
    println(s"gpuTelemetry=$lastTelemetry")
    println(s"gate=${if gatePass then "PASS" else "FAIL"}")

    if gatePass then sys.exit(0) else sys.exit(2)

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

  private def runCpu(
      batch: BatchData,
      mode: HeadsUpEquityTable.Mode,
      cpuParallelism: Int,
      seedBase: Long
  ): (Array[EquityResultWithError], Double) =
    HandEvaluator.clearCaches()
    val started = System.nanoTime()
    val result =
      HeadsUpEquityTable.computeBatchCpu(
        mode = mode,
        packedKeys = batch.packedKeys,
        keyMaterial = batch.keyMaterial,
        parallelism = cpuParallelism,
        monteCarloSeedBase = seedBase,
        progress = None
      )
    val elapsed = (System.nanoTime() - started).toDouble / 1_000_000_000.0
    (result, elapsed)

  private def runGpu(
      batch: BatchData,
      mode: HeadsUpEquityTable.Mode,
      seedBase: Long
  ): Either[String, (Array[EquityResultWithError], Double)] =
    HandEvaluator.clearCaches()
    val started = System.nanoTime()
    HeadsUpGpuRuntime.computeBatch(batch.packedKeys, batch.keyMaterial, mode, seedBase).map { values =>
      val elapsed = (System.nanoTime() - started).toDouble / 1_000_000_000.0
      (values, elapsed)
    }

  private def validate(
      mode: HeadsUpEquityTable.Mode,
      cpu: Array[EquityResultWithError],
      gpu: Array[EquityResultWithError]
  ): ValidationSummary =
    require(cpu.length == gpu.length, "CPU and GPU result lengths must match")
    var checked = 0
    var violations = 0
    var idx = 0
    while idx < cpu.length do
      val c = cpu(idx)
      val g = gpu(idx)
      val bad =
        mode match
          case HeadsUpEquityTable.Mode.Exact =>
            math.abs(c.win - g.win) > 1e-9 ||
            math.abs(c.tie - g.tie) > 1e-9 ||
            math.abs(c.loss - g.loss) > 1e-9
          case HeadsUpEquityTable.Mode.MonteCarlo(_) =>
            val diff = math.abs(c.equity - g.equity)
            val stderrBound = math.max(c.stderr, g.stderr) * 6.0
            val tolerance = math.max(stderrBound, 0.05)
            diff > tolerance
      checked += 1
      if bad then violations += 1
      idx += 1
    ValidationSummary(checked = checked, violations = violations)

  private def mergeValidation(a: ValidationSummary, b: ValidationSummary): ValidationSummary =
    ValidationSummary(checked = a.checked + b.checked, violations = a.violations + b.violations)

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
        case "speedupThreshold" => cfg.copy(speedupThreshold = value.toDouble)
        case "warmupRuns" => cfg.copy(warmupRuns = value.toInt)
        case "runs" => cfg.copy(runs = value.toInt)
        case "seed" => cfg.copy(seed = value.toLong)
        case other => throw new IllegalArgumentException(s"unknown option '$other'")
    }
