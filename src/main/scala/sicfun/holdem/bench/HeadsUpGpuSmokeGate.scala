package sicfun.holdem.bench
import sicfun.holdem.*
import sicfun.holdem.equity.*
import sicfun.holdem.gpu.*
import sicfun.holdem.cli.*
import sicfun.holdem.bench.BenchSupport.loadBatch

/** Fast fail gate that verifies the native GPU runtime is usable and that
  * execution actually happens on CUDA (not CPU fallback).
  *
  * Usage:
  * {{{
  * HeadsUpGpuSmokeGate [--key=value ...]
  * }}}
  *
  * Options:
  *   - `--table=canonical|full` (default: canonical)
  *   - `--trials=<int>` (default: 200)
  *   - `--maxMatchups=<long>` (default: 128)
  *   - `--seed=<long>` (default: 1)
  *   - `--nativePath=<absolute-path-to-dll-or-so>` (optional)
  */
object HeadsUpGpuSmokeGate:
  private val ProviderProperty = "sicfun.gpu.provider"
  private val NativePathProperty = "sicfun.gpu.native.path"
  private val NativeEngineProperty = "sicfun.gpu.native.engine"
  private val FallbackToCpuProperty = "sicfun.gpu.fallbackToCpu"

  private final case class Config(
      table: String = "canonical",
      trials: Int = 200,
      maxMatchups: Long = 128L,
      seed: Long = 1L,
      nativePath: Option[String] = None
  )

  private val AllowedOptionKeys = Set("table", "trials", "maxMatchups", "seed", "nativePath")

  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    require(config.trials > 0, "trials must be positive")
    require(config.maxMatchups > 0, "maxMatchups must be positive")

    configureNativeRuntime(config)
    val availability = HeadsUpGpuRuntime.availability
    println("gpu smoke gate")
    println(
      s"table=${config.table}, trials=${config.trials}, maxMatchups=${config.maxMatchups}, seed=${config.seed}, " +
        s"provider=${availability.provider}, providerAvailable=${availability.available}"
    )
    println(s"providerDetail=${availability.detail}")

    if !availability.available || availability.provider != "native" then
      fail("provider_unavailable_or_not_native")

    val batch = loadBatch(config.table, config.maxMatchups)
    if batch.size <= 0 then
      fail("empty_batch")

    val mode = HeadsUpEquityTable.Mode.MonteCarlo(config.trials)
    HeadsUpGpuRuntime.computeBatch(batch.packedKeys, batch.keyMaterial, mode, config.seed) match
      case Left(reason) =>
        fail("gpu_runtime_error", Some(reason))
      case Right(values) =>
        if values.length != batch.size then
          fail("result_size_mismatch", Some(s"expected=${batch.size} actual=${values.length}"))

        val telemetry = HeadsUpGpuRuntime.lastBatchTelemetry
        val packedFloatIo = telemetry.exists(_.detail.contains("io=packed-f32-seed-on-device"))
        val sumTolerance = if packedFloatIo then 1e-5 else 1e-9

        var idx = 0
        var invalidCount = 0
        while idx < values.length do
          val value = values(idx)
          val sum = value.win + value.tie + value.loss
          val valid =
            java.lang.Double.isFinite(value.win) &&
              java.lang.Double.isFinite(value.tie) &&
              java.lang.Double.isFinite(value.loss) &&
              java.lang.Double.isFinite(value.stderr) &&
              math.abs(sum - 1.0) <= sumTolerance
          if !valid then invalidCount += 1
          idx += 1

        val telemetryText = telemetry.map(t => s"provider=${t.provider}, success=${t.success}, detail=${t.detail}").getOrElse("none")
        println(s"telemetry=$telemetryText")

        val usedCuda = telemetry.exists(t => t.success && t.detail.contains("nativeEngine=cuda"))
        if invalidCount > 0 then
          fail("invalid_results", Some(s"count=$invalidCount"))
        if !usedCuda then
          fail("not_using_cuda_engine")

        println("gate=PASS")

  private def fail(reason: String, detail: Option[String] = None): Nothing =
    val suffix = detail.map(value => s" detail=$value").getOrElse("")
    println(s"gate=FAIL reason=$reason$suffix")
    throw new IllegalStateException(s"gate failed: reason=$reason$suffix")

  private def configureNativeRuntime(config: Config): Unit =
    sys.props.update(ProviderProperty, "native")
    sys.props.update(NativeEngineProperty, "cuda")
    sys.props.update(FallbackToCpuProperty, "false")
    config.nativePath match
      case Some(path) => sys.props.update(NativePathProperty, path)
      case None => ()

  private def parseArgs(args: Vector[String]): Config =
    val options = CliHelpers.requireOptions(args)
    CliHelpers.requireNoUnknownOptions(options, AllowedOptionKeys)
    Config(
      table = options.getOrElse("table", "canonical").trim.toLowerCase,
      trials = CliHelpers.requireIntOption(options, "trials", 200),
      maxMatchups = CliHelpers.requireLongOption(options, "maxMatchups", 128L),
      seed = CliHelpers.requireLongOption(options, "seed", 1L),
      nativePath = options.get("nativePath")
    )
