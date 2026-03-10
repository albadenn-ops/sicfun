package sicfun.holdem.bench
import sicfun.holdem.*
import sicfun.holdem.equity.*
import sicfun.holdem.gpu.*
import sicfun.holdem.cli.*

import scala.util.Random

/** Exact parity gate that compares native CPU vs CUDA results over a small
  * canonical slice and fails unless all deltas are exactly zero.
  *
  * Usage:
  * {{{
  * HeadsUpGpuExactParityGate [--key=value ...]
  * }}}
  *
  * Options:
  *   - `--maxMatchups=<long>` (default: 8)
  *   - `--seed=<long>` (default: 1)
  *   - `--parallelism=<int>` (default: available processors)
  *   - `--nativePath=<absolute-path-to-dll-or-so>` (optional)
  */
object HeadsUpGpuExactParityGate:
  private val ProviderProperty = "sicfun.gpu.provider"
  private val NativePathProperty = "sicfun.gpu.native.path"
  private val NativeEngineProperty = "sicfun.gpu.native.engine"
  private val FallbackToCpuProperty = "sicfun.gpu.fallbackToCpu"
  private val NativeCudaBlockSizeProperty = "sicfun.gpu.native.cuda.blockSize"
  private val NativeCudaMaxChunkMatchupsProperty = "sicfun.gpu.native.cuda.maxChunkMatchups"

  private final case class Config(
      maxMatchups: Long = 8L,
      seed: Long = 1L,
      parallelism: Int = math.max(1, Runtime.getRuntime.availableProcessors()),
      nativePath: Option[String] = None
  )

  private val AllowedOptionKeys = Set("maxMatchups", "seed", "parallelism", "nativePath")

  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    require(config.maxMatchups > 0L, "maxMatchups must be positive")
    require(config.parallelism > 0, "parallelism must be positive")

    configureNativeRuntime(config)
    val availability = HeadsUpGpuRuntime.availability
    println("gpu exact parity gate")
    println(
      s"maxMatchups=${config.maxMatchups}, seed=${config.seed}, parallelism=${config.parallelism}, " +
        s"provider=${availability.provider}, providerAvailable=${availability.available}"
    )
    println(s"providerDetail=${availability.detail}")

    if !availability.available || availability.provider != "native" then
      fail("provider_unavailable_or_not_native")

    println(s"cudaDeviceCount(best_effort)=${readCudaDeviceCount()}")

    val cpuTable = buildSlice(config, engine = "cpu")
    val cudaTable = buildSlice(config, engine = "cuda")

    val cpuKeys = cpuTable.values.keySet
    val cudaKeys = cudaTable.values.keySet
    if cpuKeys != cudaKeys then
      fail("keyset_mismatch", Some(s"cpuKeys=${cpuKeys.size} cudaKeys=${cudaKeys.size}"))

    var maxWinDelta = 0.0
    var maxTieDelta = 0.0
    var maxLossDelta = 0.0
    var maxEqDelta = 0.0
    cpuKeys.foreach { key =>
      val cpu = cpuTable.values(key)
      val cuda = cudaTable.values(key)
      maxWinDelta = math.max(maxWinDelta, math.abs(cpu.win - cuda.win))
      maxTieDelta = math.max(maxTieDelta, math.abs(cpu.tie - cuda.tie))
      maxLossDelta = math.max(maxLossDelta, math.abs(cpu.loss - cuda.loss))
      maxEqDelta = math.max(maxEqDelta, math.abs(cpu.equity - cuda.equity))
    }

    println(f"maxAbsWinDelta=$maxWinDelta%.18f")
    println(f"maxAbsTieDelta=$maxTieDelta%.18f")
    println(f"maxAbsLossDelta=$maxLossDelta%.18f")
    println(f"maxAbsEqDelta=$maxEqDelta%.18f")

    val telemetry = HeadsUpGpuRuntime.lastBatchTelemetry
      .map(t => s"provider=${t.provider}, success=${t.success}, detail=${t.detail}")
      .getOrElse("none")
    println(s"telemetry=$telemetry")

    val usedCuda = HeadsUpGpuRuntime.lastBatchTelemetry.exists(t => t.success && t.detail.contains("nativeEngine=cuda"))
    val deltasAreZero = maxWinDelta == 0.0 && maxTieDelta == 0.0 && maxLossDelta == 0.0 && maxEqDelta == 0.0
    if !usedCuda then
      fail("not_using_cuda_engine")
    if !deltasAreZero then
      fail("non_zero_delta")

    println("gate=PASS")

  private def fail(reason: String, detail: Option[String] = None): Nothing =
    val suffix = detail.map(value => s" detail=$value").getOrElse("")
    println(s"gate=FAIL reason=$reason$suffix")
    throw new IllegalStateException(s"gate failed: reason=$reason$suffix")

  private def buildSlice(config: Config, engine: String): HeadsUpEquityCanonicalTable =
    sys.props.update(NativeEngineProperty, engine)
    if engine == "cuda" then
      // Keep tiny exact slices watchdog-safe on WDDM devices during parity checks.
      sys.props.update(NativeCudaBlockSizeProperty, "32")
      sys.props.update(NativeCudaMaxChunkMatchupsProperty, "1")
    else
      sys.props.remove(NativeCudaBlockSizeProperty)
      sys.props.remove(NativeCudaMaxChunkMatchupsProperty)

    HeadsUpEquityCanonicalTable.buildAll(
      mode = HeadsUpEquityTable.Mode.Exact,
      rng = new Random(config.seed),
      maxMatchups = config.maxMatchups,
      progress = None,
      parallelism = config.parallelism,
      backend = HeadsUpEquityTable.ComputeBackend.Gpu
    )

  private def configureNativeRuntime(config: Config): Unit =
    sys.props.update(ProviderProperty, "native")
    sys.props.update(FallbackToCpuProperty, "false")
    config.nativePath match
      case Some(path) => sys.props.update(NativePathProperty, path)
      case None => ()

  private def readCudaDeviceCount(): Int =
    try HeadsUpGpuNativeBindings.cudaDeviceCount()
    catch
      case _: UnsatisfiedLinkError => 0
      case _: Throwable => 0

  private def parseArgs(args: Vector[String]): Config =
    val options = CliHelpers.requireOptions(args)
    CliHelpers.requireNoUnknownOptions(options, AllowedOptionKeys)
    Config(
      maxMatchups = CliHelpers.requireLongOption(options, "maxMatchups", 8L),
      seed = CliHelpers.requireLongOption(options, "seed", 1L),
      parallelism = CliHelpers.requireIntOption(options, "parallelism", math.max(1, Runtime.getRuntime.availableProcessors())),
      nativePath = options.get("nativePath")
    )
