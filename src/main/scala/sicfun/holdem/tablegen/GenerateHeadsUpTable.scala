package sicfun.holdem.tablegen
import sicfun.holdem.*
import sicfun.holdem.equity.*
import sicfun.holdem.gpu.*
import sicfun.holdem.bench.*

import scala.util.Random

/** CLI entry point for generating a full (non-canonical) heads-up equity table.
  *
  * Computes equity for every non-overlapping hole-card pair (up to `maxMatchups`)
  * using either exact enumeration or Monte Carlo sampling, then serializes the
  * result to a binary file via [[HeadsUpEquityTableIO]].
  *
  * '''Usage:'''
  * {{{
  * GenerateHeadsUpTable <outputPath> <mode:exact|mc> <trials> <maxMatchups> [seed] [parallelism] [backend:cpu|gpu]
  * }}}
  *
  * Progress is printed to stdout every 50,000 matchups.
  */
object GenerateHeadsUpTable:
  private val ProviderProperty = "sicfun.gpu.provider"
  private val ProviderEnv = "sicfun_GPU_PROVIDER"

  /** Entry point. Parses CLI arguments, selects the best available compute backend,
    * builds the full equity table, and writes it to disk.
    *
    * Unlike [[GenerateHeadsUpCanonicalTable]], this generates the full non-canonical table
    * with 8-byte packed (lowId, highId) keys. The full table has C(1326,2) = 878,475 entries
    * vs ~14,196 canonical entries, so it is much larger but avoids suit normalization overhead
    * at lookup time.
    *
    * @param args positional: outputPath, mode, trials, maxMatchups, [seed], [parallelism], [backend]
    */
  def main(args: Array[String]): Unit =
    if args.length < 4 then
      System.err.println(
        "Usage: GenerateHeadsUpTable <outputPath> <mode:exact|mc> <trials> <maxMatchups> [seed] [parallelism] [backend:cpu|gpu]"
      )
      sys.exit(1)

    val outputPath = args(0)
    val modeStr = args(1).toLowerCase
    val trials = args(2).toInt
    val maxMatchups = args(3).toLong
    val seed = if args.length >= 5 then args(4).toLong else 1L
    val parallelism = if args.length >= 6 then args(5).toInt else math.max(1, Runtime.getRuntime.availableProcessors())
    val backend =
      if args.length >= 7 then HeadsUpEquityTable.ComputeBackend.parse(args(6))
      else preferredDefaultBackend()

    val mode =
      modeStr match
        case "exact" => HeadsUpEquityTable.Mode.Exact
        case "mc" | "montecarlo" => HeadsUpEquityTable.Mode.MonteCarlo(trials)
        case other =>
          System.err.println(s"Unknown mode: $other")
          sys.exit(1)
          HeadsUpEquityTable.Mode.MonteCarlo(trials)

    val outFile = java.io.File(outputPath)
    val parent = outFile.getParentFile
    if parent != null then parent.mkdirs()

    val progressEvery = 50_000L
    var last = 0L
    val progress = (done: Long, totalAll: Long) =>
      if done - last >= progressEvery || done == totalAll then
        last = done
        val pct = (done.toDouble / totalAll.toDouble) * 100.0
        println(f"progress: $done / $totalAll ($pct%.2f%%)")

    HeadsUpBackendAutoTuner.configureForGeneration(
      tableKind = "full",
      mode = mode,
      maxMatchups = maxMatchups,
      backend = backend
    )

    val table = HeadsUpEquityTable.buildAll(
      mode = mode,
      rng = new Random(seed),
      maxMatchups = maxMatchups,
      progress = Some(progress),
      parallelism = parallelism,
      backend = backend
    )
    val meta = HeadsUpEquityTableMeta(
      formatVersion = HeadsUpEquityTableFormat.Version,
      mode = modeStr,
      trials = trials,
      seed = seed,
      maxMatchups = maxMatchups,
      totalMatchups = HeadsUpEquityTable.totalMatchups,
      count = table.values.size,
      canonical = false,
      createdAtMillis = System.currentTimeMillis()
    )
    HeadsUpEquityTableIO.write(outputPath, table, meta)

  /** Auto-detects the best available compute backend (GPU preferred, CPU fallback).
    * Probes "native" then "hybrid" GPU providers before falling back to pure CPU.
    */
  private def preferredDefaultBackend(): HeadsUpEquityTable.ComputeBackend =
    val configuredProvider = GpuRuntimeSupport.resolveNonEmptyLower(ProviderProperty, ProviderEnv)
    configuredProvider match
      case Some(provider) =>
        if providerAvailability(provider).available then HeadsUpEquityTable.ComputeBackend.Gpu
        else HeadsUpEquityTable.ComputeBackend.Cpu
      case None =>
        val nativeAvailability = providerAvailability("native")
        if nativeAvailability.available then HeadsUpEquityTable.ComputeBackend.Gpu
        else
          val hybridAvailability = providerAvailability("hybrid")
          if hybridAvailability.available then HeadsUpEquityTable.ComputeBackend.Gpu
          else HeadsUpEquityTable.ComputeBackend.Cpu

  private def providerAvailability(provider: String): HeadsUpGpuRuntime.Availability =
    val previous = sys.props.get(ProviderProperty)
    sys.props.update(ProviderProperty, provider)
    try HeadsUpGpuRuntime.availability
    finally
      previous match
        case Some(value) => sys.props.update(ProviderProperty, value)
        case None => sys.props.remove(ProviderProperty)
