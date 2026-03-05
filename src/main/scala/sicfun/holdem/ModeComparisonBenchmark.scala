package sicfun.holdem

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale
import scala.util.Random

/** Unbiased throughput comparison for CPU-only, CUDA-only, and hybrid modes.
  *
  * Benchmark method:
  *   1) Build one fixed canonical matchup batch and seed set.
  *   2) Warm each mode for `warmupRuns`.
  *   3) Execute `measureRuns` in randomized mode order per run.
  *   4) Report distribution stats and export CSV + SVG boxplot.
  *
  * Positional args:
  *   - arg0: maxMatchups (default 20000)
  *   - arg1: trials (default 200)
  *   - arg2: warmupRuns (default 2)
  *   - arg3: measureRuns (default 12)
  *   - arg4: seed (default 42)
  */
object ModeComparisonBenchmark:

  private final case class Config(
      maxMatchups: Long = 20000L,
      trials: Int = 200,
      warmupRuns: Int = 2,
      measureRuns: Int = 12,
      seed: Long = 42L
  )

  private final case class ModeSample(
      mode: String,
      run: Int,
      elapsedNanos: Long,
      matchupsPerSec: Double,
      statesPerSec: Double,
      detail: String
  ):
    def elapsedMs: Double = elapsedNanos.toDouble / 1_000_000.0

  private final case class ModeStats(
      count: Int,
      mean: Double,
      median: Double,
      min: Double,
      max: Double,
      q1: Double,
      q3: Double,
      stddev: Double
  )

  private final case class Runner(
      name: String,
      runOnce: Int => Either[String, ModeSample]
  )

  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    require(config.maxMatchups > 0L, "maxMatchups must be positive")
    require(config.trials > 0, "trials must be positive")
    require(config.warmupRuns >= 0, "warmupRuns must be non-negative")
    require(config.measureRuns > 0, "measureRuns must be positive")

    val devices = HeadsUpHybridDispatcher.devices
    val cudaOpt = devices.find(_.kind == "cuda")
    val cpuOpt = devices.find(_.kind == "cpu")
    val openclOpt = devices.find(_.kind == "opencl")
    require(cudaOpt.nonEmpty, "CUDA device not available")
    require(cpuOpt.nonEmpty, "CPU device not available")

    println("=== Mode Comparison Benchmark ===")
    println(
      s"config: maxMatchups=${config.maxMatchups}, trials=${config.trials}, warmupRuns=${config.warmupRuns}, " +
        s"measureRuns=${config.measureRuns}, seed=${config.seed}"
    )
    println("discovered devices:")
    devices.foreach { d =>
      println(s"  ${d.id} | kind=${d.kind} | name=${d.name} | exact=${d.supportsExact}")
    }
    println()

    val batch = HeadsUpEquityCanonicalTable.selectCanonicalBatch(config.maxMatchups)
    val n = batch.packedKeys.length
    val statesPerRun = n.toDouble * config.trials.toDouble
    println(s"batch size: $n matchups")
    println(f"states per run: $statesPerRun%.0f")

    val lowIds = new Array[Int](n)
    val highIds = new Array[Int](n)
    val seeds = new Array[Long](n)
    var idx = 0
    while idx < n do
      val packed = batch.packedKeys(idx)
      lowIds(idx) = HeadsUpEquityTable.unpackLowId(packed)
      highIds(idx) = HeadsUpEquityTable.unpackHighId(packed)
      seeds(idx) = HeadsUpEquityTable.monteCarloSeed(config.seed, batch.keyMaterial(idx))
      idx += 1

    val cpuRunner = runnerForDevice("cpu-only", cpuOpt.get, n, config.trials, statesPerRun, lowIds, highIds, seeds)
    val cudaRunner = runnerForDevice("cuda-only", cudaOpt.get, n, config.trials, statesPerRun, lowIds, highIds, seeds)
    val hybridRunner = runnerForHybrid("hybrid", n, config.trials, statesPerRun, lowIds, highIds, seeds)
    val openclRunnerOpt =
      openclOpt.map(device =>
        runnerForDevice("opencl-only", device, n, config.trials, statesPerRun, lowIds, highIds, seeds)
      )

    val runners = Vector(Some(cpuRunner), Some(cudaRunner), openclRunnerOpt, Some(hybridRunner)).flatten
    println("benchmarked modes: " + runners.map(_.name).mkString(", "))
    println()

    // Warmup in deterministic order before measurement.
    var warm = 0
    while warm < config.warmupRuns do
      runners.foreach { runner =>
        runner.runOnce(-(warm + 1)) match
          case Left(err) =>
            throw new IllegalStateException(s"warmup failed for ${runner.name}: $err")
          case Right(_) => ()
      }
      warm += 1

    val orderRng = new Random(config.seed ^ 0x9E3779B97F4A7C15L)
    val samples = scala.collection.mutable.ArrayBuffer.empty[ModeSample]
    var run = 1
    while run <= config.measureRuns do
      val order = orderRng.shuffle(runners)
      order.foreach { runner =>
        runner.runOnce(run) match
          case Left(err) =>
            throw new IllegalStateException(s"run $run failed for ${runner.name}: $err")
          case Right(sample) =>
            samples += sample
      }
      run += 1

    val byMode = samples.groupBy(_.mode).view.mapValues(_.toVector).toMap
    println("summary (states/s):")
    val orderedModes = Vector("cpu-only", "cuda-only", "opencl-only", "hybrid").filter(byMode.contains)
    val statsByMode = orderedModes.map(mode => mode -> computeStats(byMode(mode).map(_.statesPerSec))).toMap
    orderedModes.foreach { mode =>
      val stats = statsByMode(mode)
      println(
        f"  $mode%-11s count=${stats.count}%2d mean=${stats.mean / 1e6}%.2fM " +
          f"median=${stats.median / 1e6}%.2fM min=${stats.min / 1e6}%.2fM max=${stats.max / 1e6}%.2fM " +
          f"stdev=${stats.stddev / 1e6}%.2fM"
      )
    }

    val outDir = Paths.get("data", "benchmarks")
    Files.createDirectories(outDir)
    val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss"))
    val rawCsvPath = outDir.resolve(s"mode-comparison-$timestamp-raw.csv")
    val summaryCsvPath = outDir.resolve(s"mode-comparison-$timestamp-summary.csv")
    val svgPath = outDir.resolve(s"mode-comparison-$timestamp-states-boxplot.svg")

    writeRawCsv(rawCsvPath, samples.toVector)
    writeSummaryCsv(summaryCsvPath, orderedModes, statsByMode)
    writeBoxplotSvg(
      svgPath,
      title = "CPU vs CUDA vs Hybrid Throughput (States/s)",
      subtitle =
        s"randomized mode order per run, warmup=${config.warmupRuns}, measuredRuns=${config.measureRuns}, " +
          s"matchups=$n, trials=${config.trials}",
      orderedModes = orderedModes,
      statsByMode = statsByMode
    )

    println()
    println(s"raw csv: ${rawCsvPath.toAbsolutePath}")
    println(s"summary csv: ${summaryCsvPath.toAbsolutePath}")
    println(s"graph svg: ${svgPath.toAbsolutePath}")
    println("=== Done ===")

  private def runnerForDevice(
      mode: String,
      device: HeadsUpHybridDispatcher.ComputeDevice,
      matchups: Int,
      trials: Int,
      statesPerRun: Double,
      lowIds: Array[Int],
      highIds: Array[Int],
      seeds: Array[Long]
  ): Runner =
    val wins = new Array[Double](matchups)
    val ties = new Array[Double](matchups)
    val losses = new Array[Double](matchups)
    val stderrs = new Array[Double](matchups)
    Runner(
      name = mode,
      runOnce = runId =>
        val started = System.nanoTime()
        val status = device.computeSubBatch(lowIds, highIds, 1, trials, seeds, wins, ties, losses, stderrs)
        val elapsed = System.nanoTime() - started
        if status != 0 then
          Left(s"status=$status")
        else
          val elapsedSeconds = elapsed.toDouble / 1_000_000_000.0
          val matchupsPerSec = if elapsedSeconds > 0.0 then matchups.toDouble / elapsedSeconds else 0.0
          val statesPerSec = if elapsedSeconds > 0.0 then statesPerRun / elapsedSeconds else 0.0
          Right(
            ModeSample(
              mode = mode,
              run = runId,
              elapsedNanos = elapsed,
              matchupsPerSec = matchupsPerSec,
              statesPerSec = statesPerSec,
              detail = s"device=${device.id}"
            )
          )
    )

  private def runnerForHybrid(
      mode: String,
      matchups: Int,
      trials: Int,
      statesPerRun: Double,
      lowIds: Array[Int],
      highIds: Array[Int],
      seeds: Array[Long]
  ): Runner =
    Runner(
      name = mode,
      runOnce = runId =>
        val started = System.nanoTime()
        HeadsUpHybridDispatcher.dispatchBatch(lowIds, highIds, 1, trials, seeds) match
          case Left(error) =>
            Left(error)
          case Right(result) =>
            val elapsed = System.nanoTime() - started
            val elapsedSeconds = elapsed.toDouble / 1_000_000_000.0
            val matchupsPerSec = if elapsedSeconds > 0.0 then matchups.toDouble / elapsedSeconds else 0.0
            val statesPerSec = if elapsedSeconds > 0.0 then statesPerRun / elapsedSeconds else 0.0
            val devices =
              if result.perDevice.isEmpty then "none"
              else result.perDevice.map(t => s"${t.deviceId}:${t.matchups}").mkString("|")
            Right(
              ModeSample(
                mode = mode,
                run = runId,
                elapsedNanos = elapsed,
                matchupsPerSec = matchupsPerSec,
                statesPerSec = statesPerSec,
                detail = s"devices=$devices"
              )
            )
    )

  private def computeStats(values: Vector[Double]): ModeStats =
    val sorted = values.sorted
    val count = sorted.size
    require(count > 0, "cannot compute stats for empty sample set")
    val mean = sorted.sum / count.toDouble
    val median = quantile(sorted, 0.5)
    val q1 = quantile(sorted, 0.25)
    val q3 = quantile(sorted, 0.75)
    val min = sorted.head
    val max = sorted.last
    val variance =
      if count > 1 then
        val sumSq = sorted.iterator.map(v => (v - mean) * (v - mean)).sum
        sumSq / (count.toDouble - 1.0)
      else 0.0
    ModeStats(
      count = count,
      mean = mean,
      median = median,
      min = min,
      max = max,
      q1 = q1,
      q3 = q3,
      stddev = math.sqrt(variance)
    )

  private def quantile(sorted: Vector[Double], q: Double): Double =
    if sorted.isEmpty then 0.0
    else if sorted.size == 1 then sorted.head
    else
      val p = q * (sorted.size - 1).toDouble
      val lo = math.floor(p).toInt
      val hi = math.ceil(p).toInt
      if lo == hi then sorted(lo)
      else
        val w = p - lo.toDouble
        sorted(lo) * (1.0 - w) + sorted(hi) * w

  private def writeRawCsv(path: Path, samples: Vector[ModeSample]): Unit =
    val header = "mode,run,elapsed_ms,matchups_per_s,states_per_s,detail"
    val lines = samples.map { s =>
      s"${csv(s.mode)},${s.run},${fmt3(s.elapsedMs)},${fmt3(s.matchupsPerSec)},${fmt3(s.statesPerSec)},${csv(s.detail)}"
    }
    Files.write(path, (header +: lines).mkString("\n").getBytes(StandardCharsets.UTF_8))

  private def writeSummaryCsv(path: Path, modes: Vector[String], statsByMode: Map[String, ModeStats]): Unit =
    val header = "mode,count,mean_states_per_s,median_states_per_s,min_states_per_s,max_states_per_s,q1_states_per_s,q3_states_per_s,stddev_states_per_s"
    val lines = modes.map { mode =>
      val s = statsByMode(mode)
      s"${csv(mode)},${s.count},${fmt3(s.mean)},${fmt3(s.median)},${fmt3(s.min)},${fmt3(s.max)},${fmt3(s.q1)},${fmt3(s.q3)},${fmt3(s.stddev)}"
    }
    Files.write(path, (header +: lines).mkString("\n").getBytes(StandardCharsets.UTF_8))

  private def writeBoxplotSvg(
      path: Path,
      title: String,
      subtitle: String,
      orderedModes: Vector[String],
      statsByMode: Map[String, ModeStats]
  ): Unit =
    val width = 1100
    val height = 680
    val marginLeft = 120
    val marginRight = 60
    val marginTop = 90
    val marginBottom = 140
    val chartWidth = width - marginLeft - marginRight
    val chartHeight = height - marginTop - marginBottom

    val maxValue = orderedModes.map(mode => statsByMode(mode).max).maxOption.getOrElse(1.0)
    val yMax = if maxValue > 0 then maxValue * 1.1 else 1.0
    def y(v: Double): Double = marginTop + chartHeight - (v / yMax) * chartHeight

    val palette = Map(
      "cpu-only" -> "#3b82f6",
      "cuda-only" -> "#ef4444",
      "opencl-only" -> "#10b981",
      "hybrid" -> "#f59e0b"
    )

    val gridLines = 6
    val gridSvg = (0 to gridLines).map { i =>
      val ratio = i.toDouble / gridLines.toDouble
      val value = yMax * ratio
      val yy = y(value)
      val label = f"${value / 1e6}%.1fM"
      s"""<line x1="$marginLeft" y1="$yy%.2f" x2="${marginLeft + chartWidth}" y2="$yy%.2f" stroke="#e5e7eb" stroke-width="1"/>""" +
        s"""<text x="${marginLeft - 12}" y="${yy + 4}%.2f" text-anchor="end" font-size="12" fill="#374151">$label</text>"""
    }.mkString("\n")

    val n = orderedModes.size
    val slotWidth = chartWidth.toDouble / math.max(1, n).toDouble
    val boxWidth = math.min(80.0, slotWidth * 0.45)

    val boxSvg = orderedModes.zipWithIndex.map { case (mode, idx) =>
      val s = statsByMode(mode)
      val color = palette.getOrElse(mode, "#6b7280")
      val xCenter = marginLeft + slotWidth * (idx.toDouble + 0.5)
      val x0 = xCenter - boxWidth / 2.0
      val whiskerTop = y(s.max)
      val whiskerBottom = y(s.min)
      val q3y = y(s.q3)
      val q1y = y(s.q1)
      val medianY = y(s.median)
      val meanY = y(s.mean)
      val labelY = marginTop + chartHeight + 30
      s"""
         |<line x1="$xCenter%.2f" y1="$whiskerTop%.2f" x2="$xCenter%.2f" y2="$whiskerBottom%.2f" stroke="$color" stroke-width="2"/>
         |<line x1="${xCenter - 16}%.2f" y1="$whiskerTop%.2f" x2="${xCenter + 16}%.2f" y2="$whiskerTop%.2f" stroke="$color" stroke-width="2"/>
         |<line x1="${xCenter - 16}%.2f" y1="$whiskerBottom%.2f" x2="${xCenter + 16}%.2f" y2="$whiskerBottom%.2f" stroke="$color" stroke-width="2"/>
         |<rect x="$x0%.2f" y="$q3y%.2f" width="$boxWidth%.2f" height="${(q1y - q3y).max(1.0)}%.2f" fill="$color" fill-opacity="0.22" stroke="$color" stroke-width="2"/>
         |<line x1="$x0%.2f" y1="$medianY%.2f" x2="${x0 + boxWidth}%.2f" y2="$medianY%.2f" stroke="$color" stroke-width="3"/>
         |<circle cx="$xCenter%.2f" cy="$meanY%.2f" r="4" fill="$color"/>
         |<text x="$xCenter%.2f" y="$labelY%.2f" text-anchor="middle" font-size="13" fill="#111827">$mode</text>
         |<text x="$xCenter%.2f" y="${labelY + 18}%.2f" text-anchor="middle" font-size="12" fill="#4b5563">${f"${s.mean / 1e6}%.2f"}M avg</text>
       """.stripMargin
    }.mkString("\n")

    val svg =
      s"""<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">
         |  <rect x="0" y="0" width="$width" height="$height" fill="#ffffff"/>
         |  <text x="${width / 2}" y="38" text-anchor="middle" font-size="24" font-family="Segoe UI, Arial, sans-serif" fill="#111827">$title</text>
         |  <text x="${width / 2}" y="62" text-anchor="middle" font-size="13" font-family="Segoe UI, Arial, sans-serif" fill="#374151">$subtitle</text>
         |  $gridSvg
         |  <line x1="$marginLeft" y1="${marginTop + chartHeight}" x2="${marginLeft + chartWidth}" y2="${marginTop + chartHeight}" stroke="#9ca3af" stroke-width="2"/>
         |  <line x1="$marginLeft" y1="$marginTop" x2="$marginLeft" y2="${marginTop + chartHeight}" stroke="#9ca3af" stroke-width="2"/>
         |  $boxSvg
         |  <text x="28" y="${marginTop + chartHeight / 2}" transform="rotate(-90,28,${marginTop + chartHeight / 2})"
         |        text-anchor="middle" font-size="14" font-family="Segoe UI, Arial, sans-serif" fill="#111827">states per second</text>
         |</svg>
       """.stripMargin

    Files.write(path, svg.getBytes(StandardCharsets.UTF_8))

  private def csv(value: String): String =
    "\"" + value.replace("\"", "\"\"") + "\""

  private def fmt3(value: Double): String =
    String.format(Locale.US, "%.3f", java.lang.Double.valueOf(value))

  private def parseArgs(args: Vector[String]): Config =
    Config(
      maxMatchups = args.headOption.flatMap(v => scala.util.Try(v.toLong).toOption).getOrElse(20000L),
      trials = args.lift(1).flatMap(v => scala.util.Try(v.toInt).toOption).getOrElse(200),
      warmupRuns = args.lift(2).flatMap(v => scala.util.Try(v.toInt).toOption).getOrElse(2),
      measureRuns = args.lift(3).flatMap(v => scala.util.Try(v.toInt).toOption).getOrElse(12),
      seed = args.lift(4).flatMap(v => scala.util.Try(v.toLong).toOption).getOrElse(42L)
    )
