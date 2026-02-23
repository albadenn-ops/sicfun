package sicfun.holdem

import scala.util.Random

/** CLI entry point for generating a suit-isomorphic canonical heads-up equity table.
  *
  * Equivalent to [[GenerateHeadsUpTable]] but uses [[HeadsUpEquityCanonicalTable]] for
  * dramatically reduced table size by exploiting suit symmetry. The output file uses
  * 4-byte (Int) canonical keys instead of 8-byte (Long) pair keys.
  *
  * '''Usage:'''
  * {{{
  * GenerateHeadsUpCanonicalTable <outputPath> <mode:exact|mc> <trials> <maxMatchups> [seed] [parallelism] [backend:cpu|gpu]
  * }}}
  */
object GenerateHeadsUpCanonicalTable:
  def main(args: Array[String]): Unit =
    if args.length < 4 then
      System.err.println(
        "Usage: GenerateHeadsUpCanonicalTable <outputPath> <mode:exact|mc> <trials> <maxMatchups> [seed] [parallelism] [backend:cpu|gpu]"
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
      else HeadsUpEquityTable.ComputeBackend.Cpu

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
      tableKind = "canonical",
      mode = mode,
      maxMatchups = maxMatchups,
      backend = backend
    )

    val table = HeadsUpEquityCanonicalTable.buildAll(
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
      totalMatchups = HeadsUpEquityCanonicalTable.totalCanonicalKeys,
      count = table.values.size,
      canonical = true,
      createdAtMillis = System.currentTimeMillis()
    )
    HeadsUpEquityCanonicalTableIO.write(outputPath, table, meta)
