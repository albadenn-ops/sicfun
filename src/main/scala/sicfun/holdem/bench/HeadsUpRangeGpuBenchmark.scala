package sicfun.holdem.bench
import sicfun.holdem.*
import sicfun.holdem.equity.*
import sicfun.holdem.gpu.*
import sicfun.holdem.cli.*

import scala.collection.mutable.ArrayBuffer

/** Benchmark harness for CSR range evaluation in native GPU runtime.
  *
  * Runs the same CSR workload twice and compares memory paths:
  *   - `readonly` (`__ldg` read-only loads)
  *   - `global` (plain global loads; default runtime path)
  */
object HeadsUpRangeGpuBenchmark:
  private val ProviderProperty = "sicfun.gpu.provider"
  private val NativePathProperty = "sicfun.gpu.native.path"
  private val NativeEngineProperty = "sicfun.gpu.native.engine"
  private val RangeMemoryPathProperty = "sicfun.gpu.native.range.memoryPath"

  private final case class Config(
      heroes: Int = 128,
      entriesPerHero: Int = 64,
      trials: Int = 128,
      warmupRuns: Int = 1,
      runs: Int = 5,
      seedBase: Long = 1L,
      nativePath: Option[String] = None
  )

  private val AllowedOptionKeys = Set("heroes", "entriesPerHero", "trials", "warmupRuns", "runs", "seedBase", "nativePath")

  private final case class CsrBatch(
      heroIds: Array[Int],
      offsets: Array[Int],
      villainIds: Array[Int],
      keyMaterial: Array[Long],
      probabilities: Array[Float]
  ):
    def heroCount: Int = heroIds.length
    def entryCount: Int = villainIds.length

  private final case class Stats(
      label: String,
      avgSeconds: Double,
      heroThroughput: Double,
      entryThroughput: Double,
      checksum: Double
  )

  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    require(config.heroes > 0, "heroes must be positive")
    require(config.entriesPerHero > 0, "entriesPerHero must be positive")
    require(config.entriesPerHero <= 1225, "entriesPerHero must be <= 1225")
    require(config.trials > 0, "trials must be positive")
    require(config.warmupRuns >= 0, "warmupRuns must be non-negative")
    require(config.runs > 0, "runs must be positive")

    sys.props.update(ProviderProperty, "native")
    sys.props.update(NativeEngineProperty, "cuda")
    config.nativePath match
      case Some(path) => sys.props.update(NativePathProperty, path)
      case None => ()

    val availability = HeadsUpGpuRuntime.availability
    println("heads-up range gpu benchmark")
    println(
      s"heroes=${config.heroes}, entriesPerHero=${config.entriesPerHero}, trials=${config.trials}, " +
        s"warmupRuns=${config.warmupRuns}, runs=${config.runs}, seedBase=${config.seedBase}"
    )
    println(
      s"provider=${availability.provider}, available=${availability.available}, detail=${availability.detail}"
    )
    if !availability.available || availability.provider != "native" then
      throw new IllegalStateException(s"native provider unavailable: ${availability.detail}")

    val batch = buildSyntheticCsrBatch(config.heroes, config.entriesPerHero)
    println(s"batchHeroes=${batch.heroCount}, batchEntries=${batch.entryCount}")

    val readonlyStats = runScenario("readonly", batch, config)
    val globalStats = runScenario("global", batch, config)

    println(
      f"scenario=${readonlyStats.label} avg=${readonlyStats.avgSeconds}%.4fs " +
        f"heroes/s=${readonlyStats.heroThroughput}%.1f entries/s=${readonlyStats.entryThroughput}%.1f " +
        f"checksum=${readonlyStats.checksum}%.6f"
    )
    println(
      f"scenario=${globalStats.label} avg=${globalStats.avgSeconds}%.4fs " +
        f"heroes/s=${globalStats.heroThroughput}%.1f entries/s=${globalStats.entryThroughput}%.1f " +
        f"checksum=${globalStats.checksum}%.6f"
    )
    val speedup = readonlyStats.entryThroughput / globalStats.entryThroughput
    println(f"speedup(readonly/global)=${speedup}%.3fx")

  private def runScenario(label: String, batch: CsrBatch, config: Config): Stats =
    sys.props.update(RangeMemoryPathProperty, label)

    var warmup = 0
    while warmup < config.warmupRuns do
      val seed = config.seedBase + warmup.toLong
      HeadsUpRangeGpuRuntime.computeRangeBatchMonteCarloCsr(
        batch.heroIds,
        batch.offsets,
        batch.villainIds,
        batch.keyMaterial,
        batch.probabilities,
        config.trials,
        seed
      ) match
        case Left(reason) => throw new IllegalStateException(s"warmup failed [$label]: $reason")
        case Right(_) => ()
      warmup += 1

    val elapsed = new Array[Double](config.runs)
    var checksum = 0.0
    var run = 0
    while run < config.runs do
      val seed = config.seedBase + 1000L + run.toLong
      val started = System.nanoTime()
      val valuesEither =
        HeadsUpRangeGpuRuntime.computeRangeBatchMonteCarloCsr(
          batch.heroIds,
          batch.offsets,
          batch.villainIds,
          batch.keyMaterial,
          batch.probabilities,
          config.trials,
          seed
        )
      val sec = (System.nanoTime() - started).toDouble / 1_000_000_000.0
      elapsed(run) = sec
      valuesEither match
        case Left(reason) =>
          throw new IllegalStateException(s"run failed [$label]: $reason")
        case Right(values) =>
          checksum += values.iterator.map(v => v.win + v.tie + v.loss).sum
      run += 1

    val avgSeconds = elapsed.sum / elapsed.length.toDouble
    val heroThroughput = batch.heroCount.toDouble / avgSeconds
    val entryThroughput = batch.entryCount.toDouble / avgSeconds
    Stats(label, avgSeconds, heroThroughput, entryThroughput, checksum)

  private def buildSyntheticCsrBatch(heroes: Int, entriesPerHero: Int): CsrBatch =
    val heroIds = new Array[Int](heroes)
    val offsets = new Array[Int](heroes + 1)
    val villainBuf = ArrayBuffer.empty[Int]
    val keyMaterialBuf = ArrayBuffer.empty[Long]
    val probabilitiesBuf = ArrayBuffer.empty[Float]

    var h = 0
    while h < heroes do
      val heroId = (h * 17) % HoleCardsIndex.size
      heroIds(h) = heroId
      val heroHand = HoleCardsIndex.byId(heroId)
      var count = 0
      var candidate = 0
      while candidate < HoleCardsIndex.size && count < entriesPerHero do
        val villainId = (candidate + h * 31) % HoleCardsIndex.size
        val villainHand = HoleCardsIndex.byId(villainId)
        if villainId != heroId && HoleCardsIndex.areDisjoint(heroHand, villainHand) then
          villainBuf += villainId
          val lowId = math.min(heroId, villainId)
          val highId = math.max(heroId, villainId)
          val material =
            HeadsUpEquityTable.pack(lowId, highId) ^
              (h.toLong << 32) ^
              (count.toLong << 48)
          keyMaterialBuf += material
          probabilitiesBuf += ((count % 7) + 1).toFloat
          count += 1
        candidate += 1
      if count < entriesPerHero then
        throw new IllegalStateException(
          s"unable to build CSR batch for heroIndex=$h heroId=$heroId entriesPerHero=$entriesPerHero"
        )
      offsets(h + 1) = villainBuf.length
      h += 1

    CsrBatch(
      heroIds = heroIds,
      offsets = offsets,
      villainIds = villainBuf.toArray,
      keyMaterial = keyMaterialBuf.toArray,
      probabilities = probabilitiesBuf.toArray
    )

  private def parseArgs(args: Vector[String]): Config =
    val options = CliHelpers.requireOptions(args)
    CliHelpers.requireNoUnknownOptions(options, AllowedOptionKeys)
    Config(
      heroes = CliHelpers.requireIntOption(options, "heroes", 128),
      entriesPerHero = CliHelpers.requireIntOption(options, "entriesPerHero", 64),
      trials = CliHelpers.requireIntOption(options, "trials", 128),
      warmupRuns = CliHelpers.requireIntOption(options, "warmupRuns", 1),
      runs = CliHelpers.requireIntOption(options, "runs", 5),
      seedBase = CliHelpers.requireLongOption(options, "seedBase", 1L),
      nativePath = options.get("nativePath")
    )
