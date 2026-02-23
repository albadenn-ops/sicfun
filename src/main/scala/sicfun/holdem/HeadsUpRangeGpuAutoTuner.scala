package sicfun.holdem

import java.io.{File, FileOutputStream}
import java.util.Properties

import scala.collection.mutable.ArrayBuffer

/** Persistent auto-tuner for CSR range CUDA parameters.
  *
  * Benchmarks candidate combinations for each detected CUDA device and stores the
  * winner to a cache file. [[HeadsUpRangeGpuRuntime]] can auto-load this cache.
  */
object HeadsUpRangeGpuAutoTuner:
  private val ProviderProperty = "sicfun.gpu.provider"
  private val NativePathProperty = "sicfun.gpu.native.path"
  private val NativeEngineProperty = "sicfun.gpu.native.engine"
  private val RangeAutoTuneProperty = "sicfun.gpu.range.autotune"
  private val RangeAutoTuneCachePathProperty = "sicfun.gpu.range.autotune.cachePath"
  private val RangeNativeBlockSizeProperty = "sicfun.gpu.native.range.cuda.blockSize"
  private val RangeNativeMaxChunkHeroesProperty = "sicfun.gpu.native.range.cuda.maxChunkHeroes"
  private val RangeNativeMemoryPathProperty = "sicfun.gpu.native.range.memoryPath"
  private val CacheVersion = "1"
  private val DefaultCachePath = "data/headsup-range-autotune.properties"
  private val DefaultSeedBase = 0x38A7B35C1DF6241EL

  private final case class Config(
      heroes: Int = 256,
      entriesPerHero: Int = 128,
      trials: Int = 64,
      warmupRuns: Int = 1,
      runs: Int = 3,
      seedBase: Long = DefaultSeedBase,
      nativePath: Option[String] = None,
      cachePath: String = DefaultCachePath
  )

  private final case class Candidate(
      blockSize: Int,
      maxChunkHeroes: Int,
      memoryPath: String
  ):
    override def toString: String =
      s"block=$blockSize chunkHeroes=$maxChunkHeroes memoryPath=$memoryPath"

  private final case class CsrBatch(
      heroIds: Array[Int],
      offsets: Array[Int],
      villainIds: Array[Int],
      keyMaterial: Array[Long],
      probabilities: Array[Float]
  ):
    def heroCount: Int = heroIds.length
    def entryCount: Int = villainIds.length

  private final case class DeviceTuningResult(
      index: Int,
      fingerprint: String,
      winner: Candidate,
      entriesPerSecond: Double,
      elapsedSeconds: Double
  )

  private val BlockCandidates = Vector(32, 64, 128, 256)
  private val ChunkHeroesCandidates = Vector(128, 256, 512, 1024, 2048, 4096)
  private val MemoryPathCandidates = Vector("global", "readonly")

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
    // Prevent runtime cache from overriding candidate properties while tuning.
    sys.props.update(RangeAutoTuneProperty, "false")
    config.nativePath.foreach(path => sys.props.update(NativePathProperty, path))

    val availability = HeadsUpGpuRuntime.availability
    println("heads-up range gpu auto-tuner")
    println(
      s"heroes=${config.heroes}, entriesPerHero=${config.entriesPerHero}, trials=${config.trials}, " +
        s"warmupRuns=${config.warmupRuns}, runs=${config.runs}, seedBase=${config.seedBase}"
    )
    println(s"provider=${availability.provider}, available=${availability.available}, detail=${availability.detail}")
    if !availability.available || availability.provider != "native" then
      throw new IllegalStateException(s"native provider unavailable: ${availability.detail}")

    val deviceCount = HeadsUpGpuNativeBindings.cudaDeviceCount()
    if deviceCount <= 0 then
      throw new IllegalStateException("no CUDA devices found")

    val batch = buildSyntheticCsrBatch(config.heroes, config.entriesPerHero)
    println(s"batchHeroes=${batch.heroCount}, batchEntries=${batch.entryCount}")

    val allCandidates =
      for
        block <- BlockCandidates
        chunk <- ChunkHeroesCandidates
        memory <- MemoryPathCandidates
      yield Candidate(blockSize = block, maxChunkHeroes = chunk, memoryPath = memory)

    val results = ArrayBuffer.empty[DeviceTuningResult]
    var deviceIdx = 0
    while deviceIdx < deviceCount do
      val fingerprint = Option(HeadsUpGpuNativeBindings.cudaDeviceInfo(deviceIdx)).getOrElse("").trim
      if fingerprint.nonEmpty then
        println(s"tuning device[$deviceIdx]=$fingerprint")
        val winnerOpt = tuneDevice(deviceIdx, batch, config, allCandidates.toVector)
        winnerOpt match
          case Some(winner) =>
            results += DeviceTuningResult(
              index = deviceIdx,
              fingerprint = fingerprint,
              winner = winner._1,
              entriesPerSecond = winner._2,
              elapsedSeconds = winner._3
            )
            println(
              f"winner device[$deviceIdx]: ${winner._1} elapsed=${winner._3}%.4fs entries/s=${winner._2}%.1f"
            )
          case None =>
            println(s"no successful candidate for device[$deviceIdx], skipping cache entry")
      deviceIdx += 1

    if results.isEmpty then
      throw new IllegalStateException("no successful auto-tune result for any device")

    saveCache(new File(config.cachePath), results.toVector)
    println(s"cache written: ${new File(config.cachePath).getAbsolutePath}")

  private def tuneDevice(
      deviceIndex: Int,
      batch: CsrBatch,
      config: Config,
      candidates: Vector[Candidate]
  ): Option[(Candidate, Double, Double)] =
    var winner: Option[(Candidate, Double, Double)] = None
    candidates.foreach { candidate =>
      applyCandidate(candidate)

      var warmup = 0
      var warmupFailed = false
      while warmup < config.warmupRuns && !warmupFailed do
        HeadsUpRangeGpuRuntime.computeRangeBatchMonteCarloCsrOnDevice(
          deviceIndex,
          batch.heroIds,
          batch.offsets,
          batch.villainIds,
          batch.keyMaterial,
          batch.probabilities,
          config.trials,
          config.seedBase + warmup.toLong
        ) match
          case Left(_) => warmupFailed = true
          case Right(_) => ()
        warmup += 1

      if !warmupFailed then
        val elapsedRuns = new Array[Double](config.runs)
        var runFailed: Option[String] = None
        var run = 0
        while run < config.runs && runFailed.isEmpty do
          val started = System.nanoTime()
          HeadsUpRangeGpuRuntime.computeRangeBatchMonteCarloCsrOnDevice(
            deviceIndex,
            batch.heroIds,
            batch.offsets,
            batch.villainIds,
            batch.keyMaterial,
            batch.probabilities,
            config.trials,
            config.seedBase + 1000L + run.toLong
          ) match
            case Left(reason) => runFailed = Some(reason)
            case Right(values) =>
              if values.length != batch.heroCount then
                runFailed = Some(s"result length mismatch expected=${batch.heroCount} actual=${values.length}")
              else
                elapsedRuns(run) = (System.nanoTime() - started).toDouble / 1_000_000_000.0
          run += 1

        runFailed match
          case Some(reason) =>
            println(s"candidate fail device[$deviceIndex] $candidate reason=$reason")
          case None =>
            val avgSeconds = elapsedRuns.sum / elapsedRuns.length.toDouble
            val entriesPerSecond = batch.entryCount.toDouble / avgSeconds
            println(
              f"candidate ok device[$deviceIndex] $candidate elapsed=${avgSeconds}%.4fs entries/s=${entriesPerSecond}%.1f"
            )
            winner match
              case Some((_, currentEntriesPerSecond, _)) if currentEntriesPerSecond >= entriesPerSecond => ()
              case _ =>
                winner = Some((candidate, entriesPerSecond, avgSeconds))
    }
    winner

  private def applyCandidate(candidate: Candidate): Unit =
    sys.props.update(RangeNativeBlockSizeProperty, candidate.blockSize.toString)
    sys.props.update(RangeNativeMaxChunkHeroesProperty, candidate.maxChunkHeroes.toString)
    sys.props.update(RangeNativeMemoryPathProperty, candidate.memoryPath)

  private def saveCache(file: File, results: Vector[DeviceTuningResult]): Unit =
    val parent = file.getParentFile
    if parent != null then parent.mkdirs()
    val props = new Properties()
    props.setProperty("version", CacheVersion)
    props.setProperty("device.count", results.size.toString)
    results.zipWithIndex.foreach { case (result, idx) =>
      val prefix = s"device.$idx."
      props.setProperty(s"${prefix}index", result.index.toString)
      props.setProperty(s"${prefix}fingerprint", result.fingerprint)
      props.setProperty(s"${prefix}blockSize", result.winner.blockSize.toString)
      props.setProperty(s"${prefix}maxChunkHeroes", result.winner.maxChunkHeroes.toString)
      props.setProperty(s"${prefix}memoryPath", result.winner.memoryPath)
      props.setProperty(s"${prefix}entriesPerSecond", result.entriesPerSecond.toString)
      props.setProperty(s"${prefix}elapsedSeconds", result.elapsedSeconds.toString)
    }
    props.setProperty("updatedAtMillis", System.currentTimeMillis().toString)
    val out = new FileOutputStream(file)
    try props.store(out, "heads-up range gpu auto-tune cache")
    finally out.close()

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
    args.foldLeft(Config()) { (cfg, raw) =>
      val trimmed = raw.trim
      if !trimmed.startsWith("--") || !trimmed.contains("=") then
        throw new IllegalArgumentException(s"invalid argument '$raw' (expected --key=value)")
      val eq = trimmed.indexOf('=')
      val key = trimmed.substring(2, eq).trim
      val value = trimmed.substring(eq + 1).trim
      key match
        case "heroes" => cfg.copy(heroes = value.toInt)
        case "entriesPerHero" => cfg.copy(entriesPerHero = value.toInt)
        case "trials" => cfg.copy(trials = value.toInt)
        case "warmupRuns" => cfg.copy(warmupRuns = value.toInt)
        case "runs" => cfg.copy(runs = value.toInt)
        case "seedBase" => cfg.copy(seedBase = value.toLong)
        case "nativePath" => cfg.copy(nativePath = Some(value))
        case "cachePath" => cfg.copy(cachePath = value)
        case other => throw new IllegalArgumentException(s"unknown option '$other'")
    }
