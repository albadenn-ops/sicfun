package sicfun.holdem.bench
import sicfun.holdem.types.*
import sicfun.holdem.*
import sicfun.holdem.equity.*
import sicfun.holdem.gpu.*
import sicfun.holdem.cli.*

import sicfun.holdem.bench.BenchSupport.{card, hole}

import java.io.{File, FileOutputStream}
import java.util.Properties

import scala.collection.mutable.ArrayBuffer

/** Persistent auto-tuner for postflop CUDA launch parameters.
  *
  * Benchmarks candidate `(blockSize, maxChunkMatchups)` combinations on the
  * postflop native CUDA path and writes the winner to a cache file consumed by
  * [[HoldemPostflopNativeRuntime]].
  */
object HoldemPostflopGpuAutoTuner:
  private val ProviderProperty = "sicfun.postflop.provider"
  private val NativeEngineProperty = "sicfun.postflop.native.engine"
  private val NativeGpuPathProperty = "sicfun.postflop.native.gpu.path"
  private val PostflopAutoTuneProperty = "sicfun.postflop.autotune"
  private val PostflopCudaBlockSizeProperty = "sicfun.postflop.native.cuda.blockSize"
  private val PostflopCudaMaxChunkMatchupsProperty = "sicfun.postflop.native.cuda.maxChunkMatchups"

  private val CacheVersion = "1"
  private val DefaultCachePath = "data/postflop-autotune.properties"
  private val DefaultSeedBase = 0x6F31E52D9A4BC117L
  private val DefaultBlockCandidates = Vector(64, 96, 128, 160, 192, 256)
  private val DefaultChunkCandidates = Vector(256, 512, 1024, 2048, 4096)

  private final case class Config(
      villains: Int = 1024,
      trials: Int = 2_000,
      warmupRuns: Int = 1,
      runs: Int = 3,
      seedBase: Long = DefaultSeedBase,
      nativeGpuPath: Option[String] = None,
      cachePath: String = DefaultCachePath,
      blockCandidates: Vector[Int] = DefaultBlockCandidates,
      chunkCandidates: Vector[Int] = DefaultChunkCandidates
  )

  private final case class Candidate(
      blockSize: Int,
      maxChunkMatchups: Int
  ):
    override def toString: String =
      s"block=$blockSize chunkMatchups=$maxChunkMatchups"

  private final case class Spot(
      hero: HoleCards,
      board: Board,
      villains: Array[HoleCards]
  ):
    def villainCount: Int = villains.length

  private final case class DeviceTuningResult(
      index: Int,
      fingerprint: String,
      winner: Candidate,
      workUnitsPerSecond: Double,
      elapsedSeconds: Double,
      villains: Int,
      trials: Int
  )

  private final case class CandidateRun(
      candidate: Candidate,
      workUnitsPerSecond: Double,
      elapsedSeconds: Double
  )

  private final class TunerRunner(config: Config):
    private val propUpdates = Seq(
      ProviderProperty -> Some("native"),
      NativeEngineProperty -> Some("cuda"),
      PostflopAutoTuneProperty -> Some("false"),
      NativeGpuPathProperty -> config.nativeGpuPath
    )

    def run(): Unit =
      withSystemProperties(propUpdates) {
        HoldemPostflopNativeRuntime.resetLoadCacheForTests()
        val availability = HoldemPostflopNativeRuntime.availability

        printHeader(availability)
        ensureRuntimeAvailable(availability)

        val deviceIndex = 0
        val fingerprint = requireDeviceFingerprint(deviceIndex)
        val spot = buildSpot(config.villains, config.seedBase)
        printSpot(deviceIndex, fingerprint, spot)

        val winner = selectWinner(spot, buildCandidates()).getOrElse {
          throw new IllegalStateException("no successful postflop CUDA candidate")
        }
        println(
          f"winner device[$deviceIndex]: ${winner.candidate} elapsed=${winner.elapsedSeconds}%.4fs work/s=${winner.workUnitsPerSecond}%.1f"
        )

        val result = DeviceTuningResult(
          index = deviceIndex,
          fingerprint = fingerprint,
          winner = winner.candidate,
          workUnitsPerSecond = winner.workUnitsPerSecond,
          elapsedSeconds = winner.elapsedSeconds,
          villains = spot.villainCount,
          trials = config.trials
        )
        saveCache(new File(config.cachePath), Vector(result))
        println(s"cache written: ${new File(config.cachePath).getAbsolutePath}")
      }

    private def printHeader(availability: HoldemPostflopNativeRuntime.Availability): Unit =
      println("postflop gpu auto-tuner")
      println(
        s"villains=${config.villains}, trials=${config.trials}, warmupRuns=${config.warmupRuns}, " +
          s"runs=${config.runs}, seedBase=${config.seedBase}"
      )
      println(s"provider=${availability.provider}, available=${availability.available}, detail=${availability.detail}")

    private def ensureRuntimeAvailable(availability: HoldemPostflopNativeRuntime.Availability): Unit =
      if !availability.available || availability.provider != "native" then
        throw new IllegalStateException(s"native postflop provider unavailable: ${availability.detail}")
      val deviceCount = HoldemPostflopNativeGpuBindings.cudaDeviceCount()
      if deviceCount <= 0 then
        throw new IllegalStateException("no CUDA devices found")

    private def requireDeviceFingerprint(deviceIndex: Int): String =
      val fingerprint = Option(HoldemPostflopNativeGpuBindings.cudaDeviceInfo(deviceIndex)).getOrElse("").trim
      if fingerprint.isEmpty then
        throw new IllegalStateException(s"unable to resolve CUDA fingerprint for device index $deviceIndex")
      fingerprint

    private def printSpot(deviceIndex: Int, fingerprint: String, spot: Spot): Unit =
      println(s"tuning device[$deviceIndex]=$fingerprint")
      val boardToken = spot.board.cards.map(_.toToken).mkString
      println(s"spot: hero=${spot.hero.toToken} board=$boardToken villains=${spot.villainCount}")

    private def buildCandidates(): Vector[Candidate] =
      for
        block <- config.blockCandidates
        chunk <- config.chunkCandidates
      yield Candidate(blockSize = block, maxChunkMatchups = chunk)

    private def selectWinner(spot: Spot, candidates: Vector[Candidate]): Option[CandidateRun] =
      candidates.foldLeft(Option.empty[CandidateRun]) { (winner, candidate) =>
        measureCandidate(spot, candidate) match
          case Left(reason) =>
            println(s"candidate fail $candidate reason=$reason")
            winner
          case Right(run) =>
            println(
              f"candidate ok ${run.candidate} elapsed=${run.elapsedSeconds}%.4fs work/s=${run.workUnitsPerSecond}%.1f"
            )
            winner match
              case Some(current) if current.workUnitsPerSecond >= run.workUnitsPerSecond => winner
              case _ => Some(run)
      }

    private def measureCandidate(spot: Spot, candidate: Candidate): Either[String, CandidateRun] =
      applyCandidate(candidate)
      warmupCandidate(spot).flatMap { _ =>
        val elapsedRuns = new Array[Double](config.runs)
        var run = 0
        var runFailed = Option.empty[String]
        while run < config.runs && runFailed.isEmpty do
          val started = System.nanoTime()
          runOne(spot, trials = config.trials, seedBase = config.seedBase + 1000L + run.toLong) match
            case Left(reason) =>
              runFailed = Some(reason)
            case Right(rows) =>
              if rows.length != spot.villainCount then
                runFailed = Some(s"result length mismatch expected=${spot.villainCount} actual=${rows.length}")
              else
                elapsedRuns(run) = math.max(1L, System.nanoTime() - started).toDouble / 1_000_000_000.0
          run += 1

        runFailed match
          case Some(reason) => Left(reason)
          case None =>
            val avgSeconds = elapsedRuns.sum / elapsedRuns.length.toDouble
            val workUnits = spot.villainCount.toDouble * config.trials.toDouble
            Right(
              CandidateRun(
                candidate = candidate,
                workUnitsPerSecond = workUnits / avgSeconds,
                elapsedSeconds = avgSeconds
              )
            )
      }

    private def warmupCandidate(spot: Spot): Either[String, Unit] =
      var warmup = 0
      var warmupFailed = Option.empty[String]
      while warmup < config.warmupRuns && warmupFailed.isEmpty do
        runOne(spot, trials = config.trials, seedBase = config.seedBase + warmup.toLong) match
          case Left(reason) => warmupFailed = Some(reason)
          case Right(_) => ()
        warmup += 1
      warmupFailed match
        case Some(reason) => Left(reason)
        case None => Right(())

  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    validateConfig(config)
    new TunerRunner(config).run()

  private def validateConfig(config: Config): Unit =
    require(config.villains > 0, "villains must be positive")
    require(config.trials > 0, "trials must be positive")
    require(config.warmupRuns >= 0, "warmupRuns must be non-negative")
    require(config.runs > 0, "runs must be positive")
    require(config.blockCandidates.nonEmpty, "blockCandidates must be non-empty")
    require(config.chunkCandidates.nonEmpty, "chunkCandidates must be non-empty")

  private def runOne(
      spot: Spot,
      trials: Int,
      seedBase: Long
  ): Either[String, Array[EquityResultWithError]] =
    HoldemPostflopNativeRuntime.computePostflopBatch(
      hero = spot.hero,
      board = spot.board,
      villains = spot.villains,
      trials = trials,
      seedBase = seedBase
    )

  private def applyCandidate(candidate: Candidate): Unit =
    sys.props.update(PostflopCudaBlockSizeProperty, candidate.blockSize.toString)
    sys.props.update(PostflopCudaMaxChunkMatchupsProperty, candidate.maxChunkMatchups.toString)

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
      props.setProperty(s"${prefix}maxChunkMatchups", result.winner.maxChunkMatchups.toString)
      props.setProperty(s"${prefix}workUnitsPerSecond", result.workUnitsPerSecond.toString)
      props.setProperty(s"${prefix}elapsedSeconds", result.elapsedSeconds.toString)
      props.setProperty(s"${prefix}villains", result.villains.toString)
      props.setProperty(s"${prefix}trials", result.trials.toString)
    }
    props.setProperty("updatedAtMillis", System.currentTimeMillis().toString)
    val out = new FileOutputStream(file)
    try props.store(out, "postflop gpu auto-tune cache")
    finally out.close()

  private def buildSpot(villainCount: Int, seedBase: Long): Spot =
    val hero = hole("Ac", "Kh")
    val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    val dead = (hero.toVector ++ board.cards).map(sicfun.core.CardId.toId).toSet

    val villains = ArrayBuffer.empty[HoleCards]
    val size = HoleCardsIndex.size
    val start = (math.abs((seedBase ^ (seedBase >>> 32)).toInt) % size + size) % size
    val step = 37 // coprime with 1326 to enumerate all indices without repetition
    var i = 0
    while i < size && villains.length < villainCount do
      val id = (start + i * step) % size
      val hand = HoleCardsIndex.byId(id)
      val firstId = sicfun.core.CardId.toId(hand.first)
      val secondId = sicfun.core.CardId.toId(hand.second)
      if !dead.contains(firstId) && !dead.contains(secondId) then
        villains += hand
      i += 1

    if villains.length < villainCount then
      throw new IllegalStateException(
        s"unable to build postflop villain batch size=$villainCount with fixed hero/board"
      )

    Spot(hero = hero, board = board, villains = villains.toArray)

  private def parseArgs(args: Vector[String]): Config =
    val options = CliHelpers.requireOptions(args)
    CliHelpers.requireNoUnknownOptions(options, Set(
      "villains",
      "trials",
      "warmupRuns",
      "runs",
      "seedBase",
      "nativeGpuPath",
      "cachePath",
      "blockCandidates",
      "chunkCandidates"
    ))

    Config(
      villains = CliHelpers.requireIntOption(options, "villains", 1024),
      trials = CliHelpers.requireIntOption(options, "trials", 2_000),
      warmupRuns = CliHelpers.requireIntOption(options, "warmupRuns", 1),
      runs = CliHelpers.requireIntOption(options, "runs", 3),
      seedBase = CliHelpers.requireLongOption(options, "seedBase", DefaultSeedBase),
      nativeGpuPath = options.get("nativeGpuPath"),
      cachePath = options.getOrElse("cachePath", DefaultCachePath),
      blockCandidates = CliHelpers.optionalPositiveIntList(options, "blockCandidates").map(_.distinct).getOrElse(DefaultBlockCandidates),
      chunkCandidates = CliHelpers.optionalPositiveIntList(options, "chunkCandidates").map(_.distinct).getOrElse(DefaultChunkCandidates)
    )

  private def withSystemProperties[A](updates: Seq[(String, Option[String])])(thunk: => A): A =
    val previous = updates.map { case (key, _) => key -> sys.props.get(key) }.toMap
    updates.foreach {
      case (key, Some(value)) => sys.props.update(key, value)
      case (key, None) => sys.props.remove(key)
    }
    try thunk
    finally
      previous.foreach {
        case (key, Some(value)) => sys.props.update(key, value)
        case (key, None) => sys.props.remove(key)
      }
