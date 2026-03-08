package sicfun.holdem

import sicfun.core.Card

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

  def main(args: Array[String]): Unit =
    val config = parseArgs(args.toVector)
    require(config.villains > 0, "villains must be positive")
    require(config.trials > 0, "trials must be positive")
    require(config.warmupRuns >= 0, "warmupRuns must be non-negative")
    require(config.runs > 0, "runs must be positive")
    require(config.blockCandidates.nonEmpty, "blockCandidates must be non-empty")
    require(config.chunkCandidates.nonEmpty, "chunkCandidates must be non-empty")

    val propUpdates = Seq(
      ProviderProperty -> Some("native"),
      NativeEngineProperty -> Some("cuda"),
      PostflopAutoTuneProperty -> Some("false"),
      NativeGpuPathProperty -> config.nativeGpuPath
    )

    withSystemProperties(propUpdates) {
      HoldemPostflopNativeRuntime.resetLoadCacheForTests()
      val availability = HoldemPostflopNativeRuntime.availability

      println("postflop gpu auto-tuner")
      println(
        s"villains=${config.villains}, trials=${config.trials}, warmupRuns=${config.warmupRuns}, " +
          s"runs=${config.runs}, seedBase=${config.seedBase}"
      )
      println(s"provider=${availability.provider}, available=${availability.available}, detail=${availability.detail}")
      if !availability.available || availability.provider != "native" then
        throw new IllegalStateException(s"native postflop provider unavailable: ${availability.detail}")

      val deviceCount = HoldemPostflopNativeGpuBindings.cudaDeviceCount()
      if deviceCount <= 0 then
        throw new IllegalStateException("no CUDA devices found")

      val deviceIndex = 0
      val fingerprint = Option(HoldemPostflopNativeGpuBindings.cudaDeviceInfo(deviceIndex)).getOrElse("").trim
      if fingerprint.isEmpty then
        throw new IllegalStateException(s"unable to resolve CUDA fingerprint for device index $deviceIndex")

      val spot = buildSpot(config.villains, config.seedBase)
      println(s"tuning device[$deviceIndex]=$fingerprint")
      val boardToken = spot.board.cards.map(_.toToken).mkString
      println(s"spot: hero=${spot.hero.toToken} board=$boardToken villains=${spot.villainCount}")

      val candidates =
        for
          block <- config.blockCandidates
          chunk <- config.chunkCandidates
        yield Candidate(blockSize = block, maxChunkMatchups = chunk)

      val winnerOpt = tune(spot, config, candidates)
      val winner = winnerOpt.getOrElse {
        throw new IllegalStateException("no successful postflop CUDA candidate")
      }
      println(
        f"winner device[$deviceIndex]: ${winner._1} elapsed=${winner._3}%.4fs work/s=${winner._2}%.1f"
      )

      val result = DeviceTuningResult(
        index = deviceIndex,
        fingerprint = fingerprint,
        winner = winner._1,
        workUnitsPerSecond = winner._2,
        elapsedSeconds = winner._3,
        villains = spot.villainCount,
        trials = config.trials
      )
      saveCache(new File(config.cachePath), Vector(result))
      println(s"cache written: ${new File(config.cachePath).getAbsolutePath}")
    }

  private def tune(
      spot: Spot,
      config: Config,
      candidates: Vector[Candidate]
  ): Option[(Candidate, Double, Double)] =
    var winner: Option[(Candidate, Double, Double)] = None
    candidates.foreach { candidate =>
      applyCandidate(candidate)

      var warmup = 0
      var warmupFailed = false
      while warmup < config.warmupRuns && !warmupFailed do
        runOne(spot, trials = config.trials, seedBase = config.seedBase + warmup.toLong) match
          case Left(_) => warmupFailed = true
          case Right(_) => ()
        warmup += 1

      if !warmupFailed then
        val elapsedRuns = new Array[Double](config.runs)
        var runFailed: Option[String] = None
        var run = 0
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
          case Some(reason) =>
            println(s"candidate fail $candidate reason=$reason")
          case None =>
            val avgSeconds = elapsedRuns.sum / elapsedRuns.length.toDouble
            val workUnits = spot.villainCount.toDouble * config.trials.toDouble
            val workUnitsPerSecond = workUnits / avgSeconds
            println(
              f"candidate ok $candidate elapsed=${avgSeconds}%.4fs work/s=${workUnitsPerSecond}%.1f"
            )
            winner match
              case Some((_, currentWorkPerSecond, _)) if currentWorkPerSecond >= workUnitsPerSecond => ()
              case _ =>
                winner = Some((candidate, workUnitsPerSecond, avgSeconds))
    }
    winner

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
    def card(token: String): Card =
      Card.parse(token).getOrElse(throw new IllegalArgumentException(s"invalid card token: $token"))

    def hole(a: String, b: String): HoleCards =
      HoleCards.from(Vector(card(a), card(b)))

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
    val options = args.flatMap { token =>
      token.split("=", 2) match
        case Array(key, value) if key.startsWith("--") && value.nonEmpty =>
          Some(key.drop(2) -> value)
        case _ =>
          None
    }.toMap

    Config(
      villains = options.get("villains").flatMap(_.toIntOption).getOrElse(1024),
      trials = options.get("trials").flatMap(_.toIntOption).getOrElse(2_000),
      warmupRuns = options.get("warmupRuns").flatMap(_.toIntOption).getOrElse(1),
      runs = options.get("runs").flatMap(_.toIntOption).getOrElse(3),
      seedBase = options.get("seedBase").flatMap(_.toLongOption).getOrElse(DefaultSeedBase),
      nativeGpuPath = options.get("nativeGpuPath"),
      cachePath = options.getOrElse("cachePath", DefaultCachePath),
      blockCandidates = parsePositiveList(options.get("blockCandidates")).getOrElse(DefaultBlockCandidates),
      chunkCandidates = parsePositiveList(options.get("chunkCandidates")).getOrElse(DefaultChunkCandidates)
    )

  private def parsePositiveList(raw: Option[String]): Option[Vector[Int]] =
    raw.map(_.trim).filter(_.nonEmpty).flatMap { text =>
      val parsed = text.split(',').toVector.map(_.trim).filter(_.nonEmpty).flatMap(_.toIntOption).filter(_ > 0)
      if parsed.nonEmpty then Some(parsed.distinct) else None
    }

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
