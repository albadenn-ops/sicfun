package sicfun.holdem

import java.io.{File, FileInputStream}
import java.util.Properties
import java.util.concurrent.atomic.AtomicReference

/** Runtime wrapper for CSR-based hero-vs-range Monte Carlo evaluation.
  *
  * The CSR batch shape is:
  *   - `heroIds.length = H`
  *   - `offsets.length = H + 1`
  *   - villain entry span for hero `h` is `[offsets(h), offsets(h+1))`
  *   - `villainIds.length = keyMaterial.length = probabilities.length = offsets(H)`
  */
object HeadsUpRangeGpuRuntime:
  private val ProviderProperty = "sicfun.gpu.provider"
  private val ProviderEnv = "sicfun_GPU_PROVIDER"
  private val RangeAutoTuneProperty = "sicfun.gpu.range.autotune"
  private val RangeAutoTuneEnv = "sicfun_GPU_RANGE_AUTOTUNE"
  private val RangeAutoTuneCachePathProperty = "sicfun.gpu.range.autotune.cachePath"
  private val RangeAutoTuneCachePathEnv = "sicfun_GPU_RANGE_AUTOTUNE_CACHE_PATH"
  private val RangeNativeBlockSizeProperty = "sicfun.gpu.native.range.cuda.blockSize"
  private val RangeNativeBlockSizeEnv = "sicfun_GPU_RANGE_CUDA_BLOCK_SIZE"
  private val RangeNativeMaxChunkHeroesProperty = "sicfun.gpu.native.range.cuda.maxChunkHeroes"
  private val RangeNativeMaxChunkHeroesEnv = "sicfun_GPU_RANGE_CUDA_MAX_CHUNK_HEROES"
  private val RangeNativeMemoryPathProperty = "sicfun.gpu.native.range.memoryPath"
  private val RangeNativeMemoryPathEnv = "sicfun_GPU_RANGE_MEMORY_PATH"
  private val RangeAutoTuneCacheVersion = "1"
  private val DefaultRangeAutoTuneCachePath = "data/headsup-range-autotune.properties"
  private val appliedRangeTuneFingerprintRef = new AtomicReference[String](null)

  def computeRangeBatchMonteCarloCsr(
      heroIds: Array[Int],
      offsets: Array[Int],
      villainIds: Array[Int],
      keyMaterial: Array[Long],
      probabilities: Array[Float],
      trials: Int,
      monteCarloSeedBase: Long
  ): Either[String, Array[EquityResultWithError]] =
    try
      validateCsrShape(heroIds, offsets, villainIds, keyMaterial, probabilities, trials)
      configuredProvider match
        case "disabled" =>
          Left("GPU provider is disabled")
        case "cpu-emulated" =>
          Right(
            computeCpuEmulatedRangeBatch(
              heroIds,
              offsets,
              villainIds,
              keyMaterial,
              probabilities,
              trials,
              monteCarloSeedBase
            )
          )
        case "native" =>
          val availability = HeadsUpGpuRuntime.availability
          if !availability.available || availability.provider != "native" then
            if HeadsUpGpuRuntime.allowCpuFallbackOnGpuFailure then
              Right(
                computeCpuEmulatedRangeBatch(
                  heroIds,
                  offsets,
                  villainIds,
                  keyMaterial,
                  probabilities,
                  trials,
                  monteCarloSeedBase
                )
              )
            else
              Left(s"native provider unavailable: ${availability.detail}")
          else
            maybeApplyCachedRangeAutoTune(deviceIndex = 0)
            val wins = new Array[Float](heroIds.length)
            val ties = new Array[Float](heroIds.length)
            val losses = new Array[Float](heroIds.length)
            val stderrs = new Array[Float](heroIds.length)
            val status =
              HeadsUpGpuNativeBindings.computeRangeBatchMonteCarloCsr(
                heroIds,
                offsets,
                villainIds,
                keyMaterial,
                probabilities,
                trials,
                monteCarloSeedBase,
                wins,
                ties,
                losses,
                stderrs
              )
            if status != 0 then
              if HeadsUpGpuRuntime.allowCpuFallbackOnGpuFailure then
                Right(
                  computeCpuEmulatedRangeBatch(
                    heroIds,
                    offsets,
                    villainIds,
                    keyMaterial,
                    probabilities,
                    trials,
                    monteCarloSeedBase
                  )
                )
              else
                Left(describeNativeStatus(status))
            else
              Right(fromNativeArrays(wins, ties, losses, stderrs))
        case other =>
          Left(s"provider '$other' is not supported for CSR range evaluation")
    catch
      case ex: UnsatisfiedLinkError =>
        if HeadsUpGpuRuntime.allowCpuFallbackOnGpuFailure then
          Right(
            computeCpuEmulatedRangeBatch(
              heroIds,
              offsets,
              villainIds,
              keyMaterial,
              probabilities,
              trials,
              monteCarloSeedBase
            )
          )
        else
          Left(s"native GPU symbols not found: ${ex.getMessage}")
      case ex: Throwable =>
        val detail = Option(ex.getMessage).map(_.trim).filter(_.nonEmpty).getOrElse(ex.getClass.getSimpleName)
        Left(detail)

  def computeRangeBatchMonteCarloCsrOnDevice(
      deviceIndex: Int,
      heroIds: Array[Int],
      offsets: Array[Int],
      villainIds: Array[Int],
      keyMaterial: Array[Long],
      probabilities: Array[Float],
      trials: Int,
      monteCarloSeedBase: Long
  ): Either[String, Array[EquityResultWithError]] =
    try
      validateCsrShape(heroIds, offsets, villainIds, keyMaterial, probabilities, trials)
      val availability = HeadsUpGpuRuntime.availability
      if !availability.available || availability.provider != "native" then
        Left(s"native provider unavailable: ${availability.detail}")
      else
        maybeApplyCachedRangeAutoTune(deviceIndex = deviceIndex)
        val wins = new Array[Float](heroIds.length)
        val ties = new Array[Float](heroIds.length)
        val losses = new Array[Float](heroIds.length)
        val stderrs = new Array[Float](heroIds.length)
        val status =
          HeadsUpGpuNativeBindings.computeRangeBatchMonteCarloCsrOnDevice(
            deviceIndex,
            heroIds,
            offsets,
            villainIds,
            keyMaterial,
            probabilities,
            trials,
            monteCarloSeedBase,
            wins,
            ties,
            losses,
            stderrs
          )
        if status != 0 then Left(describeNativeStatus(status))
        else Right(fromNativeArrays(wins, ties, losses, stderrs))
    catch
      case ex: UnsatisfiedLinkError =>
        Left(s"native GPU symbols not found: ${ex.getMessage}")
      case ex: Throwable =>
        val detail = Option(ex.getMessage).map(_.trim).filter(_.nonEmpty).getOrElse(ex.getClass.getSimpleName)
        Left(detail)

  private def configuredProvider: String =
    GpuRuntimeSupport.resolveNonEmptyLower(ProviderProperty, ProviderEnv).getOrElse("native")

  private def maybeApplyCachedRangeAutoTune(deviceIndex: Int): Unit =
    if !rangeAutoTuneEnabled then ()
    else if hasExplicitRangeNativeConfig then ()
    else
      val deviceCount = safeCudaDeviceCount()
      if deviceCount <= 0 then ()
      else
        val boundedDeviceIndex = math.max(0, math.min(deviceIndex, deviceCount - 1))
        val fingerprint = safeCudaDeviceFingerprint(boundedDeviceIndex)
        if fingerprint.isEmpty then ()
        else
          val appliedKey = s"$boundedDeviceIndex|$fingerprint"
          if appliedRangeTuneFingerprintRef.get() == appliedKey then ()
          else
            loadRangeAutoTuneDecision(resolvedRangeAutoTuneCacheFile, boundedDeviceIndex, fingerprint) match
              case Some(decision) =>
                sys.props.update(RangeNativeBlockSizeProperty, decision.blockSize.toString)
                sys.props.update(RangeNativeMaxChunkHeroesProperty, decision.maxChunkHeroes.toString)
                sys.props.update(RangeNativeMemoryPathProperty, decision.memoryPath)
                appliedRangeTuneFingerprintRef.set(appliedKey)
                GpuRuntimeSupport.log(
                  s"range-autotune: applied cached config for device=$boundedDeviceIndex " +
                    s"(block=${decision.blockSize}, chunkHeroes=${decision.maxChunkHeroes}, memoryPath=${decision.memoryPath})"
                )
              case None => ()

  private final case class RangeAutoTuneDecision(
      blockSize: Int,
      maxChunkHeroes: Int,
      memoryPath: String
  )

  private def loadRangeAutoTuneDecision(
      file: File,
      deviceIndex: Int,
      fingerprint: String
  ): Option[RangeAutoTuneDecision] =
    import scala.util.boundary, boundary.break
    if !file.isFile then None
    else
      val props = new Properties()
      val in = new FileInputStream(file)
      try props.load(in)
      finally in.close()

      val version = Option(props.getProperty("version")).map(_.trim).getOrElse("")
      if version != RangeAutoTuneCacheVersion then None
      else
        val count = GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty("device.count")).getOrElse(0)
        if count <= 0 then None
        else
          boundary:
            var idx = 0
            while idx < count do
              val prefix = s"device.$idx."
              val cachedIndex =
                GpuRuntimeSupport.parseNonNegativeIntOpt(props.getProperty(s"${prefix}index")).getOrElse(-1)
              val cachedFingerprint = Option(props.getProperty(s"${prefix}fingerprint")).map(_.trim).getOrElse("")
              if cachedIndex == deviceIndex && cachedFingerprint == fingerprint then
                val blockSizeOpt = GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty(s"${prefix}blockSize"))
                val maxChunkOpt = GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty(s"${prefix}maxChunkHeroes"))
                val memoryPathOpt =
                  Option(props.getProperty(s"${prefix}memoryPath"))
                    .map(_.trim.toLowerCase)
                    .filter(path => path == "global" || path == "readonly" || path == "read-only" || path == "ldg")
                    .map {
                      case "read-only" | "ldg" => "readonly"
                      case other => other
                    }
                if blockSizeOpt.nonEmpty && maxChunkOpt.nonEmpty && memoryPathOpt.nonEmpty then
                  break(Some(
                    RangeAutoTuneDecision(
                      blockSize = blockSizeOpt.get,
                      maxChunkHeroes = maxChunkOpt.get,
                      memoryPath = memoryPathOpt.get
                    )
                  ))
              idx += 1
            None

  private def resolvedRangeAutoTuneCacheFile: File =
    GpuRuntimeSupport.resolveFile(
      RangeAutoTuneCachePathProperty,
      RangeAutoTuneCachePathEnv,
      DefaultRangeAutoTuneCachePath
    )

  private def rangeAutoTuneEnabled: Boolean =
    val raw = GpuRuntimeSupport.resolveNonEmptyLower(RangeAutoTuneProperty, RangeAutoTuneEnv)
    raw match
      case Some("0" | "false" | "no" | "off") => false
      case _ => true

  private def hasExplicitRangeNativeConfig: Boolean =
    GpuRuntimeSupport.isConfigured(RangeNativeBlockSizeProperty, RangeNativeBlockSizeEnv) ||
    GpuRuntimeSupport.isConfigured(RangeNativeMaxChunkHeroesProperty, RangeNativeMaxChunkHeroesEnv) ||
    GpuRuntimeSupport.isConfigured(RangeNativeMemoryPathProperty, RangeNativeMemoryPathEnv)

  private def safeCudaDeviceCount(): Int =
    try HeadsUpGpuNativeBindings.cudaDeviceCount()
    catch
      case _: Throwable => 0

  private def safeCudaDeviceFingerprint(deviceIndex: Int): String =
    try Option(HeadsUpGpuNativeBindings.cudaDeviceInfo(deviceIndex)).map(_.trim).getOrElse("")
    catch
      case _: Throwable => ""

  private def fromNativeArrays(
      wins: Array[Float],
      ties: Array[Float],
      losses: Array[Float],
      stderrs: Array[Float]
  ): Array[EquityResultWithError] =
    val out = new Array[EquityResultWithError](wins.length)
    var idx = 0
    while idx < wins.length do
      out(idx) = EquityResultWithError(
        wins(idx).toDouble,
        ties(idx).toDouble,
        losses(idx).toDouble,
        stderrs(idx).toDouble
      )
      idx += 1
    out

  private def validateCsrShape(
      heroIds: Array[Int],
      offsets: Array[Int],
      villainIds: Array[Int],
      keyMaterial: Array[Long],
      probabilities: Array[Float],
      trials: Int
  ): Unit =
    require(heroIds != null, "heroIds must be non-null")
    require(offsets != null, "offsets must be non-null")
    require(villainIds != null, "villainIds must be non-null")
    require(keyMaterial != null, "keyMaterial must be non-null")
    require(probabilities != null, "probabilities must be non-null")
    require(trials > 0, "trials must be positive")
    require(offsets.length == heroIds.length + 1, "offsets length must equal heroIds length + 1")
    require(villainIds.length == keyMaterial.length, "villainIds and keyMaterial length must match")
    require(villainIds.length == probabilities.length, "villainIds and probabilities length must match")
    require(offsets.nonEmpty && offsets(0) == 0, "offsets(0) must be 0")
    require(offsets(offsets.length - 1) == villainIds.length, "offsets(last) must equal villainIds length")
    var h = 0
    while h < heroIds.length do
      require(heroIds(h) >= 0 && heroIds(h) < HoleCardsIndex.size, s"invalid hero id at index=$h")
      require(offsets(h) <= offsets(h + 1), s"offsets must be non-decreasing at index=$h")
      h += 1
    var i = 0
    while i < villainIds.length do
      require(villainIds(i) >= 0 && villainIds(i) < HoleCardsIndex.size, s"invalid villain id at index=$i")
      val p = probabilities(i)
      require(java.lang.Float.isFinite(p) && p >= 0.0f, s"invalid probability at index=$i")
      i += 1

  private def computeCpuEmulatedRangeBatch(
      heroIds: Array[Int],
      offsets: Array[Int],
      villainIds: Array[Int],
      keyMaterial: Array[Long],
      probabilities: Array[Float],
      trials: Int,
      monteCarloSeedBase: Long
  ): Array[EquityResultWithError] =
    val out = new Array[EquityResultWithError](heroIds.length)
    val mode = HeadsUpEquityTable.Mode.MonteCarlo(trials)
    var h = 0
    while h < heroIds.length do
      val hero = HoleCardsIndex.byId(heroIds(h))
      val start = offsets(h)
      val end = offsets(h + 1)
      var weightedWin = 0.0
      var weightedTie = 0.0
      var weightedLoss = 0.0
      var weightedStdErrSq = 0.0
      var weightSum = 0.0
      var i = start
      while i < end do
        val p = probabilities(i).toDouble
        if p > 0.0 then
          val villain = HoleCardsIndex.byId(villainIds(i))
          require(
            HoleCardsIndex.areDisjoint(hero, villain),
            s"overlapping hole cards for heroIndex=$h entryIndex=$i"
          )
          val entryResult =
            HeadsUpEquityTable.computeEquityDeterministic(
              hero = hero,
              villain = villain,
              mode = mode,
              monteCarloSeedBase = monteCarloSeedBase,
              keyMaterial = keyMaterial(i)
            )
          weightedWin += p * entryResult.win
          weightedTie += p * entryResult.tie
          weightedLoss += p * entryResult.loss
          val weightedStdErr = p * entryResult.stderr
          weightedStdErrSq += weightedStdErr * weightedStdErr
          weightSum += p
        i += 1

      if weightSum > 0.0 then
        out(h) = EquityResultWithError(
          win = weightedWin / weightSum,
          tie = weightedTie / weightSum,
          loss = weightedLoss / weightSum,
          stderr = math.sqrt(weightedStdErrSq) / weightSum
        )
      else
        out(h) = EquityResultWithError(0.0, 0.0, 0.0, 0.0)
      h += 1
    out

  private def describeNativeStatus(status: Int): String =
    GpuRuntimeSupport.describeNativeStatus(status)
