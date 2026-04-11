package sicfun.holdem.gpu
import sicfun.holdem.types.*
import sicfun.holdem.*
import sicfun.holdem.equity.*

import java.io.{File, FileInputStream}
import java.util.Properties
import java.util.concurrent.atomic.AtomicReference

/** Runtime wrapper for CSR-based hero-vs-range Monte Carlo equity evaluation.
  *
  * This object extends the GPU batch computation model from single-matchup equity
  * ([[HeadsUpGpuRuntime]]) to ''range-vs-hero'' equity. Each hero hand is evaluated
  * against a weighted set of villain hands, with the results aggregated as a
  * probability-weighted sum.
  *
  * ==CSR Batch Layout==
  * The villain ranges are represented as a Compressed Sparse Row (CSR) structure:
  *   - `heroIds.length = H` -- number of hero hands
  *   - `offsets.length = H + 1` -- CSR row pointers
  *   - villain entry span for hero `h` is `[offsets(h), offsets(h+1))`
  *   - `villainIds.length = keyMaterial.length = probabilities.length = offsets(H)`
  *
  * This layout enables a single JNI call to evaluate all heroes against their
  * respective villain ranges in one GPU kernel launch, minimising host-device
  * round-trip overhead.
  *
  * ==Provider Selection==
  * Uses the same `sicfun.gpu.provider` setting as [[HeadsUpGpuRuntime]]:
  *   - `"native"` -- dispatches to the CUDA CSR kernel via [[HeadsUpGpuNativeBindings]]
  *   - `"cpu-emulated"` -- JVM-side per-matchup deterministic computation (for testing)
  *   - `"disabled"` -- always returns `Left`
  *
  * ==Auto-Tuning==
  * When `sicfun.gpu.range.autotune` is enabled (default: true), the runtime loads
  * cached CUDA block-size and chunk-size parameters from a properties file keyed
  * by device fingerprint, avoiding the need for manual tuning per GPU model.
  */
object HeadsUpRangeGpuRuntime:
  private val ProviderProperty = "sicfun.gpu.provider"
  private val ProviderEnv = "sicfun_GPU_PROVIDER"
  private val NativePathProperty = "sicfun.gpu.native.path"
  private val NativePathEnv = "sicfun_GPU_NATIVE_PATH"
  private val NativeLibProperty = "sicfun.gpu.native.lib"
  private val NativeLibEnv = "sicfun_GPU_NATIVE_LIB"
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
  private val DefaultNativeLibrary = "sicfun_gpu_kernel"
  private val RangeAutoTuneCacheVersion = "1"
  private val DefaultRangeAutoTuneCachePath = "data/headsup-range-autotune.properties"
  private val appliedRangeTuneFingerprintRef = new AtomicReference[String](null)

  /** Computes probability-weighted Monte Carlo equity for each hero hand
    * against its villain range, using the CSR layout.
    *
    * @param heroIds             array of hero HoleCardsIndex ids (length H)
    * @param offsets             CSR row pointers (length H+1); offsets(h)..offsets(h+1) spans hero h's villains
    * @param villainIds          flat array of villain HoleCardsIndex ids
    * @param keyMaterial         per-entry seed material for deterministic Monte Carlo
    * @param probabilities       per-entry villain probability weights (must sum > 0 per hero)
    * @param trials              number of Monte Carlo trials per matchup
    * @param monteCarloSeedBase  global seed base combined with keyMaterial for per-entry seeding
    * @return `Right(results)` with one EquityResultWithError per hero, or `Left(reason)` on failure
    */
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

  /** Like [[computeRangeBatchMonteCarloCsr]] but targets a specific CUDA device
    * by index. Used for multi-GPU setups where each device handles a subset of work.
    *
    * @param deviceIndex  CUDA device ordinal (0-based)
    */
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

  /** Loads cached auto-tune parameters (blockSize, maxChunkHeroes, memoryPath) from
    * a properties file and applies them as system properties. The cache is keyed by
    * CUDA device fingerprint and native library identity, so different GPUs or DLL
    * versions automatically get their own tuning profiles.
    *
    * This is a no-op when:
    *  - Auto-tuning is disabled via config
    *  - The user has explicit CUDA config properties set (manual override wins)
    *  - No CUDA devices are available
    *  - The same fingerprint was already applied (idempotency guard)
    */
  private def maybeApplyCachedRangeAutoTune(deviceIndex: Int): Unit =
    if !rangeAutoTuneEnabled then ()
    else if hasExplicitRangeNativeConfig then ()
    else
      val deviceCount = safeCudaDeviceCount()
      if deviceCount <= 0 then ()
      else
        val boundedDeviceIndex = math.max(0, math.min(deviceIndex, deviceCount - 1))
        val cacheFile = resolvedRangeAutoTuneCacheFile
        val cacheMtime = if cacheFile.isFile then cacheFile.lastModified() else 0L
        val fingerprint = safeCudaDeviceFingerprint(boundedDeviceIndex)
        if fingerprint.isEmpty then ()
        else
          val appliedKey =
            s"$boundedDeviceIndex|$fingerprint|${cacheFile.getAbsolutePath}|$cacheMtime|$configuredNativeLibraryIdentity"
          if appliedRangeTuneFingerprintRef.get() == appliedKey then ()
          else
            loadRangeAutoTuneDecision(cacheFile, boundedDeviceIndex, fingerprint) match
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

  /** Reads a range auto-tune properties file and searches for a matching
    * device index + fingerprint entry. Returns the tuning decision if found,
    * or `None` if the file is missing, has an incompatible version, or
    * doesn't contain an entry for this device.
    */
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
        val cachedNativeLibraryIdentity = Option(props.getProperty("nativeLibraryIdentity")).map(_.trim).getOrElse("")
        if cachedNativeLibraryIdentity.isEmpty || cachedNativeLibraryIdentity != configuredNativeLibraryIdentity then None
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

  private def configuredNativeLibraryIdentity: String =
    GpuRuntimeSupport.resolveNonEmpty(NativePathProperty, NativePathEnv) match
      case Some(path) =>
        val file = new File(path)
        val mtime = if file.exists() then file.lastModified() else 0L
        s"path=${file.getAbsolutePath}|mtime=$mtime"
      case None =>
        val lib = GpuRuntimeSupport.resolveNonEmpty(NativeLibProperty, NativeLibEnv).getOrElse(DefaultNativeLibrary)
        s"lib=$lib"

  /** Converts parallel f32 result arrays (wins, ties, losses, stderrs) from the
    * native JNI call into an array of EquityResultWithError domain objects.
    */
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

  /** Validates the CSR batch layout invariants before sending data to native code.
    * Checks non-null arrays, consistent lengths, valid offsets ordering, valid
    * HoleCardsIndex ids, positive trials, and finite non-negative probabilities.
    * Throws IllegalArgumentException on any violation.
    */
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

  /** JVM-side CPU emulation of the CSR range batch computation.
    *
    * For each hero, iterates over its villain entries and computes a probability-weighted
    * equity using [[HeadsUpEquityTable.computeEquityDeterministic]]. The stderr is
    * propagated as the root-sum-of-squares of per-entry weighted standard errors.
    *
    * Zero-weight entries are skipped. If all entries for a hero have zero weight,
    * the result is all-zeros.
    */
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
