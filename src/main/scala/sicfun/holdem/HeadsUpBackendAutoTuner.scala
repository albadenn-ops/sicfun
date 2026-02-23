package sicfun.holdem

import sicfun.core.HandEvaluator

import java.io.{File, FileInputStream, FileOutputStream}
import java.util.Properties

/** Automatic CUDA kernel parameter tuner for heads-up equity table generation.
  *
  * When GPU backend is selected, this tuner benchmarks a set of candidate CUDA configurations
  * (block size + max chunk matchups) on a small representative workload and selects the fastest.
  * Results are cached to a properties file keyed by a workload signature (OS, architecture,
  * Java version, native library identity, table kind, trial count, and entry count).
  *
  * '''Activation:'''
  *   - Enabled by default when the GPU backend is requested
  *   - Disabled via `sicfun.gpu.autotune=false` / `sicfun_GPU_AUTOTUNE=false`
  *   - Skipped if explicit CUDA block size or chunk size is already configured
  *   - Skipped in exact mode (CPU-only path)
  *
  * '''Cache:'''
  *   - Default location: `data/headsup-backend-autotune.properties`
  *   - Configurable via `sicfun.gpu.autotune.cachePath` / `sicfun_GPU_AUTOTUNE_CACHE_PATH`
  *   - Invalidated when the workload signature changes (e.g. different native library version)
  */
object HeadsUpBackendAutoTuner:
  private final case class BatchData(
      packedKeys: Array[Long],
      keyMaterial: Array[Long]
  ):
    def size: Int = packedKeys.length

  private final case class CudaConfig(
      blockSize: Int,
      maxChunkMatchups: Int
  )

  private final case class CandidateResult(
      engine: String,
      blockSize: Option[Int],
      maxChunkMatchups: Option[Int],
      seconds: Double,
      detail: String
  )

  /** The selected backend configuration, either from a fresh benchmark or loaded from cache.
    *
    * @param engine           selected compute engine (e.g. `"cuda"` or `"cpu"`)
    * @param blockSize        CUDA thread block size (None if not applicable)
    * @param maxChunkMatchups max matchups per CUDA kernel launch (None if not applicable)
    * @param source           how this decision was made (`"tuned"` or `"cache"`)
    * @param detail           timing or cache hit details
    */
  final case class Decision(
      engine: String,
      blockSize: Option[Int],
      maxChunkMatchups: Option[Int],
      source: String,
      detail: String
  )

  private val AutoTuneProperty = "sicfun.gpu.autotune"
  private val AutoTuneEnv = "sicfun_GPU_AUTOTUNE"
  private val AutoTuneCachePathProperty = "sicfun.gpu.autotune.cachePath"
  private val AutoTuneCachePathEnv = "sicfun_GPU_AUTOTUNE_CACHE_PATH"
  private val NativeEngineProperty = "sicfun.gpu.native.engine"
  private val NativeEngineEnv = "sicfun_GPU_NATIVE_ENGINE"
  private val NativeCudaBlockSizeProperty = "sicfun.gpu.native.cuda.blockSize"
  private val NativeCudaBlockSizeEnv = "sicfun_GPU_CUDA_BLOCK_SIZE"
  private val NativeCudaMaxChunkProperty = "sicfun.gpu.native.cuda.maxChunkMatchups"
  private val NativeCudaMaxChunkEnv = "sicfun_GPU_CUDA_MAX_CHUNK_MATCHUPS"
  private val NativePathProperty = "sicfun.gpu.native.path"
  private val NativePathEnv = "sicfun_GPU_NATIVE_PATH"
  private val NativeLibProperty = "sicfun.gpu.native.lib"
  private val NativeLibEnv = "sicfun_GPU_NATIVE_LIB"
  private val DefaultNativeLibrary = "sicfun_gpu_kernel"
  private val CacheVersion = "2"
  private val DefaultCachePath = "data/headsup-backend-autotune.properties"
  private val TuneMaxEntries = 12000
  private val TuneMinEntries = 2000
  private val TuneMinTrials = 200
  private val TuneMaxTrials = 2000
  private val TuneSeedBase = 0x7C3A9B4D5E6F1021L
  private val CudaCandidates = Vector(
    CudaConfig(blockSize = 64, maxChunkMatchups = 256),
    CudaConfig(blockSize = 64, maxChunkMatchups = 384),
    CudaConfig(blockSize = 96, maxChunkMatchups = 384),
    CudaConfig(blockSize = 96, maxChunkMatchups = 448),
    CudaConfig(blockSize = 96, maxChunkMatchups = 512),
    CudaConfig(blockSize = 128, maxChunkMatchups = 384)
  )

  /** Runs auto-tuning if the GPU backend is selected and Monte Carlo mode is active.
    *
    * Called by [[GenerateHeadsUpTable]] and [[GenerateHeadsUpCanonicalTable]] before
    * table generation begins. Includes a CPU candidate to allow the tuner to fall back
    * to CPU if it outperforms the GPU on the representative workload.
    */
  def configureForGeneration(
      tableKind: String,
      mode: HeadsUpEquityTable.Mode,
      maxMatchups: Long,
      backend: HeadsUpEquityTable.ComputeBackend
  ): Unit =
    if backend != HeadsUpEquityTable.ComputeBackend.Gpu then
      ()
    else mode match
      case HeadsUpEquityTable.Mode.Exact =>
        println("gpu-autotune: skipped (exact mode uses dedicated CUDA exact kernel defaults)")
      case mc: HeadsUpEquityTable.Mode.MonteCarlo =>
        runAutoTune(
          tableKind = tableKind.trim.toLowerCase,
          mode = mc,
          maxMatchups = maxMatchups,
          includeCpuCandidate = true,
          tuneScope = "any"
        )

  /** Runs auto-tuning for the backend comparison harness (CUDA candidates only, no CPU fallback). */
  def configureForComparison(
      tableKind: String,
      mode: HeadsUpEquityTable.Mode,
      maxMatchups: Long
  ): Unit =
    mode match
      case HeadsUpEquityTable.Mode.Exact =>
        println("gpu-autotune: skipped for comparison (exact mode uses dedicated CUDA exact kernel defaults)")
      case mc: HeadsUpEquityTable.Mode.MonteCarlo =>
        runAutoTune(
          tableKind = tableKind.trim.toLowerCase,
          mode = mc,
          maxMatchups = maxMatchups,
          includeCpuCandidate = false,
          tuneScope = "cuda"
        )

  private def runAutoTune(
      tableKind: String,
      mode: HeadsUpEquityTable.Mode.MonteCarlo,
      maxMatchups: Long,
      includeCpuCandidate: Boolean,
      tuneScope: String
  ): Unit =
    if !autoTuneEnabled then
      println("gpu-autotune: disabled via sicfun.gpu.autotune/sicfun_GPU_AUTOTUNE")
      return

    if hasExplicitNativeRuntimeConfig then
      println("gpu-autotune: skipped (native engine/block/chunk explicitly configured)")
      return

    val availability = HeadsUpGpuRuntime.availability
    if !availability.available then
      println(s"gpu-autotune: skipped (provider unavailable: ${availability.detail})")
      return

    val batch = loadTuneBatch(tableKind, maxMatchups)
    if batch.size <= 0 then
      println("gpu-autotune: skipped (empty tune batch)")
      return

    val tuneTrials = clamp(mode.trials, TuneMinTrials, TuneMaxTrials)
    val tuneMode: HeadsUpEquityTable.Mode.MonteCarlo = HeadsUpEquityTable.Mode.MonteCarlo(tuneTrials)
    val cacheFile = resolvedCacheFile
    val signature = workloadSignature(tableKind, tuneTrials, batch.size, tuneScope)

    loadDecisionFromCache(cacheFile, signature) match
      case Some(cached) =>
        applyDecision(cached)
        printDecision(cached, fromCache = true, tuneTrials, batch.size)
      case None =>
        val cpu =
          if includeCpuCandidate then benchmarkCandidate(batch, tuneMode, "cpu", None, None).toVector
          else Vector.empty
        val cudaResults =
          CudaCandidates
            .filter(cfg => cfg.maxChunkMatchups <= batch.size)
            .flatMap(cfg =>
              benchmarkCandidate(
                batch = batch,
                mode = tuneMode,
                engine = "cuda",
                blockSize = Some(cfg.blockSize),
                maxChunkMatchups = Some(cfg.maxChunkMatchups)
              )
            )

        val all = cpu ++ cudaResults
        if all.isEmpty then
          println("gpu-autotune: no successful candidates, leaving runtime defaults unchanged")
        else
          val winner = all.minBy(_.seconds)
          val decision = Decision(
            engine = winner.engine,
            blockSize = winner.blockSize,
            maxChunkMatchups = winner.maxChunkMatchups,
            source = "tuned",
            detail = winner.detail
          )
          applyDecision(decision)
          saveDecisionToCache(cacheFile, signature, decision)
          printDecision(decision, fromCache = false, tuneTrials, batch.size)

  private def benchmarkCandidate(
      batch: BatchData,
      mode: HeadsUpEquityTable.Mode.MonteCarlo,
      engine: String,
      blockSize: Option[Int],
      maxChunkMatchups: Option[Int]
  ): Option[CandidateResult] =
    applyEngineSettings(engine, blockSize, maxChunkMatchups)
    HandEvaluator.clearCaches()
    val started = System.nanoTime()
    HeadsUpGpuRuntime.computeBatch(batch.packedKeys, batch.keyMaterial, mode, TuneSeedBase) match
      case Left(reason) =>
        println(s"gpu-autotune: candidate engine=$engine block=${showOpt(blockSize)} chunk=${showOpt(maxChunkMatchups)} failed ($reason)")
        None
      case Right(_) =>
        val elapsed = (System.nanoTime() - started).toDouble / 1_000_000_000.0
        println(
          f"gpu-autotune: candidate engine=$engine block=${showOpt(blockSize)} chunk=${showOpt(maxChunkMatchups)} elapsed=${elapsed}%.3fs"
        )
        Some(
          CandidateResult(
            engine = engine,
            blockSize = blockSize,
            maxChunkMatchups = maxChunkMatchups,
            seconds = elapsed,
            detail = f"elapsed=${elapsed}%.3fs"
          )
        )

  private def loadTuneBatch(tableKind: String, maxMatchups: Long): BatchData =
    val normalizedMax = math.max(1L, maxMatchups)
    tableKind match
      case "full" =>
        val total = HeadsUpEquityTable.totalMatchups
        val limit = tuneEntryLimit(total = total, requested = normalizedMax)
        val batch = HeadsUpEquityTable.selectFullBatch(limit)
        BatchData(batch.packedKeys, batch.keyMaterial)
      case _ =>
        val total = HeadsUpEquityCanonicalTable.totalCanonicalKeys.toLong
        val limit = tuneEntryLimit(total = total, requested = normalizedMax)
        val batch = HeadsUpEquityCanonicalTable.selectCanonicalBatch(limit)
        BatchData(batch.packedKeys, batch.keyMaterial)

  private def tuneEntryLimit(total: Long, requested: Long): Long =
    val cap = math.min(total, requested)
    val bounded = math.min(cap, TuneMaxEntries.toLong)
    math.max(1L, math.min(cap, math.max(TuneMinEntries.toLong, bounded)))

  private def applyDecision(decision: Decision): Unit =
    applyEngineSettings(decision.engine, decision.blockSize, decision.maxChunkMatchups)

  private def applyEngineSettings(
      engine: String,
      blockSize: Option[Int],
      maxChunkMatchups: Option[Int]
  ): Unit =
    sys.props.update(NativeEngineProperty, engine)
    blockSize match
      case Some(v) => sys.props.update(NativeCudaBlockSizeProperty, v.toString)
      case None => sys.props.remove(NativeCudaBlockSizeProperty)
    maxChunkMatchups match
      case Some(v) => sys.props.update(NativeCudaMaxChunkProperty, v.toString)
      case None => sys.props.remove(NativeCudaMaxChunkProperty)

  private def printDecision(
      decision: Decision,
      fromCache: Boolean,
      tuneTrials: Int,
      tuneEntries: Int
  ): Unit =
    val source = if fromCache then "cache" else "fresh"
    println(
      s"gpu-autotune: selected engine=${decision.engine}, block=${showOpt(decision.blockSize)}, " +
        s"chunk=${showOpt(decision.maxChunkMatchups)} (source=$source, trials=$tuneTrials, entries=$tuneEntries, ${decision.detail})"
    )

  private def loadDecisionFromCache(file: File, signature: String): Option[Decision] =
    if !file.exists() then return None
    val props = new Properties()
    val in = new FileInputStream(file)
    try props.load(in)
    finally in.close()

    val version = Option(props.getProperty("version")).getOrElse("")
    val cachedSignature = Option(props.getProperty("signature")).getOrElse("")
    if version != CacheVersion || cachedSignature != signature then
      None
    else
      val engine = Option(props.getProperty("engine")).map(_.trim.toLowerCase).filter(_.nonEmpty)
      engine.map { e =>
        Decision(
          engine = e,
          blockSize = parsePositiveIntOpt(props.getProperty("blockSize")),
          maxChunkMatchups = parsePositiveIntOpt(props.getProperty("maxChunkMatchups")),
          source = "cache",
          detail = Option(props.getProperty("detail")).getOrElse("cached")
        )
      }

  private def saveDecisionToCache(file: File, signature: String, decision: Decision): Unit =
    val parent = file.getParentFile
    if parent != null then parent.mkdirs()
    val props = new Properties()
    props.setProperty("version", CacheVersion)
    props.setProperty("signature", signature)
    props.setProperty("engine", decision.engine)
    decision.blockSize.foreach(v => props.setProperty("blockSize", v.toString))
    decision.maxChunkMatchups.foreach(v => props.setProperty("maxChunkMatchups", v.toString))
    props.setProperty("detail", decision.detail)
    props.setProperty("updatedAtMillis", System.currentTimeMillis().toString)
    val out = new FileOutputStream(file)
    try props.store(out, "heads-up backend autotune cache")
    finally out.close()

  private def workloadSignature(tableKind: String, trials: Int, tuneEntries: Int, tuneScope: String): String =
    val os = System.getProperty("os.name", "unknown").trim.toLowerCase
    val arch = System.getProperty("os.arch", "unknown").trim.toLowerCase
    val javaVersion = System.getProperty("java.version", "unknown").trim.toLowerCase
    val libId = nativeLibraryIdentity
    s"v=$CacheVersion|scope=$tuneScope|table=$tableKind|trials=$trials|entries=$tuneEntries|os=$os|arch=$arch|java=$javaVersion|lib=$libId"

  private def nativeLibraryIdentity: String =
    val pathOpt =
      sys.props.get(NativePathProperty).orElse(sys.env.get(NativePathEnv)).map(_.trim).filter(_.nonEmpty)
    pathOpt match
      case Some(path) =>
        val file = new File(path)
        val mtime = if file.exists() then file.lastModified() else 0L
        s"path=${file.getAbsolutePath}|mtime=$mtime"
      case None =>
        val lib =
          sys.props
            .get(NativeLibProperty)
            .orElse(sys.env.get(NativeLibEnv))
            .map(_.trim)
            .filter(_.nonEmpty)
            .getOrElse(DefaultNativeLibrary)
        s"lib=$lib"

  private def resolvedCacheFile: File =
    val configured =
      sys.props
        .get(AutoTuneCachePathProperty)
        .orElse(sys.env.get(AutoTuneCachePathEnv))
        .map(_.trim)
        .filter(_.nonEmpty)
        .getOrElse(DefaultCachePath)
    new File(configured)

  private def autoTuneEnabled: Boolean =
    val raw =
      sys.props
        .get(AutoTuneProperty)
        .orElse(sys.env.get(AutoTuneEnv))
        .map(_.trim.toLowerCase)
    raw match
      case Some("0" | "false" | "no" | "off") => false
      case _ => true

  private def hasExplicitNativeRuntimeConfig: Boolean =
    isConfigured(NativeEngineProperty, NativeEngineEnv) ||
    isConfigured(NativeCudaBlockSizeProperty, NativeCudaBlockSizeEnv) ||
    isConfigured(NativeCudaMaxChunkProperty, NativeCudaMaxChunkEnv)

  private def isConfigured(prop: String, env: String): Boolean =
    sys.props.get(prop).exists(_.trim.nonEmpty) || sys.env.get(env).exists(_.trim.nonEmpty)

  private def parsePositiveIntOpt(raw: String): Option[Int] =
    Option(raw)
      .map(_.trim)
      .filter(_.nonEmpty)
      .flatMap(text => scala.util.Try(text.toInt).toOption)
      .filter(_ > 0)

  private def clamp(value: Int, minValue: Int, maxValue: Int): Int =
    math.max(minValue, math.min(maxValue, value))

  private def showOpt(value: Option[Int]): String =
    value.map(_.toString).getOrElse("n/a")

  // ── Hybrid multi-device calibration ─────────────────────────────

  private val HybridCacheVersion = "3"
  private val DefaultHybridCachePath = "data/headsup-hybrid-autotune.properties"
  private val HybridCachePathProperty = "sicfun.gpu.autotune.hybridCachePath"
  private val HybridCachePathEnv = "sicfun_GPU_AUTOTUNE_HYBRID_CACHE_PATH"

  /** Per-device calibration result for hybrid mode. */
  final case class HybridDeviceConfig(
      deviceId: String,
      throughput: Double,
      weight: Double
  )

  /** The hybrid calibration decision, including per-device throughput and split weights. */
  final case class HybridDecision(
      devices: Vector[HybridDeviceConfig],
      source: String,
      detail: String
  )

  /** Runs hybrid auto-calibration: benchmarks each available device independently,
    * measures throughput, and computes proportional split weights.
    *
    * Called when `sicfun.gpu.provider=hybrid` before table generation begins.
    */
  def configureForHybrid(
      tableKind: String,
      mode: HeadsUpEquityTable.Mode,
      maxMatchups: Long
  ): Unit =
    mode match
      case HeadsUpEquityTable.Mode.Exact =>
        println("hybrid-autotune: skipped (exact mode)")
      case mc: HeadsUpEquityTable.Mode.MonteCarlo =>
        runHybridAutoTune(tableKind.trim.toLowerCase, mc, maxMatchups)

  private def runHybridAutoTune(
      tableKind: String,
      mode: HeadsUpEquityTable.Mode.MonteCarlo,
      maxMatchups: Long
  ): Unit =
    if !autoTuneEnabled then
      println("hybrid-autotune: disabled via sicfun.gpu.autotune/sicfun_GPU_AUTOTUNE")
      return

    val devices = HeadsUpHybridDispatcher.devices
    if devices.isEmpty then
      println("hybrid-autotune: no devices discovered, skipping")
      return

    val batch = loadTuneBatch(tableKind, maxMatchups)
    if batch.size <= 0 then
      println("hybrid-autotune: empty tune batch, skipping")
      return

    val tuneTrials = clamp(mode.trials, TuneMinTrials, TuneMaxTrials)
    val cacheFile = resolvedHybridCacheFile
    val topology = deviceTopologyFingerprint(devices)
    val signature = hybridWorkloadSignature(tableKind, tuneTrials, batch.size, topology)

    loadHybridDecisionFromCache(cacheFile, signature) match
      case Some(cached) =>
        applyHybridDecision(cached)
        printHybridDecision(cached, fromCache = true, tuneTrials, batch.size)
      case None =>
        val tuneMode: HeadsUpEquityTable.Mode.MonteCarlo =
          HeadsUpEquityTable.Mode.MonteCarlo(tuneTrials)
        val results = devices.flatMap { device =>
          benchmarkDevice(device, batch, tuneMode)
        }
        if results.isEmpty then
          println("hybrid-autotune: all device benchmarks failed, leaving defaults")
        else
          val totalThroughput = results.map(_._2).sum
          val configs = results.map { case (device, throughput) =>
            val weight = if totalThroughput > 0 then throughput / totalThroughput else 1.0 / results.size
            HybridDeviceConfig(device.id, throughput, weight)
          }
          val detail = configs.map(c => f"${c.deviceId}:${c.throughput}%.1f/s(${c.weight * 100}%.1f%%)").mkString(", ")
          val decision = HybridDecision(configs, source = "tuned", detail = detail)
          applyHybridDecision(decision)
          saveHybridDecisionToCache(cacheFile, signature, decision)
          printHybridDecision(decision, fromCache = false, tuneTrials, batch.size)

  private def benchmarkDevice(
      device: HeadsUpHybridDispatcher.ComputeDevice,
      batch: BatchData,
      mode: HeadsUpEquityTable.Mode.MonteCarlo
  ): Option[(HeadsUpHybridDispatcher.ComputeDevice, Double)] =
    val n = batch.size
    val lowIds = new Array[Int](n)
    val highIds = new Array[Int](n)
    val seeds = new Array[Long](n)
    var idx = 0
    while idx < n do
      val packed = batch.packedKeys(idx)
      lowIds(idx) = HeadsUpEquityTable.unpackLowId(packed)
      highIds(idx) = HeadsUpEquityTable.unpackHighId(packed)
      seeds(idx) = HeadsUpEquityTable.monteCarloSeed(TuneSeedBase, batch.keyMaterial(idx))
      idx += 1

    val wins = new Array[Double](n)
    val ties = new Array[Double](n)
    val losses = new Array[Double](n)
    val stderrs = new Array[Double](n)

    try
      val started = System.nanoTime()
      val status = device.computeSubBatch(
        lowIds, highIds, 1, mode.trials, seeds, wins, ties, losses, stderrs
      )
      val elapsed = (System.nanoTime() - started).toDouble / 1_000_000_000.0
      if status != 0 then
        println(f"hybrid-autotune: device ${device.id} failed (status=$status)")
        None
      else
        val throughput = n.toDouble / elapsed
        println(f"hybrid-autotune: device ${device.id} — ${elapsed}%.3fs, ${throughput}%.0f matchups/s")
        Some((device, throughput))
    catch
      case ex: Throwable =>
        println(s"hybrid-autotune: device ${device.id} exception: ${ex.getMessage}")
        None

  private def applyHybridDecision(decision: HybridDecision): Unit =
    val weights = decision.devices.map(c => c.deviceId -> c.weight).toMap
    HeadsUpHybridDispatcher.setCalibratedWeights(weights)

  private def printHybridDecision(
      decision: HybridDecision,
      fromCache: Boolean,
      tuneTrials: Int,
      tuneEntries: Int
  ): Unit =
    val source = if fromCache then "cache" else "fresh"
    println(
      s"hybrid-autotune: selected split (source=$source, trials=$tuneTrials, entries=$tuneEntries): ${decision.detail}"
    )

  private def resolvedHybridCacheFile: File =
    val configured =
      sys.props
        .get(HybridCachePathProperty)
        .orElse(sys.env.get(HybridCachePathEnv))
        .map(_.trim)
        .filter(_.nonEmpty)
        .getOrElse(DefaultHybridCachePath)
    new File(configured)

  private def deviceTopologyFingerprint(devices: Vector[HeadsUpHybridDispatcher.ComputeDevice]): String =
    devices.map(d => s"${d.id}:${d.name}").sorted.mkString(";")

  private def hybridWorkloadSignature(
      tableKind: String,
      trials: Int,
      tuneEntries: Int,
      topology: String
  ): String =
    val os = System.getProperty("os.name", "unknown").trim.toLowerCase
    val arch = System.getProperty("os.arch", "unknown").trim.toLowerCase
    s"v=$HybridCacheVersion|scope=hybrid|table=$tableKind|trials=$trials|entries=$tuneEntries|os=$os|arch=$arch|topology=$topology"

  private def loadHybridDecisionFromCache(file: File, signature: String): Option[HybridDecision] =
    if !file.exists() then return None
    val props = new Properties()
    val in = new FileInputStream(file)
    try props.load(in)
    finally in.close()

    val version = Option(props.getProperty("version")).getOrElse("")
    val cachedSignature = Option(props.getProperty("signature")).getOrElse("")
    if version != HybridCacheVersion || cachedSignature != signature then
      None
    else
      val count = parsePositiveIntOpt(props.getProperty("device.count")).getOrElse(0)
      if count <= 0 then None
      else
        val configs = (0 until count).flatMap { i =>
          val prefix = s"device.$i."
          val deviceId = Option(props.getProperty(s"${prefix}id")).map(_.trim).filter(_.nonEmpty)
          val throughput = Option(props.getProperty(s"${prefix}throughput")).flatMap(s =>
            scala.util.Try(s.trim.toDouble).toOption
          )
          val weight = Option(props.getProperty(s"${prefix}weight")).flatMap(s =>
            scala.util.Try(s.trim.toDouble).toOption
          )
          for
            id <- deviceId
            t <- throughput
            w <- weight
          yield HybridDeviceConfig(id, t, w)
        }.toVector

        if configs.size == count then
          val detail = Option(props.getProperty("detail")).getOrElse("cached")
          Some(HybridDecision(configs, source = "cache", detail = detail))
        else None

  private def saveHybridDecisionToCache(file: File, signature: String, decision: HybridDecision): Unit =
    val parent = file.getParentFile
    if parent != null then parent.mkdirs()
    val props = new Properties()
    props.setProperty("version", HybridCacheVersion)
    props.setProperty("signature", signature)
    props.setProperty("device.count", decision.devices.size.toString)
    decision.devices.zipWithIndex.foreach { case (config, i) =>
      val prefix = s"device.$i."
      props.setProperty(s"${prefix}id", config.deviceId)
      props.setProperty(s"${prefix}throughput", config.throughput.toString)
      props.setProperty(s"${prefix}weight", config.weight.toString)
    }
    props.setProperty("detail", decision.detail)
    props.setProperty("updatedAtMillis", System.currentTimeMillis().toString)
    val out = new FileOutputStream(file)
    try props.store(out, "heads-up hybrid autotune cache")
    finally out.close()
