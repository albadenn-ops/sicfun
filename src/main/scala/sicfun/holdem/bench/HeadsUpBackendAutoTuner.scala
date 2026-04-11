package sicfun.holdem.bench
import sicfun.holdem.*
import sicfun.holdem.equity.*
import sicfun.holdem.gpu.*
import sicfun.holdem.bench.BenchSupport.BatchData

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
  *   - Supports Monte Carlo and exact mode tuning (with mode-specific candidate sets)
  *
  * '''Cache:'''
  *   - Default location: `data/headsup-backend-autotune.properties`
  *   - Configurable via `sicfun.gpu.autotune.cachePath` / `sicfun_GPU_AUTOTUNE_CACHE_PATH`
  *   - Invalidated when the workload signature changes (e.g. different native library version)
  */
object HeadsUpBackendAutoTuner:
  private final case class CudaConfig(
      blockSize: Int,
      maxChunkMatchups: Int
  )

  private final case class CandidateConfig(
      engine: String,
      blockSize: Option[Int],
      maxChunkMatchups: Option[Int]
  )

  private final case class CandidateResult(
      engine: String,
      blockSize: Option[Int],
      maxChunkMatchups: Option[Int],
      seconds: Double,
      detail: String
  )

  private final case class TuneWorkload(
      mode: HeadsUpEquityTable.Mode,
      signature: String,
      tuneEntries: Int,
      candidates: Vector[CandidateConfig]
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
      source: "tuned" | "cache",
      detail: String
  )

  private val AutoTuneProperty = "sicfun.gpu.autotune"
  private val AutoTuneEnv = "sicfun_GPU_AUTOTUNE"
  private val ProviderProperty = "sicfun.gpu.provider"
  private val ProviderEnv = "sicfun_GPU_PROVIDER"
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
  private val CacheVersion = "6"
  private val LegacyCacheVersion = "5"
  private val DefaultCachePath = "data/headsup-backend-autotune.properties"
  private val TuneMaxEntries = 12000
  private val TuneMinEntries = 2000
  private val ExactTuneMaxEntries = 768
  private val ExactTuneMinEntries = 256
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
  private val ExactCudaCandidates = Vector(
    CudaConfig(blockSize = 96, maxChunkMatchups = 160),
    CudaConfig(blockSize = 96, maxChunkMatchups = 192),
    CudaConfig(blockSize = 128, maxChunkMatchups = 160),
    CudaConfig(blockSize = 128, maxChunkMatchups = 192),
    CudaConfig(blockSize = 128, maxChunkMatchups = 224),
    CudaConfig(blockSize = 160, maxChunkMatchups = 160),
    CudaConfig(blockSize = 192, maxChunkMatchups = 160)
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
      backend: HeadsUpEquityTable.ComputeBackend,
      forceRetune: Boolean = false
  ): Unit =
    if backend != HeadsUpEquityTable.ComputeBackend.Gpu then
      ()
    else
      configuredProvider match
        case "hybrid" =>
          configureForHybrid(
            tableKind = tableKind.trim.toLowerCase,
            mode = mode,
            maxMatchups = maxMatchups,
            forceRetune = forceRetune
          )
        case _ =>
          mode match
            case HeadsUpEquityTable.Mode.Exact =>
              runAutoTuneExact(
                tableKind = tableKind.trim.toLowerCase,
                maxMatchups = maxMatchups,
                forceRetune = forceRetune
              )
            case mc: HeadsUpEquityTable.Mode.MonteCarlo =>
              runAutoTune(
                tableKind = tableKind.trim.toLowerCase,
                mode = mc,
                maxMatchups = maxMatchups,
                includeCpuCandidate = true,
                tuneScope = "any",
                forceRetune = forceRetune
              )

  /** Runs auto-tuning for the backend comparison harness (CUDA candidates only, no CPU fallback). */
  def configureForComparison(
      tableKind: String,
      mode: HeadsUpEquityTable.Mode,
      maxMatchups: Long,
      forceRetune: Boolean = false
  ): Unit =
    mode match
      case HeadsUpEquityTable.Mode.Exact =>
        runAutoTuneExact(
          tableKind = tableKind.trim.toLowerCase,
          maxMatchups = maxMatchups,
          forceRetune = forceRetune
        )
      case mc: HeadsUpEquityTable.Mode.MonteCarlo =>
        runAutoTune(
          tableKind = tableKind.trim.toLowerCase,
          mode = mc,
          maxMatchups = maxMatchups,
          includeCpuCandidate = false,
          tuneScope = "cuda",
          forceRetune = forceRetune
        )

  private def runAutoTune(
      tableKind: String,
      mode: HeadsUpEquityTable.Mode.MonteCarlo,
      maxMatchups: Long,
      includeCpuCandidate: Boolean,
      tuneScope: String,
      forceRetune: Boolean
  ): Unit =
    if !autoTuneEnabled then
      GpuRuntimeSupport.log("gpu-autotune: disabled via sicfun.gpu.autotune/sicfun_GPU_AUTOTUNE")
    else if hasExplicitNativeRuntimeConfig then
      GpuRuntimeSupport.log("gpu-autotune: skipped (native engine/block/chunk explicitly configured)")
    else
      val availability = HeadsUpGpuRuntime.availability
      if !availability.available then
        GpuRuntimeSupport.log(s"gpu-autotune: skipped (provider unavailable: ${availability.detail})")
      else
        val batch = loadTuneBatch(
          tableKind = tableKind,
          maxMatchups = maxMatchups,
          minEntries = TuneMinEntries,
          maxEntries = TuneMaxEntries
        )
        if batch.size <= 0 then
          GpuRuntimeSupport.log("gpu-autotune: skipped (empty tune batch)")
        else
          val tuneTrials = clamp(mode.trials, TuneMinTrials, TuneMaxTrials)
          val workload = TuneWorkload(
            mode = HeadsUpEquityTable.Mode.MonteCarlo(tuneTrials),
            signature = workloadSignature(tableKind, tuneTrials, batch.size, tuneScope),
            tuneEntries = batch.size,
            candidates = monteCarloCandidates(includeCpuCandidate, batch.size)
          )

          runDecisionSession(batch, workload, forceRetune) match
            case Some((decision, fromCache)) =>
              printDecision(decision, fromCache = fromCache, tuneTrials = tuneTrials, tuneEntries = batch.size)
            case None =>
              GpuRuntimeSupport.log("gpu-autotune: no successful candidates, leaving runtime defaults unchanged")

  private def runAutoTuneExact(
      tableKind: String,
      maxMatchups: Long,
      forceRetune: Boolean
  ): Unit =
    if !autoTuneEnabled then
      GpuRuntimeSupport.log("gpu-autotune: disabled via sicfun.gpu.autotune/sicfun_GPU_AUTOTUNE")
    else if hasExplicitNativeRuntimeConfig then
      GpuRuntimeSupport.log("gpu-autotune: skipped (native engine/block/chunk explicitly configured)")
    else
      val availability = HeadsUpGpuRuntime.availability
      if !availability.available then
        GpuRuntimeSupport.log(s"gpu-autotune: skipped (provider unavailable: ${availability.detail})")
      else
        val batch = loadTuneBatch(
          tableKind = tableKind,
          maxMatchups = maxMatchups,
          minEntries = ExactTuneMinEntries,
          maxEntries = ExactTuneMaxEntries
        )
        if batch.size <= 0 then
          GpuRuntimeSupport.log("gpu-autotune: skipped (empty tune batch)")
        else
          val workload = TuneWorkload(
            mode = HeadsUpEquityTable.Mode.Exact,
            signature = workloadSignatureExact(tableKind = tableKind, tuneEntries = batch.size),
            tuneEntries = batch.size,
            candidates = exactCandidates(batch.size)
          )

          runDecisionSession(batch, workload, forceRetune) match
            case Some((decision, true)) =>
              GpuRuntimeSupport.log(
                s"gpu-autotune: exact cached engine=${decision.engine}, block=${showOpt(decision.blockSize)}, " +
                  s"chunk=${showOpt(decision.maxChunkMatchups)} (entries=${batch.size}, ${decision.detail})"
              )
            case Some((decision, false)) =>
              GpuRuntimeSupport.log(
                s"gpu-autotune: exact selected engine=${decision.engine}, block=${showOpt(decision.blockSize)}, " +
                  s"chunk=${showOpt(decision.maxChunkMatchups)} (entries=${batch.size}, ${decision.detail})"
              )
            case None =>
              GpuRuntimeSupport.log("gpu-autotune: exact mode no successful candidates, leaving runtime defaults unchanged")

  private def monteCarloCandidates(
      includeCpuCandidate: Boolean,
      tuneEntries: Int
  ): Vector[CandidateConfig] =
    val cpu =
      if includeCpuCandidate then Vector(CandidateConfig(engine = "cpu", blockSize = None, maxChunkMatchups = None))
      else Vector.empty
    val cuda =
      CudaCandidates
        .filter(_.maxChunkMatchups <= tuneEntries)
        .map(cfg =>
          CandidateConfig(
            engine = "cuda",
            blockSize = Some(cfg.blockSize),
            maxChunkMatchups = Some(cfg.maxChunkMatchups)
          )
        )
    cpu ++ cuda

  private def exactCandidates(tuneEntries: Int): Vector[CandidateConfig] =
    ExactCudaCandidates
      .filter(_.maxChunkMatchups <= tuneEntries)
      .map(cfg =>
        CandidateConfig(
          engine = "cuda",
          blockSize = Some(cfg.blockSize),
          maxChunkMatchups = Some(cfg.maxChunkMatchups)
        )
      )

  private def runDecisionSession(
      batch: BatchData,
      workload: TuneWorkload,
      forceRetune: Boolean
  ): Option[(Decision, Boolean)] =
    val cacheFile = resolvedCacheFile
    if forceRetune then
      benchmarkCandidates(batch, workload).minByOption(_.seconds).map { winner =>
        val decision = decisionForResult(winner)
        applyDecision(decision)
        saveDecisionToCache(cacheFile, workload.signature, decision)
        (decision, false)
      }
    else
      loadDecisionFromCache(cacheFile, workload.signature) match
        case Some(cached) =>
          applyDecision(cached)
          Some((cached, true))
        case None =>
          benchmarkCandidates(batch, workload).minByOption(_.seconds).map { winner =>
            val decision = decisionForResult(winner)
            applyDecision(decision)
            saveDecisionToCache(cacheFile, workload.signature, decision)
            (decision, false)
          }

  private def benchmarkCandidates(
      batch: BatchData,
      workload: TuneWorkload
  ): Vector[CandidateResult] =
    workload.candidates.flatMap { candidate =>
      benchmarkCandidate(
        batch = batch,
        mode = workload.mode,
        engine = candidate.engine,
        blockSize = candidate.blockSize,
        maxChunkMatchups = candidate.maxChunkMatchups
      )
    }

  private def decisionForResult(result: CandidateResult): Decision =
    Decision(
      engine = result.engine,
      blockSize = result.blockSize,
      maxChunkMatchups = result.maxChunkMatchups,
      source = "tuned",
      detail = result.detail
    )

  private def benchmarkCandidate(
      batch: BatchData,
      mode: HeadsUpEquityTable.Mode,
      engine: String,
      blockSize: Option[Int],
      maxChunkMatchups: Option[Int]
  ): Option[CandidateResult] =
    applyEngineSettings(engine, blockSize, maxChunkMatchups)
    HandEvaluator.clearCaches()
    val started = System.nanoTime()
    HeadsUpGpuRuntime.computeBatch(batch.packedKeys, batch.keyMaterial, mode, TuneSeedBase) match
      case Left(reason) =>
        GpuRuntimeSupport.log(s"gpu-autotune: candidate engine=$engine block=${showOpt(blockSize)} chunk=${showOpt(maxChunkMatchups)} failed ($reason)")
        None
      case Right(_) =>
        val elapsed = (System.nanoTime() - started).toDouble / 1_000_000_000.0
        GpuRuntimeSupport.log(
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

  private def loadTuneBatch(
      tableKind: String,
      maxMatchups: Long,
      minEntries: Int,
      maxEntries: Int
  ): BatchData =
    val normalizedMax = math.max(1L, maxMatchups)
    tableKind match
      case "full" =>
        val total = HeadsUpEquityTable.totalMatchups
        val limit = tuneEntryLimit(total = total, requested = normalizedMax, minEntries = minEntries, maxEntries = maxEntries)
        val batch = HeadsUpEquityTable.selectFullBatch(limit)
        BatchData(batch.packedKeys, batch.keyMaterial)
      case _ =>
        val total = HeadsUpEquityCanonicalTable.totalCanonicalKeys.toLong
        val limit = tuneEntryLimit(total = total, requested = normalizedMax, minEntries = minEntries, maxEntries = maxEntries)
        val batch = HeadsUpEquityCanonicalTable.selectCanonicalBatch(limit)
        BatchData(batch.packedKeys, batch.keyMaterial)

  private def tuneEntryLimit(total: Long, requested: Long, minEntries: Int, maxEntries: Int): Long =
    val cap = math.min(total, requested)
    val bounded = math.min(cap, maxEntries.toLong)
    math.max(1L, math.min(cap, math.max(minEntries.toLong, bounded)))

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
    GpuRuntimeSupport.log(
      s"gpu-autotune: selected engine=${decision.engine}, block=${showOpt(decision.blockSize)}, " +
        s"chunk=${showOpt(decision.maxChunkMatchups)} (source=$source, trials=$tuneTrials, entries=$tuneEntries, ${decision.detail})"
    )

  private def loadDecisionFromCache(file: File, signature: String): Option[Decision] =
    if !file.exists() then None
    else
      val props = new Properties()
      val in = new FileInputStream(file)
      try props.load(in)
      finally in.close()

      loadDecisionEntries(props).collectFirst {
        case (`signature`, decision) =>
          decision.copy(source = "cache")
      }

  private def saveDecisionToCache(file: File, signature: String, decision: Decision): Unit =
    val parent = file.getParentFile
    if parent != null then parent.mkdirs()
    val props = new Properties()
    if file.isFile then
      val in = new FileInputStream(file)
      try props.load(in)
      finally in.close()
    val mergedEntries =
      (loadDecisionEntries(props).filterNot(_._1 == signature) :+ (signature -> decision.copy(source = "tuned"))).zipWithIndex
    props.clear()
    props.setProperty("version", CacheVersion)
    props.setProperty("entry.count", mergedEntries.size.toString)
    mergedEntries.foreach { case ((entrySignature, entryDecision), idx) =>
      val prefix = s"entry.$idx."
      props.setProperty(s"${prefix}signature", entrySignature)
      props.setProperty(s"${prefix}engine", entryDecision.engine)
      entryDecision.blockSize.foreach(v => props.setProperty(s"${prefix}blockSize", v.toString))
      entryDecision.maxChunkMatchups.foreach(v => props.setProperty(s"${prefix}maxChunkMatchups", v.toString))
      props.setProperty(s"${prefix}detail", entryDecision.detail)
    }
    props.setProperty("updatedAtMillis", System.currentTimeMillis().toString)
    val out = new FileOutputStream(file)
    try props.store(out, "heads-up backend autotune cache")
    finally out.close()

  private def loadDecisionEntries(props: Properties): Vector[(String, Decision)] =
    Option(props.getProperty("version")).map(_.trim).getOrElse("") match
      case CacheVersion =>
        val count = GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty("entry.count")).getOrElse(0)
        (0 until count).flatMap { idx =>
          val prefix = s"entry.$idx."
          val signature = Option(props.getProperty(s"${prefix}signature")).map(_.trim).filter(_.nonEmpty)
          val engine = Option(props.getProperty(s"${prefix}engine")).map(_.trim.toLowerCase).filter(_.nonEmpty)
          for
            entrySignature <- signature
            entryEngine <- engine
          yield entrySignature ->
            Decision(
              engine = entryEngine,
              blockSize = GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty(s"${prefix}blockSize")),
              maxChunkMatchups = GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty(s"${prefix}maxChunkMatchups")),
              source = "cache",
              detail = Option(props.getProperty(s"${prefix}detail")).getOrElse("cached")
            )
        }.toVector
      case LegacyCacheVersion =>
        val signature = Option(props.getProperty("signature")).map(_.trim).filter(_.nonEmpty)
        val engine = Option(props.getProperty("engine")).map(_.trim.toLowerCase).filter(_.nonEmpty)
        (for
          entrySignature <- signature
          entryEngine <- engine
        yield entrySignature ->
          Decision(
            engine = entryEngine,
            blockSize = GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty("blockSize")),
            maxChunkMatchups = GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty("maxChunkMatchups")),
            source = "cache",
            detail = Option(props.getProperty("detail")).getOrElse("cached")
          )).toVector
      case _ =>
        Vector.empty

  private def workloadSignature(tableKind: String, trials: Int, tuneEntries: Int, tuneScope: String): String =
    val os = System.getProperty("os.name", "unknown").trim.toLowerCase
    val arch = System.getProperty("os.arch", "unknown").trim.toLowerCase
    val javaVersion = System.getProperty("java.version", "unknown").trim.toLowerCase
    val libId = nativeLibraryIdentity
    s"v=$CacheVersion|scope=$tuneScope|table=$tableKind|trials=$trials|entries=$tuneEntries|os=$os|arch=$arch|java=$javaVersion|lib=$libId"

  private def workloadSignatureExact(tableKind: String, tuneEntries: Int): String =
    val os = System.getProperty("os.name", "unknown").trim.toLowerCase
    val arch = System.getProperty("os.arch", "unknown").trim.toLowerCase
    val javaVersion = System.getProperty("java.version", "unknown").trim.toLowerCase
    val libId = nativeLibraryIdentity
    s"v=$CacheVersion|scope=exact-cuda|table=$tableKind|entries=$tuneEntries|os=$os|arch=$arch|java=$javaVersion|lib=$libId"

  private def nativeLibraryIdentity: String =
    val pathOpt = GpuRuntimeSupport.resolveNonEmpty(NativePathProperty, NativePathEnv)
    pathOpt match
      case Some(path) =>
        val file = new File(path)
        val mtime = if file.exists() then file.lastModified() else 0L
        s"path=${file.getAbsolutePath}|mtime=$mtime"
      case None =>
        val lib = GpuRuntimeSupport.resolveNonEmpty(NativeLibProperty, NativeLibEnv).getOrElse(DefaultNativeLibrary)
        s"lib=$lib"

  private def resolvedCacheFile: File =
    GpuRuntimeSupport.resolveFile(AutoTuneCachePathProperty, AutoTuneCachePathEnv, DefaultCachePath)

  private def autoTuneEnabled: Boolean =
    val raw = GpuRuntimeSupport.resolveNonEmptyLower(AutoTuneProperty, AutoTuneEnv)
    raw match
      case Some("0" | "false" | "no" | "off") => false
      case _ => true

  private def configuredProvider: String =
    GpuRuntimeSupport.resolveNonEmptyLower(ProviderProperty, ProviderEnv).getOrElse("native")

  private def hasExplicitNativeRuntimeConfig: Boolean =
    GpuRuntimeSupport.isConfigured(NativeEngineProperty, NativeEngineEnv) ||
    GpuRuntimeSupport.isConfigured(NativeCudaBlockSizeProperty, NativeCudaBlockSizeEnv) ||
    GpuRuntimeSupport.isConfigured(NativeCudaMaxChunkProperty, NativeCudaMaxChunkEnv)

  private def clamp(value: Int, minValue: Int, maxValue: Int): Int =
    math.max(minValue, math.min(maxValue, value))

  private def showOpt(value: Option[Int]): String =
    value.map(_.toString).getOrElse("n/a")

  // ── Hybrid multi-device calibration ─────────────────────────────

  private val HybridCacheVersion = "4"
  private val LegacyHybridCacheVersion = "3"
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
      maxMatchups: Long,
      forceRetune: Boolean = false
  ): Unit =
    mode match
      case HeadsUpEquityTable.Mode.Exact =>
        GpuRuntimeSupport.log("hybrid-autotune: skipped (exact mode)")
      case mc: HeadsUpEquityTable.Mode.MonteCarlo =>
        runHybridAutoTune(tableKind.trim.toLowerCase, mc, maxMatchups, forceRetune)

  private def runHybridAutoTune(
      tableKind: String,
      mode: HeadsUpEquityTable.Mode.MonteCarlo,
      maxMatchups: Long,
      forceRetune: Boolean
  ): Unit =
    if !autoTuneEnabled then
      GpuRuntimeSupport.log("hybrid-autotune: disabled via sicfun.gpu.autotune/sicfun_GPU_AUTOTUNE")
    else
      val devices = HeadsUpHybridDispatcher.devices
      if devices.isEmpty then
        GpuRuntimeSupport.log("hybrid-autotune: no devices discovered, skipping")
      else
        val batch = loadTuneBatch(
          tableKind = tableKind,
          maxMatchups = maxMatchups,
          minEntries = TuneMinEntries,
          maxEntries = TuneMaxEntries
        )
        if batch.size <= 0 then
          GpuRuntimeSupport.log("hybrid-autotune: empty tune batch, skipping")
        else
          val tuneTrials = clamp(mode.trials, TuneMinTrials, TuneMaxTrials)
          val cacheFile = resolvedHybridCacheFile
          val topology = deviceTopologyFingerprint(devices)
          val signature = hybridWorkloadSignature(tableKind, tuneTrials, batch.size, topology)

          if !forceRetune then
            loadHybridDecisionFromCache(cacheFile, signature) match
              case Some(cached) =>
                applyHybridDecision(cached)
                printHybridDecision(cached, fromCache = true, tuneTrials, batch.size)
              case None =>
                benchmarkAndPersistHybrid(cacheFile, signature, devices, batch, tuneTrials)
          else
            benchmarkAndPersistHybrid(cacheFile, signature, devices, batch, tuneTrials)

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
        GpuRuntimeSupport.log(f"hybrid-autotune: device ${device.id} failed (status=$status)")
        None
      else
        val throughput = n.toDouble / elapsed
        GpuRuntimeSupport.log(f"hybrid-autotune: device ${device.id} — ${elapsed}%.3fs, ${throughput}%.0f matchups/s")
        Some((device, throughput))
    catch
      case ex: Throwable =>
        GpuRuntimeSupport.log(s"hybrid-autotune: device ${device.id} exception: ${ex.getMessage}")
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
    GpuRuntimeSupport.log(
      s"hybrid-autotune: selected split (source=$source, trials=$tuneTrials, entries=$tuneEntries): ${decision.detail}"
    )

  private def resolvedHybridCacheFile: File =
    GpuRuntimeSupport.resolveFile(HybridCachePathProperty, HybridCachePathEnv, DefaultHybridCachePath)

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
    if !file.exists() then None
    else
      val props = new Properties()
      val in = new FileInputStream(file)
      try props.load(in)
      finally in.close()

      loadHybridDecisionEntries(props).collectFirst {
        case (`signature`, decision) =>
          decision.copy(source = "cache")
      }

  private def saveHybridDecisionToCache(file: File, signature: String, decision: HybridDecision): Unit =
    val parent = file.getParentFile
    if parent != null then parent.mkdirs()
    val props = new Properties()
    if file.isFile then
      val in = new FileInputStream(file)
      try props.load(in)
      finally in.close()
    val mergedEntries =
      (loadHybridDecisionEntries(props).filterNot(_._1 == signature) :+ (signature -> decision.copy(source = "tuned"))).zipWithIndex
    props.clear()
    props.setProperty("version", HybridCacheVersion)
    props.setProperty("entry.count", mergedEntries.size.toString)
    mergedEntries.foreach { case ((entrySignature, entryDecision), entryIdx) =>
      val prefix = s"entry.$entryIdx."
      props.setProperty(s"${prefix}signature", entrySignature)
      props.setProperty(s"${prefix}device.count", entryDecision.devices.size.toString)
      entryDecision.devices.zipWithIndex.foreach { case (config, deviceIdx) =>
        val devicePrefix = s"${prefix}device.$deviceIdx."
        props.setProperty(s"${devicePrefix}id", config.deviceId)
        props.setProperty(s"${devicePrefix}throughput", config.throughput.toString)
        props.setProperty(s"${devicePrefix}weight", config.weight.toString)
      }
      props.setProperty(s"${prefix}detail", entryDecision.detail)
    }
    props.setProperty("updatedAtMillis", System.currentTimeMillis().toString)
    val out = new FileOutputStream(file)
    try props.store(out, "heads-up hybrid autotune cache")
    finally out.close()

  private def benchmarkAndPersistHybrid(
      cacheFile: File,
      signature: String,
      devices: Vector[HeadsUpHybridDispatcher.ComputeDevice],
      batch: BatchData,
      tuneTrials: Int
  ): Unit =
    val tuneMode: HeadsUpEquityTable.Mode.MonteCarlo =
      HeadsUpEquityTable.Mode.MonteCarlo(tuneTrials)
    val results = devices.flatMap { device =>
      benchmarkDevice(device, batch, tuneMode)
    }
    if results.isEmpty then
      GpuRuntimeSupport.log("hybrid-autotune: all device benchmarks failed, leaving defaults")
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

  private def loadHybridDecisionEntries(props: Properties): Vector[(String, HybridDecision)] =
    Option(props.getProperty("version")).map(_.trim).getOrElse("") match
      case HybridCacheVersion =>
        val count = GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty("entry.count")).getOrElse(0)
        (0 until count).flatMap { entryIdx =>
          val prefix = s"entry.$entryIdx."
          val signature = Option(props.getProperty(s"${prefix}signature")).map(_.trim).filter(_.nonEmpty)
          val deviceCount = GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty(s"${prefix}device.count")).getOrElse(0)
          if signature.isEmpty || deviceCount <= 0 then None
          else
            val configs = (0 until deviceCount).flatMap { deviceIdx =>
              val devicePrefix = s"${prefix}device.$deviceIdx."
              val deviceId = Option(props.getProperty(s"${devicePrefix}id")).map(_.trim).filter(_.nonEmpty)
              val throughput = Option(props.getProperty(s"${devicePrefix}throughput")).flatMap(s =>
                scala.util.Try(s.trim.toDouble).toOption
              )
              val weight = Option(props.getProperty(s"${devicePrefix}weight")).flatMap(s =>
                scala.util.Try(s.trim.toDouble).toOption
              )
              for
                id <- deviceId
                t <- throughput
                w <- weight
              yield HybridDeviceConfig(id, t, w)
            }.toVector
            if configs.size == deviceCount then
              Some(signature.get -> HybridDecision(configs, source = "cache", detail = Option(props.getProperty(s"${prefix}detail")).getOrElse("cached")))
            else None
        }.toVector
      case LegacyHybridCacheVersion =>
        val signature = Option(props.getProperty("signature")).map(_.trim).filter(_.nonEmpty)
        val count = GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty("device.count")).getOrElse(0)
        if signature.isEmpty || count <= 0 then Vector.empty
        else
          val configs = (0 until count).flatMap { idx =>
            val prefix = s"device.$idx."
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
            Vector(signature.get -> HybridDecision(configs, source = "cache", detail = Option(props.getProperty("detail")).getOrElse("cached")))
          else Vector.empty
      case _ =>
        Vector.empty
