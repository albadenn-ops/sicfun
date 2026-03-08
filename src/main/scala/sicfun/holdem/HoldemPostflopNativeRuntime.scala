package sicfun.holdem

import sicfun.core.CardId

import java.io.{File, FileInputStream}
import java.util.Properties
import java.util.concurrent.atomic.AtomicReference

/** Runtime wrapper for postflop native Monte Carlo batch evaluation. */
private[holdem] object HoldemPostflopNativeRuntime:
  enum Backend:
    case Cpu
    case Gpu

  final case class Availability(available: Boolean, provider: String, detail: String)
  private final case class ResolvedBackend(backend: Backend, source: String, note: Option[String] = None)

  private val ProviderProperty = "sicfun.postflop.provider"
  private val ProviderEnv = "sicfun_POSTFLOP_PROVIDER"
  private val NativeEngineProperty = "sicfun.postflop.native.engine"
  private val NativeEngineEnv = "sicfun_POSTFLOP_NATIVE_ENGINE"
  private val AutoMinGpuWorkProperty = "sicfun.postflop.native.auto.minGpuWork"
  private val AutoMinGpuWorkEnv = "sicfun_POSTFLOP_NATIVE_AUTO_MIN_GPU_WORK"
  private val DefaultAutoMinGpuWork = 300000L
  private val PostflopAutoTuneProperty = "sicfun.postflop.autotune"
  private val PostflopAutoTuneEnv = "sicfun_POSTFLOP_AUTOTUNE"
  private val PostflopAutoTuneCachePathProperty = "sicfun.postflop.autotune.cachePath"
  private val PostflopAutoTuneCachePathEnv = "sicfun_POSTFLOP_AUTOTUNE_CACHE_PATH"
  private val PostflopCudaBlockSizeProperty = "sicfun.postflop.native.cuda.blockSize"
  private val PostflopCudaBlockSizeEnv = "sicfun_POSTFLOP_CUDA_BLOCK_SIZE"
  private val PostflopCudaMaxChunkMatchupsProperty = "sicfun.postflop.native.cuda.maxChunkMatchups"
  private val PostflopCudaMaxChunkMatchupsEnv = "sicfun_POSTFLOP_CUDA_MAX_CHUNK_MATCHUPS"
  private val PostflopAutoTuneCacheVersion = "1"
  private val DefaultPostflopAutoTuneCachePath = "data/postflop-autotune.properties"

  private val LegacyPathProperty = "sicfun.postflop.native.path"
  private val LegacyPathEnv = "sicfun_POSTFLOP_NATIVE_PATH"
  private val LegacyLibProperty = "sicfun.postflop.native.lib"
  private val LegacyLibEnv = "sicfun_POSTFLOP_NATIVE_LIB"

  private val CpuPathProperty = "sicfun.postflop.native.cpu.path"
  private val CpuPathEnv = "sicfun_POSTFLOP_NATIVE_CPU_PATH"
  private val CpuLibProperty = "sicfun.postflop.native.cpu.lib"
  private val CpuLibEnv = "sicfun_POSTFLOP_NATIVE_CPU_LIB"
  private val DefaultCpuLibrary = "sicfun_postflop_native"

  private val GpuPathProperty = "sicfun.postflop.native.gpu.path"
  private val GpuPathEnv = "sicfun_POSTFLOP_NATIVE_GPU_PATH"
  private val GpuLibProperty = "sicfun.postflop.native.gpu.lib"
  private val GpuLibEnv = "sicfun_POSTFLOP_NATIVE_GPU_LIB"
  private val DefaultGpuLibrary = "sicfun_postflop_cuda"

  private val cpuLoadResultRef = new AtomicReference[Either[String, String]](null)
  private val gpuLoadResultRef = new AtomicReference[Either[String, String]](null)
  private val appliedPostflopTuneFingerprintRef = new AtomicReference[String](null)

  private[holdem] def resetLoadCacheForTests(): Unit =
    cpuLoadResultRef.set(null)
    gpuLoadResultRef.set(null)
    appliedPostflopTuneFingerprintRef.set(null)

  def isAvailable: Boolean =
    availability.available

  def availability: Availability =
    configuredProvider match
      case "disabled" =>
        Availability(available = false, provider = "disabled", detail = "postflop provider disabled")
      case "native" | "auto" =>
        resolveBackend() match
          case Right(resolved) =>
            val note = resolved.note.fold("")(n => s"; $n")
            Availability(
              available = true,
              provider = "native",
              detail = s"postflop native backend=${backendLabel(resolved.backend)} loaded (${resolved.source})$note"
            )
          case Left(reason) =>
            Availability(available = false, provider = "native", detail = reason)
      case other =>
        Availability(
          available = false,
          provider = other,
          detail = s"unsupported postflop provider '$other' (expected auto|native|disabled)"
        )

  def computePostflopBatch(
      hero: HoleCards,
      board: Board,
      villains: Array[HoleCards],
      trials: Int,
      seedBase: Long
  ): Either[String, Array[EquityResultWithError]] =
    if board.size <= 0 || board.size > 5 then
      Left(s"postflop native runtime requires board size in [1,5], found ${board.size}")
    else if trials <= 0 then
      Left(s"postflop native runtime requires positive trials, found $trials")
    else if villains == null then
      Left("postflop native runtime villains must be non-null")
    else
      resolveBackend() match
        case Left(reason) =>
          Left(reason)
        case Right(resolved) =>
          try
            val heroFirst = CardId.toId(hero.first)
            val heroSecond = CardId.toId(hero.second)
            val boardCards = board.cards.map(CardId.toId).toArray
            val n = villains.length
            val villainFirst = new Array[Int](n)
            val villainSecond = new Array[Int](n)
            val seeds = new Array[Long](n)
            val wins = new Array[Double](n)
            val ties = new Array[Double](n)
            val losses = new Array[Double](n)
            val stderrs = new Array[Double](n)

            var i = 0
            while i < n do
              val villain = villains(i)
              val vf = CardId.toId(villain.first)
              val vs = CardId.toId(villain.second)
              villainFirst(i) = vf
              villainSecond(i) = vs
              val keyMaterial =
                ((vf.toLong & 0x3fL) << 16) ^
                  ((vs.toLong & 0x3fL) << 8) ^
                  (i.toLong & 0xffffffffL)
              seeds(i) = HeadsUpEquityTable.monteCarloSeed(seedBase, keyMaterial)
              i += 1

            if resolved.backend == Backend.Gpu then
              maybeApplyCachedPostflopAutoTune(deviceIndex = 0)

            val requestedWork = n.toLong * trials.toLong
            val executionBackend =
              if resolved.backend == Backend.Gpu &&
                configuredNativeEngine == "auto" &&
                requestedWork < configuredAutoMinGpuWork then
                cpuLoadResult() match
                  case Right(_) =>
                    GpuRuntimeSupport.log(
                      s"postflop native auto-engine: routing to CPU for small workload " +
                        s"(work=$requestedWork < minGpuWork=$configuredAutoMinGpuWork)"
                    )
                    Backend.Cpu
                  case Left(_) =>
                    Backend.Gpu
              else
                resolved.backend

            val status =
              executionBackend match
                case Backend.Cpu =>
                  HoldemPostflopNativeBindings.computePostflopBatchMonteCarlo(
                    heroFirst,
                    heroSecond,
                    boardCards,
                    boardCards.length,
                    villainFirst,
                    villainSecond,
                    trials,
                    seeds,
                    wins,
                    ties,
                    losses,
                    stderrs
                  )
                case Backend.Gpu =>
                  HoldemPostflopNativeGpuBindings.computePostflopBatchMonteCarlo(
                    heroFirst,
                    heroSecond,
                    boardCards,
                    boardCards.length,
                    villainFirst,
                    villainSecond,
                    trials,
                    seeds,
                    wins,
                    ties,
                    losses,
                    stderrs
                  )

            if status != 0 then
              executionBackend match
                case Backend.Gpu if configuredNativeEngine == "auto" =>
                  tryCpuFallbackAfterGpuFailure(
                    gpuStatus = status,
                    heroFirst = heroFirst,
                    heroSecond = heroSecond,
                    boardCards = boardCards,
                    villainFirst = villainFirst,
                    villainSecond = villainSecond,
                    trials = trials,
                    seeds = seeds,
                    wins = wins,
                    ties = ties,
                    losses = losses,
                    stderrs = stderrs
                  )
                case _ =>
                  Left(s"${describeStatus(status)}; nativeEngine=${safeEngineLabel(resolved.backend)}")
            else
              Right(toResultArray(wins, ties, losses, stderrs))
          catch
            case ex: UnsatisfiedLinkError =>
              Left(s"postflop native symbols not found (${backendLabel(resolved.backend)}): ${ex.getMessage}")
            case ex: Throwable =>
              Left(
                Option(ex.getMessage)
                  .map(_.trim)
                  .filter(_.nonEmpty)
                  .getOrElse(ex.getClass.getSimpleName)
              )

  private def resolveBackend(): Either[String, ResolvedBackend] =
    configuredNativeEngine match
      case "cpu" =>
        cpuLoadResult().map(source => ResolvedBackend(Backend.Cpu, source))
      case "cuda" =>
        gpuLoadResult().map(source => ResolvedBackend(Backend.Gpu, source))
      case _ =>
        gpuLoadResult() match
          case Right(source) =>
            Right(ResolvedBackend(Backend.Gpu, source))
          case Left(gpuReason) =>
            cpuLoadResult() match
              case Right(source) =>
                Right(
                  ResolvedBackend(
                    backend = Backend.Cpu,
                    source = source,
                    note = Some(s"CUDA library unavailable, fell back to CPU ($gpuReason)")
                  )
                )
              case Left(cpuReason) =>
                Left(
                  s"failed to load postflop native GPU backend: $gpuReason; " +
                    s"failed to load postflop native CPU backend: $cpuReason"
                )

  private def cpuLoadResult(): Either[String, String] =
    val cached = cpuLoadResultRef.get()
    if cached != null then cached
    else
      val loaded = loadLibraryWithOverrides(
        label = "postflop native CPU library",
        pathCandidates =
          Seq(
            CpuPathProperty -> CpuPathEnv,
            LegacyPathProperty -> LegacyPathEnv
          ),
        libCandidates =
          Seq(
            CpuLibProperty -> CpuLibEnv,
            LegacyLibProperty -> LegacyLibEnv
          ),
        defaultLib = DefaultCpuLibrary
      )
      cpuLoadResultRef.compareAndSet(null, loaded)
      cpuLoadResultRef.get()

  private def gpuLoadResult(): Either[String, String] =
    val cached = gpuLoadResultRef.get()
    if cached != null then cached
    else
      val loaded = loadLibraryWithOverrides(
        label = "postflop native GPU library",
        pathCandidates = Seq(GpuPathProperty -> GpuPathEnv),
        libCandidates = Seq(GpuLibProperty -> GpuLibEnv),
        defaultLib = DefaultGpuLibrary
      )
      gpuLoadResultRef.compareAndSet(null, loaded)
      gpuLoadResultRef.get()

  private def loadLibraryWithOverrides(
      label: String,
      pathCandidates: Seq[(String, String)],
      libCandidates: Seq[(String, String)],
      defaultLib: String
  ): Either[String, String] =
    val pathOpt = pathCandidates.iterator.flatMap { case (p, e) => GpuRuntimeSupport.resolveNonEmpty(p, e) }.toSeq.headOption
    val libName = libCandidates.iterator.flatMap { case (p, e) => GpuRuntimeSupport.resolveNonEmpty(p, e) }.toSeq.headOption.getOrElse(defaultLib)
    pathOpt match
      case Some(path) =>
        try
          System.load(path)
          Right(s"path=$path")
        catch
          case ex: Throwable =>
            Left(s"failed to load $label '$path': ${ex.getMessage}")
      case None =>
        try
          System.loadLibrary(libName)
          Right(s"library=$libName")
        catch
          case ex: Throwable =>
            GpuRuntimeSupport.tryLoadFirstExistingPath(
              GpuRuntimeSupport.localNativeFallbackCandidates(libName)
            ) match
              case Right(source) =>
                Right(source)
              case Left(fallbackReason) =>
                Left(s"failed to load $label '$libName': ${ex.getMessage}; $fallbackReason")

  private def configuredProvider: String =
    GpuRuntimeSupport.resolveNonEmptyLower(ProviderProperty, ProviderEnv).getOrElse("auto")

  private def configuredNativeEngine: String =
    GpuRuntimeSupport.resolveNonEmptyLower(NativeEngineProperty, NativeEngineEnv) match
      case Some("cpu") => "cpu"
      case Some("cuda") => "cuda"
      case _ => "auto"

  private def configuredAutoMinGpuWork: Long =
    GpuRuntimeSupport
      .resolveNonEmpty(AutoMinGpuWorkProperty, AutoMinGpuWorkEnv)
      .flatMap(GpuRuntimeSupport.parsePositiveIntOpt)
      .map(_.toLong)
      .getOrElse(DefaultAutoMinGpuWork)

  private def maybeApplyCachedPostflopAutoTune(deviceIndex: Int): Unit =
    if !postflopAutoTuneEnabled then ()
    else if hasExplicitPostflopCudaConfig then ()
    else
      val deviceCount = safeCudaDeviceCount()
      if deviceCount <= 0 then ()
      else
        val boundedDeviceIndex = math.max(0, math.min(deviceIndex, deviceCount - 1))
        val fingerprint = safeCudaDeviceFingerprint(boundedDeviceIndex)
        if fingerprint.isEmpty then ()
        else
          val appliedKey = s"$boundedDeviceIndex|$fingerprint"
          if appliedPostflopTuneFingerprintRef.get() == appliedKey then ()
          else
            loadPostflopAutoTuneDecision(resolvedPostflopAutoTuneCacheFile, boundedDeviceIndex, fingerprint) match
              case Some(decision) =>
                sys.props.update(PostflopCudaBlockSizeProperty, decision.blockSize.toString)
                sys.props.update(PostflopCudaMaxChunkMatchupsProperty, decision.maxChunkMatchups.toString)
                appliedPostflopTuneFingerprintRef.set(appliedKey)
                GpuRuntimeSupport.log(
                  s"postflop-autotune: applied cached config for device=$boundedDeviceIndex " +
                    s"(block=${decision.blockSize}, chunkMatchups=${decision.maxChunkMatchups})"
                )
              case None => ()

  private final case class PostflopAutoTuneDecision(
      blockSize: Int,
      maxChunkMatchups: Int
  )

  private def loadPostflopAutoTuneDecision(
      file: File,
      deviceIndex: Int,
      fingerprint: String
  ): Option[PostflopAutoTuneDecision] =
    import scala.util.boundary, boundary.break
    try
      if !file.isFile then None
      else
        val props = new Properties()
        val in = new FileInputStream(file)
        try props.load(in)
        finally in.close()

        val version = Option(props.getProperty("version")).map(_.trim).getOrElse("")
        if version != PostflopAutoTuneCacheVersion then None
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
                  val maxChunkOpt =
                    GpuRuntimeSupport
                      .parsePositiveIntOpt(props.getProperty(s"${prefix}maxChunkMatchups"))
                      .orElse(GpuRuntimeSupport.parsePositiveIntOpt(props.getProperty(s"${prefix}maxChunk")))
                  if blockSizeOpt.nonEmpty && maxChunkOpt.nonEmpty then
                    break(
                      Some(
                        PostflopAutoTuneDecision(
                          blockSize = blockSizeOpt.get,
                          maxChunkMatchups = maxChunkOpt.get
                        )
                      )
                    )
                idx += 1
              None
    catch
      case _: Throwable => None

  private def resolvedPostflopAutoTuneCacheFile: File =
    GpuRuntimeSupport.resolveFile(
      PostflopAutoTuneCachePathProperty,
      PostflopAutoTuneCachePathEnv,
      DefaultPostflopAutoTuneCachePath
    )

  private def postflopAutoTuneEnabled: Boolean =
    val raw = GpuRuntimeSupport.resolveNonEmptyLower(PostflopAutoTuneProperty, PostflopAutoTuneEnv)
    raw match
      case Some("0" | "false" | "no" | "off") => false
      case _ => true

  private def hasExplicitPostflopCudaConfig: Boolean =
    GpuRuntimeSupport.isConfigured(PostflopCudaBlockSizeProperty, PostflopCudaBlockSizeEnv) ||
    GpuRuntimeSupport.isConfigured(PostflopCudaMaxChunkMatchupsProperty, PostflopCudaMaxChunkMatchupsEnv)

  private def safeCudaDeviceCount(): Int =
    try HoldemPostflopNativeGpuBindings.cudaDeviceCount()
    catch
      case _: Throwable => 0

  private def safeCudaDeviceFingerprint(deviceIndex: Int): String =
    try Option(HoldemPostflopNativeGpuBindings.cudaDeviceInfo(deviceIndex)).map(_.trim).getOrElse("")
    catch
      case _: Throwable => ""

  private def safeEngineLabel(backend: Backend): String =
    try
      val code =
        backend match
          case Backend.Cpu => HoldemPostflopNativeBindings.queryNativeEngine()
          case Backend.Gpu => HoldemPostflopNativeGpuBindings.queryNativeEngine()
      code match
        case 1 => "cpu"
        case 2 => "cuda"
        case 3 => "cpu-fallback-after-cuda-failure"
        case 0 => "unknown"
        case other => s"unknown(code=$other)"
    catch
      case _: Throwable => "unknown"

  private def backendLabel(backend: Backend): String =
    backend match
      case Backend.Cpu => "cpu"
      case Backend.Gpu => "gpu"

  private def toResultArray(
      wins: Array[Double],
      ties: Array[Double],
      losses: Array[Double],
      stderrs: Array[Double]
  ): Array[EquityResultWithError] =
    val out = new Array[EquityResultWithError](wins.length)
    var i = 0
    while i < wins.length do
      out(i) = EquityResultWithError(
        win = wins(i),
        tie = ties(i),
        loss = losses(i),
        stderr = stderrs(i)
      )
      i += 1
    out

  private def tryCpuFallbackAfterGpuFailure(
      gpuStatus: Int,
      heroFirst: Int,
      heroSecond: Int,
      boardCards: Array[Int],
      villainFirst: Array[Int],
      villainSecond: Array[Int],
      trials: Int,
      seeds: Array[Long],
      wins: Array[Double],
      ties: Array[Double],
      losses: Array[Double],
      stderrs: Array[Double]
  ): Either[String, Array[EquityResultWithError]] =
    cpuLoadResult() match
      case Left(cpuReason) =>
        Left(
          s"${describeStatus(gpuStatus)}; nativeEngine=${safeEngineLabel(Backend.Gpu)}; " +
            s"CPU fallback unavailable: $cpuReason"
        )
      case Right(_) =>
        val cpuStatus = HoldemPostflopNativeBindings.computePostflopBatchMonteCarlo(
          heroFirst,
          heroSecond,
          boardCards,
          boardCards.length,
          villainFirst,
          villainSecond,
          trials,
          seeds,
          wins,
          ties,
          losses,
          stderrs
        )
        if cpuStatus != 0 then
          Left(
            s"${describeStatus(gpuStatus)}; GPU failed and CPU fallback also failed: " +
              s"${describeStatus(cpuStatus)}"
          )
        else
          Right(toResultArray(wins, ties, losses, stderrs))

  private def describeStatus(status: Int): String =
    val detail =
      status match
        case 100 => "null JNI array argument"
        case 101 => "JNI arrays have mismatched lengths"
        case 102 => "failed reading JNI input arrays"
        case 124 => "failed writing JNI output arrays"
        case 125 => "invalid card id"
        case 126 => "invalid Monte Carlo trial count"
        case 127 => "overlapping/duplicate cards in hero-board-villain inputs"
        case 128 => "invalid board size (expected 1..5)"
        case 130 => "CUDA device/runtime unavailable"
        case 131 => "CUDA device allocation failed"
        case 132 => "CUDA host-to-device transfer failed"
        case 133 => "CUDA kernel launch failed"
        case 134 =>
          "CUDA synchronize failed (likely Windows WDDM/TDR timeout for long kernels; reduce chunk size or trials)"
        case 135 => "CUDA device-to-host transfer failed"
        case 137 => "CUDA kernel timed out (Windows WDDM/TDR watchdog)"
        case _ => "unknown postflop native status"
    s"postflop native returned status=$status ($detail)"
