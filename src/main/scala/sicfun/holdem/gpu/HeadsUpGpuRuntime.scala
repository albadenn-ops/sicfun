package sicfun.holdem.gpu
import sicfun.holdem.types.*
import sicfun.holdem.*
import sicfun.holdem.equity.*

import java.util.concurrent.atomic.AtomicReference

/** GPU compute abstraction for batch equity computation in heads-up table generation.
  *
  * ==Thread Safety==
  * This object is '''thread-safe'''. All mutable state is managed through atomic references
  * (`AtomicReference[BatchTelemetry]`, `AtomicReference[Boolean]` for packed API support).
  * Provider selection is derived from immutable system properties/environment variables.
  * Native library loading is guarded by `lazy val` (JVM-level synchronization).
  * Multiple threads may call `computeBatch` concurrently; however, note that the
  * underlying native JNI calls may serialize on the GPU device.
  *
  * ==Provider Selection==
  * Supports multiple provider backends, selected via the `sicfun.gpu.provider` system property
  * or `sicfun_GPU_PROVIDER` environment variable:
  *
  *   - `"native"` (default) - loads a native JNI library (e.g. CUDA kernel) and delegates
  *     batch computation to the GPU via [[HeadsUpGpuNativeBindings]].
  *   - `"opencl"` - loads the OpenCL native library for iGPU computation (Monte Carlo only).
  *   - `"hybrid"` - distributes work across all available devices (CUDA + OpenCL + CPU).
  *   - `"cpu-emulated"` - runs the same per-matchup computation on the JVM CPU, useful
  *     for integration testing the GPU code path without actual GPU hardware.
  *   - `"disabled"` - always returns `Left`, disabling the GPU backend entirely.
  *
  * When the GPU backend fails, CPU fallback is available only if explicitly enabled
  * via `sicfun_GPU_FALLBACK_TO_CPU=true` (or `-Dsicfun.gpu.fallbackToCpu=true`).
  *
  * Telemetry from the last batch call is stored atomically and can be queried via
  * [[lastBatchTelemetry]].
  */
object HeadsUpGpuRuntime:
  /** Describes whether a GPU provider is currently available and usable.
    *
    * @param available `true` if the provider is loaded and ready for batch calls
    * @param provider  identifier string of the active provider
    * @param detail    human-readable explanation of the availability status
    */
  final case class Availability(available: Boolean, provider: String, detail: String)
  /** Telemetry record from the most recent batch computation attempt.
    *
    * @param provider active provider ID at the time of the call
    * @param success  `true` if the batch returned `Right`
    * @param detail   result summary or error reason
    */
  final case class BatchTelemetry(provider: String, success: Boolean, detail: String)
  private final case class BatchSuccess(
      values: Array[EquityResultWithError],
      detail: Option[String] = None
  )

  // ── Configuration property/env-var keys ──────────────────────────────────────
  // Each pair follows the convention: JVM system property + OS environment variable.
  // Resolved via GpuRuntimeSupport.resolveNonEmpty, which checks ScopedRuntimeProperties
  // (test overlay) first, then system properties, then env vars.

  private val ProviderProperty = "sicfun.gpu.provider"          // Selects the active provider backend
  private val ProviderEnv = "sicfun_GPU_PROVIDER"
  private val FallbackToCpuProperty = "sicfun.gpu.fallbackToCpu" // Enables JVM CPU fallback on GPU failure
  private val FallbackToCpuEnv = "sicfun_GPU_FALLBACK_TO_CPU"
  private val NativePathProperty = "sicfun.gpu.native.path"      // Explicit absolute path to the GPU DLL
  private val NativePathEnv = "sicfun_GPU_NATIVE_PATH"
  private val NativeLibProperty = "sicfun.gpu.native.lib"        // Library name for System.loadLibrary
  private val NativeLibEnv = "sicfun_GPU_NATIVE_LIB"
  private val DefaultNativeLibrary = "sicfun_gpu_kernel"         // Default DLL name (sicfun_gpu_kernel.dll)
  private val NativePackedIoProperty = "sicfun.gpu.native.packedIo"       // Enable packed f32 I/O path
  private val NativePackedIoEnv = "sicfun_GPU_NATIVE_PACKED_IO"
  private val NativePackedExactIoProperty = "sicfun.gpu.native.packedExactIo" // Also use packed I/O for exact mode
  private val NativePackedExactIoEnv = "sicfun_GPU_NATIVE_PACKED_EXACT_IO"
  private val NativeEngineProperty = "sicfun.gpu.native.engine"  // "cpu" or "cuda" -- selects compute engine inside native code
  private val NativeEngineEnv = "sicfun_GPU_NATIVE_ENGINE"
  private val NativeWarmupProperty = "sicfun.gpu.native.warmup"  // Enable/disable GPU warmup on first availability check
  private val NativeWarmupEnv = "sicfun_GPU_NATIVE_WARMUP"
  private val NativeWarmupMatchupsProperty = "sicfun.gpu.native.warmupMatchups"
  private val NativeWarmupMatchupsEnv = "sicfun_GPU_NATIVE_WARMUP_MATCHUPS"
  private val NativeWarmupTrialsProperty = "sicfun.gpu.native.warmupTrials"
  private val NativeWarmupTrialsEnv = "sicfun_GPU_NATIVE_WARMUP_TRIALS"
  private val DefaultWarmupTrials = 8

  private val OpenCLPathProperty = "sicfun.opencl.native.path"
  private val OpenCLPathEnv = "sicfun_OPENCL_NATIVE_PATH"
  private val OpenCLLibProperty = "sicfun.opencl.native.lib"
  private val OpenCLLibEnv = "sicfun_OPENCL_NATIVE_LIB"
  private val DefaultOpenCLLibrary = "sicfun_opencl_kernel"

  /** Atomically-updated telemetry from the most recent computeBatch call.
    * Allows callers to inspect provider, success/failure, and detail after the call.
    */
  private val telemetryRef = new AtomicReference[BatchTelemetry](null)

  /** Resets all cached native library load state and telemetry -- used only in tests
    * to ensure each test gets a clean provider configuration.
    */
  private[holdem] def resetLoadCacheForTests(): Unit =
    telemetryRef.set(null)
    NativeJniProvider.resetLoadCacheForTests()

  /** Internal SPI for GPU computation backends. */
  private trait Provider:
    def id: String
    def availability: Availability
    def computeBatch(
        packedKeys: Array[Long],
        keyMaterial: Array[Long],
        mode: HeadsUpEquityTable.Mode,
        monteCarloSeedBase: Long
    ): Either[String, BatchSuccess]

  /** Always-unavailable provider used when the GPU is explicitly disabled. */
  private object DisabledProvider extends Provider:
    override val id: String = "disabled"

    override def availability: Availability =
      Availability(
        available = false,
        provider = id,
        detail = "provider disabled via sicfun.gpu.provider=disabled"
      )

    override def computeBatch(
        packedKeys: Array[Long],
        keyMaterial: Array[Long],
        mode: HeadsUpEquityTable.Mode,
        monteCarloSeedBase: Long
    ): Either[String, BatchSuccess] =
      Left("GPU provider is disabled")

  /** CPU-based emulation of the GPU batch interface, using the same per-matchup
    * deterministic computation as [[HeadsUpEquityTable.computeEquityDeterministic]].
    * Useful for integration testing without GPU hardware.
    */
  private object CpuEmulatedProvider extends Provider:
    override val id: String = "cpu-emulated"

    override def availability: Availability =
      Availability(
        available = true,
        provider = id,
        detail = "CPU emulation provider active (for integration/benchmark harness)"
      )

    override def computeBatch(
        packedKeys: Array[Long],
        keyMaterial: Array[Long],
        mode: HeadsUpEquityTable.Mode,
        monteCarloSeedBase: Long
    ): Either[String, BatchSuccess] =
      try
        validateBatchShape(packedKeys, keyMaterial)
        val out = new Array[EquityResultWithError](packedKeys.length)
        var idx = 0
        while idx < packedKeys.length do
          val packed = packedKeys(idx)
          val hero = HoleCardsIndex.byId(HeadsUpEquityTable.unpackLowId(packed))
          val villain = HoleCardsIndex.byId(HeadsUpEquityTable.unpackHighId(packed))
          out(idx) =
            HeadsUpEquityTable.computeEquityDeterministic(
              hero = hero,
              villain = villain,
              mode = mode,
              monteCarloSeedBase = monteCarloSeedBase,
              keyMaterial = keyMaterial(idx)
            )
          idx += 1
        Right(BatchSuccess(out))
      catch
        case ex: Throwable => Left(ex.getMessage)

  /** JNI-based native provider that loads a platform-specific shared library (e.g. CUDA kernel)
    * and delegates batch computation to native code via [[HeadsUpGpuNativeBindings]].
    *
    * Library resolution order:
    *   1. Explicit path via `sicfun.gpu.native.path` / `sicfun_GPU_NATIVE_PATH` - `System.load(path)`
    *   2. Library name via `sicfun.gpu.native.lib` / `sicfun_GPU_NATIVE_LIB` - `System.loadLibrary(name)`
    *   3. Default library name `"sicfun_gpu_kernel"` - `System.loadLibrary("sicfun_gpu_kernel")`
    */
  private object NativeJniProvider extends Provider:
    override val id: String = "native"
    private val nativeLoadResultRef = new AtomicReference[Either[String, String]](null)
    private val packedApiSupportRef = new AtomicReference[java.lang.Boolean](null)
    private val warmupDoneRef = new AtomicReference[java.lang.Boolean](java.lang.Boolean.FALSE)

    private def nativeLoadResult(): Either[String, String] =
      val cached = nativeLoadResultRef.get()
      if cached != null then cached
      else
        val loaded =
          GpuRuntimeSupport.loadNativeLibrary(
            pathProperty = NativePathProperty,
            pathEnv = NativePathEnv,
            libProperty = NativeLibProperty,
            libEnv = NativeLibEnv,
            defaultLib = DefaultNativeLibrary,
            label = "native GPU library"
          )
        nativeLoadResultRef.compareAndSet(null, loaded)
        nativeLoadResultRef.get()

    def resetLoadCacheForTests(): Unit =
      nativeLoadResultRef.set(null)
      packedApiSupportRef.set(null)
      warmupDoneRef.set(java.lang.Boolean.FALSE)

    override def availability: Availability =
      nativeLoadResult() match
        case Right(source) =>
          maybeWarmUpOnAvailability()
          Availability(
            available = true,
            provider = id,
            detail = s"native JNI provider is loaded ($source)"
          )
        case Left(reason) =>
          Availability(
            available = false,
            provider = id,
            detail = reason
          )

    override def computeBatch(
        packedKeys: Array[Long],
        keyMaterial: Array[Long],
        mode: HeadsUpEquityTable.Mode,
        monteCarloSeedBase: Long
    ): Either[String, BatchSuccess] =
      nativeLoadResult() match
        case Left(reason) => Left(reason)
        case Right(_) =>
          try
            validateBatchShape(packedKeys, keyMaterial)
            val (modeCode, trials) =
              mode match
                case HeadsUpEquityTable.Mode.Exact => (0, 0)
                case HeadsUpEquityTable.Mode.MonteCarlo(t) => (1, t)
            if modeCode == 1 && configuredNativeEngine != "cpu" then
              ensureNativeWarmup(trials, monteCarloSeedBase)
            val shouldUsePacked =
              isPackedIoEnabled &&
                (modeCode == 1 || isPackedExactIoEnabled)
            if shouldUsePacked then
              computeBatchPackedFastPath(
                packedKeys = packedKeys,
                keyMaterial = keyMaterial,
                modeCode = modeCode,
                trials = trials,
                monteCarloSeedBase = monteCarloSeedBase
              ) match
                case Some(result) => result
                case None => computeBatchLegacyPath(packedKeys, keyMaterial, modeCode, trials, monteCarloSeedBase)
            else
              // Preserve exact-mode fidelity on legacy double I/O path.
              computeBatchLegacyPath(packedKeys, keyMaterial, modeCode, trials, monteCarloSeedBase)
          catch
            case ex: UnsatisfiedLinkError =>
              Left(s"native GPU symbols not found: ${ex.getMessage}")
            case ex: Throwable =>
              Left(ex.getMessage)

    private def maybeWarmUpOnAvailability(): Unit =
      if configuredNativeEngine != "cpu" then
        ensureNativeWarmup(
          requestedTrials = DefaultWarmupTrials,
          monteCarloSeedBase = 0x00000000BADC0FFEL
        )

    /** Performs a one-time GPU warmup (JIT compilation, memory allocation, driver init)
      * by running a small throwaway batch through both the legacy and packed API paths.
      *
      * The warmup is guarded by `warmupDoneRef` CAS to guarantee it runs at most once.
      * On Windows with WDDM, the first CUDA kernel launch can trigger a multi-second
      * driver initialisation delay; doing this proactively avoids latency spikes during
      * real computation.
      */
    private def ensureNativeWarmup(requestedTrials: Int, monteCarloSeedBase: Long): Unit =
      if !nativeWarmupEnabled then
        ()
      else if !warmupDoneRef.compareAndSet(java.lang.Boolean.FALSE, java.lang.Boolean.TRUE) then
        ()
      else
        try
          val matchups = nativeWarmupMatchups
          val batch = HeadsUpEquityTable.selectFullBatch(matchups.toLong)
          val n = batch.packedKeys.length
          if n <= 0 then
            ()
          else
            val lowIds = new Array[Int](n)
            val highIds = new Array[Int](n)
            val seeds = new Array[Long](n)
            var idx = 0
            while idx < n do
              val packed = batch.packedKeys(idx)
              lowIds(idx) = HeadsUpEquityTable.unpackLowId(packed)
              highIds(idx) = HeadsUpEquityTable.unpackHighId(packed)
              seeds(idx) = HeadsUpEquityTable.monteCarloSeed(monteCarloSeedBase, batch.keyMaterial(idx))
              idx += 1

            val trials = nativeWarmupTrials(requestedTrials)
            val wins = new Array[Double](n)
            val ties = new Array[Double](n)
            val losses = new Array[Double](n)
            val stderrs = new Array[Double](n)
            val legacyStatus =
              HeadsUpGpuNativeBindings.computeBatch(
                lowIds,
                highIds,
                1,
                trials,
                seeds,
                wins,
                ties,
                losses,
                stderrs
              )
            if legacyStatus != 0 then
              GpuRuntimeSupport.log(s"native warmup legacy status=$legacyStatus")

            if isPackedIoEnabled then
              val packed = new Array[Int](n)
              idx = 0
              while idx < n do
                packed(idx) = batch.packedKeys(idx).toInt
                idx += 1
              val winsF = new Array[Float](n)
              val tiesF = new Array[Float](n)
              val lossesF = new Array[Float](n)
              val stderrsF = new Array[Float](n)
              try
                val packedStatus =
                  HeadsUpGpuNativeBindings.computeBatchPacked(
                    packed,
                    1,
                    trials,
                    monteCarloSeedBase,
                    batch.keyMaterial,
                    winsF,
                    tiesF,
                    lossesF,
                    stderrsF
                  )
                packedApiSupportRef.compareAndSet(null, java.lang.Boolean.TRUE)
                if packedStatus != 0 then
                  GpuRuntimeSupport.log(s"native warmup packed status=$packedStatus")
              catch
                case _: UnsatisfiedLinkError =>
                  packedApiSupportRef.set(java.lang.Boolean.FALSE)
        catch
          case ex: Throwable =>
            val detail = Option(ex.getMessage).filter(_.nonEmpty).getOrElse(ex.getClass.getSimpleName)
            GpuRuntimeSupport.log(s"native warmup skipped: $detail")

    /** Attempts the packed f32 I/O code path.
      *
      * The packed API sends `Int` packed keys (lowId|highId in a single 32-bit value)
      * and receives `Float` results, reducing JNI transfer overhead by ~50% compared
      * to the legacy f64 path. The native DLL computes per-matchup seeds on-device in
      * Monte Carlo mode, further reducing host-to-device bandwidth.
      *
      * Returns `None` if the packed API is unavailable (older DLL without the symbol),
      * causing the caller to fall back to [[computeBatchLegacyPath]].
      */
    private def computeBatchPackedFastPath(
        packedKeys: Array[Long],
        keyMaterial: Array[Long],
        modeCode: Int,
        trials: Int,
        monteCarloSeedBase: Long
    ): Option[Either[String, BatchSuccess]] =
      if java.lang.Boolean.FALSE == packedApiSupportRef.get() then
        None
      else
        val packed = new Array[Int](packedKeys.length)
        var idx = 0
        while idx < packedKeys.length do
          packed(idx) = packedKeys(idx).toInt
          idx += 1
        val wins = new Array[Float](packedKeys.length)
        val ties = new Array[Float](packedKeys.length)
        val losses = new Array[Float](packedKeys.length)
        val stderrs = new Array[Float](packedKeys.length)
        try
          val status =
            HeadsUpGpuNativeBindings.computeBatchPacked(
              packed,
              modeCode,
              trials,
              monteCarloSeedBase,
              keyMaterial,
              wins,
              ties,
              losses,
              stderrs
            )
          packedApiSupportRef.compareAndSet(null, java.lang.Boolean.TRUE)
          val nativeEngine = readNativeEngineLabel
          if status != 0 then
            Some(Left(s"${describeNativeStatus(status)}; nativeEngine=$nativeEngine"))
          else
            val out = new Array[EquityResultWithError](packedKeys.length)
            idx = 0
            while idx < packedKeys.length do
              out(idx) = EquityResultWithError(
                wins(idx).toDouble,
                ties(idx).toDouble,
                losses(idx).toDouble,
                stderrs(idx).toDouble
              )
              idx += 1
            Some(
              Right(
                BatchSuccess(
                  out,
                  detail =
                    Some(
                      if modeCode == 0 then
                        s"nativeEngine=$nativeEngine, io=packed-f32"
                      else
                        s"nativeEngine=$nativeEngine, io=packed-f32-seed-on-device"
                    )
                )
              )
            )
        catch
          case _: UnsatisfiedLinkError =>
            packedApiSupportRef.set(java.lang.Boolean.FALSE)
            None

    /** Legacy f64 JNI code path.
      *
      * Unpacks each key into separate lowId/highId arrays (both `Int`), pre-computes
      * per-matchup seeds on the JVM side, and receives results as `Double` arrays.
      * Used when the packed API is unavailable or when exact-mode packed I/O is disabled.
      */
    private def computeBatchLegacyPath(
        packedKeys: Array[Long],
        keyMaterial: Array[Long],
        modeCode: Int,
        trials: Int,
        monteCarloSeedBase: Long
    ): Either[String, BatchSuccess] =
      val lowIds = new Array[Int](packedKeys.length)
      val highIds = new Array[Int](packedKeys.length)
      val seeds = new Array[Long](packedKeys.length)
      var idx = 0
      while idx < packedKeys.length do
        val packed = packedKeys(idx)
        lowIds(idx) = HeadsUpEquityTable.unpackLowId(packed)
        highIds(idx) = HeadsUpEquityTable.unpackHighId(packed)
        seeds(idx) = HeadsUpEquityTable.monteCarloSeed(monteCarloSeedBase, keyMaterial(idx))
        idx += 1

      val wins = new Array[Double](packedKeys.length)
      val ties = new Array[Double](packedKeys.length)
      val losses = new Array[Double](packedKeys.length)
      val stderrs = new Array[Double](packedKeys.length)
      val status =
        HeadsUpGpuNativeBindings.computeBatch(
          lowIds,
          highIds,
          modeCode,
          trials,
          seeds,
          wins,
          ties,
          losses,
          stderrs
        )
      val nativeEngine = readNativeEngineLabel
      if status != 0 then
        Left(s"${describeNativeStatus(status)}; nativeEngine=$nativeEngine")
      else
        val out = new Array[EquityResultWithError](packedKeys.length)
        idx = 0
        while idx < packedKeys.length do
          out(idx) = EquityResultWithError(wins(idx), ties(idx), losses(idx), stderrs(idx))
          idx += 1
        Right(BatchSuccess(out, detail = Some(s"nativeEngine=$nativeEngine, io=legacy-f64")))

    private def isPackedIoEnabled: Boolean =
      GpuRuntimeSupport.resolveNonEmpty(NativePackedIoProperty, NativePackedIoEnv)
        .map(GpuRuntimeSupport.parseTruthy)
        .getOrElse(true)

    private def isPackedExactIoEnabled: Boolean =
      GpuRuntimeSupport.resolveNonEmpty(NativePackedExactIoProperty, NativePackedExactIoEnv)
        .map(GpuRuntimeSupport.parseTruthy)
        .getOrElse(false)

    private def nativeWarmupEnabled: Boolean =
      GpuRuntimeSupport.resolveNonEmpty(NativeWarmupProperty, NativeWarmupEnv)
        .map(GpuRuntimeSupport.parseTruthy)
        .getOrElse(true)

    private def nativeWarmupMatchups: Int =
      GpuRuntimeSupport.resolveNonEmpty(NativeWarmupMatchupsProperty, NativeWarmupMatchupsEnv)
        .flatMap(GpuRuntimeSupport.parsePositiveIntOpt)
        .getOrElse(64)

    private def nativeWarmupTrials(requestedTrials: Int): Int =
      GpuRuntimeSupport.resolveNonEmpty(NativeWarmupTrialsProperty, NativeWarmupTrialsEnv)
        .flatMap(GpuRuntimeSupport.parsePositiveIntOpt)
        .getOrElse(math.min(DefaultWarmupTrials, math.max(1, requestedTrials)))

    private def configuredNativeEngine: String =
      GpuRuntimeSupport.resolveNonEmptyLower(NativeEngineProperty, NativeEngineEnv).getOrElse("cuda")

    private def readNativeEngineLabel: String =
      try
        describeNativeEngine(HeadsUpGpuNativeBindings.lastEngineCode())
      catch
        case _: UnsatisfiedLinkError =>
          "unknown(no-lastEngineCode)"
        case ex: Throwable =>
          val detail = Option(ex.getMessage).map(_.trim).filter(_.nonEmpty).getOrElse(ex.getClass.getSimpleName)
          s"unknown($detail)"

  /** OpenCL-based provider for Intel/AMD integrated GPUs.
    *
    * Only Monte Carlo mode is supported; exact mode returns `Left`.
    * The native library dynamically loads `OpenCL.dll` at runtime, so this
    * provider gracefully degrades when OpenCL is not installed.
    */
  private object OpenCLProvider extends Provider:
    override val id: String = "opencl"

    private lazy val openclLoadResult: Either[String, String] =
      GpuRuntimeSupport.loadNativeLibrary(
        pathProperty = OpenCLPathProperty,
        pathEnv = OpenCLPathEnv,
        libProperty = OpenCLLibProperty,
        libEnv = OpenCLLibEnv,
        defaultLib = DefaultOpenCLLibrary,
        label = "OpenCL native library"
      )

    override def availability: Availability =
      openclLoadResult match
        case Right(_) =>
          Availability(available = true, provider = id, detail = "OpenCL JNI provider is loaded")
        case Left(reason) =>
          Availability(available = false, provider = id, detail = reason)

    override def computeBatch(
        packedKeys: Array[Long],
        keyMaterial: Array[Long],
        mode: HeadsUpEquityTable.Mode,
        monteCarloSeedBase: Long
    ): Either[String, BatchSuccess] =
      mode match
        case HeadsUpEquityTable.Mode.Exact =>
          Left("OpenCL provider does not support exact mode; use CUDA or CPU backend")
        case HeadsUpEquityTable.Mode.MonteCarlo(trials) =>
          openclLoadResult match
            case Left(reason) => Left(reason)
            case Right(_) =>
              try
                validateBatchShape(packedKeys, keyMaterial)
                val n = packedKeys.length
                val lowIds = new Array[Int](n)
                val highIds = new Array[Int](n)
                val seeds = new Array[Long](n)
                var idx = 0
                while idx < n do
                  val packed = packedKeys(idx)
                  lowIds(idx) = HeadsUpEquityTable.unpackLowId(packed)
                  highIds(idx) = HeadsUpEquityTable.unpackHighId(packed)
                  seeds(idx) = HeadsUpEquityTable.monteCarloSeed(monteCarloSeedBase, keyMaterial(idx))
                  idx += 1

                val wins = new Array[Double](n)
                val ties = new Array[Double](n)
                val losses = new Array[Double](n)
                val stderrs = new Array[Double](n)
                val status =
                  HeadsUpOpenCLNativeBindings.computeBatch(
                    0, lowIds, highIds, 1, trials, seeds, wins, ties, losses, stderrs
                  )
                if status != 0 then
                  Left(describeOpenCLStatus(status))
                else
                  val out = new Array[EquityResultWithError](n)
                  idx = 0
                  while idx < n do
                    out(idx) = EquityResultWithError(wins(idx), ties(idx), losses(idx), stderrs(idx))
                    idx += 1
                  Right(BatchSuccess(out, detail = Some("engine=opencl")))
              catch
                case ex: UnsatisfiedLinkError =>
                  Left(s"OpenCL native symbols not found: ${ex.getMessage}")
                case ex: Throwable =>
                  Left(ex.getMessage)

  /** Hybrid provider that distributes work across all available devices (CUDA + OpenCL + CPU)
    * using [[HeadsUpHybridDispatcher]] for proportional splitting and parallel dispatch.
    */
  private object HybridProvider extends Provider:
    override val id: String = "hybrid"

    override def availability: Availability =
      val devices = HeadsUpHybridDispatcher.devices
      if devices.nonEmpty then
        val summary = devices.map(d => s"${d.id}(${d.name})").mkString(", ")
        Availability(available = true, provider = id, detail = s"hybrid: ${devices.size} devices - $summary")
      else
        Availability(available = false, provider = id, detail = "no compute devices discovered")

    override def computeBatch(
        packedKeys: Array[Long],
        keyMaterial: Array[Long],
        mode: HeadsUpEquityTable.Mode,
        monteCarloSeedBase: Long
    ): Either[String, BatchSuccess] =
      try
        validateBatchShape(packedKeys, keyMaterial)
        val n = packedKeys.length
        val lowIds = new Array[Int](n)
        val highIds = new Array[Int](n)
        val seeds = new Array[Long](n)
        var idx = 0
        while idx < n do
          val packed = packedKeys(idx)
          lowIds(idx) = HeadsUpEquityTable.unpackLowId(packed)
          highIds(idx) = HeadsUpEquityTable.unpackHighId(packed)
          seeds(idx) = HeadsUpEquityTable.monteCarloSeed(monteCarloSeedBase, keyMaterial(idx))
          idx += 1

        val (modeCode, trials) = mode match
          case HeadsUpEquityTable.Mode.Exact => (0, 0)
          case HeadsUpEquityTable.Mode.MonteCarlo(t) => (1, t)

        HeadsUpHybridDispatcher.dispatchBatch(lowIds, highIds, modeCode, trials, seeds) match
          case Left(error) => Left(error)
          case Right(result) =>
            val deviceSummary =
              if result.perDevice.isEmpty then "none"
              else
                result.perDevice
                  .map(t => f"${t.deviceId}:${t.matchups}matchups/${t.elapsedMs}ms")
                  .mkString(", ")
            val recoverySummary =
              if result.recovery.isEmpty then "none"
              else
                result.recovery
                  .map(r => s"${r.failedDeviceId}->${r.recoveredByDeviceId}:${r.matchups}")
                  .mkString(", ")
            val checksumHex = f"${result.payloadCrc32}%08x"
            Right(
              BatchSuccess(
                result.results,
                detail = Some(
                  s"hybrid[devices=$deviceSummary; recovery=$recoverySummary; crc32=$checksumHex]"
                )
              )
            )
      catch
        case ex: Throwable => Left(ex.getMessage)

  /** Queries whether the active GPU provider is loaded and ready for computation. */
  def availability: Availability =
    activeProvider.availability

  /** Returns `true` if CPU fallback is enabled when the GPU backend fails.
    *
    * Controlled by `sicfun.gpu.fallbackToCpu` system property or `sicfun_GPU_FALLBACK_TO_CPU`
    * environment variable. Defaults to `false` (fail-fast on GPU errors).
    */
  def allowCpuFallbackOnGpuFailure: Boolean =
    GpuRuntimeSupport.resolveNonEmpty(FallbackToCpuProperty, FallbackToCpuEnv)
      .exists(GpuRuntimeSupport.parseTruthy)

  def lastBatchTelemetry: Option[BatchTelemetry] =
    Option(telemetryRef.get())

  /** Dispatches a batch equity computation to the active GPU provider.
    *
    * @param packedKeys         packed (lowId, highId) keys for each matchup
    * @param keyMaterial        per-matchup seed material for deterministic Monte Carlo
    * @param mode               exact or Monte Carlo computation mode
    * @param monteCarloSeedBase global seed base combined with `keyMaterial` for per-matchup seeding
    * @return `Right(results)` on success, `Left(reason)` on failure
    */
  def computeBatch(
      packedKeys: Array[Long],
      keyMaterial: Array[Long],
      mode: HeadsUpEquityTable.Mode,
      monteCarloSeedBase: Long
  ): Either[String, Array[EquityResultWithError]] =
    val provider = activeProvider
    val result = provider.computeBatch(packedKeys, keyMaterial, mode, monteCarloSeedBase)
    result match
      case Right(success) =>
        val suffix = success.detail.map(value => s", $value").getOrElse("")
        telemetryRef.set(
          BatchTelemetry(
            provider = provider.id,
            success = true,
            detail = s"entries=${success.values.length}$suffix"
          )
        )
      case Left(reason) =>
        telemetryRef.set(BatchTelemetry(provider = provider.id, success = false, detail = reason))
    result.map(_.values)

  /** Reads the configured provider string, defaulting to "native" (JNI/CUDA). */
  private def configuredProvider: String =
    GpuRuntimeSupport.resolveNonEmptyLower(ProviderProperty, ProviderEnv).getOrElse("native")

  /** Maps the configured provider string to the corresponding Provider object.
    * Unknown providers are logged as errors and mapped to DisabledProvider.
    */
  private def activeProvider: Provider =
    configuredProvider match
      case "native" => NativeJniProvider
      case "opencl" => OpenCLProvider
      case "hybrid" => HybridProvider
      case "cpu-emulated" => CpuEmulatedProvider
      case "disabled" => DisabledProvider
      case other =>
        telemetryRef.set(BatchTelemetry(provider = other, success = false, detail = s"unknown GPU provider '$other'"))
        DisabledProvider

  /** Maps native JNI status codes to human-readable error descriptions.
    * Status codes 100-127 are JNI/input validation errors; 130+ are CUDA runtime errors.
    */
  private def describeNativeStatus(status: Int): String =
    GpuRuntimeSupport.describeNativeStatus(status)

  /** Maps the integer engine code returned by `HeadsUpGpuNativeBindings.lastEngineCode()`
    * to a human-readable label. Codes: 1=cpu, 2=cuda, 3=cpu-fallback, 4=opencl.
    */
  private def describeNativeEngine(code: Int): String =
    code match
      case 1 => "cpu"
      case 2 => "cuda"
      case 3 => "cpu-fallback-after-cuda-failure"
      case 4 => "opencl"
      case 0 => "unknown"
      case other => s"unknown(code=$other)"

  private def describeOpenCLStatus(status: Int): String =
    GpuRuntimeSupport.describeOpenCLStatus(status)

  /** Pre-flight validation: packedKeys and keyMaterial must have equal length. */
  private def validateBatchShape(packedKeys: Array[Long], keyMaterial: Array[Long]): Unit =
    require(packedKeys.length == keyMaterial.length, "packedKeys and keyMaterial must have equal length")
