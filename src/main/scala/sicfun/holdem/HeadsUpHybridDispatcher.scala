package sicfun.holdem

import java.util.concurrent.{Callable, ExecutorService, Executors, Future, ThreadFactory}
import java.util.concurrent.atomic.{AtomicBoolean, AtomicInteger, AtomicReference}
import java.util.Locale
import java.util.zip.CRC32

/** Multi-device hybrid dispatcher that distributes batch equity computation
  * across all available compute devices: CUDA GPUs, OpenCL iGPUs, and CPU threads.
  *
  * Work is split proportionally to each device's estimated (or calibrated) throughput,
  * dispatched in parallel, and results are merged back into the original array order.
  *
  * '''Determinism:''' Seeds are pre-computed per matchup and independent of the
  * processing device, so results are identical regardless of how the batch is split.
  *
  * ==Thread Safety==
  * This object is '''thread-safe'''. Calibrated weights are stored in an `AtomicReference`.
  * Device discovery is guarded by `lazy val`. Dispatch tasks are executed via a shared
  * cached executor to avoid repeated thread construction overhead.
  *
  * ==Failover Behavior==
  * When a device fails (returns a non-zero status code or throws an exception):
  *   1. The dispatcher selects rescue candidates from the remaining active devices,
  *      ordered by recovery priority: CPU (highest) > OpenCL > CUDA (lowest),
  *      with ties broken by descending effective weight, then stable device ID order.
  *   2. Each rescue candidate is tried sequentially until one succeeds.
  *   3. On successful rescue, a [[RecoveryTelemetry]] record captures the failed device,
  *      its status code, the rescuing device, and timing for both attempts.
  *   4. If ''all'' rescue candidates also fail, the dispatch returns `Left` with a
  *      detailed error message listing every failure.
  *
  * In multi-device parallel dispatch, recovery is attempted independently per failed slice.
  * A single slice failure does not abort other slices that completed successfully.
  */
object HeadsUpHybridDispatcher:
  private val HybridWeightsProperty = "sicfun.hybrid.weights"
  private val HybridWeightsEnv = "sicfun_HYBRID_WEIGHTS"
  private val HybridMinSliceMatchupsProperty = "sicfun.hybrid.minSliceMatchups"
  private val HybridMinSliceMatchupsEnv = "sicfun_HYBRID_MIN_SLICE_MATCHUPS"
  private val HybridMinRelativeWeightProperty = "sicfun.hybrid.minRelativeWeight"
  private val HybridMinRelativeWeightEnv = "sicfun_HYBRID_MIN_RELATIVE_WEIGHT"
  private val HybridCpuOnlyBelowProperty = "sicfun.hybrid.cpuOnlyBelow"
  private val HybridCpuOnlyBelowEnv = "sicfun_HYBRID_CPU_ONLY_BELOW"
  private val HybridAdaptiveWeightsProperty = "sicfun.hybrid.adaptiveWeights"
  private val HybridAdaptiveWeightsEnv = "sicfun_HYBRID_ADAPTIVE_WEIGHTS"
  private val HybridAdaptiveAlphaProperty = "sicfun.hybrid.adaptiveAlpha"
  private val HybridAdaptiveAlphaEnv = "sicfun_HYBRID_ADAPTIVE_ALPHA"
  private val HybridWarmupProperty = "sicfun.hybrid.warmup"
  private val HybridWarmupEnv = "sicfun_HYBRID_WARMUP"
  private val HybridWarmupMatchupsProperty = "sicfun.hybrid.warmupMatchups"
  private val HybridWarmupMatchupsEnv = "sicfun_HYBRID_WARMUP_MATCHUPS"
  private val HybridWarmupTrialsProperty = "sicfun.hybrid.warmupTrials"
  private val HybridWarmupTrialsEnv = "sicfun_HYBRID_WARMUP_TRIALS"
  private val HybridCudaPrimaryProperty = "sicfun.hybrid.cudaPrimary"
  private val HybridCudaPrimaryEnv = "sicfun_HYBRID_CUDA_PRIMARY"
  private val HybridIncludeCpuWithGpuProperty = "sicfun.hybrid.includeCpuWithGpu"
  private val HybridIncludeCpuWithGpuEnv = "sicfun_HYBRID_INCLUDE_CPU_WITH_GPU"
  private val HybridHelperMinRelativeToCudaProperty = "sicfun.hybrid.helperMinRelativeToCuda"
  private val HybridHelperMinRelativeToCudaEnv = "sicfun_HYBRID_HELPER_MIN_RELATIVE_TO_CUDA"

  private inline val DefaultMinSliceMatchups = 64
  private inline val DefaultMinRelativeWeight = 0.0
  private inline val DefaultCpuOnlyBelow = 8_192
  private inline val DefaultAdaptiveAlpha = 0.35
  private inline val DefaultAdaptiveStepFactor = 1.35
  private inline val DefaultAdaptiveMinElapsedNanos = 5_000_000L // 5 ms
  private inline val DefaultWarmupMatchups = 64
  private inline val DefaultWarmupTrials = 8
  private inline val DefaultCudaPrimaryEnabled = true
  private inline val DefaultIncludeCpuWithGpu = false
  private inline val DefaultHelperMinRelativeToCuda = 0.0


  // ------ Device abstraction ------------------------------------------------------------------------------------------------------------------------------

  /** A compute device capable of processing equity batch sub-ranges. */
  sealed trait ComputeDevice:
    def id: String
    def kind: String
    def name: String
    def supportsExact: Boolean
    def estimatedWeight: Double

    /** Computes equity for the given sub-batch (arrays are pre-sliced).
      * @return 0 on success, non-zero status code on failure
      */
    def computeSubBatch(
        lowIds: Array[Int],
        highIds: Array[Int],
        modeCode: Int,
        trials: Int,
        seeds: Array[Long],
        wins: Array[Double],
        ties: Array[Double],
        losses: Array[Double],
        stderrs: Array[Double]
    ): Int

  final case class CudaComputeDevice(
      index: Int,
      override val name: String,
      smCount: Int,
      clockMHz: Int
  ) extends ComputeDevice:
    override val id: String = s"cuda:$index"
    override val kind: String = "cuda"
    override val supportsExact: Boolean = true
    override val estimatedWeight: Double = smCount.toDouble * clockMHz.toDouble

    override def computeSubBatch(
        lowIds: Array[Int],
        highIds: Array[Int],
        modeCode: Int,
        trials: Int,
        seeds: Array[Long],
        wins: Array[Double],
        ties: Array[Double],
        losses: Array[Double],
        stderrs: Array[Double]
    ): Int =
      HeadsUpGpuNativeBindings.computeBatchOnDevice(
        index, lowIds, highIds, modeCode, trials, seeds, wins, ties, losses, stderrs
      )

  final case class OpenCLComputeDevice(
      index: Int,
      override val name: String,
      computeUnits: Int,
      clockMHz: Int
  ) extends ComputeDevice:
    override val id: String = s"opencl:$index"
    override val kind: String = "opencl"
    override val supportsExact: Boolean = false
    override val estimatedWeight: Double = computeUnits.toDouble * clockMHz.toDouble * 0.3

    override def computeSubBatch(
        lowIds: Array[Int],
        highIds: Array[Int],
        modeCode: Int,
        trials: Int,
        seeds: Array[Long],
        wins: Array[Double],
        ties: Array[Double],
        losses: Array[Double],
        stderrs: Array[Double]
    ): Int =
      HeadsUpOpenCLNativeBindings.computeBatch(
        index, lowIds, highIds, modeCode, trials, seeds, wins, ties, losses, stderrs
      )

  final case class CpuComputeDevice(threadCount: Int) extends ComputeDevice:
    override val id: String = "cpu"
    override val kind: String = "cpu"
    override val name: String = s"CPU ($threadCount threads)"
    override val supportsExact: Boolean = true
    override val estimatedWeight: Double = threadCount.toDouble * 1000.0

    override def computeSubBatch(
        lowIds: Array[Int],
        highIds: Array[Int],
        modeCode: Int,
        trials: Int,
        seeds: Array[Long],
        wins: Array[Double],
        ties: Array[Double],
        losses: Array[Double],
        stderrs: Array[Double]
    ): Int =
      try
        HeadsUpGpuNativeBindings.computeBatchCpuOnly(
          lowIds, highIds, modeCode, trials, seeds, wins, ties, losses, stderrs
        )
      catch
        case _: UnsatisfiedLinkError =>
          // Older native libraries may not provide computeBatchCpuOnly.
          HeadsUpGpuNativeBindings.computeBatch(
            lowIds, highIds, modeCode, trials, seeds, wins, ties, losses, stderrs
          )

  /** Test-only in-memory compute device for deterministic dispatch/failover scenarios. */
  private[holdem] final case class FunctionalComputeDevice(
      override val id: String,
      override val kind: String,
      override val name: String,
      override val supportsExact: Boolean,
      override val estimatedWeight: Double,
      computeFn: (
          Array[Int],
          Array[Int],
          Int,
          Int,
          Array[Long],
          Array[Double],
          Array[Double],
          Array[Double],
          Array[Double]
      ) => Int
  ) extends ComputeDevice:
    override def computeSubBatch(
        lowIds: Array[Int],
        highIds: Array[Int],
        modeCode: Int,
        trials: Int,
        seeds: Array[Long],
        wins: Array[Double],
        ties: Array[Double],
        losses: Array[Double],
        stderrs: Array[Double]
    ): Int =
      computeFn(lowIds, highIds, modeCode, trials, seeds, wins, ties, losses, stderrs)

  // ------ Per-device telemetry ------------------------------------------------------------------------------------------------------------------------

  final case class DeviceTelemetry(
      deviceId: String,
      deviceName: String,
      matchups: Int,
      elapsedMs: Long,
      elapsedNanos: Long = 0L
  ):
    def throughput: Double =
      val nanos =
        if elapsedNanos > 0L then elapsedNanos
        else if elapsedMs > 0L then elapsedMs * 1_000_000L
        else 0L
      if nanos > 0L then matchups.toDouble / (nanos.toDouble / 1_000_000_000.0) else 0.0

  final case class RecoveryTelemetry(
      failedDeviceId: String,
      failedStatus: Int,
      recoveredByDeviceId: String,
      matchups: Int,
      failedElapsedMs: Long,
      recoveredElapsedMs: Long
  )

  // ------ Batch split descriptor ------------------------------------------------------------------------------------------------------------------

  private final case class SubBatchSlice(
      device: ComputeDevice,
      startIdx: Int,
      count: Int
  )

  private final case class SliceAttempt(
      slice: SubBatchSlice,
      status: Int,
      elapsedMs: Long,
      elapsedNanos: Long,
      error: Option[String]
  )

  // ------ Calibrated weights (updated by auto-tuner) ---------------------------------------------------

  private val calibratedWeightsRef =
    new AtomicReference[Map[String, Double]](Map.empty)
  private val warmupDone = new AtomicBoolean(false)
  private val dispatchThreadCounter = new AtomicInteger(1)
  private val dispatchPool: ExecutorService =
    Executors.newCachedThreadPool(new ThreadFactory {
      override def newThread(runnable: Runnable): Thread =
        val thread = new Thread(runnable, s"sicfun-hybrid-dispatch-${dispatchThreadCounter.getAndIncrement()}")
        thread.setDaemon(true)
        thread
    })

  def setCalibratedWeights(weights: Map[String, Double]): Unit =
    calibratedWeightsRef.set(weights)

  def calibratedWeights: Map[String, Double] =
    calibratedWeightsRef.get()

  private def configuredAdaptiveWeightsEnabled: Boolean =
    GpuRuntimeSupport.resolveNonEmpty(HybridAdaptiveWeightsProperty, HybridAdaptiveWeightsEnv)
      .map(GpuRuntimeSupport.parseTruthy)
      .getOrElse(true)

  private def configuredAdaptiveAlpha: Double =
    val parsed =
      GpuRuntimeSupport.resolveNonEmpty(HybridAdaptiveAlphaProperty, HybridAdaptiveAlphaEnv)
        .flatMap(raw => scala.util.Try(raw.toDouble).toOption)
        .filter(value => java.lang.Double.isFinite(value) && value > 0.0 && value <= 1.0)
    parsed.getOrElse(DefaultAdaptiveAlpha)

  private def configuredCpuOnlyBelow: Int =
    val parsed =
      GpuRuntimeSupport.resolveNonEmpty(HybridCpuOnlyBelowProperty, HybridCpuOnlyBelowEnv)
        .flatMap(GpuRuntimeSupport.parseNonNegativeIntOpt)
    parsed.getOrElse(DefaultCpuOnlyBelow)

  private def configuredWarmupEnabled: Boolean =
    GpuRuntimeSupport.resolveNonEmpty(HybridWarmupProperty, HybridWarmupEnv)
      .map(GpuRuntimeSupport.parseTruthy)
      .getOrElse(true)

  private def configuredWarmupMatchups: Int =
    val parsed =
      GpuRuntimeSupport.resolveNonEmpty(HybridWarmupMatchupsProperty, HybridWarmupMatchupsEnv)
        .flatMap(GpuRuntimeSupport.parsePositiveIntOpt)
    parsed.getOrElse(DefaultWarmupMatchups)

  private def configuredWarmupTrials(requestedTrials: Int): Int =
    val configured =
      GpuRuntimeSupport.resolveNonEmpty(HybridWarmupTrialsProperty, HybridWarmupTrialsEnv)
        .flatMap(GpuRuntimeSupport.parsePositiveIntOpt)
    configured.getOrElse(math.min(DefaultWarmupTrials, math.max(1, requestedTrials)))

  private def configuredCudaPrimaryEnabled: Boolean =
    GpuRuntimeSupport.resolveNonEmpty(HybridCudaPrimaryProperty, HybridCudaPrimaryEnv)
      .map(GpuRuntimeSupport.parseTruthy)
      .getOrElse(DefaultCudaPrimaryEnabled)

  private def configuredIncludeCpuWithGpu: Boolean =
    GpuRuntimeSupport.resolveNonEmpty(HybridIncludeCpuWithGpuProperty, HybridIncludeCpuWithGpuEnv)
      .map(GpuRuntimeSupport.parseTruthy)
      .getOrElse(DefaultIncludeCpuWithGpu)

  private def configuredHelperMinRelativeToCuda: Double =
    val parsed =
      GpuRuntimeSupport.resolveNonEmpty(HybridHelperMinRelativeToCudaProperty, HybridHelperMinRelativeToCudaEnv)
        .flatMap(raw => scala.util.Try(raw.toDouble).toOption)
    parsed
      .filter(value => java.lang.Double.isFinite(value) && value >= 0.0 && value <= 1.0)
      .getOrElse(DefaultHelperMinRelativeToCuda)

  private def applyCudaPrimaryPolicy(activeDevices: Vector[ComputeDevice]): Vector[ComputeDevice] =
    if !configuredCudaPrimaryEnabled then
      activeDevices
    else
      val hasCuda = activeDevices.exists(_.kind == "cuda")
      if !hasCuda then
        activeDevices
      else
        val includeCpuWithGpu = configuredIncludeCpuWithGpu
        val baseSelection =
          activeDevices.filter { device =>
            device.kind == "cuda" || device.kind == "opencl" || (includeCpuWithGpu && device.kind == "cpu")
          }
        val helperCutoff = configuredHelperMinRelativeToCuda
        if helperCutoff <= 0.0 then
          baseSelection
        else
          val bestCudaWeight = activeDevices.iterator.filter(_.kind == "cuda").map(effectiveWeight).max
          val minWeight = bestCudaWeight * helperCutoff
          val filtered =
            baseSelection.filter(device => device.kind == "cuda" || effectiveWeight(device) >= minWeight)
          if filtered.exists(_.kind == "cuda") then filtered else baseSelection.filter(_.kind == "cuda")

  private def maybeWarmUpDevices(modeCode: Int, trials: Int, activeDevices: Vector[ComputeDevice]): Unit =
    if !configuredWarmupEnabled || activeDevices.isEmpty then
      ()
    else if !warmupDone.compareAndSet(false, true) then
      ()
    else
      try
        val warmMatchups = configuredWarmupMatchups
        val batch = HeadsUpEquityTable.selectFullBatch(warmMatchups.toLong)
        val n = batch.packedKeys.length
        if n > 0 then
          val lowIds = new Array[Int](n)
          val highIds = new Array[Int](n)
          val seeds = new Array[Long](n)
          val measuredThroughput = scala.collection.mutable.HashMap.empty[String, Double]
          val warmTrials = configuredWarmupTrials(trials)
          var idx = 0
          while idx < n do
            val packed = batch.packedKeys(idx)
            lowIds(idx) = HeadsUpEquityTable.unpackLowId(packed)
            highIds(idx) = HeadsUpEquityTable.unpackHighId(packed)
            seeds(idx) = HeadsUpEquityTable.monteCarloSeed(0x00C0FFEE1234L, batch.keyMaterial(idx))
            idx += 1
          val warmModeCode = 1
          activeDevices.foreach { device =>
            val wins = new Array[Double](n)
            val ties = new Array[Double](n)
            val losses = new Array[Double](n)
            val stderrs = new Array[Double](n)
            val started = System.nanoTime()
            val status =
              device.computeSubBatch(
                lowIds,
                highIds,
                warmModeCode,
                warmTrials,
                seeds,
                wins,
                ties,
                losses,
                stderrs
              )
            val elapsedNanos = System.nanoTime() - started
            if status != 0 then
              GpuRuntimeSupport.log(s"hybrid warmup: device ${device.id} returned status=$status")
            else if configuredAdaptiveWeightsEnabled && elapsedNanos >= DefaultAdaptiveMinElapsedNanos then
              val throughput = n.toDouble / (elapsedNanos.toDouble / 1_000_000_000.0)
              measuredThroughput.update(device.id, throughput)
          }
          if configuredAdaptiveWeightsEnabled && measuredThroughput.nonEmpty then
            val current = calibratedWeightsRef.get()
            calibratedWeightsRef.set(current ++ measuredThroughput.toMap)
      catch
        case ex: Throwable =>
          val detail = Option(ex.getMessage).filter(_.nonEmpty).getOrElse(ex.getClass.getSimpleName)
          GpuRuntimeSupport.log(s"hybrid warmup skipped: $detail")

  private def maybeUpdateCalibratedWeights(perDevice: Vector[DeviceTelemetry]): Unit =
    if !configuredAdaptiveWeightsEnabled || perDevice.isEmpty then
      ()
    else
      val alpha = configuredAdaptiveAlpha
      val minAcceptedMatchups = math.max(256, configuredMinSliceMatchups)
      val measured =
        perDevice
          .filter { t =>
            val nanos =
              if t.elapsedNanos > 0L then t.elapsedNanos
              else if t.elapsedMs > 0L then t.elapsedMs * 1_000_000L
              else 0L
            nanos >= DefaultAdaptiveMinElapsedNanos &&
            t.matchups >= minAcceptedMatchups &&
            t.throughput > 0.0
          }
          .map(t => t.deviceId -> t.throughput)
      if measured.nonEmpty then
        val current = calibratedWeightsRef.get()
        val updated =
          measured.foldLeft(current) { case (acc, (deviceId, throughput)) =>
            acc.get(deviceId) match
              case Some(baseline) =>
                val unclamped = ((1.0 - alpha) * baseline) + (alpha * throughput)
                val lower = baseline / DefaultAdaptiveStepFactor
                val upper = baseline * DefaultAdaptiveStepFactor
                val blended = math.max(lower, math.min(upper, unclamped))
                acc.updated(deviceId, blended)
              case None =>
                // First observation switches this device into throughput-calibrated space.
                acc.updated(deviceId, throughput)
          }
        calibratedWeightsRef.set(updated)

  // ------ Device discovery ------------------------------------------------------------------------------------------------------------------------------------

  private lazy val allDevices: Vector[ComputeDevice] = discoverDevices()

  /** Returns all discovered compute devices. */
  def devices: Vector[ComputeDevice] = allDevices

  private def discoverDevices(): Vector[ComputeDevice] =
    ensureNativeLibrariesLoaded()
    val cudaDevices = discoverCudaDevices()
    val openclDevices = discoverOpenCLDevices(cudaDevices)
    val cpuDevice = discoverCpuDevice()
    cudaDevices ++ openclDevices ++ cpuDevice

  /** Attempts to load native libraries so that JNI discovery calls succeed.
    * Silently ignores failures --- discovery methods already handle UnsatisfiedLinkError.
    */
  private def ensureNativeLibrariesLoaded(): Unit =
    // Load CUDA native library (sicfun_gpu_kernel)
    tryLoadNativeLib("sicfun.gpu.native.path", "sicfun_GPU_NATIVE_PATH",
                     "sicfun.gpu.native.lib", "sicfun_GPU_NATIVE_LIB", "sicfun_gpu_kernel")
    // Load OpenCL native library (sicfun_opencl_kernel)
    tryLoadNativeLib("sicfun.opencl.native.path", "sicfun_OPENCL_NATIVE_PATH",
                     "sicfun.opencl.native.lib", "sicfun_OPENCL_NATIVE_LIB", "sicfun_opencl_kernel")

  private def tryLoadNativeLib(
      pathProp: String, pathEnv: String,
      libProp: String, libEnv: String,
      defaultLib: String
  ): Unit =
    try
      GpuRuntimeSupport.loadNativeLibrary(
        pathProperty = pathProp,
        pathEnv = pathEnv,
        libProperty = libProp,
        libEnv = libEnv,
        defaultLib = defaultLib,
        label = s"$defaultLib native library"
      )
      ()
    catch
      case _: Throwable => ()

  private def discoverCudaDevices(): Vector[CudaComputeDevice] =
    try
      val count = HeadsUpGpuNativeBindings.cudaDeviceCount()
      (0 until count).toVector.flatMap { i =>
        val info = HeadsUpGpuNativeBindings.cudaDeviceInfo(i)
        parseCudaDeviceInfo(i, info)
      }
    catch
      case _: UnsatisfiedLinkError => Vector.empty
      case _: Throwable => Vector.empty

  private def discoverOpenCLDevices(cudaDevices: Vector[CudaComputeDevice]): Vector[OpenCLComputeDevice] =
    try
      val count = HeadsUpOpenCLNativeBindings.openclDeviceCount()
      val cudaNames = cudaDevices.map(_.name.toLowerCase(Locale.ROOT)).toSet
      (0 until count).toVector.flatMap { i =>
        val info = HeadsUpOpenCLNativeBindings.openclDeviceInfo(i)
        parseOpenCLDeviceInfo(i, info).filterNot { dev =>
          // Exclude NVIDIA GPUs already covered by CUDA
          val vendor = extractOpenCLVendor(info).toLowerCase(Locale.ROOT)
          vendor.contains("nvidia") || cudaNames.exists(n => dev.name.toLowerCase(Locale.ROOT).contains(n))
        }
      }
    catch
      case _: UnsatisfiedLinkError => Vector.empty
      case _: Throwable => Vector.empty

  private def discoverCpuDevice(): Vector[CpuComputeDevice] =
    try
      HeadsUpGpuNativeBindings.lastEngineCode() // test if native lib is loaded
      Vector(CpuComputeDevice(Runtime.getRuntime.availableProcessors()))
    catch
      case _: UnsatisfiedLinkError => Vector.empty
      case _: Throwable =>
        Vector(CpuComputeDevice(Runtime.getRuntime.availableProcessors()))

  private def parseCudaDeviceInfo(index: Int, info: String): Option[CudaComputeDevice] =
    val parts = info.split("\\|", -1)
    if parts.length >= 4 then
      try
        Some(CudaComputeDevice(
          index = index,
          name = parts(0).trim,
          smCount = parts(1).trim.toInt,
          clockMHz = parts(2).trim.toInt
        ))
      catch case _: NumberFormatException => None
    else None

  private def parseOpenCLDeviceInfo(index: Int, info: String): Option[OpenCLComputeDevice] =
    val parts = info.split("\\|", -1)
    if parts.length >= 3 then
      try
        Some(OpenCLComputeDevice(
          index = index,
          name = parts(0).trim,
          computeUnits = parts(1).trim.toInt,
          clockMHz = parts(2).trim.toInt
        ))
      catch case _: NumberFormatException => None
    else None

  private def extractOpenCLVendor(info: String): String =
    val parts = info.split("\\|", -1)
    if parts.length >= 5 then parts(4).trim else ""

  // ------ Proportional splitting ------------------------------------------------------------------------------------------------------------------

  private def configuredWeightOverrides: Map[String, Double] =
    val raw =
      sys.props
        .get(HybridWeightsProperty)
        .orElse(sys.env.get(HybridWeightsEnv))
        .map(_.trim)
        .filter(_.nonEmpty)
    raw match
      case Some(value) =>
        value
          .split(",")
          .flatMap { entry =>
            entry.split("=", 2) match
              case Array(key, v) =>
                scala.util.Try(v.trim.toDouble).toOption.map(key.trim -> _)
              case _ => None
          }
          .toMap
      case None => Map.empty

  private def configuredMinSliceMatchups: Int =
    val parsed =
      sys.props
        .get(HybridMinSliceMatchupsProperty)
        .orElse(sys.env.get(HybridMinSliceMatchupsEnv))
        .flatMap(raw => scala.util.Try(raw.trim.toInt).toOption)
    parsed.filter(_ > 0).getOrElse(DefaultMinSliceMatchups)

  private def configuredMinRelativeWeight: Double =
    val parsed =
      sys.props
        .get(HybridMinRelativeWeightProperty)
        .orElse(sys.env.get(HybridMinRelativeWeightEnv))
        .flatMap(raw => scala.util.Try(raw.trim.toDouble).toOption)
    parsed
      .filter(value => java.lang.Double.isFinite(value) && value >= 0.0 && value <= 1.0)
      .getOrElse(DefaultMinRelativeWeight)

  private def effectiveWeight(device: ComputeDevice): Double =
    val overrides = configuredWeightOverrides
    val calibrated = calibratedWeightsRef.get()
    val raw =
      overrides.getOrElse(
        device.id,
        calibrated.getOrElse(device.id, device.estimatedWeight)
      )
    sanitizeWeight(raw)

  private def sanitizeWeight(raw: Double): Double =
    if java.lang.Double.isFinite(raw) && raw > 0.0 then raw else 0.0

  /** Chooses how many devices to engage for a batch so each selected device has enough
    * work to amortize dispatch overhead.
    *
    * @param totalItems        number of matchups in the batch
    * @param deviceWeights     (deviceId, effectiveWeight) in preferred/stable order
    * @param minSliceMatchups  target minimum matchups per selected device
    * @return selected device IDs in stable order
    */
  private[holdem] def selectDeviceIdsForBatch(
      totalItems: Int,
      deviceWeights: Vector[(String, Double)],
      minSliceMatchups: Int
  ): Vector[String] =
    if totalItems <= 0 || deviceWeights.isEmpty then
      Vector.empty
    else if deviceWeights.size == 1 then
      Vector(deviceWeights.head._1)
    else
      val normalizedMinSlice = math.max(1, minSliceMatchups)
      val maxDevicesByBatch = math.max(1, totalItems / normalizedMinSlice)
      if maxDevicesByBatch >= deviceWeights.size then
        deviceWeights.map(_._1)
      else
        val ranked = deviceWeights.zipWithIndex.sortBy { case ((_, weight), idx) =>
          (-sanitizeWeight(weight), idx)
        }
        val selectedIndexSet = ranked.take(maxDevicesByBatch).map(_._2).toSet
        deviceWeights.zipWithIndex.collect {
          case ((id, _), idx) if selectedIndexSet.contains(idx) => id
        }

  /** Computes per-device counts that sum exactly to `totalItems`.
    *
    * Uses weighted allocation with largest-remainder correction. If all weights are
    * non-positive, falls back to deterministic near-even split.
    */
  private[holdem] def proportionalCounts(
      totalItems: Int,
      deviceWeights: Vector[(String, Double)]
  ): Vector[(String, Int)] =
    if totalItems <= 0 || deviceWeights.isEmpty then
      Vector.empty
    else if deviceWeights.size == 1 then
      Vector(deviceWeights.head._1 -> totalItems)
    else
      val normalizedWeights = deviceWeights.map((id, w) => id -> sanitizeWeight(w))
      val totalWeight = normalizedWeights.map(_._2).sum
      if totalWeight <= 0.0 then
        val base = totalItems / normalizedWeights.size
        val remainder = totalItems % normalizedWeights.size
        normalizedWeights.zipWithIndex.map { case ((id, _), idx) =>
          val extra = if idx < remainder then 1 else 0
          id -> (base + extra)
        }
      else
        val scaled = normalizedWeights.map((id, w) => id -> (w / totalWeight * totalItems.toDouble))
        val baseCounts = scaled.map((id, raw) => id -> math.floor(raw).toInt)
        val baseTotal = baseCounts.map(_._2).sum
        val need = math.max(0, totalItems - baseTotal)
        val fractionOrder = scaled.zipWithIndex
          .map { case ((_, raw), idx) =>
            val floor = math.floor(raw)
            (raw - floor, idx)
          }
          .sortBy { case (fraction, idx) => (-fraction, idx) }
        val extras = Array.fill(baseCounts.size)(0)
        var i = 0
        while i < need && i < fractionOrder.length do
          val (_, idx) = fractionOrder(i)
          extras(idx) += 1
          i += 1
        baseCounts.zipWithIndex.map { case ((id, base), idx) =>
          id -> (base + extras(idx))
        }

  private def proportionalSplit(
      totalItems: Int,
      activeDevices: Vector[ComputeDevice],
      applyCpuOnlyThreshold: Boolean
  ): Vector[SubBatchSlice] =
    if activeDevices.isEmpty || totalItems <= 0 then
      Vector.empty
    else
      val weightedDevices = activeDevices.map(device => device -> effectiveWeight(device))
      val candidateDevices =
        if weightedDevices.size <= 1 then
          weightedDevices
        else
          val minRelative = configuredMinRelativeWeight
          if minRelative <= 0.0 then
            weightedDevices
          else
            val bestWeight = weightedDevices.iterator.map(_._2).max
            if bestWeight <= 0.0 then weightedDevices
            else
              val cutoff = bestWeight * minRelative
              val kept = weightedDevices.filter { case (_, weight) => sanitizeWeight(weight) >= cutoff }
              if kept.nonEmpty then kept else Vector(weightedDevices.maxBy(_._2))
      val selectedIds =
        if applyCpuOnlyThreshold && configuredCpuOnlyBelow > 0 && totalItems <= configuredCpuOnlyBelow then
          val ranked = candidateDevices.zipWithIndex.sortBy { case ((_, weight), idx) =>
            (-sanitizeWeight(weight), idx)
          }
          ranked.headOption.map(entry => Vector(entry._1._1.id)).getOrElse(Vector.empty)
        else
          selectDeviceIdsForBatch(
            totalItems,
            candidateDevices.map((device, weight) => device.id -> weight),
            configuredMinSliceMatchups
          )
      if selectedIds.isEmpty then
        Vector.empty
      else
        val selectedIdSet = selectedIds.toSet
        val selectedDevices = candidateDevices.map(_._1).filter(device => selectedIdSet.contains(device.id))
        val countsById =
          proportionalCounts(
            totalItems,
            selectedDevices.map(device => device.id -> effectiveWeight(device))
          ).toMap

        var offset = 0
        selectedDevices.flatMap { device =>
          val count = countsById.getOrElse(device.id, 0)
          if count > 0 then
            val slice = SubBatchSlice(device, offset, count)
            offset += count
            Some(slice)
          else None
        }

  // ------ Parallel dispatch ---------------------------------------------------------------------------------------------------------------------------------

  /** Result of a hybrid batch computation. */
  final case class HybridResult(
      results: Array[EquityResultWithError],
      perDevice: Vector[DeviceTelemetry],
      recovery: Vector[RecoveryTelemetry],
      payloadCrc32: Long
  )

  /** Dispatches a batch across all available compute devices in parallel.
    *
    * @param lowIds             hero hole-card pair indices
    * @param highIds            villain hole-card pair indices
    * @param modeCode           0 = exact, 1 = Monte Carlo
    * @param trials             MC trials per matchup
    * @param seeds              per-matchup PRNG seeds
    * @return `Right(result)` on success, `Left(error)` if any device fails
    */
  def dispatchBatch(
      lowIds: Array[Int],
      highIds: Array[Int],
      modeCode: Int,
      trials: Int,
      seeds: Array[Long]
  ): Either[String, HybridResult] =
    val discoveredDevices =
      if modeCode == 0 then allDevices.filter(_.supportsExact)
      else allDevices
    val activeDevices = applyCudaPrimaryPolicy(discoveredDevices)
    maybeWarmUpDevices(modeCode, trials, activeDevices)
    dispatchBatchWithDevices(
      lowIds,
      highIds,
      modeCode,
      trials,
      seeds,
      activeDevices,
      applyCpuOnlyThreshold = true
    ).map { result =>
      maybeUpdateCalibratedWeights(result.perDevice)
      result
    }

  private[holdem] def dispatchBatchWithDevices(
      lowIds: Array[Int],
      highIds: Array[Int],
      modeCode: Int,
      trials: Int,
      seeds: Array[Long],
      activeDevices: Vector[ComputeDevice],
      applyCpuOnlyThreshold: Boolean = false
  ): Either[String, HybridResult] =
    val n = lowIds.length
    require(highIds.length == n, "highIds must have same length as lowIds")
    require(seeds.length == n, "seeds must have same length as lowIds")
    val checksum = payloadCrc32(lowIds, highIds, modeCode, trials, seeds)
    if n == 0 then
      Right(HybridResult(Array.empty, Vector.empty, Vector.empty, checksum))
    else if activeDevices.isEmpty then
      Left("no compute devices available for hybrid dispatch")
    else
      val splits = proportionalSplit(n, activeDevices, applyCpuOnlyThreshold)
      if splits.isEmpty then
        Left("batch splitting produced no slices")
      else
        val wins = new Array[Double](n)
        val ties = new Array[Double](n)
        val losses = new Array[Double](n)
        val stderrs = new Array[Double](n)
        val telemetryBuilder = Vector.newBuilder[DeviceTelemetry]
        val recoveryBuilder = Vector.newBuilder[RecoveryTelemetry]

        if splits.size == 1 then
          val slice = splits.head
          val firstAttempt =
            runSlice(slice, lowIds, highIds, modeCode, trials, seeds, wins, ties, losses, stderrs)
          if firstAttempt.status == 0 then
            telemetryBuilder +=
              DeviceTelemetry(
                slice.device.id,
                slice.device.name,
                slice.count,
                firstAttempt.elapsedMs,
                firstAttempt.elapsedNanos
              )
            Right(HybridResult(
              buildResultArray(wins, ties, losses, stderrs),
              telemetryBuilder.result(),
              recoveryBuilder.result(),
              checksum
            ))
          else
            attemptRecovery(
              firstAttempt,
              activeDevices,
              lowIds,
              highIds,
              modeCode,
              trials,
              seeds,
              wins,
              ties,
              losses,
              stderrs
            ) match
              case Left(error) => Left(error)
              case Right((telemetry, recovery)) =>
                telemetryBuilder += telemetry
                recoveryBuilder += recovery
                Right(HybridResult(
                  buildResultArray(wins, ties, losses, stderrs),
                  telemetryBuilder.result(),
                  recoveryBuilder.result(),
                  checksum
                ))
        else
          val futures: Vector[Future[SliceAttempt]] = splits.map { slice =>
            dispatchPool.submit(new Callable[SliceAttempt] {
              override def call(): SliceAttempt =
                runSlice(slice, lowIds, highIds, modeCode, trials, seeds, wins, ties, losses, stderrs)
            })
          }

          val attempts = futures.zip(splits).map { case (future, slice) =>
            try future.get()
            catch
              case ex: Throwable =>
                SliceAttempt(
                  slice = slice,
                  status = Int.MinValue,
                  elapsedMs = 0L,
                  elapsedNanos = 0L,
                  error = Some(
                    Option(ex.getMessage).filter(_.nonEmpty).getOrElse(ex.getClass.getSimpleName)
                  )
                )
          }

          var firstError: Option[String] = None
          attempts.foreach { attempt =>
            if firstError.isEmpty then
              if attempt.status == 0 then
                telemetryBuilder += DeviceTelemetry(
                  attempt.slice.device.id,
                  attempt.slice.device.name,
                  attempt.slice.count,
                  attempt.elapsedMs,
                  attempt.elapsedNanos
                )
              else
                attemptRecovery(
                  attempt,
                  activeDevices,
                  lowIds,
                  highIds,
                  modeCode,
                  trials,
                  seeds,
                  wins,
                  ties,
                  losses,
                  stderrs
                ) match
                  case Left(error) =>
                    firstError = Some(error)
                  case Right((telemetry, recovery)) =>
                    telemetryBuilder += telemetry
                    recoveryBuilder += recovery
          }

          firstError match
            case Some(error) => Left(error)
            case None =>
              Right(HybridResult(
                buildResultArray(wins, ties, losses, stderrs),
                telemetryBuilder.result(),
                recoveryBuilder.result(),
                checksum
              ))

  private def deviceRecoveryPriority(device: ComputeDevice): Int =
    device.kind match
      case "cpu" => 0
      case "opencl" => 1
      case "cuda" => 2
      case _ => 3

  private def rescueCandidates(
      failedDevice: ComputeDevice,
      activeDevices: Vector[ComputeDevice]
  ): Vector[ComputeDevice] =
    activeDevices
      .filterNot(_.id == failedDevice.id)
      .sortBy(device => (deviceRecoveryPriority(device), -effectiveWeight(device), device.id))

  private def errorDetail(attempt: SliceAttempt): String =
    attempt.error match
      case Some(message) if message.nonEmpty =>
        if attempt.status == Int.MinValue then s"failed: $message"
        else s"returned status ${attempt.status}: $message"
      case _ =>
        if attempt.status == Int.MinValue then "failed with unknown error"
        else s"returned status ${attempt.status}"

  private def runSlice(
      slice: SubBatchSlice,
      lowIds: Array[Int],
      highIds: Array[Int],
      modeCode: Int,
      trials: Int,
      seeds: Array[Long],
      wins: Array[Double],
      ties: Array[Double],
      losses: Array[Double],
      stderrs: Array[Double]
  ): SliceAttempt =
    val startedAt = System.nanoTime()
    try
      val status =
        if slice.startIdx == 0 && slice.count == lowIds.length then
          slice.device.computeSubBatch(
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
        else
          val subLow = java.util.Arrays.copyOfRange(lowIds, slice.startIdx, slice.startIdx + slice.count)
          val subHigh = java.util.Arrays.copyOfRange(highIds, slice.startIdx, slice.startIdx + slice.count)
          val subSeeds = java.util.Arrays.copyOfRange(seeds, slice.startIdx, slice.startIdx + slice.count)
          val subWins = new Array[Double](slice.count)
          val subTies = new Array[Double](slice.count)
          val subLosses = new Array[Double](slice.count)
          val subStderrs = new Array[Double](slice.count)
          val subStatus = slice.device.computeSubBatch(
            subLow,
            subHigh,
            modeCode,
            trials,
            subSeeds,
            subWins,
            subTies,
            subLosses,
            subStderrs
          )
          if subStatus == 0 then
            System.arraycopy(subWins, 0, wins, slice.startIdx, slice.count)
            System.arraycopy(subTies, 0, ties, slice.startIdx, slice.count)
            System.arraycopy(subLosses, 0, losses, slice.startIdx, slice.count)
            System.arraycopy(subStderrs, 0, stderrs, slice.startIdx, slice.count)
          subStatus
      val elapsedNanos = System.nanoTime() - startedAt
      val elapsedMs = elapsedNanos / 1_000_000L
      SliceAttempt(slice, status, elapsedMs, elapsedNanos, None)
    catch
      case ex: Throwable =>
        val elapsedNanos = System.nanoTime() - startedAt
        val elapsedMs = elapsedNanos / 1_000_000L
        SliceAttempt(
          slice,
          Int.MinValue,
          elapsedMs,
          elapsedNanos,
          Some(Option(ex.getMessage).filter(_.nonEmpty).getOrElse(ex.getClass.getSimpleName))
        )

  private def attemptRecovery(
      failedAttempt: SliceAttempt,
      activeDevices: Vector[ComputeDevice],
      lowIds: Array[Int],
      highIds: Array[Int],
      modeCode: Int,
      trials: Int,
      seeds: Array[Long],
      wins: Array[Double],
      ties: Array[Double],
      losses: Array[Double],
      stderrs: Array[Double]
  ): Either[String, (DeviceTelemetry, RecoveryTelemetry)] =
    import scala.util.boundary, boundary.break
    val failedSlice = failedAttempt.slice
    val candidates = rescueCandidates(failedSlice.device, activeDevices)
    if candidates.isEmpty then
      Left(
        s"device ${failedSlice.device.id} ${errorDetail(failedAttempt)} and no rescue candidates are available"
      )
    else
      val rescueFailures = Vector.newBuilder[String]
      boundary:
        var idx = 0
        while idx < candidates.length do
          val candidate = candidates(idx)
          val rescueAttempt = runSlice(
            failedSlice.copy(device = candidate),
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
          if rescueAttempt.status == 0 then
            val rescueTelemetry =
              DeviceTelemetry(
                candidate.id,
                candidate.name,
                failedSlice.count,
                rescueAttempt.elapsedMs,
                rescueAttempt.elapsedNanos
              )
            val recovery = RecoveryTelemetry(
              failedDeviceId = failedSlice.device.id,
              failedStatus = failedAttempt.status,
              recoveredByDeviceId = candidate.id,
              matchups = failedSlice.count,
              failedElapsedMs = failedAttempt.elapsedMs,
              recoveredElapsedMs = rescueAttempt.elapsedMs
            )
            break(Right((rescueTelemetry, recovery)))
          rescueFailures += s"${candidate.id} ${errorDetail(rescueAttempt)}"
          idx += 1
        Left(
          s"device ${failedSlice.device.id} ${errorDetail(failedAttempt)}; recovery failed: ${rescueFailures.result().mkString("; ")}"
        )

  private[holdem] def payloadCrc32(
      lowIds: Array[Int],
      highIds: Array[Int],
      modeCode: Int,
      trials: Int,
      seeds: Array[Long]
  ): Long =
    val crc = new CRC32()
    updateCrcInt(crc, lowIds.length)
    updateCrcInt(crc, modeCode)
    updateCrcInt(crc, trials)
    var i = 0
    while i < lowIds.length do
      updateCrcInt(crc, lowIds(i))
      updateCrcInt(crc, highIds(i))
      updateCrcLong(crc, seeds(i))
      i += 1
    crc.getValue

  private def updateCrcInt(crc: CRC32, value: Int): Unit =
    crc.update(value & 0xff)
    crc.update((value >>> 8) & 0xff)
    crc.update((value >>> 16) & 0xff)
    crc.update((value >>> 24) & 0xff)

  private def updateCrcLong(crc: CRC32, value: Long): Unit =
    updateCrcInt(crc, (value & 0xffffffffL).toInt)
    updateCrcInt(crc, ((value >>> 32) & 0xffffffffL).toInt)
  private def buildResultArray(
      wins: Array[Double],
      ties: Array[Double],
      losses: Array[Double],
      stderrs: Array[Double]
  ): Array[EquityResultWithError] =
    val n = wins.length
    val out = new Array[EquityResultWithError](n)
    var i = 0
    while i < n do
      out(i) = EquityResultWithError(wins(i), ties(i), losses(i), stderrs(i))
      i += 1
    out
