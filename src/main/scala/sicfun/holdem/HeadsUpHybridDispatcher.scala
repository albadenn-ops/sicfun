package sicfun.holdem

import java.io.File
import java.util.concurrent.{Callable, Executors, Future}
import java.util.concurrent.atomic.AtomicReference
import java.util.zip.CRC32

/** Multi-device hybrid dispatcher that distributes batch equity computation
  * across all available compute devices: CUDA GPUs, OpenCL iGPUs, and CPU threads.
  *
  * Work is split proportionally to each device's estimated (or calibrated) throughput,
  * dispatched in parallel, and results are merged back into the original array order.
  *
  * '''Determinism:''' Seeds are pre-computed per matchup and independent of the
  * processing device, so results are identical regardless of how the batch is split.
  */
object HeadsUpHybridDispatcher:
  private val HybridWeightsProperty = "sicfun.hybrid.weights"
  private val HybridWeightsEnv = "sicfun_HYBRID_WEIGHTS"
  private val HybridMinSliceMatchupsProperty = "sicfun.hybrid.minSliceMatchups"
  private val HybridMinSliceMatchupsEnv = "sicfun_HYBRID_MIN_SLICE_MATCHUPS"
  private val DefaultMinSliceMatchups = 64


  // â”€â”€ Device abstraction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

  // â”€â”€ Per-device telemetry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

  final case class DeviceTelemetry(
      deviceId: String,
      deviceName: String,
      matchups: Int,
      elapsedMs: Long
  ):
    def throughput: Double =
      if elapsedMs > 0 then matchups.toDouble / (elapsedMs.toDouble / 1000.0) else 0.0

  final case class RecoveryTelemetry(
      failedDeviceId: String,
      failedStatus: Int,
      recoveredByDeviceId: String,
      matchups: Int,
      failedElapsedMs: Long,
      recoveredElapsedMs: Long
  )

  // â”€â”€ Batch split descriptor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

  private final case class SubBatchSlice(
      device: ComputeDevice,
      startIdx: Int,
      count: Int
  )

  private final case class SliceAttempt(
      slice: SubBatchSlice,
      status: Int,
      elapsedMs: Long,
      error: Option[String]
  )

  // â”€â”€ Calibrated weights (updated by auto-tuner) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

  private val calibratedWeightsRef =
    new AtomicReference[Map[String, Double]](Map.empty)

  def setCalibratedWeights(weights: Map[String, Double]): Unit =
    calibratedWeightsRef.set(weights)

  def calibratedWeights: Map[String, Double] =
    calibratedWeightsRef.get()

  // â”€â”€ Device discovery â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    * Silently ignores failures â€” discovery methods already handle UnsatisfiedLinkError.
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
      // Already loaded? Quick check via a known symbol.
      val _ = Class.forName("sicfun.holdem.HeadsUpGpuNativeBindings")
    catch case _: Throwable => ()
    val pathOpt = sys.props.get(pathProp).orElse(sys.env.get(pathEnv)).map(_.trim).filter(_.nonEmpty)
    val libName = sys.props.get(libProp).orElse(sys.env.get(libEnv)).map(_.trim).filter(_.nonEmpty).getOrElse(defaultLib)
    pathOpt match
      case Some(path) =>
        try System.load(path)
        catch case _: Throwable => ()
      case None =>
        try System.loadLibrary(libName)
        catch
          case _: Throwable =>
            // Try local build directory fallback
            val buildDir = new File(System.getProperty("user.dir", "."), "src/main/native/build")
            val osName = System.getProperty("os.name", "").toLowerCase
            val fileName =
              if osName.contains("win") then s"$libName.dll"
              else if osName.contains("mac") then s"lib$libName.dylib"
              else s"lib$libName.so"
            val candidate = new File(buildDir, fileName)
            if candidate.isFile then
              try System.load(candidate.getAbsolutePath)
              catch case _: Throwable => ()

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
      val cudaNames = cudaDevices.map(_.name.toLowerCase).toSet
      (0 until count).toVector.flatMap { i =>
        val info = HeadsUpOpenCLNativeBindings.openclDeviceInfo(i)
        parseOpenCLDeviceInfo(i, info).filterNot { dev =>
          // Exclude NVIDIA GPUs already covered by CUDA
          val vendor = extractOpenCLVendor(info).toLowerCase
          vendor.contains("nvidia") || cudaNames.exists(n => dev.name.toLowerCase.contains(n))
        }
      }
    catch
      case _: UnsatisfiedLinkError => Vector.empty
      case _: Throwable => Vector.empty

  private def discoverCpuDevice(): Vector[CpuComputeDevice] =
    try
      val native = HeadsUpGpuNativeBindings.lastEngineCode() // test if native lib is loaded
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

  // â”€â”€ Proportional splitting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
      activeDevices: Vector[ComputeDevice]
  ): Vector[SubBatchSlice] =
    if activeDevices.isEmpty || totalItems <= 0 then
      return Vector.empty
    val weightedDevices = activeDevices.map(device => device -> effectiveWeight(device))
    val selectedIds = selectDeviceIdsForBatch(
      totalItems,
      weightedDevices.map((device, weight) => device.id -> weight),
      configuredMinSliceMatchups
    )
    if selectedIds.isEmpty then
      return Vector.empty

    val selectedIdSet = selectedIds.toSet
    val selectedDevices = activeDevices.filter(device => selectedIdSet.contains(device.id))
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

  // â”€â”€ Parallel dispatch â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    val activeDevices =
      if modeCode == 0 then allDevices.filter(_.supportsExact)
      else allDevices
    dispatchBatchWithDevices(lowIds, highIds, modeCode, trials, seeds, activeDevices)

  private[holdem] def dispatchBatchWithDevices(
      lowIds: Array[Int],
      highIds: Array[Int],
      modeCode: Int,
      trials: Int,
      seeds: Array[Long],
      activeDevices: Vector[ComputeDevice]
  ): Either[String, HybridResult] =
    val n = lowIds.length
    require(highIds.length == n, "highIds must have same length as lowIds")
    require(seeds.length == n, "seeds must have same length as lowIds")
    val checksum = payloadCrc32(lowIds, highIds, modeCode, trials, seeds)
    if n == 0 then
      return Right(HybridResult(Array.empty, Vector.empty, Vector.empty, checksum))

    if activeDevices.isEmpty then
      return Left("no compute devices available for hybrid dispatch")

    val splits = proportionalSplit(n, activeDevices)
    if splits.isEmpty then
      return Left("batch splitting produced no slices")

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
          DeviceTelemetry(slice.device.id, slice.device.name, slice.count, firstAttempt.elapsedMs)
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
          case Left(error) => return Left(error)
          case Right((telemetry, recovery)) =>
            telemetryBuilder += telemetry
            recoveryBuilder += recovery
      return Right(HybridResult(
        buildResultArray(wins, ties, losses, stderrs),
        telemetryBuilder.result(),
        recoveryBuilder.result(),
        checksum
      ))

    val pool = Executors.newFixedThreadPool(splits.size)
    try
      val futures: Vector[Future[SliceAttempt]] = splits.map { slice =>
        pool.submit(new Callable[SliceAttempt] {
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
              attempt.elapsedMs
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
    finally
      pool.shutdown()

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
    val subLow = java.util.Arrays.copyOfRange(lowIds, slice.startIdx, slice.startIdx + slice.count)
    val subHigh = java.util.Arrays.copyOfRange(highIds, slice.startIdx, slice.startIdx + slice.count)
    val subSeeds = java.util.Arrays.copyOfRange(seeds, slice.startIdx, slice.startIdx + slice.count)
    val subWins = new Array[Double](slice.count)
    val subTies = new Array[Double](slice.count)
    val subLosses = new Array[Double](slice.count)
    val subStderrs = new Array[Double](slice.count)
    val startedAt = System.nanoTime()
    try
      val status = slice.device.computeSubBatch(
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
      val elapsedMs = (System.nanoTime() - startedAt) / 1_000_000L
      if status == 0 then
        System.arraycopy(subWins, 0, wins, slice.startIdx, slice.count)
        System.arraycopy(subTies, 0, ties, slice.startIdx, slice.count)
        System.arraycopy(subLosses, 0, losses, slice.startIdx, slice.count)
        System.arraycopy(subStderrs, 0, stderrs, slice.startIdx, slice.count)
      SliceAttempt(slice, status, elapsedMs, None)
    catch
      case ex: Throwable =>
        val elapsedMs = (System.nanoTime() - startedAt) / 1_000_000L
        SliceAttempt(
          slice,
          Int.MinValue,
          elapsedMs,
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
    val failedSlice = failedAttempt.slice
    val candidates = rescueCandidates(failedSlice.device, activeDevices)
    if candidates.isEmpty then
      Left(
        s"device ${failedSlice.device.id} ${errorDetail(failedAttempt)} and no rescue candidates are available"
      )
    else
      val rescueFailures = Vector.newBuilder[String]
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
            DeviceTelemetry(candidate.id, candidate.name, failedSlice.count, rescueAttempt.elapsedMs)
          val recovery = RecoveryTelemetry(
            failedDeviceId = failedSlice.device.id,
            failedStatus = failedAttempt.status,
            recoveredByDeviceId = candidate.id,
            matchups = failedSlice.count,
            failedElapsedMs = failedAttempt.elapsedMs,
            recoveredElapsedMs = rescueAttempt.elapsedMs
          )
          return Right((rescueTelemetry, recovery))
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
