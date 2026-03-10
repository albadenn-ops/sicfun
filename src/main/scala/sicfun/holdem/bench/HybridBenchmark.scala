package sicfun.holdem.bench
import sicfun.holdem.*
import sicfun.holdem.equity.*
import sicfun.holdem.gpu.*

/** Quick benchmark to test hybrid multi-device dispatch.
  *
  * Exercises the HybridProvider (CUDA + OpenCL + CPU in parallel) and compares
  * throughput against single-device runs.
  */
object HybridBenchmark:

  private final case class DeviceRun(elapsedMs: Long, throughput: Double):
    def elapsedSeconds: Double = elapsedMs.toDouble / 1000.0

  private def runSingleDevice(
      device: HeadsUpHybridDispatcher.ComputeDevice,
      lowIds: Array[Int],
      highIds: Array[Int],
      trials: Int,
      seeds: Array[Long]
  ): Either[Int, DeviceRun] =
    val n = lowIds.length
    val t0 = System.nanoTime()
    val status = device.computeSubBatch(
      lowIds,
      highIds,
      1,
      trials,
      seeds,
      new Array[Double](n),
      new Array[Double](n),
      new Array[Double](n),
      new Array[Double](n)
    )
    val elapsedMs = (System.nanoTime() - t0) / 1_000_000L
    if status != 0 then Left(status)
    else
      val throughput =
        if elapsedMs > 0L then n.toDouble / (elapsedMs.toDouble / 1000.0) else 0.0
      Right(DeviceRun(elapsedMs, throughput))

  def main(args: Array[String]): Unit =
    val maxMatchups = args.headOption.flatMap(s => scala.util.Try(s.toLong).toOption).getOrElse(4000L)
    val trials = args.lift(1).flatMap(s => scala.util.Try(s.toInt).toOption).getOrElse(200)
    val seed = 42L

    println("=== Hybrid Multi-Device Benchmark ===")
    println()

    // Discover devices
    val devices = HeadsUpHybridDispatcher.devices
    println(s"Discovered ${devices.size} compute device(s):")
    devices.foreach { dev =>
      println(f"  ${dev.id}%-12s  ${dev.name}%-40s  weight=${dev.estimatedWeight}%.0f  exact=${dev.supportsExact}")
    }
    println()

    // Load batch
    val batch = HeadsUpEquityCanonicalTable.selectCanonicalBatch(maxMatchups)
    val n = batch.packedKeys.length
    val statesPerRun = n.toDouble * trials.toDouble
    println(s"Batch: $n canonical matchups, mode=MC($trials), seed=$seed")
    println(f"Per run states: $statesPerRun%.0f")
    println()

    // Prepare inputs
    val lowIds = new Array[Int](n)
    val highIds = new Array[Int](n)
    val seeds = new Array[Long](n)
    var idx = 0
    while idx < n do
      val packed = batch.packedKeys(idx)
      lowIds(idx) = HeadsUpEquityTable.unpackLowId(packed)
      highIds(idx) = HeadsUpEquityTable.unpackHighId(packed)
      seeds(idx) = HeadsUpEquityTable.monteCarloSeed(seed, batch.keyMaterial(idx))
      idx += 1

    // --- Single-device CUDA (cold + warm) ---
    print("CUDA-only:    ")
    val cudaDevices = devices.filter(_.kind == "cuda")
    if cudaDevices.nonEmpty then
      val cuda = cudaDevices.head
      val cold = runSingleDevice(cuda, lowIds, highIds, trials, seeds)
      val warm = runSingleDevice(cuda, lowIds, highIds, trials, seeds)
      (cold, warm) match
        case (Right(coldRun), Right(warmRun)) =>
          val coldStatesPerSecond = if coldRun.elapsedSeconds > 0 then statesPerRun / coldRun.elapsedSeconds else 0.0
          val warmStatesPerSecond = if warmRun.elapsedSeconds > 0 then statesPerRun / warmRun.elapsedSeconds else 0.0
          println(
            f"cold=${coldRun.elapsedMs}ms (${coldRun.throughput}%.0f matchups/s, $coldStatesPerSecond%.0f states/s)  " +
              f"warm=${warmRun.elapsedMs}ms (${warmRun.throughput}%.0f matchups/s, $warmStatesPerSecond%.0f states/s)"
          )
        case (Left(status), _) =>
          println(s"cold-status=$status")
        case (_, Left(status)) =>
          println(s"warm-status=$status")
    else
      println("(not available)")

    // --- Single-device OpenCL ---
    print("OpenCL-only:  ")
    val openclDevices = devices.filter(_.kind == "opencl")
    if openclDevices.nonEmpty then
      val t0 = System.nanoTime()
      val oclResult = openclDevices.head.computeSubBatch(
        lowIds, highIds, 1, trials, seeds,
        new Array[Double](n), new Array[Double](n), new Array[Double](n), new Array[Double](n)
      )
      val oclMs = (System.nanoTime() - t0) / 1_000_000L
      println(f"${oclMs}ms  (${n.toDouble / (oclMs.toDouble / 1000.0)}%.0f matchups/s)  status=$oclResult")
    else
      println("(not available)")

    // --- Single-device CPU (cold + warm) ---
    print("CPU-only:     ")
    val cpuDevices = devices.filter(_.kind == "cpu")
    if cpuDevices.nonEmpty then
      val cpu = cpuDevices.head
      val cold = runSingleDevice(cpu, lowIds, highIds, trials, seeds)
      val warm = runSingleDevice(cpu, lowIds, highIds, trials, seeds)
      (cold, warm) match
        case (Right(coldRun), Right(warmRun)) =>
          val coldStatesPerSecond = if coldRun.elapsedSeconds > 0 then statesPerRun / coldRun.elapsedSeconds else 0.0
          val warmStatesPerSecond = if warmRun.elapsedSeconds > 0 then statesPerRun / warmRun.elapsedSeconds else 0.0
          println(
            f"cold=${coldRun.elapsedMs}ms (${coldRun.throughput}%.0f matchups/s, $coldStatesPerSecond%.0f states/s)  " +
              f"warm=${warmRun.elapsedMs}ms (${warmRun.throughput}%.0f matchups/s, $warmStatesPerSecond%.0f states/s)"
          )
        case (Left(status), _) =>
          println(s"cold-status=$status")
        case (_, Left(status)) =>
          println(s"warm-status=$status")
    else
      println("(not available)")

    println()

    // --- Hybrid dispatch (all devices) ---
    println("Hybrid (all devices in parallel):")
    val t0 = System.nanoTime()
    val hybridResult = HeadsUpHybridDispatcher.dispatchBatch(lowIds, highIds, 1, trials, seeds)
    val hybridMs = (System.nanoTime() - t0) / 1_000_000L

    hybridResult match
      case Left(error) =>
        println(s"  FAILED: $error")
      case Right(result) =>
        println(f"  Total: ${hybridMs}ms  (${n.toDouble / (hybridMs.toDouble / 1000.0)}%.0f matchups/s)")
        println(s"  Per-device breakdown:")
        result.perDevice.foreach { t =>
          println(f"    ${t.deviceId}%-12s  ${t.deviceName}%-40s  ${t.matchups}%5d matchups  ${t.elapsedMs}%5dms  ${t.throughput}%.0f/s")
        }

    // --- Second hybrid run (warmed up) ---
    println()
    println("Hybrid run 2 (warmed up):")
    val t1 = System.nanoTime()
    val hybridResult2 = HeadsUpHybridDispatcher.dispatchBatch(lowIds, highIds, 1, trials, seeds)
    val hybridMs2 = (System.nanoTime() - t1) / 1_000_000L
    hybridResult2 match
      case Left(error) =>
        println(s"  FAILED: $error")
      case Right(result) =>
        println(f"  Total: ${hybridMs2}ms  (${n.toDouble / (hybridMs2.toDouble / 1000.0)}%.0f matchups/s)")
        println(s"  Per-device breakdown:")
        result.perDevice.foreach { t =>
          println(f"    ${t.deviceId}%-12s  ${t.deviceName}%-40s  ${t.matchups}%5d matchups  ${t.elapsedMs}%5dms  ${t.throughput}%.0f/s")
        }

    println()
    println("=== Done ===")
