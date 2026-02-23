package sicfun.holdem

/** Quick benchmark to test hybrid multi-device dispatch.
  *
  * Exercises the HybridProvider (CUDA + OpenCL + CPU in parallel) and compares
  * throughput against single-device runs.
  */
object HybridBenchmark:

  def main(args: Array[String]): Unit =
    val maxMatchups = args.headOption.flatMap(s => scala.util.Try(s.toLong).toOption).getOrElse(4000L)
    val trials = 200
    val seed = 42L
    val mode = HeadsUpEquityTable.Mode.MonteCarlo(trials)

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
    println(s"Batch: $n canonical matchups, mode=MC($trials), seed=$seed")
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

    // --- Single-device CUDA ---
    print("CUDA-only:    ")
    val cudaDevices = devices.filter(_.kind == "cuda")
    if cudaDevices.nonEmpty then
      val t0 = System.nanoTime()
      val cudaResult = cudaDevices.head.computeSubBatch(
        lowIds, highIds, 1, trials, seeds,
        new Array[Double](n), new Array[Double](n), new Array[Double](n), new Array[Double](n)
      )
      val cudaMs = (System.nanoTime() - t0) / 1_000_000L
      println(f"${cudaMs}ms  (${n.toDouble / (cudaMs.toDouble / 1000.0)}%.0f matchups/s)  status=$cudaResult")
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

    // --- Single-device CPU ---
    print("CPU-only:     ")
    val cpuDevices = devices.filter(_.kind == "cpu")
    if cpuDevices.nonEmpty then
      val t0 = System.nanoTime()
      val cpuResult = cpuDevices.head.computeSubBatch(
        lowIds, highIds, 1, trials, seeds,
        new Array[Double](n), new Array[Double](n), new Array[Double](n), new Array[Double](n)
      )
      val cpuMs = (System.nanoTime() - t0) / 1_000_000L
      println(f"${cpuMs}ms  (${n.toDouble / (cpuMs.toDouble / 1000.0)}%.0f matchups/s)  status=$cpuResult")
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
