package sicfun.holdem

import munit.FunSuite

class HeadsUpHybridDispatcherPlanningTest extends FunSuite:
  override def beforeEach(context: BeforeEach): Unit =
    HeadsUpHybridDispatcher.setCalibratedWeights(Map.empty)
    sys.props.remove("sicfun.hybrid.minRelativeWeight")
    super.beforeEach(context)

  test("selectDeviceIdsForBatch keeps all devices when batch is large enough") {
    val ids = HeadsUpHybridDispatcher.selectDeviceIdsForBatch(
      totalItems = 512,
      deviceWeights = Vector(
        "cuda:0" -> 10.0,
        "opencl:0" -> 4.0,
        "cpu" -> 2.0
      ),
      minSliceMatchups = 64
    )
    assertEquals(ids, Vector("cuda:0", "opencl:0", "cpu"))
  }

  test("selectDeviceIdsForBatch trims to top weighted devices for small batches") {
    val ids = HeadsUpHybridDispatcher.selectDeviceIdsForBatch(
      totalItems = 140,
      deviceWeights = Vector(
        "cpu" -> 2.0,
        "cuda:0" -> 10.0,
        "opencl:0" -> 4.0
      ),
      minSliceMatchups = 64
    )
    assertEquals(ids, Vector("cuda:0", "opencl:0"))
  }

  test("selectDeviceIdsForBatch keeps one device for tiny batches") {
    val ids = HeadsUpHybridDispatcher.selectDeviceIdsForBatch(
      totalItems = 32,
      deviceWeights = Vector(
        "cpu" -> 2.0,
        "cuda:0" -> 10.0,
        "opencl:0" -> 4.0
      ),
      minSliceMatchups = 64
    )
    assertEquals(ids, Vector("cuda:0"))
  }

  test("proportionalCounts matches weighted ratio and totals exactly") {
    val counts = HeadsUpHybridDispatcher.proportionalCounts(
      totalItems = 100,
      deviceWeights = Vector(
        "a" -> 5.0,
        "b" -> 3.0,
        "c" -> 2.0
      )
    )
    assertEquals(counts, Vector("a" -> 50, "b" -> 30, "c" -> 20))
    assertEquals(counts.map(_._2).sum, 100)
  }

  test("proportionalCounts falls back to deterministic near-even split") {
    val counts = HeadsUpHybridDispatcher.proportionalCounts(
      totalItems = 10,
      deviceWeights = Vector(
        "a" -> 0.0,
        "b" -> -1.0,
        "c" -> Double.NaN
      )
    )
    assertEquals(counts, Vector("a" -> 4, "b" -> 3, "c" -> 3))
    assertEquals(counts.map(_._2).sum, 10)
  }

  test("proportionalCounts handles empty and zero-sized batches") {
    assertEquals(
      HeadsUpHybridDispatcher.proportionalCounts(0, Vector("a" -> 1.0, "b" -> 1.0)),
      Vector.empty
    )
    assertEquals(
      HeadsUpHybridDispatcher.proportionalCounts(10, Vector.empty),
      Vector.empty
    )
  }

  test("dispatchBatchWithDevices drops low relative-weight devices after calibration") {
    val minRelativeProp = "sicfun.hybrid.minRelativeWeight"
    val previousMinRelative = sys.props.get(minRelativeProp)
    val previousWeights = HeadsUpHybridDispatcher.calibratedWeights
    try
      sys.props.update(minRelativeProp, "0.75")
      HeadsUpHybridDispatcher.setCalibratedWeights(
        Map(
          "cpu" -> 1000.0,
          "gpu" -> 600.0
        )
      )

      val n = 256
      val lowIds = Array.tabulate(n)(i => i + 1)
      val highIds = Array.tabulate(n)(i => i + 101)
      val seeds = Array.tabulate(n)(i => i.toLong + 500L)

      val gpu = HeadsUpHybridDispatcher.FunctionalComputeDevice(
        id = "gpu",
        kind = "cuda",
        name = "Filtered GPU",
        supportsExact = true,
        estimatedWeight = 10_000.0,
        computeFn = (_, _, _, _, _, _, _, _, _) =>
          fail("GPU should be filtered out by minRelativeWeight cutoff")
          0
      )

      val cpu = HeadsUpHybridDispatcher.FunctionalComputeDevice(
        id = "cpu",
        kind = "cpu",
        name = "CPU",
        supportsExact = true,
        estimatedWeight = 1.0,
        computeFn = (_, _, _, _, _, wins, ties, losses, stderrs) =>
          java.util.Arrays.fill(wins, 0.5)
          java.util.Arrays.fill(ties, 0.0)
          java.util.Arrays.fill(losses, 0.5)
          java.util.Arrays.fill(stderrs, 0.01)
          0
      )

      val result = HeadsUpHybridDispatcher.dispatchBatchWithDevices(
        lowIds = lowIds,
        highIds = highIds,
        modeCode = 1,
        trials = 64,
        seeds = seeds,
        activeDevices = Vector(gpu, cpu)
      )
      assert(result.isRight, clues(result))
      val perDevice = result.toOption.get.perDevice.map(_.deviceId)
      assertEquals(perDevice, Vector("cpu"))
    finally
      previousMinRelative match
        case Some(value) => sys.props.update(minRelativeProp, value)
        case None => sys.props.remove(minRelativeProp)
      HeadsUpHybridDispatcher.setCalibratedWeights(previousWeights)
  }

  test("dispatchBatch validates aligned array lengths") {
    intercept[IllegalArgumentException] {
      HeadsUpHybridDispatcher.dispatchBatch(
        lowIds = Array(1, 2),
        highIds = Array(1),
        modeCode = 1,
        trials = 10,
        seeds = Array(1L, 2L)
      )
    }
    intercept[IllegalArgumentException] {
      HeadsUpHybridDispatcher.dispatchBatch(
        lowIds = Array(1, 2),
        highIds = Array(3, 4),
        modeCode = 1,
        trials = 10,
        seeds = Array(1L)
      )
    }
  }

  test("dispatchBatchWithDevices rescues failed slice on fallback device") {
    val n = 12
    val lowIds = Array.tabulate(n)(i => i + 1)
    val highIds = Array.tabulate(n)(i => i + 101)
    val seeds = Array.tabulate(n)(i => 1000L + i.toLong)

    val gpuFail = HeadsUpHybridDispatcher.FunctionalComputeDevice(
      id = "gpu",
      kind = "cuda",
      name = "Failing GPU",
      supportsExact = true,
      estimatedWeight = 100.0,
      computeFn = (_, _, _, _, _, _, _, _, _) => 77
    )

    val cpuRescue = HeadsUpHybridDispatcher.FunctionalComputeDevice(
      id = "cpu",
      kind = "cpu",
      name = "Rescue CPU",
      supportsExact = true,
      estimatedWeight = 5.0,
      computeFn = (low, high, modeCode, trials, inSeeds, wins, ties, losses, stderrs) =>
        var i = 0
        while i < low.length do
          wins(i) = low(i).toDouble / 100.0
          ties(i) = high(i).toDouble / 1000.0
          losses(i) = inSeeds(i).toDouble / 10000.0
          stderrs(i) = (modeCode + trials).toDouble
          i += 1
        0
    )

    val resultEither = HeadsUpHybridDispatcher.dispatchBatchWithDevices(
      lowIds,
      highIds,
      modeCode = 1,
      trials = 25,
      seeds = seeds,
      activeDevices = Vector(gpuFail, cpuRescue)
    )

    val result = resultEither match
      case Left(error) => fail(error)
      case Right(value) => value

    assertEquals(result.results.length, n)
    assertEquals(result.perDevice.map(_.deviceId), Vector("cpu"))
    assertEquals(result.perDevice.map(_.matchups).sum, n)
    assertEquals(result.recovery.length, 1)
    val recovery = result.recovery.head
    assertEquals(recovery.failedDeviceId, "gpu")
    assertEquals(recovery.failedStatus, 77)
    assertEquals(recovery.recoveredByDeviceId, "cpu")
    assertEquals(recovery.matchups, n)

    var i = 0
    while i < n do
      assertEqualsDouble(result.results(i).win, lowIds(i).toDouble / 100.0, 1e-12)
      assertEqualsDouble(result.results(i).tie, highIds(i).toDouble / 1000.0, 1e-12)
      assertEqualsDouble(result.results(i).loss, seeds(i).toDouble / 10000.0, 1e-12)
      assertEqualsDouble(result.results(i).stderr, 26.0, 1e-12)
      i += 1
  }

  test("dispatchBatchWithDevices fails when every rescue candidate fails") {
    val lowIds = Array(10, 11, 12, 13, 14)
    val highIds = Array(110, 111, 112, 113, 114)
    val seeds = Array(1L, 2L, 3L, 4L, 5L)

    val gpuFail = HeadsUpHybridDispatcher.FunctionalComputeDevice(
      id = "gpu",
      kind = "cuda",
      name = "Failing GPU",
      supportsExact = true,
      estimatedWeight = 100.0,
      computeFn = (_, _, _, _, _, _, _, _, _) => 11
    )

    val openclFail = HeadsUpHybridDispatcher.FunctionalComputeDevice(
      id = "opencl",
      kind = "opencl",
      name = "Failing OpenCL",
      supportsExact = true,
      estimatedWeight = 50.0,
      computeFn = (_, _, _, _, _, _, _, _, _) => 22
    )

    val result = HeadsUpHybridDispatcher.dispatchBatchWithDevices(
      lowIds,
      highIds,
      modeCode = 1,
      trials = 10,
      seeds = seeds,
      activeDevices = Vector(gpuFail, openclFail)
    )

    assert(result.isLeft)
    val error = result.swap.toOption.getOrElse("")
    assert(error.contains("gpu"), clues(error))
    assert(error.contains("status 11"), clues(error))
    assert(error.contains("opencl"), clues(error))
    assert(error.contains("status 22"), clues(error))
  }

  test("dispatchBatchWithDevices reports deterministic payload checksum") {
    val lowIds = Array(1, 4, 9, 16)
    val highIds = Array(2, 5, 10, 17)
    val seeds = Array(9L, 8L, 7L, 6L)

    val cpu = HeadsUpHybridDispatcher.FunctionalComputeDevice(
      id = "cpu",
      kind = "cpu",
      name = "Deterministic CPU",
      supportsExact = true,
      estimatedWeight = 1.0,
      computeFn = (_, _, _, _, _, wins, ties, losses, stderrs) =>
        var i = 0
        while i < wins.length do
          wins(i) = 0.5
          ties(i) = 0.0
          losses(i) = 0.5
          stderrs(i) = 0.01
          i += 1
        0
    )

    val first = HeadsUpHybridDispatcher.dispatchBatchWithDevices(
      lowIds,
      highIds,
      modeCode = 1,
      trials = 20,
      seeds = seeds,
      activeDevices = Vector(cpu)
    ).toOption.getOrElse(fail("first dispatch failed"))

    val second = HeadsUpHybridDispatcher.dispatchBatchWithDevices(
      lowIds,
      highIds,
      modeCode = 1,
      trials = 20,
      seeds = seeds,
      activeDevices = Vector(cpu)
    ).toOption.getOrElse(fail("second dispatch failed"))

    assertEquals(first.payloadCrc32, second.payloadCrc32)
    assertEquals(
      first.payloadCrc32,
      HeadsUpHybridDispatcher.payloadCrc32(lowIds, highIds, modeCode = 1, trials = 20, seeds = seeds)
    )

    val changedSeeds = seeds.clone()
    changedSeeds(0) = changedSeeds(0) + 1L
    val third = HeadsUpHybridDispatcher.dispatchBatchWithDevices(
      lowIds,
      highIds,
      modeCode = 1,
      trials = 20,
      seeds = changedSeeds,
      activeDevices = Vector(cpu)
    ).toOption.getOrElse(fail("third dispatch failed"))

    assertNotEquals(first.payloadCrc32, third.payloadCrc32)
  }

  // ── Failover coverage ─────────────────────────────────────────────

  test("failover: exception-throwing device is recovered by CPU") {
    val n = 8
    val lowIds = Array.tabulate(n)(i => i)
    val highIds = Array.tabulate(n)(i => i + 100)
    val seeds = Array.tabulate(n)(_.toLong)

    val thrower = HeadsUpHybridDispatcher.FunctionalComputeDevice(
      id = "gpu-crash",
      kind = "cuda",
      name = "Crash GPU",
      supportsExact = true,
      estimatedWeight = 200.0,
      computeFn = (_, _, _, _, _, _, _, _, _) =>
        throw new RuntimeException("simulated CUDA crash")
    )

    val cpuOk = HeadsUpHybridDispatcher.FunctionalComputeDevice(
      id = "cpu",
      kind = "cpu",
      name = "OK CPU",
      supportsExact = true,
      estimatedWeight = 1.0,
      computeFn = (_, _, _, _, _, wins, ties, losses, stderrs) =>
        java.util.Arrays.fill(wins, 0.5)
        java.util.Arrays.fill(ties, 0.0)
        java.util.Arrays.fill(losses, 0.5)
        java.util.Arrays.fill(stderrs, 0.01)
        0
    )

    val result = HeadsUpHybridDispatcher.dispatchBatchWithDevices(
      lowIds, highIds, modeCode = 1, trials = 10, seeds = seeds,
      activeDevices = Vector(thrower, cpuOk)
    )
    assert(result.isRight, clues(result))
    val r = result.toOption.get
    assertEquals(r.recovery.length, 1)
    assertEquals(r.recovery.head.failedDeviceId, "gpu-crash")
    assertEquals(r.recovery.head.recoveredByDeviceId, "cpu")
    assertEquals(r.results.length, n)
    r.results.foreach(eq => assertEqualsDouble(eq.win, 0.5, 1e-12))
  }

  test("failover: sole device failure returns Left (no rescue candidates)") {
    val lowIds = Array(1, 2, 3)
    val highIds = Array(10, 20, 30)
    val seeds = Array(1L, 2L, 3L)

    val soloFail = HeadsUpHybridDispatcher.FunctionalComputeDevice(
      id = "solo",
      kind = "cpu",
      name = "Solo Fail",
      supportsExact = true,
      estimatedWeight = 1.0,
      computeFn = (_, _, _, _, _, _, _, _, _) => 99
    )

    val result = HeadsUpHybridDispatcher.dispatchBatchWithDevices(
      lowIds, highIds, modeCode = 0, trials = 0, seeds = seeds,
      activeDevices = Vector(soloFail)
    )
    assert(result.isLeft)
    assert(result.swap.toOption.get.contains("no rescue candidates"))
  }

  test("failover: recovery priority prefers CPU over OpenCL over CUDA") {
    val lowIds = Array(1)
    val highIds = Array(2)
    val seeds = Array(42L)
    var rescuerId = ""

    val primary = HeadsUpHybridDispatcher.FunctionalComputeDevice(
      id = "primary", kind = "cuda", name = "Primary", supportsExact = true,
      estimatedWeight = 1000.0,
      computeFn = (_, _, _, _, _, _, _, _, _) => 1 // fail
    )
    val openclRescue = HeadsUpHybridDispatcher.FunctionalComputeDevice(
      id = "ocl", kind = "opencl", name = "OpenCL", supportsExact = true,
      estimatedWeight = 500.0,
      computeFn = (_, _, _, _, _, wins, ties, losses, stderrs) =>
        rescuerId = "ocl"
        wins(0) = 0.6; ties(0) = 0.0; losses(0) = 0.4; stderrs(0) = 0.0
        0
    )
    val cpuRescue = HeadsUpHybridDispatcher.FunctionalComputeDevice(
      id = "cpu", kind = "cpu", name = "CPU", supportsExact = true,
      estimatedWeight = 10.0,
      computeFn = (_, _, _, _, _, wins, ties, losses, stderrs) =>
        rescuerId = "cpu"
        wins(0) = 0.6; ties(0) = 0.0; losses(0) = 0.4; stderrs(0) = 0.0
        0
    )

    val result = HeadsUpHybridDispatcher.dispatchBatchWithDevices(
      lowIds, highIds, modeCode = 1, trials = 10, seeds = seeds,
      activeDevices = Vector(primary, openclRescue, cpuRescue)
    )
    assert(result.isRight)
    // CPU has highest recovery priority (0) so it should be tried first
    assertEquals(rescuerId, "cpu")
    assertEquals(result.toOption.get.recovery.head.recoveredByDeviceId, "cpu")
  }

  test("failover: empty device list returns Left") {
    val result = HeadsUpHybridDispatcher.dispatchBatchWithDevices(
      lowIds = Array(1), highIds = Array(2), modeCode = 1, trials = 10,
      seeds = Array(1L), activeDevices = Vector.empty
    )
    assert(result.isLeft)
    assert(result.swap.toOption.get.contains("no compute devices"))
  }
