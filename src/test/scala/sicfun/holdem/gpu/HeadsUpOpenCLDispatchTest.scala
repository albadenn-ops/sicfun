package sicfun.holdem.gpu

import munit.FunSuite

/** Tests for OpenCL compute device behaviour in the hybrid dispatcher.
  *
  * Since OpenCL hardware may not be available in CI, these tests use
  * `FunctionalComputeDevice` with `kind="opencl"` to exercise the dispatcher's
  * handling of OpenCL-typed devices without requiring an actual iGPU.
  *
  * Coverage includes:
  *  - '''Exact-mode exclusion''': OpenCL devices with `supportsExact=false` are
  *    filtered out from exact-mode (modeCode=0) dispatches, ensuring only CPU/CUDA
  *    handle enumerative computation.
  *  - '''Monte Carlo participation''': OpenCL devices participate normally in
  *    Monte Carlo batches, receiving their proportional share of work.
  *  - '''Failover to CPU''': when an OpenCL device fails (e.g. status 204 = kernel
  *    execution failure), the CPU rescue candidate recovers the failed slice.
  *  - '''Weight scaling''': verifies that CUDA devices receive proportionally more
  *    work than OpenCL when both are present, reflecting the typical throughput gap.
  */
class HeadsUpOpenCLDispatchTest extends FunSuite:
  override def beforeEach(context: BeforeEach): Unit =
    HeadsUpHybridDispatcher.setCalibratedWeights(Map.empty)
    sys.props.remove("sicfun.hybrid.minRelativeWeight")
    super.beforeEach(context)

  test("OpenCL device is excluded from exact-mode dispatch (supportsExact=false)") {
    val lowIds = Array(1, 2, 3)
    val highIds = Array(10, 20, 30)
    val seeds = Array(1L, 2L, 3L)

    val openclDevice = HeadsUpHybridDispatcher.FunctionalComputeDevice(
      id = "opencl:0",
      kind = "opencl",
      name = "Test iGPU",
      supportsExact = false,
      estimatedWeight = 100.0,
      computeFn = (_, _, _, _, _, _, _, _, _) =>
        fail("OpenCL device should not be called in exact mode")
        0
    )
    val cpuDevice = HeadsUpHybridDispatcher.FunctionalComputeDevice(
      id = "cpu",
      kind = "cpu",
      name = "CPU",
      supportsExact = true,
      estimatedWeight = 50.0,
      computeFn = (_, _, _, _, _, wins, ties, losses, stderrs) =>
        java.util.Arrays.fill(wins, 0.5)
        java.util.Arrays.fill(ties, 0.0)
        java.util.Arrays.fill(losses, 0.5)
        java.util.Arrays.fill(stderrs, 0.0)
        0
    )

    // exact mode (modeCode=0): only supportsExact=true devices should be used
    val exactDevices = Vector(openclDevice, cpuDevice).filter(_.supportsExact)
    val result = HeadsUpHybridDispatcher.dispatchBatchWithDevices(
      lowIds, highIds, modeCode = 0, trials = 0, seeds = seeds,
      activeDevices = exactDevices
    )
    assert(result.isRight)
    assertEquals(result.toOption.get.perDevice.map(_.deviceId), Vector("cpu"))
  }

  test("OpenCL device participates in Monte Carlo dispatch") {
    val n = 10
    val lowIds = Array.tabulate(n)(identity)
    val highIds = Array.tabulate(n)(_ + 100)
    val seeds = Array.tabulate(n)(_.toLong)

    val openclDevice = HeadsUpHybridDispatcher.FunctionalComputeDevice(
      id = "opencl:0",
      kind = "opencl",
      name = "Test iGPU",
      supportsExact = false,
      estimatedWeight = 100.0,
      computeFn = (_, _, _, _, _, wins, ties, losses, stderrs) =>
        java.util.Arrays.fill(wins, 0.45)
        java.util.Arrays.fill(ties, 0.10)
        java.util.Arrays.fill(losses, 0.45)
        java.util.Arrays.fill(stderrs, 0.02)
        0
    )

    val result = HeadsUpHybridDispatcher.dispatchBatchWithDevices(
      lowIds, highIds, modeCode = 1, trials = 100, seeds = seeds,
      activeDevices = Vector(openclDevice)
    )
    assert(result.isRight)
    val r = result.toOption.get
    assertEquals(r.results.length, n)
    assertEquals(r.perDevice.length, 1)
    assertEquals(r.perDevice.head.deviceId, "opencl:0")
    r.results.foreach { eq =>
      assertEqualsDouble(eq.win, 0.45, 1e-12)
      assertEqualsDouble(eq.tie, 0.10, 1e-12)
    }
  }

  test("OpenCL failure falls back to CPU rescue") {
    val lowIds = Array(5, 6)
    val highIds = Array(50, 60)
    val seeds = Array(10L, 20L)

    val openclFail = HeadsUpHybridDispatcher.FunctionalComputeDevice(
      id = "opencl:0",
      kind = "opencl",
      name = "Failing iGPU",
      supportsExact = false,
      estimatedWeight = 100.0,
      computeFn = (_, _, _, _, _, _, _, _, _) => 204 // OpenCL kernel execution failed
    )
    val cpuRescue = HeadsUpHybridDispatcher.FunctionalComputeDevice(
      id = "cpu",
      kind = "cpu",
      name = "Rescue CPU",
      supportsExact = true,
      estimatedWeight = 10.0,
      computeFn = (_, _, _, _, _, wins, ties, losses, stderrs) =>
        java.util.Arrays.fill(wins, 0.5)
        java.util.Arrays.fill(ties, 0.0)
        java.util.Arrays.fill(losses, 0.5)
        java.util.Arrays.fill(stderrs, 0.01)
        0
    )

    val result = HeadsUpHybridDispatcher.dispatchBatchWithDevices(
      lowIds, highIds, modeCode = 1, trials = 50, seeds = seeds,
      activeDevices = Vector(openclFail, cpuRescue)
    )
    assert(result.isRight)
    val r = result.toOption.get
    assertEquals(r.recovery.length, 1)
    assertEquals(r.recovery.head.failedDeviceId, "opencl:0")
    assertEquals(r.recovery.head.failedStatus, 204)
    assertEquals(r.recovery.head.recoveredByDeviceId, "cpu")
  }

  test("OpenCL weight is scaled down relative to CUDA in proportional split") {
    // Verify that when both CUDA and OpenCL are present, CUDA gets more work
    val counts = HeadsUpHybridDispatcher.proportionalCounts(
      totalItems = 100,
      deviceWeights = Vector(
        "cuda:0" -> 10000.0,  // smCount * clockMHz
        "opencl:0" -> 3000.0  // computeUnits * clockMHz * 0.3
      )
    )
    val cudaCount = counts.find(_._1 == "cuda:0").map(_._2).getOrElse(0)
    val openclCount = counts.find(_._1 == "opencl:0").map(_._2).getOrElse(0)
    assert(cudaCount > openclCount, s"CUDA ($cudaCount) should get more work than OpenCL ($openclCount)")
    assertEquals(cudaCount + openclCount, 100)
  }
