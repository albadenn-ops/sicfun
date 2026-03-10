package sicfun.holdem.gpu
import sicfun.holdem.types.*
import sicfun.holdem.*
import sicfun.holdem.equity.*

import munit.FunSuite
import java.nio.file.{Files, Paths}

class HeadsUpGpuRuntimeTest extends FunSuite:
  private val ProviderProperty = "sicfun.gpu.provider"
  private val NativePathProperty = "sicfun.gpu.native.path"
  private val NativeEngineProperty = "sicfun.gpu.native.engine"
  private val NativeCudaBlockSizeProperty = "sicfun.gpu.native.cuda.blockSize"
  private val NativeCudaMaxChunkMatchupsProperty = "sicfun.gpu.native.cuda.maxChunkMatchups"
  private val OpenCLPathProperty = "sicfun.opencl.native.path"

  private def withSystemProperties[A](updates: Seq[(String, Option[String])])(thunk: => A): A =
    TestSystemPropertyScope.withSystemProperties(updates)(thunk)

  private def withProvider[A](provider: String)(thunk: => A): A =
    withSystemProperties(Seq(ProviderProperty -> Some(provider)))(thunk)

  test("cpu-emulated provider matches CPU deterministic batch") {
    withProvider("cpu-emulated") {
      val mode = HeadsUpEquityTable.Mode.MonteCarlo(20)
      val seed = 17L
      val batch = HeadsUpEquityTable.selectFullBatch(120L)
      val cpu = HeadsUpEquityTable.computeBatchCpu(
        mode = mode,
        packedKeys = batch.packedKeys,
        keyMaterial = batch.keyMaterial,
        parallelism = 1,
        monteCarloSeedBase = seed
      )
      val gpuEither = HeadsUpGpuRuntime.computeBatch(
        packedKeys = batch.packedKeys,
        keyMaterial = batch.keyMaterial,
        mode = mode,
        monteCarloSeedBase = seed
      )
      assert(gpuEither.isRight)
      val gpu = gpuEither.toOption.getOrElse(fail("expected GPU emulated batch result"))
      assertEquals(gpu.toVector, cpu.toVector)
    }
  }

  test("disabled provider reports unavailable and rejects compute") {
    withProvider("disabled") {
      val availability = HeadsUpGpuRuntime.availability
      assert(!availability.available)
      val batch = HeadsUpEquityTable.selectFullBatch(5L)
      val result = HeadsUpGpuRuntime.computeBatch(
        packedKeys = batch.packedKeys,
        keyMaterial = batch.keyMaterial,
        mode = HeadsUpEquityTable.Mode.MonteCarlo(5),
        monteCarloSeedBase = 1L
      )
      assert(result.isLeft)
    }
  }

  test("native provider reports actual engine in telemetry when JNI library is available") {
    val nativeDll = Paths.get("src", "main", "native", "build", "sicfun_gpu_kernel.dll").toAbsolutePath.normalize()
    if !Files.isRegularFile(nativeDll) then
      println(s"Skipping native JNI integration assertion: library not found at $nativeDll")
    else
      withSystemProperties(
        Seq(
          ProviderProperty -> Some("native"),
          NativePathProperty -> Some(nativeDll.toString),
          NativeEngineProperty -> Some("cpu")
        )
      ) {
        val availability = HeadsUpGpuRuntime.availability
        if !availability.available then
          println(s"Skipping native JNI integration assertion: ${availability.detail}")
        else
          val batch = HeadsUpEquityTable.selectFullBatch(32L)
          val result = HeadsUpGpuRuntime.computeBatch(
            packedKeys = batch.packedKeys,
            keyMaterial = batch.keyMaterial,
            mode = HeadsUpEquityTable.Mode.MonteCarlo(8),
            monteCarloSeedBase = 13L
          )
          result match
            case Left(reason) =>
              println(s"Skipping native JNI integration assertion: $reason")
            case Right(_) =>
              val telemetry = HeadsUpGpuRuntime.lastBatchTelemetry.getOrElse(fail("missing telemetry after native batch"))
              assertEquals(telemetry.provider, "native")
              assert(telemetry.success)
              assert(telemetry.detail.contains("nativeEngine="))
      }
  }

  test("native provider exact mode can execute on CUDA when forced") {
    val nativeDll = Paths.get("src", "main", "native", "build", "sicfun_gpu_kernel.dll").toAbsolutePath.normalize()
    if !Files.isRegularFile(nativeDll) then
      println(s"Skipping native exact CUDA assertion: library not found at $nativeDll")
    else
      withSystemProperties(
        Seq(
          ProviderProperty -> Some("native"),
          NativePathProperty -> Some(nativeDll.toString),
          NativeEngineProperty -> Some("cuda"),
          NativeCudaBlockSizeProperty -> Some("32"),
          NativeCudaMaxChunkMatchupsProperty -> Some("1")
        )
      ) {
        val availability = HeadsUpGpuRuntime.availability
        if !availability.available then
          println(s"Skipping native exact CUDA assertion: ${availability.detail}")
        else
          val batch = HeadsUpEquityCanonicalTable.selectCanonicalBatch(1L)
          val result = HeadsUpGpuRuntime.computeBatch(
            packedKeys = batch.packedKeys,
            keyMaterial = batch.keyMaterial,
            mode = HeadsUpEquityTable.Mode.Exact,
            monteCarloSeedBase = 7L
          )
          result match
            case Left(reason) =>
              println(s"Skipping native exact CUDA assertion: $reason")
            case Right(values) =>
              assertEquals(values.length, 1)
              val telemetry = HeadsUpGpuRuntime.lastBatchTelemetry.getOrElse(
                fail("missing telemetry after exact native batch")
              )
              assertEquals(telemetry.provider, "native")
              assert(telemetry.success)
              assert(telemetry.detail.contains("nativeEngine=cuda"), clues(telemetry))
              assertEquals(values(0).stderr, 0.0)
              assert(math.abs((values(0).win + values(0).tie + values(0).loss) - 1.0) <= 1e-12)
      }
  }

  test("opencl provider computes Monte Carlo batch when OpenCL library is available") {
    val openclDll = Paths.get("src", "main", "native", "build", "sicfun_opencl_kernel.dll").toAbsolutePath.normalize()
    if !Files.isRegularFile(openclDll) then
      println(s"Skipping OpenCL provider test: library not found at $openclDll")
    else
      withSystemProperties(
        Seq(
          ProviderProperty -> Some("opencl"),
          OpenCLPathProperty -> Some(openclDll.toString)
        )
      ) {
        val availability = HeadsUpGpuRuntime.availability
        if !availability.available then
          println(s"Skipping OpenCL provider test: ${availability.detail}")
        else
          val mode = HeadsUpEquityTable.Mode.MonteCarlo(50)
          val seed = 29L
          val batch = HeadsUpEquityTable.selectFullBatch(64L)
          val result = HeadsUpGpuRuntime.computeBatch(
            packedKeys = batch.packedKeys,
            keyMaterial = batch.keyMaterial,
            mode = mode,
            monteCarloSeedBase = seed
          )
          result match
            case Left(reason) =>
              println(s"Skipping OpenCL provider test: $reason")
            case Right(values) =>
              assertEquals(values.length, batch.packedKeys.length)
              values.foreach { v =>
                assert(java.lang.Double.isFinite(v.win))
                assert(java.lang.Double.isFinite(v.tie))
                assert(java.lang.Double.isFinite(v.loss))
                assert(math.abs((v.win + v.tie + v.loss) - 1.0) <= 1e-9)
              }
              val telemetry = HeadsUpGpuRuntime.lastBatchTelemetry.getOrElse(
                fail("missing telemetry after OpenCL batch")
              )
              assertEquals(telemetry.provider, "opencl")
              assert(telemetry.success, clues(telemetry))
      }
  }

  test("opencl provider rejects exact mode") {
    val openclDll = Paths.get("src", "main", "native", "build", "sicfun_opencl_kernel.dll").toAbsolutePath.normalize()
    if !Files.isRegularFile(openclDll) then
      println(s"Skipping OpenCL exact-rejection test: library not found at $openclDll")
    else
      withSystemProperties(
        Seq(
          ProviderProperty -> Some("opencl"),
          OpenCLPathProperty -> Some(openclDll.toString)
        )
      ) {
        val availability = HeadsUpGpuRuntime.availability
        if !availability.available then
          println(s"Skipping OpenCL exact-rejection test: ${availability.detail}")
        else
          val batch = HeadsUpEquityTable.selectFullBatch(4L)
          val result = HeadsUpGpuRuntime.computeBatch(
            packedKeys = batch.packedKeys,
            keyMaterial = batch.keyMaterial,
            mode = HeadsUpEquityTable.Mode.Exact,
            monteCarloSeedBase = 1L
          )
          assert(result.isLeft, "OpenCL provider should reject exact mode")
      }
  }

  test("hybrid provider discovers devices and computes Monte Carlo batch") {
    val gpuDll = Paths.get("src", "main", "native", "build", "sicfun_gpu_kernel.dll").toAbsolutePath.normalize()
    if !Files.isRegularFile(gpuDll) then
      println(s"Skipping hybrid provider test: GPU library not found at $gpuDll")
    else
      withSystemProperties(
        Seq(
          ProviderProperty -> Some("hybrid"),
          NativePathProperty -> Some(gpuDll.toString)
        )
      ) {
        val availability = HeadsUpGpuRuntime.availability
        if !availability.available then
          println(s"Skipping hybrid provider test: ${availability.detail}")
        else
          assert(availability.detail.contains("hybrid"))

          val mode = HeadsUpEquityTable.Mode.MonteCarlo(50)
          val seed = 37L
          val batch = HeadsUpEquityTable.selectFullBatch(128L)
          val result = HeadsUpGpuRuntime.computeBatch(
            packedKeys = batch.packedKeys,
            keyMaterial = batch.keyMaterial,
            mode = mode,
            monteCarloSeedBase = seed
          )
          result match
            case Left(reason) =>
              println(s"Skipping hybrid provider test: $reason")
            case Right(values) =>
              assertEquals(values.length, batch.packedKeys.length)
              values.foreach { v =>
                assert(java.lang.Double.isFinite(v.win))
                assert(java.lang.Double.isFinite(v.tie))
                assert(java.lang.Double.isFinite(v.loss))
                assert(math.abs((v.win + v.tie + v.loss) - 1.0) <= 1e-9)
              }
              val telemetry = HeadsUpGpuRuntime.lastBatchTelemetry.getOrElse(
                fail("missing telemetry after hybrid batch")
              )
              assertEquals(telemetry.provider, "hybrid")
              assert(telemetry.success, clues(telemetry))
              assert(telemetry.detail.contains("hybrid["), clues(telemetry))
      }
  }

  test("hybrid dispatcher reports per-device telemetry") {
    val gpuDll = Paths.get("src", "main", "native", "build", "sicfun_gpu_kernel.dll").toAbsolutePath.normalize()
    if !Files.isRegularFile(gpuDll) then
      println(s"Skipping hybrid telemetry test: GPU library not found at $gpuDll")
    else
      withSystemProperties(
        Seq(
          NativePathProperty -> Some(gpuDll.toString)
        )
      ) {
        val devices = HeadsUpHybridDispatcher.devices
        if devices.isEmpty then
          println("Skipping hybrid telemetry test: no devices discovered")
        else
          val batch = HeadsUpEquityCanonicalTable.selectCanonicalBatch(64L)
          val n = batch.packedKeys.length
          val lowIds = new Array[Int](n)
          val highIds = new Array[Int](n)
          val seeds = new Array[Long](n)
          var idx = 0
          while idx < n do
            val packed = batch.packedKeys(idx)
            lowIds(idx) = HeadsUpEquityTable.unpackLowId(packed)
            highIds(idx) = HeadsUpEquityTable.unpackHighId(packed)
            seeds(idx) = HeadsUpEquityTable.monteCarloSeed(42L, batch.keyMaterial(idx))
            idx += 1

          HeadsUpHybridDispatcher.dispatchBatch(lowIds, highIds, 1, 50, seeds) match
            case Left(error) =>
              println(s"Skipping hybrid telemetry test: $error")
            case Right(result) =>
              assert(result.results.length == n)
              assert(result.perDevice.nonEmpty, "expected at least one device in telemetry")
              val totalMatchups = result.perDevice.map(_.matchups).sum
              assertEquals(totalMatchups, n)
              result.perDevice.foreach { t =>
                assert(t.matchups > 0, s"device ${t.deviceId} had zero matchups")
                assert(t.deviceName.nonEmpty, s"device ${t.deviceId} had empty name")
              }
      }
  }

  test("cpu-emulated provider rejects mismatched packedKeys/keyMaterial shape") {
    withProvider("cpu-emulated") {
      val result = HeadsUpGpuRuntime.computeBatch(
        packedKeys = Array(1L, 2L),
        keyMaterial = Array(1L),
        mode = HeadsUpEquityTable.Mode.MonteCarlo(5),
        monteCarloSeedBase = 1L
      )
      assert(result.isLeft)
      val telemetry = HeadsUpGpuRuntime.lastBatchTelemetry.getOrElse(
        fail("missing telemetry after shape validation failure")
      )
      assertEquals(telemetry.provider, "cpu-emulated")
      assert(!telemetry.success)
    }
  }

  test("unknown provider is reported as unavailable and compute is rejected") {
    withProvider("definitely-unknown-provider") {
      val availability = HeadsUpGpuRuntime.availability
      assert(!availability.available)
      val batch = HeadsUpEquityTable.selectFullBatch(2L)
      val result = HeadsUpGpuRuntime.computeBatch(
        packedKeys = batch.packedKeys,
        keyMaterial = batch.keyMaterial,
        mode = HeadsUpEquityTable.Mode.MonteCarlo(3),
        monteCarloSeedBase = 1L
      )
      assert(result.isLeft)
      val telemetry = HeadsUpGpuRuntime.lastBatchTelemetry.getOrElse(
        fail("missing telemetry after unknown provider call")
      )
      assertEquals(telemetry.provider, "disabled")
      assert(!telemetry.success)
    }
  }
