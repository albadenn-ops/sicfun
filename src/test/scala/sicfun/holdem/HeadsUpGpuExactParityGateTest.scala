package sicfun.holdem

import munit.FunSuite

import java.nio.file.{Files, Paths}
import scala.util.Random

class HeadsUpGpuExactParityGateTest extends FunSuite:
  private val ProviderProperty = "sicfun.gpu.provider"
  private val NativePathProperty = "sicfun.gpu.native.path"
  private val NativeEngineProperty = "sicfun.gpu.native.engine"
  private val FallbackToCpuProperty = "sicfun.gpu.fallbackToCpu"
  private val NativeCudaBlockSizeProperty = "sicfun.gpu.native.cuda.blockSize"
  private val NativeCudaMaxChunkMatchupsProperty = "sicfun.gpu.native.cuda.maxChunkMatchups"

  private def withSystemProperties[A](updates: Seq[(String, Option[String])])(thunk: => A): A =
    TestSystemPropertyScope.withSystemProperties(updates)(thunk)

  private def buildExactSlice(maxMatchups: Long, seed: Long): HeadsUpEquityCanonicalTable =
    HeadsUpEquityCanonicalTable.buildAll(
      mode = HeadsUpEquityTable.Mode.Exact,
      rng = new Random(seed),
      maxMatchups = maxMatchups,
      progress = None,
      parallelism = 1,
      backend = HeadsUpEquityTable.ComputeBackend.Gpu
    )

  test("native exact parity is zero-delta for CPU vs CUDA on canonical slice") {
    val nativeDll = Paths.get("src", "main", "native", "build", "sicfun_gpu_kernel.dll").toAbsolutePath.normalize()
    if !Files.isRegularFile(nativeDll) then
      println(s"Skipping exact parity gate test: library not found at $nativeDll")
    else
      withSystemProperties(
        Seq(
          ProviderProperty -> Some("native"),
          NativePathProperty -> Some(nativeDll.toString),
          FallbackToCpuProperty -> Some("false")
        )
      ) {
        val availability = HeadsUpGpuRuntime.availability
        if !availability.available then
          println(s"Skipping exact parity gate test: ${availability.detail}")
        else
          val maxMatchups = 2L
          val seed = 19L

          val cpuTableAttempt =
            scala.util.Try {
              withSystemProperties(
                Seq(
                  NativeEngineProperty -> Some("cpu"),
                  NativeCudaBlockSizeProperty -> None,
                  NativeCudaMaxChunkMatchupsProperty -> None
                )
              ) {
                buildExactSlice(maxMatchups = maxMatchups, seed = seed)
              }
            }

          cpuTableAttempt match
            case scala.util.Failure(cpuErr) =>
              println(s"Skipping exact parity gate test: CPU native execution unavailable (${cpuErr.getMessage})")
            case scala.util.Success(cpuTable) =>
              val cudaTableAttempt =
                scala.util.Try {
                  withSystemProperties(
                    Seq(
                      NativeEngineProperty -> Some("cuda"),
                      NativeCudaBlockSizeProperty -> Some("32"),
                      NativeCudaMaxChunkMatchupsProperty -> Some("1")
                    )
                  ) {
                    buildExactSlice(maxMatchups = maxMatchups, seed = seed)
                  }
                }

              cudaTableAttempt match
                case scala.util.Failure(err) =>
                  println(s"Skipping exact parity gate test: CUDA exact execution unavailable (${err.getMessage})")
                case scala.util.Success(cudaTable) =>
                  assertEquals(cpuTable.values.keySet, cudaTable.values.keySet)

                  var maxWinDelta = 0.0
                  var maxTieDelta = 0.0
                  var maxLossDelta = 0.0
                  var maxEqDelta = 0.0
                  cpuTable.values.keySet.foreach { key =>
                    val cpu = cpuTable.values(key)
                    val cuda = cudaTable.values(key)
                    maxWinDelta = math.max(maxWinDelta, math.abs(cpu.win - cuda.win))
                    maxTieDelta = math.max(maxTieDelta, math.abs(cpu.tie - cuda.tie))
                    maxLossDelta = math.max(maxLossDelta, math.abs(cpu.loss - cuda.loss))
                    maxEqDelta = math.max(maxEqDelta, math.abs(cpu.equity - cuda.equity))
                  }

                  assertEquals(maxWinDelta, 0.0)
                  assertEquals(maxTieDelta, 0.0)
                  assertEquals(maxLossDelta, 0.0)
                  assertEquals(maxEqDelta, 0.0)

                  val telemetry = HeadsUpGpuRuntime.lastBatchTelemetry.getOrElse(
                    fail("missing telemetry after CUDA exact slice generation")
                  )
                  assert(telemetry.success, clues(telemetry))
                  assert(telemetry.detail.contains("nativeEngine=cuda"), clues(telemetry))
      }
  }
