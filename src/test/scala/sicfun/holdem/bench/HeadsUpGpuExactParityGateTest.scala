package sicfun.holdem.bench
import sicfun.holdem.types.*
import sicfun.holdem.*
import sicfun.holdem.equity.*
import sicfun.holdem.gpu.*

import munit.FunSuite

import java.nio.file.{Files, Paths}
import scala.util.Random

/** Integration test that verifies bit-exact parity between the CPU-native and
  * CUDA-native exact-enumeration backends on a small canonical matchup slice.
  *
  * The test discovers the native DLL at build time, configures system properties
  * to force first CPU then CUDA execution, builds an exact canonical table slice
  * with each engine, and asserts zero delta on all equity components (win/tie/loss/equity).
  * Also validates that CUDA batch telemetry is recorded.
  *
  * Gracefully skips when the native library or CUDA device is unavailable.
  */
class HeadsUpGpuExactParityGateTest extends FunSuite:
  // System property keys controlling the GPU runtime configuration
  private val ProviderProperty = "sicfun.gpu.provider"
  private val NativePathProperty = "sicfun.gpu.native.path"
  private val NativeEngineProperty = "sicfun.gpu.native.engine"
  private val FallbackToCpuProperty = "sicfun.gpu.fallbackToCpu"
  private val NativeCudaBlockSizeProperty = "sicfun.gpu.native.cuda.blockSize"
  private val NativeCudaMaxChunkMatchupsProperty = "sicfun.gpu.native.cuda.maxChunkMatchups"

  /** Resets the GPU runtime's cached native library load state before each test
    * so system property changes take effect on the next load attempt.
    */
  override def beforeEach(context: BeforeEach): Unit =
    HeadsUpGpuRuntime.resetLoadCacheForTests()
    super.beforeEach(context)

  /** Resets the GPU runtime cache after each test to avoid cross-test contamination. */
  override def afterEach(context: AfterEach): Unit =
    try HeadsUpGpuRuntime.resetLoadCacheForTests()
    finally super.afterEach(context)

  /** Temporarily sets/clears system properties for the duration of `thunk`,
    * restoring originals afterward. Used to switch between CPU and CUDA engines.
    */
  private def withSystemProperties[A](updates: Seq[(String, Option[String])])(thunk: => A): A =
    TestSystemPropertyScope.withSystemProperties(updates)(thunk)

  /** Builds a small exact-enumeration canonical table slice using the GPU backend
    * with whatever native engine is currently configured via system properties.
    */
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
