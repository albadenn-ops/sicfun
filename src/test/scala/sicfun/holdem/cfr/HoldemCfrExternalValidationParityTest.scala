package sicfun.holdem.cfr

import munit.FunSuite
import sicfun.holdem.types.TestSystemPropertyScope

import java.nio.file.{Files, Path}
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

/** Cross-backend parity tests: verifies that native CPU and GPU CFR solvers produce
  * strategies matching the Scala reference within tight tolerances.
  *
  * The test workflow for each native backend is:
  *  1. Run the full approximation report suite with provider="scala" (reference)
  *  2. Run the same suite with provider="native-cpu" or "native-gpu"
  *  3. Use [[HoldemCfrExternalComparison.compareFiles]] to verify that policies,
  *     best actions, and EVs match within tight thresholds:
  *     - TV distance <= 1e-3 mean, 2e-3 per spot
  *     - 100% best-action agreement
  *     - EV RMSE <= 1e-3
  *
  * These thresholds are tight enough to catch algorithmic bugs but allow for
  * the small floating-point differences between JVM and native IEEE 754 evaluation.
  * Tests gracefully skip when the native backend is unavailable (no GPU, no DLL).
  *
  * The 180-second timeout accommodates the full suite solve on both backends.
  */
class HoldemCfrExternalValidationParityTest extends FunSuite:
  override val munitTimeout: Duration = 180.seconds

  // Reduced iteration count (600) with postflop lookahead — sufficient for parity
  // testing since we're comparing Scala vs native, not measuring absolute convergence.
  private val ExternalParityConfig = HoldemCfrConfig(
    iterations = 600,
    averagingDelay = 100,
    maxVillainHands = 32,
    equityTrials = 700,
    includeVillainReraises = true,
    postflopLookahead = true,
    preferNativeBatch = false,
    rngSeed = 37L
  )

  // Tight thresholds that catch algorithmic differences while allowing for
  // IEEE 754 float-order-of-operations differences between JVM and native code.
  private val ExternalParityThresholds = HoldemCfrExternalComparison.Thresholds(
    maxMeanTvDistance = Some(1e-3),
    maxSpotTvDistance = Some(2e-3),
    minBestActionAgreement = Some(1.0),
    maxMeanEvRmse = Some(1e-3)
  )

  private def withSystemProperties[A](properties: Map[String, String])(thunk: => A): A =
    TestSystemPropertyScope.withSystemProperties(properties.toSeq.map { case (key, value) => key -> Some(value) }) {
      HoldemCfrNativeRuntime.resetLoadCacheForTests()
      HoldemCfrSolver.resetAutoProviderForTests()
      try thunk
      finally
        HoldemCfrNativeRuntime.resetLoadCacheForTests()
        HoldemCfrSolver.resetAutoProviderForTests()
    }

  test("native CPU full solve passes external comparison gate against scala reference") {
    assertProviderPassesExternalValidation(
      availability = HoldemCfrNativeRuntime.availability(HoldemCfrNativeRuntime.Backend.Cpu),
      providerProperty = "native-cpu",
      expectedProviderLabel = "native-cpu"
    )
  }

  test("native GPU full solve passes external comparison gate against scala reference") {
    assertProviderPassesExternalValidation(
      availability = HoldemCfrNativeRuntime.availability(HoldemCfrNativeRuntime.Backend.Gpu),
      providerProperty = "native-gpu",
      expectedProviderLabel = "native-gpu"
    )
  }

  /** Core parity assertion: runs the approximation suite on both Scala and the target
    * native backend, then compares the two exports via external comparison with strict
    * thresholds. Skips gracefully if the native backend is unavailable. Preserves
    * artifacts on failure for debugging.
    */
  private def assertProviderPassesExternalValidation(
      availability: HoldemCfrNativeRuntime.Availability,
      providerProperty: String,
      expectedProviderLabel: String
  ): Unit =
    if availability.available then
      val root = Files.createTempDirectory(s"holdem-cfr-external-validation-$expectedProviderLabel-")
      var keepArtifacts = true
      try
        val scalaPath = runReport(
          providerProperty = "scala",
          expectedProviderLabel = "scala",
          outDir = root.resolve("scala")
        )
        val nativePath = runReport(
          providerProperty = providerProperty,
          expectedProviderLabel = expectedProviderLabel,
          outDir = root.resolve(expectedProviderLabel)
        )
        val compareOutDir = root.resolve(s"compare-$expectedProviderLabel")
        val compared = HoldemCfrExternalComparison.compareFiles(
          referencePath = scalaPath,
          externalPath = nativePath,
          thresholds = ExternalParityThresholds,
          outDir = Some(compareOutDir)
        )

        assert(
          compared.isRight,
          s"external validation failed for $expectedProviderLabel: $compared (artifacts: $compareOutDir)"
        )
        keepArtifacts = false
      finally
        if !keepArtifacts then deleteRecursively(root)
    else
      println(s"Skipping external validation parity test for $expectedProviderLabel: ${availability.detail}")

  private def runReport(
      providerProperty: String,
      expectedProviderLabel: String,
      outDir: Path
  ): Path =
    val result =
      withSystemProperties(Map("sicfun.cfr.provider" -> providerProperty)) {
        HoldemCfrApproximationReport.runSuite(
          suiteName = "default",
          spots = HoldemCfrApproximationReport.DefaultSuite,
          cfrConfig = ExternalParityConfig,
          outDir = Some(outDir)
        )
      }

    assert(
      result.isRight,
      s"expected approximation report success for $expectedProviderLabel, got $result (artifacts: $outDir)"
    )
    val runResult = result.toOption.getOrElse(fail(s"missing run result for $expectedProviderLabel"))
    assertEquals(
      runResult.aggregate.providerCounts,
      Map(expectedProviderLabel -> HoldemCfrApproximationReport.DefaultSuite.size)
    )
    outDir.resolve(HoldemCfrApproximationReport.ExternalComparisonFileName)

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val all = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        all.foreach(Files.deleteIfExists)
      finally
        stream.close()
