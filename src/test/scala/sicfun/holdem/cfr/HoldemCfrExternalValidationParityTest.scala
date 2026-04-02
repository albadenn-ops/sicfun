package sicfun.holdem.cfr

import munit.FunSuite
import sicfun.holdem.types.TestSystemPropertyScope

import java.nio.file.{Files, Path}
import scala.concurrent.duration.*
import scala.jdk.CollectionConverters.*

class HoldemCfrExternalValidationParityTest extends FunSuite:
  override val munitTimeout: Duration = 180.seconds

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
