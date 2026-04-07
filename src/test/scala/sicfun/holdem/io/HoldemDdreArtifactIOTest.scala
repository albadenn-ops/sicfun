package sicfun.holdem.io

import munit.FunSuite

import java.nio.file.{Files, Path}

/**
 * Tests for [[HoldemDdreArtifactIO]] ONNX artifact manifest persistence.
 *
 * Validates:
 *   - Save/load roundtrip preserves minimal artifacts (all optional fields absent)
 *   - Save/load roundtrip preserves fully populated artifacts (all optional fields present)
 *   - Load returns Left for non-existent directories
 *   - OnnxArtifact input validation: blank artifactId, invalid executionProvider,
 *     negative cudaDevice, non-positive thread counts
 */
class HoldemDdreArtifactIOTest extends FunSuite:
  /** Creates a temp directory, runs the test body, then recursively deletes the directory. */
  private def withTempDir(name: String)(f: Path => Unit): Unit =
    val dir = Files.createTempDirectory(s"sicfun-ddre-artifact-$name-")
    try f(dir)
    finally Files.walk(dir).sorted(java.util.Comparator.reverseOrder()).forEach(Files.deleteIfExists(_))

  /** Creates a minimal valid OnnxArtifact with all optional fields set to None/false. */
  private def minimal: HoldemDdreArtifactIO.OnnxArtifact =
    HoldemDdreArtifactIO.OnnxArtifact(
      artifactId = "test-001",
      source = "unit-test",
      createdAtEpochMillis = 1_710_000_000_000L,
      modelFile = "model.onnx",
      priorInputName = "prior",
      likelihoodInputName = "likelihoods",
      outputName = "posterior",
      executionProvider = "cpu",
      cudaDevice = 0,
      intraOpThreads = None,
      interOpThreads = None,
      validationStatus = "experimental",
      decisionDrivingAllowed = false,
      validationSampleCount = None,
      meanNll = None,
      meanKlVsBayes = None,
      blockerViolationRate = None,
      failureRate = None,
      p50LatencyMillis = None,
      p95LatencyMillis = None,
      gateMinSamples = None,
      gateMaxMeanNll = None,
      gateMaxMeanKlVsBayes = None,
      gateMaxBlockerViolationRate = None,
      gateMaxFailureRate = None,
      gateMaxP95LatencyMillis = None,
      notes = None
    )

  test("save/load roundtrip preserves minimal artifact") {
    withTempDir("minimal") { dir =>
      val artifact = minimal
      HoldemDdreArtifactIO.save(dir, artifact)
      val loaded = HoldemDdreArtifactIO.load(dir)
      assert(loaded.isRight, s"load failed: $loaded")
      val a = loaded.toOption.get
      assertEquals(a.artifactId, "test-001")
      assertEquals(a.source, "unit-test")
      assertEquals(a.modelFile, "model.onnx")
      assertEquals(a.executionProvider, "cpu")
      assertEquals(a.validationStatus, "experimental")
      assertEquals(a.decisionDrivingAllowed, false)
    }
  }

  test("save/load roundtrip preserves all optional fields") {
    withTempDir("full") { dir =>
      val artifact = minimal.copy(
        executionProvider = "cuda",
        cudaDevice = 1,
        intraOpThreads = Some(4),
        interOpThreads = Some(2),
        validationStatus = "validated",
        decisionDrivingAllowed = true,
        validationSampleCount = Some(1000),
        meanNll = Some(0.5),
        meanKlVsBayes = Some(0.01),
        blockerViolationRate = Some(0.002),
        failureRate = Some(0.001),
        p50LatencyMillis = Some(1.5),
        p95LatencyMillis = Some(3.2),
        gateMinSamples = Some(500),
        gateMaxMeanNll = Some(1.0),
        gateMaxMeanKlVsBayes = Some(0.05),
        gateMaxBlockerViolationRate = Some(0.01),
        gateMaxFailureRate = Some(0.005),
        gateMaxP95LatencyMillis = Some(10.0),
        notes = Some("test notes")
      )
      HoldemDdreArtifactIO.save(dir, artifact)
      val loaded = HoldemDdreArtifactIO.load(dir)
      assert(loaded.isRight)
      val a = loaded.toOption.get
      assertEquals(a.cudaDevice, 1)
      assertEquals(a.intraOpThreads, Some(4))
      assertEquals(a.interOpThreads, Some(2))
      assertEquals(a.decisionDrivingAllowed, true)
      assertEquals(a.validationSampleCount, Some(1000))
      assertEqualsDouble(a.meanNll.get, 0.5, 1e-12)
      assertEqualsDouble(a.meanKlVsBayes.get, 0.01, 1e-12)
      assertEquals(a.notes, Some("test notes"))
    }
  }

  test("load returns Left for non-existent directory") {
    val result = HoldemDdreArtifactIO.load(Path.of("non-existent-dir"))
    assert(result.isLeft)
  }

  test("OnnxArtifact rejects blank artifactId") {
    intercept[IllegalArgumentException] {
      minimal.copy(artifactId = "  ")
    }
  }

  test("OnnxArtifact rejects invalid executionProvider") {
    intercept[IllegalArgumentException] {
      minimal.copy(executionProvider = "tpu")
    }
  }

  test("OnnxArtifact rejects negative cudaDevice") {
    intercept[IllegalArgumentException] {
      minimal.copy(cudaDevice = -1)
    }
  }

  test("OnnxArtifact rejects non-positive intraOpThreads") {
    intercept[IllegalArgumentException] {
      minimal.copy(intraOpThreads = Some(0))
    }
  }
