package sicfun.holdem

import munit.FunSuite
import sicfun.core.Card

import java.nio.file.{Files, Path, Paths, StandardCopyOption}
import scala.util.Random

class HoldemDdreIntegrationTest extends FunSuite:
  private val DdreModeProperty = "sicfun.ddre.mode"
  private val DdreProviderProperty = "sicfun.ddre.provider"
  private val DdreAlphaProperty = "sicfun.ddre.alpha"
  private val DdreMinEntropyBitsProperty = "sicfun.ddre.minEntropyBits"
  private val DdreNativeCpuPathProperty = "sicfun.ddre.native.cpu.path"
  private val DdreNativeGpuPathProperty = "sicfun.ddre.native.gpu.path"
  private val DdreOnnxModelPathProperty = "sicfun.ddre.onnx.modelPath"
  private val DdreOnnxArtifactDirProperty = "sicfun.ddre.onnx.artifactDir"
  private val DdreOnnxAllowExperimentalProperty = "sicfun.ddre.onnx.allowExperimental"
  private val PreflopBackendProperty = "sicfun.holdem.preflopEquityBackend"

  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card token: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def withSystemProperties[A](updates: Seq[(String, Option[String])])(thunk: => A): A =
    TestSystemPropertyScope.withSystemProperties(updates) {
      HoldemDdreNativeRuntime.resetLoadCacheForTests()
      try thunk
      finally HoldemDdreNativeRuntime.resetLoadCacheForTests()
    }

  private def ddreSmokeOnnxModelPath: String =
    Option(getClass.getClassLoader.getResource("sicfun/ddre/ddre-smoke-sqrt.onnx")) match
      case Some(url) => Paths.get(url.toURI).toString
      case None => fail("missing test resource: sicfun/ddre/ddre-smoke-sqrt.onnx")

  private def createSmokeArtifactDir(
      validationStatus: String = "experimental",
      decisionDrivingAllowed: Boolean = false
  ): Path =
    val dir = Files.createTempDirectory("sicfun-ddre-smoke-artifact-")
    val modelTarget = dir.resolve("ddre-smoke-sqrt.onnx")
    Files.copy(Paths.get(ddreSmokeOnnxModelPath), modelTarget, StandardCopyOption.REPLACE_EXISTING)
    HoldemDdreArtifactIO.save(
      dir,
      HoldemDdreArtifactIO.OnnxArtifact(
        artifactId = "ddre-smoke-test",
        source = "HoldemDdreIntegrationTest",
        createdAtEpochMillis = 1_800_000_000_000L,
        modelFile = modelTarget.getFileName.toString,
        priorInputName = "prior",
        likelihoodInputName = "likelihoods",
        outputName = "posterior",
        executionProvider = "cpu",
        cudaDevice = 0,
        intraOpThreads = None,
        interOpThreads = None,
        validationStatus = validationStatus,
        decisionDrivingAllowed = decisionDrivingAllowed,
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
        notes = Some("experimental smoke artifact for integration tests")
      )
    )
    dir

  private def runPosterior(seed: Long): PosteriorInferenceResult =
    val hero = hole("Ac", "Kd")
    val board = Board.empty
    val villainPos = Position.BigBlind
    val folds = Vector(PreflopFold(Position.UTG))
    val state = GameState(
      street = Street.Preflop,
      board = board,
      pot = 6.0,
      toCall = 2.0,
      position = villainPos,
      stackSize = 98.0,
      betHistory = Vector.empty
    )
    RangeInferenceEngine.inferPosterior(
      hero = hero,
      board = board,
      folds = folds,
      tableRanges = TableRanges.defaults(TableFormat.NineMax),
      villainPos = villainPos,
      observations = Seq(VillainObservation(PokerAction.Raise(25.0), state)),
      actionModel = PokerActionModel.uniform,
      bunchingTrials = 60,
      rng = new Random(seed),
      useCache = false
    )

  private def l1Distance(a: PosteriorInferenceResult, b: PosteriorInferenceResult): Double =
    a.posterior.weights.keysIterator.map { hand =>
      math.abs(a.posterior.probabilityOf(hand) - b.posterior.probabilityOf(hand))
    }.sum

  test("DDRE shadow mode keeps Bayesian decision posterior unchanged") {
    val baseline = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("off"),
        DdreProviderProperty -> Some("disabled"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 31L)
    }

    val shadow = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("shadow"),
        DdreProviderProperty -> Some("synthetic"),
        DdreAlphaProperty -> Some("0.8"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 31L)
    }

    assertEquals(shadow.posterior.weights, baseline.posterior.weights)
    assertEqualsDouble(shadow.logEvidence, baseline.logEvidence, 1e-12)
  }

  test("DDRE blend mode falls back to Bayesian when provider is disabled") {
    val baseline = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("off"),
        DdreProviderProperty -> Some("disabled"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 41L)
    }

    val blendFallback = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("blend-canary"),
        DdreProviderProperty -> Some("disabled"),
        DdreAlphaProperty -> Some("1.0"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 41L)
    }

    assertEquals(blendFallback.posterior.weights, baseline.posterior.weights)
  }

  test("DDRE blend-primary with alpha=1 can drive non-Bayesian posterior") {
    val baseline = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("off"),
        DdreProviderProperty -> Some("disabled"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 53L)
    }

    val blend = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("blend-primary"),
        DdreProviderProperty -> Some("synthetic"),
        DdreAlphaProperty -> Some("1.0"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 53L)
    }

    val l1 = l1Distance(baseline, blend)
    assert(l1 > 1e-6, s"expected blended posterior to differ from Bayesian baseline; l1=$l1")
    assert(math.abs(blend.posterior.weights.values.sum - 1.0) < 1e-9)
  }

  test("DDRE entropy guard degradation falls back to Bayesian posterior in blend mode") {
    val baseline = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("off"),
        DdreProviderProperty -> Some("disabled"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 71L)
    }

    val entropyGuardedBlend = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("blend-primary"),
        DdreProviderProperty -> Some("synthetic"),
        DdreAlphaProperty -> Some("1.0"),
        DdreMinEntropyBitsProperty -> Some("9999.0"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 71L)
    }

    assertEquals(entropyGuardedBlend.posterior.weights, baseline.posterior.weights)
  }

  test("DDRE native-cpu provider falls back to Bayesian when native path is invalid") {
    val missingPath = Paths
      .get(System.getProperty("java.io.tmpdir"), s"sicfun-ddre-native-cpu-missing-${System.nanoTime()}.dll")
      .toString

    val baseline = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("off"),
        DdreProviderProperty -> Some("disabled"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 79L)
    }

    val nativeFallback = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("blend-primary"),
        DdreProviderProperty -> Some("native-cpu"),
        DdreAlphaProperty -> Some("1.0"),
        DdreNativeCpuPathProperty -> Some(missingPath),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 79L)
    }

    assertEquals(nativeFallback.posterior.weights, baseline.posterior.weights)
  }

  test("DDRE native-gpu provider falls back to Bayesian when native path is invalid") {
    val missingPath = Paths
      .get(System.getProperty("java.io.tmpdir"), s"sicfun-ddre-native-gpu-missing-${System.nanoTime()}.dll")
      .toString

    val baseline = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("off"),
        DdreProviderProperty -> Some("disabled"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 83L)
    }

    val nativeFallback = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("blend-primary"),
        DdreProviderProperty -> Some("native-gpu"),
        DdreAlphaProperty -> Some("1.0"),
        DdreNativeGpuPathProperty -> Some(missingPath),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 83L)
    }

    assertEquals(nativeFallback.posterior.weights, baseline.posterior.weights)
  }

  test("DDRE onnx provider falls back to Bayesian when model path is invalid") {
    val missingPath = Paths
      .get(System.getProperty("java.io.tmpdir"), s"sicfun-ddre-onnx-missing-${System.nanoTime()}.onnx")
      .toString

    val baseline = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("off"),
        DdreProviderProperty -> Some("disabled"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 89L)
    }

    val onnxFallback = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("blend-primary"),
        DdreProviderProperty -> Some("onnx"),
        DdreAlphaProperty -> Some("1.0"),
        DdreOnnxModelPathProperty -> Some(missingPath),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 89L)
    }

    assertEquals(onnxFallback.posterior.weights, baseline.posterior.weights)
  }

  test("DDRE onnx provider falls back to Bayesian when artifact is experimental") {
    val artifactDir = createSmokeArtifactDir()

    val baseline = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("off"),
        DdreProviderProperty -> Some("disabled"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 97L)
    }

    val onnxFallback = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("blend-primary"),
        DdreProviderProperty -> Some("onnx"),
        DdreAlphaProperty -> Some("1.0"),
        DdreOnnxArtifactDirProperty -> Some(artifactDir.toString),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 97L)
    }

    assertEquals(onnxFallback.posterior.weights, baseline.posterior.weights)
  }

  test("DDRE onnx provider executes smoke artifact only with explicit experimental opt-in") {
    val artifactDir = createSmokeArtifactDir()

    val baseline = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("off"),
        DdreProviderProperty -> Some("disabled"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 101L)
    }

    val onnx = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("blend-primary"),
        DdreProviderProperty -> Some("onnx"),
        DdreAlphaProperty -> Some("1.0"),
        DdreOnnxArtifactDirProperty -> Some(artifactDir.toString),
        DdreOnnxAllowExperimentalProperty -> Some("true"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 101L)
    }

    val onnxVsBaselineL1 = l1Distance(baseline, onnx)
    assert(
      onnxVsBaselineL1 > 1e-6,
      s"expected ONNX path to produce a non-fallback posterior; l1=$onnxVsBaselineL1"
    )
  }
