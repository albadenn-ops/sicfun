package sicfun.holdem.provider
import sicfun.holdem.types.*
import sicfun.holdem.engine.*
import sicfun.holdem.io.*
import sicfun.holdem.model.*
import sicfun.holdem.gpu.*
import sicfun.holdem.equity.*

import munit.FunSuite
import sicfun.core.Card

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths, StandardCopyOption}
import scala.jdk.CollectionConverters.*
import scala.util.Random

class HoldemDdreOfflineGateTest extends FunSuite:
  private val DdreModeProperty = "sicfun.ddre.mode"
  private val DdreProviderProperty = "sicfun.ddre.provider"
  private val DdreAlphaProperty = "sicfun.ddre.alpha"
  private val DdreOnnxArtifactDirProperty = "sicfun.ddre.onnx.artifactDir"
  private val PreflopBackendProperty = "sicfun.holdem.preflopEquityBackend"

  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card token: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def ddreSmokeOnnxModelPath: String =
    Option(getClass.getClassLoader.getResource("sicfun/ddre/ddre-smoke-sqrt.onnx")) match
      case Some(url) => Paths.get(url.toURI).toString
      case None => fail("missing test resource: sicfun/ddre/ddre-smoke-sqrt.onnx")

  private def createExperimentalArtifactDir(): Path =
    val dir = Files.createTempDirectory("sicfun-ddre-offline-gate-artifact-")
    val modelTarget = dir.resolve("ddre-smoke-sqrt.onnx")
    Files.copy(Paths.get(ddreSmokeOnnxModelPath), modelTarget, StandardCopyOption.REPLACE_EXISTING)
    HoldemDdreArtifactIO.save(
      dir,
      HoldemDdreArtifactIO.OnnxArtifact(
        artifactId = "ddre-offline-gate-smoke",
        source = "HoldemDdreOfflineGateTest",
        createdAtEpochMillis = 1_800_000_000_000L,
        modelFile = modelTarget.getFileName.toString,
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
        notes = Some("experimental smoke artifact for offline gate tests")
      )
    )
    dir

  private def writeDataset(path: Path): Unit =
    val hero = hole("Ac", "Kd")
    val strong = hole("Ah", "Ad")
    val medium = hole("Qs", "Qh")
    val strongId = HoleCardsIndex.idOf(strong)
    val mediumId = HoleCardsIndex.idOf(medium)
    val header =
      "hand\ttableId\tdecisionIndex\tstreet\tboard\tpotBefore\ttoCall\theroPosition\tvillainPosition\theroStackBefore\tvillainStackBefore\tbetHistory\tvillainObservations\theroHole\tvillainHole\tbayesLogEvidence\tpriorSparse\tbayesPosteriorSparse"
    val row =
      Vector(
        "1",
        "1",
        "1",
        "Preflop",
        "-",
        "6.0",
        "2.0",
        "Button",
        "BigBlind",
        "98.0",
        "98.0",
        "-",
        "st=Preflop,a=raise:25.000,pot=6.0,call=2.0,pos=BigBlind,board=-,stack=98.0,history=-",
        hero.toToken,
        strong.toToken,
        "-0.2231435513142097",
        s"$strongId:0.64|$mediumId:0.36",
        s"$strongId:0.80|$mediumId:0.20"
      ).mkString("\t")
    Files.write(path, Vector(header, row).asJava, StandardCharsets.UTF_8)

  private def withSystemProperties[A](updates: Seq[(String, Option[String])])(thunk: => A): A =
    TestSystemPropertyScope.withSystemProperties(updates) {
      HoldemDdreNativeRuntime.resetLoadCacheForTests()
      try thunk
      finally HoldemDdreNativeRuntime.resetLoadCacheForTests()
    }

  private def runPosterior(
      seed: Long,
      updates: Seq[(String, Option[String])]
  ): PosteriorInferenceResult =
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
    withSystemProperties(updates) {
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
    }

  private def l1Distance(a: PosteriorInferenceResult, b: PosteriorInferenceResult): Double =
    a.posterior.weights.keysIterator.map { hand =>
      math.abs(a.posterior.probabilityOf(hand) - b.posterior.probabilityOf(hand))
    }.sum

  test("offline gate upgrades artifact metadata for decision-driving use") {
    val root = Files.createTempDirectory("sicfun-ddre-offline-gate-")
    try
      val artifactDir = createExperimentalArtifactDir()
      val datasetPath = root.resolve("ddre-training.tsv")
      writeDataset(datasetPath)

      val baseline = runPosterior(
        seed = 11L,
        Seq(
          DdreModeProperty -> Some("off"),
          DdreProviderProperty -> Some("disabled"),
          PreflopBackendProperty -> Some("cpu")
        )
      )

      val fallback = runPosterior(
        seed = 11L,
        Seq(
          DdreModeProperty -> Some("blend-primary"),
          DdreProviderProperty -> Some("onnx"),
          DdreAlphaProperty -> Some("1.0"),
          DdreOnnxArtifactDirProperty -> Some(artifactDir.toString),
          PreflopBackendProperty -> Some("cpu")
        )
      )
      assertEquals(fallback.posterior.weights, baseline.posterior.weights)

      val summary = HoldemDdreOfflineGate.run(
        Array(
          s"--dataset=$datasetPath",
          s"--artifactDir=$artifactDir",
          "--minSamples=1",
          "--maxMeanNll=10.0",
          "--maxMeanKlVsBayes=10.0",
          "--maxBlockerViolationRate=0.0",
          "--maxFailureRate=0.0",
          "--maxP95LatencyMillis=2000.0"
        )
      )
      assert(summary.isRight, s"offline gate failed: $summary")
      assert(summary.toOption.exists(_.gatePass), s"expected gate to pass: $summary")

      val artifact = HoldemDdreArtifactIO.load(artifactDir).toOption.getOrElse(fail("expected artifact metadata"))
      assertEquals(artifact.validationStatus, "validated")
      assert(artifact.decisionDrivingAllowed)

      val admitted = runPosterior(
        seed = 11L,
        Seq(
          DdreModeProperty -> Some("blend-primary"),
          DdreProviderProperty -> Some("onnx"),
          DdreAlphaProperty -> Some("1.0"),
          DdreOnnxArtifactDirProperty -> Some(artifactDir.toString),
          PreflopBackendProperty -> Some("cpu")
        )
      )
      val l1 = l1Distance(baseline, admitted)
      assert(l1 > 1e-6, s"expected validated artifact to drive non-Bayesian posterior; l1=$l1")
    finally
      Files.deleteIfExists(root.resolve("ddre-training.tsv"))
  }

  test("offline gate rejects invalid option tokens and invalid booleans") {
    val invalidToken = HoldemDdreOfflineGate.run(Array("--dataset=foo.tsv", "oops"))
    assertEquals(invalidToken, Left("invalid argument 'oops'; expected --key=value"))

    val invalidBoolean = HoldemDdreOfflineGate.run(Array(
      "--dataset=foo.tsv",
      "--artifactDir=bar",
      "--writeMetadata=maybe"
    ))
    assertEquals(invalidBoolean, Left("--writeMetadata must be a boolean (true/false)"))
  }
