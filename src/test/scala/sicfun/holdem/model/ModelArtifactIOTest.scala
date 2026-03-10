package sicfun.holdem.model
import sicfun.holdem.types.*

import munit.FunSuite
import sicfun.core.Card

import java.nio.file.{Files, Path}
import scala.jdk.CollectionConverters.*

class ModelArtifactIOTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private val hand1 = hole("Ah", "Kh")
  private val hand2 = hole("7c", "2d")
  private val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
  private val state = GameState(
    street = Street.Flop,
    board = board,
    pot = 20.0,
    toCall = 10.0,
    position = Position.Button,
    stackSize = 200.0,
    betHistory = Vector.empty
  )

  private def trainingData(size: Int): Seq[(GameState, HoleCards, PokerAction)] =
    Seq.fill(size)(Seq(
      (state, hand1, PokerAction.Raise(20.0)),
      (state, hand2, PokerAction.Fold)
    )).flatten

  test("artifact save/load roundtrip preserves model and metadata") {
    val artifact = PokerActionModel.trainVersioned(
      trainingData = trainingData(25),
      learningRate = 0.1,
      iterations = 400,
      maxMeanBrierScore = 0.8,
      failOnGate = true,
      modelId = "roundtrip-v1",
      schemaVersion = "schema-v1",
      source = "unit-test",
      trainedAtEpochMillis = 1111L
    )

    val dir = Files.createTempDirectory("sicfun-model-artifact-")
    try
      PokerActionModelArtifactIO.save(dir, artifact)
      val loaded = PokerActionModelArtifactIO.load(dir)

      assertEquals(loaded.version, artifact.version)
      assertEquals(loaded.calibration, artifact.calibration)
      assertEquals(loaded.gate, artifact.gate)
      assertEquals(loaded.trainingSampleCount, artifact.trainingSampleCount)
      assertEquals(loaded.evaluationSampleCount, artifact.evaluationSampleCount)
      assertEquals(loaded.evaluationStrategy, artifact.evaluationStrategy)
      assertEquals(loaded.validationFraction, artifact.validationFraction)
      assertEquals(loaded.splitSeed, artifact.splitSeed)
      assertEquals(loaded.retiredAtEpochMillis, None)
      assertEquals(loaded.retirementReason, None)

      val originalProbs = artifact.model.categoryProbabilities(state, hand1)
      val loadedProbs = loaded.model.categoryProbabilities(state, hand1)
      assertEquals(originalProbs.length, loadedProbs.length)
      originalProbs.zip(loadedProbs).foreach { case (a, b) =>
        assertEqualsDouble(a, b, 1e-12)
      }
    finally
      deleteRecursively(dir)
  }

  test("artifact save/load preserves retirement state") {
    val artifact = PokerActionModel.trainVersioned(
      trainingData = trainingData(10),
      learningRate = 0.1,
      iterations = 300,
      maxMeanBrierScore = 1.0,
      failOnGate = true,
      modelId = "retire-v1",
      source = "unit-test",
      trainedAtEpochMillis = 2000L
    ).retire(atEpochMillis = 3000L, reason = "degraded in production")

    val dir = Files.createTempDirectory("sicfun-model-artifact-retired-")
    try
      PokerActionModelArtifactIO.save(dir, artifact)
      val loaded = PokerActionModelArtifactIO.load(dir)

      assert(loaded.isRetired)
      assert(!loaded.isActive)
      assertEquals(loaded.retiredAtEpochMillis, Some(3000L))
      assertEquals(loaded.retirementReason, Some("degraded in production"))
    finally
      deleteRecursively(dir)
  }

  private def deleteRecursively(path: Path): Unit =
    if Files.exists(path) then
      val stream = Files.walk(path)
      try
        val paths = stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse
        paths.foreach(Files.deleteIfExists)
      finally stream.close()
