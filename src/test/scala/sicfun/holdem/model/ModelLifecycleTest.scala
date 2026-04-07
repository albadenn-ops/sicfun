package sicfun.holdem.model
import sicfun.holdem.types.*

import munit.FunSuite
import sicfun.core.{BayesianRange, Card, DiscreteDistribution}

/**
 * Tests for the [[TrainedPokerActionModel]] lifecycle: training, calibration, gating,
 * retirement, and integration with the [[sicfun.core.BayesianRange]] inference engine.
 *
 * Validates:
 *   - [[PokerActionModel.trainVersioned]] produces artifacts with correct metadata,
 *     calibration, and passing quality gates
 *   - Holdout-split vs. external-evaluation strategies are correctly applied
 *   - Brier score gate enforcement (failOnGate=true throws on gate failure)
 *   - Retirement marks the artifact as retired with reason
 *   - Trained models integrate with BayesianRange.update and produce sensible posteriors
 */
class ModelLifecycleTest extends FunSuite:
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

  /** Generates synthetic training data: alternating (hand1=AhKh, Raise) and (hand2=7c2d, Fold) pairs. */
  private def trainingData(size: Int): Seq[(GameState, HoleCards, PokerAction)] =
    Seq.fill(size)(Seq(
      (state, hand1, PokerAction.Raise(20.0)),
      (state, hand2, PokerAction.Fold)
    )).flatten

  test("trainVersioned returns artifact with calibration and passing gate") {
    val artifact = PokerActionModel.trainVersioned(
      trainingData = trainingData(20),
      learningRate = 0.1,
      iterations = 400,
      maxMeanBrierScore = 0.6,
      failOnGate = true,
      modelId = "model-v1",
      schemaVersion = "action-schema-v1",
      source = "unit-test",
      trainedAtEpochMillis = 1000L
    )

    assertEquals(artifact.version.id, "model-v1")
    assertEquals(artifact.version.schemaVersion, "action-schema-v1")
    assertEquals(artifact.version.source, "unit-test")
    assertEquals(artifact.version.trainedAtEpochMillis, 1000L)
    assert(artifact.calibration.meanBrierScore >= 0.0)
    assertEquals(artifact.evaluationStrategy, "holdout-split")
    assertEquals(artifact.validationFraction, Some(0.2))
    assertEquals(artifact.splitSeed, Some(1L))
    assertEquals(artifact.trainingSampleCount + artifact.evaluationSampleCount, 40)
    assertEquals(artifact.calibration.sampleCount, artifact.evaluationSampleCount)
    assert(artifact.evaluationSampleCount < 40, "holdout split must evaluate on a strict subset")
    assert(artifact.gatePassed, s"expected gate to pass, got brier=${artifact.calibration.meanBrierScore}")
    assert(artifact.isActive)
  }

  test("trainVersioned uses external evaluation set when provided") {
    val evalData = Seq(
      (state, hand1, PokerAction.Raise(20.0)),
      (state, hand2, PokerAction.Fold)
    )
    val artifact = PokerActionModel.trainVersioned(
      trainingData = trainingData(10),
      evaluationData = evalData,
      learningRate = 0.1,
      iterations = 200,
      maxMeanBrierScore = 1.0,
      failOnGate = true
    )
    assertEquals(artifact.evaluationStrategy, "external-evaluation")
    assertEquals(artifact.validationFraction, None)
    assertEquals(artifact.splitSeed, None)
    assertEquals(artifact.trainingSampleCount, 20)
    assertEquals(artifact.evaluationSampleCount, 2)
    assertEquals(artifact.calibration.sampleCount, 2)
  }

  test("trainVersioned enforces Brier gate when configured") {
    intercept[IllegalArgumentException] {
      PokerActionModel.trainVersioned(
        trainingData = trainingData(5),
        learningRate = 0.01,
        iterations = 20,
        maxMeanBrierScore = 0.0,
        failOnGate = true
      )
    }
  }

  test("retire marks artifact as retired with reason") {
    val artifact = PokerActionModel.trainVersioned(
      trainingData = trainingData(10),
      learningRate = 0.1,
      iterations = 200,
      maxMeanBrierScore = 1.0,
      failOnGate = true,
      modelId = "model-v2",
      trainedAtEpochMillis = 2000L
    )

    val retired = artifact.retire(atEpochMillis = 3000L, reason = "replaced by better calibration")
    assert(retired.isRetired)
    assert(!retired.isActive)
    assertEquals(retired.retiredAtEpochMillis, Some(3000L))
    assertEquals(retired.retirementReason, Some("replaced by better calibration"))
  }

  test("versioned model integrates with BayesianRange.update") {
    val artifact = PokerActionModel.trainVersioned(
      trainingData = trainingData(20),
      learningRate = 0.1,
      iterations = 500,
      maxMeanBrierScore = 0.8,
      failOnGate = true
    )
    val prior = DiscreteDistribution(Map(hand1 -> 0.5, hand2 -> 0.5))
    val range = BayesianRange(prior)

    val (posterior, evidence) = range.update(PokerAction.Raise(20.0), state, artifact.model)
    assert(evidence > 0.0)
    assert(
      posterior.distribution.probabilityOf(hand1) > 0.5,
      s"expected raise to increase hand1 posterior, got ${posterior.distribution.probabilityOf(hand1)}"
    )
  }
