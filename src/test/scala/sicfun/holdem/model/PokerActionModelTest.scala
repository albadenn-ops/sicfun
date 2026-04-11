package sicfun.holdem.model
import sicfun.holdem.types.*

import munit.FunSuite
import sicfun.core.{Card, MultinomialLogistic}

/**
 * Comprehensive unit tests for the [[PokerActionModel]] and related domain types.
 *
 * Test coverage includes:
 *   - '''PokerActionModel.uniform''': Correct category index, feature dimension,
 *     uniform detection, and approximately equal predictions
 *   - '''Construction validation''': Mismatched sizes, out-of-range indices, wrong dimensions
 *   - '''predictFromFeatures''': Wrong feature count rejection, valid probability output
 *   - '''categoryProbabilities''': Non-standard dimension rejection, sum-to-1 invariant
 *   - '''likelihood''': MinLikelihood floor, unknown category rejection
 *   - '''isEffectivelyUniform''': False for trained models
 *   - '''train''': Correct dimension, trained model preferences
 *   - '''defaultCategoryIndex''': Complete coverage, contiguous indices
 *   - '''CalibrationSummary''': Input validation, brierSkillScore computation
 *   - '''CalibrationGate''': Threshold comparison semantics
 *   - '''CategoryMetrics''': Precision, recall, F1 edge cases
 *   - '''ModelVersion''': Input validation for all fields
 *   - '''TrainedPokerActionModel''': Lifecycle state transitions, retirement validation
 *   - '''CrossValidationResult''': Data structure integrity
 */
class PokerActionModelTest extends FunSuite:
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

  // ---- PokerActionModel.uniform ----

  test("uniform model has correct category index") {
    val model = PokerActionModel.uniform
    assertEquals(model.categoryIndex, PokerActionModel.defaultCategoryIndex)
    assertEquals(model.categoryIndex.size, PokerAction.categories.length)
  }

  test("uniform model has feature dimension matching PokerFeatures.dimension") {
    val model = PokerActionModel.uniform
    assertEquals(model.featureDimension, PokerFeatures.dimension)
  }

  test("uniform model is effectively uniform") {
    val model = PokerActionModel.uniform
    assert(model.isEffectivelyUniform)
  }

  test("uniform model predicts approximately equal probabilities") {
    val model = PokerActionModel.uniform
    val probs = model.categoryProbabilities(state, hand1)
    assertEquals(probs.length, PokerAction.categories.length)
    val expected = 1.0 / PokerAction.categories.length
    probs.foreach { p =>
      assertEqualsDouble(p, expected, 1e-6)
    }
  }

  // ---- PokerActionModel construction validation ----

  test("construction rejects mismatched categoryIndex size and logistic class count") {
    val logistic = MultinomialLogistic.zeros(4, 5)
    val badIndex = Map(PokerAction.Category.Fold -> 0, PokerAction.Category.Call -> 1)
    intercept[IllegalArgumentException] {
      PokerActionModel(logistic, badIndex, 5)
    }
  }

  test("construction rejects out-of-range categoryIndex values") {
    val logistic = MultinomialLogistic.zeros(4, 5)
    val badIndex = PokerAction.categories.zipWithIndex.toMap.updated(PokerAction.Category.Fold, 10)
    intercept[IllegalArgumentException] {
      PokerActionModel(logistic, badIndex, 5)
    }
  }

  test("construction rejects mismatched featureDimension and logistic weight dimension") {
    val logistic = MultinomialLogistic.zeros(4, 5)
    intercept[IllegalArgumentException] {
      PokerActionModel(logistic, PokerActionModel.defaultCategoryIndex, 8)
    }
  }

  // ---- predictFromFeatures ----

  test("predictFromFeatures: rejects wrong feature count") {
    val model = PokerActionModel.uniform
    intercept[IllegalArgumentException] {
      model.predictFromFeatures(Vector(0.1, 0.2, 0.3)) // too few
    }
  }

  test("predictFromFeatures: returns valid probability distribution") {
    val model = PokerActionModel.uniform
    val features = Vector.fill(PokerFeatures.dimension)(0.5)
    val probs = model.predictFromFeatures(features)
    assertEquals(probs.length, PokerAction.categories.length)
    assertEqualsDouble(probs.sum, 1.0, 1e-6)
    probs.foreach(p => assert(p >= 0.0, s"negative probability: $p"))
  }

  // ---- categoryProbabilities ----

  test("categoryProbabilities: rejects model with non-standard feature dimension") {
    val logistic = MultinomialLogistic.zeros(4, 8)
    val model = PokerActionModel(logistic, PokerActionModel.defaultCategoryIndex, 8)
    intercept[IllegalArgumentException] {
      model.categoryProbabilities(state, hand1)
    }
  }

  test("categoryProbabilities: probabilities sum to 1") {
    val model = PokerActionModel.uniform
    val probs = model.categoryProbabilities(state, hand1)
    assertEqualsDouble(probs.sum, 1.0, 1e-6)
  }

  // ---- likelihood ----

  test("likelihood: returns at least MinLikelihood for any action") {
    val model = PokerActionModel.uniform
    val lk = model.likelihood(PokerAction.Fold, state, hand1)
    assert(lk >= 1e-6, s"likelihood $lk below MinLikelihood")
  }

  test("likelihood: rejects unknown category index") {
    // Build a model with a partial categoryIndex that is missing Raise
    val logistic = MultinomialLogistic.zeros(3, 5)
    val partialIndex = Map(
      PokerAction.Category.Fold -> 0,
      PokerAction.Category.Check -> 1,
      PokerAction.Category.Call -> 2
    )
    val model = PokerActionModel(logistic, partialIndex, 5)
    intercept[IllegalArgumentException] {
      model.likelihood(PokerAction.Raise(20.0), state, hand1)
    }
  }

  // ---- isEffectivelyUniform ----

  test("isEffectivelyUniform: false for trained model") {
    val data = Seq.fill(20)(Seq(
      (state, hand1, PokerAction.Raise(20.0)),
      (state, hand2, PokerAction.Fold)
    )).flatten
    val model = PokerActionModel.train(data, learningRate = 0.1, iterations = 200)
    assert(!model.isEffectivelyUniform)
  }

  // ---- PokerActionModel.train ----

  test("train: produces a model with correct dimension and category index") {
    val data = Seq.fill(5)(Seq(
      (state, hand1, PokerAction.Raise(20.0)),
      (state, hand2, PokerAction.Fold)
    )).flatten
    val model = PokerActionModel.train(data, learningRate = 0.1, iterations = 100)
    assertEquals(model.featureDimension, PokerFeatures.dimension)
    assertEquals(model.categoryIndex, PokerActionModel.defaultCategoryIndex)
  }

  test("train: trained model gives higher probability to observed actions") {
    val data = Seq.fill(30)(Seq(
      (state, hand1, PokerAction.Raise(20.0)),
      (state, hand2, PokerAction.Fold)
    )).flatten
    val model = PokerActionModel.train(data, learningRate = 0.1, iterations = 500)
    val raiseIdx = model.categoryIndex(PokerAction.Category.Raise)
    val foldIdx = model.categoryIndex(PokerAction.Category.Fold)

    val probs1 = model.categoryProbabilities(state, hand1)
    val probs2 = model.categoryProbabilities(state, hand2)

    assert(probs1(raiseIdx) > probs1(foldIdx),
      s"hand1 should prefer Raise but got Raise=${probs1(raiseIdx)}, Fold=${probs1(foldIdx)}")
    assert(probs2(foldIdx) > probs2(raiseIdx),
      s"hand2 should prefer Fold but got Fold=${probs2(foldIdx)}, Raise=${probs2(raiseIdx)}")
  }

  // ---- defaultCategoryIndex ----

  test("defaultCategoryIndex covers all categories") {
    val idx = PokerActionModel.defaultCategoryIndex
    PokerAction.categories.foreach { cat =>
      assert(idx.contains(cat), s"missing category $cat in defaultCategoryIndex")
    }
    assertEquals(idx.size, PokerAction.categories.length)
  }

  test("defaultCategoryIndex values are contiguous from 0") {
    val idx = PokerActionModel.defaultCategoryIndex
    assertEquals(idx.values.toSet, (0 until PokerAction.categories.length).toSet)
  }

  // ---- CalibrationSummary ----

  test("CalibrationSummary: rejects negative meanBrierScore") {
    intercept[IllegalArgumentException] {
      CalibrationSummary(meanBrierScore = -0.1, sampleCount = 10)
    }
  }

  test("CalibrationSummary: rejects zero sampleCount") {
    intercept[IllegalArgumentException] {
      CalibrationSummary(meanBrierScore = 0.5, sampleCount = 0)
    }
  }

  test("CalibrationSummary: brierSkillScore is NaN when uniform baseline not set") {
    val summary = CalibrationSummary(meanBrierScore = 0.3, sampleCount = 10)
    assert(summary.brierSkillScore.isNaN)
  }

  test("CalibrationSummary: brierSkillScore computes correctly") {
    val summary = CalibrationSummary(
      meanBrierScore = 0.3,
      sampleCount = 10,
      uniformBaselineBrier = 0.6
    )
    assertEqualsDouble(summary.brierSkillScore, 1.0 - (0.3 / 0.6), 1e-9)
  }

  test("CalibrationSummary: perfect model gets brierSkillScore = 1.0") {
    val summary = CalibrationSummary(
      meanBrierScore = 0.0,
      sampleCount = 10,
      uniformBaselineBrier = 0.5
    )
    assertEqualsDouble(summary.brierSkillScore, 1.0, 1e-9)
  }

  // ---- CalibrationGate ----

  test("CalibrationGate: rejects negative maxMeanBrierScore") {
    intercept[IllegalArgumentException] {
      CalibrationGate(maxMeanBrierScore = -0.01)
    }
  }

  test("CalibrationGate: passed returns true when score is within threshold") {
    val gate = CalibrationGate(maxMeanBrierScore = 0.5)
    val summary = CalibrationSummary(meanBrierScore = 0.4, sampleCount = 10)
    assert(gate.passed(summary))
  }

  test("CalibrationGate: passed returns false when score exceeds threshold") {
    val gate = CalibrationGate(maxMeanBrierScore = 0.3)
    val summary = CalibrationSummary(meanBrierScore = 0.4, sampleCount = 10)
    assert(!gate.passed(summary))
  }

  test("CalibrationGate: passed returns true at exact threshold") {
    val gate = CalibrationGate(maxMeanBrierScore = 0.4)
    val summary = CalibrationSummary(meanBrierScore = 0.4, sampleCount = 10)
    assert(gate.passed(summary))
  }

  // ---- CategoryMetrics ----

  test("CategoryMetrics: precision with no predictions is 0") {
    val m = CategoryMetrics(PokerAction.Category.Fold, truePositives = 0, falsePositives = 0, falseNegatives = 5)
    assertEqualsDouble(m.precision, 0.0, 1e-9)
  }

  test("CategoryMetrics: recall with no actual positives is 0") {
    val m = CategoryMetrics(PokerAction.Category.Fold, truePositives = 0, falsePositives = 5, falseNegatives = 0)
    assertEqualsDouble(m.recall, 0.0, 1e-9)
  }

  test("CategoryMetrics: perfect classifier has precision, recall, f1 = 1") {
    val m = CategoryMetrics(PokerAction.Category.Fold, truePositives = 10, falsePositives = 0, falseNegatives = 0)
    assertEqualsDouble(m.precision, 1.0, 1e-9)
    assertEqualsDouble(m.recall, 1.0, 1e-9)
    assertEqualsDouble(m.f1, 1.0, 1e-9)
  }

  test("CategoryMetrics: f1 is harmonic mean of precision and recall") {
    val m = CategoryMetrics(PokerAction.Category.Call, truePositives = 5, falsePositives = 3, falseNegatives = 2)
    val p = 5.0 / 8.0
    val r = 5.0 / 7.0
    val expectedF1 = 2.0 * p * r / (p + r)
    assertEqualsDouble(m.precision, p, 1e-9)
    assertEqualsDouble(m.recall, r, 1e-9)
    assertEqualsDouble(m.f1, expectedF1, 1e-9)
  }

  test("CategoryMetrics: f1 is 0 when both precision and recall are 0") {
    val m = CategoryMetrics(PokerAction.Category.Raise, truePositives = 0, falsePositives = 0, falseNegatives = 0)
    assertEqualsDouble(m.f1, 0.0, 1e-9)
  }

  // ---- ModelVersion ----

  test("ModelVersion: rejects empty id") {
    intercept[IllegalArgumentException] {
      ModelVersion(id = "", schemaVersion = "v1", source = "test", trainedAtEpochMillis = 1000L)
    }
  }

  test("ModelVersion: rejects blank schemaVersion") {
    intercept[IllegalArgumentException] {
      ModelVersion(id = "m1", schemaVersion = "  ", source = "test", trainedAtEpochMillis = 1000L)
    }
  }

  test("ModelVersion: rejects empty source") {
    intercept[IllegalArgumentException] {
      ModelVersion(id = "m1", schemaVersion = "v1", source = "", trainedAtEpochMillis = 1000L)
    }
  }

  test("ModelVersion: rejects negative trainedAtEpochMillis") {
    intercept[IllegalArgumentException] {
      ModelVersion(id = "m1", schemaVersion = "v1", source = "test", trainedAtEpochMillis = -1L)
    }
  }

  test("ModelVersion: valid construction succeeds") {
    val v = ModelVersion(id = "m1", schemaVersion = "v1", source = "unit-test", trainedAtEpochMillis = 0L)
    assertEquals(v.id, "m1")
    assertEquals(v.trainedAtEpochMillis, 0L)
  }

  // ---- TrainedPokerActionModel ----

  /** Factory for creating test TrainedPokerActionModel instances with configurable
   * calibration, gate threshold, and lifecycle state.
   */
  private def makeTrainedModel(
      brierScore: Double = 0.3,
      maxBrier: Double = 0.5,
      trainedAt: Long = 1000L,
      retiredAt: Option[Long] = None,
      retirementReason: Option[String] = None
  ): TrainedPokerActionModel =
    TrainedPokerActionModel(
      version = ModelVersion("m1", "v1", "test", trainedAt),
      model = PokerActionModel.uniform,
      calibration = CalibrationSummary(brierScore, 10),
      gate = CalibrationGate(maxBrier),
      trainingSampleCount = 100,
      evaluationSampleCount = 20,
      evaluationStrategy = "holdout-split",
      validationFraction = Some(0.2),
      splitSeed = Some(42L),
      retiredAtEpochMillis = retiredAt,
      retirementReason = retirementReason
    )

  test("TrainedPokerActionModel: gatePassed when brier under threshold") {
    val m = makeTrainedModel(brierScore = 0.3, maxBrier = 0.5)
    assert(m.gatePassed)
  }

  test("TrainedPokerActionModel: gate fails when brier over threshold") {
    val m = makeTrainedModel(brierScore = 0.6, maxBrier = 0.5)
    assert(!m.gatePassed)
  }

  test("TrainedPokerActionModel: isActive when passing gate and not retired") {
    val m = makeTrainedModel(brierScore = 0.3, maxBrier = 0.5)
    assert(m.isActive)
    assert(!m.isRetired)
  }

  test("TrainedPokerActionModel: not active when gate fails") {
    val m = makeTrainedModel(brierScore = 0.6, maxBrier = 0.5)
    assert(!m.isActive)
  }

  test("TrainedPokerActionModel: not active when retired") {
    val m = makeTrainedModel(
      brierScore = 0.3, maxBrier = 0.5,
      retiredAt = Some(2000L), retirementReason = Some("replaced")
    )
    assert(m.isRetired)
    assert(!m.isActive)
  }

  test("TrainedPokerActionModel: retire sets fields and marks as retired") {
    val m = makeTrainedModel(brierScore = 0.3, maxBrier = 0.5, trainedAt = 1000L)
    val retired = m.retire(atEpochMillis = 2000L, reason = "superseded")
    assert(retired.isRetired)
    assertEquals(retired.retiredAtEpochMillis, Some(2000L))
    assertEquals(retired.retirementReason, Some("superseded"))
  }

  test("TrainedPokerActionModel: retire rejects already-retired model") {
    val m = makeTrainedModel(
      retiredAt = Some(2000L), retirementReason = Some("replaced")
    )
    intercept[IllegalArgumentException] {
      m.retire(atEpochMillis = 3000L, reason = "again")
    }
  }

  test("TrainedPokerActionModel: retire rejects timestamp before training") {
    val m = makeTrainedModel(trainedAt = 5000L)
    intercept[IllegalArgumentException] {
      m.retire(atEpochMillis = 4000L, reason = "too early")
    }
  }

  test("TrainedPokerActionModel: retire rejects empty reason") {
    val m = makeTrainedModel()
    intercept[IllegalArgumentException] {
      m.retire(atEpochMillis = 2000L, reason = "  ")
    }
  }

  test("TrainedPokerActionModel: rejects zero trainingSampleCount") {
    intercept[IllegalArgumentException] {
      TrainedPokerActionModel(
        version = ModelVersion("m1", "v1", "test", 1000L),
        model = PokerActionModel.uniform,
        calibration = CalibrationSummary(0.3, 10),
        gate = CalibrationGate(0.5),
        trainingSampleCount = 0,
        evaluationSampleCount = 10,
        evaluationStrategy = "holdout-split",
        validationFraction = Some(0.2),
        splitSeed = Some(1L)
      )
    }
  }

  test("TrainedPokerActionModel: rejects validationFraction out of (0, 1)") {
    intercept[IllegalArgumentException] {
      makeTrainedModel().copy(validationFraction = Some(0.0))
    }
    intercept[IllegalArgumentException] {
      makeTrainedModel().copy(validationFraction = Some(1.0))
    }
  }

  test("TrainedPokerActionModel: rejects retiredAt without retirementReason") {
    intercept[IllegalArgumentException] {
      TrainedPokerActionModel(
        version = ModelVersion("m1", "v1", "test", 1000L),
        model = PokerActionModel.uniform,
        calibration = CalibrationSummary(0.3, 10),
        gate = CalibrationGate(0.5),
        trainingSampleCount = 100,
        evaluationSampleCount = 20,
        evaluationStrategy = "holdout-split",
        validationFraction = Some(0.2),
        splitSeed = Some(1L),
        retiredAtEpochMillis = Some(2000L),
        retirementReason = None
      )
    }
  }

  test("TrainedPokerActionModel: rejects retirementReason without retiredAt") {
    intercept[IllegalArgumentException] {
      TrainedPokerActionModel(
        version = ModelVersion("m1", "v1", "test", 1000L),
        model = PokerActionModel.uniform,
        calibration = CalibrationSummary(0.3, 10),
        gate = CalibrationGate(0.5),
        trainingSampleCount = 100,
        evaluationSampleCount = 20,
        evaluationStrategy = "holdout-split",
        validationFraction = Some(0.2),
        splitSeed = Some(1L),
        retiredAtEpochMillis = None,
        retirementReason = Some("orphaned")
      )
    }
  }

  // ---- CrossValidationResult ----

  test("CrossValidationResult: stores fold scores") {
    val cvr = CrossValidationResult(
      foldBrierScores = Vector(0.3, 0.4, 0.35),
      meanBrierScore = 0.35,
      stdBrierScore = 0.05,
      foldCount = 3
    )
    assertEquals(cvr.foldCount, 3)
    assertEquals(cvr.foldBrierScores.length, 3)
  }
