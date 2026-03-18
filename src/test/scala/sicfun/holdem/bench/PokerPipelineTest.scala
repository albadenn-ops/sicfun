package sicfun.holdem.bench
import sicfun.holdem.types.*
import sicfun.holdem.*
import sicfun.holdem.model.*

import munit.FunSuite
import sicfun.core.{BayesianRange, Card, CollapseMetrics, DiscreteDistribution, MultinomialLogistic}
import sicfun.holdem.bench.BenchSupport.{card, hole}

class PokerPipelineTest extends FunSuite:
  test("uniform model preserves prior") {
    val villainHands = Seq(hole("Ah", "Kh"), hole("7c", "2d"), hole("Qs", "Jd"))
    val prior = DiscreteDistribution.uniform(villainHands)
    val range = BayesianRange(prior)
    val model = PokerActionModel.uniform

    val state = GameState(
      street = Street.Flop,
      board = Board.from(Seq(card("Ts"), card("9h"), card("8d"))),
      pot = 20.0, toCall = 10.0,
      position = Position.Button, stackSize = 200.0,
      betHistory = Vector.empty
    )

    val (posterior, evidence) = range.update(PokerAction.Raise(20.0), state, model)
    assert(evidence > 0.0)
    villainHands.foreach { hand =>
      assert(math.abs(posterior.distribution.probabilityOf(hand) - 1.0 / 3.0) < 1e-6)
    }
  }

  test("trained model shifts posterior based on action-hand correlation") {
    val hand1 = hole("Ah", "Kh")
    val hand2 = hole("7c", "2d")

    val b = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    val state = GameState(Street.Flop, b, pot = 20.0, toCall = 10.0,
      position = Position.Button, stackSize = 200.0, betHistory = Vector.empty)

    val trainingData = Seq.fill(20)(
      Seq(
        (state, hand1, PokerAction.Raise(20.0)),
        (state, hand2, PokerAction.Fold)
      )
    ).flatten

    val model = PokerActionModel.train(trainingData, learningRate = 0.1, iterations = 500)

    val prior = DiscreteDistribution(Map(hand1 -> 0.5, hand2 -> 0.5))
    val range = BayesianRange(prior)

    val (posterior, _) = range.update(PokerAction.Raise(20.0), state, model)
    assert(posterior.distribution.probabilityOf(hand1) > 0.5,
      s"expected hand1 > 0.5, got ${posterior.distribution.probabilityOf(hand1)}")
  }

  test("collapse metrics reflect narrowing after update") {
    val hand1 = hole("Ah", "Kh")
    val hand2 = hole("7c", "2d")
    val hand3 = hole("Qs", "Jd")

    val b = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    val state = GameState(Street.Flop, b, pot = 20.0, toCall = 10.0,
      position = Position.Button, stackSize = 200.0, betHistory = Vector.empty)

    val trainingData = Seq.fill(20)(
      Seq(
        (state, hand1, PokerAction.Raise(20.0)),
        (state, hand2, PokerAction.Fold),
        (state, hand3, PokerAction.Call)
      )
    ).flatten

    val model = PokerActionModel.train(trainingData, learningRate = 0.1, iterations = 500)
    val prior = DiscreteDistribution.uniform(Seq(hand1, hand2, hand3))
    val range = BayesianRange(prior)
    val (posterior, _) = range.update(PokerAction.Raise(20.0), state, model)

    val entropyDrop = CollapseMetrics.entropyReduction(prior, posterior.distribution)
    val kl = CollapseMetrics.klDivergence(prior, posterior.distribution)
    val priorSupport = CollapseMetrics.effectiveSupport(prior)
    val postSupport = CollapseMetrics.effectiveSupport(posterior.distribution)
    val ratio = CollapseMetrics.collapseRatio(prior, posterior.distribution)

    assert(entropyDrop > 0.0, s"expected positive entropy reduction, got $entropyDrop")
    assert(kl > 0.0, s"expected positive KL divergence, got $kl")
    assert(postSupport < priorSupport, s"expected posterior support < prior")
    assert(ratio > 0.0, s"expected positive collapse ratio, got $ratio")
  }

  test("Brier score calibration for trained model") {
    val hand1 = hole("Ah", "Kh")
    val hand2 = hole("7c", "2d")

    val b = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    val state = GameState(Street.Flop, b, pot = 20.0, toCall = 10.0,
      position = Position.Button, stackSize = 200.0, betHistory = Vector.empty)

    val trainingData = Seq.fill(30)(
      Seq(
        (state, hand1, PokerAction.Raise(20.0)),
        (state, hand2, PokerAction.Fold)
      )
    ).flatten

    val model = PokerActionModel.train(trainingData, learningRate = 0.1, iterations = 500)

    val catIndex = PokerActionModel.defaultCategoryIndex
    val predictions = trainingData.map { case (s, h, action) =>
      val features = PokerFeatures.extract(s, h)
      val probs = model.logistic.predict(features.values)
      val actual = catIndex(action.category)
      (probs, actual)
    }
    val brier = CollapseMetrics.meanBrierScore(predictions)
    assert(brier < 1.0, s"expected Brier score < 1.0 on training data, got $brier")
  }

  test("PokerActionModel rejects mismatched logistic and categoryIndex") {
    val logistic = MultinomialLogistic.zeros(3, PokerFeatures.dimension)
    intercept[IllegalArgumentException] {
      PokerActionModel(logistic, PokerActionModel.defaultCategoryIndex, PokerFeatures.dimension) // 3 classes vs 4 categories
    }
  }

  test("PokerActionModel rejects featureDimension mismatch with logistic weights") {
    val logistic = MultinomialLogistic.zeros(4, 8)
    intercept[IllegalArgumentException] {
      PokerActionModel(logistic, PokerActionModel.defaultCategoryIndex, 5) // 8D weights vs 5D claim
    }
  }

  test("categoryProbabilities rejects call on 8D model") {
    val logistic = MultinomialLogistic.zeros(4, FeatureExtractor.dimension)
    val model = PokerActionModel(logistic, PokerActionModel.defaultCategoryIndex, FeatureExtractor.dimension)
    val state = GameState(Street.Flop, Board.from(Seq(card("Ts"), card("9h"), card("8d"))),
      pot = 20.0, toCall = 10.0, position = Position.Button, stackSize = 200.0, betHistory = Vector.empty)
    intercept[IllegalArgumentException] {
      model.categoryProbabilities(state, hole("Ah", "Kh"))
    }
  }

  test("predictFromFeatures works with 8D model") {
    val logistic = MultinomialLogistic.zeros(4, FeatureExtractor.dimension)
    val model = PokerActionModel(logistic, PokerActionModel.defaultCategoryIndex, FeatureExtractor.dimension)
    val probs = model.predictFromFeatures(Vector.fill(FeatureExtractor.dimension)(0.5))
    assertEquals(probs.length, 4)
    assertEqualsDouble(probs.head, 0.25, 1e-6)
  }

  test("calibration summary includes baselines and trained model beats uniform") {
    val hand1 = hole("Ah", "Kh")
    val hand2 = hole("7c", "2d")

    val b = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    val state = GameState(Street.Flop, b, pot = 20.0, toCall = 10.0,
      position = Position.Button, stackSize = 200.0, betHistory = Vector.empty)

    val trainingData = Seq.fill(30)(
      Seq(
        (state, hand1, PokerAction.Raise(20.0)),
        (state, hand2, PokerAction.Fold)
      )
    ).flatten

    val model = PokerActionModel.train(trainingData, learningRate = 0.1, iterations = 500)
    val summary = PokerActionModel.calibrationSummary(model, trainingData)

    assert(summary.uniformBaselineBrier > 0.0,
      s"uniform baseline should be positive, got ${summary.uniformBaselineBrier}")
    assert(summary.majorityBaselineBrier > 0.0,
      s"majority baseline should be positive, got ${summary.majorityBaselineBrier}")
    assert(summary.meanBrierScore < summary.uniformBaselineBrier,
      s"trained model (${summary.meanBrierScore}) should beat uniform baseline (${summary.uniformBaselineBrier})")
    assert(summary.brierSkillScore > 0.0,
      s"brierSkillScore should be positive, got ${summary.brierSkillScore}")
  }

  test("evaluation report: separable data yields high accuracy and precision/recall") {
    val hand1 = hole("Ah", "Kh")
    val hand2 = hole("7c", "2d")

    val b = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    val state = GameState(Street.Flop, b, pot = 20.0, toCall = 10.0,
      position = Position.Button, stackSize = 200.0, betHistory = Vector.empty)

    val trainingData = Seq.fill(30)(
      Seq(
        (state, hand1, PokerAction.Raise(20.0)),
        (state, hand2, PokerAction.Fold)
      )
    ).flatten

    val model = PokerActionModel.train(trainingData, learningRate = 0.1, iterations = 500)
    val report = PokerActionModel.evaluate(model, trainingData)

    assert(report.overallAccuracy > 0.8,
      s"expected accuracy > 0.8 on separable data, got ${report.overallAccuracy}")

    val matrixTotal = report.confusionMatrix.map(_.sum).sum
    assertEquals(matrixTotal, trainingData.length)

    report.categoryMetrics.foreach { cm =>
      assert(cm.precision >= 0.0 && cm.precision <= 1.0,
        s"precision out of range for ${cm.category}: ${cm.precision}")
      assert(cm.recall >= 0.0 && cm.recall <= 1.0,
        s"recall out of range for ${cm.category}: ${cm.recall}")
    }
  }

  test("evaluation report: uniform model accuracy is ~0.25") {
    val hand1 = hole("Ah", "Kh")
    val hand2 = hole("7c", "2d")
    val checkState = GameState(Street.Flop,
      Board.from(Seq(card("Ts"), card("9h"), card("8d"))),
      pot = 20.0, toCall = 0.0,
      position = Position.Button, stackSize = 200.0, betHistory = Vector.empty)
    val callState = checkState.copy(toCall = 10.0)

    val evalData = Seq(
      (callState, hand1, PokerAction.Fold),
      (checkState, hand1, PokerAction.Check),
      (callState, hand2, PokerAction.Call),
      (callState, hand2, PokerAction.Raise(20.0))
    )

    val model = PokerActionModel.uniform
    val report = PokerActionModel.evaluate(model, evalData)
    // Uniform model picks class 0 (or whichever has max among equal probs).
    // Accuracy should be low.
    assert(report.overallAccuracy <= 0.5,
      s"uniform model accuracy should be <= 0.5, got ${report.overallAccuracy}")
  }

  test("stratified k-fold CV returns k fold scores") {
    val hand1 = hole("Ah", "Kh")
    val hand2 = hole("7c", "2d")

    val b = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    val state = GameState(Street.Flop, b, pot = 20.0, toCall = 10.0,
      position = Position.Button, stackSize = 200.0, betHistory = Vector.empty)

    val data = Seq.fill(20)(
      Seq(
        (state, hand1, PokerAction.Raise(20.0)),
        (state, hand2, PokerAction.Fold)
      )
    ).flatten

    val result = PokerActionModel.stratifiedKFoldCV(data, k = 5, iterations = 100)
    assertEquals(result.foldCount, 5)
    assertEquals(result.foldBrierScores.length, 5)
    assert(result.meanBrierScore >= 0.0, s"mean Brier should be non-negative: ${result.meanBrierScore}")
    assert(result.stdBrierScore >= 0.0, s"std Brier should be non-negative: ${result.stdBrierScore}")
  }
