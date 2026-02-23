package sicfun.holdem

import munit.FunSuite
import sicfun.core.{BayesianRange, Card, CollapseMetrics, DiscreteDistribution, MultinomialLogistic}

class PokerPipelineTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

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
      val actual = catIndex(PokerAction.categoryOf(action))
      (probs, actual)
    }
    val brier = CollapseMetrics.meanBrierScore(predictions)
    assert(brier < 1.0, s"expected Brier score < 1.0 on training data, got $brier")
  }

  test("PokerActionModel rejects mismatched logistic and categoryIndex") {
    val logistic = MultinomialLogistic.zeros(3, PokerFeatures.dimension)
    intercept[IllegalArgumentException] {
      PokerActionModel(logistic, PokerActionModel.defaultCategoryIndex) // 3 classes vs 4 categories
    }
  }
