package sicfun.core

import munit.FunSuite
import sicfun.holdem.types.*
import sicfun.holdem.model.*

class ActionMetricsTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

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

  private val actions: Seq[PokerAction] = Seq(
    PokerAction.Fold, PokerAction.Check, PokerAction.Call, PokerAction.Raise(20.0)
  )

  // On this flop, QJ is already a made straight, giving a strong separation signal.
  private val hand1 = hole("Qs", "Jd")
  private val hand2 = hole("7c", "2d")
  private val hand3 = hole("Ah", "Kh")
  private val hand4 = hole("9c", "8c")

  private val uniformPosterior = DiscreteDistribution.uniform(Seq(hand1, hand2, hand3, hand4))

  // Real production uniform model: returns 0.25 for every action category
  private val uniformModel: PokerActionModel = PokerActionModel.uniform

  // Real production trained model on strongly separable data
  private lazy val trainedModel: PokerActionModel =
    val data = Seq.fill(100)(Seq(
      (state, hand1, PokerAction.Raise(20.0)),
      (state, hand2, PokerAction.Fold)
    )).flatten
    PokerActionModel.train(data, learningRate = 0.1, iterations = 1500, l2Lambda = 0.0)

  private val trainedPosterior = DiscreteDistribution.uniform(Seq(hand1, hand2))

  test("uniform model yields action entropy = log2(numActions)") {
    val h = ActionMetrics.actionEntropy(state, uniformPosterior, uniformModel, actions)
    assertEqualsDouble(h, 2.0, 1e-9) // log2(4) = 2
  }

  test("trained model conditional entropy is well below marginal entropy") {
    val h = ActionMetrics.actionEntropy(state, trainedPosterior, trainedModel, actions)
    val ce = ActionMetrics.conditionalActionEntropy(state, trainedPosterior, trainedModel, actions)
    assert(ce < h,
      s"conditional entropy ($ce) should be smaller than marginal entropy ($h) for separable data")
    assert(ce < 1.0,
      s"conditional entropy ($ce) should be well below maximum for a strongly trained model")
  }

  test("mutual information is non-negative") {
    val mi = ActionMetrics.mutualInformation(state, uniformPosterior, uniformModel, actions)
    assert(mi >= -1e-9, s"mutual information should be non-negative, got $mi")
  }

  test("trained model mutual information is positive and close to action entropy") {
    val h = ActionMetrics.actionEntropy(state, trainedPosterior, trainedModel, actions)
    val mi = ActionMetrics.mutualInformation(state, trainedPosterior, trainedModel, actions)
    assert(mi > 0.0,
      s"expected positive mutual information for separable data, got $mi")
    assert(mi > h * 0.2,
      s"mutual information ($mi) should capture a meaningful fraction of action entropy ($h)")
  }

  test("uniform model yields zero mutual information") {
    val mi = ActionMetrics.mutualInformation(state, uniformPosterior, uniformModel, actions)
    assertEqualsDouble(mi, 0.0, 1e-9)
  }
