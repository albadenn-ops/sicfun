package sicfun.core

import munit.FunSuite
import sicfun.holdem.types.*
import sicfun.holdem.model.*

/** Tests for [[BayesianRange]] sequential Bayesian inference over discrete hypothesis spaces.
  *
  * Validates the core Bayesian update mechanics:
  *  - A single update shifts probability mass toward the hypothesis consistent with
  *    the observed action (e.g., observing a raise increases P(strong hand)).
  *  - The posterior remains normalized (probabilities sum to 1) after updates.
  *  - Sequential updates via `updateAll` accumulate evidence and progressively narrow
  *    the distribution (repeated raises concentrate belief on the strong hand).
  *  - An empty observation sequence leaves the prior unchanged.
  *  - `BayesianRange.uniform` initializes equal weights across all hypotheses.
  *
  * Uses a trained PokerActionModel on a T-9-8 flop where QJ (made straight) maps to
  * Raise and 72o (junk) maps to Fold.
  */
class BayesianTest extends FunSuite:
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

  // On this flop, QJ is a made straight and should dominate raise likelihood.
  private val strongHand = hole("Qs", "Jd")
  private val weakHand = hole("7c", "2d")

  // Real production trained model: strongHand → Raise, weakHand → Fold
  private lazy val trainedModel: PokerActionModel =
    val data = Seq.fill(30)(Seq(
      (state, strongHand, PokerAction.Raise(20.0)),
      (state, weakHand, PokerAction.Fold)
    )).flatten
    PokerActionModel.train(data, learningRate = 0.1, iterations = 1000, l2Lambda = 0.0)

  test("single update shifts probability toward hypothesis consistent with action") {
    val prior = BayesianRange(DiscreteDistribution(Map(strongHand -> 0.5, weakHand -> 0.5)))
    val (posterior, evidence) = prior.update(PokerAction.Raise(20.0), state, trainedModel)
    assert(posterior.distribution.probabilityOf(strongHand) > 0.5)
    assert(evidence > 0.0)
  }

  test("update preserves normalization") {
    val prior = BayesianRange(DiscreteDistribution(Map(strongHand -> 0.5, weakHand -> 0.5)))
    val (posterior, _) = prior.update(PokerAction.Raise(20.0), state, trainedModel)
    val total = posterior.distribution.weights.values.sum
    assert(math.abs(total - 1.0) < 1e-9)
  }

  test("updateAll applies sequential updates and returns log-evidence") {
    val prior = BayesianRange(DiscreteDistribution(Map(strongHand -> 0.5, weakHand -> 0.5)))
    val actions = Seq(
      (PokerAction.Raise(20.0), state),
      (PokerAction.Raise(20.0), state),
      (PokerAction.Raise(20.0), state)
    )
    val (posterior, logEvidence) = prior.updateAll(actions, trainedModel)
    assert(posterior.distribution.probabilityOf(strongHand) > 0.8)
    assert(logEvidence < 0.0, "log-evidence should be negative (evidence < 1)")
  }

  test("updateAll with empty sequence returns prior unchanged") {
    val prior = BayesianRange(DiscreteDistribution(Map(strongHand -> 0.5, weakHand -> 0.5)))
    val (posterior, logEvidence) = prior.updateAll(Seq.empty, trainedModel)
    assertEquals(posterior.distribution.probabilityOf(strongHand), 0.5)
    assertEquals(logEvidence, 0.0)
  }

  test("BayesianRange.uniform creates equal weights") {
    val range = BayesianRange.uniform(Seq(strongHand, weakHand, hole("Ah", "Kh")))
    val probs = range.distribution.weights.values.toSeq
    probs.foreach(p => assert(math.abs(p - 1.0 / 3.0) < 1e-9))
  }
