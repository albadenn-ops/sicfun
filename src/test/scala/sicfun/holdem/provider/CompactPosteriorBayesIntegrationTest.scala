package sicfun.holdem.provider

import munit.FunSuite
import sicfun.core.{DiscreteDistribution, MultinomialLogistic, Prob}
import Prob.*
import sicfun.holdem.types.*
import sicfun.holdem.model.*
import sicfun.holdem.equity.HoldemEquity

/** Integration tests for the compact posterior representation in [[HoldemBayesProvider]].
  *
  * The compact posterior (`HoldemEquity.CompactPosterior`) is a fixed-size array
  * representation of the posterior distribution, designed for hot-path equity
  * computation without `Map` overhead. These tests verify that:
  *
  *  - The compact posterior included in `UpdateResult` is non-empty and matches
  *    the full `DiscreteDistribution` posterior entry-by-entry.
  *  - Equity computed via the compact posterior matches equity computed via the
  *    full Map-based posterior to within 1e-3.
  *  - Empty observations (no updates) produce a compact posterior matching the
  *    normalised prior.
  *
  * All tests force the Scala provider to isolate compact-posterior construction
  * from native backend availability.
  */
class CompactPosteriorBayesIntegrationTest extends FunSuite:
  /** Forces the Scala Bayesian provider and resets auto-selection state for test isolation. */
  private def withScalaBayesProvider[A](thunk: => A): A =
    TestSystemPropertyScope.withSystemProperties(Seq("sicfun.bayes.provider" -> Some("scala"))) {
      HoldemBayesProvider.resetAutoProviderForTests()
      try thunk
      finally HoldemBayesProvider.resetAutoProviderForTests()
    }

  private def card(token: String) =
    sicfun.core.Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  /** Builds a simple action model with small deterministic weights.
    * The weights are non-trivial enough to produce different likelihoods per hypothesis
    * but small enough that no single action dominates overwhelmingly.
    */
  private def trivialActionModel: PokerActionModel =
    val classCount = PokerAction.categories.length
    val featureCount = PokerFeatures.dimension
    val weights = Vector.tabulate(classCount)(c =>
      Vector.tabulate(featureCount)(f => 0.01 * (c + 1) * (f + 1))
    )
    val bias = Vector.tabulate(classCount)(c => 0.05 * c)
    PokerActionModel(
      logistic = MultinomialLogistic(weights, bias),
      categoryIndex = PokerActionModel.defaultCategoryIndex,
      featureDimension = PokerFeatures.dimension
    )

  test("UpdateResult.compact is present and matches posterior") {
    withScalaBayesProvider {
      val h1 = hole("As", "Ks")
      val h2 = hole("Qh", "Jd")
      val h3 = hole("Tc", "9c")
      val prior = DiscreteDistribution(Map(h1 -> 0.4, h2 -> 0.3, h3 -> 0.3))
      val state = GameState(
        street = Street.Preflop,
        board = Board.empty,
        pot = 8.0, toCall = 2.0,
        position = Position.Button,
        stackSize = 100.0,
        betHistory = Vector.empty
      )
      val observations = Seq(PokerAction.Call -> state)

      val result = HoldemBayesProvider.updatePosterior(prior, observations, trivialActionModel)

      // compact must be present
      val compact = result.compact
      assert(compact.size > 0)
      assert(compact.size <= 3)

      // compact.distribution must match result.posterior
      val compactDist = compact.distribution
      prior.support.foreach { hand =>
        assertEqualsDouble(
          compactDist.probabilityOf(hand),
          result.posterior.probabilityOf(hand),
          1e-7
        )
      }
    }
  }

  test("compact posterior produces same equity as Map posterior") {
    withScalaBayesProvider {
      val hero = hole("Ah", "Kh")
      val h1 = hole("Jc", "Jd")
      val h2 = hole("9s", "8s")
      val prior = DiscreteDistribution(Map(h1 -> 0.5, h2 -> 0.5))
      val state = GameState(
        street = Street.Flop,
        board = Board.from(Seq(card("2c"), card("7d"), card("Ts"))),
        pot = 10.0, toCall = 3.0,
        position = Position.Button,
        stackSize = 100.0,
        betHistory = Vector.empty
      )
      val observations = Seq(PokerAction.Raise(6.0) -> state)

      val result = HoldemBayesProvider.updatePosterior(prior, observations, trivialActionModel)
      val b = state.board

      val fromMap = HoldemEquity.equityExactProb(hero, b, result.posterior)
      val fromCompact = HoldemEquity.equityExactProb(hero, b, result.compact)

      assertEqualsDouble(fromCompact.equity, fromMap.equity, 1e-3)
    }
  }

  test("empty observations returns compact matching prior") {
    withScalaBayesProvider {
      val h1 = hole("As", "Ks")
      val prior = DiscreteDistribution(Map(h1 -> 1.0))

      val result = HoldemBayesProvider.updatePosterior(prior, Seq.empty, trivialActionModel)

      assertEquals(result.compact.size, 1)
      assert(math.abs(Prob(result.compact.probWeights(0)).toDouble - 1.0) < 1e-8)
    }
  }
