package sicfun.holdem.strategic
import sicfun.holdem.strategic.types.*
import sicfun.holdem.strategic.state.*

import sicfun.holdem.types.{PokerAction, Street}
import sicfun.core.DiscreteDistribution

class PosteriorAttributedBaselineTest extends munit.FunSuite:

  private val priors: Map[(StrategicClass, PokerAction.Category), Double] = {
    import PokerAction.Category.*
    Map(
      (StrategicClass.Value, Fold) -> 0.05, (StrategicClass.Value, Check) -> 0.35,
      (StrategicClass.Value, Call) -> 0.40, (StrategicClass.Value, Raise) -> 0.20,
      (StrategicClass.Bluff, Fold) -> 0.10, (StrategicClass.Bluff, Check) -> 0.10,
      (StrategicClass.Bluff, Call) -> 0.15, (StrategicClass.Bluff, Raise) -> 0.65,
      (StrategicClass.StructuralBluff, Fold) -> 0.05, (StrategicClass.StructuralBluff, Check) -> 0.15,
      (StrategicClass.StructuralBluff, Call) -> 0.30, (StrategicClass.StructuralBluff, Raise) -> 0.50,
      (StrategicClass.Mixed, Fold) -> 0.15, (StrategicClass.Mixed, Check) -> 0.40,
      (StrategicClass.Mixed, Call) -> 0.35, (StrategicClass.Mixed, Raise) -> 0.10
    )
  }

  private val baseline = new PosteriorAttributedBaseline(priors)

  private val uniformBelief = StrategicRivalBelief.uniform

  private val valueBelief = StrategicRivalBelief(DiscreteDistribution(Map(
    StrategicClass.Value -> 0.85,
    StrategicClass.Bluff -> 0.05,
    StrategicClass.StructuralBluff -> 0.05,
    StrategicClass.Mixed -> 0.05
  )))

  private val pubState: PublicState = {
    import sicfun.holdem.types.{Board, Position}
    val hero = PlayerId("__test__")
    PublicState(
      street = Street.Flop,
      board = Board.empty,
      pot = Chips(100.0),
      stacks = TableMap(
        hero = hero,
        seats = Vector(Seat(hero, Position.SmallBlind, SeatStatus.Active, Chips(500.0)))
      ),
      actionHistory = Vector.empty
    )
  }

  test("uniform posterior returns pi0 unchanged"):
    for
      cls <- StrategicClass.values
      cat <- PokerAction.Category.values
    do
      val expected = priors.getOrElse((cls, cat), 0.25)
      val actual = baseline.probability(cls, cat, None, pubState, uniformBelief)
      assertEqualsDouble(actual, expected, 1e-10,
        s"uniform posterior should return pi0 for ($cls, $cat)")

  test("probabilities sum to 1.0 for each (class, belief)"):
    val beliefs = Vector(uniformBelief, valueBelief)
    for
      cls <- StrategicClass.values
      belief <- beliefs
    do
      val sum = PokerAction.Category.values.map { cat =>
        baseline.probability(cls, cat, None, pubState, belief)
      }.sum
      assertEqualsDouble(sum, 1.0, 1e-10,
        s"sum for $cls should be 1.0, got $sum")

  test("degenerate posterior on Value skews toward Value-typical actions"):
    val callProb = baseline.probability(
      StrategicClass.Mixed, PokerAction.Category.Call, None, pubState, valueBelief)
    val foldProb = baseline.probability(
      StrategicClass.Mixed, PokerAction.Category.Fold, None, pubState, valueBelief)
    val callPi0 = priors((StrategicClass.Mixed, PokerAction.Category.Call))
    val foldPi0 = priors((StrategicClass.Mixed, PokerAction.Category.Fold))
    assert(callProb / foldProb > callPi0 / foldPi0,
      s"Value-heavy belief should increase call/fold ratio: attributed=${callProb / foldProb}, pi0=${callPi0 / foldPi0}")

  test("non-StrategicRivalBelief returns pi0 unchanged"):
    val dummyRival = new RivalBeliefState:
      def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState = this
    for
      cls <- StrategicClass.values
      cat <- PokerAction.Category.values
    do
      val expected = priors.getOrElse((cls, cat), 0.25)
      val actual = baseline.probability(cls, cat, None, pubState, dummyRival)
      assertEqualsDouble(actual, expected, 1e-10,
        s"non-SRB should return pi0 for ($cls, $cat)")

  test("epsilon floor prevents division by zero"):
    val zeroPriors: Map[(StrategicClass, PokerAction.Category), Double] = {
      import PokerAction.Category.*
      Map(
        (StrategicClass.Value, Fold) -> 0.0, (StrategicClass.Value, Check) -> 0.0,
        (StrategicClass.Value, Call) -> 0.0, (StrategicClass.Value, Raise) -> 1.0,
        (StrategicClass.Bluff, Fold) -> 0.0, (StrategicClass.Bluff, Check) -> 0.0,
        (StrategicClass.Bluff, Call) -> 0.0, (StrategicClass.Bluff, Raise) -> 1.0,
        (StrategicClass.StructuralBluff, Fold) -> 0.0, (StrategicClass.StructuralBluff, Check) -> 0.0,
        (StrategicClass.StructuralBluff, Call) -> 0.0, (StrategicClass.StructuralBluff, Raise) -> 1.0,
        (StrategicClass.Mixed, Fold) -> 0.0, (StrategicClass.Mixed, Check) -> 0.0,
        (StrategicClass.Mixed, Call) -> 0.0, (StrategicClass.Mixed, Raise) -> 1.0
      )
    }
    val zeroBaseline = new PosteriorAttributedBaseline(zeroPriors)
    val sum = PokerAction.Category.values.map { cat =>
      zeroBaseline.probability(StrategicClass.Value, cat, None, pubState, valueBelief)
    }.sum
    assertEqualsDouble(sum, 1.0, 1e-10)