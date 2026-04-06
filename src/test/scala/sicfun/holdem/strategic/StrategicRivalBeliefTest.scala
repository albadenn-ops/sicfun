package sicfun.holdem.strategic

import munit.FunSuite
import sicfun.core.DiscreteDistribution
import sicfun.holdem.types.{Position, Street, Board}

class StrategicRivalBeliefTest extends FunSuite:

  private val hero = PlayerId("hero")
  private val rival = PlayerId("v1")

  private val uniformPrior = DiscreteDistribution(Map(
    StrategicClass.Value    -> 0.25,
    StrategicClass.Bluff    -> 0.25,
    StrategicClass.SemiBluff -> 0.25,
    StrategicClass.Marginal  -> 0.25
  ))

  private def mkPublicState: PublicState =
    PublicState(
      street = Street.Flop,
      board = Board.empty,
      pot = Chips(100.0),
      stacks = TableMap(
        hero = hero,
        seats = Vector(
          Seat(hero,  Position.SmallBlind, SeatStatus.Active, Chips(500.0)),
          Seat(rival, Position.BigBlind,   SeatStatus.Active, Chips(500.0))
        )
      ),
      actionHistory = Vector.empty
    )

  test("StrategicRivalBelief initializes with uniform prior"):
    val belief = StrategicRivalBelief(uniformPrior)
    assertEqualsDouble(belief.typePosterior.probabilityOf(StrategicClass.Value), 0.25, 1e-10)
    assertEqualsDouble(belief.typePosterior.probabilityOf(StrategicClass.Bluff), 0.25, 1e-10)
    assertEqualsDouble(belief.typePosterior.probabilityOf(StrategicClass.SemiBluff), 0.25, 1e-10)
    assertEqualsDouble(belief.typePosterior.probabilityOf(StrategicClass.Marginal), 0.25, 1e-10)

  test("update returns same belief (identity update)"):
    val belief = StrategicRivalBelief(uniformPrior)
    val signal = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Raise,
      sizing = None,
      timing = None,
      stage = Street.Flop
    )
    val updated = belief.update(signal, mkPublicState)
    assertEquals(updated, belief)

  test("update return type is StrategicRivalBelief"):
    val belief = StrategicRivalBelief(uniformPrior)
    val signal = ActionSignal(
      action = sicfun.holdem.types.PokerAction.Category.Call,
      sizing = None,
      timing = None,
      stage = Street.Turn
    )
    val updated = belief.update(signal, mkPublicState)
    assert(updated.isInstanceOf[StrategicRivalBelief])

  test("toParticles produces correct count and valid ordinals"):
    val belief = StrategicRivalBelief(DiscreteDistribution(Map(
      StrategicClass.Value -> 0.7,
      StrategicClass.Bluff -> 0.3
    )))
    val (types, weights) = belief.toParticles(numParticles = 100, handBucket = 5)
    assertEquals(types.length, 100)
    assertEquals(weights.length, 100)
    val validOrdinals = StrategicClass.values.map(_.ordinal).toSet
    assert(types.forall(t => validOrdinals.contains(t)))

  test("toParticles weights sum to 1.0"):
    val belief = StrategicRivalBelief(uniformPrior)
    val (_, weights) = belief.toParticles(numParticles = 100, handBucket = 3)
    assertEqualsDouble(weights.sum, 1.0, 1e-10)

  test("uniform factory creates equal distribution over all four classes"):
    val belief = StrategicRivalBelief.uniform
    for cls <- StrategicClass.values do
      assertEqualsDouble(belief.typePosterior.probabilityOf(cls), 0.25, 1e-10)

  test("toParticles distributes proportionally for pure Value distribution"):
    val belief = StrategicRivalBelief(DiscreteDistribution(Map(
      StrategicClass.Value -> 1.0
    )))
    val (types, _) = belief.toParticles(100, 5)
    assert(types.forall(_ == StrategicClass.Value.ordinal))

  test("updater companion produces StrategicRivalBelief with given posterior"):
    val newPosterior = DiscreteDistribution(Map(
      StrategicClass.Bluff -> 0.8,
      StrategicClass.Value -> 0.2
    ))
    val base = StrategicRivalBelief.uniform
    val updated = StrategicRivalBelief.updater(base, newPosterior)
    assertEqualsDouble(updated.typePosterior.probabilityOf(StrategicClass.Bluff), 0.8, 1e-10)
    assertEqualsDouble(updated.typePosterior.probabilityOf(StrategicClass.Value), 0.2, 1e-10)

  test("toParticles uniform weight is 1/numParticles"):
    val belief = StrategicRivalBelief.uniform
    val n = 200
    val (_, weights) = belief.toParticles(n, 0)
    val expected = 1.0 / n
    for w <- weights do assertEqualsDouble(w, expected, 1e-12)
