package sicfun.holdem.strategic

import sicfun.holdem.types.{Position, Street, Board, HoleCards}
import sicfun.core.{CardId, DiscreteDistribution}

class AugmentedStateTest extends munit.FunSuite:

  private val hero = PlayerId("hero")
  private val v1 = PlayerId("v1")
  private val v2 = PlayerId("v2")

  private def mkPublicState(nRivals: Int): PublicState =
    val rivalSeats = (1 to nRivals).map { i =>
      val pos = Position.values(i % Position.values.length)
      Seat(PlayerId(s"v$i"), pos, SeatStatus.Active, Chips(500.0))
    }.toVector
    PublicState(
      street = Street.Flop,
      board = Board.empty,
      pot = Chips(100.0),
      stacks = TableMap(
        hero = hero,
        seats = Seat(hero, Position.SmallBlind, SeatStatus.Active, Chips(500.0)) +: rivalSeats
      ),
      actionHistory = Vector.empty
    )

  private val dummyHoleCards = HoleCards.canonical(CardId.fromId(0), CardId.fromId(1))

  private val dummyBeliefState = new RivalBeliefState:
    def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState = this

  private val dummyOMS = OpponentModelState(
    typeDistribution = DiscreteDistribution.uniform(Seq("TAG", "LAG")),
    beliefState = dummyBeliefState,
    attributedBaseline = None
  )

  test("PublicState pot must be non-negative"):
    intercept[IllegalArgumentException]:
      PublicState(
        street = Street.Preflop,
        board = Board.empty,
        pot = Chips(-1.0),
        stacks = TableMap(
          hero = hero,
          seats = Vector(
            Seat(hero, Position.SmallBlind, SeatStatus.Active, Chips(100.0)),
            Seat(v1, Position.BigBlind, SeatStatus.Active, Chips(100.0))
          )
        ),
        actionHistory = Vector.empty
      )

  test("PublicState is an own type, not GameState"):
    val ps = mkPublicState(1)
    assertEquals(ps.pot, Chips(100.0))

  test("AugmentedState stores public, private, opponents, ownEvidence"):
    val opponents = RivalMap(Vector(
      Seat(v1, Position.BigBlind, SeatStatus.Active, dummyOMS),
      Seat(v2, Position.UTG, SeatStatus.Active, dummyOMS)
    ))
    val state = AugmentedState(
      publicState = mkPublicState(2),
      privateHand = dummyHoleCards,
      opponents = opponents,
      ownEvidence = OwnEvidence.empty
    )
    assertEquals(state.opponents.size, 2)

  test("AugmentedState is multiway-native (L7): heads-up is |R|=1, not special"):
    val huOpponents = RivalMap(Vector(
      Seat(v1, Position.BigBlind, SeatStatus.Active, dummyOMS)
    ))
    val huState = AugmentedState(
      publicState = mkPublicState(1),
      privateHand = dummyHoleCards,
      opponents = huOpponents,
      ownEvidence = OwnEvidence.empty
    )
    assertEquals(huState.opponents.size, 1)
    val mwOpponents = RivalMap(Vector(
      Seat(v1, Position.BigBlind, SeatStatus.Active, dummyOMS),
      Seat(v2, Position.UTG, SeatStatus.Active, dummyOMS)
    ))
    val mwState = huState.copy(opponents = mwOpponents, publicState = mkPublicState(2))
    assertEquals(mwState.opponents.size, 2)

  test("OperativeBelief wraps DiscreteDistribution over AugmentedState"):
    val opponents = RivalMap(Vector(
      Seat(v1, Position.BigBlind, SeatStatus.Active, dummyOMS)
    ))
    val state1 = AugmentedState(
      publicState = mkPublicState(1),
      privateHand = dummyHoleCards,
      opponents = opponents,
      ownEvidence = OwnEvidence.empty
    )
    val belief = OperativeBelief(DiscreteDistribution(Map(state1 -> 1.0)))
    assertEquals(belief.distribution.probabilityOf(state1), 1.0)

  test("OwnEvidence.empty has no data"):
    val oe = OwnEvidence.empty
    assert(oe.globalSummary.isEmpty)
    assert(oe.perRivalSummary.isEmpty)
    assert(oe.relationalSummary.isEmpty)

  test("OpponentModelState has no playerId field (identity in RivalMap)"):
    val oms = dummyOMS
    assertEquals(oms.typeDistribution.support.size, 2)
    assertEquals(oms.attributedBaseline, None)
