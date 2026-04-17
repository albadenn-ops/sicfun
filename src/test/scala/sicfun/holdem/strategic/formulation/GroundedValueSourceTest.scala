package sicfun.holdem.strategic.formulation

import munit.FunSuite
import sicfun.core.{Card, Rank, Suit}
import sicfun.holdem.engine.{HandStrengthEstimator, PokerPomcpFormulation}
import sicfun.holdem.strategic.types.{BridgeResult, Ev}
import sicfun.holdem.types.*

class GroundedValueSourceTest extends FunSuite:

  private val aceKingSuited = HoleCards(
    Card(Rank.Ace, Suit.Spades),
    Card(Rank.King, Suit.Spades)
  )

  private val twoSevenOff = HoleCards(
    Card(Rank.Two, Suit.Clubs),
    Card(Rank.Seven, Suit.Hearts)
  )

  private def makeSpot(
      cards: HoleCards,
      street: Street = Street.Preflop
  ): FormulationSpot =
    FormulationSpot(
      gameState = GameState(
        street = street,
        board = Board.empty,
        pot = 100.0,
        toCall = 20.0,
        position = Position.Button,
        stackSize = 1000.0,
        betHistory = Vector.empty
      ),
      candidateActions = Vector(
        PokerAction.Fold,
        PokerAction.Call,
        PokerAction.Raise(50.0)
      ),
      heroValueInput = HeroValueInput.ExactHoleCards(cards),
      rivalBeliefs = Map.empty
    )

  private def extractDouble(result: BridgeResult[Double]): Double =
    result match
      case BridgeResult.Exact(value) => value
      case BridgeResult.Approximate(value, _) => value
      case BridgeResult.Absent(reason) => fail(reason)

  private def extractEv(result: BridgeResult[Ev]): Double =
    result match
      case BridgeResult.Exact(value) => value.value
      case BridgeResult.Approximate(value, _) => value.value
      case BridgeResult.Absent(reason) => fail(reason)

  test("estimateSpotEquity returns Exact for ExactHoleCards"):
    GroundedValueSource.estimateSpotEquity(makeSpot(aceKingSuited)) match
      case BridgeResult.Exact(eq) =>
        assert(eq > 0.0 && eq <= 1.0, s"equity $eq out of range")
      case other => fail(s"Expected Exact, got $other")

  test("estimateSpotEquity: AKs > 72o"):
    val aksEquity = extractDouble(
      GroundedValueSource.estimateSpotEquity(makeSpot(aceKingSuited))
    )
    val tsoEquity = extractDouble(
      GroundedValueSource.estimateSpotEquity(makeSpot(twoSevenOff))
    )
    assert(aksEquity > tsoEquity,
      s"AKs equity ($aksEquity) should exceed 72o equity ($tsoEquity)")

  test("estimateSpotEquity falls back to Approximate for StrengthHint"):
    val spot = makeSpot(aceKingSuited).copy(
      heroValueInput = HeroValueInput.StrengthHint(bucket = 7, source = "test")
    )
    GroundedValueSource.estimateSpotEquity(spot) match
      case BridgeResult.Approximate(eq, _) =>
        assertEqualsDouble(eq, 7.0 / 9.0, 1e-10)
      case other => fail(s"Expected Approximate for StrengthHint, got $other")

  test("showdownEquityTable returns calibrated table with exact hero row"):
    val exactStrength = HandStrengthEstimator.fastGtoStrength(
      aceKingSuited,
      Board.empty,
      Street.Preflop
    )
    val heroBucket = math.min(9, math.max(0, (exactStrength * 10.0).toInt))
    val linear = PokerPomcpFormulation.buildLinearShowdownEquity(10, 10)
    GroundedValueSource.showdownEquityTable(makeSpot(aceKingSuited), 10, 10) match
      case BridgeResult.Approximate(table, note) =>
        assertEquals(table.length, 100)
        assert(table.forall(equity => equity >= 0.0 && equity <= 1.0))
        assert(!note.contains("linear"), s"unexpected note: $note")
        assert(!table.sameElements(linear), "grounded showdown table should not use linear heuristic")
        val row = table.slice(heroBucket * 10, heroBucket * 10 + 10)
        assert(row.sliding(2).forall {
          case Array(left, right) => left >= right
          case _ => true
        }, s"exact hero row should be monotone decreasing: ${row.toVector}")
        assertEqualsDouble(
          row(heroBucket),
          PokerPomcpFormulation.calibratedBucketEquity(exactStrength, heroBucket, 10),
          1e-10
        )
      case other => fail(s"Expected Approximate, got $other")

  test("estimateActionValue: fold has negative value, raise exceeds fold for strong hand"):
    val spot = makeSpot(aceKingSuited)
    val foldValue = extractEv(
      GroundedValueSource.estimateActionValue(spot, PokerAction.Fold)
    )
    val raiseValue = extractEv(
      GroundedValueSource.estimateActionValue(spot, PokerAction.Raise(50.0))
    )
    assert(foldValue < 0.0, s"fold value ($foldValue) should be negative")
    assert(raiseValue > foldValue,
      s"raise value ($raiseValue) should exceed fold ($foldValue)")

  test("estimateActionValue uses potOdds breakeven, not hardcoded 0.5"):
    val callValue = extractEv(
      GroundedValueSource.estimateActionValue(makeSpot(aceKingSuited), PokerAction.Call)
    )
    assert(callValue > 0.0,
      s"call value ($callValue) should be positive for AKs with potOdds=0.167")
