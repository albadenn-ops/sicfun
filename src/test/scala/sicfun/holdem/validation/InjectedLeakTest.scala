package sicfun.holdem.validation

import munit.FunSuite
import sicfun.core.{Card, Rank, Suit}
import sicfun.holdem.types.{Board, Position, Street, PokerAction}
import scala.util.Random

/** Tests for all six [[InjectedLeak]] types and the [[NoLeak]] sentinel.
  *
  * Each leak type is tested along two axes:
  *   1. '''Predicate correctness''' (`applies`): verifies that the leak
  *      fires only in the intended spot context (correct street, board
  *      texture, hand category, pot geometry, and range position) and
  *      does NOT fire when any guard condition is violated.
  *   2. '''Deviation correctness''' (`deviate`): verifies the action
  *      produced when the leak fires (e.g. OverfoldsToAggression -> Fold,
  *      Overcalls -> Call, OverbluffsTurnBarrel -> Raise at 60% of stack).
  *
  * Additionally tests the statistical fire-rate mechanism: severity=0.3
  * should produce approximately 30% deviations, and severity=0.5 should
  * produce approximately 50% deviations across all leak types.
  *
  * Leak types covered:
  *   - [[OverfoldsToAggression]] — folds to large bets on wet rivers with weak/capped hands
  *   - [[Overcalls]] — calls large bets with air/weak hands
  *   - [[OverbluffsTurnBarrel]] — barrel-bluffs on wet turns with air
  *   - [[PassiveInBigPots]] — checks strong hands in big pots (low SPR)
  *   - [[PreflopTooLoose]] — calls preflop with weak/air hands
  *   - [[PreflopTooTight]] — folds preflop with medium-strength hands
  */
class InjectedLeakTest extends FunSuite:

  // ── Helpers ──

  private def riverSpot(
      betToPot: Double = 0.8,
      wet: Boolean = true,
      handCategory: HandCategory = HandCategory.Weak,
      rangePos: RangePosition = RangePosition.Capped
  ): SpotContext =
    val board = if wet then
      Board.from(Vector(
        Card(Rank.Nine, Suit.Hearts), Card(Rank.Eight, Suit.Hearts),
        Card(Rank.Two, Suit.Clubs), Card(Rank.Five, Suit.Diamonds),
        Card(Rank.Jack, Suit.Hearts)
      ))
    else
      Board.from(Vector(
        Card(Rank.Ace, Suit.Hearts), Card(Rank.Eight, Suit.Diamonds),
        Card(Rank.Five, Suit.Clubs), Card(Rank.Jack, Suit.Spades),
        Card(Rank.Two, Suit.Hearts)
      ))
    SpotContext(
      street = Street.River,
      board = board,
      boardTexture = BoardTexture.from(board),
      potGeometry = PotGeometry(spr = 1.0, potOdds = betToPot / (1.0 + betToPot),
        betToPotRatio = betToPot, effectiveStack = 50.0),
      position = Position.BigBlind,
      facingAction = Some(PokerAction.Raise(betToPot * 100.0)),
      facingSizing = Some(betToPot),
      lineRepresented = ActionLine(Vector(PokerAction.Call)),
      handStrengthVsBoard = handCategory,
      rangeAdvantage = rangePos
    )

  private def turnSpotIP(handCat: HandCategory = HandCategory.Air): SpotContext =
    val board = Board.from(Vector(
      Card(Rank.Nine, Suit.Hearts), Card(Rank.Eight, Suit.Hearts),
      Card(Rank.Two, Suit.Clubs), Card(Rank.Five, Suit.Diamonds)
    ))
    SpotContext(
      street = Street.Turn, board = board,
      boardTexture = BoardTexture.from(board),
      potGeometry = PotGeometry(spr = 3.0, potOdds = 0.0, betToPotRatio = 0.0, effectiveStack = 90.0),
      position = Position.Button,
      facingAction = None, facingSizing = None,
      lineRepresented = ActionLine(Vector(PokerAction.Raise(6.0))),
      handStrengthVsBoard = handCat,
      rangeAdvantage = RangePosition.Uncapped
    )

  private def preflopSpot(handCat: HandCategory): SpotContext =
    SpotContext(
      street = Street.Preflop, board = Board.empty,
      boardTexture = BoardTexture.from(Board.empty),
      potGeometry = PotGeometry(spr = 50.0, potOdds = 0.33, betToPotRatio = 0.5, effectiveStack = 100.0),
      position = Position.Cutoff,
      facingAction = None, facingSizing = None,
      lineRepresented = ActionLine(Vector.empty),
      handStrengthVsBoard = handCat,
      rangeAdvantage = RangePosition.Uncapped
    )

  // ── OverfoldsToAggression ──

  test("OverfoldsToAggression applies on wet river facing large bet with capped weak hand"):
    val leak = OverfoldsToAggression(severity = 1.0)
    val spot = riverSpot(betToPot = 0.8, wet = true, handCategory = HandCategory.Weak, rangePos = RangePosition.Capped)
    assert(leak.applies(spot))

  test("OverfoldsToAggression does NOT apply on dry board"):
    val leak = OverfoldsToAggression(severity = 1.0)
    val spot = riverSpot(betToPot = 0.8, wet = false, handCategory = HandCategory.Weak, rangePos = RangePosition.Capped)
    assert(!leak.applies(spot))

  test("OverfoldsToAggression does NOT apply with strong hand"):
    val leak = OverfoldsToAggression(severity = 1.0)
    val spot = riverSpot(betToPot = 0.8, wet = true, handCategory = HandCategory.Strong, rangePos = RangePosition.Capped)
    assert(!leak.applies(spot))

  test("OverfoldsToAggression does NOT apply with small bet"):
    val leak = OverfoldsToAggression(severity = 1.0)
    val spot = riverSpot(betToPot = 0.3, wet = true, handCategory = HandCategory.Weak, rangePos = RangePosition.Capped)
    assert(!leak.applies(spot))

  test("OverfoldsToAggression deviates to Fold"):
    val leak = OverfoldsToAggression(severity = 1.0)
    val spot = riverSpot()
    val deviated = leak.deviate(PokerAction.Call, spot, new Random(42))
    assertEquals(deviated, PokerAction.Fold)

  test("OverfoldsToAggression severity controls fire rate"):
    val leak = OverfoldsToAggression(severity = 0.3)
    val spot = riverSpot()
    val rng = new Random(42)
    var foldCount = 0
    val trials = 10000
    for _ <- 0 until trials do
      val action = leak.deviate(PokerAction.Call, spot, rng)
      if action == PokerAction.Fold then foldCount += 1
    val rate = foldCount.toDouble / trials
    assertEqualsDouble(rate, 0.3, 0.03) // within 3% of expected

  // ── Overcalls ──

  test("Overcalls applies when facing large bet with weak hand"):
    val leak = Overcalls(severity = 1.0)
    val spot = riverSpot(betToPot = 1.0, wet = true, handCategory = HandCategory.Air, rangePos = RangePosition.Capped)
    assert(leak.applies(spot))

  test("Overcalls does NOT apply with strong hand"):
    val leak = Overcalls(severity = 1.0)
    val spot = riverSpot(betToPot = 1.0, wet = true, handCategory = HandCategory.Strong)
    assert(!leak.applies(spot))

  test("Overcalls does NOT apply with small bet"):
    val leak = Overcalls(severity = 1.0)
    val spot = riverSpot(betToPot = 0.3, wet = true, handCategory = HandCategory.Air)
    assert(!leak.applies(spot))

  test("Overcalls deviates to Call"):
    val leak = Overcalls(severity = 1.0)
    val spot = riverSpot(betToPot = 1.0, wet = true, handCategory = HandCategory.Air)
    assertEquals(leak.deviate(PokerAction.Fold, spot, new Random(42)), PokerAction.Call)

  // ── OverbluffsTurnBarrel ──

  test("OverbluffsTurnBarrel applies IP on turn with air and wet board"):
    val leak = OverbluffsTurnBarrel(severity = 1.0)
    assert(leak.applies(turnSpotIP(HandCategory.Air)))

  test("OverbluffsTurnBarrel does NOT apply with strong hand"):
    val leak = OverbluffsTurnBarrel(severity = 1.0)
    assert(!leak.applies(turnSpotIP(HandCategory.Strong)))

  test("OverbluffsTurnBarrel does NOT apply on river"):
    val leak = OverbluffsTurnBarrel(severity = 1.0)
    val spot = turnSpotIP().copy(street = Street.River)
    assert(!leak.applies(spot))

  test("OverbluffsTurnBarrel applies OOP (position-agnostic)"):
    val leak = OverbluffsTurnBarrel(severity = 1.0)
    val spot = turnSpotIP().copy(position = Position.BigBlind)
    assert(leak.applies(spot))

  test("OverbluffsTurnBarrel deviates to Raise"):
    val leak = OverbluffsTurnBarrel(severity = 1.0)
    val spot = turnSpotIP()
    val deviated = leak.deviate(PokerAction.Check, spot, new Random(42))
    deviated match
      case PokerAction.Raise(amt) => assertEqualsDouble(amt, 90.0 * 0.6, 0.01)
      case other => fail(s"expected Raise, got $other")

  // ── PassiveInBigPots ──

  test("PassiveInBigPots applies in big pot with strong hand"):
    val spot = riverSpot(betToPot = 0.0, wet = true, handCategory = HandCategory.Strong, rangePos = RangePosition.Uncapped)
      .copy(potGeometry = PotGeometry(spr = 1.2, potOdds = 0.0, betToPotRatio = 0.0, effectiveStack = 40.0))
    val leak = PassiveInBigPots(severity = 1.0)
    assert(leak.applies(spot))

  test("PassiveInBigPots applies with Nuts hand"):
    val spot = riverSpot(betToPot = 0.0, wet = true, handCategory = HandCategory.Nuts)
      .copy(potGeometry = PotGeometry(spr = 0.5, potOdds = 0.0, betToPotRatio = 0.0, effectiveStack = 20.0))
    val leak = PassiveInBigPots(severity = 1.0)
    assert(leak.applies(spot))

  test("PassiveInBigPots does NOT apply with small pot"):
    val spot = riverSpot(handCategory = HandCategory.Strong)
      .copy(potGeometry = PotGeometry(spr = 5.0, potOdds = 0.0, betToPotRatio = 0.0, effectiveStack = 100.0))
    val leak = PassiveInBigPots(severity = 1.0)
    assert(!leak.applies(spot))

  test("PassiveInBigPots deviates to Check"):
    val spot = riverSpot().copy(
      handStrengthVsBoard = HandCategory.Strong,
      potGeometry = PotGeometry(spr = 1.0, potOdds = 0.0, betToPotRatio = 0.0, effectiveStack = 40.0)
    )
    val leak = PassiveInBigPots(severity = 1.0)
    assertEquals(leak.deviate(PokerAction.Raise(30.0), spot, new Random(42)), PokerAction.Check)

  // ── PreflopTooLoose ──

  test("PreflopTooLoose applies with weak hand"):
    val leak = PreflopTooLoose(severity = 1.0)
    assert(leak.applies(preflopSpot(HandCategory.Weak)))

  test("PreflopTooLoose applies with air hand"):
    val leak = PreflopTooLoose(severity = 1.0)
    assert(leak.applies(preflopSpot(HandCategory.Air)))

  test("PreflopTooLoose does NOT apply with medium hand"):
    val leak = PreflopTooLoose(severity = 1.0)
    assert(!leak.applies(preflopSpot(HandCategory.Medium)))

  test("PreflopTooLoose does NOT apply postflop"):
    val leak = PreflopTooLoose(severity = 1.0)
    val spot = preflopSpot(HandCategory.Weak).copy(street = Street.Flop)
    assert(!leak.applies(spot))

  test("PreflopTooLoose deviates Fold to Call"):
    val leak = PreflopTooLoose(severity = 1.0)
    assertEquals(leak.deviate(PokerAction.Fold, preflopSpot(HandCategory.Weak), new Random(42)), PokerAction.Call)

  // ── PreflopTooTight ──

  test("PreflopTooTight applies with medium hand preflop"):
    val leak = PreflopTooTight(severity = 1.0)
    assert(leak.applies(preflopSpot(HandCategory.Medium)))

  test("PreflopTooTight does NOT apply with weak hand"):
    val leak = PreflopTooTight(severity = 1.0)
    assert(!leak.applies(preflopSpot(HandCategory.Weak)))

  test("PreflopTooTight does NOT apply postflop"):
    val leak = PreflopTooTight(severity = 1.0)
    val spot = preflopSpot(HandCategory.Medium).copy(street = Street.Turn)
    assert(!leak.applies(spot))

  test("PreflopTooTight deviates Call/Raise to Fold"):
    val leak = PreflopTooTight(severity = 1.0)
    assertEquals(leak.deviate(PokerAction.Call, preflopSpot(HandCategory.Medium), new Random(42)), PokerAction.Fold)

  // ── Severity fire rate ──

  test("severity=0.5 fires roughly half the time across leaks"):
    val leaks: Vector[InjectedLeak] = Vector(
      OverfoldsToAggression(severity = 0.5),
      Overcalls(severity = 0.5),
      PreflopTooLoose(severity = 0.5),
      PreflopTooTight(severity = 0.5)
    )
    val rng = new Random(123)
    val trials = 5000
    for leak <- leaks do
      var fired = 0
      for _ <- 0 until trials do
        val spot = leak match
          case _: OverfoldsToAggression => riverSpot()
          case _: Overcalls => riverSpot(betToPot = 1.0, handCategory = HandCategory.Air)
          case _: PreflopTooLoose => preflopSpot(HandCategory.Weak)
          case _: PreflopTooTight => preflopSpot(HandCategory.Medium)
          case _ => preflopSpot(HandCategory.Medium)
        if leak.applies(spot) then
          val original = leak match
            case _: Overcalls       => PokerAction.Fold // GTO says fold, leak deviates to Call
            case _: PreflopTooLoose => PokerAction.Fold // GTO says fold, leak deviates to Call
            case _: PreflopTooTight => PokerAction.Call // GTO says call, leak deviates to Fold
            case _ => PokerAction.Call // OverfoldsToAggression: GTO says call, leak deviates to Fold
          val result = leak.deviate(original, spot, rng)
          if result != original then fired += 1
      val rate = fired.toDouble / trials
      assertEqualsDouble(rate, 0.5, 0.05, s"${leak.id} fire rate $rate not near 0.5")
