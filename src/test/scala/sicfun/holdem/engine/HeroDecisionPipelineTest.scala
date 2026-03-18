package sicfun.holdem.engine

import munit.FunSuite
import sicfun.holdem.types.*

class HeroDecisionPipelineTest extends FunSuite:

  private def ctx(
      stackRemaining: Int,
      toCall: Int,
      lastBetSize: Int,
      pot: Int,
      street: Street,
      streetLastBetTo: Int,
      bigBlindChips: Int = 100
  ): HeroDecisionPipeline.RaiseSizingContext =
    HeroDecisionPipeline.RaiseSizingContext(
      stackRemainingChips = stackRemaining,
      toCallChips = toCall,
      lastBetSizeChips = lastBetSize,
      potChips = pot,
      currentStreet = street,
      streetLastBetToChips = streetLastBetTo,
      bigBlindChips = bigBlindChips
    )

  test("legalRaiseCandidates returns empty when no room to raise"):
    val result = HeroDecisionPipeline.legalRaiseCandidates(
      ctx(stackRemaining = 100, toCall = 100, lastBetSize = 100, pot = 300, street = Street.Flop, streetLastBetTo = 200)
    )
    assertEquals(result, Vector.empty[PokerAction])

  test("legalRaiseCandidates preflop facing open returns 1.5bb and 2bb sizes"):
    val result = HeroDecisionPipeline.legalRaiseCandidates(
      ctx(stackRemaining = 10000, toCall = 100, lastBetSize = 100, pot = 300, street = Street.Preflop, streetLastBetTo = 100)
    )
    assertEquals(result.length, 2)
    assertEquals(result(0), PokerAction.Raise(1.5))
    assertEquals(result(1), PokerAction.Raise(2.0))

  test("legalRaiseCandidates preflop unraised returns 2bb and 3bb sizes"):
    val result = HeroDecisionPipeline.legalRaiseCandidates(
      ctx(stackRemaining = 10000, toCall = 0, lastBetSize = 0, pot = 150, street = Street.Preflop, streetLastBetTo = 100)
    )
    assertEquals(result.length, 2)
    assertEquals(result(0), PokerAction.Raise(2.0))
    assertEquals(result(1), PokerAction.Raise(3.0))

  test("legalRaiseCandidates postflop check-to-act returns 50% and 75% pot sizes"):
    val result = HeroDecisionPipeline.legalRaiseCandidates(
      ctx(stackRemaining = 10000, toCall = 0, lastBetSize = 0, pot = 400, street = Street.Flop, streetLastBetTo = 0)
    )
    assertEquals(result.length, 2)
    assertEquals(result(0), PokerAction.Raise(2.0))
    assertEquals(result(1), PokerAction.Raise(3.0))

  test("legalRaiseCandidates postflop facing bet returns min legal and 75% pot"):
    val result = HeroDecisionPipeline.legalRaiseCandidates(
      ctx(stackRemaining = 10000, toCall = 200, lastBetSize = 200, pot = 600, street = Street.Flop, streetLastBetTo = 400)
    )
    assertEquals(result.length, 2)
    assertEquals(result(0), PokerAction.Raise(2.0))
    assertEquals(result(1), PokerAction.Raise(4.5))

  test("heroCandidates when no call needed returns Check + raises"):
    val raises = Vector(PokerAction.Raise(2.0), PokerAction.Raise(3.0))
    val result = HeroDecisionPipeline.heroCandidates(toCallChips = 0, raises = raises)
    assertEquals(result, Vector(PokerAction.Check, PokerAction.Raise(2.0), PokerAction.Raise(3.0)))

  test("heroCandidates when facing bet returns Fold + Call + raises"):
    val raises = Vector(PokerAction.Raise(4.0))
    val result = HeroDecisionPipeline.heroCandidates(toCallChips = 200, raises = raises)
    assertEquals(result, Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(4.0)))
