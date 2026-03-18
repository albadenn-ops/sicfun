package sicfun.holdem.history

import sicfun.holdem.types.*
import sicfun.holdem.engine.*
import sicfun.holdem.equity.*
import sicfun.holdem.model.*

import munit.FunSuite
import sicfun.core.Card

class ShowdownIntegrationTest extends FunSuite:

  private def card(t: String): Card =
    Card.parse(t).getOrElse(throw new IllegalArgumentException(s"bad card: $t"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  test("showdown cards flow through import -> profile -> range inference") {
    // 1. Parse a hand history with showdown
    val text = """PokerStars Hand #999: Hold'em No Limit ($1/$2) - 2025/01/01 12:00:00 ET
Table 'TestTable' 2-max Seat #1 is the button
Seat 1: Hero ($200 in chips)
Seat 2: Villain ($200 in chips)
Hero: posts small blind $1
Villain: posts big blind $2
*** HOLE CARDS ***
Dealt to Hero [Ac Kh]
Hero: raises $4 to $6
Villain: calls $4
*** FLOP *** [Ts 9h 8d]
Hero: bets $8
Villain: calls $8
*** TURN *** [Ts 9h 8d] [2c]
Hero: checks
Villain: checks
*** RIVER *** [Ts 9h 8d 2c] [3s]
Hero: checks
Villain: checks
*** SHOW DOWN ***
Villain: shows [Qh Qs] (a pair of Queens)
Hero: shows [Ac Kh] (high card Ace)
Villain collected $28 from pot
*** SUMMARY ***
Total pot $28 | Rake $0
"""
    val parsed = HandHistoryImport.parseText(text, Some(HandHistorySite.PokerStars), Some("Hero"))
    assert(parsed.isRight, s"parse failed: ${parsed.left.getOrElse("")}")
    val hands = parsed.toOption.get
    assertEquals(hands.length, 1)

    val hand = hands.head
    val villainCards = hand.showdownCards.get("Villain")
    assert(villainCards.isDefined, "Villain showdown cards should be present")
    assertEquals(villainCards.get, hole("Qh", "Qs"))

    // 2. Build opponent profile
    val profiles = OpponentProfile.fromImportedHands("pokerstars", hands, Set("Hero"))
    assertEquals(profiles.length, 1)
    val profile = profiles.head
    assertEquals(profile.playerName, "Villain")

    // 3. Assert showdownHands populated
    assertEquals(profile.showdownHands.length, 1)
    assertEquals(profile.showdownHands.head.handId, "999")
    assertEquals(profile.showdownHands.head.cards, hole("Qh", "Qs"))

    // 4. Use the profile's showdown data in range inference — delta posterior
    val revealedCards = profile.showdownHands.head.cards
    val hero = hole("Ah", "Kh")
    val result = RangeInferenceEngine.inferPosterior(
      hero = hero,
      board = Board.empty,
      folds = Vector.empty,
      tableRanges = TableRanges.defaults(TableFormat.HeadsUp),
      villainPos = Position.BigBlind,
      observations = Seq.empty,
      actionModel = PokerActionModel.uniform,
      revealedCards = Some(revealedCards),
      useCache = false
    )

    // 5. Assert delta posterior
    val prob = result.posterior.probabilityOf(revealedCards)
    assert(prob > 0.99, s"expected ~1.0 for revealed hand, got $prob")
    assertEquals(result.posterior.support.size, 1)
    assertEquals(result.collapse.effectiveSupportPosterior, 1.0)
  }
