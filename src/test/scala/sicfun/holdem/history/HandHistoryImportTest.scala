package sicfun.holdem.history
import sicfun.holdem.types.*

import munit.FunSuite

class HandHistoryImportTest extends FunSuite:
  private val sampleHand =
    """PokerStars Hand #1001:  Hold'em No Limit ($0.50/$1.00 USD) - 2026/03/10 12:00:00 ET
      |Table 'Alpha' 2-max Seat #1 is the button
      |Seat 1: Hero ($100.00 in chips)
      |Seat 2: Villain ($100.00 in chips)
      |Hero: posts small blind $0.50
      |Villain: posts big blind $1.00
      |*** HOLE CARDS ***
      |Dealt to Hero [Ac Kh]
      |Hero: raises $1.50 to $2.00
      |Villain: calls $1.00
      |*** FLOP *** [Ts 9h 8d]
      |Villain: checks
      |Hero: bets $3.00
      |Villain: raises $6.00 to $9.00
      |Hero: folds
      |Uncalled bet ($6.00) returned to Villain
      |Villain collected $8.50 from pot
      |*** SUMMARY ***
      |""".stripMargin

  test("parses PokerStars heads-up hand into normalized poker events") {
    val parsed = HandHistoryImport.parseText(
      sampleHand,
      site = Some(HandHistorySite.PokerStars),
      heroName = Some("Hero")
    )

    assert(parsed.isRight, s"parse failed: $parsed")
    val hand = parsed.toOption.get.head
    assertEquals(hand.tableName, "Alpha")
    assertEquals(hand.players.map(_.name), Vector("Hero", "Villain"))
    assertEquals(hand.players.map(_.position), Vector(Position.SmallBlind, Position.BigBlind))
    assertEquals(hand.heroName, Some("Hero"))
    assertEquals(hand.heroHoleCards.map(_.toToken), Some("AcKh"))
    assertEquals(hand.events.length, 6)

    val heroRaise = hand.events(0)
    assertEquals(heroRaise.playerId, "Hero")
    assertEquals(heroRaise.position, Position.SmallBlind)
    assertEquals(heroRaise.street, Street.Preflop)
    assertEquals(heroRaise.potBefore, 1.5)
    assertEquals(heroRaise.toCall, 0.5)
    assertEquals(heroRaise.stackBefore, 99.5)
    assertEquals(heroRaise.action, PokerAction.Raise(2.0))

    val villainCall = hand.events(1)
    assertEquals(villainCall.playerId, "Villain")
    assertEquals(villainCall.potBefore, 3.0)
    assertEquals(villainCall.toCall, 1.0)
    assertEquals(villainCall.stackBefore, 99.0)
    assertEquals(villainCall.action, PokerAction.Call)

    val villainRaise = hand.events(4)
    assertEquals(villainRaise.street, Street.Flop)
    assertEquals(villainRaise.board.cards.map(_.toToken), Vector("Ts", "9h", "8d"))
    assertEquals(villainRaise.potBefore, 7.0)
    assertEquals(villainRaise.toCall, 3.0)
    assertEquals(villainRaise.action, PokerAction.Raise(9.0))
    assertEquals(villainRaise.betHistory.lastOption.map(_.action), Some(PokerAction.Raise(3.0)))
  }
