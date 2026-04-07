package sicfun.holdem.history
import sicfun.holdem.types.*
import sicfun.core.{Card, Rank, Suit}

import munit.FunSuite

/** Tests for the multi-site hand history parser [[HandHistoryImport]].
  *
  * Validates parsing of three poker site formats into normalized
  * [[ImportedHand]] / [[PokerEvent]] structures:
  *
  *   - '''PokerStars''': standard heads-up hand with explicit site hint;
  *     verifies table name, player positions, hero hole cards, event count,
  *     and detailed pot/stack/action fields for raises and calls
  *   - '''Winamax''': European format with euro amounts (comma decimals),
  *     no-colon action syntax; auto-detects site; verifies hand ID, pot
  *     tracking, and raise amounts
  *   - '''GGPoker''': short raise syntax ("raises to $X"); auto-detects
  *     site; verifies all-in handling, board cards on each street, and
  *     correct event sequencing through river
  *   - '''Forum hero alias normalization''': PokerStars hands pasted from
  *     forums where the hero name has a "(HERO)" suffix; verifies the
  *     alias is stripped and actions resolve to the canonical name
  *   - '''Showdown card extraction''': hands reaching showdown produce
  *     correct `showdownCards` maps; hands folded before showdown have
  *     empty maps; mucked/hidden hands are excluded
  */
class HandHistoryImportTest extends FunSuite:
  private val pokerStarsHand =
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

  private val winamaxHand =
    """Winamax Poker - CashGame - HandId: #2002-101-1700000000 - Holdem no limit (0,02 €/0,05 €) - 2026/03/10 18:00:00 UTC
      |Table: 'Bravo' 2-max (real money) Seat #1 is the button
      |Seat 1: Hero (10,00 €)
      |Seat 2: Villain (10,00 €)
      |Hero posts small blind 0,02 €
      |Villain posts big blind 0,05 €
      |*** HOLE CARDS ***
      |Dealt to Hero [Ac Kh]
      |Hero raises 0,10 € to 0,15 €
      |Villain calls 0,10 €
      |*** FLOP *** [Ts 9h 8d]
      |Villain checks
      |Hero bets 0,20 €
      |Villain raises 0,40 € to 0,60 €
      |Hero folds
      |*** SUMMARY ***
      |""".stripMargin

  private val ggPokerHand =
    """Hand #GG3003: Hold'em No Limit ($0.05/$0.10 USD) - 2026-03-10 20:30:00 UTC
      |Table 'Charlie' 2-max Seat #2 is the button
      |Seat 1: Hero ($10.00 in chips)
      |Seat 2: Villain ($10.00 in chips)
      |Hero: Posts small blind $0.05
      |Villain: Posts big blind $0.10
      |*** HOLE CARDS ***
      |Dealt to Hero [Qs Qd]
      |Hero: raises to $0.30
      |Villain: calls $0.20
      |*** FLOP ***[Qc 7h 2s]
      |Villain: checks
      |Hero: bets $0.45
      |Villain: raises to $1.50
      |Hero: calls $1.05
      |*** TURN *** [Qc 7h 2s] [9c]
      |Villain: bets $2.00
      |Hero: calls $2.00
      |*** RIVER *** [Qc 7h 2s 9c] [2d]
      |Villain: checks
      |Hero: bets $3.50 and is all-in
      |Villain: folds
      |*** SUMMARY ***
      |""".stripMargin

  private val pokerStarsForumHeroAliasHand =
    """PokerStars Hand #166193791537: Hold'em No Limit ($0.02/$0.05 USD) - 2017/02/14 19:34:00 CUST [2017/02/14 7:34:00 ET]
      |Table 'Hekatostos' 6-max Seat #6 is the button
      |Seat 1: TUNGLIMING(HERO) ($6.21 in chips)
      |Seat 3: blvm ($5 in chips)
      |Seat 4: ISniffBluffs ($7.05 in chips)
      |Seat 5: JoeDavola27 ($5.05 in chips)
      |Seat 6: vitales08 ($5 in chips)
      |TUNGLIMING: posts small blind $0.02
      |RETR0-RUS98: is sitting out
      |blvm: posts big blind $0.05
      |*** HOLE CARDS ***
      |Dealt to TUNGLIMING [7d 7c]
      |ISniffBluffs: folds
      |RETR0-RUS98 leaves the table
      |JoeDavola27: folds
      |vitales08: folds
      |TUNGLIMING(HERO): raises $0.10 to $0.15
      |blvm: calls $0.10
      |*** FLOP *** [8d 3s 7s]
      |TUNGLIMING: checks
      |blvm: checks
      |*** TURN *** [8d 3s 7s] [6s]
      |TUNGLIMING: bets $0.15
      |blvm: raises $0.25 to $0.40
      |TUNGLIMING: calls $0.25
      |*** RIVER *** [8d 3s 7s 6s] [Td]
      |TUNGLIMING: checks
      |blvm: bets $0.55
      |TUNGLIMING: calls $0.55
      |*** SHOW DOWN ***
      |""".stripMargin

  test("parses PokerStars heads-up hand into normalized poker events") {
    val parsed = HandHistoryImport.parseText(
      pokerStarsHand,
      site = Some(HandHistorySite.PokerStars),
      heroName = Some("Hero")
    )

    assert(parsed.isRight, s"parse failed: $parsed")
    val hand = parsed.toOption.get.head
    assertEquals(hand.tableName, "Alpha")
    assertEquals(hand.players.map(_.name), Vector("Hero", "Villain"))
    assertEquals(hand.players.map(_.position), Vector(Position.Button, Position.BigBlind))
    assertEquals(hand.heroName, Some("Hero"))
    assertEquals(hand.heroHoleCards.map(_.toToken), Some("AcKh"))
    assertEquals(hand.events.length, 6)

    val heroRaise = hand.events(0)
    assertEquals(heroRaise.playerId, "Hero")
    assertEquals(heroRaise.position, Position.Button)
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

  test("parses Winamax hand histories with euro amounts and no-colon actions") {
    val parsed = HandHistoryImport.parseText(
      winamaxHand,
      site = None,
      heroName = Some("Hero")
    )

    assert(parsed.isRight, s"parse failed: $parsed")
    val hand = parsed.toOption.get.head
    assertEquals(hand.site, HandHistorySite.Winamax)
    assertEquals(hand.handId, "2002-101-1700000000")
    assertEquals(hand.tableName, "Bravo")
    assertEquals(hand.players.map(_.name), Vector("Hero", "Villain"))
    assertEquals(hand.heroHoleCards.map(_.toToken), Some("AcKh"))
    assertEquals(hand.events.length, 6)

    val heroRaise = hand.events(0)
    assertEquals(heroRaise.potBefore, 0.07)
    assertEquals(heroRaise.toCall, 0.03)
    assertEquals(heroRaise.stackBefore, 9.98)
    assertEquals(heroRaise.action, PokerAction.Raise(0.15))

    val villainRaise = hand.events(4)
    assertEquals(villainRaise.street, Street.Flop)
    assertEquals(villainRaise.board.cards.map(_.toToken), Vector("Ts", "9h", "8d"))
    assertEquals(villainRaise.potBefore, 0.5)
    assertEquals(villainRaise.toCall, 0.2)
    assertEquals(villainRaise.stackBefore, 9.85)
    assertEquals(villainRaise.action, PokerAction.Raise(0.6))
  }

  test("parses GGPoker short raise syntax and auto-detects the site") {
    val parsed = HandHistoryImport.parseText(
      ggPokerHand,
      site = None,
      heroName = Some("Hero")
    )

    assert(parsed.isRight, s"parse failed: $parsed")
    val hand = parsed.toOption.get.head
    assertEquals(hand.site, HandHistorySite.GGPoker)
    assertEquals(hand.handId, "GG3003")
    assertEquals(hand.tableName, "Charlie")
    assertEquals(hand.heroHoleCards.map(_.toToken), Some("QdQs"))
    assertEquals(hand.events.length, 11)

    val heroRaise = hand.events(0)
    assertEquals(heroRaise.street, Street.Preflop)
    assertEquals(heroRaise.potBefore, 0.15)
    assertEquals(heroRaise.toCall, 0.05)
    assertEquals(heroRaise.stackBefore, 9.95)
    assertEquals(heroRaise.action, PokerAction.Raise(0.3))

    val villainRaise = hand.events(4)
    assertEquals(villainRaise.street, Street.Flop)
    assertEquals(villainRaise.board.cards.map(_.toToken), Vector("Qc", "7h", "2s"))
    assertEquals(villainRaise.potBefore, 1.05)
    assertEquals(villainRaise.toCall, 0.45)
    assertEquals(villainRaise.stackBefore, 9.7)
    assertEquals(villainRaise.action, PokerAction.Raise(1.5))

    val heroRiverBet = hand.events(9)
    assertEquals(heroRiverBet.street, Street.River)
    assertEquals(heroRiverBet.board.cards.map(_.toToken), Vector("Qc", "7h", "2s", "9c", "2d"))
    assertEquals(heroRiverBet.action, PokerAction.Raise(3.5))
  }

  test("normalizes forum hero markers so PokerStars public hands still import") {
    val parsed = HandHistoryImport.parseText(
      pokerStarsForumHeroAliasHand,
      site = Some(HandHistorySite.PokerStars),
      heroName = Some("TUNGLIMING(HERO)")
    )

    assert(parsed.isRight, s"parse failed: $parsed")
    val hand = parsed.toOption.get.head
    assertEquals(hand.heroName, Some("TUNGLIMING"))
    assert(hand.players.exists(_.name == "TUNGLIMING"))
    // First action is ISniffBluffs folds; TUNGLIMING acts later
    assert(hand.events.exists(_.playerId == "TUNGLIMING"))
    assert(hand.events.exists(event => event.playerId == "TUNGLIMING" && event.action == PokerAction.Call))
  }

  test("parseText extracts showdown cards"):
    val text = """PokerStars Hand #999: Hold'em No Limit ($1/$2) - 2025/01/01 12:00:00 ET
      |Table 'TestTable' 2-max Seat #1 is the button
      |Seat 1: Hero ($200 in chips)
      |Seat 2: Villain ($200 in chips)
      |Hero: posts small blind $1
      |Villain: posts big blind $2
      |*** HOLE CARDS ***
      |Dealt to Hero [Ac Kh]
      |Hero: raises $4 to $6
      |Villain: calls $4
      |*** FLOP *** [Ts 9h 8d]
      |Hero: bets $8
      |Villain: calls $8
      |*** TURN *** [Ts 9h 8d] [2c]
      |Hero: checks
      |Villain: checks
      |*** RIVER *** [Ts 9h 8d 2c] [3s]
      |Hero: checks
      |Villain: checks
      |*** SHOW DOWN ***
      |Villain: shows [Qh Qs] (a pair of Queens)
      |Hero: shows [Ac Kh] (high card Ace)
      |Villain collected $28 from pot
      |*** SUMMARY ***
      |Total pot $28 | Rake $0
      |""".stripMargin
    val parsed = HandHistoryImport.parseText(text, Some(HandHistorySite.PokerStars), Some("Hero"))
    assert(parsed.isRight, s"parse failed: ${parsed.left.getOrElse("")}")
    val hand = parsed.toOption.get.head
    assertEquals(hand.showdownCards.get("Villain"), Some(HoleCards.from(Seq(Card(Rank.Queen, Suit.Hearts), Card(Rank.Queen, Suit.Spades)))))
    assertEquals(hand.showdownCards.get("Hero"), Some(HoleCards.from(Seq(Card(Rank.Ace, Suit.Clubs), Card(Rank.King, Suit.Hearts)))))
    assertEquals(hand.showdownCards.size, 2)

  test("parseText handles hand with no showdown (fold before river)"):
    // Use the existing pokerStarsHand which has a fold — showdownCards should be empty
    val parsed = HandHistoryImport.parseText(pokerStarsHand, Some(HandHistorySite.PokerStars), Some("Hero"))
    assert(parsed.isRight)
    val hand = parsed.toOption.get.head
    assert(hand.showdownCards.isEmpty)

  test("parseText ignores mucked hands in showdown"):
    val text = """PokerStars Hand #997: Hold'em No Limit ($1/$2) - 2025/01/01 12:00:00 ET
      |Table 'TestTable' 2-max Seat #1 is the button
      |Seat 1: Hero ($200 in chips)
      |Seat 2: Villain ($200 in chips)
      |Hero: posts small blind $1
      |Villain: posts big blind $2
      |*** HOLE CARDS ***
      |Dealt to Hero [Ac Kh]
      |Hero: raises $4 to $6
      |Villain: calls $4
      |*** FLOP *** [Ts 9h 8d]
      |Hero: bets $8
      |Villain: calls $8
      |*** TURN *** [Ts 9h 8d] [2c]
      |Hero: checks
      |Villain: checks
      |*** RIVER *** [Ts 9h 8d 2c] [3s]
      |Hero: bets $10
      |Villain: folds
      |Uncalled bet ($10) returned to Hero
      |Hero collected $28 from pot
      |Hero: doesn't show hand
      |*** SUMMARY ***
      |Total pot $28 | Rake $0
      |""".stripMargin
    val parsed = HandHistoryImport.parseText(text, Some(HandHistorySite.PokerStars), Some("Hero"))
    assert(parsed.isRight)
    val hand = parsed.toOption.get.head
    assert(hand.showdownCards.isEmpty, "muck/doesn't show should not produce entries")
