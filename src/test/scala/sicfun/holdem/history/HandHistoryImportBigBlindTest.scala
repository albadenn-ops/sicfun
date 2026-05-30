package sicfun.holdem.history

class HandHistoryImportBigBlindTest extends munit.FunSuite:
  private val sample =
    """PokerStars Hand #900000001:  Hold'em No Limit ($1/$2 USD) - 2026/01/01 12:00:00 ET
      |Table 'P0Test' 9-max Seat #1 is the button
      |Seat 1: Hero ($200 in chips)
      |Seat 2: Villain2 ($200 in chips)
      |Villain2: posts small blind $1
      |Hero: posts big blind $2
      |*** HOLE CARDS ***
      |Dealt to Hero [As Ks]
      |Villain2: folds
      |Uncalled bet ($1) returned to Hero
      |Hero collected $2 from pot
      |*** SUMMARY ***
      |Total pot $2 | Rake $0
      |""".stripMargin

  test("importer captures the big blind amount") {
    val hands = HandHistoryImport.parseText(sample, Some(HandHistorySite.PokerStars), Some("Hero"))
    assert(hands.isRight, hands.toString)
    val h = hands.toOption.get.head
    assertEqualsDouble(h.bigBlind, 2.0, 1e-9)
  }

  test("big blind is captured per-hand and does not leak across hands") {
    val twoHands =
      """PokerStars Hand #900000001:  Hold'em No Limit ($1/$2 USD) - 2026/01/01 12:00:00 ET
        |Table 'P0Test' 9-max Seat #1 is the button
        |Seat 1: Hero ($200 in chips)
        |Seat 2: Villain2 ($200 in chips)
        |Villain2: posts small blind $1
        |Hero: posts big blind $2
        |*** HOLE CARDS ***
        |Dealt to Hero [As Ks]
        |Villain2: folds
        |Uncalled bet ($1) returned to Hero
        |Hero collected $2 from pot
        |*** SUMMARY ***
        |Total pot $2 | Rake $0
        |PokerStars Hand #900000002:  Hold'em No Limit ($2/$5 USD) - 2026/01/01 12:05:00 ET
        |Table 'P0Test' 9-max Seat #1 is the button
        |Seat 1: Hero ($500 in chips)
        |Seat 2: Villain2 ($500 in chips)
        |Villain2: posts small blind $2
        |Hero: posts big blind $5
        |*** HOLE CARDS ***
        |Dealt to Hero [Ad Kd]
        |Villain2: folds
        |Uncalled bet ($3) returned to Hero
        |Hero collected $5 from pot
        |*** SUMMARY ***
        |Total pot $5 | Rake $0
        |""".stripMargin

    val hands = HandHistoryImport.parseText(twoHands, Some(HandHistorySite.PokerStars), Some("Hero"))
    assert(hands.isRight, hands.toString)
    val hs = hands.toOption.get
    assertEquals(hs.length, 2)
    assertEqualsDouble(hs(0).bigBlind, 2.0, 1e-9)
    assertEqualsDouble(hs(1).bigBlind, 5.0, 1e-9) // proves per-hand reset, not leak from hand 1
  }
