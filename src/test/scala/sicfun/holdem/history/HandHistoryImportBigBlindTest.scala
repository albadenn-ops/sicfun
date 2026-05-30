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
