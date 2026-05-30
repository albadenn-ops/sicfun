package sicfun.holdem.history

import sicfun.holdem.types.Street

/** Verifies the checked-in synthetic 9-max hand-history corpus used by P0 Track B.
  *
  * Track B parses this resource and runs the multiway analyzer over it, so the fixture
  * must (a) parse cleanly with this repo's PokerStars importer, (b) carry a positive
  * big blind on every hand (needed for chip-EV -> bb/100 conversion), (c) include Hero
  * hole cards on every hand, and (d) contain at least one hand where Hero reaches a
  * post-flop street and acts, so the analyzer has hero decisions to score.
  */
class P0SampleResourceTest extends munit.FunSuite:
  test("p0 synthetic 9-max resource parses into hands with bigBlind and hero cards") {
    val txt = scala.io.Source.fromResource("handhistory/p0-sample-9max.txt").mkString
    val hands = HandHistoryImport.parseText(txt, Some(HandHistorySite.PokerStars), Some("Hero"))
    assert(hands.isRight, hands.toString)
    val hs = hands.toOption.get
    assert(hs.nonEmpty, "expected >=1 hand")
    hs.foreach(h => assert(h.bigBlind > 0.0, s"hand ${h.handId} missing bigBlind"))
    assert(hs.forall(_.heroHoleCards.isDefined), "every hand should have Hero hole cards")
    // at least one hand reaches post-flop (events beyond preflop) so Track B has decisions to score
    assert(
      hs.exists(_.events.exists(e => e.street != Street.Preflop)),
      "expected >=1 hand with post-flop action for the analyzer"
    )
    // stronger: Hero specifically must take a post-flop action somewhere in the corpus
    assert(
      hs.exists(h => h.events.exists(e => e.playerId == "Hero" && e.street != Street.Preflop)),
      "expected >=1 hand with a post-flop HERO action"
    )
  }
