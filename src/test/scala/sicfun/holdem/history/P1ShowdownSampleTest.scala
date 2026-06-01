package sicfun.holdem.history

import munit.FunSuite
import java.nio.file.Paths

class P1ShowdownSampleTest extends FunSuite:
  test("p1 showdown sample imports with at least 2 hands that reveal holdings"):
    val url = getClass.getResource("/handhistory/p1-showdown-sample.txt")
    assert(url != null, "p1-showdown-sample.txt resource must exist")
    val hands = HandHistoryImport.parseFile(Paths.get(url.toURI)) match
      case Right(hs) => hs
      case Left(err) => fail(s"parse failed: $err")
    val withShowdown = hands.filter(_.showdownCards.nonEmpty)
    assert(withShowdown.size >= 2, s"expected ≥2 showdown hands, got ${withShowdown.size}")
    withShowdown.foreach { h =>
      h.showdownCards.keys.foreach { name =>
        assert(h.events.exists(_.playerId == name), s"revealed player $name should have events")
      }
    }
    val postflopRevealed = withShowdown.flatMap(h => h.events.filter(e =>
      h.showdownCards.contains(e.playerId) && e.street != sicfun.holdem.types.Street.Preflop))
    assert(postflopRevealed.nonEmpty, "need ≥1 postflop decision by a revealed player")
