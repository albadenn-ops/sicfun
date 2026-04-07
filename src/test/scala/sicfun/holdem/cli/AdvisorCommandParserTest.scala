package sicfun.holdem.cli

import sicfun.holdem.types.*

import munit.FunSuite

/**
  * Tests for [[AdvisorCommandParser]] REPL input parsing.
  *
  * This suite provides comprehensive coverage of the advisor REPL command grammar,
  * organized by command category:
  *
  * '''Singleton commands''': `new`, `advise`/`?`/`advice`, `review`, `session`/`stats`,
  * `undo`, `help`, `quit`/`exit`/`q` -- verified with case-insensitivity and synonyms.
  *
  * '''Hero hole cards''': Parsing with "hero"/"h" prefix, separate tokens ("h Ac Kh"),
  * concatenated tokens ("h AcKh"), bare 4-character tokens ("AcKh"), and invalid cards.
  *
  * '''Hero actions''': `h raise 6`, `h bet 8.5` (bet = raise alias), `h call`, `h fold`,
  * `h check`, with error cases for missing/invalid/zero/negative raise amounts.
  *
  * '''Villain actions''': Mirrors hero actions with "v"/"villain" prefix, plus showdown
  * parsing ("v show QhQs", "v show Qh Qs", "v shows", "v showdown").
  *
  * '''Board''': Concatenated ("board Ts9h8d") and spaced ("board Ts 9h 8d") notations,
  * "b" prefix shorthand, single-card boards, and invalid card detection.
  *
  * '''Edge cases''': Empty input, whitespace-only input, unrecognized commands, leading/
  * trailing whitespace trimming, standalone "h"/"v"/"board" without sub-commands.
  */
class AdvisorCommandParserTest extends FunSuite:

  // ---- Singleton commands ----

  test("parses 'new' as NewHand") {
    assertEquals(AdvisorCommandParser.parse("new"), AdvisorCommand.NewHand)
  }

  test("parses 'new' case-insensitively") {
    assertEquals(AdvisorCommandParser.parse("NEW"), AdvisorCommand.NewHand)
    assertEquals(AdvisorCommandParser.parse("New"), AdvisorCommand.NewHand)
  }

  test("parses advise synonyms") {
    assertEquals(AdvisorCommandParser.parse("?"), AdvisorCommand.Advise)
    assertEquals(AdvisorCommandParser.parse("advise"), AdvisorCommand.Advise)
    assertEquals(AdvisorCommandParser.parse("advice"), AdvisorCommand.Advise)
  }

  test("parses review") {
    assertEquals(AdvisorCommandParser.parse("review"), AdvisorCommand.Review)
  }

  test("parses session stats synonyms") {
    assertEquals(AdvisorCommandParser.parse("session"), AdvisorCommand.SessionStats)
    assertEquals(AdvisorCommandParser.parse("stats"), AdvisorCommand.SessionStats)
  }

  test("parses undo") {
    assertEquals(AdvisorCommandParser.parse("undo"), AdvisorCommand.Undo)
  }

  test("parses help") {
    assertEquals(AdvisorCommandParser.parse("help"), AdvisorCommand.Help)
  }

  test("parses quit synonyms") {
    assertEquals(AdvisorCommandParser.parse("quit"), AdvisorCommand.Quit)
    assertEquals(AdvisorCommandParser.parse("exit"), AdvisorCommand.Quit)
    assertEquals(AdvisorCommandParser.parse("q"), AdvisorCommand.Quit)
  }

  // ---- Hero hole cards ----

  test("parses hero hole cards with 'hero' prefix") {
    val result = AdvisorCommandParser.parse("hero AcKh")
    result match
      case AdvisorCommand.HeroCards(hc) => assertEquals(hc.toToken, "AcKh")
      case other => fail(s"expected HeroCards, got $other")
  }

  test("parses hero hole cards with 'h' prefix") {
    val result = AdvisorCommandParser.parse("h AcKh")
    result match
      case AdvisorCommand.HeroCards(hc) => assertEquals(hc.toToken, "AcKh")
      case other => fail(s"expected HeroCards, got $other")
  }

  test("parses hero hole cards as separate tokens") {
    val result = AdvisorCommandParser.parse("h Ac Kh")
    result match
      case AdvisorCommand.HeroCards(hc) => assertEquals(hc.toToken, "AcKh")
      case other => fail(s"expected HeroCards, got $other")
  }

  test("parses bare 4-char hole card token as HeroCards") {
    val result = AdvisorCommandParser.parse("AcKh")
    result match
      case AdvisorCommand.HeroCards(hc) => assertEquals(hc.toToken, "AcKh")
      case other => fail(s"expected HeroCards, got $other")
  }

  test("invalid hero hole cards produce Unknown") {
    val result = AdvisorCommandParser.parse("h XxYy")
    result match
      case AdvisorCommand.Unknown(_, reason) => assert(reason.contains("invalid"), s"reason was: $reason")
      case other => fail(s"expected Unknown, got $other")
  }

  test("invalid bare 4-char token produces Unknown") {
    val result = AdvisorCommandParser.parse("ZzZz")
    result match
      case AdvisorCommand.Unknown(_, _) => () // ok
      case other => fail(s"expected Unknown, got $other")
  }

  // ---- Hero actions ----

  test("parses hero raise with amount") {
    val result = AdvisorCommandParser.parse("h raise 6")
    assertEquals(result, AdvisorCommand.HeroAction(PokerAction.Raise(6.0)))
  }

  test("parses hero bet as raise") {
    val result = AdvisorCommandParser.parse("h bet 8.5")
    assertEquals(result, AdvisorCommand.HeroAction(PokerAction.Raise(8.5)))
  }

  test("parses hero call") {
    assertEquals(AdvisorCommandParser.parse("h call"), AdvisorCommand.HeroAction(PokerAction.Call))
  }

  test("parses hero fold") {
    assertEquals(AdvisorCommandParser.parse("h fold"), AdvisorCommand.HeroAction(PokerAction.Fold))
  }

  test("parses hero check") {
    assertEquals(AdvisorCommandParser.parse("h check"), AdvisorCommand.HeroAction(PokerAction.Check))
  }

  test("hero raise without amount produces Unknown") {
    val result = AdvisorCommandParser.parse("h raise")
    result match
      case AdvisorCommand.Unknown(_, reason) => assert(reason.contains("amount"), s"reason was: $reason")
      case other => fail(s"expected Unknown, got $other")
  }

  test("hero raise with non-numeric amount produces Unknown") {
    val result = AdvisorCommandParser.parse("h raise abc")
    result match
      case AdvisorCommand.Unknown(_, reason) => assert(reason.contains("invalid number"), s"reason was: $reason")
      case other => fail(s"expected Unknown, got $other")
  }

  test("hero raise with zero amount produces Unknown") {
    val result = AdvisorCommandParser.parse("h raise 0")
    result match
      case AdvisorCommand.Unknown(_, reason) => assert(reason.contains("positive"), s"reason was: $reason")
      case other => fail(s"expected Unknown, got $other")
  }

  test("hero raise with negative amount produces Unknown") {
    val result = AdvisorCommandParser.parse("h raise -5")
    result match
      case AdvisorCommand.Unknown(_, reason) => assert(reason.contains("positive"), s"reason was: $reason")
      case other => fail(s"expected Unknown, got $other")
  }

  // ---- Villain actions ----

  test("parses villain raise with amount") {
    val result = AdvisorCommandParser.parse("v raise 20")
    assertEquals(result, AdvisorCommand.VillainAction(PokerAction.Raise(20.0)))
  }

  test("parses villain bet as raise") {
    val result = AdvisorCommandParser.parse("v bet 12")
    assertEquals(result, AdvisorCommand.VillainAction(PokerAction.Raise(12.0)))
  }

  test("parses villain call") {
    assertEquals(AdvisorCommandParser.parse("v call"), AdvisorCommand.VillainAction(PokerAction.Call))
  }

  test("parses villain fold") {
    assertEquals(AdvisorCommandParser.parse("v fold"), AdvisorCommand.VillainAction(PokerAction.Fold))
  }

  test("parses villain check") {
    assertEquals(AdvisorCommandParser.parse("v check"), AdvisorCommand.VillainAction(PokerAction.Check))
  }

  test("parses 'villain' prefix same as 'v'") {
    assertEquals(AdvisorCommandParser.parse("villain call"), AdvisorCommand.VillainAction(PokerAction.Call))
    val result = AdvisorCommandParser.parse("villain raise 15")
    assertEquals(result, AdvisorCommand.VillainAction(PokerAction.Raise(15.0)))
  }

  test("villain with invalid sub-command produces Unknown") {
    val result = AdvisorCommandParser.parse("v garbage")
    result match
      case AdvisorCommand.Unknown(_, reason) => assert(reason.contains("expected"), s"reason was: $reason")
      case other => fail(s"expected Unknown, got $other")
  }

  test("villain raise without amount produces Unknown") {
    val result = AdvisorCommandParser.parse("v raise")
    result match
      case AdvisorCommand.Unknown(_, reason) => assert(reason.contains("amount"), s"reason was: $reason")
      case other => fail(s"expected Unknown, got $other")
  }

  // ---- Board ----

  test("parses board from single concatenated token") {
    val result = AdvisorCommandParser.parse("board Ts9h8d")
    result match
      case AdvisorCommand.DealBoard(cards) =>
        assertEquals(cards.length, 3)
        assertEquals(cards.map(_.toToken), Vector("Ts", "9h", "8d"))
      case other => fail(s"expected DealBoard, got $other")
  }

  test("parses board from separate tokens") {
    val result = AdvisorCommandParser.parse("board Ts 9h 8d")
    result match
      case AdvisorCommand.DealBoard(cards) =>
        assertEquals(cards.length, 3)
        assertEquals(cards.map(_.toToken), Vector("Ts", "9h", "8d"))
      case other => fail(s"expected DealBoard, got $other")
  }

  test("parses board with 'b' prefix") {
    val result = AdvisorCommandParser.parse("b Ts9h8d")
    result match
      case AdvisorCommand.DealBoard(cards) => assertEquals(cards.length, 3)
      case other => fail(s"expected DealBoard, got $other")
  }

  test("board with single card works") {
    val result = AdvisorCommandParser.parse("board Ah")
    result match
      case AdvisorCommand.DealBoard(cards) =>
        assertEquals(cards.length, 1)
        assertEquals(cards.head.toToken, "Ah")
      case other => fail(s"expected DealBoard, got $other")
  }

  test("board with invalid card produces Unknown") {
    val result = AdvisorCommandParser.parse("board Xx9h8d")
    result match
      case AdvisorCommand.Unknown(_, reason) => assert(reason.contains("invalid card"), s"reason was: $reason")
      case other => fail(s"expected Unknown, got $other")
  }

  // ---- Villain showdown ----

  test("parses 'v show QhQs' as VillainShowdown") {
    val cmd = AdvisorCommandParser.parse("v show QhQs")
    cmd match
      case AdvisorCommand.VillainShowdown(cards) =>
        assertEquals(cards.toToken, "QhQs")
      case other => fail(s"expected VillainShowdown, got $other")
  }

  test("parses 'v show Qh Qs' (spaced) as VillainShowdown") {
    val cmd = AdvisorCommandParser.parse("v show Qh Qs")
    cmd match
      case AdvisorCommand.VillainShowdown(cards) =>
        assertEquals(cards.toToken, "QhQs")
      case other => fail(s"expected VillainShowdown, got $other")
  }

  test("parses 'v shows QhQs' as VillainShowdown") {
    assertEquals(
      AdvisorCommandParser.parse("v shows QhQs"),
      AdvisorCommandParser.parse("v show QhQs")
    )
  }

  test("parses 'v showdown QhQs' as VillainShowdown") {
    assertEquals(
      AdvisorCommandParser.parse("v showdown QhQs"),
      AdvisorCommandParser.parse("v show QhQs")
    )
  }

  test("v show without cards produces Unknown") {
    val cmd = AdvisorCommandParser.parse("v show")
    cmd match
      case AdvisorCommand.Unknown(_, reason) =>
        assert(reason.contains("showdown requires cards"), s"reason was: $reason")
      case other => fail(s"expected Unknown, got $other")
  }

  test("v show with invalid cards produces Unknown") {
    val cmd = AdvisorCommandParser.parse("v show XxYy")
    cmd match
      case AdvisorCommand.Unknown(_, reason) =>
        assert(reason.contains("invalid hole cards"), s"reason was: $reason")
      case other => fail(s"expected Unknown, got $other")
  }

  // ---- Edge cases ----

  test("empty input produces Unknown with empty input reason") {
    val result = AdvisorCommandParser.parse("")
    result match
      case AdvisorCommand.Unknown("", reason) => assert(reason.contains("empty"), s"reason was: $reason")
      case other => fail(s"expected Unknown with empty input, got $other")
  }

  test("whitespace-only input produces Unknown") {
    val result = AdvisorCommandParser.parse("   ")
    result match
      case AdvisorCommand.Unknown("", reason) => assert(reason.contains("empty"), s"reason was: $reason")
      case other => fail(s"expected Unknown, got $other")
  }

  test("unrecognized command produces Unknown") {
    val result = AdvisorCommandParser.parse("foobar")
    result match
      case AdvisorCommand.Unknown(_, reason) => assert(reason.contains("unrecognized"), s"reason was: $reason")
      case other => fail(s"expected Unknown, got $other")
  }

  test("leading and trailing whitespace is trimmed") {
    assertEquals(AdvisorCommandParser.parse("  new  "), AdvisorCommand.NewHand)
    assertEquals(AdvisorCommandParser.parse("  h call  "), AdvisorCommand.HeroAction(PokerAction.Call))
  }

  test("hero without sub-command produces Unknown") {
    // "h" alone has parts.length == 1, so it falls through to the bare-token branch
    val result = AdvisorCommandParser.parse("h")
    result match
      case AdvisorCommand.Unknown(_, _) => () // ok
      case other => fail(s"expected Unknown, got $other")
  }

  test("villain without sub-command produces Unknown") {
    val result = AdvisorCommandParser.parse("v")
    result match
      case AdvisorCommand.Unknown(_, _) => () // ok
      case other => fail(s"expected Unknown, got $other")
  }

  test("board without cards produces Unknown or requires sub-command") {
    // "board" alone has parts.length == 1, so it falls to default case
    val result = AdvisorCommandParser.parse("board")
    result match
      case AdvisorCommand.Unknown(_, _) => () // ok
      case other => fail(s"expected Unknown, got $other")
  }
