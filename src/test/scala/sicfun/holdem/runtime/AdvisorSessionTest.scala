package sicfun.holdem.runtime
import sicfun.holdem.types.*
import sicfun.holdem.model.*
import sicfun.holdem.engine.*
import sicfun.holdem.equity.*
import sicfun.holdem.cli.*

import munit.FunSuite
import sicfun.core.Card

import scala.util.Random

class AdvisorSessionTest extends FunSuite:

  // Shared low-trial-count engine for fast tests
  private lazy val (engine, tableRanges) = buildTestEngine()

  private def buildTestEngine(): (RealTimeAdaptiveEngine, TableRanges) =
    val baseState = GameState(
      street = Street.Flop,
      board = Board.from(Seq(card("Ts"), card("9h"), card("8d"))),
      pot = 20.0, toCall = 10.0, position = Position.BigBlind,
      stackSize = 180.0, betHistory = Vector.empty
    )
    val checkState = baseState.copy(toCall = 0.0)
    val strong = HoleCards.from(Vector(card("Ah"), card("Ad")))
    val medium = HoleCards.from(Vector(card("Qc"), card("Jc")))
    val weak = HoleCards.from(Vector(card("7c"), card("2d")))
    val data: Seq[(GameState, HoleCards, PokerAction)] =
      Vector.fill(24)((baseState, strong, PokerAction.Raise(25.0))) ++
        Vector.fill(24)((baseState, medium, PokerAction.Call)) ++
        Vector.fill(24)((baseState, weak, PokerAction.Fold)) ++
        Vector.fill(12)((checkState, medium, PokerAction.Check))
    val artifact = PokerActionModel.trainVersioned(
      trainingData = data,
      learningRate = 0.1, iterations = 200, l2Lambda = 0.001,
      validationFraction = 0.25, splitSeed = 42L,
      maxMeanBrierScore = 2.0, failOnGate = false,
      modelId = "test-advisor", schemaVersion = "v1",
      source = "test", trainedAtEpochMillis = 1_000_000_000_000L
    )
    val tr = TableRanges.defaults(TableFormat.NineMax)
    val eng = new RealTimeAdaptiveEngine(
      tableRanges = tr, actionModel = artifact.model,
      bunchingTrials = 50, defaultEquityTrials = 200, minEquityTrials = 100
    )
    (eng, tr)

  private def freshSession(config: SessionConfig = SessionConfig()): AdvisorSession =
    new AdvisorSession(
      config = config,
      engine = engine,
      tableRanges = tableRanges,
      hand = None,
      stats = AdvisorSessionStats(),
      rng = new Random(42)
    )

  private def card(t: String): Card =
    Card.parse(t).getOrElse(throw new IllegalArgumentException(s"bad card: $t"))

  // ---- Command parser tests ----

  test("parse 'new' returns NewHand") {
    assertEquals(AdvisorCommandParser.parse("new"), AdvisorCommand.NewHand)
  }

  test("parse 'h AcKh' returns HeroCards") {
    val cmd = AdvisorCommandParser.parse("h AcKh")
    cmd match
      case AdvisorCommand.HeroCards(hc) =>
        assertEquals(hc.toToken, CliHelpers.parseHoleCards("AcKh").toToken)
      case other => fail(s"expected HeroCards, got $other")
  }

  test("parse 'hero Ac Kh' returns HeroCards (spaced)") {
    val cmd = AdvisorCommandParser.parse("hero Ac Kh")
    cmd match
      case AdvisorCommand.HeroCards(hc) =>
        assertEquals(hc.toToken, CliHelpers.parseHoleCards("AcKh").toToken)
      case other => fail(s"expected HeroCards, got $other")
  }

  test("parse 'h raise 6' returns HeroAction(Raise(6.0))") {
    assertEquals(AdvisorCommandParser.parse("h raise 6"), AdvisorCommand.HeroAction(PokerAction.Raise(6.0)))
  }

  test("parse 'v call' returns VillainAction(Call)") {
    assertEquals(AdvisorCommandParser.parse("v call"), AdvisorCommand.VillainAction(PokerAction.Call))
  }

  test("parse 'v bet 8' returns VillainAction(Raise(8.0))") {
    assertEquals(AdvisorCommandParser.parse("v bet 8"), AdvisorCommand.VillainAction(PokerAction.Raise(8.0)))
  }

  test("parse 'board Ts9h8d' returns DealBoard with 3 cards") {
    val cmd = AdvisorCommandParser.parse("board Ts9h8d")
    cmd match
      case AdvisorCommand.DealBoard(cards) =>
        assertEquals(cards.length, 3)
        assertEquals(cards.map(_.toToken), Vector("Ts", "9h", "8d"))
      case other => fail(s"expected DealBoard, got $other")
  }

  test("parse 'board Ts 9h 8d' returns DealBoard with 3 cards (spaced)") {
    val cmd = AdvisorCommandParser.parse("board Ts 9h 8d")
    cmd match
      case AdvisorCommand.DealBoard(cards) =>
        assertEquals(cards.length, 3)
      case other => fail(s"expected DealBoard, got $other")
  }

  test("parse '?' returns Advise") {
    assertEquals(AdvisorCommandParser.parse("?"), AdvisorCommand.Advise)
  }

  test("parse garbage returns Unknown") {
    val cmd = AdvisorCommandParser.parse("asdfqwer")
    assert(cmd.isInstanceOf[AdvisorCommand.Unknown])
  }

  // ---- Session state machine tests ----

  test("new hand posts blinds and sets pot correctly") {
    val session = freshSession()
    val result = session.execute(AdvisorCommand.NewHand)
    val h = result.session.hand.get
    assertEquals(h.handNumber, 1)
    assertEquals(h.pot, 3.0) // 1 + 2
    assertEquals(h.heroStack, 199.0) // 200 - 1 (SB)
    assertEquals(h.villainStack, 198.0) // 200 - 2 (BB)
    assertEquals(h.heroPosition, Position.SmallBlind)
    assert(result.output.exists(_.contains("Hand #1")))
  }

  test("hero cards set correctly") {
    val session = freshSession()
    val r1 = session.execute(AdvisorCommand.NewHand)
    val hc = CliHelpers.parseHoleCards("AcKh")
    val r2 = r1.session.execute(AdvisorCommand.HeroCards(hc))
    assertEquals(r2.session.hand.get.heroCards, Some(hc))
    assert(r2.output.exists(_.contains("Ac")))
  }

  test("hero raise updates pot and stacks") {
    val s = freshSession()
    val r1 = s.execute(AdvisorCommand.NewHand)
    // Hero is SB, toCall = 1 (BB - SB). Hero raises to 6.
    val r2 = r1.session.execute(AdvisorCommand.HeroAction(PokerAction.Raise(6.0)))
    val h = r2.session.hand.get
    // Hero already put in 1 (SB), now raises to 6, so additional 5 chips
    assertEquals(h.heroStack, 194.0) // 199 - 5
    assertEquals(h.pot, 8.0) // 3 + 5
    assert(h.toCall > 0.0, "villain should have something to call")
    assert(r2.output.exists(_.contains("raises")))
  }

  test("villain call updates pot and resets toCall") {
    val s = freshSession()
    val r1 = s.execute(AdvisorCommand.NewHand)
    val r2 = r1.session.execute(AdvisorCommand.HeroAction(PokerAction.Raise(6.0)))
    val r3 = r2.session.execute(AdvisorCommand.VillainAction(PokerAction.Call))
    val h = r3.session.hand.get
    assertEquals(h.toCall, 0.0)
    assertEquals(h.pot, 12.0) // hero 6 + villain 6
    assert(r3.output.exists(_.contains("calls")))
  }

  test("board command advances street") {
    val s = freshSession()
    val r1 = s.execute(AdvisorCommand.NewHand)
    val r2 = r1.session.execute(AdvisorCommand.HeroAction(PokerAction.Raise(6.0)))
    val r3 = r2.session.execute(AdvisorCommand.VillainAction(PokerAction.Call))
    val flop = Vector("Ts", "9h", "8d").map(card)
    val r4 = r3.session.execute(AdvisorCommand.DealBoard(flop))
    val h = r4.session.hand.get
    assertEquals(h.street, Street.Flop)
    assertEquals(h.board.size, 3)
    assertEquals(h.toCall, 0.0)
    assert(r4.output.exists(_.contains("Flop")))
  }

  test("fold ends hand and updates stats") {
    val s = freshSession()
    val r1 = s.execute(AdvisorCommand.NewHand)
    // Hero (SB) folds preflop
    val r2 = r1.session.execute(AdvisorCommand.HeroAction(PokerAction.Fold))
    val h = r2.session.hand.get
    assert(h.finished)
    assertEquals(r2.session.stats.handsPlayed, 1)
    assertEquals(r2.session.stats.heroLosses, 1)
    assert(r2.session.stats.heroNetChips < 0.0)
  }

  test("villain fold gives hero the pot") {
    val s = freshSession()
    val r1 = s.execute(AdvisorCommand.NewHand)
    val r2 = r1.session.execute(AdvisorCommand.HeroAction(PokerAction.Raise(6.0)))
    val r3 = r2.session.execute(AdvisorCommand.VillainAction(PokerAction.Fold))
    val h = r3.session.hand.get
    assert(h.finished)
    assert(h.heroNetResult > 0.0, s"hero should have positive result, got ${h.heroNetResult}")
    assertEquals(r3.session.stats.heroWins, 1)
  }

  test("position alternates between hands") {
    val s = freshSession()
    val r1 = s.execute(AdvisorCommand.NewHand)
    assertEquals(r1.session.hand.get.heroPosition, Position.SmallBlind)

    // Fold to finish hand 1
    val r2 = r1.session.execute(AdvisorCommand.HeroAction(PokerAction.Fold))
    val r3 = r2.session.execute(AdvisorCommand.NewHand)
    assertEquals(r3.session.hand.get.heroPosition, Position.BigBlind)

    // Fold to finish hand 2
    val r4 = r3.session.execute(AdvisorCommand.VillainAction(PokerAction.Fold))
    val r5 = r4.session.execute(AdvisorCommand.NewHand)
    assertEquals(r5.session.hand.get.heroPosition, Position.SmallBlind)
  }

  test("advise returns recommendation when hero cards are set".tag(munit.Slow)) {
    val s = freshSession()
    val r1 = s.execute(AdvisorCommand.NewHand)
    val hc = CliHelpers.parseHoleCards("AcKh")
    val r2 = r1.session.execute(AdvisorCommand.HeroCards(hc))
    val r3 = r2.session.execute(AdvisorCommand.HeroAction(PokerAction.Raise(6.0)))
    val r4 = r3.session.execute(AdvisorCommand.VillainAction(PokerAction.Call))
    val flop = Vector("Ts", "9h", "8d").map(card)
    val r5 = r4.session.execute(AdvisorCommand.DealBoard(flop))
    val r6 = r5.session.execute(AdvisorCommand.VillainAction(PokerAction.Raise(8.0)))
    val r7 = r6.session.execute(AdvisorCommand.Advise)

    assert(r7.output.exists(_.contains("Equity")), s"expected equity output, got: ${r7.output}")
    assert(r7.output.exists(_.contains("Villain")), s"expected archetype output, got: ${r7.output}")
  }

  test("advise without hero cards returns error") {
    val s = freshSession()
    val r1 = s.execute(AdvisorCommand.NewHand)
    val r2 = r1.session.execute(AdvisorCommand.Advise)
    assert(r2.output.exists(_.contains("Set hero cards")))
  }

  test("undo reverses the last action") {
    val s = freshSession()
    val r1 = s.execute(AdvisorCommand.NewHand)
    val r2 = r1.session.execute(AdvisorCommand.HeroAction(PokerAction.Raise(6.0)))
    assert(r2.session.hand.get.pot > 3.0, "pot should have increased after raise")
    val r3 = r2.session.execute(AdvisorCommand.Undo)
    val h = r3.session.hand.get
    assertEquals(h.pot, 3.0) // back to blind pot
    assertEquals(h.heroStack, 199.0)
    assert(r3.output.exists(_.contains("Undone")))
  }

  test("undo after board reverts street") {
    val s = freshSession()
    val r1 = s.execute(AdvisorCommand.NewHand)
    val r2 = r1.session.execute(AdvisorCommand.HeroAction(PokerAction.Raise(6.0)))
    val r3 = r2.session.execute(AdvisorCommand.VillainAction(PokerAction.Call))
    val flop = Vector("Ts", "9h", "8d").map(card)
    val r4 = r3.session.execute(AdvisorCommand.DealBoard(flop))
    assertEquals(r4.session.hand.get.street, Street.Flop)
    val r5 = r4.session.execute(AdvisorCommand.Undo)
    assertEquals(r5.session.hand.get.street, Street.Preflop)
    assertEquals(r5.session.hand.get.board, Board.empty)
  }

  test("session stats reflect completed hands") {
    val s = freshSession()
    val r1 = s.execute(AdvisorCommand.NewHand)
    val r2 = r1.session.execute(AdvisorCommand.HeroAction(PokerAction.Fold))
    val r3 = r2.session.execute(AdvisorCommand.SessionStats)
    assert(r3.output.exists(_.contains("Hands played: 1")))
  }

  test("check is rejected when toCall > 0") {
    val s = freshSession()
    val r1 = s.execute(AdvisorCommand.NewHand)
    // Hero is SB, toCall = 1
    val r2 = r1.session.execute(AdvisorCommand.HeroAction(PokerAction.Check))
    assert(r2.output.exists(_.contains("Cannot check")))
  }

  test("call is rejected when toCall == 0") {
    val s = freshSession(SessionConfig(heroStartsAsSB = false))
    val r1 = s.execute(AdvisorCommand.NewHand)
    // Hero is BB with heroStartsAsSB=false, but we need a situation where toCall == 0
    // After villain limps (calls), hero has option (toCall = 0)
    val r2 = r1.session.execute(AdvisorCommand.VillainAction(PokerAction.Call))
    val r3 = r2.session.execute(AdvisorCommand.HeroAction(PokerAction.Call))
    assert(r3.output.exists(_.contains("Nothing to call")))
  }

  test("invalid board size is rejected") {
    val s = freshSession()
    val r1 = s.execute(AdvisorCommand.NewHand)
    // Try dealing 2 cards (not valid: need 3 for flop)
    val cards = Vector("Ts", "9h").map(card)
    val r2 = r1.session.execute(AdvisorCommand.DealBoard(cards))
    assert(r2.output.exists(_.contains("Invalid board deal")))
  }

  test("help returns command list") {
    val s = freshSession()
    val r = s.execute(AdvisorCommand.Help)
    assert(r.output.exists(_.contains("new")))
    assert(r.output.exists(_.contains("advise")))
  }

  test("unknown command gives helpful message") {
    val s = freshSession()
    val r = s.execute(AdvisorCommand.Unknown("asdf", "unrecognized"))
    assert(r.output.exists(_.contains("Unknown")))
  }
