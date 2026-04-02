package sicfun.holdem.validation

import munit.FunSuite
import sicfun.core.{Card, Rank, Suit}
import sicfun.holdem.history.{HandHistoryImport, HandHistorySite}
import sicfun.holdem.types.{Board, HoleCards, PokerAction, Street}

class PokerStarsExporterTest extends FunSuite:

  private def makeRecord(
      handNumber: Int = 1,
      heroCards: HoleCards = HoleCards(Card(Rank.Ace, Suit.Spades), Card(Rank.King, Suit.Spades)),
      villainCards: HoleCards = HoleCards(Card(Rank.Queen, Suit.Hearts), Card(Rank.Jack, Suit.Hearts)),
      board: Board = Board.from(Vector(
        Card(Rank.Ten, Suit.Clubs), Card(Rank.Nine, Suit.Diamonds),
        Card(Rank.Two, Suit.Hearts), Card(Rank.Five, Suit.Spades),
        Card(Rank.Eight, Suit.Clubs)
      )),
      actions: Vector[RecordedAction] = Vector.empty,
      heroNet: Double = 5.0,
      heroIsButton: Boolean = true,
      villainName: String = "Villain"
  ): HandRecord =
    HandRecord(
      handId = f"SIM-${handNumber}%08d",
      handNumber = handNumber,
      heroCards = heroCards,
      villainCards = villainCards,
      board = board,
      actions = if actions.nonEmpty then actions else if heroIsButton then defaultActionsHeroButton else defaultActionsHeroBigBlind,
      heroNet = heroNet,
      streetsPlayed = 4,
      heroSeat = if heroIsButton then 1 else 2,
      villainSeat = if heroIsButton then 2 else 1,
      heroIsButton = heroIsButton,
      villainName = villainName
    )

  private val defaultActionsHeroButton: Vector[RecordedAction] = Vector(
    RecordedAction(Street.Preflop, "Hero", PokerAction.Call, 1.5, 0.5, 99.5, leakFired = false, leakId = None),
    RecordedAction(Street.Preflop, "Villain", PokerAction.Check, 2.0, 0.0, 99.0, leakFired = false, leakId = None),
    RecordedAction(Street.Flop, "Villain", PokerAction.Raise(4.0), 2.0, 0.0, 99.0, leakFired = false, leakId = None),
    RecordedAction(Street.Flop, "Hero", PokerAction.Call, 6.0, 4.0, 99.0, leakFired = false, leakId = None),
    RecordedAction(Street.Turn, "Villain", PokerAction.Check, 10.0, 0.0, 95.0, leakFired = false, leakId = None),
    RecordedAction(Street.Turn, "Hero", PokerAction.Check, 10.0, 0.0, 95.0, leakFired = false, leakId = None),
    RecordedAction(Street.River, "Villain", PokerAction.Raise(8.0), 10.0, 0.0, 95.0, leakFired = false, leakId = None),
    RecordedAction(Street.River, "Hero", PokerAction.Fold, 18.0, 8.0, 95.0, leakFired = false, leakId = None)
  )

  private val defaultActionsHeroBigBlind: Vector[RecordedAction] = Vector(
    RecordedAction(Street.Preflop, "Villain", PokerAction.Call, 1.5, 0.5, 99.5, leakFired = false, leakId = None),
    RecordedAction(Street.Preflop, "Hero", PokerAction.Check, 2.0, 0.0, 99.0, leakFired = false, leakId = None),
    RecordedAction(Street.Flop, "Hero", PokerAction.Check, 2.0, 0.0, 99.0, leakFired = false, leakId = None),
    RecordedAction(Street.Flop, "Villain", PokerAction.Raise(4.0), 2.0, 0.0, 99.0, leakFired = false, leakId = None),
    RecordedAction(Street.Flop, "Hero", PokerAction.Call, 6.0, 4.0, 99.0, leakFired = false, leakId = None),
    RecordedAction(Street.Turn, "Hero", PokerAction.Check, 10.0, 0.0, 95.0, leakFired = false, leakId = None),
    RecordedAction(Street.Turn, "Villain", PokerAction.Check, 10.0, 0.0, 95.0, leakFired = false, leakId = None),
    RecordedAction(Street.River, "Hero", PokerAction.Check, 10.0, 0.0, 95.0, leakFired = false, leakId = None),
    RecordedAction(Street.River, "Villain", PokerAction.Raise(8.0), 10.0, 0.0, 95.0, leakFired = false, leakId = None),
    RecordedAction(Street.River, "Hero", PokerAction.Fold, 18.0, 8.0, 95.0, leakFired = false, leakId = None)
  )

  test("exportHands produces PokerStars-format text"):
    val record = makeRecord()
    val text = PokerStarsExporter.exportHands(Vector(record), "Hero", "Villain")
    assert(text.contains("PokerStars Hand #"), "must have PokerStars header")
    assert(text.contains("Hold'em No Limit"), "must identify game type")
    assert(text.contains("Hero"), "must mention hero")
    assert(text.contains("Villain"), "must mention villain")
    assert(text.contains("*** HOLE CARDS ***"), "must have hole cards section")
    assert(text.contains("*** FLOP ***"), "must have flop section")

  test("exportHands includes fold action"):
    val record = makeRecord()
    val text = PokerStarsExporter.exportHands(Vector(record), "Hero", "Villain")
    assert(text.contains("Hero: folds"), "fold action must appear")

  test("exportHands shows showdown when no fold"):
    val showdownActions = Vector(
      RecordedAction(Street.Preflop, "Hero", PokerAction.Call, 1.5, 0.5, 99.5, leakFired = false, leakId = None),
      RecordedAction(Street.Preflop, "Villain", PokerAction.Check, 2.0, 0.0, 99.0, leakFired = false, leakId = None)
    )
    val record = makeRecord(actions = showdownActions)
    val text = PokerStarsExporter.exportHands(Vector(record), "Hero", "Villain")
    assert(text.contains("*** SHOW DOWN ***"), "must show showdown when no fold")

  test("exportChunked splits into correct chunk count"):
    val records = (1 to 5).map(i => makeRecord(handNumber = i)).toVector
    val chunks = PokerStarsExporter.exportChunked(records, "Hero", "Villain", chunkSize = 2)
    assertEquals(chunks.size, 3) // 2 + 2 + 1
    assertEquals(chunks(0).handCount, 2)
    assertEquals(chunks(1).handCount, 2)
    assertEquals(chunks(2).handCount, 1)
    assertEquals(chunks(0).chunkIndex, 0)
    assertEquals(chunks(1).chunkIndex, 1)

  test("exported text roundtrips through HandHistoryImport.parseText"):
    val record = makeRecord()
    val text = PokerStarsExporter.exportHands(Vector(record), "Hero", "Villain")
    val parsed = HandHistoryImport.parseText(text, Some(HandHistorySite.PokerStars), Some("Hero"))
    parsed match
      case Right(hands) =>
        assert(hands.nonEmpty, "parseText must return at least one hand")
      case Left(err) =>
        fail(s"parseText failed: $err\n\nExported text:\n$text")

  test("multi-hand export roundtrips"):
    val records = (1 to 3).map(i => makeRecord(handNumber = i)).toVector
    val text = PokerStarsExporter.exportHands(records, "Hero", "Villain")
    val parsed = HandHistoryImport.parseText(text, Some(HandHistorySite.PokerStars), Some("Hero"))
    parsed match
      case Right(hands) =>
        assertEquals(hands.size, 3, "all 3 hands should parse")
      case Left(err) =>
        fail(s"parseText failed on multi-hand export: $err")

  test("exportHands honors mirrored heads-up seat metadata"):
    val record = makeRecord(heroIsButton = false)
    val text = PokerStarsExporter.exportHands(Vector(record), "Hero", "Villain")

    assert(text.contains("Seat #1 is the button"), "villain should hold the button when hero is big blind")
    assert(text.contains("Seat 1: Villain"), "villain should move to seat 1 in mirrored leg")
    assert(text.contains("Seat 2: Hero"), "hero should move to seat 2 in mirrored leg")
    assert(text.contains("Villain: posts small blind $0.50"), "villain should post the small blind")
    assert(text.contains("Hero: posts big blind $1.00"), "hero should post the big blind")

  test("multi-opponent export uses the per-record villain identity"):
    val first = makeRecord(handNumber = 1, villainName = "VillainA")
    val second = makeRecord(handNumber = 2, heroIsButton = false, villainName = "VillainB")
    val text = PokerStarsExporter.exportHands(Vector(first, second), "Hero", "IgnoredVillain")

    assert(text.contains("Seat 2: VillainA"), "first hand should keep its own villain identity")
    assert(text.contains("Seat 1: VillainB"), "second hand should use its own mirrored villain identity")
    assert(text.contains("VillainB: posts small blind $0.50"), "second hand should use the per-record villain name")

  test("exportHands converts simulator raise amounts into correct total-to text"):
    val raiseActions = Vector(
      RecordedAction(Street.Preflop, "Hero", PokerAction.Raise(2.5), 1.5, 0.5, 99.5, leakFired = false, leakId = None),
      RecordedAction(Street.Preflop, "Villain", PokerAction.Fold, 4.0, 2.0, 99.0, leakFired = false, leakId = None)
    )
    val record = makeRecord(actions = raiseActions)
    val text = PokerStarsExporter.exportHands(Vector(record), "Hero", "Villain")

    assert(text.contains("Hero: raises $2.00 to $3.00"), "raise text should reflect the total contribution after calling")

  test("mirrored-seat export roundtrips through HandHistoryImport.parseText"):
    val record = makeRecord(heroIsButton = false)
    val text = PokerStarsExporter.exportHands(Vector(record), "Hero", "Villain")
    val parsed = HandHistoryImport.parseText(text, Some(HandHistorySite.PokerStars), Some("Hero"))

    parsed match
      case Right(hands) =>
        assertEquals(hands.size, 1)
        assertEquals(hands.head.buttonSeatNumber, 1)
      case Left(err) =>
        fail(s"parseText failed on mirrored-seat export: $err")
