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
      heroNet: Double = 5.0
  ): HandRecord =
    HandRecord(
      handId = f"SIM-${handNumber}%08d",
      handNumber = handNumber,
      heroCards = heroCards,
      villainCards = villainCards,
      board = board,
      actions = if actions.nonEmpty then actions else defaultActions,
      heroNet = heroNet,
      streetsPlayed = 4
    )

  private val defaultActions: Vector[RecordedAction] = Vector(
    RecordedAction(Street.Preflop, "Hero", PokerAction.Call, 1.5, 0.5, 99.5, leakFired = false, leakId = None),
    RecordedAction(Street.Preflop, "Villain", PokerAction.Check, 2.0, 0.0, 99.0, leakFired = false, leakId = None),
    RecordedAction(Street.Flop, "Villain", PokerAction.Raise(4.0), 2.0, 0.0, 99.0, leakFired = false, leakId = None),
    RecordedAction(Street.Flop, "Hero", PokerAction.Call, 6.0, 4.0, 99.0, leakFired = false, leakId = None),
    RecordedAction(Street.Turn, "Villain", PokerAction.Check, 10.0, 0.0, 95.0, leakFired = false, leakId = None),
    RecordedAction(Street.Turn, "Hero", PokerAction.Check, 10.0, 0.0, 95.0, leakFired = false, leakId = None),
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
