package sicfun.holdem.model
import sicfun.holdem.types.*

import sicfun.core.Card

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters.*

/**
 * TSV reader for supervised poker action training data files in the sicfun system.
 *
 * This is the input format for the [[TrainPokerActionModel]] CLI and the
 * [[PokerActionModel.train]] method. Each row represents a single observed poker decision
 * with full game state context, the player's hole cards, and the action taken.
 *
 * The TSV format is designed for easy generation from hand history parsers or self-play
 * simulations. Required columns are: street, board, potBefore, toCall, position,
 * stackBefore, action, holeCards.
 *
 * Key design decisions:
 *   - '''Flexible column ordering''': Columns are matched by header name (case-insensitive),
 *     not position, so files can include extra columns without breaking the parser.
 *   - '''Multiple card notation formats''': Board and hole cards accept space-separated,
 *     comma-separated, or concatenated 2-character tokens (e.g. "As Kd" or "AsKd").
 *   - '''Board-street consistency validation''': Each row validates that the board size
 *     matches the declared street (e.g. Flop must have exactly 3 board cards).
 *   - '''Flexible action parsing''': Raise amounts can use "raise:12.5" or "raise(12.5)" notation.
 *
 * TSV reader for supervised poker action training rows. */
object PokerActionTrainingDataIO:
  /** The set of column names that must be present in the TSV header (all lowercase). */
  private val RequiredColumns: Set[String] = Set(
    "street",
    "board",
    "potbefore",
    "tocall",
    "position",
    "stackbefore",
    "action",
    "holecards"
  )

  /** Convenience overload accepting a string path. */
  def readTsv(path: String): Vector[(GameState, HoleCards, PokerAction)] =
    readTsv(Paths.get(path))

  /** Reads a TSV training data file and returns (GameState, HoleCards, PokerAction) tuples.
   *
   * Validates that all required columns are present, then parses each data row
   * with full type and range validation.
   *
   * @param path the TSV file path
   * @return one tuple per valid data row, ready for model training
   * @throws IllegalArgumentException if the file is missing, empty, or has missing required columns
   */
  def readTsv(path: Path): Vector[(GameState, HoleCards, PokerAction)] =
    require(Files.exists(path), s"training file not found: $path")
    val lines = Files.readAllLines(path, StandardCharsets.UTF_8).asScala.toVector
    require(lines.nonEmpty, s"training file is empty: $path")

    val header = splitTsv(lines.head)
    val headerIndex = header.zipWithIndex.map { case (name, index) =>
      normalizeColumn(name) -> index
    }.toMap
    val missing = RequiredColumns.diff(headerIndex.keySet)
    require(missing.isEmpty, s"missing required columns: ${missing.toVector.sorted.mkString(", ")}")

    val rows = lines.drop(1).filter(_.trim.nonEmpty)
    rows.zipWithIndex.map { case (line, index) =>
      val rowNum = index + 2
      parseRow(path, rowNum, line, headerIndex)
    }

  /** Parses a single TSV row into a (GameState, HoleCards, PokerAction) tuple.
   * Constructs the game state with an empty bet history (not available in this format).
   */
  private def parseRow(
      path: Path,
      rowNum: Int,
      line: String,
      index: Map[String, Int]
  ): (GameState, HoleCards, PokerAction) =
    val fields = splitTsv(line)
    def value(name: String): String =
      val idx = index(name)
      if idx >= fields.length then
        throw new IllegalArgumentException(s"${path.toString}:$rowNum missing value for column $name")
      fields(idx).trim

    val street = parseStreet(value("street"), path, rowNum)
    val board = parseBoard(value("board"), street, path, rowNum)
    val position = parsePosition(value("position"), path, rowNum)
    val action = parseAction(value("action"), path, rowNum)
    val hole = parseHoleCards(value("holecards"), path, rowNum)
    val potBefore = parseDouble(value("potbefore"), "potBefore", path, rowNum)
    val toCall = parseDouble(value("tocall"), "toCall", path, rowNum)
    val stackBefore = parseDouble(value("stackbefore"), "stackBefore", path, rowNum)

    val state = GameState(
      street = street,
      board = board,
      pot = potBefore,
      toCall = toCall,
      position = position,
      stackSize = stackBefore,
      betHistory = Vector.empty
    )
    (state, hole, action)

  private def parseStreet(raw: String, path: Path, rowNum: Int): Street =
    Street.values.find(_.toString.equalsIgnoreCase(raw.trim)).getOrElse {
      throw new IllegalArgumentException(s"${path.toString}:$rowNum invalid street: $raw")
    }

  /** Parses a position string, tolerating underscores and hyphens (e.g. "Small_Blind" -> SmallBlind). */
  private def parsePosition(raw: String, path: Path, rowNum: Int): Position =
    val normalized = raw.trim.replace("_", "").replace("-", "")
    Position.values.find { p =>
      p.toString.equalsIgnoreCase(normalized) ||
      p.toString.replace("_", "").replace("-", "").equalsIgnoreCase(normalized)
    }.getOrElse {
      throw new IllegalArgumentException(s"${path.toString}:$rowNum invalid position: $raw")
    }

  /** Parses board cards and validates that the count matches the street's expected board size. */
  private def parseBoard(raw: String, street: Street, path: Path, rowNum: Int): Board =
    val cards = parseCardList(raw, "board", path, rowNum)
    val board = Board.from(cards)
    val expected = street.expectedBoardSize
    if board.size != expected then
      throw new IllegalArgumentException(
        s"${path.toString}:$rowNum street $street expects board size $expected, got ${board.size}"
      )
    board

  private def parseHoleCards(raw: String, path: Path, rowNum: Int): HoleCards =
    val cards = parseCardList(raw, "holeCards", path, rowNum)
    if cards.length != 2 then
      throw new IllegalArgumentException(s"${path.toString}:$rowNum holeCards must have 2 cards, got ${cards.length}")
    HoleCards.from(cards)

  /** Parses a poker action, accepting "fold", "check", "call", "raise:X", or "raise(X)". */
  private def parseAction(raw: String, path: Path, rowNum: Int): PokerAction =
    val lower = raw.trim.toLowerCase
    if lower == "fold" then PokerAction.Fold
    else if lower == "check" then PokerAction.Check
    else if lower == "call" then PokerAction.Call
    else if lower.startsWith("raise") then
      val tail = lower.stripPrefix("raise").trim
      val cleaned =
        if tail.startsWith("(") && tail.endsWith(")") then tail.drop(1).dropRight(1).trim
        else if tail.startsWith(":") then tail.drop(1).trim
        else tail
      if cleaned.isEmpty then
        throw new IllegalArgumentException(
          s"${path.toString}:$rowNum raise action requires amount, e.g. raise:12.5"
        )
      val amount = parseDouble(cleaned, "raiseAmount", path, rowNum)
      if amount <= 0.0 then
        throw new IllegalArgumentException(s"${path.toString}:$rowNum raise amount must be positive, got $amount")
      PokerAction.Raise(amount)
    else
      throw new IllegalArgumentException(s"${path.toString}:$rowNum invalid action: $raw")

  /** Parses a list of cards from various formats:
   *   - Space or comma separated: "As Kd 7c" or "As,Kd,7c"
   *   - Concatenated 2-char tokens: "AsKd7c"
   * Returns empty vector for "-" or empty string.
   */
  private def parseCardList(raw: String, column: String, path: Path, rowNum: Int): Vector[Card] =
    val token = raw.trim
    if token.isEmpty || token == "-" then Vector.empty
    else
      val parts =
        if token.contains(",") || token.contains(" ") then
          token.split("[,\\s]+").toVector.filter(_.nonEmpty)
        else if token.length % 2 == 0 then
          token.grouped(2).toVector
        else
          throw new IllegalArgumentException(
            s"${path.toString}:$rowNum invalid card list for $column: $raw"
          )

      parts.map { part =>
        Card.parse(part).getOrElse {
          throw new IllegalArgumentException(
            s"${path.toString}:$rowNum invalid card '$part' in $column"
          )
        }
      }

  private def parseDouble(raw: String, fieldName: String, path: Path, rowNum: Int): Double =
    val value =
      try raw.toDouble
      catch
        case _: NumberFormatException =>
          throw new IllegalArgumentException(s"${path.toString}:$rowNum invalid number for $fieldName: $raw")
    if value.isNaN || value.isInfinite then
      throw new IllegalArgumentException(s"${path.toString}:$rowNum non-finite number for $fieldName: $raw")
    value

  /** Splits a TSV line on tabs, preserving empty trailing fields with limit=-1. */
  private def splitTsv(line: String): Vector[String] =
    line.split("\t", -1).toVector

  /** Normalizes a column name to lowercase for case-insensitive matching. */
  private def normalizeColumn(name: String): String =
    name.trim.toLowerCase
