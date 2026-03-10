package sicfun.holdem.io
import sicfun.holdem.types.*
import sicfun.holdem.engine.*
import sicfun.holdem.cli.*
import sicfun.holdem.equity.*

import sicfun.core.{Card, DiscreteDistribution}

import java.net.URLDecoder
import java.nio.charset.StandardCharsets
import java.nio.file.Path
import scala.jdk.CollectionConverters.*

/** Reads DDRE self-play datasets exported by `TexasHoldemPlayingHall`. */
object HoldemDdreDatasetIO:
  final case class Sample(
      hand: Int,
      tableId: Int,
      decisionIndex: Int,
      state: GameState,
      villainPosition: Position,
      villainStackBefore: Double,
      observations: Vector[VillainObservation],
      heroHole: HoleCards,
      villainHole: HoleCards,
      bayesLogEvidence: Double,
      prior: DiscreteDistribution[HoleCards],
      bayesPosterior: DiscreteDistribution[HoleCards]
  )

  private val ExpectedColumns = Vector(
    "hand",
    "tableId",
    "decisionIndex",
    "street",
    "board",
    "potBefore",
    "toCall",
    "heroPosition",
    "villainPosition",
    "heroStackBefore",
    "villainStackBefore",
    "betHistory",
    "villainObservations",
    "heroHole",
    "villainHole",
    "bayesLogEvidence",
    "priorSparse",
    "bayesPosteriorSparse"
  )

  def read(path: Path): Vector[Sample] =
    require(java.nio.file.Files.isRegularFile(path), s"DDRE dataset file not found: $path")
    val lines = java.nio.file.Files.readAllLines(path, StandardCharsets.UTF_8).asScala.toVector
    require(lines.nonEmpty, s"DDRE dataset file is empty: $path")
    val header = lines.head.split("\t", -1).toVector
    require(header == ExpectedColumns, s"DDRE dataset header mismatch: ${header.mkString(",")}")

    lines.drop(1).zipWithIndex.collect {
      case (line, index) if line.trim.nonEmpty =>
        parseRow(path, rowNumber = index + 2, line)
    }

  private def parseRow(path: Path, rowNumber: Int, line: String): Sample =
    val cols = line.split("\t", -1).toVector
    require(cols.length == ExpectedColumns.length, s"$path:$rowNumber expected ${ExpectedColumns.length} columns, got ${cols.length}")

    def col(i: Int): String = cols(i).trim

    val state = GameState(
      street = parseStreet(col(3), path, rowNumber),
      board = parseBoard(col(4), path, rowNumber),
      pot = parseDouble(col(5), path, rowNumber, "potBefore"),
      toCall = parseDouble(col(6), path, rowNumber, "toCall"),
      position = parsePosition(col(7), path, rowNumber, "heroPosition"),
      stackSize = parseDouble(col(9), path, rowNumber, "heroStackBefore"),
      betHistory = parseBetHistory(col(11), path, rowNumber)
    )

    Sample(
      hand = parseInt(col(0), path, rowNumber, "hand"),
      tableId = parseInt(col(1), path, rowNumber, "tableId"),
      decisionIndex = parseInt(col(2), path, rowNumber, "decisionIndex"),
      state = state,
      villainPosition = parsePosition(col(8), path, rowNumber, "villainPosition"),
      villainStackBefore = parseDouble(col(10), path, rowNumber, "villainStackBefore"),
      observations = parseObservations(col(12), path, rowNumber),
      heroHole = CliHelpers.parseHoleCards(col(13)),
      villainHole = CliHelpers.parseHoleCards(col(14)),
      bayesLogEvidence = parseDouble(col(15), path, rowNumber, "bayesLogEvidence"),
      prior = parseSparsePosterior(col(16), path, rowNumber, "priorSparse"),
      bayesPosterior = parseSparsePosterior(col(17), path, rowNumber, "bayesPosteriorSparse")
    )

  private def parseSparsePosterior(
      raw: String,
      path: Path,
      rowNumber: Int,
      field: String
  ): DiscreteDistribution[HoleCards] =
    if raw == "-" || raw.isEmpty then
      throw new IllegalArgumentException(s"$path:$rowNumber $field must be non-empty")
    val weights = raw.split("\\|").toVector.map { part =>
      val pieces = part.split(":", 2)
      if pieces.length != 2 then
        throw new IllegalArgumentException(s"$path:$rowNumber invalid $field entry: $part")
      val id = parseInt(pieces(0), path, rowNumber, s"$field.id")
      val probability = parseDouble(pieces(1), path, rowNumber, s"$field.probability")
      HoleCardsIndex.byId(id) -> probability
    }.toMap
    DiscreteDistribution(weights).normalized

  private def parseObservations(raw: String, path: Path, rowNumber: Int): Vector[VillainObservation] =
    if raw == "-" || raw.isEmpty then Vector.empty
    else
      raw.split("\\|").toVector.map { item =>
        val fields = item.split(",").toVector.map { part =>
          val pieces = part.split("=", 2)
          if pieces.length != 2 then
            throw new IllegalArgumentException(s"$path:$rowNumber invalid observation token: $part")
          pieces(0).trim -> pieces(1).trim
        }.toMap

        val street = parseStreet(requiredField(fields, "st", path, rowNumber), path, rowNumber)
        val action = parseAction(requiredField(fields, "a", path, rowNumber), path, rowNumber)
        val pot = parseDouble(requiredField(fields, "pot", path, rowNumber), path, rowNumber, "pot")
        val toCall = parseDouble(requiredField(fields, "call", path, rowNumber), path, rowNumber, "call")
        val position = parsePosition(requiredField(fields, "pos", path, rowNumber), path, rowNumber, "pos")
        val board = parseBoard(requiredField(fields, "board", path, rowNumber), path, rowNumber)
        val stackSize = fields.get("stack").flatMap(_.toDoubleOption).getOrElse(0.0)
        val betHistory = fields
          .get("history")
          .map(raw => URLDecoder.decode(raw, StandardCharsets.UTF_8))
          .map(parseBetHistory(_, path, rowNumber))
          .getOrElse(Vector.empty)
        VillainObservation(
          action = action,
          state = GameState(
            street = street,
            board = board,
            pot = pot,
            toCall = toCall,
            position = position,
            stackSize = stackSize,
            betHistory = betHistory
          )
        )
      }

  private def parseBetHistory(raw: String, path: Path, rowNumber: Int): Vector[BetAction] =
    if raw == "-" || raw.isEmpty then Vector.empty
    else
      raw.split("\\|").toVector.map { item =>
        val fields = item.split(",").toVector.map { part =>
          val pieces = part.split("=", 2)
          if pieces.length != 2 then
            throw new IllegalArgumentException(s"$path:$rowNumber invalid betHistory token: $part")
          pieces(0).trim -> pieces(1).trim
        }.toMap
        BetAction(
          player = parseInt(requiredField(fields, "p", path, rowNumber), path, rowNumber, "betHistory.player"),
          action = parseAction(requiredField(fields, "a", path, rowNumber), path, rowNumber)
        )
      }

  private def parseAction(raw: String, path: Path, rowNumber: Int): PokerAction =
    raw.trim.toLowerCase match
      case "fold" => PokerAction.Fold
      case "check" => PokerAction.Check
      case "call" => PokerAction.Call
      case value if value.startsWith("raise:") =>
        val amount = value.drop(6).toDoubleOption.getOrElse(
          throw new IllegalArgumentException(s"$path:$rowNumber invalid raise token: $raw")
        )
        PokerAction.Raise(amount)
      case _ =>
        throw new IllegalArgumentException(s"$path:$rowNumber invalid action token: $raw")

  private def parseBoard(raw: String, path: Path, rowNumber: Int): Board =
    if raw == "-" || raw.isEmpty then Board.empty
    else
      val cards = raw.split("\\s+").toVector.map { token =>
        Card.parse(token).getOrElse(
          throw new IllegalArgumentException(s"$path:$rowNumber invalid card token: $token")
        )
      }
      Board.from(cards)

  private def parseStreet(raw: String, path: Path, rowNumber: Int): Street =
    Street.values.find(_.toString == raw).getOrElse(
      throw new IllegalArgumentException(s"$path:$rowNumber invalid street: $raw")
    )

  private def parsePosition(raw: String, path: Path, rowNumber: Int, field: String): Position =
    Position.values.find(_.toString == raw).getOrElse(
      throw new IllegalArgumentException(s"$path:$rowNumber invalid $field: $raw")
    )

  private def parseInt(raw: String, path: Path, rowNumber: Int, field: String): Int =
    raw.toIntOption.getOrElse(
      throw new IllegalArgumentException(s"$path:$rowNumber invalid $field: $raw")
    )

  private def parseDouble(raw: String, path: Path, rowNumber: Int, field: String): Double =
    raw.toDoubleOption.getOrElse(
      throw new IllegalArgumentException(s"$path:$rowNumber invalid $field: $raw")
    )

  private def requiredField(
      fields: Map[String, String],
      key: String,
      path: Path,
      rowNumber: Int
  ): String =
    fields.getOrElse(
      key,
      throw new IllegalArgumentException(s"$path:$rowNumber missing field '$key'")
    )
