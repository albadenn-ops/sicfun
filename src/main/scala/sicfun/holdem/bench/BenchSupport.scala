package sicfun.holdem.bench

import sicfun.core.Card
import sicfun.holdem.types.HoleCards
import sicfun.holdem.equity.{HeadsUpEquityTable, HeadsUpEquityCanonicalTable}

/** Shared test/bench helpers for card parsing, hole card construction, and batch loading.
  *
  * Consolidates private helpers duplicated across 15 bench and test files.
  */
private[holdem] object BenchSupport:

  def card(token: String): Card =
    Card.parse(token).getOrElse(throw new IllegalArgumentException(s"bad card: $token"))

  def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  final case class BatchData(
      packedKeys: Array[Long],
      keyMaterial: Array[Long]
  ):
    def size: Int = packedKeys.length

  def loadBatch(table: String, maxMatchups: Long): BatchData =
    table.trim.toLowerCase match
      case "full" =>
        val batch = HeadsUpEquityTable.selectFullBatch(maxMatchups)
        BatchData(batch.packedKeys, batch.keyMaterial)
      case "canonical" =>
        val batch = HeadsUpEquityCanonicalTable.selectCanonicalBatch(maxMatchups)
        BatchData(batch.packedKeys, batch.keyMaterial)
      case other =>
        throw new IllegalArgumentException(s"unknown table '$other' (expected canonical or full)")
