package sicfun.holdem.bench

import sicfun.core.Card
import sicfun.holdem.types.HoleCards
import sicfun.holdem.equity.{HeadsUpEquityTable, HeadsUpEquityCanonicalTable}

/** Shared test/bench helpers for card parsing, hole card construction, and batch loading.
  *
  * Consolidates private helpers duplicated across 15 bench and test files.
  */
private[holdem] object BenchSupport:

  /** Parses a two-character card token like "Ac" or "Td" into a [[Card]].
    * Throws [[IllegalArgumentException]] on invalid tokens.
    */
  def card(token: String): Card =
    Card.parse(token).getOrElse(throw new IllegalArgumentException(s"bad card: $token"))

  /** Constructs a [[HoleCards]] from two card tokens (e.g., hole("Ac", "Kh")). */
  def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  /** Flat-array batch of packed matchup keys and associated key material for GPU dispatch. */
  final case class BatchData(
      packedKeys: Array[Long],
      keyMaterial: Array[Long]
  ):
    def size: Int = packedKeys.length

  /** Loads a batch of matchup keys from either the "full" (878K entries) or
    * "canonical" (~14K entries) table index, capped at `maxMatchups`.
    */
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
