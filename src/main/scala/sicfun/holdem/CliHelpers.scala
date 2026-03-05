package sicfun.holdem

import sicfun.core.{Card, DiscreteDistribution}
import java.util.Locale

/** Shared utilities for CLI comparison tools. */
private[holdem] object CliHelpers:
  def parseHoleCards(token: String): HoleCards =
    val t = token.trim
    require(t.length == 4, s"expected 4-char token like AcAs, got '$token'")
    val c1 = Card.parse(t.substring(0, 2)).getOrElse(
      throw new IllegalArgumentException(s"invalid card in '$token'")
    )
    val c2 = Card.parse(t.substring(2, 4)).getOrElse(
      throw new IllegalArgumentException(s"invalid card in '$token'")
    )
    HoleCards.canonical(c1, c2)

  def parseRangeDistribution(token: String): DiscreteDistribution[HoleCards] =
    RangeParser.parse(token.trim) match
      case Right(dist) => dist
      case Left(err) => throw new IllegalArgumentException(s"failed to parse range token '$token': $err")

  def fmt2(value: Double): String =
    String.format(Locale.ROOT, "%6.2f", java.lang.Double.valueOf(value))

  def fmt5(value: Double): String =
    String.format(Locale.ROOT, "%8.5f", java.lang.Double.valueOf(value))
