package sicfun.holdem.bench

import sicfun.core.Card
import sicfun.holdem.types.HoleCards

/** Shared test/bench helpers for card parsing, hole card construction, and batch loading.
  *
  * Consolidates private helpers duplicated across 15 bench and test files.
  */
private[holdem] object BenchSupport:

  def card(token: String): Card =
    Card.parse(token).getOrElse(throw new IllegalArgumentException(s"bad card: $token"))

  def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))
