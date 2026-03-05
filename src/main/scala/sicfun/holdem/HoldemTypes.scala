package sicfun.holdem

import sicfun.core.Card
import sicfun.core.CardId

/** A player's two private hole cards in Texas Hold'em.
  *
  * Cards are stored in canonical order (by deck index) so that
  * `HoleCards(Ah, Ks)` and `HoleCards(Ks, Ah)` share a single
  * representation. Construct via [[HoleCards.canonical]] or
  * [[HoleCards.from]] to guarantee ordering.
  *
  * @param first  the card with the lower deck index
  * @param second the card with the higher deck index
  */
final case class HoleCards(first: Card, second: Card):
  require(first != second, "hole cards must be distinct")

  /** Returns both cards as a two-element vector. */
  inline def toVector: Vector[Card] = Vector(first, second)

  /** Returns the canonical four-character token (e.g., "AhKs"). */
  inline def toToken: String = s"${first.toToken}${second.toToken}"

  /** Returns both cards as a set (useful for dead-card filtering). */
  inline def asSet: Set[Card] = Set(first, second)

  /** Checks whether this hand contains the given card. */
  inline def contains(card: Card): Boolean = first == card || second == card

  /** Returns true when the two hands do not share any card. */
  inline def isDisjointFrom(other: HoleCards): Boolean =
    first != other.first &&
      first != other.second &&
      second != other.first &&
      second != other.second

/** Factory methods for [[HoleCards]] with canonical ordering. */
object HoleCards:
  /** Constructs a [[HoleCards]] with the two cards in canonical (deck-index) order.
    *
    * @param a first card (in any order)
    * @param b second card (in any order)
    * @return a [[HoleCards]] where `first.deckIndex <= second.deckIndex`
    */
  def canonical(a: Card, b: Card): HoleCards =
    if CardId.toId(a) <= CardId.toId(b) then HoleCards(a, b) else HoleCards(b, a)

  /** Constructs canonical [[HoleCards]] from an arbitrary two-card sequence.
    *
    * @throws IllegalArgumentException if `cards` does not contain exactly 2 distinct cards
    */
  def from(cards: Seq[Card]): HoleCards =
    require(cards.length == 2, s"HoleCards.from expects 2 cards, got ${cards.length}")
    require(cards.distinct.length == 2, "hole cards must be distinct")
    canonical(cards(0), cards(1))

/** The community board in Texas Hold'em, containing 0 to 5 cards.
  *
  * A board with 0 cards represents preflop, 3 cards is the flop,
  * 4 is the turn, and 5 is the river.
  *
  * @param cards the community cards (must be distinct, length 0-5)
  */
final case class Board(cards: Vector[Card]):
  require(cards.length <= 5, s"board cannot exceed 5 cards, got ${cards.length}")
  require(cards.distinct.length == cards.length, "board cards must be distinct")

  /** Number of community cards currently dealt. */
  inline def size: Int = cards.length

  /** Number of board cards still to come (5 minus current size). */
  inline def missing: Int = 5 - cards.length

  /** Returns the board cards as a set (useful for dead-card filtering). */
  inline def asSet: Set[Card] = cards.toSet

/** Factory methods for [[Board]]. */
object Board:
  /** An empty preflop board. */
  val empty: Board = Board(Vector.empty)

  /** Constructs a board from an arbitrary card sequence. */
  def from(cards: Seq[Card]): Board = Board(cards.toVector)

/** Exact win/tie/loss probabilities for a heads-up equity calculation.
  *
  * All values are in the range [0, 1] and sum to 1.0.
  *
  * @param win  probability that hero wins outright
  * @param tie  probability of a split pot
  * @param loss probability that the villain wins
  */
final case class EquityResult(win: Double, tie: Double, loss: Double):
  /** Hero's equity: wins count fully, ties count as half. */
  inline def equity: Double = win + (tie / 2.0)

  /** Sum of all outcome probabilities (should be ~1.0 after normalization). */
  inline def total: Double = win + tie + loss

/** Win/tie/loss probabilities with an associated standard error, typically
  * produced by Monte Carlo estimation.
  *
  * @param win    estimated probability of hero winning
  * @param tie    estimated probability of a split pot
  * @param loss   estimated probability of hero losing
  * @param stderr standard error of the equity estimate
  */
final case class EquityResultWithError(win: Double, tie: Double, loss: Double, stderr: Double):
  /** Hero's equity: wins count fully, ties count as half. */
  inline def equity: Double = win + (tie / 2.0)

  /** Sum of all outcome probabilities (should be ~1.0). */
  inline def total: Double = win + tie + loss

  /** Drops the error bound, returning a plain [[EquityResult]]. */
  def toEquityResult: EquityResult = EquityResult(win, tie, loss)

/** Full Monte Carlo equity estimate with variance statistics.
  *
  * The `mean` field represents hero's equity (wins = 1.0, ties = 0.5, losses = 0.0),
  * computed via Welford's online algorithm for numerical stability.
  *
  * @param mean     estimated equity (expected share of the pot)
  * @param variance sample variance of per-trial outcomes
  * @param stderr   standard error of the mean (`sqrt(variance / trials)`)
  * @param trials   number of Monte Carlo iterations performed
  * @param winRate  fraction of trials where hero won outright
  * @param tieRate  fraction of trials resulting in a split pot
  * @param lossRate fraction of trials where hero lost
  */
final case class EquityEstimate(
    mean: Double,
    variance: Double,
    stderr: Double,
    trials: Int,
    winRate: Double,
    tieRate: Double,
    lossRate: Double
)

/** Multi-way equity result that includes the hero's expected pot share.
  *
  * In multiway pots, `share` accounts for split pots by dividing equity among
  * all tied players (e.g., a three-way tie yields share = 1/3 for that trial).
  * This is more informative than a simple win/tie/loss breakdown when multiple
  * opponents are involved.
  *
  * @param win   probability that hero has the sole best hand
  * @param tie   probability that hero ties with at least one villain
  * @param loss  probability that a villain holds the best hand
  * @param share hero's expected fraction of the pot (accounts for split pots)
  */
final case class EquityShareResult(win: Double, tie: Double, loss: Double, share: Double):
  /** Sum of all outcome probabilities (should be ~1.0 after normalization). */
  inline def total: Double = win + tie + loss
