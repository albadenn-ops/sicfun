package sicfun.holdem

import sicfun.core.{Card, Deck, HandEvaluator, HandRank}
import java.util.concurrent.ConcurrentHashMap

/** A normalized feature vector extracted from a [[GameState]] and the player's hole cards.
  *
  * All values are scaled to approximately [0, 1] for consumption by ML models.
  *
  * @param values the ordered feature values; indices correspond to [[PokerFeatures.featureNames]]
  */
final case class PokerFeatures(values: Vector[Double]):
  /** The number of features in this vector. */
  def dimension: Int = values.length

/** Feature extraction logic that combines game-state signals with a hand-strength
  * estimate to produce a fixed-dimension feature vector for model training and inference.
  *
  * The five features are:
  *   - '''potOdds''' -- call price relative to the new pot, in [0, 1)
  *   - '''stackToPot''' -- stack-to-pot ratio clamped to [0, 10] then normalized to [0, 1]
  *   - '''streetOrdinal''' -- street ordinal / 3, mapping Preflop..River to [0, 1]
  *   - '''positionOrdinal''' -- position ordinal / 7, mapping SB..BTN to [0, 1]
  *   - '''handStrengthProxy''' -- equity vs. the full opponent range, in [0, 1]
  *
  * Hand strength results are cached in a thread-safe [[ConcurrentHashMap]] keyed
  * on `(Board, HoleCards)` to avoid redundant exhaustive evaluations.
  */
object PokerFeatures:
  /** Human-readable names for each feature dimension, in order. */
  val featureNames: Vector[String] = Vector(
    "potOdds",
    "stackToPot",
    "streetOrdinal",
    "positionOrdinal",
    "handStrengthProxy"
  )

  /** Number of features produced by [[extract]]. */
  inline val dimension = 5

  /** Thread-safe cache of hand-strength evaluations to avoid recomputing
    * exhaustive equity for the same board+hand combination.
    *
    * Bounded to avoid unbounded memory growth in long-running sessions.
    * When the cache exceeds [[MaxCacheSize]], it is cleared entirely
    * (simple but effective given the access pattern is typically bursty
    * within a single hand analysis).
    */
  private inline val MaxCacheSize = 100_000
  private val strengthCache = new ConcurrentHashMap[(Board, HoleCards), java.lang.Double]()

  /** Extracts a normalized feature vector from the given game state and hole cards.
    *
    * @param state the observable game state at the decision point
    * @param hand  the player's private hole cards
    * @return a [[PokerFeatures]] with all values scaled to approximately [0, 1]
    */
  def extract(state: GameState, hand: HoleCards): PokerFeatures =
    val strength = handStrengthProxy(state.board, hand)
    PokerFeatures(Vector(
      state.potOdds,
      math.min(state.stackToPot, 10.0) / 10.0,   // clamp SPR to [0, 10], normalize to [0, 1]
      state.street.ordinal.toDouble / 3.0,          // Preflop=0.0, River=1.0
      state.position.ordinal.toDouble / 7.0,        // SB=0.0, BTN=1.0
      strength
    ))

  /** Returns the hand's equity against the full opponent range for the given board.
    *
    * Pre-flop (board size < 3) returns a neutral 0.5 since exhaustive enumeration
    * would be prohibitively expensive and board texture is unknown.
    * Results are memoized in [[strengthCache]].
    */
  private[holdem] def handStrengthProxy(board: Board, hand: HoleCards): Double =
    if board.size < 3 then 0.5  // pre-flop: return neutral equity
    else
      val key = (board, hand)
      val cached = strengthCache.get(key)
      if cached != null then cached.doubleValue
      else
        val computed = computeHandStrength(board, hand)
        if strengthCache.size() >= MaxCacheSize then strengthCache.clear()
        strengthCache.putIfAbsent(key, java.lang.Double.valueOf(computed))
        computed

  /** Computes exact equity by exhaustively enumerating all possible opponent hole-card
    * combinations from the remaining deck, comparing hero's best 5-card hand to each.
    *
    * Ties count as half a win. Returns 0.5 if no opponent combinations exist.
    *
    * Optimization: board cards are pre-allocated into an Array to avoid
    * repeated Vector concatenation in the inner loop.
    */
  private def computeHandStrength(board: Board, hand: HoleCards): Double =
    val dead = hand.asSet ++ board.asSet
    val remaining = Deck.full.filterNot(dead.contains).toIndexedSeq
    val opponents = HoldemCombinator.holeCardsFrom(remaining)
    if opponents.isEmpty then 0.5
    else
      val heroCards = hand.toVector ++ board.cards
      val heroRank = evaluateBest(heroCards)
      val boardCards = board.cards
      var wins = 0.0
      var ties = 0.0
      var total = 0.0
      opponents.foreach { opp =>
        val oppCards = opp.toVector ++ boardCards
        val oppRank = evaluateBest(oppCards)
        val cmp = heroRank.compare(oppRank)
        if cmp > 0 then wins += 1.0
        else if cmp == 0 then ties += 1.0
        total += 1.0
      }
      (wins + ties * 0.5) / total  // ties count as half a win

  /** Selects the best 5-card hand rank from 5, 6, or 7 cards. */
  private def evaluateBest(cards: Vector[Card]): HandRank =
    cards.length match
      case 7 => HandEvaluator.evaluate7Cached(cards)
      case 6 => bestOf6(cards)
      case 5 => HandEvaluator.evaluate5Cached(cards)
      case n => throw new IllegalArgumentException(s"expected 5-7 cards, got $n")

  /** Evaluates all 6 possible 5-card subsets of a 6-card hand, returning the best rank.
    * Each subset is formed by removing one card at a time.
    */
  private def bestOf6(cards: Vector[Card]): HandRank =
    var best = HandEvaluator.evaluate5Cached(cards.patch(0, Nil, 1))
    var i = 1
    while i < 6 do
      val rank = HandEvaluator.evaluate5Cached(cards.patch(i, Nil, 1))
      if rank > best then best = rank
      i += 1
    best
