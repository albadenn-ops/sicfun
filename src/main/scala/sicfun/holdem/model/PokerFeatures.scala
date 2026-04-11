package sicfun.holdem.model
import sicfun.holdem.types.*
import sicfun.holdem.equity.*
import sicfun.holdem.engine.HandStrengthEstimator
import sicfun.holdem.gpu.*

import sicfun.core.{Card, Deck, HandEvaluator, HandRank}
import java.util.concurrent.ConcurrentHashMap

/**
 * Full feature extraction (including hand strength) for the sicfun Bayesian inference pipeline.
 *
 * This file provides the 5-dimensional feature vector used by [[PokerActionModel]] for
 * Bayesian range inference. Unlike [[FeatureExtractor]] (which uses only observable features),
 * this extractor includes a hand-strength estimate derived from the player's private hole cards,
 * making it suitable for models that operate within the Bayesian range updater where the hero's
 * cards are known.
 *
 * The hand strength computation is the most expensive operation in the feature extraction
 * pipeline. Two modes are available, controlled by the `sicfun.poker.handStrength.mode`
 * system property (or `sicfun_POKER_HAND_STRENGTH_MODE` environment variable):
 *
 *   - '''heuristic''' (default): Fast rank-based proxy using hand category and tiebreak scores.
 *     Suitable for real-time loops where low latency matters more than exact equity.
 *   - '''exact''': Exhaustive enumeration of all opponent hole-card combinations from the
 *     remaining deck, computing true equity. Used for high-accuracy offline analysis.
 *
 * Results are cached in a thread-safe [[ConcurrentHashMap]] to avoid redundant computation.
 * The cache is bounded at 100K entries and cleared entirely when full (simple but effective
 * for the typical bursty access pattern within a single hand analysis).
 */

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
  *   - '''positionOrdinal''' -- normalized position ordinal, mapping SB..BTN to [0, 1]
  *   - '''handStrengthProxy''' -- equity vs. the full opponent range, in [0, 1]
  *
  * Hand strength results are cached in a thread-safe [[ConcurrentHashMap]] keyed
  * on `(Board, HoleCards)` to avoid redundant exhaustive evaluations.
  */
object PokerFeatures:
  private val HandStrengthModeProperty = "sicfun.poker.handStrength.mode"
  private val HandStrengthModeEnv = "sicfun_POKER_HAND_STRENGTH_MODE"
  private val HandStrengthModeHeuristic = "heuristic"
  private val HandStrengthModeExact = "exact"

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
      state.position.ordinal.toDouble / (Position.values.length.toDouble - 1.0),
      strength
    ))

  /** Returns the hand's equity against the full opponent range for the given board.
    *
    * Pre-flop (board size < 3) delegates to [[HandStrengthEstimator.preflopStrength]]
    * which uses rank-based heuristics to estimate preflop hand strength.
    * Results are memoized in [[strengthCache]].
    */
  private[holdem] def handStrengthProxy(board: Board, hand: HoleCards): Double =
    if board.size < 3 then HandStrengthEstimator.preflopStrength(hand)
    else
      val key = (board, hand)
      val cached = strengthCache.get(key)
      if cached != null then cached.doubleValue
      else
        val computed =
          if configuredHandStrengthMode == HandStrengthModeExact then computeHandStrengthExact(board, hand)
          else computeHandStrengthHeuristic(board, hand)
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
  private def computeHandStrengthExact(board: Board, hand: HoleCards): Double =
    val boardCards = board.cards
    val remaining = Deck.full.filterNot(c => hand.contains(c) || boardCards.contains(c)).toIndexedSeq
    val opponents = HoldemCombinator.holeCardsFrom(remaining)
    if opponents.isEmpty then 0.5
    else if boardCards.length == 5 then
      // River: 7 cards — use packed direct evaluation (no Vector/cache/HandRank overhead)
      val b0 = boardCards(0); val b1 = boardCards(1); val b2 = boardCards(2)
      val b3 = boardCards(3); val b4 = boardCards(4)
      val heroPacked = HandEvaluator.evaluate7PackedDirect(hand.first, hand.second, b0, b1, b2, b3, b4)
      var wins = 0.0
      var ties = 0.0
      var total = 0.0
      opponents.foreach { opp =>
        val oppPacked = HandEvaluator.evaluate7PackedDirect(opp.first, opp.second, b0, b1, b2, b3, b4)
        val cmp = Integer.compare(heroPacked, oppPacked)
        if cmp > 0 then wins += 1.0
        else if cmp == 0 then ties += 1.0
        total += 1.0
      }
      (wins + ties * 0.5) / total
    else
      // Flop/Turn: 5-6 cards
      val heroRank = evaluateBest(hand.toVector ++ boardCards)
      var wins = 0.0
      var ties = 0.0
      var total = 0.0
      opponents.foreach { opp =>
        val oppRank = evaluateBest(opp.toVector ++ boardCards)
        val cmp = heroRank.compare(oppRank)
        if cmp > 0 then wins += 1.0
        else if cmp == 0 then ties += 1.0
        total += 1.0
      }
      (wins + ties * 0.5) / total

  /** Fast rank-based proxy for postflop hand strength.
    *
    * Uses current best hand category plus tiebreak ranks to avoid exhaustive
    * opponent enumeration in real-time loops.
    */
  private def computeHandStrengthHeuristic(board: Board, hand: HoleCards): Double =
    val boardCards = board.cards
    val rank =
      if boardCards.length == 5 then
        // River: skip Vector concat + cache lookup, construct HandRank from packed int
        HandRank(HandEvaluator.evaluate7PackedDirect(
          hand.first, hand.second, boardCards(0), boardCards(1), boardCards(2), boardCards(3), boardCards(4)))
      else evaluateBest(hand.toVector ++ boardCards)
    val categoryScore = rank.category.strength.toDouble / 8.0
    val tieScore = normalizedTieScore(rank)
    clamp01((0.82 * categoryScore) + (0.18 * tieScore))

  private def normalizedTieScore(rank: HandRank): Double =
    val len = rank.tiebreakLength
    if len == 0 then 0.0
    else
      var weightedSum = 0.0
      var weight = 1.0
      var normalizer = 0.0
      var idx = 0
      while idx < len && idx < 5 do
        val value = rank.tiebreak(idx)
        val normalized = (math.max(2, math.min(14, value)).toDouble - 2.0) / 12.0
        weightedSum += weight * normalized
        normalizer += weight
        weight *= 0.5
        idx += 1
      if normalizer > 0.0 then weightedSum / normalizer else 0.0

  private def clamp01(value: Double): Double =
    math.max(0.0, math.min(1.0, value))

  private def configuredHandStrengthMode: String =
    configuredHandStrengthModeCached

  private lazy val configuredHandStrengthModeCached: String =
    GpuRuntimeSupport
      .resolveNonEmptyLower(HandStrengthModeProperty, HandStrengthModeEnv)
      .getOrElse(HandStrengthModeHeuristic)

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
