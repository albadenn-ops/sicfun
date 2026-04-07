package sicfun.holdem.equity
import sicfun.holdem.types.*

import sicfun.core.{Card, CardId, Deck, DiscreteDistribution}

import java.util.concurrent.ConcurrentHashMap
import scala.util.Random

/**
  * Represents a preflop fold event for a specific table position.
  *
  * Used by the bunching engine to track which positions folded before the opener,
  * allowing the engine to condition the villain's range on the information revealed
  * by those folds. For example, if UTG folds, hands like AA/KK become less likely
  * in UTG's range, which in turn shifts the conditional probabilities of what
  * remaining players might hold.
  *
  * @param position the table position that folded preflop
  */
final case class PreflopFold(position: Position)

/**
  * Complete result of a bunching-aware equity computation against a positional villain range.
  *
  * Contains both the naive (ignoring folds) and bunching-adjusted villain ranges, their
  * corresponding equity estimates, and the bunching delta (the equity shift caused by
  * accounting for the information revealed by earlier folds).
  *
  * A positive `bunchingDelta` means the hero's equity is higher when bunching is considered
  * (folded players were more likely to hold hands that would have competed with the villain),
  * while a negative delta means the adjusted equity is lower than naive.
  *
  * @param adjustedRange the villain's range after conditioning on observed fold actions
  * @param naiveRange    the villain's range without any fold conditioning (dead-card filtered only)
  * @param adjustedEquity equity estimate computed against the bunching-adjusted range
  * @param naiveEquity    equity estimate computed against the naive (unmodified) range
  * @param bunchingDelta  difference in mean equity: adjustedEquity.mean - naiveEquity.mean
  * @param trials         number of Monte Carlo fold-sampling trials used
  */
final case class BunchingResult(
    adjustedRange: DiscreteDistribution[HoleCards],
    naiveRange: DiscreteDistribution[HoleCards],
    adjustedEquity: EquityEstimate,
    naiveEquity: EquityEstimate,
    bunchingDelta: Double,
    trials: Int
)

/**
  * Monte Carlo bunching-effect engine for Texas Hold'em poker.
  *
  * In multi-way pots, when players fold preflop, they reveal information about their
  * likely holdings. This "bunching effect" shifts the remaining players' effective ranges.
  * For example, if five tight players fold before the button opens, the button's actual
  * range (conditioned on those folds) differs from its nominal opening range, because
  * the folds make it less likely that premium cards are still in play.
  *
  * The algorithm works in two phases:
  *   1. '''Fold sampling''': For each trial, sample folded hole cards for each fold position
  *      in preflop action order, weighted by each position's fold probability per hand.
  *      Each sampled fold must not conflict with hero cards, board cards, or previously
  *      sampled folds (card exclusion via bitmask).
  *   2. '''Range accumulation''': For each successful fold configuration, accumulate the
  *      villain's hand weights (only hands compatible with the sampled dead cards).
  *      After all trials, normalize to produce the adjusted villain range.
  *
  * The engine uses 64-bit card bitmasks for O(1) card-conflict detection, and caches
  * weighted hand vectors per (TableRanges, Position) pair in concurrent hash maps to
  * amortize repeated lookups.
  *
  * @see [[TableRanges]] for the per-position range configuration
  * @see [[TableFormat]] for supported table sizes and preflop action orderings
  */
object BunchingEffect:
  /** Maximum ratio of total attempts to requested trials before giving up on fold sampling. */
  private val DefaultMaxAttemptFactor = 3
  /** Upper bound on cache entries before eviction (simple clear-all strategy). */
  private val MaxWeightedHandsCacheSize = 128

  /**
    * A hole-card hand with its associated weight (fold or open probability) and
    * precomputed 64-bit card bitmask for fast conflict detection.
    *
    * @param hand   the two-card holding
    * @param weight probability weight (fold probability for fold candidates, open probability for villain hands)
    * @param mask   bitmask with bits set at positions corresponding to the two card IDs
    */
  private final case class WeightedHand(hand: HoleCards, weight: Double, mask: Long)

  /**
    * Cache key using reference identity for the TableRanges object (not deep equality)
    * combined with position. This is safe because TableRanges instances are typically
    * reused across calls within the same analysis session.
    */
  private final class WeightedHandsCacheKey private (
      val tableRangesRef: TableRanges,
      val position: Position
  ):
    override def hashCode(): Int =
      (System.identityHashCode(tableRangesRef) * 31) + position.hashCode()

    override def equals(obj: Any): Boolean =
      obj match
        case that: WeightedHandsCacheKey =>
          (this.position == that.position) && (this.tableRangesRef eq that.tableRangesRef)
        case _ => false

  private object WeightedHandsCacheKey:
    def apply(tableRanges: TableRanges, position: Position): WeightedHandsCacheKey =
      new WeightedHandsCacheKey(tableRanges, position)

  /** All 1326 canonical hole-card hands with precomputed bitmasks, lazily initialized once. */
  private lazy val allHoleCardsWithMasks: Vector[(HoleCards, Long)] =
    HoldemCombinator.holeCardsFrom(Deck.full).map { hand =>
      hand -> handMask(hand)
    }

  /** Thread-safe cache for fold-candidate weighted hands, keyed by (TableRanges ref, Position). */
  private val foldHandsCache = new ConcurrentHashMap[WeightedHandsCacheKey, Vector[WeightedHand]]()
  /** Thread-safe cache for villain weighted hands, keyed by (TableRanges ref, Position). */
  private val villainHandsCache = new ConcurrentHashMap[WeightedHandsCacheKey, Vector[WeightedHand]]()

  /**
    * Performs a full bunching analysis: computes both the adjusted and naive villain ranges,
    * runs Monte Carlo equity estimation against each, and returns the bunching delta.
    *
    * @param hero         the hero's hole cards
    * @param board        the community cards (may be empty for preflop analysis)
    * @param folds        the sequence of observed preflop fold events
    * @param tableRanges  per-position range configuration with open/fold frequencies
    * @param villainPos   the villain's table position
    * @param trials       number of Monte Carlo fold-sampling trials (default: 10,000)
    * @param equityTrials number of Monte Carlo equity trials per range (default: 50,000)
    * @param rng          random number generator for reproducibility
    * @return a [[BunchingResult]] containing both ranges, equities, and the delta
    */
  def compute(
      hero: HoleCards,
      board: Board,
      folds: Vector[PreflopFold],
      tableRanges: TableRanges,
      villainPos: Position,
      trials: Int = 10_000,
      equityTrials: Int = 50_000,
      rng: Random = new Random()
  ): BunchingResult =
    require(equityTrials > 0, "equityTrials must be positive")
    validateInputs(hero, board, folds, tableRanges, villainPos, trials)
    val adjusted = sampleAdjustedRange(hero, board, folds, tableRanges, villainPos, trials, rng)
    val naive = naiveVillainRange(hero, board, tableRanges, villainPos)

    val adjustedEquity = HoldemEquity.equityMonteCarlo(
      hero = hero,
      board = board,
      villainRange = adjusted,
      trials = equityTrials,
      rng = new Random(rng.nextLong())
    )
    val naiveEquity = HoldemEquity.equityMonteCarlo(
      hero = hero,
      board = board,
      villainRange = naive,
      trials = equityTrials,
      rng = new Random(rng.nextLong())
    )
    val delta = adjustedEquity.mean - naiveEquity.mean
    BunchingResult(adjusted, naive, adjustedEquity, naiveEquity, delta, trials)

  /**
    * Computes only the bunching-adjusted villain range without running equity calculations.
    *
    * Useful when the caller needs just the range adjustment (e.g., for display or further
    * analysis) without the computational cost of equity Monte Carlo.
    *
    * @param hero        the hero's hole cards
    * @param board       the community cards (may be empty for preflop)
    * @param folds       observed preflop fold events
    * @param tableRanges per-position range and frequency configuration
    * @param villainPos  the villain's table position
    * @param trials      number of fold-sampling Monte Carlo trials
    * @param rng         random number generator
    * @return the bunching-adjusted villain range as a normalized distribution
    */
  def adjustedRange(
      hero: HoleCards,
      board: Board,
      folds: Vector[PreflopFold],
      tableRanges: TableRanges,
      villainPos: Position,
      trials: Int = 10_000,
      rng: Random = new Random()
  ): DiscreteDistribution[HoleCards] =
    validateInputs(hero, board, folds, tableRanges, villainPos, trials)
    sampleAdjustedRange(hero, board, folds, tableRanges, villainPos, trials, rng)

  /**
    * Computes just the bunching delta as a single scalar (equity shift due to bunching).
    *
    * Convenience method that runs the full `compute` pipeline and extracts only the delta.
    * Positive delta means hero benefits from bunching, negative means hero is worse off.
    *
    * @return the equity difference: adjusted_equity - naive_equity
    */
  def bunchingDelta(
      hero: HoleCards,
      board: Board,
      folds: Vector[PreflopFold],
      tableRanges: TableRanges,
      villainPos: Position,
      trials: Int = 10_000,
      equityTrials: Int = 50_000,
      rng: Random = new Random()
  ): Double =
    compute(hero, board, folds, tableRanges, villainPos, trials, equityTrials, rng).bunchingDelta

  /**
    * Convenience API for the common "hero opens from openerPos after everyone before folded" scenario.
    *
    * Automatically derives the fold list from the table format's preflop action order:
    * all positions that act before `openerPos` are assumed to have folded.
    * Uses an empty board (preflop analysis).
    *
    * @param hero        the hero's hole cards
    * @param tableRanges per-position range configuration
    * @param openerPos   the position from which hero opens (raise first in)
    * @param villainPos  the villain's position (must be different from openerPos and not in folds)
    * @param trials      fold-sampling trial count
    * @param equityTrials equity Monte Carlo trial count
    * @param rng         random number generator
    * @return a full [[BunchingResult]]
    */
  def computeForOpener(
      hero: HoleCards,
      tableRanges: TableRanges,
      openerPos: Position,
      villainPos: Position,
      trials: Int = 10_000,
      equityTrials: Int = 50_000,
      rng: Random = new Random()
  ): BunchingResult =
    val folds = tableRanges.format.foldsBeforeOpener(openerPos).map(PreflopFold(_))
    compute(
      hero = hero,
      board = Board.empty,
      folds = folds,
      tableRanges = tableRanges,
      villainPos = villainPos,
      trials = trials,
      equityTrials = equityTrials,
      rng = rng
    )

  /** Validates all preconditions for bunching computation: positive trials, non-empty folds,
    * valid villain position, no card overlaps, unique fold positions, etc.
    */
  private def validateInputs(
      hero: HoleCards,
      board: Board,
      folds: Vector[PreflopFold],
      tableRanges: TableRanges,
      villainPos: Position,
      trials: Int
  ): Unit =
    require(trials > 0, "trials must be positive")
    require(folds.nonEmpty, "folds must be non-empty")
    val formatOrder = tableRanges.format.preflopOrder
    require(formatOrder.contains(villainPos), s"villain position $villainPos is not part of ${tableRanges.format}")

    val all = hero.toVector ++ board.cards
    require(all.distinct.length == all.length, "hero and board must not share cards")

    val foldPositions = folds.map(_.position)
    require(foldPositions.distinct.length == foldPositions.length, "fold positions must be unique")
    require(foldPositions.forall(formatOrder.contains), "all fold positions must belong to table format")
    require(!foldPositions.contains(villainPos), s"villain position $villainPos cannot be in folds")

  /**
    * Core fold-sampling loop that produces the bunching-adjusted villain range.
    *
    * Algorithm:
    *   1. Order fold positions by the table's preflop action order.
    *   2. For each trial, sample folded hands sequentially (each fold must not conflict
    *      with hero, board, or previously sampled folds). This uses weighted random
    *      selection from each fold position's candidate hands.
    *   3. For each successful fold configuration, iterate all villain candidate hands
    *      and accumulate those compatible with the dead-card mask.
    *   4. After all trials, normalize the accumulated weights into a probability distribution.
    *
    * The `-1L` sentinel from `sampleFoldedCardsMask` indicates a failed fold sample
    * (no compatible hand found for some fold position), which is silently skipped.
    *
    * @return normalized villain range conditioned on the sampled fold configurations
    */
  private def sampleAdjustedRange(
      hero: HoleCards,
      board: Board,
      folds: Vector[PreflopFold],
      tableRanges: TableRanges,
      villainPos: Position,
      trials: Int,
      rng: Random
  ): DiscreteDistribution[HoleCards] =
    val orderedFolds = orderFolds(folds, tableRanges.format)
    // Base dead-card mask: hero cards + board cards
    val deadBaseMask = handMask(hero) | cardsMask(board.cards)
    val villainWeightedHands = cachedVillainHands(tableRanges, villainPos)
    val orderedFoldCandidates = orderedFolds.map(fold => cachedFoldHands(tableRanges, fold.position))

    // Per-villain-hand accumulator: sums weights across all successful fold samples
    val accumulator = Array.fill[Double](villainWeightedHands.length)(0.0)
    val maxAttempts = math.max(trials * DefaultMaxAttemptFactor, trials)

    var successes = 0
    var attempts = 0

    while successes < trials && attempts < maxAttempts do
      attempts += 1
      // Sample folded cards for all fold positions; returns -1L on failure
      val deadAfterFoldsMask = sampleFoldedCardsMask(deadBaseMask, orderedFoldCandidates, rng)
      if deadAfterFoldsMask != -1L then
        // Accumulate villain hand weights for hands compatible with this dead-card config
        var i = 0
        while i < villainWeightedHands.length do
          val weighted = villainWeightedHands(i)
          // Bitmask AND: if zero, no card conflicts between villain hand and dead cards
          if (weighted.mask & deadAfterFoldsMask) == 0L then
            accumulator(i) = accumulator(i) + weighted.weight
          i += 1
        successes += 1

    require(
      successes > 0,
      "unable to sample any fold-consistent dead-card configuration; ranges may be too restrictive"
    )
    // Build the adjusted range from accumulated weights
    val accumulatedWeights = Map.newBuilder[HoleCards, Double]
    var i = 0
    while i < villainWeightedHands.length do
      if accumulator(i) > 0.0 then
        accumulatedWeights += villainWeightedHands(i).hand -> accumulator(i)
      i += 1
    DiscreteDistribution(accumulatedWeights.result()).normalized

  /**
    * Sequentially samples folded hole cards for each fold position, building up the
    * dead-card bitmask. Returns `-1L` if any fold position fails to find a compatible hand.
    *
    * The sequential ordering matters: earlier folds constrain later folds through the
    * growing dead-card mask, modeling the real information flow of preflop action.
    *
    * @param initialDeadMask      bitmask of hero + board cards
    * @param orderedFoldCandidates fold candidates per position, in preflop action order
    * @param rng                  random number generator
    * @return the combined dead-card mask after all folds, or -1L on sampling failure
    */
  private def sampleFoldedCardsMask(
      initialDeadMask: Long,
      orderedFoldCandidates: Vector[Vector[WeightedHand]],
      rng: Random
  ): Long =
    var deadMask = initialDeadMask
    var i = 0
    var failed = false
    while i < orderedFoldCandidates.length && !failed do
      val sampled = sampleWeightedCompatible(orderedFoldCandidates(i), deadMask, rng)
      if sampled != null then
        deadMask = deadMask | sampled.asInstanceOf[WeightedHand].mask
      else
        failed = true
      i += 1
    if failed then -1L else deadMask

  /**
    * Computes the naive (non-bunching-adjusted) villain range by filtering only for
    * card conflicts with hero and board. Does not condition on any fold information.
    * Used as the baseline for computing the bunching delta.
    */
  private def naiveVillainRange(
      hero: HoleCards,
      board: Board,
      tableRanges: TableRanges,
      villainPos: Position
  ): DiscreteDistribution[HoleCards] =
    val deadMask = handMask(hero) | cardsMask(board.cards)
    val filtered = tableRanges.rangeFor(villainPos).weights.collect {
      case (hand, weight) if weight > 0.0 && (handMask(hand) & deadMask) == 0L => hand -> weight
    }
    require(filtered.nonEmpty, "villain range is empty after hero/board filtering")
    DiscreteDistribution(filtered).normalized

  /** Sorts fold events by their position in the preflop action order, ensuring
    * the sequential fold-sampling respects the actual action sequence at the table.
    */
  private def orderFolds(folds: Vector[PreflopFold], format: TableFormat): Vector[PreflopFold] =
    val preflopOrder = format.preflopOrder
    folds.sortBy(f => preflopOrder.indexOf(f.position))

  /** Builds the vector of villain candidate hands with their opening-range weights and bitmasks.
    * Iterates all 1326 hands and includes only those with positive weight in the villain's range.
    */
  private def weightedVillainHands(
      villainRange: DiscreteDistribution[HoleCards]
  ): Vector[WeightedHand] =
    val builder = Vector.newBuilder[WeightedHand]
    var idx = 0
    while idx < allHoleCardsWithMasks.length do
      val (hand, mask) = allHoleCardsWithMasks(idx)
      val weight = villainRange.probabilityOf(hand)
      if weight > 0.0 then builder += WeightedHand(hand, weight, mask)
      idx += 1
    builder.result()

  /** Builds the vector of fold candidate hands for a position, weighted by fold probability.
    * P(fold|hand) = 1 - P(open|hand) from the TableRanges configuration.
    */
  private def weightedFoldHands(
      position: Position,
      tableRanges: TableRanges
  ): Vector[WeightedHand] =
    val builder = Vector.newBuilder[WeightedHand]
    var idx = 0
    while idx < allHoleCardsWithMasks.length do
      val (hand, mask) = allHoleCardsWithMasks(idx)
      val weight = tableRanges.foldProbability(position, hand)
      if weight > 0.0 then builder += WeightedHand(hand, weight, mask)
      idx += 1
    builder.result()

  /** Returns cached villain hands for the given position, computing and caching on first access.
    * Uses a simple eviction strategy: clears the entire cache when it exceeds MaxWeightedHandsCacheSize.
    */
  private def cachedVillainHands(
      tableRanges: TableRanges,
      villainPos: Position
  ): Vector[WeightedHand] =
    val key = WeightedHandsCacheKey(tableRanges, villainPos)
    val cached = villainHandsCache.get(key)
    if cached != null then cached
    else
      val computed = weightedVillainHands(tableRanges.rangeFor(villainPos))
      if villainHandsCache.size() >= MaxWeightedHandsCacheSize then villainHandsCache.clear()
      villainHandsCache.putIfAbsent(key, computed)
      val published = villainHandsCache.get(key)
      if published != null then published else computed

  /** Returns cached fold candidate hands for the given position, computing and caching on first access. */
  private def cachedFoldHands(
      tableRanges: TableRanges,
      position: Position
  ): Vector[WeightedHand] =
    val key = WeightedHandsCacheKey(tableRanges, position)
    val cached = foldHandsCache.get(key)
    if cached != null then cached
    else
      val computed = weightedFoldHands(position, tableRanges)
      if foldHandsCache.size() >= MaxWeightedHandsCacheSize then foldHandsCache.clear()
      foldHandsCache.putIfAbsent(key, computed)
      val published = foldHandsCache.get(key)
      if published != null then published else computed

  /**
    * Samples one hand from the candidate list, weighted by probability, excluding hands
    * that conflict with the dead-card mask. Returns `null` if no compatible hand exists.
    *
    * Algorithm: two-pass approach.
    *   Pass 1: Sum weights of all compatible (non-conflicting) candidates.
    *   Pass 2: Draw a uniform random number in [0, totalWeight), then walk
    *           through compatible candidates accumulating weight until the target is reached.
    *
    * @param items    candidate hands with weights and bitmasks
    * @param deadMask bitmask of cards already in use (hero, board, earlier folds)
    * @param rng      random number generator
    * @return a sampled hand, or null if all candidates conflict with deadMask
    */
  private def sampleWeightedCompatible(
      items: Vector[WeightedHand],
      deadMask: Long,
      rng: Random
  ): WeightedHand | Null =
    var total = 0.0
    var i = 0
    while i < items.length do
      if (items(i).mask & deadMask) == 0L then total += items(i).weight
      i += 1
    if total <= 0.0 then null
    else
      val target = rng.nextDouble() * total
      var acc = 0.0
      var result: WeightedHand | Null = null
      i = 0
      while i < items.length do
        val item = items(i)
        if (item.mask & deadMask) == 0L then
          result = item
          acc += item.weight
          if target <= acc then i = items.length // break
        i += 1
      result

  /** Computes a 64-bit bitmask for a hole-card hand (two bits set, one per card). */
  private inline def handMask(hand: HoleCards): Long =
    cardMask(hand.first) | cardMask(hand.second)

  /** Computes a 64-bit bitmask for a sequence of cards (one bit per card). */
  private def cardsMask(cards: Seq[Card]): Long =
    var mask = 0L
    var i = 0
    while i < cards.length do
      mask = mask | cardMask(cards(i))
      i += 1
    mask

  /** Maps a card to a single-bit 64-bit mask using the card's integer ID as the bit position.
    * Cards are numbered 0-51, so this always fits within a Long (64 bits).
    */
  private inline def cardMask(card: Card): Long =
    1L << CardId.toId(card)
