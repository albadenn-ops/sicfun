package sicfun.holdem.equity
import sicfun.holdem.types.*

import sicfun.core.{Card, CardId, Deck, DiscreteDistribution}

import java.util.concurrent.ConcurrentHashMap
import scala.util.Random

/** A preflop fold event for a specific table position. */
final case class PreflopFold(position: Position)

/** Result of bunching-aware equity computation vs a positional villain range. */
final case class BunchingResult(
    adjustedRange: DiscreteDistribution[HoleCards],
    naiveRange: DiscreteDistribution[HoleCards],
    adjustedEquity: EquityEstimate,
    naiveEquity: EquityEstimate,
    bunchingDelta: Double,
    trials: Int
)

/** Monte Carlo bunching engine.
  *
  * Samples folded hole cards conditioned on position-specific fold probabilities,
  * then integrates over resulting dead-card configurations to estimate the
  * villain's adjusted range and equity impact.
  */
object BunchingEffect:
  private val DefaultMaxAttemptFactor = 3
  private val MaxWeightedHandsCacheSize = 128
  private final case class WeightedHand(hand: HoleCards, weight: Double, mask: Long)
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

  private lazy val allHoleCardsWithMasks: Vector[(HoleCards, Long)] =
    HoldemCombinator.holeCardsFrom(Deck.full).map { hand =>
      hand -> handMask(hand)
    }
  private val foldHandsCache = new ConcurrentHashMap[WeightedHandsCacheKey, Vector[WeightedHand]]()
  private val villainHandsCache = new ConcurrentHashMap[WeightedHandsCacheKey, Vector[WeightedHand]]()

  /** Full bunching analysis: adjusted range, naive range, both equities, and delta. */
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

  /** Computes only the bunching-adjusted villain range. */
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

  /** Quick bunching delta as a single scalar. */
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

  /** Convenience API for "hero opens from openerPos after everyone before folded". */
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
    val deadBaseMask = handMask(hero) | cardsMask(board.cards)
    val villainWeightedHands = cachedVillainHands(tableRanges, villainPos)
    val orderedFoldCandidates = orderedFolds.map(fold => cachedFoldHands(tableRanges, fold.position))

    val accumulator = Array.fill[Double](villainWeightedHands.length)(0.0)
    val maxAttempts = math.max(trials * DefaultMaxAttemptFactor, trials)

    var successes = 0
    var attempts = 0

    while successes < trials && attempts < maxAttempts do
      attempts += 1
      val deadAfterFoldsMask = sampleFoldedCardsMask(deadBaseMask, orderedFoldCandidates, rng)
      if deadAfterFoldsMask != -1L then
        var i = 0
        while i < villainWeightedHands.length do
          val weighted = villainWeightedHands(i)
          if (weighted.mask & deadAfterFoldsMask) == 0L then
            accumulator(i) = accumulator(i) + weighted.weight
          i += 1
        successes += 1

    require(
      successes > 0,
      "unable to sample any fold-consistent dead-card configuration; ranges may be too restrictive"
    )
    val accumulatedWeights = Map.newBuilder[HoleCards, Double]
    var i = 0
    while i < villainWeightedHands.length do
      if accumulator(i) > 0.0 then
        accumulatedWeights += villainWeightedHands(i).hand -> accumulator(i)
      i += 1
    DiscreteDistribution(accumulatedWeights.result()).normalized

  /** Returns the dead-card mask after sampling folds, or -1L if no valid configuration found. */
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

  private def orderFolds(folds: Vector[PreflopFold], format: TableFormat): Vector[PreflopFold] =
    val preflopOrder = format.preflopOrder
    folds.sortBy(f => preflopOrder.indexOf(f.position))

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

  /** Returns a compatible weighted hand sampled by weight, or null if none compatible. */
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

  private inline def handMask(hand: HoleCards): Long =
    cardMask(hand.first) | cardMask(hand.second)

  private def cardsMask(cards: Seq[Card]): Long =
    var mask = 0L
    var i = 0
    while i < cards.length do
      mask = mask | cardMask(cards(i))
      i += 1
    mask

  private inline def cardMask(card: Card): Long =
    1L << CardId.toId(card)
