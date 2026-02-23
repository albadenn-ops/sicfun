package sicfun.holdem

import sicfun.core.{Card, Deck, DiscreteDistribution}

import scala.collection.mutable
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
    val villainBase = tableRanges.rangeFor(villainPos)
    val orderedFolds = orderFolds(folds, tableRanges.format)
    val deadBase = hero.asSet ++ board.asSet

    val accumulator = mutable.Map.empty[HoleCards, Double].withDefaultValue(0.0)
    val maxAttempts = math.max(trials * DefaultMaxAttemptFactor, trials)

    var successes = 0
    var attempts = 0

    while successes < trials && attempts < maxAttempts do
      attempts += 1
      sampleFoldedCards(deadBase, orderedFolds, tableRanges, rng) match
        case Some(deadAfterFolds) =>
          val remaining = Deck.full.filterNot(deadAfterFolds.contains).toIndexedSeq
          val villainCandidates = HoldemCombinator.holeCardsFrom(remaining)
          villainCandidates.foreach { hand =>
            val p = villainBase.probabilityOf(hand)
            if p > 0.0 then
              accumulator.update(hand, accumulator(hand) + p)
          }
          successes += 1
        case None =>
          ()

    require(
      successes > 0,
      "unable to sample any fold-consistent dead-card configuration; ranges may be too restrictive"
    )
    DiscreteDistribution(accumulator.toMap).normalized

  private def sampleFoldedCards(
      initialDead: Set[Card],
      orderedFolds: Vector[PreflopFold],
      tableRanges: TableRanges,
      rng: Random
  ): Option[Set[Card]] =
    var dead = initialDead
    var i = 0
    while i < orderedFolds.length do
      val fold = orderedFolds(i)
      val remaining = Deck.full.filterNot(dead.contains).toIndexedSeq
      val candidates = HoldemCombinator.holeCardsFrom(remaining)
      val weighted = candidates.flatMap { hand =>
        val weight = tableRanges.foldProbability(fold.position, hand)
        if weight > 0.0 then Some(hand -> weight) else None
      }
      if weighted.isEmpty then return None
      val sampled = sampleWeighted(weighted, rng)
      dead = dead ++ sampled.asSet
      i += 1
    Some(dead)

  private def naiveVillainRange(
      hero: HoleCards,
      board: Board,
      tableRanges: TableRanges,
      villainPos: Position
  ): DiscreteDistribution[HoleCards] =
    val dead = hero.asSet ++ board.asSet
    val filtered = tableRanges.rangeFor(villainPos).weights.collect {
      case (hand, weight) if weight > 0.0 && !hand.asSet.exists(dead.contains) => hand -> weight
    }
    require(filtered.nonEmpty, "villain range is empty after hero/board filtering")
    DiscreteDistribution(filtered).normalized

  private def orderFolds(folds: Vector[PreflopFold], format: TableFormat): Vector[PreflopFold] =
    val order = format.preflopOrder.zipWithIndex.toMap
    folds.sortBy(f => order(f.position))

  private def sampleWeighted[A](items: IndexedSeq[(A, Double)], rng: Random): A =
    val total = items.foldLeft(0.0)(_ + _._2)
    require(total > 0.0, "cannot sample from zero-weight distribution")
    val r = rng.nextDouble() * total
    var acc = 0.0
    var i = 0
    while i < items.length do
      acc += items(i)._2
      if r <= acc then return items(i)._1
      i += 1
    items.last._1
