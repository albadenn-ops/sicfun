package sicfun.holdem.equity
import sicfun.holdem.types.*
import sicfun.holdem.gpu.*

import sicfun.core.{Card, Deck, DiscreteDistribution, HandEvaluator, Prob}
import Prob.*

import scala.util.Random
import scala.annotation.targetName
import scala.collection.mutable

/**
  * Core equity calculation engine for Texas Hold'em poker.
  *
  * Provides the primary API for computing hand equity (probability of winning) across
  * multiple computation strategies:
  *
  * '''Exact enumeration''' (`equityExact`, `equityExactProb`, `equityExactMulti`):
  *   Exhaustively evaluates all possible board run-outs. Practical for river (0 cards missing),
  *   turn (1 card, ~45 run-outs), and flop (2 cards, ~990 run-outs). Uses the 7-card hand
  *   evaluator to compare hero vs villain for each board completion. The `Prob` variant uses
  *   fixed-point integer arithmetic (Int32 at 2^30 scale) for deterministic, cache-friendly
  *   evaluation in the Bayes-to-equity hot path.
  *
  * '''Monte Carlo simulation''' (`equityMonteCarlo`, `equityMonteCarloMulti`):
  *   Randomly samples villain hands from a weighted range and board run-outs, using
  *   Welford's online algorithm for numerically stable variance/stderr estimation.
  *   Supports optional GPU/native acceleration for preflop and postflop scenarios.
  *
  * '''Multi-villain''' (`equityExactMulti`, `equityMonteCarloMulti`):
  *   Handles multi-way pots by iterating over all non-overlapping villain hand combinations,
  *   computing hero's share (win = 1.0, tie = 1/(1+tied), loss = 0.0).
  *
  * '''GPU acceleration''':
  *   Preflop equity can be accelerated via native CUDA/CPU range runtime (CSR format) or
  *   batch GPU runtime. Postflop uses the native postflop Monte Carlo bridge. An acceleration
  *   guard (ThreadLocal boolean) prevents recursive re-entry when GPU providers call back
  *   into HoldemEquity.
  *
  * @see [[HeadsUpEquityTable]] for precomputed pairwise equity lookups
  * @see [[RangeParser]] for parsing string-based range notation
  * @see [[BunchingEffect]] for fold-conditioned range adjustment
  */
object HoldemEquity:
  /** Maximum evaluation count for exact multi-villain enumeration before requiring explicit override. */
  private val DefaultExactMultiMaxEvaluations: Long = 5_000_000L

  // -- System property / environment variable keys for configuring acceleration backends --
  private val PreflopEquityBackendProperty = "sicfun.holdem.preflopEquityBackend"
  private val PreflopEquityBackendEnv = "sicfun_HOLDEM_PREFLOP_EQUITY_BACKEND"
  private val GpuProviderProperty = "sicfun.gpu.provider"
  private val GpuProviderEnv = "sicfun_GPU_PROVIDER"

  // -- Backend identifiers --
  private val PreflopEquityBackendAuto = "auto"
  private val PreflopEquityBackendCpu = "cpu"
  private val PreflopEquityBackendRange = "range"
  private val PreflopEquityBackendBatch = "batch"

  /** Thread-local guard to prevent recursive GPU acceleration when GPU providers
    * call back into HoldemEquity (e.g., native runtime computing equity internally).
    */
  private val accelerationGuard = new ThreadLocal[java.lang.Boolean]:
    override def initialValue(): java.lang.Boolean = java.lang.Boolean.FALSE
  /**
    * Pre-processed villain range in flat-array form for cache-friendly Monte Carlo iteration.
    * Hands are sorted by HoleCardsIndex ID for deterministic ordering.
    *
    * @param hands   concrete hole-card hands (dead-card filtered, canonicalized)
    * @param weights normalized probability weights summing to 1.0
    * @param handIds corresponding HoleCardsIndex integer IDs (for GPU key construction)
    */
  private final case class PreparedRange(
      hands: Array[HoleCards],
      weights: Array[Double],
      handIds: Array[Int]
  )

  /**
    * Fixed-point variant of PreparedRange using Prob (Int32 @ 2^30 scale) weights.
    * Used by equityExactProb for deterministic integer arithmetic in the Bayes hot path.
    *
    * @param hands   concrete hole-card hands
    * @param weights Prob raw values (Int32 with 2^30 denominator), normalized
    * @param size    number of valid entries (may be less than array length)
    */
  private final case class PreparedRangeProb(
      hands: Array[HoleCards],
      weights: Array[Int], // Prob raw values (Int32 @ 2^30)
      size: Int
  )
  /** Flat-array posterior for the Bayes → Equity hot path.
    * Built from arrays already computed in HoldemBayesProvider.scalaUpdate/nativeUpdate.
    * Equity methods consume this directly without Map allocation.
    * Non-hot consumers access the lazy `distribution` field.
    */
  final class CompactPosterior(
      val hands: Array[HoleCards],
      val probWeights: Array[Int], // Prob raw values (Int32 @ 2^30), normalized
      val size: Int
  ):
    lazy val distribution: DiscreteDistribution[HoleCards] =
      val builder = Map.newBuilder[HoleCards, Double]
      builder.sizeHint(size)
      var i = 0
      while i < size do
        builder += hands(i) -> Prob(probWeights(i)).toDouble
        i += 1
      DiscreteDistribution(builder.result())

  /** Builds a CompactPosterior from flat arrays (as produced by Bayesian update).
    * Skips zero-weight hypotheses. Normalizes weights to sum to Prob.Scale.
    */
  def buildCompactPosterior(
      hypotheses: Vector[HoleCards],
      posterior: Array[Double]
  ): CompactPosterior =
    require(hypotheses.length == posterior.length,
      s"hypotheses.length (${hypotheses.length}) != posterior.length (${posterior.length})")
    var total = 0.0
    var positiveCount = 0
    var i = 0
    while i < hypotheses.length do
      val w = math.max(0.0, posterior(i))
      if w > 0.0 then
        total += w
        positiveCount += 1
      i += 1
    require(total > 0.0, "all-zero posterior")
    val invTotal = 1.0 / total
    val hands = new Array[HoleCards](positiveCount)
    val weights = new Array[Int](positiveCount)
    var j = 0
    i = 0
    while i < hypotheses.length do
      val w = math.max(0.0, posterior(i))
      if w > 0.0 then
        hands(j) = hypotheses(i)
        weights(j) = Prob.fromDouble(w * invTotal).raw
        j += 1
      i += 1
    new CompactPosterior(hands, weights, positiveCount)

  /** Thread-local scratch array for weight accumulation during range preparation.
    * Sized to HoleCardsIndex.size (1326) and zeroed after each use to avoid allocation per call.
    */
  private val preparedRangeWeightScratch = new ThreadLocal[Array[Double]]:
    override def initialValue(): Array[Double] = new Array[Double](HoleCardsIndex.size)
  /** Thread-local scratch array tracking which hand IDs were touched during range preparation.
    * Used to efficiently zero out only the modified entries in the weight scratch array.
    */
  private val preparedRangeTouchedIdsScratch = new ThreadLocal[Array[Int]]:
    override def initialValue(): Array[Int] = new Array[Int](HoleCardsIndex.size)

  /**
    * Fixed-point exact equity using CompactPosterior — bypasses Map allocation entirely.
    *
    * Iterates the compact posterior's flat arrays directly, using integer (Long) accumulators
    * for win/tie/loss to avoid floating-point drift. Weight per board is computed as
    * `probWeight / boardCount` (integer division), introducing at most 1 LSB truncation
    * per evaluation (~4e-8 relative error).
    *
    * @param hero    the hero's hole cards
    * @param board   community cards (0-5 cards)
    * @param compact flat-array posterior from Bayesian update
    * @return exact equity result (win/tie/loss fractions summing to 1.0)
    */
  def equityExactProb(
      hero: HoleCards,
      board: Board,
      compact: CompactPosterior
  ): EquityResult =
    validateHeroBoard(hero, board)
    val dead = hero.asSet ++ board.asSet
    val prepared = prepareRangeProbFromCompact(compact, dead)
    val missing = board.missing

    var winL = 0L
    var tieL = 0L
    var lossL = 0L

    var i = 0
    while i < prepared.size do
      val villain = prepared.hands(i)
      val weightRaw = prepared.weights(i)
      if weightRaw > 0 then
        val remaining = Deck.full.filterNot(card => dead.contains(card) || villain.contains(card)).toIndexedSeq
        val boardCount = combinationsCount(remaining.length, missing)
        val perBoardWeight = weightRaw.toLong / boardCount

        val boards =
          if missing == 0 then Iterator.single(Vector.empty[Card])
          else HoldemCombinator.combinations(remaining, missing)

        boards.foreach { extra =>
          val fullBoard = board.cards ++ extra
          val heroRank = HandEvaluator.evaluate7Cached(hero.toVector ++ fullBoard)
          val villainRank = HandEvaluator.evaluate7Cached(villain.toVector ++ fullBoard)
          val cmp = heroRank.compare(villainRank)
          if cmp > 0 then winL += perBoardWeight
          else if cmp == 0 then tieL += perBoardWeight
          else lossL += perBoardWeight
        }
      i += 1

    val total = winL + tieL + lossL
    if total == 0L then EquityResult(0.0, 0.0, 0.0)
    else
      val invTotal = 1.0 / total.toDouble
      EquityResult(winL * invTotal, tieL * invTotal, lossL * invTotal)

  /**
    * Exact equity computation against a villain range using floating-point accumulation.
    *
    * For each villain hand in the range (weighted by probability), enumerates all possible
    * board completions, evaluates both 7-card hands, and accumulates weighted win/tie/loss.
    * Each villain hand's contribution is weighted by `handWeight / boardCount` so that
    * different villain hands are correctly weighted even when they produce different
    * numbers of remaining board cards (due to card blocking).
    *
    * @param hero         the hero's hole cards
    * @param board        community cards (0-5 cards)
    * @param villainRange probability distribution over possible villain hole cards
    * @return exact equity result
    */
  def equityExact(
      hero: HoleCards,
      board: Board,
      villainRange: DiscreteDistribution[HoleCards]
  ): EquityResult =
    validateHeroBoard(hero, board)
    val dead = hero.asSet ++ board.asSet
    val range = sanitizeRange(villainRange, dead)
    val missing = board.missing

    var win = 0.0
    var tie = 0.0
    var loss = 0.0

    range.weights.foreach { case (villain, weight) =>
      if weight > 0.0 then
        val remaining = Deck.full.filterNot(card => dead.contains(card) || villain.contains(card)).toIndexedSeq
        val boardCount = combinationsCount(remaining.length, missing)
        val perBoardWeight = weight / boardCount.toDouble

        val boards =
          if missing == 0 then Iterator.single(Vector.empty[Card])
          else HoldemCombinator.combinations(remaining, missing)

        boards.foreach { extra =>
          val fullBoard = board.cards ++ extra
          val heroRank = HandEvaluator.evaluate7Cached(hero.toVector ++ fullBoard)
          val villainRank = HandEvaluator.evaluate7Cached(villain.toVector ++ fullBoard)
          val cmp = heroRank.compare(villainRank)
          if cmp > 0 then win += perBoardWeight
          else if cmp == 0 then tie += perBoardWeight
          else loss += perBoardWeight
        }
    }

    val total = win + tie + loss
    EquityResult(win / total, tie / total, loss / total)

  /** Fixed-point variant of equityExact. Uses PreparedRangeProb (flat arrays with
    * Int32 Prob weights) for cache-friendly iteration and deterministic arithmetic.
    *
    * Phase 2: replaces Map.foreach with while-loop over sorted arrays.
    * Integer division of weight / boardCount introduces truncation error of at most
    * 1 LSB per board evaluation (~4e-8 relative error), negligible for poker equity needs.
    */
  def equityExactProb(
      hero: HoleCards,
      board: Board,
      villainRange: DiscreteDistribution[HoleCards]
  ): EquityResult =
    validateHeroBoard(hero, board)
    val dead = hero.asSet ++ board.asSet
    val prepared = prepareRangeProb(villainRange, dead)
    val missing = board.missing

    // Accumulate as Long. Max: 1326 × 1081 × (2^30/1) ≈ 1.5e15, Long max 9.2e18.
    var winL = 0L
    var tieL = 0L
    var lossL = 0L

    var i = 0
    while i < prepared.size do
      val villain = prepared.hands(i)
      val weightRaw = prepared.weights(i)
      if weightRaw > 0 then
        val remaining = Deck.full.filterNot(card => dead.contains(card) || villain.contains(card)).toIndexedSeq
        val boardCount = combinationsCount(remaining.length, missing)
        val perBoardWeight = weightRaw.toLong / boardCount

        val boards =
          if missing == 0 then Iterator.single(Vector.empty[Card])
          else HoldemCombinator.combinations(remaining, missing)

        boards.foreach { extra =>
          val fullBoard = board.cards ++ extra
          val heroRank = HandEvaluator.evaluate7Cached(hero.toVector ++ fullBoard)
          val villainRank = HandEvaluator.evaluate7Cached(villain.toVector ++ fullBoard)
          val cmp = heroRank.compare(villainRank)
          if cmp > 0 then winL += perBoardWeight
          else if cmp == 0 then tieL += perBoardWeight
          else lossL += perBoardWeight
        }
      i += 1

    val total = winL + tieL + lossL
    if total == 0L then EquityResult(0.0, 0.0, 0.0)
    else
      val invTotal = 1.0 / total.toDouble
      EquityResult(winL * invTotal, tieL * invTotal, lossL * invTotal)

  def equityMonteCarlo(
      hero: HoleCards,
      board: Board,
      compact: CompactPosterior,
      trials: Int,
      rng: Random
  ): EquityEstimate =
    validateHeroBoard(hero, board)
    require(trials > 0, "trials must be positive")
    val deadMaskValue = deadMask(hero, board)
    val preparedRange = prepareRangeFromCompact(compact, deadMaskValue)
    equityMonteCarloPrepared(hero, board, preparedRange, deadMaskValue, trials, rng)

  def equityMonteCarlo(
      hero: HoleCards,
      board: Board,
      compact: CompactPosterior,
      trials: Int
  ): EquityEstimate =
    equityMonteCarlo(hero, board, compact, trials, new Random())

  def equityExact(
      hero: HoleCards,
      board: Board,
      range: String
  ): EquityResult =
    val dist = parseRangeOrThrow(range)
    equityExact(hero, board, dist)

  def equityExact(hero: HoleCards, board: Board): EquityResult =
    equityExact(hero, board, fullRange(hero, board))

  def equityExactMulti(
      hero: HoleCards,
      board: Board,
      villainRanges: Seq[DiscreteDistribution[HoleCards]]
  ): EquityShareResult =
    equityExactMulti(hero, board, villainRanges, DefaultExactMultiMaxEvaluations)

  /**
    * Exact equity in a multi-villain pot.
    *
    * Recursively iterates over all non-overlapping villain hand assignments, then for each
    * complete assignment enumerates board completions and evaluates the multi-way showdown.
    * Hero's equity share accounts for ties: when hero ties with K villains, hero gets
    * 1/(K+1) of the pot for that board.
    *
    * Only supports flop, turn, and river boards (3-5 cards) because preflop enumeration
    * with multiple villains would be combinatorially infeasible.
    *
    * @param hero           the hero's hole cards
    * @param board          community cards (3-5 cards)
    * @param villainRanges  one distribution per villain seat
    * @param maxEvaluations safety cap to prevent runaway computation
    * @return equity share result with win/tie/loss/share fractions
    */
  def equityExactMulti(
      hero: HoleCards,
      board: Board,
      villainRanges: Seq[DiscreteDistribution[HoleCards]],
      maxEvaluations: Long
  ): EquityShareResult =
    validateHeroBoard(hero, board)
    require(villainRanges.nonEmpty, "villainRanges must be non-empty")
    require(board.missing <= 2, "equityExactMulti supports only flop, turn, or river (3-5 board cards)")
    require(maxEvaluations > 0L, "maxEvaluations must be positive")

    val dead = hero.asSet ++ board.asSet
    val sanitized = villainRanges.map(range => sanitizeRange(range, dead))
    ensureExactFeasible(board, sanitized, maxEvaluations)
    val ranges = sanitized.map(range => sortedWeights(range.weights)).toVector

    var win = 0.0
    var tie = 0.0
    var loss = 0.0
    var share = 0.0

    def evaluateBoard(fullBoard: Vector[Card], weight: Double, villains: Vector[HoleCards]): Unit =
      val heroRank = HandEvaluator.evaluate7Cached(hero.toVector ++ fullBoard)
      val villainRanks = villains.map(hand => HandEvaluator.evaluate7Cached(hand.toVector ++ fullBoard))
      val bestRank = (heroRank +: villainRanks).max
      if heroRank == bestRank then
        val tiedVillains = villainRanks.count(_ == bestRank)
        if tiedVillains == 0 then win += weight else tie += weight
        share += weight * (1.0 / (tiedVillains + 1).toDouble)
      else loss += weight

    def loop(index: Int, used: Set[Card], weight: Double, villains: Vector[HoleCards]): Unit =
      if index == ranges.length then
        board.missing match
          case 0 =>
            evaluateBoard(board.cards, weight, villains)
          case 1 =>
            val remaining = Deck.full.filterNot(used.contains)
            val count = remaining.length
            if count > 0 then
              val perCardWeight = weight / count.toDouble
              remaining.foreach { card =>
                evaluateBoard(board.cards :+ card, perCardWeight, villains)
              }
          case 2 =>
            val remaining = Deck.full.filterNot(used.contains).toIndexedSeq
            val count = combinationsCount(remaining.length, 2)
            if count > 0 then
              val perBoardWeight = weight / count.toDouble
              HoldemCombinator.combinations(remaining, 2).foreach { extra =>
                evaluateBoard(board.cards ++ extra, perBoardWeight, villains)
              }
          case _ =>
            throw new IllegalArgumentException("equityExactMulti supports only flop, turn, or river")
      else
        ranges(index).foreach { case (hand, w) =>
          if w > 0.0 &&
            !used.contains(hand.first) &&
            !used.contains(hand.second)
          then
            loop(
              index + 1,
              used + hand.first + hand.second,
              weight * w,
              villains :+ hand
            )
        }

    loop(0, dead, 1.0, Vector.empty)
    val total = win + tie + loss
    require(total > 0.0, "equityExactMulti produced zero total weight")
    EquityShareResult(win / total, tie / total, loss / total, share / total)

  @targetName("equityExactMultiRanges")
  def equityExactMulti(
      hero: HoleCards,
      board: Board,
      ranges: Seq[String]
  ): EquityShareResult =
    val distributions = ranges.map(parseRangeOrThrow)
    equityExactMulti(hero, board, distributions, DefaultExactMultiMaxEvaluations)

  @targetName("equityExactMultiRangesWithLimit")
  def equityExactMulti(
      hero: HoleCards,
      board: Board,
      ranges: Seq[String],
      maxEvaluations: Long
  ): EquityShareResult =
    val distributions = ranges.map(parseRangeOrThrow)
    equityExactMulti(hero, board, distributions, maxEvaluations)

  /**
    * Monte Carlo equity estimation against a single villain range.
    *
    * Each trial: sample a villain hand from the weighted range, sample missing board cards,
    * evaluate both 7-card hands, and record the outcome. Uses Welford's online algorithm
    * for numerically stable running mean and variance computation.
    *
    * For preflop boards, may delegate to GPU/native acceleration if configured.
    *
    * @param hero         the hero's hole cards
    * @param board        community cards (0-5 cards)
    * @param villainRange probability distribution over villain hands
    * @param trials       number of Monte Carlo samples
    * @param rng          random number generator (seeded for reproducibility)
    * @return equity estimate with mean, variance, stderr, and win/tie/loss rates
    */
  def equityMonteCarlo(
      hero: HoleCards,
      board: Board,
      villainRange: DiscreteDistribution[HoleCards],
      trials: Int,
      rng: Random = new Random()
  ): EquityEstimate =
    validateHeroBoard(hero, board)
    require(trials > 0, "trials must be positive")
    val deadMaskValue = deadMask(hero, board)
    val preparedRange = prepareRange(villainRange, deadMaskValue)
    equityMonteCarloPrepared(hero, board, preparedRange, deadMaskValue, trials, rng)

  /**
    * Inner Monte Carlo loop with prepared (flat-array) range data.
    *
    * First attempts GPU/native acceleration; falls back to the JVM-based loop if unavailable.
    * The JVM loop is optimized for the common case:
    *   - River (0 missing cards): directly evaluates with no board sampling or allocation.
    *   - Pre-river: uses a reusable board array and rejection-sampled card drawing to
    *     avoid per-trial allocations.
    *
    * Welford's online algorithm tracks mean and M2 (sum of squared deviations) for
    * numerically stable variance estimation: variance = M2 / (n-1), stderr = sqrt(var/n).
    */
  private def equityMonteCarloPrepared(
      hero: HoleCards,
      board: Board,
      preparedRange: PreparedRange,
      deadMaskValue: Long,
      trials: Int,
      rng: Random
  ): EquityEstimate =
    maybeAcceleratedMonteCarlo(hero, board, preparedRange, trials, rng) match
      case Some(estimate) =>
        estimate
      case None =>
        val sampler = WeightedSampler.fromArrays(preparedRange.hands, preparedRange.weights)

        var winCount = 0
        var tieCount = 0
        var lossCount = 0

        var mean = 0.0
        var m2 = 0.0
        var i = 0

        // Pre-filter deck once (remove hero + board), then per-iteration only
        // remove the 2 villain cards instead of filtering all 52 each time.
        val deckMinusDead = Deck.full.filterNot(card => (deadMaskValue & cardMask(card)) != 0L).toIndexedSeq
        val boardCards = board.cards
        val boardMissing = board.missing
        // Reusable board array for non-river cases (pre-fill known cards once).
        val boardArr =
          if boardMissing > 0 then
            val arr = new Array[Card](5)
            var k = 0
            while k < boardCards.length do
              arr(k) = boardCards(k)
              k += 1
            arr
          else null
        val sampledDeckIndexes =
          if boardMissing > 0 then new Array[Int](boardMissing)
          else null

        while i < trials do
          val villain = sampler.sample(rng)
          var heroRank = 0
          var villainRank = 0

          if boardMissing == 0 then
            // River: evaluate directly with no sampling or allocation.
            heroRank = HandEvaluator.evaluate7PackedDirect(
              hero.first, hero.second, boardCards(0), boardCards(1), boardCards(2), boardCards(3), boardCards(4)
            )
            villainRank = HandEvaluator.evaluate7PackedDirect(
              villain.first, villain.second, boardCards(0), boardCards(1), boardCards(2), boardCards(3), boardCards(4)
            )
          else
            fillBoardWithRandomCards(
              boardArr = boardArr,
              knownBoardSize = boardCards.length,
              deckMinusDead = deckMinusDead,
              blockedFirst = villain.first,
              blockedSecond = villain.second,
              missing = boardMissing,
              sampledDeckIndexes = sampledDeckIndexes,
              rng = rng
            )
            heroRank = HandEvaluator.evaluate7PackedDirect(
              hero.first, hero.second, boardArr(0), boardArr(1), boardArr(2), boardArr(3), boardArr(4)
            )
            villainRank = HandEvaluator.evaluate7PackedDirect(
              villain.first, villain.second, boardArr(0), boardArr(1), boardArr(2), boardArr(3), boardArr(4)
            )

          val outcome =
            if heroRank > villainRank then
              winCount += 1
              1.0
            else if heroRank == villainRank then
              tieCount += 1
              0.5
            else
              lossCount += 1
              0.0

          val delta = outcome - mean
          mean += delta / (i + 1)
          val delta2 = outcome - mean
          m2 += delta * delta2
          i += 1

        val variance = if trials > 1 then m2 / (trials - 1) else 0.0
        val stderr = math.sqrt(variance / trials)
        EquityEstimate(
          mean = mean,
          variance = variance,
          stderr = stderr,
          trials = trials,
          winRate = winCount.toDouble / trials,
          tieRate = tieCount.toDouble / trials,
          lossRate = lossCount.toDouble / trials
        )

  def equityMonteCarlo(
      hero: HoleCards,
      board: Board,
      range: String,
      trials: Int,
      rng: Random
  ): EquityEstimate =
    val dist = parseRangeOrThrow(range)
    equityMonteCarlo(hero, board, dist, trials, rng)

  def equityMonteCarlo(
      hero: HoleCards,
      board: Board,
      range: String,
      trials: Int
  ): EquityEstimate =
    equityMonteCarlo(hero, board, range, trials, new Random())

  def equityMonteCarlo(hero: HoleCards, board: Board, trials: Int): EquityEstimate =
    equityMonteCarlo(hero, board, fullRange(hero, board), trials)

  /**
    * Monte Carlo equity estimation in a multi-villain pot.
    *
    * Each trial: sample one hand per villain (rejecting overlapping samples), sample
    * missing board cards, evaluate all hands, and compute hero's share. Uses rejection
    * sampling to ensure no two villains hold the same card.
    *
    * @param hero          hero's hole cards
    * @param board         community cards (0-5 cards)
    * @param villainRanges one distribution per villain seat
    * @param trials        number of Monte Carlo samples
    * @param rng           random number generator
    * @return equity estimate with mean share, variance, stderr, and rate breakdown
    */
  def equityMonteCarloMulti(
      hero: HoleCards,
      board: Board,
      villainRanges: Seq[DiscreteDistribution[HoleCards]],
      trials: Int,
      rng: Random
  ): EquityEstimate =
    validateHeroBoard(hero, board)
    require(villainRanges.nonEmpty, "villainRanges must be non-empty")
    require(trials > 0, "trials must be positive")
    val dead = hero.asSet ++ board.asSet
    val deadMaskValue = deadMask(hero, board)
    val preparedRanges = villainRanges.map(range => prepareRange(range, deadMaskValue))
    val samplers = preparedRanges.map(range => WeightedSampler.fromArrays(range.hands, range.weights)).toVector

    var winCount = 0
    var tieCount = 0
    var lossCount = 0

    var mean = 0.0
    var m2 = 0.0
    var i = 0

    while i < trials do
      val villainHands = sampleVillainsNoOverlap(samplers, dead, rng)
      val used = dead ++ villainHands.flatMap(_.asSet)
      val remaining = Deck.full.filterNot(used.contains).toIndexedSeq
      val extra = if board.missing == 0 then Vector.empty[Card] else sampleWithoutReplacement(remaining, board.missing, rng)
      val fullBoard = board.cards ++ extra

      val heroRank = HandEvaluator.evaluate7Cached(hero.toVector ++ fullBoard)
      val villainRanks = villainHands.map(hand => HandEvaluator.evaluate7Cached(hand.toVector ++ fullBoard))
      val bestRank = (heroRank +: villainRanks).max

      val outcome =
        if heroRank == bestRank then
          val tiedVillains = villainRanks.count(_ == bestRank)
          if tiedVillains == 0 then winCount += 1 else tieCount += 1
          1.0 / (tiedVillains + 1).toDouble
        else
          lossCount += 1
          0.0

      val delta = outcome - mean
      mean += delta / (i + 1)
      val delta2 = outcome - mean
      m2 += delta * delta2
      i += 1

    val variance = if trials > 1 then m2 / (trials - 1) else 0.0
    val stderr = math.sqrt(variance / trials)
    EquityEstimate(
      mean = mean,
      variance = variance,
      stderr = stderr,
      trials = trials,
      winRate = winCount.toDouble / trials,
      tieRate = tieCount.toDouble / trials,
      lossRate = lossCount.toDouble / trials
    )

  def equityMonteCarloMulti(
      hero: HoleCards,
      board: Board,
      villainRanges: Seq[DiscreteDistribution[HoleCards]],
      trials: Int
  ): EquityEstimate =
    equityMonteCarloMulti(hero, board, villainRanges, trials, new Random())

  @targetName("equityMonteCarloMultiRanges")
  def equityMonteCarloMulti(
      hero: HoleCards,
      board: Board,
      ranges: Seq[String],
      trials: Int,
      rng: Random
  ): EquityEstimate =
    val distributions = ranges.map(parseRangeOrThrow)
    equityMonteCarloMulti(hero, board, distributions, trials, rng)

  @targetName("equityMonteCarloMultiRangesDefaultRng")
  def equityMonteCarloMulti(
      hero: HoleCards,
      board: Board,
      ranges: Seq[String],
      trials: Int
  ): EquityEstimate =
    equityMonteCarloMulti(hero, board, ranges, trials, new Random())

  /** Computes the expected value of calling a bet.
    *
    * Formula: EV(call) = equity * (pot + call) - call
    * A positive EV means calling is profitable in the long run.
    *
    * @param potBeforeCall the pot size before hero's call (must be non-negative)
    * @param callSize      the amount hero must call (must be non-negative)
    * @param equity        hero's probability of winning the hand (0.0 to 1.0)
    * @return the expected value in chips; positive = profitable call
    */
  def evCall(potBeforeCall: Double, callSize: Double, equity: Double): Double =
    require(potBeforeCall >= 0.0, "potBeforeCall must be non-negative")
    require(callSize >= 0.0, "callSize must be non-negative")
    require(equity >= 0.0 && equity <= 1.0, "equity must be between 0 and 1")
    equity * (potBeforeCall + callSize) - callSize

  /** Constructs a uniform distribution over all hole cards not blocked by hero or board.
    * Used as the default "random hand" range when no specific villain range is provided.
    */
  def fullRange(hero: HoleCards, board: Board): DiscreteDistribution[HoleCards] =
    val dead = hero.asSet ++ board.asSet
    val hands = allHoleCardsExcluding(dead)
    DiscreteDistribution.uniform(hands)

  private def validateHeroBoard(hero: HoleCards, board: Board): Unit =
    val all = hero.toVector ++ board.cards
    require(all.distinct.length == all.length, "hero and board must not share cards")

  private def parseRangeOrThrow(range: String): DiscreteDistribution[HoleCards] =
    RangeParser.parse(range) match
      case Right(dist) => dist
      case Left(err) => throw new IllegalArgumentException(s"invalid range: $err")

  /** Optional native acceleration path.
    *
    * Preflop routes to the existing preflop engines, while postflop uses the
    * dedicated native postflop Monte Carlo bridge.
    */
  private def maybeAcceleratedMonteCarlo(
      hero: HoleCards,
      board: Board,
      range: PreparedRange,
      trials: Int,
      rng: Random
  ): Option[EquityEstimate] =
    if board.size == 0 then maybeAcceleratedPreflopMonteCarlo(hero, board, range, trials, rng)
    else maybeAcceleratedPostflopMonteCarlo(hero, board, range, trials, rng)

  /** Optional preflop acceleration path.
    *
    * Uses the native CSR range runtime when possible and falls back to the
    * generic GPU/OpenCL/hybrid batch runtime. Any failure falls back to JVM MC.
    */
  private def maybeAcceleratedPreflopMonteCarlo(
      hero: HoleCards,
      board: Board,
      range: PreparedRange,
      trials: Int,
      rng: Random
  ): Option[EquityEstimate] =
    if board.size != 0 then None
    else if accelerationGuard.get().booleanValue() then None
    else
      val backend = configuredPreflopEquityBackend
      if backend == PreflopEquityBackendCpu then None
      else
        withAccelerationGuard {
          backend match
            case PreflopEquityBackendRange =>
              attemptRangeRuntimePreflop(hero, range, trials, rng)
            case PreflopEquityBackendBatch =>
              attemptBatchRuntimePreflop(hero, range, trials, rng)
            case _ =>
              configuredGpuProvider match
                case "native" =>
                  attemptRangeRuntimePreflop(hero, range, trials, rng)
                    .orElse(attemptBatchRuntimePreflop(hero, range, trials, rng))
                case "opencl" | "hybrid" | "cpu-emulated" =>
                  attemptBatchRuntimePreflop(hero, range, trials, rng)
                case _ =>
                  None
        }

  private def maybeAcceleratedPostflopMonteCarlo(
      hero: HoleCards,
      board: Board,
      range: PreparedRange,
      trials: Int,
      rng: Random
  ): Option[EquityEstimate] =
    if board.size == 0 then None
    else if accelerationGuard.get().booleanValue() then None
    else
      val villains = range.hands
      val weights = range.weights
      if villains.isEmpty then None
      else
        withAccelerationGuard {
          var i = 0

          HoldemPostflopNativeRuntime.computePostflopBatch(
            hero = hero,
            board = board,
            villains = villains,
            trials = trials,
            seedBase = rng.nextLong()
          ) match
            case Right(values) if values.length == villains.length =>
              var weightedWin = 0.0
              var weightedTie = 0.0
              var weightedLoss = 0.0
              var weightedStdErrSq = 0.0
              var weightSum = 0.0
              i = 0
              while i < values.length do
                val p = weights(i)
                val row = values(i)
                weightedWin += p * row.win
                weightedTie += p * row.tie
                weightedLoss += p * row.loss
                val weightedStdErr = p * row.stderr
                weightedStdErrSq += weightedStdErr * weightedStdErr
                weightSum += p
                i += 1
              if weightSum <= 0.0 then None
              else
                Some(
                  estimateFromAggregatedRates(
                    winRate = weightedWin / weightSum,
                    tieRate = weightedTie / weightSum,
                    lossRate = weightedLoss / weightSum,
                    stderr = math.sqrt(weightedStdErrSq) / weightSum,
                    trials = trials
                  )
                )
            case Right(_) =>
              None
            case Left(reason) =>
              GpuRuntimeSupport.log(s"postflop native runtime unavailable: $reason")
              None
        }

  private def withAccelerationGuard[A](thunk: => Option[A]): Option[A] =
    accelerationGuard.set(java.lang.Boolean.TRUE)
    try thunk
    finally accelerationGuard.set(java.lang.Boolean.FALSE)

  private def attemptRangeRuntimePreflop(
      hero: HoleCards,
      range: PreparedRange,
      trials: Int,
      rng: Random
  ): Option[EquityEstimate] =
    if range.handIds.isEmpty then None
    else
      val heroId = HoleCardsIndex.idOf(hero)
      val villainIds = range.handIds
      val keyMaterial = new Array[Long](villainIds.length)
      val probabilities = new Array[Float](villainIds.length)
      var i = 0
      while i < villainIds.length do
        val villainId = villainIds(i)
        val low = math.min(heroId, villainId)
        val high = math.max(heroId, villainId)
        keyMaterial(i) = HeadsUpEquityTable.pack(low, high) ^ (i.toLong << 32)
        probabilities(i) = range.weights(i).toFloat
        i += 1

      HeadsUpRangeGpuRuntime.computeRangeBatchMonteCarloCsr(
        heroIds = Array(heroId),
        offsets = Array(0, villainIds.length),
        villainIds = villainIds,
        keyMaterial = keyMaterial,
        probabilities = probabilities,
        trials = trials,
        monteCarloSeedBase = rng.nextLong()
      ) match
        case Right(values) if values.nonEmpty =>
          val row = values(0)
          Some(estimateFromAggregatedRates(row.win, row.tie, row.loss, row.stderr, trials))
        case Right(_) =>
          None
        case Left(reason) =>
          GpuRuntimeSupport.log(s"preflop range runtime unavailable: $reason")
          None

  private def attemptBatchRuntimePreflop(
      hero: HoleCards,
      range: PreparedRange,
      trials: Int,
      rng: Random
  ): Option[EquityEstimate] =
    val villainIds = range.handIds
    if villainIds.isEmpty then None
    else
      val heroId = HoleCardsIndex.idOf(hero)
      val packedKeys = new Array[Long](villainIds.length)
      val keyMaterial = new Array[Long](villainIds.length)
      val weights = range.weights
      val flipped = new Array[Boolean](villainIds.length)
      var i = 0
      while i < villainIds.length do
        val villainId = villainIds(i)
        val low = math.min(heroId, villainId)
        val high = math.max(heroId, villainId)
        packedKeys(i) = HeadsUpEquityTable.pack(low, high)
        keyMaterial(i) = packedKeys(i) ^ (i.toLong << 33)
        flipped(i) = heroId > villainId
        i += 1

      HeadsUpGpuRuntime.computeBatch(
        packedKeys = packedKeys,
        keyMaterial = keyMaterial,
        mode = HeadsUpEquityTable.Mode.MonteCarlo(trials),
        monteCarloSeedBase = rng.nextLong()
      ) match
        case Right(values) if values.length == villainIds.length =>
          var weightedWin = 0.0
          var weightedTie = 0.0
          var weightedLoss = 0.0
          var weightedStdErrSq = 0.0
          var weightSum = 0.0
          i = 0
          while i < values.length do
            val p = weights(i)
            val row = HeadsUpEquityTable.flipIfNeeded(values(i), flipped(i))
            weightedWin += p * row.win
            weightedTie += p * row.tie
            weightedLoss += p * row.loss
            val weightedStdErr = p * row.stderr
            weightedStdErrSq += weightedStdErr * weightedStdErr
            weightSum += p
            i += 1
          if weightSum <= 0.0 then None
          else
            Some(
              estimateFromAggregatedRates(
                winRate = weightedWin / weightSum,
                tieRate = weightedTie / weightSum,
                lossRate = weightedLoss / weightSum,
                stderr = math.sqrt(weightedStdErrSq) / weightSum,
                trials = trials
              )
            )
        case Right(_) =>
          None
        case Left(reason) =>
          GpuRuntimeSupport.log(s"preflop batch runtime unavailable: $reason")
          None

  private def configuredPreflopEquityBackend: String =
    configuredPreflopEquityBackendCached

  private lazy val configuredPreflopEquityBackendCached: String =
    GpuRuntimeSupport.resolveNonEmptyLower(PreflopEquityBackendProperty, PreflopEquityBackendEnv) match
      case Some("cpu") => PreflopEquityBackendCpu
      case Some("range" | "native-range") => PreflopEquityBackendRange
      case Some("batch" | "gpu-batch") => PreflopEquityBackendBatch
      case Some("auto") => PreflopEquityBackendAuto
      case _ => PreflopEquityBackendAuto

  private def configuredGpuProvider: String =
    configuredGpuProviderCached

  private lazy val configuredGpuProviderCached: String =
    GpuRuntimeSupport.resolveNonEmptyLower(GpuProviderProperty, GpuProviderEnv).getOrElse("native")

  private inline def cardMask(card: Card): Long =
    1L << card.id

  private inline def handMask(hand: HoleCards): Long =
    cardMask(hand.first) | cardMask(hand.second)

  private def deadMask(hero: HoleCards, board: Board): Long =
    var mask = handMask(hero)
    val cards = board.cards
    var i = 0
    while i < cards.length do
      mask |= cardMask(cards(i))
      i += 1
    mask

  private def prepareRange(
      range: DiscreteDistribution[HoleCards],
      deadMaskValue: Long
  ): PreparedRange =
    val weightScratch = preparedRangeWeightScratch.get()
    val touchedIds = preparedRangeTouchedIdsScratch.get()
    var touchedCount = 0
    try
      range.weights.foreach { case (hand, weight) =>
        if weight > 0.0 && ((handMask(hand) & deadMaskValue) == 0L) then
          val handId = HoleCardsIndex.fastIdOf(hand)
          if weightScratch(handId) == 0.0 then
            touchedIds(touchedCount) = handId
            touchedCount += 1
          weightScratch(handId) += weight
      }
      require(touchedCount > 0, "villain range is empty after filtering")
      java.util.Arrays.sort(touchedIds, 0, touchedCount)

      val hands = new Array[HoleCards](touchedCount)
      val weights = new Array[Double](touchedCount)
      val handIds = new Array[Int](touchedCount)
      var total = 0.0
      var i = 0
      while i < touchedCount do
        val handId = touchedIds(i)
        val weight = weightScratch(handId)
        hands(i) = HoleCardsIndex.byIdUnchecked(handId)
        weights(i) = weight
        handIds(i) = handId
        total += weight
        i += 1
      require(total > 0.0, "villain range is empty after filtering")
      val invTotal = 1.0 / total
      i = 0
      while i < weights.length do
        weights(i) *= invTotal
        i += 1
      PreparedRange(hands, weights, handIds)
    finally
      var i = 0
      while i < touchedCount do
        weightScratch(touchedIds(i)) = 0.0
        i += 1

  private def prepareRangeFromCompact(
      compact: CompactPosterior,
      deadMaskValue: Long
  ): PreparedRange =
    val weightScratch = preparedRangeWeightScratch.get()
    val touchedIds = preparedRangeTouchedIdsScratch.get()
    var touchedCount = 0
    try
      var i = 0
      while i < compact.size do
        val rawWeight = compact.probWeights(i)
        if rawWeight > 0 then
          val canonical = HoleCards.canonical(compact.hands(i).first, compact.hands(i).second)
          if (handMask(canonical) & deadMaskValue) == 0L then
            val handId = HoleCardsIndex.fastIdOf(canonical)
            if weightScratch(handId) == 0.0 then
              touchedIds(touchedCount) = handId
              touchedCount += 1
            weightScratch(handId) += rawWeight.toDouble
        i += 1
      require(touchedCount > 0, "villain range is empty after filtering")
      java.util.Arrays.sort(touchedIds, 0, touchedCount)

      val hands = new Array[HoleCards](touchedCount)
      val weights = new Array[Double](touchedCount)
      val handIds = new Array[Int](touchedCount)
      var total = 0.0
      i = 0
      while i < touchedCount do
        val handId = touchedIds(i)
        val weight = weightScratch(handId)
        hands(i) = HoleCardsIndex.byIdUnchecked(handId)
        weights(i) = weight
        handIds(i) = handId
        total += weight
        i += 1
      require(total > 0.0, "villain range is empty after filtering")
      val invTotal = 1.0 / total
      i = 0
      while i < weights.length do
        weights(i) *= invTotal
        i += 1
      PreparedRange(hands, weights, handIds)
    finally
      var i = 0
      while i < touchedCount do
        weightScratch(touchedIds(i)) = 0.0
        i += 1

  /** Like prepareRange but outputs Prob (Int32) weights and filters by dead card Set.
    * Used by equityExactProb which needs Set-based dead card filtering (not bitmask).
    */
  private def prepareRangeProbFromCompact(
      compact: CompactPosterior,
      dead: Set[Card]
  ): PreparedRangeProb =
    val weightScratch = preparedRangeWeightScratch.get()
    val touchedIds = preparedRangeTouchedIdsScratch.get()
    var touchedCount = 0
    try
      var i = 0
      while i < compact.size do
        val rawWeight = compact.probWeights(i)
        if rawWeight > 0 then
          val canonical = HoleCards.canonical(compact.hands(i).first, compact.hands(i).second)
          if !dead.contains(canonical.first) && !dead.contains(canonical.second) then
            val handId = HoleCardsIndex.fastIdOf(canonical)
            if weightScratch(handId) == 0.0 then
              touchedIds(touchedCount) = handId
              touchedCount += 1
            weightScratch(handId) += rawWeight.toDouble
        i += 1
      require(touchedCount > 0, "villain range is empty after filtering")
      java.util.Arrays.sort(touchedIds, 0, touchedCount)

      val hands = new Array[HoleCards](touchedCount)
      val probWeights = new Array[Int](touchedCount)
      var total = 0.0
      i = 0
      while i < touchedCount do
        total += weightScratch(touchedIds(i))
        i += 1
      require(total > 0.0, "villain range is empty after filtering")
      val invTotal = 1.0 / total
      i = 0
      while i < touchedCount do
        val handId = touchedIds(i)
        hands(i) = HoleCardsIndex.byIdUnchecked(handId)
        probWeights(i) = Prob.fromDouble(weightScratch(handId) * invTotal).raw
        i += 1
      PreparedRangeProb(hands, probWeights, touchedCount)
    finally
      var i = 0
      while i < touchedCount do
        weightScratch(touchedIds(i)) = 0.0
        i += 1

  private def prepareRangeProb(
      range: DiscreteDistribution[HoleCards],
      dead: Set[Card]
  ): PreparedRangeProb =
    val weightScratch = preparedRangeWeightScratch.get()
    val touchedIds = preparedRangeTouchedIdsScratch.get()
    var touchedCount = 0
    try
      range.weights.foreach { case (hand, weight) =>
        if weight > 0.0 &&
          !dead.contains(hand.first) &&
          !dead.contains(hand.second)
        then
          val canonical = HoleCards.canonical(hand.first, hand.second)
          val handId = HoleCardsIndex.fastIdOf(canonical)
          if weightScratch(handId) == 0.0 then
            touchedIds(touchedCount) = handId
            touchedCount += 1
          weightScratch(handId) += weight
      }
      require(touchedCount > 0, "villain range is empty after filtering")
      java.util.Arrays.sort(touchedIds, 0, touchedCount)

      val hands = new Array[HoleCards](touchedCount)
      val probWeights = new Array[Int](touchedCount)
      var total = 0.0
      var i = 0
      while i < touchedCount do
        total += weightScratch(touchedIds(i))
        i += 1
      require(total > 0.0, "villain range is empty after filtering")
      val invTotal = 1.0 / total
      i = 0
      while i < touchedCount do
        val handId = touchedIds(i)
        hands(i) = HoleCardsIndex.byIdUnchecked(handId)
        probWeights(i) = Prob.fromDouble(weightScratch(handId) * invTotal).raw
        i += 1
      PreparedRangeProb(hands, probWeights, touchedCount)
    finally
      var i = 0
      while i < touchedCount do
        weightScratch(touchedIds(i)) = 0.0
        i += 1

  private def estimateFromAggregatedRates(
      winRate: Double,
      tieRate: Double,
      lossRate: Double,
      stderr: Double,
      trials: Int
  ): EquityEstimate =
    val total = winRate + tieRate + lossRate
    val (normalizedWin, normalizedTie, normalizedLoss) =
      if total > 0.0 then
        (
          winRate / total,
          tieRate / total,
          lossRate / total
        )
      else
        (0.0, 0.0, 1.0)
    val mean = normalizedWin + (normalizedTie / 2.0)
    val safeStdErr = math.max(0.0, stderr)
    val variance = safeStdErr * safeStdErr * trials.toDouble
    EquityEstimate(
      mean = mean,
      variance = variance,
      stderr = safeStdErr,
      trials = trials,
      winRate = normalizedWin,
      tieRate = normalizedTie,
      lossRate = normalizedLoss
    )

  /**
    * Filters and canonicalizes a villain range distribution by removing hands that
    * overlap with dead cards (hero + board), canonicalizing hand ordering, and
    * deduplicating entries that collapse to the same canonical hand.
    *
    * @param range the raw villain range distribution
    * @param dead  set of cards that are already in play (hero + board)
    * @return a clean, normalized distribution with no dead-card conflicts
    */
  private def sanitizeRange(
      range: DiscreteDistribution[HoleCards],
      dead: Set[Card]
  ): DiscreteDistribution[HoleCards] =
    // Use mutable HashMap for accumulation (needed for canonical dedup),
    // but with sizeHint to reduce rehashing.
    val collapsed = mutable.HashMap.empty[HoleCards, Double]
    collapsed.sizeHint(range.weights.size)
    range.weights.foreach { case (hand, weight) =>
      if weight > 0.0 &&
        !dead.contains(hand.first) &&
        !dead.contains(hand.second)
      then
        val canonical = HoleCards.canonical(hand.first, hand.second)
        collapsed.update(canonical, collapsed.getOrElse(canonical, 0.0) + weight)
    }
    require(collapsed.nonEmpty, "villain range is empty after filtering")
    DiscreteDistribution(collapsed.toMap).normalized

  /** Safety check: estimates the total number of evaluations for exact multi-villain
    * enumeration and rejects the request if it exceeds maxEvaluations. This prevents
    * accidentally launching a computation that would take hours or days.
    */
  private def ensureExactFeasible(
      board: Board,
      ranges: Seq[DiscreteDistribution[HoleCards]],
      maxEvaluations: Long
  ): Unit =
    val remainingUpper = 52 - 2 - board.cards.length
    val boardCombos = combinationsCount(remainingUpper, board.missing)
    var estimate = boardCombos
    ranges.foreach { range =>
      val size = range.weights.size.toLong
      estimate = safeMultiply(estimate, size, maxEvaluations)
    }
    require(
      estimate <= maxEvaluations,
      s"exact enumeration upper bound $estimate exceeds maxEvaluations $maxEvaluations; use MonteCarlo or narrower ranges"
    )

  /** Overflow-safe multiplication that clamps to limit+1 instead of wrapping on overflow. */
  private def safeMultiply(a: Long, b: Long, limit: Long): Long =
    if a == 0L || b == 0L then 0L
    else if a > limit / b then limit + 1L
    else a * b

  /** Rejection-samples one hand per villain such that no two hands share any cards.
    * Retries up to maxAttempts times; throws if no valid combination is found.
    * This is necessary in multi-villain Monte Carlo to maintain physical consistency.
    */
  private def sampleVillainsNoOverlap(
      samplers: Vector[WeightedSampler[HoleCards]],
      dead: Set[Card],
      rng: Random,
      maxAttempts: Int = 1000
  ): Vector[HoleCards] =
    import scala.util.boundary, boundary.break
    require(samplers.nonEmpty, "samplers must be non-empty")
    boundary:
      var attempt = 0
      while attempt < maxAttempts do
        var used = dead
        val hands = new Array[HoleCards](samplers.length)
        var ok = true
        var i = 0
        while i < samplers.length && ok do
          val hand = samplers(i).sample(rng)
          if used.contains(hand.first) || used.contains(hand.second) then ok = false
          else
            hands(i) = hand
            used = used + hand.first + hand.second
            i += 1
        if ok then break(hands.toVector)
        attempt += 1
      throw new IllegalArgumentException("unable to sample non-overlapping villain hands; ranges may be too restrictive")

  /** Sorts range weights by HoleCardsIndex ID for deterministic iteration order.
    * This ensures exact multi-villain enumeration produces identical results regardless
    * of Map iteration order.
    */
  private def sortedWeights(weights: Map[HoleCards, Double]): Vector[(HoleCards, Double)] =
    if weights.isEmpty then Vector.empty
    else
      val entries = new Array[(HoleCards, Double, Int)](weights.size)
      var i = 0
      weights.foreach { case (hand, weight) =>
        entries(i) = (hand, weight, HoleCardsIndex.fastIdOf(hand))
        i += 1
      }
      java.util.Arrays.sort(entries, Ordering.by[(HoleCards, Double, Int), Int](_._3))
      // Build result array directly and wrap to Vector, avoiding
      // Vector.newBuilder overhead for known-size output.
      val result = new Array[(HoleCards, Double)](entries.length)
      var j = 0
      while j < entries.length do
        val e = entries(j)
        result(j) = e._1 -> e._2
        j += 1
      result.toVector

  /** Generates all canonical hole cards from the deck excluding dead cards. */
  private def allHoleCardsExcluding(dead: Set[Card]): Vector[HoleCards] =
    val remaining = Deck.full.filterNot(dead.contains).toIndexedSeq
    HoldemCombinator.holeCardsFrom(remaining)

  /** Computes C(n,k) = n! / (k! * (n-k)!) using iterative multiplication to avoid overflow.
    * Uses the identity C(n,k) = C(n, n-k) to minimize the number of multiplications.
    */
  private def combinationsCount(n: Int, k: Int): Long =
    require(k >= 0 && k <= n)
    if k == 0 || k == n then 1L
    else
      val kk = math.min(k, n - k)
      var numer = 1L
      var denom = 1L
      var i = 1
      while i <= kk do
        numer *= (n - kk + i).toLong
        denom *= i.toLong
        i += 1
      numer / denom

  /** Fisher-Yates partial shuffle to sample k items without replacement from the collection.
    * Only shuffles the first k positions, then extracts them.
    */
  private def sampleWithoutReplacement[A](items: IndexedSeq[A], k: Int, rng: Random): Vector[A] =
    require(k >= 0 && k <= items.length)
    val buffer = scala.collection.mutable.ArrayBuffer.from(items)
    var i = 0
    while i < k do
      val j = i + rng.nextInt(buffer.length - i)
      val tmp = buffer(i)
      buffer(i) = buffer(j)
      buffer(j) = tmp
      i += 1
    // Build Vector directly from first k elements instead of
    // buffer.take(k).toVector which allocates an intermediate ArrayBuffer.
    val result = Vector.newBuilder[A]
    result.sizeHint(k)
    i = 0
    while i < k do
      result += buffer(i)
      i += 1
    result.result()

  /** Fills the missing board cards by uniformly sampling without replacement from
    * deckMinusDead while excluding the current villain hole cards.
    * Uses a small fixed-size index scratch buffer to avoid per-trial allocations.
    */
  private def fillBoardWithRandomCards(
      boardArr: Array[Card],
      knownBoardSize: Int,
      deckMinusDead: IndexedSeq[Card],
      blockedFirst: Card,
      blockedSecond: Card,
      missing: Int,
      sampledDeckIndexes: Array[Int],
      rng: Random
  ): Unit =
    var draw = 0
    while draw < missing do
      var accepted = false
      while !accepted do
        val idx = rng.nextInt(deckMinusDead.length)
        val card = deckMinusDead(idx)
        if card != blockedFirst && card != blockedSecond then
          var duplicate = false
          var j = 0
          while j < draw && !duplicate do
            if sampledDeckIndexes(j) == idx then duplicate = true
            j += 1
          if !duplicate then
            sampledDeckIndexes(draw) = idx
            boardArr(knownBoardSize + draw) = card
            accepted = true
      draw += 1

  /** Weighted random sampler using cumulative distribution function (CDF) with binary search.
    * Constructed once per range preparation, then sampled O(log n) per trial.
    */
  private object WeightedSampler:
    /** Builds a sampler from parallel arrays of items and their probability weights.
      * Constructs a cumulative weight array for O(log n) binary-search sampling.
      */
    def fromArrays[A](items: Array[A], weights: Array[Double]): WeightedSampler[A] =
      require(items.nonEmpty, "cannot sample from empty range")
      require(items.length == weights.length, "sampler items/weights length mismatch")
      val cumulative = new Array[Double](weights.length)
      var acc = 0.0
      var i = 0
      while i < weights.length do
        acc += weights(i)
        cumulative(i) = acc
        i += 1
      new WeightedSampler(items.asInstanceOf[Array[Any]], cumulative)

  /**
    * Weighted random sampler backed by a cumulative distribution array.
    * Uses binary search over the CDF to achieve O(log n) sampling per call.
    *
    * @param values     the items to sample from (type-erased to Any for array covariance)
    * @param cumulative running sum of weights; cumulative(i) = sum(weights[0..i])
    */
  private final class WeightedSampler[A] private (
      values: Array[Any],
      cumulative: Array[Double]
  ):
    private val total: Double = cumulative.last

    /** Draws one sample using inverse CDF method with binary search. */
    def sample(rng: Random): A =
      // Draw uniform in [0, total), then find the first index where cumulative >= r
      val r = rng.nextDouble() * total
      var low = 0
      var high = cumulative.length - 1
      while low < high do
        val mid = (low + high) >>> 1
        if r <= cumulative(mid) then high = mid else low = mid + 1
      values(low).asInstanceOf[A]
