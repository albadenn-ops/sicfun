package sicfun.holdem.engine

import sicfun.core.{Card, DiscreteDistribution}
import sicfun.holdem.types.*
import sicfun.holdem.cfr.{HoldemCfrConfig, HoldemCfrSolver}

import scala.collection.mutable
import scala.util.Random

/** Unified GTO (Game Theory Optimal) decision system with cached CFR solving and a
  * fast heuristic fallback path.
  *
  * This object provides the GTO decision-making layer for the playing hall. It supports
  * two modes:
  *
  *   - '''Fast mode''': Uses [[HandStrengthEstimator.fastGtoStrength]] with threshold-based
  *     heuristics to make quick decisions without running a solver. Suitable for low-latency
  *     or high-volume simulation contexts.
  *
  *   - '''Exact mode''': Runs [[HoldemCfrSolver.solveShallowDecisionPolicy]] to compute a
  *     proper mixed-strategy equilibrium at the decision point. Results are cached by a
  *     canonical suit-invariant key to avoid redundant solves for strategically equivalent
  *     spots (e.g., AsKs and AhKh on isomorphic boards produce the same cache key).
  *
  * Key design decisions:
  *   - '''Suit canonicalization''': All 24 suit permutations are tried to find the
  *     lexicographically smallest (hero, board) representation. This dramatically improves
  *     cache hit rates by collapsing suit-isomorphic situations into one key.
  *   - '''Quantized cache keys''': Pot, toCall, and stack are quantized into buckets to
  *     increase cache reuse for nearby bet sizes.
  *   - '''CFR parametrization''': Iteration count, villain hand count, and equity trials
  *     are scaled by street and candidate count. Preflop gets more resources; binary
  *     (2-candidate) spots get reduced budgets.
  *   - '''LRU-ish eviction''': When the cache exceeds [[MaxGtoCacheEntries]], it is
  *     fully cleared (simple but effective for bounded memory).
  *
  * Extracted from TexasHoldemPlayingHall where gtoHeroResponds and gtoVillainResponds
  * were near-identical (only difference: posterior source). The unified [[gtoResponds]]
  * takes the opponent posterior as a parameter.
  */
private[holdem] object GtoSolveEngine:

  /** Controls which GTO decision path is used.
    *   - Fast: heuristic threshold-based decisions using [[HandStrengthEstimator]]
    *   - Exact: full CFR solve via [[HoldemCfrSolver]]
    */
  enum GtoMode:
    case Fast
    case Exact

  /** Cache key for exact GTO solutions. Uses canonical (suit-invariant) representations
    * of hero cards and board, plus quantized pot/stack/toCall buckets, to maximize
    * cache reuse across strategically equivalent spots.
    */
  private[holdem] final case class GtoSolveCacheKey(
      perspective: Int,
      canonicalHeroPacked: Long,
      streetOrdinal: Int,
      canonicalBoardPacked: Long,
      potBucket: Int,
      toCallBucket: Int,
      stackBucket: Int,
      candidateHash: Int,
      baseEquityTrials: Int
  )

  /** Result of suit canonicalization: the lexicographically minimal hero+board encoding
    * and the suit permutation that produced it.
    */
  private[holdem] final case class CanonicalSuitContext(
      canonicalHeroPacked: Long,
      canonicalBoardPacked: Long,
      suitMap: Array[Int]
  )

  /** Cached CFR solution: the action probability distribution and best action, plus
    * the provider string identifying which solver produced it (e.g., "native-batch", "jvm").
    */
  private[holdem] final case class GtoCachedPolicy(
      orderedActionProbabilities: Vector[(PokerAction, Double)],
      bestAction: PokerAction,
      provider: String
  )

  /** Mutable telemetry counters tracking cache hit/miss rates and which solver providers
    * served or computed each result. Used for performance diagnostics.
    */
  private[holdem] final case class GtoCacheStats(
      var hits: Long = 0L,
      var misses: Long = 0L,
      servedByProvider: mutable.Map[String, Long] = mutable.HashMap.empty[String, Long].withDefaultValue(0L),
      solvedByProvider: mutable.Map[String, Long] = mutable.HashMap.empty[String, Long].withDefaultValue(0L)
  ):
    def total: Long = hits + misses
    def hitRate: Double = if total > 0 then hits.toDouble / total.toDouble else 0.0
    def recordHit(provider: String): Unit =
      hits += 1L
      increment(servedByProvider, provider)
    def recordMiss(provider: String): Unit =
      misses += 1L
      increment(servedByProvider, provider)
      increment(solvedByProvider, provider)
    def servedByProviderSnapshot: Map[String, Long] = servedByProvider.toMap
    def solvedByProviderSnapshot: Map[String, Long] = solvedByProvider.toMap
    private def increment(counter: mutable.Map[String, Long], provider: String): Unit =
      counter.update(provider, counter(provider) + 1L)

  /** Upper bound on cached GTO policies. When exceeded, the entire cache is cleared.
    * 500K entries balances memory usage against hit rate for long-running match sessions.
    */
  private[holdem] val MaxGtoCacheEntries = 500000

  /** GTO threshold configuration for the fast heuristic decision path.
    * Default values are initial estimates pending calibration from CFR solver output.
    */
  final case class GtoThresholds(
      raiseThreshold: Map[Street, Double],
      foldMargin: Map[Street, Double],
      raiseGap: Map[Street, Double]
  )

  /** Default thresholds preserving the original hardcoded values. */
  val defaultThresholds: GtoThresholds = GtoThresholds(
    raiseThreshold = Map(
      Street.Preflop -> 0.78,
      Street.Flop    -> 0.74,
      Street.Turn    -> 0.71,
      Street.River   -> 0.68
    ),
    foldMargin = Map(
      Street.Preflop -> 0.05,
      Street.Flop    -> 0.03,
      Street.Turn    -> 0.01,
      Street.River   -> -0.01
    ),
    raiseGap = Map(
      Street.Preflop -> 0.27,
      Street.Flop    -> 0.24,
      Street.Turn    -> 0.22,
      Street.River   -> 0.20
    )
  )

  /** Unified GTO decision dispatcher. Replaces the near-identical
    * gtoHeroResponds / gtoVillainResponds pair.
    *
    * @param candidates pre-computed legal actions (Fold/Call/Check + raises)
    * @param opponentPosterior opponent's range (caller passes tableRanges.rangeFor(opponentPosition))
    * @param perspective 0 for hero, 1 for villain (used in cache key and RNG seed)
    */
  private[holdem] def gtoResponds(
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      mode: GtoMode,
      opponentPosterior: DiscreteDistribution[HoleCards],
      baseEquityTrials: Int,
      rng: Random,
      perspective: Int,
      exactGtoCache: mutable.HashMap[GtoSolveCacheKey, GtoCachedPolicy],
      exactGtoCacheStats: GtoCacheStats
  ): PokerAction =
    if candidates.length <= 1 then candidates.head
    else
      mode match
        case GtoMode.Fast =>
          fastGtoResponds(
            hand = hand,
            state = state,
            candidates = candidates,
            rng = rng
          )
        case GtoMode.Exact =>
          solveGtoByCfr(
            hand = hand,
            state = state,
            candidates = candidates,
            villainPosterior = opponentPosterior,
            baseEquityTrials = baseEquityTrials,
            rng = rng,
            perspective = perspective,
            exactGtoCache = exactGtoCache,
            exactGtoCacheStats = exactGtoCacheStats
          )

  /** Solves a GTO decision point using CFR with caching.
    *
    * Flow:
    *   1. Compute the canonical suit-invariant (hero, board) signature.
    *   2. Build a cache key from the canonical signature + quantized game state.
    *   3. If cached, sample an action from the stored policy.
    *   4. If not cached, configure and run [[HoldemCfrSolver.solveShallowDecisionPolicy]],
    *      then cache the result and sample an action.
    *   5. On solver failure, fall back to uniform random selection to preserve run continuity.
    */
  private def solveGtoByCfr(
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      villainPosterior: DiscreteDistribution[HoleCards],
      baseEquityTrials: Int,
      rng: Random,
      perspective: Int,
      exactGtoCache: mutable.HashMap[GtoSolveCacheKey, GtoCachedPolicy],
      exactGtoCacheStats: GtoCacheStats
  ): PokerAction =
    val canonicalSignature = canonicalHeroBoardSignature(hand = hand, board = state.board)
    val key = buildGtoSolveCacheKey(
      perspective = perspective,
      hand = hand,
      state = state,
      candidates = candidates,
      baseEquityTrials = baseEquityTrials,
      canonicalSignature = canonicalSignature
    )
    exactGtoCache.get(key) match
      case Some(cached) =>
        exactGtoCacheStats.recordHit(cached.provider)
        sampleActionByPolicy(
          ordered = cached.orderedActionProbabilities,
          fallback = cached.bestAction,
          rng = rng
        )
      case None =>
        val config = HoldemCfrConfig(
          iterations = gtoIterations(state.street, baseEquityTrials, candidates.length),
          maxVillainHands = gtoMaxVillainHands(state.street, candidates.length),
          equityTrials = gtoEquityTrials(state.street, baseEquityTrials, candidates.length),
          postflopLookahead = false,
          rngSeed = exactEquitySeed(
            perspective = perspective,
            baseEquityTrials = baseEquityTrials,
            boardSize = state.board.size,
            canonicalSignature = canonicalSignature
          )
        )
        try
          val solution = HoldemCfrSolver.solveShallowDecisionPolicy(
            hero = hand,
            state = state,
            villainPosterior = villainPosterior,
            candidateActions = candidates,
            config = config
          )
          val actionProbs =
            orderedPositiveProbabilities(
              actions = candidates,
              probabilities = solution.actionProbabilities
            )
          exactGtoCacheStats.recordMiss(solution.provider)
          if exactGtoCache.size >= MaxGtoCacheEntries then exactGtoCache.clear()
          exactGtoCache.update(
            key,
            GtoCachedPolicy(
              orderedActionProbabilities = actionProbs,
              bestAction = solution.bestAction,
              provider = solution.provider
            )
          )
          sampleActionByPolicy(
            ordered = actionProbs,
            fallback = solution.bestAction,
            rng = rng
          )
        catch
          case _: Throwable =>
            // Preserve run continuity if a specific CFR solve fails.
            exactGtoCacheStats.recordMiss("random-fallback")
            candidates(rng.nextInt(candidates.length))

  /** Fast heuristic GTO decision path (no solver).
    *
    * Uses [[HandStrengthEstimator.fastGtoStrength]] to get a deterministic strength
    * estimate, then applies street-dependent thresholds with mixed-strategy regions:
    *
    * '''Check-to-act (toCall <= 0):'''
    *   - Pure raise above pureRaiseThreshold, mixed raise in a transition zone,
    *     check otherwise.
    *
    * '''Facing a bet (toCall > 0):'''
    *   - Fold below foldThreshold (pot-odds + street margin).
    *   - Call between foldThreshold and raiseThreshold.
    *   - Mixed raise above raiseThreshold with probability proportional to excess strength.
    */
  private def fastGtoResponds(
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      rng: Random
  ): PokerAction =
    val strength = HandStrengthEstimator.fastGtoStrength(hand, state.board, state.street)
    // No allowRaise guard needed: callers pre-filter raises out of candidates
    // via heroCandidates(state, raiseSize, allowRaise) before calling gtoResponds.
    val raiseCandidate = candidates.collectFirst { case action @ PokerAction.Raise(_) => action }
    val callCandidate = candidates.find(_ == PokerAction.Call)
    val foldCandidate = candidates.find(_ == PokerAction.Fold)
    if state.toCall <= 0.0 then
      raiseCandidate match
        case None => PokerAction.Check
        case Some(raiseAction) =>
          val pureRaiseThreshold = fastGtoRaiseThreshold(state.street)
          val mixRaiseThreshold = pureRaiseThreshold - 0.18
          if strength >= pureRaiseThreshold then raiseAction
          else if strength >= mixRaiseThreshold then
            val mix = HandStrengthEstimator.clamp(0.18 + ((strength - mixRaiseThreshold) * 1.7), 0.05, 0.80)
            if rng.nextDouble() < mix then raiseAction else PokerAction.Check
          else PokerAction.Check
    else
      val potOdds = state.potOdds
      val foldThreshold = HandStrengthEstimator.clamp(potOdds + fastGtoFoldMargin(state.street), 0.06, 0.95)
      val raiseThreshold = HandStrengthEstimator.clamp(foldThreshold + fastGtoRaiseGap(state.street), 0.24, 0.98)
      if raiseCandidate.nonEmpty && strength >= raiseThreshold then
        val raiseMix = HandStrengthEstimator.clamp(0.20 + ((strength - raiseThreshold) * 1.3), 0.10, 0.92)
        if rng.nextDouble() < raiseMix then raiseCandidate.get
        else callCandidate.getOrElse(PokerAction.Call)
      else if strength >= foldThreshold then
        callCandidate.getOrElse(PokerAction.Call)
      else
        foldCandidate.getOrElse(PokerAction.Fold)

  /** Minimum strength for a pure (always) raise in the fast GTO path.
    * Decreases from preflop (0.78) to river (0.68) as board information increases.
    */
  private def fastGtoRaiseThreshold(street: Street, thresholds: GtoThresholds = defaultThresholds): Double =
    thresholds.raiseThreshold.getOrElse(street, 0.65)

  /** Extra margin added to pot odds to determine the fold threshold.
    * Positive = tighter than pot odds (preflop); negative = looser (river).
    */
  private def fastGtoFoldMargin(street: Street, thresholds: GtoThresholds = defaultThresholds): Double =
    thresholds.foldMargin.getOrElse(street, 0.02)

  /** Gap between fold threshold and raise threshold in the fast GTO path.
    * Narrower on later streets as decisions become more polarized.
    */
  private def fastGtoRaiseGap(street: Street, thresholds: GtoThresholds = defaultThresholds): Double =
    thresholds.raiseGap.getOrElse(street, 0.22)

  // --- CFR parametrization ---
  // These methods scale CFR solver resources (iterations, villain hands, equity trials)
  // by street and candidate count. The rationale: preflop has the widest ranges and
  // needs more iterations; binary decisions (2 candidates) can converge faster.

  /** Computes the number of CFR iterations for a GTO solve.
    *
    * Base iterations scale linearly with baseEquityTrials / 3, clamped to [72, 224].
    * Street adjustments: preflop gets +32; turn and river get progressive discounts.
    * Binary decisions (2 candidates) apply a 0.60x reduction with per-street floors.
    */
  private[holdem] def gtoIterations(
      street: Street,
      baseEquityTrials: Int,
      candidateCount: Int
  ): Int =
    val base = math.max(72, math.min(224, math.round(baseEquityTrials / 3.0).toInt))
    val streetBase =
      street match
        case Street.Preflop => base + 32
        case Street.Flop    => base
        case Street.Turn    => math.max(72, math.round(base * 0.85).toInt)
        case Street.River   => math.max(56, math.round(base * 0.70).toInt)
    if candidateCount <= 2 then
      val floor =
        street match
          case Street.Preflop => 88
          case Street.Flop    => 64
          case Street.Turn    => 56
          case Street.River   => 48
      math.max(floor, math.round(streetBase * 0.60).toInt)
    else
      streetBase

  /** Maximum number of villain hands sampled from the posterior for CFR solving.
    * Decreases by street (56 preflop -> 16 river) since later streets have narrower ranges.
    * Binary decisions reduce by 12 with a floor of 16.
    */
  private[holdem] def gtoMaxVillainHands(
      street: Street,
      candidateCount: Int
  ): Int =
    val base =
      street match
        case Street.Preflop => 56
        case Street.Flop    => 32
        case Street.Turn    => 24
        case Street.River   => 16
    if candidateCount <= 2 then math.max(16, base - 12) else base

  /** Number of Monte Carlo equity trials used inside each CFR iteration.
    * Scales with baseEquityTrials but divides more aggressively on later streets.
    * Binary decisions apply 0.65x reduction with per-street floors.
    */
  private[holdem] def gtoEquityTrials(
      street: Street,
      baseEquityTrials: Int,
      candidateCount: Int
  ): Int =
    val base =
      street match
        case Street.Preflop => math.max(80, baseEquityTrials / 3)
        case Street.Flop    => math.max(48, baseEquityTrials / 6)
        case Street.Turn    => math.max(32, baseEquityTrials / 8)
        case Street.River   => 24
    if candidateCount <= 2 then
      val floor =
        street match
          case Street.Preflop => 64
          case Street.Flop    => 36
          case Street.Turn    => 24
          case Street.River   => 16
      math.max(floor, math.round(base * 0.65).toInt)
    else
      base

  // --- Cache key construction ---

  /** Builds a cache key for a GTO solve, using the canonical (suit-invariant) hero+board
    * signature and quantized game-state buckets to maximize cache reuse.
    */
  private[holdem] def buildGtoSolveCacheKey(
      perspective: Int,
      hand: HoleCards,
      state: GameState,
      candidates: Vector[PokerAction],
      baseEquityTrials: Int,
      canonicalSignature: (Long, Long)
  ): GtoSolveCacheKey =
    val (canonicalHeroPacked, canonicalBoardPacked) = canonicalSignature
    GtoSolveCacheKey(
      perspective = perspective,
      canonicalHeroPacked = canonicalHeroPacked,
      streetOrdinal = state.street.ordinal,
      canonicalBoardPacked = canonicalBoardPacked,
      potBucket = quantizePot(state.pot),
      toCallBucket = quantizeToCall(state.toCall),
      stackBucket = quantizeStack(state.stackSize),
      candidateHash = hashActions(candidates),
      baseEquityTrials = baseEquityTrials
    )

  /** Quantize pot into ~1bb buckets up to 20bb, then ~5bb buckets beyond. */
  private def quantizePot(pot: Double): Int =
    if pot <= 20.0 then math.round(pot).toInt
    else 20 + math.round((pot - 20.0) / 5.0).toInt

  /** Quantize toCall into half-bb buckets. */
  private def quantizeToCall(toCall: Double): Int =
    math.round(toCall * 2.0).toInt

  /** Quantize stack into ~5bb buckets. */
  private def quantizeStack(stack: Double): Int =
    math.round(stack / 5.0).toInt

  /** Deterministic RNG seed for CFR equity calculations, derived from the canonical
    * game signature. Ensures reproducibility for the same strategic situation.
    */
  private def exactEquitySeed(
      perspective: Int,
      baseEquityTrials: Int,
      boardSize: Int,
      canonicalSignature: (Long, Long)
  ): Long =
    val (canonicalHeroPacked, canonicalBoardPacked) = canonicalSignature
    mix64(
      canonicalHeroPacked ^
        java.lang.Long.rotateLeft(canonicalBoardPacked, 11) ^
        (perspective.toLong << 48) ^
        (baseEquityTrials.toLong << 16) ^
        boardSize.toLong
    )

  // --- Canonical board signature ---
  // Suit canonicalization collapses suit-isomorphic situations into a single
  // canonical form. For example, AsKs on Qs-Jd-3h is strategically identical to
  // AhKh on Qh-Jc-3s. By trying all 24 suit permutations and picking the
  // lexicographically smallest encoding, we ensure these map to the same cache key.

  /** All 24 permutations of 4 suits (0=spades, 1=hearts, 2=diamonds, 3=clubs). */
  private val SuitPermutations: Array[Array[Int]] =
    Array(
      Array(0, 1, 2, 3), Array(0, 1, 3, 2), Array(0, 2, 1, 3), Array(0, 2, 3, 1),
      Array(0, 3, 1, 2), Array(0, 3, 2, 1), Array(1, 0, 2, 3), Array(1, 0, 3, 2),
      Array(1, 2, 0, 3), Array(1, 2, 3, 0), Array(1, 3, 0, 2), Array(1, 3, 2, 0),
      Array(2, 0, 1, 3), Array(2, 0, 3, 1), Array(2, 1, 0, 3), Array(2, 1, 3, 0),
      Array(2, 3, 0, 1), Array(2, 3, 1, 0), Array(3, 0, 1, 2), Array(3, 0, 2, 1),
      Array(3, 1, 0, 2), Array(3, 1, 2, 0), Array(3, 2, 0, 1), Array(3, 2, 1, 0)
    )

  /** Finds the lexicographically smallest (hero, board) encoding across all 24 suit permutations.
    *
    * For each permutation, remaps hero cards and board cards, sorts the board, packs both
    * into Long values, and keeps the permutation that yields the smallest (heroPacked, boardPacked)
    * pair under lexicographic ordering. This is the core of suit canonicalization for caching.
    *
    * @return a CanonicalSuitContext with the minimal encoding and the winning suit map
    */
  private[holdem] def canonicalHeroBoardContext(hand: HoleCards, board: Board): CanonicalSuitContext =
    val boardSize = board.cards.length
    val remappedBoardIds = new Array[Int](boardSize)
    var bestHeroPacked = Long.MaxValue
    var bestBoardPacked = Long.MaxValue
    var bestSuitMap = SuitPermutations(0)
    var permIdx = 0
    while permIdx < SuitPermutations.length do
      val suitMap = SuitPermutations(permIdx)
      val heroFirstId = remapCardId(hand.first, suitMap)
      val heroSecondId = remapCardId(hand.second, suitMap)
      val lowHero = math.min(heroFirstId, heroSecondId)
      val highHero = math.max(heroFirstId, heroSecondId)
      val heroPacked = packRemappedHoleCards(lowHero, highHero)

      var idx = 0
      while idx < boardSize do
        remappedBoardIds(idx) = remapCardId(board.cards(idx), suitMap)
        idx += 1
      java.util.Arrays.sort(remappedBoardIds)
      var boardPacked = boardSize.toLong
      idx = 0
      while idx < boardSize do
        boardPacked = (boardPacked << 6) | remappedBoardIds(idx).toLong
        idx += 1

      if heroPacked < bestHeroPacked || (heroPacked == bestHeroPacked && boardPacked < bestBoardPacked) then
        bestHeroPacked = heroPacked
        bestBoardPacked = boardPacked
        bestSuitMap = suitMap
      permIdx += 1
    CanonicalSuitContext(
      canonicalHeroPacked = bestHeroPacked,
      canonicalBoardPacked = bestBoardPacked,
      suitMap = bestSuitMap
    )

  /** Convenience wrapper: returns just the canonical (heroPacked, boardPacked) pair. */
  private[holdem] def canonicalHeroBoardSignature(hand: HoleCards, board: Board): (Long, Long) =
    val context = canonicalHeroBoardContext(hand, board)
    (context.canonicalHeroPacked, context.canonicalBoardPacked)

  /** Remaps hole cards through a suit permutation and packs into a combinatorial index. */
  private[holdem] def canonicalizeHoleCards(hand: HoleCards, suitMap: Array[Int]): Int =
    val firstId = remapCardId(hand.first, suitMap)
    val secondId = remapCardId(hand.second, suitMap)
    packHoleCardsId(firstId, secondId)

  /** Remaps a card's suit through the permutation and returns a unique card ID (0..51). */
  private def remapCardId(card: Card, suitMap: Array[Int]): Int =
    val mappedSuit = suitMap(card.suit.ordinal)
    (mappedSuit * 13) + card.rank.ordinal

  /** Packs two ordered card IDs into a compact Long for comparison. */
  private def packRemappedHoleCards(lowId: Int, highId: Int): Long =
    ((lowId.toLong << 6) | highId.toLong) & 0xFFFL

  /** Packs two card IDs into a unique combinatorial index (order-independent).
    * Uses the triangular number formula: C(52,2) = 1326 possible hole card combos.
    */
  private def packHoleCardsId(firstId: Int, secondId: Int): Int =
    if firstId < secondId then
      (firstId * (103 - firstId)) / 2 + (secondId - firstId - 1)
    else
      (secondId * (103 - secondId)) / 2 + (firstId - secondId - 1)

  // --- Action hashing ---

  /** Deterministic hash of a candidate action vector for cache key construction.
    * Uses the standard polynomial hash (multiply by 31, add element hash).
    */
  private[holdem] def hashActions(actions: Vector[PokerAction]): Int =
    var hash = 1
    var idx = 0
    while idx < actions.length do
      hash = 31 * hash + hashAction(actions(idx))
      idx += 1
    hash

  /** Maps a single PokerAction to a stable integer for hashing. */
  private def hashAction(action: PokerAction): Int =
    action match
      case PokerAction.Fold => 1
      case PokerAction.Check => 2
      case PokerAction.Call => 3
      case PokerAction.Raise(amount) =>
        31 * 4 + java.lang.Double.hashCode(amount)

  // --- Policy sampling ---

  /** Filters an action probability map to only positive-probability actions, preserving
    * the original candidate ordering. Zero, negative, and non-finite probabilities are dropped.
    */
  private[holdem] def orderedPositiveProbabilities(
      actions: Vector[PokerAction],
      probabilities: Map[PokerAction, Double]
  ): Vector[(PokerAction, Double)] =
    actions.flatMap { action =>
      val probability = probabilities.getOrElse(action, 0.0)
      if probability.isFinite && probability > 0.0 then Some(action -> probability)
      else None
    }

  /** Samples an action from a mixed-strategy policy using the inverse CDF method.
    *
    * Generates a uniform random target in [0, total_probability), then walks through
    * the ordered (action, probability) pairs accumulating mass until the target is reached.
    * Returns the fallback action if the ordered vector is empty or total mass is zero.
    */
  private[holdem] def sampleActionByPolicy(
      ordered: Vector[(PokerAction, Double)],
      fallback: PokerAction,
      rng: Random
  ): PokerAction =
    var total = 0.0
    var i = 0
    while i < ordered.length do
      total += ordered(i)._2
      i += 1
    if total <= 0.0 then fallback
    else
      val target = rng.nextDouble() * total
      var cumulative = 0.0
      var idx = 0
      while idx < ordered.length do
        val (action, probability) = ordered(idx)
        cumulative += probability
        if target <= cumulative then return action
        idx += 1
      ordered.last._1

  // --- Utilities ---

  /** Stafford variant of MurmurHash3 finalizer for 64-bit mixing.
    * Produces a well-distributed hash from an arbitrary Long input.
    * Used to generate deterministic RNG seeds for CFR equity calculations.
    */
  private def mix64(value: Long): Long =
    var z = value + 0x9E3779B97F4A7C15L
    z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L
    z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL
    z ^ (z >>> 31)
