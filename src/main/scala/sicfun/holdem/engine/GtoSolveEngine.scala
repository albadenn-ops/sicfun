package sicfun.holdem.engine

import sicfun.core.{Card, DiscreteDistribution}
import sicfun.holdem.types.*
import sicfun.holdem.cfr.{HoldemCfrConfig, HoldemCfrSolver}

import scala.collection.mutable
import scala.util.Random

/** Unified GTO decision system with cached CFR solving and fast heuristic path.
  *
  * Extracted from TexasHoldemPlayingHall where gtoHeroResponds and
  * gtoVillainResponds were near-identical (only difference: posterior source).
  * Both villainPosteriorForHeroGto and heroPosteriorForGto just called
  * tableRanges.rangeFor(position), so the unified gtoResponds takes the
  * opponent posterior as a parameter.
  */
private[holdem] object GtoSolveEngine:

  enum GtoMode:
    case Fast
    case Exact

  private[holdem] final case class GtoSolveCacheKey(
      perspective: Int,
      canonicalHeroPacked: Long,
      streetOrdinal: Int,
      canonicalBoardPacked: Long,
      potBits: Long,
      toCallBits: Long,
      stackBits: Long,
      candidateHash: Int,
      baseEquityTrials: Int
  )

  private[holdem] final case class GtoCachedPolicy(
      orderedActionProbabilities: Vector[(PokerAction, Double)],
      bestAction: PokerAction,
      provider: String
  )

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

  private[holdem] val MaxGtoCacheEntries = 500000

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

  private def fastGtoRaiseThreshold(street: Street): Double =
    street match
      case Street.Preflop => 0.78
      case Street.Flop    => 0.74
      case Street.Turn    => 0.71
      case Street.River   => 0.68

  private def fastGtoFoldMargin(street: Street): Double =
    street match
      case Street.Preflop => 0.05
      case Street.Flop    => 0.03
      case Street.Turn    => 0.01
      case Street.River   => -0.01

  private def fastGtoRaiseGap(street: Street): Double =
    street match
      case Street.Preflop => 0.27
      case Street.Flop    => 0.24
      case Street.Turn    => 0.22
      case Street.River   => 0.20

  // --- CFR parametrization ---

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

  private def buildGtoSolveCacheKey(
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
      potBits = java.lang.Double.doubleToLongBits(state.pot),
      toCallBits = java.lang.Double.doubleToLongBits(state.toCall),
      stackBits = java.lang.Double.doubleToLongBits(state.stackSize),
      candidateHash = hashActions(candidates),
      baseEquityTrials = baseEquityTrials
    )

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

  private val SuitPermutations: Array[Array[Int]] =
    Array(
      Array(0, 1, 2, 3), Array(0, 1, 3, 2), Array(0, 2, 1, 3), Array(0, 2, 3, 1),
      Array(0, 3, 1, 2), Array(0, 3, 2, 1), Array(1, 0, 2, 3), Array(1, 0, 3, 2),
      Array(1, 2, 0, 3), Array(1, 2, 3, 0), Array(1, 3, 0, 2), Array(1, 3, 2, 0),
      Array(2, 0, 1, 3), Array(2, 0, 3, 1), Array(2, 1, 0, 3), Array(2, 1, 3, 0),
      Array(2, 3, 0, 1), Array(2, 3, 1, 0), Array(3, 0, 1, 2), Array(3, 0, 2, 1),
      Array(3, 1, 0, 2), Array(3, 1, 2, 0), Array(3, 2, 0, 1), Array(3, 2, 1, 0)
    )

  private[holdem] def canonicalHeroBoardSignature(hand: HoleCards, board: Board): (Long, Long) =
    val boardSize = board.cards.length
    val remappedBoardIds = new Array[Int](boardSize)
    var bestHeroPacked = Long.MaxValue
    var bestBoardPacked = Long.MaxValue
    var permIdx = 0
    while permIdx < SuitPermutations.length do
      val suitMap = SuitPermutations(permIdx)
      val heroFirstId = remapCardId(hand.first, suitMap)
      val heroSecondId = remapCardId(hand.second, suitMap)
      val lowHero = math.min(heroFirstId, heroSecondId)
      val highHero = math.max(heroFirstId, heroSecondId)
      val heroPacked = ((lowHero.toLong << 6) | highHero.toLong) & 0xFFFL

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
      permIdx += 1
    (bestHeroPacked, bestBoardPacked)

  private def remapCardId(card: Card, suitMap: Array[Int]): Int =
    val mappedSuit = suitMap(card.suit.ordinal)
    (mappedSuit * 13) + card.rank.ordinal

  // --- Action hashing ---

  private[holdem] def hashActions(actions: Vector[PokerAction]): Int =
    var hash = 1
    var idx = 0
    while idx < actions.length do
      hash = 31 * hash + hashAction(actions(idx))
      idx += 1
    hash

  private def hashAction(action: PokerAction): Int =
    action match
      case PokerAction.Fold => 1
      case PokerAction.Check => 2
      case PokerAction.Call => 3
      case PokerAction.Raise(amount) =>
        31 * 4 + java.lang.Double.hashCode(amount)

  // --- Policy sampling ---

  private[holdem] def orderedPositiveProbabilities(
      actions: Vector[PokerAction],
      probabilities: Map[PokerAction, Double]
  ): Vector[(PokerAction, Double)] =
    actions.flatMap { action =>
      val probability = probabilities.getOrElse(action, 0.0)
      if probability.isFinite && probability > 0.0 then Some(action -> probability)
      else None
    }

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

  private def mix64(value: Long): Long =
    var z = value + 0x9E3779B97F4A7C15L
    z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L
    z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL
    z ^ (z >>> 31)
