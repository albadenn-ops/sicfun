package sicfun.holdem.equity
import sicfun.holdem.types.*
import sicfun.holdem.gpu.*

import sicfun.core.{Card, Deck, DiscreteDistribution, HandEvaluator, Prob}
import Prob.*

import scala.util.Random
import scala.annotation.targetName
import scala.collection.mutable

object HoldemEquity:
  private val DefaultExactMultiMaxEvaluations: Long = 5_000_000L
  private val PreflopEquityBackendProperty = "sicfun.holdem.preflopEquityBackend"
  private val PreflopEquityBackendEnv = "sicfun_HOLDEM_PREFLOP_EQUITY_BACKEND"
  private val GpuProviderProperty = "sicfun.gpu.provider"
  private val GpuProviderEnv = "sicfun_GPU_PROVIDER"
  private val PreflopEquityBackendAuto = "auto"
  private val PreflopEquityBackendCpu = "cpu"
  private val PreflopEquityBackendRange = "range"
  private val PreflopEquityBackendBatch = "batch"
  // Prevent recursive acceleration when GPU providers call back into HoldemEquity.
  private val accelerationGuard = new ThreadLocal[java.lang.Boolean]:
    override def initialValue(): java.lang.Boolean = java.lang.Boolean.FALSE
  private final case class PreparedRange(
      hands: Array[HoleCards],
      weights: Array[Double],
      handIds: Array[Int]
  )
  private val preparedRangeWeightScratch = new ThreadLocal[Array[Double]]:
    override def initialValue(): Array[Double] = new Array[Double](HoleCardsIndex.size)
  private val preparedRangeTouchedIdsScratch = new ThreadLocal[Array[Int]]:
    override def initialValue(): Array[Int] = new Array[Int](HoleCardsIndex.size)

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

  /** Fixed-point variant of equityExact. Uses Int32 weights with Long accumulation
    * for deterministic arithmetic and better cache utilization.
    *
    * Integer division of weight.raw by boardCount introduces truncation error of at most
    * 1 LSB per board evaluation (~4e-8 relative error), negligible for poker equity needs.
    */
  def equityExactProb(
      hero: HoleCards,
      board: Board,
      villainRange: DiscreteDistribution[HoleCards]
  ): EquityResult =
    validateHeroBoard(hero, board)
    val dead = hero.asSet ++ board.asSet
    val range = sanitizeRange(villainRange, dead)
    val missing = board.missing

    // Accumulate as Long. Max: 1326 × 1081 × (2^30/1) ≈ 1.5e15, Long max 9.2e18.
    var winL = 0L
    var tieL = 0L
    var lossL = 0L

    range.weights.foreach { case (villain, weightD) =>
      if weightD > 0.0 then
        val weight = Prob.fromDouble(weightD)
        val remaining = Deck.full.filterNot(card => dead.contains(card) || villain.contains(card)).toIndexedSeq
        val boardCount = combinationsCount(remaining.length, missing)
        val perBoardWeight = weight.raw.toLong / boardCount

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
    }

    val total = winL + tieL + lossL
    if total == 0L then EquityResult(0.0, 0.0, 0.0)
    else
      val invTotal = 1.0 / total.toDouble
      EquityResult(winL * invTotal, tieL * invTotal, lossL * invTotal)

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
            val remaining = deckMinusDead.filterNot(card => card == villain.first || card == villain.second)
            val extra = sampleWithoutReplacement(remaining, boardMissing, rng)
            // Fill sampled cards into reusable board array.
            var k = boardCards.length
            var extraIdx = 0
            while k < 5 do
              boardArr(k) = extra(extraIdx)
              extraIdx += 1
              k += 1
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

  def evCall(potBeforeCall: Double, callSize: Double, equity: Double): Double =
    require(potBeforeCall >= 0.0, "potBeforeCall must be non-negative")
    require(callSize >= 0.0, "callSize must be non-negative")
    require(equity >= 0.0 && equity <= 1.0, "equity must be between 0 and 1")
    equity * (potBeforeCall + callSize) - callSize

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

  private def safeMultiply(a: Long, b: Long, limit: Long): Long =
    if a == 0L || b == 0L then 0L
    else if a > limit / b then limit + 1L
    else a * b

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

  private def allHoleCardsExcluding(dead: Set[Card]): Vector[HoleCards] =
    val remaining = Deck.full.filterNot(dead.contains).toIndexedSeq
    HoldemCombinator.holeCardsFrom(remaining)

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

  private object WeightedSampler:
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

  private final class WeightedSampler[A] private (
      values: Array[Any],
      cumulative: Array[Double]
  ):
    private val total: Double = cumulative.last

    def sample(rng: Random): A =
      val r = rng.nextDouble() * total
      var low = 0
      var high = cumulative.length - 1
      while low < high do
        val mid = (low + high) >>> 1
        if r <= cumulative(mid) then high = mid else low = mid + 1
      values(low).asInstanceOf[A]
