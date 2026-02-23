package sicfun.holdem

import sicfun.core.{Card, Deck, DiscreteDistribution, HandEvaluator}

import scala.util.Random
import scala.annotation.targetName

object HoldemEquity:
  private val deckIndex: Map[Card, Int] = Deck.full.zipWithIndex.toMap
  private val DefaultExactMultiMaxEvaluations: Long = 5_000_000L

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
          if w > 0.0 && !hand.asSet.exists(used.contains) then
            loop(index + 1, used ++ hand.asSet, weight * w, villains :+ hand)
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
    val dead = hero.asSet ++ board.asSet
    val range = sanitizeRange(villainRange, dead)
    val sampler = WeightedSampler(sortedWeights(range.weights))

    var winCount = 0
    var tieCount = 0
    var lossCount = 0

    var mean = 0.0
    var m2 = 0.0
    var i = 0

    while i < trials do
      val villain = sampler.sample(rng)
      val remaining = Deck.full.filterNot(card => dead.contains(card) || villain.contains(card)).toIndexedSeq
      val extra = if board.missing == 0 then Vector.empty[Card] else sampleWithoutReplacement(remaining, board.missing, rng)
      val fullBoard = board.cards ++ extra
      val heroRank = HandEvaluator.evaluate7Cached(hero.toVector ++ fullBoard)
      val villainRank = HandEvaluator.evaluate7Cached(villain.toVector ++ fullBoard)
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
    val sanitized = villainRanges.map(range => sanitizeRange(range, dead))
    val samplers = sanitized.map(range => WeightedSampler(sortedWeights(range.weights))).toVector

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

  private def sanitizeRange(
      range: DiscreteDistribution[HoleCards],
      dead: Set[Card]
  ): DiscreteDistribution[HoleCards] =
    val collapsed = range.weights.foldLeft(Map.empty[HoleCards, Double]) {
      case (acc, (hand, weight)) =>
        if weight <= 0.0 then acc
        else if hand.asSet.exists(dead.contains) then acc
        else
          val canonical = HoleCards.canonical(hand.first, hand.second)
          acc.updated(canonical, acc.getOrElse(canonical, 0.0) + weight)
    }
    require(collapsed.nonEmpty, "villain range is empty after filtering")
    DiscreteDistribution(collapsed).normalized

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
    require(samplers.nonEmpty, "samplers must be non-empty")
    var attempt = 0
    while attempt < maxAttempts do
      var used = dead
      val hands = new Array[HoleCards](samplers.length)
      var ok = true
      var i = 0
      while i < samplers.length && ok do
        val hand = samplers(i).sample(rng)
        if hand.asSet.exists(used.contains) then ok = false
        else
          hands(i) = hand
          used = used ++ hand.asSet
          i += 1
      if ok then return hands.toVector
      attempt += 1
    throw new IllegalArgumentException("unable to sample non-overlapping villain hands; ranges may be too restrictive")

  private def sortedWeights(weights: Map[HoleCards, Double]): Vector[(HoleCards, Double)] =
    weights.toVector.sortBy { case (hand, _) => handKey(hand) }

  private def handKey(hand: HoleCards): (Int, Int) =
    val a = deckIndex(hand.first)
    val b = deckIndex(hand.second)
    if a <= b then (a, b) else (b, a)

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
    buffer.take(k).toVector

  private final case class WeightedSampler[A](items: Vector[(A, Double)]):
    require(items.nonEmpty, "cannot sample from empty range")
    private val values: Vector[A] = items.map(_._1)
    private val cumulative: Array[Double] =
      val arr = new Array[Double](items.length)
      var acc = 0.0
      var i = 0
      while i < items.length do
        acc += items(i)._2
        arr(i) = acc
        i += 1
      arr
    private val total: Double = cumulative.last

    def sample(rng: Random): A =
      val r = rng.nextDouble() * total
      var low = 0
      var high = cumulative.length - 1
      while low < high do
        val mid = (low + high) >>> 1
        if r <= cumulative(mid) then high = mid else low = mid + 1
      values(low)
