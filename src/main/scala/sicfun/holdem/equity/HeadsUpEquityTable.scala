package sicfun.holdem.equity
import sicfun.holdem.types.*
import sicfun.holdem.gpu.*

import sicfun.core.{Deck, DiscreteDistribution}

import java.io.{BufferedOutputStream, DataInputStream, DataOutputStream, FileOutputStream}
import java.util.Arrays
import java.util.Locale
import java.util.concurrent.{ConcurrentHashMap, Executors}
import java.util.concurrent.atomic.{AtomicInteger, AtomicLong}
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.duration.Duration
import scala.util.Random

/** Bijective mapping between all 1326 canonical hole-card combinations and dense integer IDs.
  *
  * Each unique unordered two-card hand drawn from a standard 52-card deck is assigned
  * a stable integer index in `[0, 1326)`. This index space is used as the key domain for
  * the precomputed heads-up equity tables.
  */
object HoleCardsIndex:
  /** All 1326 canonical hole-card hands, ordered by the deck-based enumeration. */
  private val allHands: Vector[HoleCards] =
    HoldemCombinator.holeCardsFrom(Deck.full)

  /** Reverse lookup: canonical HoleCards to its integer ID. */
  private val idByHand: Map[HoleCards, Int] = allHands.zipWithIndex.toMap

  /** O(1) card-pair to hole-card-ID lookup table. Indexed by `card1.id * 52 + card2.id`. */
  private val idByCardPair: Array[Int] =
    val arr = new Array[Int](52 * 52)
    java.util.Arrays.fill(arr, -1)
    var id = 0
    while id < allHands.length do
      val hand = allHands(id)
      val c1 = hand.first.id
      val c2 = hand.second.id
      arr(c1 * 52 + c2) = id
      arr(c2 * 52 + c1) = id
      id += 1
    arr

  /** Precomputed parallel arrays of hero/villain IDs for all disjoint (non-overlapping) pairs. */
  private lazy val disjointPairs: (Array[Int], Array[Int]) =
    val heroBuilder = new scala.collection.mutable.ArrayBuilder.ofInt()
    val villainBuilder = new scala.collection.mutable.ArrayBuilder.ofInt()
    var i = 0
    while i < size do
      val hero = byIdUnchecked(i)
      var j = i + 1
      while j < size do
        val villain = byIdUnchecked(j)
        if hero.isDisjointFrom(villain) then
          heroBuilder += i
          villainBuilder += j
        j += 1
      i += 1
    (heroBuilder.result(), villainBuilder.result())

  /** Total number of canonical two-card hands (C(52,2) = 1326). */
  val size: Int = allHands.length

  /** Number of non-overlapping hand pairs (precomputed). */
  lazy val disjointPairCount: Int = disjointPairs._1.length

  /** Returns all canonical hole-card hands in index order. */
  def all: Vector[HoleCards] = allHands

  /** Returns the integer ID for the given hole cards (canonicalized internally).
    *
    * @throws IllegalArgumentException if the hand is not recognized
    */
  def idOf(hand: HoleCards): Int =
    idByHand.getOrElse(HoleCards.canonical(hand.first, hand.second), throw new IllegalArgumentException("unknown hand"))

  /** O(1) hole-card ID lookup from precomputed card-pair table. No Map overhead. */
  inline def fastIdOf(hand: HoleCards): Int =
    idByCardPair(hand.first.id * 52 + hand.second.id)

  /** Retrieves the hole cards at the given index. Package-private for internal table construction. */
  private[holdem] def byId(id: Int): HoleCards =
    require(id >= 0 && id < allHands.length, s"invalid hole cards id: $id")
    allHands(id)

  /** Fast path for already-validated IDs used in tight inner loops. */
  private[holdem] inline def byIdUnchecked(id: Int): HoleCards =
    allHands(id)

  /** Returns `true` if the two hands share no cards. */
  inline def areDisjoint(hero: HoleCards, villain: HoleCards): Boolean =
    hero.isDisjointFrom(villain)

  /** SAM type for unboxed iteration over integer pairs. */
  trait IntPairConsumer:
    def accept(a: Int, b: Int): Unit

  /** Iterates all precomputed disjoint pairs by ID only. No boxing, no HoleCards allocation. */
  def foreachDisjointPairById(f: IntPairConsumer): Unit =
    val (heroIds, villainIds) = disjointPairs
    var k = 0
    while k < heroIds.length do
      f.accept(heroIds(k), villainIds(k))
      k += 1

  /** Iterates over all ordered (i < j) non-overlapping hand pairs, invoking `f` until it returns `false`.
    *
    * The callback receives `(heroId, heroHand, villainId, villainHand)`. Early termination
    * occurs when `f` returns `false`.
    */
  def foreachNonOverlappingPairWhile(f: (Int, HoleCards, Int, HoleCards) => Boolean): Unit =
    var i = 0
    var keepGoing = true
    while i < size && keepGoing do
      val hero = byIdUnchecked(i)
      var j = i + 1
      while j < size && keepGoing do
        val villain = byIdUnchecked(j)
        if areDisjoint(hero, villain) then
          keepGoing = f(i, hero, j, villain)
        j += 1
      i += 1

  /** Iterates over all ordered (i < j) non-overlapping hand pairs. */
  def foreachNonOverlappingPair(f: (Int, HoleCards, Int, HoleCards) => Unit): Unit =
    foreachNonOverlappingPairWhile { (i, hero, j, villain) =>
      f(i, hero, j, villain)
      true
    }

  /** Counts the total number of non-overlapping hand pairs. */
  def countNonOverlappingPairs: Long = disjointPairCount.toLong


/** Precomputed heads-up equity lookup table mapping all non-overlapping hole-card pairs to
  * their equity results.
  *
  * Keys are packed Long values encoding an ordered `(lowId, highId)` pair using 11-bit fields
  * (see [[HeadsUpEquityTable.pack]]). The table always stores results from the perspective of
  * the lower-ID hand; when the caller's hero has the higher ID, the result is "flipped"
  * (win/loss swapped) transparently.
  *
  * @param values Map from packed key to the equity result stored in low-ID-hero perspective
  */
final case class HeadsUpEquityTable(values: Map[Long, EquityResultWithError]):
  /** Number of matchups stored in this table. */
  def size: Int = values.size

  /** Returns the equity for `hero` vs `villain`, flipping perspective as needed.
    *
    * @throws NoSuchElementException if the matchup is not in the table
    */
  def equity(hero: HoleCards, villain: HoleCards): EquityResultWithError =
    lookup(hero, villain).getOrElse(throw new NoSuchElementException("matchup not found"))

  /** Optional lookup; returns `None` if the matchup is absent. */
  def get(hero: HoleCards, villain: HoleCards): Option[EquityResultWithError] =
    lookup(hero, villain)

  /** Core lookup: computes the normalized key, fetches the stored result, and flips if the
    * caller's hero was the higher-ID hand (the "flipped key" concept).
    */
  private def lookup(hero: HoleCards, villain: HoleCards): Option[EquityResultWithError] =
    val key = HeadsUpEquityTable.keyFor(hero, villain)
    values.get(key.value.raw).map(base => HeadsUpEquityTable.flipIfNeeded(base, key.flipped))

/** Companion object providing factory methods, key encoding, batch computation, and I/O
  * for full (non-canonical) heads-up equity tables.
  */
object HeadsUpEquityTable:
  /** Number of bits used per hole-card ID in the packed key representation. 11 bits can
    * address up to 2048 values, comfortably covering the 1326 canonical hands.
    */
  private val IdBits = 11
  /** Bitmask for extracting the lower (highId) field from a packed key. */
  private val IdMask = (1 << IdBits) - 1

  /** Type-safe packed matchup key `(lowId, highId)` represented as a 64-bit value. */
  opaque type PackedKey = Long

  object PackedKey:
    inline def apply(lowId: Int, highId: Int): PackedKey =
      (lowId.toLong << IdBits) | (highId.toLong & IdMask)

    inline def fromLong(value: Long): PackedKey = value

  extension (inline key: PackedKey)
    inline def raw: Long = key
    inline def lowId: Int = (key >>> IdBits).toInt
    inline def highId: Int = (key & IdMask).toInt

  /** A normalized lookup key with a flag indicating whether hero/villain were swapped.
    *
    * The "flipped" flag enables hero/villain ordering normalization: the table always
    * stores results from the lower-ID hand's perspective, and `flipped = true` signals
    * that the caller's hero had the higher ID, so win/loss must be swapped on retrieval.
    *
    * @param value   packed Long key with lowId in upper bits, highId in lower bits
    * @param flipped `true` if the original hero had the higher ID and results need swapping
    */
  final case class Key(value: PackedKey, flipped: Boolean)

  /** Internal batch descriptor carrying parallel arrays of packed keys and key material
    * (used for deterministic Monte Carlo seed derivation).
    */
  private[holdem] final case class FullBatch(packedKeys: Array[Long], keyMaterial: Array[Long])

  /** Equity computation strategy. */
  enum Mode:
    /** Exhaustive enumeration of all remaining board run-outs. */
    case Exact
    /** Monte Carlo sampling with the specified number of random trials. */
    case MonteCarlo(trials: Int)

  /** Selects CPU or GPU computation backend. */
  enum ComputeBackend:
    case Cpu
    case Gpu

  /** Parses a backend name from a CLI string. */
  object ComputeBackend:
    def parse(raw: String): ComputeBackend =
      raw.trim.toLowerCase(Locale.ROOT) match
        case "cpu" => ComputeBackend.Cpu
        case "gpu" => ComputeBackend.Gpu
        case other => throw new IllegalArgumentException(s"unknown backend: $other (expected cpu or gpu)")

  private val DefaultParallelism = math.max(1, Runtime.getRuntime.availableProcessors())
  /** Minimum work items before switching from sequential to parallel execution. */
  private val ParallelMinWorkItems = 1_000
  /** Work-stealing granularity factor.  Each worker gets `totalItems / (workers * ChunkDivisor)` items
   * per atomic grab. A higher divisor yields finer chunks, improving load balance at the cost of
   * slightly more atomic contention. 16 is a good empirical trade-off.
   */
  private val ChunkDivisor = 16

  /** Lazily computed array of packed keys for every non-overlapping pair, in enumeration order. */
  private lazy val nonOverlappingPairKeys: Array[Long] =
    val count = HoleCardsIndex.disjointPairCount
    val arr = new Array[Long](count)
    var idx = 0
    HoleCardsIndex.foreachDisjointPairById { (i, j) =>
      arr(idx) = pack(i, j)
      idx += 1
    }
    arr

  /** Selects up to `maxMatchups` non-overlapping pairs and bundles them into a [[FullBatch]]
    * ready for CPU or GPU computation. The `keyMaterial` array mirrors the packed keys and is
    * used to derive deterministic per-matchup Monte Carlo seeds.
    */
  private[holdem] def selectFullBatch(maxMatchups: Long): FullBatch =
    require(maxMatchups > 0L, "maxMatchups must be positive")
    val total = totalMatchups
    val limit = math.min(maxMatchups, total)
    require(limit <= Int.MaxValue.toLong, s"max supported matchups is ${Int.MaxValue}")
    val limitInt = limit.toInt
    val pairKeys =
      if limitInt == nonOverlappingPairKeys.length then nonOverlappingPairKeys
      else Arrays.copyOf(nonOverlappingPairKeys, limitInt)
    val keyMaterial = new Array[Long](pairKeys.length)
    var idx = 0
    while idx < pairKeys.length do
      keyMaterial(idx) = pairKeys(idx)
      idx += 1
    FullBatch(packedKeys = pairKeys, keyMaterial = keyMaterial)

  /** Computes a normalized [[Key]] for the given hero/villain pair.
    *
    * The key always encodes `(lowId, highId)` where `lowId < highId`. If the hero's ID is
    * greater than the villain's, the key is flagged as `flipped = true` so that the
    * stored equity result (from the low-ID perspective) can be inverted at query time.
    *
    * @throws IllegalArgumentException if the hands overlap or are identical
    */
  def keyFor(hero: HoleCards, villain: HoleCards): Key =
    if !HoleCardsIndex.areDisjoint(hero, villain) then
      throw new IllegalArgumentException("hands must be non-overlapping")
    val heroId = HoleCardsIndex.idOf(hero)
    val villainId = HoleCardsIndex.idOf(villain)
    if heroId == villainId then throw new IllegalArgumentException("hands must be distinct")
    // Always store with the lower ID first; flag when hero/villain are swapped.
    if heroId < villainId then Key(PackedKey(heroId, villainId), flipped = false)
    else Key(PackedKey(villainId, heroId), flipped = true)

  /** Builds the full heads-up equity table, computing equity for every non-overlapping pair
    * (up to `maxMatchups`) using the specified mode and backend.
    *
    * GPU failures are fail-fast by default. CPU fallback is available only when explicitly
    * enabled via `sicfun_GPU_FALLBACK_TO_CPU=true` (or `-Dsicfun.gpu.fallbackToCpu=true`).
    *
    * @param mode         exact enumeration or Monte Carlo with a given trial count
    * @param rng          source of randomness; a seed base is drawn once for deterministic replay
    * @param maxMatchups  cap on the number of matchups to compute (default: all)
    * @param progress     optional `(completed, total)` callback invoked periodically
    * @param parallelism  number of CPU worker threads (only used for CPU backend)
    * @param backend      CPU or GPU computation backend
    * @return a populated [[HeadsUpEquityTable]]
    */
  def buildAll(
      mode: Mode,
      rng: Random = new Random(),
      maxMatchups: Long = Long.MaxValue,
      progress: Option[(Long, Long) => Unit] = None,
      parallelism: Int = DefaultParallelism,
      backend: ComputeBackend = ComputeBackend.Cpu
  ): HeadsUpEquityTable =
    require(parallelism > 0, "parallelism must be positive")
    val batch = selectFullBatch(maxMatchups)
    val total = totalMatchups
    val monteCarloSeedBase = rng.nextLong()
    val cpuCompute = () =>
      computeBatchCpu(
        mode = mode,
        packedKeys = batch.packedKeys,
        keyMaterial = batch.keyMaterial,
        parallelism = parallelism,
        monteCarloSeedBase = monteCarloSeedBase,
        progress = progress,
        totalForProgress = total
      )

    val results =
      backend match
        case ComputeBackend.Cpu =>
          cpuCompute()
        case ComputeBackend.Gpu =>
          HeadsUpGpuRuntime.computeBatch(batch.packedKeys, batch.keyMaterial, mode, monteCarloSeedBase) match
            case Right(gpuResults) =>
              progress.foreach(callback => callback(batch.packedKeys.length.toLong, total))
              gpuResults
            case Left(reason) =>
              if HeadsUpGpuRuntime.allowCpuFallbackOnGpuFailure then
                GpuRuntimeSupport.warn(s"GPU backend unavailable ($reason); using CPU workers")
                cpuCompute()
              else
                throw new IllegalStateException(
                  s"GPU backend failed ($reason). CPU fallback is disabled; set sicfun_GPU_FALLBACK_TO_CPU=true to re-enable it."
                )

    buildFromBatchResults(batch.packedKeys, results)

  def cache(mode: Mode, rng: Random = new Random()): HeadsUpEquityCache =
    new HeadsUpEquityCache(mode, rng.nextLong())

  def totalMatchups: Long =
    nonOverlappingPairKeys.length.toLong

  private[holdem] def computeEquityDeterministic(
      hero: HoleCards,
      villain: HoleCards,
      mode: Mode,
      monteCarloSeedBase: Long,
      keyMaterial: Long
  ): EquityResultWithError =
    mode match
      case Mode.Exact => computeExactAgainstFixedVillain(hero, villain)
      case Mode.MonteCarlo(trials) =>
        val localSeed = monteCarloSeed(monteCarloSeedBase, keyMaterial)
        computeMonteCarloAgainstFixedVillain(hero, villain, trials, new Random(localSeed))

  private[holdem] inline def pack(lowId: Int, highId: Int): Long =
    PackedKey(lowId, highId).raw

  private[holdem] inline def unpackLowId(packedKey: Long): Int =
    PackedKey.fromLong(packedKey).lowId

  private[holdem] inline def unpackHighId(packedKey: Long): Int =
    PackedKey.fromLong(packedKey).highId

  private[holdem] inline def flipIfNeeded(result: EquityResultWithError, flipped: Boolean): EquityResultWithError =
    if flipped then EquityResultWithError(result.loss, result.tie, result.win, result.stderr) else result

  private[holdem] inline def monteCarloSeed(monteCarloSeedBase: Long, keyMaterial: Long): Long =
    mix64(monteCarloSeedBase ^ keyMaterial)

  private[holdem] def computeBatchCpu(
      mode: Mode,
      packedKeys: Array[Long],
      keyMaterial: Array[Long],
      parallelism: Int,
      monteCarloSeedBase: Long,
      progress: Option[(Long, Long) => Unit] = None,
      totalForProgress: Long = 0L
  ): Array[EquityResultWithError] =
    require(parallelism > 0, "parallelism must be positive")
    require(packedKeys.length == keyMaterial.length, "packedKeys and keyMaterial must have equal length")
    val results = new Array[EquityResultWithError](packedKeys.length)
    val progressTotal = if totalForProgress > 0L then totalForProgress else packedKeys.length.toLong
    if shouldRunParallel(packedKeys.length, parallelism) then
      val workers = math.min(parallelism, packedKeys.length)
      val chunkSize = math.max(1, packedKeys.length / (workers * ChunkDivisor))
      val nextStart = new AtomicInteger(0)
      val done = new AtomicLong(0L)
      withExecutionContext(workers) {
        val tasks = (0 until workers).map { _ =>
          Future {
            var start = nextStart.getAndAdd(chunkSize)
            while start < packedKeys.length do
              val end = math.min(start + chunkSize, packedKeys.length)
              var idx = start
              while idx < end do
                val packed = packedKeys(idx)
                val hero = HoleCardsIndex.byId(unpackLowId(packed))
                val villain = HoleCardsIndex.byId(unpackHighId(packed))
                results(idx) =
                  computeEquityDeterministic(
                    hero = hero,
                    villain = villain,
                    mode = mode,
                    monteCarloSeedBase = monteCarloSeedBase,
                    keyMaterial = keyMaterial(idx)
                  )
                idx += 1
              val completed = done.addAndGet((end - start).toLong)
              progress.foreach(callback => callback(completed, progressTotal))
              start = nextStart.getAndAdd(chunkSize)
          }
        }
        Await.result(Future.sequence(tasks), Duration.Inf)
      }
    else
      var idx = 0
      while idx < packedKeys.length do
        val packed = packedKeys(idx)
        val hero = HoleCardsIndex.byId(unpackLowId(packed))
        val villain = HoleCardsIndex.byId(unpackHighId(packed))
        results(idx) =
          computeEquityDeterministic(
            hero = hero,
            villain = villain,
            mode = mode,
            monteCarloSeedBase = monteCarloSeedBase,
            keyMaterial = keyMaterial(idx)
          )
        val done = idx + 1L
        progress.foreach(callback => callback(done, progressTotal))
        idx += 1
    results

  private def withExecutionContext[A](workers: Int)(body: ExecutionContext ?=> A): A =
    val executor = Executors.newFixedThreadPool(workers)
    given ExecutionContext = ExecutionContext.fromExecutorService(executor)
    try body
    finally executor.shutdown()

  private def shouldRunParallel(workSize: Int, parallelism: Int): Boolean =
    parallelism > 1 && workSize >= ParallelMinWorkItems

  private def buildFromBatchResults(
      packedKeys: Array[Long],
      results: Array[EquityResultWithError]
  ): HeadsUpEquityTable =
    require(packedKeys.length == results.length, "packedKeys and results must have equal length")
    val map = scala.collection.mutable.HashMap.empty[Long, EquityResultWithError]
    map.sizeHint(packedKeys.length)
    var idx = 0
    while idx < packedKeys.length do
      map.put(packedKeys(idx), results(idx))
      idx += 1
    HeadsUpEquityTable(map.toMap)

  private def computeExactAgainstFixedVillain(hero: HoleCards, villain: HoleCards): EquityResultWithError =
    val dist = DiscreteDistribution(Map(villain -> 1.0))
    val result = HoldemEquity.equityExact(hero, Board.empty, dist)
    EquityResultWithError(result.win, result.tie, result.loss, 0.0)

  private def computeMonteCarloAgainstFixedVillain(
      hero: HoleCards,
      villain: HoleCards,
      trials: Int,
      rng: Random
  ): EquityResultWithError =
    val dist = DiscreteDistribution(Map(villain -> 1.0))
    val estimate = HoldemEquity.equityMonteCarlo(hero, Board.empty, dist, trials, rng)
    EquityResultWithError(estimate.winRate, estimate.tieRate, estimate.lossRate, estimate.stderr)

  /** SplitMix64 hash function for deriving deterministic per-matchup Monte Carlo seeds.
    * Takes a combined key (seedBase XOR keyMaterial) and produces a well-distributed 64-bit hash.
    * This ensures each matchup gets a unique, reproducible random sequence regardless of
    * computation order or parallelism level.
    *
    * Constants are from the SplitMix64 PRNG by Sebastiano Vigna.
    */
  private inline def mix64(value: Long): Long =
    var z = value + 0x9E3779B97F4A7C15L   // golden ratio constant
    z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L
    z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL
    z ^ (z >>> 31)

/**
  * Thread-safe lazy cache for full (non-canonical) heads-up equity lookups.
  *
  * Computes equity on first access for a given hero/villain pair and caches the result
  * in a [[ConcurrentHashMap]] keyed by packed Long. Subsequent lookups for the same
  * pair (in either order) return the cached result with appropriate perspective flipping.
  *
  * @param mode                computation mode (exact or Monte Carlo)
  * @param monteCarloSeedBase  base seed for deterministic per-matchup RNG derivation
  */
final class HeadsUpEquityCache(mode: HeadsUpEquityTable.Mode, monteCarloSeedBase: Long):
  private val cache = new ConcurrentHashMap[Long, EquityResultWithError]()

  def size: Int = cache.size()

  def equity(hero: HoleCards, villain: HoleCards): EquityResultWithError =
    val key = HeadsUpEquityTable.keyFor(hero, villain)
    val packed = key.value.raw
    val cached = cache.get(packed)
    val base =
      if cached != null then cached
      else
        val heroCanonical = HoleCardsIndex.byId(HeadsUpEquityTable.unpackLowId(packed))
        val villainCanonical = HoleCardsIndex.byId(HeadsUpEquityTable.unpackHighId(packed))
        val computed =
          HeadsUpEquityTable.computeEquityDeterministic(
            heroCanonical,
            villainCanonical,
            mode,
            monteCarloSeedBase,
            packed
          )
        val prev = cache.putIfAbsent(packed, computed)
        if prev != null then prev else computed
    HeadsUpEquityTable.flipIfNeeded(base, key.flipped)

/**
  * Binary I/O for [[HeadsUpEquityTable]] (full, non-canonical) files.
  *
  * Uses the shared binary format defined in [[HeadsUpEquityTableFormat]] with 8-byte (Long)
  * packed keys. Each entry is: key (Long) + win (Double) + tie (Double) + loss (Double) + stderr (Double).
  *
  * @see [[HeadsUpEquityCanonicalTableIO]] for the canonical table variant (4-byte Int keys)
  * @see [[HeadsUpEquityTableIOUtil]] for the shared header/entry read/write routines
  */
object HeadsUpEquityTableIO:
  /** Writes a full heads-up equity table to a binary file.
    *
    * @param path  output file path
    * @param table the table to serialize
    * @param meta  header metadata (must have `canonical = false`)
    */
  def write(path: String, table: HeadsUpEquityTable, meta: HeadsUpEquityTableMeta): Unit =
    require(!meta.canonical, "meta.canonical must be false for full heads-up table")
    val out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)))
    try
      HeadsUpEquityTableIOUtil.writeHeader(out, meta)
      HeadsUpEquityTableIOUtil.writeEntries(out, table.values, (o, key: Long) => o.writeLong(key))
    finally out.close()

  def read(path: String): HeadsUpEquityTable =
    readWithMeta(path)._1

  def readMeta(path: String): HeadsUpEquityTableMeta =
    HeadsUpEquityTableIOUtil.withFileInputStream(path) { in =>
      val meta = HeadsUpEquityTableIOUtil.readHeader(in)
      require(!meta.canonical, "expected non-canonical heads-up table")
      meta
    }

  def readWithMeta(path: String): (HeadsUpEquityTable, HeadsUpEquityTableMeta) =
    HeadsUpEquityTableIOUtil.withFileInputStream(path)(readWithMeta)

  def readResource(resourcePath: String): HeadsUpEquityTable =
    readResourceWithMeta(resourcePath)._1

  def readResourceMeta(resourcePath: String): HeadsUpEquityTableMeta =
    HeadsUpEquityTableIOUtil.withResourceInputStream(resourcePath, getClass) { in =>
      val meta = HeadsUpEquityTableIOUtil.readHeader(in)
      require(!meta.canonical, "expected non-canonical heads-up table")
      meta
    }

  def readResourceWithMeta(resourcePath: String): (HeadsUpEquityTable, HeadsUpEquityTableMeta) =
    HeadsUpEquityTableIOUtil.withResourceInputStream(resourcePath, getClass)(readWithMeta)

  private def readWithMeta(in: DataInputStream): (HeadsUpEquityTable, HeadsUpEquityTableMeta) =
    val meta = HeadsUpEquityTableIOUtil.readHeader(in)
    require(!meta.canonical, "expected non-canonical heads-up table")
    val map = HeadsUpEquityTableIOUtil.readEntries(in, meta.count, _.readLong())
    (HeadsUpEquityTable(map), meta)
