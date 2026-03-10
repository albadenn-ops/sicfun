package sicfun.holdem.equity
import sicfun.holdem.types.*
import sicfun.holdem.gpu.*

import sicfun.core.Card

import java.io.{BufferedOutputStream, DataInputStream, DataOutputStream, FileOutputStream}
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import scala.util.Random

/** Precomputed heads-up equity table using suit-isomorphic canonical keys.
  *
  * Unlike [[HeadsUpEquityTable]] which stores one entry per concrete hand pair,
  * this table exploits suit symmetry to collapse equivalent matchups into a single
  * canonical representative. For example, As Kd vs Qh Jc is equivalent to
  * Ah Ks vs Qd Jc under suit relabeling. This dramatically reduces table size
  * (from ~800K entries to ~170K canonical matchups).
  *
  * Keys are 24-bit integers encoding suit-normalized rank and suit information
  * for both hands (see [[HeadsUpEquityCanonicalTable.encode]]).
  *
  * @param values map from canonical integer key to equity result (stored in canonical hero perspective)
  */
final case class HeadsUpEquityCanonicalTable(values: Map[Int, EquityResultWithError]):
  /** Number of canonical matchups stored in this table. */
  def size: Int = values.size

  /** Returns the equity for `hero` vs `villain`, transparently mapping to the canonical key.
    *
    * @throws NoSuchElementException if the matchup is not in the table
    */
  def equity(hero: HoleCards, villain: HoleCards): EquityResultWithError =
    lookup(hero, villain).getOrElse(throw new NoSuchElementException("matchup not found"))

  /** Optional lookup; returns `None` if the matchup is absent. */
  def get(hero: HoleCards, villain: HoleCards): Option[EquityResultWithError] =
    lookup(hero, villain)

  /** Core lookup with automatic canonical key resolution and perspective flipping. */
  private def lookup(hero: HoleCards, villain: HoleCards): Option[EquityResultWithError] =
    val key = HeadsUpEquityCanonicalTable.keyFor(hero, villain)
    values.get(key.value.raw).map(base => HeadsUpEquityCanonicalTable.flipIfNeeded(base, key.flipped))

/** Companion object providing canonical key computation, batch building, factory methods,
  * and I/O for suit-isomorphic heads-up equity tables.
  */
object HeadsUpEquityCanonicalTable:
  private val CanonicalProfileProperty = "sicfun.canonical.profile"

  /** Type-safe canonical lookup key represented as a 24-bit packed Int. */
  opaque type CanonicalKey = Int

  object CanonicalKey:
    inline def apply(value: Int): CanonicalKey = value

  extension (inline key: CanonicalKey)
    inline def raw: Int = key

  /** A canonical lookup key with a flag indicating whether hero/villain were swapped.
    *
    * @param value   24-bit integer encoding the suit-normalized matchup
    * @param flipped `true` if the original hero maps to the canonical villain (results need swapping)
    */
  final case class Key(value: CanonicalKey, flipped: Boolean)
  /** Internal batch descriptor for parallel computation of canonical matchups. */
  private[holdem] final case class CanonicalBatch(
      keys: Array[Key],
      packedKeys: Array[Long],
      keyMaterial: Array[Long]
  )

  private val DefaultParallelism = math.max(1, Runtime.getRuntime.availableProcessors())

  private lazy val canonicalRepresentatives: Vector[(Key, HoleCards, HoleCards)] =
    val seen = scala.collection.mutable.HashSet.empty[Int]
    val reps = Vector.newBuilder[(Key, HoleCards, HoleCards)]
    HoleCardsIndex.foreachNonOverlappingPair { (_, hero, _, villain) =>
      val key = keyFor(hero, villain)
      if seen.add(key.value.raw) then reps += ((key, hero, villain))
    }
    reps.result()

  /** Computes the canonical key for a hero/villain matchup.
    *
    * Both `canonicalKey(hero, villain)` and `canonicalKey(villain, hero)` are evaluated;
    * the smaller value becomes the canonical representative. The `flipped` flag tracks
    * which orientation was chosen so equity results can be correctly inverted on lookup.
    *
    * @throws IllegalArgumentException if the hands share any cards
    */
  def keyFor(hero: HoleCards, villain: HoleCards): Key =
    if !HoleCardsIndex.areDisjoint(hero, villain) then
      throw new IllegalArgumentException("hands must be non-overlapping")
    val key1 = canonicalKey(hero, villain)
    val key2 = canonicalKey(villain, hero)
    if key1 <= key2 then Key(CanonicalKey(key1), flipped = false)
    else Key(CanonicalKey(key2), flipped = true)

  /** Builds the canonical heads-up equity table, computing equity for one representative
    * per suit-isomorphic equivalence class (up to `maxMatchups`).
    *
    * @param mode         exact enumeration or Monte Carlo with a given trial count
    * @param rng          source of randomness for Monte Carlo seed derivation
    * @param maxMatchups  cap on canonical matchups to compute (default: all)
    * @param progress     optional `(completed, total)` callback for progress reporting
    * @param parallelism  number of CPU worker threads (CPU backend only)
    * @param backend      CPU or GPU computation backend
    * @return a populated [[HeadsUpEquityCanonicalTable]]
    */
  def buildAll(
      mode: HeadsUpEquityTable.Mode,
      rng: Random = new Random(),
      maxMatchups: Long = Long.MaxValue,
      progress: Option[(Long, Long) => Unit] = None,
      parallelism: Int = DefaultParallelism,
      backend: HeadsUpEquityTable.ComputeBackend = HeadsUpEquityTable.ComputeBackend.Cpu
  ): HeadsUpEquityCanonicalTable =
    require(parallelism > 0, "parallelism must be positive")
    val profileEnabled =
      sys.props
        .get(CanonicalProfileProperty)
        .map(_.trim.toLowerCase(Locale.ROOT))
        .exists(v => v == "1" || v == "true" || v == "yes" || v == "on")
    val startedAt = if profileEnabled then System.nanoTime() else 0L
    val batch = selectCanonicalBatch(maxMatchups)
    val afterSelectBatch = if profileEnabled then System.nanoTime() else 0L
    val total = totalCanonicalKeys.toLong
    val monteCarloSeedBase = rng.nextLong()

    val cpuCompute = () =>
      HeadsUpEquityTable.computeBatchCpu(
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
        case HeadsUpEquityTable.ComputeBackend.Cpu =>
          cpuCompute()
        case HeadsUpEquityTable.ComputeBackend.Gpu =>
          HeadsUpGpuRuntime.computeBatch(batch.packedKeys, batch.keyMaterial, mode, monteCarloSeedBase) match
            case Right(gpuResults) =>
              progress.foreach(callback => callback(batch.keys.length.toLong, total))
              gpuResults
            case Left(reason) =>
              if HeadsUpGpuRuntime.allowCpuFallbackOnGpuFailure then
                GpuRuntimeSupport.warn(s"GPU backend unavailable ($reason); using CPU workers")
                cpuCompute()
              else
                throw new IllegalStateException(
                  s"GPU backend failed ($reason). CPU fallback is disabled; set sicfun_GPU_FALLBACK_TO_CPU=true to re-enable it."
                )
    val afterCompute = if profileEnabled then System.nanoTime() else 0L

    val table = buildFromBatchResults(batch.keys, results)
    if profileEnabled then
      val finishedAt = System.nanoTime()
      val selectBatchSeconds = (afterSelectBatch - startedAt).toDouble / 1_000_000_000.0
      val computeSeconds = (afterCompute - afterSelectBatch).toDouble / 1_000_000_000.0
      val mapSeconds = (finishedAt - afterCompute).toDouble / 1_000_000_000.0
      val totalSeconds = (finishedAt - startedAt).toDouble / 1_000_000_000.0
      GpuRuntimeSupport.log(
        f"canonicalProfile selectBatch=$selectBatchSeconds%.3f compute=$computeSeconds%.3f " +
          f"map=$mapSeconds%.3f total=$totalSeconds%.3f"
      )
    table

  def cache(mode: HeadsUpEquityTable.Mode, rng: Random = new Random()): HeadsUpEquityCanonicalCache =
    new HeadsUpEquityCanonicalCache(mode, rng.nextLong())

  def totalCanonicalKeys: Int =
    canonicalRepresentatives.length

  private[holdem] def selectCanonicalBatch(maxMatchups: Long): CanonicalBatch =
    require(maxMatchups > 0L, "maxMatchups must be positive")
    val total = totalCanonicalKeys.toLong
    val limit = math.min(maxMatchups, total)
    require(limit <= Int.MaxValue.toLong, s"max supported matchups is ${Int.MaxValue}")
    val limitInt = limit.toInt
    val reps =
      if limitInt == canonicalRepresentatives.length then canonicalRepresentatives
      else canonicalRepresentatives.take(limitInt)
    val keys = new Array[Key](reps.length)
    val packedKeys = new Array[Long](reps.length)
    val keyMaterial = new Array[Long](reps.length)
    var idx = 0
    while idx < reps.length do
      val (key, hero, villain) = reps(idx)
      keys(idx) = key
      packedKeys(idx) = HeadsUpEquityTable.keyFor(hero, villain).value.raw
      keyMaterial(idx) = key.value.raw.toLong
      idx += 1
    CanonicalBatch(keys = keys, packedKeys = packedKeys, keyMaterial = keyMaterial)

  /** Encodes a single hero-villain orientation into a suit-normalized integer key.
    *
    * Considers all four card-ordering permutations (swapping cards within each hand)
    * and returns the smallest encoding. Suit normalization is achieved by mapping suits
    * to integers in first-occurrence order, making the key invariant to suit relabeling.
    */
  private def canonicalKey(hero: HoleCards, villain: HoleCards): Int =
    val h1 = hero.first
    val h2 = hero.second
    val v1 = villain.first
    val v2 = villain.second
    var best = encode(h1, h2, v1, v2)
    val k2 = encode(h2, h1, v1, v2)
    if k2 < best then best = k2
    val k3 = encode(h1, h2, v2, v1)
    if k3 < best then best = k3
    val k4 = encode(h2, h1, v2, v1)
    if k4 < best then best = k4
    best

  /** Encodes four cards into a 24-bit integer: 4 bits per rank (16 bits total) followed
    * by 2 bits per suit (8 bits total). Suits are remapped to first-occurrence order
    * so that the encoding is invariant under suit permutations.
    */
  private def encode(h1: Card, h2: Card, v1: Card, v2: Card): Int =
    val suitMap = Array.fill(4)(-1)
    var nextSuit = 0
    val ranks = new Array[Int](4)
    val suits = new Array[Int](4)
    val cards = Array(h1, h2, v1, v2)
    var i = 0
    while i < 4 do
      val card = cards(i)
      ranks(i) = card.rank.ordinal
      val s = card.suit.ordinal
      var mapped = suitMap(s)
      if mapped < 0 then
        mapped = nextSuit
        suitMap(s) = mapped
        nextSuit += 1
      suits(i) = mapped
      i += 1
    var key = 0
    i = 0
    while i < 4 do
      key = (key << 4) | (ranks(i) & 0x0f)
      i += 1
    i = 0
    while i < 4 do
      key = (key << 2) | (suits(i) & 0x03)
      i += 1
    key

  private[holdem] def flipIfNeeded(result: EquityResultWithError, flipped: Boolean): EquityResultWithError =
    if flipped then EquityResultWithError(result.loss, result.tie, result.win, result.stderr) else result

  private def buildFromBatchResults(
      keys: Array[Key],
      results: Array[EquityResultWithError]
  ): HeadsUpEquityCanonicalTable =
    require(keys.length == results.length, "keys and results must have equal length")
    val map = scala.collection.mutable.HashMap.empty[Int, EquityResultWithError]
    map.sizeHint(keys.length)
    var idx = 0
    while idx < keys.length do
      map.put(keys(idx).value.raw, flipIfNeeded(results(idx), keys(idx).flipped))
      idx += 1
    HeadsUpEquityCanonicalTable(map.toMap)

/** Thread-safe lazy cache for canonical heads-up equity lookups.
  *
  * Computes equity on first access and caches the result in a [[ConcurrentHashMap]].
  * Uses the same deterministic seed derivation as batch computation for consistency.
  *
  * @param mode                 computation mode (exact or Monte Carlo)
  * @param monteCarloSeedBase   base seed for deterministic per-matchup RNG derivation
  */
final class HeadsUpEquityCanonicalCache(mode: HeadsUpEquityTable.Mode, monteCarloSeedBase: Long):
  private val cache = new ConcurrentHashMap[Int, EquityResultWithError]()

  def size: Int = cache.size()

  def equity(hero: HoleCards, villain: HoleCards): EquityResultWithError =
    val key = HeadsUpEquityCanonicalTable.keyFor(hero, villain)
    val canonical = key.value.raw
    val cached = cache.get(canonical)
    val base =
      if cached != null then cached
      else
        val computed =
          HeadsUpEquityTable.computeEquityDeterministic(
            hero,
            villain,
            mode,
            monteCarloSeedBase,
            canonical.toLong
          )
        val stored = HeadsUpEquityCanonicalTable.flipIfNeeded(computed, key.flipped)
        val prev = cache.putIfAbsent(canonical, stored)
        if prev != null then prev else stored
    HeadsUpEquityCanonicalTable.flipIfNeeded(base, key.flipped)

/** Binary I/O for [[HeadsUpEquityCanonicalTable]] files.
  *
  * Uses the same binary format as [[HeadsUpEquityTableIO]] but with `canonical = true`
  * in the header and 4-byte (Int) keys instead of 8-byte (Long) keys.
  */
object HeadsUpEquityCanonicalTableIO:
  /** Writes a canonical table to a binary file.
    *
    * @param path  output file path
    * @param table the canonical table to serialize
    * @param meta  header metadata (must have `canonical = true`)
    */
  def write(path: String, table: HeadsUpEquityCanonicalTable, meta: HeadsUpEquityTableMeta): Unit =
    require(meta.canonical, "meta.canonical must be true for canonical heads-up table")
    val out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)))
    try
      HeadsUpEquityTableIOUtil.writeHeader(out, meta)
      HeadsUpEquityTableIOUtil.writeEntries(out, table.values, (o, key: Int) => o.writeInt(key))
    finally out.close()

  /** Writes canonical entries directly from aligned key/result arrays, avoiding intermediate map materialization. */
  def writeFromBatch(
      path: String,
      keys: Array[HeadsUpEquityCanonicalTable.Key],
      results: Array[EquityResultWithError],
      meta: HeadsUpEquityTableMeta
  ): Unit =
    require(meta.canonical, "meta.canonical must be true for canonical heads-up table")
    require(keys.length == results.length, "keys and results must have equal length")
    require(meta.count == keys.length, "meta.count must match keys/results length")
    val out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(path)))
    try
      HeadsUpEquityTableIOUtil.writeHeader(out, meta)
      var idx = 0
      while idx < keys.length do
        val key = keys(idx)
        val result = HeadsUpEquityCanonicalTable.flipIfNeeded(results(idx), key.flipped)
        out.writeInt(key.value.raw)
        out.writeDouble(result.win)
        out.writeDouble(result.tie)
        out.writeDouble(result.loss)
        out.writeDouble(result.stderr)
        idx += 1
    finally out.close()

  def read(path: String): HeadsUpEquityCanonicalTable =
    readWithMeta(path)._1

  def readMeta(path: String): HeadsUpEquityTableMeta =
    HeadsUpEquityTableIOUtil.withFileInputStream(path) { in =>
      val meta = HeadsUpEquityTableIOUtil.readHeader(in)
      require(meta.canonical, "expected canonical heads-up table")
      meta
    }

  def readWithMeta(path: String): (HeadsUpEquityCanonicalTable, HeadsUpEquityTableMeta) =
    HeadsUpEquityTableIOUtil.withFileInputStream(path)(readWithMeta)

  def readResource(resourcePath: String): HeadsUpEquityCanonicalTable =
    readResourceWithMeta(resourcePath)._1

  def readResourceMeta(resourcePath: String): HeadsUpEquityTableMeta =
    HeadsUpEquityTableIOUtil.withResourceInputStream(resourcePath, getClass) { in =>
      val meta = HeadsUpEquityTableIOUtil.readHeader(in)
      require(meta.canonical, "expected canonical heads-up table")
      meta
    }

  def readResourceWithMeta(resourcePath: String): (HeadsUpEquityCanonicalTable, HeadsUpEquityTableMeta) =
    HeadsUpEquityTableIOUtil.withResourceInputStream(resourcePath, getClass)(readWithMeta)

  private def readWithMeta(in: DataInputStream): (HeadsUpEquityCanonicalTable, HeadsUpEquityTableMeta) =
    val meta = HeadsUpEquityTableIOUtil.readHeader(in)
    require(meta.canonical, "expected canonical heads-up table")
    val map = HeadsUpEquityTableIOUtil.readEntries(in, meta.count, _.readInt())
    (HeadsUpEquityCanonicalTable(map), meta)
