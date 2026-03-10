package sicfun.holdem.equity
import sicfun.holdem.types.*

import sicfun.core.Card

/** Combinatorial utilities for enumerating card subsets used by the equity engine. */
object HoldemCombinator:
  /** Lazily generates all k-element subsets of `items` in lexicographic order.
    *
    * Produces C(n, k) combinations without materializing them all in memory.
    * Each subset is returned as a `Vector[A]` preserving the original item order.
    *
    * @tparam A the element type
    * @param items the source collection (indexed for efficient random access)
    * @param k     subset size; must satisfy `0 <= k <= items.length`
    * @return a lazy iterator over all k-subsets
    */
  def combinations[A](items: IndexedSeq[A], k: Int): Iterator[Vector[A]] =
    require(k >= 0 && k <= items.length)
    // Recursive descent: pick one element, then recurse for the remaining slots.
    def loop(start: Int, kLeft: Int, acc: Vector[A]): Iterator[Vector[A]] =
      if kLeft == 0 then Iterator.single(acc)
      else
        val maxStart = items.length - kLeft // upper bound to ensure enough elements remain
        (start to maxStart).iterator.flatMap { i =>
          loop(i + 1, kLeft - 1, acc :+ items(i))
        }
    loop(0, k, Vector.empty)

  /** Generates all canonical [[HoleCards]] from a set of remaining deck cards.
    *
    * This is equivalent to `combinations(remaining, 2)` with each pair wrapped
    * in [[HoleCards.canonical]] to ensure deterministic ordering.
    *
    * @param remaining deck cards not yet dealt (hero, board, or other dead cards excluded)
    * @return all possible two-card hands in canonical order
    */
  def holeCardsFrom(remaining: IndexedSeq[Card]): Vector[HoleCards] =
    val out = Vector.newBuilder[HoleCards]
    var i = 0
    while i < remaining.length do
      val first = remaining(i)
      var j = i + 1
      while j < remaining.length do
        out += HoleCards.canonical(first, remaining(j))
        j += 1
      i += 1
    out.result()
