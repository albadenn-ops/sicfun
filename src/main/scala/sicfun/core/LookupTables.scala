package sicfun.core

/** Precomputed combinatorial index tables used by [[HandEvaluator]] for fast
  * best-of-seven hand evaluation.
  *
  * In Texas Hold'em, the best five-card hand is selected from seven cards (two hole
  * cards plus five community cards). Rather than generating the C(7,5) = 21 subsets
  * at evaluation time, they are stored here as a constant array for zero-allocation
  * lookup on the hot path.
  */
object LookupTables:
  /** All 21 ways to choose 5 indices from 7, in lexicographic order.
    *
    * Each inner array contains 5 indices into a 7-element card array.
    * Used by [[HandEvaluator]] to enumerate all five-card subsets of a seven-card hand.
    */
  val combos7of5: Array[Array[Int]] = Array(
    Array(0, 1, 2, 3, 4),
    Array(0, 1, 2, 3, 5),
    Array(0, 1, 2, 3, 6),
    Array(0, 1, 2, 4, 5),
    Array(0, 1, 2, 4, 6),
    Array(0, 1, 2, 5, 6),
    Array(0, 1, 3, 4, 5),
    Array(0, 1, 3, 4, 6),
    Array(0, 1, 3, 5, 6),
    Array(0, 1, 4, 5, 6),
    Array(0, 2, 3, 4, 5),
    Array(0, 2, 3, 4, 6),
    Array(0, 2, 3, 5, 6),
    Array(0, 2, 4, 5, 6),
    Array(0, 3, 4, 5, 6),
    Array(1, 2, 3, 4, 5),
    Array(1, 2, 3, 4, 6),
    Array(1, 2, 3, 5, 6),
    Array(1, 2, 4, 5, 6),
    Array(1, 3, 4, 5, 6),
    Array(2, 3, 4, 5, 6)
  )
