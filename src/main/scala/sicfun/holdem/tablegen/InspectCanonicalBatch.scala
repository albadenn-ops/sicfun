package sicfun.holdem.tablegen
import sicfun.holdem.*
import sicfun.holdem.equity.*

/** Small utility to inspect canonical batch endpoint density.
  *
  * Prints how many unique hole-card IDs are touched by the selected canonical
  * matchup slice.
  */
object InspectCanonicalBatch:
  /** Entry point. Selects a canonical batch of the given size and reports how many
    * distinct hole-card IDs appear in the low position, the high position, and overall.
    *
    * This is useful for understanding how well a given batch size covers the 1326 possible
    * hole-card indices, which affects GPU occupancy and cache behavior. A batch that touches
    * all 1326 IDs is "endpoint-complete"; smaller batches may cluster around certain IDs.
    *
    * @param args optional positional: maxMatchups (default 5000)
    */
  def main(args: Array[String]): Unit =
    val maxMatchups = args.headOption.map(_.toLong).getOrElse(5000L)
    val batch = HeadsUpEquityCanonicalTable.selectCanonicalBatch(maxMatchups)
    // Track which hole-card IDs appear in low, high, and combined positions.
    val lowSeen = scala.collection.mutable.BitSet.empty
    val highSeen = scala.collection.mutable.BitSet.empty
    val allSeen = scala.collection.mutable.BitSet.empty

    var idx = 0
    while idx < batch.packedKeys.length do
      val packed = batch.packedKeys(idx)
      val low = HeadsUpEquityTable.unpackLowId(packed)
      val high = HeadsUpEquityTable.unpackHighId(packed)
      lowSeen += low
      highSeen += high
      allSeen += low
      allSeen += high
      idx += 1

    println(s"maxMatchups=$maxMatchups")
    println(s"entries=${batch.packedKeys.length}")
    println(s"uniqueLow=${lowSeen.size}")
    println(s"uniqueHigh=${highSeen.size}")
    println(s"uniqueEndpoints=${allSeen.size}")
