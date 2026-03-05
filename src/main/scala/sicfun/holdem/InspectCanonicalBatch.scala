package sicfun.holdem

/** Small utility to inspect canonical batch endpoint density.
  *
  * Prints how many unique hole-card IDs are touched by the selected canonical
  * matchup slice.
  */
object InspectCanonicalBatch:
  def main(args: Array[String]): Unit =
    val maxMatchups = args.headOption.map(_.toLong).getOrElse(5000L)
    val batch = HeadsUpEquityCanonicalTable.selectCanonicalBatch(maxMatchups)
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
