package sicfun.holdem.bench

import sicfun.core.{Card, Deck, DiscreteDistribution}
import sicfun.holdem.types.*
import sicfun.holdem.equity.{HoldemEquity, HoldemCombinator}

/** A/B benchmark: equityExact (Double) vs equityExactProb (Int32 fixed-point).
  *
  * Usage: sbt "runMain sicfun.holdem.bench.ProbEquityBenchmark [warmup] [runs] [mode]"
  *   mode: "turn" (default) or "flop"
  */
object ProbEquityBenchmark:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(throw new IllegalArgumentException(s"bad card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def board(tokens: String*): Board =
    Board.from(tokens.map(card))

  def main(args: Array[String]): Unit =
    val warmup = if args.length > 0 then args(0).toInt else 5
    val runs = if args.length > 1 then args(1).toInt else 20
    val mode = if args.length > 2 then args(2) else "turn"

    val hero = hole("Ah", "Kh")
    val (b, label) = mode match
      case "flop" => (board("2c", "7d", "Ts"), "flop (2 missing)")
      case _      => (board("2c", "7d", "Ts", "Qc"), "turn (1 missing)")

    // Build a realistic range (~20 hands with varied weights)
    val remaining = Deck.full.filterNot(card => hero.contains(card) || b.asSet.contains(card))
    val allHands = HoldemCombinator.holeCardsFrom(remaining.toIndexedSeq).take(20)
    val weights = allHands.zipWithIndex.map { case (h, i) => h -> (1.0 + i * 0.1) }
    val range = DiscreteDistribution(weights.toMap).normalized

    println(s"=== Prob Equity A/B Benchmark ===")
    println(s"Board: $label, Range size: ${range.weights.size} hands")
    println(s"Warmup: $warmup, Runs: $runs")
    println()

    // Warmup both paths
    var w = 0
    while w < warmup do
      HoldemEquity.equityExact(hero, b, range)
      HoldemEquity.equityExactProb(hero, b, range)
      w += 1

    // Interleaved measurement to reduce systematic bias (JVM warmup, GC, thermals)
    val doubleTimes = new Array[Long](runs)
    val probTimes = new Array[Long](runs)
    var r = 0
    while r < runs do
      if r % 2 == 0 then
        doubleTimes(r) = timeOnce { HoldemEquity.equityExact(hero, b, range) }
        probTimes(r) = timeOnce { HoldemEquity.equityExactProb(hero, b, range) }
      else
        probTimes(r) = timeOnce { HoldemEquity.equityExactProb(hero, b, range) }
        doubleTimes(r) = timeOnce { HoldemEquity.equityExact(hero, b, range) }
      r += 1

    // Correctness check
    val dblResult = HoldemEquity.equityExact(hero, b, range)
    val probResult = HoldemEquity.equityExactProb(hero, b, range)

    // Report
    val dblMedian = median(doubleTimes)
    val probMedian = median(probTimes)
    val speedup = dblMedian.toDouble / probMedian.toDouble

    println("--- Double (baseline) ---")
    printStats("Double", doubleTimes)
    println()
    println("--- Prob (fixed-point) ---")
    printStats("Prob  ", probTimes)
    println()
    println(f"Speedup (median): ${speedup}%.3fx")
    println()
    println("--- Correctness ---")
    println(f"Double: win=${dblResult.win}%.6f tie=${dblResult.tie}%.6f loss=${dblResult.loss}%.6f equity=${dblResult.equity}%.6f")
    println(f"Prob:   win=${probResult.win}%.6f tie=${probResult.tie}%.6f loss=${probResult.loss}%.6f equity=${probResult.equity}%.6f")
    val eqDiff = math.abs(dblResult.equity - probResult.equity)
    println(f"Equity diff: $eqDiff%.9f")

  private def timeOnce(body: => Any): Long =
    val t0 = System.nanoTime()
    body
    System.nanoTime() - t0

  private def median(arr: Array[Long]): Long =
    val sorted = arr.sorted
    sorted(sorted.length / 2)

  private def printStats(label: String, times: Array[Long]): Unit =
    val sorted = times.sorted
    val n = sorted.length
    val med = sorted(n / 2)
    val min = sorted.head
    val max = sorted.last
    val mean = times.map(_.toDouble).sum / n
    val p25 = sorted(n / 4)
    val p75 = sorted(3 * n / 4)
    println(f"$label median=${med / 1e6}%.2f ms  p25=${p25 / 1e6}%.2f  p75=${p75 / 1e6}%.2f  min=${min / 1e6}%.2f  max=${max / 1e6}%.2f  mean=${mean / 1e6}%.2f ms")
