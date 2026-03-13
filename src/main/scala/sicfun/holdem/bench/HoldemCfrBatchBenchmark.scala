package sicfun.holdem.bench

import sicfun.core.{Deck, DiscreteDistribution}
import sicfun.holdem.types.*
import sicfun.holdem.cfr.*
import sicfun.holdem.equity.HoldemCombinator

/** Measures batch CFR GPU throughput vs sequential CPU baseline. */
object HoldemCfrBatchBenchmark:
  def main(args: Array[String]): Unit =
    val batchSize = if args.length > 0 then args(0).toInt else 100
    val iterations = if args.length > 1 then args(1).toInt else 1500
    val warmup = 3
    val runs = 10

    val state = GameState(
      street = Street.Preflop, board = Board.empty,
      pot = 6.0, toCall = 2.0, position = Position.Button,
      stackSize = 100.0, betHistory = Vector.empty
    )
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(6.0))
    val allHoles = HoldemCombinator.holeCardsFrom(Deck.full)
    val allHeroes = allHoles.take(batchSize)
    val config = HoldemCfrConfig(iterations = iterations, maxVillainHands = 48)
    val posterior = DiscreteDistribution.uniform(allHoles)

    println(s"Batch CFR benchmark: $batchSize trees, $iterations iterations")
    println(s"GPU available: ${HoldemCfrNativeRuntime.availability(HoldemCfrNativeRuntime.Backend.Gpu)}")

    // Warmup
    for _ <- 0 until warmup do
      HoldemCfrSolver.solveBatchDecisionPolicies(
        heroHands = allHeroes, state = state,
        villainPosterior = posterior,
        candidateActions = actions, config = config
      )

    // Timed runs
    val times = (0 until runs).map { _ =>
      val start = System.nanoTime()
      HoldemCfrSolver.solveBatchDecisionPolicies(
        heroHands = allHeroes, state = state,
        villainPosterior = posterior,
        candidateActions = actions, config = config
      )
      (System.nanoTime() - start) / 1e6
    }.sorted

    println(s"\n=== BATCH RESULTS (median of $runs) ===")
    println(f"Batch $batchSize trees: ${times(runs / 2)}%.1f ms")
    println(f"Per tree: ${times(runs / 2) / batchSize}%.3f ms")

    // Sequential baseline
    val seqTimes = (0 until math.min(runs, 3)).map { _ =>
      val start = System.nanoTime()
      allHeroes.foreach { hero =>
        HoldemCfrSolver.solveDecisionPolicy(
          hero = hero, state = state,
          villainPosterior = posterior,
          candidateActions = actions, config = config
        )
      }
      (System.nanoTime() - start) / 1e6
    }.sorted

    println(f"\nSequential $batchSize trees: ${seqTimes(seqTimes.length / 2)}%.1f ms")
    println(f"Speedup: ${seqTimes(seqTimes.length / 2) / times(runs / 2)}%.1fx")
