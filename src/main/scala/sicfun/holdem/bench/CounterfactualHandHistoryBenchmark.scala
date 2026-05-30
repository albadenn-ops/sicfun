package sicfun.holdem.bench

import sicfun.holdem.equity.{TableFormat, TableRanges}
import sicfun.holdem.history.{HandHistoryImport, ImportedHand}
import sicfun.holdem.model.PokerActionModel
import sicfun.holdem.runtime.HandHistoryAnalyzer
import sicfun.holdem.engine.villain.RealTimeAdaptiveEngine

import java.util.Random

/** Track B: counterfactual EV of following sicfun vs the actually-played line on real hands,
  * aggregated to a bb/100 delta with a bootstrap CI. Reuses [[HandHistoryAnalyzer]]; no engine change.
  *
  * [[HandHistoryAnalyzer.AnalyzedDecision.evDifference]] is `actualEv - recommendedEv` (chips), so the
  * counterfactual gain of switching from the played action to sicfun's recommended action is
  * `recommendedEv - actualEv = -evDifference`. Chip gains are normalized to big blinds via
  * [[ImportedHand.bigBlind]] (guarded `> 0`) and aggregated to bb/100 with a percentile bootstrap CI.
  *
  * INTERPRETATION (read before trusting the number): this is a SELF-ASSESSED counterfactual.
  * `recommendedEv` is the engine's OWN argmax over its candidate EVs, so the per-decision gain
  * (`recommendedEv - actualEv`) is >= 0 by construction — the engine never recommends an action it
  * scored below the one actually played. The bb/100 figure therefore measures "EV the played line left
  * on the table vs. what THIS engine would have done," NOT a calibrated win rate and NOT an
  * exploitability bound. On a small corpus the figure (and its wide CI) is a smoke/sanity signal only.
  * A real win-rate estimate comes from Track A (hall self-play vs a canonical field), not this.
  */
object CounterfactualHandHistoryBenchmark:

  final case class Result(hands: Int, decisions: Int, ci: WinRateStats.WinRateCI)

  /** Per-hand counterfactual hero gain in bb: sum over hero decisions of
    * `(recommendedEv - actualEv)/bigBlind = sum(-evDifference)/bigBlind`. Returns the gain together
    * with the number of hero decisions analyzed (0 when the hand lacks hero cards/name or a posted BB).
    */
  private def perHandGainsBb(hand: ImportedHand, seed: Long): (Double, Int) =
    if hand.heroHoleCards.isEmpty || hand.heroName.isEmpty || hand.bigBlind <= 0.0 then (0.0, 0)
    else
      val tableRanges = TableRanges.defaults(TableFormat.forPlayerCount(hand.players.length))
      val engine = new RealTimeAdaptiveEngine(
        tableRanges = tableRanges,
        actionModel = PokerActionModel.uniform,
        bunchingTrials = 1,
        defaultEquityTrials = 400,
        minEquityTrials = 200
      )
      val decisions = HandHistoryAnalyzer.analyzeWithHeroCards(
        events = hand.events,
        heroPlayerId = hand.heroName.get,
        heroCards = hand.heroHoleCards.get,
        engine = engine,
        tableRanges = tableRanges,
        availablePositions = hand.players.iterator.map(_.position).toSet,
        budgetMs = 2000L,
        rng = new Random(seed ^ hand.handId.hashCode.toLong)
      )
      val gainBb = decisions.iterator.map(d => (-d.evDifference) / hand.bigBlind).sum
      (gainBb, decisions.length)

  def runHands(hands: Vector[ImportedHand], seed: Long, resamples: Int, ciLevel: Double): Result =
    val perHand = hands.map(h => perHandGainsBb(h, seed))
    val gains = perHand.collect { case (g, n) if n > 0 => g }
    val totalDecisions = perHand.map(_._2).sum
    Result(hands.length, totalDecisions, WinRateStats.bbPer100CI(gains, resamples, ciLevel, seed))

  def runText(
      text: String,
      heroName: String,
      seed: Long = 42L,
      resamples: Int = 2000,
      ciLevel: Double = 0.95
  ): Either[String, Result] =
    HandHistoryImport.parseText(text, None, Some(heroName)).map(hands => runHands(hands, seed, resamples, ciLevel))

  def runFile(
      path: java.nio.file.Path,
      heroName: String,
      seed: Long = 42L,
      resamples: Int = 2000,
      ciLevel: Double = 0.95
  ): Either[String, Result] =
    HandHistoryImport.parseFile(path, None, Some(heroName)).map(hands => runHands(hands, seed, resamples, ciLevel))

  def main(args: Array[String]): Unit =
    if args.length < 2 then
      System.err.println("usage: CounterfactualHandHistoryBenchmark <corpusFile> <heroName> [seed]")
      sys.exit(1)
    val seed = args.lift(2).flatMap(_.toLongOption).getOrElse(42L)
    runFile(java.nio.file.Path.of(args(0)), args(1), seed) match
      case Right(r) =>
        println(
          f"counterfactual (self-assessed, not a win rate) bb/100 = ${r.ci.pointEstimate}%.3f  CI=[${r.ci.lower}%.3f, ${r.ci.upper}%.3f]  (hands=${r.hands}, decisions=${r.decisions}, perHandSamples=${r.ci.sampleSize})"
        )
      case Left(err) =>
        System.err.println(s"failed: $err")
        sys.exit(1)
