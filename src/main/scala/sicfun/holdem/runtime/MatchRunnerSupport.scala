package sicfun.holdem.runtime

import sicfun.holdem.types.*

import java.io.BufferedWriter
import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}

/** Shared infrastructure for match runner statistics, logging, and summaries.
  *
  * Extracted from AcpcMatchRunner.Runner and SlumbotMatchRunner.Runner where
  * recordOutcome, buildSummary, writeSummary, appendDecisionLog, and RunSummary
  * were structurally identical (only differing in chip type: Double vs Int).
  */
private[holdem] object MatchRunnerSupport:

  /** Immutable snapshot of aggregate match statistics, used for file output and return values.
    *
    * @param handsPlayed      Total number of hands completed.
    * @param heroNetChips     Hero's cumulative signed chip result.
    * @param heroBbPer100     Win rate in big blinds per 100 hands.
    * @param heroWins         Number of hands where hero had positive net chips.
    * @param heroTies         Number of hands where hero broke even (net = 0).
    * @param heroLosses       Number of hands where hero had negative net chips.
    * @param buttonHands      Hands played from the button position.
    * @param buttonNetChips   Net chips accumulated from button position.
    * @param bigBlindHands    Hands played from the big blind position.
    * @param bigBlindNetChips Net chips accumulated from big blind position.
    * @param heroMode         Decision mode used (Adaptive or Gto).
    * @param modelId          Identifier of the action model artifact used.
    * @param outDir           Output directory where artifacts were written.
    */
  final case class RunSummary(
      handsPlayed: Int,
      heroNetChips: Double,
      heroBbPer100: Double,
      heroWins: Int,
      heroTies: Int,
      heroLosses: Int,
      buttonHands: Int,
      buttonNetChips: Double,
      bigBlindHands: Int,
      bigBlindNetChips: Double,
      heroMode: HeroMode,
      modelId: String,
      outDir: Path
  )

  /** Mutable match statistics accumulator. Thread-unsafe — used within a single Runner. */
  final class MatchStatistics:
    private var handsPlayed    = 0
    private var heroNetChips   = 0.0
    private var heroWins       = 0
    private var heroTies       = 0
    private var heroLosses     = 0
    private var buttonHands    = 0
    private var buttonNetChips = 0.0
    private var bigBlindHands    = 0
    private var bigBlindNetChips = 0.0

    def currentHandsPlayed: Int = handsPlayed
    def currentHeroNetChips: Double = heroNetChips

    /** Record the result of a completed hand, updating all counters and position breakdowns.
      *
      * @param heroPosition  Hero's position this hand (Button or BigBlind).
      * @param heroNetChips  Hero's signed chip result for this hand.
      */
    def recordOutcome(heroPosition: Position, heroNetChips: Double): Unit =
      this.handsPlayed += 1
      this.heroNetChips += heroNetChips
      if heroNetChips > 0.0 then this.heroWins += 1
      else if heroNetChips < 0.0 then this.heroLosses += 1
      else this.heroTies += 1
      heroPosition match
        case Position.Button =>
          this.buttonHands += 1
          this.buttonNetChips += heroNetChips
        case Position.BigBlind =>
          this.bigBlindHands += 1
          this.bigBlindNetChips += heroNetChips
        case other =>
          throw new IllegalStateException(s"unexpected hero position in match runner: $other")

    /** Compute the current bb/100 win rate: (netChips / bbSize / hands) * 100. */
    def currentBbPer100(bigBlindChips: Double): Double =
      if handsPlayed > 0 then (heroNetChips / bigBlindChips / handsPlayed.toDouble) * 100.0
      else 0.0

    /** Freeze the current mutable statistics into an immutable [[RunSummary]] snapshot. */
    def buildSummary(heroMode: HeroMode, modelId: String, outDir: Path, bigBlindChips: Int = 100): RunSummary =
      RunSummary(
        handsPlayed      = handsPlayed,
        heroNetChips     = heroNetChips,
        heroBbPer100     = currentBbPer100(bigBlindChips.toDouble),
        heroWins         = heroWins,
        heroTies         = heroTies,
        heroLosses       = heroLosses,
        buttonHands      = buttonHands,
        buttonNetChips   = buttonNetChips,
        bigBlindHands    = bigBlindHands,
        bigBlindNetChips = bigBlindNetChips,
        heroMode         = heroMode,
        modelId          = modelId,
        outDir           = outDir
      )

  /** Write a human-readable summary file with all match statistics.
    *
    * @param path    Output file path.
    * @param label   Section header label (e.g., "ACPC Match Runner", "Slumbot Match Runner").
    * @param summary The run summary to write.
    */
  def writeSummary(path: Path, label: String, summary: RunSummary): Unit =
    val lines = Vector(
      s"=== $label ===",
      s"handsPlayed: ${summary.handsPlayed}",
      s"heroNetChips: ${PokerFormatting.fmtDouble(summary.heroNetChips, 3)}",
      s"heroBbPer100: ${PokerFormatting.fmtDouble(summary.heroBbPer100, 3)}",
      s"heroWins: ${summary.heroWins}",
      s"heroTies: ${summary.heroTies}",
      s"heroLosses: ${summary.heroLosses}",
      s"buttonHands: ${summary.buttonHands}",
      s"buttonNetChips: ${PokerFormatting.fmtDouble(summary.buttonNetChips, 3)}",
      s"bigBlindHands: ${summary.bigBlindHands}",
      s"bigBlindNetChips: ${PokerFormatting.fmtDouble(summary.bigBlindNetChips, 3)}",
      s"heroMode: ${PokerFormatting.heroModeLabel(summary.heroMode)}",
      s"modelId: ${summary.modelId}"
    )
    Files.write(path, lines.mkString(System.lineSeparator()).getBytes(StandardCharsets.UTF_8))

  /** Append a tab-separated decision log row to the decisions.tsv file.
    *
    * Columns: hand, decisionIndex, street, position, pot, toCall, stack,
    * candidates (comma-separated), chosenAction, wireAction.
    */
  def appendDecisionLog(
      writer: BufferedWriter,
      handId: Int,
      decisionIndex: Int,
      state: GameState,
      candidates: Vector[PokerAction],
      chosenAction: PokerAction,
      wireAction: String
  ): Unit =
    writer.write(
      Vector(
        handId.toString,
        decisionIndex.toString,
        state.street.toString,
        state.position.toString,
        PokerFormatting.fmtDouble(state.pot, 3),
        PokerFormatting.fmtDouble(state.toCall, 3),
        PokerFormatting.fmtDouble(state.stackSize, 3),
        candidates.map(PokerFormatting.renderAction).mkString(","),
        PokerFormatting.renderAction(chosenAction),
        wireAction
      ).mkString("\t")
    )
    writer.newLine()
    writer.flush()
