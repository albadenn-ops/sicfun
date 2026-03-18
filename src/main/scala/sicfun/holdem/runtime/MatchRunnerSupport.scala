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

    def currentBbPer100(bigBlindChips: Double): Double =
      if handsPlayed > 0 then (heroNetChips / bigBlindChips / handsPlayed.toDouble) * 100.0
      else 0.0

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
