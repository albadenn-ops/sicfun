package sicfun.holdem.types

import java.util.Locale

/**
  * Shared formatting utilities for poker actions, chip amounts, and mode labels.
  *
  * This object centralizes display-formatting logic that was previously duplicated across
  * at least 9 call sites (AcpcMatchRunner, SlumbotMatchRunner, PlayingHall, AdvisorSession,
  * AlwaysOnDecisionLoop, etc.). Having a single canonical formatter prevents divergence
  * in how actions and chip values are rendered in logs, REPL output, and ACPC protocol messages.
  *
  * All formatters use `Locale.ROOT` to guarantee that decimal separators are always periods,
  * regardless of the JVM's default locale. This matters for protocol compliance (ACPC)
  * and TSV serialization.
  *
  * Package-private to `holdem` because these are internal display helpers, not part of the
  * public API surface.
  */
private[holdem] object PokerFormatting:

  /** Renders a [[PokerAction]] as a human-readable string.
    *
    * Simple actions produce their name ("Fold", "Check", "Call").
    * Raises include the amount formatted to 2 decimal places, e.g. "Raise:20.00".
    *
    * @param action the poker action to render
    * @return a display string suitable for logs, REPL output, or protocol messages
    */
  def renderAction(action: PokerAction): String =
    action match
      case PokerAction.Fold          => "Fold"
      case PokerAction.Check         => "Check"
      case PokerAction.Call          => "Call"
      case PokerAction.Raise(amount) => s"Raise:${fmtDouble(amount, 2)}"

  /** Formats a double-precision value with a fixed number of decimal digits.
    *
    * Uses `Locale.ROOT` to ensure a period decimal separator regardless of JVM locale.
    *
    * @param value  the numeric value to format
    * @param digits number of decimal places (e.g. 2 yields "3.14")
    * @return the formatted string
    */
  def fmtDouble(value: Double, digits: Int): String =
    String.format(Locale.ROOT, s"%.${digits}f", java.lang.Double.valueOf(value))

  /** Rounds a chip value to the nearest multiple of 50, with a minimum of 50.
    *
    * Used to convert continuous bet-sizing recommendations into discrete chip
    * denominations that look natural in a poker context (e.g. 50, 100, 150, ...).
    *
    * @param value the raw chip amount (may be fractional)
    * @return the rounded chip count, always >= 50
    */
  def roundedChips(value: Double): Int =
    math.max(50, math.round(value / 50.0).toInt * 50)

  /** Returns a lowercase label string for the given [[HeroMode]].
    *
    * Used in log messages and REPL status displays to indicate which decision
    * engine is currently active.
    *
    * @param mode the hero's decision mode
    * @return "adaptive" or "gto"
    */
  def heroModeLabel(mode: HeroMode): String =
    mode match
      case HeroMode.Adaptive  => "adaptive"
      case HeroMode.Gto       => "gto"
      case HeroMode.Strategic => "strategic"
