package sicfun.holdem.types

import java.util.Locale

/** Shared formatting utilities for poker actions, chip amounts, and mode labels.
  *
  * Consolidates duplicated private helpers from AcpcMatchRunner (9 copies of renderAction),
  * SlumbotMatchRunner, PlayingHall, AdvisorSession, AlwaysOnDecisionLoop, etc.
  */
private[holdem] object PokerFormatting:

  def renderAction(action: PokerAction): String =
    action match
      case PokerAction.Fold          => "Fold"
      case PokerAction.Check         => "Check"
      case PokerAction.Call          => "Call"
      case PokerAction.Raise(amount) => s"Raise:${fmtDouble(amount, 2)}"

  def fmtDouble(value: Double, digits: Int): String =
    String.format(Locale.ROOT, s"%.${digits}f", java.lang.Double.valueOf(value))

  def roundedChips(value: Double): Int =
    math.max(50, math.round(value / 50.0).toInt * 50)

  def heroModeLabel(mode: HeroMode): String =
    mode match
      case HeroMode.Adaptive => "adaptive"
      case HeroMode.Gto      => "gto"
