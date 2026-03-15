package sicfun.holdem.validation

import sicfun.holdem.types.{PokerAction, Position, Street}
import scala.util.Random

/** A deliberate exploitable deviation from competent play.
  *
  * Each leak defines:
  * - a predicate ([[applies]]) that fires in specific spots
  * - a deviation ([[deviate]]) that replaces the competent action with a leaky one
  * - a [[severity]] in [0, 1] controlling how often the deviation fires when applicable
  */
trait InjectedLeak:
  def id: String
  def description: String
  def severity: Double
  def applies(spot: SpotContext): Boolean
  def deviate(competentAction: PokerAction, spot: SpotContext, rng: Random): PokerAction

  /** Roll against severity. If hit, return deviatedAction; otherwise return competentAction unchanged. */
  protected def rollAndDeviate(
      competentAction: PokerAction,
      deviatedAction: PokerAction,
      rng: Random
  ): PokerAction =
    if rng.nextDouble() < severity then deviatedAction else competentAction

/** Overfolds to large bets on wet boards when range is capped and hand is medium/weak.
  *
  * Fires on river only. Replaces Call with Fold.
  */
final case class OverfoldsToAggression(severity: Double) extends InjectedLeak:
  val id = "overfold-river-aggression"
  val description = "Overfolds to large bets on wet boards when range is capped and hand is medium/weak"

  def applies(spot: SpotContext): Boolean =
    spot.street == Street.River &&
    spot.boardTexture.isWet &&
    spot.potGeometry.betToPotRatio >= 0.7 &&
    spot.rangeAdvantage == RangePosition.Capped &&
    (spot.handStrengthVsBoard == HandCategory.Weak || spot.handStrengthVsBoard == HandCategory.Medium)

  def deviate(competentAction: PokerAction, spot: SpotContext, rng: Random): PokerAction =
    rollAndDeviate(competentAction, PokerAction.Fold, rng)

/** Calls large bets with weak/air hands when GTO should fold. */
final case class Overcalls(severity: Double) extends InjectedLeak:
  val id = "overcall-big-bets"
  val description = "Calls large bets with weak/air hands when GTO should fold"

  def applies(spot: SpotContext): Boolean =
    spot.potGeometry.betToPotRatio >= 0.8 &&
    (spot.handStrengthVsBoard == HandCategory.Weak || spot.handStrengthVsBoard == HandCategory.Air)

  def deviate(competentAction: PokerAction, spot: SpotContext, rng: Random): PokerAction =
    rollAndDeviate(competentAction, PokerAction.Call, rng)

/** Bets/raises turn with air on wet boards when in position -- GTO would check back. */
final case class OverbluffsTurnBarrel(severity: Double) extends InjectedLeak:
  val id = "overbluff-turn-barrel"
  val description = "Bets/raises turn with air on wet boards when IP -- GTO would check back"

  def applies(spot: SpotContext): Boolean =
    spot.street == Street.Turn &&
    spot.boardTexture.isWet &&
    spot.handStrengthVsBoard == HandCategory.Air &&
    (spot.position == Position.Button || spot.position == Position.Cutoff)

  def deviate(competentAction: PokerAction, spot: SpotContext, rng: Random): PokerAction =
    val betSize = spot.potGeometry.effectiveStack * 0.6
    rollAndDeviate(competentAction, PokerAction.Raise(betSize), rng)

/** Checks strong hands in big pots (SPR < 2) instead of value betting. */
final case class PassiveInBigPots(severity: Double) extends InjectedLeak:
  val id = "passive-big-pots"
  val description = "Checks strong hands in big pots (SPR<2) instead of value betting"

  def applies(spot: SpotContext): Boolean =
    spot.potGeometry.isBigPot &&
    (spot.handStrengthVsBoard == HandCategory.Strong || spot.handStrengthVsBoard == HandCategory.Nuts)

  def deviate(competentAction: PokerAction, spot: SpotContext, rng: Random): PokerAction =
    rollAndDeviate(competentAction, PokerAction.Check, rng)

/** Calls/opens preflop with hands outside GTO range (weak/air). */
final case class PreflopTooLoose(severity: Double) extends InjectedLeak:
  val id = "preflop-too-loose"
  val description = "Calls/opens preflop with hands outside GTO range"

  def applies(spot: SpotContext): Boolean =
    spot.street == Street.Preflop &&
    (spot.handStrengthVsBoard == HandCategory.Weak || spot.handStrengthVsBoard == HandCategory.Air)

  def deviate(competentAction: PokerAction, spot: SpotContext, rng: Random): PokerAction =
    rollAndDeviate(competentAction, PokerAction.Call, rng)

/** Folds playable hands preflop that GTO would open/defend (medium hands). */
final case class PreflopTooTight(severity: Double) extends InjectedLeak:
  val id = "preflop-too-tight"
  val description = "Folds playable hands preflop that GTO would open/defend"

  def applies(spot: SpotContext): Boolean =
    spot.street == Street.Preflop &&
    spot.handStrengthVsBoard == HandCategory.Medium

  def deviate(competentAction: PokerAction, spot: SpotContext, rng: Random): PokerAction =
    rollAndDeviate(competentAction, PokerAction.Fold, rng)
