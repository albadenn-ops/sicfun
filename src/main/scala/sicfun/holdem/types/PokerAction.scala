package sicfun.holdem.types

/**
  * Poker action types and coarsened category enumeration for ML classification.
  *
  * This file defines the [[PokerAction]] enum representing the four possible actions
  * a player can take at any decision point in Texas Hold'em, plus a companion
  * [[PokerAction.Category]] enum that erases continuous raise amounts into a discrete
  * four-class label space suitable for multinomial classification.
  *
  * The separation between [[PokerAction]] and [[PokerAction.Category]] is intentional:
  *   - The full [[PokerAction]] type (with `Raise(amount)`) is used in game simulation,
  *     event sourcing, and bet-history tracking where the exact raise size matters.
  *   - The coarsened [[PokerAction.Category]] is used as the label type in ML models
  *     ([[sicfun.holdem.model.PokerActionModel]]), where predicting the action category
  *     is a multinomial classification problem and the raise amount is handled separately.
  *
  * The `category` method on [[PokerAction]] is `inline def` to eliminate the overhead
  * of the pattern match at call sites -- important because it is called per-sample
  * during training data preparation and model evaluation.
  */

/** Represents a discrete poker action that a player can take at a decision point in a hand.
  *
  * Each variant corresponds to a standard Texas Hold'em action. [[Raise]] carries a
  * bet amount, while the other actions are parameterless.
  */
enum PokerAction:
  /** Surrender the hand and forfeit any chips already committed to the pot. */
  case Fold
  /** Decline to bet when no outstanding bet is owed (toCall == 0). */
  case Check
  /** Match the current outstanding bet to stay in the hand. */
  case Call
  /** Increase the current bet by the specified `amount` (must be positive). */
  case Raise(amount: Double)

  /** Returns the coarsened [[PokerAction.Category]] for this action. */
  inline def category: PokerAction.Category = this match
    case Fold     => PokerAction.Category.Fold
    case Check    => PokerAction.Category.Check
    case Call     => PokerAction.Category.Call
    case Raise(_) => PokerAction.Category.Raise

/** Companion utilities for [[PokerAction]], including a coarsened [[Category]]
  * enumeration used as the label space for ML classification.
  */
object PokerAction:
  /** Coarsened action categories that erase the continuous raise amount,
    * reducing the action space to four discrete classes suitable for
    * multinomial classification.
    */
  enum Category:
    case Fold, Check, Call, Raise

  /** All categories in ordinal order, used as the canonical label ordering
    * for model training and prediction.
    */
  val categories: Vector[Category] = Category.values.toVector
