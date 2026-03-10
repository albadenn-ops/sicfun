package sicfun.holdem.types

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
