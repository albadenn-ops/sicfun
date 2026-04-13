package sicfun.holdem.strategic

import sicfun.holdem.types.PokerAction

/** Private strategic classes of SICFUN (Def 1).
  *
  * C^S = C^V | C^B | C^M | C^SB
  *
  * Exhaustive partition: every hand in a spot belongs to exactly one class.
  * Classification depends on spot -- it is NOT a static property of the hand.
  */
enum StrategicClass:
  case Value     // C^V: hands played for value
  case Bluff     // C^B: hands played as bluffs
  case Mixed     // C^M: mixed classes — combine value-holding and bluffing in a single strategic posture
  case StructuralBluff // C^SB: structural-bluff classes (position-based aggression)

object StrategicClass:

  /** Aggressive-wager predicate (Def 3).
    * Agg: U -> {0,1}
    * True iff the action involves a bet/raise.
    */
  def isAggressiveWager(action: PokerAction): Boolean = action match
    case PokerAction.Raise(_) => true
    case _                    => false

  /** Structural-bluff predicate (Def 4).
    * StructuralBluff(c, u) = 1 iff (c in C^B) AND Agg(u) = 1.
    */
  def isStructuralBluff(cls: StrategicClass, action: PokerAction): Boolean =
    cls == StrategicClass.Bluff && isAggressiveWager(action)

  /** Exploitative-bluff predicate (derived from Def 4 + Def 42).
    * Exploitative bluff iff structural bluff AND deltaManip > 0.
    * Law L2: isExploitativeBluff -> isStructuralBluff (always).
    */
  def isExploitativeBluff(cls: StrategicClass, action: PokerAction, deltaManip: Ev): Boolean =
    isStructuralBluff(cls, action) && deltaManip > Ev.Zero
