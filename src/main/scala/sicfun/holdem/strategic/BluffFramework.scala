package sicfun.holdem.strategic

import sicfun.holdem.types.PokerAction

/** Bluff framework (Defs 35-39).
  *
  * Structural bluff (Def 35): StructuralBluff(c, u) = 1 iff c in C^B AND Agg(u).
  * Feasible action correspondence (Def 36): A: Delta(X~) => U.
  * Feasible non-bluff set (Def 37): U_nf(b~) = {u in A(b~) | StructuralBluff(c,u) = 0}.
  * Bluff gain (Def 38): Gain_bluff(b~, u; u_cf) = Q^{*,attrib}(b~,u) - Q^{*,ref}(b~,u_cf).
  * Exploitative bluff (Def 39): structural bluff AND sup_{u_cf in U_nf} Gain_bluff > 0.
  */
object BluffFramework:

  /** Def 35: Structural bluff predicate. Delegates to StrategicClass.isStructuralBluff. */
  def isStructuralBluff(cls: StrategicClass, action: PokerAction): Boolean =
    StrategicClass.isStructuralBluff(cls, action)

  /** Def 36: Feasible action correspondence.
    * In the current model, all legal actions at a decision point are feasible.
    * This is an identity pass-through; future versions may restrict by belief.
    */
  def feasibleActions(legalActions: Vector[PokerAction]): Vector[PokerAction] =
    legalActions

  /** Def 37: Feasible non-bluff actions for the same hand.
    * U_nf(b~) = {u in A(b~) | StructuralBluff(c, u) = 0}.
    */
  def feasibleNonBluffActions(cls: StrategicClass, legalActions: Vector[PokerAction]): Vector[PokerAction] =
    legalActions.filterNot(a => isStructuralBluff(cls, a))

  /** Def 38: Bluff gain (aggregate multiway form).
    * Gain_bluff(b~, u; u_cf) = Q^{*,attrib}(b~, u) - Q^{*,ref}(b~, u_cf).
    */
  def bluffGain(qAttribAction: Ev, qRefCounterfactual: Ev): Ev =
    qAttribAction - qRefCounterfactual

  /** Def 39: Exploitative bluff predicate.
    * True iff:
    *  1. isStructuralBluff(cls, action)
    *  2. bestGainOverNonBluffs > 0 (i.e., sup_{u_cf in U_nf} Gain_bluff > 0)
    *
    * The caller is responsible for computing the supremum over non-bluff actions
    * and passing it as bestGainOverNonBluffs.
    *
    * Corollary 2: exploitative bluff => structural bluff (holds by construction).
    */
  def isExploitativeBluff(cls: StrategicClass, action: PokerAction, bestGainOverNonBluffs: Ev): Boolean =
    isStructuralBluff(cls, action) && bestGainOverNonBluffs > Ev.Zero
