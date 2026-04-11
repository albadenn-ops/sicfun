package sicfun.holdem.types

/**
  * Decision strategy selector for the hero (the player being advised).
  *
  * This file defines the [[HeroMode]] enumeration, which controls which decision engine
  * the advisor session and playing hall use to generate recommendations for the hero.
  *
  * The two modes represent fundamentally different poker philosophies:
  *   - '''Adaptive''' exploits observed villain tendencies via Bayesian range inference,
  *     producing villain-specific adjustments. Best against predictable opponents.
  *   - '''GTO''' (Game Theory Optimal) follows the Nash equilibrium strategy computed by
  *     the CFR (Counterfactual Regret Minimization) solver. Unexploitable but does not
  *     adjust to opponent weaknesses.
  *
  * The mode is typically selected at session startup and can be toggled between hands.
  * It is used by [[sicfun.holdem.runtime.PokerAdvisor]] and
  * [[sicfun.holdem.runtime.PlayingHall]] to dispatch to the appropriate engine.
  */

/** Decision mode for hero play: adaptive (Bayesian engine), GTO (CFR solver), or Strategic (POMDP solver). */
enum HeroMode:
  /** Adaptive mode: uses Bayesian inference to estimate villain ranges and exploit tendencies. */
  case Adaptive
  /** GTO mode: follows Nash equilibrium strategy from the CFR solver, ignoring villain-specific reads. */
  case Gto
  /** Strategic mode: formal POMDP via WPomcp solver with factored particle beliefs. */
  case Strategic
