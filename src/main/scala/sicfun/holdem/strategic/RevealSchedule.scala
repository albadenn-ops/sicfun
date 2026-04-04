package sicfun.holdem.strategic

import sicfun.holdem.types.Street

/** Stage-indexed reveal schedule (Def 51).
  *
  * For each decision stage tau and rival i, defines the optimal reveal threshold
  * tau_tau^{*,i} such that:
  * - Below threshold: conceal (passive action)
  * - At threshold: randomize (mixed strategy)
  * - Above threshold: reveal (aggressive action for value)
  *
  * Threshold optimality holds exactly when rival i's type posterior admits a binary
  * partition and the information disclosure is single-dimensional. In the general
  * multi-dimensional case the threshold is an approximation.
  */
final case class RevealSchedule(
    entries: Map[(PlayerId, Street), RevealThreshold]
):
  /** Look up the reveal threshold for a specific rival and stage. */
  def threshold(rival: PlayerId, stage: Street): Option[RevealThreshold] =
    entries.get((rival, stage))

  /** Classify an action decision relative to the threshold. */
  def classify(rival: PlayerId, stage: Street, posteriorEquity: Ev): RevealDecision =
    threshold(rival, stage) match
      case None => RevealDecision.Unknown
      case Some(t) =>
        if posteriorEquity < t.threshold then RevealDecision.Conceal
        else if posteriorEquity > t.threshold then RevealDecision.Reveal
        else RevealDecision.Randomize

/** A single threshold entry for one rival at one stage. */
final case class RevealThreshold(
    threshold: Ev,
    isExact: Boolean  // true if binary partition holds, false if approximation
)

/** Decision classification from the reveal schedule. */
enum RevealDecision:
  case Conceal    // below threshold: passive action
  case Randomize  // at threshold: mixed strategy
  case Reveal     // above threshold: aggressive for value
  case Unknown    // no threshold available for this rival/stage
