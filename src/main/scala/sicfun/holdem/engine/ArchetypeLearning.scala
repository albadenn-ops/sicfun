package sicfun.holdem.engine
import sicfun.holdem.types.*

/** Accumulator for villain fold/call/raise response counts when facing aggression.
  *
  * Used to build a compact summary of observed villain behavior in response to hero
  * raises. These counts feed into [[ArchetypeLearning.posteriorFromCounts]] to produce
  * a Bayesian posterior over villain archetypes without requiring the full observation
  * sequence to be replayed.
  *
  * All counts are immutable and non-negative. The [[observe]] method returns a new
  * instance with the relevant counter incremented. Non-response actions (e.g., Check)
  * are ignored as they carry no signal for raise-response modeling.
  */
final case class RaiseResponseCounts(
    folds: Int = 0,
    calls: Int = 0,
    raises: Int = 0
):
  require(folds >= 0, "folds must be non-negative")
  require(calls >= 0, "calls must be non-negative")
  require(raises >= 0, "raises must be non-negative")

  /** Total number of observed raise-response actions (fold + call + raise). */
  def total: Int = folds + calls + raises

  /** Records one villain action, incrementing the matching counter.
    *
    * @param action the observed villain action. Only Fold, Call, and Raise carry signal;
    *               Check and other actions are no-ops (the same instance is returned).
    * @return a new RaiseResponseCounts with the appropriate counter incremented
    */
  def observe(action: PokerAction): RaiseResponseCounts =
    action match
      case PokerAction.Fold => copy(folds = folds + 1)
      case PokerAction.Call => copy(calls = calls + 1)
      case PokerAction.Raise(_) => copy(raises = raises + 1)
      case _ => this

/** Shared Bayesian learning utilities for villain archetype inference.
  *
  * This object implements the core Bayesian update loop for classifying a villain
  * into one of five archetypes (Nit, Tag, Lag, CallingStation, Maniac) based on
  * their observed fold/call/raise behavior when facing aggression.
  *
  * The model works as follows:
  *   1. Each archetype has a fixed likelihood profile (the probability of fold/call/raise
  *      given that archetype). These are stored in [[raiseResponseByArchetype]].
  *   2. Given a prior distribution over archetypes and an observed villain action,
  *      [[updatePosterior]] applies Bayes' rule: posterior(archetype) ~ prior(archetype) * likelihood(action | archetype).
  *   3. [[blendedRaiseResponse]] produces a mixture response profile by weighting each
  *      archetype's profile by its posterior probability, used for EV calculations.
  *   4. [[posteriorFromCounts]] is a batch variant that replays a count summary through
  *      sequential Bayesian updates, equivalent to observing actions one at a time.
  *
  * Design decision: The likelihood profiles are hand-tuned constants rather than learned
  * parameters. This keeps the model simple and fast for real-time use while still
  * providing meaningful archetype separation (e.g., Nit folds 68% vs. Maniac raises 45%).
  */
object ArchetypeLearning:
  /** Epsilon to guard against division-by-zero in normalization. */
  private inline val Eps = 1e-12

  /** Internal enum mapping poker actions to a simplified three-outcome space
    * for the archetype likelihood model.
    */
  private enum ResponseOutcome:
    case Fold
    case Call
    case Raise

  /** Fixed likelihood profiles: P(fold/call/raise | archetype) for each villain archetype.
    *
    * These define how each archetype is expected to respond when facing a hero raise:
    * - Nit: folds most of the time (0.68), rarely raises (0.04)
    * - Tag (tight-aggressive): balanced fold/call, moderate raise frequency
    * - Lag (loose-aggressive): calls often, raises more than Tag
    * - CallingStation: calls 73% of the time, rarely folds or raises
    * - Maniac: raises 45% of the time, folds only 25%
    */
  private val raiseResponseByArchetype: Map[PlayerArchetype, VillainResponseProfile] = Map(
    PlayerArchetype.Nit -> VillainResponseProfile(0.68, 0.28, 0.04),
    PlayerArchetype.Tag -> VillainResponseProfile(0.48, 0.42, 0.10),
    PlayerArchetype.Lag -> VillainResponseProfile(0.34, 0.50, 0.16),
    PlayerArchetype.CallingStation -> VillainResponseProfile(0.20, 0.73, 0.07),
    PlayerArchetype.Maniac -> VillainResponseProfile(0.25, 0.30, 0.45)
  )

  /** Produces a mixture response profile by weighting each archetype's fixed profile
    * by its posterior probability.
    *
    * This is used by the adaptive engine to predict how a villain will respond to a
    * hero raise, given the current archetype belief distribution. The result is a
    * single VillainResponseProfile whose fold/call/raise probabilities sum to 1.0.
    *
    * @param posterior the current Bayesian posterior over villain archetypes
    * @return a blended VillainResponseProfile. Falls back to pure-call (0, 1, 0) if
    *         the weighted total is effectively zero (degenerate posterior).
    */
  def blendedRaiseResponse(posterior: ArchetypePosterior): VillainResponseProfile =
    // Accumulate weighted fold/call/raise probabilities across all archetypes
    val weighted = PlayerArchetype.values.foldLeft((0.0, 0.0, 0.0)) { case ((fold, call, raise), archetype) =>
      val weight = posterior.probabilityOf(archetype)
      val profile = raiseResponseByArchetype(archetype)
      (
        fold + (weight * profile.foldProbability),
        call + (weight * profile.callProbability),
        raise + (weight * profile.raiseProbability)
      )
    }
    val (fold, call, raise) = weighted
    val total = fold + call + raise
    // Degenerate case: if total mass is ~0, default to pure call to avoid NaN
    if total <= Eps then VillainResponseProfile(0.0, 1.0, 0.0)
    else
      // Re-normalize to ensure probabilities sum to exactly 1.0
      val inv = 1.0 / total
      VillainResponseProfile(fold * inv, call * inv, raise * inv)

  /** Single-step Bayesian update of the archetype posterior given one observed villain action.
    *
    * Applies Bayes' rule: for each archetype, multiply the prior probability by the
    * likelihood of observing the given action under that archetype's response profile,
    * then normalize. Actions that do not map to fold/call/raise (e.g., Check) are
    * ignored and the posterior is returned unchanged.
    *
    * @param current the current archetype posterior distribution
    * @param villainAction the observed villain action (Fold, Call, Raise carry signal; others are no-ops)
    * @return the updated (normalized) archetype posterior
    */
  def updatePosterior(
      current: ArchetypePosterior,
      villainAction: PokerAction
  ): ArchetypePosterior =
    responseOutcome(villainAction) match
      case None => current // Action carries no signal (e.g., Check)
      case Some(outcome) =>
        // Bayes' rule: posterior(archetype) ~ prior(archetype) * P(outcome | archetype)
        val weights = PlayerArchetype.values.map { archetype =>
          val prior = current.probabilityOf(archetype)
          val likelihood = responseLikelihood(archetype, outcome)
          archetype -> (prior * likelihood)
        }.toMap
        normalize(weights)

  /** Batch Bayesian update: replays fold/call/raise counts as sequential observations.
    *
    * This is equivalent to calling [[updatePosterior]] once for each observed action,
    * but accepts a compact count summary instead of the full action sequence. The order
    * of updates (all folds, then all calls, then all raises) does not affect the final
    * posterior because Bayesian updates with independent observations are order-invariant
    * when likelihoods are independent of history.
    *
    * @param counts the observed fold/call/raise counts
    * @param prior the starting archetype posterior (defaults to uniform)
    * @return the posterior after processing all observations
    */
  def posteriorFromCounts(
      counts: RaiseResponseCounts,
      prior: ArchetypePosterior = ArchetypePosterior.uniform
  ): ArchetypePosterior =
    var posterior = prior
    // Process all fold observations
    var idx = 0
    while idx < counts.folds do
      posterior = updatePosterior(posterior, PokerAction.Fold)
      idx += 1
    // Process all call observations
    idx = 0
    while idx < counts.calls do
      posterior = updatePosterior(posterior, PokerAction.Call)
      idx += 1
    // Process all raise observations (raise amount is irrelevant for archetype learning)
    idx = 0
    while idx < counts.raises do
      posterior = updatePosterior(posterior, PokerAction.Raise(1.0))
      idx += 1
    posterior

  /** Maps a PokerAction to its simplified ResponseOutcome for likelihood lookup.
    *
    * @return Some(outcome) for fold/call/raise, None for actions that carry no
    *         raise-response signal (Check, etc.)
    */
  private def responseOutcome(action: PokerAction): Option[ResponseOutcome] =
    action match
      case PokerAction.Fold => Some(ResponseOutcome.Fold)
      case PokerAction.Call => Some(ResponseOutcome.Call)
      case PokerAction.Raise(_) => Some(ResponseOutcome.Raise)
      case _ => None

  /** Returns P(outcome | archetype) from the fixed likelihood table.
    *
    * @param archetype the villain archetype hypothesis
    * @param outcome the observed response outcome (Fold, Call, or Raise)
    * @return the probability of observing this outcome given this archetype
    */
  private def responseLikelihood(
      archetype: PlayerArchetype,
      outcome: ResponseOutcome
  ): Double =
    val profile = raiseResponseByArchetype(archetype)
    outcome match
      case ResponseOutcome.Fold => profile.foldProbability
      case ResponseOutcome.Call => profile.callProbability
      case ResponseOutcome.Raise => profile.raiseProbability

  /** Normalizes raw archetype weights into a valid ArchetypePosterior.
    *
    * If the total weight is effectively zero (all likelihoods collapsed to zero),
    * falls back to a uniform posterior to avoid NaN/Inf propagation.
    *
    * @param weights unnormalized archetype -> weight map
    * @return a normalized ArchetypePosterior summing to 1.0
    */
  private def normalize(
      weights: Map[PlayerArchetype, Double]
  ): ArchetypePosterior =
    val total = weights.values.sum
    if total <= Eps then ArchetypePosterior.uniform
    else
      val inv = 1.0 / total
      ArchetypePosterior(weights.view.mapValues(_ * inv).toMap)
