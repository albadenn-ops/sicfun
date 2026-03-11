package sicfun.holdem.engine
import sicfun.holdem.types.*

/** Count summary of how a player responds when facing aggression. */
final case class RaiseResponseCounts(
    folds: Int = 0,
    calls: Int = 0,
    raises: Int = 0
):
  require(folds >= 0, "folds must be non-negative")
  require(calls >= 0, "calls must be non-negative")
  require(raises >= 0, "raises must be non-negative")

  def total: Int = folds + calls + raises

  def observe(action: PokerAction): RaiseResponseCounts =
    action match
      case PokerAction.Fold => copy(folds = folds + 1)
      case PokerAction.Call => copy(calls = calls + 1)
      case PokerAction.Raise(_) => copy(raises = raises + 1)
      case _ => this

/** Shared Bayesian learning utilities for villain archetype inference. */
object ArchetypeLearning:
  private inline val Eps = 1e-12

  private enum ResponseOutcome:
    case Fold
    case Call
    case Raise

  private val raiseResponseByArchetype: Map[PlayerArchetype, VillainResponseProfile] = Map(
    PlayerArchetype.Nit -> VillainResponseProfile(0.68, 0.28, 0.04),
    PlayerArchetype.Tag -> VillainResponseProfile(0.48, 0.42, 0.10),
    PlayerArchetype.Lag -> VillainResponseProfile(0.34, 0.50, 0.16),
    PlayerArchetype.CallingStation -> VillainResponseProfile(0.20, 0.73, 0.07),
    PlayerArchetype.Maniac -> VillainResponseProfile(0.25, 0.30, 0.45)
  )

  def blendedRaiseResponse(posterior: ArchetypePosterior): VillainResponseProfile =
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
    if total <= Eps then VillainResponseProfile(0.0, 1.0, 0.0)
    else
      val inv = 1.0 / total
      VillainResponseProfile(fold * inv, call * inv, raise * inv)

  def updatePosterior(
      current: ArchetypePosterior,
      villainAction: PokerAction
  ): ArchetypePosterior =
    responseOutcome(villainAction) match
      case None => current
      case Some(outcome) =>
        val weights = PlayerArchetype.values.map { archetype =>
          val prior = current.probabilityOf(archetype)
          val likelihood = responseLikelihood(archetype, outcome)
          archetype -> (prior * likelihood)
        }.toMap
        normalize(weights)

  def posteriorFromCounts(
      counts: RaiseResponseCounts,
      prior: ArchetypePosterior = ArchetypePosterior.uniform
  ): ArchetypePosterior =
    var posterior = prior
    var idx = 0
    while idx < counts.folds do
      posterior = updatePosterior(posterior, PokerAction.Fold)
      idx += 1
    idx = 0
    while idx < counts.calls do
      posterior = updatePosterior(posterior, PokerAction.Call)
      idx += 1
    idx = 0
    while idx < counts.raises do
      posterior = updatePosterior(posterior, PokerAction.Raise(1.0))
      idx += 1
    posterior

  private def responseOutcome(action: PokerAction): Option[ResponseOutcome] =
    action match
      case PokerAction.Fold => Some(ResponseOutcome.Fold)
      case PokerAction.Call => Some(ResponseOutcome.Call)
      case PokerAction.Raise(_) => Some(ResponseOutcome.Raise)
      case _ => None

  private def responseLikelihood(
      archetype: PlayerArchetype,
      outcome: ResponseOutcome
  ): Double =
    val profile = raiseResponseByArchetype(archetype)
    outcome match
      case ResponseOutcome.Fold => profile.foldProbability
      case ResponseOutcome.Call => profile.callProbability
      case ResponseOutcome.Raise => profile.raiseProbability

  private def normalize(
      weights: Map[PlayerArchetype, Double]
  ): ArchetypePosterior =
    val total = weights.values.sum
    if total <= Eps then ArchetypePosterior.uniform
    else
      val inv = 1.0 / total
      ArchetypePosterior(weights.view.mapValues(_ * inv).toMap)
