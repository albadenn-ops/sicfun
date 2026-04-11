package sicfun.holdem.engine
import sicfun.holdem.types.*
import sicfun.holdem.equity.*

/** Probability triple representing how a villain responds to a hero action.
  *
  * This is the fundamental output of all villain response models in the engine.
  * The three probabilities (fold, call, raise) must sum to 1.0 (within floating-point
  * tolerance). The [[continueProbability]] convenience accessor returns call + raise,
  * which is the probability that the villain stays in the pot (used for raise EV
  * calculations in [[RangeInferenceEngine.responseAwareRaiseEv]]).
  *
  * Used by:
  *   - [[VillainResponseModel]] and [[UniformVillainResponseModel]] (trait implementations)
  *   - [[ArchetypeLearning]] (archetype-specific fixed profiles)
  *   - [[RangeInferenceEngine]] (response-aware raise EV estimation)
  */
final case class VillainResponseProfile(
    foldProbability: Double,
    callProbability: Double,
    raiseProbability: Double
):
  require(foldProbability >= 0.0, "foldProbability must be non-negative")
  require(callProbability >= 0.0, "callProbability must be non-negative")
  require(raiseProbability >= 0.0, "raiseProbability must be non-negative")
  require(
    math.abs(foldProbability + callProbability + raiseProbability - 1.0) < 1e-9,
    s"response probabilities must sum to 1.0, got ${foldProbability + callProbability + raiseProbability}"
  )

  /** Probability that the villain continues (does not fold). Equal to call + raise. */
  def continueProbability: Double = callProbability + raiseProbability

/** Likelihood model for villain responses conditioned on villain hand and hero action.
  *
  * This is the most general response model interface. Implementations may return
  * different profiles for different villain hands (hand-dependent response modeling).
  * For models where the response does not depend on the specific villain hand, use
  * the [[UniformVillainResponseModel]] subtype instead, which enables fast-path EV
  * evaluation without per-hand aggregation.
  */
trait VillainResponseModel:
  def response(villainHand: HoleCards, state: GameState, heroAction: PokerAction): VillainResponseProfile

/** Hand-independent villain response model.
  *
  * Implementations define one profile per `(state, heroAction)`, shared across all
  * villain hands. This enables fast-path EV evaluation without per-hand aggregation:
  * when the response profile is the same for every hand, the raise EV simplifies to
  * a single equity calculation weighted by the uniform fold/continue probabilities,
  * instead of requiring a per-hand aggregation loop.
  *
  * The [[RealTimeAdaptiveEngine]] uses this trait for its archetype-blended response
  * model, where the profile depends only on the current archetype posterior (not on
  * the specific villain hand).
  */
trait UniformVillainResponseModel extends VillainResponseModel:
  def responseProfile(state: GameState, heroAction: PokerAction): VillainResponseProfile

  final override def response(
      villainHand: HoleCards,
      state: GameState,
      heroAction: PokerAction
  ): VillainResponseProfile =
    responseProfile(state, heroAction)

object VillainResponseModel:
  /** Binary (deterministic) hand-range response model.
    *
    * Assigns each hand a pure-strategy response based on set membership:
    *   - Hands in `raiseRange` always raise (probability 1.0).
    *   - Hands in `callRange` but NOT in `raiseRange` always call (probability 1.0).
    *   - All other hands always fold (probability 1.0).
    *
    * For non-raise hero actions (Check, Call, Fold), assumes the villain always
    * continues (calls with probability 1.0), since this model only defines
    * raise-defense behavior.
    *
    * This is the simplest possible response model and is used as the default
    * BB-vs-SB defense model in [[sbVsBbOpenDefault]].
    */
  final case class BinaryHandRanges(
      raiseRange: Set[HoleCards],
      callRange: Set[HoleCards]
  ) extends VillainResponseModel:
    def response(
        villainHand: HoleCards,
        state: GameState,
        heroAction: PokerAction
    ): VillainResponseProfile =
      heroAction match
        case PokerAction.Raise(_) =>
          if raiseRange.contains(villainHand) then VillainResponseProfile(0.0, 0.0, 1.0)
          else if callRange.contains(villainHand) then VillainResponseProfile(0.0, 1.0, 0.0)
          else VillainResponseProfile(1.0, 0.0, 0.0)
        case _ =>
          // Outside explicit raise-defense modeling, assume villain continues.
          VillainResponseProfile(0.0, 1.0, 0.0)

  /** Builds a binary response model from range strings. */
  def fromRanges(
      raiseRange: String,
      callRange: String
  ): Either[String, BinaryHandRanges] =
    for
      raise <- RangeParser.parseWithHands(raiseRange).left.map(err => s"raiseRange: $err")
      call <- RangeParser.parseWithHands(callRange).left.map(err => s"callRange: $err")
    yield BinaryHandRanges(raise.hands, call.hands)

  /** Default BB defend model versus a SB open (raise-first-in from SB).
    *
    * This is intentionally simple (binary per hand) and can be replaced by
    * richer mixed-frequency models later.
    */
  lazy val sbVsBbOpenDefault: BinaryHandRanges =
    fromRanges(
      raiseRange = "66+, A5s-A2s, A8s+, KTs+, QTs+, JTs, ATo+, KQo",
      callRange =
        "22-55, A2s-A7s, K2s-K9s, Q6s-Q9s, J7s-J9s, T7s-T9s, 97s+, 86s+, 75s+, 65s, 54s, A2o-A9o, K7o-KJo, Q8o-QJo, J8o-JTo, T8o+, 98o"
    ) match
      case Right(model) => model
      case Left(err) =>
        throw new IllegalStateException(s"failed to parse SB-vs-BB default response ranges: $err")

/** Hand-strength-aware villain response model.
  *
  * Modulates a base archetype-blended fold/call/raise profile by each villain
  * hand's strength (via [[HandStrengthEstimator.fastGtoStrength]]) and the
  * pot-odds implied by the hero's raise sizing. Strong hands fold less and
  * raise more; weak hands fold more. When averaged across a uniform strength
  * distribution, the aggregate profile approximates the base profile.
  *
  * This replaces the hand-agnostic [[UniformVillainResponseModel]] used by
  * [[RealTimeAdaptiveEngine]], fixing the critical bug where the engine
  * assigned identical fold equity to raises with AA and 23o.
  *
  * @param baseProfileFn live function returning the current blended archetype
  *                      profile (called per invocation so the archetype
  *                      posterior stays fresh as observations accumulate)
  */
final class HandStrengthResponseModel(
    baseProfileFn: () => VillainResponseProfile
) extends VillainResponseModel:

  override def response(
      villainHand: HoleCards,
      state: GameState,
      heroAction: PokerAction
  ): VillainResponseProfile =
    heroAction match
      case PokerAction.Raise(amount) =>
        val base = baseProfileFn()
        val strength = HandStrengthEstimator.fastGtoStrength(villainHand, state.board, state.street)
        val potOddsFactor = math.max(0.5, math.min(2.0, amount / math.max(1.0, state.pot)))
        HandStrengthResponseModel.modulate(base, strength, potOddsFactor)
      case _ =>
        VillainResponseProfile(0.0, 1.0, 0.0)

object HandStrengthResponseModel:
  /** Modulates a base profile by hand strength and pot-odds sizing.
    *
    * For a hand at strength `s` ∈ [0, 1]:
    *   - fold(s)  = baseFold × potOddsFactor × 2 × (1 − s)
    *   - raise(s) = baseRaise × 2 × s
    *   - call(s)  = remainder
    *
    * The `2 × (1 − s)` / `2 × s` weighting ensures that when integrated over
    * a uniform strength distribution, the average fold/raise rates match the
    * base profile (at potOddsFactor = 1.0).
    *
    * `potOddsFactor` adjusts for raise sizing: bigger raises (relative to pot)
    * increase the fold rate proportionally.
    */
  def modulate(
      base: VillainResponseProfile,
      strength: Double,
      potOddsFactor: Double
  ): VillainResponseProfile =
    val rawFold = base.foldProbability * potOddsFactor * 2.0 * (1.0 - strength)
    val rawRaise = base.raiseProbability * 2.0 * strength
    // Clamp fold + raise so they don't exceed 1.0
    val fold = math.max(0.0, math.min(1.0, rawFold))
    val raise = math.max(0.0, math.min(1.0 - fold, rawRaise))
    val call = math.max(0.0, 1.0 - fold - raise)
    // Normalize (handles edge cases where rounding leaves a gap)
    val total = fold + call + raise
    if total <= 1e-9 then VillainResponseProfile(0.0, 1.0, 0.0)
    else VillainResponseProfile(fold / total, call / total, raise / total)
