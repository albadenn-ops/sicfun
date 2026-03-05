package sicfun.holdem

/** Per-hand response probabilities to a hero action. */
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

  def continueProbability: Double = callProbability + raiseProbability

/** Likelihood model for villain responses conditioned on villain hand and hero action. */
trait VillainResponseModel:
  def response(villainHand: HoleCards, state: GameState, heroAction: PokerAction): VillainResponseProfile

object VillainResponseModel:
  /** Binary hand-range response model:
    * - hands in raiseRange always raise
    * - hands in callRange (and not in raiseRange) always call
    * - all other hands fold
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
