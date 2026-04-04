package sicfun.holdem.strategic.bridge

import sicfun.holdem.strategic.*
import sicfun.holdem.types.{GameState, PokerAction, Street}
import sicfun.core.DiscreteDistribution

/** Bundled output of all bridge conversions for a single decision point.
  *
  * This is the primary public API for the strategic layer's bridge — it
  * collects all bridge results into a single immutable snapshot that
  * orchestrators (ValidationRunner, AdaptiveProofHarness) can consume
  * without calling individual bridges.
  *
  * All fields are unwrapped from BridgeResult for direct access.
  * The fidelity metadata is available via [[fidelitySummary]].
  */
final case class StrategicSnapshot(
    // Public state (from PublicStateBridge)
    street: Street,
    pot: Chips,
    heroStack: Chips,
    toCall: Chips,
    // Signal (from SignalBridge)
    actionSignal: ActionSignal,
    // Classification (from ClassificationBridge)
    strategicClass: StrategicClass,
    // Value decomposition (from ValueBridge)
    fourWorld: FourWorld,
    // Baseline (from BaselineBridge)
    baseline: Ev,
    // Opponent model (from OpponentModelBridge, optional)
    opponentClassPosterior: Option[DiscreteDistribution[StrategicClass]]
):
  /** Human-readable fidelity summary from BridgeManifest. */
  def fidelitySummary: String = BridgeManifest.summary

object StrategicSnapshot:

  /** Build a StrategicSnapshot from engine data by calling all bridges.
    *
    * This is the single entry point for computing strategic context from
    * engine outputs. All bridge calls are made here; callers never need
    * to call individual bridges.
    *
    * @param gameState        current game state from engine
    * @param heroAction       the action hero is taking
    * @param heroEquity       hero's equity estimate (0.0 to 1.0)
    * @param engineEv         engine's EV estimate (maps to V^{1,1})
    * @param staticEquity     equity without adaptation (maps to V^{0,0})
    * @param hasDrawPotential whether hero has draw potential
    * @param opponentVpip     optional VPIP stat for opponent model bridge
    * @param opponentPfr      optional PFR stat for opponent model bridge
    * @param opponentAf       optional AF stat for opponent model bridge
    */
  def build(
      gameState: GameState,
      heroAction: PokerAction,
      heroEquity: Double,
      engineEv: Double,
      staticEquity: Double,
      hasDrawPotential: Boolean,
      opponentVpip: Option[Double] = None,
      opponentPfr: Option[Double] = None,
      opponentAf: Option[Double] = None
  ): StrategicSnapshot =
    // Public state
    val street = unwrapExact(PublicStateBridge.extractStreet(gameState))
    val pot = unwrapExact(PublicStateBridge.extractPot(gameState))
    val heroStack = unwrapExact(PublicStateBridge.extractHeroStack(gameState))
    val toCall = unwrapExact(PublicStateBridge.extractToCall(gameState))

    // Signal
    val actionSignal = unwrapValue(SignalBridge.toActionSignal(heroAction, street, pot))

    // Classification
    val strategicClass = unwrapValue(ClassificationBridge.classify(heroEquity, hasDrawPotential))

    // Value decomposition
    val fourWorld = unwrapValue(ValueBridge.toFourWorld(engineEv, staticEquity))

    // Baseline
    val baseline = unwrapValue(BaselineBridge.toRealBaseline(heroEquity))

    // Opponent model (optional)
    val opponentClassPosterior = for
      vpip <- opponentVpip
      pfr  <- opponentPfr
      af   <- opponentAf
    yield
      OpponentModelBridge.statsToClassPosterior(vpip, pfr, af).fold(
        onExact = identity,
        onApprox = (v, _) => v,
        onAbsent = _ => DiscreteDistribution.uniform(StrategicClass.values.toVector)
      )

    StrategicSnapshot(
      street = street,
      pot = pot,
      heroStack = heroStack,
      toCall = toCall,
      actionSignal = actionSignal,
      strategicClass = strategicClass,
      fourWorld = fourWorld,
      baseline = baseline,
      opponentClassPosterior = opponentClassPosterior
    )

  /** Unwrap an Exact BridgeResult. Only valid for bridges that always return Exact. */
  private def unwrapExact[A](result: BridgeResult[A]): A =
    result.fold(
      onExact = identity,
      onApprox = (v, _) => v,
      onAbsent = reason => throw new IllegalStateException(s"Expected Exact, got Absent: $reason")
    )

  /** Unwrap any non-Absent BridgeResult to its value. */
  private def unwrapValue[A](result: BridgeResult[A]): A =
    result.fold(
      onExact = identity,
      onApprox = (v, _) => v,
      onAbsent = reason => throw new IllegalStateException(s"Bridge returned Absent: $reason")
    )
