package sicfun.holdem.strategic.formulation

import sicfun.holdem.types.*
import sicfun.holdem.strategic.types.*
import sicfun.holdem.strategic.state.*
import sicfun.holdem.engine.PokerPomcpFormulation

/** Legacy rival prior formats from both formulation paths. */
final case class LegacyRivalPriors(
    pftActionPriors: Map[(StrategicClass, PokerAction.Category), Double],
    pomcpClassPriors: Map[StrategicClass, (Double, Double, Double)]
)

/** Allows PFT formulation to access raw PFT action priors from the legacy adapter. */
trait LegacyPftPriorsAccessor:
  def pftActionPriors: Map[(StrategicClass, PokerAction.Category), Double]

/** Compatibility adapter that wraps toy assumptions behind the A2 interfaces. */
object LegacyToyFormulationInput:

  def from(
      gameState: GameState,
      candidateActions: Vector[PokerAction],
      rivalBeliefs: Map[PlayerId, StrategicRivalBelief],
      heroBucket: Int,
      rivalPriors: LegacyRivalPriors
  ): FormulationInput =
    val spot = FormulationSpot(
      gameState = gameState,
      candidateActions = candidateActions,
      heroValueInput = HeroValueInput.StrengthHint(
        bucket = heroBucket,
        source = "LegacyToyFormulationInput"
      ),
      rivalBeliefs = rivalBeliefs
    )
    FormulationInput(
      spot = spot,
      valueSource = LegacyValueSource(heroBucket),
      rivalPolicySource = LegacyRivalPolicySource(rivalPriors),
      actionSource = LegacyActionSource
    )

  /** Value source backed by heroBucket/9.0 and linear showdown equity. */
  private class LegacyValueSource(heroBucket: Int) extends FormulationValueSource:
    override def estimateSpotEquity(spot: FormulationSpot): BridgeResult[Double] =
      BridgeResult.Approximate(
        heroBucket / 9.0,
        "toy: heroBucket / 9.0"
      )

    override def showdownEquityTable(
        spot: FormulationSpot,
        numHeroBuckets: Int,
        numRivalBuckets: Int
    ): BridgeResult[Array[Double]] =
      BridgeResult.Approximate(
        PokerPomcpFormulation.buildLinearShowdownEquity(numHeroBuckets, numRivalBuckets),
        "toy: linear showdown equity heuristic"
      )

    override def estimateActionValue(
        spot: FormulationSpot,
        action: PokerAction
    ): BridgeResult[Ev] =
      val equity = heroBucket / 9.0
      val gs = spot.gameState
      val potFraction = gs.pot / math.max(gs.stackSize, 1.0)
      val callFraction = gs.toCall / math.max(gs.stackSize, 1.0)
      val value = action match
        case PokerAction.Fold => -(equity * potFraction)
        case PokerAction.Call => (equity - 0.5) * callFraction
        case PokerAction.Check => 0.0
        case r: PokerAction.Raise =>
          val raiseFraction = r.amount / math.max(gs.stackSize, 1.0)
          (equity - 0.3) * raiseFraction
      BridgeResult.Approximate(Ev(value), "toy: bucket-derived action value")

  /** Rival policy backed by static WPomcp class priors. */
  private class LegacyRivalPolicySource(priors: LegacyRivalPriors)
      extends FormulationRivalPolicySource with LegacyPftPriorsAccessor:

    override def pftActionPriors: Map[(StrategicClass, PokerAction.Category), Double] =
      priors.pftActionPriors

    override def actionPolicy(
        cls: StrategicClass,
        spot: FormulationSpot
    ): BridgeResult[Vector[Double]] =
      val numActions = spot.candidateActions.size
      val (foldP, passiveP, raiseP) = priors.pomcpClassPriors.getOrElse(cls, (0.25, 0.50, 0.25))
      val raw = new Array[Double](numActions)
      raw(0) = foldP
      if numActions > 1 then raw(1) = passiveP
      val raiseSlots = math.max(1, numActions - 2)
      for i <- 2 until numActions do raw(i) = raiseP / raiseSlots
      val sum = raw.sum
      if sum > 0 then
        for i <- raw.indices do raw(i) /= sum
      BridgeResult.Approximate(raw.toVector, "toy: static class prior table")

  /** Action semantics backed by the current toy terminal/action logic. */
  private object LegacyActionSource extends FormulationActionSource:
    override def semanticsFor(
        spot: FormulationSpot,
        action: PokerAction
    ): BridgeResult[FormulationActionSemantics] =
      val gs = spot.gameState
      val isRiver = gs.street == Street.River
      val sem = action match
        case PokerAction.Fold =>
          FormulationActionSemantics(
            chipsCommitted = 0.0,
            potDeltaChips = 0.0,
            isAllIn = false,
            terminal = FormulationTerminalKind.HeroFold,
            advancesStreet = false
          )
        case PokerAction.Check =>
          FormulationActionSemantics(
            chipsCommitted = 0.0,
            potDeltaChips = 0.0,
            isAllIn = false,
            terminal = if isRiver then FormulationTerminalKind.Showdown
                       else FormulationTerminalKind.Continue,
            advancesStreet = !isRiver
          )
        case PokerAction.Call =>
          FormulationActionSemantics(
            chipsCommitted = gs.toCall,
            potDeltaChips = gs.toCall,
            isAllIn = false,
            terminal = if isRiver then FormulationTerminalKind.Showdown
                       else FormulationTerminalKind.Continue,
            advancesStreet = !isRiver
          )
        case r: PokerAction.Raise =>
          FormulationActionSemantics(
            chipsCommitted = r.amount,
            potDeltaChips = r.amount,
            isAllIn = r.amount >= gs.stackSize,
            terminal = if isRiver then FormulationTerminalKind.Showdown
                       else FormulationTerminalKind.Continue,
            advancesStreet = !isRiver
          )
      BridgeResult.Approximate(sem, "toy: static action semantics")
