package sicfun.holdem.strategic.formulation

import sicfun.holdem.types.PokerAction
import sicfun.holdem.strategic.types.{BridgeResult, Ev, StrategicClass}

/** Value source for formulation-level reasoning. */
trait FormulationValueSource:
  /** Scalar poker equity estimate for the current spot. */
  def estimateSpotEquity(spot: FormulationSpot): BridgeResult[Double]

  /** Full bucket-vs-bucket showdown equity table for WPomcp-style consumers. */
  def showdownEquityTable(
      spot: FormulationSpot,
      numHeroBuckets: Int,
      numRivalBuckets: Int
  ): BridgeResult[Array[Double]]

  /** Per-action value estimate for the current spot and action. */
  def estimateActionValue(
      spot: FormulationSpot,
      action: PokerAction
  ): BridgeResult[Ev]

/** Rival policy source for formulation-level reasoning. */
trait FormulationRivalPolicySource:
  /** Returns action weights aligned with spot.candidateActions. */
  def actionPolicy(
      cls: StrategicClass,
      spot: FormulationSpot
  ): BridgeResult[Vector[Double]]

/** Action semantics source for formulation-level reasoning. */
trait FormulationActionSource:
  def semanticsFor(
      spot: FormulationSpot,
      action: PokerAction
  ): BridgeResult[FormulationActionSemantics]

/** Complete formulation input: spot plus provider interfaces. */
final case class FormulationInput(
    spot: FormulationSpot,
    valueSource: FormulationValueSource,
    rivalPolicySource: FormulationRivalPolicySource,
    actionSource: FormulationActionSource
)
