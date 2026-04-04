package sicfun.holdem.strategic.bridge

import sicfun.holdem.strategic.*

/** Bridge: hand strength -> StrategicClass.
  *
  * Maps the engine's equity/hand-strength estimate to the 4-class partition.
  * Thresholds are configurable; defaults follow standard poker theory.
  *
  * Fidelity: Approximate (equity-based classification is a simplification
  * of the full spot-conditioned classification in Def 2).
  */
object ClassificationBridge:

  /** Default thresholds for equity-based classification.
    * Value: equity >= 0.65
    * SemiBluff: 0.35 <= equity < 0.65 AND draw potential
    * Marginal: 0.35 <= equity < 0.65 AND no draw potential
    * Bluff: equity < 0.35
    */
  final case class ClassificationThresholds(
      valueFloor: Double = 0.65,
      bluffCeiling: Double = 0.35
  ):
    require(valueFloor > bluffCeiling, "valueFloor must exceed bluffCeiling")

  /** Classify a hand based on equity and draw potential. */
  def classify(
      equity: Double,
      hasDrawPotential: Boolean,
      thresholds: ClassificationThresholds = ClassificationThresholds()
  ): BridgeResult[StrategicClass] =
    val cls =
      if equity >= thresholds.valueFloor then StrategicClass.Value
      else if equity < thresholds.bluffCeiling then StrategicClass.Bluff
      else if hasDrawPotential then StrategicClass.SemiBluff
      else StrategicClass.Marginal
    BridgeResult.Approximate(cls, "equity-based classification; Def 2 requires full spot context")
