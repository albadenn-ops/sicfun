package sicfun.holdem.strategic.types

import sicfun.holdem.types.PokerAction
import sicfun.holdem.strategic.decomposition.{FourWorld, DeltaVocabulary, RiskDecomposition}
import sicfun.holdem.strategic.exploitation.RevealDecision
import sicfun.holdem.strategic.safety.OperationalBaseline

/** Opaque identifier for a joint rival profile (all rivals assigned one
  * StrategicClass). Distinct from StrategicClass to prevent misuse.
  */
opaque type JointRivalProfileId = Int
object JointRivalProfileId:
  def apply(ordinal: Int): JointRivalProfileId = ordinal
  extension (id: JointRivalProfileId) def ordinal: Int = id

/** Result from a single solver invocation under one rival profile. */
final case class SolverResult(
    bestAction: Int,
    actionValues: Array[Double]
)

/** Certification result — determines which evaluation layer produced the bundle. */
enum CertificationResult:
  /** Root-local budget screening (WPomcp approximate path).
    * NOT Defs 61-66.
    */
  case LocalRobustScreening(
      rootLosses: Array[Double],
      budgetEstimate: Double,
      withinTolerance: Boolean
  )
  /** Conservative tabular approximation of Defs 58-66.
    * B* computed on latent states, lifted to belief by particle expectation.
    */
  case TabularCertification(
      bStar: Array[Double],
      requiredBudget: Double,
      safeActionIndices: IndexedSeq[Int],
      certificateValid: Boolean,
      withinTolerance: Boolean
  )
  case Unavailable(reason: String)

/** Per-action bluff annotation (Defs 35-39). */
final case class BluffAnnotation(
    isStructuralBluff: Boolean,
    bluffGain: Option[Ev],
    isExploitativeBluff: Boolean
)

/** Decision outcome from the certification pipeline. */
enum DecisionOutcome:
  case Certified(action: PokerAction, bundle: DecisionEvaluationBundle)
  case BaselineFallback(action: PokerAction, reason: String)

/** The single authoritative runtime artifact for all formal safety computations. */
final case class DecisionEvaluationBundle(
    profileResults: Map[JointRivalProfileId, SolverResult],
    robustActionLowerBounds: Array[Double],
    baselineActionValues: Array[Double],
    baselineValue: Double,
    adversarialRootGap: Option[Ev],
    pointwiseExploitability: Option[Ev],
    deploymentExploitability: Option[Ev],
    certification: CertificationResult,
    chainWorldValues: Map[ChainWorld, Ev],
    fourWorld: Option[FourWorld] = None,
    deltaVocabulary: Option[DeltaVocabulary] = None,
    bluffAnnotations: Vector[BluffAnnotation] = Vector.empty,
    chainRiskProfile: Option[RiskDecomposition.ChainRiskProfile] = None,
    polarizationProfile: Map[Int, Double] = Map.empty,
    revealDecision: Option[RevealDecision] = None,
    operationalBaseline: Option[OperationalBaseline] = None,
    notes: Vector[String]
)
