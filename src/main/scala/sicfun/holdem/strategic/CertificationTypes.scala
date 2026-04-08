package sicfun.holdem.strategic

import sicfun.holdem.types.PokerAction

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
    notes: Vector[String]
)
