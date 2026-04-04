package sicfun.holdem.strategic

/** Manifest documenting the 10 assumptions from the SICFUN v0.30.2 canonical spec.
  *
  * Assumptions fall into two categories:
  * - Structural: constraints on the problem formulation (not executable)
  * - Encoded: have computational representations in the formal layer
  */
object AssumptionManifest:

  enum EnforcementLevel:
    case Encoded    // has a computational representation enforced in code
    case Structural // constraint on the model; verified by design, not code
    case Recovered  // generalized (primed version) subsumes the original

  final case class AssumptionEntry(
      id: String,
      name: String,
      enforcement: EnforcementLevel,
      location: String,
      notes: String
  )

  val entries: Vector[AssumptionEntry] = Vector(
    AssumptionEntry(
      "A1", "Finite action space",
      EnforcementLevel.Structural,
      "PokerAction enum (4 categories)",
      "Hold'em has a finite action set by definition; enforced by PokerAction.Category enum"
    ),
    AssumptionEntry(
      "A2", "Common prior over rival types",
      EnforcementLevel.Structural,
      "OpponentModelState.typeDistribution",
      "All rivals share the same prior type space; the prior is stored in OpponentModelState"
    ),
    AssumptionEntry(
      "A3'", "Non-stationary rival types with changepoint detection",
      EnforcementLevel.Recovered,
      "ChangepointDetector.scala",
      "Generalizes A3 (stationary types). A3 recovered when hazardRate -> 0"
    ),
    AssumptionEntry(
      "A4'", "Own statistical sufficiency",
      EnforcementLevel.Structural,
      "AugmentedState.scala: OwnEvidence",
      "xi^S is finite-dimensional; OwnEvidence stores Map[String, Double] summaries"
    ),
    AssumptionEntry(
      "A5", "Conditional independence of signals given type",
      EnforcementLevel.Structural,
      "RivalKernel.scala: ActionKernel, ShowdownKernel",
      "Action and showdown channels are conditionally independent given rival type; enforced by separate kernel traits"
    ),
    AssumptionEntry(
      "A6'", "First-order interactive POMDP with detection",
      EnforcementLevel.Encoded,
      "DetectionPredicate.scala",
      "Rivals do not model SICFUN's modeling of them, but SICFUN detects if they do. NeverDetect/AlwaysDetect/FrequencyAnomaly"
    ),
    AssumptionEntry(
      "A7", "Bounded reward",
      EnforcementLevel.Structural,
      "Chips opaque type",
      "R_max is finite; poker pots are bounded by stack depths. Used in Corollary 4 bound"
    ),
    AssumptionEntry(
      "A8", "Discount factor gamma in [0,1)",
      EnforcementLevel.Structural,
      "PftDpwRuntime, WPomcpRuntime configs",
      "gamma < 1 required by POMDP solvers; validated in native solver configs"
    ),
    AssumptionEntry(
      "A9", "Spot-conditioned polarization",
      EnforcementLevel.Encoded,
      "SpotPolarization.scala",
      "Polarization is spot-dependent. UniformPolarization and PosteriorDivergencePolarization implementations"
    ),
    AssumptionEntry(
      "A10", "Adaptation safety",
      EnforcementLevel.Encoded,
      "AdaptationSafety.scala",
      "Exploit(pi^S_beta) <= epsilon_NE + delta_adapt. Enforced by betaBar clamping (Theorem 8)"
    )
  )

  def encoded: Vector[AssumptionEntry] =
    entries.filter(_.enforcement == EnforcementLevel.Encoded)

  def structural: Vector[AssumptionEntry] =
    entries.filter(_.enforcement == EnforcementLevel.Structural)

  def summary: String =
    val enc = entries.count(_.enforcement == EnforcementLevel.Encoded)
    val str = entries.count(_.enforcement == EnforcementLevel.Structural)
    val rec = entries.count(_.enforcement == EnforcementLevel.Recovered)
    s"AssumptionManifest: $enc encoded, $str structural, $rec recovered (${entries.size} total)"
