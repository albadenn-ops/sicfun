package sicfun.holdem.strategic

/** Manifest documenting the 11 assumptions from the SICFUN v0.31.1 canonical spec.
  *
  * Assumptions fall into several categories:
  * - Structural: constraints on the problem formulation (not executable)
  * - Encoded: have computational representations enforced in code
  * - Recovered: generalized (primed) version subsumes the original
  * - Approximated: conservative computational approximation with explicit fidelity
  * - Deferred: out of constitutive scope; documented but not implemented
  */
object AssumptionManifest:

  enum EnforcementLevel:
    case Encoded      // has a computational representation enforced in code
    case Structural   // constraint on the model; verified by design, not code
    case Recovered    // generalized (primed version) subsumes the original
    case Approximated // has a conservative computational approximation with explicit fidelity
    case Deferred     // out of constitutive scope; documented but not implemented

  final case class AssumptionEntry(
      id: String,
      name: String,
      enforcement: EnforcementLevel,
      location: String,
      notes: String
  )

  val entries: Vector[AssumptionEntry] = Vector(
    AssumptionEntry(
      "A1'", "Abstraction with guarantees",
      EnforcementLevel.Structural,
      "PokerAction enum + HandStrengthEstimator (sizing quantization)",
      "Abstraction maps alpha_X, alpha_A, alpha_Y with bounded errors epsilon_R, epsilon_T, epsilon_O. " +
      "Value error bounded by epsilon_R/(1-gamma) + gamma*R_max*(epsilon_T + epsilon_O)/(1-gamma)^2. " +
      "Subsumes A1 with explicit sizing quantization via Sizing type"
    ),
    AssumptionEntry(
      "A2", "Closed Markovianity",
      EnforcementLevel.Structural,
      "AugmentedState.scala: augmented state transition",
      "X_tilde_{t+1} ~ T(.|X_tilde_t, u_t). Markov property on augmented state space"
    ),
    AssumptionEntry(
      "A3'", "Rival type non-stationarity",
      EnforcementLevel.Recovered,
      "ChangepointDetector.scala",
      "theta_{t+1}^{R,i} ~ K^i(.|theta_t, zeta_t) with hazard rate h^i. " +
      "Generalizes A3 (rival latent type). A3 recovered when hazardRate = 0 and K^i = identity"
    ),
    AssumptionEntry(
      "A4'", "Own statistical sufficiency (conditioned)",
      EnforcementLevel.Structural,
      "AugmentedState.scala: OwnEvidence",
      "Finite xi_t^S in Xi^S sufficient for SICFUN's accumulated evidence relative to A1', A6, " +
      "and parametric family F^R. OwnEvidence stores Map[String, Double] summaries"
    ),
    AssumptionEntry(
      "A5", "Bounded reward and discounting",
      EnforcementLevel.Structural,
      "Chips opaque type, PftDpwRuntime/WPomcpRuntime configs",
      "|r(x,u)| <= R_max, gamma in (0,1). R_max finite by poker pot/stack bounds. " +
      "gamma < 1 validated in solver configs"
    ),
    AssumptionEntry(
      "A6", "First-order interactive sufficiency",
      EnforcementLevel.Structural,
      "AugmentedState.scala: RivalBeliefState",
      "For each rival i, future policy depends on public history only through m_t^{R,i} in M^{R,i}. " +
      "First-order truncation: rivals do not model SICFUN's modeling of them"
    ),
    AssumptionEntry(
      "A6'", "Detection-aware exploitation",
      EnforcementLevel.Encoded,
      "DetectionPredicate.scala",
      "For each rival i, exists DetectModeling^i: H_t -> {0,1}. " +
      "Returns 0 when rival play indistinguishable from baseline under A6; " +
      "returns 1 with probability -> 1 when rival conditions on beta. " +
      "Implementations: NeverDetect, AlwaysDetect, FrequencyAnomalyDetection"
    ),
    AssumptionEntry(
      "A7", "Well-defined full rival update",
      EnforcementLevel.Structural,
      "RivalKernel.scala: ActionKernel, ShowdownKernel, FullKernel; KernelConstructor.scala",
      "For each rival i, exists Gamma^{full,(omega^act,omega^sd),i}: M^{R,i} x Y x X^pub -> M^{R,i} " +
      "parameterized by chain world (omega^act, omega^sd) in Omega^chain. " +
      "Enforced by FullKernel composition in KernelConstructor"
    ),
    AssumptionEntry(
      "A8", "Strategically relevant repetition",
      EnforcementLevel.Structural,
      "Structural (poker session guarantees repeat play)",
      "Exists p_lower > 0 such that Pr(future interaction with rival i | X_t) >= p_lower for all t. " +
      "Satisfied structurally: poker sessions involve repeated hands with same opponents"
    ),
    AssumptionEntry(
      "A9", "Spot-conditioned polarization (per-rival)",
      EnforcementLevel.Encoded,
      "SpotPolarization.scala",
      "Pol_t^i(lambda) = Pol_t^i(lambda | x_t^pub, pi^{0,S}, m_t^{R,i}). " +
      "PosteriorDivergencePolarization computes true KL divergence D_KL(posterior || prior) " +
      "when TemperedLikelihoodFn provided (Fidelity.Exact); falls back to sizing-extremity proxy otherwise"
    ),
    AssumptionEntry(
      "A10", "Adaptation safety baseline (revised)",
      EnforcementLevel.Encoded,
      "AdaptationSafety.scala, SafetyBellman.scala, Exploitability.scala",
      "Exists approximate baseline pi_bar with Exploit_{B_dep}(pi_bar) <= epsilon_base. " +
      "AS-strong (Def 57): for all b, sigma^{-S}, J(b;pi,sigma) >= J(b;pi_bar,sigma) - epsilon_adapt. " +
      "Bellman-safe certificates (Defs 58-66), TotalVulnerability (Corollary 9.3)"
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
