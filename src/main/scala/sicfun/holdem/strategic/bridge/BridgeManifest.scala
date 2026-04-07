package sicfun.holdem.strategic.bridge

import sicfun.holdem.strategic.{Fidelity, Severity}

/** Fidelity declaration for a single formal object in the bridge. */
final case class BridgeEntry(
    formalObject: String,
    specDef: String,
    fidelity: Fidelity,
    severity: Severity,
    notes: String
)

/** BridgeManifest: declares fidelity for EVERY formal object bridged
  * between the engine and the formal layer.
  *
  * This manifest is the single source of truth for what the bridge can
  * and cannot faithfully represent. Any consumer of bridge results should
  * consult this manifest to understand the fidelity of each conversion.
  */
object BridgeManifest:

  val entries: Vector[BridgeEntry] = Vector(
    // -- Signal bridge --
    BridgeEntry("ActionSignal.action",      "Def 5",  Fidelity.Exact,       Severity.Cosmetic,    "action category maps 1:1"),
    BridgeEntry("ActionSignal.sizing",      "Def 5",  Fidelity.Approximate, Severity.Behavioral,  "pot fraction requires pot context at action time"),
    BridgeEntry("ActionSignal.timing",      "Def 5",  Fidelity.Absent,      Severity.Behavioral,  "no timing data in current engine"),
    BridgeEntry("ActionSignal.stage",       "Def 5",  Fidelity.Exact,       Severity.Cosmetic,    "street maps 1:1"),
    BridgeEntry("TotalSignal",              "Def 6",  Fidelity.Approximate, Severity.Behavioral,  "timing component absent"),
    BridgeEntry("ShowdownSignal",           "Def 7",  Fidelity.Exact,       Severity.Cosmetic,    "showdown data fully available"),

    // -- Classification bridge --
    BridgeEntry("StrategicClass",           "Def 2",  Fidelity.Approximate, Severity.Behavioral,  "equity-based heuristic, not full spot-conditioned classification"),

    // -- Public state bridge --
    BridgeEntry("Street",                   "--",     Fidelity.Exact,       Severity.Cosmetic,    "direct mapping"),
    BridgeEntry("Pot",                      "--",     Fidelity.Exact,       Severity.Cosmetic,    "direct mapping"),
    BridgeEntry("TableMap",                 "--",     Fidelity.Approximate, Severity.Behavioral,  "rival seats from initSession; hero-only fallback when no rival info"),

    // -- Opponent model bridge --
    BridgeEntry("ClassPosterior",           "Def 14", Fidelity.Approximate, Severity.Behavioral, "kernel pipeline posterior when Strategic mode; heuristic fallback otherwise"),

    // -- Baseline bridge --
    BridgeEntry("RealBaseline",             "Def 9",  Fidelity.Approximate, Severity.Behavioral,  "Monte Carlo equity"),
    BridgeEntry("AttributedBaseline",       "Def 10", Fidelity.Approximate, Severity.Structural,  "requires kernel decomposition"),

    // -- Value bridge --
    BridgeEntry("FourWorld.V11",            "Def 44", Fidelity.Approximate, Severity.Behavioral,  "engine EV is best available"),
    BridgeEntry("FourWorld.V10",            "Def 44", Fidelity.Approximate, Severity.Structural,  "interpolated estimate"),
    BridgeEntry("FourWorld.V01",            "Def 44", Fidelity.Approximate, Severity.Structural,  "interpolated estimate"),
    BridgeEntry("FourWorld.V00",            "Def 44", Fidelity.Approximate, Severity.Behavioral,  "static equity approximation"),
    BridgeEntry("DeltaVocabulary",          "Def 50", Fidelity.Approximate, Severity.Structural,  "derived from approximate four-world"),

    // -- Decomposition (not bridged, computed in formal layer) --
    BridgeEntry("PerRivalDelta",            "Defs 40-42", Fidelity.Exact,   Severity.Cosmetic,   "pure computation over Q-function values"),
    BridgeEntry("PerRivalSignalSubDecomp",  "Defs 48-49", Fidelity.Exact,   Severity.Cosmetic,   "pure computation over Q-function values"),
    BridgeEntry("BluffFramework",           "Defs 35-39", Fidelity.Exact,   Severity.Cosmetic,   "pure predicates over formal types"),
    BridgeEntry("AdaptationSafety",         "Defs 52-53, 57/57A-C (v0.31.1)", Fidelity.Exact, Severity.Cosmetic, "pure computation; v0.31.1 AS-strong and legacy scalar both available"),
    BridgeEntry("RevealSchedule",           "Def 51",     Fidelity.Exact,   Severity.Cosmetic,   "pure computation"),

    // -- v0.31.1 formal objects (Wave 3-5) --
    BridgeEntry("SecurityValue",            "Def 55",     Fidelity.Exact,       Severity.Cosmetic,   "pure min over rival profiles; computation is exact"),
    BridgeEntry("PointwiseExploitability",  "Def 55A",    Fidelity.Exact,       Severity.Cosmetic,   "pure clamped subtraction; exact given inputs"),
    BridgeEntry("DeploymentExploitability", "Def 55B",    Fidelity.Exact,       Severity.Cosmetic,   "max over finite belief set; exact given inputs"),
    BridgeEntry("DeploymentBaseline",       "A10",        Fidelity.Approximate, Severity.Behavioral, "baseline exploitability from CFR; conservative estimate"),
    BridgeEntry("SafetyBellman.T_safe",     "Def 60",     Fidelity.Exact,       Severity.Cosmetic,   "pure Bellman operator; exact computation"),
    BridgeEntry("SafetyBellman.B*",         "Def 61",     Fidelity.Approximate, Severity.Behavioral, "iterative fixed point; convergence tolerance 1e-10"),
    BridgeEntry("SafetyBellman.Certificate","Def 65",     Fidelity.Exact,       Severity.Cosmetic,   "structural validity checks are exact"),
    BridgeEntry("TotalVulnerability",       "Corollary 9.3", Fidelity.Approximate, Severity.Behavioral, "sum of two conservative estimates"),
    BridgeEntry("ChainWorld",               "Def 28",     Fidelity.Exact,       Severity.Cosmetic,   "enumerated world algebra; exact"),
    BridgeEntry("GridWorld",                "Def 44",     Fidelity.Exact,       Severity.Cosmetic,   "keyed grid coordinates; exact"),
    BridgeEntry("RiskDecomposition",        "Defs 56A-C", Fidelity.Exact,       Severity.Cosmetic,   "pure telescopic computation; exact given inputs"),
    BridgeEntry("GridWorldValues",          "Def 44 (v0.31.1)", Fidelity.Approximate, Severity.Behavioral, "V^{1,0} and V^{0,1} available via PftDpw when solver loaded; interpolated fallback otherwise")
  )

  /** All objects with Structural severity -- these degrade the formal model's coherence. */
  def structuralGaps: Vector[BridgeEntry] =
    entries.filter(_.severity == Severity.Structural)

  /** All Absent objects -- these have no engine representation at all. */
  def absentObjects: Vector[BridgeEntry] =
    entries.filter(_.fidelity == Fidelity.Absent)

  /** Summary statistics. */
  def summary: String =
    val exact = entries.count(_.fidelity == Fidelity.Exact)
    val approx = entries.count(_.fidelity == Fidelity.Approximate)
    val absent = entries.count(_.fidelity == Fidelity.Absent)
    s"BridgeManifest: $exact exact, $approx approximate, $absent absent (${entries.size} total)"
