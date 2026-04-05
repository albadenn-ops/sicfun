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
    BridgeEntry("TableMap",                 "--",     Fidelity.Absent,      Severity.Structural,  "GameState is hero-only; no player list for TableMap"),

    // -- Opponent model bridge --
    BridgeEntry("ClassPosterior",           "Def 14", Fidelity.Approximate, Severity.Structural,  "heuristic from VPIP/PFR/AF, not Bayesian update"),

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
    BridgeEntry("AdaptationSafety",         "Defs 52-53 (v0.30.2); upgrade to Defs 57/57A-C pending Wave 4", Fidelity.Exact,   Severity.Cosmetic,   "pure computation; def numbering will change in Wave 4"),
    BridgeEntry("RevealSchedule",           "Def 51",     Fidelity.Exact,   Severity.Cosmetic,   "pure computation")
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
