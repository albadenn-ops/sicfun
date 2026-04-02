# SICFUN-v0.26 Model Integration Design

**Date:** 2026-04-02
**Status:** Approved (design only — no implementation)
**Approach:** A' — Formal parallel layer, multiway-native, with anti-corruption layer and minimal algebraic core

## 0. Purpose

This document defines the design for preparing the sicfun codebase to integrate
the SICFUN-v0.26 formal model: an interactive first-order POMDP with augmented
hidden state, per-rival type inference, strategic signaling value decomposition
(Q^my + Delta_learn + Delta_sig), structural/exploitative bluff primitives,
spot-conditioned polarization, and dynamic reputational state tracking.

**Scope:** Design and type system only. No implementation. The existing engine
continues to function unchanged. The formal layer is strictly additive.

---

## 1. Architectural Decision and Perimeter

### 1.1 Approach

**A' — Formal parallel layer with anti-corruption layer and minimal algebraic core.**

A new namespace (`sicfun.holdem.strategic`) defines the semantic types, invariants,
and canonical operators of the v0.26 model. Integration with the existing engine
is performed through an explicit anti-corruption layer (`sicfun.holdem.strategic.bridge`)
that documents exact, approximate, or absent correspondences.

The formal layer is not a DTO cemetery. It includes:
- ADTs with invariants
- Canonical operators (mini-algebra)
- Coherence laws

### 1.2 Dependencies

```
sicfun.holdem.strategic  ->  sicfun.holdem.types   (ONLY stable primitives: PokerAction, Street, Position, HoleCards, Board)
sicfun.holdem.strategic  ->  sicfun.core           (DiscreteDistribution, Probability)
sicfun.holdem.strategic  X-> sicfun.holdem.types.GameState  (PROHIBITED - GameState does NOT enter)
sicfun.holdem.strategic  X-> sicfun.holdem.engine
sicfun.holdem.strategic  X-> sicfun.holdem.runtime
```

Anti-corruption layer:

```
sicfun.holdem.strategic.bridge  ->  sicfun.holdem.strategic  (formal types)
sicfun.holdem.strategic.bridge  ->  sicfun.holdem.engine     (current engine)
sicfun.holdem.strategic.bridge  ->  sicfun.holdem.types      (GameState, for conversion)
```

Only `bridge` sees both worlds. The conversion `GameState -> PublicState` lives
in bridge, not in strategic.

### 1.3 Dependency Rule (non-negotiable)

```
Current engine  X->  Formal semantics
Formal semantics  ->  Bridge contracts  ->  Current engine
```

The engine never imports `strategic`. Only `bridge` connects both worlds.

### 1.4 Multiway-Native Mandate

> **Heads-up is the subcase |P\{S}| = 1 of the multiway domain.**
>
> **No implicit collapse of multiway to pairwise is permitted within the formal
> semantics or the bridge**, except via a local, named transformation explicitly
> marked as an approximation.

`Pid` in `AugmentedState[Pid]` and `OperativeBelief[Pid]` is not a cosmetic
parameter. It is the index of a **finite family of active rivals**. Instantiating
a singular `Pid` to re-introduce heads-up through the back door is a violation.

### 1.5 Bridge Fidelity Policy

Each formal-to-engine correspondence declares one of:

| Fidelity | Meaning |
|---|---|
| **Exact** | Direct, semantically faithful representation in the current engine |
| **Approximate** | Mapping with documented loss or simplification |
| **Absent** | No realization in the engine; bridge provides typed stub or error |

### 1.6 What Is Inside the Formal Layer

| Formal object | Scala type | Spec ref |
|---|---|---|
| Player identity | `PlayerId` opaque type | -- |
| Table structure | `TableMap[Pid, A]` | -- |
| Rival family | `RivalMap[A]` | -- |
| Public state x^pub | `PublicState` (own type) | Def. 9 |
| Private state x^priv | `HoleCards` (reuse from types) | Def. 9 |
| Rival latent type theta^{R,i} | `OpponentType` trait, per-rival | A3 |
| Own statistical sufficiency xi^S | `OwnEvidence` (global + per-rival + relational) | A4 |
| Rival mental state m^{R,i} | `OpponentModelState` per-rival | A6-A7 |
| Augmented state X_tilde | `AugmentedState` | Def. 8-9 |
| Operative belief b_tilde | `OperativeBelief` | Def. 10 |
| Private strategic classes C^S | `StrategicClass` enum | Def. 1-2 |
| Size-aware public signal Y^act | `PublicSignal` (action + sizing + timing) | Def. 3-4 |
| Real baseline pi^{0,S} | `RealBaseline` trait (singular) | Def. 5 |
| Attributed baseline pi_hat^{0,S,i} | `AttributedBaseline` per-rival | Def. 6 |
| Derived reputation phi^{S,i} | `ReputationProjection` per-rival, computed | Def. 7 |
| Counterfactual baseline pi^{cf,S} | `CounterfactualBaseline` trait | Def. 14 |
| Spot-conditioned polarization | `SpotPolarization` | Def. 13, A9 |
| Strategic value Q^{Gamma^R} | `StrategicValue` (Qmy + Delta_learn + Delta_sig) | Def. 17-21 |
| Signal decomposition Delta_sig | `SignalDecomposition` (Delta_pass + Delta_manip) | Def. 20 |
| Structural/exploitative bluff | **Derived predicates**, not primitive types | Def. 15-16 |

### 1.7 What Is Outside the Formal Layer

- `GameState` and its current shape
- Equity computation (GPU/CPU runtimes)
- Concrete CFR solver
- Caching, scheduling, telemetry
- Persistence, training data IO
- UI/CLI/debug

---

## 2. Formal Types, Invariants, and Coherence Laws

### 2.1 Identity and Table Structure

```scala
opaque type PlayerId = String

/** Full table: includes hero. Preserves positional order. */
final case class TableMap[A](
    hero: PlayerId,
    seats: Vector[(PlayerId, Position, SeatStatus, A)]
)

/** Rivals only: NEVER includes hero. */
final case class RivalMap[A](
    rivals: Vector[(PlayerId, Position, SeatStatus, A)]
)

enum SeatStatus:
  case Active, Folded, AllIn, SittingOut
```

**Invariants:**
- `TableMap`: `hero` always in `seats`; `PlayerId` uniqueness; positional order SB->BTN
- `RivalMap`: NEVER contains hero; `PlayerId` uniqueness; positional order preserved
- `RivalMap` derived from `TableMap` by excluding hero + filtering status

**Law L1 (cardinality):** `|RivalMap.rivals.filter(active)| >= 1`. Heads-up is `== 1`.

### 2.2 Domain Types

```scala
opaque type Chips >: Nothing = Double
opaque type PotFraction >: Nothing = Double
opaque type Ev >: Nothing = Double
```

Within `strategic`, these types are used instead of raw `Double`. The bridge
converts `Double` from the current engine to/from these opaques.

### 2.3 Public State

```scala
final case class PublicState(
    street: Street,
    board: Board,
    pot: Chips,
    stacks: TableMap[Chips],
    bettingRound: BettingContext,
    actionHistory: Vector[PublicAction]
)

final case class PublicAction(
    actor: PlayerId,
    signal: PublicSignal
)

final case class PublicSignal(
    action: PokerAction.Category,
    sizing: Option[Sizing],
    timing: Option[TimingBucket]
):
  def isAggressiveWager: Boolean = sizing.isDefined

final case class Sizing(
    absolute: Chips,
    fractionOfPot: PotFraction
)
```

**Invariants:**
- `pot >= 0`, all stacks `>= 0`
- `actionHistory` chronological and consistent with `street`
- `sizing.isDefined <-> isAggressiveWager` (biconditional)

`PublicState` is an own type of `strategic`. NOT an alias of `GameState`.

### 2.4 Private Strategic Classes

```scala
enum StrategicClass:
  case Value      // C^V
  case Bluff      // C^B
  case Marginal   // C^M
  case SemiBluff  // C^SB
```

**Invariants:**
- Exhaustive partition: every hand in a spot belongs to exactly one class
- Classification depends on spot: `classifyHand(hand, publicState, baseline) -> StrategicClass`
- Not a static property of the hand; it is a function of context
- `classifyHand` must have a **deterministic tie-breaking rule** at ambiguous boundaries

**Derived predicates (not new types):**

```scala
def isStructuralBluff(cls: StrategicClass, signal: PublicSignal): Boolean =
  cls == StrategicClass.Bluff && signal.isAggressiveWager

def isExploitativeBluff(cls: StrategicClass, signal: PublicSignal, deltaManip: Ev): Boolean =
  isStructuralBluff(cls, signal) && deltaManip > Ev(0.0)
```

**Law L2 (exploitative bluff subset of structural):**
`isExploitativeBluff(c, s, d) -> isStructuralBluff(c, s)` -- always.

### 2.5 Per-Rival State (Multiway Families)

```scala
trait OpponentType:
  def space: String

final case class OpponentModelState(
    opponentType: DiscreteDistribution[OpponentType],
    beliefState: RivalBeliefState,
    attributedBaseline: AttributedBaseline
)

trait RivalBeliefState:
  /** Deterministic update: m_{t+1} = Gamma^R(m_t, Y_t) */
  def update(signal: PublicSignal, context: PublicState): RivalBeliefState
```

**Key decision:** `playerId` eliminated from `OpponentModelState`. Identity lives
in the outer container (`RivalMap`), not in the contained object.

**Invariants:**
- `opponentType` is a valid distribution (sum = 1, all >= 0)
- `beliefState.update` is deterministic given (signal, context) -- A7
- Each rival has its own `OpponentModelState`; they are not shared

**Law L3 (sufficiency of m^{R,i}):** The future policy of rival `i` depends on
past public history only through `beliefState`.

### 2.6 Derived Reputation

```scala
trait ReputationProjection:
  def project(rivalState: OpponentModelState): ReputationView

final case class ReputationView(
    perceivedTightness: PotFraction,
    perceivedAggression: PotFraction,
    perceivedBluffFrequency: PotFraction,
    raw: Map[String, Double] = Map.empty
)
```

**Law L4 (derivability):** phi^{S,i} is NEVER stored as independent state.
Always computed from `OpponentModelState.beliefState`.
`m1 == m2 -> project(m1) == project(m2)`.

**Law L5 (per-rival):** In multiway, each rival may have a different image of
SICFUN. No global reputation exists unless explicitly derived as a named aggregation.

### 2.7 Own Statistical Sufficiency (xi^S)

```scala
final case class OwnEvidence(
    global: GlobalEvidence,
    perRival: RivalMap[RivalEvidence],
    relational: Map[(PlayerId, PlayerId), RelationalEvidence]
)

trait GlobalEvidence
trait RivalEvidence
trait RelationalEvidence
```

**Law L6 (joint sufficiency, corrected):** The pair `(PublicState, xi^S)` contains
all information necessary for SICFUN's own update. `xi^S` is reserved for:
- Accumulated evidence NOT contained in the current public state
- Private information (past hole cards, private Bayesian inferences)
- Own historical summaries

`xi^S` does NOT duplicate information already present in `PublicState`.

### 2.8 Augmented State

```scala
final case class AugmentedState(
    publicState: PublicState,
    privateHand: HoleCards,
    opponents: RivalMap[OpponentModelState],
    ownEvidence: OwnEvidence
)
```

**Invariants:**
- `opponents` has exactly the active rivals (Active + AllIn)
- PlayerId in `opponents` is a subset of PlayerId in `publicState.stacks`
- Does NOT contain phi^{S,i} as a field -- it is derived
- Does NOT contain `playerId` duplicated inside `OpponentModelState`

**Law L7 (multiway-native):** `AugmentedState` has no singular "the opponent"
field. Everything is an indexed family via `RivalMap`. Heads-up is `|opponents| == 1`.

**Law L8 (no implicit collapse):** No operator on `AugmentedState` may silently
reduce the rival family to a single opponent. If pairwise simplification is needed,
it must be a local, named transformation marked as `Approximate`.

### 2.9 Strategic Value Decomposition

```scala
final case class StrategicValue(
    qImmediate: Ev,
    deltaLearn: Ev,
    deltaSignal: SignalDecomposition
):
  def total: Ev = qImmediate + deltaLearn + deltaSignal.total

final case class SignalDecomposition(
    deltaPassive: Ev,
    deltaManipulate: Ev
):
  def total: Ev = deltaPassive + deltaManipulate
```

**Law L9 (exact decomposition):** `Q^{Gamma^R} = Q^my + Delta_learn + Delta_pass + Delta_manip`.
Exact sum, no residual.

**Law L10 (damaging leak):** `deltaPassive < 0 -> leaked information harms SICFUN`.

**Law L11 (prototypical exploitative bluff):**
`class == Bluff && deltaManipulate > 0 -> exploitative bluff`. Consistent with L2.

### 2.10 Canonical Operators (Mini-Algebra)

| Operator | Conceptual signature | Spec ref |
|---|---|---|
| `updateBelief` | `(OperativeBelief, action, observation) -> OperativeBelief` | Def. 11 |
| `updateRivalBelief` | `(OpponentModelState, PublicSignal, PublicState) -> OpponentModelState` | Def. 12, A7 |
| `classifyHand` | `(HoleCards, PublicState, RealBaseline) -> StrategicClass` | Def. 1-2 |
| `projectReputation` | `(OpponentModelState) -> ReputationView` | Def. 7 |
| `computePolarization` | `(Sizing, PublicState, RealBaseline, OpponentModelState) -> SpotPolarization` | Def. 13, A9 |
| `decomposeValue` | `(AugmentedState, PokerAction, Sizing?) -> StrategicValue` | Def. 17-21 |
| `decomposeSignal` | `(AugmentedState, PublicSignal) -> SignalDecomposition` | Def. 20 |
| `isStructuralBluff` | `(StrategicClass, PublicSignal) -> Boolean` | Def. 15 |
| `isExploitativeBluff` | `(StrategicClass, PublicSignal, Ev) -> Boolean` | Def. 16 |
| `collapseToHeadsUpLocal` | `(AugmentedState) -> Approximation[AugmentedState]` | **Typed as approximation** |

**Approximation wrapper:**

```scala
final case class Approximation[A](
    value: A,
    fidelity: Fidelity,
    lossDescription: String
)

enum Fidelity:
  case Exact, Approximate, Absent
```

`collapseToHeadsUpLocal` returns `Approximation[AugmentedState]`, NEVER a bare
`AugmentedState`. This makes it impossible for a simplification to re-enter the
system as if it were canonical state.

**Law L12 (determinism of Gamma^R):** `updateRivalBelief` is a pure function.
Same input -> same output. No external mutable state.

**Law L13 (conditional independence of rival update, corrected):** The update of
`m^{R,i}` is a pure function of `(m^{R,i}, Y_t, PublicState)`. Any dependency
on other rivals must enter ONLY explicitly: through `PublicState`, or through
an additional formalized shared state. No hidden coupling between `m^{R,i}` and
`m^{R,j}`.

---

## 3. Anti-Corruption Layer and Gap Matrix

### 3.1 ACL Principles

1. **The engine does not redefine semantics.** If a bridge cannot map faithfully,
   it declares the loss -- it does not invent semantics.
2. **Each bridge declares fidelity and severity.** No grey bridges.
3. **Approximate requires documented loss.** Absent is not an error -- it is
   explicit declaration.
4. **Absence is represented as absence, not as zero.** No bridge fills absent
   values with neutrals (0, implicit None, etc.) as if they were truth.
5. **One-way only.** Bridges convert `engine -> formal` and `formal -> engine`.
   They never modify the formal layer to fit.
6. **Fidelity is declared relative to a concrete formal realization** (e.g.,
   "archetype-5 profile"), not relative to the open ideal of the trait.

### 3.2 BridgeResult (Typed)

```scala
sealed trait BridgeResult[+A]:
  def fidelity: Fidelity

object BridgeResult:
  final case class Exact[A](value: A) extends BridgeResult[A]:
    def fidelity: Fidelity = Fidelity.Exact

  final case class Approximate[A](value: A, lossDescription: String) extends BridgeResult[A]:
    def fidelity: Fidelity = Fidelity.Approximate

  final case class Absent(reason: String) extends BridgeResult[Nothing]:
    def fidelity: Fidelity = Fidelity.Absent
```

A bridge `Approximate` or `Absent` NEVER returns a bare canonical object. The
loss travels with the value.

**Consumption safety:** Consuming a `BridgeResult.Approximate` or `BridgeResult.Absent`
as if it were `Exact` without explicit check is prohibited. Enforcement is either:
- **Disciplinary/runtime:** public constructors, consumers must pattern match.
- **Type-closed:** private constructors, API via `fold[B](onExact, onApprox, onAbsent)`.

The choice is fixed at implementation time. The design contract is:

> **No `BridgeResult.Approximate` or `BridgeResult.Absent` may be consumed as
> `Exact` without explicit check.**

### 3.3 Bridge Contract

```scala
/** Source is a declarative aggregate per bridge, not a single engine object. */
trait BridgeContract[Source, Target]:
  def convert(source: Source): BridgeResult[Target]

/** Reversibility only for Exact bridges. */
trait ExactBridge[Source, Target] extends BridgeContract[Source, Target]:
  def reverse(target: Target): Source
```

Example source aggregates:

```scala
final case class PublicStateSource(
    gameState: GameState,
    seatInfo: Vector[(Int, Position, SeatStatus)],
    playerIdMapping: Map[Int, PlayerId],
    actionLog: Vector[BetAction]
)

final case class OpponentModelSource(
    archPosterior: ArchetypePosterior,
    raiseResponses: RaiseResponseCounts,
    showdownHistory: Vector[ShowdownRecord],
    villainObservations: Seq[VillainObservation]
)

final case class ValueSource(
    chipEvResult: Double,
    action: PokerAction,
    state: GameState,
    heroEquity: Double
)
```

### 3.4 Severity

Separate from fidelity. Measures the impact of loss on the formal model's ability
to produce correct decisions.

```scala
enum Severity:
  case Cosmetic    // loss does not affect decisions
  case Behavioral  // affects decision quality but not reasoning structure
  case Structural  // affects reasoning structure or model coherence
  case Critical    // invalidates a semantic law or makes a canonical operator inoperable
```

### 3.5 ACL Structure

```
sicfun.holdem.strategic.bridge/
  PublicStateBridge         // PublicStateSource -> PublicState
  OpponentTypeBridge        // PlayerArchetype -> OpponentType (profile: archetype-5)
  OpponentModelBridge       // OpponentModelSource -> OpponentModelState
  BaselineBridge            // EquilibriumBaselineConfig + HoldemCfrSolution -> RealBaseline
  EvidenceBridge            // OpponentProfile + observations -> OwnEvidence
  ValueBridge               // ValueSource -> StrategicValue (partial, no deltas)
  SignalBridge              // PokerAction + pot context -> PublicSignal
  ClassificationBridge      // (equity, board, hand, baseline) -> StrategicClass
  PolarizationBridge        // (no antecedent -- Absent stub)
  BridgeManifest            // typed registry of all correspondences + fidelities + severities
```

### 3.6 Gap Matrix

| Formal object | Engine source | Bridge | Fidelity | Severity | Loss / Note |
|---|---|---|---|---|---|
| PublicState | `GameState` + seat info + action log | PublicStateBridge | Approximate | Behavioral | `GameState` uses `Int` seat index, not `PlayerId`. `actionHistory` reconstructed from `betHistory` with identity loss. No timing. |
| PublicSignal | `PokerAction` + pot context | SignalBridge | Approximate | Behavioral | `fractionOfPot` computed exactly from (amount, pot). Timing absent (`None`). |
| Sizing | `Raise(amount)` + `GameState.pot` | SignalBridge | Exact* | Cosmetic | `absolute` is original data; `fractionOfPot` is exact derivation. (*Exact within "chips + pot-fraction" profile.) |
| TimingBucket | -- | -- | Absent | Behavioral | No timing data collected. Stub: `None`. |
| StrategicClass | -- | ClassificationBridge | Absent | Critical | No hand classifier exists. `classifyHand`, `isStructuralBluff`, `isExploitativeBluff` inoperable. |
| OpponentType | `PlayerArchetype` (5 enums) | OpponentTypeBridge | Exact* | Cosmetic | Exact within "archetype-5" profile. Does not exhaust the abstract space. |
| OpponentModelState | `ArchetypePosterior` + `RaiseResponseCounts` + showdowns + observations | OpponentModelBridge | Approximate | Critical | `ArchetypePosterior` captures theta^R partially. No real `RivalBeliefState` (m^R). `AttributedBaseline` absent. |
| RivalBeliefState | `ArchetypeLearning.updatePosterior` | OpponentModelBridge | Approximate | Critical | Update operates only on action category (fold/call/raise), ignores sizing and spot context. Kernel Gamma^R severely reduced. |
| AttributedBaseline | -- | -- | Absent | Structural | No model of what baseline each rival attributes to us. |
| RealBaseline | `EquilibriumBaselineConfig` + CFR blend | BaselineBridge | Approximate | Structural | Baseline is a configurable adaptive/GTO blend, not a pure GTO policy. |
| CounterfactualBaseline | -- | -- | Absent | Structural | No counterfactual non-manipulative reference. Needed for Delta_manip. |
| ReputationProjection | -- | -- | Absent | Structural | Derivable when `OpponentModelState` has real `beliefState`. Currently inoperable. |
| OwnEvidence | `OpponentProfile` (partial) | EvidenceBridge | Approximate | Behavioral | `actionSummary` + `raiseResponses` + `showdownHands` cover `RivalEvidence` partially. `GlobalEvidence` and `RelationalEvidence` absent. |
| OperativeBelief | Dispersed beliefs (`DiscreteDistribution[HoleCards]`, `ArchetypePosterior`) | -- | Absent | Critical | No unified belief over `AugmentedState`. |
| StrategicValue | `ActionValueModel.ChipEv` | ValueBridge | Approximate | Critical | Only `qImmediate` via `ChipEv.expectedValue`. `deltaLearn` and `deltaSignal` are `Option.None`, NOT zero. |
| SignalDecomposition | -- | -- | Absent | Critical | No signal decomposition. Blocked by CounterfactualBaseline. |
| SpotPolarization | -- | -- | Absent | Structural | No spot-conditioned polarization concept. |

### 3.7 Summary by Severity

| Severity | Exact | Approximate | Absent |
|---|---|---|---|
| Cosmetic | Sizing, OpponentType | -- | -- |
| Behavioral | -- | PublicState, PublicSignal, OwnEvidence | TimingBucket |
| Structural | -- | RealBaseline | AttributedBaseline, CounterfactualBaseline, ReputationProjection, SpotPolarization |
| Critical | -- | OpponentModelState, RivalBeliefState, StrategicValue | StrategicClass, OperativeBelief, SignalDecomposition |

**Reading:** 6 objects are Critical (3 Approximate, 3 Absent). These are the net
new work of v0.26: without them, strategic value decomposition and exploitative
bluff detection are inoperable.

### 3.8 BridgeManifest (Auditable Code)

```scala
final case class ManifestEntry(
    formalObject: String,
    fidelity: Fidelity,
    severity: Severity,
    bridge: Option[String],
    lossDescription: String
)

object BridgeManifest:
  val entries: Vector[ManifestEntry] = Vector(
    ManifestEntry("PublicState",            Fidelity.Approximate, Severity.Behavioral,  Some("PublicStateBridge"),       "Int seat index, not PlayerId; action history reconstructed; no timing"),
    ManifestEntry("PublicSignal",           Fidelity.Approximate, Severity.Behavioral,  Some("SignalBridge"),            "fractionOfPot derived exactly; timing absent"),
    ManifestEntry("Sizing",                 Fidelity.Exact,       Severity.Cosmetic,    Some("SignalBridge"),            "Exact within chips+pot-fraction profile"),
    ManifestEntry("TimingBucket",           Fidelity.Absent,      Severity.Behavioral,  None,                           "No timing data collected"),
    ManifestEntry("StrategicClass",         Fidelity.Absent,      Severity.Critical,    Some("ClassificationBridge"),    "No hand classifier exists; operators inoperable"),
    ManifestEntry("OpponentType",           Fidelity.Exact,       Severity.Cosmetic,    Some("OpponentTypeBridge"),      "Exact within archetype-5 profile"),
    ManifestEntry("OpponentModelState",     Fidelity.Approximate, Severity.Critical,    Some("OpponentModelBridge"),     "Partial theta^R; no real m^R; no attributed baseline"),
    ManifestEntry("RivalBeliefState",       Fidelity.Approximate, Severity.Critical,    Some("OpponentModelBridge"),     "Update ignores sizing and spot context"),
    ManifestEntry("AttributedBaseline",     Fidelity.Absent,      Severity.Structural,  None,                           "No per-rival attributed baseline"),
    ManifestEntry("RealBaseline",           Fidelity.Approximate, Severity.Structural,  Some("BaselineBridge"),          "Adaptive/GTO blend, not pure GTO policy"),
    ManifestEntry("CounterfactualBaseline", Fidelity.Absent,      Severity.Structural,  None,                           "No counterfactual reference; blocks Delta_manip"),
    ManifestEntry("ReputationProjection",   Fidelity.Absent,      Severity.Structural,  None,                           "Derivable when beliefState is real; currently inoperable"),
    ManifestEntry("OwnEvidence",            Fidelity.Approximate, Severity.Behavioral,  Some("EvidenceBridge"),          "OpponentProfile partial; global/relational absent"),
    ManifestEntry("OperativeBelief",        Fidelity.Absent,      Severity.Critical,    None,                           "No unified belief over augmented state"),
    ManifestEntry("StrategicValue",         Fidelity.Approximate, Severity.Critical,    Some("ValueBridge"),             "Only Q^my; Delta_learn and Delta_sig are Option.None, NOT zero"),
    ManifestEntry("SignalDecomposition",    Fidelity.Absent,      Severity.Critical,    None,                           "No signal decomposition; blocked by CounterfactualBaseline"),
    ManifestEntry("SpotPolarization",       Fidelity.Absent,      Severity.Structural,  None,                           "No polarization concept"),
  )
```

Testable: a test traverses the manifest and verifies that each `Absent` entry
has a stub returning `BridgeResult.Absent`, and each `Approximate` entry has a
non-empty `lossDescription`.

---

## 4. Validation Plan

### 4.1 Formal Law Tests

Tests on the pure formal layer, no engine or bridges involved.

| Test | Law | Validates | Enforcement |
|---|---|---|---|
| CardinalityLaw | L1 | `RivalMap` with 0 rivals fails construction | Runtime: require |
| BluffSubsetLaw | L2 | `isExploitativeBluff -> isStructuralBluff` | Property-based |
| ReputationDerivabilityLaw | L4 | `m1 == m2 -> project(m1) == project(m2)` | Property-based |
| ReputationPerRivalLaw | L5 | Two rivals with different states can have different reputations | Explicit counterexample |
| OwnEvidenceNoPublicDuplication | L6 | `OwnEvidence` has no board/pot/stacks | Structural inspection |
| MultiwayNativeLaw | L7 | No singular `.opponent` field | Compile-time |
| NoImplicitCollapseLaw | L8 | `collapseToHeadsUpLocal` returns `Approximation` | Compile-time |
| ValueDecompositionExact | L9 | `total == qImmediate + deltaLearn + deltaSignal.total` | Arithmetic (epsilon) |
| PassiveLeakDetection | L10 | `deltaPassive < 0` identified as leak | Constructed case |
| ExploitativeBluffCorollary | L11 | `Bluff && deltaManipulate > 0 -> exploitative` | Consistency with L2 |
| RivalUpdateDeterminism | L12 | `updateRivalBelief(m, s, ctx)` is **deterministic** | Same input -> same output |
| RivalUpdatePurity | L12 | `updateRivalBelief` does not read global state, mutable caches, or singletons | Runtime: same input, multiple repetitions, same serialized output in clean context |
| RivalUpdateIndependence | L13 | `update` signature does not accept other `OpponentModelState` | Compile-time |
| StrategicClassPartition | Inv 2.4 | `classifyHand` assigns exactly one class | Property-based |
| ClassifyDeterministicTiebreak | Inv 2.4 | `classifyHand` is deterministic at boundaries | Same input -> same class |

### 4.2 Bridge Tests (ACL)

| Test | Bridge | Validates |
|---|---|---|
| PublicStateBridgeRoundTrip | PublicStateBridge | Known GameState -> PublicState -> verifiable fields |
| PublicStateBridgeLoss | PublicStateBridge | Result is `Approximate` with non-empty lossDescription |
| SignalBridgeSizingDerivation | SignalBridge | `Raise(100)` with pot=200 -> `Sizing(100, 0.5)` |
| SignalBridgeTimingAbsent | SignalBridge | Timing always `None`; fidelity notes absence |
| OpponentTypeBridgeExactProfile | OpponentTypeBridge | Each `PlayerArchetype` -> `OpponentType` with `space == "archetype-5"`, result `Exact` |
| OpponentModelBridgeApproximate | OpponentModelBridge | Result is `Approximate` with documented loss |
| ValueBridgeNoFakeZeros | ValueBridge | `deltaLearn` and `deltaSignal` are `Option.None`, NOT `Some(Ev(0.0))` |
| ValueBridgeFidelity | ValueBridge | Result is `Approximate` with severity `Critical` |
| ClassificationBridgeAbsent | ClassificationBridge | Returns `BridgeResult.Absent` with reason |
| AbsentBridgesFailCleanly | All Absent | Every `Absent` entry in `BridgeManifest` produces `BridgeResult.Absent`, not exception |
| ManifestCompleteness | BridgeManifest | Every formal object in section 2 has a manifest entry |

### 4.3 Non-Regression Tests

Two modes to prove the formal layer is strictly additive:

- **Mode 1 -- ACL present, not consulted:** `strategic` and `bridge` exist in
  classpath but no engine path invokes them. Exact parity.
- **Mode 2 -- ACL in shadow mode:** Bridges execute in parallel (construct
  `AugmentedState`, compute partial `StrategicValue`) but output does not
  influence decisions. Exact parity.

| Test | Mode | Validates |
|---|---|---|
| AdaptiveEngineDecisionParity | 1 + 2 | `RealTimeAdaptiveEngine.decide()` same action with fixed seed |
| CfrBaselineStability | 1 | `GtoSolveEngine` same solution |
| ArchetypeLearningStability | 1 | `ArchetypeLearning.updatePosterior` unchanged |
| PlayingHallOutputParity | 1 + 2 | `TexasHoldemPlayingHall` session with fixed seed same result |
| ShadowModeNoSideEffects | 2 | Shadow mode does not mutate engine state, caches, or timing |

### 4.4 Validation Corpus

> **Every divergence harness executes on a reproducible, versioned corpus with
> fixed seeds.**

Minimum corpus dimensions:

| Dimension | Required values |
|---|---|
| Table format | Heads-up, 6-max, 9-max |
| Street | Preflop, Flop, Turn, River |
| Spot type | Single raise, 3-bet, limp, check-through |
| Sizing | Small (1/3 pot), Medium (2/3 pot), Large (pot+), Overbet (2x+) |
| Stack depth | Short (<20bb), Medium (50-100bb), Deep (200bb+) |
| Terminal | Fold, showdown, all-in side pots |
| Showdown data | With and without prior rival showdowns |
| Seeds | Fixed, documented, at least 3 per configuration |

**Location:** `data/v026-validation-corpus/` -- versioned in git.

**Format:** Each spot is a serializable Scala fixture producing a `GameState` +
complete context for the bridge.

### 4.5 Divergence Harness

Not pass/fail. Instrumentation that generates metric reports.

| Harness | Measures | Criterion |
|---|---|---|
| ValueDecompositionDivergence | Delta between `ChipEv.expectedValue` (engine) vs `StrategicValue.qImmediate` (bridge) over full corpus. When Delta_learn and Delta_sig are implemented: how much the optimal decision changes. | Distribution of deltas (mean, p50, p95, max). Spots where decision changes. |
| ClassificationCoverage | When classifier exists: (a) **structural coverage** -- every hand gets exactly one class, deterministic, no multiple classes; (b) **empirical plausibility** -- distribution does not collapse to one category, responds to spot, changes with board texture/equity/baseline. | (a) hard pass/fail. (b) metric + smoke test. |
| RivalBeliefDrift | Delta between `ArchetypeLearning.updatePosterior` (engine, ignores sizing) vs `updateRivalBelief` (formal, with sizing/spot). Accumulated posterior divergence. | KL divergence per hand, accumulated drift per session. |
| BaselinePolicyGap | Action frequency by spot: current CFR blend vs pure `RealBaseline`. | Frequency distributions, spots with highest discrepancy. |
| MultiwayShadow | Constructs `AugmentedState` via bridges during multiway sessions. Verifies: (a) `opponents.rivals.size == activeOrAllInPlayersExcludingHero`; (b) `RivalMap` never contains hero; (c) no implicit collapse to heads-up; (d) positional order preserved; (e) rival identities stable across updates. | (a-e) hard pass/fail. Coverage metrics by format (6-max, 9-max). |
| ReputationPhantom | When `ReputationProjection` exists: convergence of derived reputation; two rivals with different histories produce different reputations. Smoke test of L4 + L5. | Convergence: metric. Differentiation: pass/fail. |

**Harness output:** Metric report with distributions, not binary verdict. Feeds
the implementation roadmap: gaps with highest divergence add most value when closed.

### 4.6 Implementation Order

1. **Formal law tests (4.1)** -- written alongside section 2 types. Many are compile-time.
2. **Corpus (4.4)** -- minimum fixtures before writing bridges.
3. **Bridge tests (4.2)** -- written alongside each bridge.
4. **Non-regression (4.3)** -- snapshot engine behavior before connecting bridges. Mode 1 first, mode 2 after.
5. **Divergence harness (4.5)** -- last. Requires functional bridges and corpus. Generates data, not verdicts.
