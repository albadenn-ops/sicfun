# Attributed Baseline (Def 10) — Design Spec

**Date:** 2026-04-13
**Branch:** `feat/adaptive-proof-harness-9max`
**Goal:** Close the last `Severity.Structural` gap in `BridgeManifest` by implementing a spec-consistent `AttributedBaseline` (Def 10) with kernel coupling (Def 18).

---

## 1. Problem Statement

The `BridgeManifest` declares `AttributedBaseline` (Def 10) as `Severity.Structural` —
the only remaining structural gap. The spec defines:

```
hat{pi}^{0,S,i}(a, lambda | c, x^pub, m^{R,i})
```

A per-rival, state-conditioned baseline policy that conditions on the rival's belief
state. It feeds into the attrib action kernel (Def 18):

```
Gamma^{act,attrib,i}(m,y,x^pub) = BuildRivalKernel^i_{kappa,delta}(hat{pi}^{0,S,i}_{x,m})(m,y,x^pub)
```

Currently:
- `trait AttributedBaseline` exists with a signature (Baseline.scala:32)
- `OpponentModelState.attributedBaseline` field exists — always `None`
- `BaselineBridge.toAttributedBaselines` does a naive equity split, not kernel-based attribution
- No concrete `AttributedBaseline` implementation exists

## 2. Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Full kernel coupling (Def 18) | Spec-literal: attrib kernel uses attributed baseline |
| Fixed-point strategy | B1: layered initialization | Causal ordering: baseline at t uses m_t, kernel produces m_{t+1} |
| Per-rival mechanism | Call-time differentiation | Single shared baseline instance; per-rival behavior from call-time rivalState |
| Kernel coupling point | buildKernelProfile() attrib likelihood | Only the attrib likelihood varies per rival; ref/blind/design kernels stay shared |
| State ownership | Stateless over belief; derive on demand | No mutable field; avoids stale-state bugs across SessionState rebuilds |
| Trait boundary | Generic RivalBeliefState; narrow in implementation | AttributedBaseline trait stays open; PosteriorAttributedBaseline pattern-matches to StrategicRivalBelief |

## 3. Core Formula

For rival i with belief state m^{R,i}, class c, public state x^pub, action a:

```
hat_pi(a | c, x, m) = pi0(a | c, x) * w(a, x, m) / Z(c, x, m)
```

Where:

- `pi0(a | c, x)` = `config.actionPriors((c, a.category))` — the real baseline (Def 9)
- `w(a, x, m)` = posterior-predictive uplift:

```
w(a, x, m) = p_pred(a | x, m) / p_ref(a | x)

p_pred(a | x, m) = sum_{c'} P(c' | m) * pi0(a | c', x)
p_ref(a | x)     = (1/|C|) * sum_{c'} pi0(a | c', x)
```

- `Z(c, x, m)` = normalization constant:

```
Z(c, x, m) = sum_{a'} pi0(a' | c, x) * w(a', x, m)
```

**Normalization support:** All four coarse action categories `{Fold, Call, Check, Raise}`.
Same support in `p_pred`, `p_ref`, and `Z`. Legal-action filtering is a downstream
concern in the solver, not in the baseline.

**Properties:**
- Uniform posterior → w = 1 for all actions → hat_pi = pi0 (reduces to real baseline)
- Point mass on c* → uplift emphasizes c*-typical actions across all classes
- For fixed (c, x, m), sum over actions = 1 (proper distribution)
- Action-dependent: w varies by action a (no cancellation under normalization)
- No class posterior injected into hat_pi → no double-counting with Def 15B

**Guard:** Both `p_pred` and `p_ref` get an epsilon floor of `1e-10` before division.

**Reference marginal:** Uses `(1/|C|)` matching the ref world's uniform class prior
(current engine convention). If the ref world's reference prior changes, `p_ref` must
use the same prior `rho_ref(c)`.

**Sizing:** Initial implementation uses action category only (matching current
`actionPriors` granularity). The formula generalizes to `(a, lambda)` support when
sizing-conditioned priors are available.

## 4. Components

### 4.1 PosteriorAttributedBaseline (new file)

`src/main/scala/sicfun/holdem/strategic/PosteriorAttributedBaseline.scala`

```scala
class PosteriorAttributedBaseline(
    actionPriors: Map[(StrategicClass, PokerAction.Category), Double]
) extends AttributedBaseline
```

- **Stateless over belief.** Captures only `actionPriors` (immutable config).
- Per-rival differentiation from call-time `rivalState`.
- Pattern-matches `rivalState`:
  - `StrategicRivalBelief` → uses `typePosterior` for uplift computation
  - other `RivalBeliefState` → returns `pi0(a | c)` unchanged (no attribution)
- Single instance reusable across all rivals.

### 4.2 AttributedBaseline trait widening (breaking migration)

`src/main/scala/sicfun/holdem/strategic/Baseline.scala`

Current:
```scala
trait AttributedBaseline:
  def probability(cls, action, sizing, street: Street, rivalState: RivalBeliefState): Double
```

New:
```scala
trait AttributedBaseline:
  def probability(cls, action, sizing, publicState: PublicState, rivalState: RivalBeliefState): Double
```

Call sites to migrate:
- `OpponentModelState.attributedBaseline: Option[AttributedBaseline]` — type unchanged, `None` callers unaffected
- `BaselineBridge.toAttributedBaselines` — reworked (see §4.4)
- Test fixtures in `AugmentedStateTest`, `DynamicsTest` — pass `None`, no change needed
- `BridgeTest` — `toAttributedBaselines` tests updated

### 4.3 Per-rival attrib likelihood in buildKernelProfile

`src/main/scala/sicfun/holdem/engine/StrategicEngine.scala`

**New helper:**
```
buildAttribLikelihoodFromBaseline(baseline: AttributedBaseline): TemperedLikelihoodFn
```

Returns a `TemperedLikelihoodFn` whose `logLikelihood(cls, signal)` calls through to
the attributed baseline at call time. The returned closure captures only the baseline
(stateless); `PublicState` and `RivalBeliefState` arrive at call time via the
`TemperedLikelihoodFn` interface.

**In `buildKernelProfile()`:**
- Single shared attrib likelihood from `buildAttribLikelihoodFromBaseline`
- Only the attrib likelihood varies by rival at call time (via `rivalState`)
- Per-rival differentiation in the interpolated kernel assembly is from beta (exploitation), not from different likelihood instances

### 4.4 BaselineBridge rework

`src/main/scala/sicfun/holdem/strategic/bridge/BaselineBridge.scala`

Current: `toAttributedBaselines(perRivalEquity: Map[PlayerId, Double]): BridgeResult[Map[PlayerId, Ev]]`

New: `toAttributedBaseline(baseline: AttributedBaseline): BridgeResult[AttributedBaseline]`

Returns `BridgeResult.Approximate(baseline, "kernel-coupled posterior-predictive attribution")`.
The bridge annotates fidelity; it no longer transforms the data.

### 4.5 StrategicSnapshot exposure

`src/main/scala/sicfun/holdem/strategic/bridge/StrategicSnapshot.scala`

New field:
```scala
attributionEnabled: Boolean  // whether the engine used kernel-coupled attributed baselines
```

Simple boolean flag reflecting engine configuration. Not derived from posterior shape.
The flag means "the decision was made with `PosteriorAttributedBaseline` wired into the
attrib kernel path."

`buildSnapshot()` sets this flag based on whether the engine has an attributed baseline
configured (always true once wired; false only if the feature is explicitly disabled).

### 4.6 BridgeManifest downgrade

`src/main/scala/sicfun/holdem/strategic/bridge/BridgeManifest.scala`

**Gated on §4.4 existing and tested.** Final step:

```scala
BridgeEntry("AttributedBaseline", "Def 10", Fidelity.Approximate, Severity.Behavioral,
  "kernel-coupled posterior-predictive attribution; per-rival via PosteriorAttributedBaseline")
```

Zero structural gaps remaining after this.

## 5. Lifecycle (B1 Layered)

### 5.1 observeAction (action-channel update)

1. Enter with current beliefs `m_t^{R,i}`
2. `buildKernelProfile()` builds attrib kernel using the shared `PosteriorAttributedBaseline`
3. At call time, the attrib likelihood evaluates `hat_pi(a | c, x, m_t)` using `m_t` (pre-update)
4. `Dynamics.fullStep` uses those kernels → produces `m_{t+1}`
5. Session state updated with `m_{t+1}`

The kernel for step t uses the baseline derived from `m_t`. Post-update `m_{t+1}` is
only used by the *next* step.

### 5.2 endHand(showdown)

Showdown is a direct classification + posterior shift, **independent** of the
action-channel baseline. The showdown path updates beliefs (`m_{t+1}`), and the
next `observeAction` call derives the attributed baseline from those updated beliefs.

### 5.3 decide

`decide()` invokes the solver paths (WPomcp/PftDpw) which consume `rivalBeliefs` and
`actionPriors` directly. The attributed baseline does not change the solver formulations —
it changes how beliefs *evolved* to reach this point (via the kernel updates in prior
`observeAction` calls). The solver sees the result of kernel-coupled belief updates,
not the attributed baseline itself.

### 5.4 initSession

No special initialization. `PosteriorAttributedBaseline(config.actionPriors)` is
constructed once and stored in the engine. Under uniform initial beliefs, the attributed
baseline reduces to the real baseline (uplift w = 1).

## 6. Testing Strategy

1. **Unit: PosteriorAttributedBaseline**
   - Uniform posterior → returns `pi0(a | c)` for all classes and actions
   - Degenerate posterior on Value → uplift skews all classes toward Value-typical actions
   - Probabilities sum to 1.0 for each fixed (c, x, m)
   - Epsilon floor prevents division by zero when `p_ref` is near zero
   - Pattern-match fallback: non-StrategicRivalBelief returns `pi0` unchanged

2. **Unit: attrib likelihood from baseline**
   - Different rivalState posteriors → different likelihood values for the same observation
   - Uniform rivalState → matches the existing shared attrib likelihood output

3. **Integration: observeAction with attributed baseline**
   - After observeAction, beliefs updated by kernel using pre-update attributed baseline
   - Post-update beliefs differ (slightly) when using attributed vs. shared baselines
   - Multiple observeAction steps show progressive differentiation

4. **Integration: engine end-to-end**
   - Full hand: initSession → startHand → observeAction × N → decide → endHand
   - No crash, valid action returned, attributed baseline in effect

5. **Regression: all existing tests pass**
   - Engine (211), Strategic (614), CFR (48), Runtime/Types/Equity/GPU/Core (352), Web/Model (131)
   - No behavioral change for uniform-prior rivals (attributed baseline = real baseline)

6. **Closure: FormalClosureValidationTest**
   - `BridgeManifest.structuralGaps` returns empty vector
   - Manifest downgrade verified
   - All theorem/corollary tests still pass

7. **Migration: trait widening**
   - All call sites compile with `publicState: PublicState`
   - Existing `None` fixtures unchanged

## 7. File Map

| File | Change | Section |
|---|---|---|
| `src/main/scala/sicfun/holdem/strategic/PosteriorAttributedBaseline.scala` | **New** | §4.1 |
| `src/main/scala/sicfun/holdem/strategic/Baseline.scala` | Widen trait | §4.2 |
| `src/main/scala/sicfun/holdem/engine/StrategicEngine.scala` | Add helper, wire into buildKernelProfile | §4.3 |
| `src/main/scala/sicfun/holdem/strategic/bridge/BaselineBridge.scala` | Rework toAttributedBaseline | §4.4 |
| `src/main/scala/sicfun/holdem/strategic/bridge/StrategicSnapshot.scala` | Add attributionEnabled | §4.5 |
| `src/main/scala/sicfun/holdem/strategic/bridge/BridgeManifest.scala` | Downgrade severity | §4.6 |
| `src/test/scala/sicfun/holdem/strategic/PosteriorAttributedBaselineTest.scala` | **New** | §6.1 |
| `src/test/scala/sicfun/holdem/strategic/bridge/BridgeTest.scala` | Update baseline tests | §6.2 |
| `src/test/scala/sicfun/holdem/engine/StrategicEngineTest.scala` | Integration tests | §6.3-4 |
| `src/test/scala/sicfun/holdem/strategic/FormalClosureValidationTest.scala` | Closure assertion | §6.6 |

## 8. Non-Goals

- **Sizing-conditioned attribution:** Initial implementation uses coarse action categories.
  Generalizing to `(a, lambda)` support is future work when sizing-conditioned priors exist.
- **Per-class uplift `w(a, c, x, m)`:** Current uplift is shared across classes. Generalizing
  to class-dependent uplift uses the same normalization framework and is backward-compatible.
- **Solver formulation changes:** The attributed baseline does not change PftDpw/WPomcp
  solver formulations. It changes how beliefs evolve to reach the solver.
- **OpponentModelState population:** The `attributedBaseline` field in `OpponentModelState`
  is populated where the snapshot layer needs it, not eagerly in every dynamics step.
