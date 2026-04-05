# Strategic Decision Pipeline Design

## Goal

Wire the strategic module's formal POMDP machinery (Dynamics, Kernels, TemperedLikelihood, WPomcp solver) into the actual decision path as `HeroMode.Strategic` — a third mode alongside Adaptive and GTO. Fix the C++ WPomcp solver to be a proper factored POMCP with particle reweighting, then build the Scala orchestrator that feeds it.

## Context

The `sicfun.holdem.strategic` package contains ~5,100 LOC implementing a formal POMDP framework (Defs 15-56): tempered likelihood, rival kernels, belief dynamics, exploitation interpolation, detection predicates, and three native solvers (PFT-DPW, WPomcp, Wasserstein DRO). All components are built and tested in isolation, but none participate in the actual decision path.

The current decision path (`HeroDecisionPipeline` -> `RealTimeAdaptiveEngine`) uses Bayesian range inference + Monte Carlo equity + optional CFR blending. It has zero imports from the strategic module.

The C++ WPomcp solver (`WPomcpSolver.hpp`) has a critical gap: rival particles are static during tree search (never reweighted by observations), the observation hash is fake (`street * 1000 + hero_action`), and `update_factored_belief()` exists but is never called from `simulate()`. It's a lightweight MCTS over public state space, not a real factored POMCP.

## Architecture

```
sicfun.holdem.engine (modified + new)
  StrategicEngine (new)         — session/hand lifecycle, orchestrates the pipeline
  PokerPomcpFormulation (new)   — builds factored tabular model for the solver
  HeroDecisionPipeline (modified) — new Strategic branch
        |
        | imports
        v
sicfun.holdem.strategic (existing, unchanged)
  Dynamics, KernelConstructor, ExploitationInterpolation,
  TemperedLikelihood, DetectionPredicate, WPomcpRuntime, etc.
        |
        | imports (rule preserved: strategic never imports engine)
        v
sicfun.holdem.types + sicfun.core (shared types)
```

Dependency direction: `engine -> strategic -> types`. Strategic module's one-way dependency rule is preserved.

## C++ WPomcp Solver Fix

Four changes to `WPomcpSolver.hpp`:

### Change 1: Type-conditioned rival actions

Current: `rivalActionProbs[rival][action]` — flat, same distribution for all particles of a rival.

New: `rivalPolicy[type][pub_state][action]` — conditioned on the particle's `rival_type` and public state. During simulation, each particle's rival action is sampled from its type-specific distribution.

### Change 2: Real observations

Current: `obs_hash = street * 1000 + hero_action` — deterministic, not an observation.

New: observation is the joint rival action tuple. Hero observes what rivals do. Child nodes branch by `(hero_action, joint_rival_action_hash)`. This is the real information structure of poker.

### Change 3: Particle reweighting

Current: `update_factored_belief()` exists but is never called from `simulate()`.

New: After observing rival actions, call `update_factored_belief()` to reweight each rival's particles by `P(observed_action | particle.rival_type, pub_state)` from the rival policy table. Trigger `systematic_resample()` when ESS drops below threshold. The resampling code is already implemented and correct.

### Change 4: Factored tabular reward model

Current: `rewards[hero_action]` — pre-baked, no game dynamics.

New: Receive a factored tabular model from Scala with:
- `action_effects[action]` -> `(pot_delta_frac, is_fold, is_allin)` per action
- `showdown_equity[hero_bucket][rival_bucket]` -> equity for terminal reward
- `terminal_flags[pub_state][hero_action]` -> terminal condition

Public state advances by composing hero + rival action effects. Terminal rewards computed from showdown equity table.

### Revised simulate() flow

```
simulate(node, pub_state, factored_belief, model, depth):
  1. Hero picks action (UCB1 / progressive widening)
  2. For each rival particle:
     sample rival action from rival_policy[particle.type][pub_state]
  3. Observe joint rival action (one sampled action per rival, weighted by particle mass)
  4. Reweight particles: w_j *= rival_policy[particle_j.type][pub_state][observed_action]
  5. Resample if ESS < threshold (systematic_resample — already implemented)
  6. Compose next_pub_state from action_effects
  7. Check terminal -> if terminal, compute reward from showdown_equity
  8. If not terminal: find/create child keyed by real observation, recurse
  9. Backpropagate: total = reward + discount * future
```

## Factored Tabular Model

Pre-computed by Scala, passed as flat JNI arrays.

### State abstraction

| Dimension | Discretization | Count |
|-----------|---------------|-------|
| Street | preflop/flop/turn/river | 4 |
| Pot bucket | fraction of starting stacks | ~8 |
| Stack ratio | hero effective stack / starting | ~6 |
| **Public states total** | | **~192** |
| Rival type | StrategicClass ordinal (Value/Bluff/SemiBluff/Marginal) | 4 |
| Rival hand bucket | equity decile | 10 |
| **Private states per rival** | | **40** |

### Model arrays

1. **rival_policy** `[type x pub_state x action]` — `P(action | type, pub_state)`. Serves dual purpose: sampling rival actions during simulation AND computing observation likelihood for particle reweighting. Built by the kernel pipeline (TemperedLikelihood -> KernelConstructor -> ExploitationInterpolation).

2. **action_effects** `[action]` — `(pot_delta_frac, is_fold, is_allin)`. Compact description of what each action does to the public state. Derived from legal action enumeration.

3. **showdown_equity** `[hero_bucket x rival_bucket]` — terminal reward when hand reaches showdown. Pre-computed from HoldemEquity.

4. **terminal_flags** `[pub_state x hero_action]` — `{Continue, HeroFold, RivalFold, Showdown}`. When to stop simulation and compute terminal reward.

## Scala Formulation Bridge

`PokerPomcpFormulation` in `sicfun.holdem.engine` builds the factored tabular model:

### Building rival_policy (the kernel pipeline)

```
TemperedLikelihood.computeLikelihoods()
  -> KernelConstructor.buildActionKernelFull()
  -> ExploitationInterpolation.buildInterpolatedKernel()
  -> For each (type x pub_state x action): kernel predicts P(action | type, pub_state)
  -> flat array
```

### Building showdown_equity

```
For each (hero_bucket, rival_bucket):
  HoldemEquity.headsUpEquity(hero_range, rival_range) -> equity
  -> flat array
```

### Building action_effects

```
For each candidate action (fold, check, call, raise sizes):
  pot_delta = action.chipAmount / pot
  is_fold = action == Fold
  is_allin = action.chipAmount >= stack
  -> compact array
```

### Building rival particles

```
From StrategicSessionState (maintained across hands by Dynamics):
  per-rival belief state -> sample C particles
  each particle: (type=StrategicClass.ordinal, priv_state=hand_bucket, weight)
```

## State Lifecycle

### Session-level (persists across hands)

`StrategicSessionState`:
- `rivalBeliefs: Map[PlayerId, RivalBeliefState]` — accumulated from observations
- `exploitationStates: Map[PlayerId, ExploitationState]` — beta per rival
- `kernelProfile: JointKernelProfile[M]` — cached, rebuilt only on config change
- `detectionPredicate: DetectionPredicate` — cached

Loaded from `OpponentProfileStore` at session start. Flushed back at session end.

### Hand-level (reset per hand)

- New PublicState (street=preflop, pot=blinds, stacks=starting)
- Hero's private hand
- Empty action history for current hand

### Update cycle

Each observed rival action:
1. `SignalBridge`: `PokerAction` -> `TotalSignal`
2. `PublicStateBridge`: `GameState` -> strategic `PublicState`
3. `Dynamics.fullStep()`: updates rival beliefs + exploitation states

Hero's turn:
1. `PokerPomcpFormulation.buildModel()` from current session state
2. `WPomcpRuntime.solve()`
3. Map `bestAction` index -> `PokerAction`

Hand end:
1. `ShowdownKernel` update if cards revealed
2. Persist beliefs to `OpponentProfileStore`

## Pipeline Integration

### HeroMode

```scala
enum HeroMode:
  case Adaptive
  case Gto
  case Strategic   // formal POMDP via WPomcp
```

### StrategicEngine

New object in `sicfun.holdem.engine`:

```
initSession(config, opponentProfiles?)  — load/init beliefs, build kernels
startHand(heroCards)                    — reset hand-local state
observeAction(actor, action, gameState) — Dynamics.fullStep()
decide(gameState, legalActions)         — formulate + solve + map action
endHand(showdownResult?)               — showdown update, persist beliefs
```

### HeroDecisionPipeline

New branch in `decideHero()` for `HeroMode.Strategic` that delegates to `StrategicEngine.decide()`.

### HeroDecisionContext

Add optional `strategicEngine` field. Match runners construct one at session start for Strategic mode.

### Match runner integration

When mode is Strategic:
- `engine.initSession()` at match start
- `engine.startHand()` at each hand start
- `engine.observeAction()` on each observed rival action
- `decideHero()` routes to `engine.decide()`
- `engine.endHand()` at showdown

Same pattern for `HeadsUpSimulator` in validation harness.

## Concrete RivalBeliefState

The `RivalBeliefState` trait exists but has no implementation. One new type needed:

`StrategicRivalBelief` — holds `DiscreteDistribution[StrategicClass]` (type posterior) and updates via the kernel's tempered likelihood when `update()` is called. This is the concrete M that parameterizes `Dynamics[M]`, `KernelProfile[M]`, etc.

## Testing Strategy

### Level 1: C++ solver correctness (native tests)

- Trivial MDPs with known Q-values (existing PftDpwRuntimeTest pattern)
- Type-conditioned policies: particles with different types produce different rival actions
- Particle reweighting: observing action consistent with type A increases type A particle weights
- ESS-triggered resampling fires when beliefs concentrate
- Terminal reward correctness: fold/showdown paths

### Level 2: Formulation bridge (pure Scala, no native)

- `PokerPomcpFormulation.buildModel()` produces correct flat arrays
- Rival policy table sums to 1.0 per (type, pub_state) row
- Showdown equity table consistent with HoldemEquity
- Action effects compose correctly (pot arithmetic, terminal detection)
- Particle sampling from session beliefs produces valid arrays

### Level 3: Integration (end-to-end, requires native DLL)

- StrategicEngine against deterministic villain: finds exploitative action
- Belief persistence: 50 hands against a Nit, type posterior concentrates correctly
- Strategic vs Adaptive vs GTO against injected-leak villains via HeadsUpSimulator
- Regression: Adaptive and GTO modes unchanged

### Validation harness

Add `HeroMode.Strategic` to `AdaptiveProofHarness` mode options. Same leak villains, same scorecard, direct mode comparison.

## Invariants

- **Multiway-native**: WPomcp with |R| >= 1, heads-up is |R| = 1, no special case
- **Strategic never imports engine**: dependency direction preserved
- **Existing modes untouched**: Adaptive and GTO behavior unchanged
- **Belief persistence**: rival beliefs accumulate across hands via OpponentProfileStore
- **Cache stable state**: kernels/detection rebuilt only on config change, not per-hand
