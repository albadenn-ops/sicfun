# Phase 4a: Kernel Engine + Dynamics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement rival kernel construction, exploitation interpolation, and belief dynamics
**Architecture:** Pure Scala implementations over Phase 1 type signatures. KernelConstructor fills in the RivalKernel traits defined in Phase 1. Dynamics uses kernels for belief updates.
**Tech Stack:** Scala 3.8.1, munit 1.2.2
**Depends on:** Phase 1 (types), Phase 2 (tempered likelihood for exploitation)
**Unlocks:** Phase 4b (decomposition uses kernels and dynamics)

---

## File Map

| File | Responsibility | Spec Defs |
|------|---------------|-----------|
| `src/main/scala/sicfun/holdem/strategic/KernelConstructor.scala` | BuildRivalKernel, action/showdown/design kernels | Defs 16-21 impl |
| `src/main/scala/sicfun/holdem/strategic/ExploitationInterpolation.scala` | beta^{i,exploit} + retreat + safety constraint | Def 15C |
| `src/main/scala/sicfun/holdem/strategic/Dynamics.scala` | Belief update, rival-state update, polarization | Defs 22-25 |
| `src/main/scala/sicfun/holdem/strategic/DetectionPredicate.scala` | DetectModeling^i (A6') | A6' |
| `src/main/scala/sicfun/holdem/strategic/SpotPolarization.scala` | Spot-conditioned polarization | Def 25, A9 |
| `src/test/scala/sicfun/holdem/strategic/KernelConstructorTest.scala` | | |
| `src/test/scala/sicfun/holdem/strategic/ExploitationInterpolationTest.scala` | | |
| `src/test/scala/sicfun/holdem/strategic/DynamicsTest.scala` | | |

---

## Dependency Rule (NON-NEGOTIABLE)

Every file in `sicfun.holdem.strategic` may import ONLY:

```
import sicfun.holdem.types.{PokerAction, Street, Position, HoleCards, Board}
import sicfun.core.{DiscreteDistribution, Probability}
```

NO imports from `sicfun.holdem.engine`, `sicfun.holdem.runtime`, or any other `sicfun.holdem.*` subpackage.

---

## Task Execution Order

| Task | File | Depends On | Defs |
|------|------|-----------|------|
| 1 | DetectionPredicate.scala | Phase 1 (TableStructure, Signal) | A6' |
| 2 | SpotPolarization.scala | Phase 1 (DomainTypes, Signal, RivalKernel) | Def 25, A9 |
| 3 | KernelConstructor.scala + test | Phase 1 (RivalKernel traits), Phase 2 (TemperedLikelihood) | Defs 16-21 impl |
| 4 | ExploitationInterpolation.scala + test | Tasks 1, 3 | Def 15C |
| 5 | Dynamics.scala + test | Tasks 2, 3, 4 | Defs 22-25 |
| 6 | Full verification | All | -- |

---

### Task 1: DetectionPredicate (A6')

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/DetectionPredicate.scala`
- (Tested indirectly through ExploitationInterpolationTest in Task 4)

- [ ] **Step 1: Write the implementation**

```scala
package sicfun.holdem.strategic

/** Detection predicate for rival modeling awareness (A6').
  *
  * DetectModeling^i : H_t -> {0, 1}
  *
  * Returns true when rival i is inferred to be actively modeling
  * SICFUN's strategy. Triggers exploitation retreat (Def 15C).
  *
  * This is a trait because the detection mechanism is pluggable:
  * concrete implementations may use frequency anomaly detection,
  * timing tells, strategy deviation signatures, etc.
  */
trait DetectionPredicate:
  /** Evaluate whether rival i appears to be modeling SICFUN.
    *
    * @param rivalId the rival under evaluation
    * @param history the observable action history (public actions only)
    * @param publicState current public game state
    * @return true if modeling is detected
    */
  def detectModeling(
      rivalId: PlayerId,
      history: Vector[PublicAction],
      publicState: PublicState
  ): Boolean

/** Always-false detection (default: never detect modeling).
  * Used when detection is disabled or as a test stub.
  * With this predicate, beta never retreats via detection.
  */
object NeverDetect extends DetectionPredicate:
  def detectModeling(
      rivalId: PlayerId,
      history: Vector[PublicAction],
      publicState: PublicState
  ): Boolean = false

/** Always-true detection (test stub: always detect modeling).
  * Forces immediate retreat on every update.
  */
object AlwaysDetect extends DetectionPredicate:
  def detectModeling(
      rivalId: PlayerId,
      history: Vector[PublicAction],
      publicState: PublicState
  ): Boolean = true

/** Frequency-anomaly detection: detects modeling when rival's action
  * distribution deviates from expected baseline by more than a threshold.
  *
  * This is a concrete implementation suitable for initial deployment.
  * Looks at the last `window` actions and checks if the fraction of
  * counter-exploitative adjustments exceeds `threshold`.
  *
  * A "counter-exploitative adjustment" is detected when the rival's
  * action frequency shifts toward actions that specifically counter
  * SICFUN's current strategy (e.g., increased 3-bet frequency when
  * SICFUN has been opening wide).
  */
final class FrequencyAnomalyDetection(
    window: Int,
    threshold: Double
) extends DetectionPredicate:
  require(window > 0, "window must be positive")
  require(threshold > 0.0 && threshold <= 1.0, "threshold must be in (0, 1]")

  def detectModeling(
      rivalId: PlayerId,
      history: Vector[PublicAction],
      publicState: PublicState
  ): Boolean =
    if history.size < window then false
    else
      val recentActions = history.takeRight(window)
      val rivalActions = recentActions.filter(_.playerId == rivalId)
      if rivalActions.isEmpty then false
      else
        // Count aggressive actions (raises/reraises) as proxy for counter-exploitation
        val aggressiveCount = rivalActions.count(_.isAggressive)
        val aggressiveFraction = aggressiveCount.toDouble / rivalActions.size.toDouble
        aggressiveFraction > threshold
```

- [ ] **Step 2: Compile check**

Run: `sbt compile`
Expected: Compiles (no test file yet — tested via Task 4)

- [ ] **Step 3: Commit**

---

### Task 2: SpotPolarization (Def 25, A9)

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/SpotPolarization.scala`
- (Tested indirectly through DynamicsTest in Task 5)

- [ ] **Step 1: Write the implementation**

```scala
package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution

/** Spot-conditioned polarization profile (Def 25, A9).
  *
  * Pol_t^i(lambda | x_t^pub, pi^{0,S}, m_t^{R,i})
  *
  * For each rival i, the polarization profile quantifies how much
  * information about rival i's private state is revealed by a particular
  * sizing lambda, conditioned on the public spot.
  *
  * This is a trait because the polarization computation depends on
  * the specific rival model and is computed differently for different
  * kernel variants.
  */
trait SpotPolarization:
  /** Compute the polarization value for a given sizing in a spot.
    *
    * Higher polarization means the sizing reveals more about the
    * rival's range/type — i.e., the action is more informative.
    *
    * @param sizing the bet sizing to evaluate
    * @param publicState current public game state
    * @param rivalState rival's current belief state
    * @return polarization value in [0, 1]
    */
  def polarization(
      sizing: Sizing,
      publicState: PublicState,
      rivalState: RivalBeliefState
  ): Double

  /** Compute the full polarization profile over a set of candidate sizings.
    *
    * @param candidates candidate sizings to evaluate
    * @param publicState current public game state
    * @param rivalState rival's current belief state
    * @return map from sizing to polarization value
    */
  def profile(
      candidates: Vector[Sizing],
      publicState: PublicState,
      rivalState: RivalBeliefState
  ): Map[Sizing, Double] =
    candidates.map(s => s -> polarization(s, publicState, rivalState)).toMap

/** Uniform polarization (stub): all sizings are equally informative.
  * Returns 0.5 for every sizing. Used as a baseline or when
  * polarization analysis is disabled.
  */
object UniformPolarization extends SpotPolarization:
  def polarization(
      sizing: Sizing,
      publicState: PublicState,
      rivalState: RivalBeliefState
  ): Double = 0.5

/** Posterior-divergence polarization: measures polarization as the
  * KL divergence between the posterior-on-class after observing the
  * sizing vs. the prior, normalized to [0, 1].
  *
  * Pol(lambda) = 1 - exp(-D_KL(posterior_lambda || prior))
  *
  * This is the canonical implementation for Def 25.
  */
final class PosteriorDivergencePolarization(
    prior: DiscreteDistribution[StrategicClass]
) extends SpotPolarization:
  def polarization(
      sizing: Sizing,
      publicState: PublicState,
      rivalState: RivalBeliefState
  ): Double =
    // KL divergence is computed by the likelihood update in the kernel.
    // Here we use the sizing fraction as a proxy: extreme sizings
    // (very small or very large relative to pot) are more polarizing.
    val f = sizing.fractionOfPot.value
    val extremity = math.abs(2.0 * f - 1.0) // 0 at half-pot, 1 at 0 or full-pot
    // Sigmoid-like transform to [0, 1]
    1.0 - math.exp(-2.0 * extremity)
```

- [ ] **Step 2: Compile check**

Run: `sbt compile`
Expected: Compiles

- [ ] **Step 3: Commit**

---

### Task 3: KernelConstructor (Defs 16-21 implementation)

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/KernelConstructor.scala`
- Create: `src/test/scala/sicfun/holdem/strategic/KernelConstructorTest.scala`

Phase 1 defined `RivalKernel.scala` with trait signatures only. This task provides
concrete implementations that fill those traits using tempered likelihood from Phase 2.

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution
import sicfun.holdem.types.{PokerAction, Street, Board}

class KernelConstructorTest extends munit.FunSuite:

  private inline val Tol = 1e-12

  // ---- Shared fixtures ----

  /** Minimal rival belief state for testing. */
  private class TestRivalState(
      val posterior: DiscreteDistribution[StrategicClass],
      val updateCount: Int = 0
  ) extends RivalBeliefState:
    def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState =
      TestRivalState(posterior, updateCount + 1)

  private val uniformPrior = DiscreteDistribution(Map(
    StrategicClass.Villain  -> 0.25,
    StrategicClass.Bluffer  -> 0.25,
    StrategicClass.Moderate -> 0.25,
    StrategicClass.SemiBluffer -> 0.25
  ))

  private val dummyPublicState = PublicState(
    street = Street.Flop,
    board = Board.empty,
    pot = Chips(100.0),
    stacks = TableMap(
      hero = PlayerId("hero"),
      seats = Vector(
        Seat(PlayerId("hero"), sicfun.holdem.types.Position.SmallBlind, SeatStatus.Active, Chips(500.0)),
        Seat(PlayerId("v1"), sicfun.holdem.types.Position.BigBlind, SeatStatus.Active, Chips(500.0))
      )
    ),
    actionHistory = Vector.empty
  )

  private val raiseSignal = ActionSignal(
    action = PokerAction.Category.Raise,
    sizing = Some(Sizing(Chips(75.0), PotFraction(0.75))),
    timing = None,
    stage = Street.Flop
  )

  private val callSignal = ActionSignal(
    action = PokerAction.Category.Call,
    sizing = None,
    timing = None,
    stage = Street.Flop
  )

  private val showdownSignal = ShowdownSignal(
    revealedHands = Vector(
      RevealedHand(PlayerId("v1"), sicfun.holdem.types.HoleCards.empty)
    )
  )

  // ---- Def 16: StateEmbeddingUpdater ----

  test("StateEmbeddingUpdater: embeds posterior into rival state"):
    val updater: StateEmbeddingUpdater[TestRivalState] =
      (state, posterior) => TestRivalState(posterior, state.updateCount + 1)

    val initial = TestRivalState(uniformPrior)
    val shiftedPosterior = DiscreteDistribution(Map(
      StrategicClass.Villain -> 0.7,
      StrategicClass.Bluffer -> 0.1,
      StrategicClass.Moderate -> 0.1,
      StrategicClass.SemiBluffer -> 0.1
    ))
    val result = updater(initial, shiftedPosterior)
    assertEqualsDouble(result.posterior.probabilityOf(StrategicClass.Villain), 0.7, Tol)
    assertEquals(result.updateCount, 1)

  // ---- Def 17: BuildRivalKernel ----

  test("BuildRivalKernel produces an ActionKernel from a policy reference"):
    val updater: StateEmbeddingUpdater[TestRivalState] =
      (state, posterior) => TestRivalState(posterior, state.updateCount + 1)

    // Stub likelihood: always returns uniform posterior regardless of signal
    val stubLikelihood: TemperedLikelihoodFn = (signal, publicState, state) => uniformPrior

    val kernel = KernelConstructor.buildActionKernel(updater, stubLikelihood)
    val initial = TestRivalState(uniformPrior)
    val result = kernel.apply(initial, raiseSignal).asInstanceOf[TestRivalState]
    assertEquals(result.updateCount, 1)

  // ---- Def 18: Reference vs Attributed vs Blind ----

  test("RefActionKernel updates state using reference posterior"):
    val updater: StateEmbeddingUpdater[TestRivalState] =
      (state, posterior) => TestRivalState(posterior, state.updateCount + 1)
    val refLikelihood: TemperedLikelihoodFn = (_, _, _) =>
      DiscreteDistribution(Map(
        StrategicClass.Villain -> 0.6,
        StrategicClass.Bluffer -> 0.1,
        StrategicClass.Moderate -> 0.2,
        StrategicClass.SemiBluffer -> 0.1
      ))

    val kernel = KernelConstructor.buildActionKernel(updater, refLikelihood)
    val initial = TestRivalState(uniformPrior)
    val result = kernel.apply(initial, raiseSignal).asInstanceOf[TestRivalState]
    assertEqualsDouble(result.posterior.probabilityOf(StrategicClass.Villain), 0.6, Tol)

  test("BlindActionKernel returns exact same state (identity)"):
    val blind = BlindActionKernel[TestRivalState]()
    val initial = TestRivalState(uniformPrior)
    val result = blind.apply(initial, raiseSignal)
    assert(result eq initial, "Blind kernel must return the exact same state object")

  // ---- Def 19: ShowdownKernel ----

  test("ShowdownKernel updates state from showdown revelation"):
    val sdKernel = new ShowdownKernel[TestRivalState]:
      def apply(state: TestRivalState, showdown: ShowdownSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 10) // showdown = big update

    val initial = TestRivalState(uniformPrior)
    val result = sdKernel.apply(initial, showdownSignal)
    assertEquals(result.updateCount, 10)

  // ---- Def 19A: DesignSignalKernel ----

  test("DesignSignalKernel uses only action component, not sizing/timing"):
    var capturedSignal: Option[ActionSignal] = None
    val designLikelihood: TemperedLikelihoodFn = (signal, _, _) =>
      capturedSignal = Some(signal)
      uniformPrior

    val updater: StateEmbeddingUpdater[TestRivalState] =
      (state, posterior) => TestRivalState(posterior, state.updateCount + 1)

    val kernel = KernelConstructor.buildDesignKernel(updater, designLikelihood)
    val initial = TestRivalState(uniformPrior)
    kernel.apply(initial, raiseSignal)

    // Design kernel should strip sizing/timing before passing to likelihood
    assert(capturedSignal.isDefined)
    assert(capturedSignal.get.sizing.isEmpty,
      "Design kernel must strip sizing (use only action category)")
    assert(capturedSignal.get.timing.isEmpty,
      "Design kernel must strip timing (use only action category)")
    assertEquals(capturedSignal.get.action, PokerAction.Category.Raise)

  // ---- Def 20: FullKernel composition ----

  test("FullKernel composes action + showdown when showdown present"):
    val actionKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 1)

    val sdKernel = new ShowdownKernel[TestRivalState]:
      def apply(state: TestRivalState, showdown: ShowdownSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 100)

    val full = KernelConstructor.composeFullKernel(actionKernel, sdKernel)
    val initial = TestRivalState(uniformPrior)

    // With showdown: action first, then showdown
    val withSd = TotalSignal(raiseSignal, Some(showdownSignal))
    val result = full.apply(initial, withSd, dummyPublicState).asInstanceOf[TestRivalState]
    assertEquals(result.updateCount, 101) // 1 from action + 100 from showdown

  test("FullKernel applies only action when no showdown"):
    val actionKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 1)

    val sdKernel = new ShowdownKernel[TestRivalState]:
      def apply(state: TestRivalState, showdown: ShowdownSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 100)

    val full = KernelConstructor.composeFullKernel(actionKernel, sdKernel)
    val initial = TestRivalState(uniformPrior)

    // Without showdown: action only
    val noSd = TotalSignal(raiseSignal, None)
    val result = full.apply(initial, noSd, dummyPublicState).asInstanceOf[TestRivalState]
    assertEquals(result.updateCount, 1) // only action

  test("FullKernel with blind variant returns identity regardless of signals"):
    val blindFull = KernelConstructor.composeBlindFullKernel[TestRivalState]()
    val initial = TestRivalState(uniformPrior)

    val withSd = TotalSignal(raiseSignal, Some(showdownSignal))
    val result = blindFull.apply(initial, withSd, dummyPublicState)
    assert(result eq initial, "Blind full kernel must return exact same state")

  // ---- Def 21: Joint kernel profiles ----

  test("JointKernelProfile maps each rival to a FullKernel"):
    val blindFull = KernelConstructor.composeBlindFullKernel[TestRivalState]()
    val profile = JointKernelProfile(Map(
      PlayerId("v1") -> blindFull,
      PlayerId("v2") -> blindFull
    ))
    assertEquals(profile.kernels.size, 2)

  test("JointKernelProfile.apply dispatches to correct rival kernel"):
    var v1Updated = false
    var v2Updated = false

    val v1Kernel = new FullKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: TotalSignal, pub: PublicState): TestRivalState =
        v1Updated = true
        TestRivalState(state.posterior, state.updateCount + 1)

    val v2Kernel = new FullKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: TotalSignal, pub: PublicState): TestRivalState =
        v2Updated = true
        TestRivalState(state.posterior, state.updateCount + 2)

    val profile = JointKernelProfile(Map(
      PlayerId("v1") -> v1Kernel,
      PlayerId("v2") -> v2Kernel
    ))

    val initial = TestRivalState(uniformPrior)
    val signal = TotalSignal(raiseSignal, None)

    profile.kernels(PlayerId("v1")).apply(initial, signal, dummyPublicState)
    assert(v1Updated)
    assert(!v2Updated)

    profile.kernels(PlayerId("v2")).apply(initial, signal, dummyPublicState)
    assert(v2Updated)

  // ---- Backward compatibility: beta=1 recovers attributed world ----

  test("backward compat: with full attribution, kernel == attributed kernel"):
    val attribLikelihood: TemperedLikelihoodFn = (_, _, _) =>
      DiscreteDistribution(Map(
        StrategicClass.Villain -> 0.8,
        StrategicClass.Bluffer -> 0.05,
        StrategicClass.Moderate -> 0.1,
        StrategicClass.SemiBluffer -> 0.05
      ))
    val updater: StateEmbeddingUpdater[TestRivalState] =
      (state, posterior) => TestRivalState(posterior, state.updateCount + 1)

    val attribKernel = KernelConstructor.buildActionKernel(updater, attribLikelihood)
    val initial = TestRivalState(uniformPrior)
    val result = attribKernel.apply(initial, raiseSignal).asInstanceOf[TestRivalState]
    // beta=1 means full attributed: posterior should be the attributed one
    assertEqualsDouble(result.posterior.probabilityOf(StrategicClass.Villain), 0.8, Tol)
```

- [ ] **Step 2: Run test — expect compilation failure (RED)**

Run: `sbt "testOnly sicfun.holdem.strategic.KernelConstructorTest"`
Expected: Compilation error (KernelConstructor, TemperedLikelihoodFn, JointKernelProfile do not exist)

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution

/** Tempered likelihood function type.
  *
  * Given an action signal, public state, and rival belief state,
  * returns the updated posterior over strategic classes.
  *
  * This abstracts over the specific tempered likelihood computation
  * (Def 15A-15B), which lives in Phase 2. KernelConstructor depends
  * only on this function type, not on TemperedLikelihood internals.
  */
type TemperedLikelihoodFn = (
    ActionSignal,
    PublicState,
    RivalBeliefState
) => DiscreteDistribution[StrategicClass]

/** Kernel constructor (Defs 16-21 implementation).
  *
  * Builds concrete rival kernels from tempered likelihood functions
  * and state-embedding updaters. Each method corresponds to a
  * specific definition in §4.2.
  */
object KernelConstructor:

  /** Build an action kernel from an updater and likelihood (Def 17).
    *
    * BuildRivalKernel^i_{kappa,delta}(pi)(m, y, x^pub)
    *   := U^{R,i}_pi(m, mu^{R,i,pi,kappa,delta}_{t+1})
    *
    * The likelihood function encapsulates the policy reference (pi)
    * and tempering parameters (kappa, delta).
    */
  def buildActionKernel[M <: RivalBeliefState](
      updater: StateEmbeddingUpdater[M],
      likelihood: TemperedLikelihoodFn
  ): ActionKernel[M] =
    new ActionKernel[M]:
      def apply(state: M, signal: ActionSignal): M =
        val dummyPublic = PublicState(
          street = signal.stage,
          board = sicfun.holdem.types.Board.empty,
          pot = Chips(0.0),
          stacks = TableMap(
            hero = PlayerId("_"),
            seats = Vector.empty
          ),
          actionHistory = Vector.empty
        )
        val posterior = likelihood(signal, dummyPublic, state)
        updater(state, posterior)

  /** Build an action kernel with explicit public state threading (Def 17, full form).
    *
    * This is the production form where public state flows through.
    */
  def buildActionKernelFull[M <: RivalBeliefState](
      updater: StateEmbeddingUpdater[M],
      likelihood: TemperedLikelihoodFn
  ): ActionKernelFull[M] =
    new ActionKernelFull[M]:
      def apply(state: M, signal: ActionSignal, publicState: PublicState): M =
        val posterior = likelihood(signal, publicState, state)
        updater(state, posterior)

  /** Build a design-signal kernel (Def 19A).
    *
    * Strips sizing and timing from the signal before computing likelihood.
    * Uses only the action category component a_t.
    *
    * Marginalization order: temper-then-marginalize (canonical).
    * Since we strip to just the action category before passing to the
    * likelihood function, the likelihood receives a signal with
    * sizing=None, timing=None. The likelihood function is responsible
    * for summing over (lambda, tau) internally if needed.
    */
  def buildDesignKernel[M <: RivalBeliefState](
      updater: StateEmbeddingUpdater[M],
      likelihood: TemperedLikelihoodFn
  ): ActionKernel[M] =
    new ActionKernel[M]:
      def apply(state: M, signal: ActionSignal): M =
        // Strip sizing and timing — keep only action category and stage
        val designSignal = ActionSignal(
          action = signal.action,
          sizing = None,
          timing = None,
          stage = signal.stage
        )
        val dummyPublic = PublicState(
          street = signal.stage,
          board = sicfun.holdem.types.Board.empty,
          pot = Chips(0.0),
          stacks = TableMap(
            hero = PlayerId("_"),
            seats = Vector.empty
          ),
          actionHistory = Vector.empty
        )
        val posterior = likelihood(designSignal, dummyPublic, state)
        updater(state, posterior)

  /** Compose a full kernel from action + showdown channels (Def 20).
    *
    * Gamma^{full,bullet,i}(m, Y, x^pub) =
    *   if Y^sd != empty and bullet != blind:
    *     showdown(action(m, Y^act), Y^sd)
    *   if Y^sd == empty and bullet != blind:
    *     action(m, Y^act)
    *   if bullet == blind:
    *     m
    */
  def composeFullKernel[M <: RivalBeliefState](
      actionKernel: ActionKernel[M],
      showdownKernel: ShowdownKernel[M]
  ): FullKernel[M] =
    new FullKernel[M]:
      def apply(state: M, signal: TotalSignal, publicState: PublicState): M =
        val afterAction = actionKernel.apply(state, signal.actionSignal)
        signal.showdown match
          case Some(sd) => showdownKernel.apply(afterAction, sd)
          case None     => afterAction

  /** Compose a blind full kernel (Def 20, bullet=blind).
    *
    * Always returns the input state unchanged.
    */
  def composeBlindFullKernel[M <: RivalBeliefState](): FullKernel[M] =
    new FullKernel[M]:
      def apply(state: M, signal: TotalSignal, publicState: PublicState): M = state

/** Action kernel with explicit public state (production form of Def 17). */
trait ActionKernelFull[M]:
  def apply(state: M, signal: ActionSignal, publicState: PublicState): M

/** Joint kernel profile (Def 21 implementation).
  *
  * Gamma^{attrib} := {Gamma^{full,attrib,i}}_{i in R}
  * Gamma^{ref}    := {Gamma^{full,ref,i}}_{i in R}
  * Gamma^{blind}  := {Gamma^{full,blind,i}}_{i in R}
  *
  * Extends the Phase 1 KernelProfile (which was action-only) to
  * full kernels (action + showdown composition).
  */
final case class JointKernelProfile[M <: RivalBeliefState](
    kernels: Map[PlayerId, FullKernel[M]]
):
  /** Apply the profile to update all rivals simultaneously (Def 23 helper). */
  def updateAll(
      states: Map[PlayerId, M],
      signal: TotalSignal,
      publicState: PublicState
  ): Map[PlayerId, M] =
    states.map { case (id, state) =>
      kernels.get(id) match
        case Some(kernel) => id -> kernel.apply(state, signal, publicState)
        case None         => id -> state // no kernel for this rival, preserve state
    }
```

- [ ] **Step 4: Run test — expect all pass (GREEN)**

Run: `sbt "testOnly sicfun.holdem.strategic.KernelConstructorTest"`
Expected: All tests pass

- [ ] **Step 5: Commit**

---

### Task 4: ExploitationInterpolation (Def 15C)

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/ExploitationInterpolation.scala`
- Create: `src/test/scala/sicfun/holdem/strategic/ExploitationInterpolationTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution
import sicfun.holdem.types.{PokerAction, Street, Board}

class ExploitationInterpolationTest extends munit.FunSuite:

  private inline val Tol = 1e-12

  // ---- Shared fixtures ----

  private class TestRivalState(
      val posterior: DiscreteDistribution[StrategicClass],
      val updateCount: Int = 0
  ) extends RivalBeliefState:
    def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState =
      TestRivalState(posterior, updateCount + 1)

  private val uniformPrior = DiscreteDistribution(Map(
    StrategicClass.Villain  -> 0.25,
    StrategicClass.Bluffer  -> 0.25,
    StrategicClass.Moderate -> 0.25,
    StrategicClass.SemiBluffer -> 0.25
  ))

  private val dummyPublicState = PublicState(
    street = Street.Flop,
    board = Board.empty,
    pot = Chips(100.0),
    stacks = TableMap(
      hero = PlayerId("hero"),
      seats = Vector(
        Seat(PlayerId("hero"), sicfun.holdem.types.Position.SmallBlind, SeatStatus.Active, Chips(500.0)),
        Seat(PlayerId("v1"), sicfun.holdem.types.Position.BigBlind, SeatStatus.Active, Chips(500.0))
      )
    ),
    actionHistory = Vector.empty
  )

  // ---- Def 15C: ExploitationState ----

  test("ExploitationState.initial: beta starts at configured initial value"):
    val config = ExploitationConfig(
      initialBeta = 0.5,
      retreatRate = 0.1,
      adaptationTolerance = 0.05
    )
    val state = ExploitationState.initial(config)
    assertEqualsDouble(state.beta, 0.5, Tol)

  test("ExploitationState.initial: beta=1 is full attribution"):
    val config = ExploitationConfig(initialBeta = 1.0, retreatRate = 0.1, adaptationTolerance = 0.05)
    val state = ExploitationState.initial(config)
    assertEqualsDouble(state.beta, 1.0, Tol)

  test("ExploitationState.initial: beta=0 is pure reference"):
    val config = ExploitationConfig(initialBeta = 0.0, retreatRate = 0.1, adaptationTolerance = 0.05)
    val state = ExploitationState.initial(config)
    assertEqualsDouble(state.beta, 0.0, Tol)

  // ---- Detection-triggered retreat ----

  test("retreat: beta decreases by retreatRate when modeling detected"):
    val config = ExploitationConfig(initialBeta = 0.8, retreatRate = 0.2, adaptationTolerance = 0.05)
    val state = ExploitationState(beta = 0.8)
    val retreated = ExploitationInterpolation.retreat(state, config)
    assertEqualsDouble(retreated.beta, 0.6, Tol)

  test("retreat: beta floors at 0.0"):
    val config = ExploitationConfig(initialBeta = 0.8, retreatRate = 0.5, adaptationTolerance = 0.05)
    val state = ExploitationState(beta = 0.3)
    val retreated = ExploitationInterpolation.retreat(state, config)
    assertEqualsDouble(retreated.beta, 0.0, Tol)

  test("retreat: beta=0 stays at 0"):
    val config = ExploitationConfig(initialBeta = 0.0, retreatRate = 0.1, adaptationTolerance = 0.05)
    val state = ExploitationState(beta = 0.0)
    val retreated = ExploitationInterpolation.retreat(state, config)
    assertEqualsDouble(retreated.beta, 0.0, Tol)

  // ---- Interpolated posterior ----

  test("interpolate: beta=0 returns pure reference posterior"):
    val refPosterior = DiscreteDistribution(Map(
      StrategicClass.Villain -> 0.4,
      StrategicClass.Bluffer -> 0.3,
      StrategicClass.Moderate -> 0.2,
      StrategicClass.SemiBluffer -> 0.1
    ))
    val attribPosterior = DiscreteDistribution(Map(
      StrategicClass.Villain -> 0.8,
      StrategicClass.Bluffer -> 0.05,
      StrategicClass.Moderate -> 0.1,
      StrategicClass.SemiBluffer -> 0.05
    ))
    val result = ExploitationInterpolation.interpolatePosterior(
      beta = 0.0,
      refPosterior = refPosterior,
      attribPosterior = attribPosterior
    )
    assertEqualsDouble(result.probabilityOf(StrategicClass.Villain), 0.4, Tol)
    assertEqualsDouble(result.probabilityOf(StrategicClass.Bluffer), 0.3, Tol)

  test("interpolate: beta=1 returns pure attributed posterior"):
    val refPosterior = DiscreteDistribution(Map(
      StrategicClass.Villain -> 0.4,
      StrategicClass.Bluffer -> 0.3,
      StrategicClass.Moderate -> 0.2,
      StrategicClass.SemiBluffer -> 0.1
    ))
    val attribPosterior = DiscreteDistribution(Map(
      StrategicClass.Villain -> 0.8,
      StrategicClass.Bluffer -> 0.05,
      StrategicClass.Moderate -> 0.1,
      StrategicClass.SemiBluffer -> 0.05
    ))
    val result = ExploitationInterpolation.interpolatePosterior(
      beta = 1.0,
      refPosterior = refPosterior,
      attribPosterior = attribPosterior
    )
    assertEqualsDouble(result.probabilityOf(StrategicClass.Villain), 0.8, Tol)
    assertEqualsDouble(result.probabilityOf(StrategicClass.Bluffer), 0.05, Tol)

  test("interpolate: beta=0.5 returns midpoint of reference and attributed"):
    val refPosterior = DiscreteDistribution(Map(
      StrategicClass.Villain -> 0.4,
      StrategicClass.Bluffer -> 0.2,
      StrategicClass.Moderate -> 0.2,
      StrategicClass.SemiBluffer -> 0.2
    ))
    val attribPosterior = DiscreteDistribution(Map(
      StrategicClass.Villain -> 0.8,
      StrategicClass.Bluffer -> 0.0,
      StrategicClass.Moderate -> 0.2,
      StrategicClass.SemiBluffer -> 0.0
    ))
    val result = ExploitationInterpolation.interpolatePosterior(
      beta = 0.5,
      refPosterior = refPosterior,
      attribPosterior = attribPosterior
    )
    // (1-0.5)*0.4 + 0.5*0.8 = 0.6
    assertEqualsDouble(result.probabilityOf(StrategicClass.Villain), 0.6, Tol)
    // (1-0.5)*0.2 + 0.5*0.0 = 0.1
    assertEqualsDouble(result.probabilityOf(StrategicClass.Bluffer), 0.1, Tol)

  test("interpolated posterior sums to 1.0"):
    val refPosterior = DiscreteDistribution(Map(
      StrategicClass.Villain -> 0.3,
      StrategicClass.Bluffer -> 0.3,
      StrategicClass.Moderate -> 0.2,
      StrategicClass.SemiBluffer -> 0.2
    ))
    val attribPosterior = DiscreteDistribution(Map(
      StrategicClass.Villain -> 0.9,
      StrategicClass.Bluffer -> 0.02,
      StrategicClass.Moderate -> 0.05,
      StrategicClass.SemiBluffer -> 0.03
    ))
    val result = ExploitationInterpolation.interpolatePosterior(
      beta = 0.7,
      refPosterior = refPosterior,
      attribPosterior = attribPosterior
    )
    val total = StrategicClass.values.map(result.probabilityOf).sum
    assertEqualsDouble(total, 1.0, Tol)

  // ---- Integration: detection + retreat + interpolation ----

  test("full cycle: detect modeling -> retreat -> interpolated posterior shifts to reference"):
    val config = ExploitationConfig(initialBeta = 1.0, retreatRate = 0.3, adaptationTolerance = 0.05)
    val state0 = ExploitationState.initial(config)
    assertEqualsDouble(state0.beta, 1.0, Tol)

    // Modeling detected: retreat
    val state1 = ExploitationInterpolation.retreat(state0, config)
    assertEqualsDouble(state1.beta, 0.7, Tol)

    // Detected again: retreat further
    val state2 = ExploitationInterpolation.retreat(state1, config)
    assertEqualsDouble(state2.beta, 0.4, Tol)

    // Detected again
    val state3 = ExploitationInterpolation.retreat(state2, config)
    assertEqualsDouble(state3.beta, 0.1, Tol)

    // Detected again: floors at 0
    val state4 = ExploitationInterpolation.retreat(state3, config)
    assertEqualsDouble(state4.beta, 0.0, Tol)

  // ---- Backward compatibility: beta=1 recovers attributed world (v0.30.2 §12.2) ----

  test("backward compat: beta=1 for all rivals recovers v0.29.1 behavior"):
    val config = ExploitationConfig(initialBeta = 1.0, retreatRate = 0.0, adaptationTolerance = Double.MaxValue)
    val state = ExploitationState.initial(config)
    // beta=1, retreatRate=0 means no retreat ever happens
    val afterRetreat = ExploitationInterpolation.retreat(state, config)
    assertEqualsDouble(afterRetreat.beta, 1.0, Tol) // stays at 1.0

  // ---- A6': DetectionPredicate integration ----

  test("NeverDetect: modeling never detected, beta never retreats"):
    val config = ExploitationConfig(initialBeta = 0.8, retreatRate = 0.2, adaptationTolerance = 0.05)
    val state = ExploitationState.initial(config)
    val detected = NeverDetect.detectModeling(
      PlayerId("v1"), Vector.empty, dummyPublicState
    )
    assert(!detected)
    // No retreat happens because detection is false

  test("AlwaysDetect: modeling always detected"):
    val detected = AlwaysDetect.detectModeling(
      PlayerId("v1"), Vector.empty, dummyPublicState
    )
    assert(detected)

  // ---- Adaptation safety constraint ----

  test("safety constraint: beta is clamped when exploitability exceeds bound"):
    val config = ExploitationConfig(initialBeta = 1.0, retreatRate = 0.1, adaptationTolerance = 0.05)
    // Stub exploitability: returns beta * 0.1 (higher beta = more exploitable)
    val exploitability: Double => Double = beta => beta * 0.1
    val epsilonNE = 0.02

    val safeBeta = ExploitationInterpolation.clampForSafety(
      beta = 1.0,
      exploitabilityFn = exploitability,
      epsilonNE = epsilonNE,
      deltaAdapt = config.adaptationTolerance
    )
    // Exploit(pi^S_{beta}) <= epsilon_NE + delta_adapt = 0.02 + 0.05 = 0.07
    // beta * 0.1 <= 0.07 => beta <= 0.7
    assert(safeBeta <= 0.7 + Tol)
    assert(safeBeta >= 0.0)
```

- [ ] **Step 2: Run test — expect compilation failure (RED)**

Run: `sbt "testOnly sicfun.holdem.strategic.ExploitationInterpolationTest"`
Expected: Compilation error

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution

/** Exploitation interpolation configuration (Def 15C parameters). */
final case class ExploitationConfig(
    /** Initial value of beta^{i,exploit} in [0, 1].
      * beta=0 is pure reference; beta=1 is full attribution.
      */
    initialBeta: Double,
    /** Retreat rate delta_retreat > 0.
      * How much beta decreases when DetectModeling fires.
      */
    retreatRate: Double,
    /** Adaptation safety tolerance delta_adapt >= 0.
      * Exploit(pi^S_beta) <= epsilon_NE + delta_adapt.
      */
    adaptationTolerance: Double
):
  require(initialBeta >= 0.0 && initialBeta <= 1.0,
    s"initialBeta must be in [0, 1], got $initialBeta")
  require(retreatRate >= 0.0,
    s"retreatRate must be non-negative, got $retreatRate")
  require(adaptationTolerance >= 0.0,
    s"adaptationTolerance must be non-negative, got $adaptationTolerance")

/** Per-rival exploitation state tracking beta^{i,exploit}_t. */
final case class ExploitationState(
    beta: Double
):
  require(beta >= 0.0 && beta <= 1.0, s"beta must be in [0, 1], got $beta")

object ExploitationState:
  def initial(config: ExploitationConfig): ExploitationState =
    ExploitationState(beta = config.initialBeta)

/** Exploitation interpolation operations (Def 15C). */
object ExploitationInterpolation:

  /** Detection-triggered retreat (Def 15C).
    *
    * beta_{t+1} <- max(0, beta_t - delta_retreat)
    *
    * Called when DetectModeling^i(H_t) = 1.
    */
  def retreat(state: ExploitationState, config: ExploitationConfig): ExploitationState =
    ExploitationState(beta = math.max(0.0, state.beta - config.retreatRate))

  /** Interpolate reference and attributed posteriors (Def 15C).
    *
    * mu^interp = (1 - beta) * mu^{ref} + beta * mu^{attrib}
    *
    * beta=0 recovers reference world; beta=1 recovers attributed world.
    *
    * The result is always a valid distribution (convex combination of distributions).
    */
  def interpolatePosterior(
      beta: Double,
      refPosterior: DiscreteDistribution[StrategicClass],
      attribPosterior: DiscreteDistribution[StrategicClass]
  ): DiscreteDistribution[StrategicClass] =
    require(beta >= 0.0 && beta <= 1.0, s"beta must be in [0, 1], got $beta")
    if beta == 0.0 then refPosterior
    else if beta == 1.0 then attribPosterior
    else
      // Convex combination: (1 - beta) * ref + beta * attrib
      val allClasses = (refPosterior.support ++ attribPosterior.support).toSet
      val combined = allClasses.map { c =>
        val refP = refPosterior.probabilityOf(c)
        val attribP = attribPosterior.probabilityOf(c)
        c -> ((1.0 - beta) * refP + beta * attribP)
      }.toMap
      DiscreteDistribution(combined)

  /** Build an interpolated action kernel (Def 15C formal mechanism).
    *
    * Gamma^{act,interp,i}(m, y, x^pub)
    *   := U_pi(m, (1 - beta) * mu^{ref} + beta * mu^{attrib})
    *
    * This composes reference and attributed likelihood functions
    * with the interpolation parameter.
    */
  def buildInterpolatedKernel[M <: RivalBeliefState](
      updater: StateEmbeddingUpdater[M],
      refLikelihood: TemperedLikelihoodFn,
      attribLikelihood: TemperedLikelihoodFn,
      beta: Double
  ): ActionKernel[M] =
    new ActionKernel[M]:
      def apply(state: M, signal: ActionSignal): M =
        val dummyPublic = PublicState(
          street = signal.stage,
          board = sicfun.holdem.types.Board.empty,
          pot = Chips(0.0),
          stacks = TableMap(
            hero = PlayerId("_"),
            seats = Vector.empty
          ),
          actionHistory = Vector.empty
        )
        val refPost = refLikelihood(signal, dummyPublic, state)
        val attribPost = attribLikelihood(signal, dummyPublic, state)
        val interpolated = interpolatePosterior(beta, refPost, attribPost)
        updater(state, interpolated)

  /** Clamp beta to satisfy the adaptation safety constraint (Def 15C).
    *
    * Exploit(pi^S_{beta}) <= epsilon_NE + delta_adapt
    *
    * Uses binary search to find the maximum safe beta.
    *
    * @param beta the desired beta value
    * @param exploitabilityFn function that estimates exploitability for a given beta
    * @param epsilonNE baseline exploitability
    * @param deltaAdapt adaptation safety tolerance
    * @return the clamped beta, possibly lower than input
    */
  def clampForSafety(
      beta: Double,
      exploitabilityFn: Double => Double,
      epsilonNE: Double,
      deltaAdapt: Double
  ): Double =
    val bound = epsilonNE + deltaAdapt
    if exploitabilityFn(beta) <= bound then beta
    else
      // Binary search for the maximum safe beta
      var lo = 0.0
      var hi = beta
      var iter = 0
      while iter < 50 && (hi - lo) > 1e-10 do
        val mid = (lo + hi) / 2.0
        if exploitabilityFn(mid) <= bound then lo = mid
        else hi = mid
        iter += 1
      lo

  /** Full update step: detect, retreat if needed, clamp for safety.
    *
    * Combines A6' detection with Def 15C retreat and safety constraint.
    */
  def updateExploitation(
      state: ExploitationState,
      config: ExploitationConfig,
      rivalId: PlayerId,
      history: Vector[PublicAction],
      publicState: PublicState,
      detector: DetectionPredicate,
      exploitabilityFn: Double => Double,
      epsilonNE: Double
  ): ExploitationState =
    val afterDetection =
      if detector.detectModeling(rivalId, history, publicState)
      then retreat(state, config)
      else state
    val safeBeta = clampForSafety(
      afterDetection.beta,
      exploitabilityFn,
      epsilonNE,
      config.adaptationTolerance
    )
    ExploitationState(beta = safeBeta)
```

- [ ] **Step 4: Run test — expect all pass (GREEN)**

Run: `sbt "testOnly sicfun.holdem.strategic.ExploitationInterpolationTest"`
Expected: All tests pass

- [ ] **Step 5: Commit**

---

### Task 5: Dynamics (Defs 22-25)

**Files:**
- Create: `src/main/scala/sicfun/holdem/strategic/Dynamics.scala`
- Create: `src/test/scala/sicfun/holdem/strategic/DynamicsTest.scala`

- [ ] **Step 1: Write the failing test**

```scala
package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution
import sicfun.holdem.types.{PokerAction, Street, Board, Position, HoleCards}

class DynamicsTest extends munit.FunSuite:

  private inline val Tol = 1e-12

  // ---- Shared fixtures ----

  private class TestRivalState(
      val posterior: DiscreteDistribution[StrategicClass],
      val updateCount: Int = 0,
      val label: String = ""
  ) extends RivalBeliefState:
    def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState =
      TestRivalState(posterior, updateCount + 1, label)

  private val uniformPrior = DiscreteDistribution(Map(
    StrategicClass.Villain  -> 0.25,
    StrategicClass.Bluffer  -> 0.25,
    StrategicClass.Moderate -> 0.25,
    StrategicClass.SemiBluffer -> 0.25
  ))

  private val dummyPublicState = PublicState(
    street = Street.Flop,
    board = Board.empty,
    pot = Chips(100.0),
    stacks = TableMap(
      hero = PlayerId("hero"),
      seats = Vector(
        Seat(PlayerId("hero"), Position.SmallBlind, SeatStatus.Active, Chips(500.0)),
        Seat(PlayerId("v1"), Position.BigBlind, SeatStatus.Active, Chips(500.0)),
        Seat(PlayerId("v2"), Position.Button, SeatStatus.Active, Chips(500.0))
      )
    ),
    actionHistory = Vector.empty
  )

  private val raiseSignal = ActionSignal(
    action = PokerAction.Category.Raise,
    sizing = Some(Sizing(Chips(75.0), PotFraction(0.75))),
    timing = None,
    stage = Street.Flop
  )

  private val totalSignalNoSd = TotalSignal(raiseSignal, showdown = None)

  private val showdownSignal = ShowdownSignal(
    revealedHands = Vector(
      RevealedHand(PlayerId("v1"), HoleCards.empty)
    )
  )

  private val totalSignalWithSd = TotalSignal(raiseSignal, showdown = Some(showdownSignal))

  // ---- Def 22: Belief update ----

  test("Def 22: beliefUpdate produces updated augmented state"):
    val oppState1 = OpponentModelState(PlayerId("v1"), uniformPrior)
    val oppState2 = OpponentModelState(PlayerId("v2"), uniformPrior)
    val rivals = RivalMap(Map(
      PlayerId("v1") -> oppState1,
      PlayerId("v2") -> oppState2
    ))
    val augState = AugmentedState(
      publicState = dummyPublicState,
      privateHand = HoleCards.empty,
      opponents = rivals,
      ownEvidence = OwnEvidence.empty
    )
    val belief = OperativeBelief(DiscreteDistribution(Map(augState -> 1.0)))

    // After a belief update, the belief should still be a valid distribution
    val updated = Dynamics.beliefUpdate(
      belief,
      totalSignalNoSd,
      dummyPublicState,
      updater = (b, _, _) => b // identity updater for test
    )
    val total = updated.distribution.support.map(updated.distribution.probabilityOf).sum
    assertEqualsDouble(total, 1.0, Tol)

  // ---- Def 23: Full rival-state update ----

  test("Def 23: fullRivalUpdate applies kernel to each rival"):
    val v1State = TestRivalState(uniformPrior, 0, "v1")
    val v2State = TestRivalState(uniformPrior, 0, "v2")

    val v1Kernel = new FullKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: TotalSignal, pub: PublicState): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 1, state.label)

    val v2Kernel = new FullKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: TotalSignal, pub: PublicState): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 10, state.label)

    val profile = JointKernelProfile(Map(
      PlayerId("v1") -> v1Kernel,
      PlayerId("v2") -> v2Kernel
    ))

    val states = Map[PlayerId, TestRivalState](
      PlayerId("v1") -> v1State,
      PlayerId("v2") -> v2State
    )

    val updated = Dynamics.fullRivalUpdate(states, totalSignalNoSd, dummyPublicState, profile)
    assertEquals(updated(PlayerId("v1")).updateCount, 1)
    assertEquals(updated(PlayerId("v2")).updateCount, 10)

  test("Def 23: fullRivalUpdate preserves state for rivals without kernel"):
    val v1State = TestRivalState(uniformPrior, 0, "v1")
    val v3State = TestRivalState(uniformPrior, 5, "v3-no-kernel")

    val v1Kernel = new FullKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: TotalSignal, pub: PublicState): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 1, state.label)

    val profile = JointKernelProfile[TestRivalState](Map(
      PlayerId("v1") -> v1Kernel
      // no kernel for v3
    ))

    val states = Map(
      PlayerId("v1") -> v1State,
      PlayerId("v3") -> v3State
    )

    val updated = Dynamics.fullRivalUpdate(states, totalSignalNoSd, dummyPublicState, profile)
    assertEquals(updated(PlayerId("v1")).updateCount, 1)
    assertEquals(updated(PlayerId("v3")).updateCount, 5) // preserved

  // ---- Def 23: variant selection ----

  test("Def 23: different kernel variants produce different updates"):
    val v1State = TestRivalState(uniformPrior, 0, "v1")

    val attribKernel = new FullKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: TotalSignal, pub: PublicState): TestRivalState =
        TestRivalState(state.posterior, 100, "attrib")

    val refKernel = new FullKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: TotalSignal, pub: PublicState): TestRivalState =
        TestRivalState(state.posterior, 200, "ref")

    val blindKernel = KernelConstructor.composeBlindFullKernel[TestRivalState]()

    val attribProfile = JointKernelProfile(Map(PlayerId("v1") -> attribKernel))
    val refProfile = JointKernelProfile(Map(PlayerId("v1") -> refKernel))
    val blindProfile = JointKernelProfile(Map(PlayerId("v1") -> blindKernel))

    val states = Map(PlayerId("v1") -> v1State)

    val attribResult = Dynamics.fullRivalUpdate(states, totalSignalNoSd, dummyPublicState, attribProfile)
    val refResult = Dynamics.fullRivalUpdate(states, totalSignalNoSd, dummyPublicState, refProfile)
    val blindResult = Dynamics.fullRivalUpdate(states, totalSignalNoSd, dummyPublicState, blindProfile)

    assertEquals(attribResult(PlayerId("v1")).updateCount, 100)
    assertEquals(refResult(PlayerId("v1")).updateCount, 200)
    assertEquals(blindResult(PlayerId("v1")).updateCount, 0) // blind = identity

  // ---- Def 24: Counterfactual reference world ----

  test("Def 24: referenceWorld uses joint reference profile"):
    val v1State = TestRivalState(uniformPrior, 0, "v1")

    val refKernel = new FullKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: TotalSignal, pub: PublicState): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 1, "ref-updated")

    val refProfile = JointKernelProfile(Map(PlayerId("v1") -> refKernel))
    val states = Map(PlayerId("v1") -> v1State)

    val cfWorld = Dynamics.counterfactualReferenceWorld(
      states, totalSignalNoSd, dummyPublicState, refProfile
    )
    assertEquals(cfWorld(PlayerId("v1")).label, "ref-updated")
    assertEquals(cfWorld(PlayerId("v1")).updateCount, 1)

  // ---- Def 25: Spot-conditioned polarization ----

  test("Def 25: spotPolarization returns polarization for a sizing in a spot"):
    val pol = UniformPolarization
    val sizing = Sizing(Chips(50.0), PotFraction(0.5))
    val rivalState = TestRivalState(uniformPrior)
    val result = pol.polarization(sizing, dummyPublicState, rivalState)
    assertEqualsDouble(result, 0.5, Tol) // uniform returns 0.5

  test("Def 25: PosteriorDivergencePolarization varies with sizing extremity"):
    val pol = PosteriorDivergencePolarization(uniformPrior)
    val rivalState = TestRivalState(uniformPrior)
    val halfPot = Sizing(Chips(50.0), PotFraction(0.5))
    val fullPot = Sizing(Chips(100.0), PotFraction(1.0))
    val minBet = Sizing(Chips(2.0), PotFraction(0.02))

    val polHalf = pol.polarization(halfPot, dummyPublicState, rivalState)
    val polFull = pol.polarization(fullPot, dummyPublicState, rivalState)
    val polMin = pol.polarization(minBet, dummyPublicState, rivalState)

    // Extreme sizings should be more polarizing than half-pot
    assert(polFull > polHalf, s"full pot ($polFull) should be more polarizing than half pot ($polHalf)")
    assert(polMin > polHalf, s"min bet ($polMin) should be more polarizing than half pot ($polHalf)")

  test("Def 25: polarization profile computes for all candidates"):
    val pol = UniformPolarization
    val rivalState = TestRivalState(uniformPrior)
    val candidates = Vector(
      Sizing(Chips(25.0), PotFraction(0.25)),
      Sizing(Chips(50.0), PotFraction(0.5)),
      Sizing(Chips(100.0), PotFraction(1.0))
    )
    val profile = pol.profile(candidates, dummyPublicState, rivalState)
    assertEquals(profile.size, 3)

  // ---- Multiway: Dynamics must handle |R| > 1 ----

  test("multiway: dynamics updates all rivals independently"):
    val v1State = TestRivalState(uniformPrior, 0, "v1")
    val v2State = TestRivalState(uniformPrior, 0, "v2")
    val v3State = TestRivalState(uniformPrior, 0, "v3")

    val kernel = new FullKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: TotalSignal, pub: PublicState): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 1, state.label + "-updated")

    val profile = JointKernelProfile(Map(
      PlayerId("v1") -> kernel,
      PlayerId("v2") -> kernel,
      PlayerId("v3") -> kernel
    ))

    val states = Map(
      PlayerId("v1") -> v1State,
      PlayerId("v2") -> v2State,
      PlayerId("v3") -> v3State
    )

    val updated = Dynamics.fullRivalUpdate(states, totalSignalNoSd, dummyPublicState, profile)
    assertEquals(updated.size, 3)
    assertEquals(updated(PlayerId("v1")).updateCount, 1)
    assertEquals(updated(PlayerId("v2")).updateCount, 1)
    assertEquals(updated(PlayerId("v3")).updateCount, 1)
    assertEquals(updated(PlayerId("v1")).label, "v1-updated")

  // ---- Integration: full dynamics step ----

  test("fullStep: combines rival update + exploitation retreat + belief update"):
    val v1State = TestRivalState(uniformPrior, 0, "v1")
    val config = ExploitationConfig(initialBeta = 0.8, retreatRate = 0.2, adaptationTolerance = 0.1)
    val exploitState = ExploitationState.initial(config)

    val kernel = new FullKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: TotalSignal, pub: PublicState): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 1, "stepped")

    val profile = JointKernelProfile(Map(PlayerId("v1") -> kernel))
    val rivalStates = Map(PlayerId("v1") -> v1State)
    val exploitStates = Map(PlayerId("v1") -> exploitState)

    val step = Dynamics.fullStep(
      rivalStates = rivalStates,
      exploitStates = exploitStates,
      signal = totalSignalNoSd,
      publicState = dummyPublicState,
      kernelProfile = profile,
      exploitConfigs = Map(PlayerId("v1") -> config),
      detector = NeverDetect,
      exploitabilityFn = _ => 0.0,
      epsilonNE = 0.0
    )
    assertEquals(step.updatedRivals(PlayerId("v1")).updateCount, 1)
    assertEqualsDouble(step.updatedExploitation(PlayerId("v1")).beta, 0.8, Tol) // no retreat

  test("fullStep with detection: retreat happens"):
    val v1State = TestRivalState(uniformPrior, 0, "v1")
    val config = ExploitationConfig(initialBeta = 1.0, retreatRate = 0.4, adaptationTolerance = 0.1)
    val exploitState = ExploitationState.initial(config)

    val kernel = KernelConstructor.composeBlindFullKernel[TestRivalState]()
    val profile = JointKernelProfile(Map(PlayerId("v1") -> kernel))
    val rivalStates = Map(PlayerId("v1") -> v1State)
    val exploitStates = Map(PlayerId("v1") -> exploitState)

    val step = Dynamics.fullStep(
      rivalStates = rivalStates,
      exploitStates = exploitStates,
      signal = totalSignalNoSd,
      publicState = dummyPublicState,
      kernelProfile = profile,
      exploitConfigs = Map(PlayerId("v1") -> config),
      detector = AlwaysDetect,
      exploitabilityFn = _ => 0.0,
      epsilonNE = 0.0
    )
    assertEqualsDouble(step.updatedExploitation(PlayerId("v1")).beta, 0.6, Tol) // retreated from 1.0

  // ---- Backward compatibility (v0.30.2 §12.2) ----

  test("backward compat: beta=1, no detection, blind kernel = identity dynamics"):
    val v1State = TestRivalState(uniformPrior, 0, "v1")
    val config = ExploitationConfig(initialBeta = 1.0, retreatRate = 0.0, adaptationTolerance = Double.MaxValue)
    val exploitState = ExploitationState.initial(config)

    val blindKernel = KernelConstructor.composeBlindFullKernel[TestRivalState]()
    val profile = JointKernelProfile(Map(PlayerId("v1") -> blindKernel))
    val rivalStates = Map(PlayerId("v1") -> v1State)
    val exploitStates = Map(PlayerId("v1") -> exploitState)

    val step = Dynamics.fullStep(
      rivalStates = rivalStates,
      exploitStates = exploitStates,
      signal = totalSignalNoSd,
      publicState = dummyPublicState,
      kernelProfile = profile,
      exploitConfigs = Map(PlayerId("v1") -> config),
      detector = NeverDetect,
      exploitabilityFn = _ => 0.0,
      epsilonNE = 0.0
    )
    // Blind kernel = identity, beta stays 1, no retreat
    assert(step.updatedRivals(PlayerId("v1")) eq v1State)
    assertEqualsDouble(step.updatedExploitation(PlayerId("v1")).beta, 1.0, Tol)
```

- [ ] **Step 2: Run test — expect compilation failure (RED)**

Run: `sbt "testOnly sicfun.holdem.strategic.DynamicsTest"`
Expected: Compilation error

- [ ] **Step 3: Write the implementation**

```scala
package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution

/** Result of a full dynamics step.
  *
  * Contains updated rival states and exploitation states for all rivals.
  */
final case class DynamicsStepResult[M <: RivalBeliefState](
    updatedRivals: Map[PlayerId, M],
    updatedExploitation: Map[PlayerId, ExploitationState]
)

/** Belief and rival-state dynamics (Defs 22-25). */
object Dynamics:

  /** Belief update (Def 22).
    *
    * b_{t+1} = tau_tilde(b_t, u_t, Z_{t+1})
    *
    * The belief update is parameterized by an updater function that
    * transforms the operative belief given the observed signal and
    * public state. This is abstract because the specific Bayesian
    * update depends on the state representation.
    */
  def beliefUpdate(
      belief: OperativeBelief,
      signal: TotalSignal,
      publicState: PublicState,
      updater: (OperativeBelief, TotalSignal, PublicState) => OperativeBelief
  ): OperativeBelief =
    updater(belief, signal, publicState)

  /** Full rival-state update (Def 23).
    *
    * m_{t+1}^{R,i} = Gamma^{full,bullet,i}(m_t^{R,i}, Y_t, x_t^pub)
    *
    * Applies the kernel profile to update all rival states.
    * The bullet (variant) is determined by which profile is passed in.
    * Rivals without a kernel in the profile are preserved unchanged.
    */
  def fullRivalUpdate[M <: RivalBeliefState](
      rivalStates: Map[PlayerId, M],
      signal: TotalSignal,
      publicState: PublicState,
      profile: JointKernelProfile[M]
  ): Map[PlayerId, M] =
    rivalStates.map { case (id, state) =>
      profile.kernels.get(id) match
        case Some(kernel) =>
          id -> kernel.apply(state, signal, publicState).asInstanceOf[M]
        case None =>
          id -> state
    }

  /** Counterfactual reference world (Def 24).
    *
    * The non-manipulative counterfactual world is the joint reference
    * profile Gamma^{ref}. The symbol pi_t^{cf,S} is retired.
    *
    * This is simply fullRivalUpdate with the reference profile.
    */
  def counterfactualReferenceWorld[M <: RivalBeliefState](
      rivalStates: Map[PlayerId, M],
      signal: TotalSignal,
      publicState: PublicState,
      refProfile: JointKernelProfile[M]
  ): Map[PlayerId, M] =
    fullRivalUpdate(rivalStates, signal, publicState, refProfile)

  /** Full dynamics step: rival update + exploitation update.
    *
    * Combines:
    * - Def 23: rival state update via kernel profile
    * - Def 15C: exploitation interpolation + detection retreat + safety clamp
    *
    * This is the main entry point for a single time step of the dynamics.
    */
  def fullStep[M <: RivalBeliefState](
      rivalStates: Map[PlayerId, M],
      exploitStates: Map[PlayerId, ExploitationState],
      signal: TotalSignal,
      publicState: PublicState,
      kernelProfile: JointKernelProfile[M],
      exploitConfigs: Map[PlayerId, ExploitationConfig],
      detector: DetectionPredicate,
      exploitabilityFn: Double => Double,
      epsilonNE: Double
  ): DynamicsStepResult[M] =
    // Step 1: Update rival states via kernel profile (Def 23)
    val updatedRivals = fullRivalUpdate(rivalStates, signal, publicState, kernelProfile)

    // Step 2: Update exploitation states (Def 15C + A6')
    val updatedExploit = exploitStates.map { case (rivalId, state) =>
      val config = exploitConfigs.getOrElse(rivalId,
        ExploitationConfig(initialBeta = 1.0, retreatRate = 0.0, adaptationTolerance = Double.MaxValue)
      )
      val updated = ExploitationInterpolation.updateExploitation(
        state = state,
        config = config,
        rivalId = rivalId,
        history = publicState.actionHistory,
        publicState = publicState,
        detector = detector,
        exploitabilityFn = exploitabilityFn,
        epsilonNE = epsilonNE
      )
      rivalId -> updated
    }

    DynamicsStepResult(updatedRivals, updatedExploit)
```

- [ ] **Step 4: Run test — expect all pass (GREEN)**

Run: `sbt "testOnly sicfun.holdem.strategic.DynamicsTest"`
Expected: All tests pass

- [ ] **Step 5: Commit**

---

### Task 6: Full Verification

- [ ] **Step 1: Run all Phase 4a tests**

```bash
sbt "testOnly sicfun.holdem.strategic.KernelConstructorTest sicfun.holdem.strategic.ExploitationInterpolationTest sicfun.holdem.strategic.DynamicsTest"
```

Expected: All tests pass.

- [ ] **Step 2: Run full strategic package tests (Phases 1 + 4a)**

```bash
sbt "testOnly sicfun.holdem.strategic.*"
```

Expected: All strategic package tests pass (Phase 1 tests still green).

- [ ] **Step 3: Verify no prohibited imports**

```bash
grep -rn "import sicfun.holdem.engine" src/main/scala/sicfun/holdem/strategic/
grep -rn "import sicfun.holdem.runtime" src/main/scala/sicfun/holdem/strategic/
```

Expected: No matches. If any match, fix immediately.

- [ ] **Step 4: Run existing test suite (regression check)**

```bash
sbt test
```

Expected: All existing tests still pass. The formal layer is strictly additive.

- [ ] **Step 5: Final commit**

---

## Coverage Matrix

| Spec Def | File | What | Test |
|----------|------|------|------|
| A6' | `DetectionPredicate.scala` | DetectModeling^i predicate | ExploitationInterpolationTest |
| Def 15C | `ExploitationInterpolation.scala` | beta^{i,exploit}, retreat, safety | ExploitationInterpolationTest |
| Def 16 | `KernelConstructor.scala` | StateEmbeddingUpdater (impl) | KernelConstructorTest |
| Def 17 | `KernelConstructor.scala` | BuildRivalKernel (impl) | KernelConstructorTest |
| Def 18 | `KernelConstructor.scala` | Ref/Attrib/Blind action kernels | KernelConstructorTest |
| Def 19 | `KernelConstructor.scala` | ShowdownKernel composition | KernelConstructorTest |
| Def 19A | `KernelConstructor.scala` | DesignSignalKernel (strips sizing/timing) | KernelConstructorTest |
| Def 20 | `KernelConstructor.scala` | FullKernel composition (action+showdown) | KernelConstructorTest |
| Def 21 | `KernelConstructor.scala` | JointKernelProfile | KernelConstructorTest |
| Def 22 | `Dynamics.scala` | beliefUpdate | DynamicsTest |
| Def 23 | `Dynamics.scala` | fullRivalUpdate | DynamicsTest |
| Def 24 | `Dynamics.scala` | counterfactualReferenceWorld | DynamicsTest |
| Def 25 | `SpotPolarization.scala` | Spot-conditioned polarization | DynamicsTest |
| A9 | `SpotPolarization.scala` | Polarization profile trait | DynamicsTest |

## Laws Verified

| Law | Where | How |
|-----|-------|-----|
| L12 | KernelConstructorTest | BlindActionKernel returns `eq` same object |
| L13 | KernelConstructorTest (via Phase 1) | ActionKernel.apply has no cross-rival param |
| L7 | DynamicsTest | Multiway test with |R|=3 |
| L8 | DynamicsTest | No implicit collapse to single opponent |

## Backward Compatibility Tests

| Condition | Test | Location |
|-----------|------|----------|
| beta=1 for all rivals | `backward compat: beta=1 recovers v0.29.1` | ExploitationInterpolationTest |
| beta=1 + blind kernel | `backward compat: beta=1, blind kernel = identity` | DynamicsTest |
| Full attribution kernel | `backward compat: full attribution kernel` | KernelConstructorTest |

## Design Notes

1. **TemperedLikelihoodFn is a function type, not a class.** KernelConstructor depends on Phase 2 only through this type alias, keeping the dependency minimal.

2. **JointKernelProfile extends the Phase 1 KernelProfile.** Phase 1 defined `KernelProfile` with `ActionKernel` values. Phase 4a introduces `JointKernelProfile` with `FullKernel` values (action + showdown composition). Both coexist.

3. **Dynamics.fullStep is the main orchestration point.** It combines rival state update (Def 23) with exploitation update (Def 15C + A6'). Higher-level orchestration (belief update, value computation) is in Phase 4b.

4. **SpotPolarization is a trait with pluggable implementations.** `UniformPolarization` is the stub. `PosteriorDivergencePolarization` is the canonical implementation. The bridge layer will provide production implementations that use actual posterior computation.

5. **DetectionPredicate is a trait.** `NeverDetect` and `AlwaysDetect` are test stubs. `FrequencyAnomalyDetection` is a concrete implementation. Production detection may use more sophisticated methods.

6. **No PublicState dummy proliferation.** `buildActionKernel` creates a minimal dummy PublicState internally. The production form `buildActionKernelFull` threads PublicState properly. Tests use `buildActionKernel` for simplicity.

7. **Design kernel (Def 19A) enforces temper-then-marginalize.** The kernel strips sizing/timing before passing to the likelihood function. The likelihood function receives a signal with `sizing=None, timing=None` and is responsible for the correct marginalization order internally.
