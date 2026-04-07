package sicfun.holdem.strategic

import sicfun.core.DiscreteDistribution
import sicfun.holdem.types.{Board, PokerAction, Position, Street}

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
    StrategicClass.Value     -> 0.25,
    StrategicClass.Bluff     -> 0.25,
    StrategicClass.Marginal  -> 0.25,
    StrategicClass.SemiBluff -> 0.25
  ))

  private val dummyPublicState = PublicState(
    street = Street.Flop,
    board = Board.empty,
    pot = Chips(100.0),
    stacks = TableMap(
      hero = PlayerId("hero"),
      seats = Vector(
        Seat(PlayerId("hero"), Position.SmallBlind, SeatStatus.Active, Chips(500.0)),
        Seat(PlayerId("v1"), Position.BigBlind, SeatStatus.Active, Chips(500.0))
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

  private val showdownSignal = ShowdownSignal(
    revealedHands = Vector(
      RevealedHand(PlayerId("v1"), Vector.empty)
    )
  )

  // ---- Def 16: StateEmbeddingUpdater ----

  test("StateEmbeddingUpdater: embeds posterior into rival state"):
    val updater: StateEmbeddingUpdater[TestRivalState] =
      (state, posterior) => TestRivalState(posterior, state.updateCount + 1)

    val initial = TestRivalState(uniformPrior)
    val shiftedPosterior = DiscreteDistribution(Map(
      StrategicClass.Value     -> 0.7,
      StrategicClass.Bluff     -> 0.1,
      StrategicClass.Marginal  -> 0.1,
      StrategicClass.SemiBluff -> 0.1
    ))
    val result = updater(initial, shiftedPosterior)
    assertEqualsDouble(result.posterior.probabilityOf(StrategicClass.Value), 0.7, Tol)
    assertEquals(result.updateCount, 1)

  // ---- Def 17: BuildRivalKernel ----

  test("BuildRivalKernel produces an ActionKernel from a policy reference"):
    val updater: StateEmbeddingUpdater[TestRivalState] =
      (state, posterior) => TestRivalState(posterior, state.updateCount + 1)

    // Stub likelihood: always returns uniform posterior regardless of signal
    val stubLikelihood: TemperedLikelihoodFn = (signal, publicState, state) => uniformPrior

    val kernel = KernelConstructor.buildActionKernelFull(updater, stubLikelihood)
    val initial = TestRivalState(uniformPrior)
    val result = kernel.apply(initial, raiseSignal, dummyPublicState).asInstanceOf[TestRivalState]
    assertEquals(result.updateCount, 1)

  // ---- Def 18: Reference vs Attributed vs Blind ----

  test("RefActionKernel updates state using reference posterior"):
    val updater: StateEmbeddingUpdater[TestRivalState] =
      (state, posterior) => TestRivalState(posterior, state.updateCount + 1)
    val refLikelihood: TemperedLikelihoodFn = (_, _, _) =>
      DiscreteDistribution(Map(
        StrategicClass.Value     -> 0.6,
        StrategicClass.Bluff     -> 0.1,
        StrategicClass.Marginal  -> 0.2,
        StrategicClass.SemiBluff -> 0.1
      ))

    val kernel = KernelConstructor.buildActionKernelFull(updater, refLikelihood)
    val initial = TestRivalState(uniformPrior)
    val result = kernel.apply(initial, raiseSignal, dummyPublicState).asInstanceOf[TestRivalState]
    assertEqualsDouble(result.posterior.probabilityOf(StrategicClass.Value), 0.6, Tol)

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

  // ---- composeFullKernelForWorld: chain-world routing ----

  test("composeFullKernelForWorld: (Blind, Off) returns identity"):
    val actionKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 1)

    val designKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 50)

    val sdKernel = new ShowdownKernel[TestRivalState]:
      def apply(state: TestRivalState, showdown: ShowdownSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 100)

    val world = ChainWorld(LearningChannel.Blind, ShowdownMode.Off)
    val full = KernelConstructor.composeFullKernelForWorld(world, actionKernel, designKernel, sdKernel)
    val initial = TestRivalState(uniformPrior)

    val withSd = TotalSignal(raiseSignal, Some(showdownSignal))
    val result = full.apply(initial, withSd, dummyPublicState)
    assert(result eq initial, "(Blind, Off) must be identity")

  test("composeFullKernelForWorld: (Blind, On) returns identity"):
    val actionKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 1)

    val designKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 50)

    val sdKernel = new ShowdownKernel[TestRivalState]:
      def apply(state: TestRivalState, showdown: ShowdownSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 100)

    val world = ChainWorld(LearningChannel.Blind, ShowdownMode.On)
    val full = KernelConstructor.composeFullKernelForWorld(world, actionKernel, designKernel, sdKernel)
    val initial = TestRivalState(uniformPrior)

    val withSd = TotalSignal(raiseSignal, Some(showdownSignal))
    val result = full.apply(initial, withSd, dummyPublicState)
    assert(result eq initial, "(Blind, On) must also be identity")

  test("composeFullKernelForWorld: showdown skipped under ShowdownMode.Off"):
    val actionKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 1)

    val designKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 50)

    val sdKernel = new ShowdownKernel[TestRivalState]:
      def apply(state: TestRivalState, showdown: ShowdownSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 100)

    val world = ChainWorld(LearningChannel.Attrib, ShowdownMode.Off)
    val full = KernelConstructor.composeFullKernelForWorld(world, actionKernel, designKernel, sdKernel)
    val initial = TestRivalState(uniformPrior)

    // Even with showdown present in signal, Off world must not apply it
    val withSd = TotalSignal(raiseSignal, Some(showdownSignal))
    val result = full.apply(initial, withSd, dummyPublicState).asInstanceOf[TestRivalState]
    assertEquals(result.updateCount, 1) // action only, no showdown

  test("composeFullKernelForWorld: showdown applied under ShowdownMode.On"):
    val actionKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 1)

    val designKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 50)

    val sdKernel = new ShowdownKernel[TestRivalState]:
      def apply(state: TestRivalState, showdown: ShowdownSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 100)

    val world = ChainWorld(LearningChannel.Attrib, ShowdownMode.On)
    val full = KernelConstructor.composeFullKernelForWorld(world, actionKernel, designKernel, sdKernel)
    val initial = TestRivalState(uniformPrior)

    val withSd = TotalSignal(raiseSignal, Some(showdownSignal))
    val result = full.apply(initial, withSd, dummyPublicState).asInstanceOf[TestRivalState]
    assertEquals(result.updateCount, 101) // action (1) + showdown (100)

  test("composeFullKernelForWorld: design channel strips sizing/timing AND composes with showdown when On"):
    val actionKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 1)

    val designKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 50)

    val sdKernel = new ShowdownKernel[TestRivalState]:
      def apply(state: TestRivalState, showdown: ShowdownSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 100)

    // Design + On: should use design kernel AND apply showdown
    val world = ChainWorld(LearningChannel.Design, ShowdownMode.On)
    val full = KernelConstructor.composeFullKernelForWorld(world, actionKernel, designKernel, sdKernel)
    val initial = TestRivalState(uniformPrior)

    val withSd = TotalSignal(raiseSignal, Some(showdownSignal))
    val result = full.apply(initial, withSd, dummyPublicState).asInstanceOf[TestRivalState]
    assertEquals(result.updateCount, 150) // design (50) + showdown (100)

  test("composeFullKernelForWorld: (Ref, Off) applies action only, no showdown"):
    val actionKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 1)

    val designKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 50)

    val sdKernel = new ShowdownKernel[TestRivalState]:
      def apply(state: TestRivalState, showdown: ShowdownSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 100)

    val world = ChainWorld(LearningChannel.Ref, ShowdownMode.Off)
    val full = KernelConstructor.composeFullKernelForWorld(world, actionKernel, designKernel, sdKernel)
    val initial = TestRivalState(uniformPrior)

    val withSd = TotalSignal(raiseSignal, Some(showdownSignal))
    val result = full.apply(initial, withSd, dummyPublicState).asInstanceOf[TestRivalState]
    assertEquals(result.updateCount, 1) // action only

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
        StrategicClass.Value     -> 0.8,
        StrategicClass.Bluff     -> 0.05,
        StrategicClass.Marginal  -> 0.1,
        StrategicClass.SemiBluff -> 0.05
      ))
    val updater: StateEmbeddingUpdater[TestRivalState] =
      (state, posterior) => TestRivalState(posterior, state.updateCount + 1)

    val attribKernel = KernelConstructor.buildActionKernelFull(updater, attribLikelihood)
    val initial = TestRivalState(uniformPrior)
    val result = attribKernel.apply(initial, raiseSignal, dummyPublicState).asInstanceOf[TestRivalState]
    // beta=1 means full attributed: posterior should be the attributed one
    assertEqualsDouble(result.posterior.probabilityOf(StrategicClass.Value), 0.8, Tol)
