package sicfun.holdem.strategic

import sicfun.core.{CardId, DiscreteDistribution}
import sicfun.holdem.types.{Board, HoleCards, PokerAction, Position, Street}

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

  // ---- Def 22: Belief update ----

  test("Def 22: beliefUpdate produces updated augmented state"):
    val dummyBeliefState = new RivalBeliefState:
      def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState = this

    val dummyOMS = OpponentModelState(
      typeDistribution = DiscreteDistribution.uniform(Seq("TAG", "LAG")),
      beliefState = dummyBeliefState,
      attributedBaseline = None
    )

    val dummyHoleCards = HoleCards.canonical(CardId.fromId(0), CardId.fromId(1))

    val rivals = RivalMap(Vector(
      Seat(PlayerId("v1"), Position.BigBlind, SeatStatus.Active, dummyOMS),
      Seat(PlayerId("v2"), Position.Button, SeatStatus.Active, dummyOMS)
    ))
    val augState = AugmentedState(
      publicState = dummyPublicState,
      privateHand = dummyHoleCards,
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

  test("Def 25: PosteriorDivergencePolarization with real KL divergence"):
    // Likelihood that shifts posterior toward Value for large sizings
    val klLikelihood: TemperedLikelihoodFn = (signal, _, _) =>
      val sizingFrac = signal.sizing.map(_.fractionOfPot.value).getOrElse(0.5)
      if sizingFrac > 0.7 then
        // Large sizing → strong signal toward Value
        DiscreteDistribution(Map(
          StrategicClass.Value -> 0.7, StrategicClass.Bluff -> 0.1,
          StrategicClass.Marginal -> 0.1, StrategicClass.SemiBluff -> 0.1
        ))
      else
        // Small sizing → stays near uniform (low divergence)
        DiscreteDistribution(Map(
          StrategicClass.Value -> 0.28, StrategicClass.Bluff -> 0.24,
          StrategicClass.Marginal -> 0.24, StrategicClass.SemiBluff -> 0.24
        ))

    val pol = PosteriorDivergencePolarization(uniformPrior, Some(klLikelihood))
    assertEquals(pol.fidelity, Fidelity.Exact)
    val rivalState = TestRivalState(uniformPrior)
    val smallSizing = Sizing(Chips(30.0), PotFraction(0.3))
    val largeSizing = Sizing(Chips(100.0), PotFraction(1.0))

    val polSmall = pol.polarization(smallSizing, dummyPublicState, rivalState)
    val polLarge = pol.polarization(largeSizing, dummyPublicState, rivalState)

    // Large sizing should have higher polarization (more KL divergence)
    assert(polLarge > polSmall, s"large ($polLarge) should be more polarizing than small ($polSmall)")
    // Both should be in [0, 1]
    assert(polSmall >= 0.0 && polSmall <= 1.0)
    assert(polLarge >= 0.0 && polLarge <= 1.0)

  test("Def 25: KL divergence is zero for identical distributions"):
    val kl = PosteriorDivergencePolarization.klDivergence(uniformPrior, uniformPrior)
    assertEqualsDouble(kl, 0.0, 1e-12)

  test("Def 25: KL divergence is positive for different distributions"):
    val skewed = DiscreteDistribution(Map(
      StrategicClass.Value -> 0.7, StrategicClass.Bluff -> 0.1,
      StrategicClass.Marginal -> 0.1, StrategicClass.SemiBluff -> 0.1
    ))
    val kl = PosteriorDivergencePolarization.klDivergence(skewed, uniformPrior)
    assert(kl > 0.0, s"KL divergence should be positive, got $kl")

  test("Def 25: proxy fallback when no likelihood provided"):
    val pol = PosteriorDivergencePolarization(uniformPrior, None)
    assertEquals(pol.fidelity, Fidelity.Approximate)
    // Should still compute via proxy without error
    val rivalState = TestRivalState(uniformPrior)
    val sizing = Sizing(Chips(50.0), PotFraction(0.5))
    val result = pol.polarization(sizing, dummyPublicState, rivalState)
    assert(result >= 0.0 && result <= 1.0)

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

  // ---- Chain-world aware dynamics ----

  test("same signal under (Attrib,Off) and (Attrib,On) yields different results only via showdown"):
    val v1State = TestRivalState(uniformPrior, 0, "v1")

    val actionKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 1, "action")

    val designKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 50, "design")

    val sdKernel = new ShowdownKernel[TestRivalState]:
      def apply(state: TestRivalState, showdown: ShowdownSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 100, state.label + "+sd")

    val attribOff = ChainWorld(LearningChannel.Attrib, ShowdownMode.Off)
    val attribOn = ChainWorld(LearningChannel.Attrib, ShowdownMode.On)

    val offKernel = KernelConstructor.composeFullKernelForWorld(attribOff, actionKernel, designKernel, sdKernel)
    val onKernel = KernelConstructor.composeFullKernelForWorld(attribOn, actionKernel, designKernel, sdKernel)

    val worldProfile = WorldIndexedKernelProfile(Map(
      (PlayerId("v1"), attribOff) -> offKernel,
      (PlayerId("v1"), attribOn) -> onKernel
    ))

    val states = Map(PlayerId("v1") -> v1State)

    val showdownSig = ShowdownSignal(Vector(RevealedHand(PlayerId("v1"), Vector.empty)))
    val signalWithSd = TotalSignal(raiseSignal, Some(showdownSig))

    val resultOff = Dynamics.fullRivalUpdate(states, signalWithSd, dummyPublicState, worldProfile, attribOff)
    val resultOn = Dynamics.fullRivalUpdate(states, signalWithSd, dummyPublicState, worldProfile, attribOn)

    // Off: action only (1), On: action + showdown (1 + 100 = 101)
    assertEquals(resultOff(PlayerId("v1")).updateCount, 1)
    assertEquals(resultOn(PlayerId("v1")).updateCount, 101)

  test("counterfactualReferenceWorld with explicit ChainWorld(Ref, Off) produces expected result"):
    val v1State = TestRivalState(uniformPrior, 0, "v1")

    val actionKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 1, "ref-action")

    val designKernel = new ActionKernel[TestRivalState]:
      def apply(state: TestRivalState, signal: ActionSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 50, "design")

    val sdKernel = new ShowdownKernel[TestRivalState]:
      def apply(state: TestRivalState, showdown: ShowdownSignal): TestRivalState =
        TestRivalState(state.posterior, state.updateCount + 100, state.label + "+sd")

    val refOff = ChainWorld(LearningChannel.Ref, ShowdownMode.Off)
    val refKernel = KernelConstructor.composeFullKernelForWorld(refOff, actionKernel, designKernel, sdKernel)

    val worldProfile = WorldIndexedKernelProfile(Map(
      (PlayerId("v1"), refOff) -> refKernel
    ))

    val states = Map(PlayerId("v1") -> v1State)
    val showdownSig = ShowdownSignal(Vector(RevealedHand(PlayerId("v1"), Vector.empty)))
    val signalWithSd = TotalSignal(raiseSignal, Some(showdownSig))

    val cfResult = Dynamics.counterfactualReferenceWorld(
      states, signalWithSd, dummyPublicState, worldProfile, refOff
    )

    // Ref + Off: action only, showdown gated off
    assertEquals(cfResult(PlayerId("v1")).updateCount, 1)
    assertEquals(cfResult(PlayerId("v1")).label, "ref-action")

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
