package sicfun.holdem.strategic

import sicfun.holdem.types.{Position, Street, Board}
import sicfun.core.DiscreteDistribution

class RivalKernelLawTest extends munit.FunSuite:

  private val dummyPublicState = PublicState(
    street = Street.River,
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

  private val dummySignal = ActionSignal(
    action = sicfun.holdem.types.PokerAction.Category.Raise,
    sizing = Some(Sizing(Chips(100.0), PotFraction(0.5))),
    timing = None,
    stage = Street.Flop
  )

  test("StateEmbeddingUpdater type alias exists and compiles"):
    val updater: StateEmbeddingUpdater[RivalBeliefState] =
      (state: RivalBeliefState, posterior: DiscreteDistribution[StrategicClass]) => state
    assert(updater != null)

  test("BlindActionKernel returns exact same state object"):
    val belief = new RivalBeliefState:
      def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState = this
    val kernel = BlindActionKernel[RivalBeliefState]()
    val result = kernel.apply(belief, dummySignal)
    assert(result eq belief, "Blind kernel must return the exact same state object")

  test("FullKernel trait has apply(state, totalSignal, publicState)"):
    val dummy = new FullKernel[RivalBeliefState]:
      def apply(
          state: RivalBeliefState,
          signal: TotalSignal,
          publicState: PublicState
      ): RivalBeliefState = state

    val belief = new RivalBeliefState:
      def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState = this

    val totalSignal = TotalSignal(dummySignal, showdown = None)
    val result = dummy.apply(belief, totalSignal, dummyPublicState)
    assert(result eq belief)

  test("KernelProfile maps PlayerId to kernels"):
    val k1 = BlindActionKernel[RivalBeliefState]()
    val k2 = BlindActionKernel[RivalBeliefState]()
    val profile = KernelProfile(Map(
      PlayerId("v1") -> k1,
      PlayerId("v2") -> k2
    ))
    assertEquals(profile.kernels.size, 2)
    assert(profile.kernels.contains(PlayerId("v1")))
    assert(profile.kernels.contains(PlayerId("v2")))

  test("ActionKernel.apply signature has exactly (M, ActionSignal) -- no cross-rival arg"):
    val kernel: ActionKernel[RivalBeliefState] = BlindActionKernel[RivalBeliefState]()
    val belief = new RivalBeliefState:
      def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState = this
    val _ = kernel.apply(belief, dummySignal)

  test("KernelVariant has exactly four variants"):
    assertEquals(KernelVariant.values.length, 4)
    val names = KernelVariant.values.map(_.toString).toSet
    assertEquals(names, Set("Ref", "Attrib", "Blind", "Design"))

  test("KernelVariant.toLearningChannel round-trips correctly"):
    assertEquals(KernelVariant.Ref.toLearningChannel, LearningChannel.Ref)
    assertEquals(KernelVariant.Attrib.toLearningChannel, LearningChannel.Attrib)
    assertEquals(KernelVariant.Blind.toLearningChannel, LearningChannel.Blind)
    assertEquals(KernelVariant.Design.toLearningChannel, LearningChannel.Design)

  test("blind worlds (Blind, Off) and (Blind, On) are behaviorally equivalent (both identity)"):
    val belief = new RivalBeliefState:
      def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState = this

    val actionKernel = new ActionKernel[RivalBeliefState]:
      def apply(state: RivalBeliefState, signal: ActionSignal): RivalBeliefState =
        new RivalBeliefState:
          def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState = this

    val designKernel = new ActionKernel[RivalBeliefState]:
      def apply(state: RivalBeliefState, signal: ActionSignal): RivalBeliefState =
        new RivalBeliefState:
          def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState = this

    val sdKernel = new ShowdownKernel[RivalBeliefState]:
      def apply(state: RivalBeliefState, showdown: ShowdownSignal): RivalBeliefState =
        new RivalBeliefState:
          def update(signal: ActionSignal, publicState: PublicState): RivalBeliefState = this

    val blindOff = KernelConstructor.composeFullKernelForWorld(
      ChainWorld(LearningChannel.Blind, ShowdownMode.Off), actionKernel, designKernel, sdKernel
    )
    val blindOn = KernelConstructor.composeFullKernelForWorld(
      ChainWorld(LearningChannel.Blind, ShowdownMode.On), actionKernel, designKernel, sdKernel
    )

    val sd = ShowdownSignal(Vector(RevealedHand(PlayerId("v1"), Vector.empty)))
    val signal = TotalSignal(dummySignal, Some(sd))

    val resultOff = blindOff.apply(belief, signal, dummyPublicState)
    val resultOn = blindOn.apply(belief, signal, dummyPublicState)

    assert(resultOff eq belief, "Blind Off must return same state")
    assert(resultOn eq belief, "Blind On must return same state")
