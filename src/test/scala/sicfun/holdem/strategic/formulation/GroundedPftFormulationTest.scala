package sicfun.holdem.strategic.formulation

import munit.FunSuite
import sicfun.core.{Card, Rank, Suit}
import sicfun.holdem.engine.{PokerPftFormulation, PokerPomcpFormulation, StrategicEngine}
import sicfun.holdem.strategic.types.StrategicClass
import sicfun.holdem.types.*

class GroundedPftFormulationTest extends FunSuite:

  private val aceKingSuited = HoleCards(
    Card(Rank.Ace, Suit.Spades),
    Card(Rank.King, Suit.Spades)
  )

  private val twoSevenOff = HoleCards(
    Card(Rank.Two, Suit.Clubs),
    Card(Rank.Seven, Suit.Hearts)
  )

  private val defaultActions = Vector(
    PokerAction.Fold,
    PokerAction.Call,
    PokerAction.Raise(50.0)
  )

  private def preflopState: GameState =
    GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 100.0,
      toCall = 20.0,
      position = Position.Button,
      stackSize = 1000.0,
      betHistory = Vector.empty
    )

  private def makeGroundedInput(
      cards: HoleCards,
      gs: GameState = preflopState,
      actions: Vector[PokerAction] = defaultActions
  ): FormulationInput =
    val rivalPriors = LegacyRivalPriors(
      pftActionPriors = StrategicEngine.defaultActionPriors,
      pomcpClassPriors = PokerPomcpFormulation.defaultClassPriors
    )
    val legacyBase = LegacyToyFormulationInput.from(
      gs,
      actions,
      Map.empty,
      heroBucket = 5,
      rivalPriors
    )
    FormulationInput(
      spot = FormulationSpot(
        gameState = gs,
        candidateActions = actions,
        heroValueInput = HeroValueInput.ExactHoleCards(cards),
        rivalBeliefs = Map.empty
      ),
      valueSource = GroundedValueSource,
      rivalPolicySource = legacyBase.rivalPolicySource,
      actionSource = legacyBase.actionSource
    )

  test("grounded model has 5 states (4 streets + terminal sink)"):
    val model = PokerPftFormulation.buildTabularModel(
      makeGroundedInput(aceKingSuited),
      profileClass = None
    )
    assertEquals(model.numStates, 5)
    assertEquals(model.numActions, 3)
    assertEquals(model.numObs, StrategicClass.values.length)
    assertEquals(model.transitionTable.length, 15)
    assertEquals(model.rewardTable.length, 15)
    assertEquals(model.obsLikelihood.length, 15 * model.numObs)

  test("grounded transitions: fold goes to terminal sink at every street"):
    val model = PokerPftFormulation.buildTabularModel(
      makeGroundedInput(aceKingSuited),
      profileClass = None
    )
    val numActions = model.numActions
    val terminalState = 4
    for s <- 0 until 4 do
      assertEquals(model.transitionTable(s * numActions), terminalState)

  test("grounded transitions: river non-fold goes to terminal sink"):
    val model = PokerPftFormulation.buildTabularModel(
      makeGroundedInput(aceKingSuited),
      profileClass = None
    )
    val numActions = model.numActions
    val riverState = 3
    val terminalState = 4
    assertEquals(model.transitionTable(riverState * numActions + 1), terminalState)
    assertEquals(model.transitionTable(riverState * numActions + 2), terminalState)

  test("grounded transitions: preflop non-fold advances street"):
    val model = PokerPftFormulation.buildTabularModel(
      makeGroundedInput(aceKingSuited),
      profileClass = None
    )
    assertEquals(model.transitionTable(1), 1)
    assertEquals(model.transitionTable(2), 1)

  test("grounded transitions: terminal sink self-loops"):
    val model = PokerPftFormulation.buildTabularModel(
      makeGroundedInput(aceKingSuited),
      profileClass = None
    )
    val numActions = model.numActions
    val terminalState = 4
    for a <- 0 until numActions do
      assertEquals(
        model.transitionTable(terminalState * numActions + a),
        terminalState
      )

  test("grounded rewards: terminal sink has zero rewards"):
    val model = PokerPftFormulation.buildTabularModel(
      makeGroundedInput(aceKingSuited),
      profileClass = None
    )
    val numActions = model.numActions
    val terminalState = 4
    for a <- 0 until numActions do
      assertEqualsDouble(
        model.rewardTable(terminalState * numActions + a),
        0.0,
        1e-15
      )

  test("grounded rewards: fold is negative for strong hand at street states"):
    val model = PokerPftFormulation.buildTabularModel(
      makeGroundedInput(aceKingSuited),
      profileClass = None
    )
    val numActions = model.numActions
    for s <- 0 until 4 do
      assert(model.rewardTable(s * numActions) < 0.0)

  test("grounded rewards: strong hand gets higher rewards than weak hand"):
    val strongModel = PokerPftFormulation.buildTabularModel(
      makeGroundedInput(aceKingSuited),
      profileClass = None
    )
    val weakModel = PokerPftFormulation.buildTabularModel(
      makeGroundedInput(twoSevenOff),
      profileClass = None
    )
    val strongCallReward = strongModel.rewardTable(1)
    val weakCallReward = weakModel.rewardTable(1)
    assert(strongCallReward > weakCallReward,
      s"AKs call reward ($strongCallReward) should exceed 72o ($weakCallReward)")

  test("grounded rewards remain finite across street states"):
    val model = PokerPftFormulation.buildTabularModel(
      makeGroundedInput(aceKingSuited),
      profileClass = None
    )
    val numActions = model.numActions
    for s <- 0 until 4 do
      for a <- 0 until numActions do
        val reward = model.rewardTable(s * numActions + a)
        assert(!reward.isNaN && !reward.isInfinite)

  test("grounded rewards differ from the legacy bucket model"):
    val grounded = PokerPftFormulation.buildTabularModel(
      makeGroundedInput(aceKingSuited),
      profileClass = None
    )
    val legacy = PokerPftFormulation.buildTabularModel(
      preflopState,
      Map.empty,
      defaultActions,
      5,
      StrategicEngine.defaultActionPriors
    )
    val groundedState0 = grounded.rewardTable.take(defaultActions.size)
    val legacyState0 = legacy.rewardTable.take(defaultActions.size)
    assert(!groundedState0.sameElements(legacyState0))

  test("grounded obs likelihood sums to 1 per (state, action)"):
    val model = PokerPftFormulation.buildTabularModel(
      makeGroundedInput(aceKingSuited),
      profileClass = None
    )
    val numObs = model.numObs
    for s <- 0 until model.numStates do
      for a <- 0 until model.numActions do
        val base = (s * model.numActions + a) * numObs
        val obsSum = (0 until numObs).map(o => model.obsLikelihood(base + o)).sum
        assertEqualsDouble(obsSum, 1.0, 1e-10)

  test("profile-conditioned model has different obs and rewards from mixed model"):
    val input = makeGroundedInput(aceKingSuited)
    val mixedModel = PokerPftFormulation.buildTabularModel(input, profileClass = None)
    val valueModel = PokerPftFormulation.buildTabularModel(
      input,
      profileClass = Some(StrategicClass.Value)
    )
    assert(!mixedModel.obsLikelihood.sameElements(valueModel.obsLikelihood))
    assert(!mixedModel.rewardTable.sameElements(valueModel.rewardTable))

  test("profile conditioning keeps fold reward but adjusts active actions"):
    val input = makeGroundedInput(aceKingSuited)
    val baseline = PokerPftFormulation.buildTabularModel(input, profileClass = None)
    val profiled = PokerPftFormulation.buildTabularModel(
      input,
      profileClass = Some(StrategicClass.Value)
    )
    assertEqualsDouble(baseline.rewardTable(0), profiled.rewardTable(0), 1e-12)
    assert(baseline.rewardTable(1) != profiled.rewardTable(1))
    assert(baseline.rewardTable(2) != profiled.rewardTable(2))
    val terminalState = 4
    for a <- 0 until profiled.numActions do
      assertEqualsDouble(
        profiled.rewardTable(terminalState * profiled.numActions + a),
        0.0,
        1e-15
      )

  test("grounded open-loop model has uniform obs"):
    val model = PokerPftFormulation.buildOpenLoopModel(
      makeGroundedInput(aceKingSuited),
      profileClass = None
    )
    val uniformP = 1.0 / model.numObs
    for i <- 0 until model.numStates * model.numActions do
      for o <- 0 until model.numObs do
        assertEqualsDouble(model.obsLikelihood(i * model.numObs + o), uniformP, 1e-12)

  test("grounded blind model has same rewards as baseline"):
    val input = makeGroundedInput(aceKingSuited)
    val baseline = PokerPftFormulation.buildTabularModel(input, profileClass = None)
    val blind = PokerPftFormulation.buildBlindKernelModel(input)
    assertEquals(blind.numStates, 5)
    baseline.rewardTable.zip(blind.rewardTable).foreach { (base, blindReward) =>
      assertEqualsDouble(base, blindReward, 1e-12)
    }

  test("grounded design-kernel model standardizes raise rewards"):
    val actions = Vector(
      PokerAction.Fold,
      PokerAction.Call,
      PokerAction.Raise(50.0),
      PokerAction.Raise(100.0)
    )
    val input = makeGroundedInput(aceKingSuited, actions = actions)
    val normal = PokerPftFormulation.buildTabularModel(input, profileClass = None)
    val design = PokerPftFormulation.buildDesignKernelModel(input)
    val numActions = design.numActions
    for s <- 0 until 4 do
      assertEqualsDouble(
        design.rewardTable(s * numActions + 2),
        design.rewardTable(s * numActions + 3),
        1e-12
      )
      assertEqualsDouble(
        normal.rewardTable(s * numActions),
        design.rewardTable(s * numActions),
        1e-12
      )
      assertEqualsDouble(
        normal.rewardTable(s * numActions + 1),
        design.rewardTable(s * numActions + 1),
        1e-12
      )
    val terminalState = 4
    for a <- 0 until numActions do
      assertEqualsDouble(
        design.rewardTable(terminalState * numActions + a),
        0.0,
        1e-15
      )

  test("grounded four-world models all have 5 states and a terminal sink"):
    val models = StrategicEngine.buildFourWorldModels(makeGroundedInput(aceKingSuited))
    val numActions = models.baseline.numActions
    val terminalState = 4
    for model <- Seq(models.baseline, models.openLoop, models.blind, models.blindOpenLoop) do
      assertEquals(model.numStates, 5)
      for s <- 0 until 4 do
        assertEquals(model.transitionTable(s * numActions), terminalState)
      for a <- 1 until numActions do
        assertEquals(model.transitionTable(3 * numActions + a), terminalState)
      for a <- 0 until numActions do
        assertEquals(model.transitionTable(terminalState * numActions + a), terminalState)
        assertEqualsDouble(model.rewardTable(terminalState * numActions + a), 0.0, 1e-15)
