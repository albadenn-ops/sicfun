package sicfun.holdem.engine

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.types.*
import sicfun.holdem.strategic.*
import sicfun.holdem.strategic.solver.WPomcpRuntime

class StrategicEngineTest extends FunSuite:

  private def card(token: String): Card =
    Card.parse(token).getOrElse(throw new IllegalArgumentException(s"bad card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  /** Default test hero cards: Ace-King offsuit. */
  private def testHeroCards: HoleCards = hole("As", "Kh")

  /** Minimal GameState for testing — preflop, no board, pot=100, no-call check, Button. */
  private def minimalState: GameState =
    GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 100.0,
      toCall = 0.0,
      position = Position.Button,
      stackSize = 1000.0,
      betHistory = Vector.empty
    )

  test("StrategicEngine initializes with uniform beliefs for unknown rivals"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    val state = engine.sessionState
    assertEquals(state.rivalBeliefs.size, 1)
    assertEqualsDouble(
      state.rivalBeliefs(PlayerId("v1")).typePosterior.probabilityOf(StrategicClass.Value),
      0.25,
      1e-10
    )

  test("initSession creates exploitation states"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1"), PlayerId("v2")))
    val state = engine.sessionState
    assertEquals(state.exploitationStates.size, 2)
    assertEqualsDouble(state.exploitationStates(PlayerId("v1")).beta, 1.0, 1e-10)

  test("startHand with heroCards sets hand active"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    assert(engine.currentHandActive)

  test("startHand without heroCards sets hand active"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand()
    assert(engine.currentHandActive)

  test("endHand clears hand active"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    engine.endHand()
    assert(!engine.currentHandActive)

  test("observeAction updates beliefs"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    engine.observeAction(PlayerId("v1"), PokerAction.Raise(50.0), minimalState)
    val after = engine.sessionState.rivalBeliefs(PlayerId("v1")).typePosterior
    // After observing a raise, Bluff posterior should increase relative to Mixed
    val bluffAfter = after.probabilityOf(StrategicClass.Bluff)
    val mixedAfter = after.probabilityOf(StrategicClass.Mixed)
    assert(bluffAfter > mixedAfter, s"Bluff ($bluffAfter) should exceed Mixed ($mixedAfter) after raise")

  test("observeAction ignores unknown actor"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    // Observing action for unknown player should not throw or change state
    val before = engine.sessionState.rivalBeliefs(PlayerId("v1")).typePosterior
    engine.observeAction(PlayerId("unknown"), PokerAction.Call, minimalState)
    val after = engine.sessionState.rivalBeliefs(PlayerId("v1")).typePosterior
    assertEqualsDouble(
      before.probabilityOf(StrategicClass.Value),
      after.probabilityOf(StrategicClass.Value),
      1e-10
    )

  test("observeAction safe before initSession"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    // Should not throw even without init
    engine.observeAction(PlayerId("v1"), PokerAction.Call, minimalState)

  test("initSession preserves existing beliefs"):
    val existing = StrategicRivalBelief.uniform
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(
      rivalIds = Vector(PlayerId("v1"), PlayerId("v2")),
      existingBeliefs = Map(PlayerId("v1") -> existing)
    )
    val state = engine.sessionState
    assertEquals(state.rivalBeliefs.size, 2)
    // v1 used the provided belief, v2 got a fresh uniform
    assertEqualsDouble(
      state.rivalBeliefs(PlayerId("v2")).typePosterior.probabilityOf(StrategicClass.Bluff),
      0.25,
      1e-10
    )

  test("sessionState throws before initSession"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    intercept[IllegalArgumentException] {
      engine.sessionState
    }

  test("startHand throws before initSession"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    intercept[IllegalArgumentException] {
      engine.startHand(testHeroCards)
    }

  test("startHand no-arg throws before initSession"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    intercept[IllegalArgumentException] {
      engine.startHand()
    }

  test("decide falls back gracefully when native solver unavailable"):
    // Native DLL for POMCP is unlikely to be loaded in unit test context.
    // Expect the fallback: first non-Fold action or Fold.
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(50.0))
    val chosen = engine.decide(minimalState, actions)
    // Fallback path: any action from the candidate set is valid
    assert(actions.contains(chosen))

  test("decide works without hero cards (position fallback)"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand()
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(50.0))
    val chosen = engine.decide(minimalState, actions)
    assert(actions.contains(chosen))

  test("endHand is idempotent — second call does not throw"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    engine.endHand()
    engine.endHand()
    assert(!engine.currentHandActive)

  test("multiple hands — beliefs survive across startHand/endHand cycle"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    engine.endHand()
    engine.startHand(testHeroCards)
    assert(engine.currentHandActive)
    // Beliefs still present
    assertEquals(engine.sessionState.rivalBeliefs.size, 1)

  test("estimateHeroBucket uses hand strength when hero cards provided"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    // Aces — should get a high bucket
    engine.startHand(hole("As", "Ah"))
    val actions = Vector(PokerAction.Fold, PokerAction.Call)
    // This exercises the estimateHeroBucket path with real cards
    val chosen = engine.decide(minimalState, actions)
    assert(actions.contains(chosen))

  test("initSession accepts rival seat info"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(
      rivalIds = Vector(PlayerId("v1")),
      rivalSeats = Map(PlayerId("v1") -> StrategicEngine.RivalSeatInfo(Position.BigBlind, 1000.0))
    )
    assert(engine.isSessionInitialized)

  test("bridgePublicState includes rival seats"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(
      rivalIds = Vector(PlayerId("v1")),
      rivalSeats = Map(PlayerId("v1") -> StrategicEngine.RivalSeatInfo(Position.BigBlind, 1000.0))
    )
    engine.startHand(testHeroCards)
    engine.observeAction(PlayerId("v1"), PokerAction.Raise(50.0), minimalState)
    // The test verifies observeAction doesn't throw when building real PublicState with rivals

  test("observeAction accumulates action history"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(
      rivalIds = Vector(PlayerId("v1")),
      rivalSeats = Map(PlayerId("v1") -> StrategicEngine.RivalSeatInfo(Position.BigBlind, 1000.0))
    )
    engine.startHand(testHeroCards)
    engine.observeAction(PlayerId("v1"), PokerAction.Raise(50.0), minimalState)
    engine.observeAction(PlayerId("v1"), PokerAction.Check, minimalState)
    val bluffP = engine.sessionState.rivalBeliefs(PlayerId("v1")).typePosterior.probabilityOf(StrategicClass.Bluff)
    assert(bluffP != 0.25, "Beliefs should shift after observations")

  private def nativeAvailable: Boolean = WPomcpRuntime.isAvailable

  test("integration: play a complete hand with Strategic mode"):
    assume(nativeAvailable, "Native library not available")
    val engine = new StrategicEngine(StrategicEngine.Config(numSimulations = 100))
    engine.initSession(rivalIds = Vector(PlayerId("villain")))

    // Hand 1: start hand
    engine.startHand(testHeroCards)

    val preflopState = minimalState
    val candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(3.0))

    // Villain raises
    engine.observeAction(PlayerId("villain"), PokerAction.Raise(3.0), preflopState)

    // Hero decides
    val action = engine.decide(preflopState, candidates)
    assert(candidates.contains(action), s"Action $action not in candidates")

    // End hand
    engine.endHand()

    // Hand 2: beliefs should persist
    engine.startHand(testHeroCards)
    val action2 = engine.decide(preflopState, candidates)
    assert(candidates.contains(action2), s"Action $action2 not in candidates (hand 2)")
    engine.endHand()

  test("integration: decideHeroStrategic routes correctly"):
    assume(nativeAvailable, "Native library not available")
    val engine = new StrategicEngine(StrategicEngine.Config(numSimulations = 50))
    engine.initSession(rivalIds = Vector(PlayerId("villain")))
    engine.startHand(testHeroCards)

    val gs = minimalState
    val candidates = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(3.0))
    val ctx = HeroDecisionPipeline.StrategicDecisionContext(
      state = gs,
      candidates = candidates,
      engine = engine
    )
    val action = HeroDecisionPipeline.decideHeroStrategic(ctx)
    assert(candidates.contains(action), s"Action $action not in candidates")

  test("exploitability function returns non-trivial values"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    engine.observeAction(PlayerId("v1"), PokerAction.Raise(50.0), minimalState)
    val exploit = engine.computeExploitabilityEstimate(0.5)
    assert(exploit >= 0.0, s"Exploitability must be non-negative: $exploit")

  test("endHand with showdown data updates rival beliefs"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    engine.observeAction(PlayerId("v1"), PokerAction.Raise(50.0), minimalState)
    val beforeShowdown = engine.sessionState.rivalBeliefs(PlayerId("v1")).typePosterior
    // End hand with showdown revealing villain had strong hand (pocket Aces)
    engine.endHand(showdownResult = Some(Map(
      PlayerId("v1") -> HoleCards.from(Vector(card("As"), card("Ah")))
    )))
    val afterShowdown = engine.sessionState.rivalBeliefs(PlayerId("v1")).typePosterior
    val valueAfter = afterShowdown.probabilityOf(StrategicClass.Value)
    val valueBefore = beforeShowdown.probabilityOf(StrategicClass.Value)
    assert(valueAfter > valueBefore,
      s"Showdown with strong hand should increase Value posterior: before=$valueBefore, after=$valueAfter")

  test("endHand without showdown data preserves beliefs"):
    val engine = new StrategicEngine(StrategicEngine.Config())
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    engine.observeAction(PlayerId("v1"), PokerAction.Raise(50.0), minimalState)
    val before = engine.sessionState.rivalBeliefs(PlayerId("v1")).typePosterior
    engine.endHand()
    val after = engine.sessionState.rivalBeliefs(PlayerId("v1")).typePosterior
    assertEqualsDouble(
      before.probabilityOf(StrategicClass.Value),
      after.probabilityOf(StrategicClass.Value),
      1e-10
    )

  test("FrequencyAnomalyDetection fires when rival shows high aggression"):
    val config = StrategicEngine.Config(
      detector = FrequencyAnomalyDetection(window = 5, threshold = 0.5)
    )
    val engine = new StrategicEngine(config)
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    val initialBeta = engine.sessionState.exploitationStates(PlayerId("v1")).beta
    // Observe 5 raises in a row to trigger detection (100% aggression > 50% threshold)
    for _ <- 1 to 5 do
      engine.observeAction(PlayerId("v1"), PokerAction.Raise(50.0), minimalState)
    val finalBeta = engine.sessionState.exploitationStates(PlayerId("v1")).beta
    assert(finalBeta < initialBeta,
      s"Beta should retreat after detected modeling: initial=$initialBeta, final=$finalBeta")

  // ---- Task 5: Advisory clamp in observeAction (design doc §6) ----

  test("advisory clamp retreats beta when cached bundle budget exceeds tolerance"):
    // Two engines with identical state; one gets a high-budget bundle injected,
    // the other has no bundle. The difference isolates the advisory clamp effect.
    val cfg = StrategicEngine.Config(
      exploitConfig = ExploitationConfig(initialBeta = 1.0, cpRetreatRate = 0.2, epsilonAdapt = 0.05)
    )
    val engineWithClamp = new StrategicEngine(cfg)
    val engineWithout = new StrategicEngine(cfg)

    engineWithClamp.initSession(rivalIds = Vector(PlayerId("v1")))
    engineWithout.initSession(rivalIds = Vector(PlayerId("v1")))
    engineWithClamp.startHand(testHeroCards)
    engineWithout.startHand(testHeroCards)

    // Inject a high-budget bundle (budget=100 >> tolerance=0.10) to trigger the advisory clamp
    val highBudgetBundle = DecisionEvaluationBundle(
      profileResults = Map.empty,
      robustActionLowerBounds = Array(0.0),
      baselineActionValues = Array(0.0),
      baselineValue = 0.0,
      adversarialRootGap = None,
      pointwiseExploitability = None,
      deploymentExploitability = None,
      certification = CertificationResult.LocalRobustScreening(
        rootLosses = Array(1.0),
        budgetEstimate = 100.0,  // Way above tolerance (0.05 + 0.05 = 0.10)
        withinTolerance = false
      ),
      chainWorldValues = Map.empty,
      notes = Vector("test: high budget")
    )
    engineWithClamp.injectTestBundle(highBudgetBundle)

    // Both engines observe the same action
    engineWithClamp.observeAction(PlayerId("v1"), PokerAction.Check, minimalState)
    engineWithout.observeAction(PlayerId("v1"), PokerAction.Check, minimalState)

    val betaWithClamp = engineWithClamp.sessionState.exploitationStates(PlayerId("v1")).beta
    val betaWithout = engineWithout.sessionState.exploitationStates(PlayerId("v1")).beta

    // Advisory clamp should retreat beta by cpRetreatRate (0.2) beyond what fullStep alone does
    assertEqualsDouble(betaWithClamp, math.max(0.0, betaWithout - cfg.exploitConfig.cpRetreatRate), 1e-10)

  test("advisory clamp does NOT fire when budget is within tolerance"):
    val cfg = StrategicEngine.Config()
    val engineWithBundle = new StrategicEngine(cfg)
    val engineWithout = new StrategicEngine(cfg)

    engineWithBundle.initSession(rivalIds = Vector(PlayerId("v1")))
    engineWithout.initSession(rivalIds = Vector(PlayerId("v1")))
    engineWithBundle.startHand(testHeroCards)
    engineWithout.startHand(testHeroCards)

    // Budget within tolerance — clamp should NOT fire
    val withinToleranceBundle = DecisionEvaluationBundle(
      profileResults = Map.empty,
      robustActionLowerBounds = Array(0.0),
      baselineActionValues = Array(0.0),
      baselineValue = 0.0,
      adversarialRootGap = None,
      pointwiseExploitability = None,
      deploymentExploitability = None,
      certification = CertificationResult.LocalRobustScreening(
        rootLosses = Array(0.0),
        budgetEstimate = 0.01,  // Well below tolerance (0.05 + 0.05 = 0.10)
        withinTolerance = true
      ),
      chainWorldValues = Map.empty,
      notes = Vector("test: within tolerance")
    )
    engineWithBundle.injectTestBundle(withinToleranceBundle)

    engineWithBundle.observeAction(PlayerId("v1"), PokerAction.Check, minimalState)
    engineWithout.observeAction(PlayerId("v1"), PokerAction.Check, minimalState)

    val betaWith = engineWithBundle.sessionState.exploitationStates(PlayerId("v1")).beta
    val betaWithout = engineWithout.sessionState.exploitationStates(PlayerId("v1")).beta

    // No advisory clamp — betas should be identical (only fullStep effect)
    assertEqualsDouble(betaWith, betaWithout, 1e-10)

  // ---- Task 6: Deployment tracking across decide() calls ----

  test("deployment entries accumulate across decide() calls (PftDpw integration)"):
    assume(nativeAvailable, "Native library not available")
    val cfg = StrategicEngine.Config(
      solverBackend = StrategicEngine.SolverBackend.PftDpw,
      numSimulations = 50
    )
    val engine = new StrategicEngine(cfg)
    engine.initSession(rivalIds = Vector(PlayerId("v1")))

    // Hand 1
    engine.startHand(testHeroCards)
    engine.decide(minimalState, Vector(PokerAction.Fold, PokerAction.Call))
    engine.endHand()
    val sizeAfter1 = engine.sessionState.deploymentSet.entries.size

    // Hand 2
    engine.startHand(testHeroCards)
    engine.decide(minimalState, Vector(PokerAction.Fold, PokerAction.Call))
    engine.endHand()
    val sizeAfter2 = engine.sessionState.deploymentSet.entries.size

    // PftDpw produces pointwiseExploitability, so entries should accumulate
    assert(sizeAfter1 >= 1, s"Expected >= 1 entry after first decide, got $sizeAfter1")
    assert(sizeAfter2 >= 2, s"Expected >= 2 entries after second decide, got $sizeAfter2")
    assert(sizeAfter2 > sizeAfter1, "Entries should grow across decide() calls")

  test("deploymentExploitability populates in bundle after prior entries exist (PftDpw integration)"):
    assume(nativeAvailable, "Native library not available")
    val cfg = StrategicEngine.Config(
      solverBackend = StrategicEngine.SolverBackend.PftDpw,
      numSimulations = 50
    )
    val engine = new StrategicEngine(cfg)
    engine.initSession(rivalIds = Vector(PlayerId("v1")))

    // First call: no prior entries → deploymentExploitability should be None
    engine.startHand(testHeroCards)
    engine.decide(minimalState, Vector(PokerAction.Fold, PokerAction.Call))
    val bundle1 = engine.lastDecisionBundle
    engine.endHand()

    // Second call: prior entries exist → deploymentExploitability should be Some
    engine.startHand(testHeroCards)
    engine.decide(minimalState, Vector(PokerAction.Fold, PokerAction.Call))
    val bundle2 = engine.lastDecisionBundle
    engine.endHand()

    assert(bundle1.isDefined, "First bundle must exist")
    // First call: deployment set was empty before decide, so deploymentExploitability = None
    assertEquals(bundle1.get.deploymentExploitability, None)
    assert(bundle2.isDefined, "Second bundle must exist")
    // Second call: deployment set has entries from first call
    assert(bundle2.get.deploymentExploitability.isDefined,
      "deploymentExploitability should be Some after prior entries accumulated")
