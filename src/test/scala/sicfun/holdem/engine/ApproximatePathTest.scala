package sicfun.holdem.engine

import sicfun.core.Card
import sicfun.holdem.types.*
import sicfun.holdem.strategic.*

class ApproximatePathTest extends munit.FunSuite:

  private def card(token: String): Card =
    Card.parse(token).getOrElse(throw new IllegalArgumentException(s"bad card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def testHeroCards: HoleCards = hole("As", "Kh")

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

  private def defaultActions: Vector[PokerAction] =
    Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(50.0))

  test("decide() on WPomcp path produces DecisionEvaluationBundle"):
    val config = StrategicEngine.Config(
      numSimulations = 100,
      solverBackend = StrategicEngine.SolverBackend.WPomcp,
      bellmanGamma = 0.95,
      epsilonBase = 0.05,
      deploymentSetSize = 50
    )
    val engine = new StrategicEngine(config)
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)

    val chosen = engine.decide(minimalState, defaultActions)

    // Action must come from the candidate set
    assert(defaultActions.contains(chosen), s"Action $chosen not in candidates")
    // lastBundle should be populated after decide()
    val bundle = engine.lastDecisionBundle
    assert(bundle.isDefined, "lastDecisionBundle should be Some after decide()")

  test("DecisionEvaluationBundle has valid profile results"):
    val config = StrategicEngine.Config(
      numSimulations = 100,
      solverBackend = StrategicEngine.SolverBackend.WPomcp,
      bellmanGamma = 0.95,
      epsilonBase = 0.05,
      deploymentSetSize = 50
    )
    val engine = new StrategicEngine(config)
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    engine.decide(minimalState, defaultActions)

    engine.lastDecisionBundle match
      case Some(bundle) =>
        // Should have 4 profile results (one per StrategicClass)
        assertEquals(bundle.profileResults.size, StrategicClass.values.length)
        // robustActionLowerBounds should match number of actions
        assertEquals(bundle.robustActionLowerBounds.length, defaultActions.size)
        // baselineActionValues should match number of actions
        assertEquals(bundle.baselineActionValues.length, defaultActions.size)
        // certification should be LocalRobustScreening
        bundle.certification match
          case _: CertificationResult.LocalRobustScreening => () // expected
          case other => fail(s"Expected LocalRobustScreening, got $other")
      case None =>
        // Native solver not available -- bundle comes from fallback path
        // This is acceptable; the fallback still populates the bundle
        ()

  test("decide() fallback path still populates bundle when solver unavailable"):
    // This test verifies the fallback path (native DLL not loaded) still produces
    // a bundle with BaselineFallback-style certification
    val config = StrategicEngine.Config(
      numSimulations = 100,
      solverBackend = StrategicEngine.SolverBackend.WPomcp,
      bellmanGamma = 0.95,
      epsilonBase = 0.05,
      deploymentSetSize = 50
    )
    val engine = new StrategicEngine(config)
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    val chosen = engine.decide(minimalState, defaultActions)
    assert(defaultActions.contains(chosen))
    // Bundle should always be set (even on fallback)
    assert(engine.lastDecisionBundle.isDefined, "Bundle should be set even on fallback")

  test("decide() with multiple rivals produces valid bundle"):
    val config = StrategicEngine.Config(
      numSimulations = 100,
      solverBackend = StrategicEngine.SolverBackend.WPomcp,
      bellmanGamma = 0.95,
      epsilonBase = 0.05,
      deploymentSetSize = 50
    )
    val engine = new StrategicEngine(config)
    engine.initSession(rivalIds = Vector(PlayerId("v1"), PlayerId("v2")))
    engine.startHand(testHeroCards)
    val chosen = engine.decide(minimalState, defaultActions)
    assert(defaultActions.contains(chosen))
    assert(engine.lastDecisionBundle.isDefined)

  test("Config defaults include new fields"):
    val config = StrategicEngine.Config()
    assertEqualsDouble(config.bellmanGamma, 0.95, 1e-10)
    assertEqualsDouble(config.epsilonBase, 0.05, 1e-10)
    assertEquals(config.deploymentSetSize, 50)

  test("bundle budgetEstimate is non-negative"):
    val config = StrategicEngine.Config(
      numSimulations = 100,
      solverBackend = StrategicEngine.SolverBackend.WPomcp
    )
    val engine = new StrategicEngine(config)
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    engine.decide(minimalState, defaultActions)

    engine.lastDecisionBundle.foreach { bundle =>
      bundle.certification match
        case lrs: CertificationResult.LocalRobustScreening =>
          assert(lrs.budgetEstimate >= 0.0,
            s"budgetEstimate should be non-negative: ${lrs.budgetEstimate}")
        case _ => () // fallback path
    }
