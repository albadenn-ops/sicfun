package sicfun.holdem.engine

import sicfun.core.Card
import sicfun.holdem.types.*
import sicfun.holdem.strategic.*

class FormalPathTest extends munit.FunSuite:

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

  test("decide() on PftDpw path returns valid action (fail-closed when native unavailable)"):
    val config = StrategicEngine.Config(
      numSimulations = 100,
      solverBackend = StrategicEngine.SolverBackend.PftDpw,
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
    // Bundle should always be set (even on fallback)
    assert(engine.lastDecisionBundle.isDefined, "Bundle should be set even on PftDpw fallback")

  test("PftDpw fallback bundle has Unavailable certification when native not loaded"):
    val config = StrategicEngine.Config(
      numSimulations = 100,
      solverBackend = StrategicEngine.SolverBackend.PftDpw,
      bellmanGamma = 0.95,
      epsilonBase = 0.05
    )
    val engine = new StrategicEngine(config)
    engine.initSession(rivalIds = Vector(PlayerId("v1")))
    engine.startHand(testHeroCards)
    engine.decide(minimalState, defaultActions)

    engine.lastDecisionBundle match
      case Some(bundle) =>
        bundle.certification match
          case _: CertificationResult.TabularCertification =>
            // Native solver was available — unexpected in CI, but valid
            ()
          case _: CertificationResult.Unavailable =>
            // Expected path: native solver not loaded
            ()
          case other =>
            fail(s"Expected TabularCertification or Unavailable, got $other")
      case None =>
        fail("Bundle should be set after PftDpw decide()")

  test("PftDpw path does not crash with multiple rivals"):
    val config = StrategicEngine.Config(
      numSimulations = 100,
      solverBackend = StrategicEngine.SolverBackend.PftDpw
    )
    val engine = new StrategicEngine(config)
    engine.initSession(rivalIds = Vector(PlayerId("v1"), PlayerId("v2")))
    engine.startHand(testHeroCards)

    val chosen = engine.decide(minimalState, defaultActions)
    assert(defaultActions.contains(chosen))
    assert(engine.lastDecisionBundle.isDefined)
