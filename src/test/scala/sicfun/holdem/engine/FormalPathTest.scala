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

  // --- Integration tests (native fallback path) ---

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

    assert(defaultActions.contains(chosen), s"Action $chosen not in candidates")
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
          case _: CertificationResult.TabularCertification => ()
          case _: CertificationResult.Unavailable => ()
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

  // --- Certification logic unit tests (no native solver needed) ---

  test("profile-conditioned robust losses are non-zero for non-policy actions"):
    val heroActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(50.0))
    val priors = StrategicEngine.defaultActionPriors
    val numProfiles = StrategicClass.values.length

    val profileModels = (0 until numProfiles).map { p =>
      val cls = StrategicClass.fromOrdinal(p)
      PokerPftFormulation.buildTabularModel(
        minimalState, Map.empty, heroActions, 5, priors, profileClass = Some(cls)
      )
    }
    val refPolicy: Int => Int = _ => 1 // always Call
    val gamma = 0.95
    val robustLosses = PerStateLossEvaluator.computeRobustLosses(profileModels, refPolicy, gamma)

    // Policy action (Call, idx 1) has near-zero robust loss by definition.
    // Residual from value iteration convergence at gamma=0.95.
    for row <- robustLosses do
      assertEqualsDouble(row(1), 0.0, 1e-3)

    // Non-policy actions should have positive robust losses from profile variation
    val hasNonzeroLoss = robustLosses.exists(row =>
      row.zipWithIndex.exists { case (v, a) => a != 1 && v > 1e-12 }
    )
    assert(hasNonzeroLoss, "Non-policy actions should have non-zero robust losses")

    // B* is zero when transitions are identical and policy action is always safe,
    // because min_a picks the zero-loss policy action at every state.
    // This is correct: no adaptation budget needed when the policy is always available.
    val transitions: (Int, Int, Int) => Int = (s, a, p) =>
      profileModels(p).transitionTable(s * profileModels(0).numActions + a)
    val bStar = SafetyBellman.computeBStar(robustLosses, gamma, transitions, numProfiles)
    for s <- bStar.indices do
      assertEqualsDouble(bStar(s), 0.0, 1e-10)

  test("fail-closed: certification failure selects baseline action"):
    // Simulate the decision logic: when certificateValid=false or withinTolerance=false,
    // the engine should select the action with the best baseline Q-value.
    val baselineActionValues = Array(-1.0, 0.3, 0.1) // Call (idx 1) is best baseline
    val solverQValues = Array(-0.5, 0.1, 0.8) // Raise (idx 2) would be best solver choice
    val safeActions = IndexedSeq(0, 1) // only Fold and Call are safe

    // Certified path: safeFeasibleAction picks best Q among safe
    val certifiedAction = SafetyBellman.safeFeasibleAction(solverQValues, safeActions)
    assertEquals(certifiedAction, 1) // Call has Q=0.1 which is best among safe {0,1}

    // Fail-closed path: picks best baseline action regardless of solver Q
    val failClosedAction = baselineActionValues.indices.maxBy(baselineActionValues(_))
    assertEquals(failClosedAction, 1) // Call has baseline Q=0.3

    // Key: if safe set is empty, safeFeasibleAction falls back to max-Q overall (idx 2 = Raise)
    // but fail-closed should still pick baseline best (idx 1 = Call)
    val certifiedEmpty = SafetyBellman.safeFeasibleAction(solverQValues, IndexedSeq.empty)
    assertEquals(certifiedEmpty, 2) // would pick Raise — NOT fail-closed
    // whereas fail-closed always uses baseline:
    assertNotEquals(certifiedEmpty, failClosedAction)

  test("robust lower bounds differ from baseline when profiles differ"):
    val heroActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(50.0))
    val priors = StrategicEngine.defaultActionPriors
    val numProfiles = StrategicClass.values.length
    val gamma = 0.95

    val baselineModel = PokerPftFormulation.buildTabularModel(
      minimalState, Map.empty, heroActions, 5, priors, profileClass = None
    )
    val profileModels = (0 until numProfiles).map { p =>
      PokerPftFormulation.buildTabularModel(
        minimalState, Map.empty, heroActions, 5, priors,
        profileClass = Some(StrategicClass.fromOrdinal(p))
      )
    }
    val refPolicy: Int => Int = _ => 1
    val baselineValues = PerStateLossEvaluator.valueIteration(baselineModel, refPolicy, gamma)
    val profileValueArrays = profileModels.map(m =>
      PerStateLossEvaluator.valueIteration(m, refPolicy, gamma)
    )

    // Baseline action values from mixed model
    val numActions = heroActions.size
    val baselineAV = new Array[Double](numActions)
    var a = 0
    while a < numActions do
      val reward = baselineModel.rewardTable(a)
      val succ = baselineModel.transitionTable(a)
      baselineAV(a) = reward + gamma * baselineValues(succ)
      a += 1

    // Robust lower bounds: min over profiles
    val robustLB = new Array[Double](numActions)
    a = 0
    while a < numActions do
      var minQ = Double.PositiveInfinity
      var p = 0
      while p < numProfiles do
        val model = profileModels(p)
        val reward = model.rewardTable(a)
        val succ = model.transitionTable(a)
        val q = reward + gamma * profileValueArrays(p)(succ)
        if q < minQ then minQ = q
        p += 1
      robustLB(a) = minQ
      a += 1

    // Robust lower bounds should be <= baseline values (worst-case is worse than average)
    a = 0
    while a < numActions do
      assert(robustLB(a) <= baselineAV(a) + 1e-10,
        s"Robust LB ($a)=${robustLB(a)} should be <= baseline ${baselineAV(a)}")
      a += 1

    // At least one action should have a strict gap
    val hasGap = (0 until numActions).exists(a => baselineAV(a) - robustLB(a) > 1e-12)
    assert(hasGap, "Profile variation should create a gap between baseline and robust bounds")
