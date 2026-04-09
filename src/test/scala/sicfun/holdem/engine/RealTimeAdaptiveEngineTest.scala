package sicfun.holdem.engine
import sicfun.holdem.types.*
import sicfun.holdem.model.*
import sicfun.holdem.equity.*

import munit.FunSuite
import sicfun.core.{Card, DiscreteDistribution}

import java.util.concurrent.CountDownLatch
import scala.util.Random

/** Tests for [[RealTimeAdaptiveEngine]].
  *
  * Validates the real-time adaptive decision engine:
  *   - '''Archetype learning''': repeated folds -> Nit MAP; repeated raises -> Maniac MAP.
  *   - '''Response-aware EV''': fold-trained engine produces higher raise EV than
  *     raise-trained engine (villain expected to fold more often).
  *   - '''Inference caching''': second call with identical context is a cache hit.
  *   - '''Latency budget''': 0ms budget clamps equity trials to minimum.
  *   - '''Equilibrium baseline''': CFR solution is attached when enabled; blend weight
  *     is applied correctly.
  *   - '''Baseline guardrail''': adaptive action overridden when action regret exceeds threshold.
  *   - '''Trust gate''': baseline disabled when local exploitability exceeds threshold.
  *   - '''Concurrency''': archetype posterior stays normalized under concurrent updates.
  *   - '''Revealed cards''': delta posterior with single-hand support.
  */
class RealTimeAdaptiveEngineTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private val table = TableRanges.defaults(TableFormat.NineMax)
  private val model = PokerActionModel.uniform

  test("archetype posterior shifts toward Nit after repeated folds to raises") {
    val engine = new RealTimeAdaptiveEngine(
      tableRanges = table,
      actionModel = model,
      bunchingTrials = 150,
      defaultEquityTrials = 3000,
      minEquityTrials = 300
    )

    var i = 0
    while i < 30 do
      engine.observeVillainResponseToRaise(PokerAction.Fold)
      i += 1

    val posterior = engine.archetypePosterior
    assertEquals(posterior.mapEstimate, PlayerArchetype.Nit)
    assert(
      posterior.probabilityOf(PlayerArchetype.Nit) >
        posterior.probabilityOf(PlayerArchetype.Maniac)
    )
  }

  test("archetype posterior shifts toward Maniac after repeated 3-bets") {
    val engine = new RealTimeAdaptiveEngine(
      tableRanges = table,
      actionModel = model,
      bunchingTrials = 150,
      defaultEquityTrials = 3000,
      minEquityTrials = 300
    )

    var i = 0
    while i < 30 do
      engine.observeVillainResponseToRaise(PokerAction.Raise(7.5))
      i += 1

    val posterior = engine.archetypePosterior
    assertEquals(posterior.mapEstimate, PlayerArchetype.Maniac)
    assert(
      posterior.probabilityOf(PlayerArchetype.Maniac) >
        posterior.probabilityOf(PlayerArchetype.Nit)
    )
  }

  test("adaptive response learning changes raise EV for same posterior spot") {
    val hero = hole("9c", "8d")
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 1.5,
      toCall = 0.5,
      position = Position.SmallBlind,
      stackSize = 99.5,
      betHistory = Vector.empty
    )
    val posterior = DiscreteDistribution(Map(hole("Ks", "Qh") -> 0.5, hole("Ts", "9h") -> 0.5))
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(2.5))

    val foldTrained = new RealTimeAdaptiveEngine(
      tableRanges = table,
      actionModel = model,
      bunchingTrials = 100,
      defaultEquityTrials = 6_000,
      minEquityTrials = 500
    )
    val raiseTrained = new RealTimeAdaptiveEngine(
      tableRanges = table,
      actionModel = model,
      bunchingTrials = 100,
      defaultEquityTrials = 6_000,
      minEquityTrials = 500
    )

    var i = 0
    while i < 24 do
      foldTrained.observeVillainResponseToRaise(PokerAction.Fold)
      raiseTrained.observeVillainResponseToRaise(PokerAction.Raise(8.0))
      i += 1

    val foldResult = foldTrained.recommendAgainstPosterior(
      hero = hero,
      state = state,
      posterior = posterior,
      candidateActions = actions,
      rng = new Random(71)
    )
    val raiseResult = raiseTrained.recommendAgainstPosterior(
      hero = hero,
      state = state,
      posterior = posterior,
      candidateActions = actions,
      rng = new Random(71)
    )

    val foldRaiseEv = foldResult.recommendation.actionEvaluations
      .find(_.action == PokerAction.Raise(2.5))
      .map(_.expectedValue)
      .getOrElse(fail("missing raise EV for fold-trained engine"))

    val raiseRaiseEv = raiseResult.recommendation.actionEvaluations
      .find(_.action == PokerAction.Raise(2.5))
      .map(_.expectedValue)
      .getOrElse(fail("missing raise EV for raise-trained engine"))

    assert(foldRaiseEv > raiseRaiseEv)
  }

  test("decide caches posterior inference for identical context") {
    val hero = hole("Ac", "Kh")
    val board = Board.from(Seq(card("Ts"), card("9h"), card("8d")))
    val state = GameState(
      street = Street.Flop,
      board = board,
      pot = 24.0,
      toCall = 8.0,
      position = Position.Button,
      stackSize = 150.0,
      betHistory = Vector.empty
    )
    val folds = TableFormat.NineMax.foldsBeforeOpener(Position.Cutoff).map(PreflopFold(_))

    val engine = new RealTimeAdaptiveEngine(
      tableRanges = table,
      actionModel = model,
      bunchingTrials = 300,
      defaultEquityTrials = 1500,
      minEquityTrials = 400
    )

    val first = engine.decide(
      hero = hero,
      state = state,
      folds = folds,
      villainPos = Position.BigBlind,
      observations = Seq.empty,
      candidateActions = Vector(PokerAction.Fold, PokerAction.Call),
      rng = new Random(1001)
    )
    assertEquals(first.cacheStats.inferenceMisses, 1L)
    assertEquals(first.cacheStats.inferenceHits, 0L)

    val second = engine.decide(
      hero = hero,
      state = state,
      folds = folds,
      villainPos = Position.BigBlind,
      observations = Seq.empty,
      candidateActions = Vector(PokerAction.Fold, PokerAction.Call),
      rng = new Random(1002)
    )
    assertEquals(second.cacheStats.inferenceMisses, 1L)
    assertEquals(second.cacheStats.inferenceHits, 1L)
  }

  test("latency budget lower bound clamps equity trials") {
    val engine = new RealTimeAdaptiveEngine(
      tableRanges = table,
      actionModel = model,
      bunchingTrials = 150,
      defaultEquityTrials = 3200,
      minEquityTrials = 250
    )

    val hero = hole("As", "Kd")
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 6.0,
      toCall = 2.0,
      position = Position.Button,
      stackSize = 100.0,
      betHistory = Vector.empty
    )
    val posterior = DiscreteDistribution(Map(hole("7c", "2d") -> 1.0))

    val result = engine.recommendAgainstPosterior(
      hero = hero,
      state = state,
      posterior = posterior,
      candidateActions = Vector(PokerAction.Fold, PokerAction.Call),
      decisionBudgetMillis = Some(0L),
      rng = new Random(77)
    )

    assertEquals(result.equityTrialsUsed, 250)
  }

  test("equilibrium baseline attaches CFR solution when enabled") {
    val engine = new RealTimeAdaptiveEngine(
      tableRanges = table,
      actionModel = model,
      bunchingTrials = 100,
      defaultEquityTrials = 1_200,
      minEquityTrials = 300,
      equilibriumBaselineConfig = Some(
        EquilibriumBaselineConfig(
          iterations = 600,
          blendWeight = 0.4,
          maxVillainHands = 24,
          equityTrials = 800,
          includeVillainReraises = true
        )
      )
    )

    val hero = hole("Ac", "Kh")
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 6.0,
      toCall = 2.0,
      position = Position.Button,
      stackSize = 100.0,
      betHistory = Vector.empty
    )
    val posterior = DiscreteDistribution(
      Map(
        hole("7c", "2d") -> 0.6,
        hole("Qs", "Qh") -> 0.4
      )
    )

    val result = engine.recommendAgainstPosterior(
      hero = hero,
      state = state,
      posterior = posterior,
      candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(8.0)),
      rng = new Random(99)
    )

    val baseline = result.equilibriumBaseline.getOrElse(fail("expected CFR baseline to be present"))
    assert(baseline.actionProbabilities.nonEmpty)
    assert(baseline.actionEvaluations.nonEmpty)
    assertEquals(result.adaptationTrace.source, AdaptationDecisionSource.BlendedWithBaseline)
    assertEquals(result.adaptationTrace.effectiveBlendWeight, 0.4)
    assert(result.adaptationTrace.baselineLocalExploitability.nonEmpty)
  }

  test("equilibrium guardrail can clamp adaptive action back to CFR baseline") {
    val hero = hole("9c", "8d")
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 1.5,
      toCall = 0.5,
      position = Position.SmallBlind,
      stackSize = 99.5,
      betHistory = Vector.empty
    )
    val posterior = DiscreteDistribution(Map(hole("Ks", "Qh") -> 0.5, hole("Ts", "9h") -> 0.5))
    val actions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(2.5))

    val adaptiveOnly = new RealTimeAdaptiveEngine(
      tableRanges = table,
      actionModel = model,
      bunchingTrials = 100,
      defaultEquityTrials = 6_000,
      minEquityTrials = 500
    )
    val guardrailed = new RealTimeAdaptiveEngine(
      tableRanges = table,
      actionModel = model,
      bunchingTrials = 100,
      defaultEquityTrials = 6_000,
      minEquityTrials = 500,
      equilibriumBaselineConfig = Some(
        EquilibriumBaselineConfig(
          iterations = 600,
          blendWeight = 0.0,
          maxBaselineActionRegret = 0.0,
          maxVillainHands = 24,
          equityTrials = 800,
          includeVillainReraises = true
        )
      )
    )

    var i = 0
    while i < 24 do
      adaptiveOnly.observeVillainResponseToRaise(PokerAction.Fold)
      guardrailed.observeVillainResponseToRaise(PokerAction.Fold)
      i += 1

    val adaptiveOnlyResult = adaptiveOnly.recommendAgainstPosterior(
      hero = hero,
      state = state,
      posterior = posterior,
      candidateActions = actions,
      rng = new Random(71)
    )
    val guardrailedResult = guardrailed.recommendAgainstPosterior(
      hero = hero,
      state = state,
      posterior = posterior,
      candidateActions = actions,
      rng = new Random(71)
    )

    val baseline = guardrailedResult.equilibriumBaseline.getOrElse(fail("expected CFR baseline to be present"))
    assertNotEquals(adaptiveOnlyResult.recommendation.bestAction, baseline.bestAction)
    assertEquals(guardrailedResult.recommendation.bestAction, baseline.bestAction)
    assertEquals(guardrailedResult.adaptationTrace.source, AdaptationDecisionSource.BaselineGuardrail)
    assertEquals(guardrailedResult.adaptationTrace.reason, Some("baseline_action_regret_exceeds_threshold"))
  }

  test("equilibrium baseline trust gate can disable baseline blending when solve quality is below threshold") {
    val engine = new RealTimeAdaptiveEngine(
      tableRanges = table,
      actionModel = model,
      bunchingTrials = 100,
      defaultEquityTrials = 1_200,
      minEquityTrials = 300,
      equilibriumBaselineConfig = Some(
        EquilibriumBaselineConfig(
          iterations = 600,
          blendWeight = 0.4,
          maxLocalExploitabilityForTrust = 0.0,
          maxVillainHands = 24,
          equityTrials = 800,
          includeVillainReraises = true
        )
      )
    )

    val hero = hole("Ac", "Kh")
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 6.0,
      toCall = 2.0,
      position = Position.Button,
      stackSize = 100.0,
      betHistory = Vector.empty
    )
    val posterior = DiscreteDistribution(
      Map(
        hole("7c", "2d") -> 0.6,
        hole("Qs", "Qh") -> 0.4
      )
    )

    val result = engine.recommendAgainstPosterior(
      hero = hero,
      state = state,
      posterior = posterior,
      candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(8.0)),
      rng = new Random(99)
    )

    val baseline = result.equilibriumBaseline.getOrElse(fail("expected CFR baseline to be present"))
    assert(baseline.localExploitability > 0.0)
    assertEquals(result.adaptationTrace.source, AdaptationDecisionSource.AdaptiveOnly)
    assertEquals(result.adaptationTrace.effectiveBlendWeight, 0.0)
    assertEquals(result.adaptationTrace.reason, Some("baseline_local_exploitability_exceeds_trust_threshold"))
  }

  test("adaptive engine keeps archetype posterior normalized under concurrent updates") {
    val engine = new RealTimeAdaptiveEngine(
      tableRanges = table,
      actionModel = model,
      bunchingTrials = 1,
      defaultEquityTrials = 120,
      minEquityTrials = 40
    )

    val responseCycle = Array(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(8.0))

    val done = new CountDownLatch(4)
    val threads =
      Vector.tabulate(4) { workerId =>
        Thread(() =>
          try
            var i = 0
            while i < 100 do
              engine.observeVillainResponseToRaise(responseCycle((workerId + i) % responseCycle.length))
              engine.archetypePosterior
              i += 1
          finally
            done.countDown()
        )
      }
    try
      threads.foreach(_.start())
      threads.foreach(_.join(10_000L))
      assertEquals(done.getCount, 0L)
      val archetypePosterior = engine.archetypePosterior
      val total = PlayerArchetype.values.map(archetypePosterior.probabilityOf).sum
      assert(math.abs(total - 1.0) < 1e-9, s"posterior must stay normalized, got $total")
    finally
      threads.foreach(_.interrupt())
  }

  test("decide with revealedCards produces delta posterior") {
    val hero = hole("Ac", "Kh")
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 6.0,
      toCall = 2.0,
      position = Position.Button,
      stackSize = 194.0,
      betHistory = Vector.empty
    )
    val revealed = hole("Qh", "Qs")
    val engine = new RealTimeAdaptiveEngine(
      tableRanges = TableRanges.defaults(TableFormat.HeadsUp),
      actionModel = model,
      bunchingTrials = 150,
      defaultEquityTrials = 1500,
      minEquityTrials = 300
    )
    val result = engine.decide(
      hero = hero,
      state = state,
      folds = Vector.empty,
      villainPos = Position.BigBlind,
      observations = Seq.empty,
      candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(6.0)),
      revealedCards = Some(revealed),
      rng = new Random(42)
    )
    val prob = result.decision.posteriorInference.posterior.probabilityOf(revealed)
    assert(prob > 0.99, s"expected ~1.0 for revealed hand, got $prob")
    assertEquals(result.decision.posteriorInference.posterior.support.size, 1)
  }
