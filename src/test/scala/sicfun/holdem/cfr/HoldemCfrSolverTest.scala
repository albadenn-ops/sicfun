package sicfun.holdem.cfr
import sicfun.holdem.types.*
import sicfun.holdem.equity.*

import munit.FunSuite
import sicfun.core.{Card, DiscreteDistribution}

import java.nio.file.Paths

class HoldemCfrSolverTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def withSystemProperties(properties: Map[String, String])(thunk: => Unit): Unit =
    val previous = properties.keys.iterator.map { key =>
      key -> sys.props.get(key)
    }.toVector

    properties.foreach { case (key, value) =>
      System.setProperty(key, value)
    }

    try thunk
    finally
      previous.foreach { case (key, oldValue) =>
        oldValue match
          case Some(value) => System.setProperty(key, value)
          case None => System.clearProperty(key)
      }
      HoldemCfrNativeRuntime.resetLoadCacheForTests()
      HoldemCfrSolver.resetAutoProviderForTests()

  test("CFR baseline returns normalized policy and non-fold best action for premium hero hand") {
    val hero = hole("Ac", "Ad")
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
        hole("Kc", "Qd") -> 0.3,
        hole("Ks", "Kh") -> 0.1
      )
    )
    val candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(8.0))

    val solution = HoldemCfrSolver.solve(
      hero = hero,
      state = state,
      villainPosterior = posterior,
      candidateActions = candidateActions,
      config = HoldemCfrConfig(
        iterations = 1_200,
        averagingDelay = 100,
        maxVillainHands = 32,
        equityTrials = 1_000,
        preferNativeBatch = true,
        rngSeed = 11L
      )
    )

    val probSum = solution.actionProbabilities.values.sum
    assert(math.abs(probSum - 1.0) < 1e-6, s"policy must sum to 1, got $probSum")
    assert(solution.bestAction != PokerAction.Fold, s"unexpected fold best action: ${solution.bestAction}")
    assert(solution.provider.nonEmpty)
    assert(solution.villainSupport > 0)
  }

  test("CFR baseline favors fold against very strong range and expensive call") {
    val hero = hole("7c", "2d")
    val state = GameState(
      street = Street.Preflop,
      board = Board.empty,
      pot = 5.0,
      toCall = 28.0,
      position = Position.Button,
      stackSize = 80.0,
      betHistory = Vector.empty
    )
    val posterior = DiscreteDistribution(
      Map(
        hole("Ah", "As") -> 0.7,
        hole("Kh", "Ks") -> 0.3
      )
    )
    val candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(40.0))

    val solution = HoldemCfrSolver.solve(
      hero = hero,
      state = state,
      villainPosterior = posterior,
      candidateActions = candidateActions,
      config = HoldemCfrConfig(
        iterations = 1_200,
        averagingDelay = 100,
        maxVillainHands = 32,
        equityTrials = 1_000,
        preferNativeBatch = true,
        rngSeed = 13L
      )
    )

    assertEquals(solution.bestAction, PokerAction.Fold)
    val foldProbability = solution.actionProbabilities.getOrElse(PokerAction.Fold, 0.0)
    assert(foldProbability > 0.65, s"expected high fold frequency, got $foldProbability")
    assert(solution.provider.nonEmpty)
  }

  test("native CPU provider falls back to scala when native path is invalid") {
    val missingNativePath = Paths
      .get(System.getProperty("java.io.tmpdir"), s"sicfun-cfr-native-missing-${System.nanoTime()}.dll")
      .toString

    withSystemProperties(
      Map(
        "sicfun.cfr.provider" -> "native-cpu",
        "sicfun.cfr.native.cpu.path" -> missingNativePath
      )
    ) {
      HoldemCfrNativeRuntime.resetLoadCacheForTests()
      HoldemCfrSolver.resetAutoProviderForTests()

      val hero = hole("Ac", "Ad")
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
          hole("7c", "2d") -> 0.5,
          hole("Kc", "Qd") -> 0.5
        )
      )
      val candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(8.0))

      val solution = HoldemCfrSolver.solve(
        hero = hero,
        state = state,
        villainPosterior = posterior,
        candidateActions = candidateActions,
        config = HoldemCfrConfig(
          iterations = 600,
          averagingDelay = 100,
          maxVillainHands = 16,
          equityTrials = 800,
          preferNativeBatch = true,
          rngSeed = 19L
        )
      )

      assertEquals(solution.provider, "scala")
    }
  }

  test("decision-only solve matches full solve root policy") {
    withSystemProperties(Map("sicfun.cfr.provider" -> "scala")) {
      val hero = hole("Ac", "Kd")
      val state = GameState(
        street = Street.Turn,
        board = Board.from(Vector(card("2c"), card("7d"), card("Jh"), card("Qc"))),
        pot = 18.0,
        toCall = 4.0,
        position = Position.Button,
        stackSize = 82.0,
        betHistory = Vector(BetAction(1, PokerAction.Raise(4.0)))
      )
      val posterior = DiscreteDistribution(
        Map(
          hole("As", "Qd") -> 0.35,
          hole("Ts", "9s") -> 0.30,
          hole("Qh", "Js") -> 0.20,
          hole("7c", "7s") -> 0.15
        )
      )
      val candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(12.0))
      val config = HoldemCfrConfig(
        iterations = 700,
        averagingDelay = 100,
        maxVillainHands = 32,
        equityTrials = 700,
        preferNativeBatch = true,
        rngSeed = 23L
      )

      val full = HoldemCfrSolver.solve(
        hero = hero,
        state = state,
        villainPosterior = posterior,
        candidateActions = candidateActions,
        config = config
      )
      val decision = HoldemCfrSolver.solveDecisionPolicy(
        hero = hero,
        state = state,
        villainPosterior = posterior,
        candidateActions = candidateActions,
        config = config
      )

      candidateActions.foreach { action =>
        val pFull = full.actionProbabilities.getOrElse(action, 0.0)
        val pDecision = decision.actionProbabilities.getOrElse(action, 0.0)
        assert(
          math.abs(pFull - pDecision) < 1e-9,
          s"policy mismatch for $action: full=$pFull decision=$pDecision"
        )
      }
    }
  }

  test("decision-only solve uses direct shortcut when root has no raise branch") {
    val hero = hole("Ac", "Kd")
    val state = GameState(
      street = Street.Turn,
      board = Board.from(Vector(card("2c"), card("7d"), card("Jh"), card("Qc"))),
      pot = 18.0,
      toCall = 4.0,
      position = Position.Button,
      stackSize = 82.0,
      betHistory = Vector(BetAction(1, PokerAction.Raise(4.0)))
    )
    val posterior = DiscreteDistribution(
      Map(
        hole("As", "Qd") -> 0.35,
        hole("Ts", "9s") -> 0.30,
        hole("Qh", "Js") -> 0.20,
        hole("7c", "7s") -> 0.15
      )
    )
    val decision = HoldemCfrSolver.solveDecisionPolicy(
      hero = hero,
      state = state,
      villainPosterior = posterior,
      candidateActions = Vector(PokerAction.Fold, PokerAction.Call),
      config = HoldemCfrConfig(
        iterations = 700,
        averagingDelay = 100,
        maxVillainHands = 32,
        equityTrials = 700,
        preferNativeBatch = true,
        rngSeed = 23L
      )
    )

    assertEquals(decision.provider, "direct")
    val probabilitySum = decision.actionProbabilities.values.sum
    assert(math.abs(probabilitySum - 1.0) < 1e-9, s"policy must sum to 1, got $probabilitySum")
    assert(
      decision.actionProbabilities.keySet == Set(PokerAction.Fold, PokerAction.Call),
      s"unexpected actions in direct policy: ${decision.actionProbabilities.keySet}"
    )
  }

  test("shallow decision solver bypasses CFR on one-reraise hall tree") {
    val hero = hole("Ac", "Kd")
    val state = GameState(
      street = Street.Turn,
      board = Board.from(Vector(card("2c"), card("7d"), card("Jh"), card("Qc"))),
      pot = 18.0,
      toCall = 4.0,
      position = Position.Button,
      stackSize = 82.0,
      betHistory = Vector(BetAction(1, PokerAction.Raise(4.0)))
    )
    val posterior = DiscreteDistribution(
      Map(
        hole("As", "Qd") -> 0.35,
        hole("Ts", "9s") -> 0.30,
        hole("Qh", "Js") -> 0.20,
        hole("7c", "7s") -> 0.15
      )
    )
    val candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(12.0))
    val config = HoldemCfrConfig(
      iterations = 2_000,
      averagingDelay = 100,
      maxVillainHands = 32,
      equityTrials = 700,
      preferNativeBatch = true,
      rngSeed = 23L
    )

    val full = HoldemCfrSolver.solve(
      hero = hero,
      state = state,
      villainPosterior = posterior,
      candidateActions = candidateActions,
      config = config
    )
    val direct = HoldemCfrSolver.solveShallowDecisionPolicy(
      hero = hero,
      state = state,
      villainPosterior = posterior,
      candidateActions = candidateActions,
      config = config
    )

    assertEquals(direct.provider, "direct-shallow")
    val probabilitySum = direct.actionProbabilities.values.sum
    assert(math.abs(probabilitySum - 1.0) < 1e-9, s"policy must sum to 1, got $probabilitySum")
    assertEquals(direct.bestAction, full.bestAction)
  }

  test("shallow decision solver supports multiple reraises without CFR fallback") {
    withSystemProperties(Map("sicfun.cfr.provider" -> "scala")) {
      val hero = hole("Ac", "Kd")
      val state = GameState(
        street = Street.Turn,
        board = Board.from(Vector(card("2c"), card("7d"), card("Jh"), card("Qc"))),
        pot = 18.0,
        toCall = 4.0,
        position = Position.Button,
        stackSize = 82.0,
        betHistory = Vector(BetAction(1, PokerAction.Raise(4.0)))
      )
      val posterior = DiscreteDistribution(
        Map(
          hole("As", "Qd") -> 0.35,
          hole("Ts", "9s") -> 0.30,
          hole("Qh", "Js") -> 0.20,
          hole("7c", "7s") -> 0.15
        )
      )
      val candidateActions = Vector(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(12.0))
      val config = HoldemCfrConfig(
        iterations = 3_000,
        averagingDelay = 100,
        maxVillainHands = 32,
        equityTrials = 700,
        includeVillainReraises = true,
        villainReraiseMultipliers = Vector(1.5, 2.0, 2.5),
        preferNativeBatch = true,
        rngSeed = 29L
      )

      val full = HoldemCfrSolver.solve(
        hero = hero,
        state = state,
        villainPosterior = posterior,
        candidateActions = candidateActions,
        config = config
      )
      val direct = HoldemCfrSolver.solveShallowDecisionPolicy(
        hero = hero,
        state = state,
        villainPosterior = posterior,
        candidateActions = candidateActions,
        config = config
      )

      assertEquals(direct.provider, "direct-shallow")
      val probabilitySum = direct.actionProbabilities.values.sum
      assert(math.abs(probabilitySum - 1.0) < 1e-9, s"policy must sum to 1, got $probabilitySum")
      assertEquals(direct.bestAction, full.bestAction)
    }
  }

  test("exact postflop equity falls back on turn") {
    val hero = hole("Ac", "Kd")
    val board = Board.from(Vector(card("2c"), card("7d"), card("Jh"), card("Qc")))
    val villains = Vector(
      hole("As", "Qd"),
      hole("Ts", "9s"),
      hole("Qh", "Js")
    )

    assertEquals(HoldemCfrSolver.exactPostflopEquity(hero, board, villains), None)
  }

  test("exact postflop equity falls back on flop") {
    val hero = hole("Ac", "Kd")
    val board = Board.from(Vector(card("2c"), card("7d"), card("Jh")))
    val villains = Vector(
      hole("As", "Qd"),
      hole("Ts", "9s"),
      hole("Qh", "Js")
    )

    assertEquals(HoldemCfrSolver.exactPostflopEquity(hero, board, villains), None)
  }

  test("exact postflop equity matches exact evaluator on river") {
    val hero = hole("Ac", "Kd")
    val board = Board.from(Vector(card("2c"), card("7d"), card("Jh"), card("Qc"), card("3s")))
    val villains = Vector(
      hole("As", "Qd"),
      hole("Ts", "9s"),
      hole("Qh", "Js")
    )

    val lookup = HoldemCfrSolver.exactPostflopEquity(hero, board, villains)
      .getOrElse(fail("expected exact river lookup"))

    villains.foreach { villain =>
      val exact = HoldemEquity.equityExact(
        hero = hero,
        board = board,
        villainRange = DiscreteDistribution(Map(villain -> 1.0))
      ).equity
      val observed = lookup(HoleCardsIndex.fastIdOf(villain))
      if observed.isNaN then fail(s"missing lookup for ${villain.toToken}")
      assert(
        math.abs(observed - exact) < 1e-12,
        s"river exact equity mismatch for ${villain.toToken}: observed=$observed exact=$exact"
      )
    }
  }
