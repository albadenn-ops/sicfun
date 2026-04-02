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

  private def defaultSpot(id: String): HoldemCfrApproximationReport.DiagnosticSpot =
    HoldemCfrApproximationReport.DefaultSuite.find(_.id == id).getOrElse(
      fail(s"missing default diagnostic spot: $id")
    )

  private def withSystemProperties[A](properties: Map[String, String])(thunk: => A): A =
    TestSystemPropertyScope.withSystemProperties(properties.toSeq.map { case (key, value) => key -> Some(value) }) {
      HoldemCfrNativeRuntime.resetLoadCacheForTests()
      HoldemCfrSolver.resetAutoProviderForTests()
      try thunk
      finally
        HoldemCfrNativeRuntime.resetLoadCacheForTests()
        HoldemCfrSolver.resetAutoProviderForTests()
    }

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

  test("scala-fixed full solve stays close to scala on preflop spot") {
    withSystemProperties(Map("sicfun.cfr.provider" -> "scala")) {
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
      val config = HoldemCfrConfig(
        iterations = 1_200,
        averagingDelay = 100,
        maxVillainHands = 32,
        equityTrials = 1_000,
        preferNativeBatch = true,
        rngSeed = 37L
      )

      val baseline = HoldemCfrSolver.solve(hero, state, posterior, candidateActions, config)

      withSystemProperties(Map("sicfun.cfr.provider" -> "scala-fixed")) {
        val fixed = HoldemCfrSolver.solve(hero, state, posterior, candidateActions, config)

        assertEquals(fixed.provider, "scala-fixed")
        assertEquals(fixed.bestAction, baseline.bestAction)
        assertEqualsDouble(fixed.expectedValuePlayer0, baseline.expectedValuePlayer0, 0.15)
        candidateActions.foreach { action =>
          val pBaseline = baseline.actionProbabilities.getOrElse(action, 0.0)
          val pFixed = fixed.actionProbabilities.getOrElse(action, 0.0)
          assert(
            math.abs(pFixed - pBaseline) <= 0.08,
            s"policy mismatch for $action: baseline=$pBaseline fixed=$pFixed"
          )
        }
      }
    }
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

  test("native CPU decision-only solve falls back to scala when native path is invalid") {
    val missingNativePath = Paths
      .get(System.getProperty("java.io.tmpdir"), s"sicfun-cfr-native-decision-missing-${System.nanoTime()}.dll")
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

      val decision = HoldemCfrSolver.solveDecisionPolicy(
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
          rngSeed = 29L
        )
      )

      assertEquals(decision.provider, "scala")
    }
  }

  test("native CPU fixed provider falls back to scala when native path is invalid") {
    val missingNativePath = Paths
      .get(System.getProperty("java.io.tmpdir"), s"sicfun-cfr-native-fixed-missing-${System.nanoTime()}.dll")
      .toString

    withSystemProperties(
      Map(
        "sicfun.cfr.provider" -> "native-cpu-fixed",
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
          rngSeed = 43L
        )
      )

      assertEquals(solution.provider, "scala")
    }
  }

  test("native GPU fixed provider falls back to scala when native path is invalid") {
    val missingNativePath = Paths
      .get(System.getProperty("java.io.tmpdir"), s"sicfun-cfr-gpu-fixed-missing-${System.nanoTime()}.dll")
      .toString
    val missingCpuPath = Paths
      .get(System.getProperty("java.io.tmpdir"), s"sicfun-cfr-gpu-fixed-cpu-missing-${System.nanoTime()}.dll")
      .toString

    withSystemProperties(
      Map(
        "sicfun.cfr.provider" -> "native-gpu-fixed",
        "sicfun.cfr.native.gpu.path" -> missingNativePath,
        "sicfun.cfr.native.cpu.path" -> missingCpuPath
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
          rngSeed = 47L
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

  test("native CPU decision-only solve matches native full solve root policy") {
    val availability = HoldemCfrNativeRuntime.availability(HoldemCfrNativeRuntime.Backend.Cpu)
    if availability.available then
      withSystemProperties(Map("sicfun.cfr.provider" -> "native-cpu")) {
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
          rngSeed = 53L
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

        assertEquals(full.provider, "native-cpu")
        assertEquals(decision.provider, "native-cpu")
        assertEquals(decision.bestAction, full.bestAction)
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

  test("native GPU decision-only solve matches native full solve root policy") {
    val availability = HoldemCfrNativeRuntime.availability(HoldemCfrNativeRuntime.Backend.Gpu)
    if availability.available then
      withSystemProperties(Map("sicfun.cfr.provider" -> "native-gpu")) {
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
          rngSeed = 59L
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

        assertEquals(full.provider, "native-gpu")
        assertEquals(decision.provider, "native-gpu")
        assertEquals(decision.bestAction, full.bestAction)
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

  test("scala-fixed decision-only solve stays close to scala root policy") {
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
        rngSeed = 41L
      )

      val baseline = HoldemCfrSolver.solveDecisionPolicy(hero, state, posterior, candidateActions, config)

      withSystemProperties(Map("sicfun.cfr.provider" -> "scala-fixed")) {
        val fixed = HoldemCfrSolver.solveDecisionPolicy(hero, state, posterior, candidateActions, config)

        assertEquals(fixed.provider, "scala-fixed")
        assertEquals(fixed.bestAction, baseline.bestAction)
        candidateActions.foreach { action =>
          val pBaseline = baseline.actionProbabilities.getOrElse(action, 0.0)
          val pFixed = fixed.actionProbabilities.getOrElse(action, 0.0)
          assert(
            math.abs(pFixed - pBaseline) <= 0.08,
            s"policy mismatch for $action: baseline=$pBaseline fixed=$pFixed"
          )
        }
      }
    }
  }

  test("native CPU fixed decision-only solve stays aligned with scala-fixed across averaging-delay boundary") {
    val availability = HoldemCfrNativeRuntime.availability(HoldemCfrNativeRuntime.Backend.Cpu)
    if availability.available then
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
      val configs = Vector(
        HoldemCfrConfig(
          iterations = 50,
          averagingDelay = 100,
          maxVillainHands = 32,
          equityTrials = 700,
          preferNativeBatch = true,
          rngSeed = 61L
        ),
        HoldemCfrConfig(
          iterations = 101,
          averagingDelay = 100,
          maxVillainHands = 32,
          equityTrials = 700,
          preferNativeBatch = true,
          rngSeed = 67L
        )
      )

      configs.foreach { config =>
        val scalaFixed =
          withSystemProperties(Map("sicfun.cfr.provider" -> "scala-fixed")) {
            HoldemCfrSolver.solveDecisionPolicy(hero, state, posterior, candidateActions, config)
          }
        val nativeFixed =
          withSystemProperties(Map("sicfun.cfr.provider" -> "native-cpu-fixed")) {
            HoldemCfrSolver.solveDecisionPolicy(hero, state, posterior, candidateActions, config)
          }

        assertEquals(nativeFixed.provider, "native-cpu-fixed")
        assertEquals(nativeFixed.bestAction, scalaFixed.bestAction)
        candidateActions.foreach { action =>
          assertEqualsDouble(
            nativeFixed.actionProbabilities.getOrElse(action, 0.0),
            scalaFixed.actionProbabilities.getOrElse(action, 0.0),
            5e-4
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

  test("shallow decision solver preserves decision-only CFR root policy on one-reraise hall tree") {
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
        rngSeed = 23L
      )

      val decisionOnly = HoldemCfrSolver.solveDecisionPolicy(
        hero = hero,
        state = state,
        villainPosterior = posterior,
        candidateActions = candidateActions,
        config = config
      )
      val shallow = HoldemCfrSolver.solveShallowDecisionPolicy(
        hero = hero,
        state = state,
        villainPosterior = posterior,
        candidateActions = candidateActions,
        config = config
      )

      val probabilitySum = shallow.actionProbabilities.values.sum
      assert(math.abs(probabilitySum - 1.0) < 1e-9, s"policy must sum to 1, got $probabilitySum")
      assertEquals(shallow.provider, decisionOnly.provider)
      assertEquals(shallow.bestAction, decisionOnly.bestAction)
      candidateActions.foreach { action =>
        assertEqualsDouble(
          shallow.actionProbabilities.getOrElse(action, 0.0),
          decisionOnly.actionProbabilities.getOrElse(action, 0.0),
          1e-9
        )
      }
    }
  }

  test("shallow decision solver preserves decision-only CFR root policy with multiple reraises") {
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

      val decisionOnly = HoldemCfrSolver.solveDecisionPolicy(
        hero = hero,
        state = state,
        villainPosterior = posterior,
        candidateActions = candidateActions,
        config = config
      )
      val shallow = HoldemCfrSolver.solveShallowDecisionPolicy(
        hero = hero,
        state = state,
        villainPosterior = posterior,
        candidateActions = candidateActions,
        config = config
      )

      val probabilitySum = shallow.actionProbabilities.values.sum
      assert(math.abs(probabilitySum - 1.0) < 1e-9, s"policy must sum to 1, got $probabilitySum")
      assertEquals(shallow.provider, decisionOnly.provider)
      assertEquals(shallow.bestAction, decisionOnly.bestAction)
      candidateActions.foreach { action =>
        assertEqualsDouble(
          shallow.actionProbabilities.getOrElse(action, 0.0),
          decisionOnly.actionProbabilities.getOrElse(action, 0.0),
          1e-9
        )
      }
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

  test("postflop lookahead flips the turn probe benchmark from call to fold") {
    withSystemProperties(Map("sicfun.cfr.provider" -> "scala")) {
      val spot = defaultSpot("hu_turn_button_vs_probe")
      val baseConfig = HoldemCfrConfig(
        iterations = 1_200,
        averagingDelay = 100,
        maxVillainHands = 32,
        equityTrials = 700,
        includeVillainReraises = true,
        villainReraiseMultipliers = Vector(1.5, 2.0),
        preferNativeBatch = false,
        rngSeed = 29L
      )

      val shallow = HoldemCfrSolver.solve(
        hero = spot.hero,
        state = spot.state,
        villainPosterior = spot.villainRange,
        candidateActions = spot.candidateActions,
        config = baseConfig
      )
      val lookahead = HoldemCfrSolver.solve(
        hero = spot.hero,
        state = spot.state,
        villainPosterior = spot.villainRange,
        candidateActions = spot.candidateActions,
        config = baseConfig.copy(postflopLookahead = true)
      )

      assertEquals(shallow.bestAction, PokerAction.Call)
      assertEquals(lookahead.bestAction, PokerAction.Fold)
      assert(
        lookahead.actionProbabilities.getOrElse(PokerAction.Fold, 0.0) >
          lookahead.actionProbabilities.getOrElse(PokerAction.Call, 0.0),
        s"expected lookahead to prefer fold, got ${lookahead.actionProbabilities}"
      )
    }
  }

  test("postflop lookahead is a no-op on river spots") {
    withSystemProperties(Map("sicfun.cfr.provider" -> "scala")) {
      val spot = defaultSpot("hu_river_bluffcatch")
      val baseConfig = HoldemCfrConfig(
        iterations = 900,
        averagingDelay = 100,
        maxVillainHands = 32,
        equityTrials = 700,
        preferNativeBatch = false,
        rngSeed = 31L
      )

      val baseline = HoldemCfrSolver.solveDecisionPolicy(
        hero = spot.hero,
        state = spot.state,
        villainPosterior = spot.villainRange,
        candidateActions = spot.candidateActions,
        config = baseConfig
      )
      val lookahead = HoldemCfrSolver.solveDecisionPolicy(
        hero = spot.hero,
        state = spot.state,
        villainPosterior = spot.villainRange,
        candidateActions = spot.candidateActions,
        config = baseConfig.copy(postflopLookahead = true)
      )

      assertEquals(lookahead.provider, baseline.provider)
      assertEquals(lookahead.bestAction, baseline.bestAction)
      spot.candidateActions.foreach { action =>
        assertEqualsDouble(
          lookahead.actionProbabilities.getOrElse(action, 0.0),
          baseline.actionProbabilities.getOrElse(action, 0.0),
          1e-9
        )
      }
    }
  }

  test("postflop lookahead can run on native CPU and stays close to scala") {
    val availability = HoldemCfrNativeRuntime.availability(HoldemCfrNativeRuntime.Backend.Cpu)
    if availability.available then
      val spot = defaultSpot("hu_turn_button_vs_probe")
      val config = HoldemCfrConfig(
        iterations = 1_200,
        averagingDelay = 100,
        maxVillainHands = 32,
        equityTrials = 700,
        includeVillainReraises = true,
        villainReraiseMultipliers = Vector(1.5, 2.0),
        postflopLookahead = true,
        preferNativeBatch = false,
        rngSeed = 37L
      )

      val scalaSolution =
        withSystemProperties(Map("sicfun.cfr.provider" -> "scala")) {
          HoldemCfrSolver.solve(
            hero = spot.hero,
            state = spot.state,
            villainPosterior = spot.villainRange,
            candidateActions = spot.candidateActions,
            config = config
          )
        }

      val nativeSolution =
        withSystemProperties(Map("sicfun.cfr.provider" -> "native-cpu")) {
          HoldemCfrSolver.solve(
            hero = spot.hero,
            state = spot.state,
            villainPosterior = spot.villainRange,
            candidateActions = spot.candidateActions,
            config = config
          )
        }

      assertEquals(nativeSolution.provider, "native-cpu")
      assertEquals(nativeSolution.bestAction, scalaSolution.bestAction)
      spot.candidateActions.foreach { action =>
        assertEqualsDouble(
          nativeSolution.actionProbabilities.getOrElse(action, 0.0),
          scalaSolution.actionProbabilities.getOrElse(action, 0.0),
          5e-4
        )
      }
  }
