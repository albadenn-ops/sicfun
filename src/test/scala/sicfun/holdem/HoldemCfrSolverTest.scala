package sicfun.holdem

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
