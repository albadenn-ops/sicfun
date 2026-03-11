package sicfun.holdem.provider
import sicfun.holdem.types.*
import sicfun.holdem.model.*
import sicfun.holdem.gpu.*

import munit.FunSuite
import sicfun.core.{Card, DiscreteDistribution}
import sicfun.holdem.equity.HoldemEquity

import java.nio.file.Paths

class HoldemBayesProviderTest extends FunSuite:
  private val CpuProviderProperty = "sicfun.bayes.provider"
  private val CpuPathProperty = "sicfun.bayes.native.cpu.path"

  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card token: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def withSystemProperties(properties: Map[String, String])(thunk: => Unit): Unit =
    TestSystemPropertyScope.withSystemProperties(
      properties.toVector.map { case (key, value) => key -> Some(value) }
    ) {
      HoldemBayesNativeRuntime.resetLoadCacheForTests()
      HoldemBayesProvider.resetAutoProviderForTests()
      try thunk
      finally
        HoldemBayesNativeRuntime.resetLoadCacheForTests()
        HoldemBayesProvider.resetAutoProviderForTests()
    }

  test("forced native CPU falls back to scala when native path is invalid") {
    val missingPath = Paths
      .get(System.getProperty("java.io.tmpdir"), s"sicfun-bayes-native-missing-${System.nanoTime()}.dll")
      .toString

    withSystemProperties(
      Map(
        CpuProviderProperty -> "native-cpu",
        CpuPathProperty -> missingPath
      )
    ) {
      val prior = DiscreteDistribution(
        Map(
          hole("Ac", "Kd") -> 0.55,
          hole("7c", "2d") -> 0.45
        )
      )
      val state = GameState(
        street = Street.Preflop,
        board = Board.empty,
        pot = 8.0,
        toCall = 2.0,
        position = Position.Button,
        stackSize = 100.0,
        betHistory = Vector.empty
      )
      val result = HoldemBayesProvider.updatePosterior(
        prior = prior,
        observations = Seq(PokerAction.Raise(6.0) -> state),
        actionModel = PokerActionModel.uniform
      )

      assertEquals(result.provider, HoldemBayesProvider.Provider.Scala)
      val sum = result.posterior.weights.values.sum
      assert(math.abs(sum - 1.0) < 1e-9, s"posterior must sum to 1.0, got $sum")
    }
  }

  test("explicit scala provider keeps scala backend") {
    withSystemProperties(Map(CpuProviderProperty -> "scala")) {
      val prior = DiscreteDistribution(
        Map(
          hole("As", "Ah") -> 0.5,
          hole("Kc", "Qd") -> 0.5
        )
      )
      val state = GameState(
        street = Street.Flop,
        board = Board.from(Seq(card("2c"), card("7d"), card("Jh"))),
        pot = 18.0,
        toCall = 6.0,
        position = Position.Button,
        stackSize = 94.0,
        betHistory = Vector.empty
      )

      val result = HoldemBayesProvider.updatePosterior(
        prior = prior,
        observations = Seq(PokerAction.Call -> state),
        actionModel = PokerActionModel.uniform
      )
      assertEquals(result.provider, HoldemBayesProvider.Provider.Scala)
    }
  }

  test("shadow config parses explicit property overrides") {
    withSystemProperties(
      Map(
        "sicfun.bayes.shadow.enabled" -> "true",
        "sicfun.bayes.shadow.failClosed" -> "true",
        "sicfun.bayes.shadow.posteriorMaxAbsDiff" -> "1e-8",
        "sicfun.bayes.shadow.logEvidenceMaxAbsDiff" -> "1e-10"
      )
    ) {
      val config = HoldemBayesProvider.configuredShadowConfig()
      assertEquals(config.enabled, true)
      assertEquals(config.failClosed, true)
      assertEqualsDouble(config.posteriorMaxAbsDiff, 1e-8, 0.0)
      assertEqualsDouble(config.logEvidenceMaxAbsDiff, 1e-10, 0.0)
    }
  }

  test("computePosteriorDrift reports max posterior and log evidence deltas") {
    val h1 = hole("As", "Ah")
    val h2 = hole("Kc", "Qd")
    val h3 = hole("7c", "2d")
    val hypotheses = Vector(h1, h2, h3)
    val candidateDist = DiscreteDistribution(
      Map(h1 -> 0.20, h2 -> 0.30, h3 -> 0.50)
    )
    val candidateCompact = HoldemEquity.buildCompactPosterior(
      Vector(h1, h2, h3), Array(0.20, 0.30, 0.50)
    )
    val candidate = HoldemBayesProvider.UpdateResult(
      posterior = candidateDist,
      compact = candidateCompact,
      logEvidence = -3.0,
      provider = HoldemBayesProvider.Provider.NativeCpu
    )
    val referenceDist = DiscreteDistribution(
      Map(h1 -> 0.21, h2 -> 0.29, h3 -> 0.50)
    )
    val referenceCompact = HoldemEquity.buildCompactPosterior(
      Vector(h1, h2, h3), Array(0.21, 0.29, 0.50)
    )
    val reference = HoldemBayesProvider.UpdateResult(
      posterior = referenceDist,
      compact = referenceCompact,
      logEvidence = -2.95,
      provider = HoldemBayesProvider.Provider.Scala
    )

    val drift = HoldemBayesProvider.computePosteriorDrift(hypotheses, candidate, reference)
    assertEqualsDouble(drift.posteriorMaxAbsDiff, 0.01, 1e-12)
    assertEqualsDouble(drift.logEvidenceAbsDiff, 0.05, 1e-12)
  }
