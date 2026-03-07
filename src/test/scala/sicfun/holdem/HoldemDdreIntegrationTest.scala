package sicfun.holdem

import munit.FunSuite
import sicfun.core.Card

import scala.util.Random

class HoldemDdreIntegrationTest extends FunSuite:
  private val DdreModeProperty = "sicfun.ddre.mode"
  private val DdreProviderProperty = "sicfun.ddre.provider"
  private val DdreAlphaProperty = "sicfun.ddre.alpha"
  private val DdreMinEntropyBitsProperty = "sicfun.ddre.minEntropyBits"
  private val PreflopBackendProperty = "sicfun.holdem.preflopEquityBackend"

  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card token: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def withSystemProperties[A](updates: Seq[(String, Option[String])])(thunk: => A): A =
    TestSystemPropertyScope.withSystemProperties(updates)(thunk)

  private def runPosterior(seed: Long): PosteriorInferenceResult =
    val hero = hole("Ac", "Kd")
    val board = Board.empty
    val villainPos = Position.BigBlind
    val folds = Vector(PreflopFold(Position.UTG))
    val state = GameState(
      street = Street.Preflop,
      board = board,
      pot = 6.0,
      toCall = 2.0,
      position = villainPos,
      stackSize = 98.0,
      betHistory = Vector.empty
    )
    RangeInferenceEngine.inferPosterior(
      hero = hero,
      board = board,
      folds = folds,
      tableRanges = TableRanges.defaults(TableFormat.NineMax),
      villainPos = villainPos,
      observations = Seq(VillainObservation(PokerAction.Raise(25.0), state)),
      actionModel = PokerActionModel.uniform,
      bunchingTrials = 60,
      rng = new Random(seed),
      useCache = false
    )

  private def l1Distance(a: PosteriorInferenceResult, b: PosteriorInferenceResult): Double =
    a.posterior.weights.keysIterator.map { hand =>
      math.abs(a.posterior.probabilityOf(hand) - b.posterior.probabilityOf(hand))
    }.sum

  test("DDRE shadow mode keeps Bayesian decision posterior unchanged") {
    val baseline = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("off"),
        DdreProviderProperty -> Some("disabled"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 31L)
    }

    val shadow = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("shadow"),
        DdreProviderProperty -> Some("synthetic"),
        DdreAlphaProperty -> Some("0.8"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 31L)
    }

    assertEquals(shadow.posterior.weights, baseline.posterior.weights)
    assertEqualsDouble(shadow.logEvidence, baseline.logEvidence, 1e-12)
  }

  test("DDRE blend mode falls back to Bayesian when provider is disabled") {
    val baseline = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("off"),
        DdreProviderProperty -> Some("disabled"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 41L)
    }

    val blendFallback = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("blend-canary"),
        DdreProviderProperty -> Some("disabled"),
        DdreAlphaProperty -> Some("1.0"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 41L)
    }

    assertEquals(blendFallback.posterior.weights, baseline.posterior.weights)
  }

  test("DDRE blend-primary with alpha=1 can drive non-Bayesian posterior") {
    val baseline = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("off"),
        DdreProviderProperty -> Some("disabled"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 53L)
    }

    val blend = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("blend-primary"),
        DdreProviderProperty -> Some("synthetic"),
        DdreAlphaProperty -> Some("1.0"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 53L)
    }

    val l1 = l1Distance(baseline, blend)
    assert(l1 > 1e-6, s"expected blended posterior to differ from Bayesian baseline; l1=$l1")
    assert(math.abs(blend.posterior.weights.values.sum - 1.0) < 1e-9)
  }

  test("DDRE entropy guard degradation falls back to Bayesian posterior in blend mode") {
    val baseline = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("off"),
        DdreProviderProperty -> Some("disabled"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 71L)
    }

    val entropyGuardedBlend = withSystemProperties(
      Seq(
        DdreModeProperty -> Some("blend-primary"),
        DdreProviderProperty -> Some("synthetic"),
        DdreAlphaProperty -> Some("1.0"),
        DdreMinEntropyBitsProperty -> Some("9999.0"),
        PreflopBackendProperty -> Some("cpu")
      )
    ) {
      runPosterior(seed = 71L)
    }

    assertEquals(entropyGuardedBlend.posterior.weights, baseline.posterior.weights)
  }
