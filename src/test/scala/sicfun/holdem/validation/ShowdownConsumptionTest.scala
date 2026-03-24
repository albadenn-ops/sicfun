package sicfun.holdem.validation

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.engine.RangeInferenceEngine
import sicfun.holdem.equity.{TableFormat, TableRanges}
import sicfun.holdem.history.{HandHistoryImport, HandHistorySite, OpponentProfile, ShowdownRecord}
import sicfun.holdem.model.PokerActionModel
import sicfun.holdem.types.*

import scala.util.Random

class ShowdownConsumptionTest extends FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(600, "s")

  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card: $token"))

  private def hole(a: String, b: String): HoleCards =
    HoleCards.from(Vector(card(a), card(b)))

  private def showdown(handId: String, a: String, b: String): ShowdownRecord =
    ShowdownRecord(handId, hole(a, b))

  private def simulateProfile(
      leak: InjectedLeak,
      handsToPlay: Int,
      seed: Long
  ): OpponentProfile =
    val villain = LeakInjectedVillain(
      name = s"test_${leak.id}_$seed",
      leaks = Vector(leak),
      baselineNoise = if leak.id == NoLeak.Id then 0.0 else 0.03,
      seed = seed
    )
    val sim = new HeadsUpSimulator(
      heroEngine = None,
      villain = villain,
      seed = seed,
      villainStrategy = ValidationRunner.villainStrategyFor(leak)
    )
    val records = (1 to handsToPlay).map(sim.playHand).toVector
    val text = PokerStarsExporter.exportHands(records, "Hero", villain.name)
    val parsed = HandHistoryImport.parseText(text, Some(HandHistorySite.PokerStars), Some("Hero"))
    assert(parsed.isRight, s"parse failed: ${parsed.left.getOrElse("")}")
    val profiles = OpponentProfile.fromImportedHands("simulated", parsed.toOption.get, Set("Hero"))
    profiles.headOption.getOrElse(fail(s"missing profile for ${leak.id}"))

  private def isShowdownHint(text: String): Boolean =
    text.contains("shown down") ||
      text.contains("premium hands frequently") ||
      text.contains("pair-heavy")

  test("overcall leak players surface showdown weak-hand hints") {
    val profile = simulateProfile(Overcalls(0.9), handsToPlay = 3_000, seed = 42L)

    assert(profile.showdownHands.nonEmpty, "expected simulated leak player to reach showdown")
    assert(
      profile.exploitHints.exists(_.contains("shown down weak hands")),
      s"expected showdown weak-hand hint, got: ${profile.exploitHints}"
    )
  }

  test("GTO control does not produce showdown-specific false positives") {
    val profile = simulateProfile(NoLeak(), handsToPlay = 2_000, seed = 77L)

    assert(profile.showdownHands.nonEmpty, "expected GTO control to reach showdown")
    assert(
      !profile.exploitHints.exists(isShowdownHint),
      s"expected no showdown-specific hints for GTO control, got: ${profile.exploitHints}"
    )
  }

  test("historical showdown bias shifts premium prior only mildly") {
    RangeInferenceEngine.clearPosteriorCache()

    val hero = hole("Jc", "Td")
    val table = TableRanges.defaults(TableFormat.HeadsUp)
    val premiumHistory = Vector.tabulate(10)(idx => showdown(s"sd-${idx + 1}", "Ah", "As"))

    val withoutHistory = RangeInferenceEngine.inferPosterior(
      hero = hero,
      board = Board.empty,
      folds = Vector.empty,
      tableRanges = table,
      villainPos = Position.BigBlind,
      observations = Seq.empty,
      actionModel = PokerActionModel.uniform,
      rng = new Random(201),
      useCache = false
    )
    val withHistory = RangeInferenceEngine.inferPosterior(
      hero = hero,
      board = Board.empty,
      folds = Vector.empty,
      tableRanges = table,
      villainPos = Position.BigBlind,
      observations = Seq.empty,
      actionModel = PokerActionModel.uniform,
      rng = new Random(202),
      useCache = false,
      showdownHistory = premiumHistory
    )

    val premium = hole("Ah", "As")
    val baseProb = withoutHistory.posterior.probabilityOf(premium)
    val biasedProb = withHistory.posterior.probabilityOf(premium)
    val relativeChange = (biasedProb - baseProb) / baseProb

    assert(biasedProb > baseProb, s"expected premium combo to gain weight: base=$baseProb biased=$biasedProb")
    assert(relativeChange < 0.15, s"bias should stay mild: relativeChange=$relativeChange")
  }
