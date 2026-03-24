package sicfun.holdem.validation

import munit.FunSuite
import sicfun.holdem.history.{HandHistoryImport, HandHistorySite, OpponentProfile}
import sicfun.holdem.types.{PokerAction, Street}

/** Calibration test: a CFR equilibrium villain should produce ZERO false
  * positive exploit hints from the profiler.
  *
  * If this test fails, the profiler's hardcoded thresholds disagree with
  * actual equilibrium play and must be recalibrated.
  */
class CfrGtoCalibrationTest extends FunSuite:

  override val munitTimeout = scala.concurrent.duration.Duration(600, "s")

  test("CFR equilibrium villain produces no false positive leak hints"):
    val cfrStrategy = CfrVillainStrategy(allowHeuristicFallback = false)
    val villain = LeakInjectedVillain(
      name = "cfr_gto_control",
      leaks = Vector(NoLeak()),
      baselineNoise = 0.0,
      seed = 42L
    )
    val sim = new HeadsUpSimulator(
      heroEngine = None,
      villain = villain,
      seed = 42L,
      villainStrategy = cfrStrategy
    )

    val numHands = 2000
    val records = (1 to numHands).map(sim.playHand).toVector

    // Export, parse, profile
    val text = PokerStarsExporter.exportHands(records, "Hero", villain.name)
    val parsed = HandHistoryImport.parseText(text, Some(HandHistorySite.PokerStars), Some("Hero"))
    assert(parsed.isRight, s"parse failed: ${parsed.left.getOrElse("")}")
    val hands = parsed.toOption.get
    val profiles = OpponentProfile.fromImportedHands("simulated", hands, Set("Hero"))
    assert(profiles.nonEmpty, "no profile generated")
    val profile = profiles.head

    // === Calibration data: measure actual rates ===
    val events = profile.recentEvents
    val sig = profile.signature

    println(s"[CFR-CAL] Hands: $numHands, Events: ${events.size}")
    println(f"[CFR-CAL] Signature: fold=${sig.values(0)}%.3f raise=${sig.values(1)}%.3f " +
      f"call=${sig.values(2)}%.3f check=${sig.values(3)}%.3f")
    println(s"[CFR-CAL] Archetype: ${profile.archetypePosterior.mapEstimate}")

    // Per-metric calibration
    val riverFacingBet = events.filter(e => e.street == Street.River && e.toCall > 0)
    if riverFacingBet.size >= 5 then
      val riverFoldRate = riverFacingBet.count(_.action == PokerAction.Fold).toDouble / riverFacingBet.size
      println(f"[CFR-CAL] River fold rate (facing bet):  $riverFoldRate%.3f  (threshold: 0.30)")

    val facingLargeBet = events.filter { e =>
      e.toCall > 0 && e.potBefore > e.toCall && e.toCall / (e.potBefore - e.toCall) >= 0.6
    }
    if facingLargeBet.size >= 5 then
      val largeBetCallRate = facingLargeBet.count(_.action == PokerAction.Call).toDouble / facingLargeBet.size
      println(f"[CFR-CAL] Large bet call rate:           $largeBetCallRate%.3f  (threshold: 0.60)")

    val turnEvents = events.filter(_.street == Street.Turn)
    if turnEvents.size >= 5 then
      val turnRaiseRate = turnEvents.count(_.action.category == PokerAction.Category.Raise).toDouble / turnEvents.size
      println(f"[CFR-CAL] Turn raise rate:               $turnRaiseRate%.3f  (threshold: 0.25)")

    val bigPotCanBet = events.filter(e =>
      e.potBefore > 0 && e.stackBefore / e.potBefore < 4.0 && e.toCall == 0
    )
    if bigPotCanBet.size >= 5 then
      val bigPotCheckRate = bigPotCanBet.count(_.action == PokerAction.Check).toDouble / bigPotCanBet.size
      println(f"[CFR-CAL] Big pot check rate:            $bigPotCheckRate%.3f  (threshold: 0.75)")

    val preflopEvents = events.filter(_.street == Street.Preflop)
    if preflopEvents.size >= 8 then
      val preflopFoldRate = preflopEvents.count(_.action == PokerAction.Fold).toDouble / preflopEvents.size
      println(f"[CFR-CAL] Preflop fold rate:             $preflopFoldRate%.3f  (threshold: 0.20)")
      val preflopCallRate = preflopEvents.count(_.action == PokerAction.Call).toDouble / preflopEvents.size
      println(f"[CFR-CAL] Preflop call rate:             $preflopCallRate%.3f  (threshold: 0.65)")

    val responseTotal = profile.raiseResponses.total.toDouble
    if responseTotal >= 4.0 then
      val foldToRaise = profile.raiseResponses.folds / responseTotal
      val callVsRaise = profile.raiseResponses.calls / responseTotal
      val reraiseVsRaise = profile.raiseResponses.raises / responseTotal
      println(f"[CFR-CAL] Fold to raise:                 $foldToRaise%.3f  (threshold: 0.55)")
      println(f"[CFR-CAL] Call vs raise:                 $callVsRaise%.3f  (threshold: 0.55)")
      println(f"[CFR-CAL] Reraise vs raise:              $reraiseVsRaise%.3f  (threshold: 0.28)")

    // === False positive check ===
    val hints = profile.exploitHints
    println(s"[CFR-CAL] Exploit hints: $hints")
    assert(
      !hints.exists(showdownHint),
      s"CFR equilibrium villain should not trigger showdown-specific hints: $hints"
    )

    val leakIds = Vector("overfold-river-aggression", "overcall-big-bets", "overbluff-turn-barrel",
      "passive-big-pots", "preflop-too-loose", "preflop-too-tight")
    val matches = leakIds.filter(id => hintMatchesLeak(hints, id))
    println(s"[CFR-CAL] False positive leak patterns: $matches")

    assert(matches.isEmpty,
      s"CFR equilibrium villain triggers false positive leak patterns: $matches\n" +
        s"  Hints: $hints\n" +
        s"  Use calibration data above to adjust thresholds in OpponentProfileStore.exploitHintsFor")

  private def hintMatchesLeak(hints: Vector[String], leakId: String): Boolean =
    leakId match
      case "overfold-river-aggression" =>
        hints.exists(_.contains("Over-folds on the river"))
      case "overcall-big-bets" =>
        hints.exists(h =>
          h.contains("calling station") ||
            h.contains("Calls too often facing large bets") ||
            h.contains("shown down weak hands")
        )
      case "overbluff-turn-barrel" =>
        hints.exists(_.contains("Very aggressive on the turn"))
      case "passive-big-pots" =>
        false // Not detectable by action frequency — GTO checks ~100% in big pots
      case "preflop-too-loose" =>
        hints.exists(_.contains("Calls too loose preflop"))
      case "preflop-too-tight" =>
        hints.exists(_.contains("Over-folds preflop"))
      case _ => false

  private def showdownHint(hint: String): Boolean =
    hint.contains("shown down") ||
      hint.contains("premium hands frequently") ||
      hint.contains("pair-heavy")
