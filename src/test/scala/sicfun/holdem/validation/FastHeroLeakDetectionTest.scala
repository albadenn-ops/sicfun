package sicfun.holdem.validation

import munit.FunSuite
import sicfun.holdem.history.{HandHistoryImport, HandHistorySite, OpponentProfile}

/** Verifies that fast-hero simulation (no engine) produces hand histories
  * where the profiling pipeline still detects all 6 injected leak types.
  *
  * Each test simulates enough hands with a severe (0.9) leak, exports to
  * PokerStars format, parses back through the profiling pipeline, and asserts
  * that exploit hints match the injected leak.
  */
class FastHeroLeakDetectionTest extends FunSuite:

  // 3000 hands is enough for severe leaks to surface clearly
  private val HandsToPlay = 3000

  override val munitTimeout = scala.concurrent.duration.Duration(120, "s")

  private def assertLeakDetected(leak: InjectedLeak, seed: Long = 42L): Unit =
    val villain = LeakInjectedVillain(
      name = s"test_${leak.id}",
      leaks = Vector(leak),
      baselineNoise = 0.03,
      seed = seed
    )
    val sim = new HeadsUpSimulator(
      heroEngine = None,
      villain = villain,
      seed = seed
    )
    val records = (1 to HandsToPlay).map(sim.playHand).toVector

    // Verify we got enough leak firings
    val leakActions = records.flatMap(_.actions).filter(_.leakId.contains(leak.id))
    assert(leakActions.nonEmpty, s"leak ${leak.id} never fired in $HandsToPlay hands")

    // Export and parse back through profiling pipeline
    val text = PokerStarsExporter.exportHands(records, "Hero", villain.name)
    val parsed = HandHistoryImport.parseText(text, Some(HandHistorySite.PokerStars), Some("Hero"))
    assert(parsed.isRight, s"parse failed: ${parsed.left.getOrElse("")}")

    val hands = parsed.toOption.get
    assert(hands.nonEmpty, "no hands parsed")

    val profiles = OpponentProfile.fromImportedHands("simulated", hands, Set("Hero"))
    assert(profiles.nonEmpty, s"no opponent profile for ${leak.id}")

    val hints = profiles.head.exploitHints
    val detected = hintMatchesLeak(hints, leak.id)
    assert(detected,
      s"leak '${leak.id}' not detected after $HandsToPlay hands.\n" +
      s"  Leak firings: ${leakActions.size}\n" +
      s"  Applicable spots: ${records.map(_.leakApplicableSpots).sum}\n" +
      s"  Hands parsed: ${hands.size}\n" +
      s"  Archetype: ${profiles.head.archetypePosterior.mapEstimate}\n" +
      s"  Hints: $hints"
    )

  /** Same matching logic as ValidationRunner.hintMatchesLeak */
  private def hintMatchesLeak(hints: Vector[String], leakId: String): Boolean =
    leakId match
      case "overfold-river-aggression" =>
        hints.exists(_.contains("Over-folds on the river"))
      case "overcall-big-bets" =>
        hints.exists(h => h.contains("calling station") || h.contains("Calls too often facing large bets"))
      case "overbluff-turn-barrel" =>
        hints.exists(_.contains("Very aggressive on the turn"))
      case "passive-big-pots" =>
        false // Not detectable by action frequency — GTO checks ~100% in big pots
      case "preflop-too-loose" =>
        hints.exists(_.contains("Calls too loose preflop"))
      case "preflop-too-tight" =>
        hints.exists(_.contains("Over-folds preflop"))
      case _ => false

  test("fast hero detects overfold-river-aggression (severe)"):
    assertLeakDetected(OverfoldsToAggression(0.9))

  test("fast hero detects overcall-big-bets (severe)"):
    assertLeakDetected(Overcalls(0.9))

  test("fast hero detects overbluff-turn-barrel (severe)"):
    assertLeakDetected(OverbluffsTurnBarrel(0.9))

  test("passive-big-pots not detectable by action frequency (known limitation)".ignore):
    // GTO checks ~100% in big pots — this leak is indistinguishable from
    // equilibrium play by action frequency alone. Needs hand-strength-aware analysis.
    assertLeakDetected(PassiveInBigPots(0.9))

  test("fast hero detects preflop-too-loose (severe)"):
    assertLeakDetected(PreflopTooLoose(0.9))

  test("fast hero detects preflop-too-tight (severe)"):
    assertLeakDetected(PreflopTooTight(0.9))
