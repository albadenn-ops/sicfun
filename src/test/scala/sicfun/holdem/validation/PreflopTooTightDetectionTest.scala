package sicfun.holdem.validation

import munit.FunSuite
import sicfun.holdem.history.{HandHistoryImport, HandHistorySite, OpponentProfile}
import sicfun.holdem.types.PokerAction

/** Focused test for preflop-too-tight detection at mild and moderate severity.
  * These were NOT DETECTED in the 18-player validation run despite
  * 1,537+ leak firings. Diagnoses why the profiler misses them.
  */
class PreflopTooTightDetectionTest extends FunSuite:

  override val munitTimeout = scala.concurrent.duration.Duration(120, "s")

  private val HandsToPlay = 10_000

  private def runAndDiagnose(leak: InjectedLeak, label: String): Unit =
    val villain = LeakInjectedVillain(
      name = s"test_${leak.id}_$label",
      leaks = Vector(leak),
      baselineNoise = 0.03,
      seed = 42L
    )
    val sim = new HeadsUpSimulator(
      heroEngine = None,
      villain = villain,
      seed = 42L
    )
    val records = (1 to HandsToPlay).map(sim.playHand).toVector

    val leakActions = records.flatMap(_.actions).filter(_.leakId.contains(leak.id))
    val applicableSpots = records.map(_.leakApplicableSpots).sum
    println(s"[$label] Leak firings: ${leakActions.size} / $applicableSpots applicable")

    // Look at raw villain actions to measure actual preflop fold rate
    val allVillainActions = records.flatMap(_.actions).filter(_.player != "Hero")
    val villainPreflopActions = allVillainActions.filter(_.street == sicfun.holdem.types.Street.Preflop)
    val villainPreflopFolds = villainPreflopActions.count(_.action == PokerAction.Fold)
    val rawPreflopFoldRate = if villainPreflopActions.nonEmpty then
      villainPreflopFolds.toDouble / villainPreflopActions.size else 0.0
    println(f"[$label] Raw preflop fold rate: $rawPreflopFoldRate%.3f " +
      f"($villainPreflopFolds/${villainPreflopActions.size})")

    // Export, parse, profile
    val text = PokerStarsExporter.exportHands(records, "Hero", villain.name)
    val parsed = HandHistoryImport.parseText(text, Some(HandHistorySite.PokerStars), Some("Hero"))
    assert(parsed.isRight, s"parse failed: ${parsed.left.getOrElse("")}")
    val hands = parsed.toOption.get

    val profiles = OpponentProfile.fromImportedHands("simulated", hands, Set("Hero"))
    assert(profiles.nonEmpty, "no profile")
    val profile = profiles.head

    // Diagnose: what does the profiler actually see?
    val events = profile.recentEvents
    val preflopEvents = events.filter(_.street == sicfun.holdem.types.Street.Preflop)
    val preflopFolds = preflopEvents.count(_.action == PokerAction.Fold)
    val preflopFoldRate = if preflopEvents.nonEmpty then
      preflopFolds.toDouble / preflopEvents.size else 0.0

    println(f"[$label] Profile recentEvents: ${events.size} total, ${preflopEvents.size} preflop")
    println(f"[$label] Profile preflop fold rate: $preflopFoldRate%.3f ($preflopFolds/${preflopEvents.size})")
    println(f"[$label] Archetype: ${profile.archetypePosterior.mapEstimate}")

    val sig = profile.signature
    println(f"[$label] Signature fold=${sig.values(0)}%.3f raise=${sig.values(1)}%.3f " +
      f"call=${sig.values(2)}%.3f check=${sig.values(3)}%.3f")

    val hints = profile.exploitHints
    println(s"[$label] Exploit hints: $hints")

    val detected = hints.exists(_.contains("Over-folds preflop"))
    assert(detected,
      s"leak '${leak.id}' ($label) not detected.\n" +
      s"  Preflop fold rate: $preflopFoldRate (threshold: 0.20)\n" +
      s"  Signature foldRate: ${sig.values(0)} (threshold: 0.45)\n" +
      s"  Hints: $hints"
    )

  test("detects preflop-too-tight at mild severity (0.3)"):
    runAndDiagnose(PreflopTooTight(0.3), "mild")

  test("detects preflop-too-tight at moderate severity (0.6)"):
    runAndDiagnose(PreflopTooTight(0.6), "moderate")
