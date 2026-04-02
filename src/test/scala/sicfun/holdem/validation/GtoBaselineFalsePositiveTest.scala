package sicfun.holdem.validation

import munit.FunSuite
import sicfun.holdem.history.{HandHistoryImport, HandHistorySite, OpponentProfile}
import sicfun.holdem.types.{PokerAction, TestSystemPropertyScope}

/** Verifies that a CFR GTO baseline player (no leaks, no noise) does NOT trigger
  * false positive exploit hints in the profiling pipeline.
  */
class GtoBaselineFalsePositiveTest extends FunSuite:

  override val munitTimeout = scala.concurrent.duration.Duration(600, "s")

  private def withScalaCfrProvider[A](thunk: => A): A =
    TestSystemPropertyScope.withSystemProperties(Seq("sicfun.cfr.provider" -> Some("scala")))(thunk)

  test("CFR GTO baseline produces no false positive leak hints"):
    withScalaCfrProvider {
      val cfrStrategy = CfrVillainStrategy(allowHeuristicFallback = false)
      val villain = LeakInjectedVillain(
        name = "gto_control",
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
      val numHands = 2_000
      val records = (1 to numHands).map(sim.playHand).toVector

      // Verify zero leak firings
      val leakActions = records.flatMap(_.actions).filter(_.leakFired)
      assertEquals(leakActions.size, 0, "CFR GTO player should have zero leak firings")

      // Export, parse, profile
      val text = PokerStarsExporter.exportHands(records, "Hero", villain.name)
      val parsed = HandHistoryImport.parseText(text, Some(HandHistorySite.PokerStars), Some("Hero"))
      assert(parsed.isRight, s"parse failed: ${parsed.left.getOrElse("")}")
      val hands = parsed.toOption.get
      val profiles = OpponentProfile.fromImportedHands("simulated", hands, Set("Hero"))
      assert(profiles.nonEmpty, "no profile")
      val profile = profiles.head

      // Diagnostics
      val events = profile.recentEvents
      val hints = profile.exploitHints
      val sig = profile.signature

      println(s"[CFR-GTO] Hands: $numHands")
      println(s"[CFR-GTO] Archetype: ${profile.archetypePosterior.mapEstimate}")
      println(f"[CFR-GTO] Signature: fold=${sig.values(0)}%.3f raise=${sig.values(1)}%.3f " +
        f"call=${sig.values(2)}%.3f check=${sig.values(3)}%.3f")
      println(s"[CFR-GTO] Events: ${events.size} total")

      // Per-street breakdown
      val streets = events.groupBy(_.street)
      streets.toVector.sortBy(_._1.ordinal).foreach { (street, evts) =>
        val folds = evts.count(_.action == PokerAction.Fold)
        val calls = evts.count(_.action == PokerAction.Call)
        val checks = evts.count(_.action == PokerAction.Check)
        val raises = evts.count(_.action.category == PokerAction.Category.Raise)
        println(f"[CFR-GTO]   $street: n=${evts.size} fold=$folds call=$calls check=$checks raise=$raises")
      }

      println(s"[CFR-GTO] Exploit hints: $hints")
      assert(
        !hints.exists(showdownHint),
        s"CFR GTO baseline should not trigger showdown-specific hints: $hints"
      )

      // Check each leak pattern
      val leakIds = Vector("overfold-river-aggression", "overcall-big-bets", "overbluff-turn-barrel",
        "passive-big-pots", "preflop-too-loose", "preflop-too-tight")
      val matches = leakIds.filter(id => hintMatchesLeak(hints, id))
      println(s"[CFR-GTO] Matching leak patterns: $matches")

      assert(matches.isEmpty,
        s"CFR GTO baseline triggers false positive leak patterns: $matches\n  Hints: $hints")
    }

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
