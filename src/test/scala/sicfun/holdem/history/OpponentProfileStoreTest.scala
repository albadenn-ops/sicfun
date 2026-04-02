package sicfun.holdem.history

import munit.FunSuite
import sicfun.core.Card
import sicfun.holdem.engine.RaiseResponseCounts
import sicfun.holdem.types.*

import java.nio.file.Files
import scala.jdk.CollectionConverters.*

class OpponentProfileStoreTest extends FunSuite:
  private def card(token: String): Card =
    Card.parse(token).getOrElse(fail(s"invalid card token: $token"))

  private def foldHand(handId: Int, minute: Int): String =
    f"""PokerStars Hand #$handId%d:  Hold'em No Limit ($$0.50/$$1.00 USD) - 2026/03/10 12:$minute%02d:00 ET
       |Table 'Alpha' 2-max Seat #1 is the button
       |Seat 1: Hero ($$100.00 in chips)
       |Seat 2: Villain ($$100.00 in chips)
       |Hero: posts small blind $$0.50
       |Villain: posts big blind $$1.00
       |*** HOLE CARDS ***
       |Dealt to Hero [Ac Kh]
       |Hero: raises $$2.50 to $$3.00
       |Villain: folds
       |Uncalled bet ($$2.00) returned to Hero
       |Hero collected $$1.50 from pot
       |*** SUMMARY ***
       |""".stripMargin

  private def callThenFoldHand(handId: Int, minute: Int): String =
    f"""PokerStars Hand #$handId%d:  Hold'em No Limit ($$0.50/$$1.00 USD) - 2026/03/10 12:$minute%02d:00 ET
       |Table 'Alpha' 2-max Seat #1 is the button
       |Seat 1: Hero ($$100.00 in chips)
       |Seat 2: Villain ($$100.00 in chips)
       |Hero: posts small blind $$0.50
       |Villain: posts big blind $$1.00
       |*** HOLE CARDS ***
       |Dealt to Hero [Ad Qd]
       |Hero: raises $$2.50 to $$3.00
       |Villain: calls $$2.00
       |*** FLOP *** [Ts 9h 8d]
       |Villain: checks
       |Hero: bets $$4.00
       |Villain: folds
       |Uncalled bet ($$4.00) returned to Hero
       |Hero collected $$6.00 from pot
       |*** SUMMARY ***
       |""".stripMargin

  private val importText =
    Vector(
      foldHand(2001, 0),
      foldHand(2002, 1),
      callThenFoldHand(2003, 2),
      foldHand(2004, 3)
    ).mkString("\n")

  test("builds, deduplicates, saves, and reloads opponent profiles") {
    val parsed = HandHistoryImport.parseText(
      importText,
      site = Some(HandHistorySite.PokerStars),
      heroName = Some("Hero")
    )
    assert(parsed.isRight, s"parse failed: $parsed")

    val profiles = OpponentProfile.fromImportedHands(
      site = "pokerstars",
      hands = parsed.toOption.get,
      excludePlayers = Set("Hero")
    )
    assertEquals(profiles.length, 1)

    val villain = profiles.head
    assertEquals(villain.playerName, "Villain")
    assertEquals(villain.handsObserved, 4)
    assertEquals(villain.raiseResponses.folds, 4)
    assertEquals(villain.raiseResponses.calls, 1)
    assertEquals(villain.archetypePosterior.mapEstimate, sicfun.holdem.engine.PlayerArchetype.Nit)
    assert(villain.exploitHints.exists(_.contains("bluff pressure")), s"missing exploit hint: ${villain.exploitHints}")

    val once = OpponentProfileStore.empty.upsertAll(profiles)
    val twice = once.upsertAll(profiles)
    assertEquals(twice.find("pokerstars", "Villain").map(_.handsObserved), Some(4))

    val dir = Files.createTempDirectory("opponent-profile-store-test-")
    val path = dir.resolve("profiles.json")
    try
      OpponentProfileStore.save(path, once)
      val loaded = OpponentProfileStore.load(path)
      val loadedVillain = loaded.find("pokerstars", "Villain").getOrElse(fail("missing villain profile after reload"))
      assertEquals(loadedVillain.handsObserved, 4)
      assertEquals(loadedVillain.raiseResponses, villain.raiseResponses)
      assertEquals(loadedVillain.recentEvents.length, villain.recentEvents.length)
      assertEquals(loadedVillain.exploitHints, villain.exploitHints)
    finally
      val stream = Files.walk(dir)
      try
        stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse.foreach(path => Files.deleteIfExists(path))
      finally
        stream.close()
  }

  test("observeEvent updates an existing profile without double counting") {
    val initial = OpponentProfile(
      site = "pokerstars",
      playerName = "Villain",
      handsObserved = 4,
      firstSeenEpochMillis = 1_700_000_000_000L,
      lastSeenEpochMillis = 1_700_000_000_100L,
      actionSummary = OpponentActionSummary(folds = 2, raises = 1, calls = 1, checks = 1),
      raiseResponses = RaiseResponseCounts(folds = 0, calls = 1, raises = 0),
      recentEvents = Vector.empty,
      seenHandIds = Vector("h1", "h2", "h3", "h4")
    )
    val event = PokerEvent(
      handId = "h5",
      sequenceInHand = 2L,
      playerId = "Villain",
      occurredAtEpochMillis = 1_700_000_000_200L,
      street = Street.Flop,
      position = Position.BigBlind,
      board = Board.from(Seq(card("Ts"), card("9h"), card("8d"))),
      potBefore = 12.0,
      toCall = 4.0,
      stackBefore = 96.0,
      action = PokerAction.Fold,
      decisionTimeMillis = Some(300L),
      betHistory = Vector(BetAction(0, PokerAction.Raise(4.0)))
    )

    val once = OpponentProfileStore(Vector(initial)).observeEvent(
      site = "pokerstars",
      playerName = "Villain",
      event = event,
      facedRaiseResponse = true
    )
    val updated = once.find("pokerstars", "Villain").getOrElse(fail("missing updated profile"))
    assertEquals(updated.handsObserved, 5)
    assertEquals(updated.actionSummary.folds, 3)
    assertEquals(updated.raiseResponses.folds, 1)

    val twice = once.observeEvent(
      site = "pokerstars",
      playerName = "Villain",
      event = event,
      facedRaiseResponse = true
    )
    assertEquals(twice.find("pokerstars", "Villain"), Some(updated))
  }

  test("registerSicfunPlayer derives stable player UIDs and shared behavior profile UIDs") {
    val source = OpponentProfile(
      site = "pokerstars",
      playerName = "Template",
      handsObserved = 12,
      firstSeenEpochMillis = 1_700_000_000_000L,
      lastSeenEpochMillis = 1_700_000_000_500L,
      actionSummary = OpponentActionSummary(folds = 4, raises = 3, calls = 3, checks = 2),
      raiseResponses = RaiseResponseCounts(folds = 1, calls = 2, raises = 1),
      recentEvents = Vector.empty,
      seenHandIds = Vector("t-1", "t-2", "t-3")
    )

    val populated = OpponentProfileStore.empty
      .upsert(source)
      .registerSicfunPlayer("bot-alpha", "model-a", source)
      .registerSicfunPlayer("bot-alpha-alt-alias", "model-a", source)
      .registerSicfunPlayer("bot-beta", "model-b", source)

    val alpha = populated.findPlayer(OpponentIdentity.SicfunLocalSite, "bot-alpha").getOrElse(fail("missing bot-alpha"))
    val alphaAlias = populated.findPlayer(OpponentIdentity.SicfunLocalSite, "bot-alpha-alt-alias").getOrElse(fail("missing bot-alpha alias"))
    val beta = populated.findPlayer(OpponentIdentity.SicfunLocalSite, "bot-beta").getOrElse(fail("missing bot-beta"))

    assertEquals(alpha.playerUid, alphaAlias.playerUid)
    assertEquals(alpha.profileUid, alphaAlias.profileUid)
    assertNotEquals(alpha.playerUid, beta.playerUid)
    assertEquals(alpha.profileUid, beta.profileUid)

    val remembered = populated.find(OpponentIdentity.SicfunLocalSite, "bot-beta").getOrElse(fail("missing remembered bot-beta profile"))
    assertEquals(remembered.site, OpponentIdentity.SicfunLocalSite)
    assert(remembered.behaviorUid.nonEmpty)
    assertEquals(remembered.profileUid, Some(beta.profileUid))
  }

  private def showdownHand(handId: Int, minute: Int): String =
    f"""PokerStars Hand #$handId%d:  Hold'em No Limit ($$0.50/$$1.00 USD) - 2026/03/10 12:$minute%02d:00 ET
       |Table 'Alpha' 2-max Seat #1 is the button
       |Seat 1: Hero ($$100.00 in chips)
       |Seat 2: Villain ($$100.00 in chips)
       |Hero: posts small blind $$0.50
       |Villain: posts big blind $$1.00
       |*** HOLE CARDS ***
       |Dealt to Hero [Ac Kh]
       |Hero: raises $$2.50 to $$3.00
       |Villain: calls $$2.00
       |*** FLOP *** [Ts 9h 8d]
       |Hero: bets $$4.00
       |Villain: calls $$4.00
       |*** TURN *** [Ts 9h 8d] [2c]
       |Hero: checks
       |Villain: checks
       |*** RIVER *** [Ts 9h 8d 2c] [3s]
       |Hero: checks
       |Villain: checks
       |*** SHOW DOWN ***
       |Villain: shows [Qh Qs] (a pair of Queens)
       |Hero: shows [Ac Kh] (high card Ace)
       |Villain collected $$14.00 from pot
       |*** SUMMARY ***
       |""".stripMargin

  test("OpponentProfile tracks showdown-revealed hands") {
    val text = showdownHand(3001, 10)
    val parsed = HandHistoryImport.parseText(
      text,
      site = Some(HandHistorySite.PokerStars),
      heroName = Some("Hero")
    )
    assert(parsed.isRight, s"parse failed: $parsed")
    val hands = parsed.toOption.get
    assert(hands.head.showdownCards.nonEmpty, s"showdownCards should be populated by parser: ${hands.head.showdownCards}")

    val profiles = OpponentProfile.fromImportedHands("pokerstars", hands, Set("Hero"))
    assert(profiles.nonEmpty, "should have villain profile")
    val villainProfile = profiles.find(_.playerName == "Villain").get
    assert(villainProfile.showdownHands.nonEmpty, "should have showdown records")
    assertEquals(villainProfile.showdownHands.head.handId, "3001")
    assertEquals(
      villainProfile.showdownHands.head.cards,
      HoleCards.from(Seq(card("Qh"), card("Qs")))
    )
  }

  test("showdown hands survive merge") {
    val text = Vector(showdownHand(3001, 10), showdownHand(3002, 11)).mkString("\n")
    val parsed = HandHistoryImport.parseText(
      text,
      site = Some(HandHistorySite.PokerStars),
      heroName = Some("Hero")
    )
    assert(parsed.isRight, s"parse failed: $parsed")
    val hands = parsed.toOption.get

    val profilesA = OpponentProfile.fromImportedHands("pokerstars", hands.take(1), Set("Hero"))
    val profilesB = OpponentProfile.fromImportedHands("pokerstars", hands.drop(1), Set("Hero"))
    val merged = OpponentProfile.merge(profilesA.head, profilesB.head)
    assertEquals(merged.showdownHands.length, 2)
    assertEquals(merged.showdownHands.map(_.handId).toSet, Set("3001", "3002"))
  }

  test("showdown hands survive JSON roundtrip") {
    val text = showdownHand(3001, 10)
    val parsed = HandHistoryImport.parseText(
      text,
      site = Some(HandHistorySite.PokerStars),
      heroName = Some("Hero")
    )
    assert(parsed.isRight, s"parse failed: $parsed")
    val profiles = OpponentProfile.fromImportedHands("pokerstars", parsed.toOption.get, Set("Hero"))
    val store = OpponentProfileStore.empty.upsertAll(profiles)

    val dir = Files.createTempDirectory("showdown-roundtrip-")
    val path = dir.resolve("profiles.json")
    try
      OpponentProfileStore.save(path, store)
      val loaded = OpponentProfileStore.load(path)
      val loadedVillain = loaded.find("pokerstars", "Villain").getOrElse(fail("missing villain after reload"))
      assertEquals(loadedVillain.showdownHands.length, 1)
      assertEquals(loadedVillain.showdownHands.head.handId, "3001")
      assertEquals(
        loadedVillain.showdownHands.head.cards,
        HoleCards.from(Seq(card("Qh"), card("Qs")))
      )
    finally
      val stream = Files.walk(dir)
      try
        stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse.foreach(path => Files.deleteIfExists(path))
      finally
        stream.close()
  }

  test("exploit hints include showdown-derived premium hand reads") {
    val profile = OpponentProfile(
      site = "pokerstars",
      playerName = "Villain",
      handsObserved = 5,
      firstSeenEpochMillis = 1_700_000_000_000L,
      lastSeenEpochMillis = 1_700_000_000_500L,
      actionSummary = OpponentActionSummary(),
      raiseResponses = RaiseResponseCounts(),
      recentEvents = Vector.empty,
      seenHandIds = Vector("sd-1", "sd-2", "sd-3", "sd-4", "sd-5"),
      showdownHands = Vector(
        ShowdownRecord("sd-1", HoleCards.from(Vector(card("Ah"), card("As")))),
        ShowdownRecord("sd-2", HoleCards.from(Vector(card("Kh"), card("Ks")))),
        ShowdownRecord("sd-3", HoleCards.from(Vector(card("Qh"), card("Qs")))),
        ShowdownRecord("sd-4", HoleCards.from(Vector(card("Ac"), card("Kd")))),
        ShowdownRecord("sd-5", HoleCards.from(Vector(card("Jh"), card("Th"))))
      )
    )

    val hints = profile.exploitHints
    val hintDetails = profile.exploitHintDetails
    assert(
      hints.exists(_.contains("premium hands frequently")),
      s"expected premium showdown hint, got: $hints"
    )
    val premiumHint = hintDetails.find(_.ruleId == "showdown-premium-heavy").getOrElse(fail(s"missing structured premium hint: $hintDetails"))
    assertEquals(premiumHint.metrics.length, 4)
    assert(premiumHint.deviationRatioFromGto > 0.0, s"expected positive deviation metric: ${premiumHint.metrics}")
  }

  test("exploit hints include showdown-derived weak hand reads") {
    val profile = OpponentProfile(
      site = "pokerstars",
      playerName = "Villain",
      handsObserved = 5,
      firstSeenEpochMillis = 1_700_000_000_000L,
      lastSeenEpochMillis = 1_700_000_000_500L,
      actionSummary = OpponentActionSummary(calls = 6, raises = 1, checks = 3),
      raiseResponses = RaiseResponseCounts(calls = 3, folds = 1, raises = 0),
      recentEvents = Vector.empty,
      seenHandIds = Vector("sd-w1", "sd-w2", "sd-w3", "sd-w4", "sd-w5"),
      showdownHands = Vector(
        ShowdownRecord("sd-w1", HoleCards.from(Vector(card("7c"), card("2d")))),
        ShowdownRecord("sd-w2", HoleCards.from(Vector(card("6c"), card("3d")))),
        ShowdownRecord("sd-w3", HoleCards.from(Vector(card("8c"), card("4d")))),
        ShowdownRecord("sd-w4", HoleCards.from(Vector(card("9c"), card("5d")))),
        ShowdownRecord("sd-w5", HoleCards.from(Vector(card("Kh"), card("Qs"))))
      )
    )

    val hints = profile.exploitHints
    val hintDetails = profile.exploitHintDetails
    assert(
      hints.exists(_.contains("shown down weak hands")),
      s"expected weak showdown hint, got: $hints"
    )
    val weakHint = hintDetails.find(_.ruleId == "showdown-weak-heavy").getOrElse(fail(s"missing structured weak hint: $hintDetails"))
    assertEquals(weakHint.metrics.length, 4)
    assert(weakHint.leakDetectionConfidence > 0.0, s"expected positive confidence metric: ${weakHint.metrics}")
  }

  test("manual player collapse can also collapse profiles and survives JSON roundtrip") {
    val canonical = OpponentProfile(
      site = "pokerstars",
      playerName = "VillainA",
      handsObserved = 20,
      firstSeenEpochMillis = 1_700_000_000_000L,
      lastSeenEpochMillis = 1_700_000_000_500L,
      actionSummary = OpponentActionSummary(folds = 10, raises = 2, calls = 6, checks = 2),
      raiseResponses = RaiseResponseCounts(folds = 4, calls = 1, raises = 0),
      recentEvents = Vector.empty,
      seenHandIds = Vector("a-1", "a-2", "a-3")
    )
    val alias = OpponentProfile(
      site = "gg",
      playerName = "VillainB",
      handsObserved = 15,
      firstSeenEpochMillis = 1_700_000_001_000L,
      lastSeenEpochMillis = 1_700_000_001_500L,
      actionSummary = OpponentActionSummary(folds = 2, raises = 7, calls = 4, checks = 2),
      raiseResponses = RaiseResponseCounts(folds = 0, calls = 2, raises = 3),
      recentEvents = Vector.empty,
      seenHandIds = Vector("b-1", "b-2")
    )

    val collapsed = OpponentProfileStore.empty
      .upsert(canonical)
      .upsert(alias)
      .collapsePlayers(
        canonicalSite = "pokerstars",
        canonicalName = "VillainA",
        aliasSite = "gg",
        aliasName = "VillainB",
        collapseProfiles = true,
        assertedAtEpochMillis = 123L
      )

    val canonicalPlayer = collapsed.findPlayer("pokerstars", "VillainA").getOrElse(fail("missing canonical player"))
    val aliasPlayer = collapsed.findPlayer("gg", "VillainB").getOrElse(fail("missing alias player"))
    val aliasProfile = collapsed.find("gg", "VillainB").getOrElse(fail("missing collapsed alias profile"))

    assertEquals(aliasPlayer.playerUid, canonicalPlayer.playerUid)
    assertEquals(aliasProfile.profileUid, Some(canonicalPlayer.profileUid))
    assert(collapsed.playerCollapses.nonEmpty)
    assert(collapsed.profileCollapses.nonEmpty)

    val dir = Files.createTempDirectory("opponent-profile-collapse-roundtrip-")
    val path = dir.resolve("profiles.json")
    try
      OpponentProfileStore.save(path, collapsed)
      val reloaded = OpponentProfileStore.load(path)
      val reloadedAlias = reloaded.find("gg", "VillainB").getOrElse(fail("missing reloaded alias profile"))
      val reloadedAliasPlayer = reloaded.findPlayer("gg", "VillainB").getOrElse(fail("missing reloaded alias player"))
      assertEquals(reloadedAliasPlayer.playerUid, canonicalPlayer.playerUid)
      assertEquals(reloadedAlias.profileUid, Some(canonicalPlayer.profileUid))
      assertEquals(reloaded.playerCollapses.length, 1)
      assertEquals(reloaded.profileCollapses.length, 1)
    finally
      val stream = Files.walk(dir)
      try
        stream.iterator().asScala.toVector.sortBy(_.toString.length).reverse.foreach(path => Files.deleteIfExists(path))
      finally
        stream.close()
  }
end OpponentProfileStoreTest
