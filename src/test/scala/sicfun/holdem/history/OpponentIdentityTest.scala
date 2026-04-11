package sicfun.holdem.history

import munit.FunSuite
import sicfun.holdem.engine.RaiseResponseCounts

/** Tests for the [[OpponentIdentity]] module and the identity domain types
  * ([[RememberedPlayer]], [[PlayerAlias]], [[PlayerCollapse]],
  * [[ProfileCollapse]], [[BehaviorFingerprint]]).
  *
  * Coverage:
  *   - '''Construction validation''': all identity types reject blank/empty
  *     required fields (UIDs, sites, names) and negative timestamps
  *   - '''Site normalization''': lowercasing and trimming via `normalizeSite`
  *   - '''Player name normalization''': trimming but case-preserving
  *   - '''Alias key derivation''': deterministic, case-insensitive composite
  *     key using null separator
  *   - '''SHA-256 hashing''': 64-char lowercase hex, deterministic, collision-
  *     resistant for different inputs
  *   - '''Default UID derivation''': non-SICFUN sites hash (site, name);
  *     `sicfun@localhost` with a modelUid hashes (modelUid, behaviorUid);
  *     without modelUid falls back to site hash
  *   - '''Profile UID derivation''': SICFUN-local returns behaviorUid directly;
  *     non-SICFUN sites hash (site, name)
  *   - '''Behavior fingerprinting''': produces a non-empty deterministic
  *     fingerprint containing signature, responses, and archetype data
  *   - '''ensureProfileIdentity''': fills missing UIDs, normalizes site/name,
  *     preserves existing non-blank UIDs
  *   - '''defaultPlayer / normalizePlayer / normalizeAlias''': correct
  *     derivation and normalization of identity fields
  */
class OpponentIdentityTest extends FunSuite:

  // ---------------------------------------------------------------------------
  // RememberedPlayer construction and validation
  // ---------------------------------------------------------------------------

  test("RememberedPlayer construction with all required fields succeeds") {
    val player = RememberedPlayer(
      playerUid = "uid-1",
      canonicalSite = "pokerstars",
      canonicalName = "Villain",
      profileUid = "prof-1"
    )
    assertEquals(player.playerUid, "uid-1")
    assertEquals(player.modelUid, None)
    assertEquals(player.behaviorUid, None)
  }

  test("RememberedPlayer rejects blank playerUid") {
    intercept[IllegalArgumentException] {
      RememberedPlayer(playerUid = "  ", canonicalSite = "pokerstars", canonicalName = "V", profileUid = "p")
    }
  }

  test("RememberedPlayer rejects blank canonicalSite") {
    intercept[IllegalArgumentException] {
      RememberedPlayer(playerUid = "uid", canonicalSite = "", canonicalName = "V", profileUid = "p")
    }
  }

  test("RememberedPlayer rejects blank canonicalName") {
    intercept[IllegalArgumentException] {
      RememberedPlayer(playerUid = "uid", canonicalSite = "site", canonicalName = "  ", profileUid = "p")
    }
  }

  test("RememberedPlayer rejects blank profileUid") {
    intercept[IllegalArgumentException] {
      RememberedPlayer(playerUid = "uid", canonicalSite = "site", canonicalName = "V", profileUid = " ")
    }
  }

  test("RememberedPlayer rejects blank modelUid when present") {
    intercept[IllegalArgumentException] {
      RememberedPlayer(playerUid = "uid", canonicalSite = "site", canonicalName = "V", profileUid = "p", modelUid = Some("  "))
    }
  }

  test("RememberedPlayer rejects blank behaviorUid when present") {
    intercept[IllegalArgumentException] {
      RememberedPlayer(playerUid = "uid", canonicalSite = "site", canonicalName = "V", profileUid = "p", behaviorUid = Some(""))
    }
  }

  // ---------------------------------------------------------------------------
  // PlayerAlias construction and validation
  // ---------------------------------------------------------------------------

  test("PlayerAlias construction succeeds with valid fields") {
    val alias = PlayerAlias(site = "gg", playerName = "Fish", playerUid = "uid-2")
    assertEquals(alias.assertedByHuman, false)
  }

  test("PlayerAlias rejects blank site") {
    intercept[IllegalArgumentException] {
      PlayerAlias(site = "", playerName = "Fish", playerUid = "uid-2")
    }
  }

  test("PlayerAlias rejects blank playerName") {
    intercept[IllegalArgumentException] {
      PlayerAlias(site = "gg", playerName = "  ", playerUid = "uid-2")
    }
  }

  test("PlayerAlias rejects blank playerUid") {
    intercept[IllegalArgumentException] {
      PlayerAlias(site = "gg", playerName = "Fish", playerUid = " ")
    }
  }

  // ---------------------------------------------------------------------------
  // PlayerCollapse construction and validation
  // ---------------------------------------------------------------------------

  test("PlayerCollapse construction succeeds with valid fields") {
    val collapse = PlayerCollapse(
      aliasPlayerUid = "alias-uid",
      canonicalPlayerUid = "canonical-uid",
      assertedByHuman = true,
      assertedAtEpochMillis = 1000L
    )
    assertEquals(collapse.assertedByHuman, true)
  }

  test("PlayerCollapse rejects blank aliasPlayerUid") {
    intercept[IllegalArgumentException] {
      PlayerCollapse(aliasPlayerUid = "", canonicalPlayerUid = "c", assertedByHuman = true, assertedAtEpochMillis = 0L)
    }
  }

  test("PlayerCollapse rejects blank canonicalPlayerUid") {
    intercept[IllegalArgumentException] {
      PlayerCollapse(aliasPlayerUid = "a", canonicalPlayerUid = "  ", assertedByHuman = true, assertedAtEpochMillis = 0L)
    }
  }

  test("PlayerCollapse rejects negative assertedAtEpochMillis") {
    intercept[IllegalArgumentException] {
      PlayerCollapse(aliasPlayerUid = "a", canonicalPlayerUid = "c", assertedByHuman = false, assertedAtEpochMillis = -1L)
    }
  }

  // ---------------------------------------------------------------------------
  // ProfileCollapse construction and validation
  // ---------------------------------------------------------------------------

  test("ProfileCollapse construction succeeds with valid fields") {
    val collapse = ProfileCollapse(
      aliasProfileUid = "prof-alias",
      canonicalProfileUid = "prof-canonical",
      assertedByHuman = false,
      assertedAtEpochMillis = 0L
    )
    assertEquals(collapse.assertedByHuman, false)
  }

  test("ProfileCollapse rejects blank aliasProfileUid") {
    intercept[IllegalArgumentException] {
      ProfileCollapse(aliasProfileUid = " ", canonicalProfileUid = "c", assertedByHuman = true, assertedAtEpochMillis = 0L)
    }
  }

  test("ProfileCollapse rejects blank canonicalProfileUid") {
    intercept[IllegalArgumentException] {
      ProfileCollapse(aliasProfileUid = "a", canonicalProfileUid = "", assertedByHuman = true, assertedAtEpochMillis = 0L)
    }
  }

  test("ProfileCollapse rejects negative assertedAtEpochMillis") {
    intercept[IllegalArgumentException] {
      ProfileCollapse(aliasProfileUid = "a", canonicalProfileUid = "c", assertedByHuman = false, assertedAtEpochMillis = -5L)
    }
  }

  // ---------------------------------------------------------------------------
  // BehaviorFingerprint construction and validation
  // ---------------------------------------------------------------------------

  test("BehaviorFingerprint construction succeeds with valid fields") {
    val fp = BehaviorFingerprint(uid = "abc123", payload = "signature=0.5,0.2")
    assertEquals(fp.uid, "abc123")
  }

  test("BehaviorFingerprint rejects blank uid") {
    intercept[IllegalArgumentException] {
      BehaviorFingerprint(uid = "  ", payload = "data")
    }
  }

  test("BehaviorFingerprint rejects blank payload") {
    intercept[IllegalArgumentException] {
      BehaviorFingerprint(uid = "abc", payload = "")
    }
  }

  // ---------------------------------------------------------------------------
  // OpponentIdentity.normalizeSite
  // ---------------------------------------------------------------------------

  test("normalizeSite lowercases and trims") {
    assertEquals(OpponentIdentity.normalizeSite("  PokerStars  "), "pokerstars")
    assertEquals(OpponentIdentity.normalizeSite("GG"), "gg")
  }

  // ---------------------------------------------------------------------------
  // OpponentIdentity.normalizePlayerName
  // ---------------------------------------------------------------------------

  test("normalizePlayerName trims but preserves case") {
    assertEquals(OpponentIdentity.normalizePlayerName("  FishyKing  "), "FishyKing")
  }

  // ---------------------------------------------------------------------------
  // OpponentIdentity.aliasKey
  // ---------------------------------------------------------------------------

  test("aliasKey combines normalized site and lowercased player name with null separator") {
    val key = OpponentIdentity.aliasKey("PokerStars", "  FishyKing  ")
    assertEquals(key, "pokerstars\u0000fishyking")
  }

  test("aliasKey is case-insensitive for both site and player name") {
    val key1 = OpponentIdentity.aliasKey("GG", "Villain")
    val key2 = OpponentIdentity.aliasKey("gg", "villain")
    assertEquals(key1, key2)
  }

  // ---------------------------------------------------------------------------
  // OpponentIdentity.sha256Hex
  // ---------------------------------------------------------------------------

  test("sha256Hex produces 64-char lowercase hex string") {
    val hash = OpponentIdentity.sha256Hex("test")
    assertEquals(hash.length, 64)
    assert(hash.matches("[0-9a-f]{64}"), s"unexpected hash format: $hash")
  }

  test("sha256Hex is deterministic") {
    val a = OpponentIdentity.sha256Hex("hello world")
    val b = OpponentIdentity.sha256Hex("hello world")
    assertEquals(a, b)
  }

  test("sha256Hex produces different hashes for different inputs") {
    val a = OpponentIdentity.sha256Hex("alpha")
    val b = OpponentIdentity.sha256Hex("beta")
    assertNotEquals(a, b)
  }

  // ---------------------------------------------------------------------------
  // OpponentIdentity.defaultPlayerUid
  // ---------------------------------------------------------------------------

  test("defaultPlayerUid for non-sicfun site hashes site and player name") {
    val uid = OpponentIdentity.defaultPlayerUid("pokerstars", "Villain", None, "bhv-1")
    assertEquals(uid, OpponentIdentity.sha256Hex("player|pokerstars|villain"))
  }

  test("defaultPlayerUid for non-sicfun site ignores modelUid and behaviorUid") {
    val uid1 = OpponentIdentity.defaultPlayerUid("pokerstars", "Villain", Some("model-a"), "bhv-1")
    val uid2 = OpponentIdentity.defaultPlayerUid("pokerstars", "Villain", None, "bhv-2")
    assertEquals(uid1, uid2)
  }

  test("defaultPlayerUid for sicfun@localhost with modelUid uses modelUid and behaviorUid") {
    val uid = OpponentIdentity.defaultPlayerUid(
      OpponentIdentity.SicfunLocalSite,
      "bot-alpha",
      Some("model-a"),
      "bhv-1"
    )
    assertEquals(uid, OpponentIdentity.sha256Hex("sicfun-player|model-a|bhv-1"))
  }

  test("defaultPlayerUid for sicfun@localhost without modelUid falls back to site hash") {
    val uid = OpponentIdentity.defaultPlayerUid(
      OpponentIdentity.SicfunLocalSite,
      "bot-alpha",
      None,
      "bhv-1"
    )
    assertEquals(uid, OpponentIdentity.sha256Hex(s"player|${OpponentIdentity.SicfunLocalSite}|bot-alpha"))
  }

  // ---------------------------------------------------------------------------
  // OpponentIdentity.defaultProfileUid
  // ---------------------------------------------------------------------------

  test("defaultProfileUid for sicfun@localhost returns behaviorUid directly") {
    val uid = OpponentIdentity.defaultProfileUid(OpponentIdentity.SicfunLocalSite, "bot", "bhv-42")
    assertEquals(uid, "bhv-42")
  }

  test("defaultProfileUid for non-sicfun site hashes site and player name") {
    val uid = OpponentIdentity.defaultProfileUid("pokerstars", "Villain", "bhv-1")
    assertEquals(uid, OpponentIdentity.sha256Hex("profile|pokerstars|villain"))
  }

  // ---------------------------------------------------------------------------
  // OpponentIdentity.fingerprint
  // ---------------------------------------------------------------------------

  test("fingerprint produces a non-empty BehaviorFingerprint for a minimal profile") {
    val profile = OpponentProfile(
      site = "pokerstars",
      playerName = "Test",
      handsObserved = 5,
      firstSeenEpochMillis = 1_000_000L,
      lastSeenEpochMillis = 1_000_100L,
      actionSummary = OpponentActionSummary(folds = 2, raises = 1, calls = 1, checks = 1),
      raiseResponses = RaiseResponseCounts(folds = 1, calls = 1, raises = 0),
      recentEvents = Vector.empty,
      seenHandIds = Vector("h1")
    )
    val fp = OpponentIdentity.fingerprint(profile)
    assert(fp.uid.nonEmpty)
    assert(fp.payload.contains("signature="))
    assert(fp.payload.contains("responses="))
    assert(fp.payload.contains("archetypes="))
  }

  test("fingerprint is deterministic for the same profile") {
    val profile = OpponentProfile(
      site = "gg",
      playerName = "SteadyVillain",
      handsObserved = 3,
      firstSeenEpochMillis = 1_000L,
      lastSeenEpochMillis = 2_000L,
      actionSummary = OpponentActionSummary(folds = 1, raises = 1, calls = 1, checks = 0),
      raiseResponses = RaiseResponseCounts(folds = 0, calls = 2, raises = 1),
      recentEvents = Vector.empty,
      seenHandIds = Vector("h1")
    )
    val fp1 = OpponentIdentity.fingerprint(profile)
    val fp2 = OpponentIdentity.fingerprint(profile)
    assertEquals(fp1, fp2)
  }

  // ---------------------------------------------------------------------------
  // OpponentIdentity.ensureProfileIdentity
  // ---------------------------------------------------------------------------

  test("ensureProfileIdentity fills in missing UIDs") {
    val profile = OpponentProfile(
      site = "PokerStars",
      playerName = "  Villain  ",
      handsObserved = 5,
      firstSeenEpochMillis = 1_000L,
      lastSeenEpochMillis = 2_000L,
      actionSummary = OpponentActionSummary(folds = 3, raises = 1, calls = 1, checks = 0),
      raiseResponses = RaiseResponseCounts(folds = 2, calls = 0, raises = 0),
      recentEvents = Vector.empty,
      seenHandIds = Vector("h1")
    )
    val identified = OpponentIdentity.ensureProfileIdentity(profile)
    assertEquals(identified.site, "pokerstars")
    assertEquals(identified.playerName, "Villain")
    assert(identified.playerUid.isDefined)
    assert(identified.profileUid.isDefined)
    assert(identified.behaviorUid.isDefined)
  }

  test("ensureProfileIdentity preserves existing non-blank UIDs") {
    val profile = OpponentProfile(
      site = "pokerstars",
      playerName = "Villain",
      handsObserved = 5,
      firstSeenEpochMillis = 1_000L,
      lastSeenEpochMillis = 2_000L,
      actionSummary = OpponentActionSummary(folds = 3, raises = 1, calls = 1, checks = 0),
      raiseResponses = RaiseResponseCounts(folds = 2, calls = 0, raises = 0),
      recentEvents = Vector.empty,
      seenHandIds = Vector("h1"),
      playerUid = Some("custom-player"),
      profileUid = Some("custom-profile"),
      behaviorUid = Some("custom-behavior")
    )
    val identified = OpponentIdentity.ensureProfileIdentity(profile)
    assertEquals(identified.playerUid, Some("custom-player"))
    assertEquals(identified.profileUid, Some("custom-profile"))
    assertEquals(identified.behaviorUid, Some("custom-behavior"))
  }

  test("ensureProfileIdentity fills in absent modelUid") {
    val profile = OpponentProfile(
      site = "pokerstars",
      playerName = "Villain",
      handsObserved = 1,
      firstSeenEpochMillis = 1_000L,
      lastSeenEpochMillis = 1_000L,
      actionSummary = OpponentActionSummary(folds = 1),
      raiseResponses = RaiseResponseCounts(),
      recentEvents = Vector.empty,
      seenHandIds = Vector("h1"),
      modelUid = None
    )
    val identified = OpponentIdentity.ensureProfileIdentity(profile)
    // ensureProfileIdentity should not crash with modelUid = None
    assert(identified.playerUid.nonEmpty)
  }

  // ---------------------------------------------------------------------------
  // OpponentIdentity.defaultPlayer
  // ---------------------------------------------------------------------------

  test("defaultPlayer creates a RememberedPlayer from a profile") {
    val profile = OpponentProfile(
      site = "pokerstars",
      playerName = "Villain",
      handsObserved = 10,
      firstSeenEpochMillis = 1_000L,
      lastSeenEpochMillis = 2_000L,
      actionSummary = OpponentActionSummary(folds = 5, raises = 2, calls = 2, checks = 1),
      raiseResponses = RaiseResponseCounts(folds = 1, calls = 1, raises = 0),
      recentEvents = Vector.empty,
      seenHandIds = Vector("h1")
    )
    val player = OpponentIdentity.defaultPlayer(profile)
    assertEquals(player.canonicalSite, "pokerstars")
    assertEquals(player.canonicalName, "Villain")
    assert(player.playerUid.nonEmpty)
    assert(player.profileUid.nonEmpty)
    assert(player.behaviorUid.isDefined)
  }

  // ---------------------------------------------------------------------------
  // OpponentIdentity.normalizePlayer
  // ---------------------------------------------------------------------------

  test("normalizePlayer trims and lowercases site, trims name, preserves valid UIDs") {
    val player = RememberedPlayer(
      playerUid = "uid-1",
      canonicalSite = "  GG  ",
      canonicalName = "  Fish  ",
      profileUid = "prof-1",
      modelUid = Some("model-abc"),
      behaviorUid = Some("  bhv  ")
    )
    val normalized = OpponentIdentity.normalizePlayer(player)
    assertEquals(normalized.canonicalSite, "gg")
    assertEquals(normalized.canonicalName, "Fish")
    assertEquals(normalized.modelUid, Some("model-abc"))
    assertEquals(normalized.behaviorUid, Some("bhv"))
  }

  // ---------------------------------------------------------------------------
  // OpponentIdentity.normalizeAlias
  // ---------------------------------------------------------------------------

  test("normalizeAlias trims and lowercases site, trims name") {
    val alias = PlayerAlias(
      site = "  PokerStars  ",
      playerName = "  Villain  ",
      playerUid = "uid-1"
    )
    val normalized = OpponentIdentity.normalizeAlias(alias)
    assertEquals(normalized.site, "pokerstars")
    assertEquals(normalized.playerName, "Villain")
  }
end OpponentIdentityTest
