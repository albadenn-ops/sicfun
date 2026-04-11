package sicfun.holdem.history

import sicfun.holdem.engine.PlayerArchetype

import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.Locale

/** Canonical remembered-player identity record used across aliases/profiles.
  *
  * A RememberedPlayer is the top-level entity in the identity system: it represents
  * a single real-world player who may have multiple screen names (aliases) across
  * different sites. The `playerUid` is a deterministic SHA-256 hash derived from
  * the site and normalized name, ensuring stable identity across store operations.
  *
  * @param playerUid     deterministic unique identifier (SHA-256 hash)
  * @param canonicalSite the primary site this player is known from
  * @param canonicalName the primary screen name for this player
  * @param profileUid    UID of the associated behavioral profile
  * @param modelUid      optional UID of a SICFUN-local trained model
  * @param behaviorUid   optional UID of the behavior fingerprint
  */
final case class RememberedPlayer(
    playerUid: String,
    canonicalSite: String,
    canonicalName: String,
    profileUid: String,
    modelUid: Option[String] = None,
    behaviorUid: Option[String] = None
):
  require(playerUid.trim.nonEmpty, "playerUid must be non-empty")
  require(canonicalSite.trim.nonEmpty, "canonicalSite must be non-empty")
  require(canonicalName.trim.nonEmpty, "canonicalName must be non-empty")
  require(profileUid.trim.nonEmpty, "profileUid must be non-empty")
  modelUid.foreach(value => require(value.trim.nonEmpty, "modelUid must be non-empty when present"))
  behaviorUid.foreach(value => require(value.trim.nonEmpty, "behaviorUid must be non-empty when present"))

/** Alias mapping from site-local name to canonical player UID. */
final case class PlayerAlias(
    site: String,
    playerName: String,
    playerUid: String,
    assertedByHuman: Boolean = false
):
  require(site.trim.nonEmpty, "site must be non-empty")
  require(playerName.trim.nonEmpty, "playerName must be non-empty")
  require(playerUid.trim.nonEmpty, "playerUid must be non-empty")

/** Player-level collapse assertion (`alias -> canonical`). */
final case class PlayerCollapse(
    aliasPlayerUid: String,
    canonicalPlayerUid: String,
    assertedByHuman: Boolean,
    assertedAtEpochMillis: Long
):
  require(aliasPlayerUid.trim.nonEmpty, "aliasPlayerUid must be non-empty")
  require(canonicalPlayerUid.trim.nonEmpty, "canonicalPlayerUid must be non-empty")
  require(assertedAtEpochMillis >= 0L, "assertedAtEpochMillis must be non-negative")

/** Profile-level collapse assertion (`alias -> canonical`). */
final case class ProfileCollapse(
    aliasProfileUid: String,
    canonicalProfileUid: String,
    assertedByHuman: Boolean,
    assertedAtEpochMillis: Long
):
  require(aliasProfileUid.trim.nonEmpty, "aliasProfileUid must be non-empty")
  require(canonicalProfileUid.trim.nonEmpty, "canonicalProfileUid must be non-empty")
  require(assertedAtEpochMillis >= 0L, "assertedAtEpochMillis must be non-negative")

/** Stable behavior fingerprint used for identity fallback and dedup support. */
final case class BehaviorFingerprint(
    uid: String,
    payload: String
):
  require(uid.trim.nonEmpty, "uid must be non-empty")
  require(payload.trim.nonEmpty, "payload must be non-empty")

/** Identity normalization and deterministic UID/fingerprint derivation helpers.
  *
  * The identity system works at three levels:
  *   1. '''Player identity''' — maps a (site, name) pair to a stable `playerUid` via SHA-256.
  *      SICFUN-local players additionally incorporate the model UID and behavior UID.
  *   2. '''Profile identity''' — maps behavioral data to a stable `profileUid`. For
  *      SICFUN-local players, this equals the behavior UID; for external players,
  *      it is a SHA-256 hash of (site, name).
  *   3. '''Behavior fingerprint''' — a content-addressable hash of the player's
  *      statistical signature (action frequencies, archetype posteriors, raise responses)
  *      used for dedup and identity fallback.
  *
  * All UIDs are deterministic (same inputs always produce the same hash), making the
  * system idempotent across repeated imports.
  */
object OpponentIdentity:
  val SicfunLocalSite = "sicfun@localhost"

  /** Normalize a site name to lowercase for consistent identity derivation. */
  def normalizeSite(site: String): String =
    site.trim.toLowerCase(Locale.ROOT)

  def normalizePlayerName(playerName: String): String =
    playerName.trim

  /** Compute the alias lookup key: "normalized_site\0lowercase_name". */
  def aliasKey(site: String, playerName: String): String =
    s"${normalizeSite(site)}\u0000${playerName.trim.toLowerCase(Locale.ROOT)}"

  /** Compute a content-addressable behavior fingerprint from profile statistics.
    *
    * Hashes the player's signature vector, raise response counts, and archetype
    * posterior probabilities into a deterministic SHA-256 UID. Two profiles with
    * identical behavioral statistics will produce the same fingerprint.
    */
  def fingerprint(profile: OpponentProfile): BehaviorFingerprint =
    val signatureKey = profile.signature.values.map(formatDouble).mkString(",")
    val responseKey = s"${profile.raiseResponses.folds}:${profile.raiseResponses.calls}:${profile.raiseResponses.raises}"
    val archetypeKey = PlayerArchetype.values.toVector
      .map(archetype => s"${archetype.toString}:${formatDouble(profile.archetypePosterior.probabilityOf(archetype))}")
      .mkString(",")
    val payload = s"signature=$signatureKey|responses=$responseKey|archetypes=$archetypeKey"
    BehaviorFingerprint(uid = sha256Hex(s"behavior|$payload"), payload = payload)

  /** Derive the deterministic player UID.
    *
    * For SICFUN-local players (site = "sicfun@localhost") with a model UID,
    * the hash incorporates the model UID and behavior UID for uniqueness.
    * For external players, the hash uses site + lowercase name.
    */
  def defaultPlayerUid(
      site: String,
      playerName: String,
      modelUid: Option[String],
      behaviorUid: String
  ): String =
    val normalizedSite = normalizeSite(site)
    if normalizedSite == SicfunLocalSite && modelUid.exists(_.trim.nonEmpty) then
      sha256Hex(s"sicfun-player|${modelUid.get.trim}|$behaviorUid")
    else
      sha256Hex(s"player|$normalizedSite|${playerName.trim.toLowerCase(Locale.ROOT)}")

  /** Derive the deterministic profile UID.
    *
    * SICFUN-local players use the behavior UID directly (profile = behavior).
    * External players use SHA-256("profile|site|name").
    */
  def defaultProfileUid(
      site: String,
      playerName: String,
      behaviorUid: String
  ): String =
    val normalizedSite = normalizeSite(site)
    if normalizedSite == SicfunLocalSite then behaviorUid
    else sha256Hex(s"profile|$normalizedSite|${playerName.trim.toLowerCase(Locale.ROOT)}")

  /** Ensure a profile has all identity fields populated (playerUid, profileUid, behaviorUid).
    *
    * If any UID is missing, it is derived deterministically from the profile's
    * site, name, and behavioral statistics. This is idempotent — calling it on
    * an already-identified profile returns the same values.
    */
  def ensureProfileIdentity(profile: OpponentProfile): OpponentProfile =
    val normalizedSite = normalizeSite(profile.site)
    val normalizedName = normalizePlayerName(profile.playerName)
    val cleanedModelUid = profile.modelUid.map(_.trim).filter(_.nonEmpty)
    val behaviorUid = profile.behaviorUid.filter(_.trim.nonEmpty).getOrElse(fingerprint(profile).uid)
    val playerUid = profile.playerUid.filter(_.trim.nonEmpty)
      .getOrElse(defaultPlayerUid(normalizedSite, normalizedName, cleanedModelUid, behaviorUid))
    val profileUid = profile.profileUid.filter(_.trim.nonEmpty)
      .getOrElse(defaultProfileUid(normalizedSite, normalizedName, behaviorUid))
    profile.copy(
      site = normalizedSite,
      playerName = normalizedName,
      playerUid = Some(playerUid),
      profileUid = Some(profileUid),
      behaviorUid = Some(behaviorUid),
      modelUid = cleanedModelUid
    )

  /** Create a RememberedPlayer from a profile, ensuring identity fields are populated. */
  def defaultPlayer(profile: OpponentProfile): RememberedPlayer =
    val identified = ensureProfileIdentity(profile)
    RememberedPlayer(
      playerUid = identified.playerUid.getOrElse(
        throw new IllegalArgumentException("identified profile must define playerUid")
      ),
      canonicalSite = identified.site,
      canonicalName = identified.playerName,
      profileUid = identified.profileUid.getOrElse(
        throw new IllegalArgumentException("identified profile must define profileUid")
      ),
      modelUid = identified.modelUid,
      behaviorUid = identified.behaviorUid
    )

  def normalizePlayer(player: RememberedPlayer): RememberedPlayer =
    player.copy(
      canonicalSite = normalizeSite(player.canonicalSite),
      canonicalName = normalizePlayerName(player.canonicalName),
      modelUid = player.modelUid.map(_.trim).filter(_.nonEmpty),
      behaviorUid = player.behaviorUid.map(_.trim).filter(_.nonEmpty)
    )

  def normalizeAlias(alias: PlayerAlias): PlayerAlias =
    alias.copy(
      site = normalizeSite(alias.site),
      playerName = normalizePlayerName(alias.playerName)
    )

  def sha256Hex(value: String): String =
    val digest = MessageDigest.getInstance("SHA-256")
    val bytes = digest.digest(value.getBytes(StandardCharsets.UTF_8))
    bytes.map("%02x".format(_)).mkString

  private def formatDouble(value: Double): String =
    f"$value%.6f"
