package sicfun.holdem.history

import sicfun.holdem.engine.PlayerArchetype

import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.Locale

/** Canonical remembered-player identity record used across aliases/profiles. */
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

/** Identity normalization and deterministic UID/fingerprint derivation helpers. */
object OpponentIdentity:
  val SicfunLocalSite = "sicfun@localhost"

  def normalizeSite(site: String): String =
    site.trim.toLowerCase(Locale.ROOT)

  def normalizePlayerName(playerName: String): String =
    playerName.trim

  def aliasKey(site: String, playerName: String): String =
    s"${normalizeSite(site)}\u0000${playerName.trim.toLowerCase(Locale.ROOT)}"

  def fingerprint(profile: OpponentProfile): BehaviorFingerprint =
    val signatureKey = profile.signature.values.map(formatDouble).mkString(",")
    val responseKey = s"${profile.raiseResponses.folds}:${profile.raiseResponses.calls}:${profile.raiseResponses.raises}"
    val archetypeKey = PlayerArchetype.values.toVector
      .map(archetype => s"${archetype.toString}:${formatDouble(profile.archetypePosterior.probabilityOf(archetype))}")
      .mkString(",")
    val payload = s"signature=$signatureKey|responses=$responseKey|archetypes=$archetypeKey"
    BehaviorFingerprint(uid = sha256Hex(s"behavior|$payload"), payload = payload)

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

  def defaultProfileUid(
      site: String,
      playerName: String,
      behaviorUid: String
  ): String =
    val normalizedSite = normalizeSite(site)
    if normalizedSite == SicfunLocalSite then behaviorUid
    else sha256Hex(s"profile|$normalizedSite|${playerName.trim.toLowerCase(Locale.ROOT)}")

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
