package sicfun.holdem.history
import sicfun.holdem.analysis.*
import sicfun.holdem.engine.*
import sicfun.holdem.types.*

import sicfun.core.Metrics

import ujson.{Arr, Num, Obj, Str, Value}

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.collection.mutable

final case class OpponentActionSummary(
    folds: Int = 0,
    raises: Int = 0,
    calls: Int = 0,
    checks: Int = 0,
    callPotOddsTotal: Double = 0.0
):
  require(folds >= 0, "folds must be non-negative")
  require(raises >= 0, "raises must be non-negative")
  require(calls >= 0, "calls must be non-negative")
  require(checks >= 0, "checks must be non-negative")
  require(callPotOddsTotal >= 0.0, "callPotOddsTotal must be non-negative")

  def totalActions: Int = folds + raises + calls + checks

  def merge(other: OpponentActionSummary): OpponentActionSummary =
    OpponentActionSummary(
      folds = folds + other.folds,
      raises = raises + other.raises,
      calls = calls + other.calls,
      checks = checks + other.checks,
      callPotOddsTotal = callPotOddsTotal + other.callPotOddsTotal
    )

  def signature: PlayerSignature =
    val total = math.max(1, totalActions).toDouble
    val foldRate = folds / total
    val raiseRate = raises / total
    val callRate = calls / total
    val checkRate = checks / total
    val avgActionEntropy = Metrics.entropy(Vector(foldRate, raiseRate, callRate, checkRate))
    val avgPotOddsWhenCalling =
      if calls <= 0 then 0.0
      else callPotOddsTotal / calls.toDouble
    PlayerSignature(Vector(foldRate, raiseRate, callRate, checkRate, avgActionEntropy, avgPotOddsWhenCalling))

object OpponentActionSummary:
  def observe(event: PokerEvent): OpponentActionSummary =
    event.action match
      case PokerAction.Fold => OpponentActionSummary(folds = 1)
      case PokerAction.Check => OpponentActionSummary(checks = 1)
      case PokerAction.Call =>
        val denominator = event.potBefore + event.toCall
        val potOdds = if denominator <= 0.0 then 0.0 else event.toCall / denominator
        OpponentActionSummary(calls = 1, callPotOddsTotal = potOdds)
      case PokerAction.Raise(_) => OpponentActionSummary(raises = 1)

final case class StabilitySnapshot(
    windowCount: Int,
    significantChanges: Int,
    maxDrift: Double,
    meanDrift: Double
):
  require(windowCount >= 0, "windowCount must be non-negative")
  require(significantChanges >= 0, "significantChanges must be non-negative")
  require(maxDrift >= 0.0, "maxDrift must be non-negative")
  require(meanDrift >= 0.0, "meanDrift must be non-negative")

final case class OpponentProfile(
    site: String,
    playerName: String,
    handsObserved: Int,
    firstSeenEpochMillis: Long,
    lastSeenEpochMillis: Long,
    actionSummary: OpponentActionSummary,
    raiseResponses: RaiseResponseCounts,
    recentEvents: Vector[PokerEvent],
    seenHandIds: Vector[String],
    playerUid: Option[String] = None,
    profileUid: Option[String] = None,
    behaviorUid: Option[String] = None,
    modelUid: Option[String] = None
):
  require(site.trim.nonEmpty, "site must be non-empty")
  require(playerName.trim.nonEmpty, "playerName must be non-empty")
  require(handsObserved >= 0, "handsObserved must be non-negative")
  require(firstSeenEpochMillis >= 0L, "firstSeenEpochMillis must be non-negative")
  require(lastSeenEpochMillis >= 0L, "lastSeenEpochMillis must be non-negative")
  require(lastSeenEpochMillis >= firstSeenEpochMillis, "lastSeenEpochMillis must be >= firstSeenEpochMillis")
  playerUid.foreach(value => require(value.trim.nonEmpty, "playerUid must be non-empty when present"))
  profileUid.foreach(value => require(value.trim.nonEmpty, "profileUid must be non-empty when present"))
  behaviorUid.foreach(value => require(value.trim.nonEmpty, "behaviorUid must be non-empty when present"))
  modelUid.foreach(value => require(value.trim.nonEmpty, "modelUid must be non-empty when present"))

  def signature: PlayerSignature =
    actionSummary.signature

  def archetypePosterior: ArchetypePosterior =
    ArchetypeLearning.posteriorFromCounts(raiseResponses)

  def recentObservations: Vector[VillainObservation] =
    recentEvents
      .filter(_.action != PokerAction.Fold)
      .map(event => VillainObservation(event.action, OpponentProfile.eventToGameState(event)))

  def stability: Option[StabilitySnapshot] =
    OpponentProfile.computeStability(recentEvents, playerName)

  def exploitHints: Vector[String] =
    OpponentProfile.exploitHintsFor(this)

  def identified: OpponentProfile =
    OpponentIdentity.ensureProfileIdentity(this)

object OpponentProfile:
  val MaxRecentEvents = 256
  val MaxSeenHandIds = 2048

  def fromSingleEvent(
      site: String,
      playerName: String,
      event: PokerEvent,
      facedRaiseResponse: Boolean
  ): OpponentProfile =
    OpponentProfile(
      site = OpponentIdentity.normalizeSite(site),
      playerName = playerName,
      handsObserved = 1,
      firstSeenEpochMillis = event.occurredAtEpochMillis,
      lastSeenEpochMillis = event.occurredAtEpochMillis,
      actionSummary = OpponentActionSummary.observe(event),
      raiseResponses = if facedRaiseResponse then RaiseResponseCounts().observe(event.action) else RaiseResponseCounts(),
      recentEvents = Vector(event),
      seenHandIds = Vector(event.handId)
    )

  def fromImportedHands(
      site: String,
      hands: Seq[ImportedHand],
      excludePlayers: Set[String] = Set.empty
  ): Vector[OpponentProfile] =
    val builders = mutable.Map.empty[String, ProfileBuilder]
    hands.foreach { hand =>
      val raiseSeenByStreet = mutable.Map.empty[Street, Boolean].withDefaultValue(false)
      hand.events.sortBy(_.sequenceInHand).foreach { event =>
        if !excludePlayers.contains(event.playerId) then
          val builder = builders.getOrElseUpdate(event.playerId, new ProfileBuilder(site, event.playerId))
          builder.observeHand(event.handId, event.occurredAtEpochMillis)
          builder.observeEvent(event)
          if event.toCall > 0.0 && raiseSeenByStreet(event.street) then
            builder.observeRaiseResponse(event.action)
        if event.action.category == PokerAction.Category.Raise then
          raiseSeenByStreet.update(event.street, true)
      }
    }
    builders.valuesIterator.map(_.build()).toVector.sortBy(profile => (-profile.handsObserved, profile.playerName))

  def merge(existing: OpponentProfile, incoming: OpponentProfile): OpponentProfile =
    val existingSeen = existing.seenHandIds.toSet
    val uniqueIncomingHandIds = incoming.seenHandIds.filterNot(existingSeen.contains)
    if uniqueIncomingHandIds.isEmpty then existing
    else
      val uniqueIncomingSet = uniqueIncomingHandIds.toSet
      val mergedEvents = (existing.recentEvents ++ incoming.recentEvents.filter(event => uniqueIncomingSet.contains(event.handId)))
        .sortBy(event => (event.occurredAtEpochMillis, event.handId, event.sequenceInHand))
        .takeRight(MaxRecentEvents)
      val mergedSeen = dedupePreserveOrder(existing.seenHandIds ++ uniqueIncomingHandIds).takeRight(MaxSeenHandIds)
      OpponentProfile(
        site = existing.site,
        playerName = existing.playerName,
        handsObserved = existing.handsObserved + uniqueIncomingHandIds.length,
        firstSeenEpochMillis = math.min(existing.firstSeenEpochMillis, incoming.firstSeenEpochMillis),
        lastSeenEpochMillis = math.max(existing.lastSeenEpochMillis, incoming.lastSeenEpochMillis),
        actionSummary = existing.actionSummary.merge(incoming.actionSummary),
        raiseResponses = RaiseResponseCounts(
          folds = existing.raiseResponses.folds + incoming.raiseResponses.folds,
          calls = existing.raiseResponses.calls + incoming.raiseResponses.calls,
          raises = existing.raiseResponses.raises + incoming.raiseResponses.raises
        ),
        recentEvents = mergedEvents,
        seenHandIds = mergedSeen,
        playerUid =
          if existing.playerUid == incoming.playerUid then existing.playerUid
          else existing.playerUid.orElse(incoming.playerUid),
        profileUid =
          if existing.profileUid == incoming.profileUid then existing.profileUid
          else existing.profileUid.orElse(incoming.profileUid),
        behaviorUid =
          if existing.behaviorUid == incoming.behaviorUid then existing.behaviorUid
          else existing.behaviorUid.orElse(incoming.behaviorUid),
        modelUid =
          if existing.modelUid == incoming.modelUid then existing.modelUid
          else existing.modelUid.orElse(incoming.modelUid)
      )

  def observeEvent(
      profile: OpponentProfile,
      event: PokerEvent,
      facedRaiseResponse: Boolean
  ): OpponentProfile =
    val alreadySeenEvent = profile.recentEvents.exists(existing =>
      existing.handId == event.handId &&
        existing.sequenceInHand == event.sequenceInHand &&
        existing.playerId == event.playerId
    )
    if alreadySeenEvent then profile
    else
      val isNewHand = !profile.seenHandIds.contains(event.handId)
      OpponentProfile(
        site = profile.site,
        playerName = profile.playerName,
        handsObserved = profile.handsObserved + (if isNewHand then 1 else 0),
        firstSeenEpochMillis = math.min(profile.firstSeenEpochMillis, event.occurredAtEpochMillis),
        lastSeenEpochMillis = math.max(profile.lastSeenEpochMillis, event.occurredAtEpochMillis),
        actionSummary = profile.actionSummary.merge(OpponentActionSummary.observe(event)),
        raiseResponses =
          if facedRaiseResponse then profile.raiseResponses.observe(event.action)
          else profile.raiseResponses,
        recentEvents = (profile.recentEvents :+ event)
          .sortBy(existing => (existing.occurredAtEpochMillis, existing.handId, existing.sequenceInHand))
          .takeRight(MaxRecentEvents),
        seenHandIds =
          if isNewHand then (profile.seenHandIds :+ event.handId).takeRight(MaxSeenHandIds)
          else profile.seenHandIds,
        playerUid = profile.playerUid,
        profileUid = profile.profileUid,
        behaviorUid = profile.behaviorUid,
        modelUid = profile.modelUid
      )

  private final class ProfileBuilder(site: String, playerName: String):
    private var handsObserved = 0
    private var firstSeen = Long.MaxValue
    private var lastSeen = 0L
    private var actionSummary = OpponentActionSummary()
    private var raiseResponses = RaiseResponseCounts()
    private val seenHandIds = mutable.LinkedHashSet.empty[String]
    private val recentEvents = mutable.ArrayBuffer.empty[PokerEvent]

    def observeHand(handId: String, occurredAtEpochMillis: Long): Unit =
      if seenHandIds.add(handId) then
        handsObserved += 1
      firstSeen = math.min(firstSeen, occurredAtEpochMillis)
      lastSeen = math.max(lastSeen, occurredAtEpochMillis)

    def observeEvent(event: PokerEvent): Unit =
      actionSummary = actionSummary.merge(OpponentActionSummary.observe(event))
      recentEvents += event
      if recentEvents.length > MaxRecentEvents then
        recentEvents.remove(0, recentEvents.length - MaxRecentEvents)

    def observeRaiseResponse(action: PokerAction): Unit =
      raiseResponses = raiseResponses.observe(action)

    def build(): OpponentProfile =
      OpponentProfile(
        site = OpponentIdentity.normalizeSite(site),
        playerName = playerName,
        handsObserved = handsObserved,
        firstSeenEpochMillis = if firstSeen == Long.MaxValue then 0L else firstSeen,
        lastSeenEpochMillis = lastSeen,
        actionSummary = actionSummary,
        raiseResponses = raiseResponses,
        recentEvents = recentEvents.toVector,
        seenHandIds = seenHandIds.toVector.takeRight(MaxSeenHandIds)
      )

  private def dedupePreserveOrder(values: Seq[String]): Vector[String] =
    val seen = mutable.HashSet.empty[String]
    values.iterator.filter { value =>
      if seen.contains(value) then false
      else
        seen += value
        true
    }.toVector

  private def computeStability(
      recentEvents: Vector[PokerEvent],
      playerName: String
  ): Option[StabilitySnapshot] =
    if recentEvents.length < 6 then None
    else
      val ordered = recentEvents.sortBy(event => (event.occurredAtEpochMillis, event.handId, event.sequenceInHand))
      val span = math.max(1L, ordered.last.occurredAtEpochMillis - ordered.head.occurredAtEpochMillis)
      val window = math.max(60_000L, span / 2L)
      val slide = math.max(30_000L, window / 2L)
      val minEvents = math.max(3, math.min(12, ordered.length / 3))
      try
        val report = LongitudinalAnalysis.analyze(
          events = ordered,
          playerId = playerName,
          config = LongitudinalConfig(
            windowSizeMillis = window,
            slideStepMillis = slide,
            driftThreshold = 0.18,
            minEventsPerWindow = minEvents
          )
        )
        Some(
          StabilitySnapshot(
            windowCount = report.windows.length,
            significantChanges = report.drifts.count(_.significantChange),
            maxDrift = report.maxDrift,
            meanDrift = report.meanDrift
          )
        )
      catch
        case _: IllegalArgumentException => None

  private def exploitHintsFor(profile: OpponentProfile): Vector[String] =
    val hints = Vector.newBuilder[String]
    val signature = profile.signature
    val responseTotal = profile.raiseResponses.total.toDouble
    if responseTotal >= 4.0 then
      val foldToRaise = profile.raiseResponses.folds / responseTotal
      val callVsRaise = profile.raiseResponses.calls / responseTotal
      val reraiseVsRaise = profile.raiseResponses.raises / responseTotal
      if foldToRaise >= 0.55 then
        hints += "Increase bluff pressure when they face a raise."
      if callVsRaise >= 0.55 then
        hints += "Value bet thinner and cut back low-equity bluffs."
      if reraiseVsRaise >= 0.28 then
        hints += "Expect re-raises; trap stronger hands and avoid thin bluffs."

    val foldRate = signature.values(0)
    val raiseRate = signature.values(1)
    val callRate = signature.values(2)
    val checkRate = signature.values(3)
    if foldRate >= 0.45 then
      hints += "Open slightly wider against them; they over-fold overall."
    if callRate >= 0.42 && raiseRate <= 0.12 then
      hints += "Treat them like a calling station until shown otherwise."
    if raiseRate >= 0.33 then
      hints += "They choose aggressive lines often; bluff-catch more selectively."
    if checkRate >= 0.30 && raiseRate <= 0.10 then
      hints += "Respect sudden aggression from their passive lines."
    profile.stability.foreach { stability =>
      if stability.significantChanges > 0 || stability.meanDrift >= 0.20 then
        hints += "Profile is drifting; lower exploit confidence and stay closer to baseline."
    }
    val built = hints.result().distinct
    if built.nonEmpty then built.take(3)
    else Vector("No high-confidence exploit yet; use baseline strategy with mild archetype bias.")

  private[history] def eventToGameState(event: PokerEvent): GameState =
    GameState(
      street = event.street,
      board = event.board,
      pot = event.potBefore,
      toCall = event.toCall,
      position = event.position,
      stackSize = event.stackBefore,
      betHistory = event.betHistory
    )

final case class OpponentProfileStore(
    profiles: Vector[OpponentProfile],
    players: Vector[RememberedPlayer] = Vector.empty,
    aliases: Vector[PlayerAlias] = Vector.empty,
    playerCollapses: Vector[PlayerCollapse] = Vector.empty,
    profileCollapses: Vector[ProfileCollapse] = Vector.empty
):
  def population: Vector[RememberedPlayer] =
    OpponentProfileStore.normalize(this).players

  def find(site: String, playerName: String): Option[OpponentProfile] =
    OpponentProfileStore.findProfile(OpponentProfileStore.normalize(this), site, playerName)

  def findPlayer(site: String, playerName: String): Option[RememberedPlayer] =
    OpponentProfileStore.findPlayer(OpponentProfileStore.normalize(this), site, playerName)

  def upsert(profile: OpponentProfile): OpponentProfileStore =
    OpponentProfileStore.upsertNormalized(OpponentProfileStore.normalize(this), profile)

  def upsertAll(incoming: Seq[OpponentProfile]): OpponentProfileStore =
    incoming.foldLeft(OpponentProfileStore.normalize(this))((store, profile) => OpponentProfileStore.upsertNormalized(store, profile))

  def observeEvent(
      site: String,
      playerName: String,
      event: PokerEvent,
      facedRaiseResponse: Boolean
  ): OpponentProfileStore =
    OpponentProfileStore.observeEventNormalized(
      OpponentProfileStore.normalize(this),
      site = site,
      playerName = playerName,
      event = event,
      facedRaiseResponse = facedRaiseResponse
    )

  def registerSicfunPlayer(
      playerName: String,
      modelUid: String,
      profile: OpponentProfile
  ): OpponentProfileStore =
    upsert(
      profile.copy(
        site = OpponentIdentity.SicfunLocalSite,
        playerName = playerName.trim,
        playerUid = None,
        profileUid = None,
        behaviorUid = None,
        modelUid = Some(modelUid.trim)
      )
    )

  def collapsePlayers(
      canonicalSite: String,
      canonicalName: String,
      aliasSite: String,
      aliasName: String,
      collapseProfiles: Boolean = false,
      assertedAtEpochMillis: Long = System.currentTimeMillis()
  ): OpponentProfileStore =
    OpponentProfileStore.collapsePlayersNormalized(
      OpponentProfileStore.normalize(this),
      canonicalSite = canonicalSite,
      canonicalName = canonicalName,
      aliasSite = aliasSite,
      aliasName = aliasName,
      collapseProfiles = collapseProfiles,
      assertedAtEpochMillis = assertedAtEpochMillis
    )

  def collapseProfiles(
      canonicalSite: String,
      canonicalName: String,
      aliasSite: String,
      aliasName: String,
      assertedAtEpochMillis: Long = System.currentTimeMillis()
  ): OpponentProfileStore =
    OpponentProfileStore.collapseProfilesNormalized(
      OpponentProfileStore.normalize(this),
      canonicalSite = canonicalSite,
      canonicalName = canonicalName,
      aliasSite = aliasSite,
      aliasName = aliasName,
      assertedAtEpochMillis = assertedAtEpochMillis
    )

object OpponentProfileStore:
  private val FormatVersion = "opponent-profile-store-v2"
  private val LegacyFormatVersion = "opponent-profile-store-v1"

  val empty: OpponentProfileStore = OpponentProfileStore(Vector.empty)

  def load(path: Path): OpponentProfileStore =
    if !Files.exists(path) then empty
    else
      val json = ujson.read(Files.readString(path, StandardCharsets.UTF_8))
      readStore(json)

  def save(path: Path, store: OpponentProfileStore): Unit =
    Option(path.getParent).foreach(parent => Files.createDirectories(parent))
    Files.writeString(path, ujson.write(writeStore(normalize(store)), indent = 2), StandardCharsets.UTF_8)

  private[history] def key(site: String, playerName: String): String =
    OpponentIdentity.aliasKey(site, playerName)

  private[history] def normalize(store: OpponentProfileStore): OpponentProfileStore =
    val identifiedProfiles = store.profiles.map(OpponentIdentity.ensureProfileIdentity)
    val mergedProfiles = mutable.LinkedHashMap.empty[String, OpponentProfile]
    identifiedProfiles.foreach { profile =>
      val profileUid = profile.profileUid.getOrElse(
        throw new IllegalArgumentException("identified profile must define profileUid")
      )
      val merged =
        mergedProfiles.get(profileUid) match
          case Some(existing) =>
            OpponentIdentity.ensureProfileIdentity(
              OpponentProfile.merge(existing, profile).copy(
                site = existing.site,
                playerName = existing.playerName,
                playerUid = existing.playerUid.orElse(profile.playerUid),
                profileUid = Some(profileUid),
                behaviorUid = existing.behaviorUid.orElse(profile.behaviorUid),
                modelUid =
                  (existing.modelUid, profile.modelUid) match
                    case (Some(left), Some(right)) if left != right => None
                    case (Some(left), _) => Some(left)
                    case (_, Some(right)) => Some(right)
                    case _ => None
              )
            )
          case None => profile
      mergedProfiles.update(profileUid, merged)
    }

    val playerByUid = mutable.LinkedHashMap.empty[String, RememberedPlayer]
    identifiedProfiles.foreach { profile =>
      val defaultPlayer = OpponentIdentity.defaultPlayer(profile)
      playerByUid.update(defaultPlayer.playerUid, mergePlayers(playerByUid.get(defaultPlayer.playerUid), defaultPlayer))
    }
    store.players.map(OpponentIdentity.normalizePlayer).foreach { player =>
      playerByUid.update(player.playerUid, mergePlayers(playerByUid.get(player.playerUid), player))
    }

    val aliasByKey = mutable.LinkedHashMap.empty[String, PlayerAlias]
    identifiedProfiles.foreach { profile =>
      val defaultAlias = PlayerAlias(
        site = profile.site,
        playerName = profile.playerName,
        playerUid = profile.playerUid.getOrElse(
          throw new IllegalArgumentException("identified profile must define playerUid")
        )
      )
      aliasByKey.update(key(defaultAlias.site, defaultAlias.playerName), OpponentIdentity.normalizeAlias(defaultAlias))
    }
    playerByUid.values.foreach { player =>
      val canonicalAlias = PlayerAlias(
        site = player.canonicalSite,
        playerName = player.canonicalName,
        playerUid = player.playerUid
      )
      aliasByKey.update(key(canonicalAlias.site, canonicalAlias.playerName), OpponentIdentity.normalizeAlias(canonicalAlias))
    }
    store.aliases.map(OpponentIdentity.normalizeAlias).foreach { alias =>
      aliasByKey.update(key(alias.site, alias.playerName), alias)
    }

    OpponentProfileStore(
      profiles = mergedProfiles.values.toVector.sortBy(profile => (profile.site, profile.playerName, profile.profileUid.getOrElse(""))),
      players = playerByUid.values.toVector.sortBy(player => (player.canonicalSite, player.canonicalName, player.playerUid)),
      aliases = aliasByKey.values.toVector.sortBy(alias => (alias.site, alias.playerName, alias.playerUid)),
      playerCollapses = dedupePlayerCollapses(store.playerCollapses),
      profileCollapses = dedupeProfileCollapses(store.profileCollapses)
    )

  private def findProfile(store: OpponentProfileStore, site: String, playerName: String): Option[OpponentProfile] =
    rawPlayerUid(store, site, playerName)
      .flatMap { aliasPlayerUid =>
        val canonicalPlayerUid = resolvePlayerUid(store, aliasPlayerUid)
        playerByUid(store, canonicalPlayerUid)
          .orElse(playerByUid(store, aliasPlayerUid))
          .flatMap { player =>
            val canonicalProfileUid = resolveProfileUid(store, player.profileUid)
            profileByUid(store, canonicalProfileUid)
              .orElse(profileByUid(store, player.profileUid))
          }
      }

  private def findPlayer(store: OpponentProfileStore, site: String, playerName: String): Option[RememberedPlayer] =
    rawPlayerUid(store, site, playerName)
      .flatMap { aliasPlayerUid =>
        val canonicalPlayerUid = resolvePlayerUid(store, aliasPlayerUid)
        playerByUid(store, canonicalPlayerUid).orElse(playerByUid(store, aliasPlayerUid))
      }

  private def upsertNormalized(store: OpponentProfileStore, profile: OpponentProfile): OpponentProfileStore =
    val incoming = OpponentIdentity.ensureProfileIdentity(profile)
    val aliasPlayerUid = rawPlayerUid(store, incoming.site, incoming.playerName).getOrElse(
      incoming.playerUid.getOrElse(
        throw new IllegalArgumentException("identified profile must define playerUid")
      )
    )
    val canonicalPlayerUid = resolvePlayerUid(store, aliasPlayerUid)
    val canonicalPlayer =
      playerByUid(store, canonicalPlayerUid)
        .orElse(
          if canonicalPlayerUid == aliasPlayerUid then Some(OpponentIdentity.defaultPlayer(incoming.copy(playerUid = Some(canonicalPlayerUid))))
          else None
        )
        .getOrElse(
          OpponentIdentity.defaultPlayer(incoming.copy(playerUid = Some(canonicalPlayerUid)))
        )
    val targetProfileUid = resolveProfileUid(store, canonicalPlayer.profileUid)
    val mergedProfile =
      profileByUid(store, targetProfileUid) match
        case Some(existing) =>
          OpponentIdentity.ensureProfileIdentity(
            OpponentProfile.merge(
              existing,
              incoming.copy(
                site = existing.site,
                playerName = existing.playerName,
                profileUid = Some(targetProfileUid)
              )
            ).copy(
              playerUid = Some(canonicalPlayerUid),
              profileUid = Some(targetProfileUid),
              modelUid = canonicalPlayer.modelUid.orElse(incoming.modelUid)
            )
          )
        case None =>
          OpponentIdentity.ensureProfileIdentity(
            incoming.copy(
              site = canonicalPlayer.canonicalSite,
              playerName = canonicalPlayer.canonicalName,
              playerUid = Some(canonicalPlayerUid),
              profileUid = Some(targetProfileUid),
              modelUid = canonicalPlayer.modelUid.orElse(incoming.modelUid)
            )
          )

    val aliasPlayer = playerByUid(store, aliasPlayerUid).getOrElse(
      OpponentIdentity.defaultPlayer(incoming.copy(playerUid = Some(aliasPlayerUid)))
    )
    val updatedCanonicalPlayer = mergePlayers(
      playerByUid(store, canonicalPlayerUid),
      canonicalPlayer.copy(
        profileUid = targetProfileUid,
        modelUid = canonicalPlayer.modelUid.orElse(incoming.modelUid),
        behaviorUid = mergedProfile.behaviorUid
      )
    )

    normalize(
      store.copy(
        profiles = replaceProfile(store.profiles, mergedProfile),
        players =
          replacePlayer(
            replacePlayer(store.players, aliasPlayer),
            updatedCanonicalPlayer
          ),
        aliases = replaceAlias(
          store.aliases,
          PlayerAlias(
            site = incoming.site,
            playerName = incoming.playerName,
            playerUid = aliasPlayerUid,
            assertedByHuman = store.aliases.exists(existing =>
              key(existing.site, existing.playerName) == key(incoming.site, incoming.playerName) && existing.assertedByHuman
            )
          )
        )
      )
    )

  private def observeEventNormalized(
      store: OpponentProfileStore,
      site: String,
      playerName: String,
      event: PokerEvent,
      facedRaiseResponse: Boolean
  ): OpponentProfileStore =
    findProfile(store, site, playerName) match
      case Some(existing) =>
        val observed = OpponentProfile.observeEvent(existing, event, facedRaiseResponse)
        if observed == existing then store
        else
          val identified = OpponentIdentity.ensureProfileIdentity(
            observed.copy(
              playerUid = existing.playerUid.orElse(observed.playerUid),
              profileUid = existing.profileUid.orElse(observed.profileUid),
              behaviorUid = None,
              modelUid = existing.modelUid.orElse(observed.modelUid)
            )
          )
          val updatedPlayers =
            existing.profileUid match
              case Some(previousProfileUid) =>
                store.players.map { player =>
                  if player.profileUid == previousProfileUid then
                    player.copy(
                      profileUid = identified.profileUid.getOrElse(player.profileUid),
                      behaviorUid = identified.behaviorUid.orElse(player.behaviorUid)
                    )
                  else player
                }
              case None => store.players
          normalize(
            store.copy(
              profiles = replaceProfile(store.profiles, identified),
              players = updatedPlayers
            )
          )
      case None =>
        upsertNormalized(
          store,
          OpponentProfile.fromSingleEvent(site, playerName, event, facedRaiseResponse)
        )

  private def collapsePlayersNormalized(
      store: OpponentProfileStore,
      canonicalSite: String,
      canonicalName: String,
      aliasSite: String,
      aliasName: String,
      collapseProfiles: Boolean,
      assertedAtEpochMillis: Long
  ): OpponentProfileStore =
    val rawCanonical = rawPlayerUid(store, canonicalSite, canonicalName)
    val rawAlias = rawPlayerUid(store, aliasSite, aliasName)
    (rawCanonical, rawAlias) match
      case (Some(canonicalPlayerUid), Some(aliasPlayerUid)) if canonicalPlayerUid != aliasPlayerUid =>
        val collapsed = normalize(
          store.copy(
            playerCollapses =
              store.playerCollapses.filterNot(_.aliasPlayerUid == aliasPlayerUid) :+
                PlayerCollapse(
                  aliasPlayerUid = aliasPlayerUid,
                  canonicalPlayerUid = canonicalPlayerUid,
                  assertedByHuman = true,
                  assertedAtEpochMillis = math.max(0L, assertedAtEpochMillis)
                )
          )
        )
        if collapseProfiles then
          collapseProfilesNormalized(
            collapsed,
            canonicalSite = canonicalSite,
            canonicalName = canonicalName,
            aliasSite = aliasSite,
            aliasName = aliasName,
            assertedAtEpochMillis = assertedAtEpochMillis
          )
        else collapsed
      case _ => store

  private def collapseProfilesNormalized(
      store: OpponentProfileStore,
      canonicalSite: String,
      canonicalName: String,
      aliasSite: String,
      aliasName: String,
      assertedAtEpochMillis: Long
  ): OpponentProfileStore =
    val canonicalProfileUid = rawPlayerUid(store, canonicalSite, canonicalName)
      .flatMap(playerByUid(store, _))
      .map(_.profileUid)
    val aliasProfileUid = rawPlayerUid(store, aliasSite, aliasName)
      .flatMap(playerByUid(store, _))
      .map(_.profileUid)
    (canonicalProfileUid, aliasProfileUid) match
      case (Some(canonicalUid), Some(aliasUid)) if canonicalUid != aliasUid =>
        normalize(
          store.copy(
            profileCollapses =
              store.profileCollapses.filterNot(_.aliasProfileUid == aliasUid) :+
                ProfileCollapse(
                  aliasProfileUid = aliasUid,
                  canonicalProfileUid = canonicalUid,
                  assertedByHuman = true,
                  assertedAtEpochMillis = math.max(0L, assertedAtEpochMillis)
                )
          )
        )
      case _ => store

  private def rawPlayerUid(store: OpponentProfileStore, site: String, playerName: String): Option[String] =
    val aliasId = key(site, playerName)
    store.aliases.find(alias => key(alias.site, alias.playerName) == aliasId).map(_.playerUid)
      .orElse(
        store.players.find(player => key(player.canonicalSite, player.canonicalName) == aliasId).map(_.playerUid)
      )

  private def resolvePlayerUid(store: OpponentProfileStore, playerUid: String): String =
    resolveUid(
      seed = playerUid,
      next = uid => store.playerCollapses.find(_.aliasPlayerUid == uid).map(_.canonicalPlayerUid)
    )

  private def resolveProfileUid(store: OpponentProfileStore, profileUid: String): String =
    resolveUid(
      seed = profileUid,
      next = uid => store.profileCollapses.find(_.aliasProfileUid == uid).map(_.canonicalProfileUid)
    )

  private def resolveUid(seed: String, next: String => Option[String]): String =
    val seen = mutable.HashSet.empty[String]
    var current = seed
    var continue = true
    while continue && !seen.contains(current) do
      seen += current
      next(current) match
        case Some(updated) if updated.trim.nonEmpty && updated != current =>
          current = updated
        case _ =>
          continue = false
    current

  private def playerByUid(store: OpponentProfileStore, playerUid: String): Option[RememberedPlayer] =
    store.players.find(_.playerUid == playerUid)

  private def profileByUid(store: OpponentProfileStore, profileUid: String): Option[OpponentProfile] =
    store.profiles.find(_.profileUid.contains(profileUid))

  private def mergePlayers(existing: Option[RememberedPlayer], incoming: RememberedPlayer): RememberedPlayer =
    existing match
      case Some(current) =>
        current.copy(
          canonicalSite = if current.canonicalSite.trim.nonEmpty then current.canonicalSite else incoming.canonicalSite,
          canonicalName = if current.canonicalName.trim.nonEmpty then current.canonicalName else incoming.canonicalName,
          profileUid = if current.profileUid.trim.nonEmpty then current.profileUid else incoming.profileUid,
          modelUid = current.modelUid.orElse(incoming.modelUid),
          behaviorUid = current.behaviorUid.orElse(incoming.behaviorUid)
        )
      case None => incoming

  private def replaceProfile(existing: Vector[OpponentProfile], updated: OpponentProfile): Vector[OpponentProfile] =
    val profileUid = updated.profileUid.getOrElse(
      throw new IllegalArgumentException("updated profile must define profileUid")
    )
    if existing.exists(_.profileUid.contains(profileUid)) then
      existing.map { current =>
        if current.profileUid.contains(profileUid) then updated else current
      }
    else existing :+ updated

  private def replacePlayer(existing: Vector[RememberedPlayer], updated: RememberedPlayer): Vector[RememberedPlayer] =
    if existing.exists(_.playerUid == updated.playerUid) then
      existing.map(player => if player.playerUid == updated.playerUid then updated else player)
    else existing :+ updated

  private def replaceAlias(existing: Vector[PlayerAlias], updated: PlayerAlias): Vector[PlayerAlias] =
    val aliasId = key(updated.site, updated.playerName)
    if existing.exists(alias => key(alias.site, alias.playerName) == aliasId) then
      existing.map(alias => if key(alias.site, alias.playerName) == aliasId then updated else alias)
    else existing :+ updated

  private def dedupePlayerCollapses(collapses: Vector[PlayerCollapse]): Vector[PlayerCollapse] =
    val byAlias = mutable.LinkedHashMap.empty[String, PlayerCollapse]
    collapses.foreach(collapse => byAlias.update(collapse.aliasPlayerUid, collapse))
    byAlias.values.toVector.sortBy(collapse => (collapse.aliasPlayerUid, collapse.canonicalPlayerUid))

  private def dedupeProfileCollapses(collapses: Vector[ProfileCollapse]): Vector[ProfileCollapse] =
    val byAlias = mutable.LinkedHashMap.empty[String, ProfileCollapse]
    collapses.foreach(collapse => byAlias.update(collapse.aliasProfileUid, collapse))
    byAlias.values.toVector.sortBy(collapse => (collapse.aliasProfileUid, collapse.canonicalProfileUid))

  private[history] def writeStore(store: OpponentProfileStore): Value =
    Obj(
      "formatVersion" -> Str(FormatVersion),
      "profiles" -> Arr.from(store.profiles.map(writeProfile)),
      "players" -> Arr.from(store.players.map(writePlayer)),
      "aliases" -> Arr.from(store.aliases.map(writeAlias)),
      "playerCollapses" -> Arr.from(store.playerCollapses.map(writePlayerCollapse)),
      "profileCollapses" -> Arr.from(store.profileCollapses.map(writeProfileCollapse))
    )

  private[history] def readStore(json: Value): OpponentProfileStore =
    val obj = json.obj
    val version = obj.get("formatVersion").map(_.str).getOrElse(LegacyFormatVersion)
    version match
      case FormatVersion =>
        normalize(
          OpponentProfileStore(
            profiles = obj.get("profiles").map(_.arr.toVector.map(readProfile)).getOrElse(Vector.empty),
            players = obj.get("players").map(_.arr.toVector.map(readPlayer)).getOrElse(Vector.empty),
            aliases = obj.get("aliases").map(_.arr.toVector.map(readAlias)).getOrElse(Vector.empty),
            playerCollapses = obj.get("playerCollapses").map(_.arr.toVector.map(readPlayerCollapse)).getOrElse(Vector.empty),
            profileCollapses = obj.get("profileCollapses").map(_.arr.toVector.map(readProfileCollapse)).getOrElse(Vector.empty)
          )
        )
      case LegacyFormatVersion =>
        normalize(
          OpponentProfileStore(
            profiles = obj.get("profiles").map(_.arr.toVector.map(readProfile)).getOrElse(Vector.empty)
          )
        )
      case other =>
        throw new IllegalArgumentException(s"unsupported opponent profile store version: $other")

  private[history] def writeProfile(profile: OpponentProfile): Value =
    Obj(
      "site" -> Str(profile.site),
      "playerName" -> Str(profile.playerName),
      "handsObserved" -> Num(profile.handsObserved),
      "firstSeenEpochMillis" -> Str(profile.firstSeenEpochMillis.toString),
      "lastSeenEpochMillis" -> Str(profile.lastSeenEpochMillis.toString),
      "actionSummary" -> Obj(
        "folds" -> Num(profile.actionSummary.folds),
        "raises" -> Num(profile.actionSummary.raises),
        "calls" -> Num(profile.actionSummary.calls),
        "checks" -> Num(profile.actionSummary.checks),
        "callPotOddsTotal" -> Num(profile.actionSummary.callPotOddsTotal)
      ),
      "raiseResponses" -> Obj(
        "folds" -> Num(profile.raiseResponses.folds),
        "calls" -> Num(profile.raiseResponses.calls),
        "raises" -> Num(profile.raiseResponses.raises)
      ),
      "recentEvents" -> Arr.from(profile.recentEvents.map(writeEvent)),
      "seenHandIds" -> Arr.from(profile.seenHandIds.map(Str(_))),
      "playerUid" -> profile.playerUid.map(Str(_)).getOrElse(ujson.Null),
      "profileUid" -> profile.profileUid.map(Str(_)).getOrElse(ujson.Null),
      "behaviorUid" -> profile.behaviorUid.map(Str(_)).getOrElse(ujson.Null),
      "modelUid" -> profile.modelUid.map(Str(_)).getOrElse(ujson.Null)
    )

  private[history] def readProfile(json: Value): OpponentProfile =
    val obj = json.obj
    OpponentProfile(
      site = obj("site").str,
      playerName = obj("playerName").str,
      handsObserved = obj("handsObserved").num.toInt,
      firstSeenEpochMillis = obj("firstSeenEpochMillis").str.toLong,
      lastSeenEpochMillis = obj("lastSeenEpochMillis").str.toLong,
      actionSummary = {
        val action = obj("actionSummary").obj
        OpponentActionSummary(
          folds = action("folds").num.toInt,
          raises = action("raises").num.toInt,
          calls = action("calls").num.toInt,
          checks = action("checks").num.toInt,
          callPotOddsTotal = action("callPotOddsTotal").num
        )
      },
      raiseResponses = {
        val responses = obj("raiseResponses").obj
        RaiseResponseCounts(
          folds = responses("folds").num.toInt,
          calls = responses("calls").num.toInt,
          raises = responses("raises").num.toInt
        )
      },
      recentEvents = obj.get("recentEvents").map(_.arr.toVector.map(readEvent)).getOrElse(Vector.empty),
      seenHandIds = obj.get("seenHandIds").map(_.arr.toVector.map(_.str)).getOrElse(Vector.empty),
      playerUid = obj.get("playerUid").filterNot(_ == ujson.Null).map(_.str),
      profileUid = obj.get("profileUid").filterNot(_ == ujson.Null).map(_.str),
      behaviorUid = obj.get("behaviorUid").filterNot(_ == ujson.Null).map(_.str),
      modelUid = obj.get("modelUid").filterNot(_ == ujson.Null).map(_.str)
    )

  private[history] def writePlayer(player: RememberedPlayer): Value =
    Obj(
      "playerUid" -> Str(player.playerUid),
      "canonicalSite" -> Str(player.canonicalSite),
      "canonicalName" -> Str(player.canonicalName),
      "profileUid" -> Str(player.profileUid),
      "modelUid" -> player.modelUid.map(Str(_)).getOrElse(ujson.Null),
      "behaviorUid" -> player.behaviorUid.map(Str(_)).getOrElse(ujson.Null)
    )

  private[history] def readPlayer(json: Value): RememberedPlayer =
    val obj = json.obj
    RememberedPlayer(
      playerUid = obj("playerUid").str,
      canonicalSite = obj("canonicalSite").str,
      canonicalName = obj("canonicalName").str,
      profileUid = obj("profileUid").str,
      modelUid = obj.get("modelUid").filterNot(_ == ujson.Null).map(_.str),
      behaviorUid = obj.get("behaviorUid").filterNot(_ == ujson.Null).map(_.str)
    )

  private[history] def writeAlias(alias: PlayerAlias): Value =
    Obj(
      "site" -> Str(alias.site),
      "playerName" -> Str(alias.playerName),
      "playerUid" -> Str(alias.playerUid),
      "assertedByHuman" -> ujson.Bool(alias.assertedByHuman)
    )

  private[history] def readAlias(json: Value): PlayerAlias =
    val obj = json.obj
    PlayerAlias(
      site = obj("site").str,
      playerName = obj("playerName").str,
      playerUid = obj("playerUid").str,
      assertedByHuman = obj.get("assertedByHuman").exists(_.bool)
    )

  private[history] def writePlayerCollapse(collapse: PlayerCollapse): Value =
    Obj(
      "aliasPlayerUid" -> Str(collapse.aliasPlayerUid),
      "canonicalPlayerUid" -> Str(collapse.canonicalPlayerUid),
      "assertedByHuman" -> ujson.Bool(collapse.assertedByHuman),
      "assertedAtEpochMillis" -> Str(collapse.assertedAtEpochMillis.toString)
    )

  private[history] def readPlayerCollapse(json: Value): PlayerCollapse =
    val obj = json.obj
    PlayerCollapse(
      aliasPlayerUid = obj("aliasPlayerUid").str,
      canonicalPlayerUid = obj("canonicalPlayerUid").str,
      assertedByHuman = obj.get("assertedByHuman").exists(_.bool),
      assertedAtEpochMillis = obj("assertedAtEpochMillis").str.toLong
    )

  private[history] def writeProfileCollapse(collapse: ProfileCollapse): Value =
    Obj(
      "aliasProfileUid" -> Str(collapse.aliasProfileUid),
      "canonicalProfileUid" -> Str(collapse.canonicalProfileUid),
      "assertedByHuman" -> ujson.Bool(collapse.assertedByHuman),
      "assertedAtEpochMillis" -> Str(collapse.assertedAtEpochMillis.toString)
    )

  private[history] def readProfileCollapse(json: Value): ProfileCollapse =
    val obj = json.obj
    ProfileCollapse(
      aliasProfileUid = obj("aliasProfileUid").str,
      canonicalProfileUid = obj("canonicalProfileUid").str,
      assertedByHuman = obj.get("assertedByHuman").exists(_.bool),
      assertedAtEpochMillis = obj("assertedAtEpochMillis").str.toLong
    )

  private def writeEvent(event: PokerEvent): Value =
    Obj(
      "handId" -> Str(event.handId),
      "sequenceInHand" -> Str(event.sequenceInHand.toString),
      "playerId" -> Str(event.playerId),
      "occurredAtEpochMillis" -> Str(event.occurredAtEpochMillis.toString),
      "street" -> Str(event.street.toString),
      "position" -> Str(event.position.toString),
      "board" -> Arr.from(event.board.cards.map(card => Str(card.toToken))),
      "potBefore" -> Num(event.potBefore),
      "toCall" -> Num(event.toCall),
      "stackBefore" -> Num(event.stackBefore),
      "action" -> writeAction(event.action),
      "decisionTimeMillis" -> event.decisionTimeMillis.map(value => Str(value.toString)).getOrElse(ujson.Null),
      "betHistory" -> Arr.from(event.betHistory.map(writeBetAction))
    )

  private def readEvent(json: Value): PokerEvent =
    val obj = json.obj
    PokerEvent(
      handId = obj("handId").str,
      sequenceInHand = obj("sequenceInHand").str.toLong,
      playerId = obj("playerId").str,
      occurredAtEpochMillis = obj("occurredAtEpochMillis").str.toLong,
      street = Street.valueOf(obj("street").str),
      position = Position.valueOf(obj("position").str),
      board = Board.from(obj("board").arr.toVector.map(value =>
        sicfun.core.Card.parse(value.str).getOrElse(
          throw new IllegalArgumentException(s"invalid card token '${value.str}' in opponent profile store")
        )
      )),
      potBefore = obj("potBefore").num,
      toCall = obj("toCall").num,
      stackBefore = obj("stackBefore").num,
      action = readAction(obj("action")),
      decisionTimeMillis =
        obj.get("decisionTimeMillis")
          .filterNot(_ == ujson.Null)
          .map(_.str.toLong),
      betHistory = obj.get("betHistory").map(_.arr.toVector.map(readBetAction)).getOrElse(Vector.empty)
    )

  private def writeBetAction(action: BetAction): Value =
    Obj(
      "player" -> Num(action.player),
      "action" -> writeAction(action.action)
    )

  private def readBetAction(json: Value): BetAction =
    BetAction(
      player = json("player").num.toInt,
      action = readAction(json("action"))
    )

  private def writeAction(action: PokerAction): Value =
    action match
      case PokerAction.Fold => Obj("kind" -> Str("Fold"))
      case PokerAction.Check => Obj("kind" -> Str("Check"))
      case PokerAction.Call => Obj("kind" -> Str("Call"))
      case PokerAction.Raise(amount) => Obj("kind" -> Str("Raise"), "amount" -> Num(amount))

  private def readAction(json: Value): PokerAction =
    json("kind").str match
      case "Fold" => PokerAction.Fold
      case "Check" => PokerAction.Check
      case "Call" => PokerAction.Call
      case "Raise" => PokerAction.Raise(json("amount").num)
      case other => throw new IllegalArgumentException(s"invalid poker action '$other' in opponent profile store")
