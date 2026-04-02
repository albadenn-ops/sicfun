package sicfun.holdem.history
import sicfun.holdem.analysis.*
import sicfun.holdem.engine.*
import sicfun.holdem.types.*

import sicfun.core.{Card, Metrics}

import ujson.{Arr, Num, Obj, Str, Value}

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path}
import scala.collection.mutable
import scala.util.control.NonFatal

/** Aggregated per-action counters used to derive compact behavioral signatures. */
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

final case class ShowdownRecord(
    handId: String,
    cards: HoleCards
):
  require(handId.trim.nonEmpty, "handId must be non-empty")

final case class ExploitHint(
    ruleId: String,
    text: String,
    metrics: Vector[Double]
):
  require(ruleId.trim.nonEmpty, "ruleId must be non-empty")
  require(text.trim.nonEmpty, "text must be non-empty")
  require(metrics.length == ExploitHint.MetricCount, s"metrics must contain exactly ${ExploitHint.MetricCount} values")
  require(metrics.forall(value => value.isFinite && !value.isNaN), "metrics must be finite")

  def deviationRatioFromGto: Double = metrics(0)

  def evLosing: Double = metrics(1)

  def leakDetectionConfidence: Double = metrics(2)

  def exploitEdgeConfidence: Double = metrics(3)

object ExploitHint:
  val MetricCount = 4

  def fromMetrics(
      ruleId: String,
      text: String,
      deviationRatioFromGto: Double,
      evLosing: Double,
      leakDetectionConfidence: Double,
      exploitEdgeConfidence: Double
  ): ExploitHint =
    ExploitHint(
      ruleId = ruleId,
      text = text,
      metrics = Vector(
        roundMetric(clamp01(deviationRatioFromGto)),
        roundMetric(clampNonNegative(evLosing)),
        roundMetric(clamp01(leakDetectionConfidence)),
        roundMetric(clamp01(exploitEdgeConfidence))
      )
    )

  private def clamp01(value: Double): Double =
    if !value.isFinite || value.isNaN then 0.0
    else math.max(0.0, math.min(1.0, value))

  private def clampNonNegative(value: Double): Double =
    if !value.isFinite || value.isNaN then 0.0
    else math.max(0.0, value)

  private def roundMetric(value: Double): Double =
    math.round(value * 1000.0) / 1000.0

/** Persisted long-term profile for a remembered opponent. */
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
    showdownHands: Vector[ShowdownRecord] = Vector.empty,
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

  def exploitHintDetails: Vector[ExploitHint] =
    OpponentProfile.exploitHintDetailsFor(this)

  def exploitHints: Vector[String] =
    exploitHintDetails.map(_.text)

  def identified: OpponentProfile =
    OpponentIdentity.ensureProfileIdentity(this)

/** In-memory store of remembered opponent profiles, aliases, and collapse assertions. */
object OpponentProfile:
  val MaxRecentEvents = 1024
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
      hand.showdownCards.foreach { (playerName, cards) =>
        if !excludePlayers.contains(playerName) then
          val builder = builders.getOrElseUpdate(playerName, new ProfileBuilder(site, playerName))
          builder.addShowdownRecord(hand.handId, cards)
      }
    }
    builders.valuesIterator.map(_.build()).toVector.sortBy(profile => (-profile.handsObserved, profile.playerName))

  def merge(existing: OpponentProfile, incoming: OpponentProfile): OpponentProfile =
    val existingSeen = existing.seenHandIds.toSet
    val uniqueIncomingHandIds = incoming.seenHandIds.filterNot(existingSeen.contains)
    val mergedShowdownHands = (existing.showdownHands ++ incoming.showdownHands).distinctBy(_.handId)
    val mergedPlayerUid =
      if existing.playerUid == incoming.playerUid then existing.playerUid
      else existing.playerUid.orElse(incoming.playerUid)
    val mergedProfileUid =
      if existing.profileUid == incoming.profileUid then existing.profileUid
      else existing.profileUid.orElse(incoming.profileUid)
    val mergedBehaviorUid =
      if existing.behaviorUid == incoming.behaviorUid then existing.behaviorUid
      else existing.behaviorUid.orElse(incoming.behaviorUid)
    val mergedModelUid =
      if existing.modelUid == incoming.modelUid then existing.modelUid
      else existing.modelUid.orElse(incoming.modelUid)
    if uniqueIncomingHandIds.isEmpty then
      existing.copy(
        firstSeenEpochMillis = math.min(existing.firstSeenEpochMillis, incoming.firstSeenEpochMillis),
        lastSeenEpochMillis = math.max(existing.lastSeenEpochMillis, incoming.lastSeenEpochMillis),
        showdownHands = mergedShowdownHands,
        playerUid = mergedPlayerUid,
        profileUid = mergedProfileUid,
        behaviorUid = mergedBehaviorUid,
        modelUid = mergedModelUid
      )
    else
      val incomingScale = uniqueIncomingHandIds.length.toDouble / math.max(1, incoming.seenHandIds.length).toDouble
      val scaledIncomingActionSummary = scaleActionSummary(incoming.actionSummary, incomingScale)
      val scaledIncomingRaiseResponses = scaleRaiseResponses(incoming.raiseResponses, incomingScale)
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
        actionSummary = existing.actionSummary.merge(scaledIncomingActionSummary),
        raiseResponses = RaiseResponseCounts(
          folds = existing.raiseResponses.folds + scaledIncomingRaiseResponses.folds,
          calls = existing.raiseResponses.calls + scaledIncomingRaiseResponses.calls,
          raises = existing.raiseResponses.raises + scaledIncomingRaiseResponses.raises
        ),
        recentEvents = mergedEvents,
        seenHandIds = mergedSeen,
        showdownHands = mergedShowdownHands,
        playerUid = mergedPlayerUid,
        profileUid = mergedProfileUid,
        behaviorUid = mergedBehaviorUid,
        modelUid = mergedModelUid
      )

  private def scaleActionSummary(summary: OpponentActionSummary, scale: Double): OpponentActionSummary =
    OpponentActionSummary(
      folds = scaleCount(summary.folds, scale),
      raises = scaleCount(summary.raises, scale),
      calls = scaleCount(summary.calls, scale),
      checks = scaleCount(summary.checks, scale),
      callPotOddsTotal = summary.callPotOddsTotal * scale
    )

  private def scaleRaiseResponses(counts: RaiseResponseCounts, scale: Double): RaiseResponseCounts =
    RaiseResponseCounts(
      folds = scaleCount(counts.folds, scale),
      calls = scaleCount(counts.calls, scale),
      raises = scaleCount(counts.raises, scale)
    )

  private def scaleCount(count: Int, scale: Double): Int =
    math.round(count.toDouble * scale).toInt

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
        showdownHands = profile.showdownHands,
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
    private val showdownRecords = mutable.ArrayBuffer.empty[ShowdownRecord]

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

    def addShowdownRecord(handId: String, cards: HoleCards): Unit =
      showdownRecords += ShowdownRecord(handId, cards)

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
        seenHandIds = seenHandIds.toVector.takeRight(MaxSeenHandIds),
        showdownHands = showdownRecords.toVector
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

  private def exploitHintDetailsFor(profile: OpponentProfile): Vector[ExploitHint] =
    val hints = Vector.newBuilder[ExploitHint]

    // ── Per-street / per-context analysis from recent events ──
    val totalActions = math.max(1, profile.actionSummary.totalActions)
    val events = profile.recentEvents
    if events.size >= 10 then
      // River fold rate when facing bets — CFR equilibrium folds ~39% (MDF-correct
      // for mixed bet sizings). 45%+ is exploitably high in HU.
      val riverFacingBet = events.filter(e => e.street == Street.River && e.toCall > 0)
      if riverFacingBet.size >= 5 then
        val riverFoldRate = riverFacingBet.count(_.action == PokerAction.Fold).toDouble / riverFacingBet.size
        if riverFoldRate >= 0.45 then
          hints += exploitHint(
            ruleId = "overfold-river-aggression",
            text = "Over-folds on the river facing aggression; increase bluff pressure on late streets.",
            deviationRatioFromGto = upwardDeviationRatio(riverFoldRate, gtoBaseline = 0.39),
            sampleSize = riverFacingBet.size,
            minSample = 5,
            evScaleBb = 0.95
          )

      // Call rate facing large bets (bet/pot-before-bet >= 0.6)
      val facingLargeBet = events.filter { e =>
        e.toCall > 0 && e.potBefore > e.toCall && e.toCall / (e.potBefore - e.toCall) >= 0.6
      }
      if facingLargeBet.size >= 5 then
        val largeBetCallRate = facingLargeBet.count(_.action == PokerAction.Call).toDouble / facingLargeBet.size
        if largeBetCallRate >= 0.60 then
          hints += exploitHint(
            ruleId = "overcall-big-bets",
            text = "Calls too often facing large bets; value bet thinner and cut bluffs.",
            deviationRatioFromGto = upwardDeviationRatio(largeBetCallRate, gtoBaseline = 0.48),
            sampleSize = facingLargeBet.size,
            minSample = 5,
            evScaleBb = 0.9
          )

      // Turn aggression rate — 25%+ suggests excessive turn betting (HU baseline ~20%)
      val turnEvents = events.filter(_.street == Street.Turn)
      if turnEvents.size >= 5 then
        val turnRaiseRate = turnEvents.count(_.action.category == PokerAction.Category.Raise).toDouble / turnEvents.size
        if turnRaiseRate >= 0.25 then
          hints += exploitHint(
            ruleId = "overbluff-turn-barrel",
            text = "Very aggressive on the turn; bluff-catch more selectively against turn barrels.",
            deviationRatioFromGto = upwardDeviationRatio(turnRaiseRate, gtoBaseline = 0.18),
            sampleSize = turnEvents.size,
            minSample = 5,
            evScaleBb = 0.85
          )

      // Big pot passivity removed (2026-03-17, CFR calibration):
      // HU equilibrium play checks ~100% in big pots (SPR < 4, toCall = 0) — checking
      // to induce, protect range, and manage pot is the GTO strategy. The PassiveInBigPots
      // leak (checking strong hands) is not distinguishable from GTO by action frequency
      // alone — it requires hand-strength-aware analysis.

      // Preflop fold rate — baseline ~13%, anything above 20% is exploitably tight
      val preflopEvents = events.filter(_.street == Street.Preflop)
      if preflopEvents.size >= 8 then
        val preflopFoldRate = preflopEvents.count(_.action == PokerAction.Fold).toDouble / preflopEvents.size
        if preflopFoldRate >= 0.20 then
          hints += exploitHint(
            ruleId = "preflop-too-tight",
            text = "Over-folds preflop; Open slightly wider against this player.",
            deviationRatioFromGto = upwardDeviationRatio(preflopFoldRate, gtoBaseline = 0.13),
            sampleSize = preflopEvents.size,
            minSample = 8,
            evScaleBb = 0.45
          )
        // Preflop call rate — HU BB baseline ~55%, 65%+ means too passive/calling
        val preflopCallRate = preflopEvents.count(_.action == PokerAction.Call).toDouble / preflopEvents.size
        if preflopCallRate >= 0.65 then
          hints += exploitHint(
            ruleId = "preflop-too-loose",
            text = "Calls too loose preflop; value bet wider preflop ranges.",
            deviationRatioFromGto = upwardDeviationRatio(preflopCallRate, gtoBaseline = 0.55),
            sampleSize = preflopEvents.size,
            minSample = 8,
            evScaleBb = 0.5
          )

    // ── Aggregate raise response analysis ──
    val signature = profile.signature
    val responseTotal = profile.raiseResponses.total.toDouble
    if responseTotal >= 4.0 then
      val foldToRaise = profile.raiseResponses.folds / responseTotal
      val callVsRaise = profile.raiseResponses.calls / responseTotal
      val reraiseVsRaise = profile.raiseResponses.raises / responseTotal
      if foldToRaise >= 0.55 then
        hints += exploitHint(
          ruleId = "fold-to-raise",
          text = "Increase bluff pressure when they face a raise.",
          deviationRatioFromGto = upwardDeviationRatio(foldToRaise, gtoBaseline = 0.45),
          sampleSize = responseTotal.toInt,
          minSample = 4,
          evScaleBb = 0.7
        )
      if callVsRaise >= 0.55 then
        hints += exploitHint(
          ruleId = "call-vs-raise",
          text = "Value bet thinner and cut back low-equity bluffs.",
          deviationRatioFromGto = upwardDeviationRatio(callVsRaise, gtoBaseline = 0.45),
          sampleSize = responseTotal.toInt,
          minSample = 4,
          evScaleBb = 0.75
        )
      if reraiseVsRaise >= 0.28 then
        hints += exploitHint(
          ruleId = "reraise-vs-raise",
          text = "Expect re-raises; trap stronger hands and avoid thin bluffs.",
          deviationRatioFromGto = upwardDeviationRatio(reraiseVsRaise, gtoBaseline = 0.18),
          sampleSize = responseTotal.toInt,
          minSample = 4,
          evScaleBb = 0.65
        )

    showdownHintsFor(profile).foreach(hints += _)

    // ── Overall signature analysis ──
    val foldRate = signature.values(0)
    val raiseRate = signature.values(1)
    val callRate = signature.values(2)
    val checkRate = signature.values(3)
    if foldRate >= 0.45 then
      hints += exploitHint(
        ruleId = "overall-overfold",
        text = "Open slightly wider against them; they over-fold overall.",
        deviationRatioFromGto = upwardDeviationRatio(foldRate, gtoBaseline = 0.35),
        sampleSize = totalActions,
        minSample = 10,
        evScaleBb = 0.45
      )
    if callRate >= 0.42 && raiseRate <= 0.12 then
      hints += exploitHint(
        ruleId = "calling-station-profile",
        text = "Treat them like a calling station until shown otherwise.",
        deviationRatioFromGto = minDeviationRatio(
          upwardDeviationRatio(callRate, gtoBaseline = 0.32),
          downwardDeviationRatio(raiseRate, gtoBaseline = 0.18)
        ),
        sampleSize = totalActions,
        minSample = 10,
        evScaleBb = 0.8
      )
    if raiseRate >= 0.33 then
      hints += exploitHint(
        ruleId = "aggressive-lines",
        text = "They choose aggressive lines often; bluff-catch more selectively.",
        deviationRatioFromGto = upwardDeviationRatio(raiseRate, gtoBaseline = 0.23),
        sampleSize = totalActions,
        minSample = 10,
        evScaleBb = 0.65
      )
    if checkRate >= 0.30 && raiseRate <= 0.10 then
      hints += exploitHint(
        ruleId = "passive-lines",
        text = "Respect sudden aggression from their passive lines.",
        deviationRatioFromGto = minDeviationRatio(
          upwardDeviationRatio(checkRate, gtoBaseline = 0.20),
          downwardDeviationRatio(raiseRate, gtoBaseline = 0.16)
        ),
        sampleSize = totalActions,
        minSample = 10,
        evScaleBb = 0.4
      )
    profile.stability.foreach { stability =>
      if stability.significantChanges > 0 || stability.meanDrift >= 0.20 then
        hints += exploitHint(
          ruleId = "profile-drifting",
          text = "Profile is drifting; lower exploit confidence and stay closer to baseline.",
          deviationRatioFromGto = math.max(
            upwardDeviationRatio(stability.meanDrift, gtoBaseline = 0.10),
            clamp01(stability.significantChanges.toDouble / math.max(1.0, stability.windowCount.toDouble))
          ),
          sampleSize = events.size,
          minSample = 6,
          evScaleBb = 0.3
        )
    }
    val built = hints.result()
      .distinctBy(_.ruleId)
      .sortBy(hint => (-hint.leakDetectionConfidence, -hint.exploitEdgeConfidence, hint.ruleId))
    if built.nonEmpty then built.take(5)
    else
      Vector(
        ExploitHint.fromMetrics(
          ruleId = "baseline-no-exploit",
          text = "No high-confidence exploit yet; use baseline strategy with mild archetype bias.",
          deviationRatioFromGto = 0.0,
          evLosing = 0.0,
          leakDetectionConfidence = 0.0,
          exploitEdgeConfidence = 0.0
        )
      )

  private def showdownHintsFor(profile: OpponentProfile): Vector[ExploitHint] =
    val showdowns = profile.showdownHands
    if showdowns.size < 3 then Vector.empty
    else
      val total = showdowns.size.toDouble
      val classifications = showdowns.map(record => ShowdownHandClass.classify(record.cards))
      val hints = Vector.newBuilder[ExploitHint]
      val premiumRatio = classifications.count(_ == ShowdownHandClass.PremiumPair).toDouble / total
      val weakRatio = classifications.count(_ == ShowdownHandClass.Weak).toDouble / total
      val pairRatio = showdowns.count(record => ShowdownHandClass.isPair(record.cards)).toDouble / total
      val responseTotal = profile.raiseResponses.total.toDouble
      val callVsRaise =
        if responseTotal <= 0.0 then 0.0
        else profile.raiseResponses.calls / responseTotal
      val callRate = profile.signature.values(2)
      val raiseRate = profile.signature.values(1)
      val weakShowdownCorroborated =
        (responseTotal >= 4.0 && callVsRaise >= 0.45) ||
          (callRate >= 0.18 && raiseRate <= 0.18)

      if premiumRatio >= 0.5 then
        hints += exploitHint(
          ruleId = "showdown-premium-heavy",
          text = "Shows down premium hands frequently; likely playing a tight value-heavy range.",
          deviationRatioFromGto = upwardDeviationRatio(premiumRatio, gtoBaseline = 0.25),
          sampleSize = showdowns.size,
          minSample = 3,
          evScaleBb = 0.55
        )

      if showdowns.size >= 5 && weakRatio >= 0.60 && weakShowdownCorroborated then
        hints += exploitHint(
          ruleId = "showdown-weak-heavy",
          text = "Has shown down weak hands; likely stations or bluffs reaching showdown - value bet thinner.",
          deviationRatioFromGto = minDeviationRatio(
            upwardDeviationRatio(weakRatio, gtoBaseline = 0.25),
            math.max(
              upwardDeviationRatio(callVsRaise, gtoBaseline = 0.30),
              minDeviationRatio(
                upwardDeviationRatio(callRate, gtoBaseline = 0.12),
                downwardDeviationRatio(raiseRate, gtoBaseline = 0.22)
              )
            )
          ),
          sampleSize = showdowns.size,
          minSample = 5,
          evScaleBb = 0.8
        )

      if showdowns.size >= 5 && pairRatio >= 0.50 then
        hints += exploitHint(
          ruleId = "showdown-pair-heavy",
          text = "Showdown history is pair-heavy; range may be set-mining oriented.",
          deviationRatioFromGto = upwardDeviationRatio(pairRatio, gtoBaseline = 0.20),
          sampleSize = showdowns.size,
          minSample = 5,
          evScaleBb = 0.45
        )

      hints.result()

  private def exploitHint(
      ruleId: String,
      text: String,
      deviationRatioFromGto: Double,
      sampleSize: Int,
      minSample: Int,
      evScaleBb: Double
  ): ExploitHint =
    val normalizedDeviation = clamp01(deviationRatioFromGto)
    val sampleReliability = clamp01(sampleSize.toDouble / math.max(1.0, minSample.toDouble * 3.0))
    val detectionConfidence = clamp01(math.sqrt(sampleReliability) * (0.35 + (0.65 * normalizedDeviation)))
    val exploitEdgeConfidence = clamp01((0.55 * detectionConfidence) + (0.45 * normalizedDeviation))
    ExploitHint.fromMetrics(
      ruleId = ruleId,
      text = text,
      deviationRatioFromGto = normalizedDeviation,
      evLosing = evScaleBb * normalizedDeviation * exploitEdgeConfidence,
      leakDetectionConfidence = detectionConfidence,
      exploitEdgeConfidence = exploitEdgeConfidence
    )

  private def upwardDeviationRatio(observed: Double, gtoBaseline: Double): Double =
    clamp01((observed - gtoBaseline) / math.max(1e-9, 1.0 - gtoBaseline))

  private def downwardDeviationRatio(observed: Double, gtoBaseline: Double): Double =
    clamp01((gtoBaseline - observed) / math.max(1e-9, gtoBaseline))

  private def minDeviationRatio(parts: Double*): Double =
    if parts.isEmpty then 0.0
    else clamp01(parts.min)

  private def clamp01(value: Double): Double =
    if !value.isFinite || value.isNaN then 0.0
    else math.max(0.0, math.min(1.0, value))

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
  private[history] var isNormalized: Boolean = false

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

  val empty: OpponentProfileStore = markNormalized(OpponentProfileStore(Vector.empty))

  def load(path: Path): OpponentProfileStore =
    if !Files.exists(path) then empty
    else
      try
        val json = ujson.read(Files.readString(path, StandardCharsets.UTF_8))
        readStore(json)
      catch
        case NonFatal(error) =>
          System.err.println(s"warning: failed to load opponent profile store from $path: ${readErrorMessage(error)}")
          empty

  def save(path: Path, store: OpponentProfileStore): Unit =
    Option(path.getParent).foreach(parent => Files.createDirectories(parent))
    Files.writeString(path, ujson.write(writeStore(normalize(store)), indent = 2), StandardCharsets.UTF_8)

  private[history] def key(site: String, playerName: String): String =
    OpponentIdentity.aliasKey(site, playerName)

  private[history] def normalize(store: OpponentProfileStore): OpponentProfileStore =
    if store.isNormalized then store
    else
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

      markNormalized(
        OpponentProfileStore(
          profiles = mergedProfiles.values.toVector.sortBy(profile => (profile.site, profile.playerName, profile.profileUid.getOrElse(""))),
          players = playerByUid.values.toVector.sortBy(player => (player.canonicalSite, player.canonicalName, player.playerUid)),
          aliases = aliasByKey.values.toVector.sortBy(alias => (alias.site, alias.playerName, alias.playerUid)),
          playerCollapses = dedupePlayerCollapses(store.playerCollapses),
          profileCollapses = dedupeProfileCollapses(store.profileCollapses)
        )
      )

  private def markNormalized(store: OpponentProfileStore): OpponentProfileStore =
    store.isNormalized = true
    store

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
    try
      val obj = json.obj
      readStoreVersion(obj) match
        case Some(FormatVersion) =>
          normalize(
            OpponentProfileStore(
              profiles = readProfiles(obj),
              players = readPlayers(obj),
              aliases = readAliases(obj),
              playerCollapses = readPlayerCollapses(obj),
              profileCollapses = readProfileCollapses(obj)
            )
          )
        case Some(LegacyFormatVersion) =>
          normalize(
            OpponentProfileStore(
              profiles = readProfiles(obj)
            )
          )
        case Some(other) =>
          System.err.println(s"warning: unsupported opponent profile store version '$other'; loading empty store")
          empty
        case None =>
          empty
    catch
      case NonFatal(error) =>
        System.err.println(s"warning: invalid opponent profile store payload: ${readErrorMessage(error)}")
        empty

  private def readStoreVersion(obj: collection.Map[String, Value]): Option[String] =
    obj.get("formatVersion") match
      case None => Some(LegacyFormatVersion)
      case Some(value) =>
        try Some(value.str)
        catch
          case NonFatal(error) =>
            System.err.println(s"warning: invalid opponent profile store formatVersion: ${readErrorMessage(error)}")
            None

  private def readProfiles(obj: collection.Map[String, Value]): Vector[OpponentProfile] =
    readEntries(obj, fieldName = "profiles", entryLabel = "opponent profile")(readProfile)

  private def readPlayers(obj: collection.Map[String, Value]): Vector[RememberedPlayer] =
    readEntries(obj, fieldName = "players", entryLabel = "remembered player")(readPlayer)

  private def readAliases(obj: collection.Map[String, Value]): Vector[PlayerAlias] =
    readEntries(obj, fieldName = "aliases", entryLabel = "player alias")(readAlias)

  private def readPlayerCollapses(obj: collection.Map[String, Value]): Vector[PlayerCollapse] =
    readEntries(obj, fieldName = "playerCollapses", entryLabel = "player collapse")(readPlayerCollapse)

  private def readProfileCollapses(obj: collection.Map[String, Value]): Vector[ProfileCollapse] =
    readEntries(obj, fieldName = "profileCollapses", entryLabel = "profile collapse")(readProfileCollapse)

  private def readEntries[T](
      obj: collection.Map[String, Value],
      fieldName: String,
      entryLabel: String
  )(reader: Value => T): Vector[T] =
    obj.get(fieldName) match
      case None => Vector.empty
      case Some(value) =>
        val entries =
          try value.arr.toVector
          catch
            case NonFatal(error) =>
              System.err.println(
                s"warning: skipping corrupt opponent profile store field '$fieldName': ${readErrorMessage(error)}"
              )
              return Vector.empty
        entries.zipWithIndex.flatMap { case (entryJson, index) =>
          try Some(reader(entryJson))
          catch
            case NonFatal(error) =>
              System.err.println(s"warning: skipping corrupt $entryLabel at index $index: ${readErrorMessage(error)}")
              None
        }

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
      "showdownHands" -> Arr.from(profile.showdownHands.map(record =>
        Obj("handId" -> Str(record.handId), "cards" -> Str(record.cards.toToken))
      )),
      "playerUid" -> profile.playerUid.map(Str(_)).getOrElse(ujson.Null),
      "profileUid" -> profile.profileUid.map(Str(_)).getOrElse(ujson.Null),
      "behaviorUid" -> profile.behaviorUid.map(Str(_)).getOrElse(ujson.Null),
      "modelUid" -> profile.modelUid.map(Str(_)).getOrElse(ujson.Null)
    )

  private[history] def readProfile(json: Value): OpponentProfile =
    val obj = json.obj
    val site = obj("site").str
    val playerName = obj("playerName").str
    OpponentProfile(
      site = site,
      playerName = playerName,
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
      recentEvents = obj.get("recentEvents").map(value => readRecentEvents(value.arr, site, playerName)).getOrElse(Vector.empty),
      seenHandIds = obj.get("seenHandIds").map(_.arr.toVector.map(_.str)).getOrElse(Vector.empty),
      showdownHands = obj.get("showdownHands").map(_.arr.toVector.map { recordJson =>
        val recordObj = recordJson.obj
        val token = recordObj("cards").str
        if token.length != 4 then throw new IllegalArgumentException(s"invalid showdown card token length: $token")
        val cards = Card.parseAll(Seq(token.substring(0, 2), token.substring(2, 4))).map(HoleCards.from).getOrElse(
          throw new IllegalArgumentException(s"invalid showdown cards token: $token")
        )
        ShowdownRecord(handId = recordObj("handId").str, cards = cards)
      }).getOrElse(Vector.empty),
      playerUid = obj.get("playerUid").filterNot(_ == ujson.Null).map(_.str),
      profileUid = obj.get("profileUid").filterNot(_ == ujson.Null).map(_.str),
      behaviorUid = obj.get("behaviorUid").filterNot(_ == ujson.Null).map(_.str),
      modelUid = obj.get("modelUid").filterNot(_ == ujson.Null).map(_.str)
    )

  private def readRecentEvents(json: collection.IndexedSeq[Value], site: String, playerName: String): Vector[PokerEvent] =
    json.toVector.zipWithIndex.flatMap { case (eventJson, index) =>
      try Some(readEvent(eventJson))
      catch
        case NonFatal(error) =>
          System.err.println(
            s"warning: skipping corrupt opponent event for $site/$playerName at index $index: ${readErrorMessage(error)}"
          )
          None
    }

  private def readErrorMessage(error: Throwable): String =
    Option(error.getMessage).map(_.trim).filter(_.nonEmpty).getOrElse(error.getClass.getSimpleName)

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
