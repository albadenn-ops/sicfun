package sicfun.holdem.web

import sicfun.holdem.engine.RealTimeAdaptiveEngine
import sicfun.holdem.equity.{TableFormat, TableRanges}
import sicfun.holdem.history.{ExploitHint, HandHistoryImport, HandHistorySite, ImportedHand, OpponentProfile}
import sicfun.holdem.model.{PokerActionModel, PokerActionModelArtifactIO}
import sicfun.holdem.runtime.HandHistoryAnalyzer
import sicfun.holdem.types.PokerAction

import ujson.{Arr, Num, Obj, Str, Value}

import java.nio.charset.StandardCharsets
import java.nio.file.Path
import java.util.Locale
import scala.collection.mutable
import scala.util.Random

/** Core hand-history analysis service behind the review web API.
  *
  * Orchestrates the full analysis pipeline when a user uploads a hand history:
  *   1. Normalize hero name (strip forum suffixes)
  *   2. Parse the hand history text via [[HandHistoryImport]]
  *   3. For each imported hand: resolve hero identity, extract hero decisions,
  *      build a [[RealTimeAdaptiveEngine]] per hand, compute per-decision EV analysis
  *   4. Build opponent profiles from the imported hands
  *   5. Assemble the [[AnalysisResponse]] with decisions, opponents, warnings, and trace
  *
  * The service is stateless (no mutable state between analyze() calls) and thread-safe.
  * Each call creates fresh engine instances with the configured parameters.
  *
  * @param config      service configuration (MC trial counts, time budget, max decisions)
  * @param actionModel the trained PokerActionModel for range inference
  * @param modelSource human-readable description of the model source (e.g. "artifact:v2")
  */
final class HandHistoryReviewService private (
    config: HandHistoryReviewService.ServiceConfig,
    actionModel: PokerActionModel,
    modelSource: String
):
  import HandHistoryReviewService.*

  /** Analyze a hand history upload and return a structured response.
    *
    * @param request the analysis request containing the raw hand history text, optional site, and hero name
    * @return Right(response) with per-decision analysis, opponent profiles, and trace data,
    *         or Left(error) if parsing or analysis fails
    */
  def analyze(request: AnalysisRequest): Either[String, AnalysisResponse] =
    val normalizedHeroName = request.heroName.map(HandHistoryImport.normalizePlayerName).filter(_.nonEmpty)
    val normalizedRequest = request.copy(
      heroName = normalizedHeroName
    )
    val requestTrace = RequestTrace(
      rawHeroName = request.heroName,
      normalizedHeroName = normalizedHeroName,
      requestedSite = request.site.map(_.toString),
      handHistoryBytes = normalizedRequest.handHistoryText.getBytes(StandardCharsets.UTF_8).length
    )
    if normalizedRequest.handHistoryText.trim.isEmpty then Left("handHistoryText must be non-empty")
    else
      for
        imported <- HandHistoryImport.parseText(
          text = normalizedRequest.handHistoryText,
          site = normalizedRequest.site,
          heroName = normalizedRequest.heroName
        )
        response <- buildResponse(imported, normalizedRequest, requestTrace)
      yield response

  /** Serialize an AnalysisResponse to a ujson Value for HTTP response rendering. */
  def writeJson(response: AnalysisResponse): Value =
    Obj(
      "site" -> Str(response.site),
      "heroName" -> response.heroName.map(Str(_)).getOrElse(ujson.Null),
      "handsImported" -> Num(response.handsImported),
      "handsAnalyzed" -> Num(response.handsAnalyzed),
      "handsSkipped" -> Num(response.handsSkipped),
      "decisionsAnalyzed" -> Num(response.decisionsAnalyzed),
      "mistakes" -> Num(response.mistakes),
      "totalEvLost" -> Num(response.totalEvLost),
      "biggestMistakeEv" -> Num(response.biggestMistakeEv),
      "modelSource" -> Str(response.modelSource),
      "warnings" -> Arr.from(response.warnings.map(Str(_))),
      "decisions" -> Arr.from(response.decisions.map(writeDecision)),
      "opponents" -> Arr.from(response.opponents.map(writeOpponent)),
      "trace" -> writeTrace(response.trace)
    )

  /** Build the full analysis response from imported hands.
    *
    * Analyzes each hand individually (skipping hands without hero hole cards),
    * aggregates decisions and warnings, builds opponent profiles, and assembles
    * the comprehensive trace for debugging and verification.
    */
  private def buildResponse(
      imported: Vector[ImportedHand],
      request: AnalysisRequest,
      requestTrace: RequestTrace
  ): Either[String, AnalysisResponse] =
    if imported.isEmpty then Left("no hands were imported from the uploaded text")
    else
      try
        val warnings = mutable.ArrayBuffer.empty[String]
        val rng = new Random(config.seed)

        val handResults = imported.map { hand =>
          analyzeHand(hand, request.heroName, warnings, rng)
        }
        val analyzedHands = handResults.filter(_.trace.status == "analyzed")
        val allDecisions = handResults.flatMap(_.decisions)
        val handsAnalyzed = analyzedHands.length
        val handsSkipped = math.max(0, imported.length - handsAnalyzed)
        val effectiveHero = request.heroName.orElse(resolveHeroName(imported))
        val excludedHeroes = request.heroName.toSet ++ imported.flatMap(_.heroName).toSet
        val profiles = OpponentProfile.fromImportedHands(
          site = imported.head.site.toString.toLowerCase,
          hands = imported,
          excludePlayers = excludedHeroes
        )
        val totalEvLostRaw = math.abs(allDecisions.iterator.map(d => math.min(0.0, d.evDifference)).sum)
        val biggestMistake =
          if allDecisions.isEmpty then 0.0
          else allDecisions.iterator.map(d => math.abs(d.evDifference)).max
        val warningList = warnings.toVector.distinct
        val totalEvLost = roundChips(totalEvLostRaw)
        val biggestMistakeRounded = roundChips(biggestMistake)
        val trace = AnalysisTrace(
          request = requestTrace,
          importStage = ImportTrace(
            handsImported = imported.length,
            siteResolved = imported.headOption.map(_.site.toString),
            heroNameResolved = effectiveHero,
            distinctPlayersObserved = imported.iterator.flatMap(_.players.iterator.map(_.name)).toSet.size
          ),
          hands = handResults.map(_.trace),
          summary = SummaryTrace(
            handsImported = imported.length,
            handsAnalyzed = handsAnalyzed,
            handsSkipped = handsSkipped,
            decisionsAnalyzed = allDecisions.length,
            mistakes = allDecisions.count(HandHistoryAnalyzer.countsAsMistake),
            totalEvLost = totalEvLost,
            biggestMistakeEv = biggestMistakeRounded,
            warningCount = warningList.length,
            opponentsProfiled = profiles.length
          )
        )

        Right(
          AnalysisResponse(
            site = imported.head.site.toString,
            heroName = effectiveHero,
            handsImported = imported.length,
            handsAnalyzed = handsAnalyzed,
            handsSkipped = handsSkipped,
            decisionsAnalyzed = allDecisions.length,
            mistakes = allDecisions.count(HandHistoryAnalyzer.countsAsMistake),
            totalEvLost = totalEvLost,
            biggestMistakeEv = biggestMistakeRounded,
            modelSource = modelSource,
            warnings = warningList,
            decisions = allDecisions
              .sortBy(d => (-math.abs(d.evDifference), d.handId, d.street.toString))
              .take(config.maxDecisions)
              .map(writeDecisionView),
            opponents = profiles.take(5).map(writeOpponentView),
            trace = trace
          )
        )
      catch
        case e: Exception => Left(s"analysis failed: ${e.getMessage}")

  private final case class HandResult(
      handId: String,
      decisions: Vector[HandHistoryAnalyzer.AnalyzedDecision],
      trace: HandTrace
  )

  private def analyzeHand(
      hand: ImportedHand,
      heroNameOverride: Option[String],
      warnings: mutable.ArrayBuffer[String],
      rng: Random
  ): HandResult =
    val heroName = resolveHeroPlayerId(hand, heroNameOverride).orElse(hand.heroName)
    (heroName, hand.heroHoleCards) match
      case (None, _) =>
        val warning = s"hand ${hand.handId}: skipped because hero name could not be inferred"
        warnings += warning
        HandResult(
          handId = hand.handId,
          decisions = Vector.empty,
          trace = HandTrace(
            handId = hand.handId,
            status = "skipped",
            playerCount = hand.players.length,
            heroNameResolved = None,
            heroCardsPresent = false,
            decisionsAnalyzed = 0,
            skipReason = Some("hero_name_unresolved"),
            warning = Some(warning)
          )
        )
      case (_, None) =>
        val warning = s"hand ${hand.handId}: skipped because hero hole cards were not present in the uploaded history"
        warnings += warning
        HandResult(
          handId = hand.handId,
          decisions = Vector.empty,
          trace = HandTrace(
            handId = hand.handId,
            status = "skipped",
            playerCount = hand.players.length,
            heroNameResolved = heroName,
            heroCardsPresent = false,
            decisionsAnalyzed = 0,
            skipReason = Some("hero_hole_cards_missing"),
            warning = Some(warning)
          )
        )
      case (Some(heroPlayerId), Some(heroCards)) =>
        val tableRanges = TableRanges.defaults(TableFormat.forPlayerCount(hand.players.length))
        val availablePositions = hand.players.iterator.map(_.position).toSet
        val decisions = HandHistoryAnalyzer.analyzeWithHeroCards(
          events = hand.events,
          heroPlayerId = heroPlayerId,
          heroCards = heroCards,
          engine = newEngine(tableRanges),
          tableRanges = tableRanges,
          availablePositions = availablePositions,
          budgetMs = config.budgetMs,
          rng = new Random(rng.nextLong())
        )
        val warning =
          if decisions.isEmpty then
            Some(s"hand ${hand.handId}: imported but no hero decisions could be analyzed")
          else None
        warning.foreach(warnings += _)
        HandResult(
          handId = hand.handId,
          decisions = decisions,
          trace = HandTrace(
            handId = hand.handId,
            status = "analyzed",
            playerCount = hand.players.length,
            heroNameResolved = Some(heroPlayerId),
            heroCardsPresent = true,
            decisionsAnalyzed = decisions.length,
            skipReason = None,
            warning = warning
          )
        )

  private def resolveHeroPlayerId(
      hand: ImportedHand,
      heroNameOverride: Option[String]
  ): Option[String] =
    heroNameOverride
      .map(HandHistoryImport.normalizePlayerName)
      .flatMap { requested =>
        hand.players
          .find(_.name == requested)
          .map(_.name)
          .orElse(hand.players.find(_.name.equalsIgnoreCase(requested)).map(_.name))
      }

  private def newEngine(tableRanges: TableRanges): RealTimeAdaptiveEngine =
    new RealTimeAdaptiveEngine(
      tableRanges = tableRanges,
      actionModel = actionModel,
      bunchingTrials = config.bunchingTrials,
      defaultEquityTrials = config.equityTrials,
      minEquityTrials = math.max(200, config.equityTrials / 10)
    )

  private def resolveHeroName(imported: Vector[ImportedHand]): Option[String] =
    imported.flatMap(_.heroName).distinct match
      case Vector(single) => Some(single)
      case _ => None

  private def writeDecisionView(
      decision: HandHistoryAnalyzer.AnalyzedDecision
  ): DecisionView =
    DecisionView(
      handId = decision.handId,
      street = decision.street.toString,
      heroCards = decision.heroCards.map(_.toToken),
      actualAction = renderAction(decision.actualAction),
      recommendedAction = renderAction(decision.recommendedAction),
      actualEv = roundChips(decision.actualEv),
      recommendedEv = roundChips(decision.recommendedEv),
      evDifference = roundChips(decision.evDifference),
      heroEquityMean = roundChips(decision.heroEquityMean)
    )

  private def writeOpponentView(profile: OpponentProfile): OpponentView =
    OpponentView(
      playerName = profile.playerName,
      handsObserved = profile.handsObserved,
      archetype = profile.archetypePosterior.mapEstimate.toString,
      hints = profile.exploitHintDetails.take(3).map(writeHintView)
    )

  private def writeHintView(hint: ExploitHint): HintView =
    HintView(
      ruleId = hint.ruleId,
      text = hint.text,
      metrics = hint.metrics
    )

  private def writeDecision(decision: DecisionView): Value =
    Obj(
      "handId" -> Str(decision.handId),
      "street" -> Str(decision.street),
      "heroCards" -> decision.heroCards.map(Str(_)).getOrElse(ujson.Null),
      "actualAction" -> Str(decision.actualAction),
      "recommendedAction" -> Str(decision.recommendedAction),
      "actualEv" -> Num(decision.actualEv),
      "recommendedEv" -> Num(decision.recommendedEv),
      "evDifference" -> Num(decision.evDifference),
      "heroEquityMean" -> Num(decision.heroEquityMean)
    )

  private def writeOpponent(opponent: OpponentView): Value =
    Obj(
      "playerName" -> Str(opponent.playerName),
      "handsObserved" -> Num(opponent.handsObserved),
      "archetype" -> Str(opponent.archetype),
      "hints" -> Arr.from(opponent.hints.map(writeHint))
    )

  private def writeHint(hint: HintView): Value =
    Obj(
      "ruleId" -> Str(hint.ruleId),
      "text" -> Str(hint.text),
      "metrics" -> Arr.from(hint.metrics.map(Num(_)))
    )

  private def writeTrace(trace: AnalysisTrace): Value =
    Obj(
      "request" -> Obj(
        "rawHeroName" -> trace.request.rawHeroName.map(Str(_)).getOrElse(ujson.Null),
        "normalizedHeroName" -> trace.request.normalizedHeroName.map(Str(_)).getOrElse(ujson.Null),
        "requestedSite" -> trace.request.requestedSite.map(Str(_)).getOrElse(ujson.Null),
        "handHistoryBytes" -> Num(trace.request.handHistoryBytes)
      ),
      "import" -> Obj(
        "handsImported" -> Num(trace.importStage.handsImported),
        "siteResolved" -> trace.importStage.siteResolved.map(Str(_)).getOrElse(ujson.Null),
        "heroNameResolved" -> trace.importStage.heroNameResolved.map(Str(_)).getOrElse(ujson.Null),
        "distinctPlayersObserved" -> Num(trace.importStage.distinctPlayersObserved)
      ),
      "hands" -> Arr.from(trace.hands.map(writeHandTrace)),
      "summary" -> Obj(
        "handsImported" -> Num(trace.summary.handsImported),
        "handsAnalyzed" -> Num(trace.summary.handsAnalyzed),
        "handsSkipped" -> Num(trace.summary.handsSkipped),
        "decisionsAnalyzed" -> Num(trace.summary.decisionsAnalyzed),
        "mistakes" -> Num(trace.summary.mistakes),
        "totalEvLost" -> Num(trace.summary.totalEvLost),
        "biggestMistakeEv" -> Num(trace.summary.biggestMistakeEv),
        "warningCount" -> Num(trace.summary.warningCount),
        "opponentsProfiled" -> Num(trace.summary.opponentsProfiled)
      )
    )

  private def writeHandTrace(trace: HandTrace): Value =
    Obj(
      "handId" -> Str(trace.handId),
      "status" -> Str(trace.status),
      "playerCount" -> Num(trace.playerCount),
      "heroNameResolved" -> trace.heroNameResolved.map(Str(_)).getOrElse(ujson.Null),
      "heroCardsPresent" -> ujson.Bool(trace.heroCardsPresent),
      "decisionsAnalyzed" -> Num(trace.decisionsAnalyzed),
      "skipReason" -> trace.skipReason.map(Str(_)).getOrElse(ujson.Null),
      "warning" -> trace.warning.map(Str(_)).getOrElse(ujson.Null)
    )

  private def renderAction(action: PokerAction): String =
    action match
      case PokerAction.Fold => "Fold"
      case PokerAction.Check => "Check"
      case PokerAction.Call => "Call"
      case PokerAction.Raise(amount) => String.format(Locale.ROOT, "Raise(%.2f)", java.lang.Double.valueOf(amount))

  private def roundChips(amount: Double): Double =
    math.round(amount * 100.0) / 100.0

object HandHistoryReviewService:
  /** Runtime configuration for [[HandHistoryReviewService]]. */
  final case class ServiceConfig(
      modelDir: Option[Path] = None,
      seed: Long = 42L,
      bunchingTrials: Int = 200,
      equityTrials: Int = 2000,
      budgetMs: Long = 1500L,
      maxDecisions: Int = 12
  )

  final case class AnalysisRequest(
      handHistoryText: String,
      site: Option[HandHistorySite],
      heroName: Option[String]
  )

  final case class RequestTrace(
      rawHeroName: Option[String],
      normalizedHeroName: Option[String],
      requestedSite: Option[String],
      handHistoryBytes: Int
  )

  final case class ImportTrace(
      handsImported: Int,
      siteResolved: Option[String],
      heroNameResolved: Option[String],
      distinctPlayersObserved: Int
  )

  final case class HandTrace(
      handId: String,
      status: String,
      playerCount: Int,
      heroNameResolved: Option[String],
      heroCardsPresent: Boolean,
      decisionsAnalyzed: Int,
      skipReason: Option[String],
      warning: Option[String]
  )

  final case class SummaryTrace(
      handsImported: Int,
      handsAnalyzed: Int,
      handsSkipped: Int,
      decisionsAnalyzed: Int,
      mistakes: Int,
      totalEvLost: Double,
      biggestMistakeEv: Double,
      warningCount: Int,
      opponentsProfiled: Int
  )

  final case class AnalysisTrace(
      request: RequestTrace,
      importStage: ImportTrace,
      hands: Vector[HandTrace],
      summary: SummaryTrace
  )

  final case class DecisionView(
      handId: String,
      street: String,
      heroCards: Option[String],
      actualAction: String,
      recommendedAction: String,
      actualEv: Double,
      recommendedEv: Double,
      evDifference: Double,
      heroEquityMean: Double
  )

  final case class OpponentView(
      playerName: String,
      handsObserved: Int,
      archetype: String,
      hints: Vector[HintView]
  )

  final case class HintView(
      ruleId: String,
      text: String,
      metrics: Vector[Double]
  )

  final case class AnalysisResponse(
      site: String,
      heroName: Option[String],
      handsImported: Int,
      handsAnalyzed: Int,
      handsSkipped: Int,
      decisionsAnalyzed: Int,
      mistakes: Int,
      totalEvLost: Double,
      biggestMistakeEv: Double,
      modelSource: String,
      warnings: Vector[String],
      decisions: Vector[DecisionView],
      opponents: Vector[OpponentView],
      trace: AnalysisTrace
  )

  def create(config: ServiceConfig): Either[String, HandHistoryReviewService] =
    try
      val actionModel = config.modelDir match
        case Some(path) => PokerActionModelArtifactIO.load(path).model
        case None => PokerActionModel.uniform
      val source = config.modelDir.map(_.toAbsolutePath.normalize().toString).getOrElse("uniform fallback")
      Right(new HandHistoryReviewService(config, actionModel, source))
    catch
      case e: Exception => Left(s"failed to initialize hand-history review service: ${e.getMessage}")
