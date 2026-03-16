package sicfun.holdem.web

import sicfun.holdem.engine.RealTimeAdaptiveEngine
import sicfun.holdem.equity.{TableFormat, TableRanges}
import sicfun.holdem.history.{HandHistoryImport, HandHistorySite, ImportedHand, OpponentProfile}
import sicfun.holdem.model.{PokerActionModel, PokerActionModelArtifactIO}
import sicfun.holdem.runtime.HandHistoryAnalyzer
import sicfun.holdem.types.PokerAction

import ujson.{Arr, Num, Obj, Str, Value}

import java.nio.file.Path
import java.util.Locale
import scala.collection.mutable
import scala.util.Random

final class HandHistoryReviewService private (
    config: HandHistoryReviewService.ServiceConfig,
    actionModel: PokerActionModel,
    modelSource: String
):
  import HandHistoryReviewService.*

  def analyze(request: AnalysisRequest): Either[String, AnalysisResponse] =
    val normalizedRequest = request.copy(
      heroName = request.heroName.map(HandHistoryImport.normalizePlayerName).filter(_.nonEmpty)
    )
    if normalizedRequest.handHistoryText.trim.isEmpty then Left("handHistoryText must be non-empty")
    else
      for
        imported <- HandHistoryImport.parseText(
          text = normalizedRequest.handHistoryText,
          site = normalizedRequest.site,
          heroName = normalizedRequest.heroName
        )
        response <- buildResponse(imported, normalizedRequest)
      yield response

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
      "opponents" -> Arr.from(response.opponents.map(writeOpponent))
    )

  private def buildResponse(
      imported: Vector[ImportedHand],
      request: AnalysisRequest
  ): Either[String, AnalysisResponse] =
    if imported.isEmpty then Left("no hands were imported from the uploaded text")
    else
      try
        val warnings = mutable.ArrayBuffer.empty[String]
        val rng = new Random(config.seed)

        val analyzedHands = imported.flatMap { hand =>
          analyzeHand(hand, request.heroName, warnings, rng)
        }
        val allDecisions = analyzedHands.flatMap(_.decisions)
        val handsAnalyzed = analyzedHands.length
        val handsSkipped = math.max(0, imported.length - handsAnalyzed)
        val effectiveHero = request.heroName.orElse(resolveHeroName(imported))
        val excludedHeroes = request.heroName.toSet ++ imported.flatMap(_.heroName).toSet
        val profiles = OpponentProfile.fromImportedHands(
          site = imported.head.site.toString.toLowerCase,
          hands = imported,
          excludePlayers = excludedHeroes
        )
        val totalEvLost = math.abs(allDecisions.iterator.map(d => math.min(0.0, d.evDifference)).sum)
        val biggestMistake =
          if allDecisions.isEmpty then 0.0
          else allDecisions.iterator.map(d => math.abs(d.evDifference)).max

        Right(
          AnalysisResponse(
            site = imported.head.site.toString,
            heroName = effectiveHero,
            handsImported = imported.length,
            handsAnalyzed = handsAnalyzed,
            handsSkipped = handsSkipped,
            decisionsAnalyzed = allDecisions.length,
            mistakes = allDecisions.count(HandHistoryAnalyzer.countsAsMistake),
            totalEvLost = roundChips(totalEvLost),
            biggestMistakeEv = roundChips(biggestMistake),
            modelSource = modelSource,
            warnings = warnings.toVector.distinct,
            decisions = allDecisions
              .sortBy(d => (-math.abs(d.evDifference), d.handId, d.street.toString))
              .take(config.maxDecisions)
              .map(writeDecisionView),
            opponents = profiles.take(5).map(writeOpponentView)
          )
        )
      catch
        case e: Exception => Left(s"analysis failed: ${e.getMessage}")

  private final case class HandResult(
      handId: String,
      decisions: Vector[HandHistoryAnalyzer.AnalyzedDecision]
  )

  private def analyzeHand(
      hand: ImportedHand,
      heroNameOverride: Option[String],
      warnings: mutable.ArrayBuffer[String],
      rng: Random
  ): Option[HandResult] =
    val heroName = resolveHeroPlayerId(hand, heroNameOverride).orElse(hand.heroName)
    (heroName, hand.heroHoleCards) match
      case (None, _) =>
        warnings += s"hand ${hand.handId}: skipped because hero name could not be inferred"
        None
      case (_, None) =>
        warnings += s"hand ${hand.handId}: skipped because hero hole cards were not present in the uploaded history"
        None
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
        if decisions.isEmpty then
          warnings += s"hand ${hand.handId}: imported but no hero decisions could be analyzed"
        Some(HandResult(hand.handId, decisions))

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
      hints = profile.exploitHints.take(3)
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
      "hints" -> Arr.from(opponent.hints.map(Str(_)))
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
      hints: Vector[String]
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
      opponents: Vector[OpponentView]
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
