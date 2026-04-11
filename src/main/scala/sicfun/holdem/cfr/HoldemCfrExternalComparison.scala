package sicfun.holdem.cfr

import sicfun.holdem.cli.CliHelpers
import sicfun.core.Card

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Path, Paths}
import java.util.Locale
import scala.jdk.CollectionConverters.*
import scala.collection.mutable
import ujson.{Arr, Num, Obj, Str, Value}

/** Compares SICFUN CFR spot exports against an external solver/provider export.
  *
  * This is the cross-validation framework for verifying that different CFR
  * implementations (Scala, native CPU, native GPU, or external solvers like
  * TexasSolver) produce consistent strategies. It operates on JSON exports
  * rather than live solver calls, enabling offline batch comparison.
  *
  * The expected input is the `external-comparison.json` emitted by
  * [[HoldemCfrApproximationReport]] plus a provider file with the same spot ids
  * and at least a `policy` object per spot. `bestAction` and `actionEvs` are
  * optional on the provider side.
  *
  * '''Comparison metrics:'''
  *  - '''TV distance''' (Total Variation distance): Half the sum of absolute
  *    differences between reference and external policy probabilities across all
  *    actions. TV distance of 0 means identical policies; 1 means completely
  *    disjoint support. This is the primary convergence metric.
  *  - '''Best-action agreement''': Whether the reference and external agree on
  *    the highest-EV (or highest-probability) action. Ties are handled by checking
  *    whether the best-action sets intersect.
  *  - '''EV RMSE''': Root mean squared error of per-action expected values, when
  *    both sides provide `actionEvs`. This catches cases where policies agree but
  *    the underlying value estimates diverge.
  *  - '''Spot signature''': A canonical string encoding the full spot specification
  *    (board, hero, villain range, bet history, actions) so that the comparison
  *    can verify both sides solved the same problem, not just matching by spot id.
  *
  * '''Gate evaluation:''' The comparison enforces configurable thresholds on the
  * above metrics. A gate PASS requires all thresholds to be satisfied and at
  * least one threshold to be configured (to prevent vacuous passes).
  *
  * Usage: `runMain sicfun.holdem.cfr.HoldemCfrExternalComparison --reference=... --external=... --maxMeanTv=0.001`
  */
object HoldemCfrExternalComparison:
  /** Epsilon for floating-point comparison when validating that a declared bestAction
    * is consistent with the action values. Allows for minor rounding differences.
    */
  private val BestActionConsistencyEpsilon = 1e-4

  /** A single spot parsed from either the reference or external JSON dataset.
    *
    * @param id               Unique identifier for the spot (e.g., "hu_preflop_button_open")
    * @param spotSignature    Optional canonical encoding of the full spot specification for
    *                         cross-validation that both sides solved the same problem
    * @param candidateActions Union of all actions seen in policy/actionEvs/declared actions
    * @param policy           Normalized probability distribution over actions (sums to 1.0)
    * @param actionEvs        Optional per-action expected values (empty if not provided)
    * @param bestAction       The single declared or derived best action
    * @param bestActions      Set of actions tied for best (may be larger than 1 due to ties)
    */
  final case class ParsedSpot(
      id: String,
      spotSignature: Option[String],
      candidateActions: Vector[String],
      policy: Map[String, Double],
      actionEvs: Map[String, Double],
      bestAction: String,
      bestActions: Set[String]
  ):
    require(id.trim.nonEmpty, "spot id must be non-empty")
    require(candidateActions.nonEmpty, s"spot '$id' must have at least one candidate action")
    require(policy.nonEmpty, s"spot '$id' must have at least one positive policy weight")
    require(bestActions.nonEmpty, s"spot '$id' must have at least one best action")
    require(bestActions.contains(bestAction), s"spot '$id' bestAction must belong to bestActions")

  final case class ParsedDataset(
      label: String,
      spots: Vector[ParsedSpot]
  ):
    require(label.trim.nonEmpty, "dataset label must be non-empty")

  /** Configurable quality gate thresholds. At least one threshold must be set
    * for a gate evaluation to pass (prevents vacuous success).
    *
    * @param maxMeanTvDistance       Maximum allowed mean TV distance across all matched spots
    * @param maxSpotTvDistance       Maximum allowed TV distance for any single spot
    * @param minBestActionAgreement  Minimum fraction of spots where best actions agree [0,1]
    * @param maxMeanEvRmse          Maximum allowed mean EV RMSE across matched spots
    */
  final case class Thresholds(
      maxMeanTvDistance: Option[Double],
      maxSpotTvDistance: Option[Double],
      minBestActionAgreement: Option[Double],
      maxMeanEvRmse: Option[Double]
  )

  /** Per-spot comparison result between reference and external policies. */
  final case class SpotComparison(
      id: String,
      spotSignatureStatus: String,
      referenceBestAction: String,
      externalBestAction: String,
      bestActionMatches: Boolean,
      comparedActions: Vector[String],
      tvDistance: Double,
      maxActionProbabilityGap: Double,
      evRmse: Option[Double],
      evMaxGap: Option[Double],
      actionsMissingInExternal: Vector[String],
      extraActionsInExternal: Vector[String]
  )

  /** Aggregate statistics across all matched spots. */
  final case class AggregateComparison(
      referenceSpotCount: Int,
      externalSpotCount: Int,
      matchedSpotCount: Int,
      matchingSpotSignatureCount: Int,
      meanTvDistance: Double,
      maxTvDistance: Double,
      meanMaxActionProbabilityGap: Double,
      bestActionAgreementCount: Int,
      bestActionAgreementRate: Double,
      meanEvRmse: Option[Double],
      maxEvGap: Option[Double],
      spotSignatureMismatchIds: Vector[String],
      missingExternalSpotSignatureIds: Vector[String],
      unmatchedReferenceSpotIds: Vector[String],
      unmatchedExternalSpotIds: Vector[String]
  )

  final case class GateResult(
      passed: Boolean,
      failures: Vector[String]
  )

  final case class RunResult(
      referenceLabel: String,
      externalLabel: String,
      aggregate: AggregateComparison,
      gate: GateResult,
      spotComparisons: Vector[SpotComparison],
      outDir: Option[Path]
  )

  private final case class CliConfig(
      referencePath: Path,
      externalPath: Path,
      selectedSpotIds: Option[Set[String]],
      thresholds: Thresholds,
      outDir: Option[Path]
  )

  private[cfr] val SpotComparisonTsvHeader =
    "spotId\tspotSignatureStatus\treferenceBestAction\texternalBestAction\tbestActionMatches\tcomparedActions\t" +
      "tvDistance\tmaxActionProbabilityGap\tevRmse\tevMaxGap\tactionsMissingInExternal\textraActionsInExternal"
  private[cfr] val ComparisonJsonFileName = "comparison.json"

  def main(args: Array[String]): Unit =
    val wantsHelp = args.contains("--help") || args.contains("-h")
    run(args) match
      case Right(result) =>
        println("=== Holdem CFR External Comparison ===")
        println(s"reference: ${result.referenceLabel}")
        println(s"external: ${result.externalLabel}")
        println(s"matchedSpots: ${result.aggregate.matchedSpotCount}/${result.aggregate.referenceSpotCount}")
        println(
          s"matchingSpotSignatures: ${result.aggregate.matchingSpotSignatureCount}/${result.aggregate.matchedSpotCount}"
        )
        println(s"meanTvDistance: ${formatDouble(result.aggregate.meanTvDistance, 6)}")
        println(s"maxTvDistance: ${formatDouble(result.aggregate.maxTvDistance, 6)}")
        println(s"meanMaxActionProbabilityGap: ${formatDouble(result.aggregate.meanMaxActionProbabilityGap, 6)}")
        println(
          s"bestActionAgreement: ${result.aggregate.bestActionAgreementCount}/${result.aggregate.matchedSpotCount} " +
            s"(${formatPercent(result.aggregate.bestActionAgreementRate)})"
        )
        println(
          s"meanEvRmse: ${result.aggregate.meanEvRmse.map(formatDouble(_, 6)).getOrElse("n/a")}"
        )
        println(
          s"maxEvGap: ${result.aggregate.maxEvGap.map(formatDouble(_, 6)).getOrElse("n/a")}"
        )
        println(s"gate: ${if result.gate.passed then "PASS" else "FAIL"}")
        if result.gate.failures.nonEmpty then
          result.gate.failures.foreach(reason => println(s"  - $reason"))
        result.outDir.foreach(path => println(s"outDir: ${path.toAbsolutePath.normalize()}"))
      case Left(error) =>
        if wantsHelp then println(error)
        else
          System.err.println(error)
          sys.exit(1)

  def run(args: Array[String]): Either[String, RunResult] =
    for
      config <- parseArgs(args)
      result <- compareFiles(
        referencePath = config.referencePath,
        externalPath = config.externalPath,
        selectedSpotIds = config.selectedSpotIds,
        thresholds = config.thresholds,
        outDir = config.outDir
      )
    yield result

  /** Main comparison entry point used by both the CLI and programmatic callers.
    *
    * Loads both JSON datasets, optionally filters by spot ids, runs the comparison,
    * writes output artifacts, and returns the result (or Left on gate failure).
    */
  private[cfr] def compareFiles(
      referencePath: Path,
      externalPath: Path,
      selectedSpotIds: Option[Set[String]] = None,
      thresholds: Thresholds,
      outDir: Option[Path]
  ): Either[String, RunResult] =
    try
      val reference = loadDataset(referencePath)
      val external = loadDataset(externalPath)
      val selectedReference = selectDatasetSpots(reference, selectedSpotIds, "reference", requireNonEmpty = true)
      val selectedExternal = selectDatasetSpots(external, selectedSpotIds, "external", requireNonEmpty = false)
      val result = compareDatasets(selectedReference, selectedExternal, thresholds, outDir)
      outDir.foreach(dir => writeOutputs(dir, result))
      if result.gate.passed then Right(result)
      else Left(s"holdem CFR external comparison gate failed: ${result.gate.failures.mkString("; ")}")
    catch
      case e: Exception =>
        Left(s"holdem CFR external comparison failed: ${e.getMessage}")

  private[cfr] def loadDataset(path: Path): ParsedDataset =
    require(Files.exists(path), s"dataset file does not exist: $path")
    val json = ujson.read(Files.readString(path, StandardCharsets.UTF_8))
    parseDataset(json, fallbackLabel = path.getFileName.toString)

  /** Compares two parsed datasets by joining on spot id, computing per-spot
    * metrics, aggregating, and evaluating the quality gate.
    */
  private[cfr] def compareDatasets(
      reference: ParsedDataset,
      external: ParsedDataset,
      thresholds: Thresholds,
      outDir: Option[Path]
  ): RunResult =
    val externalById = external.spots.iterator.map(spot => spot.id -> spot).toMap
    val referenceIds = reference.spots.map(_.id)
    val externalIds = external.spots.map(_.id)

    val spotComparisons = reference.spots.flatMap { refSpot =>
      externalById.get(refSpot.id).map(compareSpot(refSpot, _))
    }

    val aggregate = buildAggregate(
      referenceIds = referenceIds,
      externalIds = externalIds,
      spotComparisons = spotComparisons
    )
    val gate = evaluateGate(aggregate, thresholds)

    RunResult(
      referenceLabel = reference.label,
      externalLabel = external.label,
      aggregate = aggregate,
      gate = gate,
      spotComparisons = spotComparisons,
      outDir = outDir
    )

  private def parseDataset(value: Value, fallbackLabel: String): ParsedDataset =
    val obj = value.obj
    val label =
      obj.get("providerName").map(_.str)
        .orElse(obj.get("label").map(_.str))
        .orElse(obj.get("suiteName").map(_.str))
        .getOrElse(stripJsonSuffix(fallbackLabel))
    val spotsValue = obj.get("spots").getOrElse(
      throw new IllegalArgumentException("dataset JSON must contain a 'spots' array")
    )
    val spots = spotsValue.arr.iterator.map(parseSpot).toVector
    val duplicate = spots.groupBy(_.id).collectFirst { case (id, matches) if matches.size > 1 => id }
    require(duplicate.isEmpty, s"duplicate spot id '${duplicate.get}' in dataset '$label'")
    ParsedDataset(label = label, spots = spots)

  private def selectDatasetSpots(
      dataset: ParsedDataset,
      selectedSpotIds: Option[Set[String]],
      datasetRole: String,
      requireNonEmpty: Boolean
  ): ParsedDataset =
    selectedSpotIds match
      case None => dataset
      case Some(ids) =>
        if requireNonEmpty then
          val availableIds = dataset.spots.map(_.id).toSet
          val missingIds = ids.toVector.sorted.filterNot(availableIds.contains)
          require(
            missingIds.isEmpty,
            s"$datasetRole dataset is missing selected spot ids: ${missingIds.mkString(", ")}"
          )
        val filtered = dataset.spots.filter(spot => ids.contains(spot.id))
        if requireNonEmpty then
          require(filtered.nonEmpty, s"$datasetRole dataset does not contain any selected spot ids")
        dataset.copy(spots = filtered)

  /** Parses a single spot from the JSON dataset.
    *
    * Handles the following:
    *  - Extracts and normalizes the policy distribution
    *  - Parses optional actionEvs, candidateActions, and bestAction
    *  - Canonicalizes all action labels (e.g., "raise:20" -> "RAISE:20.000")
    *  - Derives bestActions from actionEvs (if present) or policy (as fallback)
    *  - Validates bestAction consistency with the derived best actions
    *  - Computes and validates the spot signature for cross-dataset matching
    */
  private def parseSpot(value: Value): ParsedSpot =
    val obj = value.obj
    val id = obj.get("id").map(_.str).getOrElse(
      throw new IllegalArgumentException("spot entry missing 'id'")
    )
    val rawPolicy = obj.get("policy").map(parseActionDoubleMap(_, s"spot '$id' policy")).getOrElse(
      throw new IllegalArgumentException(s"spot '$id' missing policy object")
    )
    val policy = normalizeDistribution(rawPolicy, s"spot '$id' policy")
    val actionEvs = obj.get("actionEvs").map(parseActionDoubleMap(_, s"spot '$id' actionEvs")).getOrElse(Map.empty)
    val declaredActions = obj.get("candidateActions").map(parseActionVector(_, s"spot '$id' candidateActions")).getOrElse(Vector.empty)
    val candidateActions = dedupePreservingOrder(declaredActions ++ policy.keys.toVector.sorted ++ actionEvs.keys.toVector.sorted)
    val providedBestAction = obj.get("bestAction").map(parseActionString(_, s"spot '$id' bestAction")).filter(_.trim.nonEmpty)
    providedBestAction.foreach { action =>
      require(
        candidateActions.contains(action) || policy.contains(action) || actionEvs.contains(action),
        s"spot '$id' bestAction '$action' is not present in candidateActions/policy/actionEvs"
      )
    }
    val bestActions =
      if actionEvs.nonEmpty then argMaxActions(actionEvs, candidateActions.filter(actionEvs.contains))
      else argMaxActions(policy, candidateActions)
    providedBestAction.foreach { action =>
      if actionEvs.nonEmpty then
        validateBestAction(action, actionEvs, candidateActions, s"spot '$id' bestAction", "actionEvs")
      else
        validateBestAction(action, policy, candidateActions, s"spot '$id' bestAction", "policy")
    }
    val bestAction = providedBestAction.getOrElse(bestActions.head)

    val providedSignature = parseOptionalSpotSignature(obj, id)
    val computedSignature = maybeBuildSpotSignature(obj, id, candidateActions)
    val spotSignature =
      (providedSignature, computedSignature) match
        case (Some(provided), Some(computed)) =>
          require(
            provided == computed,
            s"spot '$id' spotSignature does not match the canonical signature derived from its payload"
          )
          Some(computed)
        case (Some(provided), None) => Some(provided)
        case (None, Some(computed)) => Some(computed)
        case (None, None)           => None

    ParsedSpot(
      id = id,
      spotSignature = spotSignature,
      candidateActions = candidateActions,
      policy = policy,
      actionEvs = actionEvs,
      bestAction = bestAction,
      bestActions = bestActions.toSet
    )

  private def parseActionDoubleMap(value: Value, label: String): Map[String, Double] =
    val builder = Map.newBuilder[String, Double]
    val seen = mutable.HashSet.empty[String]
    value.obj.toVector.foreach { case (rawKey, raw) =>
      val key = canonicalizeAction(rawKey, label)
      val number =
        raw match
          case Num(v) if v.isFinite => v
          case other =>
            throw new IllegalArgumentException(s"$label contains non-finite value for '$key': $other")
      require(seen.add(key), s"$label contains duplicate action '$key' after normalization")
      builder += key -> number
    }
    builder.result()

  private def parseActionVector(value: Value, label: String): Vector[String] =
    value.arr.iterator.map {
      case Str(text) if text.trim.nonEmpty => canonicalizeAction(text, label)
      case other =>
        throw new IllegalArgumentException(s"$label must contain non-empty strings, found $other")
    }.toVector

  /** Normalizes a probability distribution: filters out zero/negative weights,
    * validates at least one positive entry exists, and divides by the total so
    * the result sums to 1.0. This handles raw policy weights from external solvers
    * that may not be pre-normalized.
    */
  private def normalizeDistribution(values: Map[String, Double], label: String): Map[String, Double] =
    val negative = values.collectFirst { case (action, probability) if probability < 0.0 => action }
    require(negative.isEmpty, s"$label contains negative weight for '${negative.get}'")
    val filtered = values.collect { case (action, probability) if probability > 0.0 =>
      action -> probability
    }
    require(filtered.nonEmpty, s"$label must contain at least one positive finite weight")
    val total = filtered.values.sum
    filtered.view.mapValues(_ / total).toMap

  private def parseActionString(value: Value, label: String): String =
    value match
      case Str(text) if text.trim.nonEmpty => canonicalizeAction(text, label)
      case other =>
        throw new IllegalArgumentException(s"$label must be a non-empty string, found $other")

  private def parseOptionalSpotSignature(obj: collection.Map[String, Value], id: String): Option[String] =
    obj.get("spotSignature")
      .orElse(obj.get("stateSignature"))
      .map {
        case Str(text) if text.trim.nonEmpty => text.trim
        case other =>
          throw new IllegalArgumentException(s"spot '$id' spotSignature must be a non-empty string, found $other")
      }

  private def maybeBuildSpotSignature(
      obj: collection.Map[String, Value],
      id: String,
      candidateActions: Vector[String]
  ): Option[String] =
    val hasState = obj.contains("state")
    val hasHero = obj.contains("hero")
    val hasVillainRange = obj.contains("villainRange")
    val anySpotPayload = hasState || hasHero || hasVillainRange
    if !anySpotPayload then None
    else
      require(hasState, s"spot '$id' must include state to derive a canonical spotSignature")
      require(hasHero, s"spot '$id' must include hero to derive a canonical spotSignature")
      require(hasVillainRange, s"spot '$id' must include villainRange to derive a canonical spotSignature")

      val state = obj("state").obj
      val street = requiredStringField(state, "street", s"spot '$id' state")
      val position = requiredStringField(state, "position", s"spot '$id' state")
      val board = state.get("board").map(parseCardVector(_, s"spot '$id' state.board")).getOrElse(Vector.empty)
      val pot = requiredNumberField(state, "pot", s"spot '$id' state")
      val toCall = requiredNumberField(state, "toCall", s"spot '$id' state")
      val stackSize = requiredNumberField(state, "stackSize", s"spot '$id' state")
      val betHistory = state.get("betHistory").map(parseBetHistory(_, s"spot '$id' state.betHistory")).getOrElse(Vector.empty)
      val hero = canonicalizeHoleCards(requiredStringField(obj, "hero", s"spot '$id'"))
      val villainRange = parseVillainRange(obj("villainRange"), s"spot '$id' villainRange")

      Some(
        buildSpotSignature(
          street = street,
          position = position,
          hero = hero,
          board = board,
          pot = pot,
          toCall = toCall,
          stackSize = stackSize,
          betHistory = betHistory,
          villainRange = villainRange,
          candidateActions = candidateActions
        )
      )

  private def requiredStringField(
      obj: collection.Map[String, Value],
      key: String,
      label: String
  ): String =
    obj.get(key) match
      case Some(Str(text)) if text.trim.nonEmpty => text.trim
      case Some(other) =>
        throw new IllegalArgumentException(s"$label.$key must be a non-empty string, found $other")
      case None =>
        throw new IllegalArgumentException(s"$label missing '$key'")

  private def requiredNumberField(
      obj: collection.Map[String, Value],
      key: String,
      label: String
  ): Double =
    obj.get(key) match
      case Some(Num(value)) if value.isFinite => value
      case Some(other) =>
        throw new IllegalArgumentException(s"$label.$key must be a finite number, found $other")
      case None =>
        throw new IllegalArgumentException(s"$label missing '$key'")

  private def parseCardVector(value: Value, label: String): Vector[String] =
    value.arr.iterator.map {
      case Str(text) if text.trim.nonEmpty =>
        Card.parse(text.trim).map(_.toToken).getOrElse(
          throw new IllegalArgumentException(s"$label contains invalid card token '$text'")
        )
      case other =>
        throw new IllegalArgumentException(s"$label must contain non-empty card strings, found $other")
    }.toVector.sorted

  private def parseBetHistory(value: Value, label: String): Vector[(Int, String)] =
    value.arr.iterator.map { entry =>
      val obj = entry.obj
      val player = requiredNumberField(obj, "player", label)
      require(player.isWhole && player >= 0.0, s"$label.player must be a non-negative integer")
      val action = parseActionString(obj.getOrElse(
        "action",
        throw new IllegalArgumentException(s"$label entry missing 'action'")
      ), s"$label action")
      player.toInt -> action
    }.toVector

  private def parseVillainRange(value: Value, label: String): Vector[(String, Double)] =
    val weights = mutable.LinkedHashMap.empty[String, Double].withDefaultValue(0.0)
    value.arr.iterator.foreach { entry =>
      val obj = entry.obj
      val hand = canonicalizeHoleCards(requiredStringField(obj, "hand", label))
      val probability = requiredNumberField(obj, "probability", label)
      require(probability >= 0.0, s"$label contains negative probability for '$hand'")
      weights.update(hand, weights(hand) + probability)
    }
    val normalized = normalizeDistribution(weights.toMap, label)
    normalized.toVector.sortBy(_._1)

  /** Builds a canonical spot signature string from the spot's full specification.
    *
    * The signature encodes: street, position, hero hand, sorted board cards, pot/toCall/stack
    * amounts (to 6 decimal places), bet history, sorted candidate actions, and the normalized
    * villain range (hands sorted alphabetically with probabilities to 8 decimal places).
    *
    * This deterministic encoding ensures two datasets that describe the same poker situation
    * will produce identical signatures regardless of JSON key ordering or float formatting.
    */
  private def buildSpotSignature(
      street: String,
      position: String,
      hero: String,
      board: Vector[String],
      pot: Double,
      toCall: Double,
      stackSize: Double,
      betHistory: Vector[(Int, String)],
      villainRange: Vector[(String, Double)],
      candidateActions: Vector[String]
  ): String =
    val normalizedHistory = betHistory.map { case (player, action) => s"$player:$action" }.mkString(";")
    val normalizedRange = villainRange
      .map { case (hand, probability) => s"$hand@${formatDouble(probability, 8)}" }
      .mkString(";")
    val normalizedActions = candidateActions.sorted.mkString(",")
    s"street=${street.trim.toUpperCase(Locale.ROOT)}|" +
      s"position=${position.trim.toUpperCase(Locale.ROOT)}|hero=$hero|board=${board.mkString("")}|" +
      s"pot=${formatDouble(pot, 6)}|toCall=${formatDouble(toCall, 6)}|stackSize=${formatDouble(stackSize, 6)}|" +
      s"betHistory=$normalizedHistory|candidateActions=$normalizedActions|villainRange=$normalizedRange"

  private def canonicalizeHoleCards(token: String): String =
    CliHelpers.parseHoleCards(token.trim).toToken

  /** Canonicalizes action labels into a consistent format: FOLD, CHECK, CALL,
    * or RAISE:amount (with 3 decimal places). Handles variations like "bet:20",
    * "RAISE=20", "Raise 20", etc. from different solver output formats.
    */
  private def canonicalizeAction(raw: String, label: String): String =
    val trimmed = raw.trim
    require(trimmed.nonEmpty, s"$label must not contain blank action labels")
    val upper = trimmed.toUpperCase(Locale.ROOT)
    upper match
      case "FOLD"  => "FOLD"
      case "CHECK" => "CHECK"
      case "CALL"  => "CALL"
      case _ =>
        val raisePattern = """^(RAISE|BET)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)$""".r
        upper match
          case raisePattern(_, amountText) =>
            val amount = amountText.toDouble
            require(amount.isFinite && amount > 0.0, s"$label contains invalid raise amount '$raw'")
            renderRaise(amount)
          case _ =>
            throw new IllegalArgumentException(s"$label contains unsupported action label '$raw'")

  private def validateBestAction(
      action: String,
      scores: Map[String, Double],
      candidateActions: Vector[String],
      label: String,
      source: String
  ): Unit =
    val actions = candidateActions.filter(scores.contains)
    require(actions.nonEmpty, s"$label cannot be validated because $source is empty")
    val maxValue = actions.map(scores).max
    val chosenValue = scores.getOrElse(action, Double.NegativeInfinity)
    require(
      chosenValue + BestActionConsistencyEpsilon >= maxValue,
      s"$label '$action' is inconsistent with the reported $source"
    )

  private def renderRaise(amount: Double): String =
    String.format(Locale.ROOT, "RAISE:%.3f", java.lang.Double.valueOf(amount))

  /** Computes per-spot comparison metrics between a reference and external spot.
    *
    * TV distance = 0.5 * sum(|ref_policy[a] - ext_policy[a]|) for all actions a.
    * This is a standard measure of distribution divergence bounded in [0, 1].
    * Actions present in one policy but not the other are treated as having 0 probability.
    *
    * EV RMSE is computed only over actions where both sides provide EV values.
    * Best-action agreement uses set intersection to handle ties gracefully.
    */
  private def compareSpot(reference: ParsedSpot, external: ParsedSpot): SpotComparison =
    val comparedActions = dedupePreservingOrder(
      reference.candidateActions ++
        external.candidateActions ++
        reference.policy.keys.toVector ++
        external.policy.keys.toVector ++
        reference.actionEvs.keys.toVector ++
        external.actionEvs.keys.toVector
    )
    val referenceBestAction = reference.bestAction
    val externalBestAction = external.bestAction
    val spotSignatureStatus =
      (reference.spotSignature, external.spotSignature) match
        case (Some(ref), Some(ext)) if ref == ext => "match"
        case (Some(_), Some(_))                   => "mismatch"
        case (Some(_), None)                      => "missing-external"
        case (None, Some(_))                      => "missing-reference"
        case (None, None)                         => "missing-both"
    val diffs = comparedActions.map { action =>
      math.abs(reference.policy.getOrElse(action, 0.0) - external.policy.getOrElse(action, 0.0))
    }
    val tvDistance = diffs.sum * 0.5
    val maxActionProbabilityGap = if diffs.nonEmpty then diffs.max else 0.0
    val evActions = comparedActions.filter(action => reference.actionEvs.contains(action) && external.actionEvs.contains(action))
    val evGaps = evActions.map(action => math.abs(reference.actionEvs(action) - external.actionEvs(action)))
    val evRmse =
      if evGaps.nonEmpty then
        Some(math.sqrt(evGaps.map(gap => gap * gap).sum / evGaps.size.toDouble))
      else None
    val evMaxGap = if evGaps.nonEmpty then Some(evGaps.max) else None

    val externalDeclaredActions = external.candidateActions.toSet ++ external.policy.keySet
    val referenceDeclaredActions = reference.candidateActions.toSet ++ reference.policy.keySet

    SpotComparison(
      id = reference.id,
      spotSignatureStatus = spotSignatureStatus,
      referenceBestAction = referenceBestAction,
      externalBestAction = externalBestAction,
      bestActionMatches = reference.bestActions.intersect(external.bestActions).nonEmpty,
      comparedActions = comparedActions,
      tvDistance = tvDistance,
      maxActionProbabilityGap = maxActionProbabilityGap,
      evRmse = evRmse,
      evMaxGap = evMaxGap,
      actionsMissingInExternal = reference.candidateActions.filterNot(externalDeclaredActions.contains),
      extraActionsInExternal = external.candidateActions.filterNot(referenceDeclaredActions.contains)
    )

  /** Aggregates per-spot comparison results into summary statistics.
    * Also identifies unmatched spots (present in one dataset but not the other)
    * and spots with mismatched or missing signatures.
    */
  private def buildAggregate(
      referenceIds: Vector[String],
      externalIds: Vector[String],
      spotComparisons: Vector[SpotComparison]
  ): AggregateComparison =
    val matchedSpotCount = spotComparisons.size
    val matchingSpotSignatureCount = spotComparisons.count(_.spotSignatureStatus == "match")
    val meanTvDistance =
      if matchedSpotCount == 0 then 0.0
      else spotComparisons.map(_.tvDistance).sum / matchedSpotCount.toDouble
    val maxTvDistance =
      if matchedSpotCount == 0 then 0.0
      else spotComparisons.map(_.tvDistance).max
    val meanMaxActionProbabilityGap =
      if matchedSpotCount == 0 then 0.0
      else spotComparisons.map(_.maxActionProbabilityGap).sum / matchedSpotCount.toDouble
    val bestActionAgreementCount = spotComparisons.count(_.bestActionMatches)
    val bestActionAgreementRate =
      if matchedSpotCount == 0 then 0.0
      else bestActionAgreementCount.toDouble / matchedSpotCount.toDouble
    val evRmses = spotComparisons.flatMap(_.evRmse)
    val evMaxGaps = spotComparisons.flatMap(_.evMaxGap)
    val referenceSet = referenceIds.toSet
    val externalSet = externalIds.toSet

    AggregateComparison(
      referenceSpotCount = referenceIds.size,
      externalSpotCount = externalIds.size,
      matchedSpotCount = matchedSpotCount,
      matchingSpotSignatureCount = matchingSpotSignatureCount,
      meanTvDistance = meanTvDistance,
      maxTvDistance = maxTvDistance,
      meanMaxActionProbabilityGap = meanMaxActionProbabilityGap,
      bestActionAgreementCount = bestActionAgreementCount,
      bestActionAgreementRate = bestActionAgreementRate,
      meanEvRmse = if evRmses.nonEmpty then Some(evRmses.sum / evRmses.size.toDouble) else None,
      maxEvGap = if evMaxGaps.nonEmpty then Some(evMaxGaps.max) else None,
      spotSignatureMismatchIds = spotComparisons.collect { case spot if spot.spotSignatureStatus == "mismatch" => spot.id },
      missingExternalSpotSignatureIds =
        spotComparisons.collect { case spot if spot.spotSignatureStatus == "missing-external" => spot.id },
      unmatchedReferenceSpotIds = referenceIds.filterNot(externalSet.contains),
      unmatchedExternalSpotIds = externalIds.filterNot(referenceSet.contains)
    )

  /** Evaluates the quality gate against the aggregate comparison statistics.
    *
    * A gate passes only when ALL of the following hold:
    *  - At least one spot was matched between datasets
    *  - No reference spots are missing from the external dataset
    *  - No spot signature mismatches detected
    *  - At least one proof threshold is configured
    *  - All configured thresholds are satisfied
    *
    * Returns a GateResult with passed=true if all checks pass, otherwise
    * passed=false with a list of failure reasons.
    */
  private def evaluateGate(aggregate: AggregateComparison, thresholds: Thresholds): GateResult =
    val failures = Vector.newBuilder[String]

    if aggregate.matchedSpotCount == 0 then
      failures += "no matching spot ids were found between the reference and external datasets"
    if aggregate.unmatchedReferenceSpotIds.nonEmpty then
      failures += s"external dataset is missing ${aggregate.unmatchedReferenceSpotIds.size} reference spot(s)"
    if aggregate.spotSignatureMismatchIds.nonEmpty then
      failures += s"spot signature mismatch for ${aggregate.spotSignatureMismatchIds.mkString(", ")}"
    if aggregate.missingExternalSpotSignatureIds.nonEmpty then
      failures += s"external dataset is missing spot signatures for ${aggregate.missingExternalSpotSignatureIds.mkString(", ")}"
    if thresholds.maxMeanTvDistance.isEmpty &&
      thresholds.maxSpotTvDistance.isEmpty &&
      thresholds.minBestActionAgreement.isEmpty &&
      thresholds.maxMeanEvRmse.isEmpty
    then
      failures +=
        "no proof thresholds configured; set at least one of maxMeanTv/maxSpotTv/minBestActionAgreement/maxMeanEvRmse"

    thresholds.maxMeanTvDistance.foreach { limit =>
      if aggregate.meanTvDistance > limit then
        failures +=
          s"mean TV distance ${formatDouble(aggregate.meanTvDistance, 6)} exceeded limit ${formatDouble(limit, 6)}"
    }
    thresholds.maxSpotTvDistance.foreach { limit =>
      if aggregate.maxTvDistance > limit then
        failures +=
          s"max TV distance ${formatDouble(aggregate.maxTvDistance, 6)} exceeded limit ${formatDouble(limit, 6)}"
    }
    thresholds.minBestActionAgreement.foreach { floor =>
      if aggregate.bestActionAgreementRate < floor then
        failures +=
          s"best-action agreement ${formatPercent(aggregate.bestActionAgreementRate)} fell below ${formatPercent(floor)}"
    }
    thresholds.maxMeanEvRmse.foreach { limit =>
      aggregate.meanEvRmse match
        case Some(value) if value > limit =>
          failures += s"mean EV RMSE ${formatDouble(value, 6)} exceeded limit ${formatDouble(limit, 6)}"
        case None =>
          failures += "mean EV RMSE threshold requested but no overlapping action EVs were available"
        case _ => ()
    }

    val built = failures.result()
    GateResult(passed = built.isEmpty, failures = built)

  private def writeOutputs(outDir: Path, result: RunResult): Unit =
    Files.createDirectories(outDir)
    Files.write(
      outDir.resolve("summary.txt"),
      buildSummaryLines(result).asJava,
      StandardCharsets.UTF_8
    )
    Files.write(
      outDir.resolve("spot-comparison.tsv"),
      buildSpotRows(result.spotComparisons).asJava,
      StandardCharsets.UTF_8
    )
    Files.writeString(
      outDir.resolve(ComparisonJsonFileName),
      ujson.write(writeResult(result), indent = 2),
      StandardCharsets.UTF_8
    )

  private def buildSummaryLines(result: RunResult): Vector[String] =
    val aggregate = result.aggregate
    Vector(
      "Holdem CFR External Comparison",
      s"reference: ${result.referenceLabel}",
      s"external: ${result.externalLabel}",
      s"matchedSpots: ${aggregate.matchedSpotCount}/${aggregate.referenceSpotCount}",
      s"matchingSpotSignatures: ${aggregate.matchingSpotSignatureCount}/${aggregate.matchedSpotCount}",
      s"meanTvDistance: ${formatDouble(aggregate.meanTvDistance, 8)}",
      s"maxTvDistance: ${formatDouble(aggregate.maxTvDistance, 8)}",
      s"meanMaxActionProbabilityGap: ${formatDouble(aggregate.meanMaxActionProbabilityGap, 8)}",
      s"bestActionAgreement: ${aggregate.bestActionAgreementCount}/${aggregate.matchedSpotCount} (${formatPercent(aggregate.bestActionAgreementRate)})",
      s"meanEvRmse: ${aggregate.meanEvRmse.map(formatDouble(_, 8)).getOrElse("n/a")}",
      s"maxEvGap: ${aggregate.maxEvGap.map(formatDouble(_, 8)).getOrElse("n/a")}",
      s"gate: ${if result.gate.passed then "PASS" else "FAIL"}",
      s"gateFailures: ${if result.gate.failures.nonEmpty then result.gate.failures.mkString(" | ") else "none"}",
      s"spotSignatureMismatchIds: ${if aggregate.spotSignatureMismatchIds.nonEmpty then aggregate.spotSignatureMismatchIds.mkString(",") else "none"}",
      s"missingExternalSpotSignatureIds: ${if aggregate.missingExternalSpotSignatureIds.nonEmpty then aggregate.missingExternalSpotSignatureIds.mkString(",") else "none"}",
      s"unmatchedReferenceSpots: ${if aggregate.unmatchedReferenceSpotIds.nonEmpty then aggregate.unmatchedReferenceSpotIds.mkString(",") else "none"}",
      s"unmatchedExternalSpots: ${if aggregate.unmatchedExternalSpotIds.nonEmpty then aggregate.unmatchedExternalSpotIds.mkString(",") else "none"}"
    )

  private def buildSpotRows(spotComparisons: Vector[SpotComparison]): Vector[String] =
    SpotComparisonTsvHeader +:
      spotComparisons.map { spot =>
        Vector(
          spot.id,
          spot.spotSignatureStatus,
          spot.referenceBestAction,
          spot.externalBestAction,
          spot.bestActionMatches.toString,
          spot.comparedActions.mkString(","),
          spot.tvDistance.toString,
          spot.maxActionProbabilityGap.toString,
          spot.evRmse.map(_.toString).getOrElse(""),
          spot.evMaxGap.map(_.toString).getOrElse(""),
          spot.actionsMissingInExternal.mkString(","),
          spot.extraActionsInExternal.mkString(",")
        ).mkString("\t")
      }

  private def writeResult(result: RunResult): Value =
    Obj(
      "referenceLabel" -> Str(result.referenceLabel),
      "externalLabel" -> Str(result.externalLabel),
      "aggregate" -> writeAggregate(result.aggregate),
      "gate" -> Obj(
        "passed" -> ujson.Bool(result.gate.passed),
        "failures" -> Arr.from(result.gate.failures.map(Str(_)))
      ),
      "spots" -> Arr.from(result.spotComparisons.map(writeSpotComparison))
    )

  private def writeAggregate(aggregate: AggregateComparison): Value =
    Obj(
      "referenceSpotCount" -> Num(aggregate.referenceSpotCount),
      "externalSpotCount" -> Num(aggregate.externalSpotCount),
      "matchedSpotCount" -> Num(aggregate.matchedSpotCount),
      "matchingSpotSignatureCount" -> Num(aggregate.matchingSpotSignatureCount),
      "meanTvDistance" -> Num(aggregate.meanTvDistance),
      "maxTvDistance" -> Num(aggregate.maxTvDistance),
      "meanMaxActionProbabilityGap" -> Num(aggregate.meanMaxActionProbabilityGap),
      "bestActionAgreementCount" -> Num(aggregate.bestActionAgreementCount),
      "bestActionAgreementRate" -> Num(aggregate.bestActionAgreementRate),
      "meanEvRmse" -> aggregate.meanEvRmse.map(Num(_)).getOrElse(ujson.Null),
      "maxEvGap" -> aggregate.maxEvGap.map(Num(_)).getOrElse(ujson.Null),
      "spotSignatureMismatchIds" -> Arr.from(aggregate.spotSignatureMismatchIds.map(Str(_))),
      "missingExternalSpotSignatureIds" -> Arr.from(aggregate.missingExternalSpotSignatureIds.map(Str(_))),
      "unmatchedReferenceSpotIds" -> Arr.from(aggregate.unmatchedReferenceSpotIds.map(Str(_))),
      "unmatchedExternalSpotIds" -> Arr.from(aggregate.unmatchedExternalSpotIds.map(Str(_)))
    )

  private def writeSpotComparison(spot: SpotComparison): Value =
    Obj(
      "id" -> Str(spot.id),
      "spotSignatureStatus" -> Str(spot.spotSignatureStatus),
      "referenceBestAction" -> Str(spot.referenceBestAction),
      "externalBestAction" -> Str(spot.externalBestAction),
      "bestActionMatches" -> ujson.Bool(spot.bestActionMatches),
      "comparedActions" -> Arr.from(spot.comparedActions.map(Str(_))),
      "tvDistance" -> Num(spot.tvDistance),
      "maxActionProbabilityGap" -> Num(spot.maxActionProbabilityGap),
      "evRmse" -> spot.evRmse.map(Num(_)).getOrElse(ujson.Null),
      "evMaxGap" -> spot.evMaxGap.map(Num(_)).getOrElse(ujson.Null),
      "actionsMissingInExternal" -> Arr.from(spot.actionsMissingInExternal.map(Str(_))),
      "extraActionsInExternal" -> Arr.from(spot.extraActionsInExternal.map(Str(_)))
    )

  private def parseArgs(args: Array[String]): Either[String, CliConfig] =
    if args.contains("--help") || args.contains("-h") then Left(usage)
    else
      for
        options <- CliHelpers.parseOptions(args)
        referencePath <- parseRequiredPath(options, "reference")
        externalPath <- parseRequiredPath(options, "external")
        selectedSpotIds <- parseOptionalSpotIds(options.get("spotIds"))
        maxMeanTvDistance <- parseOptionalDoubleOption(options, "maxMeanTv")
        _ <- validateOptionalNonNegative(maxMeanTvDistance, "maxMeanTv")
        maxSpotTvDistance <- parseOptionalDoubleOption(options, "maxSpotTv")
        _ <- validateOptionalNonNegative(maxSpotTvDistance, "maxSpotTv")
        minBestActionAgreement <- parseOptionalDoubleOption(options, "minBestActionAgreement")
        _ <- validateOptionalUnitInterval(minBestActionAgreement, "minBestActionAgreement")
        maxMeanEvRmse <- parseOptionalDoubleOption(options, "maxMeanEvRmse")
        _ <- validateOptionalNonNegative(maxMeanEvRmse, "maxMeanEvRmse")
      yield
        CliConfig(
          referencePath = referencePath,
          externalPath = externalPath,
          selectedSpotIds = selectedSpotIds,
          thresholds = Thresholds(
            maxMeanTvDistance = maxMeanTvDistance,
            maxSpotTvDistance = maxSpotTvDistance,
            minBestActionAgreement = minBestActionAgreement,
            maxMeanEvRmse = maxMeanEvRmse
          ),
          outDir = options.get("outDir").map(Paths.get(_))
        )

  private def parseRequiredPath(options: Map[String, String], key: String): Either[String, Path] =
    options.get(key) match
      case None => Left(s"--$key is required")
      case Some(raw) =>
        val path = Paths.get(raw)
        if Files.exists(path) then Right(path) else Left(s"--$key path '$raw' does not exist")

  private def parseOptionalDoubleOption(options: Map[String, String], key: String): Either[String, Option[Double]] =
    options.get(key) match
      case None => Right(None)
      case Some(raw) => raw.toDoubleOption.map(Some(_)).toRight(s"--$key must be a number")

  private def parseOptionalSpotIds(raw: Option[String]): Either[String, Option[Set[String]]] =
    raw match
      case None => Right(None)
      case Some(value) =>
        val ids = value.split(",").toVector.map(_.trim).filter(_.nonEmpty)
        if ids.isEmpty then Left("--spotIds must contain at least one non-empty id")
        else Right(Some(ids.toSet))

  private def validateOptionalNonNegative(value: Option[Double], key: String): Either[String, Unit] =
    value match
      case Some(number) if !number.isFinite || number < 0.0 => Left(s"--$key must be >= 0")
      case _ => Right(())

  private def validateOptionalUnitInterval(value: Option[Double], key: String): Either[String, Unit] =
    value match
      case Some(number) if !number.isFinite || number < 0.0 || number > 1.0 =>
        Left(s"--$key must be between 0 and 1")
      case _ => Right(())

  /** Finds all actions tied for the maximum value in the given scores map.
    * Uses BestActionConsistencyEpsilon to handle floating-point ties.
    * Actions are sorted alphabetically for deterministic ordering.
    */
  private def argMaxActions(policy: Map[String, Double], orderedActions: Vector[String]): Vector[String] =
    val actions =
      if orderedActions.nonEmpty then dedupePreservingOrder(orderedActions).sorted
      else policy.keys.toVector.sorted
    require(actions.nonEmpty, "cannot pick argmax from an empty action set")
    val bestValue = actions.map(action => policy.getOrElse(action, Double.NegativeInfinity)).max
    actions.filter(action => policy.getOrElse(action, Double.NegativeInfinity) + BestActionConsistencyEpsilon >= bestValue)

  private def dedupePreservingOrder(values: Vector[String]): Vector[String] =
    val seen = scala.collection.mutable.LinkedHashSet.empty[String]
    values.foreach { value =>
      if value.trim.nonEmpty then seen += value
    }
    seen.toVector

  private def stripJsonSuffix(label: String): String =
    if label.toLowerCase(Locale.ROOT).endsWith(".json") then label.dropRight(5) else label

  private def formatDouble(value: Double, digits: Int): String =
    String.format(Locale.ROOT, s"%.${digits}f", java.lang.Double.valueOf(value))

  private def formatPercent(value: Double): String =
    String.format(Locale.ROOT, "%.1f%%", java.lang.Double.valueOf(value * 100.0))

  private val usage =
    """Usage:
      |  runMain sicfun.holdem.cfr.HoldemCfrExternalComparison [--key=value ...]
      |
      |Options:
      |  --reference=<path>                SICFUN external-comparison.json export
      |  --external=<path>                 External solver/provider export JSON with
      |                                    matching spotSignature or full spot payload
      |  --spotIds=<a,b,...>               Optional subset of spot ids to compare
      |  --outDir=<path>                   Optional output dir for summary/TSV/JSON
      |  --maxMeanTv=<double>              Optional gate on mean TV distance
      |  --maxSpotTv=<double>              Optional gate on worst-spot TV distance
      |  --minBestActionAgreement=<double> Optional gate on best-action agreement [0,1]
      |  --maxMeanEvRmse=<double>          Optional gate on mean EV RMSE
      |                                    At least one threshold is required for PASS
      |""".stripMargin
