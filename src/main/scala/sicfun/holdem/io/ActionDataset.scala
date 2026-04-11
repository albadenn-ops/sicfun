package sicfun.holdem.io
import sicfun.holdem.types.*
import sicfun.holdem.model.*

/**
 * Action dataset domain model and builder for the sicfun poker analytics system.
 *
 * This file defines the data structures that represent a labeled training dataset
 * for poker action classification. Each [[ActionExample]] pairs an observed poker
 * decision point (encoded as a fixed-dimension feature vector) with a categorical
 * label (Fold/Check/Call/Raise). The [[ActionDataset]] aggregates examples with
 * [[DatasetProvenance]] metadata for reproducibility and audit.
 *
 * Key design decisions:
 *   - Features are pre-extracted at dataset build time (not lazily) to ensure
 *     consistency between training and evaluation.
 *   - The [[DatasetBuilder]] enforces temporal ordering within each hand to catch
 *     data corruption (duplicate sequences, timestamp regressions) early.
 *   - Output is deterministic regardless of input ordering: events are sorted by
 *     (handId, sequenceInHand, timestamp, playerId) before feature extraction.
 *   - The category-to-index mapping is explicit in provenance, avoiding silent
 *     label mismatches when models are trained on different dataset versions.
 */

/** A single labeled training example extracted from a poker event.
 *
 * @param handId                unique identifier of the poker hand
 * @param sequenceInHand        ordinal position of this action within the hand
 * @param playerId              identifier of the acting player
 * @param occurredAtEpochMillis wall-clock timestamp of the original action
 * @param features              pre-extracted normalized feature vector (dimension matches [[FeatureExtractor.dimension]])
 * @param label                 integer class label corresponding to the action category via [[DatasetProvenance.labelMapping]]
 */
final case class ActionExample(
    handId: String,
    sequenceInHand: Long,
    playerId: String,
    occurredAtEpochMillis: Long,
    features: Vector[Double],
    label: Int
)

/** Provenance metadata recording how a dataset was generated, enabling reproducibility.
 *
 * @param schemaVersion          version of the [[PokerEvent]] schema used during generation
 * @param source                 human-readable origin description (e.g. filename, pipeline name)
 * @param generatedAtEpochMillis wall-clock time when the dataset was built
 * @param eventCount             total number of examples in the dataset
 * @param uniqueHandCount        number of distinct poker hands represented
 * @param featureNames           ordered names of feature dimensions (matches [[FeatureExtractor.featureNames]])
 * @param labelMapping           mapping from action category to integer label index
 */
final case class DatasetProvenance(
    schemaVersion: String,
    source: String,
    generatedAtEpochMillis: Long,
    eventCount: Int,
    uniqueHandCount: Int,
    featureNames: Vector[String],
    labelMapping: Map[PokerAction.Category, Int]
)

/** Descriptive statistics for a single feature dimension across all examples.
 *
 * @param name feature name (from [[DatasetProvenance.featureNames]])
 * @param mean arithmetic mean of this feature's values
 * @param std  sample standard deviation (Bessel-corrected, n-1 denominator)
 * @param min  minimum observed value
 * @param max  maximum observed value
 */
final case class FeatureStatistics(
    name: String,
    mean: Double,
    std: Double,
    min: Double,
    max: Double
)

/** Aggregate statistics for an entire dataset, including class balance and per-feature summaries.
 *
 * @param totalExamples     total number of labeled examples
 * @param classDistribution count of examples per action category (Fold, Check, Call, Raise)
 * @param featureStatistics per-feature descriptive statistics (one entry per dimension)
 */
final case class DatasetStatistics(
    totalExamples: Int,
    classDistribution: Map[PokerAction.Category, Int],
    featureStatistics: Vector[FeatureStatistics]
)

/** A complete labeled dataset for training poker action classification models.
 *
 * Pairs a vector of [[ActionExample]] instances with [[DatasetProvenance]] metadata.
 * Provides convenience methods for extracting the training matrix and computing
 * descriptive statistics.
 *
 * @param examples   ordered vector of labeled training examples
 * @param provenance metadata describing how this dataset was generated
 */
final case class ActionDataset(
    examples: Vector[ActionExample],
    provenance: DatasetProvenance
):
  /** Returns the (features, label) pairs suitable for direct consumption by
   * [[sicfun.core.MultinomialLogistic.train]].
   */
  def trainingMatrix: Vector[(Vector[Double], Int)] =
    examples.map(example => (example.features, example.label))

  /** Compute class distribution and feature statistics for this dataset. */
  def statistics: DatasetStatistics =
    require(examples.nonEmpty, "cannot compute statistics on empty dataset")
    val numFeatures = provenance.featureNames.length
    val reverseLabel = provenance.labelMapping.map { case (cat, idx) => idx -> cat }

    // Count examples per action category by reversing the label mapping
    val classCounts = scala.collection.mutable.Map.empty[PokerAction.Category, Int]
    examples.foreach { ex =>
      val cat = reverseLabel(ex.label)
      classCounts(cat) = classCounts.getOrElse(cat, 0) + 1
    }

    // Compute per-feature descriptive statistics (mean, std, min, max)
    val featureStats = (0 until numFeatures).map { f =>
      val values = examples.map(_.features(f))
      val n = values.length
      val mean = values.sum / n
      // Bessel-corrected variance (n-1 denominator) for sample standard deviation
      val variance = if n > 1 then values.map(v => math.pow(v - mean, 2)).sum / (n - 1) else 0.0
      FeatureStatistics(
        name = provenance.featureNames(f),
        mean = mean,
        std = math.sqrt(variance),
        min = values.min,
        max = values.max
      )
    }.toVector

    DatasetStatistics(
      totalExamples = examples.length,
      classDistribution = classCounts.toMap,
      featureStatistics = featureStats
    )

/** Factory for constructing [[ActionDataset]] instances from raw [[PokerEvent]] sequences.
 *
 * The builder validates input integrity (non-empty events, unique sequences per hand,
 * monotonic timestamps within each hand), extracts features via [[FeatureExtractor]],
 * and produces a deterministic output regardless of input ordering.
 */
object DatasetBuilder:
  /** Builds an [[ActionDataset]] from a sequence of poker events.
   *
   * The method performs the following steps:
   *   1. Validates all preconditions (non-empty input, valid source/schema, complete category coverage)
   *   2. Validates temporal ordering within each hand (no duplicate sequences, no timestamp regressions)
   *   3. Sorts events deterministically by (handId, sequenceInHand, timestamp, playerId)
   *   4. Extracts features from each event via [[FeatureExtractor.extract]]
   *   5. Maps action categories to integer labels via the provided categoryIndex
   *
   * @param events                  raw poker events to convert into training examples
   * @param source                  provenance description (e.g. "hand-history-file-2026.tsv")
   * @param generatedAtEpochMillis  generation timestamp (defaults to current time)
   * @param schemaVersion           event schema version (defaults to [[PokerEvent.SchemaVersion]])
   * @param categoryIndex           mapping from action category to integer label index
   * @return a fully populated [[ActionDataset]] with provenance metadata
   * @throws IllegalArgumentException if any validation check fails
   */
  def build(
      events: Seq[PokerEvent],
      source: String,
      generatedAtEpochMillis: Long = System.currentTimeMillis(),
      schemaVersion: String = PokerEvent.SchemaVersion,
      categoryIndex: Map[PokerAction.Category, Int] = PokerActionModel.defaultCategoryIndex
  ): ActionDataset =
    require(events.nonEmpty, "events must be non-empty")
    require(source.trim.nonEmpty, "source must be non-empty")
    require(generatedAtEpochMillis >= 0L, "generatedAtEpochMillis must be non-negative")
    require(schemaVersion.trim.nonEmpty, "schemaVersion must be non-empty")
    require(categoryIndex.nonEmpty, "categoryIndex must be non-empty")
    require(categoryIndex.values.forall(_ >= 0), "categoryIndex values must be non-negative")
    require(categoryIndex.values.toSet.size == categoryIndex.size, "categoryIndex values must be unique")

    validateTemporalOrder(events)
    val ordered = sortEvents(events)

    ordered.foreach { event =>
      val category = event.action.category
      require(categoryIndex.contains(category), s"missing class index for category $category")
    }

    val examples = ordered.map { event =>
      val features = FeatureExtractor.extract(event)
      require(
        features.dimension == FeatureExtractor.dimension,
        s"feature extractor returned ${features.dimension} features, expected ${FeatureExtractor.dimension}"
      )
      val category = event.action.category
      ActionExample(
        handId = event.handId,
        sequenceInHand = event.sequenceInHand,
        playerId = event.playerId,
        occurredAtEpochMillis = event.occurredAtEpochMillis,
        features = features.values,
        label = categoryIndex(category)
      )
    }.toVector

    val provenance = DatasetProvenance(
      schemaVersion = schemaVersion.trim,
      source = source.trim,
      generatedAtEpochMillis = generatedAtEpochMillis,
      eventCount = examples.length,
      uniqueHandCount = ordered.map(_.handId).distinct.length,
      featureNames = FeatureExtractor.featureNames,
      labelMapping = categoryIndex.toVector.sortBy(_._2).toMap
    )
    ActionDataset(examples, provenance)

  /** Sorts events into a deterministic canonical order for reproducible dataset generation.
   * The sort key is (handId, sequenceInHand, timestamp, playerId), ensuring that the same
   * set of events always produces identical output regardless of input ordering.
   */
  private def sortEvents(events: Seq[PokerEvent]): Seq[PokerEvent] =
    events.sortBy(event =>
      (event.handId, event.sequenceInHand, event.occurredAtEpochMillis, event.playerId)
    )

  /** Validates temporal integrity within each hand:
   *   - No duplicate sequenceInHand values within the same hand
   *   - Timestamps are monotonically non-decreasing when ordered by sequence
   *
   * These checks detect data corruption before it silently pollutes training data.
   */
  private def validateTemporalOrder(events: Seq[PokerEvent]): Unit =
    events.groupBy(_.handId).foreach { case (handId, handEvents) =>
      val bySequence = handEvents.sortBy(_.sequenceInHand)
      val uniqueSequences = bySequence.map(_.sequenceInHand).toSet
      require(
        uniqueSequences.size == bySequence.length,
        s"hand $handId contains duplicate sequenceInHand values"
      )

      bySequence.sliding(2).foreach {
        case Seq(previous, current) =>
          require(
            previous.occurredAtEpochMillis <= current.occurredAtEpochMillis,
            s"hand $handId has timestamp regression between sequence ${previous.sequenceInHand} and ${current.sequenceInHand}"
          )
        case _ => ()
      }
    }
