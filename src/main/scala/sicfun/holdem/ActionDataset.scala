package sicfun.holdem

final case class ActionExample(
    handId: String,
    sequenceInHand: Long,
    playerId: String,
    occurredAtEpochMillis: Long,
    features: Vector[Double],
    label: Int
)

final case class DatasetProvenance(
    schemaVersion: String,
    source: String,
    generatedAtEpochMillis: Long,
    eventCount: Int,
    uniqueHandCount: Int,
    featureNames: Vector[String],
    labelMapping: Map[PokerAction.Category, Int]
)

final case class FeatureStatistics(
    name: String,
    mean: Double,
    std: Double,
    min: Double,
    max: Double
)

final case class DatasetStatistics(
    totalExamples: Int,
    classDistribution: Map[PokerAction.Category, Int],
    featureStatistics: Vector[FeatureStatistics]
)

final case class ActionDataset(
    examples: Vector[ActionExample],
    provenance: DatasetProvenance
):
  def trainingMatrix: Vector[(Vector[Double], Int)] =
    examples.map(example => (example.features, example.label))

  /** Compute class distribution and feature statistics for this dataset. */
  def statistics: DatasetStatistics =
    require(examples.nonEmpty, "cannot compute statistics on empty dataset")
    val numFeatures = provenance.featureNames.length
    val reverseLabel = provenance.labelMapping.map { case (cat, idx) => idx -> cat }

    val classCounts = scala.collection.mutable.Map.empty[PokerAction.Category, Int]
    examples.foreach { ex =>
      val cat = reverseLabel(ex.label)
      classCounts(cat) = classCounts.getOrElse(cat, 0) + 1
    }

    val featureStats = (0 until numFeatures).map { f =>
      val values = examples.map(_.features(f))
      val n = values.length
      val mean = values.sum / n
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

object DatasetBuilder:
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

  private def sortEvents(events: Seq[PokerEvent]): Seq[PokerEvent] =
    events.sortBy(event =>
      (event.handId, event.sequenceInHand, event.occurredAtEpochMillis, event.playerId)
    )

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
