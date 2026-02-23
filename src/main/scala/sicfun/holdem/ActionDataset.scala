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

final case class ActionDataset(
    examples: Vector[ActionExample],
    provenance: DatasetProvenance
):
  def trainingMatrix: Vector[(Vector[Double], Int)] =
    examples.map(example => (example.features, example.label))

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
      val category = PokerAction.categoryOf(event.action)
      require(categoryIndex.contains(category), s"missing class index for category $category")
    }

    val examples = ordered.map { event =>
      val features = FeatureExtractor.extract(event)
      require(
        features.dimension == FeatureExtractor.dimension,
        s"feature extractor returned ${features.dimension} features, expected ${FeatureExtractor.dimension}"
      )
      val category = PokerAction.categoryOf(event.action)
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
