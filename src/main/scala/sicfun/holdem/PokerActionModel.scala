package sicfun.holdem

import sicfun.core.{ActionModel, CollapseMetrics, MultinomialLogistic}

final case class PokerActionModel(
    logistic: MultinomialLogistic,
    categoryIndex: Map[PokerAction.Category, Int]
) extends ActionModel[GameState, PokerAction, HoleCards]:
  require(categoryIndex.size == logistic.weights.length,
    s"categoryIndex size (${categoryIndex.size}) must match logistic class count (${logistic.weights.length})")
  require(categoryIndex.values.forall(i => i >= 0 && i < logistic.weights.length),
    "all categoryIndex values must be valid class indices")

  private val MinLikelihood = 1e-6

  def categoryProbabilities(state: GameState, hand: HoleCards): Vector[Double] =
    val features = PokerFeatures.extract(state, hand)
    logistic.predict(features.values)

  def likelihood(action: PokerAction, state: GameState, hand: HoleCards): Double =
    val probs = categoryProbabilities(state, hand)
    val category = PokerAction.categoryOf(action)
    val idx = categoryIndex.getOrElse(category,
      throw new IllegalArgumentException(s"unknown action category: $category"))
    math.max(probs(idx), MinLikelihood)

final case class CalibrationSummary(meanBrierScore: Double, sampleCount: Int):
  require(meanBrierScore >= 0.0, "meanBrierScore must be non-negative")
  require(sampleCount > 0, "sampleCount must be positive")

final case class CalibrationGate(maxMeanBrierScore: Double):
  require(maxMeanBrierScore >= 0.0, "maxMeanBrierScore must be non-negative")

  def passed(summary: CalibrationSummary): Boolean =
    summary.meanBrierScore <= maxMeanBrierScore

final case class ModelVersion(
    id: String,
    schemaVersion: String,
    source: String,
    trainedAtEpochMillis: Long
):
  require(id.trim.nonEmpty, "id must be non-empty")
  require(schemaVersion.trim.nonEmpty, "schemaVersion must be non-empty")
  require(source.trim.nonEmpty, "source must be non-empty")
  require(trainedAtEpochMillis >= 0L, "trainedAtEpochMillis must be non-negative")

final case class TrainedPokerActionModel(
    version: ModelVersion,
    model: PokerActionModel,
    calibration: CalibrationSummary,
    gate: CalibrationGate,
    trainingSampleCount: Int,
    evaluationSampleCount: Int,
    evaluationStrategy: String,
    validationFraction: Option[Double],
    splitSeed: Option[Long],
    retiredAtEpochMillis: Option[Long] = None,
    retirementReason: Option[String] = None
):
  require(trainingSampleCount > 0, "trainingSampleCount must be positive")
  require(evaluationSampleCount > 0, "evaluationSampleCount must be positive")
  require(evaluationStrategy.trim.nonEmpty, "evaluationStrategy must be non-empty")
  validationFraction.foreach { fraction =>
    require(fraction > 0.0 && fraction < 1.0, "validationFraction must be in (0, 1)")
  }
  retiredAtEpochMillis.foreach { retiredAt =>
    require(retiredAt >= version.trainedAtEpochMillis,
      "retiredAtEpochMillis must be >= trainedAtEpochMillis")
  }
  retirementReason.foreach { reason =>
    require(reason.trim.nonEmpty, "retirementReason must be non-empty when present")
  }
  require(
    retiredAtEpochMillis.isDefined == retirementReason.isDefined,
    "retiredAtEpochMillis and retirementReason must be both defined or both empty"
  )

  def gatePassed: Boolean = gate.passed(calibration)

  def isRetired: Boolean = retiredAtEpochMillis.nonEmpty

  def isActive: Boolean = gatePassed && !isRetired

  def retire(atEpochMillis: Long, reason: String): TrainedPokerActionModel =
    require(!isRetired, "model is already retired")
    require(atEpochMillis >= version.trainedAtEpochMillis,
      "retirement timestamp must be >= training timestamp")
    val cleanReason = reason.trim
    require(cleanReason.nonEmpty, "retirement reason must be non-empty")
    copy(
      retiredAtEpochMillis = Some(atEpochMillis),
      retirementReason = Some(cleanReason)
    )

object PokerActionModel:
  private val DefaultSchemaVersion = "poker-action-model-v1"

  val defaultCategoryIndex: Map[PokerAction.Category, Int] =
    PokerAction.categories.zipWithIndex.toMap

  def apply(logistic: MultinomialLogistic): PokerActionModel =
    PokerActionModel(logistic, defaultCategoryIndex)

  def uniform: PokerActionModel =
    val numCategories = PokerAction.categories.length
    val logistic = MultinomialLogistic.zeros(numCategories, PokerFeatures.dimension)
    PokerActionModel(logistic, defaultCategoryIndex)

  def train(
      trainingData: Seq[(GameState, HoleCards, PokerAction)],
      learningRate: Double = 0.01,
      iterations: Int = 1000,
      l2Lambda: Double = 0.001
  ): PokerActionModel =
    val numCategories = PokerAction.categories.length
    val catIndex = defaultCategoryIndex
    val examples = trainingData.map { case (state, hand, action) =>
      val features = PokerFeatures.extract(state, hand)
      val label = catIndex(PokerAction.categoryOf(action))
      (features.values, label)
    }
    val logistic = MultinomialLogistic.train(
      examples, numCategories, PokerFeatures.dimension,
      learningRate, iterations, l2Lambda
    )
    PokerActionModel(logistic, catIndex)

  def calibrationSummary(
      model: PokerActionModel,
      evaluationData: Seq[(GameState, HoleCards, PokerAction)]
  ): CalibrationSummary =
    require(evaluationData.nonEmpty, "evaluationData must be non-empty")
    val predictions = evaluationData.map { case (state, hand, action) =>
      val probs = model.categoryProbabilities(state, hand)
      val actual = model.categoryIndex(PokerAction.categoryOf(action))
      (probs, actual)
    }
    CalibrationSummary(
      meanBrierScore = CollapseMetrics.meanBrierScore(predictions),
      sampleCount = predictions.length
    )

  def trainVersioned(
      trainingData: Seq[(GameState, HoleCards, PokerAction)],
      learningRate: Double = 0.01,
      iterations: Int = 1000,
      l2Lambda: Double = 0.001,
      evaluationData: Seq[(GameState, HoleCards, PokerAction)] = Seq.empty,
      validationFraction: Double = 0.2,
      splitSeed: Long = 1L,
      maxMeanBrierScore: Double = 1.0,
      failOnGate: Boolean = false,
      modelId: String = s"model-${System.currentTimeMillis()}",
      schemaVersion: String = DefaultSchemaVersion,
      source: String = "unknown",
      trainedAtEpochMillis: Long = System.currentTimeMillis()
  ): TrainedPokerActionModel =
    require(trainingData.nonEmpty, "trainingData must be non-empty")
    val (trainSet, evalSet, strategy, holdoutFraction, holdoutSeed) =
      if evaluationData.nonEmpty then
        (trainingData.toVector, evaluationData.toVector, "external-evaluation", None, None)
      else
        val (trainSplit, evalSplit) = splitTrainingValidation(trainingData, validationFraction, splitSeed)
        (trainSplit, evalSplit, "holdout-split", Some(validationFraction), Some(splitSeed))

    val model = train(trainSet, learningRate, iterations, l2Lambda)
    val eval = evalSet
    val summary = calibrationSummary(model, eval)
    val gate = CalibrationGate(maxMeanBrierScore)
    val artifact = TrainedPokerActionModel(
      version = ModelVersion(
        id = modelId.trim,
        schemaVersion = schemaVersion.trim,
        source = source.trim,
        trainedAtEpochMillis = trainedAtEpochMillis
      ),
      model = model,
      calibration = summary,
      gate = gate,
      trainingSampleCount = trainSet.length,
      evaluationSampleCount = evalSet.length,
      evaluationStrategy = strategy,
      validationFraction = holdoutFraction,
      splitSeed = holdoutSeed
    )
    if failOnGate then
      require(
        artifact.gatePassed,
        s"calibration gate failed: meanBrierScore=${summary.meanBrierScore} > maxMeanBrierScore=${gate.maxMeanBrierScore}"
      )
    artifact

  private def splitTrainingValidation[A](
      samples: Seq[A],
      validationFraction: Double,
      seed: Long
  ): (Vector[A], Vector[A]) =
    require(samples.length >= 2, "at least 2 samples are required for holdout split")
    require(validationFraction > 0.0 && validationFraction < 1.0,
      "validationFraction must be in (0, 1) when evaluationData is empty")

    val shuffled = new scala.util.Random(seed).shuffle(samples.toVector)
    val rawValidation = math.round(shuffled.length.toDouble * validationFraction).toInt
    val validationCount = math.max(1, math.min(shuffled.length - 1, rawValidation))
    val (evaluation, training) = shuffled.splitAt(validationCount)
    (training, evaluation)
