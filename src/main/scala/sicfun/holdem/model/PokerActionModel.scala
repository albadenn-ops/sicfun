package sicfun.holdem.model
import sicfun.holdem.types.*
import sicfun.holdem.io.*

import sicfun.core.{ActionModel, CollapseMetrics, MultinomialLogistic}

final case class PokerActionModel(
    logistic: MultinomialLogistic,
    categoryIndex: Map[PokerAction.Category, Int],
    featureDimension: Int
) extends ActionModel[GameState, PokerAction, HoleCards]:
  require(categoryIndex.size == logistic.weights.length,
    s"categoryIndex size (${categoryIndex.size}) must match logistic class count (${logistic.weights.length})")
  require(categoryIndex.values.forall(i => i >= 0 && i < logistic.weights.length),
    "all categoryIndex values must be valid class indices")
  require(logistic.weights.isEmpty || featureDimension == logistic.weights.head.length,
    s"featureDimension ($featureDimension) must match logistic feature dimension (${logistic.weights.headOption.map(_.length).getOrElse(0)})")

  private inline val MinLikelihood = 1e-6
  private inline val UniformTolerance = 1e-12

  /** Predict action probabilities from a pre-computed feature vector.
    * Works with any feature dimension matching this model's `featureDimension`.
    */
  def predictFromFeatures(features: Vector[Double]): Vector[Double] =
    require(features.length == featureDimension,
      s"expected $featureDimension features, got ${features.length}")
    logistic.predict(features)

  /** Predict action probabilities from game state and hole cards.
    * Only valid for models trained on [[PokerFeatures]] (5D).
    */
  def categoryProbabilities(state: GameState, hand: HoleCards): Vector[Double] =
    require(featureDimension == PokerFeatures.dimension,
      s"categoryProbabilities(GameState, HoleCards) requires ${PokerFeatures.dimension}D model, got ${featureDimension}D. Use predictFromFeatures() for other dimensions.")
    val features = PokerFeatures.extract(state, hand)
    logistic.predict(features.values)

  def likelihood(action: PokerAction, state: GameState, hand: HoleCards): Double =
    val probs = categoryProbabilities(state, hand)
    val category = action.category
    val idx = categoryIndex.getOrElse(category,
      throw new IllegalArgumentException(s"unknown action category: $category"))
    math.max(probs(idx), MinLikelihood)

  /** True when this model is effectively the bootstrap-uniform model
    * (all weights and biases are numerically zero).
    */
  def isEffectivelyUniform: Boolean =
    logistic.bias.forall(b => math.abs(b) <= UniformTolerance) &&
      logistic.weights.forall(row => row.forall(w => math.abs(w) <= UniformTolerance))

final case class CalibrationSummary(
    meanBrierScore: Double,
    sampleCount: Int,
    uniformBaselineBrier: Double = -1.0,
    majorityBaselineBrier: Double = -1.0
):
  require(meanBrierScore >= 0.0, "meanBrierScore must be non-negative")
  require(sampleCount > 0, "sampleCount must be positive")

  /** Brier skill score relative to the uniform baseline.
    * Positive means the model outperforms random guessing; 1.0 is perfect.
    * Returns NaN if the uniform baseline was not computed.
    */
  def brierSkillScore: Double =
    if uniformBaselineBrier > 0.0 then 1.0 - (meanBrierScore / uniformBaselineBrier)
    else Double.NaN

final case class CalibrationGate(maxMeanBrierScore: Double):
  require(maxMeanBrierScore >= 0.0, "maxMeanBrierScore must be non-negative")

  def passed(summary: CalibrationSummary): Boolean =
    summary.meanBrierScore <= maxMeanBrierScore

/** Per-category classification metrics (one-vs-rest). */
final case class CategoryMetrics(
    category: PokerAction.Category,
    truePositives: Int,
    falsePositives: Int,
    falseNegatives: Int
):
  def precision: Double =
    if truePositives + falsePositives == 0 then 0.0
    else truePositives.toDouble / (truePositives + falsePositives)
  def recall: Double =
    if truePositives + falseNegatives == 0 then 0.0
    else truePositives.toDouble / (truePositives + falseNegatives)
  def f1: Double =
    val p = precision; val r = recall
    if p + r <= 0.0 then 0.0 else 2.0 * p * r / (p + r)

/** Full evaluation report with confusion matrix and per-category metrics. */
final case class EvaluationReport(
    confusionMatrix: Vector[Vector[Int]],
    categoryMetrics: Vector[CategoryMetrics],
    overallAccuracy: Double,
    calibration: CalibrationSummary
)

final case class CrossValidationResult(
    foldBrierScores: Vector[Double],
    meanBrierScore: Double,
    stdBrierScore: Double,
    foldCount: Int
)

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
    PokerActionModel(logistic, defaultCategoryIndex, PokerFeatures.dimension)

  def uniform: PokerActionModel =
    val numCategories = PokerAction.categories.length
    val logistic = MultinomialLogistic.zeros(numCategories, PokerFeatures.dimension)
    PokerActionModel(logistic, defaultCategoryIndex, PokerFeatures.dimension)

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
      val label = catIndex(action.category)
      (features.values, label)
    }
    val logistic = MultinomialLogistic.train(
      examples, numCategories, PokerFeatures.dimension,
      learningRate, iterations, l2Lambda
    )
    PokerActionModel(logistic, catIndex, PokerFeatures.dimension)

  /** Train a model from an [[ActionDataset]] (8D observable features).
    *
    * Models trained this way must use [[predictFromFeatures]] at inference time,
    * not [[categoryProbabilities]], since they use [[FeatureExtractor]] dimensions.
    */
  def trainFromDataset(
      dataset: ActionDataset,
      learningRate: Double = 0.01,
      iterations: Int = 1000,
      l2Lambda: Double = 0.001
  ): PokerActionModel =
    val matrix = dataset.trainingMatrix
    require(matrix.nonEmpty, "dataset must be non-empty")
    val numCategories = PokerAction.categories.length
    val logistic = MultinomialLogistic.train(
      matrix, numCategories, FeatureExtractor.dimension,
      learningRate, iterations, l2Lambda
    )
    PokerActionModel(logistic, defaultCategoryIndex, FeatureExtractor.dimension)

  def calibrationSummary(
      model: PokerActionModel,
      evaluationData: Seq[(GameState, HoleCards, PokerAction)]
  ): CalibrationSummary =
    require(evaluationData.nonEmpty, "evaluationData must be non-empty")
    val catIndex = model.categoryIndex
    val numCategories = catIndex.size

    val predictions = evaluationData.map { case (state, hand, action) =>
      val probs = model.categoryProbabilities(state, hand)
      val actual = catIndex(action.category)
      (probs, actual)
    }

    // Uniform baseline: always predicts 1/k for each class.
    val uniformProb = Vector.fill(numCategories)(1.0 / numCategories)
    val uniformPredictions = evaluationData.map { case (_, _, action) =>
      (uniformProb, catIndex(action.category))
    }

    // Majority-class baseline: always predicts 1.0 for the most frequent class.
    val counts = Array.ofDim[Int](numCategories)
    evaluationData.foreach { case (_, _, action) => counts(catIndex(action.category)) += 1 }
    val majorityClass = counts.indices.maxBy(counts(_))
    val majorityProb = Vector.tabulate(numCategories)(i => if i == majorityClass then 1.0 else 0.0)
    val majorityPredictions = evaluationData.map { case (_, _, action) =>
      (majorityProb, catIndex(action.category))
    }

    CalibrationSummary(
      meanBrierScore = CollapseMetrics.meanBrierScore(predictions),
      sampleCount = predictions.length,
      uniformBaselineBrier = CollapseMetrics.meanBrierScore(uniformPredictions),
      majorityBaselineBrier = CollapseMetrics.meanBrierScore(majorityPredictions)
    )

  /** Compute a full evaluation report with confusion matrix and per-category metrics. */
  def evaluate(
      model: PokerActionModel,
      evaluationData: Seq[(GameState, HoleCards, PokerAction)]
  ): EvaluationReport =
    require(evaluationData.nonEmpty, "evaluationData must be non-empty")
    val catIndex = model.categoryIndex
    val categories = PokerAction.categories
    val numCategories = categories.length
    val matrix = Array.ofDim[Int](numCategories, numCategories) // [actual][predicted]

    evaluationData.foreach { case (state, hand, action) =>
      val probs = model.categoryProbabilities(state, hand)
      val actualIdx = catIndex(action.category)
      val predictedIdx = probs.indices.maxBy(probs(_))
      matrix(actualIdx)(predictedIdx) += 1
    }

    val categoryMetricsVec = categories.zipWithIndex.map { case (cat, idx) =>
      val tp = matrix(idx)(idx)
      val fp = (0 until numCategories).filter(_ != idx).map(matrix(_)(idx)).sum
      val fn = (0 until numCategories).filter(_ != idx).map(matrix(idx)(_)).sum
      CategoryMetrics(cat, tp, fp, fn)
    }

    val correct = (0 until numCategories).map(i => matrix(i)(i)).sum
    val accuracy = correct.toDouble / evaluationData.length

    EvaluationReport(
      confusionMatrix = matrix.map(_.toVector).toVector,
      categoryMetrics = categoryMetricsVec,
      overallAccuracy = accuracy,
      calibration = calibrationSummary(model, evaluationData)
    )

  /** Stratified k-fold cross-validation.
    *
    * Splits data by category for stratification, shuffles with a deterministic seed,
    * trains and evaluates on each fold, and returns aggregate statistics.
    */
  def stratifiedKFoldCV(
      data: Seq[(GameState, HoleCards, PokerAction)],
      k: Int = 5,
      learningRate: Double = 0.01,
      iterations: Int = 1000,
      l2Lambda: Double = 0.001,
      seed: Long = 1L
  ): CrossValidationResult =
    require(data.length >= k, s"need at least k=$k samples, got ${data.length}")
    require(k >= 2, "k must be >= 2")

    val rng = new scala.util.Random(seed)
    val shuffled = rng.shuffle(data.toVector)
    val foldSize = shuffled.length / k

    val brierScores = (0 until k).map { fold =>
      val testStart = fold * foldSize
      val testEnd = if fold == k - 1 then shuffled.length else testStart + foldSize
      val testSet = shuffled.slice(testStart, testEnd)
      val trainSet = shuffled.take(testStart) ++ shuffled.drop(testEnd)
      val model = train(trainSet, learningRate, iterations, l2Lambda)
      calibrationSummary(model, testSet).meanBrierScore
    }.toVector

    val mean = brierScores.sum / brierScores.length
    val std =
      if brierScores.length > 1 then
        math.sqrt(brierScores.map(s => math.pow(s - mean, 2)).sum / (brierScores.length - 1))
      else 0.0

    CrossValidationResult(
      foldBrierScores = brierScores,
      meanBrierScore = mean,
      stdBrierScore = std,
      foldCount = k
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
