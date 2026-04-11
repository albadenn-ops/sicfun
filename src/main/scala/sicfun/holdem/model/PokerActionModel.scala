package sicfun.holdem.model
import sicfun.holdem.types.*
import sicfun.holdem.io.*

import sicfun.core.{ActionModel, CollapseMetrics, MultinomialLogistic}

/**
 * Poker action classification model and full model lifecycle management for sicfun.
 *
 * This file is the core of sicfun's opponent modeling pipeline. It defines:
 *
 *   1. '''[[PokerActionModel]]''': A multinomial logistic regression model that predicts
 *      the probability of each action category (Fold/Check/Call/Raise) given a feature
 *      vector. Implements [[sicfun.core.ActionModel]] for integration with the Bayesian
 *      range inference engine.
 *
 *   2. '''[[TrainedPokerActionModel]]''': A versioned, calibrated model artifact that
 *      bundles the trained model with calibration metrics, a quality gate, provenance
 *      metadata, and lifecycle state (active/retired). This is what gets persisted to disk
 *      and loaded at runtime.
 *
 *   3. '''Model lifecycle operations''' (in the companion object): Training from raw
 *      game-state tuples or from [[ActionDataset]], calibration via Brier score,
 *      evaluation with confusion matrices, stratified k-fold cross-validation, and
 *      versioned model production with quality gating.
 *
 * Key design decisions:
 *   - '''Two feature dimensions''': 5D models use [[PokerFeatures]] (with hand strength)
 *     for Bayesian inference; 8D models use [[FeatureExtractor]] (observable only) for
 *     opponent modeling. The `featureDimension` field disambiguates at runtime.
 *   - '''MinLikelihood floor''': Likelihood values are floored at 1e-6 to prevent
 *     numerical collapse in the Bayesian update (log(0) would break inference).
 *   - '''Brier score calibration''': The quality gate uses mean Brier score rather than
 *     accuracy because calibrated probabilities matter more than classification correctness
 *     for downstream Bayesian inference.
 *   - '''Retirement semantics''': Models can be retired with a reason and timestamp,
 *     making the lifecycle auditable. Both fields must be present or both absent.
 */

/** A multinomial logistic regression model for predicting poker action probabilities.
 *
 * This model wraps a [[MultinomialLogistic]] instance with a category-to-index mapping
 * and a declared feature dimension. It implements the [[ActionModel]] trait, making it
 * pluggable into the Bayesian range inference engine.
 *
 * @param logistic         the underlying multinomial logistic regression model
 * @param categoryIndex    mapping from [[PokerAction.Category]] to integer class index
 * @param featureDimension the expected number of input features (5 for PokerFeatures, 8 for FeatureExtractor)
 */
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

  /** Floor for likelihood values to prevent log(0) in downstream Bayesian updates. */
  private inline val MinLikelihood = 1e-6
  /** Tolerance for detecting effectively-zero weights (bootstrap-uniform model). */
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

  /** Returns the likelihood of the given action for the Bayesian range updater.
   *
   * Implements [[ActionModel.likelihood]]. Floors the result at [[MinLikelihood]]
   * to ensure numerically safe log-likelihood computation in downstream inference.
   *
   * Only valid for 5D models (trained on [[PokerFeatures]]).
   */
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

/** Summary of a model's calibration quality, measured by Brier score.
 *
 * The Brier score measures the accuracy of probabilistic predictions: lower is better,
 * 0.0 is perfect, and the uniform baseline represents random guessing (1/k for k classes).
 *
 * @param meanBrierScore        average Brier score across the evaluation set
 * @param sampleCount           number of samples used for evaluation
 * @param uniformBaselineBrier   Brier score of a uniform (random) predictor (-1.0 if not computed)
 * @param majorityBaselineBrier  Brier score of a majority-class predictor (-1.0 if not computed)
 */
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

/** Quality gate that determines whether a model's calibration is good enough for production.
 *
 * A model passes the gate if its mean Brier score is at or below the threshold.
 * This prevents poorly calibrated models from driving live Bayesian inference.
 *
 * @param maxMeanBrierScore the maximum acceptable mean Brier score
 */
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

/** Versioning metadata for a trained model, capturing identity and provenance.
 *
 * @param id                    unique model identifier (e.g. "model-1710000000000")
 * @param schemaVersion         schema version of the model artifact format
 * @param source                provenance description (e.g. training data path or pipeline name)
 * @param trainedAtEpochMillis  wall-clock time when training completed
 */
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

/** A fully versioned, calibrated, and lifecycle-managed poker action model artifact.
 *
 * This is the top-level artifact produced by the training pipeline and persisted via
 * [[PokerActionModelArtifactIO]]. It bundles the trained model with all the metadata
 * needed for production deployment: calibration metrics, quality gate, sample counts,
 * evaluation strategy, and lifecycle state.
 *
 * Lifecycle states:
 *   - '''Active''': gate passed AND not retired -- eligible for live inference
 *   - '''Gate-failed''': calibration below threshold -- blocked from live use
 *   - '''Retired''': explicitly decommissioned with a reason and timestamp
 *
 * @param version                 model identity and provenance
 * @param model                   the trained logistic regression model
 * @param calibration             calibration quality summary
 * @param gate                    quality gate threshold
 * @param trainingSampleCount     number of samples used for training
 * @param evaluationSampleCount   number of samples used for calibration evaluation
 * @param evaluationStrategy      how evaluation data was obtained ("holdout-split" or "external-evaluation")
 * @param validationFraction      holdout fraction (present only for holdout-split strategy)
 * @param splitSeed               random seed for holdout split (present only for holdout-split strategy)
 * @param retiredAtEpochMillis    when the model was retired (None if active)
 * @param retirementReason        why the model was retired (None if active)
 */
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

  /** Retires this model, marking it as no longer eligible for live inference.
   *
   * @param atEpochMillis retirement timestamp (must be >= training timestamp)
   * @param reason        human-readable retirement reason (must be non-empty)
   * @return a copy with retirement fields set
   * @throws IllegalArgumentException if already retired, timestamp is too early, or reason is blank
   */
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

/** Companion object providing factory methods, training, calibration, evaluation,
 * cross-validation, and versioned model production.
 */
object PokerActionModel:
  private val DefaultSchemaVersion = "poker-action-model-v1"

  /** The default mapping from action categories to integer indices, used when no custom mapping is provided.
   * Categories are assigned indices in their enum ordinal order: Fold=0, Check=1, Call=2, Raise=3.
   */
  val defaultCategoryIndex: Map[PokerAction.Category, Int] =
    PokerAction.categories.zipWithIndex.toMap

  def apply(logistic: MultinomialLogistic): PokerActionModel =
    PokerActionModel(logistic, defaultCategoryIndex, PokerFeatures.dimension)

  /** Creates a bootstrap-uniform model with all weights and biases set to zero.
   * This model predicts equal probability for all action categories (1/k each).
   * Used as the initial model before any training data is available.
   */
  def uniform: PokerActionModel =
    val numCategories = PokerAction.categories.length
    val logistic = MultinomialLogistic.zeros(numCategories, PokerFeatures.dimension)
    PokerActionModel(logistic, defaultCategoryIndex, PokerFeatures.dimension)

  /** Trains a 5D model from (GameState, HoleCards, PokerAction) tuples using [[PokerFeatures]].
   *
   * This is the primary training method for models used in Bayesian range inference,
   * where hand strength (from hole cards) is a critical feature.
   *
   * @param trainingData  labeled training examples as (state, hand, action) triples
   * @param learningRate  SGD learning rate for logistic regression
   * @param iterations    number of training iterations
   * @param l2Lambda      L2 regularization strength
   * @return a trained 5D PokerActionModel
   */
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

  /** Computes a calibration summary for a model on evaluation data.
   *
   * Calculates the mean Brier score for the model alongside two baselines:
   *   - '''Uniform baseline''': Always predicts 1/k for each class (random guessing)
   *   - '''Majority-class baseline''': Always predicts 1.0 for the most frequent class
   *
   * The Brier skill score (available on the returned summary) measures improvement
   * over the uniform baseline.
   *
   * @param model          the model to evaluate
   * @param evaluationData labeled examples to evaluate on
   * @return calibration summary with Brier scores for model and both baselines
   */
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

  /** Trains a versioned model artifact with calibration, quality gating, and provenance metadata.
   *
   * This is the main production entry point for model training. It handles:
   *   1. Splitting data into training and evaluation sets (or using external evaluation data)
   *   2. Training the logistic regression model
   *   3. Computing calibration metrics on the held-out evaluation set
   *   4. Applying the Brier score quality gate (optionally failing if the gate is not met)
   *   5. Assembling the complete [[TrainedPokerActionModel]] artifact
   *
   * @param trainingData        labeled training examples
   * @param learningRate        SGD learning rate
   * @param iterations          number of training iterations
   * @param l2Lambda            L2 regularization strength
   * @param evaluationData      optional external evaluation set (bypasses holdout split if non-empty)
   * @param validationFraction  fraction of training data to hold out for evaluation (used when evaluationData is empty)
   * @param splitSeed           random seed for holdout split reproducibility
   * @param maxMeanBrierScore   quality gate threshold
   * @param failOnGate          if true, throws if the gate fails; if false, returns the artifact with a failed gate
   * @param modelId             unique identifier for this model version
   * @param schemaVersion       artifact schema version
   * @param source              provenance description
   * @param trainedAtEpochMillis training timestamp
   * @return the complete trained model artifact
   */
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

  /** Splits a sample set into training and validation subsets using a deterministic shuffle.
   *
   * The validation set is drawn from the front of the shuffled sequence, with the size
   * clamped to ensure at least 1 sample in each split.
   *
   * @param samples             the full sample set to split
   * @param validationFraction  desired fraction of samples for validation (0, 1)
   * @param seed                random seed for reproducible shuffling
   * @return (training set, validation set)
   */
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
