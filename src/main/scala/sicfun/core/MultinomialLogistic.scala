package sicfun.core

/** Multinomial logistic regression model (softmax classifier) implemented from scratch.
  *
  * Maps a d-dimensional feature vector to a probability distribution over k classes
  * using the softmax function: P(class=i | x) = softmax(W*x + b),,i,,
  *
  * The model is immutable; training produces a new instance via [[MultinomialLogistic.train]].
  *
  * @param weights weight matrix of shape (k, d) where k = number of classes, d = feature dimension.
  *                `weights(i)(j)` is the weight for class i, feature j.
  * @param bias    bias vector of length k, one per class
  */
final case class MultinomialLogistic(weights: Vector[Vector[Double]], bias: Vector[Double]):
  require(weights.nonEmpty, "must have at least one class")
  require(weights.length == bias.length, "weights and bias must have same number of classes")
  private val k: Int = weights.length   // number of classes
  private val d: Int = weights.head.length // feature dimension
  require(weights.forall(_.length == d), "all weight vectors must have same dimension")

  /** Predicts class probabilities for a feature vector.
    *
    * Computes logits = W*x + b, then applies the softmax function.
    *
    * @param features input feature vector of length d
    * @return probability vector of length k summing to 1.0
    */
  def predict(features: Vector[Double]): Vector[Double] =
    require(features.length == d, s"expected $d features, got ${features.length}")
    val logits = (0 until k).map { i =>
      var sum = bias(i)
      var j = 0
      while j < d do
        sum += weights(i)(j) * features(j)
        j += 1
      sum
    }.toVector
    softmax(logits)

  /** Numerically stable softmax: subtracts the max logit before exponentiating
    * to prevent overflow (exp of large positive) while preserving the output distribution.
    */
  private inline def softmax(logits: Vector[Double]): Vector[Double] =
    val maxLogit = logits.max
    val exps = logits.map(l => math.exp(l - maxLogit)) // shift for numerical stability
    val sum = exps.sum
    exps.map(_ / sum)

/** Factory and training methods for [[MultinomialLogistic]]. */
object MultinomialLogistic:
  /** Creates a zero-initialized model. Useful as a starting point or baseline. */
  def zeros(numClasses: Int, numFeatures: Int): MultinomialLogistic =
    MultinomialLogistic(
      Vector.fill(numClasses)(Vector.fill(numFeatures)(0.0)),
      Vector.fill(numClasses)(0.0)
    )

  /** Trains a multinomial logistic regression model via batch gradient descent.
    *
    * ==Training loop==
    * Each iteration processes all examples (full-batch gradient descent):
    *  1. '''Forward pass''': compute logits = W*x + b for each example
    *  2. '''Softmax''': convert logits to probabilities (numerically stable, subtracting max logit)
    *  3. '''Gradient''': compute cross-entropy gradient: dL/d(logit,,c,,) = P(c|x) - 1{c == label}
    *  4. '''Weight update''': W -= lr * (grad/n + lambda*W), b -= lr * (grad/n)
    *
    * The gradient of cross-entropy loss with respect to logits is simply (softmax output - one-hot target),
    * which is backpropagated to weights via the chain rule.
    *
    * L2 regularization (weight decay) is applied to weights but not biases,
    * penalizing large weight magnitudes to reduce overfitting.
    *
    * @param examples     training data as (feature vector, class label) pairs
    * @param numClasses   number of output classes (k >= 2)
    * @param numFeatures  dimensionality of input features (d >= 1)
    * @param learningRate step size for gradient descent (default 0.01)
    * @param iterations   number of full passes over all examples (default 1000)
    * @param l2Lambda     L2 regularization coefficient (default 0.001)
    * @return a trained [[MultinomialLogistic]] model
    */
  def train(
      examples: Seq[(Vector[Double], Int)],
      numClasses: Int,
      numFeatures: Int,
      learningRate: Double = 0.01,
      iterations: Int = 1000,
      l2Lambda: Double = 0.001
  ): MultinomialLogistic =
    require(examples.nonEmpty, "training requires at least one example")
    require(numClasses >= 2, "need at least 2 classes")
    require(numFeatures >= 1, "need at least 1 feature")
    require(iterations > 0, "iterations must be positive")
    examples.foreach { case (features, label) =>
      require(features.length == numFeatures,
        s"example has ${features.length} features, expected $numFeatures")
      require(label >= 0 && label < numClasses,
        s"label $label out of range [0, $numClasses)")
    }

    // Mutable arrays for in-place updates during training.
    val w = Array.ofDim[Double](numClasses, numFeatures)
    val b = new Array[Double](numClasses)
    val n = examples.length.toDouble

    var iter = 0
    while iter < iterations do
      // Accumulate gradients across the full batch before updating.
      val gradW = Array.ofDim[Double](numClasses, numFeatures)
      val gradB = new Array[Double](numClasses)

      examples.foreach { case (features, label) =>
        // Forward pass: compute logits for this example.
        val logits = new Array[Double](numClasses)
        var c = 0
        while c < numClasses do
          var sum = b(c)
          var f = 0
          while f < numFeatures do
            sum += w(c)(f) * features(f)
            f += 1
          logits(c) = sum
          c += 1

        // Numerically stable softmax: subtract max logit before exp.
        val maxLogit = logits.max
        val exps = logits.map(l => math.exp(l - maxLogit))
        val sumExp = exps.sum
        val probs = exps.map(_ / sumExp)

        // Cross-entropy gradient: softmax output minus one-hot target.
        c = 0
        while c < numClasses do
          val indicator = if c == label then 1.0 else 0.0
          val diff = probs(c) - indicator
          gradB(c) += diff
          var f = 0
          while f < numFeatures do
            gradW(c)(f) += diff * features(f)
            f += 1
          c += 1
      }

      // Parameter update: average gradient + L2 regularization on weights only.
      var c = 0
      while c < numClasses do
        b(c) -= learningRate * (gradB(c) / n)
        var f = 0
        while f < numFeatures do
          w(c)(f) -= learningRate * (gradW(c)(f) / n + l2Lambda * w(c)(f))
          f += 1
        c += 1

      iter += 1

    MultinomialLogistic(
      (0 until numClasses).map(c => w(c).toVector).toVector,
      b.toVector
    )
