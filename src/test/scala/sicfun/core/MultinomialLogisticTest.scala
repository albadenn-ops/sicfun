package sicfun.core

import munit.FunSuite

/** Tests for [[MultinomialLogistic]] softmax classifier (training and inference).
  *
  * Coverage includes:
  *  - '''Zero model''': a zero-weight model predicts uniform probabilities across classes.
  *  - '''Softmax normalization''': predicted probabilities always sum to 1.
  *  - '''Binary classification''': after training on linearly separable 2D data, the model
  *    assigns highest probability to the correct class for test inputs.
  *  - '''Multi-class''': training with 3 classes converges to correct dominant predictions.
  *  - '''L2 regularization''': regularized weights have smaller magnitude than unregularized.
  *  - '''Input validation''': wrong feature dimensions, out-of-range labels, non-positive
  *    iterations, empty datasets, and inconsistent weight/bias shapes are all rejected.
  */
class MultinomialLogisticTest extends FunSuite:
  test("zeros model predicts uniform probabilities") {
    val model = MultinomialLogistic.zeros(3, 2)
    val probs = model.predict(Vector(1.0, 2.0))
    probs.foreach { p =>
      assert(math.abs(p - 1.0 / 3.0) < 1e-9)
    }
  }

  test("softmax probabilities sum to 1") {
    val model = MultinomialLogistic(
      Vector(Vector(1.0, -1.0), Vector(-1.0, 1.0), Vector(0.0, 0.0)),
      Vector(0.5, -0.5, 0.0)
    )
    val probs = model.predict(Vector(2.0, 3.0))
    assert(math.abs(probs.sum - 1.0) < 1e-9)
  }

  test("predict assigns highest probability to correct class after training") {
    val examples = Seq(
      (Vector(1.0, 0.0), 0),
      (Vector(0.9, 0.1), 0),
      (Vector(0.8, 0.2), 0),
      (Vector(0.0, 1.0), 1),
      (Vector(0.1, 0.9), 1),
      (Vector(0.2, 0.8), 1)
    )
    val model = MultinomialLogistic.train(examples, numClasses = 2, numFeatures = 2,
      learningRate = 0.1, iterations = 500)
    val prob0 = model.predict(Vector(1.0, 0.0))
    val prob1 = model.predict(Vector(0.0, 1.0))
    assert(prob0(0) > prob0(1), s"expected class 0 dominant, got $prob0")
    assert(prob1(1) > prob1(0), s"expected class 1 dominant, got $prob1")
  }

  test("train with 3 classes converges") {
    val base = Seq(
      (Vector(0.9, 0.05, 0.05), 0),
      (Vector(0.05, 0.9, 0.05), 1),
      (Vector(0.05, 0.05, 0.9), 2)
    )
    val examples = Seq.fill(10)(base).flatten
    val model = MultinomialLogistic.train(examples, numClasses = 3, numFeatures = 3,
      learningRate = 0.1, iterations = 1000)
    val probs = model.predict(Vector(1.0, 0.0, 0.0))
    assertEquals(probs.indices.maxBy(probs), 0)
  }

  test("l2 regularization constrains weights") {
    val examples = Seq((Vector(100.0), 0), (Vector(-100.0), 1))
    val regularized = MultinomialLogistic.train(examples, 2, 1, l2Lambda = 1.0, iterations = 100)
    val unregularized = MultinomialLogistic.train(examples, 2, 1, l2Lambda = 0.0, iterations = 100)
    val regMaxWeight = regularized.weights.flatten.map(math.abs).max
    val unregMaxWeight = unregularized.weights.flatten.map(math.abs).max
    assert(regMaxWeight < unregMaxWeight, "regularization should constrain weights")
  }

  test("train rejects wrong feature dimension") {
    val examples = Seq((Vector(1.0, 2.0, 3.0), 0))
    intercept[IllegalArgumentException] {
      MultinomialLogistic.train(examples, numClasses = 2, numFeatures = 2)
    }
  }

  test("train rejects out-of-range label") {
    val examples = Seq((Vector(1.0), 5))
    intercept[IllegalArgumentException] {
      MultinomialLogistic.train(examples, numClasses = 2, numFeatures = 1)
    }
  }

  test("train rejects non-positive iterations") {
    // Regression: iterations <= 0 silently returned a zero-weight model.
    val examples = Seq((Vector(1.0, 0.0), 0), (Vector(0.0, 1.0), 1))
    intercept[IllegalArgumentException] {
      MultinomialLogistic.train(examples, numClasses = 2, numFeatures = 2, iterations = 0)
    }
    intercept[IllegalArgumentException] {
      MultinomialLogistic.train(examples, numClasses = 2, numFeatures = 2, iterations = -5)
    }
  }

  test("train rejects empty dataset and invalid class/feature counts") {
    intercept[IllegalArgumentException] {
      MultinomialLogistic.train(Seq.empty, numClasses = 2, numFeatures = 1)
    }
    intercept[IllegalArgumentException] {
      MultinomialLogistic.train(Seq((Vector(1.0), 0)), numClasses = 1, numFeatures = 1)
    }
    intercept[IllegalArgumentException] {
      MultinomialLogistic.train(Seq((Vector(1.0), 0)), numClasses = 2, numFeatures = 0)
    }
  }

  test("predict rejects wrong feature dimension") {
    val model = MultinomialLogistic.zeros(2, 3)
    intercept[IllegalArgumentException] {
      model.predict(Vector(1.0, 2.0))
    }
  }

  test("constructor validates weight/bias shape consistency") {
    intercept[IllegalArgumentException] {
      MultinomialLogistic(
        weights = Vector(Vector(1.0), Vector(2.0)),
        bias = Vector(0.0)
      )
    }
    intercept[IllegalArgumentException] {
      MultinomialLogistic(
        weights = Vector(Vector(1.0, 2.0), Vector(3.0)),
        bias = Vector(0.0, 0.0)
      )
    }
  }
