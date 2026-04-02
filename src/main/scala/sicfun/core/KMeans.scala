package sicfun.core

import scala.util.Random

/** Hyperparameters for k-means clustering. */
final case class KMeansConfig(
    k: Int,
    maxIterations: Int = 100,
    convergenceThreshold: Double = 1e-6,
    seed: Long = 42L
):
  require(k >= 1, "k must be >= 1")
  require(maxIterations >= 1, "maxIterations must be >= 1")
  require(convergenceThreshold >= 0.0, "convergenceThreshold must be >= 0")

/** Result of a k-means fit run. */
final case class KMeansResult(
    centroids: Vector[Vector[Double]],
    assignments: Vector[Int],
    inertia: Double,
    iterationsRun: Int,
    converged: Boolean
)

/** Lightweight k-means implementation for dense Double vectors. */
object KMeans:
  def fit(points: Vector[Vector[Double]], config: KMeansConfig): KMeansResult =
    require(points.nonEmpty, "points must be non-empty")
    val dimension = validatePoints(points)
    require(config.k <= points.length, s"k (${config.k}) cannot exceed point count (${points.length})")

    val rng = new Random(config.seed)
    var centroids = initializeCentroids(points, config.k, rng)
    var assignments = Vector.fill(points.length)(0)
    var iterations = 0
    var converged = false

    while iterations < config.maxIterations && !converged do
      assignments = points.map(point => assignInternal(point, centroids, dimension))

      val (nextCentroids, maxShift) = recomputeCentroids(
        points = points,
        assignments = assignments,
        previousCentroids = centroids,
        dimension = dimension,
        k = config.k
      )

      centroids = nextCentroids
      iterations += 1
      converged = maxShift <= config.convergenceThreshold

    val inertia = computeInertia(points, assignments, centroids, dimension)
    KMeansResult(
      centroids = centroids,
      assignments = assignments,
      inertia = inertia,
      iterationsRun = iterations,
      converged = converged
    )

  def assign(point: Vector[Double], centroids: Vector[Vector[Double]]): Int =
    require(centroids.nonEmpty, "centroids must be non-empty")
    val dimension = centroids.head.length
    require(point.length == dimension, s"point dimension ${point.length} != centroid dimension $dimension")
    centroids.foreach { centroid =>
      require(centroid.length == dimension, "all centroids must share the same dimension")
    }
    assignInternal(point, centroids, dimension)

  private def validatePoints(points: Vector[Vector[Double]]): Int =
    val dimension = points.head.length
    require(dimension > 0, "point dimension must be > 0")
    points.foreach { point =>
      require(point.length == dimension, s"inconsistent point dimensions: expected $dimension, got ${point.length}")
      point.foreach { value =>
        require(value.isFinite, "all point coordinates must be finite")
      }
    }
    dimension

  private def initializeCentroids(
      points: Vector[Vector[Double]],
      k: Int,
      rng: Random
  ): Vector[Vector[Double]] =
    rng.shuffle(points.indices.toVector).take(k).map(points)

  private def assignInternal(
      point: Vector[Double],
      centroids: Vector[Vector[Double]],
      dimension: Int
  ): Int =
    var bestIdx = 0
    var bestDist = squaredDistance(point, centroids(0), dimension)
    var i = 1
    while i < centroids.length do
      val dist = squaredDistance(point, centroids(i), dimension)
      if dist < bestDist then
        bestDist = dist
        bestIdx = i
      i += 1
    bestIdx

  private def recomputeCentroids(
      points: Vector[Vector[Double]],
      assignments: Vector[Int],
      previousCentroids: Vector[Vector[Double]],
      dimension: Int,
      k: Int
  ): (Vector[Vector[Double]], Double) =
    val sums = Array.fill(k, dimension)(0.0)
    val counts = Array.fill(k)(0)

    var i = 0
    while i < points.length do
      val cluster = assignments(i)
      counts(cluster) += 1
      var j = 0
      while j < dimension do
        sums(cluster)(j) += points(i)(j)
        j += 1
      i += 1

    val next = new Array[Vector[Double]](k)
    var maxShift = 0.0
    var c = 0
    while c < k do
      val updated =
        if counts(c) == 0 then previousCentroids(c)
        else
          val mean = new Array[Double](dimension)
          var j = 0
          while j < dimension do
            mean(j) = sums(c)(j) / counts(c).toDouble
            j += 1
          mean.toVector

      val shift = math.sqrt(squaredDistance(previousCentroids(c), updated, dimension))
      if shift > maxShift then maxShift = shift
      next(c) = updated
      c += 1

    (next.toVector, maxShift)

  private def computeInertia(
      points: Vector[Vector[Double]],
      assignments: Vector[Int],
      centroids: Vector[Vector[Double]],
      dimension: Int
  ): Double =
    var total = 0.0
    var i = 0
    while i < points.length do
      total += squaredDistance(points(i), centroids(assignments(i)), dimension)
      i += 1
    total

  private def squaredDistance(a: Vector[Double], b: Vector[Double], dimension: Int): Double =
    var sum = 0.0
    var i = 0
    while i < dimension do
      val d = a(i) - b(i)
      sum += d * d
      i += 1
    sum
