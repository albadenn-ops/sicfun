package sicfun.core

import scala.util.Random

/** Hyperparameters for k-means clustering.
  *
  * @param k                     number of clusters to produce (must be >= 1)
  * @param maxIterations         upper bound on the number of assign-recompute cycles (default 100)
  * @param convergenceThreshold  maximum centroid shift (Euclidean distance) to declare convergence (default 1e-6)
  * @param seed                  RNG seed for deterministic centroid initialization (default 42)
  */
final case class KMeansConfig(
    k: Int,
    maxIterations: Int = 100,
    convergenceThreshold: Double = 1e-6,
    seed: Long = 42L
):
  require(k >= 1, "k must be >= 1")
  require(maxIterations >= 1, "maxIterations must be >= 1")
  require(convergenceThreshold >= 0.0, "convergenceThreshold must be >= 0")

/** Result of a k-means fit run.
  *
  * @param centroids     the final cluster centers, each a d-dimensional vector
  * @param assignments   per-point cluster index (0-based), aligned with the input points
  * @param inertia       total within-cluster sum of squared distances (lower = tighter clusters)
  * @param iterationsRun number of assign-recompute cycles actually executed
  * @param converged     true if the algorithm stopped because all centroid shifts were
  *                      below the convergence threshold (rather than hitting maxIterations)
  */
final case class KMeansResult(
    centroids: Vector[Vector[Double]],
    assignments: Vector[Int],
    inertia: Double,
    iterationsRun: Int,
    converged: Boolean
)

/** Lightweight k-means (Lloyd's algorithm) implementation for dense Double vectors.
  *
  * Used in the sicfun poker system for player clustering (grouping opponents by
  * behavioral feature vectors such as aggression frequency, VPIP, fold-to-3bet, etc.).
  *
  * Design decisions:
  *  - '''Random initialization''': centroids are initialized by randomly shuffling point
  *    indices and taking the first k. This is simpler than k-means++ but sufficient for
  *    the small cluster counts (k <= 10) typical in poker player profiling.
  *  - '''While-loop hot paths''': inner loops use `while` instead of `foreach`/`map` to
  *    avoid boxing and closure allocation, which matters when clustering large datasets.
  *  - '''Empty cluster handling''': if a cluster loses all members during reassignment,
  *    its centroid is preserved from the previous iteration (no random re-seeding).
  */
object KMeans:
  /** Runs k-means clustering on the given points.
    *
    * The algorithm iterates between two steps until convergence or maxIterations:
    *  1. '''Assignment''': each point is assigned to the nearest centroid (by Euclidean distance).
    *  2. '''Update''': each centroid is recomputed as the mean of its assigned points.
    *
    * Convergence is declared when the maximum centroid shift across all clusters is
    * below `config.convergenceThreshold`.
    *
    * @param points a non-empty vector of d-dimensional feature vectors (all same dimension)
    * @param config clustering hyperparameters
    * @return a [[KMeansResult]] containing centroids, assignments, inertia, and convergence info
    */
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

  /** Assigns a single point to the nearest centroid by squared Euclidean distance.
    *
    * This is the public API for inference after training: given a new observation,
    * determine which cluster it belongs to.
    *
    * @param point     the d-dimensional feature vector to classify
    * @param centroids the cluster centers (from a [[KMeansResult]])
    * @return the 0-based index of the nearest centroid
    */
  def assign(point: Vector[Double], centroids: Vector[Vector[Double]]): Int =
    require(centroids.nonEmpty, "centroids must be non-empty")
    val dimension = centroids.head.length
    require(point.length == dimension, s"point dimension ${point.length} != centroid dimension $dimension")
    centroids.foreach { centroid =>
      require(centroid.length == dimension, "all centroids must share the same dimension")
    }
    assignInternal(point, centroids, dimension)

  /** Validates that all points share the same dimension and contain only finite values.
    * @return the common dimension of all points
    */
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

  /** Selects k initial centroids by random shuffle of point indices (simple random init). */
  private def initializeCentroids(
      points: Vector[Vector[Double]],
      k: Int,
      rng: Random
  ): Vector[Vector[Double]] =
    rng.shuffle(points.indices.toVector).take(k).map(points)

  /** Finds the index of the nearest centroid by squared Euclidean distance (internal, no validation). */
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

  /** Recomputes cluster centroids as the mean of assigned points.
    *
    * Returns the new centroids and the maximum Euclidean shift of any centroid,
    * which is used for convergence checking.
    *
    * If a cluster has zero assigned points, its centroid is carried forward unchanged
    * from the previous iteration.
    */
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

  /** Computes total within-cluster sum of squared distances (inertia / WCSS).
    * Lower inertia indicates tighter, more compact clusters.
    */
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

  /** Computes the squared Euclidean distance between two d-dimensional vectors.
    * Uses squared distance (no sqrt) for efficiency since only relative ordering matters.
    */
  private def squaredDistance(a: Vector[Double], b: Vector[Double], dimension: Int): Double =
    var sum = 0.0
    var i = 0
    while i < dimension do
      val d = a(i) - b(i)
      sum += d * d
      i += 1
    sum
