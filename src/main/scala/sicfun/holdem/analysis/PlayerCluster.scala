package sicfun.holdem.analysis

import sicfun.core.{KMeans, KMeansConfig}

import java.nio.charset.StandardCharsets
import java.security.MessageDigest

/** Configuration for K-Means player clustering.
  *
  * @param k                     number of clusters (player archetypes) to discover
  * @param maxIterations         upper bound on K-Means iterations (default 100)
  * @param convergenceThreshold  centroid movement threshold below which clustering stops (default 1e-6)
  * @param seed                  random seed for deterministic centroid initialization (default 42)
  */
final case class PlayerClusterConfig(
    k: Int,
    maxIterations: Int = 100,
    convergenceThreshold: Double = 1e-6,
    seed: Long = 42L
)

/** A player's cluster assignment paired with a content-addressable centroid version hash.
  *
  * The `centroidVersion` is a SHA-256 digest of the centroid vectors, ensuring that
  * fingerprints from different clustering runs can be compared for compatibility.
  *
  * @param clusterId       zero-based index of the assigned cluster
  * @param centroidVersion hex SHA-256 hash uniquely identifying the centroid configuration
  */
final case class Fingerprint(clusterId: Int, centroidVersion: String):
  require(clusterId >= 0, "clusterId must be >= 0")
  require(centroidVersion.trim.nonEmpty, "centroidVersion must be non-empty")

/** Complete result of a player clustering run.
  *
  * Centroids and assignments are ''canonicalized'' — centroids are sorted lexicographically
  * by their value vectors so that the cluster numbering is deterministic regardless of
  * initialization order.
  *
  * @param centroids       canonical centroid signatures, one per cluster
  * @param assignments     cluster index for each input signature (parallel to input order)
  * @param fingerprints    [[Fingerprint]] for each input signature (parallel to input order)
  * @param inertia         sum of squared distances from each point to its centroid
  * @param iterationsRun   number of K-Means iterations actually executed
  * @param converged       `true` if centroids converged within the threshold before hitting max iterations
  * @param centroidVersion SHA-256 hex digest of the canonical centroid vectors
  */
final case class PlayerClusterResult(
    centroids: Vector[PlayerSignature],
    assignments: Vector[Int],
    fingerprints: Vector[Fingerprint],
    inertia: Double,
    iterationsRun: Int,
    converged: Boolean,
    centroidVersion: String
):
  require(centroids.nonEmpty, "centroids must be non-empty")
  require(assignments.length == fingerprints.length, "assignments and fingerprints must align")
  require(centroidVersion.trim.nonEmpty, "centroidVersion must be non-empty")

/** K-Means clustering of [[PlayerSignature]] vectors into behavioral archetypes.
  *
  * Wraps [[sicfun.core.KMeans]] and adds poker-specific post-processing:
  *   - '''Canonicalization:''' centroids are sorted lexicographically so that cluster IDs
  *     are deterministic across runs with the same data.
  *   - '''Fingerprinting:''' each input signature receives a [[Fingerprint]] combining its
  *     cluster assignment with a content-addressable centroid version hash (SHA-256).
  *   - '''Assignment:''' new signatures can be classified against existing centroids without re-clustering.
  */
object PlayerCluster:
  /** Clusters the given signatures into `config.k` archetypes using K-Means.
    *
    * @param signatures input behavioral profiles (must be non-empty, same dimension)
    * @param config     clustering hyperparameters
    * @return a [[PlayerClusterResult]] with canonicalized centroids and fingerprints
    */
  def cluster(signatures: Vector[PlayerSignature], config: PlayerClusterConfig): PlayerClusterResult =
    require(signatures.nonEmpty, "signatures must be non-empty")
    require(config.k >= 1, "k must be >= 1")
    require(config.k <= signatures.length, s"k (${config.k}) cannot exceed signature count (${signatures.length})")

    val points = signatures.map(_.values)
    val dimension = points.head.length
    require(dimension > 0, "signature vectors must be non-empty")
    points.foreach { point =>
      require(point.length == dimension, "all signatures must have the same dimension")
      point.foreach { value =>
        require(value.isFinite, "signature values must be finite")
      }
    }

    val kmeansResult = KMeans.fit(
      points,
      KMeansConfig(
        k = config.k,
        maxIterations = config.maxIterations,
        convergenceThreshold = config.convergenceThreshold,
        seed = config.seed
      )
    )

    val rawCentroids = kmeansResult.centroids.map(PlayerSignature(_))
    val (canonicalCentroids, canonicalAssignments) = canonicalize(rawCentroids, kmeansResult.assignments)
    val centroidVersion = hashCentroids(canonicalCentroids)
    val fingerprints = canonicalAssignments.map(clusterId => Fingerprint(clusterId, centroidVersion))

    PlayerClusterResult(
      centroids = canonicalCentroids,
      assignments = canonicalAssignments,
      fingerprints = fingerprints,
      inertia = kmeansResult.inertia,
      iterationsRun = kmeansResult.iterationsRun,
      converged = kmeansResult.converged,
      centroidVersion = centroidVersion
    )

  /** Assigns a single signature to the nearest centroid by Euclidean distance.
    *
    * @param signature  the player's behavioral profile
    * @param centroids  the reference centroid set (e.g. from a previous clustering result)
    * @return zero-based index of the nearest centroid
    */
  def assign(signature: PlayerSignature, centroids: Vector[PlayerSignature]): Int =
    require(centroids.nonEmpty, "centroids must be non-empty")
    val dimension = centroids.head.values.length
    require(signature.values.length == dimension, "signature dimension must match centroids")
    centroids.foreach { centroid =>
      require(centroid.values.length == dimension, "all centroids must have same dimension")
    }
    KMeans.assign(signature.values, centroids.map(_.values))

  /** Assigns a single signature and wraps the result in a [[Fingerprint]].
    *
    * @param signature       the player's behavioral profile
    * @param centroids       the reference centroid set
    * @param centroidVersion version hash of the centroid set (from [[PlayerClusterResult.centroidVersion]])
    * @return a [[Fingerprint]] combining the assigned cluster ID with the centroid version
    */
  def fingerprint(
      signature: PlayerSignature,
      centroids: Vector[PlayerSignature],
      centroidVersion: String
  ): Fingerprint =
    require(centroidVersion.trim.nonEmpty, "centroidVersion must be non-empty")
    val clusterId = assign(signature, centroids)
    Fingerprint(clusterId, centroidVersion.trim)

  /** Sorts centroids lexicographically and remaps assignments for deterministic ordering. */
  private def canonicalize(
      centroids: Vector[PlayerSignature],
      assignments: Vector[Int]
  ): (Vector[PlayerSignature], Vector[Int]) =
    val order = centroids.indices.toVector.sortBy(i => (centroids(i).values, i))(using lexicographicVectorOrdering)
    val remap = order.zipWithIndex.toMap
    val canonicalCentroids = order.map(centroids)
    val canonicalAssignments = assignments.map(oldId => remap(oldId))
    (canonicalCentroids, canonicalAssignments)

  private val lexicographicVectorOrdering: Ordering[(Vector[Double], Int)] =
    new Ordering[(Vector[Double], Int)]:
      override def compare(a: (Vector[Double], Int), b: (Vector[Double], Int)): Int =
        val cmp = compareVectors(a._1, b._1)
        if cmp != 0 then cmp else a._2.compare(b._2)

  private def compareVectors(a: Vector[Double], b: Vector[Double]): Int =
    import scala.util.boundary, boundary.break
    val len = math.min(a.length, b.length)
    boundary:
      var i = 0
      while i < len do
        val cmp = java.lang.Double.compare(a(i), b(i))
        if cmp != 0 then break(cmp)
        i += 1
      a.length.compare(b.length)

  /** Computes a SHA-256 hex digest of all centroid values for content-addressable versioning. */
  private def hashCentroids(centroids: Vector[PlayerSignature]): String =
    val payload = centroids.flatMap(_.values).map(v => java.lang.Double.doubleToLongBits(v).toHexString).mkString("|")
    val digest = MessageDigest.getInstance("SHA-256")
    val bytes = digest.digest(payload.getBytes(StandardCharsets.UTF_8))
    bytes.map("%02x".format(_)).mkString
