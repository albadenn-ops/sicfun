package sicfun.holdem.analysis

import munit.FunSuite

class PlayerClusterTest extends FunSuite:
  private def signature(
      fold: Double,
      raise: Double,
      call: Double,
      check: Double,
      entropy: Double,
      potOdds: Double
  ): PlayerSignature =
    PlayerSignature(Vector(fold, raise, call, check, entropy, potOdds))

  test("cluster separates passive and aggressive archetypes") {
    val signatures = Vector(
      signature(0.75, 0.05, 0.10, 0.10, 0.60, 0.20),
      signature(0.70, 0.08, 0.12, 0.10, 0.58, 0.22),
      signature(0.10, 0.55, 0.25, 0.10, 1.30, 0.40),
      signature(0.08, 0.60, 0.22, 0.10, 1.35, 0.42)
    )

    val result = PlayerCluster.cluster(
      signatures,
      PlayerClusterConfig(k = 2, seed = 7L, maxIterations = 50)
    )

    val passiveCluster = result.assignments.take(2).toSet
    val aggressiveCluster = result.assignments.drop(2).toSet
    assertEquals(passiveCluster.size, 1)
    assertEquals(aggressiveCluster.size, 1)
    assertNotEquals(passiveCluster.head, aggressiveCluster.head)
    assert(result.centroidVersion.nonEmpty)
  }

  test("cluster produces deterministic fingerprints for same seed and data") {
    val signatures = Vector(
      signature(0.8, 0.05, 0.1, 0.05, 0.5, 0.1),
      signature(0.78, 0.06, 0.1, 0.06, 0.52, 0.12),
      signature(0.12, 0.5, 0.3, 0.08, 1.4, 0.45),
      signature(0.1, 0.52, 0.3, 0.08, 1.38, 0.44)
    )
    val config = PlayerClusterConfig(k = 2, seed = 11L, maxIterations = 40)

    val first = PlayerCluster.cluster(signatures, config)
    val second = PlayerCluster.cluster(signatures, config)

    assertEquals(first.centroids, second.centroids)
    assertEquals(first.assignments, second.assignments)
    assertEquals(first.centroidVersion, second.centroidVersion)
    assertEquals(first.fingerprints, second.fingerprints)
  }

  test("assign returns cluster nearest to provided centroid set") {
    val signatures = Vector(
      signature(0.8, 0.05, 0.1, 0.05, 0.5, 0.1),
      signature(0.75, 0.08, 0.1, 0.07, 0.55, 0.12),
      signature(0.1, 0.55, 0.25, 0.1, 1.3, 0.4),
      signature(0.12, 0.5, 0.28, 0.1, 1.25, 0.38)
    )
    val result = PlayerCluster.cluster(signatures, PlayerClusterConfig(k = 2, seed = 3L))

    val candidate = signature(0.78, 0.06, 0.1, 0.06, 0.53, 0.11)
    val assigned = PlayerCluster.assign(candidate, result.centroids)
    assertEquals(assigned, result.assignments.head)
  }

  test("fingerprint combines cluster id with centroid version") {
    val centroids = Vector(
      signature(0.8, 0.05, 0.1, 0.05, 0.5, 0.1),
      signature(0.1, 0.55, 0.25, 0.1, 1.3, 0.4)
    )
    val fp = PlayerCluster.fingerprint(
      signature(0.78, 0.06, 0.1, 0.06, 0.53, 0.11),
      centroids,
      centroidVersion = "abc123"
    )
    assertEquals(fp.centroidVersion, "abc123")
    assert(fp.clusterId == 0 || fp.clusterId == 1)
  }

  test("inertia decreases when k increases") {
    val signatures = Vector(
      signature(0.8, 0.05, 0.1, 0.05, 0.5, 0.1),
      signature(0.75, 0.08, 0.1, 0.07, 0.55, 0.12),
      signature(0.1, 0.55, 0.25, 0.1, 1.3, 0.4),
      signature(0.12, 0.5, 0.28, 0.1, 1.25, 0.38)
    )
    val k1 = PlayerCluster.cluster(signatures, PlayerClusterConfig(k = 1, seed = 2L))
    val k2 = PlayerCluster.cluster(signatures, PlayerClusterConfig(k = 2, seed = 2L))
    assert(k2.inertia <= k1.inertia + 1e-12)
  }
