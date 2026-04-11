package sicfun.core

/** Tests for [[KMeans]] clustering implementation.
  *
  * Coverage includes:
  *  - '''Input validation''': empty points, k > n, k < 1, inconsistent dimensions, NaN coords.
  *  - '''Trivial cases''': k=1 assigns everything to one cluster; k=n gives zero inertia.
  *  - '''Separation''': well-separated point groups are assigned to distinct clusters.
  *  - '''Determinism''': same seed + same input = identical output.
  *  - '''Assignment API''': public `assign` method picks the nearest centroid.
  *  - '''Convergence''': with maxIterations=1 and tight threshold, converged=false.
  *  - '''Config validation''': rejects non-positive maxIterations and negative thresholds.
  */
class KMeansTest extends munit.FunSuite:
  test("fit rejects empty input and invalid k") {
    intercept[IllegalArgumentException] {
      KMeans.fit(Vector.empty, KMeansConfig(k = 1))
    }
    intercept[IllegalArgumentException] {
      KMeans.fit(Vector(Vector(0.0, 0.0)), KMeansConfig(k = 0))
    }
    intercept[IllegalArgumentException] {
      KMeans.fit(Vector(Vector(0.0, 0.0)), KMeansConfig(k = 2))
    }
  }

  test("fit rejects inconsistent dimensions") {
    intercept[IllegalArgumentException] {
      KMeans.fit(
        Vector(Vector(0.0, 1.0), Vector(2.0)),
        KMeansConfig(k = 1)
      )
    }
  }

  test("k=1 assigns all points to same cluster") {
    val points = Vector(
      Vector(0.0, 0.0),
      Vector(1.0, 0.0),
      Vector(2.0, 0.0)
    )

    val result = KMeans.fit(points, KMeansConfig(k = 1, seed = 7L))
    assertEquals(result.centroids.length, 1)
    assertEquals(result.assignments, Vector(0, 0, 0))
    assert(result.inertia >= 0.0)
  }

  test("k=n has zero inertia for distinct points") {
    val points = Vector(
      Vector(0.0, 0.0),
      Vector(1.0, 2.0),
      Vector(5.0, 5.0)
    )
    val result = KMeans.fit(points, KMeansConfig(k = points.length, seed = 11L))
    assert(math.abs(result.inertia) < 1e-12)
  }

  test("separated groups are assigned to different clusters") {
    val points = Vector(
      Vector(0.0, 0.0),
      Vector(0.2, 0.1),
      Vector(9.8, 10.1),
      Vector(10.0, 10.0)
    )
    val result = KMeans.fit(points, KMeansConfig(k = 2, seed = 1L, maxIterations = 50))

    val left = result.assignments.take(2).toSet
    val right = result.assignments.drop(2).toSet
    assertEquals(left.size, 1)
    assertEquals(right.size, 1)
    assertNotEquals(left.head, right.head)
  }

  test("fit is deterministic for same seed and inputs") {
    val points = Vector(
      Vector(0.0, 0.0),
      Vector(0.1, 0.2),
      Vector(10.0, 10.0),
      Vector(9.9, 10.1)
    )
    val config = KMeansConfig(k = 2, seed = 42L, maxIterations = 30)

    val first = KMeans.fit(points, config)
    val second = KMeans.fit(points, config)

    assertEquals(first.centroids, second.centroids)
    assertEquals(first.assignments, second.assignments)
    assertEquals(first.inertia, second.inertia)
    assertEquals(first.iterationsRun, second.iterationsRun)
    assertEquals(first.converged, second.converged)
  }

  test("assign chooses nearest centroid") {
    val centroids = Vector(
      Vector(0.0, 0.0),
      Vector(10.0, 10.0)
    )
    assertEquals(KMeans.assign(Vector(0.1, 0.2), centroids), 0)
    assertEquals(KMeans.assign(Vector(9.7, 9.9), centroids), 1)
  }

  test("converged is false when maxIterations is too small") {
    val points = Vector(
      Vector(0.0, 0.0),
      Vector(2.0, 0.0)
    )
    val result = KMeans.fit(
      points,
      KMeansConfig(k = 1, maxIterations = 1, convergenceThreshold = 1e-12, seed = 5L)
    )
    assertEquals(result.iterationsRun, 1)
    assert(!result.converged)
  }

  test("config rejects invalid thresholds and iteration settings") {
    intercept[IllegalArgumentException] {
      KMeansConfig(k = 1, maxIterations = 0)
    }
    intercept[IllegalArgumentException] {
      KMeansConfig(k = 1, convergenceThreshold = -1e-6)
    }
  }

  test("fit rejects non-finite point coordinates") {
    intercept[IllegalArgumentException] {
      KMeans.fit(
        Vector(Vector(0.0, Double.NaN), Vector(1.0, 2.0)),
        KMeansConfig(k = 1)
      )
    }
  }

  test("assign validates centroid and point dimensions") {
    intercept[IllegalArgumentException] {
      KMeans.assign(Vector(1.0), Vector.empty)
    }
    intercept[IllegalArgumentException] {
      KMeans.assign(Vector(1.0), Vector(Vector(1.0, 2.0)))
    }
    intercept[IllegalArgumentException] {
      KMeans.assign(Vector(1.0, 2.0), Vector(Vector(0.0, 0.0), Vector(1.0)))
    }
  }
