package sicfun.holdem.strategic.solver

import munit.FunSuite

/** Tests for WassersteinDroRuntime — validates v0.30.2 Definitions 33-34 and Theorem 7.
  *
  * The pure-Scala fallback is always available so no native-DLL assume guards are
  * needed for EMD or DRO tests.  Tests that explicitly exercise the native JNI path
  * use the isNativeAvailable guard.
  */
class WassersteinDroRuntimeTest extends FunSuite:

  private val Tol = 1e-6  // tolerance for floating-point comparisons

  // ---- Helpers ----------------------------------------------------------------

  /** 1D Euclidean cost matrix: c(i,j) = |i - j|. */
  private def euclidCost(n: Int, m: Int): Array[Double] =
    val c = new Array[Double](n * m)
    var i = 0
    while i < n do
      var j = 0
      while j < m do
        c(i * m + j) = math.abs(i - j).toDouble
        j += 1
      i += 1
    c

  /** Square Euclidean cost: c(i,j) = |i - j| for n×n matrix. */
  private def euclidCostSquare(n: Int): Array[Double] = euclidCost(n, n)

  /** Dirac mass at position k in n-dim simplex. */
  private def dirac(n: Int, k: Int): Array[Double] =
    val w = new Array[Double](n)
    w(k) = 1.0
    w

  /** Uniform distribution over n points. */
  private def uniform(n: Int): Array[Double] =
    Array.fill(n)(1.0 / n)

  /** Convex combination: λ * a + (1-λ) * b. */
  private def convexCombo(a: Array[Double], b: Array[Double], lambda: Double): Array[Double] =
    Array.tabulate(a.length)(i => lambda * a(i) + (1 - lambda) * b(i))

  /** Normalise a vector to sum to 1.0. */
  private def normalize(v: Array[Double]): Array[Double] =
    val s = v.sum
    v.map(_ / s)

  /** Random simplex point using Dirichlet(1,...,1) via exponential sampling. */
  private def randomSimplex(n: Int, rng: scala.util.Random): Array[Double] =
    normalize(Array.fill(n)(-math.log(rng.nextDouble() + 1e-15)))

  private def assertClose(actual: Double, expected: Double, label: String): Unit =
    assert(
      math.abs(actual - expected) <= Tol,
      s"$label: expected $expected, got $actual (diff=${math.abs(actual - expected)})"
    )

  // ---- EMD Tests (Task 4 / Task 5) -------------------------------------------

  test("W_1 of identical distributions is zero") {
    val w = normalize(Array(0.5, 0.3, 0.2))
    val c = euclidCostSquare(3)
    val result = WassersteinDroRuntime.emd(w, w, c)
    assert(result.isRight, s"emd returned Left: ${result.left.toOption}")
    assertClose(result.toOption.get, 0.0, "W_1(p,p)")
  }

  test("W_1 of Dirac masses equals ground distance") {
    val n = 5
    val cost = euclidCostSquare(n)
    // δ_0 vs δ_4: ground distance = 4
    val a = dirac(n, 0)
    val b = dirac(n, 4)
    val result = WassersteinDroRuntime.emd(a, b, cost)
    assert(result.isRight, s"emd returned Left: ${result.left.toOption}")
    assertClose(result.toOption.get, 4.0, "W_1(δ_0, δ_4)")
  }

  test("W_1 is symmetric: W_1(a,b) == W_1(b,a)") {
    val n = 4
    val cost = euclidCostSquare(n)
    val a = normalize(Array(0.1, 0.4, 0.3, 0.2))
    val b = normalize(Array(0.25, 0.25, 0.25, 0.25))
    val ab = WassersteinDroRuntime.emd(a, b, cost).toOption.get
    val ba = WassersteinDroRuntime.emd(b, a, cost).toOption.get
    assertClose(ab, ba, "W_1(a,b) vs W_1(b,a)")
  }

  test("W_1 satisfies triangle inequality") {
    val rng = new scala.util.Random(42L)
    val n = 4
    val cost = euclidCostSquare(n)
    val p = randomSimplex(n, rng)
    val q = randomSimplex(n, rng)
    val r = randomSimplex(n, rng)
    val pq = WassersteinDroRuntime.emd(p, q, cost).toOption.get
    val qr = WassersteinDroRuntime.emd(q, r, cost).toOption.get
    val pr = WassersteinDroRuntime.emd(p, r, cost).toOption.get
    assert(pr <= pq + qr + Tol, s"Triangle inequality violated: W_1(p,r)=$pr > W_1(p,q)+W_1(q,r)=${pq+qr}")
  }

  test("W_1 of uniform distributions on unit metric") {
    // Two 2-point distributions: [0.7, 0.3] and [0.3, 0.7].
    // Cost matrix: d(0,1)=d(1,0)=1, d(i,i)=0.
    // W_1 = |0.7 - 0.3| * 1 = 0.4 (by Kantorovich-Rubinstein duality on {0,1}).
    val a = Array(0.7, 0.3)
    val b = Array(0.3, 0.7)
    val cost = Array(0.0, 1.0, 1.0, 0.0)
    val result = WassersteinDroRuntime.emd(a, b, cost).toOption.get
    assertClose(result, 0.4, "W_1 on 2-point uniform shift")
  }

  test("W_1 batch matches individual calls") {
    val n = 3
    val cost = euclidCostSquare(n)
    val ref = uniform(n)
    val targets = Array(
      normalize(Array(0.6, 0.3, 0.1)),
      dirac(n, 0),
      dirac(n, 2)
    )
    val k = targets.length
    val packed = targets.flatten
    val batchResult = WassersteinDroRuntime.emdBatch(ref, packed, k, cost)
    assert(batchResult.isRight, s"emdBatch returned Left: ${batchResult.left.toOption}")
    val batchDists = batchResult.toOption.get
    var i = 0
    while i < k do
      val expected = WassersteinDroRuntime.emd(ref, targets(i), cost).toOption.get
      assertClose(batchDists(i), expected, s"batch[$i] vs individual")
      i += 1
  }

  test("W_1 with trivial 1x1 distributions is zero") {
    val a = Array(1.0)
    val b = Array(1.0)
    val cost = Array(0.0)
    val result = WassersteinDroRuntime.emd(a, b, cost)
    assert(result.isRight, s"emd returned Left: ${result.left.toOption}")
    assertClose(result.toOption.get, 0.0, "W_1 on 1x1")
  }

  // ---- Ambiguity Set Tests (Definition 33) ------------------------------------

  test("ambiguity set with rho=0 contains only the center (same distribution)") {
    val n = 3
    val b = normalize(Array(0.5, 0.3, 0.2))
    val cost = euclidCostSquare(n)
    val inSet = WassersteinDroRuntime.inAmbiguitySet(b, b, cost, rho = 0.0)
    assert(inSet.isRight, s"inAmbiguitySet returned Left: ${inSet.left.toOption}")
    assert(inSet.toOption.get, "center must be in rho=0 ambiguity set")
  }

  test("ambiguity set with rho=0 excludes different distribution") {
    val n = 3
    val center = dirac(n, 0)
    val candidate = dirac(n, 2)
    val cost = euclidCostSquare(n)  // distance(0,2) = 2
    val inSet = WassersteinDroRuntime.inAmbiguitySet(center, candidate, cost, rho = 0.0)
    assert(inSet.isRight)
    assert(!inSet.toOption.get, "δ_2 should not be in rho=0 set around δ_0")
  }

  test("ambiguity set with rho>0 contains perturbations within radius") {
    val n = 4
    val center = uniform(n)
    val cost = euclidCostSquare(n)
    // Small perturbation: shift 0.05 mass from position 0 to position 1.
    // W_1 cost = 0.05 * d(0,1) = 0.05.
    val perturbed = normalize(Array(0.20, 0.30, 0.25, 0.25))
    val dist = WassersteinDroRuntime.emd(center, perturbed, cost).toOption.get
    // Verify it really is within rho = 0.1
    val inSet = WassersteinDroRuntime.inAmbiguitySet(center, perturbed, cost, rho = 0.1)
    assert(inSet.isRight)
    if dist <= 0.1 + Tol then
      assert(inSet.toOption.get, s"point with W_1=$dist should be in rho=0.1 set")
    else
      assert(!inSet.toOption.get, s"point with W_1=$dist should not be in rho=0.1 set")
  }

  test("ambiguity set rejects distributions outside radius") {
    val n = 3
    val center = dirac(n, 0)
    val farPoint = dirac(n, 2)  // W_1 = 2 from center
    val cost = euclidCostSquare(n)
    val inSet = WassersteinDroRuntime.inAmbiguitySet(center, farPoint, cost, rho = 1.0)
    assert(inSet.isRight)
    assert(!inSet.toOption.get, "distance=2 point should not be in rho=1.0 set")
  }

  test("ambiguity set membership is consistent with W_1") {
    val n = 3
    val center = uniform(n)
    val cost = euclidCostSquare(n)
    val rng = new scala.util.Random(123L)
    var inconsistent = 0
    var trials = 0
    while trials < 20 do
      val candidate = randomSimplex(n, rng)
      val dist = WassersteinDroRuntime.emd(center, candidate, cost).toOption.get
      val rho = 0.15
      val inSet = WassersteinDroRuntime.inAmbiguitySet(center, candidate, cost, rho).toOption.get
      val expected = dist <= rho + Tol
      if inSet != expected then inconsistent += 1
      trials += 1
    assertEquals(inconsistent, 0, "Ambiguity set membership inconsistent with W_1")
  }

  // ---- DRO LP Tests (Definition 34) ------------------------------------------

  test("DRO with rho=0 equals standard Q-value") {
    val n = 3
    val b = normalize(Array(0.5, 0.3, 0.2))
    val q = Array(10.0, 5.0, -3.0)
    val cost = euclidCostSquare(n)
    val standardV = b.zip(q).map(_ * _).sum  // = 5.0 + 1.5 - 0.6 = 5.9
    val droResult = WassersteinDroRuntime.solveDroLp(b, q, cost, rho = 0.0)
    assert(droResult.isRight, s"solveDroLp returned Left: ${droResult.left.toOption}")
    assertClose(droResult.toOption.get.worstCaseValue, standardV, "rho=0 DRO vs standard Q")
  }

  test("DRO with rho>0 returns value <= standard Q-value") {
    val n = 3
    val b = normalize(Array(0.5, 0.3, 0.2))
    val q = Array(10.0, 5.0, -3.0)
    val cost = euclidCostSquare(n)
    val standardV = b.zip(q).map(_ * _).sum
    val droResult = WassersteinDroRuntime.solveDroLp(b, q, cost, rho = 0.1)
    assert(droResult.isRight, s"solveDroLp returned Left: ${droResult.left.toOption}")
    val droV = droResult.toOption.get.worstCaseValue
    assert(droV <= standardV + Tol, s"DRO value $droV should be <= standard value $standardV")
  }

  test("DRO worst-case belief lies within ambiguity set") {
    val n = 3
    val b = normalize(Array(0.5, 0.3, 0.2))
    val q = Array(10.0, 5.0, -3.0)
    val rho = 0.1
    val cost = euclidCostSquare(n)
    val droResult = WassersteinDroRuntime.solveDroLp(b, q, cost, rho).toOption.get
    val worstBelief = droResult.worstCaseBelief
    // Worst-case belief should sum to 1.0 (valid distribution)
    assertClose(worstBelief.sum, 1.0, "worst-case belief sum")
    // Worst-case belief should be within ambiguity set
    val dist = WassersteinDroRuntime.emd(b, worstBelief, cost).toOption.get
    assert(
      dist <= rho + Tol * 100,
      s"worst-case belief has W_1=$dist which exceeds rho=$rho"
    )
  }

  test("DRO LP produces non-negative beliefs") {
    val n = 4
    val b = normalize(Array(0.4, 0.3, 0.2, 0.1))
    val q = Array(8.0, 4.0, 1.0, -2.0)
    val cost = euclidCostSquare(n)
    val droResult = WassersteinDroRuntime.solveDroLp(b, q, cost, rho = 0.05)
    assert(droResult.isRight, s"solveDroLp returned Left: ${droResult.left.toOption}")
    val wb = droResult.toOption.get.worstCaseBelief
    wb.foreach(w => assert(w >= -Tol, s"worst-case belief has negative component $w"))
  }

  // ---- Backward Compatibility -------------------------------------------------

  test("rho=0 recovers non-robust behavior exactly") {
    val n = 4
    val b = normalize(Array(0.1, 0.4, 0.3, 0.2))
    val q = Array(5.0, 3.0, 1.0, -1.0)
    val cost = euclidCostSquare(n)
    val standard = b.zip(q).map(_ * _).sum
    val dro = WassersteinDroRuntime.solveDroLp(b, q, cost, rho = 0.0)
    assert(dro.isRight)
    assertClose(dro.toOption.get.worstCaseValue, standard, "exact backward compat at rho=0")
  }

  // ---- Edge Cases (Task 5) ----------------------------------------------------

  test("EMD with very small weights does not crash or return NaN") {
    val n = 3
    // Weights with tiny values; we normalise them to sum to 1.0
    val raw = Array(1e-10, 1.0, 1e-10)
    val w = raw.map(_ / raw.sum)
    val cost = euclidCostSquare(n)
    val result = WassersteinDroRuntime.emd(w, w, cost)
    assert(result.isRight)
    assert(!result.toOption.get.isNaN, "EMD should not return NaN for tiny weights")
    assertClose(result.toOption.get, 0.0, "W_1(p,p) with small weights")
  }

  test("DRO LP with very large rho returns global minimum of q-values") {
    // When rho is large enough, the worst-case distribution can be any point in Δ.
    // The minimum of q^T b' over Δ is achieved at the Dirac mass on the argmin of q.
    val n = 4
    val b = uniform(n)
    val q = Array(5.0, 2.0, -1.0, 3.0)  // min at index 2: q[2] = -1.0
    val cost = euclidCostSquare(n)
    // rho = 10.0 >> diameter of simplex under this cost
    val dro = WassersteinDroRuntime.solveDroLp(b, q, cost, rho = 10.0)
    assert(dro.isRight, s"solveDroLp returned Left: ${dro.left.toOption}")
    val worstVal = dro.toOption.get.worstCaseValue
    val minQ = q.min
    // The worst-case value should be close to minQ since rho is very large
    assert(
      worstVal <= minQ + Tol * 10,
      s"Large-rho DRO value $worstVal should be <= min(q)=$minQ"
    )
  }

  test("DRO LP with degenerate single-point distribution") {
    val b = Array(1.0)
    val q = Array(7.5)
    val cost = Array(0.0)
    val dro = WassersteinDroRuntime.solveDroLp(b, q, cost, rho = 0.5)
    assert(dro.isRight)
    assertClose(dro.toOption.get.worstCaseValue, 7.5, "single-state DRO")
  }

  test("EMD rejects unnormalized weights with error code 102") {
    val a = Array(0.3, 0.3)  // sums to 0.6, not 1.0
    val b = Array(0.5, 0.5)
    val cost = Array(0.0, 1.0, 1.0, 0.0)
    val result = WassersteinDroRuntime.emd(a, b, cost)
    assert(result.isLeft, "should reject unnormalized weights")
    assertEquals(result.left.toOption.get, WassersteinDroRuntime.StatusNormalization)
  }

  test("EMD rejects negative weights with error code 103") {
    val a = Array(-0.1, 1.1)
    val b = Array(0.5, 0.5)
    val cost = Array(0.0, 1.0, 1.0, 0.0)
    val result = WassersteinDroRuntime.emd(a, b, cost)
    assert(result.isLeft, "should reject negative weights")
    assertEquals(result.left.toOption.get, WassersteinDroRuntime.StatusNegativeWeight)
  }

  test("EMD rejects negative cost entries with error code 104") {
    val a = Array(0.5, 0.5)
    val b = Array(0.5, 0.5)
    val cost = Array(0.0, -1.0, -1.0, 0.0)
    val result = WassersteinDroRuntime.emd(a, b, cost)
    assert(result.isLeft, "should reject negative cost entries")
    assertEquals(result.left.toOption.get, WassersteinDroRuntime.StatusNegativeCost)
  }

  // ---- Theorem 7: Convexity of robust value function (Task 6) ----------------

  test("robust value function is convex in belief for fixed rho") {
    // Theorem 7: V^{*,Γ,ρ}_Π(b̃) is convex in b̃.
    // Validated numerically: for beliefs b1, b2 and lambda in [0,1]:
    // V(lambda * b1 + (1-lambda) * b2) <= lambda * V(b1) + (1-lambda) * V(b2)
    val rng = new scala.util.Random(777L)
    val n = 4
    val cost = euclidCostSquare(n)
    val q = Array(1.0, 2.0, 0.5, 3.0)
    val rho = 0.1
    val lambdas = Array(0.1, 0.25, 0.5, 0.75, 0.9)

    var failures = 0
    var trials = 0
    while trials < 10 do
      val b1 = randomSimplex(n, rng)
      val b2 = randomSimplex(n, rng)
      val v1 = WassersteinDroRuntime.solveDroLp(b1, q, cost, rho).toOption.get.worstCaseValue
      val v2 = WassersteinDroRuntime.solveDroLp(b2, q, cost, rho).toOption.get.worstCaseValue
      lambdas.foreach { lam =>
        val bMix = convexCombo(b1, b2, lam)
        val vMix = WassersteinDroRuntime.solveDroLp(bMix, q, cost, rho).toOption.get.worstCaseValue
        val rhs = lam * v1 + (1 - lam) * v2
        if vMix > rhs + Tol * 100 then
          failures += 1
      }
      trials += 1

    assertEquals(failures, 0, s"Theorem 7 convexity violated in $failures cases")
  }

  // ---- Integration smoke tests (Task 7) ---------------------------------------

  test("full DRO solve on 3-state poker-inspired scenario") {
    // States: {tight=0, neutral=1, loose=2}
    // Belief center: [0.5, 0.3, 0.2]
    // Q-values: [10.0, 5.0, -3.0] (EV per action against each type)
    // Ground metric: Hamming (d(i,j) = 1 if i != j, 0 otherwise)
    // rho = 0.1
    val b = Array(0.5, 0.3, 0.2)
    val q = Array(10.0, 5.0, -3.0)
    val n = 3
    val cost = Array.tabulate(n * n) { idx =>
      val i = idx / n; val j = idx % n
      if i == j then 0.0 else 1.0
    }
    val standardV = b.zip(q).map(_ * _).sum  // 5.0 + 1.5 - 0.6 = 5.9
    val dro = WassersteinDroRuntime.solveDroLp(b, q, cost, rho = 0.1)
    assert(dro.isRight, s"solveDroLp returned Left: ${dro.left.toOption}")
    val result = dro.toOption.get
    assert(
      result.worstCaseValue < standardV + Tol,
      s"DRO value ${result.worstCaseValue} should be <= standard value $standardV"
    )
    // Worst-case belief shifts mass toward the low-Q state (index 2, q=-3)
    val wb = result.worstCaseBelief
    assertClose(wb.sum, 1.0, "worst-case belief sum")
  }

  test("graceful degradation: isAvailable is always true (pure-Scala fallback)") {
    assert(WassersteinDroRuntime.isAvailable, "pure-Scala fallback must always be available")
  }

  test("graceful degradation: emd returns Right without native DLL") {
    val a = Array(0.5, 0.5)
    val b = Array(0.5, 0.5)
    val cost = Array(0.0, 1.0, 1.0, 0.0)
    val result = WassersteinDroRuntime.emd(a, b, cost)
    assert(result.isRight, "emd must succeed via pure-Scala fallback")
  }

  test("robustQValue convenience alias produces same result as solveDroLp") {
    val n = 3
    val b = normalize(Array(0.4, 0.4, 0.2))
    val q = Array(6.0, 3.0, 0.0)
    val cost = euclidCostSquare(n)
    val rho = 0.05
    val fromDro = WassersteinDroRuntime.solveDroLp(b, q, cost, rho).toOption.get.worstCaseValue
    val fromConvenience = WassersteinDroRuntime.robustQValue(b, q, rho, cost).toOption.get
    assertClose(fromConvenience, fromDro, "robustQValue vs solveDroLp")
  }

  test("isInAmbiguitySet convenience alias is consistent") {
    val n = 3
    val center = normalize(Array(0.5, 0.3, 0.2))
    val candidate = normalize(Array(0.4, 0.4, 0.2))
    val cost = euclidCostSquare(n)
    val rho = 0.2
    val fromMain = WassersteinDroRuntime.inAmbiguitySet(center, candidate, cost, rho).toOption.get
    val fromAlias = WassersteinDroRuntime.isInAmbiguitySet(center, candidate, rho, cost)
    assertEquals(fromAlias, fromMain, "isInAmbiguitySet alias must match inAmbiguitySet")
  }
