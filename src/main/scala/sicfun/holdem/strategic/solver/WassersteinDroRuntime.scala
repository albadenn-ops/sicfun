package sicfun.holdem.strategic.solver

import java.util.concurrent.atomic.AtomicReference

/** Runtime for Wasserstein-1 distance computation and distributionally robust
  * optimization (DRO) LP formulation.
  *
  * Implements SICFUN v0.30.2 §6A:
  *  - Definition 33: Ambiguity set B_ρ(b̃_t) = { b' ∈ Δ(X̃) : W_1(b', b̃_t) ≤ ρ }
  *  - Definition 34: Robust Q-function Q^{*,Γ,ρ}_Π(b̃, u)
  *
  * EMD is computed via a pure-Scala SPFA-based min-cost flow (O(n² * flow) for
  * small n). The DRO inner minimization is solved as a transportation LP using a
  * pure-Scala Big-M simplex; no external LP library is required.
  *
  * The native JNI path (sicfun_wasserstein_native.dll via HoldemWassersteinBindings)
  * is attempted first; if the DLL is absent the pure-Scala fallback is used
  * transparently. Tests can check [[isNativeAvailable]] to know which path was taken.
  *
  * ==Thread Safety==
  * Library loading uses AtomicReference CAS. All EMD and LP computations are
  * stateless and safe to call concurrently.
  */
object WassersteinDroRuntime:

  // ---- Status codes (consistent with C++ JNI binding) -------------------------

  val StatusSuccess: Int          = 0
  val StatusNullArg: Int          = 100
  val StatusDimension: Int        = 101
  val StatusNormalization: Int    = 102
  val StatusNegativeWeight: Int   = 103
  val StatusNegativeCost: Int     = 104

  private val NativeLib          = "sicfun_wasserstein_native"
  private val NativeLibPathProp  = "sicfun.wasserstein.native.path"
  private val NativeLibPathEnv   = "sicfun_WASSERSTEIN_NATIVE_PATH"
  private val NativeLibNameProp  = "sicfun.wasserstein.native.lib"
  private val NativeLibNameEnv   = "sicfun_WASSERSTEIN_NATIVE_LIB"

  private val nativeLoadResultRef = new AtomicReference[Either[String, String]](null)

  private[solver] def resetNativeLoadCacheForTests(): Unit =
    nativeLoadResultRef.set(null)

  private def nativeLoadResult(): Either[String, String] =
    val cached = nativeLoadResultRef.get()
    if cached != null then cached
    else
      val loaded = tryLoadNative()
      nativeLoadResultRef.compareAndSet(null, loaded)
      nativeLoadResultRef.get()

  private def tryLoadNative(): Either[String, String] =
    val pathOpt =
      Option(System.getProperty(NativeLibPathProp))
        .orElse(Option(System.getenv(NativeLibPathEnv)))
        .map(_.trim).filter(_.nonEmpty)
    val libName =
      Option(System.getProperty(NativeLibNameProp))
        .orElse(Option(System.getenv(NativeLibNameEnv)))
        .map(_.trim).filter(_.nonEmpty)
        .getOrElse(NativeLib)
    pathOpt match
      case Some(path) =>
        try { System.load(path); Right(s"path=$path") }
        catch
          case ex: Throwable => Left(s"failed to load wasserstein native '$path': ${ex.getMessage}")
      case None =>
        try { System.loadLibrary(libName); Right(s"library=$libName") }
        catch
          case ex: Throwable =>
            tryLocalBuildFallback(libName) match
              case Right(src) => Right(src)
              case Left(fb)   => Left(s"failed to load wasserstein native '$libName': ${ex.getMessage}; $fb")

  private def tryLocalBuildFallback(libName: String): Either[String, String] =
    val userDir  = new java.io.File(System.getProperty("user.dir", "."))
    val buildDir = new java.io.File(userDir, "src/main/native/build")
    val osName   = System.getProperty("os.name", "").toLowerCase
    val candidates: Vector[java.io.File] =
      if osName.contains("win") then Vector(new java.io.File(buildDir, s"$libName.dll"))
      else if osName.contains("mac") then
        Vector(new java.io.File(buildDir, s"lib$libName.dylib"),
               new java.io.File(buildDir, s"$libName.dylib"))
      else
        Vector(new java.io.File(buildDir, s"lib$libName.so"),
               new java.io.File(buildDir, s"$libName.so"))
    val errors = scala.collection.mutable.Buffer[String]()
    var found: Option[String] = None
    val it = candidates.iterator
    while it.hasNext && found.isEmpty do
      val f = it.next()
      if f.isFile then
        try { System.load(f.getAbsolutePath); found = Some(s"path=${f.getAbsolutePath}") }
        catch case ex: Throwable => errors += s"${f.getAbsolutePath} (${ex.getMessage})"
      else errors += s"${f.getAbsolutePath} (missing)"
    found match
      case Some(src) => Right(src)
      case None      => Left(s"also tried local native paths: ${errors.mkString("; ")}")

  // ---- Public availability queries -------------------------------------------

  /** Whether the native Wasserstein JNI library is loaded and operational. */
  def isNativeAvailable: Boolean = nativeLoadResult().isRight

  /** Always true — pure Scala fallback is always available. */
  def isAvailable: Boolean = true

  /** Always true — built-in pure-Scala LP solver is always available. */
  def isGlpkAvailable: Boolean = true

  // ---- Validation helpers ----------------------------------------------------

  private def validateWeights(w: Array[Double]): Either[Int, Unit] =
    if w == null then Left(StatusNullArg)
    else if w.isEmpty then Left(StatusDimension)
    else
      var sum = 0.0
      var i = 0
      while i < w.length do
        if w(i) < 0.0 then return Left(StatusNegativeWeight)
        sum += w(i)
        i += 1
      if math.abs(sum - 1.0) > 1e-6 then Left(StatusNormalization)
      else Right(())

  private def validateCostMatrix(cost: Array[Double], n: Int, m: Int): Either[Int, Unit] =
    if cost == null then Left(StatusNullArg)
    else if cost.length != n * m then Left(StatusDimension)
    else
      var i = 0
      while i < cost.length do
        if cost(i) < 0.0 then return Left(StatusNegativeCost)
        i += 1
      Right(())

  // ---- Definition 33: Wasserstein-1 distance ---------------------------------

  /** Compute W_1(a, b) given weight vectors and a ground cost matrix.
    *
    * @param weightsA   distribution A (sums to 1.0, length n)
    * @param weightsB   distribution B (sums to 1.0, length m)
    * @param costMatrix ground metric (n*m, row-major, all entries >= 0)
    * @return Right(W_1) on success, Left(errorCode) on validation failure
    */
  def emd(
      weightsA: Array[Double],
      weightsB: Array[Double],
      costMatrix: Array[Double]
  ): Either[Int, Double] =
    for
      _ <- validateWeights(weightsA)
      _ <- validateWeights(weightsB)
      _ <- validateCostMatrix(costMatrix, weightsA.length, weightsB.length)
    yield
      if isNativeAvailable then
        val result = new Array[Double](1)
        try
          val status = sicfun.holdem.HoldemWassersteinBindings.computeEmd(
            weightsA, weightsB, costMatrix, weightsA.length, weightsB.length, result
          )
          if status == 0 then result(0)
          else PureScalaEmd.compute(weightsA, weightsB, costMatrix)
        catch
          case _: Throwable => PureScalaEmd.compute(weightsA, weightsB, costMatrix)
      else
        PureScalaEmd.compute(weightsA, weightsB, costMatrix)

  /** Batch EMD: W_1 from one reference distribution to K target distributions.
    *
    * @param weightsRef   reference distribution (length n)
    * @param weightsBatch K target distributions packed row-major (length K*m)
    * @param k            number of target distributions
    * @param costMatrix   shared ground metric (n*m, row-major)
    * @return Right(distances) of length k, or Left(errorCode) on failure
    */
  def emdBatch(
      weightsRef: Array[Double],
      weightsBatch: Array[Double],
      k: Int,
      costMatrix: Array[Double]
  ): Either[Int, Array[Double]] =
    if weightsRef == null || weightsBatch == null || costMatrix == null then Left(StatusNullArg)
    else if k <= 0 then Left(StatusDimension)
    else
      val n = weightsRef.length
      val m = if k > 0 then weightsBatch.length / k else 0
      if weightsBatch.length != k * m then Left(StatusDimension)
      else
        validateWeights(weightsRef).map { _ =>
          if isNativeAvailable then
            val results = new Array[Double](k)
            try
              val status = sicfun.holdem.HoldemWassersteinBindings.computeEmdBatch(
                weightsRef, weightsBatch, costMatrix, n, m, k, results
              )
              if status == 0 then results
              else pureScalaBatch(weightsRef, weightsBatch, k, m, costMatrix)
            catch
              case _: Throwable => pureScalaBatch(weightsRef, weightsBatch, k, m, costMatrix)
          else
            pureScalaBatch(weightsRef, weightsBatch, k, m, costMatrix)
        }

  private def pureScalaBatch(
      weightsRef: Array[Double],
      weightsBatch: Array[Double],
      k: Int,
      m: Int,
      costMatrix: Array[Double]
  ): Array[Double] =
    val out = new Array[Double](k)
    var i = 0
    while i < k do
      val wB = weightsBatch.slice(i * m, (i + 1) * m)
      out(i) = PureScalaEmd.compute(weightsRef, wB, costMatrix)
      i += 1
    out

  // ---- Definition 33: Ambiguity set membership -------------------------------

  /** Test whether b' is in B_ρ(bCenter): W_1(b', bCenter) ≤ rho. */
  def inAmbiguitySet(
      bCenter: Array[Double],
      bPrime: Array[Double],
      costMatrix: Array[Double],
      rho: Double
  ): Either[Int, Boolean] =
    emd(bCenter, bPrime, costMatrix).map(_ <= rho + 1e-10)

  /** Convenience alias matching the plan API. */
  def isInAmbiguitySet(
      center: Array[Double],
      candidate: Array[Double],
      rho: Double,
      costMatrix: Array[Double]
  ): Boolean =
    inAmbiguitySet(center, candidate, costMatrix, rho).getOrElse(false)

  // ---- Definition 34: DRO LP -------------------------------------------------

  /** Result of a DRO LP solve. */
  final case class DroResult(
      worstCaseValue: Double,
      worstCaseBelief: Array[Double],
      dualVariable: Double,
      status: SolveStatus
  )

  enum SolveStatus:
    case Optimal
    case Infeasible
    case Unbounded
    case NumericFailure

  private def standardQValue(belief: Array[Double], qValues: Array[Double]): Double =
    var v = 0.0
    var i = 0
    while i < belief.length do
      v += belief(i) * qValues(i)
      i += 1
    v

  /** Solve the DRO inner minimization (Definition 34):
    *   inf_{b' ∈ B_ρ(b)} q^T b'
    *
    * Formulated as a transportation LP with n² variables (the transport plan π_{ij}):
    *   minimize   Σ_{i,j} π_{ij} * q_j
    *   subject to Σ_j π_{ij} = b_i  for i = 0..n-1   (source marginal)
    *              Σ_{i,j} π_{ij} * c_{ij} ≤ ρ         (Wasserstein budget)
    *              π_{ij} ≥ 0
    *
    * When ρ = 0 the LP is skipped entirely (returns standard Q-value).
    * When n = 1 the LP is trivially the single Q-value.
    *
    * @param beliefCenter current belief b̃_t (length n)
    * @param qValues      Q-values per state (length n)
    * @param costMatrix   ground metric (n*n row-major)
    * @param rho          ambiguity radius (>= 0)
    */
  def solveDroLp(
      beliefCenter: Array[Double],
      qValues: Array[Double],
      costMatrix: Array[Double],
      rho: Double
  ): Either[String, DroResult] =
    if beliefCenter == null || qValues == null || costMatrix == null then
      Left("null argument")
    else if beliefCenter.length != qValues.length then
      Left(s"belief length ${beliefCenter.length} != qValues length ${qValues.length}")
    else if rho < 0.0 then
      Left(s"rho must be >= 0, got $rho")
    else
      val n = beliefCenter.length
      if costMatrix.length != n * n then
        Left(s"costMatrix must be $n×$n=${n * n} entries, got ${costMatrix.length}")
      else if rho == 0.0 then
        val v = standardQValue(beliefCenter, qValues)
        Right(DroResult(v, beliefCenter.clone(), 0.0, SolveStatus.Optimal))
      else if n == 1 then
        Right(DroResult(qValues(0), Array(1.0), 0.0, SolveStatus.Optimal))
      else
        PureScalaDroSimplex.solve(beliefCenter, qValues, costMatrix, rho)

  /** Convenience alias matching the plan API. */
  def robustQValue(
      belief: Array[Double],
      qPerState: Array[Double],
      rho: Double,
      costMatrix: Array[Double]
  ): Either[String, Double] =
    solveDroLp(belief, qPerState, costMatrix, rho).map(_.worstCaseValue)

// ---- Pure Scala EMD: SPFA-based min-cost transportation --------------------
//
// Computes W_1(a, b) via successive shortest paths on the residual graph.
// Weights are scaled to Long integers (scale = 1e9) to avoid floating-point
// precision issues in the flow algorithm.
//
// Complexity: O(F * SPFA) where F = total flow = 1e9 scaled units of mass,
// and SPFA is O((n+m)^2) per iteration. For small n, m this converges quickly
// because the number of augmenting paths equals at most min(n, m).
private object PureScalaEmd:

  def compute(wA: Array[Double], wB: Array[Double], cost: Array[Double]): Double =
    val n = wA.length
    val m = wB.length
    if n == 1 && m == 1 then return 0.0

    val Scale = 1_000_000_000L
    val supply = new Array[Long](n)
    val demand = new Array[Long](m)

    var i = 0
    var supplyTotal = 0L
    while i < n do
      supply(i) = math.round(wA(i) * Scale)
      supplyTotal += supply(i)
      i += 1

    var j = 0
    var demandTotal = 0L
    while j < m do
      demand(j) = math.round(wB(j) * Scale)
      demandTotal += demand(j)
      j += 1

    val slack = supplyTotal - demandTotal
    if slack > 0 then demand(m - 1) += slack
    else if slack < 0 then supply(n - 1) -= slack

    runSSP(supply, demand, cost, n, m, Scale)

  // Arc in the residual graph (mutable capacity).
  private final class Arc(val to: Int, var cap: Long, val cost: Double, val revIdx: Int)

  private def runSSP(
      supply: Array[Long],
      demand: Array[Long],
      cost: Array[Double],
      n: Int,
      m: Int,
      scale: Long
  ): Double =
    // Graph: super-source S, supply nodes 0..n-1, demand nodes n..n+m-1, super-sink T
    val S          = n + m
    val T          = n + m + 1
    val totalNodes = n + m + 2
    val InfCap     = Long.MaxValue / 4

    val adj = Array.fill(totalNodes)(new java.util.ArrayList[Arc]())

    def addArc(u: Int, v: Int, c: Long, w: Double): Unit =
      val fwdRevIdx = adj(v).size()
      val bwdRevIdx = adj(u).size()
      adj(u).add(new Arc(v, c, w, fwdRevIdx))
      adj(v).add(new Arc(u, 0L, -w, bwdRevIdx))

    // S -> supply nodes
    var i = 0
    while i < n do
      addArc(S, i, supply(i), 0.0)
      i += 1

    // Supply nodes -> demand nodes
    i = 0
    while i < n do
      var j = 0
      while j < m do
        addArc(i, n + j, InfCap, cost(i * m + j))
        j += 1
      i += 1

    // Demand nodes -> T
    var j = 0
    while j < m do
      addArc(n + j, T, demand(j), 0.0)
      j += 1

    val dist    = new Array[Double](totalNodes)
    val inQueue = new Array[Boolean](totalNodes)
    val prev    = new Array[Int](totalNodes)
    val prevArc = new Array[Int](totalNodes)
    val PosInf  = Double.MaxValue / 2

    var totalCost = 0.0
    var remaining = supply.sum

    while remaining > 0 do
      java.util.Arrays.fill(dist, PosInf)
      dist(S) = 0.0
      java.util.Arrays.fill(inQueue, false)
      java.util.Arrays.fill(prev, -1)

      val queue = new java.util.ArrayDeque[Int]()
      queue.addFirst(S)
      inQueue(S) = true

      while !queue.isEmpty do
        val u = queue.pollFirst()
        inQueue(u) = false
        var aIdx = 0
        val arcs = adj(u)
        val sz   = arcs.size()
        while aIdx < sz do
          val arc = arcs.get(aIdx)
          if arc.cap > 0 && dist(u) + arc.cost < dist(arc.to) - 1e-12 then
            dist(arc.to) = dist(u) + arc.cost
            prev(arc.to) = u
            prevArc(arc.to) = aIdx
            if !inQueue(arc.to) then
              inQueue(arc.to) = true
              if !queue.isEmpty && dist(arc.to) < dist(queue.peekFirst()) then
                queue.addFirst(arc.to)
              else
                queue.addLast(arc.to)
          aIdx += 1

      if dist(T) >= PosInf then
        remaining = 0  // No augmenting path (balanced supply/demand; should not happen)
      else
        var flow = remaining
        var v = T
        while v != S do
          val u   = prev(v)
          val arc = adj(u).get(prevArc(v))
          if arc.cap < flow then flow = arc.cap
          v = u

        v = T
        while v != S do
          val u   = prev(v)
          val arc = adj(u).get(prevArc(v))
          arc.cap -= flow
          adj(v).get(arc.revIdx).cap += flow
          totalCost += flow.toDouble * arc.cost
          v = u

        remaining -= flow

    totalCost / scale.toDouble

// ---- Pure Scala DRO LP Solver -----------------------------------------------
//
// Solves the DRO inner minimization as a transportation LP.
//
// LP in standard form (after adding slack s for the Wasserstein constraint):
//   Variables: π_{00}, ..., π_{(n-1)(n-1)}, s   (total = n² + 1)
//   Minimize:  Σ_{i,j} q_j * π_{ij}
//   Subject to:
//     Σ_j π_{ij} = b_i   (i = 0..n-1)         source marginal
//     Σ_{i,j} c_{ij} π_{ij} + s = ρ            Wasserstein budget (slack)
//     π_{ij} ≥ 0,  s ≥ 0
//
// Initial BFS via Big-M method: artificial variables added for each constraint.
// Simplex: full-tableau Bland's rule to avoid cycling.
private object PureScalaDroSimplex:

  private val Eps    = 1e-10
  private val MaxIter = 50_000

  def solve(
      b: Array[Double],
      q: Array[Double],
      c: Array[Double],
      rho: Double
  ): Either[String, WassersteinDroRuntime.DroResult] =
    val n     = b.length
    val nPi   = n * n          // transport variables π_{ij}
    val nVars = nPi + 1        // +1 for slack s
    val nCon  = n + 1          // n source-marginal equalities + 1 Wasserstein row

    // Objective: minimize q_j * π_{ij}; slack coefficient = 0
    val obj = new Array[Double](nVars)
    var i = 0
    while i < n do
      var j = 0
      while j < n do
        obj(i * n + j) = q(j)
        j += 1
      i += 1

    // Constraint matrix A (nCon × nVars), rhs
    val A    = Array.ofDim[Double](nCon, nVars)
    val bRhs = new Array[Double](nCon)

    // Rows 0..n-1: Σ_j π_{ij} = b_i
    i = 0
    while i < n do
      var j = 0
      while j < n do
        A(i)(i * n + j) = 1.0
        j += 1
      bRhs(i) = b(i)
      i += 1

    // Row n: Σ_{i,j} c_{ij} π_{ij} + s = rho
    i = 0
    while i < n do
      var j = 0
      while j < n do
        A(n)(i * n + j) = c(i * n + j)
        j += 1
      i += 1
    A(n)(nPi) = 1.0  // slack s
    bRhs(n)   = rho

    // Extended LP with Big-M artificials: one per constraint row
    val BigM      = 1e8
    val nArt      = nCon
    val nExt      = nVars + nArt
    val extObj    = new Array[Double](nExt)
    System.arraycopy(obj, 0, extObj, 0, nVars)
    var k = 0
    while k < nArt do
      extObj(nVars + k) = BigM
      k += 1

    val extA = Array.ofDim[Double](nCon, nExt)
    i = 0
    while i < nCon do
      System.arraycopy(A(i), 0, extA(i), 0, nVars)
      extA(i)(nVars + i) = 1.0
      i += 1

    val basis = Array.tabulate(nCon)(idx => nVars + idx)
    val x     = new Array[Double](nExt)
    i = 0
    while i < nCon do
      x(nVars + i) = bRhs(i)
      i += 1

    runSimplex(extA, bRhs, extObj, basis, x, nCon, nExt) match
      case Left(msg) => Left(msg)
      case Right(_) =>
        // Feasibility: all artificials must be zero
        var artInBasis = false
        k = 0
        while k < nCon do
          if basis(k) >= nVars && x(basis(k)) > Eps then artInBasis = true
          k += 1
        if artInBasis then
          Left("DRO LP infeasible (artificial variable remains in basis)")
        else
          // Recover worst-case belief: b'_j = Σ_i π_{ij}
          val worstBelief = new Array[Double](n)
          i = 0
          while i < n do
            var j = 0
            while j < n do
              worstBelief(j) += x(i * n + j)
              j += 1
            i += 1
          var worstVal = 0.0
          var j = 0
          while j < n do
            worstVal += q(j) * worstBelief(j)
            j += 1
          Right(
            WassersteinDroRuntime.DroResult(
              worstCaseValue = worstVal,
              worstCaseBelief = worstBelief,
              dualVariable = 0.0,
              status = WassersteinDroRuntime.SolveStatus.Optimal
            )
          )

  /** Full-tableau simplex. Modifies basis and x in-place.
    * Returns Right(()) at optimality, Left(message) on unboundedness or failure.
    */
  private def runSimplex(
      A: Array[Array[Double]],
      bRhs: Array[Double],
      c: Array[Double],
      basis: Array[Int],
      x: Array[Double],
      m: Int,
      nTotal: Int
  ): Either[String, Unit] =
    // Build full tableau: m rows × (nTotal + 1) columns [A | b]
    // We maintain the tableau under basis operations.
    val tab = Array.ofDim[Double](m, nTotal + 1)
    var r = 0
    while r < m do
      var col = 0
      while col < nTotal do
        tab(r)(col) = A(r)(col)
        col += 1
      tab(r)(nTotal) = bRhs(r)
      r += 1

    // Express the objective row in terms of non-basic variables by eliminating
    // basic variables.  We track reduced costs in a separate array.
    val rc = c.clone()

    // Pivot the initial basis columns to identity form in the tableau.
    r = 0
    while r < m do
      val bv = basis(r)
      val piv = tab(r)(bv)
      if math.abs(piv) > 1e-12 then
        var col = 0
        while col <= nTotal do
          tab(r)(col) /= piv
          col += 1
        // Eliminate bv from other rows and from rc
        var r2 = 0
        while r2 < m do
          if r2 != r then
            val factor = tab(r2)(bv)
            if math.abs(factor) > 1e-14 then
              var col2 = 0
              while col2 <= nTotal do
                tab(r2)(col2) -= factor * tab(r)(col2)
                col2 += 1
          r2 += 1
        val rcFactor = rc(bv)
        if math.abs(rcFactor) > 1e-14 then
          var col3 = 0
          while col3 < nTotal do
            rc(col3) -= rcFactor * tab(r)(col3)
            col3 += 1
      r += 1

    var iter  = 0
    var done  = false
    var error: Option[String] = None

    while iter < MaxIter && !done && error.isEmpty do
      // Find entering variable (most negative reduced cost — Bland's: smallest index)
      var enterIdx = -1
      var j = 0
      while j < nTotal do
        if rc(j) < -Eps then
          if enterIdx < 0 then enterIdx = j  // Bland's: first negative
        j += 1

      if enterIdx < 0 then
        done = true
      else
        // Ratio test: find leaving row
        var leaveRow = -1
        var minRatio = Double.MaxValue
        r = 0
        while r < m do
          val tabVal = tab(r)(enterIdx)
          if tabVal > Eps then
            val ratio = tab(r)(nTotal) / tabVal
            if ratio < minRatio - Eps then
              minRatio = ratio
              leaveRow = r
            else if math.abs(ratio - minRatio) <= Eps && leaveRow >= 0 then
              // Bland's tie-breaking: prefer lowest-index basic variable
              if basis(r) < basis(leaveRow) then leaveRow = r
          r += 1

        if leaveRow < 0 then
          error = Some("DRO LP unbounded")
        else
          // Pivot: normalize pivot row
          val pivVal = tab(leaveRow)(enterIdx)
          var col = 0
          while col <= nTotal do
            tab(leaveRow)(col) /= pivVal
            col += 1

          // Eliminate enterIdx from all other rows
          r = 0
          while r < m do
            if r != leaveRow then
              val factor = tab(r)(enterIdx)
              if math.abs(factor) > 1e-14 then
                var col2 = 0
                while col2 <= nTotal do
                  tab(r)(col2) -= factor * tab(leaveRow)(col2)
                  col2 += 1
            r += 1

          // Update reduced costs
          val rcEnter = rc(enterIdx)
          if math.abs(rcEnter) > 1e-14 then
            var col3 = 0
            while col3 < nTotal do
              rc(col3) -= rcEnter * tab(leaveRow)(col3)
              col3 += 1

          // Update basis and primal solution
          basis(leaveRow) = enterIdx
          r = 0
          while r < m do
            x(basis(r)) = tab(r)(nTotal)
            r += 1
          // Non-basic variables are zero (update all non-basics to 0)
          val basisSet = basis.toSet
          var nbi = 0
          while nbi < nTotal do
            if !basisSet.contains(nbi) then x(nbi) = 0.0
            nbi += 1

      iter += 1

    error match
      case Some(msg) => Left(msg)
      case None =>
        if iter >= MaxIter then Left(s"simplex did not converge after $MaxIter iterations")
        else Right(())
