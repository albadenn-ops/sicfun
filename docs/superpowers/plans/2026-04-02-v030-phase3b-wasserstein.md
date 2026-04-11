# Phase 3b: Wasserstein EMD + DRO LP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Wasserstein distance computation and distributionally robust optimization LP
**Architecture:** Vendored C++ EMD solver with JNI, GLPK-java for LP formulation, Scala runtime wrapper
**Tech Stack:** C++17, Scala 3.8.1, GLPK-java 1.12.0, munit 1.2.2
**Depends on:** Phase 1 (belief distribution types)
**Unlocks:** Phase 4b (adaptation safety uses DRO bounds)

---

## Formal Grounding

This phase implements two definitions from SICFUN-v0.30.2 canonical spec §6A:

**Definition 33 — Ambiguity set:**

```
B_ρ(b̃_t) := { b̃' ∈ Δ(X̃) : W_1(b̃', b̃_t) ≤ ρ }
```

where W_1 is the Wasserstein-1 (earth mover's) distance and ρ ≥ 0 is the ambiguity radius.

**Definition 34 — Robust Q-function:**

```
Q^{*,Γ,ρ}_Π(b̃, u) := sup_{π ∈ Π} inf_{b̃' ∈ B_ρ(b̃)} Q^{π,Γ}(b̃', u)
```

When ρ = 0, the robust Q-function reduces to the standard Q-function.

**Theorem 7 — Convexity of robust value function under Wasserstein ambiguity:**
V^{*,Γ,ρ}_Π(b̃) is convex in b̃ for fixed ρ. This is validated by a property test.

**Backward compatibility:** When ρ = 0, the DRO layer is a no-op. All tests must verify this degenerate case.

---

## File Map

```
src/main/native/vendor/
├── network_simplex_simple.h       VENDOR: nbonneel (MIT), network simplex EMD solver
└── full_bipartitegraph.h          VENDOR: nbonneel (MIT), complete bipartite graph

src/main/native/jni/
├── WassersteinEmd.hpp             C++ EMD engine (header-only, uses vendor headers)
└── HoldemWassersteinBindings.cpp  JNI entry points for W_1 computation

src/main/java/sicfun/holdem/
└── HoldemWassersteinBindings.java JNI class declaration (native methods)

src/main/scala/sicfun/holdem/strategic/solver/
└── WassersteinDroRuntime.scala    Scala runtime: JNI EMD + GLPK LP formulation

src/test/scala/sicfun/holdem/strategic/solver/
└── WassersteinDroRuntimeTest.scala

build.sbt                          MODIFY: add GLPK-java dependency
src/main/native/build-windows-llvm.ps1  MODIFY: add wasserstein DLL build step
```

**Estimated LOC:** ~490 total (130 C++, 30 Java, 250 Scala, 80 test)

---

## External Dependencies

| Dependency | Type | License | Source | Notes |
|---|---|---|---|---|
| nbonneel/network_simplex_simple.h | Vendored header | MIT | https://github.com/nbonneel/network_simplex | Single-header network simplex for optimal transport |
| nbonneel/full_bipartitegraph.h | Vendored header | MIT | https://github.com/nbonneel/network_simplex | Complete bipartite graph structure used by solver |
| GLPK-java 1.12.0 | Maven dependency | GPL-3.0 | `org.gnu.glpk:glpk-java:1.12.0` | LP solver for DRO inner minimization |

**GLPK native library requirement:** GLPK-java is a JNI wrapper around the native GLPK C library. The native `glpk_4_65.dll` (or equivalent) must be on the system PATH or `java.library.path`. On Windows, install via MSYS2 (`pacman -S mingw-w64-x86_64-glpk`) or download prebuilt binaries from https://winglpk.sourceforge.net/.

**Alternative (if GPL is a concern):** HiGHS via highs4j (Apache 2.0), as noted in the master plan.

---

## Task 0: Vendor File Acquisition

- [ ] **0.1** Create vendor directory: `src/main/native/vendor/`
- [ ] **0.2** Download `network_simplex_simple.h` from nbonneel's repository:
  - Source: https://github.com/nbonneel/network_simplex
  - File: `network_simplex_simple.h` (the main solver header)
  - Commit: use latest stable (check for C++17 compatibility)
  - Place at: `src/main/native/vendor/network_simplex_simple.h`
- [ ] **0.3** Download `full_bipartitegraph.h` from the same repository:
  - File: `full_bipartitegraph.h` (complete bipartite graph for EMD)
  - Place at: `src/main/native/vendor/full_bipartitegraph.h`
- [ ] **0.4** Add MIT license header comment at top of each vendored file noting provenance:
  ```
  // Vendored from: https://github.com/nbonneel/network_simplex
  // License: MIT (see LICENSE in source repository)
  // Vendored on: 2026-04-02
  ```
- [ ] **0.5** Verify both headers compile standalone with clang++ -std=c++17:
  ```powershell
  & "C:\Program Files\LLVM\bin\clang++.exe" -std=c++17 -fsyntax-only -I src/main/native/vendor src/main/native/vendor/network_simplex_simple.h
  ```

**Acceptance:** Both headers present in `src/main/native/vendor/`, compile clean with C++17.

---

## Task 1: build.sbt — Add GLPK-java Dependency

### Test-first verification

- [ ] **1.1** Before modifying, confirm current `sbt compile` succeeds (baseline green).

### Implementation

- [ ] **1.2** Add GLPK-java to `libraryDependencies` in `build.sbt`:

  In the `libraryDependencies ++= Seq(...)` block, add:
  ```scala
  "org.gnu.glpk" % "glpk-java" % "1.12.0"
  ```

  The full block becomes:
  ```scala
  libraryDependencies ++= Seq(
    "org.scalameta" %% "munit" % "1.2.2" % Test,
    "io.zonky.test" % "embedded-postgres" % embeddedPostgresVersion % Test,
    "com.lihaoyi" %% "ujson" % "3.3.1",
    "com.microsoft.onnxruntime" % "onnxruntime" % onnxRuntimeVersion,
    "org.postgresql" % "postgresql" % postgresqlJdbcVersion,
    "org.gnu.glpk" % "glpk-java" % "1.12.0"
  ),
  ```

### Verification

- [ ] **1.3** Run `sbt compile` — must resolve the GLPK-java artifact from Maven Central.
- [ ] **1.4** Verify `org.gnu.glpk.GLPK` class is on the compile classpath:
  ```
  sbt "show Compile/dependencyClasspath" | grep -i glpk
  ```

**Acceptance:** `sbt compile` green. GLPK-java JAR resolved.

---

## Task 2: C++ Wasserstein EMD Engine (WassersteinEmd.hpp)

### Test-first: write a minimal JNI caller test stub

We cannot unit-test C++ in isolation within this project's build, so the C++ is validated through the JNI binding test in Task 4. However, the C++ code must be designed for correctness first.

### Implementation

- [ ] **2.1** Create `src/main/native/jni/WassersteinEmd.hpp` — header-only C++ engine.

  **Design contract:**
  - Computes Wasserstein-1 distance between two discrete distributions over a finite metric space.
  - Input: two weight vectors `w_a[n]`, `w_b[m]` (non-negative, each summing to 1.0) and a cost matrix `cost[n*m]` (row-major, `cost[i*m + j]` = ground distance from support point i to support point j).
  - Output: scalar W_1 distance (double).
  - Uses vendored `network_simplex_simple.h` and `full_bipartitegraph.h` for the optimal transport computation.
  - The cost matrix encodes the ground metric; for belief distributions over augmented states, this is a user-supplied metric (the Scala layer constructs it).

  **Required functions:**
  ```cpp
  namespace sicfun::wasserstein {

  // Compute W_1(a, b) given weight vectors and cost matrix.
  // Preconditions: n >= 1, m >= 1, weights sum to 1.0 (within tolerance),
  //                cost is n*m row-major, all entries >= 0.
  // Returns: optimal transport cost (W_1 distance).
  double compute_emd(
      const double* w_a, int n,
      const double* w_b, int m,
      const double* cost  // n*m row-major
  );

  // Batch variant: compute W_1 for K distribution pairs against a shared reference.
  // w_ref[n] is the reference distribution.
  // w_batch[K*m] contains K distributions of size m each (row-major).
  // cost[n*m] is the shared cost matrix.
  // out[K] receives the K distance values.
  // Returns: 0 on success, non-zero status code on error.
  int compute_emd_batch(
      const double* w_ref, int n,
      const double* w_batch, int K, int m,
      const double* cost,
      double* out
  );

  } // namespace sicfun::wasserstein
  ```

  **Error status codes (consistent with existing JNI bindings):**
  - 0: success
  - 100: null pointer argument
  - 101: dimension mismatch (n < 1 or m < 1)
  - 102: weight normalization failure (sum deviates from 1.0 by more than 1e-6)
  - 103: negative weight
  - 104: negative cost entry

  **Implementation notes:**
  - The `compute_emd` function constructs a `FullBipartiteGraph` of size (n, m), populates arc costs from the cost matrix, sets supplies from the weight vectors (w_a as sources, w_b as sinks), and runs the network simplex solver.
  - Weight vectors are rescaled internally to integer supplies (multiply by a large factor, e.g., 1e9, and round) since the network simplex operates on integer capacities. The result is rescaled back.
  - The batch variant loops over K pairs, reusing the graph structure.

### Verification

- [ ] **2.2** Verify header compiles with the vendor includes:
  ```powershell
  & "C:\Program Files\LLVM\bin\clang++.exe" -std=c++17 -fsyntax-only `
    -I src/main/native/vendor `
    -I "$env:JAVA_HOME\include" `
    -I "$env:JAVA_HOME\include\win32" `
    src/main/native/jni/WassersteinEmd.hpp
  ```

**Acceptance:** Header compiles clean. All functions implemented with proper error checking.

---

## Task 3: JNI Binding Layer

### 3A: Java JNI Class Declaration

- [ ] **3.1** Create `src/main/java/sicfun/holdem/HoldemWassersteinBindings.java`:

  ```java
  package sicfun.holdem;

  /**
   * JNI bindings for Wasserstein-1 (earth mover's) distance computation.
   * Native implementation uses vendored network simplex solver (nbonneel, MIT).
   * Compiled into: sicfun_wasserstein_native.dll
   */
  public final class HoldemWassersteinBindings {
    private HoldemWassersteinBindings() {}

    /**
     * Computes Wasserstein-1 distance between two discrete distributions.
     *
     * @param weightsA  source distribution weights (length n, sums to 1.0)
     * @param weightsB  target distribution weights (length m, sums to 1.0)
     * @param costMatrix ground metric cost matrix (length n*m, row-major)
     * @param n         size of distribution A
     * @param m         size of distribution B
     * @param result    output array of length 1 (receives W_1 distance)
     * @return 0 on success, non-zero error code on failure
     */
    public static native int computeEmd(
        double[] weightsA,
        double[] weightsB,
        double[] costMatrix,
        int n,
        int m,
        double[] result
    );

    /**
     * Batch EMD: computes W_1 for K distributions against one reference.
     *
     * @param weightsRef   reference distribution (length n)
     * @param weightsBatch K target distributions (length K*m, row-major)
     * @param costMatrix   shared cost matrix (length n*m, row-major)
     * @param n            size of reference distribution
     * @param m            size of each target distribution
     * @param k            number of target distributions
     * @param results      output array of length K (receives W_1 distances)
     * @return 0 on success, non-zero error code on failure
     */
    public static native int computeEmdBatch(
        double[] weightsRef,
        double[] weightsBatch,
        double[] costMatrix,
        int n,
        int m,
        int k,
        double[] results
    );

    /**
     * Returns the native engine identifier (always 1 = CPU for this binding).
     */
    public static native int queryNativeEngine();
  }
  ```

  **Note:** The JNI class lives in `sicfun.holdem` (not `sicfun.holdem.strategic.solver`) because JNI package names are baked into native DLL function signatures. All existing JNI binding classes follow this convention.

### 3B: C++ JNI Entry Points

- [ ] **3.2** Create `src/main/native/jni/HoldemWassersteinBindings.cpp`:

  **Structure:**
  ```cpp
  #include <jni.h>
  #include "WassersteinEmd.hpp"

  extern "C" {

  JNIEXPORT jint JNICALL
  Java_sicfun_holdem_HoldemWassersteinBindings_computeEmd(
      JNIEnv* env, jclass,
      jdoubleArray weightsA,
      jdoubleArray weightsB,
      jdoubleArray costMatrix,
      jint n, jint m,
      jdoubleArray result
  ) {
      // 1. Null-check all array arguments (return 100 on null)
      // 2. Validate array lengths: weightsA.length == n, weightsB.length == m,
      //    costMatrix.length == n*m, result.length >= 1
      // 3. GetDoubleArrayElements for all input arrays
      // 4. Call sicfun::wasserstein::compute_emd(wA, n, wB, m, cost)
      // 5. Write scalar result to result[0]
      // 6. ReleaseDoubleArrayElements with JNI_ABORT (read-only inputs)
      //    and 0 (copy-back) for result
      // 7. Return status code
  }

  JNIEXPORT jint JNICALL
  Java_sicfun_holdem_HoldemWassersteinBindings_computeEmdBatch(
      JNIEnv* env, jclass,
      jdoubleArray weightsRef,
      jdoubleArray weightsBatch,
      jdoubleArray costMatrix,
      jint n, jint m, jint k,
      jdoubleArray results
  ) {
      // Same pattern as computeEmd but calls compute_emd_batch
      // Validates weightsBatch.length == k*m, results.length == k
  }

  JNIEXPORT jint JNICALL
  Java_sicfun_holdem_HoldemWassersteinBindings_queryNativeEngine(
      JNIEnv*, jclass
  ) {
      return 1; // CPU-only
  }

  } // extern "C"
  ```

  **Implementation notes:**
  - Follow the exact JNI function naming convention: `Java_sicfun_holdem_HoldemWassersteinBindings_<method>`.
  - Error handling: catch C++ exceptions from the network simplex solver, translate to JNI status codes, never propagate exceptions across the JNI boundary.
  - No global state. All calls are stateless and thread-safe.

### 3C: Build Script Modification

- [ ] **3.3** Add wasserstein DLL build step to `src/main/native/build-windows-llvm.ps1`:

  Append after the existing DDRE build block:
  ```powershell
  $wassSrc = Join-Path $PSScriptRoot "jni\HoldemWassersteinBindings.cpp"
  $wassDll = Join-Path $OutDir "sicfun_wasserstein_native.dll"
  $wassLib = Join-Path $OutDir "sicfun_wasserstein_native.lib"
  $wassExp = Join-Path $OutDir "sicfun_wasserstein_native.exp"

  if (Test-Path $wassDll) { Remove-Item $wassDll -Force }
  if (Test-Path $wassLib) { Remove-Item $wassLib -Force }
  if (Test-Path $wassExp) { Remove-Item $wassExp -Force }

  & $clang `
    -std=c++17 `
    -O3 `
    -DNDEBUG `
    -D_CRT_SECURE_NO_WARNINGS `
    -shared `
    "-I$jniInclude" `
    "-I$jniWinInclude" `
    "-I$PSScriptRoot\vendor" `
    -o $wassDll `
    $wassSrc

  if ($LASTEXITCODE -ne 0) {
    throw "Native Wasserstein build failed with exit code $LASTEXITCODE"
  }

  if (-not (Test-Path $wassDll)) {
    throw "Build did not produce $wassDll"
  }

  Write-Host "Built: $wassDll"
  ```

  **Key difference from other build steps:** the `-I$PSScriptRoot\vendor` include path, which makes the vendored headers visible.

### Verification

- [ ] **3.4** Run the build script; confirm `sicfun_wasserstein_native.dll` is produced in the build output directory.
- [ ] **3.5** Run `sbt compile` to verify the Java JNI class compiles.

**Acceptance:** DLL builds clean. Java class compiles. JNI function names match the Java native declarations.

---

## Task 4: Scala Runtime Wrapper (WassersteinDroRuntime.scala)

### Test-first

- [ ] **4.1** Create `src/test/scala/sicfun/holdem/strategic/solver/WassersteinDroRuntimeTest.scala` with the following test cases BEFORE writing the implementation.

  **Test suite structure:**

  ```
  WassersteinDroRuntimeTest
  ├── EMD Tests
  │   ├── "W_1 of identical distributions is zero"
  │   ├── "W_1 of Dirac masses equals ground distance"
  │   ├── "W_1 is symmetric: W_1(a,b) == W_1(b,a)"
  │   ├── "W_1 satisfies triangle inequality"
  │   ├── "W_1 of uniform distributions on unit metric"
  │   ├── "W_1 batch matches individual calls"
  │   └── "W_1 with trivial 1x1 distributions is zero"
  ├── Ambiguity Set Tests (Definition 33)
  │   ├── "ambiguity set with rho=0 contains only the center"
  │   ├── "ambiguity set with rho>0 contains perturbations within radius"
  │   ├── "ambiguity set rejects distributions outside radius"
  │   └── "ambiguity set membership is consistent with W_1"
  ├── DRO LP Tests (Definition 34)
  │   ├── "DRO with rho=0 equals standard Q-value"
  │   ├── "DRO with rho>0 returns value <= standard Q-value"
  │   ├── "DRO LP produces feasible dual variables"
  │   ├── "DRO LP solution satisfies weak duality"
  │   └── "DRO worst-case belief lies within ambiguity set"
  ├── Backward Compatibility
  │   └── "rho=0 recovers non-robust behavior exactly"
  └── Theorem 7 Validation
      └── "robust value function is convex in belief for fixed rho"
  ```

  **Test data construction:**
  - Small distributions (n = 2..5 support points) with known analytic W_1 values.
  - Ground metric: use simple Euclidean distance on integer-indexed support points (d(i,j) = |i - j|) for reproducibility.
  - For the Dirac-mass test: delta_i vs delta_j should give W_1 = d(i,j) exactly.
  - For the triangle inequality test: sample three random distributions, compute all three pairwise W_1, verify W_1(a,c) <= W_1(a,b) + W_1(b,c).
  - For the DRO LP tests: construct a toy 3-state problem with known Q-values per state, solve the DRO LP, verify the inner inf is achieved at a feasible point.
  - For Theorem 7 (convexity): sample several belief points b_1, ..., b_k and a convex combination lambda, verify V(sum lambda_i b_i) <= sum lambda_i V(b_i) within numerical tolerance.

  **Test structure notes:**
  - Tests that require the native DLL use `assume(WassersteinDroRuntime.isAvailable, "Wasserstein native library not loaded")` to skip gracefully when the DLL is absent (CI without native build).
  - Tests that exercise only the GLPK LP path use `assume(WassersteinDroRuntime.isGlpkAvailable, "GLPK native library not loaded")`.
  - All numeric comparisons use `assertEqualsDouble` with tolerance 1e-9.

### Implementation

- [ ] **4.2** Create `src/main/scala/sicfun/holdem/strategic/solver/WassersteinDroRuntime.scala`:

  **Public API:**

  ```scala
  package sicfun.holdem.strategic.solver

  /** Runtime for Wasserstein-1 distance computation and distributionally robust
    * optimization (DRO) LP formulation.
    *
    * Implements v0.30.2 §6A:
    *  - Definition 33: Ambiguity set B_ρ(b̃_t)
    *  - Definition 34: Robust Q-function Q^{*,Γ,ρ}_Π(b̃, u)
    *
    * EMD computed via vendored C++ network simplex (JNI).
    * DRO inner minimization solved via GLPK-java LP.
    */
  object WassersteinDroRuntime:

    /** Whether the native Wasserstein library is loaded and operational. */
    def isAvailable: Boolean

    /** Whether the GLPK LP solver is loaded and operational. */
    def isGlpkAvailable: Boolean

    // --- Definition 33: Wasserstein-1 distance ---

    /** Compute W_1(a, b) given weight vectors and a ground cost matrix.
      *
      * @param weightsA distribution A (sums to 1.0, length n)
      * @param weightsB distribution B (sums to 1.0, length m)
      * @param costMatrix ground metric (n*m, row-major)
      * @return W_1 distance, or Left(errorCode) on failure
      */
    def emd(
        weightsA: Array[Double],
        weightsB: Array[Double],
        costMatrix: Array[Double]
    ): Either[Int, Double]

    /** Batch EMD: W_1 from a reference to K target distributions. */
    def emdBatch(
        weightsRef: Array[Double],
        weightsBatch: Array[Double],
        k: Int,
        costMatrix: Array[Double]
    ): Either[Int, Array[Double]]

    // --- Definition 33: Ambiguity set membership ---

    /** Test whether b' is in B_ρ(b_center).
      * Returns true iff W_1(b', b_center) <= rho.
      */
    def inAmbiguitySet(
        bCenter: Array[Double],
        bPrime: Array[Double],
        costMatrix: Array[Double],
        rho: Double
    ): Either[Int, Boolean]

    // --- Definition 34: DRO LP ---

    /** Solve the DRO inner minimization:
      *   inf_{b' ∈ B_ρ(b)} sum_i b'_i * qValues_i
      *
      * This is a linear program:
      *   minimize  q^T b'
      *   subject to  W_1(b', b) <= rho
      *               b' ∈ Δ(X̃)
      *
      * Reformulated as a standard transportation LP using the Kantorovich
      * dual of the Wasserstein constraint.
      *
      * @param beliefCenter current belief b̃_t (length n)
      * @param qValues Q-values per state (length n)
      * @param costMatrix ground metric (n*n, row-major, square for same-support DRO)
      * @param rho ambiguity radius (>= 0)
      * @return DroResult containing worst-case value and worst-case belief
      */
    def solveDroLp(
        beliefCenter: Array[Double],
        qValues: Array[Double],
        costMatrix: Array[Double],
        rho: Double
    ): Either[String, DroResult]

    /** Result of a DRO LP solve. */
    final case class DroResult(
        worstCaseValue: Double,
        worstCaseBelief: Array[Double],
        dualVariable: Double,     // Lagrange multiplier for W_1 constraint
        status: SolveStatus
    )

    enum SolveStatus:
      case Optimal
      case Infeasible
      case Unbounded
      case NumericFailure
  ```

  **Internal design of `solveDroLp`:**

  The DRO inner minimization `inf_{b' : W_1(b', b) <= ρ} q^T b'` has a known Kantorovich dual reformulation as a finite LP. For distributions on n support points:

  **Primal (transportation form):**
  ```
  minimize   Σ_j q_j * b'_j
  subject to Σ_{i,j} π_{ij} * c_{ij}  ≤  ρ          (Wasserstein constraint)
             Σ_j π_{ij}  =  b_i        for all i    (marginal = center)
             Σ_i π_{ij}  =  b'_j       for all j    (marginal = perturbed)
             π_{ij} ≥ 0
  ```

  Substituting `b'_j = Σ_i π_{ij}` into the objective eliminates b' and yields a standard LP in the transport plan π with n^2 variables and 2n + 1 constraints.

  **GLPK LP construction:**
  - Create a GLPK problem with n^2 structural variables (π_{ij}).
  - Objective: minimize `Σ_{i,j} π_{ij} * q_j` (the q-value of the destination state, weighted by transport mass).
  - Row constraints:
    - n equality rows: `Σ_j π_{ij} = b_i` (source marginal)
    - 1 inequality row: `Σ_{i,j} π_{ij} * c_{ij} <= ρ` (Wasserstein budget)
  - Note: the sink marginal constraints are implicit (b'_j = Σ_i π_{ij}), so they do not appear as separate constraints; b' is recovered from the optimal π.
  - Bounds: all π_{ij} >= 0.
  - Solve with GLPK simplex.
  - Extract: optimal objective = worst-case value, b'_j = Σ_i π*_{ij} = worst-case belief, dual of Wasserstein constraint = Lagrange multiplier.

  **Special case ρ = 0:** Skip LP entirely, return `DroResult(q^T b, b, 0.0, Optimal)`.

  **Library loading:**
  - Native EMD library: `sicfun_wasserstein_native` (loaded via `System.loadLibrary` with `java.library.path` fallback, same pattern as `HoldemPostflopNativeRuntime`).
  - GLPK: loaded implicitly by `org.gnu.glpk.GLPK` class initialization. Catch `UnsatisfiedLinkError` to set `isGlpkAvailable = false`.

### Verification

- [ ] **4.3** Run `WassersteinDroRuntimeTest` — all tests must pass.
- [ ] **4.4** Verify `sbt compile` is green with no warnings.
- [ ] **4.5** Verify the `rho=0` backward-compatibility test passes without GLPK (pure EMD path).

**Acceptance:** All 17 test cases pass. No compilation warnings. DRO LP produces feasible, dual-consistent solutions on toy problems.

---

## Task 5: Edge Cases and Robustness

- [ ] **5.1** Add test: "EMD with very small weights (1e-15) does not crash or return NaN"
- [ ] **5.2** Add test: "DRO LP with very large rho returns global minimum of q-values"
  - When ρ is large enough to reach any distribution, the worst case is the Dirac mass on the state with minimum Q-value.
- [ ] **5.3** Add test: "DRO LP with degenerate single-point distribution"
  - n = 1: W_1 is trivially zero, DRO result equals q[0] regardless of ρ.
- [ ] **5.4** Add test: "EMD rejects unnormalized weights with error code 102"
- [ ] **5.5** Add test: "EMD rejects negative weights with error code 103"
- [ ] **5.6** Add test: "EMD rejects negative cost entries with error code 104"

**Acceptance:** All edge-case tests pass. Error codes match the documented status codes.

---

## Task 6: Theorem 7 Convexity Validation

- [ ] **6.1** Add a dedicated property-based test for Theorem 7.

  **Test design:**
  - Fix ρ > 0 (e.g., 0.1).
  - Define a ground metric on n = 4 states (e.g., d(i,j) = |i - j|).
  - Define a Q-value vector (e.g., [1.0, 2.0, 0.5, 3.0]).
  - Sample 20 random beliefs b_1, ..., b_20 from the simplex.
  - For each pair (b_i, b_j) and lambda in {0.1, 0.25, 0.5, 0.75, 0.9}:
    - Compute V(lambda * b_i + (1-lambda) * b_j) via `solveDroLp`.
    - Compute lambda * V(b_i) + (1-lambda) * V(b_j).
    - Assert V(convex_combo) <= lambda * V(b_i) + (1-lambda) * V(b_j) + 1e-9.

  **Note:** This test validates the spec's Theorem 7 claim. Failure would indicate either a bug in the LP formulation or a violation of the theorem's preconditions.

- [ ] **6.2** Test must use `assume(WassersteinDroRuntime.isGlpkAvailable)` to skip when GLPK is absent.

**Acceptance:** Convexity holds for all sampled belief pairs and all lambda values.

---

## Task 7: Integration Smoke Test

- [ ] **7.1** Add integration test: round-trip from Scala belief array through JNI EMD and back.
  - Construct two `Array[Double]` beliefs, call `emd`, verify result matches hand-computed value.
  - This validates the full stack: Scala -> Java JNI class -> C++ native -> vendored network simplex -> back.

- [ ] **7.2** Add integration test: full DRO solve on a 3-state poker-inspired scenario.
  - States represent opponent types: {tight, neutral, loose}.
  - Belief center: [0.5, 0.3, 0.2].
  - Q-values: [10.0, 5.0, -3.0] (chips EV per action against each type).
  - Ground metric: Hamming distance (d(i,j) = 1 if i != j, 0 otherwise).
  - ρ = 0.1.
  - Verify: DRO value < non-robust value (10*0.5 + 5*0.3 + (-3)*0.2 = 5.9).
  - Verify: worst-case belief shifts mass toward the low-Q state.

- [ ] **7.3** Verify native library graceful degradation: when DLL is absent, `isAvailable` returns false and `emd` returns `Left(errorCode)` without throwing.

**Acceptance:** Full-stack round-trip produces correct results. Graceful degradation works.

---

## Verification Checklist (all tasks)

- [ ] `sbt compile` green (no warnings, no errors)
- [ ] `sbt test` passes all existing tests (formal layer is additive, no regressions)
- [ ] `WassersteinDroRuntimeTest` passes all 23+ test cases
- [ ] Native DLL builds via `build-windows-llvm.ps1` without errors
- [ ] GLPK-java resolves from Maven Central
- [ ] Vendored headers have MIT license annotations
- [ ] `WassersteinDroRuntime` lives in `sicfun.holdem.strategic.solver` (correct namespace)
- [ ] `HoldemWassersteinBindings.java` lives in `sicfun.holdem` (JNI namespace constraint)
- [ ] No imports from `sicfun.holdem.engine` or `sicfun.holdem.runtime` in any new file
- [ ] ρ = 0 backward compatibility verified by explicit test

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| GLPK native DLL not found at runtime | Detect via `isGlpkAvailable`, skip LP tests with `assume`, document install steps |
| nbonneel headers use C++14 features incompatible with our C++17 build | Verified at Task 0.5; if issues arise, patch vendored headers (MIT license allows) |
| Network simplex numeric instability on near-degenerate distributions | Rescale weights to integer supplies (1e9 factor); add epsilon-perturbation for zero-weight states |
| GLPK GPL-3.0 license concern | Alternative noted in master plan: HiGHS via highs4j (Apache 2.0). Switch is localized to the LP construction in `solveDroLp` |
| Large state spaces make n^2 LP variables expensive | Phase 3b scope is limited to small belief dimensions (n <= ~50 augmented states); production use in Phase 4 may require state abstraction |
