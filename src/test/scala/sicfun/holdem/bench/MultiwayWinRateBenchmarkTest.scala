package sicfun.holdem.bench

import java.nio.file.Files

class MultiwayWinRateBenchmarkTest extends munit.FunSuite:
  // Real 9-max self-play is CPU-heavy. The 2000-hand adaptive rigged run alone measured ~809s
  // on a GTX-960M-class dev box (the determinism pair of 300-hand strategic runs ~104s), so a
  // 600s per-test deadline trips on the rigged test even though the measurement itself completes
  // correctly (adaptive vs 8 stations = +45 bb/100). Raise to 1800s (≈2.2x measured headroom for
  // cold-JVM / CI variance) — this is the harness's patience ceiling, NOT a measurement parameter.
  override val munitTimeout = scala.concurrent.duration.Duration(1800, "s")

  test("same seed produces identical bb/100 and CI (computational replay)") {
    val tmp = Files.createTempDirectory("p0-trackA-det-")
    val r1 = MultiwayWinRateBenchmark.run(CanonicalField.NineMaxExploitable, heroStyle = "strategic", hands = 300, seed = 11L, outDir = tmp.resolve("a"), equityTrials = Some(48), bunchingTrials = Some(1))
    val r2 = MultiwayWinRateBenchmark.run(CanonicalField.NineMaxExploitable, heroStyle = "strategic", hands = 300, seed = 11L, outDir = tmp.resolve("b"), equityTrials = Some(48), bunchingTrials = Some(1))
    assert(r1.isRight && r2.isRight, s"$r1 / $r2")
    assertEquals(r1.toOption.get.ci.pointEstimate, r2.toOption.get.ci.pointEstimate)
    assertEquals(r1.toOption.get.ci.lower, r2.toOption.get.ci.lower)
  }

  test("rigged all-station field: adaptive hero edge CI clears zero (harness detects an edge)") {
    val tmp = Files.createTempDirectory("p0-trackA-rig-")
    // Adaptive (real equity engine) vs 8 calling stations: a large, reliable edge — this validates
    // that the HARNESS can DETECT an edge (independent of the strategic overlay's strength).
    val res = MultiwayWinRateBenchmark.run(CanonicalField.NineMaxRiggedStations, heroStyle = "adaptive", hands = 2000, seed = 5L, outDir = tmp, equityTrials = Some(80), bunchingTrials = Some(1))
    assert(res.isRight, res.toString)
    val ci = res.toOption.get.ci
    assertEquals(ci.sampleSize, 2000)
    assert(ci.lower > 0.0, s"expected positive lower CI vs all-calling-stations, got lower=${ci.lower} point=${ci.pointEstimate}")
  }
