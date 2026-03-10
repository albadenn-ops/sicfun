package sicfun.holdem.runtime
import sicfun.holdem.types.*

import munit.FunSuite

class LiveHandSimulatorTest extends FunSuite:
  test("run executes end-to-end and returns recommendation plus signals") {
    val result = LiveHandSimulator.run(Array(
      "--hero=AcKh",
      "--villainStyle=tag",
      "--seed=7",
      "--adaptiveObservations=8",
      "--bunchingTrials=120",
      "--equityTrials=900",
      "--showTopPosterior=3",
      "--keepArtifacts=false"
    ))

    assert(result.isRight, s"simulation failed: $result")
    val summary = result.toOption.getOrElse(fail("missing simulation summary"))

    assert(summary.signalCount > 0, s"expected signals > 0, got ${summary.signalCount}")
    assert(summary.topPosterior.nonEmpty, "expected non-empty posterior summary")
    assert(summary.topPosterior.length <= 3, s"expected at most 3 posterior rows, got ${summary.topPosterior.length}")
    assertEquals(summary.artifactRoot, None)
    assert(
      Set(PokerAction.Fold, PokerAction.Call, PokerAction.Raise(20.0)).contains(summary.bestAction),
      s"unexpected best action: ${summary.bestAction}"
    )
  }

  test("run returns Left for unsupported villain style") {
    val result = LiveHandSimulator.run(Array("--villainStyle=robot"))
    assert(result.isLeft, "expected invalid villain style to fail")
  }
