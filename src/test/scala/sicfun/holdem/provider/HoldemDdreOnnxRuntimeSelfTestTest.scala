package sicfun.holdem.provider

import munit.FunSuite

/** Verifies that [[HoldemDdreOnnxRuntime.selfTest]] catches API drift in the
  * reflection bridge to ai.onnxruntime.
  *
  * The runtime is on the compile classpath (build.sbt:16), so the test classpath
  * includes it -- every check in selfTest() must resolve. If any lookup fails,
  * the assertion failure surfaces the exact class/method that drifted, which is
  * the actionable signal an onnxruntime upgrade needs (the compiler cannot warn
  * about reflection drift).
  */
class HoldemDdreOnnxRuntimeSelfTestTest extends FunSuite:

  test("selfTest passes against the on-classpath ai.onnxruntime version") {
    val report = HoldemDdreOnnxRuntime.selfTest()
    assert(
      report.allOk,
      s"ONNX reflection bridge is broken; lookup status:\n${report.summary}"
    )
  }

  test("selfTest reports every documented reflection target") {
    // Locks the count -- if a new Class.forName / getMethod gets added to runOnnx
    // without being mirrored in selfTest, the count drifts and this fails. Update
    // both in lockstep.
    val report = HoldemDdreOnnxRuntime.selfTest()
    assertEquals(
      report.checks.size,
      9,
      "selfTest should mirror runOnnx's reflection chain; if you added a lookup, also add a check."
    )
  }

  test("SelfTestReport.summary lists each check on its own line") {
    val report = HoldemDdreOnnxRuntime.selfTest()
    val lines = report.summary.split('\n')
    assertEquals(lines.length, report.checks.size)
    lines.foreach { line =>
      assert(
        line.startsWith("  OK ") || line.startsWith("  FAIL "),
        s"unexpected summary line shape: $line"
      )
    }
  }

  test("SelfTestReport.allOk is true only when every check resolved") {
    val allRight = HoldemDdreOnnxRuntime.SelfTestReport(
      Vector("a" -> Right(()), "b" -> Right(()))
    )
    assert(allRight.allOk)
    val mixed = HoldemDdreOnnxRuntime.SelfTestReport(
      Vector("a" -> Right(()), "b" -> Left("missing"))
    )
    assert(!mixed.allOk)
  }
