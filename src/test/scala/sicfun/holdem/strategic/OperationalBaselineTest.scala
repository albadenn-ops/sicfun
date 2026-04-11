package sicfun.holdem.strategic

class OperationalBaselineTest extends munit.FunSuite:

  test("DeploymentBeliefSummary stores entropy and exploitability"):
    val summary = DeploymentBeliefSummary(
      beliefEntropy = 1.5,
      exploitabilitySnapshot = Ev(0.03),
      timestamp = 1000L
    )
    assertEqualsDouble(summary.beliefEntropy, 1.5, 1e-12)

  test("EmpiricalDeploymentSet respects maxSize"):
    var set = EmpiricalDeploymentSet(Vector.empty, maxSize = 3)
    for i <- 0 until 5 do
      set = set.add(DeploymentBeliefSummary(i.toDouble, Ev(0.01 * i), i.toLong))
    assertEquals(set.entries.size, 3)
    // Oldest entries should be dropped
    assertEqualsDouble(set.entries.head.beliefEntropy, 2.0, 1e-12)

  test("EmpiricalDeploymentSet.deploymentExploitability is max"):
    val entries = Vector(
      DeploymentBeliefSummary(1.0, Ev(0.02), 1L),
      DeploymentBeliefSummary(2.0, Ev(0.05), 2L),
      DeploymentBeliefSummary(1.5, Ev(0.03), 3L)
    )
    val set = EmpiricalDeploymentSet(entries, maxSize = 10)
    val depExpl = set.deploymentExploitability
    assertEqualsDouble(depExpl.value, 0.05, 1e-12)

  test("OperationalBaseline stores epsilonBase"):
    val baseline = OperationalBaseline(
      epsilonBase = 0.05,
      deploymentSet = EmpiricalDeploymentSet(Vector.empty),
      description = "CFR-derived"
    )
    assertEqualsDouble(baseline.epsilonBase, 0.05, 1e-12)
