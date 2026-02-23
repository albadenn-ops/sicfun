package sicfun.core

import munit.FunSuite

class ActionMetricsTest extends FunSuite:

  // A uniform action model: every action equally likely regardless of hypothesis
  private object UniformModel extends ActionModel[String, Int, String]:
    def likelihood(action: Int, features: String, hypothesis: String): Double = 0.25

  // A deterministic model: each hypothesis maps to exactly one action
  private object DeterministicModel extends ActionModel[String, Int, String]:
    def likelihood(action: Int, features: String, hypothesis: String): Double =
      hypothesis match
        case "A" => if action == 0 then 1.0 else 0.0
        case "B" => if action == 1 then 1.0 else 0.0
        case "C" => if action == 2 then 1.0 else 0.0
        case "D" => if action == 3 then 1.0 else 0.0
        case _   => 0.25

  private val actions = Seq(0, 1, 2, 3)
  private val state = "any"
  private val uniformPosterior = DiscreteDistribution.uniform(Seq("A", "B", "C", "D"))

  test("uniform model yields action entropy = log2(numActions)") {
    val h = ActionMetrics.actionEntropy(state, uniformPosterior, UniformModel, actions)
    assertEqualsDouble(h, 2.0, 1e-9) // log2(4) = 2
  }

  test("deterministic model yields conditional entropy = 0") {
    val ce = ActionMetrics.conditionalActionEntropy(state, uniformPosterior, DeterministicModel, actions)
    assertEqualsDouble(ce, 0.0, 1e-9)
  }

  test("mutual information is non-negative") {
    val mi = ActionMetrics.mutualInformation(state, uniformPosterior, UniformModel, actions)
    assert(mi >= -1e-9, s"mutual information should be non-negative, got $mi")
  }

  test("mutual information equals action entropy when conditional entropy is 0") {
    val h = ActionMetrics.actionEntropy(state, uniformPosterior, DeterministicModel, actions)
    val mi = ActionMetrics.mutualInformation(state, uniformPosterior, DeterministicModel, actions)
    assertEqualsDouble(mi, h, 1e-9)
  }

  test("uniform model yields zero mutual information") {
    val mi = ActionMetrics.mutualInformation(state, uniformPosterior, UniformModel, actions)
    assertEqualsDouble(mi, 0.0, 1e-9)
  }
