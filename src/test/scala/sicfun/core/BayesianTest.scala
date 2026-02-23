package sicfun.core

import munit.FunSuite

class BayesianTest extends FunSuite:
  private object ToyModel extends ActionModel[String, String, String]:
    def likelihood(action: String, features: String, hypothesis: String): Double =
      if hypothesis == "good" then
        action match
          case "bet"  => 0.8
          case "fold" => 0.2
          case _      => 0.5
      else
        action match
          case "bet"  => 0.3
          case "fold" => 0.7
          case _      => 0.5

  test("single update shifts probability toward hypothesis consistent with action") {
    val prior = BayesianRange(DiscreteDistribution(Map("good" -> 0.5, "bad" -> 0.5)))
    val (posterior, evidence) = prior.update("bet", "ignored", ToyModel)
    assert(posterior.distribution.probabilityOf("good") > 0.5)
    assert(evidence > 0.0)
  }

  test("update preserves normalization") {
    val prior = BayesianRange(DiscreteDistribution(Map("good" -> 0.5, "bad" -> 0.5)))
    val (posterior, _) = prior.update("bet", "ignored", ToyModel)
    val total = posterior.distribution.weights.values.sum
    assert(math.abs(total - 1.0) < 1e-9)
  }

  test("updateAll applies sequential updates and returns log-evidence") {
    val prior = BayesianRange(DiscreteDistribution(Map("good" -> 0.5, "bad" -> 0.5)))
    val actions = Seq(("bet", "ignored"), ("bet", "ignored"), ("bet", "ignored"))
    val (posterior, logEvidence) = prior.updateAll(actions, ToyModel)
    assert(posterior.distribution.probabilityOf("good") > 0.8)
    assert(logEvidence < 0.0, "log-evidence should be negative (evidence < 1)")
  }

  test("updateAll with empty sequence returns prior unchanged") {
    val prior = BayesianRange(DiscreteDistribution(Map("good" -> 0.5, "bad" -> 0.5)))
    val (posterior, logEvidence) = prior.updateAll(Seq.empty, ToyModel)
    assertEquals(posterior.distribution.probabilityOf("good"), 0.5)
    assertEquals(logEvidence, 0.0)
  }

  test("BayesianRange.uniform creates equal weights") {
    val range = BayesianRange.uniform(Seq("a", "b", "c"))
    val probs = range.distribution.weights.values.toSeq
    probs.foreach(p => assert(math.abs(p - 1.0 / 3.0) < 1e-9))
  }
