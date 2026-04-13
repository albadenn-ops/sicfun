package sicfun.holdem.strategic

import sicfun.holdem.types.PokerAction

class CertificationTypesTest extends munit.FunSuite:

  test("LocalRobustScreening stores root losses and budget"):
    val cert: CertificationResult.LocalRobustScreening = CertificationResult.LocalRobustScreening(
      rootLosses = Array(0.1, 0.3, 0.2),
      budgetEstimate = 0.6,
      withinTolerance = true
    )
    assertEqualsDouble(cert.budgetEstimate, 0.6, 1e-12)
    assert(cert.withinTolerance)

  test("TabularCertification stores B* and safe actions"):
    val cert: CertificationResult.TabularCertification = CertificationResult.TabularCertification(
      bStar = Array(1.0, 2.0),
      requiredBudget = 2.0,
      safeActionIndices = IndexedSeq(0, 2),
      certificateValid = true,
      withinTolerance = true
    )
    assertEquals(cert.safeActionIndices.size, 2)
    assert(cert.certificateValid)

  test("Unavailable stores reason"):
    val cert: CertificationResult.Unavailable = CertificationResult.Unavailable("solver not loaded")
    assertEquals(cert.reason, "solver not loaded")

  test("DecisionOutcome.Certified wraps action and bundle"):
    val bundle = DecisionEvaluationBundle(
      profileResults = Map.empty,
      robustActionLowerBounds = Array(1.0),
      baselineActionValues = Array(1.0),
      baselineValue = 1.0,
      adversarialRootGap = None,
      pointwiseExploitability = None,
      deploymentExploitability = None,
      certification = CertificationResult.Unavailable("test"),
      chainWorldValues = Map.empty,
      notes = Vector.empty
    )
    val outcome: DecisionOutcome.Certified = DecisionOutcome.Certified(PokerAction.Call, bundle)
    assertEquals(outcome.action, PokerAction.Call)

  test("DecisionOutcome.BaselineFallback wraps action and reason"):
    val outcome: DecisionOutcome.BaselineFallback = DecisionOutcome.BaselineFallback(PokerAction.Fold, "solver error")
    assertEquals(outcome.reason, "solver error")
