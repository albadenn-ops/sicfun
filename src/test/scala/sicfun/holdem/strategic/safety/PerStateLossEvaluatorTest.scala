package sicfun.holdem.strategic

import sicfun.holdem.strategic.solver.TabularGenerativeModel

class PerStateLossEvaluatorTest extends munit.FunSuite:

  private inline val Tol = 1e-10

  // Simple 2-state, 2-action model
  private def twoStateModel(rewards: Array[Double]): TabularGenerativeModel =
    TabularGenerativeModel(
      transitionTable = Array(0, 1, 1, 0), // s0,a0->s0; s0,a1->s1; s1,a0->s1; s1,a1->s0
      obsLikelihood = Array.fill(2 * 2 * 1)(1.0),
      rewardTable = rewards,
      numStates = 2,
      numActions = 2,
      numObs = 1
    )

  test("valueIteration converges for reference policy"):
    val model = twoStateModel(Array(1.0, 0.5, 0.3, 0.8))
    val gamma = 0.5
    // Reference policy: always take action 0
    val refPolicy: Int => Int = _ => 0
    val values = PerStateLossEvaluator.valueIteration(model, refPolicy, gamma)
    assertEquals(values.length, 2)
    for v <- values do assert(v.isFinite)

  test("valueIteration fixed point for self-loop"):
    // s0 takes a0 -> stays at s0, reward=2.0, gamma=0.5
    // V(s0) = 2.0 + 0.5 * V(s0) => V(s0) = 4.0
    val model = TabularGenerativeModel(
      transitionTable = Array(0, 0),
      obsLikelihood = Array(1.0, 1.0), // numStates * numActions * numObs = 1 * 2 * 1 = 2
      rewardTable = Array(2.0, 1.0),
      numStates = 1,
      numActions = 2,
      numObs = 1
    )
    val values = PerStateLossEvaluator.valueIteration(model, _ => 0, 0.5)
    assertEqualsDouble(values(0), 4.0, 1e-8)

  test("computeRobustLosses produces non-negative losses"):
    val model = twoStateModel(Array(1.0, 0.5, 0.3, 0.8))
    val gamma = 0.5
    val refPolicy: Int => Int = _ => 0
    val profileModels = Vector(model) // 1 profile, same model
    val losses = PerStateLossEvaluator.computeRobustLosses(profileModels, refPolicy, gamma)
    assertEquals(losses.length, 2)
    for row <- losses; v <- row do
      assert(v >= -Tol, s"robust loss should be non-negative, got $v")

  test("computeRobustLosses zero when policy is optimal"):
    // Single state, single action: policy always picks the only action
    // Loss = max(0, V(s) - R(s,a) - gamma*V(T(s,a))) = max(0, V(s) - R(s,a) - gamma*V(s))
    //       = max(0, 0) = 0  [by definition of V^pi]
    val model = TabularGenerativeModel(
      transitionTable = Array(0),
      obsLikelihood = Array(1.0),
      rewardTable = Array(3.0),
      numStates = 1,
      numActions = 1,
      numObs = 1
    )
    val losses = PerStateLossEvaluator.computeRobustLosses(Vector(model), _ => 0, 0.5)
    assertEqualsDouble(losses(0)(0), 0.0, Tol)

  test("computeRobustLosses captures cross-profile loss"):
    // Two profiles with different rewards for action 1
    // Profile 0: R(s0,a1)=0.5
    // Profile 1: R(s0,a1)=0.0 (worse for a1)
    // Reference policy: always action 0
    val model0 = twoStateModel(Array(1.0, 0.5, 0.3, 0.8))
    val model1 = twoStateModel(Array(1.0, 0.0, 0.3, 0.8))
    val losses = PerStateLossEvaluator.computeRobustLosses(Vector(model0, model1), _ => 0, 0.5)
    // Loss for action 1 at state 0 should be larger under profile 1
    assert(losses(0)(1) > 0.0, "cross-profile loss should be positive")
