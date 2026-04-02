package sicfun.holdem.validation

import munit.FunSuite

class ValidationRunnerTest extends FunSuite:

  test("gto-baseline control uses CFR strategy"):
    val strategy = ValidationRunner.villainStrategyFor(NoLeak())
    assert(strategy.isInstanceOf[CfrVillainStrategy],
      s"expected CfrVillainStrategy for gto-baseline control, got ${strategy.getClass.getSimpleName}")
    assert(!strategy.asInstanceOf[CfrVillainStrategy].allowsHeuristicFallback,
      "gto-baseline control must fail fast rather than silently falling back to heuristic play")

  test("leak-injected players keep the fast heuristic strategy"):
    val strategy = ValidationRunner.villainStrategyFor(Overcalls(0.9))
    assert(strategy.isInstanceOf[EquityBasedStrategy],
      s"expected EquityBasedStrategy for leak players, got ${strategy.getClass.getSimpleName}")
