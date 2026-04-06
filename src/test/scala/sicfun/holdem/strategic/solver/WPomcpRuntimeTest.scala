package sicfun.holdem.strategic.solver

class WPomcpRuntimeTest extends munit.FunSuite:

  /* --- Config validation tests (pure Scala, no native required) --- */

  test("Config rejects zero simulations"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(numSimulations = 0)

  test("Config rejects discount outside (0,1)"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(discount = 0.0)
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(discount = 1.0)
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(discount = -0.5)

  test("Config rejects negative exploration"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(exploration = -1.0)

  test("Config rejects non-positive rMax"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(rMax = 0.0)

  test("Config rejects zero maxDepth"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(maxDepth = 0)

  test("Config rejects essThreshold outside (0,1]"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(essThreshold = 0.0)
    intercept[IllegalArgumentException]:
      WPomcpRuntime.Config(essThreshold = 1.5)

  test("Config accepts valid parameters"):
    val cfg = WPomcpRuntime.Config(
      numSimulations = 500,
      discount = 0.95,
      exploration = 2.0,
      rMax = 100.0,
      maxDepth = 30,
      essThreshold = 0.3,
      seed = 12345L
    )
    assertEquals(cfg.numSimulations, 500)
    assertEquals(cfg.discount, 0.95)

  /* --- RivalParticles validation tests --- */

  test("RivalParticles rejects mismatched array lengths"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.RivalParticles(
        rivalTypes = Array(0, 1),
        privStates = Array(0),
        weights = Array(0.5, 0.5)
      )

  test("RivalParticles rejects empty arrays"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.RivalParticles(
        rivalTypes = Array.empty,
        privStates = Array.empty,
        weights = Array.empty
      )

  test("RivalParticles accepts valid input"):
    val rp = WPomcpRuntime.RivalParticles(
      rivalTypes = Array(0, 1, 2),
      privStates = Array(5, 6, 7),
      weights = Array(0.3, 0.3, 0.4)
    )
    assertEquals(rp.particleCount, 3)

  /* --- SearchInput validation tests --- */

  test("SearchInput rejects empty rival list"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.SearchInput(
        publicState = WPomcpRuntime.PublicState(0, 100.0),
        rivalParticles = IndexedSeq.empty,
        heroActionCount = 2,
        rivalActionProbs = Array(0.5, 0.5),
        rewards = Array(1.0, 0.0)
      )

  test("SearchInput rejects >8 rivals"):
    val rp = WPomcpRuntime.RivalParticles(Array(0), Array(0), Array(1.0))
    intercept[IllegalArgumentException]:
      WPomcpRuntime.SearchInput(
        publicState = WPomcpRuntime.PublicState(0, 100.0),
        rivalParticles = IndexedSeq.fill(9)(rp),
        heroActionCount = 2,
        rivalActionProbs = Array.fill(18)(1.0 / 2),
        rewards = Array(1.0, 0.0)
      )

  test("SearchInput rejects zero hero actions"):
    val rp = WPomcpRuntime.RivalParticles(Array(0), Array(0), Array(1.0))
    intercept[IllegalArgumentException]:
      WPomcpRuntime.SearchInput(
        publicState = WPomcpRuntime.PublicState(0, 100.0),
        rivalParticles = IndexedSeq(rp),
        heroActionCount = 0,
        rivalActionProbs = Array(1.0),
        rewards = Array.empty
      )

  test("SearchInput rejects mismatched rewards length"):
    val rp = WPomcpRuntime.RivalParticles(Array(0), Array(0), Array(1.0))
    intercept[IllegalArgumentException]:
      WPomcpRuntime.SearchInput(
        publicState = WPomcpRuntime.PublicState(0, 100.0),
        rivalParticles = IndexedSeq(rp),
        heroActionCount = 2,
        rivalActionProbs = Array(0.5, 0.5),
        rewards = Array(1.0)  /* length 1, expected 2 */
      )

  /* --- Native tests (require DLL) --- */

  private def nativeAvailable: Boolean = WPomcpRuntime.isAvailable

  test("native: isAvailable reports status"):
    /* This test always passes -- it just documents whether native is loaded. */
    val available = WPomcpRuntime.isAvailable
    if !available then
      println("[WPomcpRuntimeTest] Native library not available, skipping native tests")

  test("native: solve with dominant action returns correct best action"):
    assume(nativeAvailable, "Native library not available")
    val rp = WPomcpRuntime.RivalParticles(
      rivalTypes = Array(0, 0, 0, 0, 0),
      privStates = Array(0, 1, 2, 3, 4),
      weights = Array(0.2, 0.2, 0.2, 0.2, 0.2)
    )
    val input = WPomcpRuntime.SearchInput(
      publicState = WPomcpRuntime.PublicState(street = 0, pot = 100.0),
      rivalParticles = IndexedSeq(rp),
      heroActionCount = 2,
      rivalActionProbs = Array(0.5, 0.5),
      rewards = Array(1.0, 0.0)  /* action 0 dominates */
    )
    val config = WPomcpRuntime.Config(numSimulations = 200, seed = 42L)

    val result = WPomcpRuntime.solve(input, config)
    assert(result.isRight, s"Expected Right, got $result")
    val sr = result.toOption.get
    assertEquals(sr.bestAction, 0, "Action 0 should dominate")
    assert(sr.actionValues(0) > sr.actionValues(1),
      s"Action 0 value ${sr.actionValues(0)} should exceed action 1 value ${sr.actionValues(1)}")

  test("native: solve with equal rewards returns valid action"):
    assume(nativeAvailable, "Native library not available")
    val rp = WPomcpRuntime.RivalParticles(
      rivalTypes = Array(0, 0, 0),
      privStates = Array(0, 1, 2),
      weights = Array(1.0 / 3, 1.0 / 3, 1.0 / 3)
    )
    val input = WPomcpRuntime.SearchInput(
      publicState = WPomcpRuntime.PublicState(0, 50.0),
      rivalParticles = IndexedSeq(rp),
      heroActionCount = 3,
      rivalActionProbs = Array(1.0 / 3, 1.0 / 3, 1.0 / 3),
      rewards = Array(1.0, 1.0, 1.0)
    )
    val config = WPomcpRuntime.Config(numSimulations = 100, seed = 7L)

    val result = WPomcpRuntime.solve(input, config)
    assert(result.isRight, s"Expected Right, got $result")
    val sr = result.toOption.get
    assert(sr.bestAction >= 0 && sr.bestAction < 3,
      s"Best action ${sr.bestAction} out of range [0,3)")

  test("native: multiway 3-rival solve"):
    assume(nativeAvailable, "Native library not available")
    val rp1 = WPomcpRuntime.RivalParticles(Array(0, 1), Array(0, 1), Array(0.5, 0.5))
    val rp2 = WPomcpRuntime.RivalParticles(Array(0, 1), Array(2, 3), Array(0.5, 0.5))
    val rp3 = WPomcpRuntime.RivalParticles(Array(0, 1), Array(4, 5), Array(0.5, 0.5))
    val input = WPomcpRuntime.SearchInput(
      publicState = WPomcpRuntime.PublicState(1, 200.0),
      rivalParticles = IndexedSeq(rp1, rp2, rp3),
      heroActionCount = 2,
      rivalActionProbs = Array.fill(6)(0.5),
      rewards = Array(2.0, 0.5)  /* action 0 dominates */
    )
    val config = WPomcpRuntime.Config(numSimulations = 300, seed = 99L)

    val result = WPomcpRuntime.solve(input, config)
    assert(result.isRight, s"Expected Right, got $result")
    val sr = result.toOption.get
    assertEquals(sr.bestAction, 0)
    assert(sr.actionValues.length == 2)

  test("native: |R|=1 heads-up degenerate case"):
    assume(nativeAvailable, "Native library not available")
    val rp = WPomcpRuntime.RivalParticles(Array(0), Array(0), Array(1.0))
    val input = WPomcpRuntime.SearchInput(
      publicState = WPomcpRuntime.PublicState(0, 50.0),
      rivalParticles = IndexedSeq(rp),
      heroActionCount = 1,
      rivalActionProbs = Array(1.0),
      rewards = Array(0.5)
    )
    val config = WPomcpRuntime.Config(numSimulations = 10, seed = 7L)

    val result = WPomcpRuntime.solve(input, config)
    assert(result.isRight, s"Expected Right, got $result")
    val sr = result.toOption.get
    assertEquals(sr.bestAction, 0)

  test("native: deterministic seed produces reproducible results"):
    assume(nativeAvailable, "Native library not available")
    val rp = WPomcpRuntime.RivalParticles(
      rivalTypes = Array(0, 1, 2, 0, 1),
      privStates = Array(0, 1, 2, 3, 4),
      weights = Array(0.2, 0.2, 0.2, 0.2, 0.2)
    )
    val input = WPomcpRuntime.SearchInput(
      publicState = WPomcpRuntime.PublicState(0, 100.0),
      rivalParticles = IndexedSeq(rp),
      heroActionCount = 3,
      rivalActionProbs = Array(0.4, 0.3, 0.3),
      rewards = Array(1.0, 0.5, 0.8)
    )
    val config = WPomcpRuntime.Config(numSimulations = 100, seed = 42L)

    val r1 = WPomcpRuntime.solve(input, config)
    val r2 = WPomcpRuntime.solve(input, config)
    assert(r1.isRight && r2.isRight)
    val sr1 = r1.toOption.get
    val sr2 = r2.toOption.get
    assertEquals(sr1.bestAction, sr2.bestAction,
      "Same seed must produce same best action")
    for i <- sr1.actionValues.indices do
      assertEqualsDouble(sr1.actionValues(i), sr2.actionValues(i), 1e-15,
        s"Action value $i differs across runs with same seed")

  test("native: library unavailable returns Left"):
    /* This test validates the error path when native is not loaded.
     * It is meaningful only in environments without the DLL. */
    if !nativeAvailable then
      val rp = WPomcpRuntime.RivalParticles(Array(0), Array(0), Array(1.0))
      val input = WPomcpRuntime.SearchInput(
        publicState = WPomcpRuntime.PublicState(0, 50.0),
        rivalParticles = IndexedSeq(rp),
        heroActionCount = 1,
        rivalActionProbs = Array(1.0),
        rewards = Array(0.5)
      )
      val config = WPomcpRuntime.Config()
      val result = WPomcpRuntime.solve(input, config)
      assert(result.isLeft, "Expected Left when native unavailable")

  test("native: more particles reduce action value variance"):
    assume(nativeAvailable, "Native library not available")

    def runWithParticles(n: Int, seed: Long): Array[Double] =
      val rp = WPomcpRuntime.RivalParticles(
        rivalTypes = Array.fill(n)(0),
        privStates = Array.tabulate(n)(identity),
        weights = Array.fill(n)(1.0 / n)
      )
      val input = WPomcpRuntime.SearchInput(
        publicState = WPomcpRuntime.PublicState(0, 100.0),
        rivalParticles = IndexedSeq(rp),
        heroActionCount = 2,
        rivalActionProbs = Array(0.6, 0.4),
        rewards = Array(1.0, 0.5)
      )
      val config = WPomcpRuntime.Config(numSimulations = 200, seed = seed)
      WPomcpRuntime.solve(input, config).toOption.get.actionValues

    /* Run multiple seeds and measure variance. */
    val seeds = (1L to 10L)
    val variance5 = {
      val runs = seeds.map(s => runWithParticles(5, s).head)
      val mean = runs.sum / runs.size
      runs.map(v => (v - mean) * (v - mean)).sum / runs.size
    }
    val variance50 = {
      val runs = seeds.map(s => runWithParticles(50, s).head)
      val mean = runs.sum / runs.size
      runs.map(v => (v - mean) * (v - mean)).sum / runs.size
    }
    /* More particles should reduce variance (or at least not increase it much). */
    assert(variance50 <= variance5 * 2.0,
      s"50-particle variance $variance50 should not greatly exceed 5-particle variance $variance5")

  test("native: output values bounded by R_max / (1-gamma)"):
    assume(nativeAvailable, "Native library not available")
    val rMax = 10.0
    val gamma = 0.9
    val bound = rMax / (1.0 - gamma)  /* = 100 */
    val rp = WPomcpRuntime.RivalParticles(
      rivalTypes = Array(0, 1, 2),
      privStates = Array(0, 1, 2),
      weights = Array(1.0 / 3, 1.0 / 3, 1.0 / 3)
    )
    val input = WPomcpRuntime.SearchInput(
      publicState = WPomcpRuntime.PublicState(0, 100.0),
      rivalParticles = IndexedSeq(rp),
      heroActionCount = 2,
      rivalActionProbs = Array(0.5, 0.5),
      rewards = Array(rMax, -rMax)
    )
    val config = WPomcpRuntime.Config(
      numSimulations = 500, rMax = rMax, discount = gamma, seed = 42L
    )
    val result = WPomcpRuntime.solve(input, config)
    assert(result.isRight)
    for v <- result.toOption.get.actionValues do
      assert(math.abs(v) <= bound * 1.01,
        s"Action value $v exceeds bound $bound")

  /* --- FactoredModel validation tests --- */

  test("FactoredModel rejects mismatched rival policy size"):
    intercept[IllegalArgumentException]:
      WPomcpRuntime.FactoredModel(
        rivalPolicy = Array(0.5, 0.5),  /* should be 4 * 2 * 2 = 16 */
        numRivalTypes = 4,
        numPubStates = 2,
        actionEffects = Array.fill(6)(0.0),
        showdownEquity = Array.fill(100)(0.5),
        numHeroBuckets = 10,
        numRivalBuckets = 10,
        terminalFlags = Array.fill(4)(0),
        potBucketSize = 50.0
      )

  test("FactoredModel accepts valid dimensions"):
    val nTypes = 4; val nPub = 2; val nAct = 2
    val model = WPomcpRuntime.FactoredModel(
      rivalPolicy = Array.fill(nTypes * nPub * nAct)(1.0 / nAct),
      numRivalTypes = nTypes,
      numPubStates = nPub,
      actionEffects = Array.fill(nAct * 3)(0.0),
      showdownEquity = Array.fill(10 * 10)(0.5),
      numHeroBuckets = 10,
      numRivalBuckets = 10,
      terminalFlags = Array.fill(nPub * nAct)(0),
      potBucketSize = 50.0
    )
    assertEquals(model.numRivalTypes, 4)

  test("native: solveV2 with dominant action returns correct best action"):
    assume(nativeAvailable, "Native library not available")
    val rp = WPomcpRuntime.RivalParticles(
      rivalTypes = Array(0, 1, 2, 3),
      privStates = Array(0, 1, 2, 3),
      weights = Array(0.25, 0.25, 0.25, 0.25)
    )
    val nTypes = 4; val nPub = 4; val nAct = 3
    val policy = Array.fill(nTypes * nPub * nAct)(0.0)
    for pub <- 0 until nPub do
      policy(0 * nPub * nAct + pub * nAct + 0) = 1.0
      for t <- 1 until nTypes do
        policy(t * nPub * nAct + pub * nAct + 1) = 1.0
    val effects = Array(
      0.0, 1.0, 0.0,   /* fold */
      0.5, 0.0, 0.0,   /* call */
      1.0, 0.0, 0.0    /* raise */
    )
    val equity = Array.tabulate(10 * 10)((idx) =>
      val hb = idx / 10; val rb = idx % 10
      if hb > rb then 0.8 else if hb == rb then 0.5 else 0.2
    )
    val terminal = Array.fill(nPub * nAct)(0)
    for a <- 0 until nAct do terminal(3 * nAct + a) = 3

    val model = WPomcpRuntime.FactoredModel(
      rivalPolicy = policy, numRivalTypes = nTypes, numPubStates = nPub,
      actionEffects = effects, showdownEquity = equity,
      numHeroBuckets = 10, numRivalBuckets = 10,
      terminalFlags = terminal, potBucketSize = 50.0
    )
    val input = WPomcpRuntime.SearchInputV2(
      publicState = WPomcpRuntime.PublicState(0, 100.0),
      rivalParticles = IndexedSeq(rp),
      model = model,
      heroBucket = 7
    )
    val config = WPomcpRuntime.Config(numSimulations = 500, seed = 42L)
    val result = WPomcpRuntime.solveV2(input, config)
    assert(result.isRight, s"Expected Right, got $result")
    val sr = result.toOption.get
    assert(sr.actionValues.length == 3)
