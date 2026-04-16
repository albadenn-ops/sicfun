package sicfun.holdem.bench

class DecisionCorpusBenchmarkTest extends munit.FunSuite:

  test("corpus has at least 6 spots covering required categories") {
    val corpus = DecisionCorpusBenchmark.corpus
    assert(corpus.size >= 6, s"corpus has only ${corpus.size} spots, need >= 6")
    // Required coverage: preflop, postflop flop, postflop turn, postflop river
    val streets = corpus.map(_.state.street).toSet
    assert(streets.contains(sicfun.holdem.types.Street.Preflop), "missing preflop spot")
    assert(streets.contains(sicfun.holdem.types.Street.Flop), "missing flop spot")
    assert(streets.contains(sicfun.holdem.types.Street.Turn), "missing turn spot")
    assert(streets.contains(sicfun.holdem.types.Street.River), "missing river spot")
  }

  test("corpus spots have non-empty candidates") {
    DecisionCorpusBenchmark.corpus.foreach { spot =>
      assert(spot.candidates.nonEmpty, s"spot ${spot.id} has empty candidates")
    }
  }

  test("replay produces deterministic results with fixed seed") {
    val results1 = DecisionCorpusBenchmark.replayAll(seed = 42L)
    val results2 = DecisionCorpusBenchmark.replayAll(seed = 42L)
    assertEquals(results1.size, results2.size)
    results1.zip(results2).foreach { (r1, r2) =>
      assertEquals(r1.spotId, r2.spotId)
      assertEquals(r1.mode, r2.mode)
      assertEquals(r1.selectedAction, r2.selectedAction,
        s"non-deterministic result for spot=${r1.spotId} mode=${r1.mode}")
    }
  }
