package sicfun.holdem.engine

import munit.FunSuite
import sicfun.holdem.types.*

/** Tests for [[RaiseResponseCounts]] and [[ArchetypeLearning]].
  *
  * Validates the Bayesian archetype learning system:
  *   - [[RaiseResponseCounts]]: immutable counter accumulation and invariant enforcement
  *   - [[ArchetypeLearning.blendedRaiseResponse]]: mixture profile computation from posteriors
  *   - [[ArchetypeLearning.updatePosterior]]: single-step Bayesian updates shift mass toward
  *     archetypes whose likelihood profiles best explain the observed action
  *   - [[ArchetypeLearning.posteriorFromCounts]]: batch updates match sequential updates
  *   - Convergence: after many observations of the same action, the MAP estimate converges
  *     to the expected archetype (folds -> Nit, calls -> CallingStation, raises -> Maniac)
  */
class ArchetypeLearningTest extends FunSuite:

  // ---------------------------------------------------------------------------
  // RaiseResponseCounts
  // ---------------------------------------------------------------------------

  test("RaiseResponseCounts default is all zeros") {
    val counts = RaiseResponseCounts()
    assertEquals(counts.folds, 0)
    assertEquals(counts.calls, 0)
    assertEquals(counts.raises, 0)
    assertEquals(counts.total, 0)
  }

  test("RaiseResponseCounts total sums all three fields") {
    val counts = RaiseResponseCounts(folds = 3, calls = 5, raises = 2)
    assertEquals(counts.total, 10)
  }

  test("RaiseResponseCounts rejects negative folds") {
    intercept[IllegalArgumentException] {
      RaiseResponseCounts(folds = -1)
    }
  }

  test("RaiseResponseCounts rejects negative calls") {
    intercept[IllegalArgumentException] {
      RaiseResponseCounts(calls = -1)
    }
  }

  test("RaiseResponseCounts rejects negative raises") {
    intercept[IllegalArgumentException] {
      RaiseResponseCounts(raises = -1)
    }
  }

  test("RaiseResponseCounts observe Fold increments folds") {
    val counts = RaiseResponseCounts().observe(PokerAction.Fold)
    assertEquals(counts.folds, 1)
    assertEquals(counts.calls, 0)
    assertEquals(counts.raises, 0)
  }

  test("RaiseResponseCounts observe Call increments calls") {
    val counts = RaiseResponseCounts().observe(PokerAction.Call)
    assertEquals(counts.folds, 0)
    assertEquals(counts.calls, 1)
    assertEquals(counts.raises, 0)
  }

  test("RaiseResponseCounts observe Raise increments raises") {
    val counts = RaiseResponseCounts().observe(PokerAction.Raise(5.0))
    assertEquals(counts.folds, 0)
    assertEquals(counts.calls, 0)
    assertEquals(counts.raises, 1)
  }

  test("RaiseResponseCounts observe Check is a no-op") {
    val counts = RaiseResponseCounts(folds = 1, calls = 2, raises = 3)
    val after = counts.observe(PokerAction.Check)
    assertEquals(after, counts)
  }

  test("RaiseResponseCounts observe accumulates over multiple actions") {
    var counts = RaiseResponseCounts()
    counts = counts.observe(PokerAction.Fold)
    counts = counts.observe(PokerAction.Fold)
    counts = counts.observe(PokerAction.Call)
    counts = counts.observe(PokerAction.Raise(3.0))
    counts = counts.observe(PokerAction.Raise(6.0))
    counts = counts.observe(PokerAction.Raise(12.0))
    assertEquals(counts.folds, 2)
    assertEquals(counts.calls, 1)
    assertEquals(counts.raises, 3)
    assertEquals(counts.total, 6)
  }

  // ---------------------------------------------------------------------------
  // ArchetypeLearning.blendedRaiseResponse
  // ---------------------------------------------------------------------------

  test("blendedRaiseResponse from uniform prior sums to 1") {
    val profile = ArchetypeLearning.blendedRaiseResponse(ArchetypePosterior.uniform)
    val sum = profile.foldProbability + profile.callProbability + profile.raiseProbability
    assertEqualsDouble(sum, 1.0, 1e-9)
  }

  test("blendedRaiseResponse from uniform prior has positive fold, call, and raise") {
    val profile = ArchetypeLearning.blendedRaiseResponse(ArchetypePosterior.uniform)
    assert(profile.foldProbability > 0.0)
    assert(profile.callProbability > 0.0)
    assert(profile.raiseProbability > 0.0)
  }

  test("blendedRaiseResponse with full Nit posterior matches Nit profile") {
    val nitPosterior = ArchetypePosterior(
      PlayerArchetype.values.map { a =>
        a -> (if a == PlayerArchetype.Nit then 1.0 else 0.0)
      }.toMap
    )
    val profile = ArchetypeLearning.blendedRaiseResponse(nitPosterior)
    // Nit profile: fold=0.68, call=0.28, raise=0.04
    assertEqualsDouble(profile.foldProbability, 0.68, 1e-9)
    assertEqualsDouble(profile.callProbability, 0.28, 1e-9)
    assertEqualsDouble(profile.raiseProbability, 0.04, 1e-9)
  }

  test("blendedRaiseResponse with full Maniac posterior matches Maniac profile") {
    val maniacPosterior = ArchetypePosterior(
      PlayerArchetype.values.map { a =>
        a -> (if a == PlayerArchetype.Maniac then 1.0 else 0.0)
      }.toMap
    )
    val profile = ArchetypeLearning.blendedRaiseResponse(maniacPosterior)
    // Maniac profile: fold=0.25, call=0.30, raise=0.45
    assertEqualsDouble(profile.foldProbability, 0.25, 1e-9)
    assertEqualsDouble(profile.callProbability, 0.30, 1e-9)
    assertEqualsDouble(profile.raiseProbability, 0.45, 1e-9)
  }

  test("blendedRaiseResponse with full CallingStation posterior has highest call probability") {
    val csPosterior = ArchetypePosterior(
      PlayerArchetype.values.map { a =>
        a -> (if a == PlayerArchetype.CallingStation then 1.0 else 0.0)
      }.toMap
    )
    val profile = ArchetypeLearning.blendedRaiseResponse(csPosterior)
    assert(profile.callProbability > profile.foldProbability)
    assert(profile.callProbability > profile.raiseProbability)
  }

  // ---------------------------------------------------------------------------
  // ArchetypeLearning.updatePosterior
  // ---------------------------------------------------------------------------

  test("updatePosterior after a Fold shifts probability toward fold-heavy archetypes") {
    val prior = ArchetypePosterior.uniform
    val updated = ArchetypeLearning.updatePosterior(prior, PokerAction.Fold)
    // Nit has the highest fold likelihood (0.68) so should gain the most mass
    assert(updated.probabilityOf(PlayerArchetype.Nit) > prior.probabilityOf(PlayerArchetype.Nit))
  }

  test("updatePosterior after a Raise shifts probability toward raise-heavy archetypes") {
    val prior = ArchetypePosterior.uniform
    val updated = ArchetypeLearning.updatePosterior(prior, PokerAction.Raise(5.0))
    // Maniac has the highest raise likelihood (0.45) so should gain the most mass
    assert(updated.probabilityOf(PlayerArchetype.Maniac) > prior.probabilityOf(PlayerArchetype.Maniac))
  }

  test("updatePosterior after a Call shifts probability toward call-heavy archetypes") {
    val prior = ArchetypePosterior.uniform
    val updated = ArchetypeLearning.updatePosterior(prior, PokerAction.Call)
    // CallingStation has the highest call likelihood (0.73) so should gain the most mass
    assert(updated.probabilityOf(PlayerArchetype.CallingStation) > prior.probabilityOf(PlayerArchetype.CallingStation))
  }

  test("updatePosterior with Check is a no-op (returns same posterior)") {
    val prior = ArchetypePosterior.uniform
    val updated = ArchetypeLearning.updatePosterior(prior, PokerAction.Check)
    for archetype <- PlayerArchetype.values do
      assertEqualsDouble(
        updated.probabilityOf(archetype),
        prior.probabilityOf(archetype),
        1e-12
      )
  }

  test("updatePosterior preserves normalization") {
    var posterior = ArchetypePosterior.uniform
    posterior = ArchetypeLearning.updatePosterior(posterior, PokerAction.Fold)
    posterior = ArchetypeLearning.updatePosterior(posterior, PokerAction.Call)
    posterior = ArchetypeLearning.updatePosterior(posterior, PokerAction.Raise(3.0))
    val sum = PlayerArchetype.values.map(posterior.probabilityOf).sum
    assertEqualsDouble(sum, 1.0, 1e-9)
  }

  test("updatePosterior converges toward Nit after many folds") {
    var posterior = ArchetypePosterior.uniform
    var i = 0
    while i < 50 do
      posterior = ArchetypeLearning.updatePosterior(posterior, PokerAction.Fold)
      i += 1
    assertEquals(posterior.mapEstimate, PlayerArchetype.Nit)
    assert(posterior.probabilityOf(PlayerArchetype.Nit) > 0.9)
  }

  test("updatePosterior converges toward Maniac after many raises") {
    var posterior = ArchetypePosterior.uniform
    var i = 0
    while i < 50 do
      posterior = ArchetypeLearning.updatePosterior(posterior, PokerAction.Raise(8.0))
      i += 1
    assertEquals(posterior.mapEstimate, PlayerArchetype.Maniac)
    assert(posterior.probabilityOf(PlayerArchetype.Maniac) > 0.9)
  }

  test("updatePosterior converges toward CallingStation after many calls") {
    var posterior = ArchetypePosterior.uniform
    var i = 0
    while i < 50 do
      posterior = ArchetypeLearning.updatePosterior(posterior, PokerAction.Call)
      i += 1
    assertEquals(posterior.mapEstimate, PlayerArchetype.CallingStation)
    assert(posterior.probabilityOf(PlayerArchetype.CallingStation) > 0.9)
  }

  // ---------------------------------------------------------------------------
  // ArchetypeLearning.posteriorFromCounts
  // ---------------------------------------------------------------------------

  test("posteriorFromCounts with zero counts returns the prior") {
    val prior = ArchetypePosterior.uniform
    val posterior = ArchetypeLearning.posteriorFromCounts(RaiseResponseCounts(), prior)
    for archetype <- PlayerArchetype.values do
      assertEqualsDouble(
        posterior.probabilityOf(archetype),
        prior.probabilityOf(archetype),
        1e-12
      )
  }

  test("posteriorFromCounts defaults to uniform prior") {
    val posterior = ArchetypeLearning.posteriorFromCounts(RaiseResponseCounts(folds = 1))
    val sum = PlayerArchetype.values.map(posterior.probabilityOf).sum
    assertEqualsDouble(sum, 1.0, 1e-9)
    // After one fold from uniform, should shift toward Nit
    assert(posterior.probabilityOf(PlayerArchetype.Nit) > posterior.probabilityOf(PlayerArchetype.Maniac))
  }

  test("posteriorFromCounts with heavy fold counts matches sequential fold updates") {
    val counts = RaiseResponseCounts(folds = 20, calls = 0, raises = 0)
    val fromCounts = ArchetypeLearning.posteriorFromCounts(counts)

    var sequential = ArchetypePosterior.uniform
    var i = 0
    while i < 20 do
      sequential = ArchetypeLearning.updatePosterior(sequential, PokerAction.Fold)
      i += 1

    for archetype <- PlayerArchetype.values do
      assertEqualsDouble(
        fromCounts.probabilityOf(archetype),
        sequential.probabilityOf(archetype),
        1e-9
      )
  }

  test("posteriorFromCounts with mixed counts matches sequential mixed updates") {
    val counts = RaiseResponseCounts(folds = 3, calls = 5, raises = 2)
    val fromCounts = ArchetypeLearning.posteriorFromCounts(counts)

    var sequential = ArchetypePosterior.uniform
    var i = 0
    while i < 3 do
      sequential = ArchetypeLearning.updatePosterior(sequential, PokerAction.Fold)
      i += 1
    i = 0
    while i < 5 do
      sequential = ArchetypeLearning.updatePosterior(sequential, PokerAction.Call)
      i += 1
    i = 0
    while i < 2 do
      sequential = ArchetypeLearning.updatePosterior(sequential, PokerAction.Raise(1.0))
      i += 1

    for archetype <- PlayerArchetype.values do
      assertEqualsDouble(
        fromCounts.probabilityOf(archetype),
        sequential.probabilityOf(archetype),
        1e-9
      )
  }

  test("posteriorFromCounts is normalized for large observation counts") {
    val counts = RaiseResponseCounts(folds = 100, calls = 80, raises = 20)
    val posterior = ArchetypeLearning.posteriorFromCounts(counts)
    val sum = PlayerArchetype.values.map(posterior.probabilityOf).sum
    assertEqualsDouble(sum, 1.0, 1e-9)
  }

  test("posteriorFromCounts with only raises converges to Maniac") {
    val counts = RaiseResponseCounts(folds = 0, calls = 0, raises = 30)
    val posterior = ArchetypeLearning.posteriorFromCounts(counts)
    assertEquals(posterior.mapEstimate, PlayerArchetype.Maniac)
  }

  test("posteriorFromCounts with only calls converges to CallingStation") {
    val counts = RaiseResponseCounts(folds = 0, calls = 30, raises = 0)
    val posterior = ArchetypeLearning.posteriorFromCounts(counts)
    assertEquals(posterior.mapEstimate, PlayerArchetype.CallingStation)
  }

  test("posteriorFromCounts with custom prior shifts MAP away from prior given enough evidence") {
    val nitPrior = ArchetypePosterior(
      PlayerArchetype.values.map { a =>
        a -> (if a == PlayerArchetype.Nit then 0.96 else 0.01)
      }.toMap
    )
    // With a strong Nit prior and heavy raise evidence, posterior should eventually shift
    val counts = RaiseResponseCounts(folds = 0, calls = 0, raises = 30)
    val posterior = ArchetypeLearning.posteriorFromCounts(counts, nitPrior)
    // 30 raises should be enough to overcome even a 0.96 Nit prior
    assertEquals(posterior.mapEstimate, PlayerArchetype.Maniac)
  }
