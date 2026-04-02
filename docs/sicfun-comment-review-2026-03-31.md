# SICFUN Comment/Doc Review - 2026-03-31

## Scope

Reviewed and improved comments/docstrings in core runtime and decision-engine paths where ambiguity risk is highest:

- `src/main/scala/sicfun/core/Prob.scala`
- `src/main/scala/sicfun/holdem/types/HandEngine.scala`
- `src/main/scala/sicfun/holdem/engine/HeroDecisionPipeline.scala`
- `src/main/scala/sicfun/holdem/runtime/AlwaysOnDecisionLoop.scala`
- `src/main/scala/sicfun/holdem/engine/RangeInferenceEngine.scala`
- `src/main/scala/sicfun/holdem/engine/RealTimeAdaptiveEngine.scala`

Second full-pass extension (same date) covered all production files without Scaladoc and additional usage/doc coherence fixes in runtime/cfr:

- `src/main/scala/sicfun/core/BenchmarkHandEvaluator.scala`
- `src/main/scala/sicfun/core/DiscreteDistribution.scala`
- `src/main/scala/sicfun/core/HandEvaluatorValidation.scala`
- `src/main/scala/sicfun/core/KMeans.scala`
- `src/main/scala/sicfun/core/Metrics.scala`
- `src/main/scala/sicfun/holdem/engine/ShowdownPriorBias.scala`
- `src/main/scala/sicfun/holdem/equity/RangeParser.scala`
- `src/main/scala/sicfun/holdem/history/HandHistoryImport.scala`
- `src/main/scala/sicfun/holdem/history/OpponentIdentity.scala`
- `src/main/scala/sicfun/holdem/history/OpponentMemoryTarget.scala`
- `src/main/scala/sicfun/holdem/history/OpponentProfileStore.scala`
- `src/main/scala/sicfun/holdem/history/OpponentProfileStorePersistence.scala`
- `src/main/scala/sicfun/holdem/model/PokerActionTrainingDataIO.scala`
- `src/main/scala/sicfun/holdem/model/TrainPokerActionModel.scala`
- `src/main/scala/sicfun/holdem/runtime/AcpcHeadsUpDealer.scala`
- `src/main/scala/sicfun/holdem/runtime/AcpcMatchRunner.scala`
- `src/main/scala/sicfun/holdem/runtime/SlumbotMatchRunner.scala`
- `src/main/scala/sicfun/holdem/web/HandHistoryReviewServer.scala`
- `src/main/scala/sicfun/holdem/web/HandHistoryReviewService.scala`
- `src/main/scala/sicfun/holdem/web/PlatformUserAuth.scala`
- `src/main/scala/sicfun/holdem/runtime/LiveHandSimulator.scala`
- `src/main/scala/sicfun/holdem/runtime/HandHistoryAnalyzer.scala`
- `src/main/scala/sicfun/holdem/runtime/PokerAdvisor.scala`
- `src/main/scala/sicfun/holdem/cfr/HoldemCfrReport.scala`
- `src/main/scala/sicfun/holdem/validation/ValidationRunner.scala`

## Corrections Made

1. Fixed incorrect usage FQCN in `AlwaysOnDecisionLoop` help text.
2. Clarified idempotency semantics in `HandEngine.applyEvents`:
   exact duplicates are skipped, conflicting duplicates throw.
3. Rewrote and expanded `Prob` contract docs:
   fixed encoding artifact, clarified no-clamp behavior and caller responsibilities.
4. Clarified units and conversion boundaries in `HeroDecisionPipeline`:
   chips-in/chips-out context vs BB action representation.
5. Added explicit behavior notes for adaptive vs GTO decision paths
   (latency-bounded runtime path vs quality-oriented CFR path).
6. Documented policy sampling fallback behavior when probabilities are partial/residual.
7. Added operational lifecycle docs to `AlwaysOnDecisionLoop.LoopRunner` and key methods.
8. Expanded DDRE decision-mode semantics and fail/degrade behavior in `RangeInferenceEngine`.
9. Expanded compact posterior truncation retention guarantees (`maxHands/minHands/minMass`).
10. Added explicit explanation of response-aware raise EV approximation model.
11. Expanded `RealTimeAdaptiveEngine` docs for low-latency shortcuts, cache-key intent,
    equity trial scaling, and CFR trust/guardrail gates.
12. Added missing object/type-level Scaladoc to every `src/main` `sicfun` file that had none.
13. Corrected additional `runMain` doc paths to fully-qualified package names:
    - `sicfun.holdem.runtime.LiveHandSimulator`
    - `sicfun.holdem.runtime.HandHistoryAnalyzer`
    - `sicfun.holdem.runtime.PokerAdvisor`
    - `sicfun.holdem.cfr.HoldemCfrReport`
14. Reworded deferred-work comment in `ValidationRunner` to remove ambiguous TODO phrasing.

## Inconsistency/Ambiguity/Omission Pass

- Inconsistency fixed:
  - `runMain sicfun.holdem.AlwaysOnDecisionLoop` -> `runMain sicfun.holdem.runtime.AlwaysOnDecisionLoop`.
- Ambiguity reduced:
  - chip vs BB units in raise candidate construction.
  - DDRE mode semantics (Off/Shadow/Blend*) and fallback behavior.
  - baseline blending trust logic and guardrail behavior.
- Omission reduced:
  - lifecycle and intent comments in always-on service runner.
  - explicit contract for probabilistic sampling fallback and fixed-point conversion.

## Remaining Work (for full repo-wide pass)

Production-code pass status: complete for `src/main` (`scala` + `java`) under `sicfun`:
- no production file remains without at least one Scaladoc block
- high-risk runtime/engine ambiguity fixes applied
- known incorrect `runMain` references corrected in touched runtime/cfr entrypoints

Test-code note:
- this pass prioritized production/runtime behavior documentation and did not normalize every `src/test` file header comment.
