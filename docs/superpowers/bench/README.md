# Phase 2 Bench Manifest

This index preserves the provenance of the 2026-04-17 Phase 2 benchmark and audit runs.
The raw run artifacts live in the primary worktree under `data/` and are currently untracked.
This file records the commit, branch, command shape, and gate outcome for each run so the
evidence remains auditable in git history.

Artifact path convention:
- Paths below are written as repo-relative paths from the primary worktree root
  `C:/Users/alexl/code/math/untitled`.
- The clean A3 and A4 benchmarking worktrees wrote their outputs back into that primary
  worktree rather than into their own checkout directories.

| Run | Date | Commit | Branch or Ref | Seed | Hall Hands | Focused Validation | Full Repo Audit | Gate Summary | Artifact Path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | `2026-04-17-1519` | `fcbc3a8` | `detached` | `42` | `1000` | not run | not run | G1 baseline captured for decision corpus, hall, and Slumbot probe | `data/phase2-baseline/2026-04-17-1519` |
| A3 rerun | `2026-04-17-1601` | `048a737` | `bench/a3-rerun-20260417-160046` | `42` | `1000` | `120 passed, 0 failed` | `1803 passed, 10 failed, 1 ignored` | G2-G5 passed as overlay-path non-regression; grounded PFT path still covered by focused formulation tests rather than Track B | `data/phase2-a3/2026-04-17-1601` |
| A4 rerun | `2026-04-17-1740` | `f26cbe3` code, `fa1347e` docs and hygiene | `bench/a4-grounding-20260417-173122` | `42` | `1000` | `121 passed, 0 failed` | `1806 passed, 8 failed, 1 ignored` at `fa1347e`; see [failures-at-fa1347e.txt](./failures-at-fa1347e.txt) | A4 gate met for the `FormulationInput` / `StrategicEngine`-driven WPomcp path; Track B remained a non-regression probe for overlay-backed runtime surfaces | `data/phase2-a4/2026-04-17-1740` |
| A5 Slumbot focus | `2026-04-17-2037` | `9c92419` | `bench/a5-certification-20260417-192934` | `42` | `n/a` | `888 passed, 0 failed` | not rerun | Narrow A5 end-to-end Slumbot probe for adaptive, GTO, and strategic modes; records stable strategic mean latency, unchanged `overlayChangeRate=0.0%` relative to A4 in this probe, and the native-CPU CFR smoke used for the GTO check | `data/phase2-a5/2026-04-17-2037-slumbot-focus` |

## Gate Notes

- Baseline capture used `seed=42`, `1000` hands per hall matchup, and `50` Slumbot hands per
  mode. Slumbot `bb/100` at that sample size is noise and should be treated as a protocol and
  latency probe only.
- A3 and A4 both matched baseline exactly on the deterministic surfaces:
  decision corpus `12/12`, hall self-play `18/18`, and overlay change / veto rates `0.0%`.
- By local code inspection, `DecisionCorpusBenchmark`, `SlumbotMatchRunner`, and
  `TexasHoldemPlayingHall` stay on the Phase 1 overlay path. The A3 and A4 Track B reruns
  therefore show that formulation changes did not bleed into runtime behavior; they do not
  directly benchmark the grounded formulation path itself.

## A5 Slumbot Focus

- Commit `9c92419` on `bench/a5-certification-20260417-192934` added a narrow Slumbot-only
  verification run under `data/phase2-a5/2026-04-17-2037-slumbot-focus/`.
- This artifact records that all three runtime modes completed end-to-end against
  Slumbot at the A5 tip, that strategic `meanLatencyMs` remained in-family
  (`31.045 ms` vs. `34.938 ms` at A4 and `30.905 ms` at A3), and that strategic
  `overlayChangeRate` remained `0.0%`, matching the A4 Slumbot probe. Treat that as
  bounded runtime-behavior evidence rather than a formal latency non-regression gate.
- The GTO Slumbot run was configured with `sicfun.cfr.provider=native-cpu-fixed`.
  A separate 1-hand verbose smoke recorded at
  `data/phase2-a5/2026-04-17-2037-slumbot-focus/slumbot-gto-verbose-smoke.log`
  showed native runtime activity (`postflop native auto-engine: routing to CPU for small workload`)
  and no `unavailable`, `fallback`, or `scala` lines. That is positive evidence of native-CPU
  execution for the smoke, not a proof that every solve in the 50-hand run used the native path.
- This artifact does not certify EV direction, quality ranking, or statistical significance.
  At `n=50`, cross-run and cross-mode `bb/100` deltas mix sampling noise with server-side
  dealer non-determinism and should be treated as protocol/behavior probes only.
- Engine-path reminder for future readers:
  `SlumbotMatchRunner` dispatches `strategic` to `decideHeroStrategic` and other modes to
  `decideHero`; `HeroDecisionPipeline` routes `adaptive` through the adaptive engine,
  `strategic` through adaptive-upstream plus overlay, and `gto` through `HoldemCfrSolver`.
  Only `gto` can reach the Scala/native CFR provider surface in Slumbot runs.

## Audit Policy

- For Track A work, the authoritative semantic signals are the focused formulation slice and
  isolated reruns of any suite that newly appears in the full-repo audit.
- Full `sbt test` pass counts are still useful repo-health telemetry, but on this branch they
  are treated as informational because the suite contains load-sensitive timeout churn.
- Proceeding into A5 therefore means accepting the known full-suite instability as separate
  repo-health debt unless a newly appearing failure reproduces in isolation as an A-track
  semantic regression.
- A5 uses `27e72d4` as its pre-change audit point. Subsequent A5 sub-slices should compare
  focused-slice and isolated-rerun behavior against that tip rather than against the earlier
  `fa1347e` full-suite run.
- Residual non-grounded surfaces after the current A5 code slices are explicit carve-outs:
  synthetic `StrategicSnapshot.build` / `ValidationRunner` reporting, legacy raw-parameter
  `PokerPftFormulation` helpers, `LegacyToyFormulationInput`, and deferred A4 items 2 and 3.
- The grounded certification path remains certification/offline-only. Runtime rollout is still
  a separate Phase 3 decision.

## Operational Notes

- Hall capture uses the preserved `capture-hall-baseline-sbt.ps1` helper because
  `scripts/match/run-hall-matchups.ps1` still trips the Java 22 deprecated Security Manager
  path when its helper invokes `sbt package`.
- Baseline hall capture retried one crashed `gto-vs-lag` button-seat JVM leg and then folded
  the successful retry back into the aggregate artifacts.
- The `fa1347e` full-repo audit changed the failing-suite set relative to `048a737`, even
  though none of the failing suites were in `strategic.*`, `engine.*`, `bench.*`, or
  `runtime.*`. That delta is logged in [failures-at-fa1347e.txt](./failures-at-fa1347e.txt).
- Follow-up isolated reruns at `0c15676` showed that
  `HoldemCfrExternalValidationParityTest` passes at `fcbc3a8`, `048a737`, and the A4 branch
  tip, while `HandHistoryReviewServerTest` passes in isolation at the A4 branch tip. The new
  `fa1347e` delta is therefore treated as non-reproducing in isolation and likely
  load-sensitive timeout churn rather than direct A3/A4 semantic evidence.
- The dirty `src/main/native/build/*.dll` and `*.exp` outputs left behind by `sbt test` are
  not a `.gitignore` miss. `src/main/native/build/` is already ignored, but those binaries are
  tracked in git today, so test runs rewrite tracked files. Cleaning that up requires
  de-tracking or relocating the generated native outputs rather than another ignore rule.
- On `bench/a5-certification-20260417-192934`, commit `27e72d4` proved the cold regeneration
  path by passing `sbt "clean; compile"` after the native build outputs were removed from the
  index. `src/main/native/build/` is now ignored rather than tracked on that branch line.
