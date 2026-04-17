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

## Operational Notes

- Hall capture uses the preserved `capture-hall-baseline-sbt.ps1` helper because
  `scripts/match/run-hall-matchups.ps1` still trips the Java 22 deprecated Security Manager
  path when its helper invokes `sbt package`.
- Baseline hall capture retried one crashed `gto-vs-lag` button-seat JVM leg and then folded
  the successful retry back into the aggregate artifacts.
- The `fa1347e` full-repo audit changed the failing-suite set relative to `048a737`, even
  though none of the failing suites were in `strategic.*`, `engine.*`, `bench.*`, or
  `runtime.*`. That delta is logged in [failures-at-fa1347e.txt](./failures-at-fa1347e.txt)
  and should be treated as a pre-A5 blocking triage item unless explicitly waived.
