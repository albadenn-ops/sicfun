# Superpowers plans and specs — index

`plans/` (32 files) and `specs/` (12 files) accumulate dated planning and design
artifacts produced during sicfun development. They are append-only by convention:
once landed, a plan or spec stays as a snapshot of the thinking at that date.
Don't edit them retroactively except to add a status header.

## Conventions for new files

Every new plan or spec should start with a YAML front-matter block:

```markdown
---
status: draft
date: 2026-04-26
slug: short-name
---

# <human-readable title>
```

`status` values:
- `draft` — being authored, not yet shared.
- `active` — being executed; expect updates.
- `landed` — shipped (or rejected; see `note`).
- `abandoned` — work stopped before completion. Add a `note:` field
  explaining why.

`date` is the authoring date. The filename keeps `YYYY-MM-DD-<slug>.md`
so the directory listing remains chronological without parsing front-matter.

Existing files (pre-2026-04-26) don't carry front-matter; their status is
inferable from git history and the live tech-debt register
([`/TECH_DEBT_AUDIT.md`](../../TECH_DEBT_AUDIT.md)). Back-filling status into
the 44 existing files would require reading each one and the corresponding
code surface. That's a separate slice; the convention is forward-looking.

## Plans (32)

- 2026-03-10 compact-posterior
- 2026-03-10 fixed-point-prob
- 2026-03-10 holdem-subpackage-restructure
- 2026-03-12 batch-cfr-cuda-kernel
- 2026-03-14 profiling-validation-harness
- 2026-03-17 cfr-gto-villain-and-threshold-calibration
- 2026-03-17 showdown-cards-integration
- 2026-03-18 bench-utility-consolidation
- 2026-03-18 decision-pipeline-runner-dedup
- 2026-03-19 playing-hall-gto-extraction
- 2026-03-24 historical-showdown-consumption
- 2026-03-30 adaptive-proof-harness
- 2026-03-30 adaptive-proof-harness-9max
- 2026-04-02 v030-formal-layer-master
- 2026-04-02 v030-phase1-foundation
- 2026-04-02 v030-phase2-tempered-inference
- 2026-04-02 v030-phase3a-pft-dpw
- 2026-04-02 v030-phase3b-wasserstein
- 2026-04-02 v030-phase3c-wpomcp
- 2026-04-02 v030-phase4a-kernels
- 2026-04-02 v030-phase4b-decomposition
- 2026-04-04 strategic-bridge-wiring
- 2026-04-05 strategic-decision-pipeline
- 2026-04-05 v0311-formal-closure
- 2026-04-07 reductionism-resolution
- 2026-04-07 runtime-spec-closure
- 2026-04-09 four-world-solver-closure
- 2026-04-09 v0311-spec-alignment
- 2026-04-13 attributed-baseline-closure
- 2026-04-13 repo-restructure
- 2026-04-13 strategic-overlay
- 2026-04-14 strategic-phase2-track-b

## Specs (12)

Specs precede plans: a spec captures *what* will be built; the corresponding
plan captures *how*. Spec files are paired with plans by date and slug.

- 2026-03-10 compact-posterior-design
- 2026-03-10 fixed-point-prob-design
- 2026-03-14 profiling-validation-harness-design
- 2026-03-30 adaptive-proof-harness-design
- 2026-04-02 sicfun-v026-model-design
- 2026-04-05 strategic-decision-pipeline-design
- 2026-04-07 reductionism-resolution-design
- 2026-04-07 runtime-spec-closure-design
- 2026-04-13 attributed-baseline-design
- 2026-04-13 repo-restructure-design
- 2026-04-13 strategic-overlay-design
- 2026-04-14 strategic-phase2-design

## Refresh procedure

When a plan or spec lands, add a status header to it (don't edit the body
beyond that). When this index drifts from the directory listings:

```sh
ls docs/superpowers/plans/  | sort > /tmp/plans
ls docs/superpowers/specs/  | sort > /tmp/specs
```

and reconcile by hand. Automating the refresh isn't worth a script; the
directory rarely changes, and the index is only useful when it's
human-readable.
