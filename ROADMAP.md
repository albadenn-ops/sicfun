# sicfun Implementation Roadmap

This roadmap maps the whitepaper sections to concrete engineering milestones.

## M0 Core math (done)
- [x] 5-of-7 evaluator with total ordering
- [x] Exact/MC equity + EV utilities
- [x] Range parsing + Bayes update scaffolding
- [x] Metrics utilities (entropy, variance)

## M1 Equity tables + traceability (done)
- [x] Heads-up combo table generator
- [x] Canonical (symmetry-reduced) table generator
- [x] Binary IO for tables
- [x] Metadata header (mode/trials/seed/version/coverage)
- [x] Per-entry uncertainty for MC tables
- [x] Coverage report helpers (percent of canonical keys present)
- [x] Decide canonical vs full table as default artifact - canonical chosen (smaller artifact: Int key vs Long key, fewer entries via suit isomorphism; exact for preflop HU equity; full table available as alternative when post-board symmetry doesn't hold)

## M2 ActionModel data contract
- [x] Event schema (actions, bets, positions, timing)
- [x] Feature extraction API (observable-only)
- [x] Dataset layout + provenance metadata

## M3 ActionModel training + calibration
- [x] Multinomial logistic regression baseline
- [x] Calibration metrics + Brier score gates
- [x] Versioning + retirement workflow
- [x] Integrate trained ActionModel into BayesianRange.update() for live posterior inference
- [x] Default holdout split for calibration when no external evaluation set is provided
- [x] Persist/load versioned model artifacts (weights + metadata + lifecycle state)
- [x] CLI training entrypoint for artifact generation from TSV datasets

## M4 Behavioral metrics pipeline
- [x] EV variance with uncertainty
- [x] Action entropy + conditional entropy
- [x] Bayesian collapse metrics (requires M3 integration)
- [x] Player signature vector

## M5 Real-time engine
- [x] Actor/state model (HandState + HandEngine)
- [x] Idempotent event ingestion (sequenceInHand deduplication, out-of-order delivery)
- [x] Snapshot + recovery (HandStateSnapshotIO: state.properties + events.tsv)
- [x] Latency targets (p95 < 1ms per applyEvent for 20-event hands, verified in HandEngineTest)

## M6 Batch analytics
- [x] Multi-shard batch training pipeline (single-process)
- [ ] Distributed training pipeline (planned; deferred while prioritizing M3/M5/M7 deliverables)
- [x] Longitudinal stability tests
- [x] Clustering + fingerprinting

## M7 Signals API + audit trail
- [x] Structured signals payload
- [x] Full reconstruction path (events + model version)
- [x] Exportable audit logs
- [x] CLI batch generation of signals from snapshot + model artifact

## M8 Validation & compliance
- [x] Hand evaluator category distribution checks
- [x] MC convergence tests vs exact
- [x] Operational regression suite
