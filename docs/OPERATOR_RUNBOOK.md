# SICFUN Operator Runbook

This is the primary day-to-day operations guide for running, validating, and stress-testing SICFUN.

Scope note:
- This runbook covers simulator, benchmark, and research-harness workflows.
- It does not imply live table integration or production deployment.

Optional interactive launcher (menu for top 5 runbook actions):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/runbook.ps1
```

One-shot launcher mode:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/runbook.ps1 -Action quick-proof
```

Dry run preview (prints command without executing):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/runbook.ps1 -Action hall-max-autotune -WhatIf
```

## 1. Daily Start

Quick health check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1 -Quick
```

Full validation sweep:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1
```

## 2. Main Workload: Playing Hall

Single-process hall run (good for functional checks and controlled experiments):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall.ps1 `
  -Hands 1000000 `
  -TableCount 8 `
  -ReportEvery 50000 `
  -LearnEveryHands 0 `
  -SaveTrainingTsv false `
  -SaveDdreTrainingTsv false `
  -OutDir data/bench-hall-single
```

Maximum hardware saturation (recommended for long stress runs):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall-max.ps1 `
  -Hands 100000000 `
  -Workers 0 `
  -TableCountPerWorker 8 `
  -NativeProfile gpu `
  -ReportEvery 500000 `
  -LearnEveryHands 0 `
  -SaveTrainingTsv false `
  -SaveDdreTrainingTsv false `
  -JvmOption "-Xms2g" "-Xmx2g" `
  -OutDir data/bench-hall-max
```

Auto-tuned hardware run (recommended default when machine load is variable):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall-max.ps1 `
  -AutoTune `
  -AutoTuneHands 500000 `
  -AutoTuneProfiles auto,cpu,gpu `
  -AutoTuneWorkerCandidates 8,12,16,20,24 `
  -Hands 100000000 `
  -TableCountPerWorker 8 `
  -ReportEvery 500000 `
  -LearnEveryHands 0 `
  -SaveTrainingTsv false `
  -SaveDdreTrainingTsv false `
  -OutDir data/bench-hall-max
```

## 2A. Benchmark Control Notes

- As of March 10, 2026, the default long-run heads-up range autotune cache on this machine is `data/headsup-range-autotune.properties` with `block=64`, `chunkHeroes=2048`, `memoryPath=readonly`.
- Use `5000` or `10000` hands for exact-mode throughput comparisons. `1000`-hand runs were highly variable and often understated the long-run throughput of the retuned range kernel.
- Current long-run reference on this machine for `-Workers 1 -TableCountPerWorker 1 -HeroStyle adaptive -GtoMode exact -VillainStyle gto -BunchingTrials 80 -EquityTrials 700` is about `127.99-130.59 hands/s` at `5000` hands and `129.27 hands/s` at `10000` hands.
- Exact-GTO cache hit rate did not materially change in these longer runs (`~18.7%` to `19.1%`), so the improvement came from the native range path, not higher exact-solve cache reuse.
- If you need short-run smoke/control comparability, override the range autotune cache with `data/headsup-range-autotune.shortrun-prev.properties`.

Short-run cache override example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall-max.ps1 `
  -Hands 1000 `
  -Workers 1 `
  -TableCountPerWorker 1 `
  -NativeProfile auto `
  -ReportEvery 1000 `
  -LearnEveryHands 0 `
  -SaveTrainingTsv false `
  -SaveDdreTrainingTsv false `
  -JvmOption "-Dsicfun.gpu.range.autotune.cachePath=data/headsup-range-autotune.shortrun-prev.properties" `
  -OutDir data/bench-hall-short-control
```

## 3. Hall Output Locations

`scripts/run-playing-hall.ps1` output:
- `<outDir>/hands.tsv`
- `<outDir>/learning.tsv`
- `<outDir>/training-selfplay.tsv` (if enabled)
- `<outDir>/ddre-training-selfplay.tsv` (if enabled)

`scripts/run-playing-hall-max.ps1` output:
- `<outDir>/run-*/aggregate-summary.txt`
- `<outDir>/run-*/worker-*/stdout.log`
- `<outDir>/run-*/worker-*/stderr.log`
- `<outDir>/autotune/autotune-results.tsv` (if `-AutoTune`)
- `<outDir>/autotune/autotune-selection.txt` (if `-AutoTune`)

## 4. DDRE Adapter Operations

Current DDRE status:
- `synthetic` is a heuristic scaffold for routing/fallback checks.
- Native DDRE CPU/GPU currently accelerate the same synthetic inference core, not a trained diffusion model.
- The checked-in ONNX smoke model only verifies adapter execution (`posterior = sqrt(prior)`); it is not a poker-quality model.
- Decision-driving ONNX requires a DDRE artifact directory whose metadata has passed the offline gate, unless you explicitly opt into experimental artifacts for tooling.

Generate ONNX smoke model:

```powershell
python scripts/generate-ddre-smoke-onnx.py --artifact-dir data/ddre-smoke-artifact
```

Run DDRE integration suite:

```powershell
sbt "testOnly sicfun.holdem.provider.HoldemDdreIntegrationTest"
```

Run DDRE adapter parity benchmark:

```powershell
sbt "runMain sicfun.holdem.bench.HoldemDdreParityBenchmark --modes=synthetic,onnx --referenceMode=synthetic --onnxArtifactDir=data/ddre-smoke-artifact --onnxAllowExperimental=true --warmupRuns=0 --measureRuns=2 --hypothesisCount=128 --maxL1Diff=1e-4 --maxAbsDiff=1e-5"
```

Run the DDRE offline validation gate against self-play data:

```powershell
sbt "runMain sicfun.holdem.provider.HoldemDdreOfflineGate --dataset=data/bench-hall-single/ddre-training-selfplay.tsv --artifactDir=data/ddre-smoke-artifact --minSamples=100 --maxMeanNll=8.0 --maxMeanKlVsBayes=8.0 --maxBlockerViolationRate=0.0 --maxFailureRate=0.0 --maxP95LatencyMillis=50.0"
```

## 5. Native Build/Runtime Operations

CPU native build:

```powershell
powershell -ExecutionPolicy Bypass -File src/main/native/build-windows-llvm.ps1
```

CUDA native build:

```powershell
powershell -ExecutionPolicy Bypass -File src/main/native/build-windows-cuda11.ps1
```

Heads-up range CUDA auto-tuner:

```powershell
sbt "runMain sicfun.holdem.bench.HeadsUpRangeGpuAutoTuner --heroes=256 --entriesPerHero=128 --trials=256 --warmupRuns=1 --runs=3 --cachePath=data/headsup-range-autotune.properties"
```

Postflop CUDA auto-tuner:

```powershell
sbt "runMain sicfun.holdem.bench.HoldemPostflopGpuAutoTuner --villains=1024 --trials=2000 --warmupRuns=1 --runs=3 --cachePath=data/postflop-autotune.properties"
```

## 6. Troubleshooting

Classpath appears stale or Java run behaves like old code:
- Re-run with `-RefreshClasspath` on `run-playing-hall.ps1` or `run-playing-hall-max.ps1`.

`sbt` lock/server issues:
- Kill stale sbt Java processes:

```powershell
Get-CimInstance Win32_Process -Filter "Name='java.exe'" |
  Where-Object { $_.CommandLine -match "sbt|sbt-launch" } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
```

GPU profile underperforms:
- Use `-AutoTune` and include both `auto` and `cpu` in `-AutoTuneProfiles`.
- Reduce `-Workers` or `-TableCountPerWorker` if GPU is oversubscribed.

Short hall benchmark regressed after retuning the range GPU cache:
- For `1000`-hand smoke/control runs, use `-JvmOption "-Dsicfun.gpu.range.autotune.cachePath=data/headsup-range-autotune.shortrun-prev.properties"`.
- For `5000+` hand long exact-mode runs on this machine, keep the default `data/headsup-range-autotune.properties`.

## 7. Minimal Command Set

The minimal set most operators need:

```powershell
# 1) quick health
powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1 -Quick

# 2) auto-tuned max run
powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall-max.ps1 -AutoTune -AutoTuneHands 500000 -AutoTuneProfiles auto,cpu,gpu -Hands 100000000 -TableCountPerWorker 8 -LearnEveryHands 0 -SaveTrainingTsv false -SaveDdreTrainingTsv false -OutDir data/bench-hall-max

# 3) inspect result
Get-Content data/bench-hall-max/autotune/autotune-selection.txt
Get-Content data/bench-hall-max/run-*/aggregate-summary.txt
```

## 8. Optional AI Sidecars

Optional delegated analysis/review helpers for read-heavy tasks:

Unified health check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action doctor
```

One-time auth:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action auth -Provider gemini
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action auth -Provider claude
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action auth -Provider gpt
```

Manual/no-browser auth is available for Gemini and GPT:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action auth -Provider gemini -NoBrowser
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action auth -Provider gpt -NoBrowser
```

Read-only delegation example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 `
  -Action delegate `
  -Provider gpt `
  -Mode analysis `
  -Task "Summarize the latest exact-mode hall benchmark deltas." `
  -ContextPath docs/OPERATOR_RUNBOOK.md,ROADMAP.md `
  -OutputFormat text
```

Notes:

- Gemini keeps its provider-specific wrapper at `scripts/gemini-sidecar.ps1`.
- Claude login is browser-based through `claude auth login`.
- GPT uses the official OpenAI Codex CLI and ChatGPT/device auth.

For setup details and more examples, see `docs/AI_MINIONS.md` and `docs/GEMINI_MINION.md`.
