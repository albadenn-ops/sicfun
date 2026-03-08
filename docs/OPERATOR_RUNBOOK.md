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
sbt "testOnly sicfun.holdem.HoldemDdreIntegrationTest"
```

Run DDRE adapter parity benchmark:

```powershell
sbt "runMain sicfun.holdem.HoldemDdreParityBenchmark --modes=synthetic,onnx --referenceMode=synthetic --onnxArtifactDir=data/ddre-smoke-artifact --onnxAllowExperimental=true --warmupRuns=0 --measureRuns=2 --hypothesisCount=128 --maxL1Diff=1e-4 --maxAbsDiff=1e-5"
```

Run the DDRE offline validation gate against self-play data:

```powershell
sbt "runMain sicfun.holdem.HoldemDdreOfflineGate --dataset=data/bench-hall-single/ddre-training-selfplay.tsv --artifactDir=data/ddre-smoke-artifact --minSamples=100 --maxMeanNll=8.0 --maxMeanKlVsBayes=8.0 --maxBlockerViolationRate=0.0 --maxFailureRate=0.0 --maxP95LatencyMillis=50.0"
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

Postflop CUDA auto-tuner:

```powershell
sbt "runMain sicfun.holdem.HoldemPostflopGpuAutoTuner --villains=1024 --trials=2000 --warmupRuns=1 --runs=3 --cachePath=data/postflop-autotune.properties"
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
