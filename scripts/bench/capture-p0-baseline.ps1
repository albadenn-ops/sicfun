<#
.SYNOPSIS  Captures the P0 multiway baseline: Track A (hall self-play bb/100 + bootstrap CI for strategic AND adaptive) and, if a corpus is given, Track B (counterfactual bb/100 on real hands).
.DESCRIPTION
  Runs the P0 benchmarks at full hall quality with a fixed seed and saves artifacts under -OutDir.
  Track A is heavy (full 9-max self-play): ~minutes-to-hours per style at large -Hands. This is the
  honest baseline capture, intended to run before any P1+ engine change (Phase 2 G1 gate).
.PARAMETER OutDir   Root output dir. Default: data/p0-baseline
.PARAMETER Hands    Hands per Track-A run. Default: 50000
.PARAMETER Seed     RNG seed. Default: 42
.PARAMETER Corpus   Optional real hand-history file for Track B (omit to skip Track B).
.PARAMETER HeroName Hero player name in the corpus (required if -Corpus is set; case-sensitive).
#>
param(
  [string]$OutDir = "data/p0-baseline",
  [int]$Hands = 50000,
  [long]$Seed = 42,
  [string]$Corpus = "",
  [string]$HeroName = ""
)
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

foreach ($style in @("strategic", "adaptive")) {
  Write-Host "== Track A: hero=$style, hands=$Hands, seed=$Seed =="
  $trackAOut = Join-Path $OutDir "trackA-$style"
  sbt "runMain sicfun.holdem.bench.MultiwayWinRateBenchmark $style $Hands $Seed $trackAOut"
  if ($LASTEXITCODE -ne 0) { throw "Track A ($style) failed (exit $LASTEXITCODE)" }
}

if ($Corpus -ne "") {
  if ($HeroName -eq "") { throw "-HeroName is required when -Corpus is set" }
  if (-not (Test-Path $Corpus)) { throw "Corpus file not found: $Corpus" }
  Write-Host "== Track B: counterfactual on $Corpus (hero=$HeroName) =="
  sbt "runMain sicfun.holdem.bench.CounterfactualHandHistoryBenchmark $Corpus $HeroName $Seed"
  if ($LASTEXITCODE -ne 0) { throw "Track B failed (exit $LASTEXITCODE)" }
} else {
  Write-Host "Track B skipped (no -Corpus). Real histories stay local; supply -Corpus + -HeroName to capture the counterfactual delta."
}

Write-Host "Baseline artifacts under $OutDir (Track A writes winrate-summary.txt per style)."
