<#
.SYNOPSIS
    Captures Phase 2 baseline benchmarks across hall, decision corpus, and optionally Slumbot.

.DESCRIPTION
    Runs all benchmark surfaces with fixed seeds and saves artifacts to a timestamped
    output directory. Must be run BEFORE any Track A formulation changes merge.

    Surfaces:
    1. Decision corpus replay (deterministic, local)
    2. Hall self-play: adaptive vs strategic vs gto (1000 hands each, seed=42)
    3. Slumbot benchmark: adaptive vs strategic (50 hands each, seed=42) [optional]

.PARAMETER OutDir
    Root output directory. Default: data/bench-baseline-phase2

.PARAMETER Hands
    Hands per hall matchup. Default: 1000

.PARAMETER Seed
    RNG seed. Default: 42

.PARAMETER SkipSlumbot
    Skip the Slumbot benchmark (requires network access).

.PARAMETER SkipHall
    Skip the hall matchups.
#>
param(
    [string]$OutDir = "data/bench-baseline-phase2",
    [int]$Hands = 1000,
    [long]$Seed = 42,
    [switch]$SkipSlumbot,
    [switch]$SkipHall
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyy-MM-dd-HHmm"
$baseDir = Join-Path $OutDir $timestamp

Write-Host "=== Phase 2 Baseline Capture ===" -ForegroundColor Cyan
Write-Host "Output: $baseDir"
Write-Host "Seed: $Seed"
Write-Host ""

# --- 1. Decision Corpus ---
Write-Host "--- Decision Corpus Benchmark ---" -ForegroundColor Yellow
$corpusDir = Join-Path $baseDir "decision-corpus"
sbt "runMain sicfun.holdem.bench.DecisionCorpusBenchmark $Seed $corpusDir"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Decision corpus benchmark failed"
    exit 1
}
Write-Host ""

# --- 2. Hall Matchups (via run-hall-matchups.ps1 for summary artifact output) ---
if (-not $SkipHall) {
    Write-Host "--- Hall Self-Play Benchmarks ---" -ForegroundColor Yellow
    $hallDir = Join-Path $baseDir "hall"
    & "$PSScriptRoot/../match/run-hall-matchups.ps1" `
        -HeroStyles "adaptive,gto,strategic" `
        -Hands $Hands `
        -Seed $Seed `
        -OutDir $hallDir
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Hall matchups failed (exit code $LASTEXITCODE)"
    }
    Write-Host ""
}

# --- 3. Slumbot ---
if (-not $SkipSlumbot) {
    Write-Host "--- Slumbot Benchmarks ---" -ForegroundColor Yellow
    $slumbotModes = @("adaptive", "strategic")
    foreach ($mode in $slumbotModes) {
        Write-Host "  Running slumbot: $mode (50 hands, seed=$Seed)" -ForegroundColor Gray
        $slumbotDir = Join-Path $baseDir "slumbot-$mode"
        sbt "runMain sicfun.holdem.runtime.protocol.SlumbotMatchRunner --hands=50 --heroMode=$mode --seed=$Seed --outDir=$slumbotDir"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Slumbot $mode failed (exit code $LASTEXITCODE)"
        }
    }
    Write-Host ""
}

# --- Summary ---
Write-Host "=== Baseline Capture Complete ===" -ForegroundColor Cyan
Write-Host "Artifacts saved to: $baseDir"

# Write metadata
$metaPath = Join-Path $baseDir "baseline-meta.txt"
@"
Phase 2 Baseline Capture
Date: $timestamp
Seed: $Seed
HallHands: $Hands
SkipSlumbot: $SkipSlumbot
SkipHall: $SkipHall
GitCommit: $(git rev-parse HEAD)
GitBranch: $(git branch --show-current)
"@ | Set-Content -Path $metaPath -Encoding UTF8

Write-Host "Metadata written to: $metaPath"
