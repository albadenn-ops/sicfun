[CmdletBinding()]
param(
  [int]$Hands = 50,
  [int]$ReportEvery = 10,
  [string]$OutDir = "data/bench-slumbot",
  [string]$HeroModes = "adaptive,gto",
  [string]$ModelArtifactDir = "",
  [int]$BunchingTrials = 1,
  [int]$EquityTrials = 600,
  [int]$CfrIterations = 180,
  [int]$CfrVillainHands = 48,
  [int]$CfrEquityTrials = 300,
  [long]$Seed = 42,
  [long]$TimeoutMillis = 15000,
  [string]$BaseUrl = "https://slumbot.com",
  [switch]$FailFast
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Split-TrimCsv {
  param([string]$Value)

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return @()
  }

  return @(
    $Value.Split(",") |
      ForEach-Object { $_.Trim().ToLowerInvariant() } |
      Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
  )
}

function Get-SummaryMap {
  param([string]$Path)

  $map = @{}
  foreach ($line in Get-Content -Path $Path) {
    if ($line -match "^\s*([^:]+):\s*(.*)$") {
      $map[$matches[1].Trim()] = $matches[2].Trim()
    }
  }
  return $map
}

function To-InvariantDouble {
  param([string]$Value)
  $normalized = $Value.Trim()
  if ($normalized.Contains(",") -and -not $normalized.Contains(".")) {
    $normalized = $normalized.Replace(",", ".")
  }
  return [double]::Parse($normalized, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Format-Invariant {
  param(
    [double]$Value,
    [int]$Digits = 3
  )
  return $Value.ToString(("F{0}" -f $Digits), [System.Globalization.CultureInfo]::InvariantCulture)
}

$modes = @(Split-TrimCsv -Value $HeroModes)
if ($modes.Count -eq 0) {
  throw "HeroModes produced no values."
}

$allowedModes = @("adaptive", "gto")
foreach ($mode in $modes) {
  if ($allowedModes -notcontains $mode) {
    throw "Unsupported hero mode '$mode'. Allowed: adaptive, gto."
  }
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$resultsPath = Join-Path $OutDir "results.tsv"
$summaryPath = Join-Path $OutDir "summary.txt"
"heroMode`thandsPlayed`theroNetChips`theroBbPer100`theroWins`theroTies`theroLosses`tbuttonHands`tbuttonNetChips`tbigBlindHands`tbigBlindNetChips`tmodelId`toutputDir" |
  Set-Content -Path $resultsPath

$rows = @()

foreach ($mode in $modes) {
  $runDir = Join-Path $OutDir $mode
  $args = @(
    "--error",
    "runMain sicfun.holdem.runtime.SlumbotMatchRunner --hands=$Hands --reportEvery=$ReportEvery --heroMode=$mode --outDir=$runDir --bunchingTrials=$BunchingTrials --equityTrials=$EquityTrials --cfrIterations=$CfrIterations --cfrVillainHands=$CfrVillainHands --cfrEquityTrials=$CfrEquityTrials --seed=$Seed --timeoutMillis=$TimeoutMillis --baseUrl=$BaseUrl"
  )
  if (-not [string]::IsNullOrWhiteSpace($ModelArtifactDir)) {
    $args[1] += " --model=$ModelArtifactDir"
  }

  try {
    & sbt @args
    if ($LASTEXITCODE -ne 0) {
      throw "sbt exited with code $LASTEXITCODE"
    }
  }
  catch {
    if ($FailFast) {
      throw
    }
    Write-Warning "Run for heroMode=$mode failed: $($_.Exception.Message)"
    continue
  }

  $runSummaryPath = Join-Path $runDir "summary.txt"
  if (-not (Test-Path $runSummaryPath)) {
    if ($FailFast) {
      throw "Missing summary file: $runSummaryPath"
    }
    Write-Warning "Missing summary file for heroMode=$mode"
    continue
  }

  $summary = Get-SummaryMap -Path $runSummaryPath
  $handsPlayed = [int]$summary["handsPlayed"]
  $heroNetChips = [int]$summary["heroNetChips"]
  $heroBbPer100 = (($heroNetChips / 100.0) / $handsPlayed) * 100.0
  $row = [PSCustomObject]@{
    heroMode = $mode
    handsPlayed = $handsPlayed
    heroNetChips = $heroNetChips
    heroBbPer100 = $heroBbPer100
    heroWins = [int]$summary["heroWins"]
    heroTies = [int]$summary["heroTies"]
    heroLosses = [int]$summary["heroLosses"]
    buttonHands = [int]$summary["buttonHands"]
    buttonNetChips = [int]$summary["buttonNetChips"]
    bigBlindHands = [int]$summary["bigBlindHands"]
    bigBlindNetChips = [int]$summary["bigBlindNetChips"]
    modelId = $summary["modelId"]
    outputDir = (Resolve-Path $runDir).Path
  }
  $rows += $row

  "$($row.heroMode)`t$($row.handsPlayed)`t$($row.heroNetChips)`t$(Format-Invariant $row.heroBbPer100 3)`t$($row.heroWins)`t$($row.heroTies)`t$($row.heroLosses)`t$($row.buttonHands)`t$($row.buttonNetChips)`t$($row.bigBlindHands)`t$($row.bigBlindNetChips)`t$($row.modelId)`t$($row.outputDir)" |
    Add-Content -Path $resultsPath
}

$summaryLines = @(
  "=== Slumbot Benchmark ===",
  "handsPerRun: $Hands",
  "heroModes: $($modes -join ',')",
  "baseUrl: $BaseUrl",
  "outDir: $((Resolve-Path $OutDir).Path)",
  ""
)

foreach ($row in $rows) {
  $summaryLines += "[$($row.heroMode)]"
  $summaryLines += "heroNetChips: $($row.heroNetChips)"
  $summaryLines += "heroBbPer100: $(Format-Invariant $row.heroBbPer100 3)"
  $summaryLines += "record: $($row.heroWins)-$($row.heroTies)-$($row.heroLosses)"
  $summaryLines += "button: $($row.buttonNetChips) chips over $($row.buttonHands) hands"
  $summaryLines += "bigBlind: $($row.bigBlindNetChips) chips over $($row.bigBlindHands) hands"
  $summaryLines += "modelId: $($row.modelId)"
  $summaryLines += "outputDir: $($row.outputDir)"
  $summaryLines += ""
}

Set-Content -Path $summaryPath -Value $summaryLines
Write-Host "Wrote $resultsPath"
Write-Host "Wrote $summaryPath"
