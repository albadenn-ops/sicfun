[CmdletBinding()]
param(
  [switch]$Quick
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Stop-StaleSbtJavaProcesses {
  $sbtJava = Get-CimInstance Win32_Process -Filter "Name='java.exe'" |
    Where-Object {
      $cmd = $_.CommandLine
      $null -ne $cmd -and ($cmd -match "sbt" -or $cmd -match "sbt-launch")
    }
  foreach ($proc in $sbtJava) {
    Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
  }
}

function Invoke-SbtWithRetry {
  param(
    [string[]]$Commands,
    [int]$MaxAttempts = 3
  )

  $attempt = 1
  while ($attempt -le $MaxAttempts) {
    & sbt @Commands
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
      return
    }
    if ($attempt -lt $MaxAttempts) {
      Stop-StaleSbtJavaProcesses
      Start-Sleep -Milliseconds 1200
      $attempt += 1
      continue
    }
    throw "sbt command failed with exit code $exitCode"
  }
}

$fullSuites = @(
  "sicfun.holdem.bench.PokerPipelineTest",
  "sicfun.holdem.types.HandEngineTest",
  "sicfun.holdem.engine.RangeInferenceEngineTest",
  "sicfun.holdem.engine.RealTimeAdaptiveEngineTest",
  "sicfun.holdem.runtime.LiveHandSimulatorTest",
  "sicfun.holdem.runtime.AlwaysOnDecisionLoopTest",
  "sicfun.holdem.runtime.TexasHoldemPlayingHallTest",
  "sicfun.holdem.model.TrainPokerActionModelCliTest",
  "sicfun.holdem.analysis.GenerateSignalsCliTest",
  "sicfun.holdem.bench.OperationalRegressionSuiteTest",
  "sicfun.holdem.web.HandHistoryReviewSimulationProofTest",
  "sicfun.holdem.web.HandHistoryReviewServerTest"
)

$quickSuites = @(
  "sicfun.holdem.bench.PokerPipelineTest",
  "sicfun.holdem.types.HandEngineTest",
  "sicfun.holdem.runtime.LiveHandSimulatorTest",
  "sicfun.holdem.runtime.AlwaysOnDecisionLoopTest",
  "sicfun.holdem.runtime.TexasHoldemPlayingHallTest",
  "sicfun.holdem.bench.OperationalRegressionSuiteTest"
)

$selectedSuites =
  if ($Quick) { $quickSuites } else { $fullSuites }

$commands = $selectedSuites | ForEach-Object { "testOnly $_" }

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$previousSbtOpts = $null
Push-Location $repoRoot
try {
  $previousSbtOpts = $env:SBT_OPTS
  $env:SBT_OPTS = "-Dsbt.server.autostart=false"

  if ($Quick) {
    Write-Host "Quick proof covers core engine/runtime smoke. Run the full proof for hand-history review end-to-end coverage."
  }
  else {
    Write-Host "Full proof includes hand-history review end-to-end coverage: hall export -> import -> analysis service -> async web job."
  }

  Stop-StaleSbtJavaProcesses
  Invoke-SbtWithRetry -Commands $commands
  Write-Host "Pipeline proof passed for $($selectedSuites.Count) suite(s)."
}
finally {
  if ($null -ne $previousSbtOpts) {
    $env:SBT_OPTS = $previousSbtOpts
  }
  else {
    Remove-Item Env:SBT_OPTS -ErrorAction SilentlyContinue
  }
  Pop-Location
}
