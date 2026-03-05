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
  "sicfun.holdem.PokerPipelineTest",
  "sicfun.holdem.HandEngineTest",
  "sicfun.holdem.RangeInferenceEngineTest",
  "sicfun.holdem.RealTimeAdaptiveEngineTest",
  "sicfun.holdem.LiveHandSimulatorTest",
  "sicfun.holdem.AlwaysOnDecisionLoopTest",
  "sicfun.holdem.TexasHoldemPlayingHallTest",
  "sicfun.holdem.TrainPokerActionModelCliTest",
  "sicfun.holdem.GenerateSignalsCliTest",
  "sicfun.holdem.OperationalRegressionSuiteTest"
)

$quickSuites = @(
  "sicfun.holdem.PokerPipelineTest",
  "sicfun.holdem.HandEngineTest",
  "sicfun.holdem.LiveHandSimulatorTest",
  "sicfun.holdem.AlwaysOnDecisionLoopTest",
  "sicfun.holdem.TexasHoldemPlayingHallTest",
  "sicfun.holdem.OperationalRegressionSuiteTest"
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
