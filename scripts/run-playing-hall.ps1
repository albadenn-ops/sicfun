[CmdletBinding()]
param(
  [int]$Hands = 1000000,
  [int]$ReportEvery = 50000,
  [int]$LearnEveryHands = 50000,
  [int]$LearningWindowSamples = 200000,
  [long]$Seed = 42,
  [string]$OutDir = "data/playing-hall",
  [ValidateSet("nit", "tag", "lag", "callingstation", "station", "maniac")]
  [string]$VillainStyle = "tag",
  [double]$HeroExplorationRate = 0.05,
  [double]$RaiseSize = 2.5,
  [int]$BunchingTrials = 80,
  [int]$EquityTrials = 700,
  [bool]$SaveTrainingTsv = $true
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
    if ($exitCode -eq 0) { return }
    if ($attempt -lt $MaxAttempts) {
      Stop-StaleSbtJavaProcesses
      Start-Sleep -Milliseconds 1200
      $attempt += 1
      continue
    }
    throw "sbt command failed with exit code $exitCode"
  }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$previousSbtOpts = $null
Push-Location $repoRoot
try {
  $previousSbtOpts = $env:SBT_OPTS
  $env:SBT_OPTS = "-Dsbt.server.autostart=false"

  $args = @(
    "--hands=$Hands",
    "--reportEvery=$ReportEvery",
    "--learnEveryHands=$LearnEveryHands",
    "--learningWindowSamples=$LearningWindowSamples",
    "--seed=$Seed",
    "--outDir=$OutDir",
    "--villainStyle=$VillainStyle",
    "--heroExplorationRate=$HeroExplorationRate",
    "--raiseSize=$RaiseSize",
    "--bunchingTrials=$BunchingTrials",
    "--equityTrials=$EquityTrials",
    "--saveTrainingTsv=$SaveTrainingTsv"
  )
  $joined = $args -join " "
  Stop-StaleSbtJavaProcesses
  Invoke-SbtWithRetry -Commands @("runMain sicfun.holdem.TexasHoldemPlayingHall $joined")
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
