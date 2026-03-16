param(
  [Alias("Host")]
  [string]$BindHost = "127.0.0.1",
  [int]$Port = 8080,
  [string]$StaticDir = "docs/site-preview-hybrid",
  [string]$Model = "",
  [long]$Seed = 42,
  [int]$BunchingTrials = 200,
  [int]$EquityTrials = 2000,
  [long]$BudgetMs = 1500,
  [int]$MaxDecisions = 12,
  [int]$MaxUploadBytes = 2097152
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-AppPath {
  param(
    [string]$PathValue
  )

  if ([string]::IsNullOrWhiteSpace($PathValue)) {
    return ""
  }

  $candidate =
    if ([System.IO.Path]::IsPathRooted($PathValue)) { $PathValue }
    else { Join-Path $repoRoot $PathValue }

  if (-not (Test-Path -LiteralPath $candidate)) {
    throw "Path not found: $PathValue"
  }

  (Resolve-Path -LiteralPath $candidate).Path
}

function Set-Or-ClearEnv {
  param(
    [string]$Name,
    [string]$Value
  )

  if ([string]::IsNullOrWhiteSpace($Value)) {
    Remove-Item "Env:$Name" -ErrorAction SilentlyContinue
  }
  else {
    Set-Item "Env:$Name" $Value
  }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$resolvedStaticDir = Resolve-AppPath -PathValue $StaticDir
$resolvedModel = ""
if (-not [string]::IsNullOrWhiteSpace($Model)) {
  $resolvedModel = Resolve-AppPath -PathValue $Model
}

$previousSbtOpts = $env:SBT_OPTS
$previousEnv = @{
  HOST = $env:HOST
  PORT = $env:PORT
  STATIC_DIR = $env:STATIC_DIR
  MODEL_DIR = $env:MODEL_DIR
  SEED = $env:SEED
  BUNCHING_TRIALS = $env:BUNCHING_TRIALS
  EQUITY_TRIALS = $env:EQUITY_TRIALS
  BUDGET_MS = $env:BUDGET_MS
  MAX_DECISIONS = $env:MAX_DECISIONS
  MAX_UPLOAD_BYTES = $env:MAX_UPLOAD_BYTES
}

try {
  $env:SBT_OPTS = "-Dsbt.server.autostart=false"
  Set-Or-ClearEnv -Name "HOST" -Value $BindHost
  Set-Or-ClearEnv -Name "PORT" -Value $Port
  Set-Or-ClearEnv -Name "STATIC_DIR" -Value $resolvedStaticDir
  Set-Or-ClearEnv -Name "MODEL_DIR" -Value $resolvedModel
  Set-Or-ClearEnv -Name "SEED" -Value $Seed
  Set-Or-ClearEnv -Name "BUNCHING_TRIALS" -Value $BunchingTrials
  Set-Or-ClearEnv -Name "EQUITY_TRIALS" -Value $EquityTrials
  Set-Or-ClearEnv -Name "BUDGET_MS" -Value $BudgetMs
  Set-Or-ClearEnv -Name "MAX_DECISIONS" -Value $MaxDecisions
  Set-Or-ClearEnv -Name "MAX_UPLOAD_BYTES" -Value $MaxUploadBytes

  Push-Location $repoRoot
  try {
    sbt "runMain sicfun.holdem.web.HandHistoryReviewServer"
  }
  finally {
    Pop-Location
  }
}
finally {
  if ($null -ne $previousSbtOpts) {
    $env:SBT_OPTS = $previousSbtOpts
  }
  else {
    Remove-Item Env:SBT_OPTS -ErrorAction SilentlyContinue
  }

  foreach ($entry in $previousEnv.GetEnumerator()) {
    if ($null -ne $entry.Value) {
      Set-Item "Env:$($entry.Key)" $entry.Value
    }
    else {
      Remove-Item "Env:$($entry.Key)" -ErrorAction SilentlyContinue
    }
  }
}
