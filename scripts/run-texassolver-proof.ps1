[CmdletBinding()]
param(
  [string]$Reference = "validation-output/cfr-proof/external-comparison.json",
  [string]$TexasDir,
  [string]$OutDir = "validation-output/texassolver-proof",
  [string]$SpotIds = "",
  [Nullable[int]]$ExpectedPlayer = $null,
  [double]$MaxMeanTv = 0.05,
  [double]$MaxSpotTv = 0.15,
  [double]$MinBestActionAgreement = 0.80
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Quote-Arg([string]$Value) {
  '"' + ($Value -replace '"', '\"') + '"'
}

function Resolve-RepoPath([string]$PathValue, [string]$RepoRoot) {
  if ([string]::IsNullOrWhiteSpace($PathValue)) {
    return $PathValue
  }
  if ([System.IO.Path]::IsPathRooted($PathValue)) {
    return $PathValue
  }
  return (Join-Path $RepoRoot $PathValue)
}

function Normalize-CsvArg([string]$Value, [string]$Label) {
  $tokens = @(
    $Value.Split(",") |
      ForEach-Object { $_.Trim() } |
      Where-Object { -not [string]::IsNullOrWhiteSpace($_) }
  )
  if ($tokens.Count -eq 0) {
    throw "-$Label must contain at least one non-empty value"
  }
  return ($tokens -join ",")
}

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
    [string]$Command,
    [string]$FailureLabel,
    [int]$MaxAttempts = 3
  )

  $attempt = 1
  while ($attempt -le $MaxAttempts) {
    & sbt $Command
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
    throw "$FailureLabel failed with sbt exit code $exitCode"
  }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$previousSbtOpts = $null
Push-Location $repoRoot
try {
  if ([string]::IsNullOrWhiteSpace($TexasDir)) {
    throw "-TexasDir is required"
  }

  $resolvedReference = Resolve-RepoPath -PathValue $Reference -RepoRoot $repoRoot
  $resolvedTexasDir = Resolve-RepoPath -PathValue $TexasDir -RepoRoot $repoRoot
  $resolvedOutDir = Resolve-RepoPath -PathValue $OutDir -RepoRoot $repoRoot
  $normalizedSpotIds =
    if ([string]::IsNullOrWhiteSpace($SpotIds)) { $null }
    else { Normalize-CsvArg -Value $SpotIds -Label "SpotIds" }

  $previousSbtOpts = $env:SBT_OPTS
  $env:SBT_OPTS = "-Dsbt.server.autostart=false"

  New-Item -ItemType Directory -Force -Path $resolvedOutDir | Out-Null
  $providerPath = Join-Path $resolvedOutDir "texassolver-provider.json"
  $comparisonOutDir = Join-Path $resolvedOutDir "comparison"
  New-Item -ItemType Directory -Force -Path $comparisonOutDir | Out-Null

  $adapterArgs = @(
    "--reference=$(Quote-Arg $resolvedReference)",
    "--texasDir=$(Quote-Arg $resolvedTexasDir)",
    "--out=$(Quote-Arg $providerPath)",
    "--providerName=TexasSolver"
  )
  if ($null -ne $normalizedSpotIds) {
    $adapterArgs += "--spotIds=$normalizedSpotIds"
  }
  if ($null -ne $ExpectedPlayer) {
    $adapterArgs += "--expectedPlayer=$ExpectedPlayer"
  }

  $compareArgs = @(
    "--reference=$(Quote-Arg $resolvedReference)",
    "--external=$(Quote-Arg $providerPath)",
    "--outDir=$(Quote-Arg $comparisonOutDir)",
    "--maxMeanTv=$MaxMeanTv",
    "--maxSpotTv=$MaxSpotTv",
    "--minBestActionAgreement=$MinBestActionAgreement"
  )
  if ($null -ne $normalizedSpotIds) {
    $compareArgs += "--spotIds=$normalizedSpotIds"
  }

  $adapterCommand = 'runMain sicfun.holdem.cfr.HoldemCfrTexasSolverJsonAdapter ' + ($adapterArgs -join ' ')
  $compareCommand = 'runMain sicfun.holdem.cfr.HoldemCfrExternalComparison ' + ($compareArgs -join ' ')

  Stop-StaleSbtJavaProcesses
  Write-Host "Adapting TexasSolver root dumps..."
  Invoke-SbtWithRetry -Command $adapterCommand -FailureLabel "TexasSolver adapter"

  Write-Host "Comparing against SICFUN reference spots..."
  Invoke-SbtWithRetry -Command $compareCommand -FailureLabel "External comparison"

  Write-Host "Provider JSON: $providerPath"
  Write-Host "Comparison output: $comparisonOutDir"
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
