param(
  [Parameter(Mandatory = $true)][string]$Server,
  [Parameter(Mandatory = $true)][string]$Port
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$heroMode = if ($env:SICFUN_ACPC_HERO_MODE) { $env:SICFUN_ACPC_HERO_MODE } else { "adaptive" }
$outDir = if ($env:SICFUN_ACPC_OUT_DIR) { $env:SICFUN_ACPC_OUT_DIR } else { "data/acpc-match-runner" }
$bunchingTrials = if ($env:SICFUN_ACPC_BUNCHING_TRIALS) { $env:SICFUN_ACPC_BUNCHING_TRIALS } else { "1" }
$equityTrials = if ($env:SICFUN_ACPC_EQUITY_TRIALS) { $env:SICFUN_ACPC_EQUITY_TRIALS } else { "600" }
$cfrIterations = if ($env:SICFUN_ACPC_CFR_ITERATIONS) { $env:SICFUN_ACPC_CFR_ITERATIONS } else { "180" }
$cfrVillainHands = if ($env:SICFUN_ACPC_CFR_VILLAIN_HANDS) { $env:SICFUN_ACPC_CFR_VILLAIN_HANDS } else { "48" }
$cfrEquityTrials = if ($env:SICFUN_ACPC_CFR_EQUITY_TRIALS) { $env:SICFUN_ACPC_CFR_EQUITY_TRIALS } else { "300" }
$seed = if ($env:SICFUN_ACPC_SEED) { $env:SICFUN_ACPC_SEED } else { "42" }
$timeoutMillis = if ($env:SICFUN_ACPC_TIMEOUT_MILLIS) { $env:SICFUN_ACPC_TIMEOUT_MILLIS } else { "15000" }
$connectTimeoutMillis = if ($env:SICFUN_ACPC_CONNECT_TIMEOUT_MILLIS) { $env:SICFUN_ACPC_CONNECT_TIMEOUT_MILLIS } else { "5000" }

function Test-ClasspathValue([string]$Value) {
  if ([string]::IsNullOrWhiteSpace($Value)) {
    return $false
  }
  return $Value -match "\.jar" -and $Value -match ";"
}

function Resolve-SicfunClasspath {
  if (Test-ClasspathValue $env:SICFUN_ACPC_CLASSPATH) {
    return $env:SICFUN_ACPC_CLASSPATH
  }

  $cachePath = Join-Path $repoRoot "data/runtime-classpath.txt"
  if (Test-Path $cachePath) {
    $cached = (Get-Content -Path $cachePath -Raw).Trim()
    if (Test-ClasspathValue $cached) {
      return $cached
    }
  }

  $previousSbtOpts = $env:SBT_OPTS
  $env:SBT_OPTS = "-Dsbt.server.autostart=false"
  try {
    $null = & sbt --error "package"
    if ($LASTEXITCODE -ne 0) {
      throw "sbt package failed with exit code $LASTEXITCODE"
    }
    $sbtOutput = & sbt --error "export Runtime / fullClasspathAsJars"
    if ($LASTEXITCODE -ne 0) {
      throw "sbt export failed with exit code $LASTEXITCODE"
    }
    $lines = $sbtOutput | ForEach-Object { "$_" }
    $joinedOutput = $lines -join [Environment]::NewLine
    $matches = [regex]::Matches($joinedOutput, '(?i)[A-Za-z]:\\[^;\r\n"]+?\.jar(?:;[A-Za-z]:\\[^;\r\n"]+?\.jar)+')
    $classpath =
      if ($matches.Count -gt 0) { $matches[$matches.Count - 1].Value }
      else {
        $lines |
          Where-Object {
            ($_ -match "\.jar") -and ($_ -match ";") -and ($_ -notmatch "^\[info\]") -and ($_ -notmatch "^\[warn\]")
          } |
          Select-Object -Last 1
      }
    if (-not (Test-ClasspathValue $classpath)) {
      throw "Could not parse runtime classpath from sbt export output"
    }
    Set-Content -Path $cachePath -Value $classpath
    return $classpath
  }
  finally {
    if ($null -ne $previousSbtOpts) {
      $env:SBT_OPTS = $previousSbtOpts
    }
    else {
      Remove-Item Env:SBT_OPTS -ErrorAction SilentlyContinue
    }
  }
}

$runnerArgs = @(
  "--server=$Server",
  "--port=$Port",
  "--heroMode=$heroMode",
  "--outDir=$outDir",
  "--bunchingTrials=$bunchingTrials",
  "--equityTrials=$equityTrials",
  "--cfrIterations=$cfrIterations",
  "--cfrVillainHands=$cfrVillainHands",
  "--cfrEquityTrials=$cfrEquityTrials",
  "--seed=$seed",
  "--timeoutMillis=$timeoutMillis",
  "--connectTimeoutMillis=$connectTimeoutMillis"
)

if ($env:SICFUN_ACPC_MODEL) {
  $runnerArgs += "--model=$($env:SICFUN_ACPC_MODEL)"
}

$classpath = Resolve-SicfunClasspath

Push-Location $repoRoot
try {
  & java -cp $classpath "sicfun.holdem.runtime.AcpcMatchRunner" @runnerArgs
}
finally {
  Pop-Location
}
