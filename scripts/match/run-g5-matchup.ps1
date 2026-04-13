param(
  [int]$Hands = 100,
  [int]$ReportEvery = 20,
  [string]$OutDir = "data/bench-g5-matchup",
  [string]$SicfunHeroMode = "gto",
  [string]$SicfunModel = "",
  [switch]$SkipBuildG5
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$dealerOutDir = Join-Path $repoRoot $OutDir
$sicfunClientOutDir = Join-Path $dealerOutDir "sicfun-client"
$classpathCache = Join-Path $repoRoot "data/runtime-classpath.txt"

function Test-ClasspathValue([string]$Value) {
  if ([string]::IsNullOrWhiteSpace($Value)) {
    return $false
  }
  return $Value -match "\.jar" -and $Value -match ";"
}

function Test-ClasspathEntriesExist([string]$Value) {
  if (-not (Test-ClasspathValue $Value)) {
    return $false
  }
  $entries = $Value.Split(';', [System.StringSplitOptions]::RemoveEmptyEntries)
  if ($entries.Count -eq 0) {
    return $false
  }
  foreach ($entry in $entries) {
    if (-not (Test-Path $entry)) {
      return $false
    }
  }
  return $true
}

function Resolve-SicfunClasspath {
  if (Test-Path $classpathCache) {
    $cached = (Get-Content -Path $classpathCache -Raw).Trim()
    if (Test-ClasspathEntriesExist $cached) {
      return $cached
    }
  }

  $null = & sbt --error "package"
  if ($LASTEXITCODE -ne 0) {
    throw "sbt package failed with exit code $LASTEXITCODE"
  }

  $sbtOutput = & sbt --error "export Runtime / fullClasspathAsJars"
  if ($LASTEXITCODE -ne 0) {
    throw "sbt export failed with exit code $LASTEXITCODE"
  }
  $lines = $sbtOutput | ForEach-Object { "$_" }
  $joined = $lines -join [Environment]::NewLine
  $matches = [regex]::Matches($joined, '(?i)[A-Za-z]:\\[^;\r\n"]+?\.jar(?:;[A-Za-z]:\\[^;\r\n"]+?\.jar)+')
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
  if (-not (Test-ClasspathEntriesExist $classpath)) {
    throw "Resolved runtime classpath contains missing jar entries"
  }
  Set-Content -Path $classpathCache -Value $classpath
  return $classpath
}

if (-not $SkipBuildG5) {
  & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "build-g5-acpc.ps1")
}

$env:SICFUN_ACPC_CLASSPATH = Resolve-SicfunClasspath
$env:SICFUN_ACPC_HERO_MODE = $SicfunHeroMode
$env:SICFUN_ACPC_OUT_DIR = $sicfunClientOutDir
if ($SicfunModel) {
  $env:SICFUN_ACPC_MODEL = (Resolve-Path $SicfunModel).Path
}
else {
  Remove-Item Env:SICFUN_ACPC_MODEL -ErrorAction SilentlyContinue
}

Push-Location $repoRoot
try {
  $command = @(
    "runMain sicfun.holdem.runtime.AcpcHeadsUpDealer",
    "--hands=$Hands",
    "--reportEvery=$ReportEvery",
    "--outDir=$OutDir",
    "--playerAName=sicfun",
    "--playerBName=g5",
    "--playerAScript=scripts/start-sicfun-acpc.cmd",
    "--playerBScript=scripts/start-g5-acpc.cmd"
  ) -join " "

  & sbt --error $command
}
finally {
  Pop-Location
}
