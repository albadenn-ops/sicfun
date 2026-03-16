[CmdletBinding()]
param(
  [int]$Hands = 1000000,
  [int]$TableCount = 1,
  [ValidateRange(2, 9)]
  [int]$PlayerCount = 2,
  [int]$ReportEvery = 50000,
  [int]$LearnEveryHands = 50000,
  [int]$LearningWindowSamples = 200000,
  [long]$Seed = 42,
  [string]$OutDir = "data/playing-hall",
  [ValidateSet("adaptive", "gto")]
  [string]$HeroStyle = "adaptive",
  [string]$HeroPosition = "",
  [ValidateSet("fast", "exact")]
  [string]$GtoMode = "exact",
  [ValidateSet("nit", "tag", "lag", "callingstation", "station", "maniac", "gto")]
  [string]$VillainStyle = "tag",
  [string]$VillainPool = "",
  [double]$HeroExplorationRate = 0.05,
  [double]$RaiseSize = 2.5,
  [int]$BunchingTrials = 80,
  [int]$EquityTrials = 700,
  [object]$SaveTrainingTsv = $true,
  [object]$SaveDdreTrainingTsv = $false,
  [object]$SaveReviewHandHistory = $false,
  [ValidateSet("java", "sbt")]
  [string]$Runner = "java",
  [switch]$RefreshClasspath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-BoolLiteral {
  param(
    [object]$Value,
    [string]$Name
  )

  if ($Value -is [bool]) {
    return $Value.ToString().ToLowerInvariant()
  }

  $raw = [string]$Value
  if ([string]::IsNullOrWhiteSpace($raw)) {
    throw "$Name must be true/false (or 1/0)."
  }

  switch ($raw.Trim().ToLowerInvariant()) {
    "true" { return "true" }
    "false" { return "false" }
    "1" { return "true" }
    "0" { return "false" }
    default { throw "$Name must be true/false (or 1/0)." }
  }
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

function Resolve-RuntimeClasspath {
  param(
    [string]$CachePath
  )

  function Test-ClasspathLine {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) {
      return $false
    }
    return $Value -match "\.jar" -and $Value -match ";" -and $Value -notmatch "\[info\]|\[warn\]|welcome to sbt"
  }

  Stop-StaleSbtJavaProcesses
  $null = & sbt --error "package"
  $packageExitCode = $LASTEXITCODE
  if ($packageExitCode -ne 0) {
    throw "Failed to package runtime classes (exit code $packageExitCode)"
  }

  if (-not $RefreshClasspath -and (Test-Path $CachePath)) {
    $cached = (Get-Content -Path $CachePath -Raw).Trim()
    if (Test-ClasspathLine -Value $cached) {
      return $cached
    }
  }

  $sbtOutput = & sbt --error "export Runtime / fullClasspathAsJars"
  $exitCode = $LASTEXITCODE
  if ($exitCode -ne 0) {
    throw "Failed to build/export runtime classpath (exit code $exitCode)"
  }
  $lines = $sbtOutput | ForEach-Object { "$_" }
  $joined = $lines -join [Environment]::NewLine
  $matches = [regex]::Matches($joined, '(?i)[A-Za-z]:\\[^;\r\n"]+?\.jar(?:;[A-Za-z]:\\[^;\r\n"]+?\.jar)+')
  $classpathLine =
    if ($matches.Count -gt 0) {
      $matches[$matches.Count - 1].Value
    }
    else {
      $lines |
        Where-Object {
          ($_ -match "\.jar") -and ($_ -match ";") -and ($_ -notmatch "^\[info\]") -and ($_ -notmatch "^\[warn\]")
        } |
        Select-Object -Last 1
    }
  if (-not (Test-ClasspathLine -Value $classpathLine)) {
    $tail = ($lines | Select-Object -Last 20) -join [Environment]::NewLine
    Write-Host "sbt output tail:"
    Write-Host $tail
    throw "Could not parse classpath from sbt export output"
  }
  Set-Content -Path $CachePath -Value $classpathLine
  return $classpathLine
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$previousSbtOpts = $null
Push-Location $repoRoot
try {
  $previousSbtOpts = $env:SBT_OPTS
  $env:SBT_OPTS = "-Dsbt.server.autostart=false"
  $saveTrainingTsvLiteral = Resolve-BoolLiteral -Value $SaveTrainingTsv -Name "SaveTrainingTsv"
  $saveDdreTrainingTsvLiteral = Resolve-BoolLiteral -Value $SaveDdreTrainingTsv -Name "SaveDdreTrainingTsv"
  $saveReviewHandHistoryLiteral = Resolve-BoolLiteral -Value $SaveReviewHandHistory -Name "SaveReviewHandHistory"

  $args = @(
    "--hands=$Hands",
    "--tableCount=$TableCount",
    "--playerCount=$PlayerCount",
    "--reportEvery=$ReportEvery",
    "--learnEveryHands=$LearnEveryHands",
    "--learningWindowSamples=$LearningWindowSamples",
    "--seed=$Seed",
    "--outDir=$OutDir",
    "--heroStyle=$HeroStyle",
    "--gtoMode=$GtoMode",
    "--villainStyle=$VillainStyle",
    "--heroExplorationRate=$HeroExplorationRate",
    "--raiseSize=$RaiseSize",
    "--bunchingTrials=$BunchingTrials",
    "--equityTrials=$EquityTrials",
    "--saveTrainingTsv=$saveTrainingTsvLiteral",
    "--saveDdreTrainingTsv=$saveDdreTrainingTsvLiteral",
    "--saveReviewHandHistory=$saveReviewHandHistoryLiteral"
  )
  if (-not [string]::IsNullOrWhiteSpace($HeroPosition)) {
    $args += "--heroPosition=$HeroPosition"
  }
  if (-not [string]::IsNullOrWhiteSpace($VillainPool)) {
    $args += "--villainPool=$VillainPool"
  }
  if ($Runner -eq "java") {
    $cacheDir = Join-Path $repoRoot "data"
    if (-not (Test-Path $cacheDir)) {
      New-Item -ItemType Directory -Path $cacheDir | Out-Null
    }
    $classpathCache = Join-Path $cacheDir "runtime-classpath.txt"
    $classpath = Resolve-RuntimeClasspath -CachePath $classpathCache
    & java -cp $classpath "sicfun.holdem.runtime.TexasHoldemPlayingHall" @args
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
      throw "java run failed with exit code $exitCode"
    }
  }
  else {
    $joined = $args -join " "
    Stop-StaleSbtJavaProcesses
    Invoke-SbtWithRetry -Commands @("runMain sicfun.holdem.runtime.TexasHoldemPlayingHall $joined")
  }
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
