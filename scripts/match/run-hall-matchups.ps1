[CmdletBinding()]
param(
  [int]$Hands = 1000,
  [int]$TableCount = 1,
  [int]$ReportEvery = 1000,
  [int]$LearnEveryHands = 0,
  [int]$LearningWindowSamples = 200000,
  [long]$Seed = 42,
  [string]$OutDir = "data/bench-hall-matchups",
  [string]$HeroStyles = "adaptive,gto",
  [string]$VillainStyles = "nit,tag,lag,station,maniac,gto",
  [ValidateSet("fast", "exact")]
  [string]$GtoMode = "fast",
  [double]$HeroExplorationRate = 0.0,
  [double]$RaiseSize = 2.5,
  [int]$BunchingTrials = 80,
  [int]$EquityTrials = 700,
  [object]$SaveTrainingTsv = $false,
  [object]$SaveDdreTrainingTsv = $false,
  [ValidateSet("auto", "cpu", "gpu")]
  [string]$NativeProfile = "auto",
  [string]$ModelArtifactDir = "",
  [ValidateSet("single", "seat-normalized")]
  [string]$SeatMode = "seat-normalized",
  [switch]$RefreshClasspath,
  [switch]$SkipExisting,
  [switch]$FailFast
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Format-InvariantNumber {
  param(
    [double]$Value,
    [int]$Digits = 3
  )

  if ([double]::IsNaN($Value)) {
    return "NaN"
  }

  return $Value.ToString(("F{0}" -f $Digits), [System.Globalization.CultureInfo]::InvariantCulture)
}

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
    & sbt --error @Commands
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
  Invoke-SbtWithRetry -Commands @("package")

  if (-not $RefreshClasspath -and (Test-Path $CachePath)) {
    $cached = (Get-Content -Path $CachePath -Raw).Trim()
    if (Test-ClasspathLine -Value $cached) {
      return $cached
    }
  }

  $sbtOutput = & sbt --error "export Runtime / fullClasspathAsJars"
  $exitCode = $LASTEXITCODE
  if ($exitCode -ne 0) {
    throw "Failed to export runtime classpath (exit code $exitCode)"
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

function Resolve-HeroStyles {
  param([string]$Csv)

  $allowed = @("adaptive", "gto")
  $styles = @(Split-TrimCsv -Value $Csv)
  if ($styles.Count -eq 0) {
    throw "HeroStyles produced no values."
  }
  foreach ($style in $styles) {
    if ($allowed -notcontains $style) {
      throw "HeroStyles contains invalid value '$style'. Allowed: adaptive,gto."
    }
  }
  return $styles
}

function Resolve-VillainStyles {
  param([string]$Csv)

  $allowed = @("nit", "tag", "lag", "callingstation", "station", "maniac", "gto")
  $styles = @(Split-TrimCsv -Value $Csv)
  if ($styles.Count -eq 0) {
    throw "VillainStyles produced no values."
  }
  foreach ($style in $styles) {
    if ($allowed -notcontains $style) {
      throw "VillainStyles contains invalid value '$style'. Allowed: nit,tag,lag,callingstation,station,maniac,gto."
    }
  }
  return $styles
}

function Get-NativeProfileJvmProps {
  param([string]$Profile)

  switch ($Profile) {
    "cpu" {
      return @(
        "-Dsicfun.bayes.provider=native-cpu",
        "-Dsicfun.postflop.provider=native",
        "-Dsicfun.postflop.native.engine=cpu",
        "-Dsicfun.gpu.provider=disabled"
      )
    }
    "gpu" {
      return @(
        "-Dsicfun.bayes.provider=native-gpu",
        "-Dsicfun.postflop.provider=native",
        "-Dsicfun.postflop.native.engine=cuda",
        "-Dsicfun.gpu.provider=native"
      )
    }
    default {
      return @(
        "-Dsicfun.bayes.provider=auto",
        "-Dsicfun.postflop.provider=auto",
        "-Dsicfun.gpu.provider=native"
      )
    }
  }
}

function Get-SafeLabel {
  param([string]$Value)

  $label = $Value.ToLowerInvariant()
  $label = [regex]::Replace($label, "[^a-z0-9]+", "-")
  return $label.Trim("-")
}

function Get-SeatLabels {
  param([string]$Mode)

  switch ($Mode) {
    "single" { return @("button") }
    "seat-normalized" { return @("button", "bigblind") }
    default { throw "Unsupported SeatMode '$Mode'." }
  }
}

function Read-MatchupMetrics {
  param(
    [string]$HandsPath,
    [string]$HeroStyle,
    [string]$VillainStyle,
    [string]$HeroSeat,
    [string]$RunDir,
    [long]$RunSeed,
    [double]$RuntimeSeconds,
    [string]$Status
  )

  if (-not (Test-Path $HandsPath)) {
    return [pscustomobject]@{
      heroStyle = $HeroStyle
      villainStyle = $VillainStyle
      heroSeat = $HeroSeat
      hands = 0
      heroNetChips = [double]::NaN
      heroBbPer100 = [double]::NaN
      heroWins = 0
      heroTies = 0
      heroLosses = 0
      heroWinRate = [double]::NaN
      avgStreetsPlayed = [double]::NaN
      modelId = ""
      seed = $RunSeed
      runtimeSeconds = $RuntimeSeconds
      status = $Status
      outDir = $RunDir
    }
  }

  $rows = Import-Csv -Path $HandsPath -Delimiter "`t"
  $hands = @($rows).Count
  $heroNet = 0.0
  $wins = 0
  $ties = 0
  $losses = 0
  $streetsTotal = 0.0
  $modelId = ""

  foreach ($row in $rows) {
    $heroNet += [double]$row.heroNet
    $streetsTotal += [double]$row.streetsPlayed
    if ([string]::IsNullOrWhiteSpace($modelId)) {
      $modelId = [string]$row.modelId
    }
    switch ([string]$row.outcome) {
      "win" { $wins += 1 }
      "tie" { $ties += 1 }
      "loss" { $losses += 1 }
    }
  }

  $bbPer100 = if ($hands -gt 0) { ($heroNet / [double]$hands) * 100.0 } else { [double]::NaN }
  $winRate = if ($hands -gt 0) { $wins / [double]$hands } else { [double]::NaN }
  $avgStreets = if ($hands -gt 0) { $streetsTotal / [double]$hands } else { [double]::NaN }

  return [pscustomobject]@{
    heroStyle = $HeroStyle
    villainStyle = $VillainStyle
    heroSeat = $HeroSeat
    hands = $hands
    heroNetChips = [math]::Round($heroNet, 4)
    heroBbPer100 = [math]::Round($bbPer100, 3)
    heroWins = $wins
    heroTies = $ties
    heroLosses = $losses
    heroWinRate = [math]::Round($winRate, 4)
    avgStreetsPlayed = [math]::Round($avgStreets, 3)
    modelId = $modelId
    seed = $RunSeed
    runtimeSeconds = [math]::Round($RuntimeSeconds, 3)
    status = $Status
    outDir = $RunDir
  }
}

function New-AggregateMatchupResult {
  param(
    [string]$HeroStyle,
    [string]$VillainStyle,
    [string]$SeatMode,
    [object[]]$LegRows
  )

  $buttonRow = $LegRows | Where-Object { $_.heroSeat -eq "button" } | Select-Object -First 1
  $bigBlindRow = $LegRows | Where-Object { $_.heroSeat -eq "bigblind" } | Select-Object -First 1
  $completedRows = @($LegRows | Where-Object { $_.status -eq "ok" })
  $completedLegs = $completedRows.Count
  $statusDetail = ($LegRows | ForEach-Object { "{0}:{1}" -f $_.heroSeat, $_.status }) -join "|"
  $modelIds = @(
    $completedRows |
      Where-Object { -not [string]::IsNullOrWhiteSpace($_.modelId) } |
      ForEach-Object { $_.modelId } |
      Select-Object -Unique
  )
  $modelId =
    if ($modelIds.Count -gt 0) { $modelIds -join "|" }
    else { "" }

  $buttonBbPer100 = if ($null -ne $buttonRow) { [double]$buttonRow.heroBbPer100 } else { [double]::NaN }
  $bigBlindBbPer100 = if ($null -ne $bigBlindRow) { [double]$bigBlindRow.heroBbPer100 } else { [double]::NaN }
  $buttonWinRate = if ($null -ne $buttonRow) { [double]$buttonRow.heroWinRate } else { [double]::NaN }
  $bigBlindWinRate = if ($null -ne $bigBlindRow) { [double]$bigBlindRow.heroWinRate } else { [double]::NaN }
  $buttonOutDir = if ($null -ne $buttonRow) { $buttonRow.outDir } else { "" }
  $bigBlindOutDir = if ($null -ne $bigBlindRow) { $bigBlindRow.outDir } else { "" }
  $buttonSeed = if ($null -ne $buttonRow) { $buttonRow.seed } else { "" }
  $bigBlindSeed = if ($null -ne $bigBlindRow) { $bigBlindRow.seed } else { "" }

  if ($SeatMode -eq "single") {
    $row = $LegRows | Select-Object -First 1
    return [pscustomobject]@{
      heroStyle = $HeroStyle
      villainStyle = $VillainStyle
      seatMode = $SeatMode
      completedLegs = $completedLegs
      hands = $row.hands
      heroNetChips = $row.heroNetChips
      heroBbPer100 = $row.heroBbPer100
      heroWins = $row.heroWins
      heroTies = $row.heroTies
      heroLosses = $row.heroLosses
      heroWinRate = $row.heroWinRate
      avgStreetsPlayed = $row.avgStreetsPlayed
      modelId = $row.modelId
      runtimeSeconds = $row.runtimeSeconds
      status = $row.status
      statusDetail = $statusDetail
      buttonBbPer100 = $buttonBbPer100
      bigBlindBbPer100 = $bigBlindBbPer100
      buttonWinRate = $buttonWinRate
      bigBlindWinRate = $bigBlindWinRate
      buttonSeed = $buttonSeed
      bigBlindSeed = $bigBlindSeed
      buttonOutDir = $buttonOutDir
      bigBlindOutDir = $bigBlindOutDir
    }
  }

  if ($completedLegs -eq 2 -and $null -ne $buttonRow -and $null -ne $bigBlindRow) {
    $handsTotal = [int]$buttonRow.hands + [int]$bigBlindRow.hands
    $heroNetTotal = [double]$buttonRow.heroNetChips + [double]$bigBlindRow.heroNetChips
    $heroWins = [int]$buttonRow.heroWins + [int]$bigBlindRow.heroWins
    $heroTies = [int]$buttonRow.heroTies + [int]$bigBlindRow.heroTies
    $heroLosses = [int]$buttonRow.heroLosses + [int]$bigBlindRow.heroLosses
    $runtimeSeconds = [double]$buttonRow.runtimeSeconds + [double]$bigBlindRow.runtimeSeconds
    $streetsWeighted = ([double]$buttonRow.avgStreetsPlayed * [double]$buttonRow.hands) + ([double]$bigBlindRow.avgStreetsPlayed * [double]$bigBlindRow.hands)
    $heroBbPer100 = if ($handsTotal -gt 0) { ($heroNetTotal / [double]$handsTotal) * 100.0 } else { [double]::NaN }
    $heroWinRate = if ($handsTotal -gt 0) { $heroWins / [double]$handsTotal } else { [double]::NaN }
    $avgStreetsPlayed = if ($handsTotal -gt 0) { $streetsWeighted / [double]$handsTotal } else { [double]::NaN }

    return [pscustomobject]@{
      heroStyle = $HeroStyle
      villainStyle = $VillainStyle
      seatMode = $SeatMode
      completedLegs = $completedLegs
      hands = $handsTotal
      heroNetChips = [math]::Round($heroNetTotal, 4)
      heroBbPer100 = [math]::Round($heroBbPer100, 3)
      heroWins = $heroWins
      heroTies = $heroTies
      heroLosses = $heroLosses
      heroWinRate = [math]::Round($heroWinRate, 4)
      avgStreetsPlayed = [math]::Round($avgStreetsPlayed, 3)
      modelId = $modelId
      runtimeSeconds = [math]::Round($runtimeSeconds, 3)
      status = "ok"
      statusDetail = $statusDetail
      buttonBbPer100 = $buttonBbPer100
      bigBlindBbPer100 = $bigBlindBbPer100
      buttonWinRate = $buttonWinRate
      bigBlindWinRate = $bigBlindWinRate
      buttonSeed = $buttonSeed
      bigBlindSeed = $bigBlindSeed
      buttonOutDir = $buttonOutDir
      bigBlindOutDir = $bigBlindOutDir
    }
  }

  $status =
    if ($completedLegs -eq 0) { "failed" }
    else { "partial" }

  return [pscustomobject]@{
    heroStyle = $HeroStyle
    villainStyle = $VillainStyle
    seatMode = $SeatMode
    completedLegs = $completedLegs
    hands = @($completedRows | Measure-Object -Property hands -Sum).Sum
    heroNetChips = [double]::NaN
    heroBbPer100 = [double]::NaN
    heroWins = 0
    heroTies = 0
    heroLosses = 0
    heroWinRate = [double]::NaN
    avgStreetsPlayed = [double]::NaN
    modelId = $modelId
    runtimeSeconds = [math]::Round((@($LegRows | Measure-Object -Property runtimeSeconds -Sum).Sum), 3)
    status = $status
    statusDetail = $statusDetail
    buttonBbPer100 = $buttonBbPer100
    bigBlindBbPer100 = $bigBlindBbPer100
    buttonWinRate = $buttonWinRate
    bigBlindWinRate = $bigBlindWinRate
    buttonSeed = $buttonSeed
    bigBlindSeed = $bigBlindSeed
    buttonOutDir = $buttonOutDir
    bigBlindOutDir = $bigBlindOutDir
  }
}

function Write-LegResultsTsv {
  param(
    [object[]]$Rows,
    [string]$Path
  )

  $header = @(
    "heroStyle",
    "villainStyle",
    "heroSeat",
    "hands",
    "heroNetChips",
    "heroBbPer100",
    "heroWins",
    "heroTies",
    "heroLosses",
    "heroWinRate",
    "avgStreetsPlayed",
    "modelId",
    "seed",
    "runtimeSeconds",
    "status",
    "outDir"
  ) -join "`t"

  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add($header) | Out-Null

  foreach ($row in $Rows) {
    $line = @(
      $row.heroStyle,
      $row.villainStyle,
      $row.heroSeat,
      $row.hands,
      (Format-InvariantNumber -Value ([double]$row.heroNetChips) -Digits 4),
      (Format-InvariantNumber -Value ([double]$row.heroBbPer100) -Digits 3),
      $row.heroWins,
      $row.heroTies,
      $row.heroLosses,
      (Format-InvariantNumber -Value ([double]$row.heroWinRate) -Digits 4),
      (Format-InvariantNumber -Value ([double]$row.avgStreetsPlayed) -Digits 3),
      $row.modelId,
      $row.seed,
      (Format-InvariantNumber -Value ([double]$row.runtimeSeconds) -Digits 3),
      $row.status,
      $row.outDir
    ) -join "`t"
    $lines.Add($line) | Out-Null
  }

  Set-Content -Path $Path -Value $lines -Encoding UTF8
}

function Write-ResultsTsv {
  param(
    [object[]]$Rows,
    [string]$Path
  )

  $header = @(
    "heroStyle",
    "villainStyle",
    "seatMode",
    "completedLegs",
    "hands",
    "heroNetChips",
    "heroBbPer100",
    "heroWins",
    "heroTies",
    "heroLosses",
    "heroWinRate",
    "avgStreetsPlayed",
    "modelId",
    "buttonSeed",
    "bigBlindSeed",
    "runtimeSeconds",
    "status",
    "statusDetail",
    "buttonBbPer100",
    "bigBlindBbPer100",
    "buttonWinRate",
    "bigBlindWinRate",
    "buttonOutDir",
    "bigBlindOutDir"
  ) -join "`t"

  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add($header) | Out-Null

  foreach ($row in $Rows) {
    $line = @(
      $row.heroStyle,
      $row.villainStyle,
      $row.seatMode,
      $row.completedLegs,
      $row.hands,
      (Format-InvariantNumber -Value ([double]$row.heroNetChips) -Digits 4),
      (Format-InvariantNumber -Value ([double]$row.heroBbPer100) -Digits 3),
      $row.heroWins,
      $row.heroTies,
      $row.heroLosses,
      (Format-InvariantNumber -Value ([double]$row.heroWinRate) -Digits 4),
      (Format-InvariantNumber -Value ([double]$row.avgStreetsPlayed) -Digits 3),
      $row.modelId,
      $row.buttonSeed,
      $row.bigBlindSeed,
      (Format-InvariantNumber -Value ([double]$row.runtimeSeconds) -Digits 3),
      $row.status,
      $row.statusDetail,
      (Format-InvariantNumber -Value ([double]$row.buttonBbPer100) -Digits 3),
      (Format-InvariantNumber -Value ([double]$row.bigBlindBbPer100) -Digits 3),
      (Format-InvariantNumber -Value ([double]$row.buttonWinRate) -Digits 4),
      (Format-InvariantNumber -Value ([double]$row.bigBlindWinRate) -Digits 4),
      $row.buttonOutDir,
      $row.bigBlindOutDir
    ) -join "`t"
    $lines.Add($line) | Out-Null
  }

  Set-Content -Path $Path -Value $lines -Encoding UTF8
}

function Write-Summary {
  param(
    [object[]]$Rows,
    [string]$Path,
    [int]$Hands,
    [string]$GtoMode,
    [string]$NativeProfile,
    [string]$SeatMode
  )

  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("SICFUN hall matchup summary") | Out-Null
  $lines.Add("generatedAt: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")") | Out-Null
  $lines.Add("handsPerMatchup: $Hands") | Out-Null
  $lines.Add("gtoMode: $GtoMode") | Out-Null
  $lines.Add("nativeProfile: $NativeProfile") | Out-Null
  $lines.Add("seatMode: $SeatMode") | Out-Null
  $lines.Add("") | Out-Null

  $completed = @($Rows | Where-Object { $_.status -eq "ok" })
  if ($completed.Count -gt 0) {
    $lines.Add("Leaderboard by heroBbPer100:") | Out-Null
    foreach ($row in ($completed | Sort-Object heroBbPer100 -Descending)) {
      if ($row.seatMode -eq "seat-normalized") {
        $lines.Add(("{0} vs {1}: bb/100={2}, net={3}, winRate={4}, runtime={5}s, button={6}, bigBlind={7}" -f `
          $row.heroStyle,
          $row.villainStyle,
          (Format-InvariantNumber -Value ([double]$row.heroBbPer100) -Digits 3),
          (Format-InvariantNumber -Value ([double]$row.heroNetChips) -Digits 4),
          (Format-InvariantNumber -Value ([double]$row.heroWinRate) -Digits 4),
          (Format-InvariantNumber -Value ([double]$row.runtimeSeconds) -Digits 3),
          (Format-InvariantNumber -Value ([double]$row.buttonBbPer100) -Digits 3),
          (Format-InvariantNumber -Value ([double]$row.bigBlindBbPer100) -Digits 3))) | Out-Null
      }
      else {
        $lines.Add(("{0} vs {1}: bb/100={2}, net={3}, winRate={4}, runtime={5}s" -f `
          $row.heroStyle,
          $row.villainStyle,
          (Format-InvariantNumber -Value ([double]$row.heroBbPer100) -Digits 3),
          (Format-InvariantNumber -Value ([double]$row.heroNetChips) -Digits 4),
          (Format-InvariantNumber -Value ([double]$row.heroWinRate) -Digits 4),
          (Format-InvariantNumber -Value ([double]$row.runtimeSeconds) -Digits 3))) | Out-Null
      }
    }
    $lines.Add("") | Out-Null
    $lines.Add("Hero-style aggregates:") | Out-Null
    foreach ($group in ($completed | Group-Object heroStyle | Sort-Object Name)) {
      $avgBb = ($group.Group | Measure-Object -Property heroBbPer100 -Average).Average
      $avgWinRate = ($group.Group | Measure-Object -Property heroWinRate -Average).Average
      $positive = @($group.Group | Where-Object { $_.heroBbPer100 -gt 0.0 }).Count
      $lines.Add(("{0}: avg bb/100={1}, avg winRate={2}, positive matchups={3}/{4}" -f `
        $group.Name,
        (Format-InvariantNumber -Value ([double]$avgBb) -Digits 3),
        (Format-InvariantNumber -Value ([double]$avgWinRate) -Digits 4),
        $positive,
        $group.Count)) | Out-Null
    }
  }
  else {
    $lines.Add("No successful matchups.") | Out-Null
  }

  $failed = @($Rows | Where-Object { $_.status -ne "ok" })
  if ($failed.Count -gt 0) {
    $lines.Add("") | Out-Null
    $lines.Add("Failed matchups:") | Out-Null
    foreach ($row in $failed) {
      $lines.Add(("{0} vs {1}: status={2}, detail={3}" -f $row.heroStyle, $row.villainStyle, $row.status, $row.statusDetail)) | Out-Null
    }
  }

  Set-Content -Path $Path -Value $lines -Encoding UTF8
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$previousSbtOpts = $null
Push-Location $repoRoot
try {
  $previousSbtOpts = $env:SBT_OPTS
  $env:SBT_OPTS = "-Dsbt.server.autostart=false"

  $resolvedHeroStyles = @(Resolve-HeroStyles -Csv $HeroStyles)
  $resolvedVillainStyles = @(Resolve-VillainStyles -Csv $VillainStyles)
  $saveTrainingTsvLiteral = Resolve-BoolLiteral -Value $SaveTrainingTsv -Name "SaveTrainingTsv"
  $saveDdreTrainingTsvLiteral = Resolve-BoolLiteral -Value $SaveDdreTrainingTsv -Name "SaveDdreTrainingTsv"

  if ($Hands -le 0) { throw "Hands must be > 0." }
  if ($TableCount -le 0) { throw "TableCount must be > 0." }
  if ($ReportEvery -le 0) { throw "ReportEvery must be > 0." }
  if ($LearnEveryHands -lt 0) { throw "LearnEveryHands must be >= 0." }
  if ($BunchingTrials -le 0) { throw "BunchingTrials must be > 0." }
  if ($EquityTrials -le 0) { throw "EquityTrials must be > 0." }

  $cacheDir = Join-Path $repoRoot "data"
  if (-not (Test-Path $cacheDir)) {
    New-Item -ItemType Directory -Path $cacheDir | Out-Null
  }
  $classpathCache = Join-Path $cacheDir "runtime-classpath.txt"
  $classpath = Resolve-RuntimeClasspath -CachePath $classpathCache
  $jvmProps = @(Get-NativeProfileJvmProps -Profile $NativeProfile)

  $resolvedOutDir = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutDir))
  if (-not (Test-Path $resolvedOutDir)) {
    New-Item -ItemType Directory -Path $resolvedOutDir | Out-Null
  }

  $seatLabels = @(Get-SeatLabels -Mode $SeatMode)
  $legResults = New-Object System.Collections.Generic.List[object]
  $results = New-Object System.Collections.Generic.List[object]
  $matchups = New-Object System.Collections.Generic.List[object]
  foreach ($heroStyle in $resolvedHeroStyles) {
    foreach ($villainStyle in $resolvedVillainStyles) {
      $matchups.Add([pscustomobject]@{
        heroStyle = $heroStyle
        villainStyle = $villainStyle
      }) | Out-Null
    }
  }

  $totalMatchups = $matchups.Count
  for ($index = 0; $index -lt $totalMatchups; $index++) {
    $matchup = $matchups[$index]
    $labelBase = "{0:D2}-{1}-vs-{2}" -f ($index + 1), (Get-SafeLabel -Value $matchup.heroStyle), (Get-SafeLabel -Value $matchup.villainStyle)
    $matchupLegRows = New-Object System.Collections.Generic.List[object]

    for ($seatIndex = 0; $seatIndex -lt $seatLabels.Count; $seatIndex++) {
      $heroSeat = $seatLabels[$seatIndex]
      $runSeed = [long]($Seed + ($index * 19946) + $seatIndex)
      $label =
        if ($SeatMode -eq "single") { $labelBase }
        else { "{0}-{1}" -f $labelBase, $heroSeat }
      $runDir = Join-Path $resolvedOutDir $label
      $handsPath = Join-Path $runDir "hands.tsv"
      $stdoutPath = Join-Path $runDir "stdout.log"
      $stderrPath = Join-Path $runDir "stderr.log"

      if ($SkipExisting -and (Test-Path $handsPath)) {
        Write-Host ("[{0}/{1}] Skipping existing {2} vs {3} (seat={4})" -f ($index + 1), $totalMatchups, $matchup.heroStyle, $matchup.villainStyle, $heroSeat)
        $result = Read-MatchupMetrics -HandsPath $handsPath -HeroStyle $matchup.heroStyle -VillainStyle $matchup.villainStyle -HeroSeat $heroSeat -RunDir $runDir -RunSeed $runSeed -RuntimeSeconds 0.0 -Status "ok"
        $legResults.Add($result) | Out-Null
        $matchupLegRows.Add($result) | Out-Null
        continue
      }

      if (-not (Test-Path $runDir)) {
        New-Item -ItemType Directory -Path $runDir | Out-Null
      }

      Write-Host ("[{0}/{1}] Running {2} vs {3} (seat={4}, seed={5})" -f ($index + 1), $totalMatchups, $matchup.heroStyle, $matchup.villainStyle, $heroSeat, $runSeed)

      $args = @(
        "--hands=$Hands",
        "--tableCount=$TableCount",
        "--reportEvery=$ReportEvery",
        "--learnEveryHands=$LearnEveryHands",
        "--learningWindowSamples=$LearningWindowSamples",
        "--seed=$runSeed",
        "--outDir=$runDir",
        "--heroStyle=$($matchup.heroStyle)",
        "--heroSeat=$heroSeat",
        "--gtoMode=$GtoMode",
        "--villainStyle=$($matchup.villainStyle)",
        "--heroExplorationRate=$HeroExplorationRate",
        "--raiseSize=$RaiseSize",
        "--bunchingTrials=$BunchingTrials",
        "--equityTrials=$EquityTrials",
        "--saveTrainingTsv=$saveTrainingTsvLiteral",
        "--saveDdreTrainingTsv=$saveDdreTrainingTsvLiteral"
      )
      if (-not [string]::IsNullOrWhiteSpace($ModelArtifactDir)) {
        $args += "--modelArtifactDir=$ModelArtifactDir"
      }

      $status = "ok"
      $runtimeSeconds = 0.0
      $sw = [System.Diagnostics.Stopwatch]::StartNew()
      try {
        & java @jvmProps -cp $classpath "sicfun.holdem.runtime.TexasHoldemPlayingHall" @args 1> $stdoutPath 2> $stderrPath
        if ($LASTEXITCODE -ne 0) {
          $status = "exit-$LASTEXITCODE"
        }
      }
      catch {
        $status = "exception"
        $_ | Out-File -FilePath $stderrPath -Append
        if ($FailFast) {
          throw
        }
      }
      finally {
        $sw.Stop()
        $runtimeSeconds = $sw.Elapsed.TotalSeconds
      }

      if ($status -ne "ok" -and $FailFast) {
        throw "Matchup failed: $($matchup.heroStyle) vs $($matchup.villainStyle) seat=$heroSeat ($status)"
      }

      $result = Read-MatchupMetrics -HandsPath $handsPath -HeroStyle $matchup.heroStyle -VillainStyle $matchup.villainStyle -HeroSeat $heroSeat -RunDir $runDir -RunSeed $runSeed -RuntimeSeconds $runtimeSeconds -Status $status
      $legResults.Add($result) | Out-Null
      $matchupLegRows.Add($result) | Out-Null

      if ($status -eq "ok") {
        Write-Host ("  seat={0} bb/100={1} net={2} winRate={3} runtime={4}s" -f `
          $heroSeat,
          (Format-InvariantNumber -Value ([double]$result.heroBbPer100) -Digits 3),
          (Format-InvariantNumber -Value ([double]$result.heroNetChips) -Digits 4),
          (Format-InvariantNumber -Value ([double]$result.heroWinRate) -Digits 4),
          (Format-InvariantNumber -Value ([double]$result.runtimeSeconds) -Digits 3))
      }
      else {
        Write-Host ("  seat={0} failed with status={1}" -f $heroSeat, $status)
      }
    }

    $aggregateResult = New-AggregateMatchupResult -HeroStyle $matchup.heroStyle -VillainStyle $matchup.villainStyle -SeatMode $SeatMode -LegRows $matchupLegRows.ToArray()
    $results.Add($aggregateResult) | Out-Null

    if ($aggregateResult.status -eq "ok" -and $SeatMode -eq "seat-normalized") {
      Write-Host ("  seat-normalized bb/100={0} net={1} winRate={2} button={3} bigBlind={4}" -f `
        (Format-InvariantNumber -Value ([double]$aggregateResult.heroBbPer100) -Digits 3),
        (Format-InvariantNumber -Value ([double]$aggregateResult.heroNetChips) -Digits 4),
        (Format-InvariantNumber -Value ([double]$aggregateResult.heroWinRate) -Digits 4),
        (Format-InvariantNumber -Value ([double]$aggregateResult.buttonBbPer100) -Digits 3),
        (Format-InvariantNumber -Value ([double]$aggregateResult.bigBlindBbPer100) -Digits 3))
    }
    elseif ($aggregateResult.status -ne "ok") {
      Write-Host ("  aggregate status={0} detail={1}" -f $aggregateResult.status, $aggregateResult.statusDetail)
    }
  }

  $legResultsArray = $legResults.ToArray()
  $resultsArray = $results.ToArray()
  $legResultsPath = Join-Path $resolvedOutDir "seat-legs.tsv"
  $resultsPath = Join-Path $resolvedOutDir "results.tsv"
  $leaderboardPath = Join-Path $resolvedOutDir "leaderboard.tsv"
  $summaryPath = Join-Path $resolvedOutDir "summary.txt"

  Write-LegResultsTsv -Rows $legResultsArray -Path $legResultsPath
  Write-ResultsTsv -Rows $resultsArray -Path $resultsPath
  $leaderboardRows = @($resultsArray | Sort-Object status, @{ Expression = "heroBbPer100"; Descending = $true })
  Write-ResultsTsv -Rows $leaderboardRows -Path $leaderboardPath
  Write-Summary -Rows $resultsArray -Path $summaryPath -Hands $Hands -GtoMode $GtoMode -NativeProfile $NativeProfile -SeatMode $SeatMode

  Write-Host ""
  Write-Host "Matchup benchmark complete."
  Write-Host "seat legs: $legResultsPath"
  Write-Host "results: $resultsPath"
  Write-Host "leaderboard: $leaderboardPath"
  Write-Host "summary: $summaryPath"
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
