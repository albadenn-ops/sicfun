[CmdletBinding()]
param(
  [int]$Hands = 10000000,
  [int]$Workers = 0,
  [int]$TableCountPerWorker = 1,
  [int]$ReportEvery = 100000,
  [int]$LearnEveryHands = 0,
  [int]$LearningWindowSamples = 200000,
  [long]$Seed = 42,
  [string]$OutDir = "data/bench-hall-max",
  [ValidateSet("adaptive", "gto")]
  [string]$HeroStyle = "adaptive",
  [ValidateSet("fast", "exact")]
  [string]$GtoMode = "exact",
  [ValidateSet("nit", "tag", "lag", "callingstation", "station", "maniac", "gto")]
  [string]$VillainStyle = "gto",
  [double]$HeroExplorationRate = 0.00,
  [double]$RaiseSize = 2.5,
  [int]$BunchingTrials = 80,
  [int]$EquityTrials = 700,
  [object]$SaveTrainingTsv = $false,
  [object]$SaveDdreTrainingTsv = $false,
  [ValidateSet("auto", "cpu", "gpu")]
  [string]$NativeProfile = "auto",
  [string[]]$JvmOption = @(),
  [switch]$RefreshClasspath,
  [switch]$FailFast,
  [int]$ProgressSeconds = 15,
  [switch]$AutoTune,
  [int]$AutoTuneHands = 200000,
  [string]$AutoTuneProfiles = "auto,cpu,gpu",
  [string]$AutoTuneWorkerCandidates = ""
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

function Get-HandsDistribution {
  param(
    [int]$TotalHands,
    [int]$WorkerCount
  )

  $base = [int]([math]::Floor($TotalHands / [double]$WorkerCount))
  $extra = $TotalHands % $WorkerCount
  $shares = New-Object System.Collections.Generic.List[int]
  for ($i = 0; $i -lt $WorkerCount; $i++) {
    $share = $base
    if ($i -lt $extra) {
      $share += 1
    }
    $shares.Add($share)
  }
  return ,$shares.ToArray()
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

function Get-WorkerFailureReason {
  param([object]$Worker)

  $handsLog = Join-Path $Worker.WorkerDir "hands.tsv"
  $stdoutText =
    if (Test-Path $Worker.StdOutPath) { (Get-Content -Path $Worker.StdOutPath -Raw) }
    else { "" }
  $stderrText =
    if (Test-Path $Worker.StdErrPath) { (Get-Content -Path $Worker.StdErrPath -Raw) }
    else { "" }

  $hasSummary = $stdoutText -match "handsPlayed:\s+\d+"
  $hasHandsLog = Test-Path $handsLog
  if ($hasSummary -and $hasHandsLog) {
    return $null
  }

  if (-not [string]::IsNullOrWhiteSpace($stderrText)) {
    $firstLine = ($stderrText -split "(`r`n|`n|`r)", 2)[0].Trim()
    if (-not [string]::IsNullOrWhiteSpace($firstLine)) {
      return $firstLine
    }
    return "stderr output present"
  }
  if (-not $hasHandsLog) {
    return "hands.tsv missing"
  }
  return "missing hall summary output"
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

function Resolve-AutoTuneProfiles {
  param([string]$Csv)

  $allowed = @("auto", "cpu", "gpu")
  $values = Split-TrimCsv -Value $Csv
  if ($values.Count -eq 0) {
    throw "AutoTuneProfiles produced no values."
  }

  $set = New-Object 'System.Collections.Generic.HashSet[string]'
  foreach ($item in $values) {
    if ($allowed -notcontains $item) {
      throw "AutoTuneProfiles contains invalid value '$item'. Allowed: auto,cpu,gpu."
    }
    $null = $set.Add($item)
  }
  return ,@($set)
}

function Resolve-AutoTuneWorkerCandidates {
  param(
    [int]$RequestedWorkers,
    [string]$Csv,
    [int]$CpuCount
  )

  $set = New-Object 'System.Collections.Generic.HashSet[int]'
  if (-not [string]::IsNullOrWhiteSpace($Csv)) {
    $tokens = Split-TrimCsv -Value $Csv
    foreach ($token in $tokens) {
      $parsed = 0
      if (-not [int]::TryParse($token, [ref]$parsed) -or $parsed -le 0) {
        throw "AutoTuneWorkerCandidates contains invalid value '$token'. Must be comma-separated positive integers."
      }
      $null = $set.Add($parsed)
    }
  }
  else {
    $base = if ($RequestedWorkers -gt 0) { $RequestedWorkers } else { $CpuCount }
    $suggested = @(
      1,
      [Math]::Max(1, [int][Math]::Floor($base * 0.5)),
      [Math]::Max(1, [int][Math]::Ceiling($base * 0.75)),
      [Math]::Max(1, [int]$base),
      [Math]::Max(1, [int][Math]::Ceiling($base * 1.25)),
      [Math]::Max(1, [int][Math]::Ceiling($base * 1.5))
    )
    foreach ($value in $suggested) {
      $null = $set.Add($value)
    }
  }

  $arr = @($set)
  [Array]::Sort($arr)
  return ,$arr
}

function Invoke-HallParallelRun {
  param(
    [string]$RepoRoot,
    [string]$RootOutDir,
    [string]$RunLabel,
    [int]$TotalHands,
    [int]$WorkerCount,
    [int]$TableCountPerWorker,
    [int]$ReportEvery,
    [int]$LearnEveryHands,
    [int]$LearningWindowSamples,
    [long]$Seed,
    [string]$HeroStyle,
    [string]$GtoMode,
    [string]$VillainStyle,
    [double]$HeroExplorationRate,
    [double]$RaiseSize,
    [int]$BunchingTrials,
    [int]$EquityTrials,
    [string]$SaveTrainingLiteral,
    [string]$SaveDdreTrainingLiteral,
    [string]$NativeProfile,
    [string[]]$JvmOption,
    [string]$Classpath,
    [switch]$FailFast,
    [int]$ProgressSeconds,
    [switch]$SuppressProgress
  )

  if ($TotalHands -le 0) {
    throw "TotalHands must be > 0."
  }
  if ($WorkerCount -le 0) {
    throw "WorkerCount must be > 0."
  }
  if ($TableCountPerWorker -le 0) {
    throw "TableCountPerWorker must be > 0."
  }
  if ($ReportEvery -le 0) {
    throw "ReportEvery must be > 0."
  }

  $workersState = New-Object System.Collections.Generic.List[object]
  $runOutDir = $null
  try {
    $handsDistribution = Get-HandsDistribution -TotalHands $TotalHands -WorkerCount $WorkerCount
    $activeWorkers = @($handsDistribution | Where-Object { $_ -gt 0 }).Count
    if ($activeWorkers -eq 0) {
      throw "No active workers after hand distribution."
    }

    $runStamp = Get-Date -Format "yyyyMMdd-HHmmss-fff"
    $runOutDir = Join-Path $RootOutDir ("$RunLabel-$runStamp")
    New-Item -ItemType Directory -Path $runOutDir -Force | Out-Null

    $profileJvmProps = Get-NativeProfileJvmProps -Profile $NativeProfile
    $sharedHallArgs = @(
      "--tableCount=$TableCountPerWorker",
      "--reportEvery=$ReportEvery",
      "--learnEveryHands=$LearnEveryHands",
      "--learningWindowSamples=$LearningWindowSamples",
      "--heroStyle=$HeroStyle",
      "--gtoMode=$GtoMode",
      "--villainStyle=$VillainStyle",
      "--heroExplorationRate=$HeroExplorationRate",
      "--raiseSize=$RaiseSize",
      "--bunchingTrials=$BunchingTrials",
      "--equityTrials=$EquityTrials",
      "--saveTrainingTsv=$SaveTrainingLiteral",
      "--saveDdreTrainingTsv=$SaveDdreTrainingLiteral"
    )

    if (-not $SuppressProgress) {
      Write-Host "[hall-max] launching workers=$activeWorkers/$WorkerCount hands=$TotalHands profile=$NativeProfile outDir=$runOutDir"
    }

    for ($i = 0; $i -lt $WorkerCount; $i++) {
      $workerHands = [int]$handsDistribution[$i]
      if ($workerHands -le 0) {
        continue
      }

      $workerIndex = $workersState.Count + 1
      $workerSeed = $Seed + (1000003L * $workerIndex)
      $workerDir = Join-Path $runOutDir ("worker-" + $workerIndex.ToString("D2"))
      New-Item -ItemType Directory -Path $workerDir -Force | Out-Null
      $stdoutPath = Join-Path $workerDir "stdout.log"
      $stderrPath = Join-Path $workerDir "stderr.log"

      $javaArgs = New-Object System.Collections.Generic.List[string]
      foreach ($prop in $profileJvmProps) {
        $javaArgs.Add($prop)
      }
      if ($JvmOption.Count -gt 0) {
        foreach ($opt in $JvmOption) {
          $javaArgs.Add($opt)
        }
      }
      $javaArgs.Add("-cp")
      $javaArgs.Add($Classpath)
      $javaArgs.Add("sicfun.holdem.TexasHoldemPlayingHall")
      $javaArgs.Add("--hands=$workerHands")
      $javaArgs.Add("--seed=$workerSeed")
      $javaArgs.Add("--outDir=$workerDir")
      foreach ($arg in $sharedHallArgs) {
        $javaArgs.Add($arg)
      }

      $quotedArgs = $javaArgs | ForEach-Object {
        if ($_ -match "\s") { '"' + $_ + '"' } else { $_ }
      }
      $launchPreview = "java " + ($quotedArgs -join " ")
      Set-Content -Path (Join-Path $workerDir "launch-command.txt") -Value $launchPreview -Encoding utf8

      $proc = Start-Process -FilePath "java" `
        -ArgumentList ([string[]]$javaArgs.ToArray()) `
        -WorkingDirectory $RepoRoot `
        -NoNewWindow `
        -PassThru `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath

      $processInfo = Get-CimInstance Win32_Process -Filter ("ProcessId=" + $proc.Id) -ErrorAction SilentlyContinue
      if ($null -ne $processInfo -and -not [string]::IsNullOrWhiteSpace($processInfo.CommandLine)) {
        Set-Content -Path (Join-Path $workerDir "actual-commandline.txt") -Value $processInfo.CommandLine -Encoding utf8
      }

      $workersState.Add([pscustomobject]@{
        WorkerIndex = $workerIndex
        Hands = $workerHands
        Seed = $workerSeed
        WorkerDir = $workerDir
        StdOutPath = $stdoutPath
        StdErrPath = $stderrPath
        Process = $proc
      })
    }

    $startAt = Get-Date
    $nextProgress = $ProgressSeconds
    while ($true) {
      $running = @($workersState | Where-Object { -not $_.Process.HasExited })
      $completed = $workersState.Count - $running.Count
      $elapsed = (Get-Date) - $startAt

      if ((-not $SuppressProgress) -and $elapsed.TotalSeconds -ge $nextProgress) {
        Write-Host ("[hall-max] running={0} completed={1}/{2} elapsed={3:n1}s" -f $running.Count, $completed, $workersState.Count, $elapsed.TotalSeconds)
        $nextProgress += $ProgressSeconds
      }

      $failedNow = @()
      foreach ($worker in $workersState) {
        if ($worker.Process.HasExited) {
          $failureReason = Get-WorkerFailureReason -Worker $worker
          if ($null -ne $failureReason) {
            $worker | Add-Member -NotePropertyName FailureReason -NotePropertyValue $failureReason -Force
            $failedNow += $worker
          }
        }
      }
      if ($FailFast -and $failedNow.Count -gt 0 -and $running.Count -gt 0) {
        foreach ($worker in $running) {
          try { Stop-Process -Id $worker.Process.Id -Force -ErrorAction SilentlyContinue } catch { }
        }
        $fail = $failedNow[0]
        throw "worker-$($fail.WorkerIndex) failed early: $($fail.FailureReason) (stderr=$($fail.StdErrPath))"
      }

      if ($running.Count -eq 0) {
        break
      }

      Start-Sleep -Milliseconds 800
      foreach ($worker in $running) {
        $worker.Process.Refresh()
      }
    }

    $failed = New-Object System.Collections.Generic.List[object]
    foreach ($worker in $workersState) {
      $reason = Get-WorkerFailureReason -Worker $worker
      if ($null -ne $reason) {
        $worker | Add-Member -NotePropertyName FailureReason -NotePropertyValue $reason -Force
        $failed.Add($worker)
      }
    }
    if ($failed.Count -gt 0) {
      foreach ($item in $failed) {
        Write-Host ("[hall-max][error] worker-{0:D2} reason={1} stderr={2}" -f $item.WorkerIndex, $item.FailureReason, $item.StdErrPath)
      }
      throw "$($failed.Count) worker(s) failed."
    }

    $elapsedAll = (Get-Date) - $startAt
    $elapsedSeconds = [math]::Max($elapsedAll.TotalSeconds, 0.001)
    $handsPerSecond = $TotalHands / $elapsedSeconds
    $elapsedText = ([double]$elapsedSeconds).ToString("F3", [System.Globalization.CultureInfo]::InvariantCulture)
    $handsPerSecondText = ([double]$handsPerSecond).ToString("F2", [System.Globalization.CultureInfo]::InvariantCulture)
    $summaryPath = Join-Path $runOutDir "aggregate-summary.txt"
    $summaryLines = @(
      "workers=$($workersState.Count)",
      "hands=$TotalHands",
      "elapsedSeconds=$elapsedText",
      "handsPerSecond=$handsPerSecondText",
      "nativeProfile=$NativeProfile",
      "heroStyle=$HeroStyle",
      "gtoMode=$GtoMode",
      "villainStyle=$VillainStyle",
      "tableCountPerWorker=$TableCountPerWorker",
      "learnEveryHands=$LearnEveryHands",
      "saveTrainingTsv=$SaveTrainingLiteral",
      "saveDdreTrainingTsv=$SaveDdreTrainingLiteral"
    )
    Set-Content -Path $summaryPath -Value $summaryLines -Encoding utf8

    if (-not $SuppressProgress) {
      Write-Host "[hall-max] completed hands=$TotalHands in ${elapsedText}s (${handsPerSecondText} hands/s)"
      Write-Host "[hall-max] aggregate summary: $summaryPath"
    }

    return [pscustomobject]@{
      Success = $true
      Hands = $TotalHands
      Workers = $workersState.Count
      NativeProfile = $NativeProfile
      ElapsedSeconds = [double]$elapsedText
      HandsPerSecond = [double]$handsPerSecondText
      RunOutDir = $runOutDir
      SummaryPath = $summaryPath
      FailureReason = ""
    }
  }
  catch {
    foreach ($worker in $workersState) {
      if ($null -ne $worker.Process -and -not $worker.Process.HasExited) {
        try { Stop-Process -Id $worker.Process.Id -Force -ErrorAction SilentlyContinue } catch { }
      }
    }
    return [pscustomobject]@{
      Success = $false
      Hands = $TotalHands
      Workers = $WorkerCount
      NativeProfile = $NativeProfile
      ElapsedSeconds = 0.0
      HandsPerSecond = 0.0
      RunOutDir = $runOutDir
      SummaryPath = ""
      FailureReason = $_.Exception.Message
    }
  }
}

if ($Hands -le 0) {
  throw "Hands must be > 0."
}
if ($TableCountPerWorker -le 0) {
  throw "TableCountPerWorker must be > 0."
}
if ($ProgressSeconds -le 0) {
  throw "ProgressSeconds must be > 0."
}
if ($AutoTune -and $AutoTuneHands -le 0) {
  throw "AutoTuneHands must be > 0 when AutoTune is enabled."
}

$requestedWorkers = $Workers
if ($Workers -le 0) {
  $Workers = [Math]::Max(1, [Environment]::ProcessorCount)
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$previousSbtOpts = $null
Push-Location $repoRoot
try {
  $previousSbtOpts = $env:SBT_OPTS
  $env:SBT_OPTS = "-Dsbt.server.autostart=false"

  $saveTrainingLiteral = Resolve-BoolLiteral -Value $SaveTrainingTsv -Name "SaveTrainingTsv"
  $saveDdreTrainingLiteral = Resolve-BoolLiteral -Value $SaveDdreTrainingTsv -Name "SaveDdreTrainingTsv"

  if (-not (Test-Path $OutDir)) {
    New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
  }

  $cacheDir = Join-Path $repoRoot "data"
  if (-not (Test-Path $cacheDir)) {
    New-Item -ItemType Directory -Path $cacheDir | Out-Null
  }
  $classpathCache = Join-Path $cacheDir "runtime-classpath.txt"
  $classpath = Resolve-RuntimeClasspath -CachePath $classpathCache

  $selectedWorkers = $Workers
  $selectedNativeProfile = $NativeProfile

  if ($AutoTune) {
    Write-Host "[hall-max][autotune] starting with hands=$AutoTuneHands per candidate (learning/log exports forced off during tuning)"
    $autoTuneProfilesResolved = Resolve-AutoTuneProfiles -Csv $AutoTuneProfiles
    $autoTuneWorkersResolved = Resolve-AutoTuneWorkerCandidates `
      -RequestedWorkers $requestedWorkers `
      -Csv $AutoTuneWorkerCandidates `
      -CpuCount ([Math]::Max(1, [Environment]::ProcessorCount))

    $profilesPretty = ($autoTuneProfilesResolved | Sort-Object) -join ","
    $workersPretty = ($autoTuneWorkersResolved | Sort-Object) -join ","
    Write-Host "[hall-max][autotune] profiles=$profilesPretty workers=$workersPretty"

    $autoTuneRoot = Join-Path $OutDir "autotune"
    New-Item -ItemType Directory -Path $autoTuneRoot -Force | Out-Null

    $candidateRows = New-Object System.Collections.Generic.List[object]
    $candidateId = 0
    foreach ($profile in ($autoTuneProfilesResolved | Sort-Object)) {
      foreach ($candidateWorkers in ($autoTuneWorkersResolved | Sort-Object)) {
        $candidateId += 1
        $candidateLabel = "candidate-" + $candidateId.ToString("D2")
        $candidateRunLabel = "$candidateLabel-p$profile-w$candidateWorkers"
        Write-Host "[hall-max][autotune] running $candidateRunLabel"

        $tuneReportEvery = [Math]::Max(1, [Math]::Min($AutoTuneHands, [int][Math]::Ceiling($AutoTuneHands / 2.0)))
        $tuneResult = Invoke-HallParallelRun `
          -RepoRoot $repoRoot `
          -RootOutDir $autoTuneRoot `
          -RunLabel $candidateRunLabel `
          -TotalHands $AutoTuneHands `
          -WorkerCount $candidateWorkers `
          -TableCountPerWorker $TableCountPerWorker `
          -ReportEvery $tuneReportEvery `
          -LearnEveryHands 0 `
          -LearningWindowSamples $LearningWindowSamples `
          -Seed ($Seed + 1000 + $candidateId) `
          -HeroStyle $HeroStyle `
          -GtoMode $GtoMode `
          -VillainStyle $VillainStyle `
          -HeroExplorationRate $HeroExplorationRate `
          -RaiseSize $RaiseSize `
          -BunchingTrials $BunchingTrials `
          -EquityTrials $EquityTrials `
          -SaveTrainingLiteral "false" `
          -SaveDdreTrainingLiteral "false" `
          -NativeProfile $profile `
          -JvmOption $JvmOption `
          -Classpath $classpath `
          -FailFast:$FailFast `
          -ProgressSeconds $ProgressSeconds `
          -SuppressProgress

        if ($tuneResult.Success) {
          Write-Host ("[hall-max][autotune] {0} => {1:n2} hands/s" -f $candidateRunLabel, $tuneResult.HandsPerSecond)
        }
        else {
          Write-Host ("[hall-max][autotune][warn] {0} failed: {1}" -f $candidateRunLabel, $tuneResult.FailureReason)
        }

        $candidateRows.Add([pscustomobject]@{
          Candidate = $candidateRunLabel
          NativeProfile = $profile
          Workers = $candidateWorkers
          Hands = $AutoTuneHands
          ElapsedSeconds = $tuneResult.ElapsedSeconds
          HandsPerSecond = $tuneResult.HandsPerSecond
          Success = $tuneResult.Success
          FailureReason = $tuneResult.FailureReason
          RunOutDir = if ($null -eq $tuneResult.RunOutDir) { "" } else { $tuneResult.RunOutDir }
        })
      }
    }

    $resultsPath = Join-Path $autoTuneRoot "autotune-results.tsv"
    $tsvLines = New-Object System.Collections.Generic.List[string]
    $tsvLines.Add("candidate\tnativeProfile\tworkers\thands\telapsedSeconds\thandsPerSecond\tsuccess\tfailureReason\trunOutDir")
    foreach ($row in $candidateRows) {
      $elapsed = ([double]$row.ElapsedSeconds).ToString("F3", [System.Globalization.CultureInfo]::InvariantCulture)
      $hps = ([double]$row.HandsPerSecond).ToString("F2", [System.Globalization.CultureInfo]::InvariantCulture)
      $reason = [string]$row.FailureReason
      if ($reason.Contains("`t")) {
        $reason = $reason.Replace("`t", " ")
      }
      $runDir = [string]$row.RunOutDir
      if ($runDir.Contains("`t")) {
        $runDir = $runDir.Replace("`t", " ")
      }
      $tsvLines.Add("$($row.Candidate)`t$($row.NativeProfile)`t$($row.Workers)`t$($row.Hands)`t$elapsed`t$hps`t$($row.Success)`t$reason`t$runDir")
    }
    Set-Content -Path $resultsPath -Value $tsvLines -Encoding utf8
    Write-Host "[hall-max][autotune] results: $resultsPath"

    $successfulCandidates = @($candidateRows | Where-Object { $_.Success } | Sort-Object HandsPerSecond -Descending)
    if ($successfulCandidates.Count -eq 0) {
      throw "AutoTune failed: no candidate completed successfully."
    }

    $best = $successfulCandidates[0]
    $selectedWorkers = [int]$best.Workers
    $selectedNativeProfile = [string]$best.NativeProfile

    $selectionPath = Join-Path $autoTuneRoot "autotune-selection.txt"
    $selectionLines = @(
      "selectedCandidate=$($best.Candidate)",
      "selectedWorkers=$selectedWorkers",
      "selectedNativeProfile=$selectedNativeProfile",
      ("selectedHandsPerSecond={0}" -f ([double]$best.HandsPerSecond).ToString("F2", [System.Globalization.CultureInfo]::InvariantCulture)),
      "resultsPath=$resultsPath"
    )
    Set-Content -Path $selectionPath -Value $selectionLines -Encoding utf8
    Write-Host "[hall-max][autotune] selected workers=$selectedWorkers profile=$selectedNativeProfile"
    Write-Host "[hall-max][autotune] selection: $selectionPath"
  }

  $finalResult = Invoke-HallParallelRun `
    -RepoRoot $repoRoot `
    -RootOutDir $OutDir `
    -RunLabel "run" `
    -TotalHands $Hands `
    -WorkerCount $selectedWorkers `
    -TableCountPerWorker $TableCountPerWorker `
    -ReportEvery $ReportEvery `
    -LearnEveryHands $LearnEveryHands `
    -LearningWindowSamples $LearningWindowSamples `
    -Seed $Seed `
    -HeroStyle $HeroStyle `
    -GtoMode $GtoMode `
    -VillainStyle $VillainStyle `
    -HeroExplorationRate $HeroExplorationRate `
    -RaiseSize $RaiseSize `
    -BunchingTrials $BunchingTrials `
    -EquityTrials $EquityTrials `
    -SaveTrainingLiteral $saveTrainingLiteral `
    -SaveDdreTrainingLiteral $saveDdreTrainingLiteral `
    -NativeProfile $selectedNativeProfile `
    -JvmOption $JvmOption `
    -Classpath $classpath `
    -FailFast:$FailFast `
    -ProgressSeconds $ProgressSeconds

  if (-not $finalResult.Success) {
    throw "Final run failed: $($finalResult.FailureReason)"
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
