[CmdletBinding()]
param(
  [string]$OutputDir = "dist/hand-history-web",
  [string]$StaticDir = "docs/site-preview-hybrid",
  [string]$ModelDir = "",
  [int]$SmokePort = 18080,
  [int]$MaxUploadBytes = 2097152
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Net.Http

function Invoke-Step {
  param(
    [string]$Label,
    [scriptblock]$Action
  )
  Write-Host "==> $Label"
  & $Action
}

function Stop-StaleSbtJavaProcesses {
  $sbtJava = Get-CimInstance Win32_Process -Filter "Name='java.exe'" |
    Where-Object {
      $cmd = $_.CommandLine
      $null -ne $cmd -and
      ($cmd -match "sbt" -or $cmd -match "sbt-launch") -and
      $cmd.Contains($repoRoot.Path)
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
    $previousErrorActionPreference = $ErrorActionPreference
    try {
      $ErrorActionPreference = "Continue"
      $output = & sbt @Commands 2>&1
      $exitCode = $LASTEXITCODE
    }
    finally {
      $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -eq 0) {
      return $output
    }

    $asText = ($output | Out-String)
    $isLockError = $asText -match "ServerAlreadyBootingException" -or $asText -match "Could not create lock for \\\\.\\pipe\\sbt-load"
    if ($isLockError -and $attempt -lt $MaxAttempts) {
      Write-Host "sbt lock collision detected, retrying (attempt $attempt/$MaxAttempts)..."
      Stop-StaleSbtJavaProcesses
      Start-Sleep -Milliseconds 1200
      $attempt += 1
      continue
    }

    $tail = (($output | ForEach-Object { "$_" }) | Select-Object -Last 30) -join [Environment]::NewLine
    Write-Host "sbt failure output tail:"
    Write-Host $tail
    throw "sbt command failed with exit code $exitCode"
  }

  throw "sbt command failed after $MaxAttempts attempts"
}

function Resolve-AppPath {
  param(
    [string]$PathValue,
    [switch]$AllowBlank
  )

  if ([string]::IsNullOrWhiteSpace($PathValue)) {
    if ($AllowBlank) {
      return ""
    }
    throw "Path must be non-empty"
  }

  $candidate =
    if ([System.IO.Path]::IsPathRooted($PathValue)) { $PathValue }
    else { Join-Path $repoRoot $PathValue }

  if (-not (Test-Path -LiteralPath $candidate)) {
    throw "Path not found: $PathValue"
  }

  (Resolve-Path -LiteralPath $candidate).Path
}

function Assert-RequiredStaticFiles {
  param(
    [string]$StaticRoot
  )

  $requiredRelativePaths = @(
    "index.html",
    "site.css",
    "site.js",
    "range-heatmap.svg"
  )

  foreach ($relativePath in $requiredRelativePaths) {
    $candidate = Join-Path $StaticRoot $relativePath
    if (-not (Test-Path -LiteralPath $candidate)) {
      throw "Required static asset missing: $candidate"
    }
  }
}

function Resolve-JobUri {
  param(
    [string]$BaseUri,
    [string]$StatusUrl
  )

  if ([string]::IsNullOrWhiteSpace($StatusUrl)) {
    throw "Analysis submission did not return a status URL"
  }

  if ($StatusUrl -match "^https?://") {
    return $StatusUrl
  }

  if ($StatusUrl.StartsWith("/")) {
    return "$BaseUri$StatusUrl"
  }

  return "$BaseUri/$StatusUrl"
}

function Wait-AnalysisJobResult {
  param(
    [string]$BaseUri,
    [string]$StatusUrl,
    [int]$TimeoutSeconds = 30
  )

  $jobUri = Resolve-JobUri -BaseUri $BaseUri -StatusUrl $StatusUrl
  $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
  $delayMs = 750

  while ([DateTime]::UtcNow -lt $deadline) {
    Start-Sleep -Milliseconds $delayMs
    $statusResponse = Invoke-WebRequest -Uri $jobUri -UseBasicParsing -TimeoutSec 10
    $statusBody = $statusResponse.Content | ConvertFrom-Json

    switch ($statusBody.status) {
      "queued" {
        if ($statusBody.pollAfterMs) {
          $delayMs = [Math]::Max(250, [Math]::Min(5000, [int]$statusBody.pollAfterMs))
        }
        continue
      }
      "running" {
        if ($statusBody.pollAfterMs) {
          $delayMs = [Math]::Max(250, [Math]::Min(5000, [int]$statusBody.pollAfterMs))
        }
        continue
      }
      "completed" {
        return $statusBody.result
      }
      "failed" {
        $errorStatus = if ($statusBody.errorStatus) { [int]$statusBody.errorStatus } else { 0 }
        throw "Packaged analysis job failed: status=$errorStatus error=$($statusBody.error)"
      }
      default {
        throw "Packaged analysis job returned unexpected status: $($statusBody.status)"
      }
    }
  }

  throw "Packaged analysis job did not complete within $TimeoutSeconds seconds"
}

function Invoke-ReleaseSmoke {
  param(
    [string]$ReleaseRoot,
    [int]$Port,
    [int]$MaxUploadBytes,
    [string]$ExpectedModelSource
  )

  $job = Start-Job -ScriptBlock {
    param($launcherPath, $portArg, $maxUploadBytesArg)
    & powershell -NoProfile -ExecutionPolicy Bypass -File $launcherPath -Host "127.0.0.1" -Port $portArg -MaxUploadBytes $maxUploadBytesArg
  } -ArgumentList (Join-Path $ReleaseRoot "bin\\run-hand-history-web.ps1"), $Port, $MaxUploadBytes

  try {
    $healthUri = "http://127.0.0.1:$Port/api/health"
    $indexUri = "http://127.0.0.1:$Port/"
    $analyzeUri = "http://127.0.0.1:$Port/api/analyze-hand-history"
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
      Start-Sleep -Milliseconds 750
      try {
        $response = Invoke-WebRequest -Uri $healthUri -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
          $ready = $true
          break
        }
      }
      catch {
      }
    }

    if (-not $ready) {
      $jobOutput = Receive-Job -Job $job -Keep | Out-String
      throw "Packaged web server failed health check. Job output:`n$jobOutput"
    }

    $indexResponse = Invoke-WebRequest -Uri $indexUri -UseBasicParsing -TimeoutSec 5
    if ($indexResponse.Content -notmatch 'id="hand-upload-form"' -or $indexResponse.Content -notmatch 'id="review-panel"') {
      throw "Packaged site smoke check failed: upload UI markers not found"
    }

    $sampleHand = @'
PokerStars Hand #1001:  Hold'em No Limit ($0.50/$1.00 USD) - 2026/03/10 12:00:00 ET
Table 'Alpha' 2-max Seat #1 is the button
Seat 1: Hero ($100.00 in chips)
Seat 2: Villain ($100.00 in chips)
Hero: posts small blind $0.50
Villain: posts big blind $1.00
*** HOLE CARDS ***
Dealt to Hero [Ac Kh]
Hero: raises $1.50 to $2.00
Villain: calls $1.00
*** FLOP *** [Ts 9h 8d]
Villain: checks
Hero: bets $3.00
Villain: raises $6.00 to $9.00
Hero: folds
*** SUMMARY ***
'@
    $payload = @{
      handHistoryText = $sampleHand
      heroName = "Hero"
      site = "auto"
    } | ConvertTo-Json

    $analysisResponse = Invoke-WebRequest -Uri $analyzeUri -Method Post -ContentType "application/json" -Body $payload -UseBasicParsing -TimeoutSec 20
    $submission = $analysisResponse.Content | ConvertFrom-Json
    if ([string]::IsNullOrWhiteSpace([string]$submission.jobId) -or [string]::IsNullOrWhiteSpace([string]$submission.statusUrl) -or $submission.status -ne "queued") {
      $submissionJson = $submission | ConvertTo-Json -Depth 8
      throw "Packaged async submission smoke check failed: $submissionJson"
    }
    $analysis =
      if ($submission.jobId) {
        $statusUrl = if ($submission.statusUrl) { [string]$submission.statusUrl } else { [string]$analysisResponse.Headers.Location }
        Wait-AnalysisJobResult -BaseUri "http://127.0.0.1:$Port" -StatusUrl $statusUrl -TimeoutSeconds 40
      }
      else {
        $submission
      }

    if ($analysis.handsImported -ne 1 -or $analysis.handsAnalyzed -ne 1) {
      $analysisJson = $analysis | ConvertTo-Json -Depth 8
      throw "Packaged analysis smoke check failed: $analysisJson"
    }

    if ([string]$analysis.modelSource -ne $ExpectedModelSource) {
      $analysisJson = $analysis | ConvertTo-Json -Depth 8
      throw "Packaged model source smoke check failed: expected '$ExpectedModelSource' got '$($analysis.modelSource)'. Payload: $analysisJson"
    }

    $tcpClient = [System.Net.Sockets.TcpClient]::new()
    try {
      $tcpClient.Connect("127.0.0.1", $Port)
      $stream = $tcpClient.GetStream()
      try {
        $declaredLength = $MaxUploadBytes + 256
        $requestLines = @(
          "POST /api/analyze-hand-history HTTP/1.1",
          "Host: 127.0.0.1:$Port",
          "Content-Type: application/json",
          "Content-Length: $declaredLength",
          "Connection: close",
          "",
          "{}"
        )
        $requestBytes = [System.Text.Encoding]::ASCII.GetBytes(($requestLines -join "`r`n"))
        $stream.Write($requestBytes, 0, $requestBytes.Length)
        $stream.Flush()
        $tcpClient.Client.Shutdown([System.Net.Sockets.SocketShutdown]::Send)

        $reader = [System.IO.StreamReader]::new($stream, [System.Text.Encoding]::UTF8)
        try {
          $rawResponse = $reader.ReadToEnd()
        }
        finally {
          $reader.Dispose()
        }

        if ($rawResponse -notmatch "413" -or $rawResponse -notmatch "max upload size") {
          throw "Packaged upload limit smoke check failed: $rawResponse"
        }
      }
      finally {
        $stream.Dispose()
      }
    }
    finally {
      $tcpClient.Dispose()
    }
  }
  finally {
    Stop-Job -Job $job -ErrorAction SilentlyContinue | Out-Null
    Receive-Job -Job $job -ErrorAction SilentlyContinue | Out-Null
    Remove-Job -Job $job -Force -ErrorAction SilentlyContinue | Out-Null
  }
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$resolvedStaticDir = Resolve-AppPath -PathValue $StaticDir
$resolvedModelDir = Resolve-AppPath -PathValue $ModelDir -AllowBlank
$releaseRoot = Join-Path $repoRoot $OutputDir
$releaseLibDir = Join-Path $releaseRoot "lib"
$releaseBinDir = Join-Path $releaseRoot "bin"
$releaseStaticDir = Join-Path $releaseRoot "static"
$releaseModelDir = Join-Path $releaseRoot "model"
$previousSbtOpts = $env:SBT_OPTS

Push-Location $repoRoot
try {
  $env:SBT_OPTS = "-Dsbt.server.autostart=false"

  Invoke-Step "Validate static site assets" {
    Assert-RequiredStaticFiles -StaticRoot $resolvedStaticDir
  }

  Invoke-Step "Build runtime jars" {
    Invoke-SbtWithRetry -Commands @(
      "package"
    ) | Out-Null
  }

  $classpathLine = Invoke-Step "Export runtime classpath" {
    $sbtOutput = Invoke-SbtWithRetry -Commands @(
      "export Runtime / fullClasspathAsJars"
    )
    $sbtLines = $sbtOutput | ForEach-Object { "$_" }
    $resolvedClasspathLine = $sbtLines |
      ForEach-Object { ($_ -replace "^\[info\]\s*", "").Trim() } |
      Where-Object {
        ($_ -match "\.jar") -and
        (($_ -match ";") -or ($_ -match "\.jar$")) -and
        ($_ -notmatch "^\[warn\]")
      } |
      Select-Object -Last 1
    if ([string]::IsNullOrWhiteSpace($resolvedClasspathLine)) {
      $tail = ($sbtLines | Select-Object -Last 20) -join [Environment]::NewLine
      Write-Host "sbt output tail:"
      Write-Host $tail
      throw "Could not parse classpath from sbt export output"
    }
    $resolvedClasspathLine
  }

  Invoke-Step "Assemble web release directory" {
    if (Test-Path $releaseRoot) {
      Remove-Item -Path $releaseRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $releaseRoot | Out-Null
    New-Item -ItemType Directory -Path $releaseLibDir | Out-Null
    New-Item -ItemType Directory -Path $releaseBinDir | Out-Null
    Copy-Item -Path $resolvedStaticDir -Destination $releaseStaticDir -Recurse -Force

    $jarPaths = @($classpathLine -split ";" | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
    if ($jarPaths.Count -eq 0) {
      throw "Parsed classpath had zero entries: $classpathLine"
    }
    foreach ($jarPath in $jarPaths) {
      if (-not (Test-Path $jarPath)) {
        throw "Classpath artifact missing: $jarPath"
      }
      Copy-Item -Path $jarPath -Destination $releaseLibDir -Force
    }

    if (-not [string]::IsNullOrWhiteSpace($resolvedModelDir)) {
      Copy-Item -Path $resolvedModelDir -Destination $releaseModelDir -Recurse -Force
    }
  }

  Invoke-Step "Write packaged web launcher" {
    $launcherPath = Join-Path $releaseBinDir "run-hand-history-web.ps1"
    $launcher = @'
[CmdletBinding()]
param(
  [Alias("Host")]
  [string]$BindHost = "127.0.0.1",
  [int]$Port = 8080,
  [string]$StaticDir = "",
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
    [string]$PathValue,
    [string]$FallbackRelative
  )

  $candidate =
    if ([string]::IsNullOrWhiteSpace($PathValue)) { Join-Path $releaseRoot $FallbackRelative }
    elseif ([System.IO.Path]::IsPathRooted($PathValue)) { $PathValue }
    else { Join-Path $releaseRoot $PathValue }

  if (-not (Test-Path -LiteralPath $candidate)) {
    throw "Path not found: $candidate"
  }

  (Resolve-Path -LiteralPath $candidate).Path
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$releaseRoot = Split-Path -Parent $scriptDir
$libWildcard = Join-Path $releaseRoot "lib\\*"
$resolvedStaticDir = Resolve-AppPath -PathValue $StaticDir -FallbackRelative "static"
$defaultModelDir = Join-Path $releaseRoot "model"
$resolvedModel = ""
if (-not [string]::IsNullOrWhiteSpace($Model)) {
  $resolvedModel = Resolve-AppPath -PathValue $Model -FallbackRelative "model"
}
elseif (Test-Path -LiteralPath $defaultModelDir) {
  $resolvedModel = (Resolve-Path -LiteralPath $defaultModelDir).Path
}

$javaArgs = @(
  "-cp", $libWildcard,
  "sicfun.holdem.web.HandHistoryReviewServer",
  "--host=$BindHost",
  "--port=$Port",
  "--staticDir=$resolvedStaticDir",
  "--seed=$Seed",
  "--bunchingTrials=$BunchingTrials",
  "--equityTrials=$EquityTrials",
  "--budgetMs=$BudgetMs",
  "--maxDecisions=$MaxDecisions",
  "--maxUploadBytes=$MaxUploadBytes"
)

if (-not [string]::IsNullOrWhiteSpace($resolvedModel)) {
  $javaArgs += "--model=$resolvedModel"
}

& java @javaArgs
exit $LASTEXITCODE
'@
    Set-Content -Path $launcherPath -Value $launcher -Encoding utf8
  }

  Invoke-Step "Packaged startup smoke check" {
    $expectedModelSource =
      if ([string]::IsNullOrWhiteSpace($resolvedModelDir)) {
        "uniform fallback"
      }
      else {
        (Resolve-Path -LiteralPath $releaseModelDir).Path
      }
    Invoke-ReleaseSmoke -ReleaseRoot $releaseRoot -Port $SmokePort -MaxUploadBytes $MaxUploadBytes -ExpectedModelSource $expectedModelSource
  }

  Invoke-Step "Write reproducibility manifest" {
    $manifestPath = Join-Path $releaseRoot "manifest.sha256"
    $files = Get-ChildItem -Path $releaseRoot -File -Recurse | Sort-Object FullName
    $lines = foreach ($file in $files) {
      $hash = (Get-FileHash -Path $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
      $relative = $file.FullName.Substring($releaseRoot.Length).TrimStart('\')
      "$hash  $relative"
    }
    Set-Content -Path $manifestPath -Value $lines -Encoding utf8
  }

  Write-Host "Hand-history web release ready: $releaseRoot"
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
