[CmdletBinding()]
param(
  [string]$OutputDir = "dist/hand-history-web",
  [string]$StaticDir = "docs/site-preview-hybrid",
  [string]$ModelDir = "",
  [int]$SmokePort = 18080,
  [int]$MaxUploadBytes = 2097152,
  [long]$AnalysisTimeoutMs = 120000
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

function New-BasicAuthHeaderValue {
  param(
    [string]$Username,
    [string]$Password
  )

  $raw = [System.Text.Encoding]::UTF8.GetBytes("${Username}:${Password}")
  $encoded = [Convert]::ToBase64String($raw)
  return "Basic $encoded"
}

function Wait-AnalysisJobResult {
  param(
    [string]$BaseUri,
    [string]$StatusUrl,
    [hashtable]$Headers = @{},
    [int]$TimeoutSeconds = 30
  )

  $jobUri = Resolve-JobUri -BaseUri $BaseUri -StatusUrl $StatusUrl
  $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
  $delayMs = 750

  while ([DateTime]::UtcNow -lt $deadline) {
    Start-Sleep -Milliseconds $delayMs
    $statusResponse = Invoke-WebRequest -Uri $jobUri -Headers $Headers -UseBasicParsing -TimeoutSec 10
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
    [long]$AnalysisTimeoutMs,
    [string]$ExpectedHealthModelSource,
    [string]$ExpectedAnalysisModelSource
  )

  $drainSignalFile = Join-Path ([System.IO.Path]::GetTempPath()) ("sicfun-hand-history-web-drain-" + [guid]::NewGuid().ToString("N") + ".signal")
  $configFile = Join-Path ([System.IO.Path]::GetTempPath()) ("sicfun-hand-history-web-smoke-" + [guid]::NewGuid().ToString("N") + ".env")
  $basicAuthUser = "operator"
  $basicAuthPassword = "smoke-" + [guid]::NewGuid().ToString("N")
  $rateLimitSubmitsPerMinute = 1
  $rateLimitStatusPerMinute = 123
  $rateLimitClientIpHeader = "X-Real-IP"
  $rateLimitTrustedProxyIps = "203.0.113.10"
  $firstClientHeaders = @{
    Authorization = New-BasicAuthHeaderValue -Username $basicAuthUser -Password $basicAuthPassword
  }
  $firstClientHeaders[$rateLimitClientIpHeader] = "198.51.100.10"
  $secondClientHeaders = @{
    Authorization = New-BasicAuthHeaderValue -Username $basicAuthUser -Password $basicAuthPassword
  }
  $secondClientHeaders[$rateLimitClientIpHeader] = "198.51.100.11"
  $thirdClientHeaders = @{
    Authorization = New-BasicAuthHeaderValue -Username $basicAuthUser -Password $basicAuthPassword
  }
  $thirdClientHeaders[$rateLimitClientIpHeader] = "198.51.100.12"
  Set-Content -Path $configFile -Encoding ascii -Value @(
    "HOST=127.0.0.1",
    "PORT=$Port",
    "MAX_UPLOAD_BYTES=$MaxUploadBytes",
    "ANALYSIS_TIMEOUT_MS=$AnalysisTimeoutMs",
    "RATE_LIMIT_SUBMITS_PER_MINUTE=$rateLimitSubmitsPerMinute",
    "RATE_LIMIT_STATUS_PER_MINUTE=$rateLimitStatusPerMinute",
    "RATE_LIMIT_CLIENT_IP_HEADER=$rateLimitClientIpHeader",
    "RATE_LIMIT_TRUSTED_PROXY_IPS=$rateLimitTrustedProxyIps",
    "DRAIN_SIGNAL_FILE=$drainSignalFile",
    "BASIC_AUTH_USER=$basicAuthUser",
    "BASIC_AUTH_PASSWORD=$basicAuthPassword"
  )
  $job = Start-Job -ScriptBlock {
    param($launcherPath, $configFileArg)
    & powershell -NoProfile -ExecutionPolicy Bypass -File $launcherPath -ConfigFile $configFileArg
  } -ArgumentList (Join-Path $ReleaseRoot "bin\\run-hand-history-web.ps1"), $configFile

  try {
    $requiredPackagedFiles = @(
      "bin\\run-hand-history-web.ps1",
      "bin\\service-common.ps1",
      "bin\\install-hand-history-web-service.ps1",
      "bin\\uninstall-hand-history-web-service.ps1",
      "bin\\drain-stop-hand-history-web-service.ps1",
      "bin\\start-hand-history-web-service.ps1",
      "conf\\hand-history-web.env"
    )
    foreach ($relativePath in $requiredPackagedFiles) {
      $candidate = Join-Path $ReleaseRoot $relativePath
      if (-not (Test-Path -LiteralPath $candidate)) {
        throw "Packaged helper missing: $candidate"
      }
    }

    $healthUri = "http://127.0.0.1:$Port/api/health"
    $readyUri = "http://127.0.0.1:$Port/api/ready"
    $indexUri = "http://127.0.0.1:$Port/"
    $analyzeUri = "http://127.0.0.1:$Port/api/analyze-hand-history"
    $ready = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
      Start-Sleep -Milliseconds 750
      try {
        $response = Invoke-WebRequest -Uri $readyUri -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
          $readyBody = $response.Content | ConvertFrom-Json
          if (-not $readyBody.ready) {
            throw "Packaged readiness response did not report ready=true: $($response.Content)"
          }
          if ([string]$readyBody.reason -ne "accepting-traffic") {
            throw "Packaged readiness response reported unexpected reason: $($response.Content)"
          }
          if ([int]$readyBody.maxConcurrentJobs -lt 1 -or [int]$readyBody.maxQueuedJobs -lt 1) {
            throw "Packaged readiness response reported invalid queue settings: $($response.Content)"
          }
          if (-not $readyBody.authenticationEnabled -or [string]$readyBody.authenticationMode -ne "basic") {
            throw "Packaged readiness response did not report basic auth enabled: $($response.Content)"
          }
          if ([int]$readyBody.rateLimitSubmitsPerMinute -ne $rateLimitSubmitsPerMinute -or [int]$readyBody.rateLimitStatusPerMinute -ne $rateLimitStatusPerMinute) {
            throw "Packaged readiness response reported unexpected rate limit settings: $($response.Content)"
          }
          if ([string]$readyBody.rateLimitClientIpSource -ne "header:$rateLimitClientIpHeader via loopback-or-allowlisted-proxy") {
            throw "Packaged readiness response reported unexpected client IP source: $($response.Content)"
          }
          $ready = $true
          break
        }
      }
      catch {
      }
    }

    if (-not $ready) {
      $jobOutput = Receive-Job -Job $job -Keep | Out-String
      throw "Packaged web server failed readiness check. Job output:`n$jobOutput"
    }

    $healthResponse = Invoke-WebRequest -Uri $healthUri -UseBasicParsing -TimeoutSec 5
    if ($healthResponse.StatusCode -ne 200) {
      throw "Packaged health endpoint returned unexpected status: $($healthResponse.StatusCode)"
    }
    $healthBody = $healthResponse.Content | ConvertFrom-Json
    if (-not $healthBody.ok) {
      throw "Packaged health response did not report ok=true: $($healthResponse.Content)"
    }
    if ([string]$healthBody.modelSource -ne $ExpectedHealthModelSource) {
      throw "Packaged health model source mismatch: expected '$ExpectedHealthModelSource' got '$($healthBody.modelSource)'"
    }
    if ([string]$healthBody.readyReason -ne "accepting-traffic") {
      throw "Packaged health response reported unexpected readiness state: $($healthResponse.Content)"
    }
    if ([long]$healthBody.analysisTimeoutMs -ne $AnalysisTimeoutMs) {
      throw "Packaged health timeout mismatch: expected '$AnalysisTimeoutMs' got '$($healthBody.analysisTimeoutMs)'"
    }
    if (-not $healthBody.authenticationEnabled -or [string]$healthBody.authenticationMode -ne "basic") {
      throw "Packaged health response did not report basic auth enabled: $($healthResponse.Content)"
    }
    if ([int]$healthBody.rateLimitSubmitsPerMinute -ne $rateLimitSubmitsPerMinute -or [int]$healthBody.rateLimitStatusPerMinute -ne $rateLimitStatusPerMinute) {
      throw "Packaged health response reported unexpected rate limit settings: $($healthResponse.Content)"
    }
    if ([string]$healthBody.rateLimitClientIpSource -ne "header:$rateLimitClientIpHeader via loopback-or-allowlisted-proxy") {
      throw "Packaged health response reported unexpected client IP source: $($healthResponse.Content)"
    }

    try {
      Invoke-WebRequest -Uri $indexUri -UseBasicParsing -TimeoutSec 5 | Out-Null
      throw "Packaged index route allowed unauthenticated access"
    }
    catch {
      if (-not $_.Exception.Response -or [int]$_.Exception.Response.StatusCode -ne 401) {
        throw
      }
    }

    $indexResponse = Invoke-WebRequest -Uri $indexUri -Headers $firstClientHeaders -UseBasicParsing -TimeoutSec 5
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

    try {
      Invoke-WebRequest -Uri $analyzeUri -Method Post -ContentType "application/json" -Body $payload -UseBasicParsing -TimeoutSec 20 | Out-Null
      throw "Packaged analyze route allowed unauthenticated access"
    }
    catch {
      if (-not $_.Exception.Response -or [int]$_.Exception.Response.StatusCode -ne 401) {
        throw
      }
    }

    $analysisResponse = Invoke-WebRequest -Uri $analyzeUri -Method Post -Headers $firstClientHeaders -ContentType "application/json" -Body $payload -UseBasicParsing -TimeoutSec 20
    $submission = $analysisResponse.Content | ConvertFrom-Json
    $statusUrl = if ($submission.statusUrl) { [string]$submission.statusUrl } else { [string]$analysisResponse.Headers.Location }
    if ([string]::IsNullOrWhiteSpace([string]$submission.jobId) -or [string]::IsNullOrWhiteSpace($statusUrl) -or $submission.status -ne "queued") {
      $submissionJson = $submission | ConvertTo-Json -Depth 8
      throw "Packaged async submission smoke check failed: $submissionJson"
    }
    $analysis =
      if ($submission.jobId) {
        try {
          Invoke-WebRequest -Uri (Resolve-JobUri -BaseUri "http://127.0.0.1:$Port" -StatusUrl $statusUrl) -UseBasicParsing -TimeoutSec 5 | Out-Null
          throw "Packaged job-status route allowed unauthenticated access"
        }
        catch {
          if (-not $_.Exception.Response -or [int]$_.Exception.Response.StatusCode -ne 401) {
            throw
          }
        }
        Wait-AnalysisJobResult -BaseUri "http://127.0.0.1:$Port" -StatusUrl $statusUrl -Headers $firstClientHeaders -TimeoutSeconds 40
      }
      else {
        $submission
      }

    if ($analysis.handsImported -ne 1 -or $analysis.handsAnalyzed -ne 1) {
      $analysisJson = $analysis | ConvertTo-Json -Depth 8
      throw "Packaged analysis smoke check failed: $analysisJson"
    }

    $secondAnalysisResponse = Invoke-WebRequest -Uri $analyzeUri -Method Post -Headers $secondClientHeaders -ContentType "application/json" -Body $payload -UseBasicParsing -TimeoutSec 20
    $secondSubmission = $secondAnalysisResponse.Content | ConvertFrom-Json
    $secondStatusUrl = if ($secondSubmission.statusUrl) { [string]$secondSubmission.statusUrl } else { [string]$secondAnalysisResponse.Headers.Location }
    if ([string]::IsNullOrWhiteSpace([string]$secondSubmission.jobId) -or [string]::IsNullOrWhiteSpace($secondStatusUrl) -or $secondSubmission.status -ne "queued") {
      $secondSubmissionJson = $secondSubmission | ConvertTo-Json -Depth 8
      throw "Packaged trusted-header submission smoke check failed: $secondSubmissionJson"
    }
    $secondAnalysis = Wait-AnalysisJobResult -BaseUri "http://127.0.0.1:$Port" -StatusUrl $secondStatusUrl -Headers $secondClientHeaders -TimeoutSeconds 40
    if ($secondAnalysis.handsImported -ne 1 -or $secondAnalysis.handsAnalyzed -ne 1) {
      $secondAnalysisJson = $secondAnalysis | ConvertTo-Json -Depth 8
      throw "Packaged trusted-header analysis smoke check failed: $secondAnalysisJson"
    }

    if ([string]$analysis.modelSource -ne $ExpectedAnalysisModelSource) {
      $analysisJson = $analysis | ConvertTo-Json -Depth 8
      throw "Packaged model source smoke check failed: expected '$ExpectedAnalysisModelSource' got '$($analysis.modelSource)'. Payload: $analysisJson"
    }

    $tcpClient = [System.Net.Sockets.TcpClient]::new()
    try {
      $tcpClient.Connect("127.0.0.1", $Port)
      $stream = $tcpClient.GetStream()
      try {
        $declaredLength = $MaxUploadBytes + 256
        $authorizationHeader = [string]$firstClientHeaders.Authorization
        $requestLines = @(
          "POST /api/analyze-hand-history HTTP/1.1",
          "Host: 127.0.0.1:$Port",
          "Authorization: $authorizationHeader",
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

    Set-Content -Path $drainSignalFile -Value "draining" -Encoding ascii
    $draining = $false
    for ($attempt = 0; $attempt -lt 10; $attempt++) {
      Start-Sleep -Milliseconds 250
      try {
        $unexpectedReady = Invoke-WebRequest -Uri $readyUri -UseBasicParsing -TimeoutSec 5
        if ($unexpectedReady.StatusCode -eq 200) {
          continue
        }
      }
      catch {
        $response = $_.Exception.Response
        if ($null -ne $response -and [int]$response.StatusCode -eq 503) {
          $draining = $true
          break
        }
      }
    }
    if (-not $draining) {
      throw "Packaged readiness endpoint did not switch to 503 after drain signal activation"
    }

    $drainHealth = Invoke-WebRequest -Uri $healthUri -UseBasicParsing -TimeoutSec 5
    $drainHealthBody = $drainHealth.Content | ConvertFrom-Json
    if ($drainHealthBody.ready -or [string]$drainHealthBody.readyReason -ne "draining" -or $drainHealthBody.acceptingAnalysisJobs -or -not $drainHealthBody.drainSignalPresent) {
      throw "Packaged health endpoint did not report drain mode after drain signal activation: $($drainHealth.Content)"
    }

    $drainRejected = $false
    try {
      Invoke-WebRequest -Uri $analyzeUri -Method Post -Headers $thirdClientHeaders -ContentType "application/json" -Body $payload -UseBasicParsing -TimeoutSec 20 | Out-Null
    }
    catch {
      $response = $_.Exception.Response
      if ($null -ne $response -and [int]$response.StatusCode -eq 503) {
        $drainRejected = $true
      }
      else {
        throw
      }
    }
    if (-not $drainRejected) {
      throw "Packaged analysis submission was not rejected while drain mode was active"
    }
  }
  finally {
    Remove-Item -Path $configFile -Force -ErrorAction SilentlyContinue
    Remove-Item -Path $drainSignalFile -Force -ErrorAction SilentlyContinue
    Stop-Job -Job $job -ErrorAction SilentlyContinue | Out-Null
    $lingeringJava = Get-CimInstance Win32_Process -Filter "Name='java.exe'" -ErrorAction SilentlyContinue |
      Where-Object {
        $cmd = $_.CommandLine
        $null -ne $cmd -and
        $cmd.Contains("sicfun.holdem.web.HandHistoryReviewServer") -and
        $cmd.Contains("--port=$Port") -and
        $cmd.Contains($ReleaseRoot)
      }
    foreach ($proc in $lingeringJava) {
      Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }
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
$releaseConfDir = Join-Path $releaseRoot "conf"
$releaseStaticDir = Join-Path $releaseRoot "static"
$releaseModelDir = Join-Path $releaseRoot "model"
$serviceTemplateDir = Join-Path $repoRoot "scripts\packaged-hand-history-web"
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
    New-Item -ItemType Directory -Path $releaseConfDir | Out-Null
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

    $serviceTemplateFiles = @(
      "service-common.ps1",
      "install-hand-history-web-service.ps1",
      "uninstall-hand-history-web-service.ps1",
      "drain-stop-hand-history-web-service.ps1",
      "start-hand-history-web-service.ps1",
      "hand-history-web.env"
    ) | ForEach-Object { Join-Path $serviceTemplateDir $_ }

    foreach ($templatePath in $serviceTemplateFiles) {
      if (-not (Test-Path -LiteralPath $templatePath)) {
        throw "Packaged service helper missing: $templatePath"
      }
    }

    Copy-Item -Path (Join-Path $serviceTemplateDir "service-common.ps1") -Destination $releaseBinDir -Force
    Copy-Item -Path (Join-Path $serviceTemplateDir "install-hand-history-web-service.ps1") -Destination $releaseBinDir -Force
    Copy-Item -Path (Join-Path $serviceTemplateDir "uninstall-hand-history-web-service.ps1") -Destination $releaseBinDir -Force
    Copy-Item -Path (Join-Path $serviceTemplateDir "drain-stop-hand-history-web-service.ps1") -Destination $releaseBinDir -Force
    Copy-Item -Path (Join-Path $serviceTemplateDir "start-hand-history-web-service.ps1") -Destination $releaseBinDir -Force
    Copy-Item -Path (Join-Path $serviceTemplateDir "hand-history-web.env") -Destination (Join-Path $releaseConfDir "hand-history-web.env") -Force
  }

  Invoke-Step "Write packaged web launcher" {
    $launcherPath = Join-Path $releaseBinDir "run-hand-history-web.ps1"
    $launcher = @'
[CmdletBinding()]
param(
  [Alias("Host")]
  [string]$BindHost = "127.0.0.1",
  [int]$Port = 8080,
  [string]$ConfigFile = "",
  [string]$StaticDir = "",
  [string]$Model = "",
  [long]$Seed = 42,
  [int]$BunchingTrials = 200,
  [int]$EquityTrials = 2000,
  [long]$BudgetMs = 1500,
  [int]$MaxDecisions = 12,
  [int]$MaxUploadBytes = 2097152,
  [long]$AnalysisTimeoutMs = 120000,
  [int]$MaxConcurrentJobs = 0,
  [int]$MaxQueuedJobs = 0,
  [long]$ShutdownGraceMs = 5000,
  [int]$RateLimitSubmitsPerMinute = 6,
  [int]$RateLimitStatusPerMinute = 240,
  [string]$RateLimitClientIpHeader = "",
  [string]$RateLimitTrustedProxyIps = "",
  [string]$DrainSignalFile = "",
  [string]$BasicAuthUser = "",
  [string]$BasicAuthPassword = ""
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

function Resolve-ConfiguredPath {
  param(
    [string]$PathValue
  )

  if ([string]::IsNullOrWhiteSpace($PathValue)) {
    return ""
  }

  $candidate =
    if ([System.IO.Path]::IsPathRooted($PathValue)) { $PathValue }
    else { Join-Path $releaseRoot $PathValue }

  [System.IO.Path]::GetFullPath($candidate)
}

function Convert-ConfigValue {
  param(
    [string]$Value
  )

  if ($null -eq $Value) {
    return ""
  }

  $trimmed = $Value.Trim()
  if ($trimmed.Length -ge 2) {
    if (($trimmed.StartsWith('"') -and $trimmed.EndsWith('"')) -or ($trimmed.StartsWith("'") -and $trimmed.EndsWith("'"))) {
      return $trimmed.Substring(1, $trimmed.Length - 2)
    }
  }

  return $trimmed
}

function Read-EnvConfig {
  param(
    [string]$PathValue
  )

  $resolvedPath = Resolve-ConfiguredPath -PathValue $PathValue
  $values = @{}

  if ([string]::IsNullOrWhiteSpace($resolvedPath)) {
    return [pscustomobject]@{
      Path = ""
      Values = $values
    }
  }

  if (-not (Test-Path -LiteralPath $resolvedPath)) {
    throw "Config file not found: $resolvedPath"
  }

  $lineNumber = 0
  foreach ($line in Get-Content -LiteralPath $resolvedPath) {
    $lineNumber += 1
    $trimmed = $line.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith("#")) {
      continue
    }

    $separator = $trimmed.IndexOf("=")
    if ($separator -lt 1) {
      throw "Invalid config line $lineNumber in $resolvedPath. Expected KEY=VALUE."
    }

    $name = $trimmed.Substring(0, $separator).Trim()
    $value = Convert-ConfigValue -Value $trimmed.Substring($separator + 1)
    if ([string]::IsNullOrWhiteSpace($name)) {
      throw "Invalid config line $lineNumber in $resolvedPath. Key must be non-empty."
    }
    $values[$name] = $value
  }

  return [pscustomobject]@{
    Path = $resolvedPath
    Values = $values
  }
}

function Get-ConfigValue {
  param(
    [hashtable]$ConfigValues,
    [string]$Name
  )

  if ($null -ne $ConfigValues -and $ConfigValues.ContainsKey($Name)) {
    return [string]$ConfigValues[$Name]
  }

  return ""
}

function Get-JavaMajorVersion {
  param(
    [string]$RawVersion
  )

  if ([string]::IsNullOrWhiteSpace($RawVersion)) {
    throw "Java version string was empty"
  }

  $firstToken = $RawVersion.Split('.', 2)[0]
  if ($firstToken -eq "1") {
    return [int]($RawVersion.Split('.')[1])
  }

  return [int]$firstToken
}

function Assert-JavaRuntime {
  $java = Get-Command java.exe -ErrorAction SilentlyContinue
  if ($null -eq $java) {
    $java = Get-Command java -ErrorAction SilentlyContinue
  }
  if ($null -eq $java -or [string]::IsNullOrWhiteSpace($java.Source)) {
    throw "Java runtime not found on PATH. Install Java 17+ (JDK 21 recommended) or add java.exe to PATH before starting the packaged service."
  }

  $quotedJavaPath = '"' + $java.Source + '"'
  $versionOutput = (& cmd.exe /d /c "$quotedJavaPath -version 2>&1" | Out-String)
  if ($LASTEXITCODE -ne 0) {
    throw "Failed to execute 'java -version' using $($java.Source)."
  }
  if ($versionOutput -notmatch 'version "(?<version>[^"]+)"') {
    throw "Could not determine Java version from 'java -version' output.`n$versionOutput"
  }

  $rawVersion = $Matches.version
  $majorVersion = Get-JavaMajorVersion -RawVersion $rawVersion
  if ($majorVersion -lt 17) {
    throw "Java 17+ is required to run the packaged SICFUN hand-history web service. Detected $rawVersion at $($java.Source)."
  }
  if ($majorVersion -lt 21) {
    Write-Warning "Detected Java $rawVersion at $($java.Source). JDK 21 is recommended for operator parity."
  }

  Write-Host "Using Java runtime: $($java.Source) ($rawVersion)"
  return $java.Source
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$releaseRoot = Split-Path -Parent $scriptDir
$libWildcard = Join-Path $releaseRoot "lib\\*"
$effectiveConfigFileInput =
  if ($PSBoundParameters.ContainsKey("ConfigFile")) { $ConfigFile }
  elseif (-not [string]::IsNullOrWhiteSpace($env:CONFIG_FILE)) { $env:CONFIG_FILE }
  else { "conf\\hand-history-web.env" }
$configBundle = Read-EnvConfig -PathValue $effectiveConfigFileInput
$configValues = $configBundle.Values
$effectiveBindHost =
  if ($PSBoundParameters.ContainsKey("BindHost")) { $BindHost }
  elseif (-not [string]::IsNullOrWhiteSpace($env:HOST)) { $env:HOST }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "HOST"))) { Get-ConfigValue -ConfigValues $configValues -Name "HOST" }
  else { "127.0.0.1" }
$effectivePort =
  if ($PSBoundParameters.ContainsKey("Port")) { $Port }
  elseif (-not [string]::IsNullOrWhiteSpace($env:PORT)) { [int]$env:PORT }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "PORT"))) { [int](Get-ConfigValue -ConfigValues $configValues -Name "PORT") }
  else { 8080 }
$effectiveStaticDirInput =
  if ($PSBoundParameters.ContainsKey("StaticDir")) { $StaticDir }
  elseif (-not [string]::IsNullOrWhiteSpace($env:STATIC_DIR)) { $env:STATIC_DIR }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "STATIC_DIR"))) { Get-ConfigValue -ConfigValues $configValues -Name "STATIC_DIR" }
  else { "" }
$effectiveModelInput =
  if ($PSBoundParameters.ContainsKey("Model")) { $Model }
  elseif (-not [string]::IsNullOrWhiteSpace($env:MODEL_DIR)) { $env:MODEL_DIR }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "MODEL_DIR"))) { Get-ConfigValue -ConfigValues $configValues -Name "MODEL_DIR" }
  else { "" }
$effectiveSeed =
  if ($PSBoundParameters.ContainsKey("Seed")) { $Seed }
  elseif (-not [string]::IsNullOrWhiteSpace($env:SEED)) { [long]$env:SEED }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "SEED"))) { [long](Get-ConfigValue -ConfigValues $configValues -Name "SEED") }
  else { 42 }
$effectiveBunchingTrials =
  if ($PSBoundParameters.ContainsKey("BunchingTrials")) { $BunchingTrials }
  elseif (-not [string]::IsNullOrWhiteSpace($env:BUNCHING_TRIALS)) { [int]$env:BUNCHING_TRIALS }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "BUNCHING_TRIALS"))) { [int](Get-ConfigValue -ConfigValues $configValues -Name "BUNCHING_TRIALS") }
  else { 200 }
$effectiveEquityTrials =
  if ($PSBoundParameters.ContainsKey("EquityTrials")) { $EquityTrials }
  elseif (-not [string]::IsNullOrWhiteSpace($env:EQUITY_TRIALS)) { [int]$env:EQUITY_TRIALS }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "EQUITY_TRIALS"))) { [int](Get-ConfigValue -ConfigValues $configValues -Name "EQUITY_TRIALS") }
  else { 2000 }
$effectiveBudgetMs =
  if ($PSBoundParameters.ContainsKey("BudgetMs")) { $BudgetMs }
  elseif (-not [string]::IsNullOrWhiteSpace($env:BUDGET_MS)) { [long]$env:BUDGET_MS }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "BUDGET_MS"))) { [long](Get-ConfigValue -ConfigValues $configValues -Name "BUDGET_MS") }
  else { 1500 }
$effectiveMaxDecisions =
  if ($PSBoundParameters.ContainsKey("MaxDecisions")) { $MaxDecisions }
  elseif (-not [string]::IsNullOrWhiteSpace($env:MAX_DECISIONS)) { [int]$env:MAX_DECISIONS }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "MAX_DECISIONS"))) { [int](Get-ConfigValue -ConfigValues $configValues -Name "MAX_DECISIONS") }
  else { 12 }
$effectiveMaxUploadBytes =
  if ($PSBoundParameters.ContainsKey("MaxUploadBytes")) { $MaxUploadBytes }
  elseif (-not [string]::IsNullOrWhiteSpace($env:MAX_UPLOAD_BYTES)) { [int]$env:MAX_UPLOAD_BYTES }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "MAX_UPLOAD_BYTES"))) { [int](Get-ConfigValue -ConfigValues $configValues -Name "MAX_UPLOAD_BYTES") }
  else { 2097152 }
$effectiveAnalysisTimeoutMs =
  if ($PSBoundParameters.ContainsKey("AnalysisTimeoutMs")) { $AnalysisTimeoutMs }
  elseif (-not [string]::IsNullOrWhiteSpace($env:ANALYSIS_TIMEOUT_MS)) { [long]$env:ANALYSIS_TIMEOUT_MS }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "ANALYSIS_TIMEOUT_MS"))) { [long](Get-ConfigValue -ConfigValues $configValues -Name "ANALYSIS_TIMEOUT_MS") }
  else { 120000 }
$effectiveMaxConcurrentJobs =
  if ($PSBoundParameters.ContainsKey("MaxConcurrentJobs")) { $MaxConcurrentJobs }
  elseif (-not [string]::IsNullOrWhiteSpace($env:MAX_CONCURRENT_JOBS)) { [int]$env:MAX_CONCURRENT_JOBS }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "MAX_CONCURRENT_JOBS"))) { [int](Get-ConfigValue -ConfigValues $configValues -Name "MAX_CONCURRENT_JOBS") }
  else { 0 }
$effectiveMaxQueuedJobs =
  if ($PSBoundParameters.ContainsKey("MaxQueuedJobs")) { $MaxQueuedJobs }
  elseif (-not [string]::IsNullOrWhiteSpace($env:MAX_QUEUED_JOBS)) { [int]$env:MAX_QUEUED_JOBS }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "MAX_QUEUED_JOBS"))) { [int](Get-ConfigValue -ConfigValues $configValues -Name "MAX_QUEUED_JOBS") }
  else { 0 }
$effectiveShutdownGraceMs =
  if ($PSBoundParameters.ContainsKey("ShutdownGraceMs")) { $ShutdownGraceMs }
  elseif (-not [string]::IsNullOrWhiteSpace($env:SHUTDOWN_GRACE_MS)) { [long]$env:SHUTDOWN_GRACE_MS }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "SHUTDOWN_GRACE_MS"))) { [long](Get-ConfigValue -ConfigValues $configValues -Name "SHUTDOWN_GRACE_MS") }
  else { 5000 }
$effectiveRateLimitSubmitsPerMinute =
  if ($PSBoundParameters.ContainsKey("RateLimitSubmitsPerMinute")) { $RateLimitSubmitsPerMinute }
  elseif (-not [string]::IsNullOrWhiteSpace($env:RATE_LIMIT_SUBMITS_PER_MINUTE)) { [int]$env:RATE_LIMIT_SUBMITS_PER_MINUTE }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "RATE_LIMIT_SUBMITS_PER_MINUTE"))) { [int](Get-ConfigValue -ConfigValues $configValues -Name "RATE_LIMIT_SUBMITS_PER_MINUTE") }
  else { 6 }
$effectiveRateLimitStatusPerMinute =
  if ($PSBoundParameters.ContainsKey("RateLimitStatusPerMinute")) { $RateLimitStatusPerMinute }
  elseif (-not [string]::IsNullOrWhiteSpace($env:RATE_LIMIT_STATUS_PER_MINUTE)) { [int]$env:RATE_LIMIT_STATUS_PER_MINUTE }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "RATE_LIMIT_STATUS_PER_MINUTE"))) { [int](Get-ConfigValue -ConfigValues $configValues -Name "RATE_LIMIT_STATUS_PER_MINUTE") }
  else { 240 }
$effectiveRateLimitClientIpHeader =
  if ($PSBoundParameters.ContainsKey("RateLimitClientIpHeader")) { $RateLimitClientIpHeader }
  elseif (-not [string]::IsNullOrWhiteSpace($env:RATE_LIMIT_CLIENT_IP_HEADER)) { $env:RATE_LIMIT_CLIENT_IP_HEADER }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "RATE_LIMIT_CLIENT_IP_HEADER"))) { Get-ConfigValue -ConfigValues $configValues -Name "RATE_LIMIT_CLIENT_IP_HEADER" }
  else { "" }
$effectiveRateLimitTrustedProxyIps =
  if ($PSBoundParameters.ContainsKey("RateLimitTrustedProxyIps")) { $RateLimitTrustedProxyIps }
  elseif (-not [string]::IsNullOrWhiteSpace($env:RATE_LIMIT_TRUSTED_PROXY_IPS)) { $env:RATE_LIMIT_TRUSTED_PROXY_IPS }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "RATE_LIMIT_TRUSTED_PROXY_IPS"))) { Get-ConfigValue -ConfigValues $configValues -Name "RATE_LIMIT_TRUSTED_PROXY_IPS" }
  else { "" }
$effectiveDrainSignalFileInput =
  if ($PSBoundParameters.ContainsKey("DrainSignalFile")) { $DrainSignalFile }
  elseif (-not [string]::IsNullOrWhiteSpace($env:DRAIN_SIGNAL_FILE)) { $env:DRAIN_SIGNAL_FILE }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "DRAIN_SIGNAL_FILE"))) { Get-ConfigValue -ConfigValues $configValues -Name "DRAIN_SIGNAL_FILE" }
  else { "" }
$effectiveBasicAuthUser =
  if ($PSBoundParameters.ContainsKey("BasicAuthUser")) { $BasicAuthUser }
  elseif (-not [string]::IsNullOrWhiteSpace($env:BASIC_AUTH_USER)) { $env:BASIC_AUTH_USER }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "BASIC_AUTH_USER"))) { Get-ConfigValue -ConfigValues $configValues -Name "BASIC_AUTH_USER" }
  else { "" }
$effectiveBasicAuthPassword =
  if ($PSBoundParameters.ContainsKey("BasicAuthPassword")) { $BasicAuthPassword }
  elseif (-not [string]::IsNullOrWhiteSpace($env:BASIC_AUTH_PASSWORD)) { $env:BASIC_AUTH_PASSWORD }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "BASIC_AUTH_PASSWORD"))) { Get-ConfigValue -ConfigValues $configValues -Name "BASIC_AUTH_PASSWORD" }
  else { "" }
$effectiveUserStorePathInput =
  if (-not [string]::IsNullOrWhiteSpace($env:USER_STORE_PATH)) { $env:USER_STORE_PATH }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "USER_STORE_PATH"))) { Get-ConfigValue -ConfigValues $configValues -Name "USER_STORE_PATH" }
  else { "" }
$effectiveUserAuthAllowRegistration =
  if (-not [string]::IsNullOrWhiteSpace($env:USER_AUTH_ALLOW_REGISTRATION)) { $env:USER_AUTH_ALLOW_REGISTRATION }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "USER_AUTH_ALLOW_REGISTRATION"))) { Get-ConfigValue -ConfigValues $configValues -Name "USER_AUTH_ALLOW_REGISTRATION" }
  else { "" }
$effectiveUserAuthSessionTtlMs =
  if (-not [string]::IsNullOrWhiteSpace($env:USER_AUTH_SESSION_TTL_MS)) { $env:USER_AUTH_SESSION_TTL_MS }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "USER_AUTH_SESSION_TTL_MS"))) { Get-ConfigValue -ConfigValues $configValues -Name "USER_AUTH_SESSION_TTL_MS" }
  else { "" }
$effectiveUserAuthCookieSecure =
  if (-not [string]::IsNullOrWhiteSpace($env:USER_AUTH_COOKIE_SECURE)) { $env:USER_AUTH_COOKIE_SECURE }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "USER_AUTH_COOKIE_SECURE"))) { Get-ConfigValue -ConfigValues $configValues -Name "USER_AUTH_COOKIE_SECURE" }
  else { "" }
$effectiveGoogleOidcClientId =
  if (-not [string]::IsNullOrWhiteSpace($env:GOOGLE_OIDC_CLIENT_ID)) { $env:GOOGLE_OIDC_CLIENT_ID }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "GOOGLE_OIDC_CLIENT_ID"))) { Get-ConfigValue -ConfigValues $configValues -Name "GOOGLE_OIDC_CLIENT_ID" }
  else { "" }
$effectiveGoogleOidcClientSecret =
  if (-not [string]::IsNullOrWhiteSpace($env:GOOGLE_OIDC_CLIENT_SECRET)) { $env:GOOGLE_OIDC_CLIENT_SECRET }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "GOOGLE_OIDC_CLIENT_SECRET"))) { Get-ConfigValue -ConfigValues $configValues -Name "GOOGLE_OIDC_CLIENT_SECRET" }
  else { "" }
$effectiveGoogleOidcRedirectUri =
  if (-not [string]::IsNullOrWhiteSpace($env:GOOGLE_OIDC_REDIRECT_URI)) { $env:GOOGLE_OIDC_REDIRECT_URI }
  elseif (-not [string]::IsNullOrWhiteSpace((Get-ConfigValue -ConfigValues $configValues -Name "GOOGLE_OIDC_REDIRECT_URI"))) { Get-ConfigValue -ConfigValues $configValues -Name "GOOGLE_OIDC_REDIRECT_URI" }
  else { "" }
$javaCommand = Assert-JavaRuntime
if (-not [string]::IsNullOrWhiteSpace($configBundle.Path)) {
  Write-Host "Using config file: $($configBundle.Path)"
}
$resolvedStaticDir = Resolve-AppPath -PathValue $effectiveStaticDirInput -FallbackRelative "static"
$defaultModelDir = Join-Path $releaseRoot "model"
$resolvedModel = ""
if (-not [string]::IsNullOrWhiteSpace($effectiveModelInput)) {
  $resolvedModel = Resolve-AppPath -PathValue $effectiveModelInput -FallbackRelative "model"
}
elseif (Test-Path -LiteralPath $defaultModelDir) {
  $resolvedModel = (Resolve-Path -LiteralPath $defaultModelDir).Path
}
$resolvedDrainSignalFile = Resolve-ConfiguredPath -PathValue $effectiveDrainSignalFileInput
$resolvedUserStorePath = Resolve-ConfiguredPath -PathValue $effectiveUserStorePathInput

$javaArgs = @(
  "-cp", $libWildcard,
  "sicfun.holdem.web.HandHistoryReviewServer",
  "--host=$effectiveBindHost",
  "--port=$effectivePort",
  "--staticDir=$resolvedStaticDir",
  "--seed=$effectiveSeed",
  "--bunchingTrials=$effectiveBunchingTrials",
  "--equityTrials=$effectiveEquityTrials",
  "--budgetMs=$effectiveBudgetMs",
  "--maxDecisions=$effectiveMaxDecisions",
  "--maxUploadBytes=$effectiveMaxUploadBytes",
  "--analysisTimeoutMs=$effectiveAnalysisTimeoutMs",
  "--shutdownGraceMs=$effectiveShutdownGraceMs",
  "--rateLimitSubmitsPerMinute=$effectiveRateLimitSubmitsPerMinute",
  "--rateLimitStatusPerMinute=$effectiveRateLimitStatusPerMinute"
)

if (-not [string]::IsNullOrWhiteSpace($effectiveRateLimitClientIpHeader)) {
  $javaArgs += "--rateLimitClientIpHeader=$effectiveRateLimitClientIpHeader"
}
if (-not [string]::IsNullOrWhiteSpace($effectiveRateLimitTrustedProxyIps)) {
  $javaArgs += "--rateLimitTrustedProxyIps=$effectiveRateLimitTrustedProxyIps"
}

if (-not [string]::IsNullOrWhiteSpace($resolvedModel)) {
  $javaArgs += "--model=$resolvedModel"
}

if ($effectiveMaxConcurrentJobs -gt 0) {
  $javaArgs += "--maxConcurrentJobs=$effectiveMaxConcurrentJobs"
}
if ($effectiveMaxQueuedJobs -gt 0) {
  $javaArgs += "--maxQueuedJobs=$effectiveMaxQueuedJobs"
}
if (-not [string]::IsNullOrWhiteSpace($resolvedDrainSignalFile)) {
  $javaArgs += "--drainSignalFile=$resolvedDrainSignalFile"
}

$previousBasicAuthUser = $env:BASIC_AUTH_USER
$previousBasicAuthPassword = $env:BASIC_AUTH_PASSWORD
$previousUserStorePath = $env:USER_STORE_PATH
$previousUserAuthAllowRegistration = $env:USER_AUTH_ALLOW_REGISTRATION
$previousUserAuthSessionTtlMs = $env:USER_AUTH_SESSION_TTL_MS
$previousUserAuthCookieSecure = $env:USER_AUTH_COOKIE_SECURE
$previousGoogleOidcClientId = $env:GOOGLE_OIDC_CLIENT_ID
$previousGoogleOidcClientSecret = $env:GOOGLE_OIDC_CLIENT_SECRET
$previousGoogleOidcRedirectUri = $env:GOOGLE_OIDC_REDIRECT_URI
$exitCode = 0

try {
  if ([string]::IsNullOrWhiteSpace($effectiveBasicAuthUser)) {
    Remove-Item Env:BASIC_AUTH_USER -ErrorAction SilentlyContinue
  }
  else {
    $env:BASIC_AUTH_USER = $effectiveBasicAuthUser
  }

  if ([string]::IsNullOrWhiteSpace($effectiveBasicAuthPassword)) {
    Remove-Item Env:BASIC_AUTH_PASSWORD -ErrorAction SilentlyContinue
  }
  else {
    $env:BASIC_AUTH_PASSWORD = $effectiveBasicAuthPassword
  }

  if ([string]::IsNullOrWhiteSpace($resolvedUserStorePath)) {
    Remove-Item Env:USER_STORE_PATH -ErrorAction SilentlyContinue
  }
  else {
    $env:USER_STORE_PATH = $resolvedUserStorePath
  }

  if ([string]::IsNullOrWhiteSpace($effectiveUserAuthAllowRegistration)) {
    Remove-Item Env:USER_AUTH_ALLOW_REGISTRATION -ErrorAction SilentlyContinue
  }
  else {
    $env:USER_AUTH_ALLOW_REGISTRATION = $effectiveUserAuthAllowRegistration
  }

  if ([string]::IsNullOrWhiteSpace($effectiveUserAuthSessionTtlMs)) {
    Remove-Item Env:USER_AUTH_SESSION_TTL_MS -ErrorAction SilentlyContinue
  }
  else {
    $env:USER_AUTH_SESSION_TTL_MS = $effectiveUserAuthSessionTtlMs
  }

  if ([string]::IsNullOrWhiteSpace($effectiveUserAuthCookieSecure)) {
    Remove-Item Env:USER_AUTH_COOKIE_SECURE -ErrorAction SilentlyContinue
  }
  else {
    $env:USER_AUTH_COOKIE_SECURE = $effectiveUserAuthCookieSecure
  }

  if ([string]::IsNullOrWhiteSpace($effectiveGoogleOidcClientId)) {
    Remove-Item Env:GOOGLE_OIDC_CLIENT_ID -ErrorAction SilentlyContinue
  }
  else {
    $env:GOOGLE_OIDC_CLIENT_ID = $effectiveGoogleOidcClientId
  }

  if ([string]::IsNullOrWhiteSpace($effectiveGoogleOidcClientSecret)) {
    Remove-Item Env:GOOGLE_OIDC_CLIENT_SECRET -ErrorAction SilentlyContinue
  }
  else {
    $env:GOOGLE_OIDC_CLIENT_SECRET = $effectiveGoogleOidcClientSecret
  }

  if ([string]::IsNullOrWhiteSpace($effectiveGoogleOidcRedirectUri)) {
    Remove-Item Env:GOOGLE_OIDC_REDIRECT_URI -ErrorAction SilentlyContinue
  }
  else {
    $env:GOOGLE_OIDC_REDIRECT_URI = $effectiveGoogleOidcRedirectUri
  }

  & $javaCommand @javaArgs
  $exitCode = $LASTEXITCODE
}
finally {
  if ($null -ne $previousBasicAuthUser) {
    $env:BASIC_AUTH_USER = $previousBasicAuthUser
  }
  else {
    Remove-Item Env:BASIC_AUTH_USER -ErrorAction SilentlyContinue
  }

  if ($null -ne $previousBasicAuthPassword) {
    $env:BASIC_AUTH_PASSWORD = $previousBasicAuthPassword
  }
  else {
    Remove-Item Env:BASIC_AUTH_PASSWORD -ErrorAction SilentlyContinue
  }

  if ($null -ne $previousUserStorePath) {
    $env:USER_STORE_PATH = $previousUserStorePath
  }
  else {
    Remove-Item Env:USER_STORE_PATH -ErrorAction SilentlyContinue
  }

  if ($null -ne $previousUserAuthAllowRegistration) {
    $env:USER_AUTH_ALLOW_REGISTRATION = $previousUserAuthAllowRegistration
  }
  else {
    Remove-Item Env:USER_AUTH_ALLOW_REGISTRATION -ErrorAction SilentlyContinue
  }

  if ($null -ne $previousUserAuthSessionTtlMs) {
    $env:USER_AUTH_SESSION_TTL_MS = $previousUserAuthSessionTtlMs
  }
  else {
    Remove-Item Env:USER_AUTH_SESSION_TTL_MS -ErrorAction SilentlyContinue
  }

  if ($null -ne $previousUserAuthCookieSecure) {
    $env:USER_AUTH_COOKIE_SECURE = $previousUserAuthCookieSecure
  }
  else {
    Remove-Item Env:USER_AUTH_COOKIE_SECURE -ErrorAction SilentlyContinue
  }

  if ($null -ne $previousGoogleOidcClientId) {
    $env:GOOGLE_OIDC_CLIENT_ID = $previousGoogleOidcClientId
  }
  else {
    Remove-Item Env:GOOGLE_OIDC_CLIENT_ID -ErrorAction SilentlyContinue
  }

  if ($null -ne $previousGoogleOidcClientSecret) {
    $env:GOOGLE_OIDC_CLIENT_SECRET = $previousGoogleOidcClientSecret
  }
  else {
    Remove-Item Env:GOOGLE_OIDC_CLIENT_SECRET -ErrorAction SilentlyContinue
  }

  if ($null -ne $previousGoogleOidcRedirectUri) {
    $env:GOOGLE_OIDC_REDIRECT_URI = $previousGoogleOidcRedirectUri
  }
  else {
    Remove-Item Env:GOOGLE_OIDC_REDIRECT_URI -ErrorAction SilentlyContinue
  }
}

exit $exitCode
'@
    Set-Content -Path $launcherPath -Value $launcher -Encoding utf8
  }

  Invoke-Step "Packaged startup smoke check" {
    $expectedHealthModelSource =
      if ([string]::IsNullOrWhiteSpace($resolvedModelDir)) {
        "uniform fallback"
      }
      else {
        "configured artifact dir"
      }
    $expectedAnalysisModelSource =
      if ([string]::IsNullOrWhiteSpace($resolvedModelDir)) {
        "uniform fallback"
      }
      else {
        (Resolve-Path -LiteralPath $releaseModelDir).Path
      }
    Invoke-ReleaseSmoke `
      -ReleaseRoot $releaseRoot `
      -Port $SmokePort `
      -MaxUploadBytes $MaxUploadBytes `
      -AnalysisTimeoutMs $AnalysisTimeoutMs `
      -ExpectedHealthModelSource $expectedHealthModelSource `
      -ExpectedAnalysisModelSource $expectedAnalysisModelSource
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
