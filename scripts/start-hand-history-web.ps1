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
  [string]$BasicAuthPassword = "",
  [string]$UserStorePath = "",
  [bool]$UserAuthAllowRegistration = $true,
  [long]$UserAuthSessionTtlMs = 43200000,
  [bool]$UserAuthCookieSecure = $false,
  [string]$GoogleOidcClientId = "",
  [string]$GoogleOidcClientSecret = "",
  [string]$GoogleOidcRedirectUri = ""
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

function Resolve-OptionalPath {
  param(
    [string]$PathValue
  )

  if ([string]::IsNullOrWhiteSpace($PathValue)) {
    return ""
  }

  $candidate =
    if ([System.IO.Path]::IsPathRooted($PathValue)) { $PathValue }
    else { Join-Path $repoRoot $PathValue }

  [System.IO.Path]::GetFullPath($candidate)
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
$resolvedDrainSignalFile = Resolve-OptionalPath -PathValue $DrainSignalFile
$resolvedUserStorePath = Resolve-OptionalPath -PathValue $UserStorePath

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
  ANALYSIS_TIMEOUT_MS = $env:ANALYSIS_TIMEOUT_MS
  MAX_CONCURRENT_JOBS = $env:MAX_CONCURRENT_JOBS
  MAX_QUEUED_JOBS = $env:MAX_QUEUED_JOBS
  SHUTDOWN_GRACE_MS = $env:SHUTDOWN_GRACE_MS
  RATE_LIMIT_SUBMITS_PER_MINUTE = $env:RATE_LIMIT_SUBMITS_PER_MINUTE
  RATE_LIMIT_STATUS_PER_MINUTE = $env:RATE_LIMIT_STATUS_PER_MINUTE
  RATE_LIMIT_CLIENT_IP_HEADER = $env:RATE_LIMIT_CLIENT_IP_HEADER
  RATE_LIMIT_TRUSTED_PROXY_IPS = $env:RATE_LIMIT_TRUSTED_PROXY_IPS
  DRAIN_SIGNAL_FILE = $env:DRAIN_SIGNAL_FILE
  BASIC_AUTH_USER = $env:BASIC_AUTH_USER
  BASIC_AUTH_PASSWORD = $env:BASIC_AUTH_PASSWORD
  USER_STORE_PATH = $env:USER_STORE_PATH
  USER_AUTH_ALLOW_REGISTRATION = $env:USER_AUTH_ALLOW_REGISTRATION
  USER_AUTH_SESSION_TTL_MS = $env:USER_AUTH_SESSION_TTL_MS
  USER_AUTH_COOKIE_SECURE = $env:USER_AUTH_COOKIE_SECURE
  GOOGLE_OIDC_CLIENT_ID = $env:GOOGLE_OIDC_CLIENT_ID
  GOOGLE_OIDC_CLIENT_SECRET = $env:GOOGLE_OIDC_CLIENT_SECRET
  GOOGLE_OIDC_REDIRECT_URI = $env:GOOGLE_OIDC_REDIRECT_URI
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
  Set-Or-ClearEnv -Name "ANALYSIS_TIMEOUT_MS" -Value $AnalysisTimeoutMs
  if ($MaxConcurrentJobs -gt 0) {
    Set-Or-ClearEnv -Name "MAX_CONCURRENT_JOBS" -Value $MaxConcurrentJobs
  }
  else {
    Set-Or-ClearEnv -Name "MAX_CONCURRENT_JOBS" -Value ""
  }
  if ($MaxQueuedJobs -gt 0) {
    Set-Or-ClearEnv -Name "MAX_QUEUED_JOBS" -Value $MaxQueuedJobs
  }
  else {
    Set-Or-ClearEnv -Name "MAX_QUEUED_JOBS" -Value ""
  }
  Set-Or-ClearEnv -Name "SHUTDOWN_GRACE_MS" -Value $ShutdownGraceMs
  Set-Or-ClearEnv -Name "RATE_LIMIT_SUBMITS_PER_MINUTE" -Value $RateLimitSubmitsPerMinute
  Set-Or-ClearEnv -Name "RATE_LIMIT_STATUS_PER_MINUTE" -Value $RateLimitStatusPerMinute
  Set-Or-ClearEnv -Name "RATE_LIMIT_CLIENT_IP_HEADER" -Value $RateLimitClientIpHeader
  Set-Or-ClearEnv -Name "RATE_LIMIT_TRUSTED_PROXY_IPS" -Value $RateLimitTrustedProxyIps
  Set-Or-ClearEnv -Name "DRAIN_SIGNAL_FILE" -Value $resolvedDrainSignalFile
  Set-Or-ClearEnv -Name "BASIC_AUTH_USER" -Value $BasicAuthUser
  Set-Or-ClearEnv -Name "BASIC_AUTH_PASSWORD" -Value $BasicAuthPassword
  Set-Or-ClearEnv -Name "USER_STORE_PATH" -Value $resolvedUserStorePath
  Set-Or-ClearEnv -Name "USER_AUTH_ALLOW_REGISTRATION" -Value $UserAuthAllowRegistration
  Set-Or-ClearEnv -Name "USER_AUTH_SESSION_TTL_MS" -Value $UserAuthSessionTtlMs
  Set-Or-ClearEnv -Name "USER_AUTH_COOKIE_SECURE" -Value $UserAuthCookieSecure
  Set-Or-ClearEnv -Name "GOOGLE_OIDC_CLIENT_ID" -Value $GoogleOidcClientId
  Set-Or-ClearEnv -Name "GOOGLE_OIDC_CLIENT_SECRET" -Value $GoogleOidcClientSecret
  Set-Or-ClearEnv -Name "GOOGLE_OIDC_REDIRECT_URI" -Value $GoogleOidcRedirectUri

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
