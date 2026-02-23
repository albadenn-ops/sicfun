[CmdletBinding()]
param(
  [ValidateSet("canonical", "full")]
  [string]$Table = "canonical",
  [int]$Trials = 200,
  [long]$MaxMatchups = 128,
  [long]$Seed = 1,
  [string]$NativePath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

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

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$previousSbtOpts = $null
Push-Location $repoRoot
try {
  $previousSbtOpts = $env:SBT_OPTS
  $env:SBT_OPTS = "-Dsbt.server.autostart=false"

  $args = @(
    "--table=$Table",
    "--trials=$Trials",
    "--maxMatchups=$MaxMatchups",
    "--seed=$Seed"
  )
  if (-not [string]::IsNullOrWhiteSpace($NativePath)) {
    $resolvedNativePath = (Resolve-Path $NativePath).Path
    $args += "--nativePath=$resolvedNativePath"
  }

  Stop-StaleSbtJavaProcesses
  $joined = $args -join " "
  Invoke-SbtWithRetry -Commands @("runMain sicfun.holdem.HeadsUpGpuSmokeGate $joined")
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
