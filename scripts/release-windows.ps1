[CmdletBinding()]
param(
  [string]$OutputDir = "dist/release",
  [string]$NativeDllPath = "src/main/native/build/sicfun_gpu_kernel.dll",
  [int]$SmokeTrials = 200,
  [long]$SmokeMatchups = 128,
  [long]$ParityMatchups = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Step {
  param(
    [string]$Label,
    [scriptblock]$Action
  )
  Write-Host "==> $Label"
  & $Action
}

function Get-ReleaseClasspath {
  param(
    [string]$LibDir
  )
  $jars = @(Get-ChildItem -Path $LibDir -Filter *.jar | Sort-Object Name)
  if ($jars.Count -eq 0) {
    throw "No jars found under $LibDir"
  }
  ($jars | ForEach-Object { $_.FullName }) -join ";"
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
    $output = & sbt @Commands 2>&1
    $exitCode = $LASTEXITCODE
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

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$previousSbtOpts = $null
Push-Location $repoRoot
try {
  $resolvedNativeDll = (Resolve-Path $NativeDllPath).Path
  if (-not (Test-Path $resolvedNativeDll)) {
    throw "Native DLL not found: $NativeDllPath"
  }

  $previousSbtOpts = $env:SBT_OPTS
  $env:SBT_OPTS = "-Dsbt.server.autostart=false"

  Invoke-Step "Clear stale sbt java processes" {
    Stop-StaleSbtJavaProcesses
  }

  Invoke-Step "GPU smoke gate (source workspace)" {
    $null = Invoke-SbtWithRetry -Commands @(
      "runMain sicfun.holdem.HeadsUpGpuSmokeGate --table=canonical --trials=$SmokeTrials --maxMatchups=$SmokeMatchups --seed=1 --nativePath=$resolvedNativeDll"
    )
  }

  Invoke-Step "GPU exact parity gate (source workspace)" {
    Stop-StaleSbtJavaProcesses
    $null = Invoke-SbtWithRetry -Commands @(
      "runMain sicfun.holdem.HeadsUpGpuExactParityGate --maxMatchups=$ParityMatchups --seed=1 --parallelism=1 --nativePath=$resolvedNativeDll"
    )
  }

  $classpathLine = Invoke-Step "Build runtime jars and export classpath" {
    Stop-StaleSbtJavaProcesses
    $sbtOutput = Invoke-SbtWithRetry -Commands @(
      "package"
      "export Runtime / fullClasspathAsJars"
    )
    $sbtLines = $sbtOutput | ForEach-Object { "$_" }
    $resolvedClasspathLine = $sbtLines |
      Where-Object {
        ($_ -match "\.jar") -and ($_ -match ";") -and ($_ -notmatch "^\[info\]") -and ($_ -notmatch "^\[warn\]")
      } |
      Select-Object -Last 1
    if ([string]::IsNullOrWhiteSpace($resolvedClasspathLine)) {
      $tail = ($sbtLines | Select-Object -Last 20) -join [Environment]::NewLine
      Write-Host "sbt output tail:"
      Write-Host $tail
      throw "Could not parse classpath from sbt export output"
    }
    Write-Host "Resolved classpath: $resolvedClasspathLine"
    $resolvedClasspathLine
  }

  $releaseRoot = Join-Path $repoRoot $OutputDir
  $releaseLibDir = Join-Path $releaseRoot "lib"
  $releaseNativeDir = Join-Path $releaseRoot "native"
  $releaseBinDir = Join-Path $releaseRoot "bin"

  Invoke-Step "Assemble release directory" {
    if (Test-Path $releaseRoot) {
      Remove-Item -Path $releaseRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $releaseRoot | Out-Null
    New-Item -ItemType Directory -Path $releaseLibDir | Out-Null
    New-Item -ItemType Directory -Path $releaseNativeDir | Out-Null
    New-Item -ItemType Directory -Path $releaseBinDir | Out-Null

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
    Copy-Item -Path $resolvedNativeDll -Destination (Join-Path $releaseNativeDir "sicfun_gpu_kernel.dll") -Force
  }

  Invoke-Step "Write release run script (startup smoke gate)" {
    $startupScriptPath = Join-Path $releaseBinDir "run-gpu-smoke-gate.ps1"
    $startupScript = @'
[CmdletBinding()]
param(
  [ValidateSet("canonical", "full")]
  [string]$Table = "canonical",
  [int]$Trials = 128,
  [long]$MaxMatchups = 64,
  [long]$Seed = 1
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$releaseRoot = Split-Path -Parent $scriptDir
$libDir = Join-Path $releaseRoot "lib"
$nativeDll = Join-Path $releaseRoot "native\\sicfun_gpu_kernel.dll"

if (-not (Test-Path $nativeDll)) {
  throw "Missing native DLL: $nativeDll"
}

$classpath = (Get-ChildItem -Path $libDir -Filter *.jar | Sort-Object Name | ForEach-Object { $_.FullName }) -join ";"
if ([string]::IsNullOrWhiteSpace($classpath)) {
  throw "No jars found under $libDir"
}

& java `
  "-Dsicfun.gpu.native.path=$nativeDll" `
  "-cp" $classpath `
  "sicfun.holdem.HeadsUpGpuSmokeGate" `
  "--table=$Table" `
  "--trials=$Trials" `
  "--maxMatchups=$MaxMatchups" `
  "--seed=$Seed"

exit $LASTEXITCODE
'@
    Set-Content -Path $startupScriptPath -Value $startupScript -Encoding ascii
  }

  Invoke-Step "Startup check from release layout" {
    $releaseClasspath = Get-ReleaseClasspath -LibDir $releaseLibDir
    $releaseNativeDll = Join-Path $releaseNativeDir "sicfun_gpu_kernel.dll"
    & java `
      "-Dsicfun.gpu.native.path=$releaseNativeDll" `
      "-cp" $releaseClasspath `
      "sicfun.holdem.HeadsUpGpuSmokeGate" `
      "--table=canonical" `
      "--trials=128" `
      "--maxMatchups=64" `
      "--seed=1"
    if ($LASTEXITCODE -ne 0) {
      throw "Release startup smoke check failed with exit code $LASTEXITCODE"
    }
  }

  Invoke-Step "Write reproducibility manifest" {
    $manifestPath = Join-Path $releaseRoot "manifest.sha256"
    $files = Get-ChildItem -Path $releaseRoot -File -Recurse | Sort-Object FullName
    $lines = foreach ($file in $files) {
      $hash = (Get-FileHash -Path $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
      $relative = $file.FullName.Substring($releaseRoot.Length).TrimStart('\')
      "$hash  $relative"
    }
    Set-Content -Path $manifestPath -Value $lines -Encoding ascii
  }

  Write-Host "Release ready: $releaseRoot"
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
