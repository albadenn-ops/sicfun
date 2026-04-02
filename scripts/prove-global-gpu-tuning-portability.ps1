[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-Condition {
  param(
    [bool]$Condition,
    [string]$Message
  )

  if (-not $Condition) {
    throw $Message
  }
}

$osName = [System.Runtime.InteropServices.RuntimeInformation]::OSDescription
$isWindowsHost = $osName.ToLowerInvariant().Contains("windows")
if (-not $isWindowsHost) {
  throw "This proof script currently supports Windows only. Host OS: $osName"
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$repoRootPath = $repoRoot.ProviderPath
$runScript = Join-Path $repoRootPath "scripts\run-global-tuning.ps1"
$nativeBuildDir = Join-Path $repoRootPath "src\main\native\build"
$requiredDlls = @(
  (Join-Path $nativeBuildDir "sicfun_gpu_kernel.dll"),
  (Join-Path $nativeBuildDir "sicfun_postflop_cuda.dll")
)
$repoCaches = @(
  (Join-Path $repoRootPath "data\headsup-backend-autotune.properties"),
  (Join-Path $repoRootPath "data\headsup-range-autotune.properties"),
  (Join-Path $repoRootPath "data\postflop-autotune.properties")
)
$proofRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("sicfun-global-gpu-proof-" + [guid]::NewGuid().ToString("N"))
$backupDir = Join-Path $proofRoot "backup"
$logPath = Join-Path $proofRoot "proof.log"
$capturedOutput = New-Object System.Collections.Generic.List[string]
$previousSbtOpts = $env:SBT_OPTS
$backups = New-Object System.Collections.Generic.List[object]

New-Item -ItemType Directory -Path $proofRoot -Force | Out-Null
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

try {
  foreach ($dllPath in $requiredDlls) {
    if (Test-Path $dllPath) {
      $backupPath = Join-Path $backupDir ([System.IO.Path]::GetFileName($dllPath) + ".bak")
      Move-Item $dllPath $backupPath -Force
      $backups.Add([pscustomobject]@{
        OriginalPath = $dllPath
        BackupPath = $backupPath
        RestoreMode = "move"
      }) | Out-Null
      Write-Host "[proof] moved $dllPath -> $backupPath"
    }
    else {
      Write-Host "[proof] runtime DLL already absent: $dllPath"
    }
  }

  foreach ($cachePath in $repoCaches) {
    if (Test-Path $cachePath) {
      $backupPath = Join-Path $backupDir ([System.IO.Path]::GetFileName($cachePath) + ".bak")
      Copy-Item $cachePath $backupPath -Force
      $backups.Add([pscustomobject]@{
        OriginalPath = $cachePath
        BackupPath = $backupPath
        RestoreMode = "copy"
      }) | Out-Null
      Write-Host "[proof] copied cache backup $cachePath -> $backupPath"
    }
  }

  $effectiveSbtOpts = @()
  if (-not [string]::IsNullOrWhiteSpace($previousSbtOpts)) {
    $effectiveSbtOpts += $previousSbtOpts.Trim()
  }
  $effectiveSbtOpts += @(
    "-Dsicfun.verbose=true",
    "-Dsicfun.repo.root=$repoRootPath"
  )
  $env:SBT_OPTS = ($effectiveSbtOpts -join " ").Trim()

  Push-Location $proofRoot
  try {
    & powershell -ExecutionPolicy Bypass -File $runScript --targets=runtime --force=true 2>&1 |
      ForEach-Object {
        $line = $_.ToString()
        $capturedOutput.Add($line) | Out-Null
        Write-Host $line
      }
    $exitCode = $LASTEXITCODE
    if ($null -ne $exitCode -and $exitCode -ne 0) {
      throw "Portability proof run exited with code $exitCode."
    }
  }
  finally {
    Pop-Location
  }

  $capturedOutput | Set-Content -Path $logPath
  $joinedOutput = $capturedOutput -join [Environment]::NewLine

  Assert-Condition ($joinedOutput.Contains("global gpu tuning: required native artifact(s) missing, running")) `
    "Proof failed: global tuning did not trigger the missing-artifact auto-build path."
  Assert-Condition (($capturedOutput | Where-Object { $_.StartsWith("[native-build]") }).Count -gt 0) `
    "Proof failed: no native build output was observed."
  foreach ($dllPath in $requiredDlls) {
    Assert-Condition (Test-Path $dllPath) "Proof failed: required runtime DLL missing after proof run: $dllPath"
  }
  Assert-Condition ($joinedOutput.Contains("gpu-autotune:")) `
    "Proof failed: backend runtime autotune code was not observed in the proof output."
  Assert-Condition ($joinedOutput.Contains("heads-up range gpu auto-tuner")) `
    "Proof failed: heads-up range tuning did not start."
  Assert-Condition ($joinedOutput.Contains("postflop gpu auto-tuner")) `
    "Proof failed: postflop tuning did not start."
  Assert-Condition (($joinedOutput -match 'range: (tuned|retuned)')) `
    "Proof failed: global summary did not report a tuned or retuned range target."
  Assert-Condition (($joinedOutput -match 'postflop: (tuned|retuned)')) `
    "Proof failed: global summary did not report a tuned or retuned postflop target."

  foreach ($backup in $backups) {
    if ($backup.RestoreMode -eq "move" -and (Test-Path $backup.BackupPath)) {
      Remove-Item $backup.BackupPath -Force
    }
  }

  Write-Host "[proof] PASS"
  Write-Host "[proof] cwd during run: $proofRoot"
  Write-Host "[proof] log: $logPath"
  foreach ($dllPath in $requiredDlls) {
    Write-Host "[proof] rebuilt: $dllPath"
  }
}
finally {
  $env:SBT_OPTS = $previousSbtOpts
  foreach ($backup in $backups) {
    if ($backup.RestoreMode -eq "copy") {
      if (Test-Path $backup.BackupPath) {
        Copy-Item $backup.BackupPath $backup.OriginalPath -Force
      }
    }
    elseif ((-not (Test-Path $backup.OriginalPath)) -and (Test-Path $backup.BackupPath)) {
      Move-Item $backup.BackupPath $backup.OriginalPath -Force
    }
  }
}
