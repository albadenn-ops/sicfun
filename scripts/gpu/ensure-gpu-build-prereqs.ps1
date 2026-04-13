[CmdletBinding()]
param(
  [switch]$InstallMissing,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$BuildArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$buildScript = Join-Path $repoRoot.ProviderPath "src\main\native\build-windows-cuda11.ps1"
$command = @(
  "-ExecutionPolicy", "Bypass",
  "-File", $buildScript,
  "-CheckPrerequisites"
)
if ($InstallMissing) {
  $command += "-InstallMissingPrerequisites"
}
if ($BuildArgs) {
  $command += $BuildArgs
}

& powershell @command
$exitCode = $LASTEXITCODE
if ($null -ne $exitCode -and $exitCode -ne 0) {
  throw "GPU build prerequisite check exited with code $exitCode."
}
