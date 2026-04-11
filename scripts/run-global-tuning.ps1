[CmdletBinding()]
param(
  [switch]$InstallMissingPrerequisites,
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ToolArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-SbtCommand {
  foreach ($candidate in @("sbt.bat", "sbt.cmd", "sbt")) {
    $command = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($null -ne $command -and -not [string]::IsNullOrWhiteSpace($command.Source)) {
      return $command.Source
    }
  }

  throw "Unable to resolve sbt on PATH. Install sbt or add it to PATH before running global GPU tuning."
}

function Format-SbtArgument {
  param(
    [string]$Value
  )

  if ($null -eq $Value) {
    return '""'
  }

  if ($Value -match '[\s"]') {
    return '"' + $Value.Replace('\', '\\').Replace('"', '\"') + '"'
  }

  return $Value
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$repoRootPath = $repoRoot.ProviderPath
$sbtCommand = Resolve-SbtCommand
$previousSbtOpts = $env:SBT_OPTS
$effectiveSbtOpts = @()
if (-not [string]::IsNullOrWhiteSpace($previousSbtOpts)) {
  $effectiveSbtOpts += $previousSbtOpts.Trim()
}
if (($effectiveSbtOpts -join " ") -notmatch '(?i)(^|\s)-Dsbt\.server\.autostart=') {
  $effectiveSbtOpts += "-Dsbt.server.autostart=false"
}
if (($effectiveSbtOpts -join " ") -notmatch '(?i)(^|\s)-Dsicfun\.repo\.root=') {
  $effectiveSbtOpts += ('"-Dsicfun.repo.root={0}"' -f $repoRootPath)
}
if ($InstallMissingPrerequisites -and (($effectiveSbtOpts -join " ") -notmatch '(?i)(^|\s)-Dsicfun\.gpu\.build\.installPrereqs=')) {
  $effectiveSbtOpts += "-Dsicfun.gpu.build.installPrereqs=true"
}
Push-Location $repoRoot
try {
  $env:SBT_OPTS = ($effectiveSbtOpts -join " ").Trim()
  $formattedToolArgs = @()
  if ($ToolArgs -and @($ToolArgs).Count -gt 0) {
    foreach ($toolArg in @($ToolArgs)) {
      $formattedToolArgs += Format-SbtArgument $toolArg
    }
  }
  $runMainArg = if ($formattedToolArgs.Count -gt 0) {
    "runMain sicfun.holdem.bench.GlobalGpuTuningTool " + ($formattedToolArgs -join " ")
  }
  else {
    "runMain sicfun.holdem.bench.GlobalGpuTuningTool"
  }
  $escapedSbtCommand = $sbtCommand.Replace('"', '""')
  $escapedRunMainArg = $runMainArg.Replace('"', '""')
  $commandLine = ('"{0}" "{1}"' -f $escapedSbtCommand, $escapedRunMainArg)
  & cmd /c $commandLine
  $exitCode = $LASTEXITCODE
  if ($null -ne $exitCode -and $exitCode -ne 0) {
    throw "GlobalGpuTuningTool exited with code $exitCode."
  }
}
finally {
  $env:SBT_OPTS = $previousSbtOpts
  Pop-Location
}
