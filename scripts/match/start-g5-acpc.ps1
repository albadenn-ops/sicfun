param(
  [Parameter(Mandatory = $true)][string]$Server,
  [Parameter(Mandatory = $true)][string]$Port,
  [string]$RuntimeDir = "data/tmp/g5-runtime-hu",
  [string]$DotnetDir = "data/tmp/toolchains/dotnet"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$runtimeDirAbs = Join-Path $repoRoot $RuntimeDir
$dotnetDirAbs = Join-Path $repoRoot $DotnetDir
$dotnet = Join-Path $dotnetDirAbs "dotnet.exe"
$g5Dll = Join-Path $runtimeDirAbs "G5.Acpc.dll"

if (-not (Test-Path $dotnet)) {
  throw "dotnet not found at $dotnet. Run scripts/build-g5-acpc.ps1 first. data/tmp/toolchains is local setup and is not source-controlled."
}
if (-not (Test-Path $g5Dll)) {
  throw "G5.Acpc.dll not found at $g5Dll. Run scripts/build-g5-acpc.ps1 first. The runtime under data/tmp/ is generated locally and is not source-controlled."
}

$env:DOTNET_ROOT = $dotnetDirAbs

Push-Location $runtimeDirAbs
try {
  & $dotnet "G5.Acpc.dll" $Server $Port
}
finally {
  Pop-Location
}
