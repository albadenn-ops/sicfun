[CmdletBinding()]
param(
  [string]$ServiceName = "sicfun-hand-history-web",
  [string]$DisplayName = "SICFUN Hand-History Review",
  [string]$Description = "SICFUN packaged hand-history review web service",
  [string]$ConfigFile = "conf\\hand-history-web.env",
  [string]$NssmPath = "",
  [ValidateSet("auto", "delayed-auto", "demand")]
  [string]$StartupType = "auto",
  [switch]$ForceReinstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "service-common.ps1")

function Invoke-Nssm {
  param(
    [string]$Command,
    [string[]]$Arguments
  )

  & $script:NssmCommand $Command @Arguments | Out-Null
  if ($LASTEXITCODE -ne 0) {
    throw "nssm $Command failed with exit code $LASTEXITCODE."
  }
}

function Invoke-Sc {
  param(
    [string[]]$Arguments
  )

  & sc.exe @Arguments | Out-Null
  if ($LASTEXITCODE -ne 0) {
    throw "sc.exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE."
  }
}

Assert-Administrator
$script:NssmCommand = Resolve-NssmCommand -NssmPath $NssmPath
$releaseRoot = Get-ReleaseRoot
$launcherPath = Resolve-ReleasePath -PathValue "bin\\run-hand-history-web.ps1"
$configPath = Resolve-ReleasePath -PathValue $ConfigFile
$logsDir = Join-Path $releaseRoot "logs"
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null

$existingService = Get-ServiceOrNull -ServiceName $ServiceName
if ($null -ne $existingService) {
  if (-not $ForceReinstall) {
    throw "Service '$ServiceName' already exists. Re-run with -ForceReinstall to replace it."
  }

  Write-Host "Replacing existing service '$ServiceName'..."
  try {
    Stop-Service -Name $ServiceName -ErrorAction SilentlyContinue
    Wait-ServiceStatus -ServiceName $ServiceName -DesiredStatus "Stopped" -TimeoutSeconds 30
  }
  catch {
  }

  Invoke-Nssm -Command "remove" -Arguments @($ServiceName, "confirm")
  Wait-ServiceStatus -ServiceName $ServiceName -DesiredStatus "Deleted" -TimeoutSeconds 30
}

$startupMode =
  switch ($StartupType) {
    "auto" { "SERVICE_AUTO_START" }
    "delayed-auto" { "SERVICE_DELAYED_AUTO_START" }
    "demand" { "SERVICE_DEMAND_START" }
    default { throw "Unsupported startup type: $StartupType" }
  }

$appArguments = "-NoProfile -ExecutionPolicy Bypass -File `"$launcherPath`" -ConfigFile `"$configPath`""
Invoke-Nssm -Command "install" -Arguments @($ServiceName, "powershell.exe", $appArguments)
Invoke-Nssm -Command "set" -Arguments @($ServiceName, "AppDirectory", $releaseRoot)
Invoke-Nssm -Command "set" -Arguments @($ServiceName, "DisplayName", $DisplayName)
Invoke-Nssm -Command "set" -Arguments @($ServiceName, "AppStdout", (Join-Path $logsDir "service.stdout.log"))
Invoke-Nssm -Command "set" -Arguments @($ServiceName, "AppStderr", (Join-Path $logsDir "service.stderr.log"))
Invoke-Nssm -Command "set" -Arguments @($ServiceName, "AppRotateFiles", "1")
Invoke-Nssm -Command "set" -Arguments @($ServiceName, "AppRotateOnline", "1")
Invoke-Nssm -Command "set" -Arguments @($ServiceName, "AppRotateBytes", "10485760")
Invoke-Nssm -Command "set" -Arguments @($ServiceName, "Start", $startupMode)

Invoke-Sc -Arguments @("description", $ServiceName, $Description)
Invoke-Sc -Arguments @("failure", $ServiceName, "reset= 86400", "actions= restart/5000/restart/5000/restart/5000")
Invoke-Sc -Arguments @("failureflag", $ServiceName, "1")

Write-Host "Installed service: $ServiceName"
Write-Host "  display name: $DisplayName"
Write-Host "  startup type: $StartupType"
Write-Host "  config file : $configPath"
Write-Host "  logs dir    : $logsDir"
Write-Host "Use .\\start-hand-history-web-service.ps1 to start it and wait for readiness."
