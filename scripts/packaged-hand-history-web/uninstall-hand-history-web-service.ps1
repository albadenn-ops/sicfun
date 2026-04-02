[CmdletBinding()]
param(
  [string]$ServiceName = "sicfun-hand-history-web"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "service-common.ps1")

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
$service = Get-ServiceOrNull -ServiceName $ServiceName
if ($null -eq $service) {
  Write-Host "Service '$ServiceName' is not installed."
  exit 0
}

try {
  Stop-Service -Name $ServiceName -ErrorAction SilentlyContinue
  Wait-ServiceStatus -ServiceName $ServiceName -DesiredStatus "Stopped" -TimeoutSeconds 30
}
catch {
}

Invoke-Sc -Arguments @("delete", $ServiceName)
Wait-ServiceStatus -ServiceName $ServiceName -DesiredStatus "Deleted" -TimeoutSeconds 30

Write-Host "Removed service: $ServiceName"
