[CmdletBinding()]
param(
  [string]$ServiceName = "sicfun-hand-history-web",
  [string]$ConfigFile = "",
  [int]$ReadyTimeoutSeconds = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "service-common.ps1")

Assert-Administrator
$service = Get-ServiceOrNull -ServiceName $ServiceName
if ($null -eq $service) {
  throw "Service '$ServiceName' is not installed."
}

$effectiveConfigFile =
  if ($PSBoundParameters.ContainsKey("ConfigFile")) { $ConfigFile }
  else {
    $installedConfig = Get-ServiceInstalledConfigFile -ServiceName $ServiceName
    if (-not [string]::IsNullOrWhiteSpace($installedConfig)) { $installedConfig }
    else { "conf\\hand-history-web.env" }
  }

$configValues = Read-EnvConfig -ConfigFile $effectiveConfigFile
$drainSignalPath = Resolve-DrainSignalPath -ConfigValues $configValues
$baseUri = Resolve-ServiceBaseUri -ConfigValues $configValues
$readyUri = "$baseUri/api/ready"

Remove-Item -Path $drainSignalPath -Force -ErrorAction SilentlyContinue
Start-Service -Name $ServiceName

$deadline = [DateTime]::UtcNow.AddSeconds($ReadyTimeoutSeconds)
while ([DateTime]::UtcNow -lt $deadline) {
  try {
    $response = Invoke-WebRequest -Uri $readyUri -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
      $ready = $response.Content | ConvertFrom-Json
      if ($ready.ready -and [string]$ready.reason -eq "accepting-traffic") {
        Write-Host "Service is ready at $baseUri"
        exit 0
      }
    }
  }
  catch {
  }

  Start-Sleep -Milliseconds 500
}

throw "Service '$ServiceName' did not become ready within $ReadyTimeoutSeconds seconds."
