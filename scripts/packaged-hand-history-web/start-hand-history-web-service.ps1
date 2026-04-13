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
$healthUri = "$baseUri/api/health"

Remove-Item -Path $drainSignalPath -Force -ErrorAction SilentlyContinue
Start-Service -Name $ServiceName

$deadline = [DateTime]::UtcNow.AddSeconds($ReadyTimeoutSeconds)
$lastReadyProbe = $null
$lastHealthProbe = $null
$nextProgressAt = [DateTime]::UtcNow.AddSeconds(5)
while ([DateTime]::UtcNow -lt $deadline) {
  $service = Get-ServiceOrNull -ServiceName $ServiceName
  if ($null -eq $service) {
    throw "Service '$ServiceName' disappeared during startup.`n$(Get-ServiceDiagnosticSummary -ServiceName $ServiceName -ReadyProbe $lastReadyProbe -HealthProbe $lastHealthProbe)"
  }
  if ($service.Status -eq [System.ServiceProcess.ServiceControllerStatus]::Stopped) {
    throw "Service '$ServiceName' stopped before readiness.`n$(Get-ServiceDiagnosticSummary -ServiceName $ServiceName -ReadyProbe $lastReadyProbe -HealthProbe $lastHealthProbe)"
  }

  $lastReadyProbe = Invoke-JsonProbe -Uri $readyUri -TimeoutSeconds 5
  if ($lastReadyProbe.Success -and $lastReadyProbe.StatusCode -eq 200) {
    $ready = $lastReadyProbe.Json
    if ($null -ne $ready -and $ready.ready -and [string]$ready.reason -eq "accepting-traffic") {
      Write-Host "Service is ready at $baseUri"
      exit 0
    }
  }

  if (-not $lastReadyProbe.Success -or $lastReadyProbe.StatusCode -ne 200) {
    $lastHealthProbe = Invoke-JsonProbe -Uri $healthUri -TimeoutSeconds 5
  }

  if ([DateTime]::UtcNow -ge $nextProgressAt) {
    $progress = Format-ProbeSummary -Label "ready" -Probe $lastReadyProbe
    if ($null -ne $lastHealthProbe) {
      $progress = "$progress; $(Format-ProbeSummary -Label "health" -Probe $lastHealthProbe)"
    }
    Write-Host "Waiting for service readiness at $baseUri (service=$($service.Status); $progress)"
    $nextProgressAt = [DateTime]::UtcNow.AddSeconds(5)
  }

  Start-Sleep -Milliseconds 500
}

throw "Service '$ServiceName' did not become ready within $ReadyTimeoutSeconds seconds.`n$(Get-ServiceDiagnosticSummary -ServiceName $ServiceName -ReadyProbe $lastReadyProbe -HealthProbe $lastHealthProbe)"
