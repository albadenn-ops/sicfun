[CmdletBinding()]
param(
  [string]$ServiceName = "sicfun-hand-history-web",
  [string]$ConfigFile = "",
  [int]$DrainTimeoutSeconds = 60,
  [int]$ServiceStopTimeoutSeconds = 30,
  [switch]$KeepDrainSignal
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
$drainDir = Split-Path -Parent $drainSignalPath
if (-not [string]::IsNullOrWhiteSpace($drainDir)) {
  New-Item -ItemType Directory -Path $drainDir -Force | Out-Null
}

try {
  Set-Content -Path $drainSignalPath -Value "draining" -Encoding ascii
  Write-Host "Drain signal created: $drainSignalPath"

  $readyFlipped = $false
  $drained = $false
  $lastObservedDrainState = $null
  $deadline = [DateTime]::UtcNow.AddSeconds($DrainTimeoutSeconds)
  while ([DateTime]::UtcNow -lt $deadline) {
    $service = Get-ServiceOrNull -ServiceName $ServiceName
    if ($null -eq $service -or $service.Status -eq [System.ServiceProcess.ServiceControllerStatus]::Stopped) {
      $drained = $true
      break
    }

    try {
      Invoke-WebRequest -Uri $readyUri -UseBasicParsing -TimeoutSec 5 | Out-Null
    }
    catch {
      $response = $_.Exception.Response
      if ($null -ne $response -and [int]$response.StatusCode -eq 503) {
        $readyFlipped = $true
      }
    }

    try {
      $healthResponse = Invoke-WebRequest -Uri $healthUri -UseBasicParsing -TimeoutSec 5
      $health = $healthResponse.Content | ConvertFrom-Json
      $lastObservedDrainState = [pscustomobject]@{
        ActiveHttpRequests = [int]$health.activeHttpRequests
        QueuedJobs = [int]$health.queuedJobs
        RunningJobs = [int]$health.runningJobs
        TimedOutWorkersInFlight = [int]$health.timedOutWorkersInFlight
      }
      # Do not stop the service while UI, health, or job-status traffic is still in flight.
      if ($lastObservedDrainState.ActiveHttpRequests -eq 0 -and
          $lastObservedDrainState.QueuedJobs -eq 0 -and
          $lastObservedDrainState.RunningJobs -eq 0 -and
          $lastObservedDrainState.TimedOutWorkersInFlight -eq 0) {
        $drained = $true
        break
      }
    }
    catch {
    }

    Start-Sleep -Milliseconds 500
  }

  if (-not $readyFlipped) {
    Write-Warning "Readiness did not flip to 503 before stop."
  }
  if (-not $drained) {
    if ($null -ne $lastObservedDrainState) {
      Write-Warning ("Service did not fully drain within {0} seconds. activeHttpRequests={1} queuedJobs={2} runningJobs={3} timedOutWorkersInFlight={4}. Stopping service anyway." -f `
        $DrainTimeoutSeconds,
        $lastObservedDrainState.ActiveHttpRequests,
        $lastObservedDrainState.QueuedJobs,
        $lastObservedDrainState.RunningJobs,
        $lastObservedDrainState.TimedOutWorkersInFlight)
    }
    else {
      Write-Warning "Service did not fully drain within $DrainTimeoutSeconds seconds. Stopping service anyway."
    }
  }

  Stop-Service -Name $ServiceName -ErrorAction Stop
  Wait-ServiceStatus -ServiceName $ServiceName -DesiredStatus "Stopped" -TimeoutSeconds $ServiceStopTimeoutSeconds
  Write-Host "Stopped service: $ServiceName"
}
finally {
  if (-not $KeepDrainSignal) {
    Remove-Item -Path $drainSignalPath -Force -ErrorAction SilentlyContinue
  }
}
