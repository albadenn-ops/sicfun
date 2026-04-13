Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-ReleaseRoot {
  Split-Path -Parent $PSScriptRoot
}

function Resolve-ReleasePath {
  param(
    [string]$PathValue,
    [switch]$AllowBlank,
    [switch]$AllowMissing
  )

  if ([string]::IsNullOrWhiteSpace($PathValue)) {
    if ($AllowBlank) {
      return ""
    }
    throw "Path must be non-empty"
  }

  $releaseRoot = Get-ReleaseRoot
  $candidate =
    if ([System.IO.Path]::IsPathRooted($PathValue)) { $PathValue }
    else { Join-Path $releaseRoot $PathValue }

  $resolved = [System.IO.Path]::GetFullPath($candidate)
  if (-not $AllowMissing -and -not (Test-Path -LiteralPath $resolved)) {
    throw "Path not found: $resolved"
  }

  return $resolved
}

function Convert-ConfigValue {
  param(
    [string]$Value
  )

  if ($null -eq $Value) {
    return ""
  }

  $trimmed = $Value.Trim()
  if ($trimmed.Length -ge 2) {
    if (($trimmed.StartsWith('"') -and $trimmed.EndsWith('"')) -or ($trimmed.StartsWith("'") -and $trimmed.EndsWith("'"))) {
      return $trimmed.Substring(1, $trimmed.Length - 2)
    }
  }

  return $trimmed
}

function Read-EnvConfig {
  param(
    [string]$ConfigFile = "conf\\hand-history-web.env"
  )

  $path = Resolve-ReleasePath -PathValue $ConfigFile
  $values = @{}
  $lineNumber = 0
  foreach ($line in Get-Content -LiteralPath $path) {
    $lineNumber += 1
    $trimmed = $line.Trim()
    if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith("#")) {
      continue
    }

    $separator = $trimmed.IndexOf("=")
    if ($separator -lt 1) {
      throw "Invalid config line $lineNumber in $path. Expected KEY=VALUE."
    }

    $name = $trimmed.Substring(0, $separator).Trim()
    if ([string]::IsNullOrWhiteSpace($name)) {
      throw "Invalid config line $lineNumber in $path. Key must be non-empty."
    }

    $values[$name] = Convert-ConfigValue -Value $trimmed.Substring($separator + 1)
  }

  return $values
}

function Get-ConfigValue {
  param(
    [hashtable]$ConfigValues,
    [string]$Name,
    [string]$Default = ""
  )

  if ($null -ne $ConfigValues -and $ConfigValues.ContainsKey($Name)) {
    return [string]$ConfigValues[$Name]
  }

  return $Default
}

function Resolve-DrainSignalPath {
  param(
    [hashtable]$ConfigValues
  )

  $raw = Get-ConfigValue -ConfigValues $ConfigValues -Name "DRAIN_SIGNAL_FILE" -Default "conf\\service-drain.signal"
  Resolve-ReleasePath -PathValue $raw -AllowMissing
}

function Resolve-ServiceBaseUri {
  param(
    [hashtable]$ConfigValues
  )

  $rawHost = Get-ConfigValue -ConfigValues $ConfigValues -Name "HOST" -Default "127.0.0.1"
  $rawPort = Get-ConfigValue -ConfigValues $ConfigValues -Name "PORT" -Default "8080"
  $probeHost =
    if ([string]::IsNullOrWhiteSpace($rawHost) -or $rawHost -eq "0.0.0.0" -or $rawHost -eq "::") { "127.0.0.1" }
    else { $rawHost }
  if ($probeHost.Contains(":") -and -not ($probeHost.StartsWith("[") -and $probeHost.EndsWith("]"))) {
    $probeHost = "[$probeHost]"
  }

  $port = [int]$rawPort
  return "http://${probeHost}:$port"
}

function Get-ServiceLogPaths {
  $releaseRoot = Get-ReleaseRoot
  $logsDir = Join-Path $releaseRoot "logs"
  [pscustomobject]@{
    LogsDir = $logsDir
    Stdout = Join-Path $logsDir "service.stdout.log"
    Stderr = Join-Path $logsDir "service.stderr.log"
  }
}

function Get-LogTailText {
  param(
    [string]$Path,
    [int]$MaxLines = 20
  )

  if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path -LiteralPath $Path)) {
    return ""
  }

  $lines = Get-Content -LiteralPath $Path -Tail $MaxLines -ErrorAction SilentlyContinue
  if ($null -eq $lines) {
    return ""
  }

  return ($lines -join [Environment]::NewLine).Trim()
}

function Get-JsonFieldValue {
  param(
    [object]$Json,
    [string]$Name
  )

  if ($null -eq $Json) {
    return ""
  }

  $property = $Json.PSObject.Properties[$Name]
  if ($null -eq $property -or $null -eq $property.Value) {
    return ""
  }

  [string]$property.Value
}

function Invoke-JsonProbe {
  param(
    [string]$Uri,
    [int]$TimeoutSeconds = 5
  )

  try {
    $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec $TimeoutSeconds
    $body = [string]$response.Content
    $json = $null
    if (-not [string]::IsNullOrWhiteSpace($body)) {
      try {
        $json = $body | ConvertFrom-Json
      }
      catch {
      }
    }

    return [pscustomobject]@{
      Success = $true
      StatusCode = [int]$response.StatusCode
      Body = $body
      Json = $json
      Error = ""
    }
  }
  catch {
    $statusCode = 0
    $body = ""
    $json = $null
    $response = $_.Exception.Response
    if ($null -ne $response) {
      try {
        $statusCode = [int]$response.StatusCode
      }
      catch {
      }
    }
    if ($null -ne $_.ErrorDetails -and -not [string]::IsNullOrWhiteSpace($_.ErrorDetails.Message)) {
      $body = [string]$_.ErrorDetails.Message
    }
    if (-not [string]::IsNullOrWhiteSpace($body)) {
      try {
        $json = $body | ConvertFrom-Json
      }
      catch {
      }
    }

    return [pscustomobject]@{
      Success = $false
      StatusCode = $statusCode
      Body = $body
      Json = $json
      Error = $_.Exception.Message
    }
  }
}

function Format-ProbeSummary {
  param(
    [string]$Label,
    [object]$Probe
  )

  if ($null -eq $Probe) {
    return "${Label}: unavailable"
  }

  $parts = @("${Label}.statusCode=$($Probe.StatusCode)")
  $stateKeys = @("reason", "readyReason", "status", "error")
  foreach ($key in $stateKeys) {
    $value = Get-JsonFieldValue -Json $Probe.Json -Name $key
    if (-not [string]::IsNullOrWhiteSpace($value)) {
      $parts += "${Label}.${key}=$value"
    }
  }
  if (-not $Probe.Success -and -not [string]::IsNullOrWhiteSpace($Probe.Error)) {
    $parts += "${Label}.error=$($Probe.Error)"
  }

  return ($parts -join " ")
}

function Get-ServiceDiagnosticSummary {
  param(
    [string]$ServiceName,
    [object]$ReadyProbe = $null,
    [object]$HealthProbe = $null,
    [int]$LogTailLines = 20
  )

  $service = Get-ServiceOrNull -ServiceName $ServiceName
  $status =
    if ($null -eq $service) { "missing" }
    else { $service.Status.ToString() }

  $logPaths = Get-ServiceLogPaths
  $stdoutTail = Get-LogTailText -Path $logPaths.Stdout -MaxLines $LogTailLines
  $stderrTail = Get-LogTailText -Path $logPaths.Stderr -MaxLines $LogTailLines

  $parts = @(
    "serviceStatus=$status",
    (Format-ProbeSummary -Label "ready" -Probe $ReadyProbe),
    (Format-ProbeSummary -Label "health" -Probe $HealthProbe)
  )

  if (-not [string]::IsNullOrWhiteSpace($stderrTail)) {
    $parts += "stderr tail ($($logPaths.Stderr)):`n$stderrTail"
  }
  elseif (Test-Path -LiteralPath $logPaths.Stderr) {
    $parts += "stderr tail ($($logPaths.Stderr)): <empty>"
  }
  else {
    $parts += "stderr log missing: $($logPaths.Stderr)"
  }

  if (-not [string]::IsNullOrWhiteSpace($stdoutTail)) {
    $parts += "stdout tail ($($logPaths.Stdout)):`n$stdoutTail"
  }
  elseif (Test-Path -LiteralPath $logPaths.Stdout) {
    $parts += "stdout tail ($($logPaths.Stdout)): <empty>"
  }
  else {
    $parts += "stdout log missing: $($logPaths.Stdout)"
  }

  return ($parts -join [Environment]::NewLine)
}

function Assert-Administrator {
  $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
  $principal = [Security.Principal.WindowsPrincipal]::new($identity)
  if (-not $principal.IsInRole([Security.Principal.WindowsBuiltinRole]::Administrator)) {
    throw "Run this command from an elevated PowerShell session."
  }
}

function Resolve-NssmCommand {
  param(
    [string]$NssmPath = ""
  )

  if (-not [string]::IsNullOrWhiteSpace($NssmPath)) {
    $resolved = Resolve-ReleasePath -PathValue $NssmPath
    return $resolved
  }

  $nssm = Get-Command nssm.exe -ErrorAction SilentlyContinue
  if ($null -eq $nssm) {
    $nssm = Get-Command nssm -ErrorAction SilentlyContinue
  }
  if ($null -eq $nssm -or [string]::IsNullOrWhiteSpace($nssm.Source)) {
    throw "nssm.exe not found. Install NSSM and add it to PATH or pass -NssmPath."
  }

  return $nssm.Source
}

function Get-ServiceOrNull {
  param(
    [string]$ServiceName
  )

  try {
    return Get-Service -Name $ServiceName -ErrorAction Stop
  }
  catch {
    return $null
  }
}

function Get-ServiceCommandLine {
  param(
    [string]$ServiceName
  )

  $escapedName = $ServiceName.Replace("'", "''")
  $service = Get-CimInstance Win32_Service -Filter "Name='$escapedName'" -ErrorAction SilentlyContinue
  if ($null -eq $service) {
    return ""
  }

  return [string]$service.PathName
}

function Get-ServiceInstalledConfigFile {
  param(
    [string]$ServiceName
  )

  $commandLine = Get-ServiceCommandLine -ServiceName $ServiceName
  if ([string]::IsNullOrWhiteSpace($commandLine)) {
    return ""
  }

  $patterns = @(
    '-ConfigFile\s+"([^"]+)"',
    "-ConfigFile\s+'([^']+)'",
    '-ConfigFile\s+(\S+)'
  )

  foreach ($pattern in $patterns) {
    if ($commandLine -match $pattern) {
      return $Matches[1]
    }
  }

  return ""
}

function Wait-ServiceStatus {
  param(
    [string]$ServiceName,
    [string]$DesiredStatus,
    [int]$TimeoutSeconds = 60
  )

  $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
  while ([DateTime]::UtcNow -lt $deadline) {
    $service = Get-ServiceOrNull -ServiceName $ServiceName
    if ($DesiredStatus -eq "Deleted") {
      if ($null -eq $service) {
        return
      }
    }
    elseif ($null -ne $service -and $service.Status.ToString() -eq $DesiredStatus) {
      return
    }

    Start-Sleep -Milliseconds 500
  }

  throw "Service '$ServiceName' did not reach state '$DesiredStatus' within $TimeoutSeconds seconds."
}
