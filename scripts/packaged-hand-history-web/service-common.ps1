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
