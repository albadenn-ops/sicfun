[CmdletBinding()]
param(
  [string]$ReleaseRoot = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ReleaseRoot)) {
  $ReleaseRoot = Split-Path -Parent $PSScriptRoot
}

$ReleaseRoot = (Resolve-Path -LiteralPath $ReleaseRoot).Path
$manifestPath = Join-Path $ReleaseRoot "manifest.sha256"

if (-not (Test-Path -LiteralPath $manifestPath)) {
  throw "Manifest not found: $manifestPath"
}

$expected = @{}
foreach ($line in Get-Content -LiteralPath $manifestPath) {
  if ([string]::IsNullOrWhiteSpace($line)) {
    continue
  }

  $match = [regex]::Match($line, '^([0-9a-fA-F]{64})  (.+)$')
  if (-not $match.Success) {
    throw "Malformed manifest line: $line"
  }

  $expected[$match.Groups[2].Value] = $match.Groups[1].Value.ToLowerInvariant()
}

$actual = @{}
$files = Get-ChildItem -Path $ReleaseRoot -File -Recurse | Where-Object { $_.FullName -ne $manifestPath }
foreach ($file in $files) {
  $relative = $file.FullName.Substring($ReleaseRoot.Length).TrimStart('\', '/').Replace('\', '/')
  $actual[$relative] = (Get-FileHash -Path $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}

$missing = [System.Collections.Generic.List[string]]::new()
$modified = [System.Collections.Generic.List[string]]::new()
$extra = [System.Collections.Generic.List[string]]::new()

foreach ($relative in $expected.Keys | Sort-Object) {
  if (-not $actual.ContainsKey($relative)) {
    $missing.Add($relative)
    continue
  }
  if ($actual[$relative] -ne $expected[$relative]) {
    $modified.Add($relative)
  }
}

foreach ($relative in $actual.Keys | Sort-Object) {
  if (-not $expected.ContainsKey($relative)) {
    $extra.Add($relative)
  }
}

if ($missing.Count -gt 0 -or $modified.Count -gt 0 -or $extra.Count -gt 0) {
  if ($missing.Count -gt 0) {
    Write-Host "Missing files:"
    $missing | ForEach-Object { Write-Host "  $_" }
  }
  if ($modified.Count -gt 0) {
    Write-Host "Modified files:"
    $modified | ForEach-Object { Write-Host "  $_" }
  }
  if ($extra.Count -gt 0) {
    Write-Host "Unexpected files:"
    $extra | ForEach-Object { Write-Host "  $_" }
  }
  throw "Release manifest verification failed."
}

Write-Host "Release manifest verified: $ReleaseRoot"
