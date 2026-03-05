[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$Title,
  [Parameter(Mandatory = $true)]
  [string]$Summary,
  [string]$Why = "",
  [string]$Files = "",
  [string]$Validation = "",
  [string]$Risks = "",
  [string]$ArchivePath = "docs/AI_CONTEXT_ARCHIVE.md"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $repoRoot
try {
  $target = Join-Path $repoRoot $ArchivePath
  $dir = Split-Path -Parent $target
  if (-not [string]::IsNullOrWhiteSpace($dir)) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
  }
  if (-not (Test-Path $target)) {
    Set-Content -Path $target -Encoding UTF8 -Value @(
      "# AI Context Archive",
      "",
      "Append-only context ledger.",
      ""
    )
  }

  $timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

  $gitHead = ""
  $gitStatus = ""
  try {
    $gitHead = (git rev-parse --short HEAD 2>$null).Trim()
    $gitStatus = ((git status --short 2>$null) -join "; ").Trim()
  }
  catch {
    $gitHead = ""
    $gitStatus = ""
  }

  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("## $timestamp - $Title")
  $lines.Add("Summary: $Summary")
  if (-not [string]::IsNullOrWhiteSpace($Why)) { $lines.Add("Why: $Why") }
  if (-not [string]::IsNullOrWhiteSpace($Files)) { $lines.Add("Files: $Files") }
  if (-not [string]::IsNullOrWhiteSpace($Validation)) { $lines.Add("Validation: $Validation") }
  if (-not [string]::IsNullOrWhiteSpace($Risks)) { $lines.Add("Risks: $Risks") }
  if (-not [string]::IsNullOrWhiteSpace($gitHead)) { $lines.Add("GitHead: $gitHead") }
  if (-not [string]::IsNullOrWhiteSpace($gitStatus)) { $lines.Add("GitStatus: $gitStatus") }
  $lines.Add("")

  Add-Content -Path $target -Encoding UTF8 -Value $lines
  Write-Host "Appended context entry to $ArchivePath"
}
finally {
  Pop-Location
}
