[CmdletBinding()]
param(
  [ValidateSet("doctor", "auth", "delegate")]
  [string]$Action = "doctor",

  [ValidateSet("analysis", "review", "draft-patch", "custom")]
  [string]$Mode = "analysis",

  [string]$Task,
  [string[]]$ContextPath = @(),
  [string[]]$IncludeDirectories = @(),
  [string]$Model,

  [ValidateSet("text", "json", "stream-json")]
  [string]$OutputFormat = "text",

  [string]$OutputPath,
  [switch]$NoBrowser,
  [switch]$WhatIf
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sidecarCacheDir = Join-Path $repoRoot ".tool-cache\gemini-sidecar"
$googleAuthType = "oauth-personal"

function Find-CommandPath {
  param([string]$Name)

  $command = Get-Command $Name -ErrorAction SilentlyContinue
  if ($null -eq $command) {
    return $null
  }

  return $command.Source
}

function Resolve-NodePath {
  $nodePath = Find-CommandPath -Name "node"
  if ([string]::IsNullOrWhiteSpace($nodePath)) {
    throw "node.exe was not found on PATH."
  }

  return $nodePath
}

function Resolve-NpmPath {
  $npmPath = Find-CommandPath -Name "npm"
  if ([string]::IsNullOrWhiteSpace($npmPath)) {
    throw "npm.cmd was not found on PATH."
  }

  return $npmPath
}

function Get-GlobalNpmRoot {
  $npmPath = Resolve-NpmPath
  $root = (& $npmPath root -g).Trim()
  if ([string]::IsNullOrWhiteSpace($root)) {
    throw "Unable to resolve the global npm root."
  }

  return $root
}

function Resolve-GeminiEntrypoint {
  $npmRoot = Get-GlobalNpmRoot
  $candidate = Join-Path $npmRoot "@google\gemini-cli\dist\index.js"
  if (-not (Test-Path $candidate)) {
    throw "Gemini CLI entrypoint not found at $candidate. Install it with: npm install -g @google/gemini-cli"
  }

  return $candidate
}

function Resolve-GeminiPackageJson {
  $npmRoot = Get-GlobalNpmRoot
  $candidate = Join-Path $npmRoot "@google\gemini-cli\package.json"
  if (-not (Test-Path $candidate)) {
    throw "Gemini CLI package.json not found at $candidate."
  }

  return $candidate
}

function Get-GeminiVersion {
  $packageJsonPath = Resolve-GeminiPackageJson
  $packageJson = Get-Content -Raw $packageJsonPath | ConvertFrom-Json
  return [string]$packageJson.version
}

function Get-GeminiSettingsPath {
  return [System.IO.Path]::Combine($HOME, ".gemini", "settings.json")
}

function Get-GeminiSelectedAuthType {
  $settingsPath = Get-GeminiSettingsPath
  if (-not (Test-Path $settingsPath)) {
    return $null
  }

  try {
    $settings = Get-Content -Raw $settingsPath | ConvertFrom-Json
  }
  catch {
    return "<unreadable>"
  }

  if ($null -eq $settings.security -or $null -eq $settings.security.auth) {
    return $null
  }

  return [string]$settings.security.auth.selectedType
}

function Write-Utf8NoBom {
  param(
    [string]$Path,
    [string]$Value
  )

  $encoding = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($Path, $Value, $encoding)
}

function Resolve-WorkspacePath {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    throw "Path must not be empty."
  }

  $candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
    $Path
  }
  else {
    Join-Path $repoRoot $Path
  }

  return [System.IO.Path]::GetFullPath($candidate)
}

function Get-RepoRelativePath {
  param([string]$AbsolutePath)

  $repoUri = [System.Uri]($repoRoot.TrimEnd("\") + "\")
  $pathUri = [System.Uri]$AbsolutePath
  if (-not $repoUri.IsBaseOf($pathUri)) {
    return $AbsolutePath
  }

  $relative = $repoUri.MakeRelativeUri($pathUri).ToString()
  return [System.Uri]::UnescapeDataString($relative).Replace("/", "\")
}

function Ensure-SidecarCacheDir {
  if (-not (Test-Path $sidecarCacheDir)) {
    New-Item -ItemType Directory -Path $sidecarCacheDir -Force | Out-Null
  }
}

function Ensure-GoogleAuthSelection {
  $settingsPath = Get-GeminiSettingsPath
  if (Test-Path $settingsPath) {
    $selectedType = Get-GeminiSelectedAuthType
    if ($selectedType -eq "<unreadable>") {
      $backupPath = "$settingsPath.bak-" + (Get-Date -Format "yyyyMMdd-HHmmss")
      Copy-Item -Path $settingsPath -Destination $backupPath -Force

      $payload = @{
        security = @{
          auth = @{
            selectedType = $googleAuthType
          }
        }
      } | ConvertTo-Json -Depth 10

      Write-Utf8NoBom -Path $settingsPath -Value $payload
      Write-Host "[gemini-sidecar] backed up unreadable settings to $backupPath"
      Write-Host "[gemini-sidecar] rewrote $settingsPath with security.auth.selectedType=$googleAuthType"
      return
    }

    if ([string]::IsNullOrWhiteSpace($selectedType)) {
      Write-Host "[gemini-sidecar] settings exist but security.auth.selectedType is not set: $settingsPath"
      Write-Host "[gemini-sidecar] if login does not start automatically, open Gemini and run /auth."
    }
    return
  }

  $settingsDir = Split-Path -Parent $settingsPath
  New-Item -ItemType Directory -Path $settingsDir -Force | Out-Null

  $payload = @{
    security = @{
      auth = @{
        selectedType = $googleAuthType
      }
    }
  } | ConvertTo-Json -Depth 10

  Write-Utf8NoBom -Path $settingsPath -Value $payload
  Write-Host "[gemini-sidecar] created $settingsPath with security.auth.selectedType=$googleAuthType"
}

function Get-DefaultArtifactPath {
  param([string]$Suffix)

  Ensure-SidecarCacheDir
  $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
  return Join-Path $sidecarCacheDir ("$timestamp-$Mode.$Suffix")
}

function Get-SharedContractPath {
  return Join-Path $repoRoot "AI_ENTRYPOINT.md"
}

function Get-ProviderContractPath {
  return Join-Path $repoRoot "GEMINI.md"
}

function Get-ContractText {
  param(
    [string]$Path,
    [string]$Label
  )

  if ([string]::IsNullOrWhiteSpace($Path) -or -not (Test-Path $Path)) {
    return "${Label}: <missing>"
  }

  $content = Get-Content -Raw $Path -ErrorAction SilentlyContinue
  if ([string]::IsNullOrWhiteSpace($content)) {
    return "${Label}: <empty>"
  }

  $repoPath = Get-RepoRelativePath -AbsolutePath $Path
  return @(
    "$Label ($repoPath):"
    $content.Trim()
  ) -join "`n`n"
}

function Get-ModeContract {
  param([string]$CurrentMode)

  switch ($CurrentMode) {
    "analysis" {
      return @(
        "- Summarize the answer for another coding agent."
        "- Cite repo-relative files and symbols for concrete claims."
        "- Focus on the shortest useful handoff."
      )
    }
    "review" {
      return @(
        "- Findings first, ordered by severity."
        "- Focus on bugs, regressions, weak assumptions, and missing validation."
        "- Cite repo-relative paths and explain impact clearly."
      )
    }
    "draft-patch" {
      return @(
        "- Propose a patch plan or diff sketch only."
        "- Do not edit files and do not claim that files were changed."
        "- Call out which files would need modification."
      )
    }
    default {
      return @(
        "- Answer the task directly and keep it concise."
        "- Stay read-only unless the prompt explicitly says otherwise."
        "- Cite repo-relative paths when making repo-specific claims."
      )
    }
  }
}

function New-DelegationPrompt {
  param(
    [string]$CurrentMode,
    [string]$CurrentTask,
    [string[]]$ResolvedContextPaths
  )

  $contextBlock = if ($ResolvedContextPaths.Count -gt 0) {
    @(
      "Inspect these paths first:"
      ($ResolvedContextPaths | ForEach-Object { "- $(Get-RepoRelativePath -AbsolutePath $_)" })
    ) -join "`n"
  }
  else {
    "Inspect the relevant repository files before answering."
  }

  $contractLines = (Get-ModeContract -CurrentMode $CurrentMode) -join "`n"
  $sharedContractBlock = Get-ContractText -Path (Get-SharedContractPath) -Label "Shared repository contract"
  $providerContractBlock = Get-ContractText -Path (Get-ProviderContractPath) -Label "Provider role overlay"

  return @"
You are a delegated worker running inside the SICFUN repository.

Provider: gemini
Mode: $CurrentMode
Working directory: $repoRoot

Task:
$CurrentTask

Shared contract:
$sharedContractBlock

Provider overlay:
$providerContractBlock

Rules:
- Stay read-only. Do not edit files and do not claim to have edited files.
- Prefer concrete repository evidence over guesses.
- Keep the response concise and directly usable by another coding agent.
- Use repo-relative file paths when you reference code or docs.

$contextBlock

Response contract:
$contractLines
"@
}

function Show-Doctor {
  $nodePath = Find-CommandPath -Name "node"
  $npmPath = Find-CommandPath -Name "npm"
  $settingsPath = Get-GeminiSettingsPath
  $selectedAuthType = Get-GeminiSelectedAuthType
  $sharedContractPath = Get-SharedContractPath
  $geminiMdPath = Get-ProviderContractPath

  Write-Host "[gemini-sidecar] repo root: $repoRoot"
  Write-Host "[gemini-sidecar] node: $(if ($nodePath) { $nodePath } else { '<missing>' })"
  Write-Host "[gemini-sidecar] npm: $(if ($npmPath) { $npmPath } else { '<missing>' })"

  try {
    $entrypoint = Resolve-GeminiEntrypoint
    $version = Get-GeminiVersion
    Write-Host "[gemini-sidecar] gemini cli entrypoint: $entrypoint"
    Write-Host "[gemini-sidecar] gemini cli version: $version"
  }
  catch {
    Write-Host "[gemini-sidecar] gemini cli: <missing>"
  }

  Write-Host "[gemini-sidecar] settings path: $settingsPath"
  Write-Host "[gemini-sidecar] selected auth type: $(if ($selectedAuthType) { $selectedAuthType } else { '<unset>' })"
  Write-Host "[gemini-sidecar] shared contract: $(if (Test-Path $sharedContractPath) { $sharedContractPath } else { '<missing>' })"
  Write-Host "[gemini-sidecar] repo GEMINI.md: $(if (Test-Path $geminiMdPath) { $geminiMdPath } else { '<missing>' })"
  Write-Host "[gemini-sidecar] cache dir: $sidecarCacheDir"
}

function Invoke-GeminiAuthAttempt {
  param(
    [string]$NodePath,
    [string]$Entrypoint,
    [switch]$Manual
  )

  if ($Manual) {
    $env:NO_BROWSER = "true"
    Write-Host "[gemini-sidecar] starting manual auth flow..."
  }
  else {
    Remove-Item Env:NO_BROWSER -ErrorAction SilentlyContinue
    Write-Host "[gemini-sidecar] starting browser auth flow..."
  }

  & $NodePath $Entrypoint
  return $LASTEXITCODE
}

function Invoke-GeminiAuth {
  $nodePath = Resolve-NodePath
  $entrypoint = Resolve-GeminiEntrypoint
  $settingsPath = Get-GeminiSettingsPath

  Write-Host "[gemini-sidecar] preparing Google login..."
  Write-Host "[gemini-sidecar] settings file: $settingsPath"
  Write-Host "[gemini-sidecar] command: node <gemini-entrypoint>"
  Write-Host "[gemini-sidecar] if prompted, keep 'Login with Google' selected."
  if ($NoBrowser) {
    Write-Host "[gemini-sidecar] NO_BROWSER=true enabled. Gemini will print a URL and ask for the authorization code."
  }
  else {
    Write-Host "[gemini-sidecar] Gemini will ask 'Do you want to continue? [Y/n]:' before attempting to open the browser."
    Write-Host "[gemini-sidecar] If Gemini hits the known consent bug, this wrapper will retry in manual auth mode automatically."
  }

  if ($WhatIf) {
    Write-Host "[gemini-sidecar] WhatIf enabled. Command not executed."
    return
  }

  Ensure-GoogleAuthSelection

  $previousDefaultAuth = $env:GEMINI_DEFAULT_AUTH_TYPE
  $previousNoBrowser = $env:NO_BROWSER
  Push-Location $repoRoot
  try {
    $env:GEMINI_DEFAULT_AUTH_TYPE = $googleAuthType
    $exitCode = Invoke-GeminiAuthAttempt -NodePath $nodePath -Entrypoint $entrypoint -Manual:$NoBrowser

    if (($exitCode -eq 41) -and (-not $NoBrowser)) {
      Write-Host "[gemini-sidecar] Gemini browser auth failed with exit code 41."
      Write-Host "[gemini-sidecar] Retrying with NO_BROWSER=true so Gemini prints the URL and accepts a pasted authorization code."
      $exitCode = Invoke-GeminiAuthAttempt -NodePath $nodePath -Entrypoint $entrypoint -Manual
    }

    if ($exitCode -ne 0) {
      throw "Gemini CLI exited with code $exitCode during auth."
    }
  }
  finally {
    if ($null -eq $previousDefaultAuth) {
      Remove-Item Env:GEMINI_DEFAULT_AUTH_TYPE -ErrorAction SilentlyContinue
    }
    else {
      $env:GEMINI_DEFAULT_AUTH_TYPE = $previousDefaultAuth
    }
    if ($null -eq $previousNoBrowser) {
      Remove-Item Env:NO_BROWSER -ErrorAction SilentlyContinue
    }
    else {
      $env:NO_BROWSER = $previousNoBrowser
    }
    Pop-Location
  }
}

function Invoke-GeminiDelegate {
  if ([string]::IsNullOrWhiteSpace($Task)) {
    throw "Delegate mode requires -Task."
  }

  $nodePath = Resolve-NodePath
  $entrypoint = Resolve-GeminiEntrypoint

  $resolvedContextPaths = @()
  foreach ($path in $ContextPath) {
    $resolvedContextPaths += Resolve-WorkspacePath -Path $path
  }

  $resolvedIncludeDirectories = @()
  foreach ($dir in $IncludeDirectories) {
    $resolvedIncludeDirectories += Resolve-WorkspacePath -Path $dir
  }

  $prompt = New-DelegationPrompt -CurrentMode $Mode -CurrentTask $Task -ResolvedContextPaths $resolvedContextPaths

  $artifactExtension = switch ($OutputFormat) {
    "json" { "json" }
    "stream-json" { "jsonl" }
    default { "txt" }
  }

  $resolvedOutputPath = if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    Get-DefaultArtifactPath -Suffix $artifactExtension
  }
  else {
    Resolve-WorkspacePath -Path $OutputPath
  }

  $promptPath = [System.IO.Path]::ChangeExtension($resolvedOutputPath, "prompt.txt")
  $outputDirectory = Split-Path -Parent $resolvedOutputPath
  if (-not (Test-Path $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
  }

  Write-Utf8NoBom -Path $promptPath -Value $prompt

  $cliArgs = @(
    $entrypoint,
    "--output-format", $OutputFormat,
    "-p", $prompt
  )

  if (-not [string]::IsNullOrWhiteSpace($Model)) {
    $cliArgs += @("-m", $Model)
  }

  foreach ($dir in $resolvedIncludeDirectories) {
    $cliArgs += @("--include-directories", $dir)
  }

  Write-Host "[gemini-sidecar] mode: $Mode"
  Write-Host "[gemini-sidecar] output format: $OutputFormat"
  Write-Host "[gemini-sidecar] prompt file: $promptPath"
  Write-Host "[gemini-sidecar] output file: $resolvedOutputPath"
  Write-Host "[gemini-sidecar] context paths: $(if ($resolvedContextPaths.Count -gt 0) { ($resolvedContextPaths | ForEach-Object { Get-RepoRelativePath -AbsolutePath $_ }) -join ', ' } else { '<none>' })"

  if ($WhatIf) {
    Write-Host "[gemini-sidecar] command: node <gemini-entrypoint> --output-format $OutputFormat -p <prompt>"
    Write-Host "[gemini-sidecar] WhatIf enabled. Command not executed."
    return
  }

  Push-Location $repoRoot
  try {
    & $nodePath @cliArgs | Tee-Object -FilePath $resolvedOutputPath
    if ($LASTEXITCODE -ne 0) {
      throw "Gemini CLI exited with code $LASTEXITCODE. See $resolvedOutputPath"
    }
  }
  finally {
    Pop-Location
  }
}

switch ($Action) {
  "doctor" {
    Show-Doctor
  }
  "auth" {
    Invoke-GeminiAuth
  }
  "delegate" {
    Invoke-GeminiDelegate
  }
}
