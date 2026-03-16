<#
.SYNOPSIS
  Unified multi-provider AI sidecar dispatcher for the SICFUN repository.

.DESCRIPTION
  Delegates read-only tasks to Gemini, Claude, or GPT/Codex. The delegated
  worker is always advisory: Codex remains responsible for edits, validation,
  and final judgment.

  Artifacts are stored under .tool-cache/ai-minions/<provider>/.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 -Action doctor

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 `
    -Action auth `
    -Provider gpt `
    -NoBrowser

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts/ai-minion.ps1 `
    -Action delegate `
    -Provider gemini `
    -Mode analysis `
    -Task "Summarize the relevant implementation and risks for this task." `
    -ContextPath README.md,ROADMAP.md `
    -OutputFormat text
#>

[CmdletBinding()]
param(
  [ValidateSet("doctor", "auth", "delegate")]
  [string]$Action = "doctor",

  [ValidateSet("all", "gemini", "claude", "gpt")]
  [string]$Provider = "all",

  [ValidateSet("analysis", "review", "draft-patch", "custom")]
  [string]$Mode = "analysis",

  [string]$Task,
  [string[]]$ContextPath = @(),
  [string[]]$IncludeDirectories = @(),
  [string]$Model,

  [ValidateSet("text", "json")]
  [string]$OutputFormat = "text",

  [string]$OutputPath,
  [switch]$InjectContext,
  [switch]$NoBrowser,
  [switch]$WhatIf
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$cacheRoot = Join-Path $repoRoot ".tool-cache\ai-minions"
$geminiSettingsPath = [System.IO.Path]::Combine($HOME, ".gemini", "settings.json")
$claudeProjectRoot = [System.IO.Path]::Combine($HOME, ".claude", "projects")
$codexConfigPath = [System.IO.Path]::Combine($HOME, ".codex", "config.toml")

function Write-Utf8NoBom {
  param(
    [string]$Path,
    [string]$Value
  )

  $encoding = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($Path, $Value, $encoding)
}

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

function Invoke-NativeCommandCapture {
  param(
    [string]$FilePath,
    [string[]]$ArgumentList
  )

  $hasPreferenceVariable = Test-Path Variable:PSNativeCommandUseErrorActionPreference
  if ($hasPreferenceVariable) {
    $previousPreference = $PSNativeCommandUseErrorActionPreference
    $script:PSNativeCommandUseErrorActionPreference = $false
  }

  try {
    return @(& $FilePath @ArgumentList 2>&1 | ForEach-Object { [string]$_ })
  }
  finally {
    if ($hasPreferenceVariable) {
      $script:PSNativeCommandUseErrorActionPreference = $previousPreference
    }
  }
}

function Get-GlobalNpmRoot {
  $npmPath = Resolve-NpmPath
  $root = (& $npmPath root -g).Trim()
  if ([string]::IsNullOrWhiteSpace($root)) {
    throw "Unable to resolve the global npm root."
  }

  return $root
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

function Expand-DelimitedArguments {
  param([string[]]$Values)

  $expanded = @()
  foreach ($value in $Values) {
    if ([string]::IsNullOrWhiteSpace($value)) {
      continue
    }

    foreach ($piece in ($value -split ",")) {
      $trimmed = $piece.Trim()
      if (-not [string]::IsNullOrWhiteSpace($trimmed)) {
        $expanded += $trimmed
      }
    }
  }

  return $expanded
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

function Ensure-ProviderCacheDir {
  param([string]$ProviderName)

  $providerDir = Join-Path $cacheRoot $ProviderName
  if (-not (Test-Path $providerDir)) {
    New-Item -ItemType Directory -Path $providerDir -Force | Out-Null
  }

  return $providerDir
}

function Get-DefaultArtifactPath {
  param(
    [string]$ProviderName,
    [string]$Extension
  )

  $providerDir = Ensure-ProviderCacheDir -ProviderName $ProviderName
  $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
  return Join-Path $providerDir ("$timestamp-$Mode.$Extension")
}

function Get-ProviderContractPath {
  param([string]$ProviderName)

  switch ($ProviderName) {
    "gemini" { return Join-Path $repoRoot "GEMINI.md" }
    "claude" { return Join-Path $repoRoot "CLAUDE.md" }
    "gpt" { return Join-Path $repoRoot "GPT.md" }
    default { return $null }
  }
}

function Get-SharedContractPath {
  return Join-Path $repoRoot "AI_ENTRYPOINT.md"
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

function Resolve-ContextPaths {
  $resolvedPaths = @()
  foreach ($path in (Expand-DelimitedArguments -Values $ContextPath)) {
    $resolvedPaths += Resolve-WorkspacePath -Path $path
  }
  if ($resolvedPaths.Count -eq 0) {
    return @()
  }
  return $resolvedPaths
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
        "- Stay read-only unless the task explicitly says otherwise."
        "- Cite repo-relative paths when making repo-specific claims."
      )
    }
  }
}

function Build-InjectedContext {
  param([string[]]$ResolvedPaths)

  $blocks = New-Object System.Collections.Generic.List[string]
  foreach ($path in $ResolvedPaths) {
    $repoPath = Get-RepoRelativePath -AbsolutePath $path
    if (Test-Path $path -PathType Container) {
      $blocks.Add("## Directory: $repoPath")
      $children = @(Get-ChildItem $path -Recurse -File -ErrorAction SilentlyContinue |
        Select-Object -First 50 |
        ForEach-Object { Get-RepoRelativePath -AbsolutePath $_.FullName })
      if ($children.Count -gt 0) {
        $blocks.Add(($children -join "`n"))
      }
      else {
        $blocks.Add("<empty directory or no readable files>")
      }
      continue
    }

    if (-not (Test-Path $path)) {
      $blocks.Add("## Missing Path: $repoPath")
      continue
    }

    $content = Get-Content -Raw $path -ErrorAction SilentlyContinue
    if ($null -eq $content) {
      $blocks.Add("## File: $repoPath")
      $blocks.Add("<unreadable>")
      continue
    }

    if ($content.Length -gt 20000) {
      $content = $content.Substring(0, 20000) + "`n... [truncated at 20000 chars]"
    }

    $blocks.Add("## File: $repoPath")
    $blocks.Add('```')
    $blocks.Add($content)
    $blocks.Add('```')
  }

  return ($blocks -join "`n`n")
}

function New-DelegationPrompt {
  param(
    [string]$ProviderName,
    [string]$CurrentMode,
    [string]$CurrentTask,
    [string[]]$ResolvedContextPaths,
    [switch]$InjectContextBlock,
    [switch]$NoTools
  )

  $contextBlock = if ($InjectContextBlock -and $ResolvedContextPaths.Count -gt 0) {
    @(
      "Inspect this injected repository context first:"
      (Build-InjectedContext -ResolvedPaths $ResolvedContextPaths)
    ) -join "`n`n"
  }
  elseif ($ResolvedContextPaths.Count -gt 0) {
    @(
      "Inspect these paths first:"
      ($ResolvedContextPaths | ForEach-Object { "- $(Get-RepoRelativePath -AbsolutePath $_)" })
    ) -join "`n"
  }
  elseif ($NoTools) {
    "No repository context was provided. You do not have file-reading tools in this delegated run, so answer only from the task text and clearly say if more repository context is needed."
  }
  else {
    "Inspect the relevant repository files before answering."
  }

  $contractBlock = (Get-ModeContract -CurrentMode $CurrentMode) -join "`n"
  $sharedContractBlock = Get-ContractText -Path (Get-SharedContractPath) -Label "Shared repository contract"
  $providerContractBlock = Get-ContractText -Path (Get-ProviderContractPath -ProviderName $ProviderName) -Label "Provider role overlay"

  return @"
You are a delegated worker inside the SICFUN repository.

Provider: $ProviderName
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
$contractBlock
"@
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

function Get-GeminiSelectedAuthType {
  if (-not (Test-Path $geminiSettingsPath)) {
    return $null
  }

  try {
    $settings = Get-Content -Raw $geminiSettingsPath | ConvertFrom-Json
  }
  catch {
    return "<unreadable>"
  }

  if ($null -eq $settings.security -or $null -eq $settings.security.auth) {
    return $null
  }

  return [string]$settings.security.auth.selectedType
}

function Invoke-GeminiDoctor {
  Write-Host "[gemini]"
  Write-Host "  cli entrypoint: $(Resolve-GeminiEntrypoint)"
  Write-Host "  cli version: $(Get-GeminiVersion)"
  Write-Host "  settings path: $geminiSettingsPath"
  Write-Host "  selected auth type: $(if (Get-GeminiSelectedAuthType) { Get-GeminiSelectedAuthType } else { '<unset>' })"
  $contractPath = Get-ProviderContractPath -ProviderName "gemini"
  Write-Host "  repo contract: $(if (Test-Path $contractPath) { $contractPath } else { '<missing>' })"
}

function Invoke-GeminiAuth {
  $scriptPath = Join-Path $repoRoot "scripts\gemini-sidecar.ps1"
  if (-not (Test-Path $scriptPath)) {
    throw "Missing $scriptPath"
  }

  $invokeParams = @{
    Action = "auth"
  }
  if ($NoBrowser) {
    $invokeParams.NoBrowser = $true
  }
  if ($WhatIf) {
    $invokeParams.WhatIf = $true
  }

  & $scriptPath @invokeParams
  if ($LASTEXITCODE -ne 0) {
    throw "Gemini auth failed with exit code $LASTEXITCODE."
  }
}

function Invoke-GeminiDelegate {
  param(
    [string]$ResolvedOutputPath,
    [string[]]$ResolvedContextPaths
  )

  if ([string]::IsNullOrWhiteSpace($Task)) {
    throw "Delegate action requires -Task."
  }

  if ($InjectContext) {
    $nodePath = Resolve-NodePath
    $entrypoint = Resolve-GeminiEntrypoint
    $prompt = New-DelegationPrompt -ProviderName "gemini" -CurrentMode $Mode -CurrentTask $Task -ResolvedContextPaths $ResolvedContextPaths -InjectContextBlock
    $promptPath = [System.IO.Path]::ChangeExtension($ResolvedOutputPath, "prompt.txt")
    Write-Utf8NoBom -Path $promptPath -Value $prompt

    # Pipe full prompt via stdin; -p gets a short trigger appended after stdin
    $cliArgs = @(
      $entrypoint,
      "--output-format", $OutputFormat,
      "-p", "Answer the task above."
    )
    if (-not [string]::IsNullOrWhiteSpace($Model)) {
      $cliArgs += @("-m", $Model)
    }

    Write-Host "[ai-minion:gemini] mode: $Mode"
    Write-Host "[ai-minion:gemini] output: $ResolvedOutputPath"
    Write-Host "[ai-minion:gemini] prompt file: $promptPath"

    if ($WhatIf) {
      Write-Host "[ai-minion:gemini] WhatIf enabled. Command not executed."
      return
    }

    Push-Location $repoRoot
    try {
      # Pipe prompt via stdin; -p "" enables non-interactive mode
      $prompt | & $nodePath @cliArgs | Tee-Object -FilePath $ResolvedOutputPath
      if ($LASTEXITCODE -ne 0) {
        throw "Gemini CLI exited with code $LASTEXITCODE."
      }
    }
    finally {
      Pop-Location
    }
    return
  }

  $scriptPath = Join-Path $repoRoot "scripts\gemini-sidecar.ps1"
  if (-not (Test-Path $scriptPath)) {
    throw "Missing $scriptPath"
  }

  $invokeParams = @{
    Action = "delegate"
    Mode = $Mode
    Task = $Task
    OutputFormat = $OutputFormat
    OutputPath = $ResolvedOutputPath
  }
  $expandedContextPaths = @(Expand-DelimitedArguments -Values $ContextPath)
  if ($expandedContextPaths.Count -gt 0) {
    $invokeParams.ContextPath = $expandedContextPaths
  }
  $expandedIncludeDirectories = @(Expand-DelimitedArguments -Values $IncludeDirectories)
  if ($expandedIncludeDirectories.Count -gt 0) {
    $invokeParams.IncludeDirectories = $expandedIncludeDirectories
  }
  if (-not [string]::IsNullOrWhiteSpace($Model)) {
    $invokeParams.Model = $Model
  }
  if ($WhatIf) {
    $invokeParams.WhatIf = $true
  }

  & $scriptPath @invokeParams
  if ($LASTEXITCODE -ne 0) {
    throw "Gemini delegate failed with exit code $LASTEXITCODE."
  }
}

function Resolve-ClaudePath {
  $claudePath = Find-CommandPath -Name "claude"
  if ([string]::IsNullOrWhiteSpace($claudePath)) {
    throw "claude.exe was not found on PATH."
  }

  return $claudePath
}

function Get-ClaudeVersion {
  $claudePath = Resolve-ClaudePath
  return (& $claudePath --version).Trim()
}

function Get-ClaudeAuthStatus {
  $claudePath = Resolve-ClaudePath
  $raw = & $claudePath auth status --json
  if ($LASTEXITCODE -ne 0) {
    throw "claude auth status failed with exit code $LASTEXITCODE."
  }
  return $raw | ConvertFrom-Json
}

function Get-ClaudeProjectSlug {
  param([string]$Path)

  return $Path.Replace(":", "-").Replace("\", "-").Replace("/", "-")
}

function Get-ClaudeSessionTranscriptPath {
  param([string]$SessionId)

  $projectDir = Join-Path $claudeProjectRoot (Get-ClaudeProjectSlug -Path $repoRoot)
  if (-not (Test-Path $projectDir)) {
    return $null
  }

  $candidate = Join-Path $projectDir "$SessionId.jsonl"
  if (Test-Path $candidate) {
    return $candidate
  }

  return $null
}

function Wait-For-ClaudeSessionTranscript {
  param([string]$SessionId)

  for ($attempt = 0; $attempt -lt 30; $attempt++) {
    $candidate = Get-ClaudeSessionTranscriptPath -SessionId $SessionId
    if ($candidate) {
      return $candidate
    }
    Start-Sleep -Milliseconds 250
  }

  return $null
}

function Get-ClaudeAssistantEvent {
  param([string]$SessionId)

  $transcriptPath = Wait-For-ClaudeSessionTranscript -SessionId $SessionId
  if (-not $transcriptPath) {
    return $null
  }

  $assistantEvent = $null
  foreach ($line in Get-Content $transcriptPath) {
    if ([string]::IsNullOrWhiteSpace($line)) {
      continue
    }

    try {
      $event = $line | ConvertFrom-Json
    }
    catch {
      continue
    }

    if ($event.type -eq "assistant" -and $null -ne $event.message) {
      $assistantEvent = $event
    }
  }

  return $assistantEvent
}

function Get-ClaudeAssistantText {
  param($AssistantEvent)

  if ($null -eq $AssistantEvent -or $null -eq $AssistantEvent.message -or $null -eq $AssistantEvent.message.content) {
    return $null
  }

  $parts = New-Object System.Collections.Generic.List[string]
  foreach ($item in $AssistantEvent.message.content) {
    if ($item.type -eq "text") {
      $parts.Add([string]$item.text)
    }
  }

  $text = ($parts -join "`n").Trim()
  if ([string]::IsNullOrWhiteSpace($text)) {
    return $null
  }

  return $text
}

function Invoke-ClaudeDoctor {
  $status = Get-ClaudeAuthStatus
  Write-Host "[claude]"
  Write-Host "  cli path: $(Resolve-ClaudePath)"
  Write-Host "  cli version: $(Get-ClaudeVersion)"
  Write-Host "  logged in: $($status.loggedIn)"
  if ($status.loggedIn) {
    Write-Host "  auth method: $($status.authMethod)"
    Write-Host "  email: $($status.email)"
  }
  Write-Host "  project root: $claudeProjectRoot"
  $sharedContractPath = Get-SharedContractPath
  Write-Host "  shared contract: $(if (Test-Path $sharedContractPath) { $sharedContractPath } else { '<missing>' })"
  $contractPath = Get-ProviderContractPath -ProviderName "claude"
  Write-Host "  repo contract: $(if (Test-Path $contractPath) { $contractPath } else { '<missing>' })"
}

function Invoke-ClaudeAuth {
  if ($NoBrowser) {
    throw "Claude CLI does not expose a manual no-browser/device auth flow. Use browser login with 'claude auth login'."
  }

  $claudePath = Resolve-ClaudePath
  Write-Host "[ai-minion:claude] command: claude auth login"
  if ($WhatIf) {
    Write-Host "[ai-minion:claude] WhatIf enabled. Command not executed."
    return
  }

  & $claudePath auth login
  if ($LASTEXITCODE -ne 0) {
    throw "Claude auth failed with exit code $LASTEXITCODE."
  }
}

function Invoke-ClaudeDelegate {
  param(
    [string]$ResolvedOutputPath,
    [string[]]$ResolvedContextPaths
  )

  if ([string]::IsNullOrWhiteSpace($Task)) {
    throw "Delegate action requires -Task."
  }

  $claudePath = Resolve-ClaudePath
  $sessionId = [guid]::NewGuid().ToString()
  $shouldInject = ($ResolvedContextPaths.Count -gt 0)
  $prompt = New-DelegationPrompt -ProviderName "claude" -CurrentMode $Mode -CurrentTask $Task -ResolvedContextPaths $ResolvedContextPaths -InjectContextBlock:$shouldInject -NoTools
  $promptPath = [System.IO.Path]::ChangeExtension($ResolvedOutputPath, "prompt.txt")
  Write-Utf8NoBom -Path $promptPath -Value $prompt

  $cliArgs = @(
    "-p",
    "--session-id", $sessionId,
    "--permission-mode", "dontAsk",
    "--tools", "",
    "--output-format", $OutputFormat
  )
  if (-not [string]::IsNullOrWhiteSpace($Model)) {
    $cliArgs += @("--model", $Model)
  }

  Write-Host "[ai-minion:claude] session id: $sessionId"
  Write-Host "[ai-minion:claude] output: $ResolvedOutputPath"
  Write-Host "[ai-minion:claude] prompt file: $promptPath"
  if ($shouldInject -and -not $InjectContext) {
    Write-Host "[ai-minion:claude] note: forcing inline context because Claude tools are disabled for delegated runs."
  }

  if ($WhatIf) {
    Write-Host "[ai-minion:claude] WhatIf enabled. Command not executed."
    return
  }

  Push-Location $repoRoot
  try {
    $stdout = $prompt | & $claudePath @cliArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
      throw "Claude delegate failed with exit code $LASTEXITCODE."
    }
  }
  finally {
    Pop-Location
  }

  $assistantEvent = Get-ClaudeAssistantEvent -SessionId $sessionId
  $rawTranscriptPath = Get-ClaudeSessionTranscriptPath -SessionId $sessionId
  if ($rawTranscriptPath) {
    $cacheTranscriptPath = "$ResolvedOutputPath.session.jsonl"
    Copy-Item -Path $rawTranscriptPath -Destination $cacheTranscriptPath -Force
    Write-Host "[ai-minion:claude] transcript: $cacheTranscriptPath"
  }

  if ($OutputFormat -eq "json") {
    if ($assistantEvent) {
      $payload = $assistantEvent | ConvertTo-Json -Depth 20
      Write-Utf8NoBom -Path $ResolvedOutputPath -Value $payload
      Write-Output $payload
      return
    }

    if (-not [string]::IsNullOrWhiteSpace(($stdout | Out-String))) {
      $fallback = ($stdout | Out-String).Trim()
      Write-Utf8NoBom -Path $ResolvedOutputPath -Value $fallback
      Write-Output $fallback
      return
    }

    throw "Claude returned no JSON output and no assistant transcript was found."
  }

  $assistantText = Get-ClaudeAssistantText -AssistantEvent $assistantEvent
  if ([string]::IsNullOrWhiteSpace($assistantText)) {
    $assistantText = ($stdout | Out-String).Trim()
  }
  if ([string]::IsNullOrWhiteSpace($assistantText)) {
    throw "Claude returned no assistant text."
  }

  Write-Utf8NoBom -Path $ResolvedOutputPath -Value $assistantText
  Write-Output $assistantText
}

function Resolve-CodexEntrypoint {
  $npmRoot = Get-GlobalNpmRoot
  $candidate = Join-Path $npmRoot "@openai\codex\bin\codex.js"
  if (-not (Test-Path $candidate)) {
    throw "Codex CLI entrypoint not found at $candidate. Install it with: npm install -g @openai/codex"
  }

  return $candidate
}

function Resolve-CodexPackageJson {
  $npmRoot = Get-GlobalNpmRoot
  $candidate = Join-Path $npmRoot "@openai\codex\package.json"
  if (-not (Test-Path $candidate)) {
    throw "Codex CLI package.json not found at $candidate."
  }

  return $candidate
}

function Get-CodexVersion {
  $packageJsonPath = Resolve-CodexPackageJson
  $packageJson = Get-Content -Raw $packageJsonPath | ConvertFrom-Json
  return [string]$packageJson.version
}

function Get-CodexLoginStatus {
  $nodePath = Resolve-NodePath
  $entrypoint = Resolve-CodexEntrypoint
  $commandLine = ('"{0}" "{1}" login status 2>&1' -f $nodePath, $entrypoint)
  $raw = @(& cmd /d /c $commandLine)
  if ($LASTEXITCODE -ne 0) {
    throw "codex login status failed with exit code $LASTEXITCODE."
  }
  return ($raw | Out-String).Trim()
}

function Invoke-GptDoctor {
  Write-Host "[gpt]"
  Write-Host "  cli entrypoint: $(Resolve-CodexEntrypoint)"
  Write-Host "  cli version: $(Get-CodexVersion)"
  Write-Host "  login status: $(Get-CodexLoginStatus)"
  Write-Host "  config path: $codexConfigPath"
  $sharedContractPath = Get-SharedContractPath
  Write-Host "  shared contract: $(if (Test-Path $sharedContractPath) { $sharedContractPath } else { '<missing>' })"
  $contractPath = Get-ProviderContractPath -ProviderName "gpt"
  Write-Host "  repo contract: $(if (Test-Path $contractPath) { $contractPath } else { '<missing>' })"
}

function Invoke-GptAuth {
  $nodePath = Resolve-NodePath
  $entrypoint = Resolve-CodexEntrypoint
  $cliArgs = @(
    $entrypoint,
    "login"
  )
  if ($NoBrowser) {
    $cliArgs += "--device-auth"
  }

  Write-Host "[ai-minion:gpt] command: node <codex-entrypoint> login$(if ($NoBrowser) { ' --device-auth' } else { '' })"
  if ($WhatIf) {
    Write-Host "[ai-minion:gpt] WhatIf enabled. Command not executed."
    return
  }

  & $nodePath @cliArgs
  if ($LASTEXITCODE -ne 0) {
    throw "GPT auth failed with exit code $LASTEXITCODE."
  }
}

function Get-CodexLastMessageText {
  param([object[]]$Events)

  foreach ($event in ($Events | Sort-Object { $_.Index } -Descending)) {
    if ($event.Payload.type -eq "item.completed" -and $null -ne $event.Payload.item -and $event.Payload.item.type -eq "agent_message") {
      return [string]$event.Payload.item.text
    }
  }

  return $null
}

function Invoke-GptDelegate {
  param(
    [string]$ResolvedOutputPath,
    [string[]]$ResolvedContextPaths
  )

  if ([string]::IsNullOrWhiteSpace($Task)) {
    throw "Delegate action requires -Task."
  }

  $prompt = New-DelegationPrompt -ProviderName "gpt" -CurrentMode $Mode -CurrentTask $Task -ResolvedContextPaths $ResolvedContextPaths -InjectContextBlock:$InjectContext
  $promptPath = [System.IO.Path]::ChangeExtension($ResolvedOutputPath, "prompt.txt")
  Write-Utf8NoBom -Path $promptPath -Value $prompt

  $nodePath = Resolve-NodePath
  $entrypoint = Resolve-CodexEntrypoint
  $cliArgs = @(
    $entrypoint,
    "exec",
    "--json",
    "-C", $repoRoot,
    "-s", "read-only",
    "-"
  )
  if (-not [string]::IsNullOrWhiteSpace($Model)) {
    $cliArgs += @("-m", $Model)
  }

  Write-Host "[ai-minion:gpt] output: $ResolvedOutputPath"
  Write-Host "[ai-minion:gpt] prompt file: $promptPath"

  if ($WhatIf) {
    Write-Host "[ai-minion:gpt] WhatIf enabled. Command not executed."
    return
  }

  Push-Location $repoRoot
  try {
    $rawLines = @(Get-Content -Raw $promptPath | & $nodePath @cliArgs 2>&1 | ForEach-Object { [string]$_ })
    if ($LASTEXITCODE -ne 0) {
      throw "Codex CLI exited with code $LASTEXITCODE."
    }
  }
  finally {
    Pop-Location
  }

  $rawPath = "$ResolvedOutputPath.raw.jsonl"
  Write-Utf8NoBom -Path $rawPath -Value ($rawLines -join [Environment]::NewLine)

  $events = New-Object System.Collections.Generic.List[object]
  for ($i = 0; $i -lt $rawLines.Count; $i++) {
    $line = $rawLines[$i]
    if ([string]::IsNullOrWhiteSpace($line)) {
      continue
    }
    try {
      $events.Add([pscustomobject]@{
        Index = $i
        Payload = ($line | ConvertFrom-Json)
      })
    }
    catch {
    }
  }

  $assistantText = Get-CodexLastMessageText -Events $events
  if ([string]::IsNullOrWhiteSpace($assistantText)) {
    throw "Codex delegate did not emit a final agent message. See $rawPath"
  }

  if ($OutputFormat -eq "json") {
    $payload = [pscustomobject]@{
      provider = "gpt"
      model = if ([string]::IsNullOrWhiteSpace($Model)) { $null } else { $Model }
      text = $assistantText
      rawPath = $rawPath
    } | ConvertTo-Json -Depth 10
    Write-Utf8NoBom -Path $ResolvedOutputPath -Value $payload
    Write-Output $payload
    return
  }

  Write-Utf8NoBom -Path $ResolvedOutputPath -Value $assistantText
  Write-Output $assistantText
}

function Show-Doctor {
  Write-Host "[ai-minion] repo root: $repoRoot"
  Write-Host "[ai-minion] cache root: $cacheRoot"
  switch ($Provider) {
    "all" {
      Invoke-GeminiDoctor
      Invoke-ClaudeDoctor
      Invoke-GptDoctor
    }
    "gemini" { Invoke-GeminiDoctor }
    "claude" { Invoke-ClaudeDoctor }
    "gpt" { Invoke-GptDoctor }
  }
}

switch ($Action) {
  "doctor" {
    Show-Doctor
  }
  "auth" {
    if ($Provider -eq "all") {
      throw "Auth requires -Provider gemini, claude, or gpt."
    }

    switch ($Provider) {
      "gemini" { Invoke-GeminiAuth }
      "claude" { Invoke-ClaudeAuth }
      "gpt" { Invoke-GptAuth }
    }
  }
  "delegate" {
    if ($Provider -eq "all") {
      throw "Delegate requires -Provider gemini, claude, or gpt."
    }
    if ([string]::IsNullOrWhiteSpace($Task)) {
      throw "Delegate action requires -Task."
    }

    $resolvedContextPaths = @(Resolve-ContextPaths)
    $extension = if ($OutputFormat -eq "json") { "json" } else { "txt" }
    $resolvedOutputPath = if ([string]::IsNullOrWhiteSpace($OutputPath)) {
      Get-DefaultArtifactPath -ProviderName $Provider -Extension $extension
    }
    else {
      Resolve-WorkspacePath -Path $OutputPath
    }

    $outputDirectory = Split-Path -Parent $resolvedOutputPath
    if (-not (Test-Path $outputDirectory)) {
      New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
    }

    Write-Host "[ai-minion] provider: $Provider"
    Write-Host "[ai-minion] mode: $Mode"
    Write-Host "[ai-minion] output: $resolvedOutputPath"
    Write-Host "[ai-minion] context paths: $(if (@($resolvedContextPaths).Count -gt 0) { (@($resolvedContextPaths) | ForEach-Object { Get-RepoRelativePath -AbsolutePath $_ }) -join ', ' } else { '<none>' })"

    switch ($Provider) {
      "gemini" { Invoke-GeminiDelegate -ResolvedOutputPath $resolvedOutputPath -ResolvedContextPaths $resolvedContextPaths }
      "claude" { Invoke-ClaudeDelegate -ResolvedOutputPath $resolvedOutputPath -ResolvedContextPaths $resolvedContextPaths }
      "gpt" { Invoke-GptDelegate -ResolvedOutputPath $resolvedOutputPath -ResolvedContextPaths $resolvedContextPaths }
    }
  }
}
