[CmdletBinding()]
param(
  [ValidateSet("menu", "quick-proof", "full-proof", "hall-max-autotune", "hall-max-gpu", "hall-single")]
  [string]$Action = "menu",
  [switch]$WhatIf
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function New-RunbookAction {
  param(
    [string]$Key,
    [string]$Label,
    [string]$CommandText,
    [scriptblock]$Command,
    [bool]$Heavy = $false
  )

  return [pscustomobject]@{
    Key = $Key
    Label = $Label
    CommandText = $CommandText
    Command = $Command
    Heavy = $Heavy
  }
}

$actions = [ordered]@{}
$actions["quick-proof"] = New-RunbookAction `
  -Key "quick-proof" `
  -Label "Quick proof pipeline" `
  -CommandText "powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1 -Quick" `
  -Command { & powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1 -Quick }

$actions["full-proof"] = New-RunbookAction `
  -Key "full-proof" `
  -Label "Full proof pipeline" `
  -CommandText "powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1" `
  -Command { & powershell -ExecutionPolicy Bypass -File scripts/prove-pipeline.ps1 }

$actions["hall-max-autotune"] = New-RunbookAction `
  -Key "hall-max-autotune" `
  -Label "Hall max run with autotune (recommended)" `
  -CommandText "powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall-max.ps1 -AutoTune -AutoTuneHands 500000 -AutoTuneProfiles auto,cpu,gpu -AutoTuneWorkerCandidates 8,12,16,20,24 -Hands 100000000 -TableCountPerWorker 8 -ReportEvery 500000 -LearnEveryHands 0 -SaveTrainingTsv false -SaveDdreTrainingTsv false -OutDir data/bench-hall-max" `
  -Command {
    & powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall-max.ps1 `
      -AutoTune `
      -AutoTuneHands 500000 `
      -AutoTuneProfiles auto,cpu,gpu `
      -AutoTuneWorkerCandidates 8,12,16,20,24 `
      -Hands 100000000 `
      -TableCountPerWorker 8 `
      -ReportEvery 500000 `
      -LearnEveryHands 0 `
      -SaveTrainingTsv false `
      -SaveDdreTrainingTsv false `
      -OutDir data/bench-hall-max
  } `
  -Heavy $true

$actions["hall-max-gpu"] = New-RunbookAction `
  -Key "hall-max-gpu" `
  -Label "Hall max run with fixed GPU profile" `
  -CommandText "powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall-max.ps1 -Hands 100000000 -Workers 0 -TableCountPerWorker 8 -NativeProfile gpu -ReportEvery 500000 -LearnEveryHands 0 -SaveTrainingTsv false -SaveDdreTrainingTsv false -JvmOption \"-Xms2g\" \"-Xmx2g\" -OutDir data/bench-hall-max" `
  -Command {
    & powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall-max.ps1 `
      -Hands 100000000 `
      -Workers 0 `
      -TableCountPerWorker 8 `
      -NativeProfile gpu `
      -ReportEvery 500000 `
      -LearnEveryHands 0 `
      -SaveTrainingTsv false `
      -SaveDdreTrainingTsv false `
      -JvmOption "-Xms2g" "-Xmx2g" `
      -OutDir data/bench-hall-max
  } `
  -Heavy $true

$actions["hall-single"] = New-RunbookAction `
  -Key "hall-single" `
  -Label "Single-process hall run" `
  -CommandText "powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall.ps1 -Hands 1000000 -TableCount 8 -ReportEvery 50000 -LearnEveryHands 0 -SaveTrainingTsv false -SaveDdreTrainingTsv false -OutDir data/bench-hall-single" `
  -Command {
    & powershell -ExecutionPolicy Bypass -File scripts/run-playing-hall.ps1 `
      -Hands 1000000 `
      -TableCount 8 `
      -ReportEvery 50000 `
      -LearnEveryHands 0 `
      -SaveTrainingTsv false `
      -SaveDdreTrainingTsv false `
      -OutDir data/bench-hall-single
  } `
  -Heavy $true

function Invoke-RunbookAction {
  param(
    [string]$Key
  )

  if (-not $actions.Contains($Key)) {
    throw "Unknown action '$Key'."
  }

  $entry = $actions[$Key]
  Write-Host "[runbook] action: $($entry.Label)"
  Write-Host "[runbook] command: $($entry.CommandText)"

  if ($entry.Heavy -and -not $WhatIf) {
    $confirm = Read-Host "This action may run for a long time and consume high hardware resources. Type YES to continue"
    if ($confirm -cne "YES") {
      Write-Host "[runbook] cancelled."
      return
    }
  }

  if ($WhatIf) {
    Write-Host "[runbook] WhatIf enabled. Command not executed."
    return
  }

  & $entry.Command
  $exitCode = $LASTEXITCODE
  if ($null -ne $exitCode -and $exitCode -ne 0) {
    throw "Action '$Key' exited with code $exitCode."
  }
}

$menuOrder = @(
  "quick-proof",
  "full-proof",
  "hall-max-autotune",
  "hall-max-gpu",
  "hall-single"
)

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $repoRoot
try {
  if ($Action -ne "menu") {
    Invoke-RunbookAction -Key $Action
    return
  }

  while ($true) {
    Write-Host ""
    Write-Host "SICFUN Runbook Menu"
    for ($i = 0; $i -lt $menuOrder.Count; $i++) {
      $key = $menuOrder[$i]
      $entry = $actions[$key]
      $heavyTag = if ($entry.Heavy) { " [HEAVY]" } else { "" }
      Write-Host ("[{0}] {1}{2}" -f ($i + 1), $entry.Label, $heavyTag)
    }
    Write-Host "[0] Exit"
    Write-Host "You can also type an action key (example: quick-proof)."

    $rawChoice = Read-Host "Select option"
    if ($null -eq $rawChoice) {
      continue
    }
    $choice = $rawChoice.Trim()
    if ([string]::IsNullOrWhiteSpace($choice)) {
      continue
    }
    if ($choice -eq "0" -or $choice.Equals("exit", [System.StringComparison]::OrdinalIgnoreCase)) {
      break
    }

    $numericChoice = 0
    if ([int]::TryParse($choice, [ref]$numericChoice)) {
      if ($numericChoice -ge 1 -and $numericChoice -le $menuOrder.Count) {
        $selectedKey = $menuOrder[$numericChoice - 1]
        Invoke-RunbookAction -Key $selectedKey
        continue
      }
      Write-Host "[runbook] invalid numeric choice: $choice"
      continue
    }

    if ($actions.Contains($choice)) {
      Invoke-RunbookAction -Key $choice
      continue
    }

    Write-Host "[runbook] unknown option: $choice"
  }
}
finally {
  Pop-Location
}
