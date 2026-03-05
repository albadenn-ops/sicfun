[CmdletBinding()]
param(
  [string]$BenchmarkDir = "data/benchmarks",
  [string]$RoadmapPath = "ROADMAP.md",
  [string]$OutputPath = "dist/parent-dashboard/index.html"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-FullPath {
  param(
    [string]$Path,
    [string]$RepoRoot,
    [bool]$MustExist = $false
  )

  $candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
    $Path
  } else {
    Join-Path $RepoRoot $Path
  }

  if ($MustExist) {
    return (Resolve-Path $candidate).Path
  }

  return [System.IO.Path]::GetFullPath($candidate)
}

function Get-LatestFile {
  param(
    [string]$DirectoryPath,
    [string]$Filter
  )

  $latest = Get-ChildItem -Path $DirectoryPath -File -Filter $Filter |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

  return $latest
}

function Parse-DoubleInvariant {
  param([string]$Value)
  if ([string]::IsNullOrWhiteSpace($Value)) {
    return $null
  }
  return [double]::Parse($Value, [System.Globalization.CultureInfo]::InvariantCulture)
}

function Html {
  param([string]$Value)
  if ($null -eq $Value) { return "" }
  return [System.Net.WebUtility]::HtmlEncode($Value)
}

function Parse-Roadmap {
  param([string]$Path)

  $lines = Get-Content -Path $Path
  $sections = @()
  $current = $null

  foreach ($line in $lines) {
    if ($line -match "^##\s+(M\d+.+)$") {
      if ($null -ne $current) {
        $sections += [pscustomobject]$current
      }

      $current = [ordered]@{
        Name  = $Matches[1].Trim()
        Done  = 0
        Total = 0
      }
      continue
    }

    if ($null -ne $current -and $line -match "^- \[(x|X| )\]\s+") {
      $current.Total += 1
      if ($Matches[1] -match "[xX]") {
        $current.Done += 1
      }
    }
  }

  if ($null -ne $current) {
    $sections += [pscustomobject]$current
  }

  $done = 0
  $total = 0
  foreach ($section in $sections) {
    $done += $section.Done
    $total += $section.Total
  }

  [pscustomobject]@{
    Sections = $sections
    Done     = $done
    Total    = $total
  }
}

function Parse-DeviceProof {
  param([string]$Path)

  $lines = Get-Content -Path $Path
  $devices = @()
  $runs = @()
  $configLine = ""
  $completedLine = ""

  foreach ($line in $lines) {
    if ($line -match "^config:\s+(.+)$") {
      $configLine = $Matches[1].Trim()
      continue
    }

    if ($line -match "^\s*(cuda:\d+|opencl:\d+|cpu)\s+kind=([^\s]+)\s+name=(.+)$") {
      $devices += [pscustomobject]@{
        Slot = $Matches[1]
        Kind = $Matches[2]
        Name = $Matches[3].Trim()
      }
      continue
    }

    if ($line -match "^(cuda\.run|opencl\.run|opencl\.direct\[\d+\]\.run):\s+deviceIndex=(\d+)\s+status=(-?\d+)\s+elapsedMs=([0-9,\.]+)\s+lastEngineCode=(\d+)\s+checksum=([0-9a-fA-F]+)$") {
      $runs += [pscustomobject]@{
        Name       = $Matches[1]
        DeviceIdx  = $Matches[2]
        Status     = $Matches[3]
        ElapsedMs  = $Matches[4]
        EngineCode = $Matches[5]
        Checksum   = $Matches[6]
      }
      continue
    }

    if ($line -match "^\[success\].*completed\s+(.+)$") {
      $completedLine = $Matches[1].Trim()
    }
  }

  [pscustomobject]@{
    Config    = $configLine
    Completed = $completedLine
    Devices   = $devices
    Runs      = $runs
  }
}

function Format-StatesRate {
  param([object]$Value)
  if ($null -eq $Value) { return "n/a" }
  $rounded = [math]::Round([double]$Value)
  return $rounded.ToString("N0", [System.Globalization.CultureInfo]::InvariantCulture)
}

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$benchmarkDirPath = Get-FullPath -Path $BenchmarkDir -RepoRoot $repoRoot -MustExist $true
$roadmapPathResolved = Get-FullPath -Path $RoadmapPath -RepoRoot $repoRoot -MustExist $true
$outputPathResolved = Get-FullPath -Path $OutputPath -RepoRoot $repoRoot
$outputDir = Split-Path -Parent $outputPathResolved

if (-not (Test-Path $benchmarkDirPath)) {
  throw "Benchmark directory not found: $benchmarkDirPath"
}

$summaryFile = Get-LatestFile -DirectoryPath $benchmarkDirPath -Filter "mode-comparison-*-summary.csv"
$rawFile = Get-LatestFile -DirectoryPath $benchmarkDirPath -Filter "mode-comparison-*-raw.csv"
$chartFile = Get-LatestFile -DirectoryPath $benchmarkDirPath -Filter "mode-comparison-*-states-boxplot.svg"
$deviceProofFile = Get-LatestFile -DirectoryPath $benchmarkDirPath -Filter "device-proof-*.txt"

if ($null -eq $summaryFile) {
  throw "No summary CSV found in $benchmarkDirPath"
}
if ($null -eq $deviceProofFile) {
  throw "No device proof file found in $benchmarkDirPath"
}

$summaryRows = Import-Csv -Path $summaryFile.FullName | ForEach-Object {
  [pscustomobject]@{
    Mode   = $_.mode.Trim('"')
    Count  = [int]$_.count
    Mean   = Parse-DoubleInvariant $_.mean_states_per_s
    Median = Parse-DoubleInvariant $_.median_states_per_s
    Min    = Parse-DoubleInvariant $_.min_states_per_s
    Max    = Parse-DoubleInvariant $_.max_states_per_s
    Q1     = Parse-DoubleInvariant $_.q1_states_per_s
    Q3     = Parse-DoubleInvariant $_.q3_states_per_s
    StdDev = Parse-DoubleInvariant $_.stddev_states_per_s
  }
}

if ($summaryRows.Count -eq 0) {
  throw "Summary CSV is empty: $($summaryFile.FullName)"
}

$summarySorted = $summaryRows | Sort-Object Mean -Descending
$fastest = $summarySorted | Select-Object -First 1
$cpuRow = $summaryRows | Where-Object { $_.Mode -eq "cpu-only" } | Select-Object -First 1
$openclRow = $summaryRows | Where-Object { $_.Mode -eq "opencl-only" } | Select-Object -First 1
$cudaRow = $summaryRows | Where-Object { $_.Mode -eq "cuda-only" } | Select-Object -First 1
$hybridRow = $summaryRows | Where-Object { $_.Mode -eq "hybrid" } | Select-Object -First 1

$speedup = $null
if ($null -ne $cpuRow -and $cpuRow.Mean -gt 0 -and $null -ne $fastest) {
  $speedup = $fastest.Mean / $cpuRow.Mean
}

$roadmap = Parse-Roadmap -Path $roadmapPathResolved
$proof = Parse-DeviceProof -Path $deviceProofFile.FullName
$progressPercent = if ($roadmap.Total -gt 0) { [math]::Round((100.0 * $roadmap.Done) / $roadmap.Total, 1) } else { 0.0 }

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$chartTargetName = ""
if ($null -ne $chartFile) {
  $chartTargetName = "states-boxplot.svg"
  Copy-Item -Path $chartFile.FullName -Destination (Join-Path $outputDir $chartTargetName) -Force
}

$generatedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$summaryStamp = $summaryFile.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
$proofStamp = $deviceProofFile.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
$rawFileName = if ($null -ne $rawFile) { $rawFile.Name } else { "n/a" }

$summaryRowsHtml = ($summarySorted | ForEach-Object {
  $isTop = if ($_.Mode -eq $fastest.Mode) { " class=`"top-row`"" } else { "" }
  "<tr$isTop><td>$(Html $_.Mode)</td><td>$($_.Count)</td><td>$(Format-StatesRate $_.Mean)</td><td>$(Format-StatesRate $_.Median)</td><td>$(Format-StatesRate $_.Min)</td><td>$(Format-StatesRate $_.Max)</td><td>$(Format-StatesRate $_.StdDev)</td></tr>"
}) -join "`n"

$milestoneRowsHtml = ($roadmap.Sections | ForEach-Object {
  $pct = if ($_.Total -gt 0) { [math]::Round((100.0 * $_.Done) / $_.Total, 1) } else { 0.0 }
  "<tr><td>$(Html $_.Name)</td><td>$($_.Done) / $($_.Total)</td><td><div class=`"mini-bar`"><span style=`"width:$pct%`"></span></div></td><td>$pct%</td></tr>"
}) -join "`n"

$deviceRowsHtml = ($proof.Devices | ForEach-Object {
  "<tr><td>$(Html $_.Slot)</td><td>$(Html $_.Kind)</td><td>$(Html $_.Name)</td></tr>"
}) -join "`n"

$runRowsHtml = ($proof.Runs | ForEach-Object {
  $statusClass = if ($_.Status -eq "0") { "ok" } else { "warn" }
  "<tr><td>$(Html $_.Name)</td><td>$(Html $_.DeviceIdx)</td><td class=`"$statusClass`">$(Html $_.Status)</td><td>$(Html $_.ElapsedMs)</td><td>$(Html $_.EngineCode)</td><td><code>$(Html $_.Checksum)</code></td></tr>"
}) -join "`n"

$openclMean = if ($null -ne $openclRow) { Format-StatesRate $openclRow.Mean } else { "n/a" }
$cudaMean = if ($null -ne $cudaRow) { Format-StatesRate $cudaRow.Mean } else { "n/a" }
$hybridMean = if ($null -ne $hybridRow) { Format-StatesRate $hybridRow.Mean } else { "n/a" }
$cpuMean = if ($null -ne $cpuRow) { Format-StatesRate $cpuRow.Mean } else { "n/a" }
$speedupText = if ($null -ne $speedup) {
  ("{0}x" -f ([double]$speedup).ToString("0.00", [System.Globalization.CultureInfo]::InvariantCulture))
} else {
  "n/a"
}

$chartHtml = ""
if (-not [string]::IsNullOrWhiteSpace($chartTargetName)) {
  $chartHtml = @"
<section class="panel">
  <h2>Distribution Chart</h2>
  <p class="muted">Copied from latest benchmark artifact.</p>
  <img class="chart" src="$chartTargetName" alt="States per second boxplot" />
</section>
"@
}

$html = @"
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sicfun Progress Dashboard</title>
  <style>
    :root {
      --bg: #f4efe3;
      --paper: #fffdf8;
      --ink: #1f2933;
      --muted: #5f6c7b;
      --accent: #0f766e;
      --accent-soft: #d7f0ed;
      --warn: #b45309;
      --ok: #166534;
      --line: #d6d0c2;
      --header: #f3e5c4;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Segoe UI", Tahoma, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 8%, #ffffff 0%, rgba(255, 255, 255, 0) 28%),
        linear-gradient(165deg, #f8f4ea 0%, #efe4cc 100%);
      min-height: 100vh;
    }

    .wrap {
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }

    .hero {
      background: linear-gradient(135deg, #0f766e 0%, #115e59 100%);
      color: #f8fafc;
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 16px 35px rgba(17, 24, 39, 0.25);
      position: relative;
      overflow: hidden;
    }

    .hero::after {
      content: "";
      position: absolute;
      right: -42px;
      top: -42px;
      width: 180px;
      height: 180px;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.16);
    }

    h1 {
      margin: 0 0 8px 0;
      font-size: clamp(1.5rem, 2.5vw, 2.4rem);
      letter-spacing: 0.02em;
    }

    p {
      margin: 0;
    }

    .meta {
      margin-top: 8px;
      color: #dbeafe;
      font-size: 0.95rem;
    }

    .grid {
      display: grid;
      gap: 16px;
      margin-top: 16px;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    }

    .card {
      background: var(--paper);
      border-radius: 14px;
      border: 1px solid var(--line);
      padding: 16px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
    }

    .label {
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .value {
      font-size: clamp(1.2rem, 2vw, 2rem);
      font-weight: 700;
    }

    .accent {
      color: var(--accent);
    }

    .panel {
      margin-top: 16px;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }

    .panel h2 {
      margin: 0 0 8px 0;
      font-size: 1.15rem;
    }

    .muted {
      color: var(--muted);
      font-size: 0.95rem;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 0.93rem;
    }

    th, td {
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }

    th {
      background: var(--header);
      color: #3f3f46;
      font-weight: 600;
    }

    .top-row td {
      background: #ecfdf5;
      font-weight: 600;
    }

    .mini-bar {
      width: 100%;
      height: 9px;
      background: #e5e7eb;
      border-radius: 999px;
      overflow: hidden;
      margin-top: 4px;
    }

    .mini-bar span {
      display: block;
      height: 100%;
      background: linear-gradient(90deg, #14b8a6, #0f766e);
    }

    .ok {
      color: var(--ok);
      font-weight: 700;
    }

    .warn {
      color: var(--warn);
      font-weight: 700;
    }

    .modes {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }

    .mode-chip {
      background: var(--accent-soft);
      border: 1px solid #b8e2dc;
      border-radius: 12px;
      padding: 10px;
    }

    .mode-chip strong {
      display: block;
      margin-bottom: 4px;
    }

    .chart {
      width: 100%;
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      padding: 6px;
    }

    .footer {
      margin-top: 16px;
      color: var(--muted);
      font-size: 0.9rem;
    }

    @media (max-width: 640px) {
      .wrap { padding: 12px; }
      .hero { padding: 16px; }
      th, td { padding: 7px 6px; font-size: 0.85rem; }
    }
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <h1>Sicfun Progress Dashboard</h1>
      <p>Evidence-backed progress and performance summary for easy non-technical review.</p>
      <p class="meta">Generated: $generatedAt | Summary snapshot: $summaryStamp | Device proof snapshot: $proofStamp</p>
    </section>

    <section class="grid">
      <article class="card">
        <p class="label">Roadmap Completion</p>
        <p class="value accent">$($roadmap.Done) / $($roadmap.Total) tasks</p>
        <p class="muted">$progressPercent% complete</p>
      </article>
      <article class="card">
        <p class="label">Fastest Mode</p>
        <p class="value">$(Html $fastest.Mode)</p>
        <p class="muted">$(Format-StatesRate $fastest.Mean) states/s (mean)</p>
      </article>
      <article class="card">
        <p class="label">Fastest vs CPU</p>
        <p class="value">$speedupText</p>
        <p class="muted">Based on latest mode comparison summary</p>
      </article>
      <article class="card">
        <p class="label">Proof Run Completed</p>
        <p class="value">$(Html $proof.Completed)</p>
        <p class="muted">$(Html $proof.Config)</p>
      </article>
    </section>

    <section class="panel">
      <h2>Current Throughput Snapshot (states/s mean)</h2>
      <div class="modes">
        <div class="mode-chip"><strong>CPU only</strong>$cpuMean</div>
        <div class="mode-chip"><strong>CUDA only</strong>$cudaMean</div>
        <div class="mode-chip"><strong>OpenCL only</strong>$openclMean</div>
        <div class="mode-chip"><strong>Hybrid</strong>$hybridMean</div>
      </div>
    </section>

    <section class="panel">
      <h2>Mode Comparison Detail</h2>
      <p class="muted">Rows sorted by mean throughput (higher is better).</p>
      <table>
        <thead>
          <tr>
            <th>Mode</th>
            <th>Runs</th>
            <th>Mean</th>
            <th>Median</th>
            <th>Min</th>
            <th>Max</th>
            <th>StdDev</th>
          </tr>
        </thead>
        <tbody>
$summaryRowsHtml
        </tbody>
      </table>
    </section>

$chartHtml

    <section class="panel">
      <h2>Roadmap Milestones</h2>
      <table>
        <thead>
          <tr>
            <th>Milestone</th>
            <th>Done</th>
            <th>Progress</th>
            <th>Percent</th>
          </tr>
        </thead>
        <tbody>
$milestoneRowsHtml
        </tbody>
      </table>
    </section>

    <section class="panel">
      <h2>Device Evidence</h2>
      <table>
        <thead>
          <tr>
            <th>Slot</th>
            <th>Kind</th>
            <th>Name</th>
          </tr>
        </thead>
        <tbody>
$deviceRowsHtml
        </tbody>
      </table>

      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>DeviceIdx</th>
            <th>Status</th>
            <th>ElapsedMs</th>
            <th>EngineCode</th>
            <th>Checksum</th>
          </tr>
        </thead>
        <tbody>
$runRowsHtml
        </tbody>
      </table>
    </section>

    <section class="panel">
      <h2>Evidence Files</h2>
      <p class="muted">Summary CSV: $(Html $summaryFile.Name)</p>
      <p class="muted">Raw CSV: $(Html $rawFileName)</p>
      <p class="muted">Device proof: $(Html $deviceProofFile.Name)</p>
      <p class="muted">Roadmap: $(Html (Split-Path -Leaf $roadmapPathResolved))</p>
    </section>

    <p class="footer">Generated by scripts/parent-dashboard.ps1</p>
  </main>
</body>
</html>
"@

Set-Content -Path $outputPathResolved -Value $html -Encoding utf8

Write-Host "Dashboard generated:"
Write-Host "  $outputPathResolved"
