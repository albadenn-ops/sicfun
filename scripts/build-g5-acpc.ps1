param(
  [string]$RepoDir = "data/tmp/g5-poker-bot",
  [string]$RuntimeDir = "data/tmp/g5-runtime-hu",
  [string]$DotnetDir = "data/tmp/toolchains/dotnet",
  [string]$DotnetChannel = "8.0"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$repoDirAbs = Join-Path $repoRoot $RepoDir
$runtimeDirAbs = Join-Path $repoRoot $RuntimeDir
$dotnetDirAbs = Join-Path $repoRoot $DotnetDir
$dotnetInstallScript = Join-Path (Split-Path $dotnetDirAbs -Parent) "dotnet-install.ps1"

if (-not (Test-Path $repoDirAbs)) {
  throw "G5 repo not found at $repoDirAbs. This script expects a local developer checkout under data/tmp/, which is not source-controlled."
}

if (-not (Test-Path $dotnetInstallScript)) {
  New-Item -ItemType Directory -Force (Split-Path $dotnetInstallScript -Parent) | Out-Null
  Invoke-WebRequest https://dot.net/v1/dotnet-install.ps1 -OutFile $dotnetInstallScript
}

if (-not (Test-Path (Join-Path $dotnetDirAbs "dotnet.exe"))) {
  & powershell -ExecutionPolicy Bypass -File $dotnetInstallScript -Channel $DotnetChannel -InstallDir $dotnetDirAbs -NoPath
}

$dotnet = Join-Path $dotnetDirAbs "dotnet.exe"
$acpcMain = Join-Path $repoDirAbs "src/G5.Acpc/AcpcMain.cs"
$acpcMainText = Get-Content $acpcMain -Raw
$updatedAcpcMain = $acpcMainText -replace "using \(var game = new AcpcGame\(TableType\.SixMax\)\)", "using (var game = new AcpcGame(TableType.HeadsUp))"
if ($updatedAcpcMain -ne $acpcMainText) {
  Set-Content -Path $acpcMain -Value $updatedAcpcMain -NoNewline
}

$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$msbuild = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\MSBuild\Current\Bin\amd64\MSBuild.exe"
if (-not (Test-Path $vcvars)) {
  throw "vcvars64.bat not found at $vcvars"
}
if (-not (Test-Path $msbuild)) {
  throw "MSBuild.exe not found at $msbuild"
}

$decisionProject = Join-Path $repoDirAbs "src/DecisionMaking/DecisionMaking.vcxproj"
$solutionDir = (Join-Path $repoDirAbs "src") + "\\"
$nativeBuildCmd = 'call "' + $vcvars + '" >nul && "' + $msbuild + '" "' + $decisionProject + '" /m /p:Configuration=Release /p:Platform=x64 /p:PlatformToolset=v145 /p:SolutionDir=' + $solutionDir + ' /verbosity:minimal'
cmd /c $nativeBuildCmd
if ($LASTEXITCODE -ne 0) {
  throw "DecisionMaking.dll build failed with exit code $LASTEXITCODE"
}

$g5Project = Join-Path $repoDirAbs "src/G5.Acpc/G5.Acpc.csproj"
$publishDir = Join-Path $repoDirAbs "src/bin/Release/G5.Acpc"
& $dotnet restore $g5Project --source https://api.nuget.org/v3/index.json
if ($LASTEXITCODE -ne 0) {
  throw "dotnet restore failed with exit code $LASTEXITCODE"
}
& $dotnet publish $g5Project -c Release -o $publishDir --no-restore
if ($LASTEXITCODE -ne 0) {
  throw "dotnet publish failed with exit code $LASTEXITCODE"
}

Remove-Item -Recurse -Force $runtimeDirAbs -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $runtimeDirAbs | Out-Null

Copy-Item (Join-Path $publishDir "*") $runtimeDirAbs -Force
Copy-Item (Join-Path $repoDirAbs "src/bin/Release/DecisionMaking.dll") $runtimeDirAbs -Force
Copy-Item (Join-Path $repoDirAbs "redist/tbb.dll") $runtimeDirAbs -Force
Copy-Item (Join-Path $repoDirAbs "redist/full_stats_list_hu.bin") $runtimeDirAbs -Force
Copy-Item (Join-Path $repoDirAbs "redist/PreFlopEquities.txt") $runtimeDirAbs -Force
Copy-Item (Join-Path $repoDirAbs "redist/PreFlopCharts") $runtimeDirAbs -Recurse -Force

Write-Output "G5 ACPC runtime ready:"
Write-Output "  dotnet:  $dotnet"
Write-Output "  runtime: $runtimeDirAbs"
Write-Output "  note:    G5 source, generated runtime, and local toolchains under data/tmp/ are developer-local assets."
Write-Output "  start:   powershell -ExecutionPolicy Bypass -File scripts/start-g5-acpc.ps1 127.0.0.1 12345"
