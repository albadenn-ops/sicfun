param(
  [string]$OutDir = "$PSScriptRoot\build",
  [string]$JavaHome = "",
  [string]$CudaRoot = "",
  [string]$VcVars = "",
  [string]$Arch = "auto",
  [string]$Architectures = "",
  [string]$DllName = "sicfun_gpu_kernel.dll",
  [switch]$CheckPrerequisites = $false,
  [switch]$InstallMissingPrerequisites = $false
)

$ErrorActionPreference = "Stop"

$processArchitecture = $env:PROCESSOR_ARCHITECTURE
if ([string]::IsNullOrWhiteSpace($processArchitecture)) {
  $processArchitecture = [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString()
}
if ($processArchitecture.ToLowerInvariant() -notin @("amd64", "x64")) {
  throw "This CUDA build script currently supports Windows x64 only because it resolves vcvars64.bat and produces x64 DLLs. Run it from an x64 PowerShell/JDK environment. Current process architecture: $processArchitecture"
}

function Get-UniqueNonEmptyValues {
  param([string[]]$Values)

  $seen = @{}
  $result = New-Object System.Collections.Generic.List[string]
  foreach ($value in $Values) {
    if ([string]::IsNullOrWhiteSpace($value)) {
      continue
    }
    $trimmed = $value.Trim()
    if (-not $seen.ContainsKey($trimmed)) {
      $seen[$trimmed] = $true
      $result.Add($trimmed)
    }
  }
  return $result
}

$AutoSupportedArchitectures = @(
  "sm_50",
  "sm_52",
  "sm_60",
  "sm_61",
  "sm_70",
  "sm_72",
  "sm_75",
  "sm_80",
  "sm_86",
  "sm_87",
  "sm_89",
  "sm_90"
)

$PreferredJdkPackageId = "Microsoft.OpenJDK.21"
$PreferredCudaPackageId = "Nvidia.CUDA"
$PreferredCudaPackageVersion = "11.8"
$PreferredBuildToolsPackageId = "Microsoft.VisualStudio.2022.BuildTools"
$PreferredBuildToolsInstallOverride = "--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

function Get-CommonJdkInstallRoots {
  $roots = New-Object System.Collections.Generic.List[string]
  foreach ($programFilesRoot in (Get-UniqueNonEmptyValues @($env:ProgramFiles, ${env:ProgramFiles(x86)}))) {
    foreach ($relativePath in @("Java", "Microsoft", "Eclipse Adoptium")) {
      $baseDir = Join-Path $programFilesRoot $relativePath
      if (-not (Test-Path $baseDir)) {
        continue
      }

      $directoryCandidates = Get-ChildItem -Path $baseDir -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
      foreach ($candidate in $directoryCandidates) {
        if (
          $candidate.Name -like "jdk*" -or
          $candidate.Name -like "msopenjdk*" -or
          $candidate.Name -like "jdk-*"
        ) {
          $roots.Add($candidate.FullName)
        }
      }
    }
  }

  return $roots
}

function Resolve-WingetCommand {
  $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
  if (-not $winget) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
  }
  if ($winget -and $winget.Source) {
    return $winget.Source
  }
  return $null
}

function Test-IsAdministrator {
  $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
  $principal = New-Object Security.Principal.WindowsPrincipal($identity)
  return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function New-PrerequisiteEntry {
  param(
    [string]$Name,
    [bool]$Ready,
    [string]$Detail,
    [bool]$Installable = $false,
    [string]$InstallCommand = ""
  )

  return [pscustomobject]@{
    Name = $Name
    Ready = $Ready
    Detail = $Detail
    Installable = $Installable
    InstallCommand = $InstallCommand
  }
}

function Resolve-JavaHome {
  param([string]$Explicit)

  $javac = Get-Command javac.exe -ErrorAction SilentlyContinue
  $java = Get-Command java.exe -ErrorAction SilentlyContinue
  $candidateRoots = @(
    $Explicit,
    $env:SICFUN_GPU_BUILD_JAVA_HOME,
    $env:JAVA_HOME,
    $(if ($javac -and $javac.Source) { Split-Path -Parent (Split-Path -Parent $javac.Source) }),
    $(if ($java -and $java.Source) { Split-Path -Parent (Split-Path -Parent $java.Source) }),
    $(Get-CommonJdkInstallRoots)
  )
  foreach ($candidate in (Get-UniqueNonEmptyValues $candidateRoots)) {
    $jniInclude = Join-Path $candidate "include"
    $jniWinInclude = Join-Path $jniInclude "win32"
    if ((Test-Path $jniInclude) -and (Test-Path $jniWinInclude)) {
      return $candidate
    }
  }
  return $null
}

function Resolve-CudaRoot {
  param([string]$Explicit)

  $cuda11EnvCandidates = Get-ChildItem Env: -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like "CUDA_PATH_V11*" -and -not [string]::IsNullOrWhiteSpace($_.Value) } |
    Sort-Object Name -Descending |
    ForEach-Object { $_.Value }

  $cuda11BaseCandidates = @()
  $otherCudaBaseCandidates = @()
  foreach ($programFilesRoot in (Get-UniqueNonEmptyValues @($env:ProgramFiles, ${env:ProgramFiles(x86)}))) {
    $cudaBase = Join-Path $programFilesRoot "NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path $cudaBase) {
      $versionDirs = Get-ChildItem -Path $cudaBase -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
      $cuda11BaseCandidates += $versionDirs | Where-Object { $_.Name -like "v11*" } | ForEach-Object { $_.FullName }
      $otherCudaBaseCandidates += $versionDirs | Where-Object { $_.Name -notlike "v11*" } | ForEach-Object { $_.FullName }
    }
  }

  $nvcc = Get-Command nvcc.exe -ErrorAction SilentlyContinue
  $candidateRoots = @(
    $Explicit,
    $env:SICFUN_GPU_BUILD_CUDA_ROOT,
    $cuda11EnvCandidates,
    $env:CUDA_PATH_V11_8,
    $cuda11BaseCandidates,
    $env:CUDA_PATH,
    $(if ($nvcc -and $nvcc.Source) { Split-Path -Parent (Split-Path -Parent $nvcc.Source) })
  ) + $otherCudaBaseCandidates

  foreach ($candidate in (Get-UniqueNonEmptyValues $candidateRoots)) {
    $resolvedNvcc = Join-Path $candidate "bin\nvcc.exe"
    if (Test-Path $resolvedNvcc) {
      return $candidate
    }
  }
  return $null
}

function Resolve-VcVars {
  param([string]$Explicit)

  foreach ($candidate in (Get-UniqueNonEmptyValues @($Explicit, $env:SICFUN_GPU_BUILD_VCVARS))) {
    if (Test-Path $candidate) {
      return $candidate
    }
  }

  foreach ($programFilesRoot in (Get-UniqueNonEmptyValues @(${env:ProgramFiles(x86)}, $env:ProgramFiles))) {
    $vswhere = Join-Path $programFilesRoot "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
      $resolved = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Auxiliary\Build\vcvars64.bat" 2>$null |
        Select-Object -First 1
      if (-not [string]::IsNullOrWhiteSpace($resolved) -and (Test-Path $resolved.Trim())) {
        return $resolved.Trim()
      }
    }
  }

  foreach ($programFilesRoot in (Get-UniqueNonEmptyValues @(${env:ProgramFiles(x86)}, $env:ProgramFiles))) {
    $visualStudioRoot = Join-Path $programFilesRoot "Microsoft Visual Studio"
    if (-not (Test-Path $visualStudioRoot)) {
      continue
    }

    $versionDirs = Get-ChildItem -Path $visualStudioRoot -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending
    foreach ($versionDir in $versionDirs) {
      foreach ($edition in @("BuildTools", "Community", "Professional", "Enterprise", "Preview")) {
        $candidate = Join-Path $versionDir.FullName "$edition\VC\Auxiliary\Build\vcvars64.bat"
        if (Test-Path $candidate) {
          return $candidate
        }
      }
    }
  }

  return $null
}

function Normalize-ArchitectureToken {
  param([string]$Raw)

  $token = $Raw.Trim().ToLowerInvariant()
  if ([string]::IsNullOrWhiteSpace($token)) {
    return $null
  }
  if ($token -match '^sm_(\d+)$') {
    return $token
  }
  if ($token -match '^(\d+)\.(\d+)$') {
    return "sm_$($Matches[1])$($Matches[2])"
  }
  if ($token -match '^(\d+)$') {
    return "sm_$($Matches[1])"
  }
  throw "Unsupported CUDA architecture token '$Raw'. Expected values like sm_50 or 8.6."
}

function Resolve-NvidiaSmi {
  $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
  if (-not $nvidiaSmi) {
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  }
  if ($nvidiaSmi -and $nvidiaSmi.Source) {
    return $nvidiaSmi.Source
  }

  foreach ($programFilesRoot in (Get-UniqueNonEmptyValues @($env:ProgramFiles, ${env:ProgramFiles(x86)}))) {
    $candidate = Join-Path $programFilesRoot "NVIDIA Corporation\NVSMI\nvidia-smi.exe"
    if (Test-Path $candidate) {
      return $candidate
    }
  }

  return $null
}

function Get-ArchitectureRank {
  param([string]$Architecture)

  return [int]$Architecture.Substring(3)
}

function Get-CudaToolkitVersionInfo {
  param([string]$ResolvedCudaRoot)

  if ([string]::IsNullOrWhiteSpace($ResolvedCudaRoot)) {
    return $null
  }

  $leaf = Split-Path $ResolvedCudaRoot -Leaf
  if ($leaf -match '^v(\d+)(?:\.(\d+))?') {
    return [pscustomobject]@{
      Major = [int]$Matches[1]
      Minor = if ($Matches[2]) { [int]$Matches[2] } else { 0 }
      Label = $leaf
    }
  }

  $nvcc = Join-Path $ResolvedCudaRoot "bin\nvcc.exe"
  if (Test-Path $nvcc) {
    $versionText = & $nvcc --version 2>$null
    foreach ($line in $versionText) {
      if ($line -match 'release (\d+)\.(\d+)') {
        return [pscustomobject]@{
          Major = [int]$Matches[1]
          Minor = [int]$Matches[2]
          Label = "$($Matches[1]).$($Matches[2])"
        }
      }
    }
  }

  return [pscustomobject]@{
    Major = $null
    Minor = $null
    Label = $leaf
  }
}

function Resolve-Architectures {
  param(
    [string]$ExplicitArchitectures,
    [string]$ExplicitArch
  )

  $rawTokens = @()
  foreach ($raw in @($ExplicitArchitectures, $env:SICFUN_GPU_BUILD_ARCHITECTURES)) {
    if (-not [string]::IsNullOrWhiteSpace($raw)) {
      $rawTokens += ($raw -split ',')
    }
  }
  if ($rawTokens.Count -eq 0 -and -not [string]::IsNullOrWhiteSpace($ExplicitArch) -and $ExplicitArch.Trim().ToLowerInvariant() -ne "auto") {
    $rawTokens += $ExplicitArch
  }
  if ($rawTokens.Count -eq 0 -and -not [string]::IsNullOrWhiteSpace($env:SICFUN_GPU_BUILD_ARCH) -and $env:SICFUN_GPU_BUILD_ARCH.Trim().ToLowerInvariant() -ne "auto") {
    $rawTokens += $env:SICFUN_GPU_BUILD_ARCH
  }

  if ($rawTokens.Count -eq 0) {
    $nvidiaSmi = Resolve-NvidiaSmi
    if ($nvidiaSmi) {
      $computeCaps = & $nvidiaSmi --query-gpu=compute_cap --format=csv,noheader 2>$null
      foreach ($computeCap in $computeCaps) {
        if (-not [string]::IsNullOrWhiteSpace($computeCap) -and $computeCap.Trim().ToUpperInvariant() -ne "N/A") {
          $rawTokens += $computeCap
        }
      }
    }
  }

  $resolved = New-Object System.Collections.Generic.List[string]
  foreach ($token in $rawTokens) {
    $normalized = Normalize-ArchitectureToken $token
    if ($null -ne $normalized -and -not $resolved.Contains($normalized)) {
      $resolved.Add($normalized)
    }
  }

  $usedAutomaticDetection = $rawTokens.Count -gt 0 -and
    [string]::IsNullOrWhiteSpace($ExplicitArchitectures) -and
    ([string]::IsNullOrWhiteSpace($ExplicitArch) -or $ExplicitArch.Trim().ToLowerInvariant() -eq "auto") -and
    [string]::IsNullOrWhiteSpace($env:SICFUN_GPU_BUILD_ARCHITECTURES) -and
    ([string]::IsNullOrWhiteSpace($env:SICFUN_GPU_BUILD_ARCH) -or $env:SICFUN_GPU_BUILD_ARCH.Trim().ToLowerInvariant() -eq "auto")

  if ($usedAutomaticDetection) {
    $supported = New-Object System.Collections.Generic.List[string]
    $unsupported = New-Object System.Collections.Generic.List[string]
    foreach ($architecture in $resolved) {
      if ($AutoSupportedArchitectures -contains $architecture) {
        $supported.Add($architecture)
      }
      else {
        $unsupported.Add($architecture)
      }
    }

    if ($unsupported.Count -gt 0) {
      if ($supported.Count -eq 0) {
        $fallback = $AutoSupportedArchitectures[-1]
        Write-Warning "Auto-detected CUDA architecture(s) '$($unsupported -join ', ')' are newer than this CUDA 11 build script supports. Falling back to $fallback. Override -Arch or -Architectures if you need a different target."
        $supported.Add($fallback)
      }
      else {
        Write-Warning "Ignoring unsupported auto-detected CUDA architecture(s) '$($unsupported -join ', ')' for this CUDA 11 build."
      }
    }

    return $supported
  }

  return $resolved
}

function Get-BuildPrerequisiteState {
  param(
    [string]$ExplicitJavaHome,
    [string]$ExplicitCudaRoot,
    [string]$ExplicitVcVars,
    [string]$ExplicitArchitectures,
    [string]$ExplicitArch
  )

  $resolvedJavaHome = Resolve-JavaHome $ExplicitJavaHome
  $resolvedCudaRoot = Resolve-CudaRoot $ExplicitCudaRoot
  $resolvedVcVars = Resolve-VcVars $ExplicitVcVars
  $resolvedArchitectures = @()
  $architectureError = $null
  try {
    $resolvedArchitectures = @(Resolve-Architectures -ExplicitArchitectures $ExplicitArchitectures -ExplicitArch $ExplicitArch)
  }
  catch {
    $architectureError = $_.Exception.Message
  }

  $cudaVersionInfo = Get-CudaToolkitVersionInfo $resolvedCudaRoot
  $cudaVersionIssue = $null
  if ($resolvedCudaRoot -and $resolvedArchitectures.Count -gt 0 -and ($resolvedArchitectures -contains "sm_50")) {
    if ($cudaVersionInfo -and $cudaVersionInfo.Major -and $cudaVersionInfo.Major -gt 11) {
      $cudaVersionIssue = "detected toolkit '$($cudaVersionInfo.Label)' at '$resolvedCudaRoot', but auto-detected architecture sm_50 requires CUDA 11.8"
    }
  }

  $entries = New-Object System.Collections.Generic.List[object]
  $entries.Add((New-PrerequisiteEntry `
    -Name "host-architecture" `
    -Ready ($processArchitecture.ToLowerInvariant() -in @("amd64", "x64")) `
    -Detail "current process architecture: $processArchitecture")) | Out-Null
  $entries.Add((New-PrerequisiteEntry `
    -Name "jdk" `
    -Ready (-not [string]::IsNullOrWhiteSpace($resolvedJavaHome)) `
    -Detail $(if ($resolvedJavaHome) { "resolved JavaHome: $resolvedJavaHome" } else { "missing JDK with JNI headers (include\\jni.h and include\\win32\\jni_md.h)" }) `
    -Installable $true `
    -InstallCommand "winget install --id $PreferredJdkPackageId --exact --accept-package-agreements --accept-source-agreements --disable-interactivity --silent")) | Out-Null
  $entries.Add((New-PrerequisiteEntry `
    -Name "cuda-toolkit" `
    -Ready ((-not [string]::IsNullOrWhiteSpace($resolvedCudaRoot)) -and [string]::IsNullOrWhiteSpace($cudaVersionIssue)) `
    -Detail $(if ($resolvedCudaRoot -and [string]::IsNullOrWhiteSpace($cudaVersionIssue)) { "resolved CUDA root: $resolvedCudaRoot" } elseif ($cudaVersionIssue) { $cudaVersionIssue } else { "missing CUDA toolkit with nvcc.exe" }) `
    -Installable $true `
    -InstallCommand "winget install --id $PreferredCudaPackageId --exact --version $PreferredCudaPackageVersion --accept-package-agreements --accept-source-agreements --disable-interactivity --silent")) | Out-Null
  $entries.Add((New-PrerequisiteEntry `
    -Name "visual-studio-build-tools" `
    -Ready (-not [string]::IsNullOrWhiteSpace($resolvedVcVars)) `
    -Detail $(if ($resolvedVcVars) { "resolved vcvars64.bat: $resolvedVcVars" } else { "missing Visual Studio Build Tools with C++ workload (vcvars64.bat)" }) `
    -Installable $true `
    -InstallCommand "winget install --id $PreferredBuildToolsPackageId --exact --accept-package-agreements --accept-source-agreements --disable-interactivity --override ""$PreferredBuildToolsInstallOverride""")) | Out-Null
  $entries.Add((New-PrerequisiteEntry `
    -Name "cuda-architecture" `
    -Ready ($resolvedArchitectures.Count -gt 0) `
    -Detail $(if ($resolvedArchitectures.Count -gt 0) { "resolved CUDA architecture: $($resolvedArchitectures -join ', ')" } elseif ($architectureError) { $architectureError } else { "unable to resolve CUDA architecture from -Architectures, -Arch, env overrides, or nvidia-smi" }))) | Out-Null

  return [pscustomobject]@{
    JavaHome = $resolvedJavaHome
    CudaRoot = if ([string]::IsNullOrWhiteSpace($cudaVersionIssue)) { $resolvedCudaRoot } else { $null }
    VcVars = $resolvedVcVars
    Architectures = $resolvedArchitectures
    Entries = $entries
  }
}

function Write-PrerequisiteSummary {
  param([object[]]$Entries)

  foreach ($entry in $Entries) {
    $status = if ($entry.Ready) { "ready" } else { "missing" }
    Write-Host "[prereq] $($entry.Name): $status ($($entry.Detail))"
    if (-not $entry.Ready -and $entry.InstallCommand) {
      Write-Host "[prereq] install $($entry.Name): $($entry.InstallCommand)"
    }
  }
}

function Install-BuildPrerequisite {
  param(
    [string]$WingetCommand,
    [object]$Entry
  )

  switch ($Entry.Name) {
    "jdk" {
      $arguments = @(
        "install",
        "--id", $PreferredJdkPackageId,
        "--exact",
        "--accept-package-agreements",
        "--accept-source-agreements",
        "--disable-interactivity",
        "--silent"
      )
    }
    "cuda-toolkit" {
      $arguments = @(
        "install",
        "--id", $PreferredCudaPackageId,
        "--exact",
        "--version", $PreferredCudaPackageVersion,
        "--accept-package-agreements",
        "--accept-source-agreements",
        "--disable-interactivity",
        "--silent"
      )
    }
    "visual-studio-build-tools" {
      $arguments = @(
        "install",
        "--id", $PreferredBuildToolsPackageId,
        "--exact",
        "--accept-package-agreements",
        "--accept-source-agreements",
        "--disable-interactivity",
        "--override", $PreferredBuildToolsInstallOverride
      )
    }
    default {
      throw "No auto-install plan is defined for prerequisite '$($Entry.Name)'."
    }
  }

  Write-Host "[prereq] installing $($Entry.Name) via winget..."
  & $WingetCommand @arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Auto-install failed for prerequisite '$($Entry.Name)' with exit code $LASTEXITCODE. Command: $($Entry.InstallCommand)"
  }
}

function Ensure-BuildPrerequisites {
  param(
    [string]$ExplicitJavaHome,
    [string]$ExplicitCudaRoot,
    [string]$ExplicitVcVars,
    [string]$ExplicitArchitectures,
    [string]$ExplicitArch,
    [bool]$ShouldInstall
  )

  $state = Get-BuildPrerequisiteState `
    -ExplicitJavaHome $ExplicitJavaHome `
    -ExplicitCudaRoot $ExplicitCudaRoot `
    -ExplicitVcVars $ExplicitVcVars `
    -ExplicitArchitectures $ExplicitArchitectures `
    -ExplicitArch $ExplicitArch
  Write-PrerequisiteSummary $state.Entries

  $missing = @($state.Entries | Where-Object { -not $_.Ready })
  if ($missing.Count -eq 0) {
    return $state
  }

  if ($ShouldInstall) {
    $wingetCommand = Resolve-WingetCommand
    if ([string]::IsNullOrWhiteSpace($wingetCommand)) {
      throw "Auto-install requested, but winget is not available on PATH."
    }
    if (-not (Test-IsAdministrator)) {
      throw "Auto-install requested, but this PowerShell session is not running as Administrator. Re-run elevated or install prerequisites manually."
    }

    foreach ($entry in $missing | Where-Object { $_.Installable }) {
      Install-BuildPrerequisite -WingetCommand $wingetCommand -Entry $entry
    }

    $state = Get-BuildPrerequisiteState `
      -ExplicitJavaHome $ExplicitJavaHome `
      -ExplicitCudaRoot $ExplicitCudaRoot `
      -ExplicitVcVars $ExplicitVcVars `
      -ExplicitArchitectures $ExplicitArchitectures `
      -ExplicitArch $ExplicitArch
    Write-PrerequisiteSummary $state.Entries
    $missing = @($state.Entries | Where-Object { -not $_.Ready })
  }

  if ($missing.Count -gt 0) {
    $details = $missing | ForEach-Object {
      if ($_.InstallCommand) {
        "$($_.Name): $($_.Detail). Install with: $($_.InstallCommand)"
      }
      else {
        "$($_.Name): $($_.Detail)"
      }
    }
    $actionHint = if ($ShouldInstall) {
      "Auto-install could not satisfy every prerequisite."
    }
    else {
      "Re-run with -InstallMissingPrerequisites to let this script install supported missing prerequisites automatically."
    }
    throw "CUDA build prerequisites are not ready. $actionHint Missing prerequisite(s): $($details -join ' | ')"
  }

  return $state
}

$prerequisiteState = Ensure-BuildPrerequisites `
  -ExplicitJavaHome $JavaHome `
  -ExplicitCudaRoot $CudaRoot `
  -ExplicitVcVars $VcVars `
  -ExplicitArchitectures $Architectures `
  -ExplicitArch $Arch `
  -ShouldInstall:$InstallMissingPrerequisites

$JavaHome = $prerequisiteState.JavaHome
$CudaRoot = $prerequisiteState.CudaRoot
$VcVars = $prerequisiteState.VcVars
$resolvedArchitectures = @($prerequisiteState.Architectures)

if ($CheckPrerequisites) {
  Write-Host "CUDA build prerequisites are ready."
  return
}

$nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
if (-not (Test-Path $nvcc)) {
  throw "nvcc not found at $nvcc"
}
if (-not (Test-Path $VcVars)) {
  throw "vcvars64.bat not found at $VcVars"
}

$jniInclude = Join-Path $JavaHome "include"
$jniWinInclude = Join-Path $jniInclude "win32"
if (-not (Test-Path $jniInclude)) {
  throw "JNI include path not found: $jniInclude"
}
if (-not (Test-Path $jniWinInclude)) {
  throw "JNI Windows include path not found: $jniWinInclude"
}

Write-Host "Resolved JavaHome: $JavaHome"
Write-Host "Resolved CudaRoot: $CudaRoot"
Write-Host "Resolved VcVars: $VcVars"
Write-Host "Resolved CUDA architectures: $($resolvedArchitectures -join ', ')"
Write-Host "Resolved process architecture: $processArchitecture"

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$src = Join-Path $PSScriptRoot "jni\HeadsUpGpuNativeBindingsCuda.cu"
$dll = Join-Path $OutDir $DllName
$gencodeArgs = @()
foreach ($resolvedArch in $resolvedArchitectures) {
  $computeArch = $resolvedArch.Substring(3)
  $gencodeArgs += @("-gencode", "arch=compute_$computeArch,code=$resolvedArch")
}
$highestArch = $resolvedArchitectures | Sort-Object { Get-ArchitectureRank $_ } | Select-Object -Last 1
if ($highestArch) {
  $highestComputeArch = $highestArch.Substring(3)
  $gencodeArgs += @("-gencode", "arch=compute_$highestComputeArch,code=compute_$highestComputeArch")
}

$nvccArgs = @(
  "-allow-unsupported-compiler"
  "-std=c++17"
  "-O3"
  "-Xcompiler", "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
  "-Xcompiler", "/EHsc"
  "-shared"
  "-I$jniInclude"
  "-I$jniWinInclude"
  "-o", $dll
  $src
)
$nvccArgs = $nvccArgs[0..6] + $gencodeArgs + $nvccArgs[7..($nvccArgs.Length - 1)]

$escapedVcVars = $VcVars.Replace('"', '""')
$escapedNvcc = $nvcc.Replace('"', '""')
$escapedArgs = ($nvccArgs | ForEach-Object {
  if ($_ -match '\s') { '"' + $_.Replace('"', '""') + '"' } else { $_ }
}) -join " "
$cmdLine = "call ""$escapedVcVars"" && ""$escapedNvcc"" $escapedArgs"

cmd.exe /c $cmdLine
if ($LASTEXITCODE -ne 0) {
  throw "CUDA native build failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path $dll)) {
  throw "Build did not produce $dll"
}

Write-Host "Built: $dll"

$cfrSrc = Join-Path $PSScriptRoot "jni\HoldemCfrNativeGpuBindings.cu"
$cfrDll = Join-Path $OutDir "sicfun_cfr_cuda.dll"
$cfrArgs = @(
  "-allow-unsupported-compiler"
  "-std=c++17"
  "-O3"
  "-Xcompiler", "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
  "-Xcompiler", "/EHsc"
  "-shared"
  "-I$jniInclude"
  "-I$jniWinInclude"
  "-o", $cfrDll
  $cfrSrc
)
$cfrArgs = $cfrArgs[0..6] + $gencodeArgs + $cfrArgs[7..($cfrArgs.Length - 1)]

$escapedCfrArgs = ($cfrArgs | ForEach-Object {
  if ($_ -match '\s') { '"' + $_.Replace('"', '""') + '"' } else { $_ }
}) -join " "
$cfrCmdLine = "call ""$escapedVcVars"" && ""$escapedNvcc"" $escapedCfrArgs"

cmd.exe /c $cfrCmdLine
if ($LASTEXITCODE -ne 0) {
  throw "CUDA CFR native build failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path $cfrDll)) {
  throw "Build did not produce $cfrDll"
}

Write-Host "Built: $cfrDll"

$bayesSrc = Join-Path $PSScriptRoot "jni\HoldemBayesNativeGpuBindings.cu"
$bayesDll = Join-Path $OutDir "sicfun_bayes_cuda.dll"
$bayesArgs = @(
  "-allow-unsupported-compiler"
  "-std=c++17"
  "-O3"
  "-Xcompiler", "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
  "-Xcompiler", "/EHsc"
  "-shared"
  "-I$jniInclude"
  "-I$jniWinInclude"
  "-o", $bayesDll
  $bayesSrc
)
$bayesArgs = $bayesArgs[0..6] + $gencodeArgs + $bayesArgs[7..($bayesArgs.Length - 1)]

$escapedBayesArgs = ($bayesArgs | ForEach-Object {
  if ($_ -match '\s') { '"' + $_.Replace('"', '""') + '"' } else { $_ }
}) -join " "
$bayesCmdLine = "call ""$escapedVcVars"" && ""$escapedNvcc"" $escapedBayesArgs"

cmd.exe /c $bayesCmdLine
if ($LASTEXITCODE -ne 0) {
  throw "CUDA Bayesian native build failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path $bayesDll)) {
  throw "Build did not produce $bayesDll"
}

Write-Host "Built: $bayesDll"

$ddreSrc = Join-Path $PSScriptRoot "jni\HoldemDdreNativeGpuBindings.cu"
$ddreDll = Join-Path $OutDir "sicfun_ddre_cuda.dll"
$ddreArgs = @(
  "-allow-unsupported-compiler"
  "-std=c++17"
  "-O3"
  "-Xcompiler", "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
  "-Xcompiler", "/EHsc"
  "-shared"
  "-I$jniInclude"
  "-I$jniWinInclude"
  "-o", $ddreDll
  $ddreSrc
)
$ddreArgs = $ddreArgs[0..6] + $gencodeArgs + $ddreArgs[7..($ddreArgs.Length - 1)]

$escapedDdreArgs = ($ddreArgs | ForEach-Object {
  if ($_ -match '\s') { '"' + $_.Replace('"', '""') + '"' } else { $_ }
}) -join " "
$ddreCmdLine = "call ""$escapedVcVars"" && ""$escapedNvcc"" $escapedDdreArgs"

cmd.exe /c $ddreCmdLine
if ($LASTEXITCODE -ne 0) {
  throw "CUDA DDRE native build failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path $ddreDll)) {
  throw "Build did not produce $ddreDll"
}

Write-Host "Built: $ddreDll"

$postflopSrc = Join-Path $PSScriptRoot "jni\HoldemPostflopNativeBindingsCuda.cu"
$postflopDll = Join-Path $OutDir "sicfun_postflop_cuda.dll"
$postflopArgs = @(
  "-allow-unsupported-compiler"
  "-std=c++17"
  "-O3"
  "-Xcompiler", "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
  "-Xcompiler", "/EHsc"
  "-shared"
  "-I$jniInclude"
  "-I$jniWinInclude"
  "-o", $postflopDll
  $postflopSrc
)
$postflopArgs = $postflopArgs[0..6] + $gencodeArgs + $postflopArgs[7..($postflopArgs.Length - 1)]

$escapedPostflopArgs = ($postflopArgs | ForEach-Object {
  if ($_ -match '\s') { '"' + $_.Replace('"', '""') + '"' } else { $_ }
}) -join " "
$postflopCmdLine = "call ""$escapedVcVars"" && ""$escapedNvcc"" $escapedPostflopArgs"

cmd.exe /c $postflopCmdLine
if ($LASTEXITCODE -ne 0) {
  throw "CUDA postflop native build failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path $postflopDll)) {
  throw "Build did not produce $postflopDll"
}

Write-Host "Built: $postflopDll"
