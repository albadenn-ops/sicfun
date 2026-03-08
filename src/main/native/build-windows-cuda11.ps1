param(
  [string]$OutDir = "$PSScriptRoot\build",
  [string]$JavaHome = $env:JAVA_HOME,
  [string]$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
  [string]$VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
  [string]$Arch = "sm_50",
  [string]$DllName = "sicfun_gpu_kernel.dll"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($JavaHome)) {
  throw "JAVA_HOME is not set"
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

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$src = Join-Path $PSScriptRoot "jni\HeadsUpGpuNativeBindingsCuda.cu"
$dll = Join-Path $OutDir $DllName
$gencode = "arch=compute_$($Arch.Substring(3)),code=$Arch"

$nvccArgs = @(
  "-allow-unsupported-compiler"
  "-std=c++17"
  "-O3"
  "-Xcompiler", "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
  "-Xcompiler", "/EHsc"
  "-shared"
  "-gencode", $gencode
  "-I$jniInclude"
  "-I$jniWinInclude"
  "-o", $dll
  $src
)

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
  "-gencode", $gencode
  "-I$jniInclude"
  "-I$jniWinInclude"
  "-o", $cfrDll
  $cfrSrc
)

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
  "-gencode", $gencode
  "-I$jniInclude"
  "-I$jniWinInclude"
  "-o", $bayesDll
  $bayesSrc
)

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
  "-gencode", $gencode
  "-I$jniInclude"
  "-I$jniWinInclude"
  "-o", $ddreDll
  $ddreSrc
)

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
  "-gencode", $gencode
  "-I$jniInclude"
  "-I$jniWinInclude"
  "-o", $postflopDll
  $postflopSrc
)

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
