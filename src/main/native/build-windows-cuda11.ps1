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
