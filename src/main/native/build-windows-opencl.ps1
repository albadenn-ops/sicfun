param(
  [string]$OutDir = "$PSScriptRoot\build",
  [string]$JavaHome = $env:JAVA_HOME,
  [string]$OpenCLInclude = "$PSScriptRoot\opencl-headers",
  [string]$OpenCLLib = "",
  [string]$VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
  [string]$DllName = "sicfun_opencl_kernel.dll"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($JavaHome)) {
  throw "JAVA_HOME is not set"
}

$jniInclude = Join-Path $JavaHome "include"
$jniWinInclude = Join-Path $jniInclude "win32"
if (-not (Test-Path $jniInclude)) {
  throw "JNI include path not found: $jniInclude"
}
if (-not (Test-Path $jniWinInclude)) {
  throw "JNI Windows include path not found: $jniWinInclude"
}
if (-not (Test-Path $VcVars)) {
  throw "vcvars64.bat not found at $VcVars"
}
if (-not (Test-Path $OpenCLInclude)) {
  throw "OpenCL headers not found at $OpenCLInclude. Vendor Khronos OpenCL-Headers into $OpenCLInclude"
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

# Generate embedded kernel source as a C string literal
$clKernel = Join-Path $PSScriptRoot "jni\HeadsUpOpenCLKernel.cl"
$embedInc = Join-Path $PSScriptRoot "jni\HeadsUpOpenCLKernel_embed.inc"
if (-not (Test-Path $clKernel)) {
  throw "OpenCL kernel source not found: $clKernel"
}
Write-Host "Embedding kernel source: $clKernel -> $embedInc"
$lines = Get-Content $clKernel -Raw
# C++ raw string literal R"sicfun_CL(...)sicfun_CL" — content is taken verbatim, no escaping needed
Set-Content -Path $embedInc -Value "R""sicfun_CL($lines)sicfun_CL""" -NoNewline
Write-Host "Generated: $embedInc"

$src = Join-Path $PSScriptRoot "jni\HeadsUpOpenCLBindings.cpp"
$dll = Join-Path $OutDir $DllName

# No OpenCL.lib needed: the DLL dynamically loads OpenCL.dll at runtime
# via LoadLibraryA/GetProcAddress, so no link-time dependency exists.

$clArgs = @(
  "/std:c++17"
  "/O2"
  "/EHsc"
  "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
  "/LD"
  "/I""$jniInclude"""
  "/I""$jniWinInclude"""
  "/I""$OpenCLInclude"""
  "/Fe""$dll"""
  """$src"""
)

$escapedVcVars = $VcVars.Replace('"', '""')
$escapedArgs = ($clArgs | ForEach-Object { $_ }) -join " "
$cmdLine = "call ""$escapedVcVars"" && cl.exe $escapedArgs"

cmd.exe /c $cmdLine
if ($LASTEXITCODE -ne 0) {
  throw "OpenCL native build failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path $dll)) {
  throw "Build did not produce $dll"
}

Write-Host "Built: $dll"
