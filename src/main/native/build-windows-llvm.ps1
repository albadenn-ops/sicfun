param(
  [string]$OutDir = "$PSScriptRoot\build",
  [string]$JavaHome = $env:JAVA_HOME
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($JavaHome)) {
  throw "JAVA_HOME is not set"
}

$clang = "C:\Program Files\LLVM\bin\clang++.exe"
if (-not (Test-Path $clang)) {
  throw "clang++ not found at $clang"
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

$src = Join-Path $PSScriptRoot "jni\HeadsUpGpuNativeBindings.cpp"
$dll = Join-Path $OutDir "sicfun_native_cpu.dll"
$lib = Join-Path $OutDir "sicfun_native_cpu.lib"
$exp = Join-Path $OutDir "sicfun_native_cpu.exp"

if (Test-Path $dll) { Remove-Item $dll -Force }
if (Test-Path $lib) { Remove-Item $lib -Force }
if (Test-Path $exp) { Remove-Item $exp -Force }

& $clang `
  -std=c++17 `
  -O3 `
  -DNDEBUG `
  -D_CRT_SECURE_NO_WARNINGS `
  -shared `
  "-I$jniInclude" `
  "-I$jniWinInclude" `
  -o $dll `
  $src

if ($LASTEXITCODE -ne 0) {
  throw "Native build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $dll)) {
  throw "Build did not produce $dll"
}

Write-Host "Built: $dll"
