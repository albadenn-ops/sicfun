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

$cfrSrc = Join-Path $PSScriptRoot "jni\HoldemCfrNativeCpuBindings.cpp"
$cfrDll = Join-Path $OutDir "sicfun_cfr_native.dll"
$cfrLib = Join-Path $OutDir "sicfun_cfr_native.lib"
$cfrExp = Join-Path $OutDir "sicfun_cfr_native.exp"

if (Test-Path $cfrDll) { Remove-Item $cfrDll -Force }
if (Test-Path $cfrLib) { Remove-Item $cfrLib -Force }
if (Test-Path $cfrExp) { Remove-Item $cfrExp -Force }

& $clang `
  -std=c++17 `
  -O3 `
  -DNDEBUG `
  -D_CRT_SECURE_NO_WARNINGS `
  -shared `
  "-I$jniInclude" `
  "-I$jniWinInclude" `
  -o $cfrDll `
  $cfrSrc

if ($LASTEXITCODE -ne 0) {
  throw "Native CFR CPU build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $cfrDll)) {
  throw "Build did not produce $cfrDll"
}

Write-Host "Built: $cfrDll"

$bayesSrc = Join-Path $PSScriptRoot "jni\HoldemBayesNativeCpuBindings.cpp"
$bayesDll = Join-Path $OutDir "sicfun_bayes_native.dll"
$bayesLib = Join-Path $OutDir "sicfun_bayes_native.lib"
$bayesExp = Join-Path $OutDir "sicfun_bayes_native.exp"

if (Test-Path $bayesDll) { Remove-Item $bayesDll -Force }
if (Test-Path $bayesLib) { Remove-Item $bayesLib -Force }
if (Test-Path $bayesExp) { Remove-Item $bayesExp -Force }

& $clang `
  -std=c++17 `
  -O3 `
  -DNDEBUG `
  -D_CRT_SECURE_NO_WARNINGS `
  -shared `
  "-I$jniInclude" `
  "-I$jniWinInclude" `
  -o $bayesDll `
  $bayesSrc

if ($LASTEXITCODE -ne 0) {
  throw "Native Bayesian CPU build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $bayesDll)) {
  throw "Build did not produce $bayesDll"
}

Write-Host "Built: $bayesDll"
