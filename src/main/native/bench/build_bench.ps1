
$ErrorActionPreference = "Stop"
$JdkRoot = "C:\Program Files\Java\jdk-22"
$VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$src = Join-Path $PSScriptRoot "native_baseline_bench.cpp"
$out = Join-Path $PSScriptRoot "native_baseline_bench.exe"

$clArgs = @(
  "/std:c++17"
  "/O2"
  "/EHsc"
  "/I", "$JdkRoot\include"
  "/I", "$JdkRoot\include\win32"
  "/Fe:$out"
  $src
  "User32.lib"
)

$escapedVcVars = $VcVars.Replace('"', '""')
$clCmd = "cl " + (($clArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join " ")
$cmdLine = "call ""$escapedVcVars"" && $clCmd"

cmd.exe /c $cmdLine
if ($LASTEXITCODE -ne 0) {
  throw "Build failed with exit code $LASTEXITCODE"
}
Write-Host "Built: $out"
