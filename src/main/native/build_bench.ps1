$ErrorActionPreference = "Stop"
$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
$src = Join-Path $PSScriptRoot "jni\cuda_throughput_bench.cu"
$out = Join-Path $PSScriptRoot "build\cuda_bench.exe"

$nvccArgs = @(
  "-allow-unsupported-compiler"
  "-std=c++17"
  "-O3"
  "-Xcompiler", "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
  "-Xcompiler", "/EHsc"
  "-gencode", "arch=compute_50,code=sm_50"
  "-o", $out
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
  throw "Build failed with exit code $LASTEXITCODE"
}
Write-Host "Built: $out"
