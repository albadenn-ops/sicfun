$ErrorActionPreference = "Continue"
$CudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
$VcVars = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
$src = Join-Path $PSScriptRoot "jni\cuda_throughput_bench.cu"
$outDir = Join-Path $PSScriptRoot "build"

$architectures = @("sm_50", "sm_75", "sm_80", "sm_86", "sm_89", "sm_90")

foreach ($arch in $architectures) {
  $gencode = "arch=compute_$($arch.Substring(3)),code=$arch"
  $outFile = Join-Path $outDir "cuda_bench_$arch.exe"

  $nvccArgs = @(
    "-allow-unsupported-compiler"
    "-std=c++17"
    "-O3"
    "-Xcompiler", "/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
    "-Xcompiler", "/EHsc"
    "-gencode", $gencode
    "-o", $outFile
    $src
  )

  $escapedVcVars = $VcVars.Replace('"', '""')
  $escapedNvcc = $nvcc.Replace('"', '""')
  $escapedArgs = ($nvccArgs | ForEach-Object {
    if ($_ -match '\s') { '"' + $_.Replace('"', '""') + '"' } else { $_ }
  }) -join " "
  $cmdLine = "call ""$escapedVcVars"" && ""$escapedNvcc"" $escapedArgs"

  cmd.exe /c $cmdLine 2>$null | Out-Null

  if (Test-Path $outFile) {
    $size = (Get-Item $outFile).Length / 1KB
    Write-Host ("OK   {0}  ({1:N0} KB)" -f $arch, $size)
  } else {
    Write-Host "FAIL $arch"
  }
}
