@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\nvcc.exe" -allow-unsupported-compiler -std=c++17 -O3 -Xcompiler /D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -Xcompiler /EHsc -gencode arch=compute_50,code=sm_50 -o "%~dp0build\cuda_bench.exe" "%~dp0jni\cuda_throughput_bench.cu"
if %ERRORLEVEL% NEQ 0 (
    echo BUILD_FAILED with code %ERRORLEVEL%
    exit /b 1
)
echo BUILD_OK
