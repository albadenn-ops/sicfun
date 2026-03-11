@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0start-g5-acpc.ps1" %*
endlocal
