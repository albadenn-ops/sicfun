@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0start-sicfun-acpc.ps1" %*
endlocal
