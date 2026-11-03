@echo off
setlocal
cd /d "%~dp0"
set "A4_PYTHON=C:\Users\17264\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%A4_PYTHON%" (
    "%A4_PYTHON%" "%~dp0a4_mosaics.py"
) else (
    python "%~dp0a4_mosaics.py"
)
if errorlevel 1 pause
