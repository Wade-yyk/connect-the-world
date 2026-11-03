@echo off
setlocal
cd /d "%~dp0"
set "A3_PYTHON=C:\Users\17264\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%A3_PYTHON%" (
    "%A3_PYTHON%" "%~dp0a3_rectify.py"
) else (
    python "%~dp0a3_rectify.py"
)
if errorlevel 1 pause
