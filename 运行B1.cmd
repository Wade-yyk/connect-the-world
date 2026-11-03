@echo off
setlocal
cd /d "%~dp0"
set "B1_PYTHON=C:\Users\17264\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%B1_PYTHON%" (
    "%B1_PYTHON%" "%~dp0b1_detect_corners.py"
) else (
    python "%~dp0b1_detect_corners.py"
)
if errorlevel 1 pause
