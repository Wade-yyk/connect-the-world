@echo off
setlocal
cd /d "%~dp0"
set "B3_PYTHON=C:\Users\17264\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%B3_PYTHON%" (
    "%B3_PYTHON%" "%~dp0b3_match_features.py"
) else (
    python "%~dp0b3_match_features.py"
)
if errorlevel 1 pause
