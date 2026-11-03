@echo off
setlocal
cd /d "%~dp0"
set "B2_PYTHON=C:\Users\17264\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%B2_PYTHON%" (
    "%B2_PYTHON%" "%~dp0b2_extract_descriptors.py"
) else (
    python "%~dp0b2_extract_descriptors.py"
)
if errorlevel 1 pause
