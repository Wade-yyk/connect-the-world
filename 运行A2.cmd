@echo off
setlocal
cd /d "%~dp0"
set "A2_PYTHON=C:\Users\17264\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%A2_PYTHON%" (
    "%A2_PYTHON%" "%~dp0a2_recover_homographies.py"
) else (
    python "%~dp0a2_recover_homographies.py"
)
if errorlevel 1 (
    pause
    exit /b 1
)
start "" "%~dp0output\a2\report.html"
