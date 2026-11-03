@echo off
setlocal
cd /d "%~dp0"
set "PREVIEW_PYTHON=C:\Users\17264\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%PREVIEW_PYTHON%" (
    "%PREVIEW_PYTHON%" "%~dp0preview_site.py"
) else (
    python "%~dp0preview_site.py"
)
if errorlevel 1 pause
