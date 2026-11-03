@echo off
setlocal
cd /d "%~dp0"
set "B4_PYTHON=C:\Users\17264\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%B4_PYTHON%" (
    "%B4_PYTHON%" "%~dp0b4_autostitch.py"
) else (
    python "%~dp0b4_autostitch.py"
)
if errorlevel 1 pause
