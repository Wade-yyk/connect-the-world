@echo off
setlocal
cd /d "%~dp0"
set "A3_PICKER_PYTHON=C:\Users\17264\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%A3_PICKER_PYTHON%" (
    "%A3_PICKER_PYTHON%" "%~dp0mark_rectification.py"
) else (
    python "%~dp0mark_rectification.py"
)
if errorlevel 1 pause
