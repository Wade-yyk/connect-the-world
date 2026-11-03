@echo off
setlocal
cd /d "%~dp0"
set "POINT_PICKER_PYTHON=C:\Users\17264\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%POINT_PICKER_PYTHON%" (
    "%POINT_PICKER_PYTHON%" "%~dp0mark_points.py"
) else (
    python "%~dp0mark_points.py"
)
if errorlevel 1 pause
