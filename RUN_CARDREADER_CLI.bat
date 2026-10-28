@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set "CR_VENV=%LOCALAPPDATA%\CardReaderRuntime\venv"
if exist "%CR_VENV%\Scripts\python.exe" (
  "%CR_VENV%\Scripts\python.exe" "%~dp0cardreader_cli.py"
) else (
  py -3 "%~dp0cardreader_cli.py"
)
echo.
pause
