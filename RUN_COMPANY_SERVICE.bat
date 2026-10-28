@echo off
setlocal
cd /d "%~dp0"

if not defined CARDREADER_WORKERS set "CARDREADER_WORKERS=2"
if not defined CARDREADER_HOST set "CARDREADER_HOST=0.0.0.0"
if not defined CARDREADER_PORT set "CARDREADER_PORT=8510"

set "PY=%LOCALAPPDATA%\CardReaderRuntime\venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================================
echo AiNationalCard - v26 GOLDEN OCR / multi-process API
echo OCR core: UNCHANGED
echo Workers : %CARDREADER_WORKERS%
echo Address : %CARDREADER_HOST%:%CARDREADER_PORT%
echo ============================================================

"%PY%" -m uvicorn company_api:app ^
  --host %CARDREADER_HOST% ^
  --port %CARDREADER_PORT% ^
  --workers %CARDREADER_WORKERS%
