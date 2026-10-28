@echo off
setlocal
cd /d "C:\Users\ext.h.bahmanighajar\Desktop\AiNationalCard_COMPANY_CLEAN"

set "PY=%LOCALAPPDATA%\CardReaderRuntime\venv\Scripts\python.exe"

echo ============================================================
echo ROLLBACK SAFE MULTI-CARD FIX
echo ============================================================

if not exist "%PY%" (
  echo ERROR: Runtime Python not found:
  echo %PY%
  pause
  exit /b 1
)

"%PY%" -c "import base64; exec(base64.b64decode('CmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAppbXBvcnQgc2h1dGlsLCBzdWJwcm9jZXNzLCBzeXMKClJPT1QgPSBQYXRoKHIiQzpcVXNlcnNcZXh0LmguYmFobWFuaWdoYWphclxEZXNrdG9wXEFpTmF0aW9uYWxDYXJkX0NPTVBBTllfQ0xFQU4iKQpiYWNrdXBfcm9vdCA9IFJPT1QgLyAiX3BhdGNoX2JhY2t1cHMiIC8gIm11bHRpY2FyZF9zYWZlIgpsYXRlc3RfZmlsZSA9IGJhY2t1cF9yb290IC8gIkxBVEVTVC50eHQiCgppZiBub3QgbGF0ZXN0X2ZpbGUuZXhpc3RzKCk6CiAgICByYWlzZSBTeXN0ZW1FeGl0KCJFUlJPUjogTm8gbXVsdGljYXJkX3NhZmUgYmFja3VwIG1hcmtlciBmb3VuZC4iKQoKc3RhbXAgPSBsYXRlc3RfZmlsZS5yZWFkX3RleHQoZW5jb2Rpbmc9InV0Zi04Iikuc3RyaXAoKQpiYWNrdXBfZGlyID0gYmFja3VwX3Jvb3QgLyBzdGFtcAoKc3JjX2RldGVjdG9yID0gYmFja3VwX2RpciAvICJkZXRlY3Rvci5weSIKc3JjX21vZHVsZSA9IGJhY2t1cF9kaXIgLyAiY2FyZHJlYWRlcl9tb2R1bGUucHkiCmRzdF9kZXRlY3RvciA9IFJPT1QgLyAiZGV0ZWN0b3IucHkiCmRzdF9tb2R1bGUgPSBST09UIC8gImNhcmRyZWFkZXJfbW9kdWxlLnB5IgoKaWYgbm90IHNyY19kZXRlY3Rvci5leGlzdHMoKSBvciBub3Qgc3JjX21vZHVsZS5leGlzdHMoKToKICAgIHJhaXNlIFN5c3RlbUV4aXQoIkVSUk9SOiBCYWNrdXAgZmlsZXMgYXJlIG1pc3NpbmcuIikKCnNodXRpbC5jb3B5MihzcmNfZGV0ZWN0b3IsIGRzdF9kZXRlY3RvcikKc2h1dGlsLmNvcHkyKHNyY19tb2R1bGUsIGRzdF9tb2R1bGUpCgpzdWJwcm9jZXNzLnJ1bigKICAgIFtzeXMuZXhlY3V0YWJsZSwgIi1tIiwgInB5X2NvbXBpbGUiLCBzdHIoZHN0X2RldGVjdG9yKSwgc3RyKGRzdF9tb2R1bGUpXSwKICAgIGNoZWNrPVRydWUKKQoKcHJpbnQoIlJPTExCQUNLIENPTVBMRVRFIikKcHJpbnQoIlJlc3RvcmVkIGJhY2t1cDoiLCBiYWNrdXBfZGlyKQpwcmludCgiZGV0ZWN0b3IucHkgYW5kIGNhcmRyZWFkZXJfbW9kdWxlLnB5IGFyZSBiYWNrIHRvIHRoZSBleGFjdCBwcmUtcGF0Y2ggc3RhdGUuIikK').decode('utf-8'))"
set ERR=%ERRORLEVEL%

echo.
if not "%ERR%"=="0" (
  echo FAILED
  pause
  exit /b %ERR%
)

pause
exit /b 0
