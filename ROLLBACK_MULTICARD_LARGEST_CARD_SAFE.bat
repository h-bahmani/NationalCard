@echo off
setlocal
cd /d "C:\Users\ext.h.bahmanighajar\Desktop\AiNationalCard_COMPANY_CLEAN"
set "PY=%LOCALAPPDATA%\CardReaderRuntime\venv\Scripts\python.exe"

echo ============================================================
echo ROLLBACK MULTI-CARD LARGEST-CARD PATCH
echo ============================================================

if not exist "%PY%" (
  echo ERROR: Runtime Python not found.
  pause
  exit /b 1
)

"%PY%" -c "import base64; exec(base64.b64decode('CmZyb20gcGF0aGxpYiBpbXBvcnQgUGF0aAppbXBvcnQgc2h1dGlsCmltcG9ydCBzdWJwcm9jZXNzCmltcG9ydCBzeXMKCnJvb3QgPSBQYXRoKHIiQzpcVXNlcnNcZXh0LmguYmFobWFuaWdoYWphclxEZXNrdG9wXEFpTmF0aW9uYWxDYXJkX0NPTVBBTllfQ0xFQU4iKQpkZXRlY3RvciA9IHJvb3QgLyAiZGV0ZWN0b3IucHkiCmJhY2t1cCA9IHJvb3QgLyAiZGV0ZWN0b3IucHkuYmVmb3JlX2xhcmdlc3RfY2FyZF9wYXRjaC5iYWsiCgppZiBub3QgYmFja3VwLmV4aXN0cygpOgogICAgcmFpc2UgU3lzdGVtRXhpdCgiRVJST1I6IHJvbGxiYWNrIGJhY2t1cCBub3QgZm91bmQiKQoKc2h1dGlsLmNvcHkyKGJhY2t1cCwgZGV0ZWN0b3IpCnN1YnByb2Nlc3MucnVuKFtzeXMuZXhlY3V0YWJsZSwgIi1tIiwgInB5X2NvbXBpbGUiLCBzdHIoZGV0ZWN0b3IpXSwgY2hlY2s9VHJ1ZSkKcHJpbnQoIlJPTExCQUNLIENPTVBMRVRFIikKcHJpbnQoIk9yaWdpbmFsIGRldGVjdG9yLnB5IHJlc3RvcmVkLiIpCg==').decode('utf-8'))"
set ERR=%ERRORLEVEL%

echo.
if not "%ERR%"=="0" (
  echo FAILED
  pause
  exit /b %ERR%
)

pause
exit /b 0
