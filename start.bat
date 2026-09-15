@echo off
REM Double-click this to run quorum.ai. Keep the window open while the
REM meeting runs — closing it stops the agent.
setlocal
cd /d "%~dp0"
title quorum.ai

echo.
echo   quorum.ai
echo   ---------
echo.

REM --- python present? -----------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
  echo   Python is not on your PATH.
  echo   Reinstall from python.org and tick "Add python.exe to PATH".
  echo.
  pause
  exit /b 1
)

REM --- already running? ----------------------------------------------------
REM Starting a second copy fails with a confusing port error, so check first.
netstat -ano | findstr /r /c:"TCP.*:8000 .*LISTENING" >nul 2>&1
if not errorlevel 1 (
  echo   Something is already listening on port 8000.
  echo   Opening the dashboard instead of starting a second copy.
  start "" http://127.0.0.1:8000
  echo.
  timeout /t 3 >nul
  exit /b 0
)

REM --- dependencies --------------------------------------------------------
python -c "import quorum.server" >nul 2>&1
if errorlevel 1 (
  echo   Installing dependencies. First run only, this takes a few minutes.
  echo.
  python -m pip install -e ".[dev,audio,voice]"
  if errorlevel 1 (
    echo.
    echo   Install failed. Scroll up for the error.
    pause
    exit /b 1
  )
)

REM --- warn about the things that silently break a meeting -----------------
tasklist /fi "imagename eq voicemeeter*" 2>nul | find /i "voicemeeter" >nul
if errorlevel 1 (
  echo   NOTE: VoiceMeeter does not appear to be running. The virtual audio
  echo         devices carry no sound until it is open.
  echo.
)

if not exist ".env" (
  echo   NOTE: No .env file yet. Add your Groq API key under Settings once
  echo         the dashboard opens.
  echo.
)

echo   Starting. The dashboard will open in your browser.
echo   Close this window to stop the agent.
echo.

REM Give uvicorn a moment to bind before the browser asks for the page.
start "" /b cmd /c "timeout /t 4 >nul & start "" http://127.0.0.1:8000"

python -m uvicorn quorum.server:app --host 127.0.0.1 --port 8000

echo.
echo   Server stopped.
pause
