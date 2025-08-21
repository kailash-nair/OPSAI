@echo off
cd /d "%~dp0"
echo Starting AI Agent (debug mode)...

:: Open server in its own console and keep it open
start "SERVER" cmd /k "cd server && npm run dev"

:: Open web in its own console and keep it open
start "WEB" cmd /k "cd web && npm run dev"

:: Run the Node launcher in this window so you see its output
node start.js

echo.
echo Launcher finished. Press any key to close this window.
pause >nul
