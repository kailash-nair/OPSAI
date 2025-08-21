@echo off
cd /d "%~dp0"
echo Starting AI Agent (silent mode)...

:: Run server silently
start "" /min cmd /c "cd server && npm run dev >nul 2>&1"

:: Run web silently
start "" /min cmd /c "cd web && npm run dev >nul 2>&1"

:: Launch the Node.js launcher silently (it opens browser)
start "" /min cmd /c "node start.js >nul 2>&1"

exit