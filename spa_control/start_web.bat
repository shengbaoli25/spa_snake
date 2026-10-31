@echo off
cd /d "%~dp0.."
if not exist "spa_control\.venv\Scripts\python.exe" (
  echo Python environment not found. See spa_control\README_web.md.
  pause
  exit /b 1
)
echo SPA workbench: http://127.0.0.1:8765
echo Leave this window open while using the website.
"spa_control\.venv\Scripts\python.exe" -m spa_control.web_server --port 8765
pause
