@echo off
echo ========================================
echo Starting GNN Traffic Simulation Server
echo ========================================
echo.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python app.py
pause



