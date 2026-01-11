@echo off
:: ============================================================
:: GNN Digital Twin City Simulation - Quick Start
:: ============================================================
:: Double-click this file to run the project
:: ============================================================

title Digital Twin City Simulation

cd /d "%~dp0"

echo.
echo ============================================================
echo    GNN Digital Twin City Simulation
echo    AI-Powered Traffic Prediction System
echo ============================================================
echo.

:: Check if virtual environment exists
if not exist "twin-city-env\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo.
    echo Please run the following command first:
    echo    powershell -ExecutionPolicy Bypass -File run_project.ps1 -Install
    echo.
    pause
    exit /b 1
)

:: Activate virtual environment and run server
echo Starting server...
echo.
echo Frontend:   http://localhost:5000
echo API Status: http://localhost:5000/api/status
echo.
echo Press CTRL+C to stop the server
echo ============================================================
echo.

call twin-city-env\Scripts\activate.bat
python backend\app.py

pause
