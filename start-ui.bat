@echo off
REM GNN City Simulator - Modern UI Startup Script
REM Run this script to start both backend and frontend servers

echo.
echo ========================================
echo   GNN City Simulator - Modern UI
echo ========================================
echo.

REM Change to project directory
cd /d "c:\Users\Akash\Desktop\EDI\Edi proj\GNN---DIGITAL_CITY_SIMULATION"

if errorlevel 1 (
    echo Error: Could not navigate to project directory
    pause
    exit /b 1
)

echo [1/4] Checking virtual environment...
if not exist ".venv" (
    echo Error: Virtual environment not found
    echo Please run: python -m venv .venv
    pause
    exit /b 1
)
echo ✓ Virtual environment found

echo.
echo [2/4] Starting Backend Server (Port 5000)...
echo Opening new window...
start "GNN Backend - Flask" cmd /k ".\.venv\Scripts\activate.bat && cd backend && echo Backend starting on port 5000... && python -m flask run --host 0.0.0.0 --port 5000"

if errorlevel 1 (
    echo Error: Could not start backend
    pause
    exit /b 1
)

timeout /t 3 /nobreak

echo.
echo [3/4] Starting Frontend Server (Port 8000)...
echo Opening new window...
start "GNN Frontend - HTTP Server" cmd /k "cd frontend && echo Frontend starting on port 8000... && python -m http.server 8000"

if errorlevel 1 (
    echo Error: Could not start frontend
    pause
    exit /b 1
)

timeout /t 2 /nobreak

echo.
echo [4/4] Opening application in browser...
timeout /t 1

REM Try to open in default browser
if exist "C:\Program Files\Google\Chrome\Application\chrome.exe" (
    start "C:\Program Files\Google\Chrome\Application\chrome.exe" "http://localhost:8000"
) else if exist "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" (
    start "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" "http://localhost:8000"
) else if exist "C:\Program Files\Mozilla Firefox\firefox.exe" (
    start "C:\Program Files\Mozilla Firefox\firefox.exe" "http://localhost:8000"
) else (
    start http://localhost:8000
)

echo.
echo ========================================
echo ✓ SERVERS STARTED SUCCESSFULLY!
echo ========================================
echo.
echo Backend:  http://localhost:5000
echo Frontend: http://localhost:8000
echo.
echo Available Endpoints:
echo   GET  http://localhost:5000/health
echo   GET  http://localhost:5000/city-data
echo   POST http://localhost:5000/predict
echo   POST http://localhost:5000/whatif
echo.
echo To stop servers: Close the command windows
echo.
pause
